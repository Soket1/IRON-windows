# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# V3 fused INT4 dequant batched GEMV operator.
# Processes M_BATCH activation rows against the same packed weight buffer
# so dequant cost is amortised. Designed for speculative-decoding
# verification batches (M_BATCH = 4 or 8).
#
# Interface changes vs v2:
#   vector buffer:  M_BATCH × K  bf16   (M_BATCH stacked rows)
#   output buffer:  N × M_BATCH  bf16   (interleaved per weight row)
# Packed-weight buffer layout: identical to v2 (no change).

import torch
import numpy as np
from ml_dtypes import bfloat16
from pathlib import Path

from iron.common import (
    AIEOperatorBase,
    AIEOperatorConstraintError,
    XclbinArtifact,
    InstsBinArtifact,
    KernelObjectArtifact,
    SourceArtifact,
    PythonGeneratedMLIRArtifact,
)
from iron.common.utils import torch_to_numpy


class AIEFusedDequantGEMVv3(AIEOperatorBase):
    """V3 fused INT4 dequant + batched GEMV for spec-dec verification.

    Identical packed-weight buffer layout to v2. Key differences:

      * Adds M_BATCH parameter: the 'vector' buffer holds M_BATCH activation
        rows (M_BATCH × K bf16) and 'output' holds N × M_BATCH bf16 results.
      * Kernel compiled with -DM_BATCH=N so the inner MB loop is unrolled.
      * Entry point signature unchanged externally: (m, row_offset, a, b, c).
    """

    def __init__(
        self,
        M,
        K,
        m_batch=4,
        num_aie_columns=4,
        tile_size_input=1,
        tile_size_output=None,
        group_size=32,
        context=None,
    ):
        if tile_size_output is None:
            tile_size_output = M // num_aie_columns

        assert (
            tile_size_output % tile_size_input == 0
            and tile_size_output >= tile_size_input
        ), "tile_size_output must be a multiple of tile_size_input"
        assert K % group_size == 0, "K must be a multiple of group_size"
        assert group_size % 32 == 0, "group_size must be a multiple of 32"
        assert M % num_aie_columns == 0, "M must be a multiple of num_aie_columns"
        assert 1 <= m_batch <= 8, "m_batch must be 1..8"

        self.M = M
        self.K = K
        self.m_batch = m_batch
        self.num_aie_columns = num_aie_columns
        self.tile_size_input = tile_size_input
        self.tile_size_output = tile_size_output
        self.group_size = group_size

        self.xclbin_artifact = None
        self.insts_artifact = None

        AIEOperatorBase.__init__(self, context=context)

    def _packed_buffer_size(self):
        """Identical to v2 — weight layout unchanged."""
        num_groups_per_row = self.K // self.group_size
        packed_tile_bytes = (
            self.tile_size_input * self.K // 2
            + self.tile_size_input * num_groups_per_row * 2
        )
        rows_per_col = self.M // self.num_aie_columns
        tiles_per_col = rows_per_col // self.tile_size_input
        return self.num_aie_columns * tiles_per_col * packed_tile_bytes

    def get_artifacts(self, prefix="fused_dequant_gemv_v3_"):
        operator_dir = Path(__file__).parent
        file_name_base = (
            f"{prefix}{self.M}x{self.K}"
            f"_{self.tile_size_input}tsi"
            f"_{self.tile_size_output}tso"
            f"_{self.num_aie_columns}col"
            f"_g{self.group_size}"
            f"_mb{self.m_batch}"
        )

        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{file_name_base}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="my_fused_dequant_matvec_v3",
            callback_args=[
                self.context.device_manager.device_type,
                self.num_aie_columns,
                self.M,
                self.K,
                self.tile_size_input,
                self.m_batch,
                self.tile_size_output,
                self.group_size,
            ],
        )

        # Per-shape + per-M_BATCH kernel object.
        kernel_obj_name = (
            f"fused_dequant_gemv_v3_{self.K}k_g{self.group_size}_mb{self.m_batch}.o"
        )

        xclbin_artifact = XclbinArtifact.new(
            f"{file_name_base}.xclbin",
            depends=[
                mlir_artifact,
                KernelObjectArtifact.new(
                    kernel_obj_name,
                    depends=[
                        SourceArtifact.new(
                            self.context.base_dir
                            / "aie_kernels"
                            / "aie2p"
                            / "fused_dequant_gemv_v3.cc"
                        )
                    ],
                    extra_flags=[
                        f"-DDIM_K={self.K}",
                        f"-DGROUP_SIZE={self.group_size}",
                        f"-DM_BATCH={self.m_batch}",
                    ],
                ),
            ],
        )

        insts_artifact = InstsBinArtifact.new(
            f"{file_name_base}.bin", depends=[mlir_artifact]
        )

        return xclbin_artifact, insts_artifact

    def set_up_artifacts(self):
        xclbin_artifact, insts_artifact = self.get_artifacts()
        self.xclbin_artifact = xclbin_artifact
        self.insts_artifact = insts_artifact
        self.add_artifacts([xclbin_artifact, insts_artifact])

    def set_up_runtime(self):
        self.add_buffer("packed_weights", self._packed_buffer_size(), dtype=np.uint8)
        self.add_buffer("vector", self.m_batch * self.K, dtype=bfloat16)
        self.add_buffer("output", self.M * self.m_batch, dtype=bfloat16)
        self.add_kernel(
            "fused_dequant_gemv_v3",
            self.xclbin_artifact,
            self.xclbin_artifact.kernel_name,
            self.insts_artifact,
        )
        self.add_to_runlist(
            "fused_dequant_gemv_v3", "packed_weights", "vector", "output"
        )

    def forward(self, vectors, packed_weights=None):
        """
        vectors:        [m_batch, K] bfloat16 tensor
        packed_weights: optional pre-packed weight bytes
        returns:        [M, m_batch] bfloat16 tensor
        """
        if vectors.shape != (self.m_batch, self.K) or vectors.dtype != torch.bfloat16:
            raise AIEOperatorConstraintError(
                f"AIEFusedDequantGEMVv3: expected bf16 tensor [{self.m_batch}, {self.K}], "
                f"got shape {vectors.shape} dtype {vectors.dtype}"
            )
        if packed_weights is not None:
            self.write_buffer("packed_weights", packed_weights)
        self.write_buffer("vector", vectors.reshape(-1))
        self.run_runlist()
        return self.read_buffer_as_torch("output", (self.M, self.m_batch))
