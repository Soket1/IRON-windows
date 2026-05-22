# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# V2 of fused INT4 dequant GEMV operator. Same packed buffer layout and
# math as the v1 operator next door, but uses the optimized kernel from
# amd/IRON PR #101:
#   - compile-time DIM_K and GROUP_SIZE (per-shape kernel object)
#   - AIE pipelining hints
#   - double-pump 2-group interleaved dequant chain
#
# Bench in PR #101: 561 us @ K=2048 N=8192 on Strix Point NPU2
# (vs ~2200 us measured in our v1 spike).

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


class AIEFusedDequantGEMVv2(AIEOperatorBase):
    """V2 fused INT4 dequant + GEMV with double-pump + compile-time DIM_K.

    Drop-in replacement for AIEFusedDequantGEMV (v1). Identical packed
    buffer layout (uint4 weights + bf16 scales per tile), identical
    semantics. Differences:

      * The kernel object is compiled PER-SHAPE with `-DDIM_K=K` and
        `-DGROUP_SIZE=G` so the inner loop bounds are constexpr.
      * The C entry point loses the runtime `k` and `group_size` args
        (now baked in at compile time): signature is
        `(m, row_offset, a, b, c)` instead of `(m, k, row_offset, a, b, c, g)`.
      * The MLIR design.py file used here (design.py next door) declares
        the kernel with the matching 5-arg signature.
    """

    def __init__(
        self,
        M,
        K,
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

        self.M = M
        self.K = K
        self.num_aie_columns = num_aie_columns
        self.tile_size_input = tile_size_input
        self.tile_size_output = tile_size_output
        self.group_size = group_size

        self.xclbin_artifact = None
        self.insts_artifact = None

        AIEOperatorBase.__init__(self, context=context)

    def _packed_buffer_size(self):
        num_groups_per_row = self.K // self.group_size
        packed_tile_bytes = (
            self.tile_size_input * self.K // 2
            + self.tile_size_input * num_groups_per_row * 2
        )
        rows_per_col = self.M // self.num_aie_columns
        tiles_per_col = rows_per_col // self.tile_size_input
        return self.num_aie_columns * tiles_per_col * packed_tile_bytes

    def get_artifacts(self, prefix="fused_dequant_gemv_v2_"):
        operator_dir = Path(__file__).parent
        file_name_base = (
            f"{prefix}{self.M}x{self.K}"
            f"_{self.tile_size_input}tsi"
            f"_{self.tile_size_output}tso"
            f"_{self.num_aie_columns}col"
            f"_g{self.group_size}"
        )

        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{file_name_base}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="my_fused_dequant_matvec_v2",
            callback_args=[
                self.context.device_manager.device_type,
                self.num_aie_columns,
                self.M,
                self.K,
                self.tile_size_input,
                self.tile_size_output,
                self.group_size,
            ],
        )

        # Per-shape kernel object name + -DDIM_K/-DGROUP_SIZE flags so
        # the compiler can specialize the inner loop bounds. Different
        # (K, group_size) combos produce distinct .o files automatically.
        kernel_obj_name = f"fused_dequant_gemv_v2_{self.K}k_g{self.group_size}.o"

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
                            / "fused_dequant_gemv_v2.cc"
                        )
                    ],
                    extra_flags=[
                        f"-DDIM_K={self.K}",
                        f"-DGROUP_SIZE={self.group_size}",
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
        self.add_buffer("vector", self.K, dtype=bfloat16)
        self.add_buffer("output", self.M, dtype=bfloat16)
        self.add_kernel(
            "fused_dequant_gemv_v2",
            self.xclbin_artifact,
            self.xclbin_artifact.kernel_name,
            self.insts_artifact,
        )
        self.add_to_runlist(
            "fused_dequant_gemv_v2", "packed_weights", "vector", "output"
        )

    def forward(self, vector, packed_weights=None):
        vector = vector.reshape(*vector.shape[-1:])
        if vector.shape[-1] != self.K or vector.dtype != torch.bfloat16:
            raise AIEOperatorConstraintError(
                f"AIEFusedDequantGEMVv2: expected bf16 vector of length "
                f"{self.K}, got shape {vector.shape} dtype {vector.dtype}"
            )
        if packed_weights is not None:
            self.write_buffer("packed_weights", packed_weights)
        self.write_buffer("vector", vector)
        self.run_runlist()
        return self.read_buffer_as_torch("output", (self.M,))
