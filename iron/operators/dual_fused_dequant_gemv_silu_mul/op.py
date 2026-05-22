# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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


def interleave_packed_int4(packed_W1, packed_W2, cols, bytes_per_col_per_weight):
    """Interleave two packed-INT4 weight buffers per-column for DMA streaming.

    Each weight (W1 = gate, W2 = up) is already packed via
    `fused_dequant_gemv.reference.quantize_and_pack` into a flat uint8
    buffer of size ``cols * bytes_per_col_per_weight``.  The output is a
    single uint8 buffer twice as big, laid out per-column as:

        [col 0 gate tiles][col 0 up tiles][col 1 gate tiles][col 1 up tiles]...

    matching the DDR access pattern expected by the design.
    """
    assert packed_W1.shape == packed_W2.shape
    assert packed_W1.dtype == np.uint8 and packed_W2.dtype == np.uint8
    assert packed_W1.size == cols * bytes_per_col_per_weight

    bytes_per_col = 2 * bytes_per_col_per_weight
    out = np.empty(cols * bytes_per_col, dtype=np.uint8)
    for col in range(cols):
        src_lo = col * bytes_per_col_per_weight
        src_hi = src_lo + bytes_per_col_per_weight
        dst_lo = col * bytes_per_col
        out[dst_lo : dst_lo + bytes_per_col_per_weight] = packed_W1[src_lo:src_hi]
        out[dst_lo + bytes_per_col_per_weight : dst_lo + bytes_per_col] = packed_W2[src_lo:src_hi]
    return out


class AIEDualFusedDequantGEMVSiLUMul(AIEOperatorBase):
    """AIE-accelerated fused dual-GEMV (INT4 dequant) + SiLU + elementwise multiply.

    Computes: output = silu(dequant(W1) @ x) * (dequant(W2) @ x)

    Mirror of `AIEDualGEMVSiLUMul` but the weights are INT4-packed (Q4_0
    layout: uint4 nibbles followed by per-group bf16 scales). The kernel
    dequantizes in-register; activation `x` and intermediate buffers
    stay bf16 in L1.
    """

    def __init__(
        self,
        M,
        K,
        num_aie_columns=4,
        tile_size_input=4,
        tile_size_output=None,
        group_size=32,
        context=None,
    ):
        if tile_size_output is None:
            tile_size_output = M // num_aie_columns
        assert tile_size_output % tile_size_input == 0
        assert tile_size_output >= tile_size_input
        assert K % group_size == 0, "K must be a multiple of group_size"
        assert group_size % 32 == 0, "group_size must be a multiple of 32"
        assert M % num_aie_columns == 0
        self.M = M
        self.K = K
        self.num_aie_columns = num_aie_columns
        self.tile_size_input = tile_size_input
        self.tile_size_output = tile_size_output
        self.group_size = group_size

        self.xclbin_artifact = None
        self.insts_artifact = None

        AIEOperatorBase.__init__(self, context=context)

    def _packed_bytes_per_col_per_weight(self):
        num_groups_per_row = self.K // self.group_size
        packed_tile_bytes = (
            self.tile_size_input * self.K // 2
            + self.tile_size_input * num_groups_per_row * 2
        )
        rows_per_col = self.M // self.num_aie_columns
        tiles_per_col = rows_per_col // self.tile_size_input
        return tiles_per_col * packed_tile_bytes

    def _packed_total_bytes(self):
        return 2 * self.num_aie_columns * self._packed_bytes_per_col_per_weight()

    def get_artifacts(self, prefix="dual_fused_dequant_gemv_silu_mul_"):
        operator_dir = Path(__file__).parent
        file_name_base = (
            f"{prefix}{self.M}x{self.K}_{self.tile_size_input}tsi_"
            f"{self.tile_size_output}tso_{self.num_aie_columns}col_g{self.group_size}"
        )

        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{file_name_base}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="my_dual_fused_dequant_gemv_silu_mul",
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

        # Per-shape kernel object name + -DDIM_K/-DGROUP_SIZE flags so the
        # AIE compiler can specialize the inner loop bounds. Different
        # (K, group_size) combos produce distinct .o files automatically.
        kernel_obj_name = (
            f"dual_fused_dequant_gemv_silu_mul_{self.K}k_g{self.group_size}.o"
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
                            / "dual_fused_dequant_gemv_silu_mul.cc"
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
        # Packed weight buffer for gate + up (uint8 bytes)
        self.add_buffer(
            "weights_packed_int4",
            self._packed_total_bytes(),
            dtype=np.uint8,
        )
        self.add_buffer("vector", self.K, dtype=bfloat16)
        self.add_buffer("output", self.M, dtype=bfloat16)
        self.add_kernel(
            "dual_fused_dequant_gemv_silu_mul",
            self.xclbin_artifact,
            self.xclbin_artifact.kernel_name,
            self.insts_artifact,
        )
        self.add_to_runlist(
            "dual_fused_dequant_gemv_silu_mul",
            "weights_packed_int4", "vector", "output",
        )

    def forward(self, vector, packed_weights=None):
        """Forward pass. If packed_weights is None, assumes the buffer was
        already populated. The packed buffer must be interleaved per
        column via `interleave_packed_int4`."""
        vector = vector.reshape(*vector.shape[-1:])
        if vector.shape[-1] != self.K or vector.dtype != torch.bfloat16:
            raise AIEOperatorConstraintError(
                f"AIEDualFusedDequantGEMVSiLUMul: expected bf16 vector of "
                f"length {self.K}, got shape {vector.shape} dtype {vector.dtype}"
            )
        if packed_weights is not None:
            self.write_buffer("weights_packed_int4", packed_weights)
        self.write_buffer("vector", vector)
        self.run_runlist()
        return self.read_buffer_as_torch("output", (self.M,))
