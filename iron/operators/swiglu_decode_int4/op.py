# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import torch
import numpy as np
from ml_dtypes import bfloat16

from iron.common import (
    AIEOperatorBase,
    XclbinArtifact,
    InstsBinArtifact,
    KernelObjectArtifact,
    SourceArtifact,
    PythonGeneratedMLIRArtifact,
)
from iron.operators.dual_fused_dequant_gemv_silu_mul.op import (
    AIEDualFusedDequantGEMVSiLUMul,
    interleave_packed_int4,
)
from iron.operators.fused_dequant_gemv_v2.op import AIEFusedDequantGEMVv2
from iron.operators.fused_dequant_gemv.reference import quantize_and_pack
from iron.common.utils import torch_to_numpy


class AIESwiGLUDecodeInt4(AIEOperatorBase):
    """AIE-accelerated SwiGLU FFN decode with INT4 quantized weights.

    Computes: output = (silu(dequant(W1) @ x) * (dequant(W2) @ x)) @ dequant(W3)^T

    Mirror of `AIESwiGLUDecode` (bf16) but uses
    `AIEDualFusedDequantGEMVSiLUMul` for the fused gate+up+silu+mul stage
    and `AIEFusedDequantGEMVv2` for the down stage. Both stages share the
    same Q4_0-compatible per-group bf16 scale format (group_size=32).
    """

    # AIE2p L1 data memory budget (64KB per tile on Strix Point / NPU2).
    _L1_BUDGET_BYTES = 64 * 1024
    # Fused gate+up kernel always uses 4 columns (dual-GEMV DMA constraint
    # inherited from the bf16 design).
    _FUSED_COLS = 4
    _TILE_CANDIDATES = [8, 4, 2, 1]

    def __init__(self, embedding_dim, hidden_dim, num_aie_columns=8,
                 group_size=32, context=None):
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_aie_columns = num_aie_columns
        self.group_size = group_size
        assert hidden_dim % self._FUSED_COLS == 0
        assert embedding_dim % num_aie_columns == 0
        assert hidden_dim % num_aie_columns == 0
        assert embedding_dim % group_size == 0
        assert hidden_dim % group_size == 0

        # User-supplied bf16 weights (M, K) -- quantized per-row at
        # set_up_runtime() time via fused_dequant_gemv.quantize_and_pack.
        self.weights_1 = None
        self.weights_2 = None
        self.weights_3 = None

        self.combined_xclbin = None
        self.fused_xclbin = None
        self.fused_insts = None
        self.gemv_2_xclbin = None
        self.gemv_2_insts = None

        super().__init__(context=context)

    @classmethod
    def _compute_fused_tile_in(cls, K, tile_out, m_output_max, group_size):
        """Largest tile_in that fits L1 for the fused INT4 stage.

        Budget components:
          - A: 2 buffers (FIFO depth=2) of packed_tile_bytes each
          - B: K bf16 (depth=1)
          - C: 2 * tile_out bf16 (depth=2)
          - static left_buf + right_buf: 2 * m_output_max bf16
        """
        bf16 = 2
        num_groups_per_row = K // group_size
        # packed bytes for one tile of m_input rows; depend on m_input
        # which is the unknown -- treat as a function of tsi below.
        bc_static = (K * bf16) + (2 * tile_out * bf16) + (2 * m_output_max * bf16)
        if bc_static >= cls._L1_BUDGET_BYTES:
            return 1
        available = cls._L1_BUDGET_BYTES - bc_static
        for tsi in cls._TILE_CANDIDATES:
            packed_tile_bytes = tsi * K // 2 + tsi * num_groups_per_row * 2
            # A buffers in L1 (depth=2)
            a_bytes = 2 * packed_tile_bytes
            if (tsi <= tile_out and a_bytes <= available
                    and tile_out % tsi == 0):
                return tsi
        return 1

    @classmethod
    def _compute_down_tile_in(cls, K, tile_out, group_size):
        """Largest tile_in for the down INT4 GEMV (no static buffers)."""
        bf16 = 2
        num_groups_per_row = K // group_size
        bc = (K * bf16) + (2 * tile_out * bf16)
        if bc >= cls._L1_BUDGET_BYTES:
            return 1
        available = cls._L1_BUDGET_BYTES - bc
        for tsi in cls._TILE_CANDIDATES:
            packed_tile_bytes = tsi * K // 2 + tsi * num_groups_per_row * 2
            a_bytes = 2 * packed_tile_bytes
            if (tsi <= tile_out and a_bytes <= available
                    and tile_out % tsi == 0):
                return tsi
        return 1

    def set_up_artifacts(self):
        artifacts = []

        # --- Fused gate+up+SiLU+mul (cols=_FUSED_COLS) ---
        fused_cols = self._FUSED_COLS
        fused_tile_out = self.hidden_dim // fused_cols
        m_output_max = fused_tile_out
        fused_tile_in = self._compute_fused_tile_in(
            K=self.embedding_dim,
            tile_out=fused_tile_out,
            m_output_max=m_output_max,
            group_size=self.group_size,
        )

        fused = AIEDualFusedDequantGEMVSiLUMul(
            M=self.hidden_dim,
            K=self.embedding_dim,
            num_aie_columns=fused_cols,
            tile_size_input=fused_tile_in,
            tile_size_output=fused_tile_out,
            group_size=self.group_size,
        )
        self.fused = fused
        fused_xclbin, fused_insts = fused.get_artifacts(
            prefix="swiglu_decode_int4_fused_"
        )
        fused_xclbin.extra_flags += [
            "--xclbin-instance-name=swiglu_fused_int4",
            "--xclbin-kernel-id=0x911",
        ]
        fused_xclbin.kernel_name = "swiglu_fused_int4"
        # Tell the kernel how large the static buffers must be.
        # dependencies[1] is the KernelObjectArtifact (dual_fused_dequant_gemv_silu_mul.o).
        fused_xclbin.dependencies[1].extra_flags.append(
            f"-DM_OUTPUT_MAX={m_output_max}"
        )
        artifacts.append(fused_insts)

        # --- Down GEMV (cols = device cols) ---
        down_cols = self.num_aie_columns
        down_tile_out = self.embedding_dim // down_cols
        down_tile_in = self._compute_down_tile_in(
            K=self.hidden_dim,
            tile_out=down_tile_out,
            group_size=self.group_size,
        )

        gemv_2 = AIEFusedDequantGEMVv2(
            M=self.embedding_dim,
            K=self.hidden_dim,
            num_aie_columns=down_cols,
            tile_size_input=down_tile_in,
            tile_size_output=down_tile_out,
            group_size=self.group_size,
        )
        self.gemv_2 = gemv_2
        gemv_2_xclbin, gemv_2_insts = gemv_2.get_artifacts(
            prefix="swiglu_decode_int4_gemv_2_"
        )
        gemv_2_xclbin.xclbin_input = fused_xclbin
        gemv_2_xclbin.extra_flags += [
            "--xclbin-instance-name=swiglu_gemv_2_int4",
            "--xclbin-kernel-id=0x912",
        ]
        gemv_2_xclbin.kernel_name = "swiglu_gemv_2_int4"
        gemv_2_xclbin.dependencies.add(fused_xclbin)
        artifacts.append(gemv_2_xclbin)
        artifacts.append(gemv_2_insts)

        self.combined_xclbin = gemv_2_xclbin
        self.fused_xclbin = fused_xclbin
        self.fused_insts = fused_insts
        self.gemv_2_xclbin = gemv_2_xclbin
        self.gemv_2_insts = gemv_2_insts

        self.add_artifacts(artifacts)

    def set_up_runtime(self):
        # Compute packed buffer sizes
        fused_packed_total = self.fused._packed_total_bytes()
        gemv_2_packed_total = self.gemv_2._packed_buffer_size()

        self.add_buffer("input", self.embedding_dim, dtype=bfloat16)
        self.add_buffer(
            "weights_gate_up_int4",
            fused_packed_total,
            dtype=np.uint8,
        )
        self.add_buffer(
            "weights_down_int4",
            gemv_2_packed_total,
            dtype=np.uint8,
        )
        self.add_buffer("intermediate", self.hidden_dim, dtype=bfloat16)
        self.add_buffer("output", self.embedding_dim, dtype=bfloat16)

        self.add_kernel(
            "swiglu_fused_int4",
            self.combined_xclbin,
            self.fused_xclbin.kernel_name,
            self.fused_insts,
        )
        self.add_kernel(
            "swiglu_gemv_2_int4",
            self.combined_xclbin,
            self.gemv_2_xclbin.kernel_name,
            self.gemv_2_insts,
        )
        self.add_to_runlist(
            "swiglu_fused_int4",
            "weights_gate_up_int4", "input", "intermediate",
        )
        self.add_to_runlist(
            "swiglu_gemv_2_int4",
            "weights_down_int4", "intermediate", "output",
        )

    def forward(self, x):
        """Forward pass for golden test. Quantize+pack weights at first
        call, then stream activations on subsequent calls."""
        x_flat = x.reshape(x.shape[-1])
        assert x_flat.shape[0] == self.embedding_dim

        # Packed weights are not auto-quantized here -- caller is expected
        # to populate the buffers via write_buffer() OR provide raw bf16
        # weights via assign_weights() prior to compile.
        self.write_buffer("input", x_flat)
        self.run_runlist()
        result = self.read_buffer_as_torch(
            "output", (self.embedding_dim,),
        ).view_as(x)
        return result
