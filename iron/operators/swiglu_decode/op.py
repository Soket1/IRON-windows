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
    KernelArchiveArtifact,
    SourceArtifact,
    PythonGeneratedMLIRArtifact,
)
from iron.operators.dual_gemv_silu_mul.op import AIEDualGEMVSiLUMul, interleave_weights
from iron.operators.gemv.op import GEMV
from iron.common.utils import torch_to_numpy


class AIESwiGLUDecode(AIEOperatorBase):

    # AIE2p L1 data memory budget (64KB per tile on Strix Point / NPU2).
    _L1_BUDGET_BYTES = 64 * 1024
    # Fused gate+up kernel always uses 4 columns (dual-GEMV DMA constraint).
    _FUSED_COLS = 4
    # Candidate tile sizes, largest first (must divide tile_out evenly).
    _TILE_CANDIDATES = [8, 4, 2, 1]

    def __init__(self, embedding_dim, hidden_dim, prio_accuracy=False,
                 num_aie_columns=8, context=None):
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.prio_accuracy = prio_accuracy
        self.num_aie_columns = num_aie_columns  # device column count
        # weights to be set by user (e.g., assign_weights in FeedForward block)
        self.weights_1 = None
        self.weights_2 = None
        self.weights_3 = None

        # Artifacts created by set_up_artifacts()
        self.combined_xclbin = None
        self.fused_xclbin = None
        self.fused_insts = None
        self.gemv_2_xclbin = None
        self.gemv_2_insts = None

        super().__init__(context=context)

    @classmethod
    def _compute_tile_in(cls, K, tile_out, has_static_bufs=False,
                         static_buf_elems=0):
        """Select largest tile_in that fits in L1.

        Accounts for A (double-buffer), B (depth=1), C (depth=2), and
        optional static kernel buffers (left_buf + right_buf).
        Returns 1 as fallback if no tile_in fits the budget (caller should
        validate total L1 separately).
        """
        bf16 = 2
        static_bytes = (2 * static_buf_elems * bf16) if has_static_bufs else 0
        bc_static = (K * bf16) + (2 * tile_out * bf16) + static_bytes
        if bc_static >= cls._L1_BUDGET_BYTES:
            return 1  # will overflow — caller validation should catch this
        available = cls._L1_BUDGET_BYTES - bc_static
        max_by_l1 = available // (2 * K * bf16)
        for tsi in cls._TILE_CANDIDATES:
            if (tsi <= tile_out and tsi <= max_by_l1
                    and tile_out % tsi == 0 and tile_out % tsi == 0):
                return tsi
        return 1  # fallback

    def set_up_artifacts(self):
        artifacts = []
        device_type = self.context.device_manager.device_type

        # --- Fused gate+up+SiLU+mul (dual-GEMV, cols=4) ---
        fused_cols = self._FUSED_COLS
        fused_tile_out = self.hidden_dim // fused_cols
        m_output_max = fused_tile_out  # for -DM_OUTPUT_MAX compile flag
        fused_tile_in = self._compute_tile_in(
            K=self.embedding_dim,
            tile_out=fused_tile_out,
            has_static_bufs=True,
            static_buf_elems=m_output_max,
        )

        fused = AIEDualGEMVSiLUMul(
            M=self.hidden_dim,
            K=self.embedding_dim,
            num_aie_columns=fused_cols,
            tile_size_input=fused_tile_in,
            tile_size_output=fused_tile_out,
        )
        self.fused = fused
        self.hidden_dim_padded = self.hidden_dim
        fused_xclbin, fused_insts = fused.get_artifacts(prefix="swiglu_decode_fused_")
        fused_xclbin.extra_flags += [
            "--xclbin-instance-name=swiglu_fused",
            "--xclbin-kernel-id=0x901",
        ]
        fused_xclbin.kernel_name = "swiglu_fused"
        # Tell the kernel how large the static buffers must be.
        # dependencies[1] is the KernelObjectArtifact (dual_gemv_silu_mul.o).
        fused_xclbin.dependencies[1].extra_flags.append(
            f"-DM_OUTPUT_MAX={m_output_max}"
        )
        artifacts.append(fused_insts)

        # --- Down GEMV (cols=device cols) ---
        down_cols = self.num_aie_columns
        down_tile_out = self.embedding_dim // down_cols
        down_tile_in = self._compute_tile_in(
            K=self.hidden_dim,
            tile_out=down_tile_out,
        )

        gemv_2 = GEMV(
            M=self.embedding_dim,
            K=self.hidden_dim,
            num_aie_columns=down_cols,
            tile_size_input=down_tile_in,
            tile_size_output=down_tile_out,
        )
        self.gemv_2 = gemv_2
        gemv_2_xclbin, gemv_2_insts = gemv_2.get_artifacts(
            prefix="swiglu_decode_gemv_2_"
        )
        gemv_2_xclbin.xclbin_input = fused_xclbin
        gemv_2_xclbin.extra_flags += [
            "--xclbin-instance-name=swiglu_gemv_2",
            "--xclbin-kernel-id=0x902",
        ]
        gemv_2_xclbin.kernel_name = "swiglu_gemv_2"
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
        self.add_buffer("input", self.embedding_dim)
        # Pre-interleave W1 and W2 for the fused dual-GEMV design
        rows_per_col = self.hidden_dim // self.fused.num_aie_columns
        w_interleaved = interleave_weights(
            self.weights_1, self.weights_2, rows_per_col, self.fused.num_aie_columns
        )
        self.add_buffer(
            "weights_gate_up",
            2 * self.embedding_dim * self.hidden_dim_padded,
            static_data=torch_to_numpy(w_interleaved),
        )
        self.add_buffer(
            "weights_3",
            self.hidden_dim_padded * self.embedding_dim,
            static_data=torch_to_numpy(self.weights_3),
        )
        self.add_buffer("intermediate", self.hidden_dim_padded)
        self.add_buffer("output", self.embedding_dim)
        self.add_kernel(
            "swiglu_fused",
            self.combined_xclbin,
            self.fused_xclbin.kernel_name,
            self.fused_insts,
        )
        self.add_kernel(
            "swiglu_gemv_2",
            self.combined_xclbin,
            self.gemv_2_xclbin.kernel_name,
            self.gemv_2_insts,
        )
        self.add_to_runlist("swiglu_fused", "weights_gate_up", "input", "intermediate")
        self.add_to_runlist("swiglu_gemv_2", "weights_3", "intermediate", "output")

    def forward(self, x):
        x_flat = x.reshape(x.shape[-1])
        assert x_flat.shape[0] == self.embedding_dim

        self.write_buffer("input", x_flat)
        self.run_runlist()
        result = self.read_buffer_as_torch(
            "output",
            (self.embedding_dim,),
        ).view_as(x)

        return result
