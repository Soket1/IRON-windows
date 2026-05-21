# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
from ml_dtypes import bfloat16

from iron.operators.fused_dequant_gemv.reference import quantize_and_pack
from iron.operators.dual_fused_dequant_gemv_silu_mul.op import (
    interleave_packed_int4,
)


def generate_golden_reference(
    embedding_dim=2048, hidden_dim=8192, num_aie_columns=8,
    fused_cols=4, group_size=32, seed=42,
):
    """Build random INT4-quantized gate/up/down weights, run the full
    SwiGLU FFN forward in fp32 for the golden output.

    Returns a dict with packed buffers (gate+up interleaved, down stand-alone),
    the bf16 input vector, the bf16 reference output, and the dequantized
    weight tensors for debugging.
    """
    torch.manual_seed(seed)

    M_fused = hidden_dim
    K_fused = embedding_dim
    M_down  = embedding_dim
    K_down  = hidden_dim

    x = torch.randn(K_fused, dtype=torch.bfloat16) * 4

    # Quantize+pack each of gate (W1) and up (W2) to the fused operator's
    # tile layout (cols = fused_cols).
    # tile_size_input must match what AIESwiGLUDecodeInt4._compute_fused_tile_in
    # would pick -- but for the test harness we pick the standard tsi=4
    # (matches the default _TILE_CANDIDATES first-fit).
    m_input_fused = 4
    packed_W1, W1_dequant = quantize_and_pack(
        M_fused, K_fused, group_size, m_input_fused, fused_cols
    )
    packed_W2, W2_dequant = quantize_and_pack(
        M_fused, K_fused, group_size, m_input_fused, fused_cols
    )

    rows_per_col_fused = M_fused // fused_cols
    tiles_per_col_fused = rows_per_col_fused // m_input_fused
    num_groups_per_row_fused = K_fused // group_size
    packed_tile_bytes_fused = (
        m_input_fused * K_fused // 2
        + m_input_fused * num_groups_per_row_fused * 2
    )
    bytes_per_col_per_weight_fused = tiles_per_col_fused * packed_tile_bytes_fused

    packed_gate_up = interleave_packed_int4(
        packed_W1, packed_W2, fused_cols, bytes_per_col_per_weight_fused
    )

    # Quantize+pack the down projection.
    m_input_down = 1
    packed_W3, W3_dequant = quantize_and_pack(
        M_down, K_down, group_size, m_input_down, num_aie_columns
    )

    # fp32 golden
    g = (W1_dequant.to(torch.float32) @ x.to(torch.float32))
    u = (W2_dequant.to(torch.float32) @ x.to(torch.float32))
    silu_g = g * torch.sigmoid(g)
    intermediate = (silu_g * u)
    output = (W3_dequant.to(torch.float32) @ intermediate).to(torch.bfloat16)

    return {
        "packed_gate_up": packed_gate_up,
        "packed_down":   packed_W3,
        "x": x,
        "output": output,
        "W1_dequant": W1_dequant,
        "W2_dequant": W2_dequant,
        "W3_dequant": W3_dequant,
    }
