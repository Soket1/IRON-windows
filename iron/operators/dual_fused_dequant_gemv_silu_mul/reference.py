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
    M=2048, K=2048, group_size=32, m_input=4, cols=4, seed=42,
):
    """Build random INT4-quantized gate (W1) and up (W2) weights, pack them
    per-column, interleave for the dual-GEMV DMA, run dequant + GEMV +
    SiLU + elementwise mul in fp32 for the golden output.

    Returns:
      packed_weights : uint8 numpy buffer ready to be written to the
                       NPU's L3 weights buffer.
      x              : bfloat16 input vector of length K.
      output         : bfloat16 reference output of length M
                       = silu(W1_dequant @ x) * (W2_dequant @ x).
      W1_dequant     : bf16 dequantized gate weights (M, K)
      W2_dequant     : bf16 dequantized up   weights (M, K)
    """
    torch.manual_seed(seed)

    x = torch.randn(K, dtype=torch.bfloat16) * 4

    packed_W1, W1_dequant = quantize_and_pack(M, K, group_size, m_input, cols)
    packed_W2, W2_dequant = quantize_and_pack(M, K, group_size, m_input, cols)

    rows_per_col = M // cols
    tiles_per_col = rows_per_col // m_input
    num_groups_per_row = K // group_size
    packed_tile_bytes = m_input * K // 2 + m_input * num_groups_per_row * 2
    bytes_per_col_per_weight = tiles_per_col * packed_tile_bytes

    packed_weights = interleave_packed_int4(
        packed_W1, packed_W2, cols, bytes_per_col_per_weight
    )

    # SiLU(x) = x * sigmoid(x)
    g = (W1_dequant.to(torch.float32) @ x.to(torch.float32))
    u = (W2_dequant.to(torch.float32) @ x.to(torch.float32))
    silu_g = g * torch.sigmoid(g)
    output = (silu_g * u).to(torch.bfloat16)

    return {
        "packed_weights": packed_weights,
        "x": x,
        "output": output,
        "W1_dequant": W1_dequant,
        "W2_dequant": W2_dequant,
    }
