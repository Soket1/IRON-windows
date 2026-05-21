#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from iron.operators.dual_fused_dequant_gemv_silu_mul.op import (
    AIEDualFusedDequantGEMVSiLUMul,
)
from iron.operators.dual_fused_dequant_gemv_silu_mul.reference import (
    generate_golden_reference,
)
from iron.common.test_utils import run_test


def generate_test_params(extensive=False):
    if not extensive:
        params = [
            # (M, K, cols, tsi, tso, group_size)
            (2048, 2048, 4, 4, 512, 32),
        ]
    else:
        params = [
            (2048, 2048, 4, 4, 512, 32),
            (8192, 2048, 4, 4, 2048, 32),
        ]

    names = [
        f"dual_fused_dequant_gemv_silu_mul_{M}x{K}"
        f"_{tsi}tsi_{tso}tso_{cols}col_g{gs}"
        for M, K, cols, tsi, tso, gs in params
    ]
    return params, names


regular_params, regular_names = generate_test_params(extensive=False)
extensive_params, extensive_names = generate_test_params(extensive=True)

all_params = [
    pytest.param(*params, id=name)
    for params, name in zip(regular_params, regular_names)
] + [
    pytest.param(*params, marks=pytest.mark.extensive, id=name)
    for params, name in zip(extensive_params, extensive_names)
]


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
)
@pytest.mark.parametrize(
    "M,K,num_aie_columns,tile_size_input,tile_size_output,group_size",
    all_params,
)
def test_dual_fused_dequant_gemv_silu_mul(
    M, K, num_aie_columns, tile_size_input, tile_size_output, group_size,
    aie_context,
):
    golden_ref = generate_golden_reference(
        M=M, K=K, group_size=group_size,
        m_input=tile_size_input, cols=num_aie_columns,
    )

    operator = AIEDualFusedDequantGEMVSiLUMul(
        M=M, K=K, num_aie_columns=num_aie_columns,
        tile_size_input=tile_size_input,
        tile_size_output=tile_size_output,
        group_size=group_size,
        context=aie_context,
    )

    packed_weights_tensor = torch.from_numpy(golden_ref["packed_weights"])
    input_buffers = {
        "weights_packed_int4": packed_weights_tensor,
        "vector": golden_ref["x"],
    }
    output_buffers = {"output": golden_ref["output"]}

    errors, latency_us, _ = run_test(
        operator,
        input_buffers,
        output_buffers,
        rel_tol=0.07,
        abs_tol=0.7,
    )

    print(f"\nLatency (us): {latency_us:.1f}")
    assert not errors, f"Test failed with errors: {errors}"
