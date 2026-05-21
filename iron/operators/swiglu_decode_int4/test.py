#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from iron.operators.swiglu_decode_int4.op import AIESwiGLUDecodeInt4
from iron.operators.swiglu_decode_int4.reference import (
    generate_golden_reference,
)
from iron.common.test_utils import run_test


def generate_test_params(extensive=False):
    if not extensive:
        params = [
            # (embedding_dim, hidden_dim, num_aie_columns, group_size)
            (2048, 8192, 8, 32),
        ]
    else:
        params = [
            (2048, 8192, 8, 32),
            (2048, 8192, 4, 32),
        ]
    names = [
        f"swiglu_decode_int4_E{e}_H{h}_{c}col_g{g}"
        for e, h, c, g in params
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
    "embedding_dim,hidden_dim,num_aie_columns,group_size",
    all_params,
)
def test_swiglu_decode_int4(
    embedding_dim, hidden_dim, num_aie_columns, group_size, aie_context,
):
    golden_ref = generate_golden_reference(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_aie_columns=num_aie_columns,
        group_size=group_size,
    )

    operator = AIESwiGLUDecodeInt4(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_aie_columns=num_aie_columns,
        group_size=group_size,
        context=aie_context,
    )

    input_buffers = {
        "input": golden_ref["x"],
        "weights_gate_up_int4": torch.from_numpy(golden_ref["packed_gate_up"]),
        "weights_down_int4":    torch.from_numpy(golden_ref["packed_down"]),
    }
    output_buffers = {"output": golden_ref["output"]}

    errors, latency_us, _ = run_test(
        operator,
        input_buffers,
        output_buffers,
        rel_tol=0.1,
        abs_tol=1.5,
    )

    print(f"\nLatency (us): {latency_us:.1f}")
    assert not errors, f"Test failed with errors: {errors}"
