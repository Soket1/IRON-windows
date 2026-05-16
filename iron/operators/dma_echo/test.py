#!/usr/bin/env python3
"""DMA echo test using IRON's test framework.

Usage:
    pytest test.py -v --iterations 1
    pytest test.py -v --iterations 1 -k test_echo_v1
    pytest test.py -v --iterations 1 -k test_echo_v2
"""
import numpy as np
import torch
import pytest
from ml_dtypes import bfloat16

from iron.operators.dma_echo.op import AIEEchoV1, AIEEchoV2
from iron.common.test_utils import run_test


def test_echo_v1(aie_context):
    """v1: single ObjectFifo memcpy — tests basic DMA path."""
    N = 256
    dtype = bfloat16

    # Known pattern as input
    input_np = np.full(N, 0.25, dtype=np.float32).astype(dtype)
    input_np[0] = np.float32(0.12).astype(dtype)
    input_np[1] = np.float32(0.23).astype(dtype)
    expected_np = input_np.copy()

    input_torch = torch.from_numpy(input_np.view(np.uint16)).to(torch.bfloat16)
    expected_torch = torch.from_numpy(expected_np.view(np.uint16)).to(torch.bfloat16)

    operator = AIEEchoV1(size=N, context=aie_context)

    input_buffers = {"input": input_torch}
    output_buffers = {"output": expected_torch}

    errors, latency_us, bandwidth_gbps = run_test(
        operator, input_buffers, output_buffers, rel_tol=0.0, abs_tol=0.0
    )

    print(f"\nv1 Latency (us): {latency_us:.1f}")
    assert not errors, f"Echo v1 failed: {errors}"
    print("Echo v1 PASSED")


def test_echo_v2(aie_context):
    """v2: dual ObjectFifo concat — tests K+V style multi-DMA."""
    N = 128
    dtype = bfloat16

    a_np = np.full(N, 0.25, dtype=np.float32).astype(dtype)
    a_np[0] = np.float32(0.12).astype(dtype)
    b_np = np.full(N, 0.50, dtype=np.float32).astype(dtype)
    b_np[0] = np.float32(0.34).astype(dtype)
    expected_np = np.concatenate([a_np, b_np])

    a_torch = torch.from_numpy(a_np.view(np.uint16)).to(torch.bfloat16)
    b_torch = torch.from_numpy(b_np.view(np.uint16)).to(torch.bfloat16)
    expected_torch = torch.from_numpy(expected_np.view(np.uint16)).to(torch.bfloat16)

    operator = AIEEchoV2(size=N, context=aie_context)

    input_buffers = {"input_a": a_torch, "input_b": b_torch}
    output_buffers = {"output": expected_torch}

    errors, latency_us, bandwidth_gbps = run_test(
        operator, input_buffers, output_buffers, rel_tol=0.0, abs_tol=0.0
    )

    print(f"\nv2 Latency (us): {latency_us:.1f}")
    assert not errors, f"Echo v2 failed: {errors}"
    print("Echo v2 PASSED")
