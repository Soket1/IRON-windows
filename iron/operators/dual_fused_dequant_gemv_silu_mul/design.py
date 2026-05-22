# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from pathlib import Path
from ml_dtypes import bfloat16
import argparse

import aie.dialects.index as index
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.scf import _for as range_
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2

"""
Dual fused-INT4-dequant matrix-vector + SiLU + elementwise multiply design.

Computes: output = silu(dequant(W1_int4) @ x) * (dequant(W2_int4) @ x)

Mirror of dual_gemv_silu_mul/design.py but the L1 weight tile carries an
INT4-packed buffer (weights then bf16 scales, same layout as
fused_dequant_gemv). The kernel's two-phase static-buffer trick is
unchanged: phase=0 writes to left_buf (gate), phase=1 writes to
right_buf (up), then silu_mul reads both and emits c.

DDR layout (per column):
  [tiles_per_col * packed_tile_bytes]  packed W1 (gate) tiles
  [tiles_per_col * packed_tile_bytes]  packed W2 (up)   tiles

Tiles for column 0 come first, then column 1, etc. The same
``quantize_and_pack`` used by fused_dequant_gemv produces each weight's
per-column packed bytes -- the operator just concatenates gate+up per
column before writing to L3.
"""


def my_dual_fused_dequant_gemv_silu_mul(dev, cols, M, K, m_input,
                                         m_output=None, group_size=32):
    if m_output is None:
        m_output = m_input

    assert m_output % m_input == 0 and m_output >= m_input
    assert m_output <= M // cols
    assert (M // cols) % m_output == 0
    assert m_input <= M // cols
    assert (M // cols) % m_input == 0
    assert K % group_size == 0
    assert group_size % 32 == 0
    assert M % cols == 0

    dtype_packed = np.dtype[np.uint8]
    dtype_vec    = np.dtype[bfloat16]
    dtype_out    = np.dtype[bfloat16]

    dev_ty = NPU1() if dev == "npu" else NPU2()

    num_groups_per_row = K // group_size
    packed_tile_bytes = m_input * K // 2 + m_input * num_groups_per_row * 2

    rows_per_col = M // cols
    tiles_per_col = rows_per_col // m_input
    bytes_per_col_per_weight = tiles_per_col * packed_tile_bytes
    bytes_per_col            = 2 * bytes_per_col_per_weight   # gate + up
    packed_total_bytes       = cols * bytes_per_col

    L1_A_ty = np.ndarray[(packed_tile_bytes,), dtype_packed]
    L1_B_ty = np.ndarray[(K,), dtype_vec]
    L1_C_ty = np.ndarray[(m_output,), dtype_out]

    L3_W_ty = np.ndarray[(packed_total_bytes,), dtype_packed]
    L3_B_ty = np.ndarray[(K,), dtype_vec]
    L3_C_ty = np.ndarray[(M,), dtype_out]

    # V2 kernel: K and group_size baked in via -D flags, dropped from
    # the runtime arg list. Per-shape kernel object name matches op.py.
    kernel_obj = f"dual_fused_dequant_gemv_silu_mul_{K}k_g{group_size}.o"

    # Dequant-GEMV writing to left_buf (phase=0) or right_buf (phase=1)
    matvec = Kernel(
        "dual_fused_dequant_gemv_bf16",
        kernel_obj,
        [np.int32, np.int32, L1_A_ty, L1_B_ty, np.int32],
    )

    # SiLU+Mul reads static buffers, writes C FIFO
    silu_mul_fn = Kernel(
        "dual_fused_dequant_gemv_silu_mul_bf16",
        kernel_obj,
        [L1_C_ty, np.int32],
    )

    A_fifos = [ObjectFifo(L1_A_ty, name=f"A_{i}", depth=2) for i in range(cols)]
    B_fifos = [ObjectFifo(L1_B_ty, name=f"B_{i}", depth=1) for i in range(cols)]
    C_fifos = [ObjectFifo(L1_C_ty, name=f"C_{i}", depth=2) for i in range(cols)]

    def core_body(A_fifo, B_fifo, C_fifo, matvec_fn, silu_mul):
        for _ in range_(0xFFFFFFFF):
            b = B_fifo.acquire(1)
            for i_idx in range_(M // m_output // cols):
                # Phase 0: W1 (gate) rows -> left_buf
                for j_idx in range_(m_output // m_input):
                    j_i32 = index.casts(T.i32(), j_idx)
                    row_offset = j_i32 * m_input
                    a = A_fifo.acquire(1)
                    matvec_fn(m_input, row_offset, a, b, 0)
                    A_fifo.release(1)
                # Phase 1: W2 (up) rows -> right_buf
                for j_idx in range_(m_output // m_input):
                    j_i32 = index.casts(T.i32(), j_idx)
                    row_offset = j_i32 * m_input
                    a = A_fifo.acquire(1)
                    matvec_fn(m_input, row_offset, a, b, 1)
                    A_fifo.release(1)
                # Phase 2: silu(left_buf) * right_buf -> output
                c = C_fifo.acquire(1)
                silu_mul(c, m_output)
                C_fifo.release(1)
            B_fifo.release(1)

    workers = [
        Worker(
            core_body,
            [
                A_fifos[i].cons(),
                B_fifos[i].cons(),
                C_fifos[i].prod(),
                matvec,
                silu_mul_fn,
            ],
        )
        for i in range(cols)
    ]

    # Each column's region is bytes_per_col contiguous bytes (gate tiles
    # then up tiles for that column). The kernel's two-phase loop
    # consumes them in order via the A FIFO.
    A_taps = [
        TensorAccessPattern(
            tensor_dims=(1, packed_total_bytes),
            offset=col * bytes_per_col,
            sizes=[1, 1, 1, bytes_per_col],
            strides=[0, 0, 0, 1],
        )
        for col in range(cols)
    ]

    C_taps = [
        TensorAccessPattern(
            tensor_dims=(1, M),
            offset=col * rows_per_col,
            sizes=[1, 1, 1, rows_per_col],
            strides=[0, 0, 0, 1],
        )
        for col in range(cols)
    ]

    rt = Runtime()
    with rt.sequence(L3_W_ty, L3_B_ty, L3_C_ty) as (W, B, C):
        rt.start(*workers)
        tg = rt.task_group()
        for i in range(cols):
            rt.fill(A_fifos[i].prod(), W, A_taps[i], task_group=tg)
            rt.fill(B_fifos[i].prod(), B, task_group=tg)
        for i in range(cols):
            rt.drain(C_fifos[i].cons(), C, C_taps[i], task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        prog="AIE Dual fused-dequant GEMV + SiLU + Mul Design",
    )
    argparser.add_argument("--dev", type=str, choices=["npu", "npu2"], default="npu")
    argparser.add_argument("-M", type=int, required=True)
    argparser.add_argument("-K", type=int, required=True)
    argparser.add_argument("-m", type=int, required=True, dest="m_input")
    argparser.add_argument("--m-output", type=int, default=None, dest="m_output")
    argparser.add_argument("--cols", type=int, required=True)
    argparser.add_argument("--group-size", type=int, default=32, dest="group_size")
    argparser.add_argument(
        "--output-file-path",
        "-o",
        type=str,
        help="Output file path for the generated MLIR module",
    )
    args = argparser.parse_args()
    module = my_dual_fused_dequant_gemv_silu_mul(
        args.dev,
        args.cols,
        args.M,
        args.K,
        args.m_input,
        args.m_output,
        args.group_size,
    )

    output_file_path = Path(args.output_file_path)
    with open(output_file_path, "w") as f:
        f.write(str(module))
