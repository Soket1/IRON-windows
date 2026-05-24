# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# MLIR design for v3 fused INT4 dequant batched GEMV.
# Extends v2: M_BATCH activation rows are processed together so the
# weight dequant cost is amortised across the batch.
#
# Buffer layout changes vs v2:
#   vector:  M_BATCH × K  bf16  (M_BATCH stacked activation rows)
#   output:  N × M_BATCH  bf16  (N weight rows × M_BATCH outputs each)
#
# Kernel signature: (m, row_offset, a, b, c) -- same as v2 externally.
# b is now M_BATCH×K and c is now m_output×M_BATCH per tile.

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


def my_fused_dequant_matvec_v3(
    dev, cols, M, K, m_input, m_batch, m_output=None, group_size=32
):
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
    assert m_batch >= 1 and m_batch <= 8

    dtype_in  = np.dtype[np.uint8]
    dtype_vec = np.dtype[bfloat16]
    dtype_out = np.dtype[bfloat16]

    dev_ty = NPU1() if dev == "npu" else NPU2()

    num_groups_per_row  = K // group_size
    packed_tile_bytes   = m_input * K // 2 + m_input * num_groups_per_row * 2

    rows_per_col    = M // cols
    tiles_per_col   = rows_per_col // m_input
    bytes_per_col   = tiles_per_col * packed_tile_bytes
    packed_total_bytes = cols * bytes_per_col

    # v3: b tile holds M_BATCH activation rows; c tile holds m_output × M_BATCH
    vec_tile_elems = m_batch * K           # activations: M_BATCH × K bf16
    out_tile_elems = m_output * m_batch    # outputs:     m_output × M_BATCH bf16

    L1_A_ty = np.ndarray[(packed_tile_bytes,), dtype_in]
    L1_B_ty = np.ndarray[(vec_tile_elems,),    dtype_vec]
    L1_C_ty = np.ndarray[(out_tile_elems,),    dtype_out]

    L3_A_ty = np.ndarray[(packed_total_bytes,), dtype_in]
    L3_B_ty = np.ndarray[(m_batch * K,),        dtype_vec]
    L3_C_ty = np.ndarray[(M * m_batch,),        dtype_out]

    kernel_obj = f"fused_dequant_gemv_v3_{K}k_g{group_size}_mb{m_batch}.o"
    fused_matvec = Kernel(
        "fused_dequant_matvec_v3_bf16",
        kernel_obj,
        [np.int32, np.int32, L1_A_ty, L1_B_ty, L1_C_ty],
    )

    A_L3L1_fifos = [
        ObjectFifo(L1_A_ty, name=f"A_L3L1_{i}", depth=2) for i in range(cols)
    ]
    B_L3L1_fifos = [
        ObjectFifo(L1_B_ty, name=f"B_L3L1_{i}", depth=1) for i in range(cols)
    ]
    C_L1L3_fifos = [
        ObjectFifo(L1_C_ty, name=f"C_L1L3_{i}", depth=2) for i in range(cols)
    ]

    N_div_n = tiles_per_col // (m_output // m_input)

    def core_body(A_L3L1_fifo, B_L3L1_fifo, C_L1L3_fifo, fused_matvec_fn):
        for _ in range_(0xFFFFFFFF):
            b = B_L3L1_fifo.acquire(1)
            for i_idx in range_(N_div_n):
                c = C_L1L3_fifo.acquire(1)
                for j_idx in range_(m_output // m_input):
                    j_i32 = index.casts(T.i32(), j_idx)
                    # row_offset × M_BATCH handled inside the kernel (c_out offset)
                    output_row_offset = j_i32 * m_input
                    a = A_L3L1_fifo.acquire(1)
                    fused_matvec_fn(m_input, output_row_offset, a, b, c)
                    A_L3L1_fifo.release(1)
                C_L1L3_fifo.release(1)
            B_L3L1_fifo.release(1)

    workers = [
        Worker(
            core_body,
            [
                A_L3L1_fifos[i].cons(),
                B_L3L1_fifos[i].cons(),
                C_L1L3_fifos[i].prod(),
                fused_matvec,
            ],
        )
        for i in range(cols)
    ]

    A_taps = [
        TensorAccessPattern(
            tensor_dims=(1, packed_total_bytes),
            offset=col * bytes_per_col,
            sizes=[1, 1, 1, bytes_per_col],
            strides=[0, 0, 0, 1],
        )
        for col in range(cols)
    ]

    # Output: each col owns rows_per_col weight rows, each producing M_BATCH outputs.
    # Flat output layout: [col0_row0_mb0, col0_row0_mb1, ..., col0_row1_mb0, ...]
    # Each col's block is rows_per_col × m_batch bf16 values.
    C_taps = [
        TensorAccessPattern(
            tensor_dims=(1, M * m_batch),
            offset=col * rows_per_col * m_batch,
            sizes=[1, 1, 1, rows_per_col * m_batch],
            strides=[0, 0, 0, 1],
        )
        for col in range(cols)
    ]

    rt = Runtime()
    with rt.sequence(L3_A_ty, L3_B_ty, L3_C_ty) as (A, B, C):
        rt.start(*workers)
        tg = rt.task_group()
        for i in range(cols):
            rt.fill(A_L3L1_fifos[i].prod(), A, A_taps[i], task_group=tg)
            rt.fill(B_L3L1_fifos[i].prod(), B, task_group=tg)
        for i in range(cols):
            rt.drain(
                C_L1L3_fifos[i].cons(), C, C_taps[i], task_group=tg, wait=True,
            )
        rt.finish_task_group(tg)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    p = argparse.ArgumentParser(prog="AIE Fused Dequant GEMV v3 Design")
    p.add_argument("--dev", choices=["npu", "npu2"], default="npu")
    p.add_argument("-M", type=int, required=True)
    p.add_argument("-K", type=int, required=True)
    p.add_argument("-m", type=int, required=True, dest="m_input")
    p.add_argument("--m-batch", type=int, default=4, dest="m_batch")
    p.add_argument("--m-output", type=int, default=None, dest="m_output")
    p.add_argument("--cols", type=int, required=True)
    p.add_argument("--group-size", type=int, default=32, dest="group_size")
    p.add_argument("-o", "--output-file-path", type=str)
    args = p.parse_args()
    mod = my_fused_dequant_matvec_v3(
        args.dev, args.cols, args.M, args.K, args.m_input,
        args.m_batch, args.m_output, args.group_size,
    )
    with open(args.output_file_path, "w") as f:
        f.write(str(mod))
