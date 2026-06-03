# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal 4-column QKV INT4 GEMV (Track A, V3 spike).

Front stage of a fused decode layer. Q/K/V are STACKED into one GEMV with
M = embed_dim + 2*kv_dim (rows 0..embed = Q, then K, then V) and K = embed_dim,
so it is a single INT4 matrix-vector: one weight tensor, one phase, `cols`
workers each computing M/cols contiguous output rows. Output-stationary,
per-column weight fill + per-column output drain (the proven gemv /
post_attn_fused pattern; no MemTile join needed).

Weights: INT4, symmetric, group_size groups, our tile layout
  packed_tile = m_input*K/2 (int4) + m_input*(K/G)*2 (bf16 scales).
Kernel: fused_dequant_matvec_v2_bf16(m, row_offset, a_in[u8], b_in[bf16], c_out[bf16])
  built with -DGROUP_SIZE=<g> -DDIM_K=<embed_dim>.

DDR args (rt.sequence order):
  0: w_qkv — INT4 stacked QKV weights (cols * tiles_per_col * packed_tile)
  1: x     — input activation (embed_dim bf16), broadcast to all cols
  2: out   — stacked output [q (embed) | k (kv) | v (kv)] bf16
"""

import numpy as np
from ml_dtypes import bfloat16
import argparse

from aie.dialects.aie import T
import aie.dialects.index as index
from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2, Tile


def my_qkv_dense(dev, cols, embed_dim, kv_dim, group_size=32,
                 m_input=2, func_prefix="", col_offset=2, compute_row=2):
    M = embed_dim + 2 * kv_dim             # stacked Q|K|V output rows
    K = embed_dim                          # shared contraction dim
    assert M % cols == 0
    assert embed_dim % group_size == 0

    dev_ty = NPU1() if dev == "npu" else NPU2()
    # FFLM puts its 4-col GEMV array on the CENTRAL columns (2-5), not
    # left-packed 0-3 — cols 0/7 hold passive LUT/helper tiles. NPU2 compute
    # tiles span cols 0-7 x rows 2-5 (verified against device.get_compute_tiles).
    # col_offset shifts the array right; compute_row pins the core row.
    assert col_offset + cols <= 8, "compute array overflows 8 columns"
    assert compute_row in (2, 3, 4, 5), "NPU2 compute rows are 2-5"
    dtype_packed = np.dtype[np.uint8]
    dtype_vec    = np.dtype[bfloat16]

    groups = K // group_size
    packed_tile = m_input * K // 2 + m_input * groups * 2   # bytes per m_input-row tile
    rows_per_col  = M // cols
    tiles_per_col = rows_per_col // m_input
    assert rows_per_col % m_input == 0
    assert tiles_per_col <= 1023

    L1_A_ty = np.ndarray[(packed_tile,), dtype_packed]
    L1_B_ty = np.ndarray[(K,), dtype_vec]
    L1_C_ty = np.ndarray[(m_input,), dtype_vec]

    total_w_bytes = cols * tiles_per_col * packed_tile
    assert total_w_bytes % 2 == 0
    L3_w   = np.ndarray[(total_w_bytes // 2,), dtype_vec]
    L3_x   = np.ndarray[(embed_dim,), dtype_vec]
    L3_out = np.ndarray[(M,), dtype_vec]

    kobj = f"{func_prefix}qkv_dense_{embed_dim}k_g{group_size}.o"
    matvec = Kernel(
        f"{func_prefix}fused_dequant_matvec_v2_bf16", kobj,
        [np.int32, np.int32, L1_A_ty, L1_B_ty, L1_C_ty],
    )

    A_fifos = [ObjectFifo(L1_A_ty, name=f"A_{i}", depth=2) for i in range(cols)]
    B_fifos = [ObjectFifo(L1_B_ty, name=f"B_{i}", depth=1) for i in range(cols)]
    C_fifos = [ObjectFifo(L1_C_ty, name=f"C_{i}", depth=2) for i in range(cols)]

    def core_body(A_fifo, B_fifo, C_fifo, fn):
        for _ in range_(0xFFFFFFFF):
            b = B_fifo.acquire(1)
            for _ in range_(rows_per_col // m_input):
                c = C_fifo.acquire(1)
                a = A_fifo.acquire(1)
                fn(m_input, 0, a, b, c)
                A_fifo.release(1)
                C_fifo.release(1)
            B_fifo.release(1)

    workers = [Worker(core_body,
                      [A_fifos[i].cons(), B_fifos[i].cons(), C_fifos[i].prod(), matvec],
                      placement=Tile(col=i + col_offset, row=compute_row))
               for i in range(cols)]

    A_taps = [TensorAccessPattern(
                  tensor_dims=(1, total_w_bytes),
                  offset=i * tiles_per_col * packed_tile,
                  sizes=[1, 1, 1, tiles_per_col * packed_tile],
                  strides=[0, 0, 0, 1])
              for i in range(cols)]
    x_tap = TensorAccessPattern(tensor_dims=(1, embed_dim), offset=0,
                                sizes=[1, 1, 1, embed_dim], strides=[0, 0, 0, 1])
    C_taps = [TensorAccessPattern(
                  tensor_dims=(1, M),
                  offset=i * rows_per_col,
                  sizes=[1, 1, tiles_per_col, m_input],
                  strides=[0, 0, m_input, 1])
              for i in range(cols)]

    rt = Runtime()
    with rt.sequence(L3_w, L3_x, L3_out) as (w, x, out):
        rt.start(*workers)
        tg = rt.task_group()
        for i in range(cols):
            rt.fill(B_fifos[i].prod(), x, x_tap, task_group=tg)
            rt.fill(A_fifos[i].prod(), w, A_taps[i], task_group=tg)
        for i in range(cols):
            rt.drain(C_fifos[i].cons(), out, C_taps[i], task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Minimal 4-col stacked QKV INT4 GEMV")
    p.add_argument("--dev", default="npu2", choices=["npu", "npu2"])
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--embed-dim", type=int, default=2048)
    p.add_argument("--kv-dim", type=int, default=512)
    p.add_argument("--group-size", type=int, default=32)
    p.add_argument("--m-input", type=int, default=2)
    p.add_argument("-o", "--output-file-path", type=str)
    a = p.parse_args()
    module = my_qkv_dense(a.dev, a.cols, a.embed_dim, a.kv_dim,
                          a.group_size, a.m_input)
    if a.output_file_path:
        with open(a.output_file_path, "w") as f:
            f.write(str(module))
    else:
        print(module)
