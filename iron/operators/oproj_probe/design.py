# SPDX-License-Identifier: Apache-2.0
"""Standalone O_proj validation probe (decode-layer step toward #17).

O_proj is an INT4 E×E GEMV: full attn_out (E bf16) → o_out (E bf16), split
E/cols output rows per column. Uses layer_fused.cc's layer_fused_o_proj_bf16
(K=EMBED_DIM baked). Geometry: 4 central columns (cols 2-5), each computes its
E/4=512-row output slice; each reads the SAME full attn_out (E=2048),
broadcast to all columns.

Validated numerically vs a CPU INT4-dequant GEMV reference (symmetric, group_size
scale, -8 bias — matching the kernel's _qkv_gemv). attn_out→O is the FFLM
post-attention step; this proves the O_proj compute + E-dim weight feed on our
cols-2-5 geometry before wiring it into the fused chain.
"""
import numpy as np
from ml_dtypes import bfloat16

import aie.dialects.index as index
from aie.dialects.aie import T
from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2, Tile


def my_oproj(dev, embed_dim=2048, group_size=32, num_cols=4, m_input=2,
             col_offset=2):
    dev_ty = NPU1() if dev == "npu" else NPU2()
    bf = np.dtype[bfloat16]; u8 = np.dtype[np.uint8]
    E = embed_dim
    rows_per_col = E // num_cols                 # 512
    groups = E // group_size
    packed_tile = m_input * E // 2 + m_input * groups * 2
    tiles_per_col = rows_per_col // m_input
    assert rows_per_col % m_input == 0

    m_output = rows_per_col                      # one C buffer = whole col slice
    L1_A_ty = np.ndarray[(packed_tile,), u8]     # weight tile
    L1_B_ty = np.ndarray[(E,), bf]               # full attn_out (broadcast)
    L1_C_ty = np.ndarray[(m_output,), bf]        # whole-column output slice

    L3_W_ty = np.ndarray[(num_cols * tiles_per_col * packed_tile,), u8]
    L3_B_ty = np.ndarray[(E,), bf]
    L3_O_ty = np.ndarray[(E,), bf]

    # O_proj is an INT4 E×E GEMV — identical to fused_dequant_gemv_v2 (which
    # offsets c_out by row_offset, enabling a single whole-column C buffer).
    o_proj = Kernel(
        "fused_dequant_matvec_v2_bf16",
        f"fused_dequant_gemv_v2_{E}k_g{group_size}.o",
        [np.int32, np.int32, L1_A_ty, L1_B_ty, L1_C_ty],
    )

    A_f = [ObjectFifo(L1_A_ty, name=f"A_{c}", depth=2) for c in range(num_cols)]
    B_f = [ObjectFifo(L1_B_ty, name=f"B_{c}", depth=1) for c in range(num_cols)]
    C_f = [ObjectFifo(L1_C_ty, name=f"C_{c}", depth=2) for c in range(num_cols)]

    workers = []
    for c in range(num_cols):
        pc = c + col_offset

        def body(af, bf_, cf, oproj_fn):
            for _ in range_(0xFFFFFFFF):
                b = bf_.acquire(1)
                cc = cf.acquire(1)                 # one C buffer = whole col slice
                for j in range_(tiles_per_col):
                    a = af.acquire(1)
                    ro = index.casts(T.i32(), j) * m_input   # row offset INTO c
                    oproj_fn(m_input, ro, a, b, cc)
                    af.release(1)
                cf.release(1)
                bf_.release(1)

        workers.append(Worker(
            body, [A_f[c].cons(), B_f[c].cons(), C_f[c].prod(), o_proj],
            placement=Tile(col=pc, row=2)))

    def a_tap(c):
        return TensorAccessPattern(
            tensor_dims=(1, num_cols * tiles_per_col * packed_tile),
            offset=c * tiles_per_col * packed_tile,
            sizes=[1, 1, 1, tiles_per_col * packed_tile], strides=[0, 0, 0, 1])

    b_tap = TensorAccessPattern(
        tensor_dims=(1, E), offset=0, sizes=[1, 1, 1, E], strides=[0, 0, 0, 1])

    def c_tap(c):
        return TensorAccessPattern(
            tensor_dims=(1, E), offset=c * rows_per_col,
            sizes=[1, 1, tiles_per_col, m_input], strides=[0, 0, m_input, 1])

    rt = Runtime()
    with rt.sequence(L3_O_ty, L3_W_ty, L3_B_ty) as (o, w, b):
        rt.start(*workers)
        tg = rt.task_group()
        for c in range(num_cols):
            rt.fill(A_f[c].prod(), w, a_tap(c), task_group=tg)
            rt.fill(B_f[c].prod(), b, b_tap, task_group=tg)
        for c in range(num_cols):
            rt.drain(C_f[c].cons(), o, c_tap(c), task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())
