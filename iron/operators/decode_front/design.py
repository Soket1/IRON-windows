# SPDX-License-Identifier: Apache-2.0
"""Minimal multi-.o link probe (Track A fusion prerequisite).

Question being retired: can ONE compute tile's core ELF link kernel functions
from TWO different .o objects at once — the INT4 GEMV kernel
(fused_dequant_gemv_v2.cc -> ..._2048k_g32.o) AND the bf16 attention kernel
(flowkv.cc -> flowkv_64d.o)? This is the prerequisite for FFLM-style
time-multiplex, where one central tile runs QKV then attention sequentially.

The worker is deliberately trivial: it runs one INT4 GEMV call and then one
flowkv score_init call, so the core must link BOTH objects. If AIECC produces
an xclbin, single-tile multi-.o linking works and the time-multiplex path is
viable. Placed on the central column 2 (FFLM geometry).
"""
import numpy as np
from ml_dtypes import bfloat16

from aie.helpers.dialects.scf import _for as range_
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2, Tile


def my_decode_front(dev, embed_dim=2048, K=2048, head_dim=64, group_size=32,
                    m_input=2, col=2, row=2):
    dev_ty = NPU1() if dev == "npu" else NPU2()
    groups = K // group_size
    packed_tile = m_input * K // 2 + m_input * groups * 2

    L1_A_ty = np.ndarray[(packed_tile,), np.dtype[np.uint8]]
    L1_B_ty = np.ndarray[(K,), np.dtype[bfloat16]]
    L1_C_ty = np.ndarray[(m_input,), np.dtype[bfloat16]]

    L3_w = np.ndarray[(packed_tile,), np.dtype[np.uint8]]
    L3_x = np.ndarray[(K,), np.dtype[bfloat16]]
    L3_o = np.ndarray[(m_input,), np.dtype[bfloat16]]

    # Kernel from object #1: INT4 dequant GEMV (-DDIM_K -DGROUP_SIZE baked).
    gemv = Kernel(
        "fused_dequant_matvec_v2_bf16",
        f"fused_dequant_gemv_v2_{K}k_g{group_size}.o",
        [np.int32, np.int32, L1_A_ty, L1_B_ty, L1_C_ty],
    )
    # Kernel from object #2: bf16 flowkv attention (-DHEAD_DIM baked).
    score_init = Kernel(
        "flowkv_score_init_bf16",
        f"flowkv_{head_dim}d.o",
        [np.int32],
    )

    A = ObjectFifo(L1_A_ty, name="A", depth=2)
    B = ObjectFifo(L1_B_ty, name="B", depth=1)
    C = ObjectFifo(L1_C_ty, name="C", depth=2)

    def body(a_fifo, b_fifo, c_fifo, gemv_fn, score_init_fn):
        for _ in range_(0xFFFFFFFF):
            b = b_fifo.acquire(1)
            c = c_fifo.acquire(1)
            a = a_fifo.acquire(1)
            gemv_fn(m_input, 0, a, b, c)          # INT4 GEMV (.o #1)
            score_init_fn(group_size)             # flowkv attn init (.o #2)
            a_fifo.release(1)
            c_fifo.release(1)
            b_fifo.release(1)

    worker = Worker(
        body,
        [A.cons(), B.cons(), C.prod(), gemv, score_init],
        placement=Tile(col=col, row=row),
    )

    rt = Runtime()
    with rt.sequence(L3_w, L3_x, L3_o) as (w, x, o):
        rt.start(worker)
        tg = rt.task_group()
        rt.fill(A.prod(), w, task_group=tg)
        rt.fill(B.prod(), x, task_group=tg)
        rt.drain(C.cons(), o, task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())
