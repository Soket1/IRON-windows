# SPDX-License-Identifier: Apache-2.0
"""decode_front — functional QKV->attention fusion, step 1 (FFLM-style).

One worker on a central tile time-multiplexes two real stages sharing ONE
on-tile L1 buffer (no DDR bounce, no fifo shape-matching — the FFLM mega-kernel
pattern):

  stage 1 (INT4 GEMV, .o #1): produces Q activations (group_size*head_dim rows)
           by writing them into the first region of a flowkv Q buffer via the
           kernel's row_offset arg.
  stage 2 (bf16 attention, .o #2): flowkv_score_init + flowkv_score_rope_q
           apply RoPE to that same Q buffer IN PLACE, on the same tile.

This proves a GEMV output feeds an attention-preprocessing kernel on-chip within
one time-multiplexed tile, reusing the verified INT4-GEMV and flowkv kernels.
NOTE: RoPE angles live at Q[gs*hd : gs*hd+hd] and are NOT yet filled here (the
FFLM way is a passive on-tile RoPE LUT) — so the rotation runs but is not yet
numerically validated vs the oracle; that is step 2. Step 1 = the on-tile
GEMV->attention handoff mechanism, compiled and NPU-dispatched.

Placed on central column 2 (FFLM geometry).
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


def my_decode_front(dev, embed_dim=2048, K=2048, head_dim=64, group_size=32,
                    attn_group=4, m_input=2, col=2, row=2):
    """attn_group = q-heads per kv-head (flowkv group_size). K = GEMV contraction
    (embed_dim). group_size = INT4 quant group. The GEMV produces
    attn_group*head_dim Q rows (one KV group's worth)."""
    dev_ty = NPU1() if dev == "npu" else NPU2()

    q_rows = attn_group * head_dim                 # 256: Q activations produced
    q_buf_elems = attn_group * head_dim + head_dim + 2   # flowkv L1_Q_ty (322)
    groups = K // group_size
    packed_tile = m_input * K // 2 + m_input * groups * 2
    tiles = q_rows // m_input
    assert q_rows % m_input == 0

    L1_A_ty = np.ndarray[(packed_tile,), np.dtype[np.uint8]]
    L1_B_ty = np.ndarray[(K,), np.dtype[bfloat16]]
    L1_Q_ty = np.ndarray[(q_buf_elems,), np.dtype[bfloat16]]   # shared on-tile buf

    L3_w = np.ndarray[(tiles * packed_tile,), np.dtype[np.uint8]]
    L3_x = np.ndarray[(K,), np.dtype[bfloat16]]
    L3_o = np.ndarray[(q_buf_elems,), np.dtype[bfloat16]]

    gemv = Kernel(
        "fused_dequant_matvec_v2_bf16",
        f"fused_dequant_gemv_v2_{K}k_g{group_size}.o",
        [np.int32, np.int32, L1_A_ty, L1_B_ty, L1_Q_ty],   # writes into Q buf
    )
    score_init = Kernel("flowkv_score_init_bf16", f"flowkv_{head_dim}d.o", [np.int32])
    score_rope_q = Kernel(
        "flowkv_score_rope_q_bf16", f"flowkv_{head_dim}d.o",
        [L1_Q_ty, np.int32, np.int32],
    )

    A = ObjectFifo(L1_A_ty, name="A", depth=2)
    B = ObjectFifo(L1_B_ty, name="B", depth=1)
    Q = ObjectFifo(L1_Q_ty, name="Q", depth=2)

    def body(a_fifo, b_fifo, q_fifo, gemv_fn, score_init_fn, score_rope_q_fn):
        for _ in range_(0xFFFFFFFF):
            b = b_fifo.acquire(1)
            q = q_fifo.acquire(1)
            score_init_fn(attn_group)
            # stage 1: INT4 GEMV fills Q[0 : q_rows] (one m_input slice per call)
            for j in range_(tiles):
                a = a_fifo.acquire(1)
                row_off = index.casts(T.i32(), j) * m_input
                gemv_fn(m_input, row_off, a, b, q)
                a_fifo.release(1)
            # stage 2: attention RoPE on the same on-tile Q buffer
            score_rope_q_fn(q, attn_group, head_dim)
            q_fifo.release(1)
            b_fifo.release(1)

    worker = Worker(
        body,
        [A.cons(), B.cons(), Q.prod(), gemv, score_init, score_rope_q],
        placement=Tile(col=col, row=row),
    )

    # A streams `tiles` packed weight tiles through the depth-2 fifo.
    A_tap = TensorAccessPattern(
        tensor_dims=(1, tiles * packed_tile),
        offset=0,
        sizes=[1, 1, 1, tiles * packed_tile],
        strides=[0, 0, 0, 1],
    )

    rt = Runtime()
    with rt.sequence(L3_w, L3_x, L3_o) as (w, x, o):
        rt.start(worker)
        tg = rt.task_group()
        rt.fill(A.prod(), w, A_tap, task_group=tg)
        rt.fill(B.prod(), x, task_group=tg)
        rt.drain(Q.cons(), o, task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())
