# SPDX-License-Identifier: Apache-2.0
"""decode_front — functional QKV->attention fusion (FFLM-style, on-tile).

One worker on a central tile time-multiplexes three real stages sharing ONE
on-tile L1 buffer (no DDR bounce, no fifo shape-matching):

  stage 1 (INT4 GEMV, .o #1): writes group_size*head_dim Q activation rows into
           the first region of a flowkv Q buffer (via the kernel row_offset arg).
  stage 2 (RoPE, .o #2): rope.cc (TWO_HALVES) rotates head 0 of that Q in place
           using a PASSIVE on-tile sin/cos LUT.
  stage 3 (attention, .o #3): flowkv score_init + score_rope_q copy the rotated Q
           to the static buffer + read seq_len.

KEY: the RoPE LUT is a passive `Buffer(initial_value=...)` placed on the SAME
tile — it is CDO-loaded once at xclbin load with ZERO runtime DMA, so it does NOT
consume a 3rd shim S2MM channel (the compute tile is capped at 2 input DMAs:
A-weights + B-vector). This is exactly FFLM's passive-LUT-tile technique.
[[reference_fflm_layer_geometry_fixed]]

Default LUT is identity (cos=1, sin=0) -> RoPE is a no-op passthrough, so the
output Q equals the GEMV output (lets us confirm the fused chain preserves data).
Pass a real interleaved cos/sin LUT to rotate. Placed on central column 2.
"""
import numpy as np
from ml_dtypes import bfloat16

import aie.dialects.index as index
from aie.dialects.aie import T
from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2, Tile


def my_decode_front(dev, embed_dim=2048, K=2048, head_dim=64, group_size=32,
                    attn_group=4, m_input=2, col=2, row=2, rope_lut=None):
    """attn_group = q-heads per kv-head (flowkv group_size). K = GEMV contraction
    (embed_dim). group_size = INT4 quant group. rope_lut = interleaved cos/sin
    bf16 array (head_dim elems); None => identity (cos=1,sin=0)."""
    dev_ty = NPU1() if dev == "npu" else NPU2()

    q_rows = attn_group * head_dim                 # 256: Q activations produced
    q_buf_elems = attn_group * head_dim + head_dim + 2   # flowkv L1_Q_ty (322)
    groups = K // group_size
    packed_tile = m_input * K // 2 + m_input * groups * 2
    tiles = q_rows // m_input
    assert q_rows % m_input == 0

    if rope_lut is None:
        rope_lut = np.zeros(head_dim, dtype=bfloat16)
        rope_lut[0::2] = bfloat16(1.0)             # cos = 1
        rope_lut[1::2] = bfloat16(0.0)             # sin = 0  -> identity rotation

    L1_A_ty = np.ndarray[(packed_tile,), np.dtype[np.uint8]]
    L1_B_ty = np.ndarray[(K,), np.dtype[bfloat16]]
    L1_Q_ty = np.ndarray[(q_buf_elems,), np.dtype[bfloat16]]   # shared on-tile buf
    L1_LUT_ty = np.ndarray[(head_dim,), np.dtype[bfloat16]]

    L3_w = np.ndarray[(tiles * packed_tile,), np.dtype[np.uint8]]
    L3_x = np.ndarray[(K,), np.dtype[bfloat16]]
    L3_o = np.ndarray[(q_buf_elems,), np.dtype[bfloat16]]

    gemv = Kernel(
        "fused_dequant_matvec_v2_bf16",
        f"fused_dequant_gemv_v2_{K}k_g{group_size}.o",
        [np.int32, np.int32, L1_A_ty, L1_B_ty, L1_Q_ty],   # writes into Q buf
    )
    rope = Kernel("rope", "rope_th.o", [L1_Q_ty, L1_LUT_ty, L1_Q_ty, np.int32])
    score_init = Kernel("flowkv_score_init_bf16", f"flowkv_{head_dim}d.o", [np.int32])
    score_rope_q = Kernel(
        "flowkv_score_rope_q_bf16", f"flowkv_{head_dim}d.o",
        [L1_Q_ty, np.int32, np.int32],
    )

    A = ObjectFifo(L1_A_ty, name="A", depth=2)
    B = ObjectFifo(L1_B_ty, name="B", depth=1)
    Q = ObjectFifo(L1_Q_ty, name="Q", depth=2)
    # PASSIVE on-tile RoPE LUT: CDO-loaded once, zero runtime DMA (no S2MM).
    # No explicit placement — it inherits the worker's tile (placing it here too
    # double-places it). CDO bakes initial_value into the xclbin.
    lut_buf = Buffer(type=L1_LUT_ty, initial_value=rope_lut, name="rope_lut")

    def body(a_fifo, b_fifo, q_fifo, lut, gemv_fn, rope_fn,
             score_init_fn, score_rope_q_fn):
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
            # stage 2: on-NPU RoPE rotates head 0 of the GEMV Q in place (passive LUT)
            rope_fn(q, lut, q, head_dim)
            # stage 3: attention reads the rotated Q (copy to static + seq_len)
            score_rope_q_fn(q, attn_group, head_dim)
            q_fifo.release(1)
            b_fifo.release(1)

    worker = Worker(
        body,
        [A.cons(), B.cons(), Q.prod(), lut_buf,
         gemv, rope, score_init, score_rope_q],
        placement=Tile(col=col, row=row),
    )

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
