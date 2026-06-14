# SPDX-License-Identifier: Apache-2.0
"""decode_front_attn — FRONT HALF of the 2-dispatch fused decode layer (#32 Option B).

Forked from decode_front. One dispatch: Q-GEMV -> REAL interleaved RoPE(Q, all heads)
-> flowkv decode-attention -> attn_out[num_cols*attn_group*head_dim] col-major (DDR).
The output BO is OUTPUT-FIRST (bo0) so xclbin_replay can dump it, and its layout
[col0_512 | col1_512 | col2_512 | col3_512] byte-matches decode_back_mono's ATTN input.

Differences vs decode_front:
  * rope kernel = rope_il.o (-DINTERLEAVED) fed a REAL per-position cos/sin LUT
    (baked from `pos`, freq_base=10000, no freq_factors) instead of the identity LUT.
    LUT covers ALL attn_group heads (q_rows-long, the 64-elem head LUT tiled attn_group×),
    and rope rotates q_rows elements (all heads), not just head_dim (one head).
  * flowkv compiled with -DMAX_Q_HEADS=attn_group (h8) so attn_group=8 statics don't overflow.
  * output-first rt.sequence so the attn_out drain lands in bo0.

  tile (c, 2): GEMV + RoPE → Qi[c]
  tile (c, 3): score_init + score_rope_q(identity) + score_chunk → Ii[c]
  tile (c, 4): value_init + value_accum + value_normalize → O_f[c] (DDR drain)
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


def my_decode_front_attn(dev, embed_dim=2048, K_gemv=2048, head_dim=64, group_size=32,
                         attn_group=8, seq_len=32, chunk_size=None,
                         m_input=4, num_cols=4, col_offset=2, pos=0):
    if chunk_size is None:
        chunk_size = seq_len
    dev_ty = NPU1() if dev == "npu" else NPU2()
    bf = np.dtype[bfloat16]
    u8 = np.dtype[np.uint8]
    assert col_offset + num_cols <= 8, "columns overflow 8-wide partition"

    q_rows = attn_group * head_dim
    q_buf_elems = q_rows + head_dim + 2
    groups = K_gemv // group_size
    packed_tile = m_input * K_gemv // 2 + m_input * groups * 2
    gemv_tiles = q_rows // m_input
    assert q_rows % m_input == 0
    num_chunks = seq_len // chunk_size
    inter_size = chunk_size * attn_group + 2 * attn_group

    L1_A_ty   = np.ndarray[(packed_tile,), u8]
    L1_B_ty   = np.ndarray[(K_gemv,), bf]
    L1_Q_ty   = np.ndarray[(q_buf_elems,), bf]
    L1_LUT_ty = np.ndarray[(q_rows,), bf]                 # real interleaved LUT, all heads
    L1_K_ty   = np.ndarray[(chunk_size * head_dim,), bf]
    L1_I_ty   = np.ndarray[(inter_size,), bf]
    L1_V_ty   = np.ndarray[(chunk_size * head_dim,), bf]
    L1_O_ty   = np.ndarray[(attn_group * head_dim,), bf]

    DTYPE_SIZE = 2
    raw_head = seq_len * head_dim * DTYPE_SIZE
    ahs = int((raw_head + 63) / 64) * 64
    ahs_elems = ahs // DTYPE_SIZE

    # DDR layouts. A-weights: per-column contiguous. K/V: per-column aligned.
    L3_W_ty = np.ndarray[(num_cols * gemv_tiles * packed_tile,), u8]
    L3_X_ty = np.ndarray[(K_gemv,), bf]                       # shared input vector
    L3_K_ty = np.ndarray[(num_cols * ahs_elems,), bf]
    L3_V_ty = np.ndarray[(num_cols * ahs_elems,), bf]
    L3_O_ty = np.ndarray[(num_cols * attn_group * head_dim,), bf]

    # -------------------------------------------------------------------
    # Kernels — declared ONCE, shared across all column workers (the flowkv
    # pattern; declaring per-column redefines the symbol → AIECC error).
    # -------------------------------------------------------------------
    gemv = Kernel(
        "fused_dequant_matvec_v2_bf16",
        f"fused_dequant_gemv_v2_{K_gemv}k_g{group_size}.o",
        [np.int32, np.int32, L1_A_ty, L1_B_ty, L1_Q_ty],
    )
    rope = Kernel("rope", "rope_il.o", [L1_Q_ty, L1_LUT_ty, L1_Q_ty, np.int32])
    fkv = f"flowkv_{head_dim}d_h{attn_group}.o"
    s_init  = Kernel("flowkv_score_init_bf16",      fkv, [np.int32])
    s_rope  = Kernel("flowkv_score_rope_q_bf16",    fkv, [L1_Q_ty, np.int32, np.int32])
    s_chunk = Kernel("flowkv_score_chunk_bf16",     fkv,
                     [L1_Q_ty, L1_K_ty, L1_I_ty, np.int32, np.int32, np.int32])
    v_init  = Kernel("flowkv_value_init_bf16",      fkv, [np.int32, np.int32])
    v_accum = Kernel("flowkv_value_accum_bf16",     fkv,
                     [L1_I_ty, L1_V_ty, np.int32, np.int32, np.int32])
    v_norm  = Kernel("flowkv_value_normalize_bf16", fkv, [L1_O_ty, np.int32, np.int32])

    A_f = [ObjectFifo(L1_A_ty, name=f"A_{c}", depth=2) for c in range(num_cols)]
    B_f = [ObjectFifo(L1_B_ty, name=f"B_{c}", depth=1) for c in range(num_cols)]
    K_f = [ObjectFifo(L1_K_ty, name=f"K_{c}", depth=2) for c in range(num_cols)]
    V_f = [ObjectFifo(L1_V_ty, name=f"V_{c}", depth=2) for c in range(num_cols)]
    Qi  = [ObjectFifo(L1_Q_ty, name=f"Qi_{c}", depth=2) for c in range(num_cols)]
    Ii  = [ObjectFifo(L1_I_ty, name=f"Ii_{c}", depth=2) for c in range(num_cols)]
    O_f = [ObjectFifo(L1_O_ty, name=f"O_{c}", depth=2) for c in range(num_cols)]

    # REAL interleaved RoPE LUT for token `pos` (freq_base=10000, no freq_factors),
    # head LUT [cos0,sin0,...,cos31,sin31] tiled attn_group× to cover all q-heads.
    theta = 10000.0
    lut_head = np.zeros(head_dim, dtype=np.float32)
    for i in range(head_dim // 2):
        ang = pos * (theta ** (-2.0 * i / head_dim))
        lut_head[2 * i] = np.cos(ang)
        lut_head[2 * i + 1] = np.sin(ang)
    rope_lut_data = np.tile(lut_head, attn_group).astype(bfloat16)
    luts = [Buffer(type=L1_LUT_ty, initial_value=rope_lut_data, name=f"rope_lut_{c}")
            for c in range(num_cols)]

    workers = []
    for c in range(num_cols):
        pc = c + col_offset

        def gemv_body(af, bf_, qf, lut, gemv_fn, rope_fn):
            for _ in range_(0xFFFFFFFF):
                b = bf_.acquire(1)
                q = qf.acquire(1)
                for j in range_(gemv_tiles):
                    a = af.acquire(1)
                    ro = index.casts(T.i32(), j) * m_input
                    gemv_fn(m_input, ro, a, b, q)
                    af.release(1)
                rope_fn(q, lut, q, q_rows)
                qf.release(1)
                bf_.release(1)

        workers.append(Worker(
            gemv_body, [A_f[c].cons(), B_f[c].cons(), Qi[c].prod(), luts[c],
                        gemv, rope],
            placement=Tile(col=pc, row=2)))

        def score_body(kf, qf, inf, init_fn, rope_fn, chunk_fn):
            for _ in range_(0xFFFFFFFF):
                init_fn(attn_group)
                q = qf.acquire(1)
                rope_fn(q, attn_group, head_dim)
                for _ in range_(num_chunks):
                    k = kf.acquire(1)
                    it = inf.acquire(1)
                    chunk_fn(q, k, it, attn_group, head_dim, chunk_size)
                    kf.release(1)
                    inf.release(1)
                qf.release(1)

        workers.append(Worker(
            score_body, [K_f[c].cons(), Qi[c].cons(), Ii[c].prod(),
                         s_init, s_rope, s_chunk],
            placement=Tile(col=pc, row=3)))

        def value_body(vf, inf, of, init_fn, accum_fn, norm_fn):
            for _ in range_(0xFFFFFFFF):
                init_fn(attn_group, head_dim)
                for _ in range_(num_chunks):
                    it = inf.acquire(1)
                    v = vf.acquire(1)
                    accum_fn(it, v, attn_group, head_dim, chunk_size)
                    inf.release(1)
                    vf.release(1)
                o = of.acquire(1)
                norm_fn(o, attn_group, head_dim)
                of.release(1)

        workers.append(Worker(
            value_body, [V_f[c].cons(), Ii[c].cons(), O_f[c].prod(),
                         v_init, v_accum, v_norm],
            placement=Tile(col=pc, row=4)))

    # -------------------------------------------------------------------
    # Runtime: per-column taps, DMA time-multiplexed across task_groups.
    # OUTPUT-FIRST: attn_out (o) is bo0 so xclbin_replay can dump it.
    # -------------------------------------------------------------------
    def a_tap(c):
        return TensorAccessPattern(
            tensor_dims=(1, num_cols * gemv_tiles * packed_tile),
            offset=c * gemv_tiles * packed_tile,
            sizes=[1, 1, 1, gemv_tiles * packed_tile], strides=[0, 0, 0, 1])

    x_tap = TensorAccessPattern(
        tensor_dims=(1, K_gemv), offset=0,
        sizes=[1, 1, 1, K_gemv], strides=[0, 0, 0, 1])

    def kv_tap(c):
        return TensorAccessPattern(
            tensor_dims=(num_cols * ahs_elems,), offset=c * ahs_elems,
            sizes=[1, 1, 1, seq_len * head_dim], strides=[0, 0, 0, 1])

    def o_tap(c):
        return TensorAccessPattern(
            tensor_dims=(num_cols * attn_group * head_dim,),
            offset=c * attn_group * head_dim,
            sizes=[1, 1, 1, attn_group * head_dim], strides=[0, 0, 0, 1])

    rt = Runtime()
    with rt.sequence(L3_O_ty, L3_W_ty, L3_X_ty, L3_K_ty, L3_V_ty) as (o, w, x, k, v):
        rt.start(*workers)
        # phase 1: gemv inputs (A per-col + B broadcast) — 2 S2MM per col
        tg1 = rt.task_group()
        for c in range(num_cols):
            rt.fill(A_f[c].prod(), w, a_tap(c), task_group=tg1)
            rt.fill(B_f[c].prod(), x, x_tap, task_group=tg1)
        rt.finish_task_group(tg1)
        # phase 2: attention inputs (K + V per-col) — 2 S2MM per col
        tg2 = rt.task_group()
        for c in range(num_cols):
            rt.fill(K_f[c].prod(), k, kv_tap(c), task_group=tg2)
            rt.fill(V_f[c].prod(), v, kv_tap(c), task_group=tg2)
        rt.finish_task_group(tg2)
        # phase 3: drain outputs (1 MM2S per col) into bo0
        tg3 = rt.task_group()
        for c in range(num_cols):
            rt.drain(O_f[c].cons(), o, o_tap(c), task_group=tg3, wait=True)
        rt.finish_task_group(tg3)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())
