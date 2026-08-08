# SPDX-License-Identifier: Apache-2.0
"""decode_front_attn — FRONT HALF of the 2-dispatch fused decode layer (#32 Option B).

One dispatch: Q-GEMV -> REAL interleaved RoPE(Q, all heads) -> flowkv decode-attention
-> attn_out[num_kv_heads*attn_group*head_dim] col-major (DDR). OUTPUT-FIRST (bo0). The
output layout [grp0_qheads | grp1 | ...] byte-matches decode_back_mono's ATTN input.

GQA via TEMPORAL BATCHING (the production flowkv scheme): num_cols central columns
(cols col_offset..col_offset+num_cols-1, default 2-5), attn_group = num_heads/num_kv_heads
query heads per column = ONE KV head per column. num_batches = num_kv_heads // num_cols;
the rt.sequence loops the batches (batch b, col c -> kv/q-head group b*num_cols+c), all in
ONE dispatch (the loop is compiled into the instruction stream). Cols 0/1 are unusable for
compute, so 4 cols × 2 batches covers 8 kv-heads correctly AND places.

RUNTIME LUT bundled in the tail of the activation BO X = [vector(K_gemv) | lut(q_rows)];
rope_bundled (rope.cc -DINTERLEAVED -DLUT_OFF=K_gemv) reads the LUT from b[LUT_OFF:], so the
GEMV vector and the LUT ride ONE shim S2MM (2-S2MM/col gemv-phase cap preserved).

  tile (c, 2): GEMV + RoPE → Qi[c]
  tile (c, 3): score → Ii[c]
  tile (c, 4): value → O_f[c] (DDR drain)
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


def my_decode_front_attn(dev, embed_dim=2048, K_gemv=2048, head_dim=64, group_size=32,
                         attn_group=4, num_kv_heads=8, seq_len=32, chunk_size=None,
                         m_input=4, num_cols=4, col_offset=2):
    if chunk_size is None:
        chunk_size = 32 if seq_len % 32 == 0 else seq_len   # cap L1 K/V fifo size
    assert seq_len % chunk_size == 0, "seq_len must be divisible by chunk_size"
    dev_ty = NPU1() if dev == "npu" else NPU2()
    bf = np.dtype[bfloat16]
    u8 = np.dtype[np.uint8]
    assert col_offset + num_cols <= 8, "columns overflow 8-wide partition"
    assert num_kv_heads % num_cols == 0, "num_kv_heads must be divisible by num_cols"

    num_batches = num_kv_heads // num_cols
    q_rows = attn_group * head_dim
    q_buf_elems = q_rows + head_dim + 2
    groups = K_gemv // group_size
    packed_tile = m_input * K_gemv // 2 + m_input * groups * 2
    gemv_tiles = q_rows // m_input
    assert q_rows % m_input == 0
    num_chunks = seq_len // chunk_size
    inter_size = chunk_size * attn_group + 2 * attn_group
    # vector + bundled LUT + actual_seq_len, padded so the shim DMA length stays
    # 4-byte (here 16-elem) aligned — an odd +1 element fails aie.dma_bd alignment.
    xb_elems = K_gemv + q_rows + 16                     # actual_seq_len at [K_gemv+q_rows]

    L1_A_ty   = np.ndarray[(packed_tile,), u8]
    L1_B_ty   = np.ndarray[(xb_elems,), bf]             # [vector | lut], one S2MM
    L1_Q_ty   = np.ndarray[(q_buf_elems,), bf]
    L1_K_ty   = np.ndarray[(chunk_size * head_dim,), bf]
    L1_I_ty   = np.ndarray[(inter_size,), bf]
    L1_V_ty   = np.ndarray[(chunk_size * head_dim,), bf]
    L1_O_ty   = np.ndarray[(attn_group * head_dim,), bf]

    DTYPE_SIZE = 2
    raw_head = seq_len * head_dim * DTYPE_SIZE
    ahs = int((raw_head + 63) / 64) * 64
    ahs_elems = ahs // DTYPE_SIZE

    # DDR layouts sized for num_kv_heads GROUPS (one per kv-head), indexed by
    # kv_head_idx = batch*num_cols + col.
    NG = num_kv_heads
    L3_W_ty = np.ndarray[(NG * gemv_tiles * packed_tile,), u8]
    L3_X_ty = np.ndarray[(xb_elems,), bf]               # shared [vector | lut]
    L3_K_ty = np.ndarray[(NG * ahs_elems,), bf]
    L3_V_ty = np.ndarray[(NG * ahs_elems,), bf]
    L3_O_ty = np.ndarray[(NG * attn_group * head_dim,), bf]

    # -------------------------------------------------------------------
    # Kernels — declared ONCE, shared across all column workers.
    # -------------------------------------------------------------------
    gemv = Kernel(
        "fused_dequant_matvec_v2_bf16",
        f"fused_dequant_gemv_v2_signed_{K_gemv}k_g{group_size}.o",
        [np.int32, np.int32, L1_A_ty, L1_B_ty, L1_Q_ty],
    )
    rope = Kernel("rope_bundled", "rope_il.o", [L1_Q_ty, L1_B_ty, L1_Q_ty, np.int32])
    fkv = f"flowkv_{head_dim}d_h{attn_group}_c{chunk_size}_r4.o"
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
    # depth=1 is REQUIRED, not a tuning choice: with depth=2 the score worker
    # runs a chunk ahead and the value tile reads a packet whose score body is
    # not yet visible, while its tail (correction/denom) already is. Measured
    # (#188): depth=2 leaves 3 of every 4 q-heads of the second temporal batch
    # exactly zero; depth=1 makes all 32 heads report canonical state. Padding
    # inter_size to 64B does NOT help, so this is ordering, not alignment.
    Ii  = [ObjectFifo(L1_I_ty, name=f"Ii_{c}", depth=1) for c in range(num_cols)]
    O_f = [ObjectFifo(L1_O_ty, name=f"O_{c}", depth=2) for c in range(num_cols)]

    workers = []
    for c in range(num_cols):
        pc = c + col_offset

        def gemv_body(af, bf_, qf, gemv_fn, rope_fn):
            for _ in range_(0xFFFFFFFF):
                b = bf_.acquire(1)                       # [vector | lut]
                q = qf.acquire(1)
                for j in range_(gemv_tiles):
                    a = af.acquire(1)
                    ro = index.casts(T.i32(), j) * m_input
                    gemv_fn(m_input, ro, a, b, q)        # reads b[0:K_gemv]
                    af.release(1)
                rope_fn(q, b, q, q_rows)                 # reads lut from b[LUT_OFF:]
                qf.release(1)
                bf_.release(1)

        workers.append(Worker(
            gemv_body, [A_f[c].cons(), B_f[c].cons(), Qi[c].prod(), gemv, rope],
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
    # Runtime: temporal-batched per-(batch,col) taps indexed by group g.
    # OUTPUT-FIRST: attn_out (o) is bo0.
    # -------------------------------------------------------------------
    def a_tap(g):
        return TensorAccessPattern(
            tensor_dims=(1, NG * gemv_tiles * packed_tile),
            offset=g * gemv_tiles * packed_tile,
            sizes=[1, 1, 1, gemv_tiles * packed_tile], strides=[0, 0, 0, 1])

    x_tap = TensorAccessPattern(
        tensor_dims=(1, xb_elems), offset=0,
        sizes=[1, 1, 1, xb_elems], strides=[0, 0, 0, 1])

    def kv_tap(g):
        return TensorAccessPattern(
            tensor_dims=(NG * ahs_elems,), offset=g * ahs_elems,
            sizes=[1, 1, 1, seq_len * head_dim], strides=[0, 0, 0, 1])

    def o_tap(g):
        return TensorAccessPattern(
            tensor_dims=(NG * attn_group * head_dim,),
            offset=g * attn_group * head_dim,
            sizes=[1, 1, 1, attn_group * head_dim], strides=[0, 0, 0, 1])

    rt = Runtime()
    with rt.sequence(L3_O_ty, L3_W_ty, L3_X_ty, L3_K_ty, L3_V_ty) as (o, w, x, k, v):
        rt.start(*workers)
        for b in range(num_batches):
            # phase 1: gemv inputs (A per-group + B=[vector|lut] broadcast) — 2 S2MM/col
            tg1 = rt.task_group()
            for c in range(num_cols):
                g = b * num_cols + c
                rt.fill(A_f[c].prod(), w, a_tap(g), task_group=tg1)
                rt.fill(B_f[c].prod(), x, x_tap, task_group=tg1)
            rt.finish_task_group(tg1)
            # phase 2: attention inputs (K + V per-group) — 2 S2MM/col
            tg2 = rt.task_group()
            for c in range(num_cols):
                g = b * num_cols + c
                rt.fill(K_f[c].prod(), k, kv_tap(g), task_group=tg2)
                rt.fill(V_f[c].prod(), v, kv_tap(g), task_group=tg2)
            rt.finish_task_group(tg2)
            # phase 3: drain outputs (1 MM2S/col) into bo0
            tg3 = rt.task_group()
            for c in range(num_cols):
                g = b * num_cols + c
                rt.drain(O_f[c].cons(), o, o_tap(g), task_group=tg3, wait=True)
            rt.finish_task_group(tg3)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())
