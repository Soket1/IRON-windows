# SPDX-License-Identifier: Apache-2.0
"""decode_attn_split — column-split variant of decode_front_attn.

Relocates the projection and the attention math onto DIFFERENT column groups to prove a
center<->edge on-chip relay: the Q-projection GEMV runs on the CENTER columns (cols
center_col_offset..), the attention math (flowkv score + value) runs on the EDGE columns
(cols edge_col_offset..), and Q flows center->edge over a plain on-chip ObjectFifo (IRON
routes the cross-column hop via stream switches / MemTile). Only worker PLACEMENT differs
from decode_front_attn; the bodies, kernels and DDR ABI are byte-identical.

  center (c,2): Q-GEMV + RoPE        -> Qi[c]   (relay center->edge)
  edge   (e,2): score                -> Ii[c]
  edge   (e,3): value                -> O_f[c]  (attn_out, DDR drain, OUTPUT-FIRST)

GQA via TEMPORAL BATCHING: num_batches = num_kv_heads // num_cols; the rt.sequence loops
the batches in ONE dispatch. Because everything but placement matches decode_front_attn,
a numeric PASS proves the center->edge Q relay + the edge score/value placement. RoPE stays
on the (now center) GEMV tile for parity; relocating it to the edge is a later refinement.

fuse_sv=True merges score+value onto ONE edge tile (K/V time-muxed on one fifo to fit the
2-S2MM cap). NOTE: this is CORRECT ONLY for a SINGLE chunk (seq <= chunk_size). With >1
chunk the per-chunk score->value interleave breaks the online-softmax (value does not
rescale its accumulator when the running max updates across chunks), so fuse_sv is a dead
end for growing context. Use the default 2-stage path for multi-chunk correctness.
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


def my_decode_attn_split(dev, embed_dim=2048, K_gemv=2048, head_dim=64, group_size=32,
                         attn_group=4, num_kv_heads=8, seq_len=32, chunk_size=None,
                         m_input=4, num_cols=2, center_col_offset=2, edge_col_offset=6,
                         fuse_sv=False):
    if chunk_size is None:
        chunk_size = 32 if seq_len % 32 == 0 else seq_len
    assert seq_len % chunk_size == 0, "seq_len must be divisible by chunk_size"
    dev_ty = NPU1() if dev == "npu" else NPU2()
    bf = np.dtype[bfloat16]
    u8 = np.dtype[np.uint8]
    # center cols [center_col_offset, +num_cols); edge cols [edge_col_offset, +num_cols)
    assert center_col_offset + num_cols <= 8 and edge_col_offset + num_cols <= 8
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
    xb_elems = K_gemv + q_rows + 16                     # [vector | lut | seq(pad16)]

    L1_A_ty   = np.ndarray[(packed_tile,), u8]
    L1_B_ty   = np.ndarray[(xb_elems,), bf]
    L1_Q_ty   = np.ndarray[(q_buf_elems,), bf]
    L1_K_ty   = np.ndarray[(chunk_size * head_dim,), bf]
    L1_I_ty   = np.ndarray[(inter_size,), bf]
    L1_V_ty   = np.ndarray[(chunk_size * head_dim,), bf]
    L1_O_ty   = np.ndarray[(attn_group * head_dim,), bf]

    DTYPE_SIZE = 2
    raw_head = seq_len * head_dim * DTYPE_SIZE
    ahs = int((raw_head + 63) / 64) * 64
    ahs_elems = ahs // DTYPE_SIZE

    NG = num_kv_heads
    L3_W_ty = np.ndarray[(NG * gemv_tiles * packed_tile,), u8]
    L3_X_ty = np.ndarray[(xb_elems,), bf]
    L3_K_ty = np.ndarray[(NG * ahs_elems,), bf]
    L3_V_ty = np.ndarray[(NG * ahs_elems,), bf]
    L3_O_ty = np.ndarray[(NG * attn_group * head_dim,), bf]

    # Kernels — identical to decode_front_attn.
    gemv = Kernel(
        "fused_dequant_matvec_v2_bf16",
        f"fused_dequant_gemv_v2_signed_{K_gemv}k_g{group_size}.o",
        [np.int32, np.int32, L1_A_ty, L1_B_ty, L1_Q_ty],
    )
    rope = Kernel("rope_bundled", "rope_il.o", [L1_Q_ty, L1_B_ty, L1_Q_ty, np.int32])
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
    if fuse_sv:
        # fused tile would need K+Q+V = 3 S2MM > 2 cap. K and V are the SAME shape
        # (chunk_size*head_dim), so time-mux them onto ONE fifo: fill K then V, the
        # fused body acquires it twice/chunk (K for score, V for value). 1 S2MM.
        KV_f = [ObjectFifo(L1_K_ty, name=f"KV_{c}", depth=4) for c in range(num_cols)]
    else:
        K_f = [ObjectFifo(L1_K_ty, name=f"K_{c}", depth=2) for c in range(num_cols)]
        V_f = [ObjectFifo(L1_V_ty, name=f"V_{c}", depth=2) for c in range(num_cols)]
    Qi  = [ObjectFifo(L1_Q_ty, name=f"Qi_{c}", depth=2) for c in range(num_cols)]  # center→edge relay
    if not fuse_sv:
        Ii = [ObjectFifo(L1_I_ty, name=f"Ii_{c}", depth=2) for c in range(num_cols)]
    else:
        # fused score+value on ONE edge tile: Ii is a tile-local scratch buffer
        # (score_chunk writes it, value_accum reads it — no inter-tile fifo).
        Ii_buf = [Buffer(type=L1_I_ty, initial_value=np.zeros(inter_size, dtype=bfloat16),
                         name=f"Ii_buf_{c}") for c in range(num_cols)]
    O_f = [ObjectFifo(L1_O_ty, name=f"O_{c}", depth=2) for c in range(num_cols)]

    workers = []
    for c in range(num_cols):
        cc = c + center_col_offset      # Q-GEMV column (center)
        ec = c + edge_col_offset        # attention column (edge)

        def gemv_body(af, bf_, qf, gemv_fn, rope_fn):
            for _ in range_(0xFFFFFFFF):
                b = bf_.acquire(1)
                q = qf.acquire(1)
                for j in range_(gemv_tiles):
                    a = af.acquire(1)
                    ro = index.casts(T.i32(), j) * m_input
                    gemv_fn(m_input, ro, a, b, q)
                    af.release(1)
                rope_fn(q, b, q, q_rows)
                qf.release(1)
                bf_.release(1)

        workers.append(Worker(
            gemv_body, [A_f[c].cons(), B_f[c].cons(), Qi[c].prod(), gemv, rope],
            placement=Tile(col=cc, row=2)))

        if not fuse_sv:
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
                placement=Tile(col=ec, row=2)))

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
                placement=Tile(col=ec, row=3)))
        else:
            # FUSED score+value on ONE edge tile (col ec, row 2). Ii is a tile-local
            # scratch buffer (no inter-tile fifo). Per chunk: score writes ii, value
            # reads it immediately (the standard flash-attention interleave).
            def attn_body(kvf, qf, of, ii, s_init_fn, s_rope_fn, s_chunk_fn,
                          v_init_fn, v_accum_fn, v_norm_fn):
                for _ in range_(0xFFFFFFFF):
                    s_init_fn(attn_group)
                    q = qf.acquire(1)
                    s_rope_fn(q, attn_group, head_dim)
                    v_init_fn(attn_group, head_dim)
                    for _ in range_(num_chunks):
                        k = kvf.acquire(1)          # K chunk (time-mux slot 1)
                        s_chunk_fn(q, k, ii, attn_group, head_dim, chunk_size)
                        kvf.release(1)
                        v = kvf.acquire(1)          # V chunk (time-mux slot 2)
                        v_accum_fn(ii, v, attn_group, head_dim, chunk_size)
                        kvf.release(1)
                    qf.release(1)
                    o = of.acquire(1)
                    v_norm_fn(o, attn_group, head_dim)
                    of.release(1)

            workers.append(Worker(
                attn_body, [KV_f[c].cons(), Qi[c].cons(), O_f[c].prod(),
                            Ii_buf[c], s_init, s_rope, s_chunk, v_init, v_accum, v_norm],
                placement=Tile(col=ec, row=2)))

    # Runtime: temporal-batched per-(batch,col) taps indexed by group g. OUTPUT-FIRST.
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
            tg1 = rt.task_group()
            for c in range(num_cols):
                g = b * num_cols + c
                rt.fill(A_f[c].prod(), w, a_tap(g), task_group=tg1)   # center shim
                rt.fill(B_f[c].prod(), x, x_tap, task_group=tg1)      # center shim
            rt.finish_task_group(tg1)
            tg2 = rt.task_group()
            for c in range(num_cols):
                g = b * num_cols + c
                if fuse_sv:
                    # time-mux K then V onto the SAME edge fifo (1 S2MM); the fused
                    # body acquires it twice/chunk (K for score, V for value).
                    rt.fill(KV_f[c].prod(), k, kv_tap(g), task_group=tg2)
                    rt.fill(KV_f[c].prod(), v, kv_tap(g), task_group=tg2)
                else:
                    rt.fill(K_f[c].prod(), k, kv_tap(g), task_group=tg2)  # edge shim
                    rt.fill(V_f[c].prod(), v, kv_tap(g), task_group=tg2)  # edge shim
            rt.finish_task_group(tg2)
            tg3 = rt.task_group()
            for c in range(num_cols):
                g = b * num_cols + c
                rt.drain(O_f[c].cons(), o, o_tap(g), task_group=tg3, wait=True)  # edge drain
            rt.finish_task_group(tg3)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())
