# SPDX-License-Identifier: Apache-2.0
"""decode_attn_spatial — spatial 8-head 2-stage attention block.

All num_kv_heads attend in PARALLEL (no temporal batching): one (score, value) tile
pair per head on the EDGE columns, with the Q-projection GEMV on the CENTER columns and
Q relayed center->edge per head. attn_out is gathered by a 2-level join (a single
num_heads-way join would need num_heads S2MM > the 6-channel MemTile cap) and drained as
two halves to DDR. This is the attention block of the projection-on-center /
attention-on-edge layer; correctness here proves the spatial placement + the Q relay +
the 2-level output join.

  center (2+h%4, 2+h//4): Q-GEMV + RoPE        -> Qi[h]   (relay center->edge)
  edge   (ec, sr):        score                -> Ii[h]
  edge   (ec, sr+1):      value                -> O_f[h]
  edge MemTiles:          O_f[0..3] -> Half0,  O_f[4..7] -> Half1  (4-way each)
  drain:                  Half0 -> out[0:H/2], Half1 -> out[H/2:H]

The (score, value) stages stay SEPARATE (2-stage) so the online-softmax is correct for
multi-chunk context; a fused single-tile variant only holds for a single chunk.
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

# edge columns used for the attention tiles, in head order (2 heads per column:
# rows 2-3 then rows 4-5). 4 columns x 4 rows = 16 tiles = 8 (score,value) pairs.
EDGE_COLS = [0, 1, 6, 7]


def my_decode_attn_oproj(dev, embed_dim=2048, K_gemv=2048, head_dim=64, group_size=32,
                         attn_group=4, num_kv_heads=8, seq_len=32, chunk_size=None,
                         m_input=4, center_col_offset=2):
    if chunk_size is None:
        chunk_size = 32 if seq_len % 32 == 0 else seq_len
    assert seq_len % chunk_size == 0, "seq_len must be divisible by chunk_size"
    dev_ty = NPU1() if dev == "npu" else NPU2()
    bf = np.dtype[bfloat16]
    u8 = np.dtype[np.uint8]
    NH = num_kv_heads
    assert NH == 8, "spatial layout maps 8 heads onto 4 edge cols x 2 (rows 2-3/4-5)"

    q_rows = attn_group * head_dim
    q_buf_elems = q_rows + head_dim + 2
    groups = K_gemv // group_size
    packed_tile = m_input * K_gemv // 2 + m_input * groups * 2
    gemv_tiles = q_rows // m_input
    assert q_rows % m_input == 0
    num_chunks = seq_len // chunk_size
    inter_size = chunk_size * attn_group + 2 * attn_group
    xb_elems = K_gemv + q_rows + 16
    E = NH * q_rows                                      # attn_out / O width = 2048
    o_tiles = E // m_input                               # O-proj weight tiles (K=E)
    o_packed = packed_tile                               # K=E == K_gemv so same packing

    L1_A_ty   = np.ndarray[(packed_tile,), u8]
    L1_B_ty   = np.ndarray[(xb_elems,), bf]
    L1_Q_ty   = np.ndarray[(q_buf_elems,), bf]
    L1_K_ty   = np.ndarray[(chunk_size * head_dim,), bf]
    L1_I_ty   = np.ndarray[(inter_size,), bf]
    L1_V_ty   = np.ndarray[(chunk_size * head_dim,), bf]
    L1_O_ty   = np.ndarray[(q_rows,), bf]
    L1_HALF_ty = np.ndarray[((NH // 2) * q_rows,), bf]   # 4 heads = 1024
    L1_E_ty   = np.ndarray[(E,), bf]                     # full attn_out / O
    L1_OW_ty  = np.ndarray[(o_packed,), u8]              # one O-proj weight tile

    DTYPE_SIZE = 2
    raw_head = seq_len * head_dim * DTYPE_SIZE
    ahs = int((raw_head + 63) / 64) * 64
    ahs_elems = ahs // DTYPE_SIZE

    # weights BO = [A region (8 heads' Q-proj) | Wo region (O-proj, K=E)]
    a_region = NH * gemv_tiles * packed_tile
    ow_region = o_tiles * o_packed
    L3_W_ty = np.ndarray[(a_region + ow_region,), u8]
    L3_X_ty = np.ndarray[(xb_elems,), bf]
    L3_K_ty = np.ndarray[(NH * ahs_elems,), bf]
    L3_V_ty = np.ndarray[(NH * ahs_elems,), bf]
    L3_O_ty = np.ndarray[(E,), bf]                       # O = Wo @ attn_out

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
    # O-proj = signed v2 GEMV, K=E, DISTINCT symbol so its c_out can be typed E (2048)
    # instead of the Q-buffer (322). attn_concat2 reassembles the two join halves -> E.
    oproj = Kernel("oproj_matvec_v2_bf16",
                   f"fused_dequant_gemv_v2_oproj_signed_{K_gemv}k_g{group_size}.o",
                   [np.int32, np.int32, L1_OW_ty, L1_E_ty, L1_E_ty])
    concat2 = Kernel("attn_concat2_bf16", "attn_concat.o",
                     [L1_HALF_ty, L1_HALF_ty, L1_E_ty, np.int32])

    # Merged DELIVERY (8 heads simultaneously would need A8+B8+K8+V8 = 32 shim input
    # streams > 16 shim MM2S channels). Reduce to 14: A stays per-head (8, on the center
    # col shims), B is BROADCAST (one x forwarded to 4 GEMV tiles per group, 2 groups so
    # the 8-way fan-out stays <= the 6-MM2S MemTile cap), K/V are ONE shim fill per 4-head
    # group .split() to per-head sub-fifos on an edge MemTile (16 -> 4).
    L1_KGRP_ty = np.ndarray[(4 * ahs_elems,), bf]
    A_f = [ObjectFifo(L1_A_ty, name=f"A_{h}", depth=2) for h in range(NH)]
    B_src = [ObjectFifo(L1_B_ty, name=f"Bsrc_{g}", depth=1) for g in range(2)]
    B_bc = [B_src[g].cons().forward(name=f"Bbc_{g}", depth=1,
                                    placement=Tile(col=center_col_offset + 1 + g, row=1))
            for g in range(2)]
    K_grp = [ObjectFifo(L1_KGRP_ty, name=f"Kgrp_{g}", depth=2) for g in range(2)]
    K_parts = [K_grp[g].cons().split(
                   offsets=[i * ahs_elems for i in range(4)],
                   placement=Tile(col=EDGE_COLS[g * 2], row=1),
                   obj_types=[L1_K_ty for _ in range(4)]) for g in range(2)]
    V_grp = [ObjectFifo(L1_KGRP_ty, name=f"Vgrp_{g}", depth=2) for g in range(2)]
    V_parts = [V_grp[g].cons().split(
                   offsets=[i * ahs_elems for i in range(4)],
                   placement=Tile(col=EDGE_COLS[g * 2 + 1], row=1),
                   obj_types=[L1_V_ty for _ in range(4)]) for g in range(2)]
    Qi  = [ObjectFifo(L1_Q_ty, name=f"Qi_{h}", depth=2) for h in range(NH)]
    Ii  = [ObjectFifo(L1_I_ty, name=f"Ii_{h}", depth=2) for h in range(NH)]
    O_f = [ObjectFifo(L1_O_ty, name=f"O_{h}", depth=2) for h in range(NH)]

    # 2-level attn_out gather: two 4-way joins (each <= 6 S2MM cap) on edge MemTiles.
    # joins on FREE center MemTiles (cols 2-5 row1; center Q-GEMV uses rows 2-3 only),
    # NOT on the contested edge cols 0/7 (those carry K/V fills + attention tiles).
    Half0 = ObjectFifo(L1_HALF_ty, name="Half0", depth=2)
    H0_parts = Half0.prod().join(
        offsets=[i * q_rows for i in range(NH // 2)],
        placement=Tile(col=center_col_offset, row=1),
        obj_types=[L1_O_ty for _ in range(NH // 2)])
    Half1 = ObjectFifo(L1_HALF_ty, name="Half1", depth=2)
    H1_parts = Half1.prod().join(
        offsets=[i * q_rows for i in range(NH // 2)],
        placement=Tile(col=center_col_offset + 3, row=1),
        obj_types=[L1_O_ty for _ in range(NH // 2)])

    # attn_out relay: a center tile concats Half0|Half1 -> AttnE (one fifo), so the
    # O-proj tile reads ONE activation input + its weights within the 2-S2MM cap.
    AttnE = ObjectFifo(L1_E_ty, name="AttnE", depth=2)
    # O-proj weights: one shim fill split per-tile (single O-proj tile here -> no split,
    # plain per-tile fifo, IRON re-tiles the contiguous Wo region by o_packed elements).
    OW_f = ObjectFifo(L1_OW_ty, name="OW", depth=2)
    O_out = ObjectFifo(L1_E_ty, name="O_out", depth=2)

    workers = []
    for h in range(NH):
        cc = center_col_offset + (h % 4)        # center Q-GEMV column
        cr = 2 + (h // 4)                        # center row (2 or 3)
        ec = EDGE_COLS[h // 2]                   # edge attention column
        sr = 2 + (h % 2) * 2                     # score row (2 or 4)

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
            gemv_body, [A_f[h].cons(), B_bc[h // 4].cons(), Qi[h].prod(), gemv, rope],
            placement=Tile(col=cc, row=cr)))

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
            score_body, [K_parts[h // 4][h % 4].cons(), Qi[h].cons(), Ii[h].prod(),
                         s_init, s_rope, s_chunk],
            placement=Tile(col=ec, row=sr)))

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

        of_prod = (H0_parts if h < NH // 2 else H1_parts)[h % (NH // 2)].prod()
        workers.append(Worker(
            value_body, [V_parts[h // 4][h % 4].cons(), Ii[h].cons(), of_prod,
                         v_init, v_accum, v_norm],
            placement=Tile(col=ec, row=sr + 1)))

    # relay: concat the two attn_out halves -> AttnE (center tile, row 4 free since
    # the Q-GEMV workers use center rows 2,3 only).
    half = (NH // 2) * q_rows
    def relay_body(h0, h1, ae, concat_fn):
        for _ in range_(0xFFFFFFFF):
            a = h0.acquire(1); b = h1.acquire(1); e = ae.acquire(1)
            concat_fn(a, b, e, half)
            h0.release(1); h1.release(1); ae.release(1)
    workers.append(Worker(
        relay_body, [Half0.cons(), Half1.cons(), AttnE.prod(), concat2],
        placement=Tile(col=center_col_offset, row=4)))

    # O-proj: signed v2 GEMV K=E, input = AttnE (full attn_out), weight = Wo tiles.
    # c_out is the full E output (each call writes m rows at row_offset).
    def oproj_body(ow, ae, oo, oproj_fn):
        for _ in range_(0xFFFFFFFF):
            a_full = ae.acquire(1)
            o = oo.acquire(1)
            for j in range_(o_tiles):
                w = ow.acquire(1)
                oproj_fn(m_input, index.casts(T.i32(), j) * m_input, w, a_full, o)
                ow.release(1)
            ae.release(1); oo.release(1)
    workers.append(Worker(
        oproj_body, [OW_f.cons(), AttnE.cons(), O_out.prod(), oproj],
        placement=Tile(col=center_col_offset + 1, row=4)))

    def a_tap(h):
        return TensorAccessPattern(
            tensor_dims=(1, NH * gemv_tiles * packed_tile),
            offset=h * gemv_tiles * packed_tile,
            sizes=[1, 1, 1, gemv_tiles * packed_tile], strides=[0, 0, 0, 1])

    x_tap = TensorAccessPattern(
        tensor_dims=(1, xb_elems), offset=0,
        sizes=[1, 1, 1, xb_elems], strides=[0, 0, 0, 1])

    def kvgrp_tap(g):
        # one contiguous fill of 4 heads' cache region (heads g*4 .. g*4+3)
        return TensorAccessPattern(
            tensor_dims=(NH * ahs_elems,), offset=g * 4 * ahs_elems,
            sizes=[1, 1, 1, 4 * ahs_elems], strides=[0, 0, 0, 1])

    ow_tap = TensorAccessPattern(           # Wo region follows the A region in the W BO
        tensor_dims=(1, a_region + ow_region), offset=a_region,
        sizes=[1, 1, 1, ow_region], strides=[0, 0, 0, 1])
    o_drain_tap = TensorAccessPattern(
        tensor_dims=(1, E), offset=0, sizes=[1, 1, 1, E], strides=[0, 0, 0, 1])

    def shim(col):
        return Tile(col=col, row=0)

    rt = Runtime()
    with rt.sequence(L3_O_ty, L3_W_ty, L3_X_ty, L3_K_ty, L3_V_ty) as (o, w, x, k, v):
        rt.start(*workers)
        tg1 = rt.task_group()
        for h in range(NH):
            cc = center_col_offset + (h % 4)
            rt.fill(A_f[h].prod(), w, a_tap(h), task_group=tg1, placement=shim(cc))
        for g in range(2):
            rt.fill(B_src[g].prod(), x, x_tap, task_group=tg1, placement=shim(g))
        # O-proj weights ride a free edge shim (col 6); the A fills already use the
        # center col shims, B uses cols 0,1.
        rt.fill(OW_f.prod(), w, ow_tap, task_group=tg1, placement=shim(6))
        rt.finish_task_group(tg1)
        tg2 = rt.task_group()
        for g in range(2):
            rt.fill(K_grp[g].prod(), k, kvgrp_tap(g), task_group=tg2,
                    placement=shim(EDGE_COLS[g * 2]))
            rt.fill(V_grp[g].prod(), v, kvgrp_tap(g), task_group=tg2,
                    placement=shim(EDGE_COLS[g * 2 + 1]))
        rt.finish_task_group(tg2)
        tg3 = rt.task_group()
        rt.drain(O_out.cons(), o, o_drain_tap, task_group=tg3, wait=True,
                 placement=shim(center_col_offset + 1))
        rt.finish_task_group(tg3)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())
