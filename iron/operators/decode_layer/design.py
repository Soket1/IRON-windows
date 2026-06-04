# SPDX-License-Identifier: Apache-2.0
"""decode_layer — fused decode layer, increment A: QKV→RoPE→flowkv→join→O_proj.

Builds on the proven 4-column decode_front chain (QKV+RoPE+flowkv on rows 2-4 of
central columns 2-5) and adds the post-attention O_proj:

  rows 2-4 (per col): GEMV+RoPE → score → value  → O_slice[c] (E/cols = 512)
  MemTile join:       4 O_slice[c] → full attn_out (E=2048)
  MemTile forward:    broadcast full attn_out to all O_proj tiles
  row 5 (per col):    O_proj INT4 GEMV (K=E, v2 kernel) → o_out_slice[c] → DDR

attn_group=8 so per-col attn slice = 8*64 = 512, ×4 = 2048 = E (O_proj K=E).
O_proj uses the v2 GEMV kernel (c_out += row_offset, no -8 bias) — same as the
oproj_probe. attn_out gather = ObjectFifo.join (gather_probe-proven). Each tile's
RMS/RoPE statics are per-tile. DMA time-multiplexed via task_group phases.
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


def my_decode_layer(dev, embed_dim=2048, head_dim=64, group_size=32,
                    attn_group=8, seq_len=32, output_first=False,
                    chunk_size=None, m_input=2, num_cols=4, col_offset=2,
                    stub_oproj=False):
    if chunk_size is None:
        chunk_size = seq_len
    dev_ty = NPU1() if dev == "npu" else NPU2()
    bf = np.dtype[bfloat16]; u8 = np.dtype[np.uint8]
    E = embed_dim
    K_gemv = E
    assert col_offset + num_cols <= 8
    assert attn_group * head_dim * num_cols == E, "per-col attn slice must tile E"

    q_rows = attn_group * head_dim                  # 512 (QKV out per col)
    q_buf_elems = q_rows + head_dim + 2
    groups = K_gemv // group_size
    packed_tile = m_input * K_gemv // 2 + m_input * groups * 2
    gemv_tiles = q_rows // m_input
    num_chunks = seq_len // chunk_size
    inter_size = chunk_size * attn_group + 2 * attn_group
    o_slice = E // num_cols                          # 512 (O_proj out per col)
    o_groups = E // group_size
    o_packed = m_input * E // 2 + m_input * o_groups * 2
    o_tiles = o_slice // m_input

    L1_A_ty   = np.ndarray[(packed_tile,), u8]
    L1_B_ty   = np.ndarray[(K_gemv,), bf]
    L1_Q_ty   = np.ndarray[(q_buf_elems,), bf]
    L1_LUT_ty = np.ndarray[(head_dim,), bf]
    L1_K_ty   = np.ndarray[(chunk_size * head_dim,), bf]
    L1_I_ty   = np.ndarray[(inter_size,), bf]
    L1_V_ty   = np.ndarray[(chunk_size * head_dim,), bf]
    L1_O_ty   = np.ndarray[(q_rows,), bf]            # attn out slice (512)
    L1_E_ty   = np.ndarray[(E,), bf]                 # full attn_out
    L1_OW_ty  = np.ndarray[(o_packed,), u8]          # O_proj weight tile (= packed_tile, K=E)
    # O_proj reuses the QKV v2-gemv kernel symbol → its C arg must match L1_Q_ty
    # (578). O_proj writes o_slice=512 rows into it (first 512 used). o_packed
    # == packed_tile and L1_E_ty == L1_B_ty (both K=E=2048), so all 3 args match.
    L1_OC_ty  = L1_Q_ty                               # o_out buffer (write first o_slice)

    DTYPE_SIZE = 2
    ahs = int((seq_len * head_dim * DTYPE_SIZE + 63) / 64) * 64
    ahs_elems = ahs // DTYPE_SIZE

    # FFLM-style ONE weight BO (like FFLM's bo1 = all layer weights together):
    # QKV/A region then O_proj region, contiguous in a single DDR buffer. This
    # keeps the kernel at 5 data BOs (FFLM's MLIR_AIE ABI: bo0-4), with the
    # weight BO holding all GEMV weights. Layout: [W (num_cols*a_per_col) |
    # OW (num_cols*ow_per_col)] — A_src reads at 0, OW_src at the W-region end.
    w_region  = num_cols * gemv_tiles * packed_tile          # all QKV weights
    ow_region = num_cols * o_tiles * o_packed                # all O weights
    ow_base   = w_region                                     # OW offset in the BO
    L3_WT_ty = np.ndarray[(w_region + ow_region,), u8]       # single weight BO
    L3_X_ty  = np.ndarray[(K_gemv,), bf]
    L3_K_ty  = np.ndarray[(num_cols * ahs_elems,), bf]
    L3_V_ty  = np.ndarray[(num_cols * ahs_elems,), bf]
    L3_O_ty  = np.ndarray[(E,), bf]                  # final o_out (E)

    # Kernels (declared once)
    gemv = Kernel("fused_dequant_matvec_v2_bf16",
                  f"fused_dequant_gemv_v2_{K_gemv}k_g{group_size}.o",
                  [np.int32, np.int32, L1_A_ty, L1_B_ty, L1_Q_ty])
    rope = Kernel("rope", "rope_th.o", [L1_Q_ty, L1_LUT_ty, L1_Q_ty, np.int32])
    fkv = f"flowkv_{head_dim}d_h{attn_group}.o"
    s_init  = Kernel("flowkv_score_init_bf16",      fkv, [np.int32])
    s_rope  = Kernel("flowkv_score_rope_q_bf16",    fkv, [L1_Q_ty, np.int32, np.int32])
    s_chunk = Kernel("flowkv_score_chunk_bf16",     fkv,
                     [L1_Q_ty, L1_K_ty, L1_I_ty, np.int32, np.int32, np.int32])
    v_init  = Kernel("flowkv_value_init_bf16",      fkv, [np.int32, np.int32])
    v_accum = Kernel("flowkv_value_accum_bf16",     fkv,
                     [L1_I_ty, L1_V_ty, np.int32, np.int32, np.int32])
    v_norm  = Kernel("flowkv_value_normalize_bf16", fkv, [L1_O_ty, np.int32, np.int32])
    # O_proj is the SAME v2 GEMV kernel (K=E=K_gemv) as QKV — reuse the `gemv`
    # Kernel object (declaring it twice redefines the symbol). The kernel's
    # buffer args are generic memrefs; OW/L1_E/OC match A/B/Q element types
    # closely enough (uint8 weights, bf16 in, bf16 out) for the same symbol.
    oproj = gemv
    # E→E relay copy kernel (layer_fused_add: d = a + a as relay stand-in; this
    # tile becomes the ANM add+rms stage later). From layer_fused.cc.
    add_k = Kernel("layer_fused_add_bf16", "layer_fused_relay.o",
                   [L1_E_ty, L1_E_ty, L1_E_ty, np.int32])

    # Weight/input delivery. A (QKV) and OW (O_proj) weights are PER-COLUMN
    # CONTIGUOUS fills (the decode_front-proven pattern): each column's weight
    # region is laid out contiguously in the WT BO and streamed into a per-
    # column fifo whose element = one packed tile; IRON re-tiles the contiguous
    # DDR region by the fifo element size. A .split() was WRONG here — its fifo
    # element is one tile-row interleaved across columns ([col0_t | col1_t | ..]),
    # which expects TILE-major DDR order, but the packing is COLUMN-major, so
    # every column got scrambled tiles. Per-column contiguous matches the pack.
    # K/V stay as .split() (each column has exactly 1 chunk, so column-major ==
    # tile-major — no scramble) and B stays a broadcast (one vector to all cols).
    # Shim budget (per column, 2-S2MM cap): cols 2-5 = A_f[c] + OW_f[c] = 2 S2MM;
    # B-source on free col1 MemTile; K/V split sources on col6/col7 MemTiles.
    kv_col  = chunk_size * head_dim                     # K/V per column (1 chunk)

    A_f  = [ObjectFifo(L1_A_ty,  name=f"A_{c}",  depth=2) for c in range(num_cols)]
    OW_f = [ObjectFifo(L1_OW_ty, name=f"OW_{c}", depth=2) for c in range(num_cols)]
    K_full = ObjectFifo(np.ndarray[(num_cols * kv_col,), bf], name="K_src", depth=2)
    K_f = K_full.cons().split(
        offsets=[c * kv_col for c in range(num_cols)], placement=Tile(col=6, row=1),
        obj_types=[L1_K_ty for _ in range(num_cols)])
    V_full = ObjectFifo(np.ndarray[(num_cols * kv_col,), bf], name="V_src", depth=2)
    V_f = V_full.cons().split(
        offsets=[c * kv_col for c in range(num_cols)], placement=Tile(col=7, row=1),
        obj_types=[L1_V_ty for _ in range(num_cols)])
    # input vector B: one shim source, broadcast (forward) to all columns. The
    # broadcast MemTile sits on free col1 (NOT col3 — col3 already carries
    # A_f[1]+OW_f[1] = 2 S2MM, so a B source there would be the 3rd, over cap).
    B_src = ObjectFifo(L1_B_ty, name="B_src", depth=1)
    B_bcast = B_src.cons().forward(name="B_bcast", depth=1, placement=Tile(col=1, row=1))

    Qi  = [ObjectFifo(L1_Q_ty, name=f"Qi_{c}", depth=2) for c in range(num_cols)]
    Ii  = [ObjectFifo(L1_I_ty, name=f"Ii_{c}", depth=2) for c in range(num_cols)]
    OC_f = [ObjectFifo(L1_OC_ty, name=f"OC_{c}", depth=2) for c in range(num_cols)]

    # attn_out join: 4 value-stage O slices → full-E fifo at MemTile col_offset.
    # Then a SEPARATE forward on a DIFFERENT MemTile broadcasts the assembled E
    # vector to the 4 O_proj workers — layer_fused keeps join and broadcast on
    # distinct MemTile columns so neither over-subscribes the 6+6 channel budget.
    # Place join + broadcast MemTiles on the FREE edge columns (1 and 6), NOT on
    # the central compute columns 2-5 — layer_fused does exactly this (join col1,
    # broadcast col6) so MemTiles don't collide with the compute-column tiles.
    # MemTile usage so far: A-split col0, OW-split col1, K-split col6, V-split
    # col7, B-forward col3. Use the remaining free MemTile cols 4,5 for the
    # attn_out join and broadcast (each MemTile is independent; compute tiles on
    # cols 4,5 are on rows 2-5, the MemTile is row 1).
    # join→relay→broadcast: a join-fifo CANNOT also be .forward()'d (both put it
    # in an ObjectFifoLinkOp → 'more than one link' AIECC error), AND a single
    # fifo that is both join-target and read by N workers DEADLOCKS on NPU. So a
    # RELAY worker sits between: it consumes the joined full-E (AttnFull) and
    # produces it into AttnBcast, which forwards/broadcasts to the N O_proj
    # workers. This relay tile is where ANM (add+rms) will live later.
    AttnFull  = ObjectFifo(L1_E_ty, name="AttnFull",  depth=2)   # join target
    AttnBcast = ObjectFifo(L1_E_ty, name="AttnBcast", depth=2)   # broadcast src
    O_parts = AttnFull.prod().join(
        offsets=[c * q_rows for c in range(num_cols)],
        placement=Tile(col=4, row=1),
        obj_types=[L1_O_ty for _ in range(num_cols)])
    AttnB = AttnBcast.cons().forward(
        name="attn_bcast", depth=2, placement=Tile(col=5, row=1))

    def relay_body(src, dst, copy_fn):
        for _ in range_(0xFFFFFFFF):
            a = src.acquire(1)
            d = dst.acquire(1)
            copy_fn(a, a, d, E)              # d = a + a (relay; becomes ANM later)
            src.release(1); dst.release(1)

    rope_lut_data = np.zeros(head_dim, dtype=bfloat16)
    rope_lut_data[0::2] = bfloat16(1.0); rope_lut_data[1::2] = bfloat16(0.0)
    luts = [Buffer(type=L1_LUT_ty, initial_value=rope_lut_data, name=f"rope_lut_{c}")
            for c in range(num_cols)]

    workers = []
    for c in range(num_cols):
        pc = c + col_offset

        def gemv_body(af, bf_, qf, lut, gemv_fn, rope_fn):
            for _ in range_(0xFFFFFFFF):
                b = bf_.acquire(1); q = qf.acquire(1)
                for j in range_(gemv_tiles):
                    a = af.acquire(1)
                    gemv_fn(m_input, index.casts(T.i32(), j) * m_input, a, b, q)
                    af.release(1)
                rope_fn(q, lut, q, head_dim)
                qf.release(1); bf_.release(1)
        workers.append(Worker(gemv_body,
            [A_f[c].cons(), B_bcast.cons(), Qi[c].prod(), luts[c], gemv, rope],
            placement=Tile(col=pc, row=2)))

        def score_body(kf, qf, inf, init_fn, rope_fn, chunk_fn):
            for _ in range_(0xFFFFFFFF):
                init_fn(attn_group)
                q = qf.acquire(1)
                rope_fn(q, attn_group, head_dim)
                for _ in range_(num_chunks):
                    k = kf.acquire(1); it = inf.acquire(1)
                    chunk_fn(q, k, it, attn_group, head_dim, chunk_size)
                    kf.release(1); inf.release(1)
                qf.release(1)
        workers.append(Worker(score_body,
            [K_f[c].cons(), Qi[c].cons(), Ii[c].prod(), s_init, s_rope, s_chunk],
            placement=Tile(col=pc, row=3)))
        # (K_f[c]/V_f[c] are split sub-fifos; .cons() is their read endpoint)

        def value_body(vf, inf, of, init_fn, accum_fn, norm_fn):
            for _ in range_(0xFFFFFFFF):
                init_fn(attn_group, head_dim)
                for _ in range_(num_chunks):
                    it = inf.acquire(1); v = vf.acquire(1)
                    accum_fn(it, v, attn_group, head_dim, chunk_size)
                    inf.release(1); vf.release(1)
                o = of.acquire(1)
                norm_fn(o, attn_group, head_dim)
                of.release(1)
        workers.append(Worker(value_body,
            [V_f[c].cons(), Ii[c].cons(), O_parts[c].prod(),
             v_init, v_accum, v_norm],
            placement=Tile(col=pc, row=4)))

        def oproj_body(ow, ab, oc, oproj_fn):
            for _ in range_(0xFFFFFFFF):
                a_full = ab.acquire(1)            # full attn_out (broadcast)
                o = oc.acquire(1)
                for j in range_(o_tiles):
                    w = ow.acquire(1)
                    oproj_fn(m_input, index.casts(T.i32(), j) * m_input,
                             w, a_full, o)
                    ow.release(1)
                ab.release(1); oc.release(1)

        # DEADLOCK-ISOLATION stub: O_proj worker that ONLY consumes the joined
        # attn_out and produces an output slice — NO OW weights, NO GEMV (just
        # acquire/release both fifos). If this COMPLETES, the hang is in the
        # O_proj GEMV / OW weight-fill path; if it still hangs, the join / attn
        # broadcast is the culprit. No kernel call needed for the sync test.
        def oproj_stub(ab, oc):
            for _ in range_(0xFFFFFFFF):
                ab.acquire(1)
                oc.acquire(1)
                ab.release(1); oc.release(1)

        if stub_oproj:
            workers.append(Worker(oproj_stub,
                [AttnB.cons(), OC_f[c].prod()],
                placement=Tile(col=pc, row=5)))
        else:
            workers.append(Worker(oproj_body,
                [OW_f[c].cons(), AttnB.cons(), OC_f[c].prod(), oproj],
                placement=Tile(col=pc, row=5)))

    # relay worker: join (AttnFull) → relay → AttnBcast (which forwards to AttnB
    # broadcast). On a free compute tile outside the cols-2-5 band (col 6 row 2).
    workers.append(Worker(relay_body,
        [AttnFull.cons(), AttnBcast.prod(), add_k],
        placement=Tile(col=6, row=2)))

    # Per-column contiguous taps (decode_front-proven). IRON auto-tiles a large
    # contiguous inner dim down to the BD-1023 limit, so a flat [.. , inner] tap
    # with inner >> 1023 is fine (decode_front's a_tap inner = 589824 works).
    # No _factor4 / inline_ops needed — those were introduced for the (wrong)
    # split delivery. A weights: column c at offset c*a_per_col. O weights:
    # column c at ow_base + c*ow_per_col (same single WT BO).
    a_per_col  = gemv_tiles * packed_tile          # A weights/col (bytes, u8)
    ow_per_col = o_tiles * o_packed                 # O weights/col (bytes, u8)

    def a_tap(c):
        return TensorAccessPattern(
            tensor_dims=(1, w_region + ow_region), offset=c * a_per_col,
            sizes=[1, 1, 1, a_per_col], strides=[0, 0, 0, 1])

    def ow_tap(c):
        return TensorAccessPattern(
            tensor_dims=(1, w_region + ow_region), offset=ow_base + c * ow_per_col,
            sizes=[1, 1, 1, ow_per_col], strides=[0, 0, 0, 1])

    x_tap = TensorAccessPattern(tensor_dims=(1, K_gemv), offset=0,
        sizes=[1, 1, 1, K_gemv], strides=[0, 0, 0, 1])

    # K/V split sources: ONE fill of the whole num_cols*ahs_elems contiguous
    # region; the MemTile split fans the per-column kv_col slices on-chip.
    def kv_src_tap():
        return TensorAccessPattern(
            tensor_dims=(num_cols * ahs_elems,), offset=0,
            sizes=[1, 1, 1, num_cols * ahs_elems], strides=[0, 0, 0, 1])

    def oc_tap(c):
        return TensorAccessPattern(tensor_dims=(1, E), offset=c * o_slice,
            sizes=[1, 1, o_tiles, m_input], strides=[0, 0, m_input, 1])

    rt = Runtime()
    # output_first reorders the DDR BO list so the final O is bo0 (arg3) — the
    # replay harness dumps bo0 to disk for off-line numeric validation. Default
    # (committed) order keeps O last (matches the op.py runlist WT,X,K,V,O).
    if output_first:
        seq_tys = (L3_O_ty, L3_WT_ty, L3_X_ty, L3_K_ty, L3_V_ty)
    else:
        seq_tys = (L3_WT_ty, L3_X_ty, L3_K_ty, L3_V_ty, L3_O_ty)
    with rt.sequence(*seq_tys) as seq_args:
        if output_first:
            o, wt, x, k, v = seq_args
        else:
            wt, x, k, v, o = seq_args
        rt.start(*workers)

        # A/OW: per-column contiguous fills (cols 2-5: 2 S2MM each, within cap).
        # K/V: one fill of the split source each (fanned on col6/col7 MemTiles).
        # B: one broadcast fill (col1 MemTile). O: per-column drains.
        tg_w = rt.task_group()
        for c in range(num_cols):
            rt.fill(A_f[c].prod(),  wt, a_tap(c),  task_group=tg_w)
            rt.fill(OW_f[c].prod(), wt, ow_tap(c), task_group=tg_w)
        rt.fill(K_full.prod(), k, kv_src_tap(), task_group=tg_w)
        rt.fill(V_full.prod(), v, kv_src_tap(), task_group=tg_w)
        rt.fill(B_src.prod(),  x, x_tap,        task_group=tg_w)
        for c in range(num_cols):
            rt.drain(OC_f[c].cons(), o, oc_tap(c), task_group=tg_w, wait=True)
        rt.finish_task_group(tg_w)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())
