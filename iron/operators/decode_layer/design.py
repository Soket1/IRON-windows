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
                    hidden_dim=8192, stub_ffn=False,
                    chunk_size=None, m_input=2, num_cols=4, col_offset=2,
                    stub_oproj=False, rms_gain_data=None):
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

    # FFN (SwiGLU) sizes — ffn_probe-proven
    Hc = hidden_dim // num_cols                      # 2048
    assert Hc * num_cols == hidden_dim
    gu_groups = E // group_size
    gu_packed = m_input * E // 2 + m_input * gu_groups * 2
    gu_tiles = Hc // m_input
    dn_groups = Hc // group_size
    dn_packed = m_input * Hc // 2 + m_input * dn_groups * 2
    dn_tiles = E // m_input

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
    L1_GUW_ty = np.ndarray[(gu_packed,), u8]          # gate/up/down weight tile
    L1_SILU_ty = np.ndarray[(Hc,), bf]                # silu(gate)*up per column

    DTYPE_SIZE = 2
    ahs = int((seq_len * head_dim * DTYPE_SIZE + 63) / 64) * 64
    ahs_elems = ahs // DTYPE_SIZE

    # FFLM Q4NX weight order in ONE BO: [QKV | O | FFN]. FFN region = per-column
    # [gate|up|down] CONTIGUOUS (one shim stream/col, 16-S2MM cap is global).
    guw_per_col = 2 * gu_tiles * gu_packed              # gate then up tiles
    dnw_per_col = dn_tiles * dn_packed
    ffnw_per_col = guw_per_col + dnw_per_col            # [gate|up|down] per col
    ffn_region  = num_cols * ffnw_per_col
    w_region   = num_cols * gemv_tiles * packed_tile    # all QKV weights
    ow_region  = num_cols * o_tiles * o_packed          # all O weights
    ow_base    = w_region
    ffn_base   = w_region + ow_region
    WT_ELEMS   = w_region + ow_region + ffn_region
    L3_WT_ty = np.ndarray[(WT_ELEMS,), u8]
    # X is an input BUNDLE (FFLM-style: pack activations into one BO, not a 6th
    # DDR arg — the 5-BO ABI is full). Layout: [Xqkv (E) | inpL (E)] where Xqkv
    # is the (pre-normed) QKV GEMV activation and inpL is the residual the ANM
    # adds to the O_proj output. RMS gain is a passive Buffer (per-layer const).
    L3_X_ty  = np.ndarray[(2 * K_gemv,), bf]
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
    # O_proj is the v2 GEMV but with a DISTINCT symbol (compiled from the same
    # source with -Dfused_dequant_matvec_v2_bf16=oproj_matvec_v2_bf16) so its C
    # output arg can be typed L1_O_ty (o_slice=512) — matching the join endpoint
    # — instead of the QKV q-buffer L1_Q_ty (578). Reusing the same symbol forces
    # the 578 type and a memref mismatch at the join. Distinct symbol + distinct
    # .o avoids the redefinition error while letting the types differ.
    oproj = Kernel("oproj_matvec_v2_bf16",
                   f"fused_dequant_gemv_v2_oproj_{K_gemv}k_g{group_size}.o",
                   [np.int32, np.int32, L1_OW_ty, L1_E_ty, L1_O_ty])
    # ANM kernels (layer_fused.cc, same .o): residual add (c=a+b) then weighted
    # RMSNorm (out = in * invsqrt(mean(in^2)+1e-5) * gain). The relay tile uses
    # add only (a+0 = copy); the post-O_proj ANM tile uses add then rms.
    add_k = Kernel("layer_fused_add_bf16", "layer_fused_relay.o",
                   [L1_E_ty, L1_E_ty, L1_E_ty, np.int32])
    rms_k = Kernel("layer_fused_rms_norm2_bf16", "layer_fused_relay.o",
                   [L1_E_ty, L1_E_ty, L1_E_ty, np.int32])
    # FFN kernels: gate_up + silu from layer_fused_relay.o; down = v2 GEMV K=Hc
    # with a distinct renamed symbol; reduce4 does fp32 accumulation of 4 partials
    # + residual on a MemTile-joined 4E buffer (FFLM-style single-MemTile aggregation).
    gate_up = Kernel("layer_fused_gate_up_bf16", "layer_fused_relay.o",
                     [np.int32, np.int32, L1_GUW_ty, L1_E_ty, np.int32])
    silu_k = Kernel("layer_fused_silu_mul_bf16", "layer_fused_relay.o",
                    [L1_SILU_ty, np.int32])
    down = Kernel("down_matvec_v2_bf16",
                  f"fused_dequant_gemv_v2_down_{Hc}k_g{group_size}.o",
                  [np.int32, np.int32, L1_GUW_ty, L1_SILU_ty, L1_E_ty])
    L1_4E_ty = np.ndarray[(4 * E,), bf]
    reduce4 = Kernel("layer_fused_reduce4_bf16", "layer_fused_relay.o",
                     [L1_4E_ty, L1_E_ty, L1_E_ty, np.int32])

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
    # Row-5 weight stream. stub_ffn → OW only (O_proj). full FFN → D1.c MERGED
    # weight fifo W_f carrying [O_proj | gate | up | down] tiles in order on ONE
    # S2MM channel (FFLM bd-multiplex: one channel, N chained tiles). The fused
    # worker acquires from W_f for every weight loop (oproj→gate→up→down). This
    # collapses 2 S2MM (OW + GUWD) → 1, hitting the 2-S2MM core-tile cap. Valid
    # because all weight tiles are byte-identical in size (o_packed==gu_packed==
    # dn_packed) for this config (asserted below); the kernels differ but each
    # consumes one same-sized packed tile per acquire.
    if stub_ffn:
        OW_f = [ObjectFifo(L1_OW_ty, name=f"OW_{c}", depth=2) for c in range(num_cols)]
    else:
        assert o_packed == gu_packed == dn_packed, \
            "D1.c merged weight fifo needs uniform tile size (Hc==E config)"
        W_f = [ObjectFifo(L1_OW_ty, name=f"W_{c}", depth=2) for c in range(num_cols)]
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
    O_parts = AttnFull.prod().join(
        offsets=[c * q_rows for c in range(num_cols)],
        placement=Tile(col=4, row=1),
        obj_types=[L1_O_ty for _ in range(num_cols)])
    # Broadcast structure differs by mode:
    #  - stub_ffn (diagnostic): a plain relay forwards attn-out → AttnB to the
    #    O_proj-only row-5 workers (FFN absent, no ffn-in stream).
    #  - full FFN (D1.b): the row-5 tile needs BOTH attn-out (O_proj phase) and
    #    ffn-in (FFN phase), but a core tile has only 2 S2MM inputs and weights
    #    take channels. So attn-out and ffn-in time-multiplex onto ONE broadcast
    #    fifo: a MUX worker (generalized relay) produces them in order [attn,
    #    ffn_in] into MergedBcast; the row-5 tile acquires it TWICE per iteration
    #    (1st=attn, 2nd=ffn_in). An ObjectFifo has ONE producer, so the two
    #    upstreams (AttnFull join + ANM ffn-in) MUST funnel through one MUX worker
    #    (IRON asserts single producer). forward() = 1 S2MM per consumer tile.
    if stub_ffn:
        AttnBcast = ObjectFifo(L1_E_ty, name="AttnBcast", depth=2)
        AttnB = AttnBcast.cons().forward(
            name="attn_bcast", depth=2, placement=Tile(col=5, row=1))
    else:
        MergedBcast = ObjectFifo(L1_E_ty, name="MergedBcast", depth=2)  # mux out
        MergedB = MergedBcast.cons().forward(
            name="merged_bcast", depth=2, placement=Tile(col=5, row=1))

    # ANM stage (post-O_proj): the 4 O_proj output slices (o_slice each) are
    # joined back to full-E on a free MemTile (col0), then a single ANM worker
    # adds the residual inpL and applies weighted RMSNorm. RMS needs the whole-E
    # sum-of-squares, so the join must assemble all 4 columns first (post_attn_
    # fused does exactly this: O_proj -> join MemTile -> ANM tile). One consumer
    # (the ANM worker) reads the join target, so NO relay is needed here.
    OProjFull = ObjectFifo(L1_E_ty, name="OProjFull", depth=2)   # join target
    Oproj_parts = OProjFull.prod().join(
        offsets=[c * o_slice for c in range(num_cols)],
        placement=Tile(col=0, row=1),
        obj_types=[L1_O_ty for _ in range(num_cols)])
    InpL  = ObjectFifo(L1_E_ty, name="InpL",  depth=2)   # residual (from X bundle)
    # ANM output = ffn_in. In stub mode it is drained directly; in full FFN mode
    # it feeds the MUX worker (D1.b), which merges it with attn-out onto one
    # broadcast. Same fifo object either way (single ANM producer).
    FfnIn = ObjectFifo(L1_E_ty, name="FfnIn", depth=2)
    anm_dual = not stub_ffn

    if not stub_ffn:
        InpFF = ObjectFifo(L1_E_ty, name="InpFF", depth=2)  # FFN final residual
        # D1 (FFLM center geometry): FFN runs on the SAME center cols 2-5 as the
        # attention/O_proj workers — NOT scattered on edge cols 0,1,6,7. Each
        # center col's O_proj worker (row 5) is FUSED with that col's FFN body
        # (one tile = one worker → O_proj phase THEN FFN phase, time-multiplexed).
        # This removes the edge-scatter cross-tile interference suspected of the
        # ~11% deficit and matches fflm_routing_analysis.md §7 (all on cols 2-5).
        # GUWD merged into W_f (D1.c) — gate/up/down tiles stream after the
        # O_proj tiles on the same per-col weight fifo. No separate GUWD fifo.
        # D1.a: SILU is a TILE-LOCAL scratch buffer, NOT an ObjectFifo. silu_mul
        # writes it, down reads it — both on the SAME row-5 tile, so no DMA is
        # needed. A self-fifo (prod+cons same tile) would burn 1 S2MM + 1 MM2S
        # (core tile cap = 2+2); a Buffer burns ZERO DMA channels. Passed to the
        # kernels directly like gain_buf/zero_buf (no acquire/release).
        silu_zero = np.zeros(Hc, dtype=bfloat16)
        SILU_buf = [Buffer(type=L1_SILU_ty, initial_value=silu_zero,
                           name=f"silu_scratch_{c}") for c in range(num_cols)]
        # FFN partial reduction: 4 E-partials join → 4E buffer on MemTile(6,1),
        # the FFLM single-MemTile aggregator (fflm_routing_analysis.md §6). col6
        # MemTile also hosts K_full split (4 outs); join adds 4 ins = 8 of 12 ch.
        PartsFull = ObjectFifo(L1_4E_ty, name="PartsFull", depth=1)
        Pp = PartsFull.prod().join(
            offsets=[i*E for i in range(num_cols)], placement=Tile(col=6, row=1),
            obj_types=[L1_E_ty for _ in range(num_cols)])
        # FfnIn (ANM output) is consumed by the MUX worker (D1.b), NOT broadcast
        # directly — the MUX funnels attn-out + ffn-in into one MergedBcast that
        # forwards to all 4 row-5 tiles. No separate ffn broadcast fifo.
        FfnOut = ObjectFifo(L1_E_ty, name="FfnOut", depth=2)

    def relay_body(src, dst, zero, copy_fn):
        for _ in range_(0xFFFFFFFF):
            a = src.acquire(1)
            d = dst.acquire(1)
            copy_fn(a, zero, d, E)
            src.release(1); dst.release(1)

    # D1.b MUX worker (full FFN): generalized relay that funnels TWO upstreams
    # into one merged broadcast, in order [attn_out, ffn_in], per iteration. The
    # row-5 tiles acquire MergedB twice — 1st=attn (O_proj), 2nd=ffn_in (FFN).
    def mux_body(attn_src, ffn_src, merged_dst, zero, copy_fn):
        for _ in range_(0xFFFFFFFF):
            a = attn_src.acquire(1)            # phase 1: attn-out
            d = merged_dst.acquire(1)
            copy_fn(a, zero, d, E)
            attn_src.release(1); merged_dst.release(1)
            f = ffn_src.acquire(1)             # phase 2: ffn-in
            d2 = merged_dst.acquire(1)
            copy_fn(f, zero, d2, E)
            ffn_src.release(1); merged_dst.release(1)

    zero_e = np.zeros(E, dtype=bfloat16)
    zero_buf = Buffer(type=L1_E_ty, initial_value=zero_e, name="relay_zero")

    if not anm_dual:
        def anm_body(oj, inpl, ffi, gain, add_fn, rms_fn):
            for _ in range_(0xFFFFFFFF):
                o = oj.acquire(1); l = inpl.acquire(1); r = ffi.acquire(1)
                add_fn(o, l, r, E); rms_fn(r, gain, r, E)
                oj.release(1); inpl.release(1); ffi.release(1)
    else:
        def anm_body(oj, inpl, ffi, pf_out, gain, add_fn, rms_fn):
            for _ in range_(0xFFFFFFFF):
                o = oj.acquire(1); l = inpl.acquire(1)
                r = ffi.acquire(1); pf = pf_out.acquire(1)
                add_fn(o, l, pf, E); rms_fn(pf, gain, r, E)
                oj.release(1); inpl.release(1); ffi.release(1); pf_out.release(1)

    if rms_gain_data is None:
        rms_gain_data = np.ones(E, dtype=bfloat16)
    gain_buf = Buffer(type=L1_E_ty, initial_value=rms_gain_data, name="rms_gain")

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

        # D1 FUSED row-5 worker: O_proj phase THEN FFN phase on the SAME tile
        # (one tile = one worker; FFN can't be a separate worker on this tile).
        # Phase 1 (O_proj): consume OW + merged-bcast(1st=attn) → Oproj_parts join.
        # Phase 2 (FFN): consume GUWD + merged-bcast(2nd=ffn_in) → g/u/silu/down → Pp.
        # SILU is a tile-local Buffer (D1.a). attn-out and ffn-in arrive on ONE
        # merged broadcast (D1.b) acquired twice. This is the FFLM time-multiplex
        # pattern (stages + streams share a tile sequentially in one iteration).
        def oproj_ffn_body(wf, mb, oj, p, silu_scratch,
                           oproj_fn, gate_up_fn, silu_fn, down_fn):
            for _ in range_(0xFFFFFFFF):
                # --- phase 1: O_proj (1st merged acquire = attn-out) ---
                # weights from merged W_f: first o_tiles tiles = O_proj
                a_full = mb.acquire(1)
                o = oj.acquire(1)
                for j in range_(o_tiles):
                    w = wf.acquire(1)
                    oproj_fn(m_input, index.casts(T.i32(), j) * m_input,
                             w, a_full, o)
                    wf.release(1)
                mb.release(1); oj.release(1)
                # --- phase 2: FFN (2nd merged acquire = ffn-in) ---
                # weights continue on W_f: gate tiles, then up, then down
                b = mb.acquire(1)
                for j in range_(gu_tiles):  # gate → lf_left
                    w = wf.acquire(1)
                    gate_up_fn(m_input, index.casts(T.i32(), j) * m_input, w, b, 0)
                    wf.release(1)
                for j in range_(gu_tiles):  # up → lf_right
                    w = wf.acquire(1)
                    gate_up_fn(m_input, index.casts(T.i32(), j) * m_input, w, b, 1)
                    wf.release(1)
                mb.release(1)
                # silu·mul writes the tile-local scratch (D1.a: Buffer, not fifo)
                silu_fn(silu_scratch, Hc)
                fo = p.acquire(1)
                for j in range_(dn_tiles):  # down over silu (reads scratch)
                    w = wf.acquire(1)
                    down_fn(m_input, index.casts(T.i32(), j) * m_input,
                            w, silu_scratch, fo)
                    wf.release(1)
                p.release(1)

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
        elif stub_ffn:
            workers.append(Worker(oproj_body,
                [OW_f[c].cons(), AttnB.cons(), Oproj_parts[c].prod(), oproj],
                placement=Tile(col=pc, row=5)))
        else:
            # FFN enabled → fused O_proj+FFN on this center row-5 tile (D1).
            # ONE merged weight fifo W_f (D1.c: O_proj|gate|up|down tiles), ONE
            # merged broadcast MergedB (D1.b: attn-out then ffn-in, 2 acquires),
            # SILU = tile-local Buffer (D1.a). Net: 2 S2MM (W_f + MergedB) +
            # 2 MM2S (Oproj_parts + Pp) = exactly the core-tile cap.
            workers.append(Worker(oproj_ffn_body,
                [W_f[c].cons(), MergedB.cons(), Oproj_parts[c].prod(),
                 Pp[c].prod(),
                 SILU_buf[c],
                 oproj, gate_up, silu_k, down],
                placement=Tile(col=pc, row=5)))

    # Tile(6,2): stub mode = plain relay (attn-out → AttnBcast → AttnB); full FFN
    # mode = MUX worker (D1.b) funneling attn-out + ffn-in → MergedBcast.
    if stub_ffn:
        workers.append(Worker(relay_body,
            [AttnFull.cons(), AttnBcast.prod(), zero_buf, add_k],
            placement=Tile(col=6, row=2)))
    else:
        workers.append(Worker(mux_body,
            [AttnFull.cons(), FfnIn.cons(), MergedBcast.prod(), zero_buf, add_k],
            placement=Tile(col=6, row=2)))

    # ANM worker: O_proj join + residual inpL + passive gain. Dual output
    # (InpFF + FfnIn) when FFN enabled.
    if not anm_dual:
        workers.append(Worker(anm_body,
            [OProjFull.cons(), InpL.cons(), FfnIn.prod(), gain_buf, add_k, rms_k],
            placement=Tile(col=7, row=2)))
    else:
        workers.append(Worker(anm_body,
            [OProjFull.cons(), InpL.cons(), FfnIn.prod(), InpFF.prod(),
             gain_buf, add_k, rms_k],
            placement=Tile(col=7, row=2)))

    if not stub_ffn:
        # FFN now runs FUSED into the row-5 oproj_ffn_body on center cols 2-5
        # (D1 — see the worker loop above). No separate edge FFN workers. Only
        # the final reduction worker remains here.

        # FFLM-style reduce: MemTile-joined 4E partials + inpFF → ffn_out
        def reduce_body(parts_f, resid_f, out_f, red_fn):
            for _ in range_(0xFFFFFFFF):
                p = parts_f.acquire(1); r = resid_f.acquire(1); o = out_f.acquire(1)
                red_fn(p, r, o, E)
                parts_f.release(1); resid_f.release(1); out_f.release(1)

        import os as _os
        if _os.environ.get("DECODE_DBG_DUMP_P0"):
            _red = Kernel("layer_fused_dump_part_bf16", "layer_fused_relay.o",
                          [L1_4E_ty, L1_E_ty, L1_E_ty, np.int32])
        else:
            _red = reduce4
        workers.append(Worker(reduce_body,
            [PartsFull.cons(), InpFF.cons(), FfnOut.prod(), _red],
            placement=Tile(col=6, row=5)))

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
            tensor_dims=(1, WT_ELEMS), offset=c * a_per_col,
            sizes=[1, 1, 1, a_per_col], strides=[0, 0, 0, 1])

    def ow_tap(c):
        return TensorAccessPattern(
            tensor_dims=(1, WT_ELEMS), offset=ow_base + c * ow_per_col,
            sizes=[1, 1, 1, ow_per_col], strides=[0, 0, 0, 1])

    def ffnw_tap(i):
        return TensorAccessPattern(
            tensor_dims=(1, WT_ELEMS), offset=ffn_base + i * ffnw_per_col,
            sizes=[1, 1, 1, ffnw_per_col], strides=[0, 0, 0, 1])

    # X bundle: Xqkv at offset 0, residual inpL at offset K_gemv (both E elems).
    x_tap = TensorAccessPattern(tensor_dims=(1, 2 * K_gemv), offset=0,
        sizes=[1, 1, 1, K_gemv], strides=[0, 0, 0, 1])
    inpL_tap = TensorAccessPattern(tensor_dims=(1, 2 * K_gemv), offset=K_gemv,
        sizes=[1, 1, 1, K_gemv], strides=[0, 0, 0, 1])

    ffn_tap = TensorAccessPattern(tensor_dims=(1, E), offset=0,
        sizes=[1, 1, 1, E], strides=[0, 0, 0, 1])

    # K/V split sources: ONE fill of the whole num_cols*ahs_elems contiguous
    # region; the MemTile split fans the per-column kv_col slices on-chip.
    def kv_src_tap():
        return TensorAccessPattern(
            tensor_dims=(num_cols * ahs_elems,), offset=0,
            sizes=[1, 1, 1, num_cols * ahs_elems], strides=[0, 0, 0, 1])

    # ffn_in (ANM output) drain: full E, contiguous.
    ffn_tap = TensorAccessPattern(tensor_dims=(1, E), offset=0,
        sizes=[1, 1, 1, E], strides=[0, 0, 0, 1])

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
        # B: QKV activation = X bundle offset 0. InpL: residual = X bundle offset
        # E (feeds the ANM tile). O: single full-E ffn_in drain (ANM output).
        tg_main = rt.task_group()
        for c in range(num_cols):
            rt.fill(A_f[c].prod(), wt, a_tap(c), task_group=tg_main)
            # Row-5 weight stream: stub → OW only; full FFN → O_proj region into
            # the merged W_f[c] (FFN gate/up/down region filled next in tg_ffn,
            # same producer, FIFO order → O_proj tiles consumed first).
            if stub_ffn:
                rt.fill(OW_f[c].prod(), wt, ow_tap(c), task_group=tg_main)
            else:
                rt.fill(W_f[c].prod(), wt, ow_tap(c), task_group=tg_main)
        rt.fill(K_full.prod(), k, kv_src_tap(), task_group=tg_main)
        rt.fill(V_full.prod(), v, kv_src_tap(), task_group=tg_main)
        rt.fill(B_src.prod(),  x, x_tap,        task_group=tg_main)
        rt.fill(InpL.prod(),   x, inpL_tap,     task_group=tg_main)
        rt.finish_task_group(tg_main)

        if stub_ffn:
            tg_out = rt.task_group()
            rt.drain(FfnIn.cons(), o, ffn_tap, task_group=tg_out, wait=True)
            rt.finish_task_group(tg_out)
        else:
            # FFN weights (gate|up|down per col) into the SAME merged W_f[c],
            # after the O_proj region filled in tg_main (D1.c bd-multiplex).
            tg_ffn = rt.task_group()
            for i in range(num_cols):
                rt.fill(W_f[i].prod(), wt, ffnw_tap(i), task_group=tg_ffn)
            rt.drain(FfnOut.cons(), o, ffn_tap, task_group=tg_ffn, wait=True)
            rt.finish_task_group(tg_ffn)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())
