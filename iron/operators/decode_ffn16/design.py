# SPDX-License-Identifier: Apache-2.0
"""decode_ffn16 — D2.2b: 16-tile parallel FFN (gate/up/silu/down), 16-way split.

Drops the real FFN body into the D2.2a skeleton topology (decode_red16). All 16
tiles (cols 2-5 × rows 2-5) run gate→up→silu→down on a 1/16 output slice of an
8192-wide FFN, then 16 down-partials reduce 2-level → layer FFN output.

DMA layout (FFLM-faithful, all ≤ caps):
  - WEIGHTS via column MemTile(c,1) split (North): a per-column interleaved stream
    (each round = the j-th weight tile of all 4 rows) is split to 4 per-tile fifos.
    Column MemTile = weight(1 S2MM fill + 4 MM2S) + down-join(4 S2MM + 1 MM2S) = 5+5.
  - ACTIVATION (ffn_in) broadcast via MemTile(1,1) (East), 4 per-row forwards —
    kept OFF the column MemTiles (which are full with weights+join).
  - REDUCE: per-column down-join → reduce4 core (col 1) → 4 col-partials join @
    MemTile(6,1) → reduce4 core (col 6) → out.
Per tile = 2 S2MM (weight North + act) + 2 MM2S (down-out South + ...). 16-way:
gate/up 512 out (gu_tiles=128 @ m=4); down K=Hc16=512 → full-E partial. down
packed (1152B) padded to gate/up size (4608B) for a uniform weight-fifo element.
See dev_notes/track_a_build/D2_design_spike.md §3b.
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


def my_decode_ffn16(dev, embed_dim=2048, hidden_dim=8192, group_size=32,
                    m_input=4, num_cols=4, col_offset=2):
    dev_ty = NPU1() if dev == "npu" else NPU2()
    bf = np.dtype[bfloat16]
    u8 = np.dtype[np.uint8]
    E, H, g, m = embed_dim, hidden_dim, group_size, m_input
    NC, R = num_cols, 4
    NT = NC * R                                   # 16 tiles

    Hc16 = H // NT                                # 512 (per-tile gate/up out, down K)
    gu_tiles = Hc16 // m                          # 128
    dn_tiles = E // m                             # 512
    PACKED = m * E // 2 + m * (E // g) * 2        # 4608 (gate/up tile, K=E)
    DN_PACKED = m * Hc16 // 2 + m * (Hc16 // g) * 2  # 1152 (down tile, K=Hc16)
    # UNPAD (D2.2c): pack DN_SUB=4 down tiles per 4608 element (4×1152=4608) so
    # the weight fifo element stays uniform WITHOUT padding. down loop slices.
    DN_SUB = PACKED // DN_PACKED                  # 4
    assert DN_SUB * DN_PACKED == PACKED
    dn_elems = dn_tiles // DN_SUB                 # 128
    assert dn_elems * DN_SUB == dn_tiles
    WT_PER_TILE = gu_tiles + gu_tiles + dn_elems  # 384 (was 768 padded)

    L1_E_ty  = np.ndarray[(E,), bf]
    L1_4E_ty = np.ndarray[(4 * E,), bf]
    L1_W_ty  = np.ndarray[(PACKED,), u8]          # one element (gate/up tile, or 4 down)
    L1_DW_ty = np.ndarray[(DN_PACKED,), u8]       # one down sub-tile (slice)
    L1_W4_ty = np.ndarray[(R * PACKED,), u8]      # one round = 4 rows' j-th element

    L3_E_ty  = np.ndarray[(E,), bf]
    L3_D_ty  = np.ndarray[(64,), bf]
    # per-column interleaved weight region: WT_PER_TILE rounds × (R × PACKED)
    L3_W_ty  = np.ndarray[(NC * WT_PER_TILE * R * PACKED,), u8]

    gate_up = Kernel("layer_fused_gate_up_bf16", "layer_fused_relay.o",
                     [np.int32, np.int32, L1_W_ty, L1_E_ty, np.int32])
    silu = Kernel("layer_fused_silu_mul_static_bf16", "layer_fused_relay.o",
                  [np.int32])
    down = Kernel("layer_fused_down_v2_x4_bf16", "layer_fused_relay.o",
                  [np.int32, np.int32, np.int32, L1_W_ty, L1_E_ty])
    reduce4 = Kernel("layer_fused_reduce4_bf16", "layer_fused_relay.o",
                     [L1_4E_ty, L1_E_ty, L1_E_ty, np.int32])
    add = Kernel("layer_fused_add_bf16", "layer_fused_relay.o",
                 [L1_E_ty, L1_E_ty, L1_E_ty, np.int32])
    import os as _os
    DBG_COPY = bool(_os.environ.get("DBG_FFN16_COPY"))

    _zn = [0]
    def mk_zero():
        b = Buffer(type=L1_E_ty, initial_value=np.zeros(E, dtype=bfloat16),
                   name=f"ffn16_zero_{_zn[0]}")
        _zn[0] += 1
        return b

    # --- activation broadcast: per-COLUMN forward (skeleton-proven) on MemTile(c,1),
    # to the column's 4 row tiles. Coexists with weight-split + join on the same
    # MemTile IF the hardware broadcast is a single stream-switch fan (1 MM2S).
    Bsrc = [ObjectFifo(L1_E_ty, name=f"Bsrc_{c}", depth=1) for c in range(NC)]
    Bcol = [Bsrc[c].cons().forward(name=f"Bcol_{c}", depth=2,
                                   placement=Tile(col=col_offset + c, row=1))
            for c in range(NC)]

    # --- weights: per-column interleaved stream → split (North) to 4 per-tile fifos
    Wsrc = [ObjectFifo(L1_W4_ty, name=f"Wsrc_{c}", depth=2) for c in range(NC)]
    Wf = [Wsrc[c].cons().split(
            offsets=[r * PACKED for r in range(R)],
            placement=Tile(col=col_offset + c, row=1),
            obj_types=[L1_W_ty for _ in range(R)])
          for c in range(NC)]

    # --- down-partial L1 join: per column, 4 row partials → 4E @ MemTile(c,1) ---
    PF, PF_parts = [], []
    for c in range(NC):
        pf = ObjectFifo(L1_4E_ty, name=f"PF_{c}", depth=1)
        parts = pf.prod().join(
            offsets=[r * E for r in range(R)],
            placement=Tile(col=col_offset + c, row=1),
            obj_types=[L1_E_ty for _ in range(R)])
        PF.append(pf); PF_parts.append(parts)

    # --- L2 join: 4 column-partials → 4E @ MemTile(6,1) ---
    FinalParts = ObjectFifo(L1_4E_ty, name="FinalParts", depth=1)
    FP_parts = FinalParts.prod().join(
        offsets=[c * E for c in range(NC)],
        placement=Tile(col=6, row=1),
        obj_types=[L1_E_ty for _ in range(NC)])
    Cout = ObjectFifo(L1_E_ty, name="Cout", depth=2)

    workers = []

    def ffn_body(wf, bc, pp, gate_up_fn, silu_fn, down_fn):
        for _ in range_(0xFFFFFFFF):
            b = bc.acquire(1)
            for j in range_(gu_tiles):                 # gate → lf_left
                w = wf.acquire(1)
                gate_up_fn(m, index.casts(T.i32(), j) * m, w, b, 0)
                wf.release(1)
            for j in range_(gu_tiles):                 # up → lf_right
                w = wf.acquire(1)
                gate_up_fn(m, index.casts(T.i32(), j) * m, w, b, 1)
                wf.release(1)
            bc.release(1)
            silu_fn(Hc16)
            o = pp.acquire(1)
            for j in range_(dn_elems):                 # 128 elems × 4 sub-tiles
                w = wf.acquire(1)                      # 4608 = 4 down sub-tiles
                ro = index.casts(T.i32(), j) * DN_SUB * m   # base output row
                down_fn(m, ro, DN_SUB, w, o)          # ONE call processes 4 sub-tiles
                wf.release(1)
            pp.release(1)

    # DEBUG: copy activation → partial, still drain all 768 weights, to isolate
    # broadcast+weight-streaming+reduce from the FFN body.
    def copy_body(wf, bc, pp, add_fn, zero):
        for _ in range_(0xFFFFFFFF):
            b = bc.acquire(1)
            for _j in range_(2 * gu_tiles + dn_elems):
                w = wf.acquire(1)
                wf.release(1)
            o = pp.acquire(1)
            add_fn(b, zero, o, E)
            bc.release(1)
            pp.release(1)

    for c in range(NC):
        for r in range(R):
            if DBG_COPY:
                workers.append(Worker(
                    copy_body,
                    [Wf[c][r].cons(), Bcol[c].cons(), PF_parts[c][r].prod(),
                     add, mk_zero()],
                    placement=Tile(col=col_offset + c, row=2 + r)))
            else:
                workers.append(Worker(
                    ffn_body,
                    [Wf[c][r].cons(), Bcol[c].cons(), PF_parts[c][r].prod(),
                     gate_up, silu, down],
                    placement=Tile(col=col_offset + c, row=2 + r)))

    def colred_body(pf, cp, zero, red_fn):
        for _ in range_(0xFFFFFFFF):
            p = pf.acquire(1); o = cp.acquire(1)
            red_fn(p, zero, o, E)
            pf.release(1); cp.release(1)

    for c in range(NC):
        workers.append(Worker(
            colred_body,
            [PF[c].cons(), FP_parts[c].prod(), mk_zero(), reduce4],
            placement=Tile(col=1, row=2 + c)))

    def finalred_body(fp, out, zero, red_fn):
        for _ in range_(0xFFFFFFFF):
            p = fp.acquire(1); o = out.acquire(1)
            red_fn(p, zero, o, E)
            fp.release(1); out.release(1)

    workers.append(Worker(
        finalred_body,
        [FinalParts.cons(), Cout.prod(), mk_zero(), reduce4],
        placement=Tile(col=6, row=2)))

    e_tap = TensorAccessPattern(tensor_dims=(1, E), offset=0,
                                sizes=[1, 1, 1, E], strides=[0, 0, 0, 1])
    wcol = WT_PER_TILE * R * PACKED

    def w_tap(c):
        return TensorAccessPattern(
            tensor_dims=(1, NC * wcol), offset=c * wcol,
            sizes=[1, 1, 1, wcol], strides=[0, 0, 0, 1])

    rt = Runtime()
    # 5-BO: bo0=out, bo1=W, bo2=ffn_in, bo3=dummy, bo4=dummy
    with rt.sequence(L3_E_ty, L3_W_ty, L3_E_ty, L3_D_ty, L3_D_ty) as (o, w, x, _d3, _d4):
        rt.start(*workers)
        tg = rt.task_group()
        for c in range(NC):
            rt.fill(Wsrc[c].prod(), w, w_tap(c), task_group=tg)
        for c in range(NC):
            rt.fill(Bsrc[c].prod(), x, e_tap, task_group=tg)
        rt.drain(Cout.cons(), o, e_tap, task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())
