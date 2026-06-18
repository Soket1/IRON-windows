# SPDX-License-Identifier: Apache-2.0
"""decode_qkvffn16_b1 — P6.4b BASELINE: QKV+FFN merged on 16 tiles, ONE reused E-join per
MemTile (cap-fit in PURE IRON, no hand-author). Throwaway scaffold to validate weight-packing
+ FFN-reduce correctness on real weights, SEPARATE from the parametric generator.

Cap fix vs decode_qkvffn16 (which AIECC-fails 10>6 at L1, 8>6 at L2): the two separate joins
(QF concat 192-typed + PF reduce E-typed) are replaced by ONE reused E-typed join per center
MemTile, used BOTH phases. The join DMA is identical for concat and reduce (proven: same 4-S2MM
gather, differs only in consumer). To keep the toy minimal: relay tiles are pure COPIES; ALL
concat-extraction (QKV) and cross-tile reduction (FFN) happen on the HOST.

Per center tile emits E (2048) both phases:
  phase1 QKV: gemv writes its 192-slice into the FIRST 192 of an E buffer (rest = pad/garbage).
  phase2 FFN: gate_up + silu + down -> full E partial.
Drains: bo0 = 16*E QKV partials (host extracts [tile*E : tile*E+SL] -> QD=3072),
        bo1 = 16*E FFN partials (host sums 16 -> E).
This is NOT FFLM-faithful (FFLM keeps QKV on-chip, no DDR concat); padding waste is fine for a
baseline number. The parametric channel-reuse generator (no pad) is the next step.
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


def my_decode_qkvffn16_b1(dev, embed_dim=2048, qkv_dim=3072, hidden_dim=8192,
                          group_size=32, m_input=4, num_cols=4, col_offset=2):
    dev_ty = NPU1() if dev == "npu" else NPU2()
    bf = np.dtype[bfloat16]
    u8 = np.dtype[np.uint8]
    E, QD, H, g, m = embed_dim, qkv_dim, hidden_dim, group_size, m_input
    NC, R = num_cols, 4
    NT = NC * R                                    # 16

    SL = QD // NT                                  # 192 (QKV slice / tile)
    qkv_tiles = SL // m                            # 48
    Hc16 = H // NT                                 # 512
    gu_tiles = Hc16 // m                           # 128
    dn_tiles = E // m                              # 512
    PACKED = m * E // 2 + m * (E // g) * 2         # 4608
    DN_PACKED = m * Hc16 // 2 + m * (Hc16 // g) * 2  # 1152
    DN_SUB = PACKED // DN_PACKED                   # 4
    dn_elems = dn_tiles // DN_SUB                  # 128
    FFN_WT = gu_tiles + gu_tiles + dn_elems        # 384
    WT_PER_TILE = qkv_tiles + FFN_WT               # 432

    L1_E_ty   = np.ndarray[(E,), bf]
    L1_S_ty   = np.ndarray[(SL,), bf]
    L1_4E_ty  = np.ndarray[(R * E,), bf]
    L1_W_ty   = np.ndarray[(PACKED,), u8]
    L1_W4_ty  = np.ndarray[(R * PACKED,), u8]

    L3_OUT_ty = np.ndarray[(2 * NT * E,), bf]      # bo0 = [16E QKV partials | 16E FFN partials]
    L3_E_ty   = np.ndarray[(E,), bf]
    L3_W_ty   = np.ndarray[(NC * WT_PER_TILE * R * PACKED,), u8]
    gemv = Kernel("fused_dequant_matvec_v2_bf16", f"fused_dequant_gemv_v2_{E}k_g{g}.o",
                  [np.int32, np.int32, L1_W_ty, L1_E_ty, L1_E_ty])
    gate_up = Kernel("layer_fused_gate_up_bf16", "layer_fused_relay.o",
                     [np.int32, np.int32, L1_W_ty, L1_E_ty, np.int32])
    silu = Kernel("layer_fused_silu_mul_static_bf16", "layer_fused_relay.o", [np.int32])
    down = Kernel("layer_fused_down_v2_x4_bf16", "layer_fused_relay.o",
                  [np.int32, np.int32, np.int32, L1_W_ty, L1_E_ty])
    cp_e = Kernel("layer_fused_add_bf16", "layer_fused_relay.o",
                  [L1_E_ty, L1_E_ty, L1_E_ty, np.int32])

    _zn = [0]
    def mk_zero(ty, n):
        b = Buffer(type=ty, initial_value=np.zeros(n, dtype=bfloat16), name=f"z_{_zn[0]}")
        _zn[0] += 1
        return b

    # activation broadcast per col (phase1 x ; phase2 ffn_in) — Bsrc filled twice
    Bsrc = [ObjectFifo(L1_E_ty, name=f"Bsrc_{c}", depth=1) for c in range(NC)]
    Bcol = [Bsrc[c].cons().forward(name=f"Bcol_{c}", depth=2,
                                   placement=Tile(col=col_offset + c, row=1))
            for c in range(NC)]
    # weights: 1 stream/col split North to 4 per-tile fifos (QKV tiles then FFN tiles)
    Wsrc = [ObjectFifo(L1_W4_ty, name=f"Wsrc_{c}", depth=2) for c in range(NC)]
    Wf = [Wsrc[c].cons().split(offsets=[r * PACKED for r in range(R)],
                               placement=Tile(col=col_offset + c, row=1),
                               obj_types=[L1_W_ty for _ in range(R)])
          for c in range(NC)]

    # ONE reused E-typed join per center MemTile (both phases) -> 4E @ MemTile(c,1)
    PF, PF_parts = [], []
    for c in range(NC):
        pf = ObjectFifo(L1_4E_ty, name=f"PF_{c}", depth=1)
        parts = pf.prod().join(offsets=[r * E for r in range(R)],
                               placement=Tile(col=col_offset + c, row=1),
                               obj_types=[L1_E_ty for _ in range(R)])
        PF.append(pf); PF_parts.append(parts)

    # ONE reused L2 join (both phases) -> 16E @ MemTile(6,1)
    Final = ObjectFifo(L1_4E_ty, name="Final", depth=1)
    F_parts = Final.prod().join(offsets=[c * E for c in range(NC)],
                                placement=Tile(col=6, row=1),
                                obj_types=[L1_E_ty for _ in range(NC)])
    # two output fifos drained from the L2 relay tile (col6,row2): QKV (phase1) & FFN (phase2)
    Qout = ObjectFifo(L1_4E_ty, name="Qout", depth=2)   # 4 col-chunks * E = NC*E per round
    Fout = ObjectFifo(L1_4E_ty, name="Fout", depth=2)

    workers = []

    def center_body(wf, bc, fpart, gemv_fn, gate_up_fn, silu_fn, down_fn):
        for _ in range_(0xFFFFFFFF):
            # phase1 QKV: gemv writes 192-slice into first SL of the E buffer
            x = bc.acquire(1); qo = fpart.acquire(1)
            for j in range_(qkv_tiles):
                w = wf.acquire(1)
                gemv_fn(m, index.casts(T.i32(), j) * m, w, x, qo)
                wf.release(1)
            bc.release(1); fpart.release(1)
            # phase2 FFN: gate, up, silu(local), down -> full E partial
            b = bc.acquire(1)
            for j in range_(gu_tiles):
                w = wf.acquire(1); gate_up_fn(m, index.casts(T.i32(), j) * m, w, b, 0); wf.release(1)
            for j in range_(gu_tiles):
                w = wf.acquire(1); gate_up_fn(m, index.casts(T.i32(), j) * m, w, b, 1); wf.release(1)
            bc.release(1)
            silu_fn(Hc16)
            fo = fpart.acquire(1)
            for j in range_(dn_elems):
                w = wf.acquire(1)
                ro = index.casts(T.i32(), j) * DN_SUB * m
                down_fn(m, ro, DN_SUB, w, fo)
                wf.release(1)
            fpart.release(1)

    for c in range(NC):
        for r in range(R):
            workers.append(Worker(
                center_body,
                [Wf[c][r].cons(), Bcol[c].cons(), PF_parts[c][r].prod(),
                 gemv, gate_up, silu, down],
                placement=Tile(col=col_offset + c, row=2 + r)))

    # NO compute relay: drain each PF[c] DIRECTLY from its MemTile to DDR (IRON drains a
    # MemTile-placed objectfifo fine — cf. decode_qkv16 QKVfull on MemTile(6,1)). PF[c]
    # yields 8E/token: 4E phase1 (QKV partials) then 4E phase2 (FFN partials). We drain
    # BOTH into a per-column 8E region; the host splits phase1/phase2 + extracts/sums.

    # bo0 layout: per column c, an 8E region = [phase1 4E QKV | phase2 4E FFN].
    # PF[c] yields 4E phase1 then 4E phase2; we drain both into col c's 8E slot.
    COLW, TOTAL = 2 * R * E, NC * 2 * R * E
    def col_tap(base, c):
        return TensorAccessPattern(tensor_dims=(1, TOTAL), offset=c * COLW + base,
                                   sizes=[1, 1, 1, R * E], strides=[0, 0, 0, 1])
    wcol = WT_PER_TILE * R * PACKED
    def w_tap(c):
        return TensorAccessPattern(tensor_dims=(1, NC * wcol), offset=c * wcol,
                                   sizes=[1, 1, 1, wcol], strides=[0, 0, 0, 1])
    e_tap = TensorAccessPattern(tensor_dims=(1, E), offset=0, sizes=[1, 1, 1, E], strides=[0, 0, 0, 1])

    rt = Runtime()
    # bo0 = NC*8E [(QKV4E|FFN4E) per col], bo1=W, bo2=x(E), bo3=ffn_in(E), bo4=dummy
    with rt.sequence(L3_OUT_ty, L3_W_ty, L3_E_ty, L3_E_ty, L3_E_ty) as (out, w, x, ffin, _d):
        rt.start(*workers)
        tg = rt.task_group()
        for c in range(NC):
            rt.fill(Wsrc[c].prod(), w, w_tap(c), task_group=tg)
        for c in range(NC):
            rt.fill(Bsrc[c].prod(), x, e_tap, task_group=tg)     # phase1 x
        for c in range(NC):
            rt.fill(Bsrc[c].prod(), ffin, e_tap, task_group=tg)  # phase2 ffn_in
        for c in range(NC):
            rt.drain(PF[c].cons(), out, col_tap(0, c), task_group=tg, wait=False)       # phase1 QKV
        for c in range(NC):
            last = (c == NC - 1)
            rt.drain(PF[c].cons(), out, col_tap(R * E, c), task_group=tg, wait=last)     # phase2 FFN
        rt.finish_task_group(tg)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())
