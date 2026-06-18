# SPDX-License-Identifier: Apache-2.0
"""decode_qkvffn16 — P6.4/50a: QKV (concat) + FFN (reduce) on the SAME 16 tiles, ONE dispatch.

The structural fix for decode_layer's FFN-on-4-tiles regression: all 16 center tiles
(cols 2-5 × rows 2-5) run BOTH phases time-muxed, weights streamed QKV-then-FFN per tile:
  phase1 QKV : gemv (v2, K=E) -> 192-slice -> CONCAT-join per col -> drained (QKV[3072])
  phase2 FFN : gate_up + silu(local) + down -> E partial -> REDUCE-tree -> Cout[E]
Reuses PROVEN bodies verbatim: decode_qkv16.qkv_body + decode_ffn16_2mm.ffn_body. All
kernel symbols distinct (gemv / gate_up / silu / down / reduce4) → no redefinition.
Cap skeleton resolves (decode_qkvffn_probe). Attention deferred (QKV drained to DDR for
CPU check; ffn_in is an independent DDR input). One Wsrc/col (split-4, 4 MM2S weight) —
correctness first; 8-MM2S (SPC=2) is a later bandwidth step.
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


def my_decode_qkvffn16(dev, embed_dim=2048, qkv_dim=3072, hidden_dim=8192,
                       group_size=32, m_input=4, num_cols=4, col_offset=2):
    dev_ty = NPU1() if dev == "npu" else NPU2()
    bf = np.dtype[bfloat16]
    u8 = np.dtype[np.uint8]
    E, QD, H, g, m = embed_dim, qkv_dim, hidden_dim, group_size, m_input
    NC, R = num_cols, 4
    NT = NC * R                                    # 16

    # ---- phase1 QKV dims (decode_qkv16) ----
    SL = QD // NT                                  # 192
    qkv_tiles = SL // m                            # 48
    COL_Q = R * SL                                 # 768
    # ---- phase2 FFN dims (decode_ffn16_2mm) ----
    Hc16 = H // NT                                 # 512
    gu_tiles = Hc16 // m                           # 128
    dn_tiles = E // m                              # 512
    PACKED = m * E // 2 + m * (E // g) * 2         # 4608
    DN_PACKED = m * Hc16 // 2 + m * (Hc16 // g) * 2  # 1152
    DN_SUB = PACKED // DN_PACKED                   # 4
    dn_elems = dn_tiles // DN_SUB                  # 128
    FFN_WT = gu_tiles + gu_tiles + dn_elems        # 384
    WT_PER_TILE = qkv_tiles + FFN_WT               # 432 (QKV then FFN, time-muxed)

    L1_E_ty   = np.ndarray[(E,), bf]
    L1_S_ty   = np.ndarray[(SL,), bf]
    L1_CQ_ty  = np.ndarray[(COL_Q,), bf]
    L1_QD_ty  = np.ndarray[(QD,), bf]
    L1_4E_ty  = np.ndarray[(R * E,), bf]
    L1_W_ty   = np.ndarray[(PACKED,), u8]
    L1_W4_ty  = np.ndarray[(R * PACKED,), u8]

    L3_O_ty  = np.ndarray[(QD + E,), bf]          # bo0 = [QKV(3072) | FFN(E)]
    L3_E_ty  = np.ndarray[(E,), bf]
    L3_W_ty  = np.ndarray[(NC * WT_PER_TILE * R * PACKED,), u8]

    gemv = Kernel("fused_dequant_matvec_v2_bf16", f"fused_dequant_gemv_v2_{E}k_g{g}.o",
                  [np.int32, np.int32, L1_W_ty, L1_E_ty, L1_S_ty])
    gate_up = Kernel("layer_fused_gate_up_bf16", "layer_fused_relay.o",
                     [np.int32, np.int32, L1_W_ty, L1_E_ty, np.int32])
    silu = Kernel("layer_fused_silu_mul_static_bf16", "layer_fused_relay.o", [np.int32])
    down = Kernel("layer_fused_down_v2_x4_bf16", "layer_fused_relay.o",
                  [np.int32, np.int32, np.int32, L1_W_ty, L1_E_ty])
    reduce4 = Kernel("layer_fused_reduce4_bf16", "layer_fused_relay.o",
                     [L1_4E_ty, L1_E_ty, L1_E_ty, np.int32])
    cp_cq = Kernel("layer_fused_add_bf16", "layer_fused_relay.o",
                   [L1_CQ_ty, L1_CQ_ty, L1_CQ_ty, np.int32])

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

    # phase1 CONCAT join: 4 slices(192) -> 768 @ MemTile(c,1)
    QF, QF_parts = [], []
    for c in range(NC):
        qf = ObjectFifo(L1_CQ_ty, name=f"QF_{c}", depth=2)
        parts = qf.prod().join(offsets=[r * SL for r in range(R)],
                               placement=Tile(col=col_offset + c, row=1),
                               obj_types=[L1_S_ty for _ in range(R)])
        QF.append(qf); QF_parts.append(parts)
    QKVfull = ObjectFifo(L1_QD_ty, name="QKVfull", depth=2)
    QKV_fp = QKVfull.prod().join(offsets=[c * COL_Q for c in range(NC)],
                                 placement=Tile(col=6, row=1),
                                 obj_types=[L1_CQ_ty for _ in range(NC)])
    # phase2 REDUCE join: 4 partials(E) -> 4E @ MemTile(c,1) -> colred -> E -> final
    PF, PF_parts = [], []
    for c in range(NC):
        pf = ObjectFifo(L1_4E_ty, name=f"PF_{c}", depth=1)
        parts = pf.prod().join(offsets=[r * E for r in range(R)],
                               placement=Tile(col=col_offset + c, row=1),
                               obj_types=[L1_E_ty for _ in range(R)])
        PF.append(pf); PF_parts.append(parts)
    FinalParts = ObjectFifo(L1_4E_ty, name="FinalParts", depth=1)
    FP_parts = FinalParts.prod().join(offsets=[c * E for c in range(NC)],
                                      placement=Tile(col=6, row=1),
                                      obj_types=[L1_E_ty for _ in range(NC)])
    Cout = ObjectFifo(L1_E_ty, name="Cout", depth=2)

    workers = []

    def center_body(wf, bc, qpart, fpart, gemv_fn, gate_up_fn, silu_fn, down_fn):
        for _ in range_(0xFFFFFFFF):
            # phase1 QKV (concat)
            x = bc.acquire(1); qo = qpart.acquire(1)
            for j in range_(qkv_tiles):
                w = wf.acquire(1)
                gemv_fn(m, index.casts(T.i32(), j) * m, w, x, qo)
                wf.release(1)
            bc.release(1); qpart.release(1)
            # phase2 FFN (reduce): gate, up, silu(local), down
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
                [Wf[c][r].cons(), Bcol[c].cons(), QF_parts[c][r].prod(), PF_parts[c][r].prod(),
                 gemv, gate_up, silu, down],
                placement=Tile(col=col_offset + c, row=2 + r)))

    # col relay: phase1 copy concat (768) -> final concat ; phase2 reduce4 -> final reduce
    def colrelay_body(qf, qfp, pf, fpp, zc, ze, cpcq, red):
        for _ in range_(0xFFFFFFFF):
            a = qf.acquire(1); o = qfp.acquire(1)
            cpcq(a, zc, o, COL_Q); qf.release(1); qfp.release(1)
            f = pf.acquire(1); r = fpp.acquire(1)
            red(f, ze, r, E); pf.release(1); fpp.release(1)
    for c in range(NC):
        workers.append(Worker(
            colrelay_body,
            [QF[c].cons(), QKV_fp[c].prod(), PF[c].cons(), FP_parts[c].prod(),
             mk_zero(L1_CQ_ty, COL_Q), mk_zero(L1_E_ty, E), cp_cq, reduce4],
            placement=Tile(col=1, row=2 + c)))

    def finalred_body(fp, out, z, red):
        for _ in range_(0xFFFFFFFF):
            f = fp.acquire(1); o = out.acquire(1)
            red(f, z, o, E); fp.release(1); out.release(1)
    workers.append(Worker(finalred_body, [FinalParts.cons(), Cout.prod(), mk_zero(L1_E_ty, E), reduce4],
                          placement=Tile(col=6, row=2)))

    e_tap = TensorAccessPattern(tensor_dims=(1, E), offset=0, sizes=[1, 1, 1, E], strides=[0, 0, 0, 1])
    # bo0 = [QKV(QD) | FFN(E)]; QKV at offset 0, FFN at offset QD
    qd_tap = TensorAccessPattern(tensor_dims=(1, QD + E), offset=0, sizes=[1, 1, 1, QD], strides=[0, 0, 0, 1])
    ff_tap = TensorAccessPattern(tensor_dims=(1, QD + E), offset=QD, sizes=[1, 1, 1, E], strides=[0, 0, 0, 1])
    wcol = WT_PER_TILE * R * PACKED
    def w_tap(c):
        return TensorAccessPattern(tensor_dims=(1, NC * wcol), offset=c * wcol,
                                   sizes=[1, 1, 1, wcol], strides=[0, 0, 0, 1])

    rt = Runtime()
    # bo0=[QKV(3072)|FFN(E)], bo1=W, bo2=x(E), bo3=ffn_in(E), bo4=dummy
    with rt.sequence(L3_O_ty, L3_W_ty, L3_E_ty, L3_E_ty, L3_E_ty) as (out, w, x, ffin, _d):
        rt.start(*workers)
        tg = rt.task_group()
        for c in range(NC):
            rt.fill(Wsrc[c].prod(), w, w_tap(c), task_group=tg)
        for c in range(NC):
            rt.fill(Bsrc[c].prod(), x, e_tap, task_group=tg)     # phase1 x
        for c in range(NC):
            rt.fill(Bsrc[c].prod(), ffin, e_tap, task_group=tg)  # phase2 ffn_in
        rt.drain(QKVfull.cons(), out, qd_tap, task_group=tg, wait=True)
        rt.drain(Cout.cons(), out, ff_tap, task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())
