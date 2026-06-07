# SPDX-License-Identifier: Apache-2.0
"""decode_back_real — #11.1b: 16-tile fused back-half (O->ANM->FFN), REAL bodies.

Real-body version of the proven decode_back_fused skeleton. Resolves the core-tile
2-S2MM wall (center would need attn + ffn_in + weights = 3 in) with the FFLM
Tile(1,2) AHUB that time-muxes BOTH activations into ONE broadcast:
  phase1 (O):   AHUB forwards attn(DDR) -> bcast -> 16 tiles O-GEMV, PADDED into a
                2048 partial at the tile's offset (matvec_v2 does c_out+=row_offset)
                -> 2-level reduce, resid=h_in folded at finalred -> inpff.
  AHUB phase2:  RMS(inpff)*gain -> ffn_in -> SAME bcast -> 16 tiles gate/up/silu/
                down (ffn16 body) -> 2-level reduce -> FFN.
  resid-hub:    out = FFN + inpff.
O padded-reduce SHARES the FFN reduce join. ONE weight stream (O 32 + gate 128 +
up 128 + down 128 = 416 chunks/tile, all 4608B) split per column.
center = bcast(MemTile1,1) + weight(MemTile c,1) = 2 S2MM + 1 MM2S.
MemTile(c,1) = weight-split(1+4) + reduce-join(4+1) = 5+5.
"""
import numpy as np
from ml_dtypes import bfloat16
import os as _os

import aie.dialects.index as index
from aie.dialects.aie import T
from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2, Tile


def my_decode_back_real(dev, embed_dim=2048, hidden_dim=8192, group_size=32,
                        m_input=4, num_cols=4, col_offset=2):
    dev_ty = NPU1() if dev == "npu" else NPU2()
    bf = np.dtype[bfloat16]
    u8 = np.dtype[np.uint8]
    E, H, g, m = embed_dim, hidden_dim, group_size, m_input
    NC, R = num_cols, 4
    NT = NC * R                                    # 16

    Hc16 = H // NT                                 # 512
    gu_tiles = Hc16 // m                           # 128
    SL_O = E // NT                                 # 128
    o_tiles = SL_O // m                            # 32
    PACKED = m * E // 2 + m * (E // g) * 2         # 4608
    DN_PACKED = m * Hc16 // 2 + m * (Hc16 // g) * 2  # 1152
    DN_SUB = PACKED // DN_PACKED                   # 4
    dn_elems = (E // m) // DN_SUB                  # 128
    WT_PER_TILE = o_tiles + gu_tiles + gu_tiles + dn_elems   # 416

    L1_E_ty = np.ndarray[(E,), bf]
    L1_4E_ty = np.ndarray[(R * E,), bf]
    L1_W_ty = np.ndarray[(PACKED,), u8]
    L1_W4_ty = np.ndarray[(R * PACKED,), u8]
    L3_E_ty = np.ndarray[(E,), bf]
    L3_W_ty = np.ndarray[(NC * WT_PER_TILE * R * PACKED,), u8]

    gemv = Kernel("layer_fused_o_scatter_bf16", "layer_fused_relay.o",
                  [np.int32, np.int32, L1_W_ty, L1_E_ty, L1_E_ty])
    gate_up = Kernel("layer_fused_gate_up_bf16", "layer_fused_relay.o",
                     [np.int32, np.int32, L1_W_ty, L1_E_ty, np.int32])
    silu = Kernel("layer_fused_silu_mul_static_bf16", "layer_fused_relay.o", [np.int32])
    down = Kernel("layer_fused_down_v2_x4_bf16", "layer_fused_relay.o",
                  [np.int32, np.int32, np.int32, L1_W_ty, L1_E_ty])
    reduce4 = Kernel("layer_fused_reduce4_bf16", "layer_fused_relay.o",
                     [L1_4E_ty, L1_E_ty, L1_E_ty, np.int32])
    add = Kernel("layer_fused_add_bf16", "layer_fused_relay.o",
                 [L1_E_ty, L1_E_ty, L1_E_ty, np.int32])
    rms = Kernel("layer_fused_rms_norm_bf16", "layer_fused_relay.o",
                 [L1_E_ty, L1_E_ty, L1_E_ty, np.int32])

    _zn = [0]
    def mk_zero():
        b = Buffer(type=L1_E_ty, initial_value=np.zeros(E, dtype=bfloat16),
                   name=f"br_zero_{_zn[0]}"); _zn[0] += 1
        return b
    gain_buf = Buffer(type=L1_E_ty, initial_value=np.ones(E, dtype=bfloat16),
                      name="br_gain")

    # AHUB (Tile 1,2): merge attn (p1) + ffn_in (p2) into ONE bcast -> MemTile(1,1)
    Attn = ObjectFifo(L1_E_ty, name="Attn", depth=1)
    Inpff = ObjectFifo(L1_E_ty, name="Inpff", depth=2)        # finalred -> AHUB + resid (bcast)
    Bc = ObjectFifo(L1_E_ty, name="Bc", depth=1)
    BcB = Bc.cons().forward(name="BcB", depth=2, placement=Tile(col=1, row=1))

    # weights: per-column interleaved -> split to 4 per-tile fifos
    Wsrc = [ObjectFifo(L1_W4_ty, name=f"Wsrc_{c}", depth=2) for c in range(NC)]
    Wf = [Wsrc[c].cons().split(offsets=[r * PACKED for r in range(R)],
                               placement=Tile(col=col_offset + c, row=1),
                               obj_types=[L1_W_ty for _ in range(R)])
          for c in range(NC)]

    # per-column reduce-join (REUSED both phases) @ MemTile(c,1)
    PF, PF_parts = [], []
    for c in range(NC):
        pf = ObjectFifo(L1_4E_ty, name=f"PF_{c}", depth=1)
        parts = pf.prod().join(offsets=[r * E for r in range(R)],
                               placement=Tile(col=col_offset + c, row=1),
                               obj_types=[L1_E_ty for _ in range(R)])
        PF.append(pf); PF_parts.append(parts)

    # L2 reduce-join (REUSED) @ MemTile(6,1)
    FinalParts = ObjectFifo(L1_4E_ty, name="FinalParts", depth=1)
    FP_parts = FinalParts.prod().join(offsets=[c * E for c in range(NC)],
                                      placement=Tile(col=6, row=1),
                                      obj_types=[L1_E_ty for _ in range(NC)])
    Hin = ObjectFifo(L1_E_ty, name="Hin", depth=1)
    FFNfifo = ObjectFifo(L1_E_ty, name="FFNfifo", depth=2)
    Out = ObjectFifo(L1_E_ty, name="Out", depth=2)

    workers = []

    def ahub_body(attn_in, inpff_in, bc_out, gain, zero, add_fn, rms_fn):
        for _ in range_(0xFFFFFFFF):
            a = attn_in.acquire(1); o = bc_out.acquire(1)
            add_fn(a, zero, o, E)                   # phase1 bcast = attn (copy)
            attn_in.release(1); bc_out.release(1)
            ip = inpff_in.acquire(1); o = bc_out.acquire(1)
            rms_fn(ip, gain, o, E)                  # phase2 bcast = rms(inpff)*gain
            inpff_in.release(1); bc_out.release(1)

    workers.append(Worker(ahub_body, [Attn.cons(), Inpff.cons(), Bc.prod(),
                                      gain_buf, mk_zero(), add, rms],
                          placement=Tile(col=1, row=2)))

    def center_body(bc, wf, pp, zero, add_fn, gemv_fn, gate_up_fn, silu_fn, down_fn, scatter):
        _dbg_ffnin = bool(_os.environ.get("DBG_BR_FFNIN"))
        for _ in range_(0xFFFFFFFF):
            # phase1: O padded-reduce
            b = bc.acquire(1); p = pp.acquire(1)
            add_fn(zero, zero, p, E)               # zero the 2048 partial
            for j in range_(o_tiles):
                w = wf.acquire(1)
                gemv_fn(m, index.casts(T.i32(), j) * m + scatter, w, b, p)
                wf.release(1)
            bc.release(1); pp.release(1)
            # phase2: FFN
            b2 = bc.acquire(1)
            if _dbg_ffnin:
                p2 = pp.acquire(1)
                add_fn(b2, zero, p2, E)            # DBG: partial = ffn_in (copy) -> reduce=16*ffn_in
                for j in range_(gu_tiles + gu_tiles + dn_elems):
                    w = wf.acquire(1); wf.release(1)   # drain weights
                bc.release(1); pp.release(1)
            else:
                for j in range_(gu_tiles):
                    w = wf.acquire(1); gate_up_fn(m, index.casts(T.i32(), j) * m, w, b2, 0); wf.release(1)
                for j in range_(gu_tiles):
                    w = wf.acquire(1); gate_up_fn(m, index.casts(T.i32(), j) * m, w, b2, 1); wf.release(1)
                bc.release(1)
                silu_fn(Hc16)
                p2 = pp.acquire(1)
                for j in range_(dn_elems):
                    w = wf.acquire(1)
                    down_fn(m, index.casts(T.i32(), j) * DN_SUB * m, DN_SUB, w, p2)
                    wf.release(1)
                pp.release(1)

    for c in range(NC):
        for r in range(R):
            scatter = (c * R + r) * SL_O
            workers.append(Worker(
                center_body,
                [BcB.cons(), Wf[c][r].cons(), PF_parts[c][r].prod(),
                 mk_zero(), add, gemv, gate_up, silu, down, scatter],
                placement=Tile(col=col_offset + c, row=2 + r)))

    def colred_body(pf, fp, zero, red_fn):
        for _ in range_(0xFFFFFFFF):
            f = pf.acquire(1); o = fp.acquire(1)
            red_fn(f, zero, o, E); pf.release(1); fp.release(1)   # phase1
            f = pf.acquire(1); o = fp.acquire(1)
            red_fn(f, zero, o, E); pf.release(1); fp.release(1)   # phase2

    for c in range(NC):
        workers.append(Worker(colred_body, [PF[c].cons(), FP_parts[c].prod(), mk_zero(), reduce4],
                              placement=Tile(col=0, row=2 + c)))

    def finalred_body(fp, hin, inpff_out, ffn_out, zero, red_fn):
        for _ in range_(0xFFFFFFFF):
            f = fp.acquire(1); h = hin.acquire(1); o = inpff_out.acquire(1)
            red_fn(f, h, o, E)                                    # inpff = O_sum + h_in
            fp.release(1); hin.release(1); inpff_out.release(1)
            f = fp.acquire(1); o = ffn_out.acquire(1)
            red_fn(f, zero, o, E)                                 # FFN = sum
            fp.release(1); ffn_out.release(1)

    workers.append(Worker(finalred_body, [FinalParts.cons(), Hin.cons(), Inpff.prod(),
                                          FFNfifo.prod(), mk_zero(), reduce4],
                          placement=Tile(col=6, row=2)))

    def resid_body(ffni, inpff_in, out, zero, add_fn):
        _dbg_inpff = bool(_os.environ.get("DBG_BR_INPFF"))
        for _ in range_(0xFFFFFFFF):
            f = ffni.acquire(1); ip = inpff_in.acquire(1); o = out.acquire(1)
            if _dbg_inpff:
                add_fn(ip, zero, o, E)                           # DBG: out = inpff (isolate O+fold)
            else:
                add_fn(f, ip, o, E)                              # out = FFN + inpff
            ffni.release(1); inpff_in.release(1); out.release(1)

    workers.append(Worker(resid_body, [FFNfifo.cons(), Inpff.cons(), Out.prod(), mk_zero(), add],
                          placement=Tile(col=6, row=3)))

    tap = TensorAccessPattern(tensor_dims=(1, E), offset=0,
                              sizes=[1, 1, 1, E], strides=[0, 0, 0, 1])
    wcol = WT_PER_TILE * R * PACKED

    def w_tap(c):
        return TensorAccessPattern(tensor_dims=(1, NC * wcol), offset=c * wcol,
                                   sizes=[1, 1, 1, wcol], strides=[0, 0, 0, 1])

    rt = Runtime()
    # bo0=out, bo1=W, bo2=attn, bo3=h_in
    with rt.sequence(L3_E_ty, L3_W_ty, L3_E_ty, L3_E_ty) as (o, w, attn, hin):
        rt.start(*workers)
        tg = rt.task_group()
        for c in range(NC):
            rt.fill(Wsrc[c].prod(), w, w_tap(c), task_group=tg)
        rt.fill(Attn.prod(), attn, tap, task_group=tg)
        rt.fill(Hin.prod(), hin, tap, task_group=tg)
        rt.drain(Out.cons(), o, tap, task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())
