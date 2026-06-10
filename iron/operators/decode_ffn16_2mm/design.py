# SPDX-License-Identifier: Apache-2.0
"""decode_ffn16_2mm — decode_ffn16 with 2 INDEPENDENT weight ingress fifos per
center column (= 2 shim MM2S/col = 8 MM2S total, FFLM's 4cols x 2MM2S layout).

ONLY the weight ingress topology differs from decode_ffn16: per column we use 2
Wsrc fifos (each split to 2 of the 4 row tiles) instead of 1 Wsrc split to 4. This
doubles the shim MM2S ingress channels (4 -> 8), which dma_fanout F=2 measured at
47.0 GB/s (vs 41.4 @ 4ch). The GEMV body, SwiGLU, down, reduce, and activation
broadcast are byte-identical to decode_ffn16. MemTile S2MM budget: 2 weight + 4
down-join = 6 = exactly the cap.

Weight W.bin layout: per column, TWO contiguous half-streams. Stream s carries the
interleaved weight rounds for tiles {2s, 2s+1} (rows 2+2s, 3+2s). Each round =
2*PACKED (the j-th block of the stream's 2 tiles). See validate_ffn16_2mm.py.
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


def my_decode_ffn16_2mm(dev, embed_dim=2048, hidden_dim=8192, group_size=32,
                        m_input=4, num_cols=4, col_offset=2):
    dev_ty = NPU1() if dev == "npu" else NPU2()
    bf = np.dtype[bfloat16]
    u8 = np.dtype[np.uint8]
    E, H, g, m = embed_dim, hidden_dim, group_size, m_input
    NC, R = num_cols, 4
    NT = NC * R                                   # 16 tiles
    SPC = 2                                        # shim ingress fifos per column
    TPS = R // SPC                                 # tiles per stream = 2

    Hc16 = H // NT
    gu_tiles = Hc16 // m
    dn_tiles = E // m
    PACKED = m * E // 2 + m * (E // g) * 2
    DN_PACKED = m * Hc16 // 2 + m * (Hc16 // g) * 2
    DN_SUB = PACKED // DN_PACKED
    assert DN_SUB * DN_PACKED == PACKED
    dn_elems = dn_tiles // DN_SUB
    assert dn_elems * DN_SUB == dn_tiles
    WT_PER_TILE = gu_tiles + gu_tiles + dn_elems  # 384

    L1_E_ty  = np.ndarray[(E,), bf]
    L1_4E_ty = np.ndarray[(4 * E,), bf]
    L1_W_ty  = np.ndarray[(PACKED,), u8]
    L1_WS_ty = np.ndarray[(TPS * PACKED,), u8]    # one round of a stream = 2 tiles' block
    L1_DW_ty = np.ndarray[(DN_PACKED,), u8]

    L3_E_ty  = np.ndarray[(E,), bf]
    L3_D_ty  = np.ndarray[(64,), bf]
    # per-column: SPC streams, each WT_PER_TILE rounds x (TPS x PACKED)
    L3_W_ty  = np.ndarray[(NC * SPC * WT_PER_TILE * TPS * PACKED,), u8]

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

    # activation broadcast: DIRECT shim -> 4 tiles per column (NO MemTile forward).
    # This frees 1 MemTile S2MM so the 2 weight streams fit (MemTile S2MM = 2 weight
    # + 4 down-join = 6 = cap; activation lands in each tile's 2nd S2MM instead).
    Bsrc = [ObjectFifo(L1_E_ty, name=f"Bsrc_{c}", depth=2) for c in range(NC)]

    # --- weights: SPC streams per column, each split (North) to TPS per-tile fifos
    Wsrc = [[ObjectFifo(L1_WS_ty, name=f"Wsrc_{c}_{s}", depth=2) for s in range(SPC)]
            for c in range(NC)]
    Wf = [[Wsrc[c][s].cons().split(
              offsets=[t * PACKED for t in range(TPS)],
              placement=Tile(col=col_offset + c, row=1),
              obj_types=[L1_W_ty for _ in range(TPS)])
           for s in range(SPC)] for c in range(NC)]

    # down-partial L1 join (unchanged): per column, 4 row partials -> 4E @ MemTile(c,1)
    PF, PF_parts = [], []
    for c in range(NC):
        pf = ObjectFifo(L1_4E_ty, name=f"PF_{c}", depth=1)
        parts = pf.prod().join(
            offsets=[r * E for r in range(R)],
            placement=Tile(col=col_offset + c, row=1),
            obj_types=[L1_E_ty for _ in range(R)])
        PF.append(pf); PF_parts.append(parts)

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
            for j in range_(gu_tiles):
                w = wf.acquire(1)
                gate_up_fn(m, index.casts(T.i32(), j) * m, w, b, 0)
                wf.release(1)
            for j in range_(gu_tiles):
                w = wf.acquire(1)
                gate_up_fn(m, index.casts(T.i32(), j) * m, w, b, 1)
                wf.release(1)
            bc.release(1)
            silu_fn(Hc16)
            o = pp.acquire(1)
            for j in range_(dn_elems):
                w = wf.acquire(1)
                ro = index.casts(T.i32(), j) * DN_SUB * m
                down_fn(m, ro, DN_SUB, w, o)
                wf.release(1)
            pp.release(1)

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

    # tile (c, 2+r); stream s feeds tiles r in {s*TPS .. s*TPS+TPS}
    for c in range(NC):
        for s in range(SPC):
            for tt in range(TPS):
                r = s * TPS + tt
                if DBG_COPY:
                    workers.append(Worker(
                        copy_body,
                        [Wf[c][s][tt].cons(), Bsrc[c].cons(), PF_parts[c][r].prod(),
                         add, mk_zero()],
                        placement=Tile(col=col_offset + c, row=2 + r)))
                else:
                    workers.append(Worker(
                        ffn_body,
                        [Wf[c][s][tt].cons(), Bsrc[c].cons(), PF_parts[c][r].prod(),
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
    # per-(col, stream) weight region: WT_PER_TILE rounds x (TPS x PACKED)
    wstream = WT_PER_TILE * TPS * PACKED
    wcol = SPC * wstream

    def w_tap(c, s):
        off = c * wcol + s * wstream
        return TensorAccessPattern(
            tensor_dims=(1, NC * wcol), offset=off,
            sizes=[1, 1, 1, wstream], strides=[0, 0, 0, 1])

    rt = Runtime()
    with rt.sequence(L3_E_ty, L3_W_ty, L3_E_ty, L3_D_ty, L3_D_ty) as (o, w, x, _d3, _d4):
        rt.start(*workers)
        tg = rt.task_group()
        for c in range(NC):
            for s in range(SPC):
                rt.fill(Wsrc[c][s].prod(), w, w_tap(c, s), task_group=tg)
        for c in range(NC):
            rt.fill(Bsrc[c].prod(), x, e_tap, task_group=tg)
        rt.drain(Cout.cons(), o, e_tap, task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())
