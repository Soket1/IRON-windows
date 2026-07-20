# SPDX-License-Identifier: Apache-2.0
"""decode_qkvffn_probe — 50a RESOLVE-ONLY cap test (minimal).

Question (reference_qkvffn_merge_concat_vs_reduce_routing): can 16 center tiles each
drive TWO MM2S outputs in two time-muxed phases, one feeding a CONCAT-join (QKV-like)
and one feeding a REDUCE-tree (FFN-like), within the 2-S2MM+2-MM2S core cap and the
MemTile 6+6 cap? Copy bodies, ALL outputs E-typed (the concat-vs-reduce difference is
carried by the JOIN structure, not the element size — keeps ONE add + ONE reduce4 symbol,
avoiding the multi-type symbol redefinition). resolve-only: returns => placement OK.

center tile DMA: Bcol(1 S2MM) + Wf(1 S2MM) + qpart(1 MM2S, concat) + fpart(1 MM2S, reduce)
 = 2 S2MM + 2 MM2S = core cap. MemTile(c,1): Bcol-fwd + Wsplit + QF-join(4) + PF-join(4)...
 (this is the budget the resolve tests).
"""
import numpy as np
from ml_dtypes import bfloat16

from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2, Tile


def my_decode_qkvffn_probe(dev, embed_dim=2048, group_size=32, m_input=4,
                           num_cols=4, col_offset=2):
    dev_ty = NPU1() if dev == "npu" else NPU2()
    bf = np.dtype[bfloat16]
    u8 = np.dtype[np.uint8]
    E, g, m = embed_dim, group_size, m_input
    NC, R = num_cols, 4
    PACKED = m * E // 2 + m * (E // g) * 2
    qkv_tiles = 48

    L1_E_ty  = np.ndarray[(E,), bf]
    L1_4E_ty = np.ndarray[(R * E,), bf]
    L1_W_ty  = np.ndarray[(PACKED,), u8]
    L1_W4_ty = np.ndarray[(R * PACKED,), u8]
    L3_4E_ty = np.ndarray[(R * E,), bf]
    L3_CC_ty = np.ndarray[(NC * R * E,), bf]   # all 4 concat columns
    L3_E_ty  = np.ndarray[(E,), bf]
    L3_W_ty  = np.ndarray[(NC * qkv_tiles * R * PACKED,), u8]

    add = Kernel("layer_fused_add_bf16", "layer_fused_relay.o",
                 [L1_E_ty, L1_E_ty, L1_E_ty, np.int32])
    reduce4 = Kernel("layer_fused_reduce4_bf16", "layer_fused_relay.o",
                     [L1_4E_ty, L1_E_ty, L1_E_ty, np.int32])

    _zn = [0]
    def zero():
        b = Buffer(type=L1_E_ty, initial_value=np.zeros(E, dtype=bfloat16),
                   name=f"z_{_zn[0]}"); _zn[0] += 1
        return b

    Bsrc = [ObjectFifo(L1_E_ty, name=f"Bsrc_{c}", depth=1) for c in range(NC)]
    Bcol = [Bsrc[c].cons().forward(name=f"Bcol_{c}", depth=2,
                                   placement=Tile(col=col_offset + c, row=1))
            for c in range(NC)]
    Wsrc = [ObjectFifo(L1_W4_ty, name=f"Wsrc_{c}", depth=2) for c in range(NC)]
    Wf = [Wsrc[c].cons().split(offsets=[r * PACKED for r in range(R)],
                               placement=Tile(col=col_offset + c, row=1),
                               obj_types=[L1_W_ty for _ in range(R)])
          for c in range(NC)]

    # phase1 CONCAT join (QKV-like): 4 tile partials -> 4E @ MemTile(c,1), drained per-col
    QF, QF_parts = [], []
    for c in range(NC):
        qf = ObjectFifo(L1_4E_ty, name=f"QF_{c}", depth=2)
        parts = qf.prod().join(offsets=[r * E for r in range(R)],
                               placement=Tile(col=col_offset + c, row=1),
                               obj_types=[L1_E_ty for _ in range(R)])
        QF.append(qf); QF_parts.append(parts)
    # phase2 REDUCE join (FFN-like): 4 partials -> 4E @ MemTile(c,1) -> colred -> E
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

    def center_body(wf, bc, qpart, fpart, zq, zf, add_fn):
        for _ in range_(0xFFFFFFFF):
            x = bc.acquire(1); q = qpart.acquire(1)
            for j in range_(qkv_tiles):
                w = wf.acquire(1); wf.release(1)
            add_fn(x, zq, q, E)                       # phase1 -> concat slot
            bc.release(1); qpart.release(1)
            x2 = bc.acquire(1); f = fpart.acquire(1)
            for j in range_(qkv_tiles):
                w = wf.acquire(1); wf.release(1)
            add_fn(x2, zf, f, E)                       # phase2 -> reduce slot
            bc.release(1); fpart.release(1)

    for c in range(NC):
        for r in range(R):
            workers.append(Worker(
                center_body,
                [Wf[c][r].cons(), Bcol[c].cons(),
                 QF_parts[c][r].prod(), PF_parts[c][r].prod(), zero(), zero(), add],
                placement=Tile(col=col_offset + c, row=2 + r)))

    # col reduce relay (phase2 only); QF concat drained directly per column
    def colred_body(pf, fp, z, red):
        for _ in range_(0xFFFFFFFF):
            p = pf.acquire(1); o = fp.acquire(1)
            red(p, z, o, E); pf.release(1); fp.release(1)
    for c in range(NC):
        workers.append(Worker(colred_body, [PF[c].cons(), FP_parts[c].prod(), zero(), reduce4],
                              placement=Tile(col=1, row=2 + c)))

    def finalred_body(fp, out, z, red):
        for _ in range_(0xFFFFFFFF):
            f = fp.acquire(1); o = out.acquire(1)
            red(f, z, o, E); fp.release(1); out.release(1)
    workers.append(Worker(finalred_body, [FinalParts.cons(), Cout.prod(), zero(), reduce4],
                          placement=Tile(col=6, row=2)))

    e_tap = TensorAccessPattern(tensor_dims=(1, E), offset=0, sizes=[1, 1, 1, E], strides=[0, 0, 0, 1])
    wcol = qkv_tiles * R * PACKED
    def w_tap(c):
        return TensorAccessPattern(tensor_dims=(1, NC * wcol), offset=c * wcol,
                                   sizes=[1, 1, 1, wcol], strides=[0, 0, 0, 1])
    def cc_tap(c):  # column c's 4E chunk within the 16E concat output
        return TensorAccessPattern(tensor_dims=(1, NC * R * E), offset=c * R * E,
                                   sizes=[1, 1, 1, R * E], strides=[0, 0, 0, 1])

    rt = Runtime()
    # bo0=QKVconcat(16E, all 4 cols), bo1=W, bo2=x, bo3=ffn_reduce(E), bo4=dummy
    with rt.sequence(L3_CC_ty, L3_W_ty, L3_E_ty, L3_E_ty, L3_E_ty) as (qo, w, x, fo, _d):
        rt.start(*workers)
        tg = rt.task_group()
        for c in range(NC):
            rt.fill(Wsrc[c].prod(), w, w_tap(c), task_group=tg)
        for c in range(NC):
            rt.fill(Bsrc[c].prod(), x, e_tap, task_group=tg)
        for c in range(NC):
            rt.drain(QF[c].cons(), qo, cc_tap(c), task_group=tg, wait=True)  # 4 concat cols
        rt.drain(Cout.cons(), fo, e_tap, task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())
