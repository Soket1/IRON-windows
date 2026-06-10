# SPDX-License-Identifier: Apache-2.0
"""dma_fanout_probe (#34) — pure weight-ingress bandwidth vs shim COLUMN count.

Streams a FIXED total (28.3MB, = decode_ffn16 FFN weights) of weight blocks from
DDR through NCOL shim columns. Per column: DDR -> MemTile(col,1) split to R=4
worker tiles (rows 2-5) that PURE-DRAIN the blocks (no compute, no join), then emit
1 token (gates host completion). GB/s = 28.3MB / warm_time.

Compare COLS=[2,3,4,5] (N=4 = our current ingress, ~38 GB/s) vs [0,2,3,4,5,7]
(N=6 = FFLM weight-ingress fan-in) vs [0..7] (N=8). If GB/s scales with N -> the
38->46.6 bandwidth gap is DMA TOPOLOGY (free via more ingress columns). If flat ->
aggregate HW/per-channel limit. Mirrors decode_ffn16's proven Wsrc->split ingress.
"""
import numpy as np
from ml_dtypes import bfloat16

from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2, Tile


def my_dma_fanout_probe(dev, cols=(2, 3, 4, 5), embed_dim=2048, group_size=32,
                        m_input=4, chans_per_col=1):
    dev_ty = NPU1() if dev == "npu" else NPU2()
    bf = np.dtype[bfloat16]
    u8 = np.dtype[np.uint8]
    E, g, m = embed_dim, group_size, m_input
    NCOL = len(cols)
    R = 4
    F = chans_per_col                          # ingress fifos (shim MM2S) per column
    assert R % F == 0, "fifos_per_col must divide R=4"
    RF = R // F                                 # tiles fed per fifo
    PACKED = m * E // 2 + m * (E // g) * 2     # 4608
    # fixed total = 1536*4*PACKED = 28.31MB regardless of NCOL or F
    assert 1536 % NCOL == 0, "NCOL must divide 1536 for a fixed 28.3MB total"
    WT = 1536 // NCOL                           # rounds per fifo (= per tile)
    Ntok = 32

    L1_W_ty  = np.ndarray[(PACKED,), u8]
    L1_WF_ty = np.ndarray[(RF * PACKED,), u8]  # one round = RF rows' j-th block
    L1_O_ty  = np.ndarray[(Ntok,), bf]
    L1_4O_ty = np.ndarray[(R * Ntok,), bf]
    wfifo = WT * RF * PACKED                     # per-fifo bytes
    wcol = F * wfifo                             # per-column bytes (= WT*R*PACKED, F-indep)
    L3_W_ty = np.ndarray[(NCOL * wcol,), u8]
    L3_O_ty = np.ndarray[(NCOL * R * Ntok,), bf]

    # per-column weight ingress: F INDEPENDENT fifos per col (= F shim MM2S channels).
    # No join() needed — each fifo splits to RF tiles. This DOES express 2 MM2S/col
    # (F=2): #37's "join-required" framing was wrong. IRON's SequentialPlacer assigns
    # the F*NCOL fills to shim MM2S channels independently of the GEMV column.
    Wsrc = [[ObjectFifo(L1_WF_ty, name=f"Wsrc_{i}_{s}", depth=2) for s in range(F)]
            for i in range(NCOL)]
    Wf = [[Wsrc[i][s].cons().split(
              offsets=[r * PACKED for r in range(RF)],
              placement=Tile(col=cols[i], row=1),
              obj_types=[L1_W_ty for _ in range(RF)])
           for s in range(F)] for i in range(NCOL)]
    # per-column token JOIN (R workers -> 1 drain/col)
    ColTok, CT_parts = [], []
    for i in range(NCOL):
        ct = ObjectFifo(L1_4O_ty, name=f"ColTok_{i}", depth=1)
        parts = ct.prod().join(
            offsets=[r * Ntok for r in range(R)],
            placement=Tile(col=cols[i], row=1),
            obj_types=[L1_O_ty for _ in range(R)])
        ColTok.append(ct); CT_parts.append(parts)

    workers = []

    def drain_body(wf, of):
        for _ in range_(0xFFFFFFFF):
            for _b in range_(WT):
                wf.acquire(1)
                wf.release(1)
            of.acquire(1)        # emit 1 token AFTER all weights drained
            of.release(1)

    # tile (i, row 2+r); fifo s feeds tiles [s*RF .. s*RF+RF)
    for i in range(NCOL):
        for s in range(F):
            for rr in range(RF):
                r = s * RF + rr
                workers.append(Worker(
                    drain_body, [Wf[i][s][rr].cons(), CT_parts[i][r].prod()],
                    placement=Tile(col=cols[i], row=2 + r)))

    def w_tap(i, s):
        return TensorAccessPattern(tensor_dims=(1, NCOL * wcol),
                                   offset=i * wcol + s * wfifo,
                                   sizes=[1, 1, 1, wfifo], strides=[0, 0, 0, 1])

    def o_tap(i):
        return TensorAccessPattern(tensor_dims=(1, NCOL * R * Ntok),
                                   offset=i * R * Ntok,
                                   sizes=[1, 1, 1, R * Ntok], strides=[0, 0, 0, 1])

    rt = Runtime()
    # bo0 = tokens (drained, gates completion), bo1 = weights
    with rt.sequence(L3_O_ty, L3_W_ty) as (o, w):
        rt.start(*workers)
        tg = rt.task_group()
        for i in range(NCOL):
            for s in range(F):
                rt.fill(Wsrc[i][s].prod(), w, w_tap(i, s), task_group=tg)
        for i in range(NCOL):
            rt.drain(ColTok[i].cons(), o, o_tap(i), task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())
