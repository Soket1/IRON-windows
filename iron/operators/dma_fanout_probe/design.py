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
    PACKED = m * E // 2 + m * (E // g) * 2     # 4608
    C = chans_per_col
    # fixed total = 1536*4*PACKED = 28.31MB regardless of NCOL or C
    assert 1536 % (NCOL * C) == 0, "NCOL*C must divide 1536 for a fixed 28.3MB total"
    ROUNDS_PER_CHAN = 1536 // (NCOL * C)       # rounds per (col, channel)
    Ntok = 32

    L1_W_ty  = np.ndarray[(PACKED,), u8]
    L1_W4_ty = np.ndarray[(R * PACKED,), u8]   # one round = 4 rows' j-th block
    L1_O_ty  = np.ndarray[(Ntok,), bf]
    L1_4O_ty = np.ndarray[(R * Ntok,), bf]     # 4 worker tokens joined per column
    wchan = ROUNDS_PER_CHAN * R * PACKED       # per-(col,channel) bytes
    wcol = C * wchan                            # per-column bytes
    L3_W_ty = np.ndarray[(NCOL * wcol,), u8]
    L3_O_ty = np.ndarray[(NCOL * R * Ntok,), bf]

    # per-column weight ingress: C shim channels per col, each fills
    # 1/C of the per-column region. IRON LIMITATION: join() is on PROD of a SINGLE
    # ObjectFifo (multi-prod via obj_types), not across DIFFERENT ObjectFifos. So
    # the C-channel fan-in cannot be expressed in IRON's high-level API at all.
    # This probe therefore tests the C=1 case (and falls back to N4/N6/N8 cols).
    # For C=2 on 4 center cols, would need raw-aiex BD chains.
    if C != 1:
        raise ValueError(f"chans_per_col={C} unsupported in IRON (only C=1): join() requires a single ObjectFifo prod.")
    Wsrc = [ObjectFifo(L1_W4_ty, name=f"Wsrc_{i}", depth=2) for i in range(NCOL)]
    Wf = [Wsrc[i].cons().split(
            offsets=[r * PACKED for r in range(R)],
            placement=Tile(col=cols[i], row=1),
            obj_types=[L1_W_ty for _ in range(R)])
          for i in range(NCOL)]
    # per-column token JOIN (4 workers -> 1 drain/col) keeps shim endpoints to
    # NCOL*C fills + NCOL drains = 12 channels @ N=6 (FFLM-faithful), not 4/col.
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
            for _b in range_(ROUNDS_PER_CHAN * C):
                wf.acquire(1)
                wf.release(1)
            of.acquire(1)        # emit 1 token AFTER all weights drained
            of.release(1)

    for i in range(NCOL):
        for r in range(R):
            workers.append(Worker(
                drain_body, [Wf[i][r].cons(), CT_parts[i][r].prod()],
                placement=Tile(col=cols[i], row=2 + r)))

    def w_tap(i):
        return TensorAccessPattern(tensor_dims=(1, NCOL * wcol),
                                   offset=i * wcol,
                                   sizes=[1, 1, 1, wcol], strides=[0, 0, 0, 1])

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
            rt.fill(Wsrc[i].prod(), w, w_tap(i), task_group=tg)
        for i in range(NCOL):
            rt.drain(ColTok[i].cons(), o, o_tap(i), task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())
