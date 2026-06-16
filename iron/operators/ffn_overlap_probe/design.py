# SPDX-License-Identifier: Apache-2.0
"""ffn_overlap_probe — DECISIVE overlap experiment (P3, path to FFLM 816us).

Question: does raw-aiex BD-chain weight delivery recover the compute/DMA overlap
that IRON ObjectFifo loses at PHASE BOUNDARIES of a fused multi-phase tile?

Background (measured): single-phase decode_ffn16_2mm is bandwidth-bound at FFLM's
46.3 GB/s ceiling. But fusing 2 stages on one tile via IRON ObjectFifo measured
1.51x SLOWER (reference_fusion_slower_than_multidispatch) — phase N+1's weight DMA
cannot prefetch behind phase N's compute because acquire/release + .split lockstep
inserts a barrier. FFLM streams all phases' weights continuously via raw BD-chains.

This probe runs the SAME validated dense FFN body TWICE (phase 0, phase 1) on the
same 16 center tiles, both reading the same broadcast activation, with DIFFERENT
weights per phase. Output = ffn(x, Wa) + ffn(x, Wb). Two weight-delivery modes
(env WMODE), byte-identical compute — the ONLY difference is the fill mechanism:
  WMODE=fifo  (default): per (col,stream) ObjectFifo, refilled for each phase via
              two rt.fill calls (the IRON baseline that loses overlap).
  WMODE=chain:           per (col,stream) ObjectFifo, BOTH phases' weight regions
              chained as N BDs on ONE shim S2MM channel via rt.inline_ops
              (inline_fill_probe / layer_dense weight_fill_chain pattern).
PASS = COMPLETED + out approx ffn(x,Wa)+ffn(x,Wb). LEVER = fifo_us - chain_us.
"""
import os
import numpy as np
from ml_dtypes import bfloat16

import aie.dialects.index as index
from aie.dialects.aie import T, EndOp
from aie.dialects.aiex import (
    dma_configure_task_for, bds, shim_dma_bd, dma_start_task, dma_await_task,
)
from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2, Tile


def my_ffn_overlap_probe(dev, embed_dim=2048, hidden_dim=8192, group_size=32,
                         m_input=4, num_cols=4, col_offset=2):
    dev_ty = NPU1() if dev == "npu" else NPU2()
    bf = np.dtype[bfloat16]
    u8 = np.dtype[np.uint8]
    E, H, g, m = embed_dim, hidden_dim, group_size, m_input
    NC, R = num_cols, 4
    NT = NC * R                                    # 16 tiles
    SPC = 2                                         # shim ingress fifos per column
    TPS = R // SPC                                  # tiles per stream = 2
    NPHASE = int(os.environ.get("NPHASE", "2"))   # bisect: NPHASE=1 isolates 2-phase plumbing

    Hc16 = H // NT
    gu_tiles = Hc16 // m
    dn_tiles = E // m
    PACKED = m * E // 2 + m * (E // g) * 2
    DN_PACKED = m * Hc16 // 2 + m * (Hc16 // g) * 2
    DN_SUB = PACKED // DN_PACKED
    assert DN_SUB * DN_PACKED == PACKED
    dn_elems = dn_tiles // DN_SUB
    assert dn_elems * DN_SUB == dn_tiles
    WT_PER_TILE = gu_tiles + gu_tiles + dn_elems   # rounds per tile per phase

    WMODE = os.environ.get("WMODE", "fifo")

    L1_E_ty  = np.ndarray[(E,), bf]
    L1_4E_ty = np.ndarray[(4 * E,), bf]
    L1_W_ty  = np.ndarray[(PACKED,), u8]
    L1_WS_ty = np.ndarray[(TPS * PACKED,), u8]     # one round of a stream = 2 tiles' block

    L3_E_ty  = np.ndarray[(E,), bf]
    # weights: NPHASE phases, per-column SPC streams, each WT_PER_TILE rounds x (TPS x PACKED)
    wstream = WT_PER_TILE * TPS * PACKED           # bytes per (phase,col,stream)
    wcol = SPC * wstream
    wphase = NC * wcol
    L3_W_ty  = np.ndarray[(NPHASE * wphase,), u8]

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

    _zn = [0]
    def mk_zero():
        b = Buffer(type=L1_E_ty, initial_value=np.zeros(E, dtype=bfloat16),
                   name=f"ovl_zero_{_zn[0]}")
        _zn[0] += 1
        return b

    # activation broadcast: DIRECT shim -> 4 tiles per column (no MemTile forward).
    Bsrc = [ObjectFifo(L1_E_ty, name=f"Bsrc_{c}", depth=2) for c in range(NC)]

    # weights: SPC streams per column, each split (North) to TPS per-tile fifos.
    # depth = NPHASE so both phases' rounds can be resident (chain-mode prefetch).
    Wsrc = [[ObjectFifo(L1_WS_ty, name=f"Wsrc_{c}_{s}", depth=2 * NPHASE)
             for s in range(SPC)] for c in range(NC)]
    Wf = [[Wsrc[c][s].cons().split(
              offsets=[t * PACKED for t in range(TPS)],
              placement=Tile(col=col_offset + c, row=1),
              obj_types=[L1_W_ty for _ in range(TPS)])
           for s in range(SPC)] for c in range(NC)]

    # per-column join PF[c] REUSED both phases: 4 row partials -> 4E @ MemTile(c,1)
    # depth=1: per-column, so phase0/phase1 of column c don't cross-contend with other
    # columns here (cross-column contention is only at the shared L2 FinalParts). MemTile(c,1)
    # also hosts the weight splits, so depth=2 here overflows its BD budget. decode_back_fused
    # runs the 2-phase skeleton at PF depth=1.
    PF, PF_parts = [], []
    for c in range(NC):
        pf = ObjectFifo(L1_4E_ty, name=f"PF_{c}", depth=1)
        parts = pf.prod().join(
            offsets=[r * E for r in range(R)],
            placement=Tile(col=col_offset + c, row=1),
            obj_types=[L1_E_ty for _ in range(R)])
        PF.append(pf); PF_parts.append(parts)

    # L2 join FinalParts: 4 col-partials -> 4E @ MemTile(6,1). depth=1 — IDENTICAL
    # to decode_ffn16_2mm: phases are accumulated AT THE TILE (below), so the reduce
    # tree sees exactly ONE element per tile per token, like the proven single-phase op.
    FinalParts = ObjectFifo(L1_4E_ty, name="FinalParts", depth=1)
    FP_parts = FinalParts.prod().join(
        offsets=[c * E for c in range(NC)],
        placement=Tile(col=6, row=1),
        obj_types=[L1_E_ty for _ in range(NC)])
    Cout = ObjectFifo(L1_E_ty, name="Cout", depth=2)

    workers = []

    # scratch (per-tile) buffer to hold phase-0's down output while phase-1 runs.
    def mk_tmp():
        b = Buffer(type=L1_E_ty, initial_value=np.zeros(E, dtype=bfloat16),
                   name=f"ovl_tmp_{_zn[0]}")
        _zn[0] += 1
        return b

    # center tile: run the dense FFN body for each phase (time-mux), ACCUMULATING the
    # phase down-outputs into ONE partial: phase0 -> o, phase>0 -> tmp then o += tmp.
    # The pp slot is acquired ONCE per token (NOT per phase) so downstream reduce is
    # byte-identical to decode_ffn16_2mm. At NPHASE=1 this collapses to that op exactly.
    def ffn_body(wf, bc, pp, tmp, gate_up_fn, silu_fn, down_fn, add_fn):
        for _ in range_(0xFFFFFFFF):
            o = pp.acquire(1)
            for ph in range(NPHASE):
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
                dst = o if ph == 0 else tmp
                for j in range_(dn_elems):
                    w = wf.acquire(1)
                    ro = index.casts(T.i32(), j) * DN_SUB * m
                    down_fn(m, ro, DN_SUB, w, dst)
                    wf.release(1)
                if ph > 0:
                    add_fn(o, tmp, o, E)         # o += tmp (this phase's down output)
            pp.release(1)

    for c in range(NC):
        for s in range(SPC):
            for tt in range(TPS):
                r = s * TPS + tt
                workers.append(Worker(
                    ffn_body,
                    [Wf[c][s][tt].cons(), Bsrc[c].cons(), PF_parts[c][r].prod(),
                     mk_tmp(), gate_up, silu, down, add],
                    placement=Tile(col=col_offset + c, row=2 + r)))

    # column reduce: PF[c] (one accumulated partial) -> FinalParts slot. Like ffn16_2mm.
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

    # final reduce: FinalParts (4 col-partials) -> Cout. Identical to ffn16_2mm.
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

    # weight tap for (phase, col, stream): region in the NPHASE*wphase DDR BO
    def w_tap(ph, c, s):
        off = ph * wphase + c * wcol + s * wstream
        return TensorAccessPattern(
            tensor_dims=(1, NPHASE * wphase), offset=off,
            sizes=[1, 1, 1, wstream], strides=[0, 0, 0, 1])

    # ELEMENT-ITERATING tap for the raw BD-chain: the Wsrc fifo element is
    # L1_WS_ty (= TPS*PACKED bytes); the .split() consumer signals its semaphore
    # ONCE PER ELEMENT. A flat single-blob BD (w_tap) delivers bytes but signals
    # only once → the split stalls. This pattern iterates WT_PER_TILE times over
    # element-sized chunks so the DMA fires the producer semaphore per element,
    # exactly as rt.fill auto-tiles. (Root cause of the chain+split hang.)
    WS_ELEM = TPS * PACKED
    def w_tap_iter(ph, c, s):
        off = ph * wphase + c * wcol + s * wstream
        return TensorAccessPattern(
            tensor_dims=(1, NPHASE * wphase), offset=off,
            sizes=[1, 1, WT_PER_TILE, WS_ELEM], strides=[0, 0, WS_ELEM, 1])

    rt = Runtime()
    with rt.sequence(L3_E_ty, L3_W_ty, L3_E_ty) as (o, w, x):
        rt.start(*workers)
        tg = rt.task_group()

        if WMODE == "chain":
            # rt.fill registers each fifo + its shim channel (proven inline_fill_probe
            # pattern: rt.fill then inline_ops chain on the SAME fifo). The chain emits
            # ALL NPHASE phases as element-ITERATING BDs so the .split() consumer gets
            # per-element semaphore signals (the flat single-blob BD stalled the split).
            for c in range(NC):
                for s in range(SPC):
                    rt.fill(Wsrc[c][s].prod(), w, w_tap_iter(0, c, s), task_group=tg)

            def fill_chain(w_rt):
                w_op = w_rt.op
                # Configure+start ALL channels FIRST, then await ALL (serial
                # start+await deadlocks: (c,0) await needs col c tiles to drain, but
                # PF[c] join also needs rows fed by (c,1) not yet started).
                tasks = []
                for c in range(NC):
                    for s in range(SPC):
                        task = dma_configure_task_for(f"Wsrc_{c}_{s}", issue_token=True)
                        with bds(task) as bd:
                            for ph in range(NPHASE):
                                tp = w_tap_iter(ph, c, s)
                                with bd[ph]:
                                    shim_dma_bd(w_op, offset=tp.offset,
                                                sizes=tp.sizes, strides=tp.strides)
                                    EndOp()
                        tasks.append(task)
                for task in tasks:
                    dma_start_task(task)
                for task in tasks:
                    dma_await_task(task)
            rt.inline_ops(fill_chain, [w])
        else:
            # FIFO baseline: two independent rt.fill per (col,stream), one per phase.
            for ph in range(NPHASE):
                for c in range(NC):
                    for s in range(SPC):
                        rt.fill(Wsrc[c][s].prod(), w, w_tap(ph, c, s), task_group=tg)

        for ph in range(NPHASE):
            for c in range(NC):
                rt.fill(Bsrc[c].prod(), x, e_tap, task_group=tg)
        rt.drain(Cout.cons(), o, e_tap, task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())
