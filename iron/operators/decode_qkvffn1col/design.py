# SPDX-License-Identifier: Apache-2.0
"""decode_qkvffn1col — P6.4a DE-RISK vehicle for mid-level aie-dialect channel time-mux.

ONE column (4 center tiles at col2 rows2-5 + MemTile(2,1)), copy bodies. Two phases
time-muxed, each with its OWN MemTile join — the SMALLEST repro of the decode_qkvffn16
MemTile cap blocker:
  phase1 (concat): 4 tiles -> QF concat-join (4 S2MM, offsets r*E) -> drain bo0 (4E)
  phase2 (reduce): 4 tiles -> PF reduce-join (4 S2MM, offsets r*E) -> reduce4 -> drain bo1 (E)
MemTile(2,1) S2MM = Bcol(1) + QF(4) + PF(4) = 9 > 6  => AIECC MUST FAIL (channel cap).

This is the de-risk baseline: build it (expect AIECC 'input DMA channel exceeded'), then
HAND-EDIT the emitted .mlir to time-mux QF+PF onto ONE 4-S2MM channel-set via explicit
aie.memtile_dma BD-chains + lock phases, re-run aiecc, dispatch, verify BOTH outputs.
Copy bodies (add-with-zero) make outputs deterministic:
  bo0 (concat) = [x1 | x1 | x1 | x1]   (Bsrc filled with x1 for phase1, broadcast to 4 tiles)
  bo1 (reduce) = 4 * x2                 (Bsrc refilled with x2 for phase2, summed over 4 tiles)
"""
import numpy as np
from ml_dtypes import bfloat16

from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2, Tile


def my_decode_qkvffn1col(dev, embed_dim=2048, col_offset=2):
    dev_ty = NPU1() if dev == "npu" else NPU2()
    bf = np.dtype[bfloat16]
    E = embed_dim
    R = 4
    L1_E_ty = np.ndarray[(E,), bf]
    L1_4E_ty = np.ndarray[(R * E,), bf]
    L3_E_ty = np.ndarray[(E,), bf]
    L3_4E_ty = np.ndarray[(R * E,), bf]

    add = Kernel("layer_fused_add_bf16", "layer_fused_relay.o",
                 [L1_E_ty, L1_E_ty, L1_E_ty, np.int32])
    reduce4 = Kernel("layer_fused_reduce4_bf16", "layer_fused_relay.o",
                     [L1_4E_ty, L1_E_ty, L1_E_ty, np.int32])

    _zn = [0]
    def mk_zero():
        b = Buffer(type=L1_E_ty, initial_value=np.zeros(E, dtype=bfloat16),
                   name=f"z_{_zn[0]}"); _zn[0] += 1
        return b

    # broadcast input per phase (x1 then x2) to the 4 tiles via MemTile(2,1) forward
    Bsrc = ObjectFifo(L1_E_ty, name="Bsrc", depth=1)
    Bcol = Bsrc.cons().forward(name="Bcol", depth=2, placement=Tile(col=col_offset, row=1))

    # phase1 CONCAT join: 4 partials (E) -> 4E @ MemTile(2,1)
    QF = ObjectFifo(L1_4E_ty, name="QF", depth=2)
    QF_parts = QF.prod().join(offsets=[r * E for r in range(R)],
                              placement=Tile(col=col_offset, row=1),
                              obj_types=[L1_E_ty for _ in range(R)])
    # phase2 REDUCE join: 4 partials (E) -> 4E @ MemTile(2,1) -> reduce4 -> E
    PF = ObjectFifo(L1_4E_ty, name="PF", depth=1)
    PF_parts = PF.prod().join(offsets=[r * E for r in range(R)],
                              placement=Tile(col=col_offset, row=1),
                              obj_types=[L1_E_ty for _ in range(R)])
    Cout = ObjectFifo(L1_E_ty, name="Cout", depth=2)

    workers = []

    def center_body(bcol, qpart, fpart, zero, add_fn):
        for _ in range_(0xFFFFFFFF):
            b = bcol.acquire(1); q = qpart.acquire(1)
            add_fn(b, zero, q, E)                     # phase1: copy x1 -> concat slot
            bcol.release(1); qpart.release(1)
            b = bcol.acquire(1); f = fpart.acquire(1)
            add_fn(b, zero, f, E)                     # phase2: copy x2 -> reduce slot
            bcol.release(1); fpart.release(1)

    for r in range(R):
        workers.append(Worker(
            center_body,
            [Bcol.cons(), QF_parts[r].prod(), PF_parts[r].prod(), mk_zero(), add],
            placement=Tile(col=col_offset, row=2 + r)))

    # reduce worker (col 1): phase2 PF(4E) -> reduce4 -> Cout(E). QF drained directly.
    def red_body(pf, out, zero, red_fn):
        for _ in range_(0xFFFFFFFF):
            p = pf.acquire(1); o = out.acquire(1)
            red_fn(p, zero, o, E); pf.release(1); out.release(1)
    workers.append(Worker(red_body, [PF.cons(), Cout.prod(), mk_zero(), reduce4],
                          placement=Tile(col=1, row=2)))

    e_tap = TensorAccessPattern(tensor_dims=(1, E), offset=0, sizes=[1, 1, 1, E], strides=[0, 0, 0, 1])
    fe_tap = TensorAccessPattern(tensor_dims=(1, R * E), offset=0, sizes=[1, 1, 1, R * E], strides=[0, 0, 0, 1])

    rt = Runtime()
    # bo0=concat(4E), bo1=reduce(E), bo2=x1, bo3=x2, bo4=dummy
    with rt.sequence(L3_4E_ty, L3_E_ty, L3_E_ty, L3_E_ty, L3_E_ty) as (oc, orr, x1, x2, _d):
        rt.start(*workers)
        tg = rt.task_group()
        rt.fill(Bsrc.prod(), x1, e_tap, task_group=tg)   # phase1
        rt.fill(Bsrc.prod(), x2, e_tap, task_group=tg)   # phase2
        rt.drain(QF.cons(), oc, fe_tap, task_group=tg, wait=True)
        rt.drain(Cout.cons(), orr, e_tap, task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())
