# SPDX-License-Identifier: Apache-2.0
"""decode_qkvffn1col_fit — P6.4a learning model: ONE reused join (fits cap, LOWERS OK).

Same 1-column shape as decode_qkvffn1col but with a SINGLE join `PF` (E-typed, 4-way)
reused across BOTH phases (decode_back_fused trick). MemTile(2,1) S2MM = Bcol(1) + PF(4)
= 5 <= 6 → AIECC lowers past AIEObjectFifoStatefulTransform. Purpose: dump the lowered
`aie.memtile_dma` + `aie.lock` form (via aie-opt --aie-objectFifo-stateful-transform) to
learn how (a) a 4-way join → memtile_dma BD-chain + locks, and (b) a 2-phase-reused fifo →
lock acquire/release phases. Then hand-write the TIME-MUXED 2-channel-set version by analogy.

center_body acquires PF_parts TWICE per loop (phase1 then phase2) = the reuse pattern.
"""
import numpy as np
from ml_dtypes import bfloat16

from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2, Tile


def my_decode_qkvffn1col_fit(dev, embed_dim=2048, col_offset=2):
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

    Bsrc = ObjectFifo(L1_E_ty, name="Bsrc", depth=1)
    Bcol = Bsrc.cons().forward(name="Bcol", depth=2, placement=Tile(col=col_offset, row=1))

    # ONE reused join PF (4-way, E-typed) for BOTH phases
    PF = ObjectFifo(L1_4E_ty, name="PF", depth=1)
    PF_parts = PF.prod().join(offsets=[r * E for r in range(R)],
                              placement=Tile(col=col_offset, row=1),
                              obj_types=[L1_E_ty for _ in range(R)])
    Oc = ObjectFifo(L1_E_ty, name="Oc", depth=2)     # phase1 drain (E)
    Cout = ObjectFifo(L1_E_ty, name="Cout", depth=2)  # phase2 drain (reduce E)

    workers = []

    def center_body(bcol, fpart, zero, add_fn):
        for _ in range_(0xFFFFFFFF):
            b = bcol.acquire(1); p = fpart.acquire(1)
            add_fn(b, zero, p, E); bcol.release(1); fpart.release(1)   # phase1
            b = bcol.acquire(1); p = fpart.acquire(1)
            add_fn(b, zero, p, E); bcol.release(1); fpart.release(1)   # phase2
    for r in range(R):
        workers.append(Worker(center_body,
            [Bcol.cons(), PF_parts[r].prod(), mk_zero(), add],
            placement=Tile(col=col_offset, row=2 + r)))

    # relay (col1): phase1 reduce 4E -> Oc ; phase2 reduce4 4E -> Cout (PF reused both)
    def relay_body(pf, oc, cout, zero, red_fn):
        for _ in range_(0xFFFFFFFF):
            p = pf.acquire(1); o = oc.acquire(1)
            # phase1: pass the 4E through (copy) — model as reduce into first E for simplicity of learning build
            red_fn(p, zero, o, E); pf.release(1); oc.release(1)
            p = pf.acquire(1); c = cout.acquire(1)
            red_fn(p, zero, c, E); pf.release(1); cout.release(1)      # phase2 reduce
    workers.append(Worker(relay_body, [PF.cons(), Oc.prod(), Cout.prod(), mk_zero(), reduce4],
                          placement=Tile(col=1, row=2)))

    e_tap = TensorAccessPattern(tensor_dims=(1, E), offset=0, sizes=[1, 1, 1, E], strides=[0, 0, 0, 1])

    rt = Runtime()
    with rt.sequence(L3_E_ty, L3_E_ty, L3_E_ty, L3_E_ty, L3_E_ty) as (oc, orr, x1, x2, _d):
        rt.start(*workers)
        tg = rt.task_group()
        rt.fill(Bsrc.prod(), x1, e_tap, task_group=tg)
        rt.fill(Bsrc.prod(), x2, e_tap, task_group=tg)
        rt.drain(Oc.cons(), oc, e_tap, task_group=tg, wait=True)
        rt.drain(Cout.cons(), orr, e_tap, task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())
