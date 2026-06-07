# SPDX-License-Identifier: Apache-2.0
"""decode_fuse_col — #11.1 probe1: 1-column back-half plumbing skeleton (copy bodies).

Adds MemTile-mediated hub coupling on top of probe0: broadcast-from-DDR (phase1)
AND broadcast-from-HUB (phase2, on-chip source) through MemTile(col,1), plus a
gather (4 tiles -> reduce -> hub / -> out). Proves the column dataflow of the
fused back-half before scaling to 16 tiles + 2-level gather + dual hubs.

  x(DDR) -bcast-> 4 tiles(copy) -join-> reduce -> HUB(copy) -bcast-> 4 tiles(copy)
                                                            -join-> reduce -> out(DDR)

ONE PF join fifo is REUSED across both phases (IRON = 1 ch/fifo, no temporal
packing, so 2 joins would be 8 S2MM > MemTile cap 6). MemTile(col,1):
  2 broadcast-forward fills (x, act2) + 1 join (4 S2MM) = 6 S2MM (= cap).
Predictable: phase1 reduce(4 copies of x)=4x -> hub -> phase2 reduce(4 of 4x)=16x.
"""
import numpy as np
from ml_dtypes import bfloat16

from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2, Tile


def my_decode_fuse_col(dev, n=256, col_offset=2):
    dev_ty = NPU1() if dev == "npu" else NPU2()
    bf = np.dtype[bfloat16]
    E = n
    R = 4
    L1_E_ty = np.ndarray[(E,), bf]
    L1_4E_ty = np.ndarray[(R * E,), bf]
    L3_ty = np.ndarray[(E,), bf]

    cp = Kernel("layer_fused_add_bf16", "layer_fused_relay.o",
                [L1_E_ty, L1_E_ty, L1_E_ty, np.int32])
    reduce4 = Kernel("layer_fused_reduce4_bf16", "layer_fused_relay.o",
                     [L1_4E_ty, L1_E_ty, L1_E_ty, np.int32])

    _zn = [0]
    def mk_zero():
        b = Buffer(type=L1_E_ty, initial_value=np.zeros(E, dtype=bfloat16),
                   name=f"fc_zero_{_zn[0]}"); _zn[0] += 1
        return b

    cc = col_offset
    # broadcast forwards through MemTile(cc,1): phase1 x (DDR), phase2 act2 (hub)
    Bsrc = ObjectFifo(L1_E_ty, name="Bsrc", depth=1)
    Bcol = Bsrc.cons().forward(name="Bcol", depth=2, placement=Tile(col=cc, row=1))
    ACT2 = ObjectFifo(L1_E_ty, name="ACT2", depth=1)            # hub -> MemTile
    ACT2col = ACT2.cons().forward(name="ACT2col", depth=2, placement=Tile(col=cc, row=1))

    # ONE join fifo, reused both phases: 4 tile partials -> 4E @ MemTile(cc,1)
    PF = ObjectFifo(L1_4E_ty, name="PF", depth=1)
    PF_parts = PF.prod().join(offsets=[r * E for r in range(R)],
                              placement=Tile(col=cc, row=1),
                              obj_types=[L1_E_ty for _ in range(R)])
    Red1 = ObjectFifo(L1_E_ty, name="Red1", depth=2)           # colred phase1 -> hub
    Out = ObjectFifo(L1_E_ty, name="Out", depth=2)             # colred phase2 -> DDR

    workers = []

    def center_body(bc, ac, pp, zero, cp_fn):
        for _ in range_(0xFFFFFFFF):
            b = bc.acquire(1); p = pp.acquire(1)
            cp_fn(b, zero, p, E)                                # phase1: x -> partial
            bc.release(1); pp.release(1)
            a = ac.acquire(1); p = pp.acquire(1)
            cp_fn(a, zero, p, E)                                # phase2: act2 -> partial
            ac.release(1); pp.release(1)

    for r in range(R):
        workers.append(Worker(
            center_body,
            [Bcol.cons(), ACT2col.cons(), PF_parts[r].prod(), mk_zero(), cp],
            placement=Tile(col=cc, row=2 + r)))

    def colred_body(pf, r1, out, zero, red_fn):
        for _ in range_(0xFFFFFFFF):
            f = pf.acquire(1); r = r1.acquire(1)
            red_fn(f, zero, r, E)                               # phase1 reduce -> hub
            pf.release(1); r1.release(1)
            f = pf.acquire(1); o = out.acquire(1)
            red_fn(f, zero, o, E)                               # phase2 reduce -> out
            pf.release(1); out.release(1)

    workers.append(Worker(colred_body, [PF.cons(), Red1.prod(), Out.prod(),
                                        mk_zero(), reduce4],
                          placement=Tile(col=1, row=2)))

    def hub_body(r1, ac, zero, cp_fn):
        for _ in range_(0xFFFFFFFF):
            r = r1.acquire(1); a = ac.acquire(1)
            cp_fn(r, zero, a, E)                                # O -> ffn_in (copy)
            r1.release(1); ac.release(1)

    workers.append(Worker(hub_body, [Red1.cons(), ACT2.prod(), mk_zero(), cp],
                          placement=Tile(col=1, row=3)))

    tap = TensorAccessPattern(tensor_dims=(1, E), offset=0,
                              sizes=[1, 1, 1, E], strides=[0, 0, 0, 1])
    rt = Runtime()
    with rt.sequence(L3_ty, L3_ty) as (o, x):
        rt.start(*workers)
        tg = rt.task_group()
        rt.fill(Bsrc.prod(), x, tap, task_group=tg)
        rt.drain(Out.cons(), o, tap, task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())
