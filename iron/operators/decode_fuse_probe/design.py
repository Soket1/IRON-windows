# SPDX-License-Identifier: Apache-2.0
"""decode_fuse_probe — #11.1 probe0: minimal on-chip HUB ROUND-TRIP in one dispatch.

Tests the core fusion unknown (NOT covered by D2.1 single-tile time-mux): can a
center tile do TWO phases time-muxed, with an on-chip hub tile processing the
intermediate BETWEEN the phases, in ONE dispatch, no deadlock?

  IN(DDR) --> center.phase1(copy) --> MID --> hub(copy) --> ACT2 -->
              center.phase2(copy) --> OUT(DDR)

center DMA: IN(S2MM) + ACT2(S2MM) = 2 in ; MID(MM2S) + OUT(MM2S) = 2 out  (= cap)
hub    DMA: MID(S2MM) ; ACT2(MM2S)                                        (1+1)
All copies via layer_fused_add_bf16(a, zero, out, n). Output must equal input.
"""
import numpy as np
from ml_dtypes import bfloat16

from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2, Tile


def my_decode_fuse_probe(dev, n=256, col_offset=2):
    dev_ty = NPU1() if dev == "npu" else NPU2()
    bf = np.dtype[bfloat16]
    N = n
    L1_ty = np.ndarray[(N,), bf]
    L3_ty = np.ndarray[(N,), bf]

    cp = Kernel("layer_fused_add_bf16", "layer_fused_relay.o",
                [L1_ty, L1_ty, L1_ty, np.int32])

    _zn = [0]
    def mk_zero():
        b = Buffer(type=L1_ty, initial_value=np.zeros(N, dtype=bfloat16),
                   name=f"fp_zero_{_zn[0]}"); _zn[0] += 1
        return b

    IN = ObjectFifo(L1_ty, name="IN", depth=2)
    MID = ObjectFifo(L1_ty, name="MID", depth=2)     # center -> hub
    ACT2 = ObjectFifo(L1_ty, name="ACT2", depth=2)   # hub -> center
    OUT = ObjectFifo(L1_ty, name="OUT", depth=2)

    def center_body(i_in, mid_out, act_in, o_out, zero, cp_fn):
        for _ in range_(0xFFFFFFFF):
            i = i_in.acquire(1); m = mid_out.acquire(1)
            cp_fn(i, zero, m, N)                       # phase1: IN -> MID
            i_in.release(1); mid_out.release(1)
            a = act_in.acquire(1); o = o_out.acquire(1)
            cp_fn(a, zero, o, N)                       # phase2: ACT2 -> OUT
            act_in.release(1); o_out.release(1)

    def hub_body(mid_in, act_out, zero, cp_fn):
        for _ in range_(0xFFFFFFFF):
            m = mid_in.acquire(1); a = act_out.acquire(1)
            cp_fn(m, zero, a, N)                       # MID -> ACT2
            mid_in.release(1); act_out.release(1)

    center = Worker(center_body,
                    [IN.cons(), MID.prod(), ACT2.cons(), OUT.prod(), mk_zero(), cp],
                    placement=Tile(col=col_offset, row=2))
    hub = Worker(hub_body, [MID.cons(), ACT2.prod(), mk_zero(), cp],
                 placement=Tile(col=col_offset, row=3))

    tap = TensorAccessPattern(tensor_dims=(1, N), offset=0,
                              sizes=[1, 1, 1, N], strides=[0, 0, 0, 1])
    rt = Runtime()
    with rt.sequence(L3_ty, L3_ty) as (o, x):
        rt.start(center, hub)
        tg = rt.task_group()
        rt.fill(IN.prod(), x, tap, task_group=tg)
        rt.drain(OUT.cons(), o, tap, task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())
