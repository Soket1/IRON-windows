# SPDX-License-Identifier: Apache-2.0
"""Standalone numerical validation of the on-NPU RoPE kernel (rope.cc).

Wraps `rope(input, lut, output, dims)` (TWO_HALVES = HF/NeoX convention, the one
ggml/Llama uses) on one central tile. Output is the FIRST rt.sequence arg so it
lands in bo0 (which the replay harness dumps to file for CPU comparison).

Two-halves math (per head, HALF = head_dim/2):
  out[j]      = in[j]*cos_j - in[HALF+j]*sin_j
  out[HALF+j] = in[j]*sin_j + in[HALF+j]*cos_j
lut is interleaved [cos_0, sin_0, cos_1, sin_1, ...] (head_dim elems = HALF pairs).
"""
import numpy as np
from ml_dtypes import bfloat16

from aie.helpers.dialects.scf import _for as range_
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2, Tile


def my_rope(dev, head_dim=64, col=2, row=2):
    dev_ty = NPU1() if dev == "npu" else NPU2()
    dims = head_dim                       # one head
    bf = np.dtype[bfloat16]

    IN_ty = np.ndarray[(dims,), bf]
    LUT_ty = np.ndarray[(dims,), bf]      # interleaved cos/sin, HALF pairs
    OUT_ty = np.ndarray[(dims,), bf]

    rope = Kernel("rope", "rope_th.o", [IN_ty, LUT_ty, OUT_ty, np.int32])

    I = ObjectFifo(IN_ty, name="I", depth=1)
    L = ObjectFifo(LUT_ty, name="L", depth=1)
    O = ObjectFifo(OUT_ty, name="O", depth=1)

    def body(i_fifo, l_fifo, o_fifo, rope_fn):
        for _ in range_(0xFFFFFFFF):
            i = i_fifo.acquire(1)
            l = l_fifo.acquire(1)
            o = o_fifo.acquire(1)
            rope_fn(i, l, o, dims)
            i_fifo.release(1)
            l_fifo.release(1)
            o_fifo.release(1)

    worker = Worker(body, [I.cons(), L.cons(), O.prod(), rope],
                    placement=Tile(col=col, row=row))

    L3_O = np.ndarray[(dims,), bf]
    L3_I = np.ndarray[(dims,), bf]
    L3_L = np.ndarray[(dims,), bf]

    rt = Runtime()
    # Output FIRST so it is bo0 (the harness dumps bo0 to file).
    with rt.sequence(L3_O, L3_I, L3_L) as (out, inp, lut):
        rt.start(worker)
        tg = rt.task_group()
        rt.fill(I.prod(), inp, task_group=tg)
        rt.fill(L.prod(), lut, task_group=tg)
        rt.drain(O.cons(), out, task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())
