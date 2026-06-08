# SPDX-License-Identifier: Apache-2.0
"""BD-chain vs ObjectFifo weight-fill latency probe (#11 lever-2 go/no-go).

Question: does the FFLM-style rt.inline_ops BD-chain (N weight bands chained on
ONE shim S2MM channel) cost less / same / more wall-clock than N independent
ObjectFifo rt.fill transfers? This is the mechanism that lets the full fused
single-dispatch layer fit 7 weight stages on the 2-S2MM/col budget. We already
know it's expressible; this measures whether it's competitive.

One compute tile sums N input buffers (so all N fills must land before compute).
  MODE=fifo (default): N separate ObjectFifo + N rt.fill (one channel each).
  MODE=chain:          1 ObjectFifo, N BDs chained via rt.inline_ops on its channel.
Same compute, same bytes — the only difference is the fill mechanism.
"""
import os
import numpy as np
from ml_dtypes import bfloat16

from aie.dialects.aie import EndOp
from aie.dialects.aiex import (
    dma_configure_task_for, bds, shim_dma_bd, dma_start_task, dma_await_task,
)
from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2, Tile


def my_inline_fill_probe(dev, n=2048, n_fills=8):
    dev_ty = NPU1() if dev == "npu" else NPU2()
    bf = np.dtype[bfloat16]
    E = n
    mode = os.environ.get("PROBE_MODE", "fifo")

    L1_E_ty = np.ndarray[(E,), bf]
    L3_E_ty = np.ndarray[(E,), bf]
    # one big DDR source holding n_fills E-vectors back-to-back
    L3_W_ty = np.ndarray[(n_fills * E,), bf]

    add_k = Kernel("layer_fused_add_bf16", "layer_fused_relay.o",
                   [L1_E_ty, L1_E_ty, L1_E_ty, np.int32])

    Out = ObjectFifo(L1_E_ty, name="Out", depth=2)

    tap1 = TensorAccessPattern(tensor_dims=(1, E), offset=0,
                               sizes=[1, 1, 1, E], strides=[0, 0, 0, 1])

    if mode == "chain":
        # ONE fifo, depth n_fills; the worker acquires/sums all n_fills slots.
        In = ObjectFifo(L1_E_ty, name="In", depth=n_fills)

        def body(inp, out, add_fn):
            for _ in range_(0xFFFFFFFF):
                o = out.acquire(1)
                acc = inp.acquire(1)
                # seed o = acc(first), then add the rest
                add_fn(acc, acc, o, E)   # o = 2*first (placeholder math; perf-only probe)
                inp.release(1)
                for _ in range(n_fills - 1):
                    v = inp.acquire(1)
                    add_fn(o, v, o, E)
                    inp.release(1)
                out.release(1)

        worker = Worker(body, [In.cons(), Out.prod(), add_k],
                        placement=Tile(col=2, row=2))

        rt = Runtime()
        with rt.sequence(L3_W_ty, L3_E_ty) as (w, out_ddr):
            rt.start(worker)

            # BD-chain: N BDs on the In fifo's single S2MM channel via inline_ops.
            def fill_chain(w_rt):
                w_op = w_rt.op
                task = dma_configure_task_for("In", issue_token=True)
                with bds(task) as bd:
                    for i in range(n_fills):
                        with bd[i]:
                            shim_dma_bd(w_op, offset=i * E,
                                        sizes=[1, 1, 1, E], strides=[0, 0, 0, 1])
                            EndOp()
                dma_start_task(task)
                dma_await_task(task)

            tg = rt.task_group()
            rt.fill(In.prod(), w, tap1, task_group=tg)   # register fifo+channel
            rt.inline_ops(fill_chain, [w])
            rt.drain(Out.cons(), out_ddr, tap1, task_group=tg, wait=True)
            rt.finish_task_group(tg)

        return Program(dev_ty, rt).resolve_program(SequentialPlacer())

    # MODE=fifo: N independent fifos + N rt.fill (baseline)
    In = [ObjectFifo(L1_E_ty, name=f"In_{i}", depth=2) for i in range(n_fills)]

    def body(*args):
        ins = list(args[:n_fills]); out = args[n_fills]; add_fn = args[n_fills + 1]
        for _ in range_(0xFFFFFFFF):
            o = out.acquire(1)
            v0 = ins[0].acquire(1)
            add_fn(v0, v0, o, E); ins[0].release(1)
            for i in range(1, n_fills):
                v = ins[i].acquire(1); add_fn(o, v, o, E); ins[i].release(1)
            out.release(1)

    worker = Worker(body, [i.cons() for i in In] + [Out.prod(), add_k],
                    placement=Tile(col=2, row=2))

    rt = Runtime()
    with rt.sequence(L3_W_ty, L3_E_ty) as (w, out_ddr):
        rt.start(worker)
        tg = rt.task_group()
        for i in range(n_fills):
            tap_i = TensorAccessPattern(tensor_dims=(1, n_fills * E), offset=i * E,
                                        sizes=[1, 1, 1, E], strides=[0, 0, 0, 1])
            rt.fill(In[i].prod(), w, tap_i, task_group=tg)
        rt.drain(Out.cons(), out_ddr, tap1, task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())
