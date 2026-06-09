# SPDX-License-Identifier: Apache-2.0
"""Tile-sized GEMV probe (#30 (B)): one tile computes out[M_OUT] = W @ x[K] over
a REAL per-tile FFN workload (M_OUT=512, K=2048) so the kernel is COMPUTE-bound,
not dispatch-floor-bound. `sym` selects the broadcast (column-major) or
dot-product (row-major) kernel; same I/O for an apples-to-apples A/B."""
import numpy as np
from ml_dtypes import bfloat16

from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2, Tile


def my_gemv_tile_probe(dev, sym="layer_fused_gemv_bcast_tile_bf16",
                       M_OUT=512, K=2048, G=32):
    dev_ty = NPU1() if dev == "npu" else NPU2()
    bf = np.dtype[bfloat16]
    u8 = np.dtype[np.uint8]
    WB = M_OUT * K // 2
    SB = M_OUT * (K // G)
    WSB = WB + SB * 2                # weights + scales(bf16) bytes (same both layouts)

    L1_WS = np.ndarray[(WSB,), u8]
    L1_X = np.ndarray[(K,), bf]
    L1_O = np.ndarray[(M_OUT,), bf]
    L3_WS = np.ndarray[(WSB,), u8]
    L3_X = np.ndarray[(K,), bf]
    L3_O = np.ndarray[(M_OUT,), bf]

    k = Kernel(sym, "layer_fused_relay.o", [L1_WS, L1_X, L1_O])

    WS = ObjectFifo(L1_WS, name="WS", depth=1)
    X = ObjectFifo(L1_X, name="X", depth=1)
    O = ObjectFifo(L1_O, name="O", depth=2)

    def body(ws, x, o, kern):
        for _ in range_(0xFFFFFFFF):
            wv = ws.acquire(1); xv = x.acquire(1); ov = o.acquire(1)
            kern(wv, xv, ov)
            ws.release(1); x.release(1); o.release(1)

    worker = Worker(body, [WS.cons(), X.cons(), O.prod(), k],
                    placement=Tile(col=2, row=2))

    def tap(n):
        return TensorAccessPattern(tensor_dims=(1, n), offset=0,
                                   sizes=[1, 1, 1, n], strides=[0, 0, 0, 1])

    rt = Runtime()
    with rt.sequence(L3_WS, L3_X, L3_O) as (ws, x, o):
        rt.start(worker)
        tg = rt.task_group()
        rt.fill(WS.prod(), ws, tap(WSB), task_group=tg)
        rt.fill(X.prod(), x, tap(K), task_group=tg)
        rt.drain(O.cons(), o, tap(M_OUT), task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())
