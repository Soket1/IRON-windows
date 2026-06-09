# SPDX-License-Identifier: Apache-2.0
"""Streaming GEMV probe (#30 (B) realize): one tile, weights STREAM in as NB blocks
of 32 outputs each via a weight fifo (real weight-DMA, NOT L1-resident), X resident,
out[M_OUT]. Answers whether the broadcast compute win survives weight streaming
(the tile-resident probe could not — it reused one L1 block). `sym` selects the
bcast (column-major block) or dot (row-major block) per-block kernel."""
import numpy as np
from ml_dtypes import bfloat16

from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2, Tile


def my_gemv_stream_probe(dev, sym="layer_fused_gemv_bcast_blk_bf16",
                         M_OUT=512, K=2048, G=32):
    dev_ty = NPU1() if dev == "npu" else NPU2()
    bf = np.dtype[bfloat16]
    u8 = np.dtype[np.uint8]
    N = 32
    NB = M_OUT // N                       # number of streamed weight blocks
    BLK = K * N // 2 + N * (K // G) * 2   # one block bytes (weights + scales)
    # shape the L1 weight block 2D so the DMA BD dims stay <=1023 (BLK=36864 in one
    # dim exceeds the 1023 limit). 36864 = 72*512. Memory is contiguous → kernel
    # reads it as a flat uint8*.
    assert BLK % 512 == 0
    BR = BLK // 512                       # 72

    L1_W = np.ndarray[(BR, 512), u8]
    L1_X = np.ndarray[(K,), bf]
    L1_O = np.ndarray[(N,), bf]
    L3_W = np.ndarray[(NB * BLK,), u8]
    L3_X = np.ndarray[(K,), bf]
    L3_O = np.ndarray[(M_OUT,), bf]

    k = Kernel(sym, "layer_fused_relay.o", [L1_W, L1_X, L1_O])

    # W depth=1: one 36KB block fits L1 (depth=2 would be 72KB > 64KB). This
    # serializes weight-DMA with compute (no double-buffer) — a conservative
    # streaming floor; real FFN16 double-buffers via MemTile.
    W = ObjectFifo(L1_W, name="W", depth=1)     # streamed weight blocks
    X = ObjectFifo(L1_X, name="X", depth=1)
    O = ObjectFifo(L1_O, name="O", depth=2)

    def body(wf, xf, of, kern):
        for _ in range_(0xFFFFFFFF):
            xv = xf.acquire(1)
            for _b in range_(NB):
                wv = wf.acquire(1); ov = of.acquire(1)
                kern(wv, xv, ov)
                wf.release(1); of.release(1)
            xf.release(1)

    worker = Worker(body, [W.cons(), X.cons(), O.prod(), k],
                    placement=Tile(col=2, row=2))

    def tap1(n):
        return TensorAccessPattern(tensor_dims=(1, n), offset=0,
                                   sizes=[1, 1, 1, n], strides=[0, 0, 0, 1])

    rt = Runtime()
    with rt.sequence(L3_W, L3_X, L3_O) as (w, x, o):
        rt.start(worker)
        tg = rt.task_group()
        rt.fill(X.prod(), x, tap1(K), task_group=tg)
        # stream NB weight blocks; each block = BR*512 bytes, dims all <=1023
        rt.fill(W.prod(), w, TensorAccessPattern(
            tensor_dims=(NB, BLK), offset=0,
            sizes=[1, NB, BR, 512], strides=[0, BLK, 512, 1]), task_group=tg)
        rt.drain(O.cons(), o, TensorAccessPattern(
            tensor_dims=(NB, N), offset=0,
            sizes=[1, 1, NB, N], strides=[0, 0, N, 1]), task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())
