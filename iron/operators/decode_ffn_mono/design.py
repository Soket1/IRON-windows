# SPDX-License-Identifier: Apache-2.0
"""decode_ffn_mono — standalone single-tile FFN via the register-resident
monolithic kernel layer_fused_ffn_mono_bf16 (#12). One tile computes one tile's
FFN down-partial: gate/up/silu in registers, down SAXPY-accumulated into the
output (no lf_left/lf_right/lf_silu L1 statics). Validates the mono kernel logic
before integrating into the fused layer.

Weight element per hidden-chunk (m hidden rows):
  gud = [ gate (PK) ][ up (PK) ][ down_nibbles (m*E/2) ][ down_scl (E bf16) ]
  PK = m*E/2 + m*(E/G)*2 ; down_scl = E per-output scales for the chunk's group.
"""
import numpy as np
from ml_dtypes import bfloat16

import aie.dialects.index as index
from aie.dialects.aie import T
from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2, Tile


def my_decode_ffn_mono(dev, embed_dim=2048, hc16=512, group_size=32,
                       m_input=4, col_offset=2):
    dev_ty = NPU1() if dev == "npu" else NPU2()
    bf = np.dtype[bfloat16]
    u8 = np.dtype[np.uint8]
    E, Hc, g, m = embed_dim, hc16, group_size, m_input
    PK = m * E // 2 + m * (E // g) * 2                 # gate/up block
    DN_NIB = m * E // 2                                # down nibbles
    DN_SCL = E * 2                                     # down per-output scales (bytes)
    GUD = 2 * PK + DN_NIB + DN_SCL                     # weight element bytes
    n_chunks = Hc // m

    L1_E_ty = np.ndarray[(E,), bf]
    L1_GUD_ty = np.ndarray[(GUD,), u8]
    L3_O_ty = np.ndarray[(E,), bf]
    L3_W_ty = np.ndarray[(n_chunks * GUD,), u8]
    L3_X_ty = np.ndarray[(E,), bf]

    mono = Kernel("layer_fused_ffn_mono_bf16", "layer_fused_relay.o",
                  [np.int32, L1_GUD_ty, L1_E_ty, L1_E_ty])
    add = Kernel("layer_fused_add_bf16", "layer_fused_relay.o",
                 [L1_E_ty, L1_E_ty, L1_E_ty, np.int32])

    zero = Buffer(type=L1_E_ty, initial_value=np.zeros(E, dtype=bfloat16), name="mono_zero")

    Xsrc = ObjectFifo(L1_E_ty, name="Xsrc", depth=1)
    Xt = Xsrc.cons().forward(name="Xt", depth=2, placement=Tile(col=col_offset, row=1))
    Wf = ObjectFifo(L1_GUD_ty, name="Wf", depth=2)
    Out = ObjectFifo(L1_E_ty, name="Out", depth=2)

    def body(xf, wf, of, z, mono_fn, add_fn):
        for _ in range_(0xFFFFFFFF):
            x = xf.acquire(1); o = of.acquire(1)
            add_fn(z, z, o, E)                          # zero down_acc
            for _j in range_(n_chunks):
                w = wf.acquire(1)
                mono_fn(m, w, x, o)                     # accumulate
                wf.release(1)
            xf.release(1); of.release(1)

    worker = Worker(body, [Xt.cons(), Wf.cons(), Out.prod(), zero, mono, add],
                    placement=Tile(col=col_offset, row=2))

    e_tap = TensorAccessPattern(tensor_dims=(1, E), offset=0,
                                sizes=[1, 1, 1, E], strides=[0, 0, 0, 1])
    w_tap = TensorAccessPattern(tensor_dims=(1, n_chunks * GUD), offset=0,
                                sizes=[1, 1, 1, n_chunks * GUD], strides=[0, 0, 0, 1])
    rt = Runtime()
    with rt.sequence(L3_O_ty, L3_W_ty, L3_X_ty) as (o, w, x):
        rt.start(worker)
        tg = rt.task_group()
        rt.fill(Wf.prod(), w, w_tap, task_group=tg)
        rt.fill(Xsrc.prod(), x, e_tap, task_group=tg)
        rt.drain(Out.cons(), o, e_tap, task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())
