# SPDX-License-Identifier: Apache-2.0
"""Standalone ANM (ADD + post-RMSNorm) validation probe — decode-layer step #17.

The post-attention "ANM" block: inpFF = o_out + inpL (residual add), then
ffn_in = rms_norm(inpFF) * gain. Uses layer_fused.cc kernels
layer_fused_add_bf16 (c=a+b) and layer_fused_rms_norm2_bf16
(out = in * invsqrt(mean(in^2)+1e-5) * gain).

This probe runs the full E-dim ADD then RMS on ONE tile (col 2) — RMS needs
the whole-E sum of squares, so the norm is over the full embedding, not a
column slice (cross-column reduction is a separate wiring step). Validated
numerically vs CPU. Establishes the ANM compute correctness.
"""
import numpy as np
from ml_dtypes import bfloat16

from aie.helpers.dialects.scf import _for as range_
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2, Tile


def my_anm(dev, embed_dim=2048, col=2, row=2, gain_data=None):
    dev_ty = NPU1() if dev == "npu" else NPU2()
    bf = np.dtype[bfloat16]
    E = embed_dim

    L1_E_ty = np.ndarray[(E,), bf]

    add = Kernel("layer_fused_add_bf16", "layer_fused_anm.o",
                 [L1_E_ty, L1_E_ty, L1_E_ty, np.int32])
    rms = Kernel("layer_fused_rms_norm2_bf16", "layer_fused_anm.o",
                 [L1_E_ty, L1_E_ty, L1_E_ty, np.int32])

    Oo  = ObjectFifo(L1_E_ty, name="Oo", depth=1)    # o_out (attn proj result)
    InpL = ObjectFifo(L1_E_ty, name="InpL", depth=1)  # residual input
    Out = ObjectFifo(L1_E_ty, name="Out", depth=1)    # ffn_in result

    # RMS gain is a STATIC weight (per-layer constant, like the RoPE LUT) →
    # passive Buffer (CDO-loaded once, zero runtime DMA), so it does NOT consume
    # a 3rd shim S2MM channel. Leaves Oo + InpL = 2 S2MM, within the cap.
    if gain_data is None:
        # deterministic known gain (matches the CPU reference in the probe)
        gain_data = ((np.arange(E) % 7) * 0.1 + 0.75).astype(bfloat16)
    gain_buf = Buffer(type=L1_E_ty, initial_value=gain_data, name="rms_gain")

    def body(oo, inpl, out, gain, add_fn, rms_fn):
        for _ in range_(0xFFFFFFFF):
            o = oo.acquire(1)
            l = inpl.acquire(1)
            r = out.acquire(1)
            add_fn(o, l, r, E)          # r = o_out + inpL (in place into out buf)
            rms_fn(r, gain, r, E)       # r = rms(r) * gain (gain = passive Buffer)
            oo.release(1); inpl.release(1); out.release(1)

    worker = Worker(body, [Oo.cons(), InpL.cons(), Out.prod(), gain_buf,
                           add, rms],
                    placement=Tile(col=col, row=row))

    L3 = np.ndarray[(E,), bf]
    rt = Runtime()
    with rt.sequence(L3, L3, L3) as (out, oo, inpl):
        rt.start(worker)
        tg = rt.task_group()
        rt.fill(Oo.prod(), oo, task_group=tg)
        rt.fill(InpL.prod(), inpl, task_group=tg)
        rt.drain(Out.cons(), out, task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())
