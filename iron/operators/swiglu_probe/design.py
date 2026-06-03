# SPDX-License-Identifier: Apache-2.0
"""Standalone SwiGLU validation probe — decode-layer step #17 (FFN block).

TWO tiles in one column (FFLM-style row split — one tile can't hold gate+up+down
weights within the 2-S2MM cap):

  tile (col, 2): gate (E->Hc) + up (E->Hc) + silu_mul → SILU (on-chip handoff)
                 inputs: W fifo (gate then up weights) + FFIN activation = 2 S2MM
  tile (col, 3): down partial GEMV (Hc->E) over SILU → OUT
                 inputs: DNW weights (1 S2MM) + SILU (on-chip from tile 2)

silu(gate)*up written to a SILU ObjectFifo that the down tile consumes on-chip.
NOTE: layer_fused gate_up/down SUBTRACT 8 (w=(nibble-8)*scale) — CPU ref uses -8.
Validated numerically vs CPU.
"""
import numpy as np
from ml_dtypes import bfloat16

import aie.dialects.index as index
from aie.dialects.aie import T
from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2, Tile


def my_swiglu(dev, embed_dim=2048, hidden_per_col=512, group_size=32,
              m_gemv=2, col=2):
    dev_ty = NPU1() if dev == "npu" else NPU2()
    bf = np.dtype[bfloat16]; u8 = np.dtype[np.uint8]
    E = embed_dim; Hc = hidden_per_col

    gu_groups = E // group_size
    gu_packed = m_gemv * E // 2 + m_gemv * gu_groups * 2
    gu_tiles = Hc // m_gemv
    dn_groups = Hc // group_size
    dn_packed = m_gemv * Hc // 2 + m_gemv * dn_groups * 2
    dn_tiles = E // m_gemv

    L1_GUW_ty = np.ndarray[(gu_packed,), u8]
    L1_FFIN_ty = np.ndarray[(E,), bf]
    L1_DNW_ty = np.ndarray[(dn_packed,), u8]
    L1_SILU_ty = np.ndarray[(Hc,), bf]
    L1_OUT_ty = np.ndarray[(E,), bf]

    gate_up = Kernel("layer_fused_gate_up_bf16", "layer_fused_swiglu.o",
                     [np.int32, np.int32, L1_GUW_ty, L1_FFIN_ty, np.int32])
    silu = Kernel("layer_fused_silu_mul_bf16", "layer_fused_swiglu.o",
                  [L1_SILU_ty, np.int32])
    down = Kernel("layer_fused_down_partial_bf16", "layer_fused_swiglu.o",
                  [np.int32, np.int32, L1_DNW_ty, L1_SILU_ty, L1_OUT_ty])

    GUW = ObjectFifo(L1_GUW_ty, name="GUW", depth=2)    # gate then up weights
    FFIN = ObjectFifo(L1_FFIN_ty, name="FFIN", depth=1)
    SILU = ObjectFifo(L1_SILU_ty, name="SILU", depth=2)  # tile2 → tile3 on-chip
    DNW = ObjectFifo(L1_DNW_ty, name="DNW", depth=2)
    OUT = ObjectFifo(L1_OUT_ty, name="OUT", depth=2)

    # tile (col,2): gate + up + silu_mul → SILU
    def gu_body(guw, ffin, silu_p, gate_up_fn, silu_fn):
        for _ in range_(0xFFFFFFFF):
            b = ffin.acquire(1)
            for j in range_(gu_tiles):                # gate → lf_left
                w = guw.acquire(1)
                gate_up_fn(m_gemv, index.casts(T.i32(), j) * m_gemv, w, b, 0)
                guw.release(1)
            for j in range_(gu_tiles):                # up → lf_right
                w = guw.acquire(1)
                gate_up_fn(m_gemv, index.casts(T.i32(), j) * m_gemv, w, b, 1)
                guw.release(1)
            ffin.release(1)
            s = silu_p.acquire(1)
            silu_fn(s, Hc)                            # silu(gate)*up → SILU
            silu_p.release(1)

    gu_worker = Worker(gu_body, [GUW.cons(), FFIN.cons(), SILU.prod(),
                                 gate_up, silu],
                       placement=Tile(col=col, row=2))

    # tile (col,3): down partial GEMV over SILU → OUT
    def dn_body(dnw, silu_c, out, down_fn):
        for _ in range_(0xFFFFFFFF):
            sd = silu_c.acquire(1)
            o = out.acquire(1)
            for j in range_(dn_tiles):
                w = dnw.acquire(1)
                down_fn(m_gemv, index.casts(T.i32(), j) * m_gemv, w, sd, o)
                dnw.release(1)
            silu_c.release(1)
            out.release(1)

    dn_worker = Worker(dn_body, [DNW.cons(), SILU.cons(), OUT.prod(), down],
                       placement=Tile(col=col, row=3))

    L3_GUW = np.ndarray[(2 * gu_tiles * gu_packed,), u8]
    L3_FFIN = np.ndarray[(E,), bf]
    L3_DNW = np.ndarray[(dn_tiles * dn_packed,), u8]
    L3_OUT = np.ndarray[(E,), bf]

    guw_tap = TensorAccessPattern(tensor_dims=(1, 2 * gu_tiles * gu_packed),
        offset=0, sizes=[1, 1, 1, 2 * gu_tiles * gu_packed], strides=[0, 0, 0, 1])
    dnw_tap = TensorAccessPattern(tensor_dims=(1, dn_tiles * dn_packed),
        offset=0, sizes=[1, 1, 1, dn_tiles * dn_packed], strides=[0, 0, 0, 1])

    rt = Runtime()
    with rt.sequence(L3_OUT, L3_GUW, L3_FFIN, L3_DNW) as (out, guw, ffin, dnw):
        rt.start(gu_worker, dn_worker)
        tg = rt.task_group()
        rt.fill(GUW.prod(), guw, guw_tap, task_group=tg)
        rt.fill(FFIN.prod(), ffin, task_group=tg)
        rt.fill(DNW.prod(), dnw, dnw_tap, task_group=tg)
        rt.drain(OUT.cons(), out, task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())
