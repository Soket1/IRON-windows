# SPDX-License-Identifier: Apache-2.0
"""ffn_probe1 — SINGLE-TILE SwiGLU FFN (gate/up+silu+down on ONE tile per col).
Isolates whether the single-tile structure itself causes the ~11% error seen in
decode_layer. Same kernels as ffn_probe (which uses 2 tiles and passes 0.99999).
If this probe also fails, single-tile is the root cause."""
import numpy as np
from ml_dtypes import bfloat16

import aie.dialects.index as index
from aie.dialects.aie import T
from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2, Tile


def my_ffn1(dev, embed_dim=2048, hidden_dim=8192, group_size=32,
            m_gemv=2, num_cols=4, col_offset=2):
    dev_ty = NPU1() if dev == "npu" else NPU2()
    bf = np.dtype[bfloat16]; u8 = np.dtype[np.uint8]
    E = embed_dim
    Hc = hidden_dim // num_cols
    gu_groups = E // group_size
    gu_packed = m_gemv * E // 2 + m_gemv * gu_groups * 2
    gu_tiles = Hc // m_gemv
    dn_groups = Hc // group_size
    dn_packed = m_gemv * Hc // 2 + m_gemv * dn_groups * 2
    dn_tiles = E // m_gemv

    L1_GUW_ty = np.ndarray[(gu_packed,), u8]
    L1_E_ty = np.ndarray[(E,), bf]
    L1_SILU_ty = np.ndarray[(Hc,), bf]

    gate_up = Kernel("layer_fused_gate_up_bf16", "layer_fused_ffn1.o",
                     [np.int32, np.int32, L1_GUW_ty, L1_E_ty, np.int32])
    silu = Kernel("layer_fused_silu_mul_bf16", "layer_fused_ffn1.o",
                  [L1_SILU_ty, np.int32])
    down = Kernel("down_matvec_v2_bf16",
                  f"fused_dequant_gemv_v2_down_{Hc}k_g{group_size}.o",
                  [np.int32, np.int32, L1_GUW_ty, L1_SILU_ty, L1_E_ty])

    GUWD = [ObjectFifo(L1_GUW_ty, name=f"GUWD_{c}", depth=2) for c in range(num_cols)]
    FFIN = ObjectFifo(L1_E_ty, name="FFIN", depth=1)
    FFB = FFIN.cons().forward(name="ffb", depth=2, placement=Tile(col=1, row=1))
    SILU = [ObjectFifo(L1_SILU_ty, name=f"S_{c}", depth=1) for c in range(num_cols)]
    OUT = [ObjectFifo(L1_E_ty, name=f"O_{c}", depth=2) for c in range(num_cols)]

    def ffn_body(guwd, ffin, p, sp, sc, gu_fn, si_fn, dn_fn):
        for _ in range_(0xFFFFFFFF):
            b = ffin.acquire(1)
            for j in range_(gu_tiles):
                w = guwd.acquire(1)
                gu_fn(m_gemv, index.casts(T.i32(), j) * m_gemv, w, b, 0)
                guwd.release(1)
            for j in range_(gu_tiles):
                w = guwd.acquire(1)
                gu_fn(m_gemv, index.casts(T.i32(), j) * m_gemv, w, b, 1)
                guwd.release(1)
            ffin.release(1)
            s = sp.acquire(1); si_fn(s, Hc); sp.release(1)
            sd = sc.acquire(1); o = p.acquire(1)
            for j in range_(dn_tiles):
                w = guwd.acquire(1)
                dn_fn(m_gemv, index.casts(T.i32(), j) * m_gemv, w, sd, o)
                guwd.release(1)
            sc.release(1); p.release(1)

    workers = []
    import os as _os
    # DECODE_DBG_EDGE: place FFN workers on edge cols 0,1,6,7 (like the full
    # decode_layer) instead of center cols 2-5, to test if edge placement breaks.
    if _os.environ.get("DECODE_DBG_EDGE"):
        place_cols = [0, 1, 6, 7]
        place_rows = [2, 2, 3, 3]
    else:
        place_cols = [c + col_offset for c in range(num_cols)]
        place_rows = [2] * num_cols
    for c in range(num_cols):
        workers.append(Worker(ffn_body,
            [GUWD[c].cons(), FFB.cons(), OUT[c].prod(),
             SILU[c].prod(), SILU[c].cons(), gate_up, silu, down],
            placement=Tile(col=place_cols[c], row=place_rows[c])))

    ffnw_per_col = 2 * gu_tiles * gu_packed + dn_tiles * dn_packed
    L3_OUT = np.ndarray[(num_cols * E,), bf]
    L3_W = np.ndarray[(num_cols * ffnw_per_col,), u8]
    L3_FFIN = np.ndarray[(E,), bf]

    def w_tap(c):
        return TensorAccessPattern(tensor_dims=(1, num_cols * ffnw_per_col),
            offset=c * ffnw_per_col, sizes=[1, 1, 1, ffnw_per_col], strides=[0, 0, 0, 1])

    def o_tap(c):
        return TensorAccessPattern(tensor_dims=(1, num_cols * E),
            offset=c * E, sizes=[1, 1, 1, E], strides=[0, 0, 0, 1])

    ffin_tap = TensorAccessPattern(tensor_dims=(1, E), offset=0,
        sizes=[1, 1, 1, E], strides=[0, 0, 0, 1])

    rt = Runtime()
    with rt.sequence(L3_OUT, L3_W, L3_FFIN) as (out, w, ffin):
        rt.start(*workers)
        tg = rt.task_group()
        for c in range(num_cols):
            rt.fill(GUWD[c].prod(), w, w_tap(c), task_group=tg)
        rt.fill(FFIN.prod(), ffin, ffin_tap, task_group=tg)
        for c in range(num_cols):
            rt.drain(OUT[c].cons(), out, o_tap(c), task_group=tg, wait=(c == num_cols - 1))
        rt.finish_task_group(tg)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())
