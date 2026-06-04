# SPDX-License-Identifier: Apache-2.0
"""ffn_probe — standalone 4-column SwiGLU FFN with cross-column reduction.

De-risks the FFN stage of the fused decode layer (decode-layer step #17, STEP C)
before integrating it into decode_layer. Mirrors the proven swiglu_probe math
but replicates it over num_cols columns and adds the cross-column reduction +
final residual add that a real tensor-parallel FFN needs:

  per column c (cols col_offset..col_offset+num_cols-1):
    tile (pc, 2): gate (E->Hc) + up (E->Hc) + silu_mul → SILU_c (on-chip)
    tile (pc, 3): down partial GEMV (Hc->E) over SILU_c → P_c (full-E partial)
  reduction tree (each column's down is a PARTIAL of the full E — layer_fused
  comment: "host sums the NUM_AIE_COLUMNS partials"; we sum on-chip):
    redA: P0 + P1 → T01      redB: P2 + P3 → T23
    redC: T01 + T23 → T      resid: T + inpFF → OUT (final residual add)

gate/up = layer_fused_gate_up_bf16 (phase 0=gate→lf_left, 1=up→lf_right,
INT4 = (nibble-8)*scale WITH -8 bias). silu_mul = silu(gate)*up,
silu(x)=x*0.5(1+tanh(x/2)). down = v2 GEMV (c_out+=row_offset, NO -8 bias).
FFIN bundle = [ffn_in (E) | inpFF (E)]: ffn_in feeds gate/up, inpFF is the
pre-FFN residual added at the end. 5-BO ABI respected (OUT,GUW,DNW,FFINbundle).
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


def my_ffn(dev, embed_dim=2048, hidden_dim=8192, group_size=32,
           m_gemv=2, num_cols=4, col_offset=2):
    dev_ty = NPU1() if dev == "npu" else NPU2()
    bf = np.dtype[bfloat16]; u8 = np.dtype[np.uint8]
    E = embed_dim
    Hc = hidden_dim // num_cols                      # 2048 hidden per column
    assert Hc * num_cols == hidden_dim
    assert col_offset + num_cols <= 8

    gu_groups = E // group_size
    gu_packed = m_gemv * E // 2 + m_gemv * gu_groups * 2     # gate/up tile bytes
    gu_tiles  = Hc // m_gemv
    dn_groups = Hc // group_size
    dn_packed = m_gemv * Hc // 2 + m_gemv * dn_groups * 2     # down tile bytes
    dn_tiles  = E // m_gemv

    L1_GUW_ty  = np.ndarray[(gu_packed,), u8]
    L1_FFIN_ty = np.ndarray[(E,), bf]
    L1_DNW_ty  = np.ndarray[(dn_packed,), u8]
    L1_SILU_ty = np.ndarray[(Hc,), bf]
    L1_E_ty    = np.ndarray[(E,), bf]

    gate_up = Kernel("layer_fused_gate_up_bf16", "layer_fused_ffn.o",
                     [np.int32, np.int32, L1_GUW_ty, L1_FFIN_ty, np.int32])
    silu = Kernel("layer_fused_silu_mul_bf16", "layer_fused_ffn.o",
                  [L1_SILU_ty, np.int32])
    # down = v2 GEMV Hc->E (c_out += row_offset → one whole-E buffer accumulates
    # all tiles; NO -8 bias). Distinct symbol so its arg types (DNW/SILU/E) are
    # clean; reuse across all 4 down tiles.
    down = Kernel("fused_dequant_matvec_v2_bf16",
                  f"fused_dequant_gemv_v2_{Hc}k_g{group_size}.o",
                  [np.int32, np.int32, L1_DNW_ty, L1_SILU_ty, L1_E_ty])
    add_k = Kernel("layer_fused_add_bf16", "layer_fused_ffn.o",
                   [L1_E_ty, L1_E_ty, L1_E_ty, np.int32])

    # Per-column fifos
    GUW  = [ObjectFifo(L1_GUW_ty,  name=f"GUW_{c}",  depth=2) for c in range(num_cols)]
    DNW  = [ObjectFifo(L1_DNW_ty,  name=f"DNW_{c}",  depth=2) for c in range(num_cols)]
    SILU = [ObjectFifo(L1_SILU_ty, name=f"SILU_{c}", depth=2) for c in range(num_cols)]
    P    = [ObjectFifo(L1_E_ty,    name=f"P_{c}",    depth=2) for c in range(num_cols)]

    # ffn_in: one source broadcast (forward) to all gate/up tiles.
    FFIN_src = ObjectFifo(L1_FFIN_ty, name="FFIN_src", depth=1)
    FFIN_b = FFIN_src.cons().forward(name="ffin_bcast", depth=1,
                                     placement=Tile(col=1, row=1))

    # reduction-tree fifos
    T01 = ObjectFifo(L1_E_ty, name="T01", depth=2)
    T23 = ObjectFifo(L1_E_ty, name="T23", depth=2)
    Tsum = ObjectFifo(L1_E_ty, name="Tsum", depth=2)
    InpFF = ObjectFifo(L1_E_ty, name="InpFF", depth=2)   # pre-FFN residual
    OUT = ObjectFifo(L1_E_ty, name="OUT", depth=2)

    workers = []
    for c in range(num_cols):
        pc = c + col_offset

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
                silu_fn(s, Hc)                            # silu(gate)*up → SILU_c
                silu_p.release(1)
        workers.append(Worker(gu_body,
            [GUW[c].cons(), FFIN_b.cons(), SILU[c].prod(), gate_up, silu],
            placement=Tile(col=pc, row=2)))

        def dn_body(dnw, silu_c, p, down_fn):
            for _ in range_(0xFFFFFFFF):
                sd = silu_c.acquire(1)
                o = p.acquire(1)
                for j in range_(dn_tiles):
                    w = dnw.acquire(1)
                    down_fn(m_gemv, index.casts(T.i32(), j) * m_gemv, w, sd, o)
                    dnw.release(1)
                silu_c.release(1)
                p.release(1)
        workers.append(Worker(dn_body,
            [DNW[c].cons(), SILU[c].cons(), P[c].prod(), down],
            placement=Tile(col=pc, row=3)))

    # reduction tree: 3 adds + 1 residual add. Each tile reads 2 fifos, 1 out.
    def add2_body(a_f, b_f, o_f, fn):
        for _ in range_(0xFFFFFFFF):
            a = a_f.acquire(1); b = b_f.acquire(1); o = o_f.acquire(1)
            fn(a, b, o, E)
            a_f.release(1); b_f.release(1); o_f.release(1)

    workers.append(Worker(add2_body, [P[0].cons(), P[1].cons(), T01.prod(), add_k],
                          placement=Tile(col=0, row=2)))
    workers.append(Worker(add2_body, [P[2].cons(), P[3].cons(), T23.prod(), add_k],
                          placement=Tile(col=1, row=2)))
    workers.append(Worker(add2_body, [T01.cons(), T23.cons(), Tsum.prod(), add_k],
                          placement=Tile(col=6, row=2)))
    workers.append(Worker(add2_body, [Tsum.cons(), InpFF.cons(), OUT.prod(), add_k],
                          placement=Tile(col=7, row=2)))

    # DDR layout. GUW/DNW per-column contiguous. FFIN bundle = [ffn_in | inpFF].
    L3_OUT = np.ndarray[(E,), bf]
    L3_GUW = np.ndarray[(num_cols * 2 * gu_tiles * gu_packed,), u8]
    L3_DNW = np.ndarray[(num_cols * dn_tiles * dn_packed,), u8]
    L3_FFIN = np.ndarray[(2 * E,), bf]                # [ffn_in | inpFF]

    guw_per_col = 2 * gu_tiles * gu_packed
    dnw_per_col = dn_tiles * dn_packed

    def guw_tap(c):
        return TensorAccessPattern(tensor_dims=(1, num_cols * guw_per_col),
            offset=c * guw_per_col, sizes=[1, 1, 1, guw_per_col], strides=[0, 0, 0, 1])

    def dnw_tap(c):
        return TensorAccessPattern(tensor_dims=(1, num_cols * dnw_per_col),
            offset=c * dnw_per_col, sizes=[1, 1, 1, dnw_per_col], strides=[0, 0, 0, 1])

    ffin_tap = TensorAccessPattern(tensor_dims=(1, 2 * E), offset=0,
        sizes=[1, 1, 1, E], strides=[0, 0, 0, 1])
    inpff_tap = TensorAccessPattern(tensor_dims=(1, 2 * E), offset=E,
        sizes=[1, 1, 1, E], strides=[0, 0, 0, 1])
    out_tap = TensorAccessPattern(tensor_dims=(1, E), offset=0,
        sizes=[1, 1, 1, E], strides=[0, 0, 0, 1])

    rt = Runtime()
    with rt.sequence(L3_OUT, L3_GUW, L3_DNW, L3_FFIN) as (out, guw, dnw, ffin):
        rt.start(*workers)
        tg = rt.task_group()
        for c in range(num_cols):
            rt.fill(GUW[c].prod(), guw, guw_tap(c), task_group=tg)
            rt.fill(DNW[c].prod(), dnw, dnw_tap(c), task_group=tg)
        rt.fill(FFIN_src.prod(), ffin, ffin_tap, task_group=tg)
        rt.fill(InpFF.prod(),    ffin, inpff_tap, task_group=tg)
        rt.drain(OUT.cons(), out, out_tap, task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())
