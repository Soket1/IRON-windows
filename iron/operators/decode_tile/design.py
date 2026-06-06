# SPDX-License-Identifier: Apache-2.0
"""decode_tile — D2.1 isolated single-tile time-mux GEMV chain.

ONE core tile runs the FULL GEMV chain of a decode layer time-multiplexed:
  QKV → O_proj → gate → up → silu → down
on the FFLM per-tile DMA budget = 2 S2MM + 1 MM2S:
  - Wf (1 S2MM): ONE weight fifo carrying [QKV|O|gate|up|down] tiles, time-mux.
  - Bf (1 S2MM): ONE activation-broadcast fifo, acquired 3× (X→QKV, attn_out→O,
                 ffn_in→gate+up). Down reads the LOCAL lf_silu static.
  - Of (1 MM2S): ONE unified output fifo, acquired 3× (QKV slice, O slice, down
                 partial), all draining to ONE DDR buffer at 3 offsets.

This proves the per-tile pattern for D2 (all-16-tiles parallel GEMV). It fuses
D1's row-2 (QKV) and row-5 (O_proj+FFN) work onto ONE tile. Attention is DEFERRED
(QKV output is drained to DDR for CPU check; O reads attn_out from DDR; gate/up
read ffn_in from DDR) — re-integrated in D2.3. Kernels are reused verbatim from
decode_layer (no recompile): v2 GEMV (QKV+O_proj), layer_fused gate_up/silu/down.

Slices use D1's 4-way column dims (q_rows=attn_group·head_dim, o=E/4, Hc=H/4) so
the proven kernels apply unchanged; D2.2 re-slices to true 16-way.
See dev_notes/track_a_build/D2_design_spike.md.
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


def my_decode_tile(dev, embed_dim=2048, head_dim=64, group_size=32, attn_group=8,
                   hidden_dim=8192, m_input=4, num_cols=4, col_offset=2):
    dev_ty = NPU1() if dev == "npu" else NPU2()
    bf = np.dtype[bfloat16]
    u8 = np.dtype[np.uint8]

    E = embed_dim
    g = group_size
    m = m_input
    groups = E // g
    # all GEMV stages read K=E=2048 (down K = Hc = H/num_cols = 2048 in 4-way) →
    # one packed-tile size for the shared weight fifo.
    Hc = hidden_dim // num_cols                # 2048 (4-way per-col hidden)
    assert Hc == E, "D2.1 reuses D1 4-way: down K must equal E for one packed size"
    packed_tile = m * E // 2 + m * groups * 2  # u8 bytes per m-row weight tile

    q_rows = attn_group * head_dim             # 512 = one column's QKV output
    o_slice = E // num_cols                     # 512 = one column's O_proj output

    qkv_tiles = q_rows // m
    o_tiles = o_slice // m
    gu_tiles = Hc // m
    dn_tiles = E // m
    assert q_rows % m == 0 and o_slice % m == 0 and Hc % m == 0 and E % m == 0

    n_wtiles = qkv_tiles + o_tiles + 2 * gu_tiles + dn_tiles

    L1_W_ty = np.ndarray[(packed_tile,), u8]
    L1_E_ty = np.ndarray[(E,), bf]              # broadcast activation AND output

    # DDR layouts — 5 buffers to match the xclbin_replay MLIR_AIE 5-BO ABI
    # (bo0=O out, bo1=W, bo2=X, bo3=attn_out, bo4=ffn_in). The ONE Bf fifo is
    # still filled 3× (from X, attn_out, ffn_in) → 1 S2MM preserved.
    L3_O_ty = np.ndarray[(3 * E,), bf]          # bo0: [qkv_out | o_out | down_out]
    L3_W_ty = np.ndarray[(n_wtiles * packed_tile,), u8]  # bo1
    L3_E_ty = np.ndarray[(E,), bf]              # bo2/3/4: X, attn_out, ffn_in

    # -------------------------------------------------------------------
    # Kernels — reused verbatim from decode_layer. Output args re-typed to
    # L1_E_ty so QKV/O/down can all share the ONE unified output fifo buffer
    # (the C functions take bfloat16* and write ≤E rows; type is the IRON slot).
    # -------------------------------------------------------------------
    qkv = Kernel("fused_dequant_matvec_v2_bf16",
                 f"fused_dequant_gemv_v2_{E}k_g{g}.o",
                 [np.int32, np.int32, L1_W_ty, L1_E_ty, L1_E_ty])
    oproj = Kernel("oproj_matvec_v2_bf16",
                   f"fused_dequant_gemv_v2_oproj_{E}k_g{g}.o",
                   [np.int32, np.int32, L1_W_ty, L1_E_ty, L1_E_ty])
    gate_up = Kernel("layer_fused_gate_up_bf16", "layer_fused_relay.o",
                     [np.int32, np.int32, L1_W_ty, L1_E_ty, np.int32])
    silu = Kernel("layer_fused_silu_mul_static_bf16", "layer_fused_relay.o",
                  [np.int32])
    down = Kernel("layer_fused_down_v2_static_bf16", "layer_fused_relay.o",
                  [np.int32, np.int32, L1_W_ty, L1_E_ty])

    Wf = ObjectFifo(L1_W_ty, name="Wf", depth=2)
    Bf = ObjectFifo(L1_E_ty, name="Bf", depth=2)
    Of = ObjectFifo(L1_E_ty, name="Of", depth=2)

    def tile_body(wf, bf_, of, qkv_fn, oproj_fn, gate_up_fn, silu_fn, down_fn):
        for _ in range_(0xFFFFFFFF):
            # --- stage 1: QKV (act = X) → drain QKV slice ---
            bx = bf_.acquire(1)
            qo = of.acquire(1)
            for j in range_(qkv_tiles):
                w = wf.acquire(1)
                qkv_fn(m, index.casts(T.i32(), j) * m, w, bx, qo)
                wf.release(1)
            of.release(1)
            bf_.release(1)
            # --- stage 2: O_proj (act = attn_out) → drain O slice ---
            ba = bf_.acquire(1)
            oo = of.acquire(1)
            for j in range_(o_tiles):
                w = wf.acquire(1)
                oproj_fn(m, index.casts(T.i32(), j) * m, w, ba, oo)
                wf.release(1)
            of.release(1)
            bf_.release(1)
            # --- stage 3: FFN gate/up (act = ffn_in) → lf_left/right statics ---
            bff = bf_.acquire(1)
            for j in range_(gu_tiles):
                w = wf.acquire(1)
                gate_up_fn(m, index.casts(T.i32(), j) * m, w, bff, 0)
                wf.release(1)
            for j in range_(gu_tiles):
                w = wf.acquire(1)
                gate_up_fn(m, index.casts(T.i32(), j) * m, w, bff, 1)
                wf.release(1)
            bf_.release(1)
            silu_fn(Hc)
            # --- stage 4: down (act = local lf_silu) → drain down partial ---
            do = of.acquire(1)
            for j in range_(dn_tiles):
                w = wf.acquire(1)
                down_fn(m, index.casts(T.i32(), j) * m, w, do)
                wf.release(1)
            of.release(1)

    worker = Worker(
        tile_body,
        [Wf.cons(), Bf.cons(), Of.prod(), qkv, oproj, gate_up, silu, down],
        placement=Tile(col=col_offset, row=2))

    # -------------------------------------------------------------------
    # Runtime: one big weight fill, 3 activation fills, 3 output drains.
    # All on one shim column → 2 S2MM (Wf, Bf) + 1 MM2S (Of), time-muxed
    # by the fifo depth + backpressure (worker enforces stage order).
    # -------------------------------------------------------------------
    e_tap = TensorAccessPattern(
        tensor_dims=(1, E), offset=0,
        sizes=[1, 1, 1, E], strides=[0, 0, 0, 1])

    def o_tap(i):  # i-th E-chunk of [qkv_out | o_out | down_out]
        return TensorAccessPattern(
            tensor_dims=(3 * E,), offset=i * E,
            sizes=[1, 1, 1, E], strides=[0, 0, 0, 1])

    w_tap = TensorAccessPattern(
        tensor_dims=(1, n_wtiles * packed_tile), offset=0,
        sizes=[1, 1, 1, n_wtiles * packed_tile], strides=[0, 0, 0, 1])

    rt = Runtime()
    # 5-BO ABI: bo0=O (output, dumped by replay), bo1=W, bo2=X, bo3=attn, bo4=ffn.
    with rt.sequence(L3_O_ty, L3_W_ty, L3_E_ty, L3_E_ty, L3_E_ty) as (o, w, x, a, f):
        rt.start(worker)
        # ONE task group: the worker INTERLEAVES drains with fills (drains
        # qkv_out at stage 1, before consuming ffn_in at stage 3). Separate
        # phased task groups would deadlock (drain blocked behind a later
        # fill that blocks on the stalled worker). With all DMAs in one group,
        # fifo depth-2 backpressure sequences them safely.
        tg = rt.task_group()
        rt.fill(Wf.prod(), w, w_tap, task_group=tg)
        rt.fill(Bf.prod(), x, e_tap, task_group=tg)   # X → QKV
        rt.fill(Bf.prod(), a, e_tap, task_group=tg)   # attn_out → O_proj
        rt.fill(Bf.prod(), f, e_tap, task_group=tg)   # ffn_in → gate/up
        for i in range(3):
            rt.drain(Of.cons(), o, o_tap(i), task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())
