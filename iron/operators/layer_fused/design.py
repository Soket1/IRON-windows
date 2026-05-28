# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LayerFused IRON design — A1.3.1 SKELETON.

This is the empty-pipeline baseline: each of the 5 BOs is fed through
a trivial 1-tile passthrough Worker so that the legacy_xclbin compile
path emits a complete xclbin + insts.bin pair and we can validate:

  * 8-column partition declared correctly
  * 5 BO arg signature passes XRT's 8-group_id cap
  * DDR_PATCH ops are auto-emitted in the .insts (count is informational
    — A2 wiring will rewrite addresses)
  * MEM_TOPOLOGY shows HOST + SRAM banks (target: match FFLM)
  * llvm-objcopy/xclbinutil pipeline succeeds for an 8-col design

Compute kernels are stubbed in layer_fused.cc as a single noop that
just copies its input to output. Real stages land in A1.3.2-A1.3.4.
"""

import numpy as np
from ml_dtypes import bfloat16

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.scf import _for as range_
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2


def my_layer_fused(
    dev,
    cols,
    embed_dim,
    hidden_dim,
    num_heads,
    num_kv_heads,
    head_dim,
    max_seq_len,
    group_size=32,
    func_prefix="",
):
    """Build the LayerFused IRON design (skeleton stage)."""

    assert embed_dim  % cols == 0
    assert hidden_dim % cols == 0
    assert embed_dim  % group_size == 0
    assert hidden_dim % group_size == 0
    assert num_heads * head_dim == embed_dim

    dev_ty = NPU1() if dev == "npu" else NPU2()
    dtype_packed = np.dtype[np.uint8]
    dtype_vec    = np.dtype[bfloat16]

    e, h, g = embed_dim, hidden_dim, group_size
    nkv, hd, mx = num_kv_heads, head_dim, max_seq_len
    kv_e = nkv * hd
    groups_e = e // g
    groups_h = h // g

    # ── Bundle byte sizes (must mirror op_mlir.py:_bundle_byte_sizes) ────────
    norm_bytes = e * 2  # bf16 gain weights size

    m_input_qkv = 1
    packed_q   = m_input_qkv * e // 2 + m_input_qkv * groups_e * 2
    total_q    = cols * (e // cols) * packed_q
    total_kv_one = cols * (kv_e // cols) * packed_q
    bo0_bytes  = norm_bytes + total_q + 2 * total_kv_one

    m_input_o  = 1
    packed_o   = m_input_o * e // 2 + m_input_o * groups_e * 2
    total_o    = cols * (e // cols) * packed_o
    bo1_bytes  = total_o

    m_input_gu = 4
    packed_gu  = m_input_gu * e // 2 + m_input_gu * groups_e * 2
    total_gu   = cols * 2 * (h // cols // m_input_gu) * packed_gu
    m_input_d  = 1
    packed_d   = m_input_d * h // 2 + m_input_d * groups_h * 2
    total_d    = cols * (e // cols) * packed_d
    bo2_bytes  = norm_bytes + total_gu + total_d

    kv_one_bytes = nkv * mx * hd * 2
    bo3_bytes    = 2 * kv_one_bytes

    bo4_elems = (
        e + 2 * mx * hd + e + e + kv_e + kv_e
        + e + e + e + e + e + h + e + e
    )
    bo4_bytes = bo4_elems * 2

    for nm, b in (("bo0", bo0_bytes), ("bo1", bo1_bytes),
                  ("bo2", bo2_bytes), ("bo3", bo3_bytes)):
        assert b % 2 == 0

    # ── BO offset table (in bf16-element units, since L3 type is bf16) ───────
    # BO0: W_norm1 lives at offset 0, then Q, K, V weights follow.
    bo0_off_W_norm1 = 0
    # BO2: W_norm2 at offset 0, then gate, up, down.
    bo2_off_W_norm2 = 0
    # BO4 layout (bf16 elements):
    bo4_off_x         = 0
    bo4_off_rope_lut  = bo4_off_x        + e
    bo4_off_scratch   = bo4_off_rope_lut + 2 * mx * hd
    bo4_off_q_rot     = bo4_off_scratch  + e
    bo4_off_k_rot     = bo4_off_q_rot    + e
    bo4_off_v         = bo4_off_k_rot    + kv_e
    bo4_off_attn_out  = bo4_off_v        + kv_e
    bo4_off_o_out     = bo4_off_attn_out + e
    bo4_off_inpFF     = bo4_off_o_out    + e
    bo4_off_normed    = bo4_off_inpFF    + e
    bo4_off_ffn_in    = bo4_off_normed   + e
    bo4_off_silu_out  = bo4_off_ffn_in   + e
    bo4_off_ffn_out   = bo4_off_silu_out + h
    bo4_off_outL      = bo4_off_ffn_out  + e
    assert bo4_off_outL + e == bo4_elems

    # ── L3 (DDR / runtime sequence) types ────────────────────────────────────
    L3_w_qkv  = np.ndarray[(bo0_bytes // 2,), dtype_vec]
    L3_w_o    = np.ndarray[(bo1_bytes // 2,), dtype_vec]
    L3_w_ffn  = np.ndarray[(bo2_bytes // 2,), dtype_vec]
    L3_kv     = np.ndarray[(bo3_bytes // 2,), dtype_vec]
    L3_act    = np.ndarray[(bo4_elems,),       dtype_vec]

    # ── L1 types ─────────────────────────────────────────────────────────────
    # Pre-RMS: leader tile reads E bf16 input + E bf16 gain, writes E bf16
    # normed.
    L1_E_ty = np.ndarray[(e,), dtype_vec]

    # Skeleton passthrough (kept around to exercise the remaining BOs until
    # real compute lands in 2b..2d). Each touches a tiny chunk to keep the
    # tile-utilisation footprint nonzero.
    chunk_elems = 32
    L1_chunk = np.ndarray[(chunk_elems,), dtype_vec]

    # ── Kernels ──────────────────────────────────────────────────────────────
    rms_norm_fn = Kernel(
        f"{func_prefix}layer_fused_rms_norm_bf16",
        f"{func_prefix}layer_fused_{e}_{h}_g{g}.o",
        [L1_E_ty, L1_E_ty, L1_E_ty, np.int32],
    )
    noop_fn = Kernel(
        f"{func_prefix}layer_fused_noop_bf16",
        f"{func_prefix}layer_fused_{e}_{h}_g{g}.o",
        [L1_chunk, L1_chunk, np.int32],
    )

    # ── Pre-RMS ObjectFifos ──────────────────────────────────────────────────
    # ANM-style: one input fifo of depth=2 fed in two phases (x, then gain),
    # one output fifo of depth=1 drained to BO4 normed slot.
    # AIE2P limit: per-tile input-DMA channels ≤ 2 → fits comfortably with
    # one acquire(2) on the input side.
    rms_in_fifo  = ObjectFifo(L1_E_ty, name="rms_in",  depth=2)
    rms_out_fifo = ObjectFifo(L1_E_ty, name="rms_out", depth=1)

    def pre_rms_body(in_fifo, out_fifo, fn):
        for _ in range_(0xFFFFFFFF):
            sub = in_fifo.acquire(2)
            x_in = sub[0]
            gain = sub[1]
            out  = out_fifo.acquire(1)
            fn(x_in, gain, out, e)
            in_fifo.release(2)
            out_fifo.release(1)

    pre_rms_worker = Worker(
        pre_rms_body,
        [rms_in_fifo.cons(), rms_out_fifo.prod(), rms_norm_fn],
    )

    # ── Skeleton passthrough fifos (BOs not yet wired into real compute) ─────
    # We only need passthroughs for BO1 (w_o) and BO3 (kv_pair) now — BO0/BO2
    # are touched by pre-RMS already (W_norm1 in BO0 prefix, W_norm2 in BO2
    # prefix), and BO4 is touched by both rms_in (x) and rms_out (normed).
    skel_in  = [ObjectFifo(L1_chunk, name=f"skel_in_{i}",  depth=2) for i in range(2)]
    skel_out = [ObjectFifo(L1_chunk, name=f"skel_out_{i}", depth=2) for i in range(2)]

    def passthrough_body(in_fifo, out_fifo, fn):
        for _ in range_(0xFFFFFFFF):
            i = in_fifo.acquire(1)
            o = out_fifo.acquire(1)
            fn(i, o, chunk_elems)
            in_fifo.release(1)
            out_fifo.release(1)

    skel_workers = [
        Worker(passthrough_body, [skel_in[i].cons(), skel_out[i].prod(), noop_fn])
        for i in range(2)
    ]

    # ── TensorAccessPatterns ─────────────────────────────────────────────────
    # Pre-RMS: read x from BO4[0..E], read W_norm1 from BO0[0..E], write
    # normed to BO4[normed..normed+E].
    x_tap = TensorAccessPattern(
        tensor_dims=(1, bo4_elems),
        offset=bo4_off_x,
        sizes=[1, 1, 1, e],
        strides=[0, 0, 0, 1],
    )
    w_norm1_tap = TensorAccessPattern(
        tensor_dims=(1, bo0_bytes // 2),
        offset=bo0_off_W_norm1,
        sizes=[1, 1, 1, e],
        strides=[0, 0, 0, 1],
    )
    normed_tap = TensorAccessPattern(
        tensor_dims=(1, bo4_elems),
        offset=bo4_off_normed,
        sizes=[1, 1, 1, e],
        strides=[0, 0, 0, 1],
    )

    # Skeleton chunk TAPs for the still-unwired BOs (BO1 w_o, BO3 kv_pair).
    def chunk_tap(total_elems):
        return TensorAccessPattern(
            tensor_dims=(1, total_elems),
            offset=0,
            sizes=[1, 1, 1, chunk_elems],
            strides=[0, 0, 0, 1],
        )

    tap_o    = chunk_tap(bo1_bytes // 2)
    tap_kv   = chunk_tap(bo3_bytes // 2)
    # Skeleton drains write back into BO4 outL slot (won't be read by any
    # real consumer at this stage).
    skel_drain_tap = TensorAccessPattern(
        tensor_dims=(1, bo4_elems),
        offset=bo4_off_outL,
        sizes=[1, 1, 1, chunk_elems],
        strides=[0, 0, 0, 1],
    )

    rt = Runtime()
    with rt.sequence(
        L3_w_qkv, L3_w_o, L3_w_ffn, L3_kv, L3_act,
    ) as (w_qkv, w_o, w_ffn, kv, act):
        rt.start(pre_rms_worker, *skel_workers)
        tg = rt.task_group()
        # Pre-RMS phase
        rt.fill(rms_in_fifo.prod(),  act,   x_tap,       task_group=tg)
        rt.fill(rms_in_fifo.prod(),  w_qkv, w_norm1_tap, task_group=tg)
        rt.drain(rms_out_fifo.cons(), act,  normed_tap,  task_group=tg)
        # Skeleton (BO1 w_o + BO3 kv_pair touch — keeps DDR_PATCH ops alive)
        rt.fill (skel_in [0].prod(), w_o, tap_o,         task_group=tg)
        rt.fill (skel_in [1].prod(), kv,  tap_kv,        task_group=tg)
        rt.drain(skel_out[0].cons(), act, skel_drain_tap, task_group=tg)
        rt.drain(skel_out[1].cons(), act, skel_drain_tap, task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="LayerFused skeleton design (A1.3.1)")
    p.add_argument("--dev", default="npu2", choices=["npu", "npu2"])
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--embed-dim", type=int, default=2048)
    p.add_argument("--hidden-dim", type=int, default=8192)
    p.add_argument("--num-heads", type=int, default=32)
    p.add_argument("--num-kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--group-size", type=int, default=32)
    p.add_argument("-o", "--output-file-path", type=str)
    a = p.parse_args()
    module = my_layer_fused(
        a.dev, a.cols, a.embed_dim, a.hidden_dim,
        a.num_heads, a.num_kv_heads, a.head_dim,
        a.max_seq_len, a.group_size,
    )
    if a.output_file_path:
        with open(a.output_file_path, "w") as f:
            f.write(str(module))
