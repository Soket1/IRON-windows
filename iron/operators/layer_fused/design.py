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
from aie.iron.device import NPU1, NPU2, Tile


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

    m_input_qkv = 2  # AIE2P shim DMA cap: drain len must be ≥ 4 bytes
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
    # BO0: W_norm1 [E elems] | W_q [total_q bytes / 2] | W_k [...] | W_v [...]
    bo0_off_W_norm1 = 0
    bo0_off_W_q     = bo0_off_W_norm1 + e
    bo0_off_W_k     = bo0_off_W_q     + total_q // 2
    bo0_off_W_v     = bo0_off_W_k     + total_kv_one // 2
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

    # Q GEMV: weight tile (packed INT4 + scales) per outer iter; activation
    # is full E bf16 (broadcast); output is m_input_qkv bf16 slice per tile.
    packed_qkv_tile_bytes = m_input_qkv * e // 2 + m_input_qkv * groups_e * 2
    L1_Aq_ty = np.ndarray[(packed_qkv_tile_bytes,), dtype_packed]
    L1_Bq_ty = np.ndarray[(e,), dtype_vec]
    L1_Cq_ty = np.ndarray[(m_input_qkv,), dtype_vec]

    # Attention KV chunk type: K_CHUNK tokens × HEAD_DIM bf16 elements.
    # Matches FFLM's ct_chunk_size=32 (mha.dll RE'd disasm) and our kernel
    # ATTN_K_CHUNK macro. Per-col attention reads its 1 kv_head slice from
    # BO3 via this fifo, streaming chunk-by-chunk.
    K_CHUNK = 32
    kv_chunk_elems = K_CHUNK * hd
    L1_kv_chunk_ty = np.ndarray[(kv_chunk_elems,), dtype_vec]

    # Skeleton passthrough (kept around to exercise the remaining BOs until
    # real compute lands in 2b..2d). Each touches a tiny chunk to keep the
    # tile-utilisation footprint nonzero.
    chunk_elems = 32
    L1_chunk = np.ndarray[(chunk_elems,), dtype_vec]

    # ── Kernels ──────────────────────────────────────────────────────────────
    # Three QKV variants:
    #   * fifo-input GEMV: cols 1-7 use this (consume bq_mem broadcast).
    #   * static-input GEMV: col 0 only (reads activation from static L1
    #     buffer that the col-0 pre-RMS kernel wrote earlier in the same
    #     worker iteration).
    #   * pre-RMS col-0: weighted RMSNorm that writes to BOTH the static
    #     L1 buffer (for col 0's own QKV phases) AND the bq fifo output
    #     buffer (broadcast via MemTile to cols 1-7).
    rms_norm_fn = Kernel(
        f"{func_prefix}layer_fused_rms_norm_bf16",
        f"{func_prefix}layer_fused_{e}_{h}_g{g}.o",
        [L1_E_ty, L1_E_ty, L1_E_ty, np.int32],
    )
    pre_rms_col0_fn = Kernel(
        f"{func_prefix}layer_fused_pre_rms_col0_bf16",
        f"{func_prefix}layer_fused_{e}_{h}_g{g}.o",
        [L1_E_ty, L1_E_ty, L1_E_ty, np.int32],
    )
    qkv_gemv_fn = Kernel(
        f"{func_prefix}layer_fused_qkv_gemv_bf16",
        f"{func_prefix}layer_fused_{e}_{h}_g{g}.o",
        [np.int32, np.int32, L1_Aq_ty, L1_Bq_ty, L1_Cq_ty],
    )
    qkv_gemv_static_fn = Kernel(
        f"{func_prefix}layer_fused_qkv_gemv_static_bf16",
        f"{func_prefix}layer_fused_{e}_{h}_g{g}.o",
        [np.int32, np.int32, L1_Aq_ty, L1_Cq_ty],
    )
    noop_fn = Kernel(
        f"{func_prefix}layer_fused_noop_bf16",
        f"{func_prefix}layer_fused_{e}_{h}_g{g}.o",
        [L1_chunk, L1_chunk, np.int32],
    )
    # Same kernel symbol but bound for the larger KV chunk shape; needed
    # because Kernel parameter types are the L1 type signature, and the
    # attention spike worker passes kv_chunk_elems-sized buffers. Uses a
    # distinct C symbol (`_kv_bf16`) to avoid IRON's "redefinition of
    # symbol" verifier error on same-symbol re-binding.
    noop_kv_fn = Kernel(
        f"{func_prefix}layer_fused_noop_kv_bf16",
        f"{func_prefix}layer_fused_{e}_{h}_g{g}.o",
        [L1_kv_chunk_ty, L1_kv_chunk_ty, np.int32],
    )

    # ── Pre-RMS input fifo ───────────────────────────────────────────────────
    # Single input fifo of depth=2 fed in two phases from rt.sequence
    # (x first, then W_norm1). col 0's body acquires both via acquire(2).
    # No separate output fifo — pre_rms_col0 kernel writes to BOTH static
    # L1 (consumed in same body's QKV phases) AND the bq fifo prod slot
    # (broadcast via MemTile to cols 1-7).
    rms_in_fifo  = ObjectFifo(L1_E_ty, name="rms_in",  depth=2)

    # ── ATTN sub-stage (deferred) ────────────────────────────────────────────
    # We tried adding a 3rd input DMA channel (K_chunk reads from BO3) to
    # the per-col qkv_worker as a phase-5 spike. SequentialPlacer rejected
    # it with "Failed to find a tile matching column 0" — confirming that
    # AIE2P compute tiles have a hard 2-input-DMA cap. Attention KV
    # streaming MUST go through MemTile gather (one row-1 tile fans the
    # K/V chunks out to compute tiles via the stream switch).
    #
    # For this checkpoint we keep the qkv_workers at 2 input channels
    # (Aqkv + Bq), and defer attention wiring to the next step which
    # adds 4 MemTile-mediated KV-streaming fifos (one per kv_head).
    # No Kch_fifos, no phase-5 in qkv_body, no Kch_taps in rt.sequence.
    # Per-col qkv_worker stays at: 2 input + 1 output DMA channels.

    # ── Q/K/V GEMV (8 cols, unified worker doing 3 phases per outer iter) ───
    # Per-col output dims:
    #   Q:  rows_per_col_q  = e // cols           (256 for cols=8, e=2048)
    #   K:  rows_per_col_kv = kv_e // cols        (64  for cols=8, kv_e=512)
    #   V:  same as K
    rows_per_col_q   = e // cols
    tiles_per_col_q  = rows_per_col_q // m_input_qkv
    bytes_col_q      = tiles_per_col_q * packed_qkv_tile_bytes
    assert tiles_per_col_q <= 1023, f"tiles_per_col_q={tiles_per_col_q} exceeds DMA cap"

    rows_per_col_kv  = kv_e // cols
    tiles_per_col_kv = rows_per_col_kv // m_input_qkv
    bytes_col_kv     = tiles_per_col_kv * packed_qkv_tile_bytes
    assert rows_per_col_kv % m_input_qkv == 0, (
        f"rows_per_col_kv={rows_per_col_kv} must be divisible by m_input_qkv={m_input_qkv}"
    )
    assert tiles_per_col_kv <= 1023, f"tiles_per_col_kv={tiles_per_col_kv} exceeds DMA cap"

    # ── Unified bq broadcast (col 0 produces, MemTile fans out to all cols) ──
    # Col 0's pre-RMS kernel writes the normed result into the bq_l3l2_fifo
    # producer slot directly. MemTile broadcast fanout publishes it to all
    # 8 cols' .cons() ports. depth=1 — one normed token per outer iteration.
    Aqkv_fifos = [ObjectFifo(L1_Aq_ty, name=f"Aqkv_{i}", depth=2) for i in range(cols)]
    bq_l3l2_fifo = ObjectFifo(L1_Bq_ty, name="bq_L3L2", depth=1)
    bq_mem_fifo  = bq_l3l2_fifo.cons().forward(
        name="bq_mem", depth=1, placement=Tile(col=0, row=1))
    Cqkv_fifos = [ObjectFifo(L1_Cq_ty, name=f"Cqkv_{i}", depth=2) for i in range(cols)]

    # ── Worker bodies ────────────────────────────────────────────────────────
    # Col 0: 4-phase body (pre-RMS → Q → K → V). Pre-RMS writes to BOTH the
    # static L1 buffer (consumed by qkv_gemv_static below) AND to the bq
    # fifo prod slot (broadcast). QKV phases use the static-input variant
    # so col 0's tile stays at 2 input channels (rms_in + Aqkv).
    def col0_body(rms_in, bq_prod, Aqkv, Cqkv, pre_rms_fn, qkv_static_fn):
        for _ in range_(0xFFFFFFFF):
            # Pre-RMS phase
            sub = rms_in.acquire(2)
            x_in  = sub[0]
            gain  = sub[1]
            bq_out = bq_prod.acquire(1)
            pre_rms_fn(x_in, gain, bq_out, e)
            rms_in.release(2)
            bq_prod.release(1)  # publish normed via MemTile to cols 1-7
            # Q phase (static-input GEMV, no B fifo acquire)
            for _ in range_(tiles_per_col_q):
                a = Aqkv.acquire(1)
                c = Cqkv.acquire(1)
                qkv_static_fn(m_input_qkv, 0, a, c)
                Aqkv.release(1)
                Cqkv.release(1)
            # K phase
            for _ in range_(tiles_per_col_kv):
                a = Aqkv.acquire(1)
                c = Cqkv.acquire(1)
                qkv_static_fn(m_input_qkv, 0, a, c)
                Aqkv.release(1)
                Cqkv.release(1)
            # V phase
            for _ in range_(tiles_per_col_kv):
                a = Aqkv.acquire(1)
                c = Cqkv.acquire(1)
                qkv_static_fn(m_input_qkv, 0, a, c)
                Aqkv.release(1)
                Cqkv.release(1)

    # Cols 1-7: 3-phase body. Acquire B (bq_mem broadcast) ONCE per outer
    # iter, reuse for all three QKV phases. (Was acquire/release per phase,
    # but with col 0 producing bq once and depth=1, multi-acquire deadlocks.)
    def colN_body(A, B, C, fn):
        for _ in range_(0xFFFFFFFF):
            b = B.acquire(1)
            # Q phase
            for _ in range_(tiles_per_col_q):
                a = A.acquire(1)
                c = C.acquire(1)
                fn(m_input_qkv, 0, a, b, c)
                A.release(1)
                C.release(1)
            # K phase
            for _ in range_(tiles_per_col_kv):
                a = A.acquire(1)
                c = C.acquire(1)
                fn(m_input_qkv, 0, a, b, c)
                A.release(1)
                C.release(1)
            # V phase
            for _ in range_(tiles_per_col_kv):
                a = A.acquire(1)
                c = C.acquire(1)
                fn(m_input_qkv, 0, a, b, c)
                A.release(1)
                C.release(1)
            B.release(1)

    col0_worker = Worker(
        col0_body,
        [rms_in_fifo.cons(), bq_l3l2_fifo.prod(),
         Aqkv_fifos[0].cons(), Cqkv_fifos[0].prod(),
         pre_rms_col0_fn, qkv_gemv_static_fn],
    )
    qkv_workers_rest = [
        Worker(colN_body,
               [Aqkv_fifos[i].cons(), bq_mem_fifo.cons(),
                Cqkv_fifos[i].prod(), qkv_gemv_fn])
        for i in range(1, cols)
    ]
    qkv_workers = [col0_worker] + qkv_workers_rest

    # ── Skeleton passthrough workers (BO1/BO2 placeholders only) ─────────────
    # Two small noop workers keep DDR_PATCH ops alive on BO1 (w_o) and
    # BO2 (w_ffn). BO3 (kv_pair) is wired via the attention path below
    # (no skel needed). Real wiring for BO1/BO2 lands in stage 3
    # (O_proj, SwiGLU). Tile budget: 0 pre_rms + 8 qkv + 2 skel + N attn
    # = 10 + N of 16. Plan: N = 8 attention workers if MemTile gather
    # design fits, else N < 8 with degradation to host-CPU fallback.
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

    # ── Attention worker spike (col 0 only, 1-tile probe) ────────────────────
    # Probes whether IRON's SequentialPlacer can place a Worker on row 3 of
    # col 0 (since col 0's qkv_worker already occupies row 2). Body: simple
    # passthrough that acquires a chunk from a MemTile-mediated KV stream
    # (no compute, just exercises the placement). If placement succeeds,
    # we have proof-of-concept that the row-3 placement works and can
    # scale to 8 attention workers.
    #
    # MemTile gather pattern: shim DMA fills `kv_l3l2_fifo` from BO3
    # (one chunk total — col 0's own kv_head slice). The fifo's MemTile-
    # placed forward fans out to col 0's attention worker on row 3.
    kv_l3l2_fifo = ObjectFifo(L1_kv_chunk_ty, name="kv_L3L2", depth=2)
    kv_mem_fifo  = kv_l3l2_fifo.cons().forward(
        name="kv_mem", depth=2, placement=Tile(col=0, row=1))
    # noop entry takes (in, out, n) — we re-use it just to consume the
    # KV chunk. The "out" goes to a small drain fifo we'll route to the
    # BO4 outL slot for shape validation only.
    attn_drain_fifo = ObjectFifo(L1_kv_chunk_ty, name="attn_drain", depth=1)

    def attn_spike_body(kv_in, drain_out, fn):
        for _ in range_(0xFFFFFFFF):
            i = kv_in.acquire(1)
            o = drain_out.acquire(1)
            fn(i, o, kv_chunk_elems)
            kv_in.release(1)
            drain_out.release(1)

    attn_spike_worker = Worker(
        attn_spike_body,
        [kv_mem_fifo.cons(), attn_drain_fifo.prod(), noop_kv_fn],
    )

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

    # Skeleton chunk TAPs for the still-unwired BOs (BO1 w_o, BO2 w_ffn,
    # BO3 kv_pair).
    def chunk_tap(total_elems):
        return TensorAccessPattern(
            tensor_dims=(1, total_elems),
            offset=0,
            sizes=[1, 1, 1, chunk_elems],
            strides=[0, 0, 0, 1],
        )

    tap_o    = chunk_tap(bo1_bytes // 2)
    tap_ffn  = chunk_tap(bo2_bytes // 2)
    # tap_kv removed — BO3 wired via attn_kv_tap (col 0 spike).
    # Skeleton drains write back into BO4 outL slot (won't be read by any
    # real consumer at this stage).
    skel_drain_tap = TensorAccessPattern(
        tensor_dims=(1, bo4_elems),
        offset=bo4_off_outL,
        sizes=[1, 1, 1, chunk_elems],
        strides=[0, 0, 0, 1],
    )

    # Attention spike: read 1 chunk (32 tokens × HEAD_DIM) from BO3
    # K_cache region for col 0's kv_head (= kv_head 0, offset 0 in BO3).
    attn_kv_tap = TensorAccessPattern(
        tensor_dims=(1, bo3_bytes // 2),
        offset=0,  # col 0 → kv_head 0 → start of K_cache
        sizes=[1, 1, 1, kv_chunk_elems],
        strides=[0, 0, 0, 1],
    )
    # Drain attn worker output into BO4 attn_out (just for the spike, no
    # real consumer downstream until stage 3 wiring).
    attn_drain_tap = TensorAccessPattern(
        tensor_dims=(1, bo4_elems),
        offset=bo4_off_attn_out,
        sizes=[1, 1, 1, kv_chunk_elems],
        strides=[0, 0, 0, 1],
    )

    # ── Q GEMV TAPs ──────────────────────────────────────────────────────────
    # Each col reads its bytes_col_q slice of W_q from BO0 (declared as
    # bf16-element halved-shape, so byte offset = element offset × 2).
    # bytes_col_q is in BYTES; the L3 type is bf16 elements; offsets and
    # sizes inside the TAP are in *element* units (the design's L3 ndarray
    # type was declared bf16 — the BD generator multiplies by itemsize for
    # the actual byte count).
    # post_attn_fused.design.py:330 also uses sizes=[1,1,1,N] linear chunk
    # to avoid the partial-merge bug; we follow.
    Aq_taps = [
        TensorAccessPattern(
            tensor_dims=(1, bo0_bytes // 2),
            offset=bo0_off_W_q + i * (bytes_col_q // 2),
            sizes=[1, 1, 1, bytes_col_q // 2],
            strides=[0, 0, 0, 1],
        )
        for i in range(cols)
    ]
    # K and V weight TAPs — same per-row tile size, fewer rows/col.
    Ak_taps = [
        TensorAccessPattern(
            tensor_dims=(1, bo0_bytes // 2),
            offset=bo0_off_W_k + i * (bytes_col_kv // 2),
            sizes=[1, 1, 1, bytes_col_kv // 2],
            strides=[0, 0, 0, 1],
        )
        for i in range(cols)
    ]
    Av_taps = [
        TensorAccessPattern(
            tensor_dims=(1, bo0_bytes // 2),
            offset=bo0_off_W_v + i * (bytes_col_kv // 2),
            sizes=[1, 1, 1, bytes_col_kv // 2],
            strides=[0, 0, 0, 1],
        )
        for i in range(cols)
    ]
    # Bq is normed (E bf16) in BO4 — broadcast via MemTile.
    bq_tap = TensorAccessPattern(
        tensor_dims=(1, bo4_elems),
        offset=bo4_off_normed,
        sizes=[1, 1, 1, e],
        strides=[0, 0, 0, 1],
    )
    # Cq drains: each col writes rows_per_col_q bf16 elements into BO4
    # q_rot region (will be RoPE'd in stage 2d).
    Cq_taps = [
        TensorAccessPattern(
            tensor_dims=(1, bo4_elems),
            offset=bo4_off_q_rot + i * rows_per_col_q,
            sizes=[1, 1, tiles_per_col_q, rows_per_col_q // tiles_per_col_q],
            strides=[0, 0, rows_per_col_q // tiles_per_col_q, 1],
        )
        for i in range(cols)
    ]
    # Ck drains: BO4 k_rot region; rows_per_col_kv per col.
    Ck_taps = [
        TensorAccessPattern(
            tensor_dims=(1, bo4_elems),
            offset=bo4_off_k_rot + i * rows_per_col_kv,
            sizes=[1, 1, tiles_per_col_kv, rows_per_col_kv // tiles_per_col_kv],
            strides=[0, 0, rows_per_col_kv // tiles_per_col_kv, 1],
        )
        for i in range(cols)
    ]
    # Cv drains: BO4 v region (V doesn't get RoPE'd, goes straight to KV slot).
    Cv_taps = [
        TensorAccessPattern(
            tensor_dims=(1, bo4_elems),
            offset=bo4_off_v + i * rows_per_col_kv,
            sizes=[1, 1, tiles_per_col_kv, rows_per_col_kv // tiles_per_col_kv],
            strides=[0, 0, rows_per_col_kv // tiles_per_col_kv, 1],
        )
        for i in range(cols)
    ]

    # ATTN Kch_taps removed — BO3 will be wired via MemTile-mediated
    # streaming in the next step (separate attention worker per col).

    rt = Runtime()
    with rt.sequence(
        L3_w_qkv, L3_w_o, L3_w_ffn, L3_kv, L3_act,
    ) as (w_qkv, w_o, w_ffn, kv, act):
        rt.start(*qkv_workers, *skel_workers, attn_spike_worker)
        # ── Phase 1: pre-RMS inputs (col 0's worker does the compute) ───────
        # No drain — pre_rms_col0 kernel writes normed into the bq_l3l2
        # fifo producer slot (broadcast via MemTile) AND into a static L1
        # buffer (consumed by col 0's own QKV phases). Synchronization to
        # subsequent Q/K/V phases happens implicitly via bq_mem.cons() on
        # cols 1-7's bodies — they block until col 0 publishes.
        tg_rms = rt.task_group()
        rt.fill(rms_in_fifo.prod(), act,   x_tap,       task_group=tg_rms)
        rt.fill(rms_in_fifo.prod(), w_qkv, w_norm1_tap, task_group=tg_rms)
        rt.finish_task_group(tg_rms)

        # ── Phase 2: Q GEMV ─────────────────────────────────────────────────
        # Unified qkv body: A=Aq, B=normed broadcast (produced by col 0),
        # C drained to BO4 q_rot region. No host-side bq fill — col 0
        # produces it inline.
        # Unified qkv_worker first phase: A = Aq, B = normed broadcast,
        # C drained to BO4 q_rot region.
        tg_q = rt.task_group()
        for i in range(cols):
            rt.fill(Aqkv_fifos[i].prod(), w_qkv, Aq_taps[i], task_group=tg_q)
        # No bq_l3l2 fill — col 0's body produces it from pre_rms_col0 output.
        for i in range(cols):
            rt.drain(Cqkv_fifos[i].cons(), act, Cq_taps[i],
                     task_group=tg_q, wait=True)
        rt.finish_task_group(tg_q)

        # ── Phase 3: K GEMV ─────────────────────────────────────────────────
        # Same unified worker, second phase: A = Ak, B reused from col-0's
        # static buffer (col 0) or held bq_mem acquire (cols 1-7), C drained
        # to BO4 k_rot region.
        tg_k = rt.task_group()
        for i in range(cols):
            rt.fill(Aqkv_fifos[i].prod(), w_qkv, Ak_taps[i], task_group=tg_k)
        for i in range(cols):
            rt.drain(Cqkv_fifos[i].cons(), act, Ck_taps[i],
                     task_group=tg_k, wait=True)
        rt.finish_task_group(tg_k)

        # ── Phase 4: V GEMV ─────────────────────────────────────────────────
        # Third phase of unified worker: A = Av, B reused, C drained to
        # BO4 v region.
        tg_v = rt.task_group()
        for i in range(cols):
            rt.fill(Aqkv_fifos[i].prod(), w_qkv, Av_taps[i], task_group=tg_v)
        for i in range(cols):
            rt.drain(Cqkv_fifos[i].cons(), act, Cv_taps[i],
                     task_group=tg_v, wait=True)
        rt.finish_task_group(tg_v)

        # ATTN phase deferred — will land in next step with MemTile gather.

        # ── Skeleton chunk fills/drains (BO1, BO2 placeholders) ─────────────
        tg_skel = rt.task_group()
        rt.fill (skel_in [0].prod(), w_o,   tap_o,         task_group=tg_skel)
        rt.fill (skel_in [1].prod(), w_ffn, tap_ffn,       task_group=tg_skel)
        rt.drain(skel_out[0].cons(), act,   skel_drain_tap, task_group=tg_skel)
        rt.drain(skel_out[1].cons(), act,   skel_drain_tap, task_group=tg_skel,
                 wait=True)
        rt.finish_task_group(tg_skel)

        # ── Attention spike (1-tile probe on col 0 row 3) ───────────────────
        # Fills 1 KV chunk (32 tokens × HEAD_DIM bf16) from BO3 K_cache
        # via the MemTile-mediated gather path; drains the noop output to
        # BO4 attn_out region. Validates that an attention worker can be
        # placed on a separate compute row from the qkv worker.
        tg_attn = rt.task_group()
        rt.fill (kv_l3l2_fifo.prod(),  kv,  attn_kv_tap,    task_group=tg_attn)
        rt.drain(attn_drain_fifo.cons(), act, attn_drain_tap, task_group=tg_attn,
                 wait=True)
        rt.finish_task_group(tg_attn)

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
