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
from aie.dialects import arith
from aie.extras import types as T
from aie.helpers.dialects.scf import _for as range_
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker, Buffer
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
    # `packed_q` is bytes for ONE tile = m_input_qkv consecutive rows.
    # Per-col total = (rows_per_col / m_input_qkv) tiles × packed_q.
    # Forgetting `/ m_input_qkv` here overcounts the Q/K/V regions by
    # m_input_qkv × and shifts K/V offsets past where the packer writes.
    total_q      = cols * (e    // cols // m_input_qkv) * packed_q
    total_kv_one = cols * (kv_e // cols // m_input_qkv) * packed_q
    bo0_bytes  = norm_bytes + total_q + 2 * total_kv_one

    m_input_o  = 2  # R2-F1: DMA BD align (4 bytes min)
    packed_o   = m_input_o * e // 2 + m_input_o * groups_e * 2
    total_o    = cols * (e // cols // m_input_o) * packed_o
    bo1_bytes  = total_o

    m_input_gu = 4
    packed_gu  = m_input_gu * e // 2 + m_input_gu * groups_e * 2
    total_gu   = cols * 2 * (h // cols // m_input_gu) * packed_gu
    m_input_d  = 2  # R2-F3: DMA BD align (4 bytes min)
    packed_d   = m_input_d * h // 2 + m_input_d * groups_h * 2
    total_d    = cols * (e // cols // m_input_d) * packed_d
    bo2_bytes  = norm_bytes + total_gu + total_d

    kv_one_bytes = nkv * mx * hd * 2
    bo3_bytes    = 2 * kv_one_bytes

    bo4_elems = (
        e + 2 * mx * hd + e + e + kv_e + kv_e
        + e + e + e + e + e + h + e + e
        + cols * e  # ffn_out_partials (decomp-B: cols partial-E vectors)
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
    bo2_off_gate    = bo2_off_W_norm2 + e
    bo2_off_up      = bo2_off_gate    + total_gu // 4  # gate+up interleaved, each is half
    bo2_off_down    = bo2_off_gate    + total_gu // 2  # after both gate+up
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
    # decomp-B: each down col drains a full-E partial; host sums the cols.
    bo4_off_ffn_part  = bo4_off_outL     + e
    assert bo4_off_ffn_part + cols * e == bo4_elems

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

    # FFLM-style QKV weight distribution via MemTile split (replaces per-col
    # shim S2MM). 2 shim sources, each split → 4 cols. Frees 6 shim S2MM
    # (was 8 per-col, now 2 per-source). Mirrors O_proj split (L466-478).
    n_shim_qkv       = 2
    cols_per_shim_qkv = cols // n_shim_qkv            # 4 cols per source
    Aqkv_src_ty = np.ndarray[(cols_per_shim_qkv * packed_qkv_tile_bytes,), dtype_packed]

    # Attention KV chunk type: K_CHUNK tokens × HEAD_DIM bf16 elements.
    # Matches FFLM's ct_chunk_size=32 (mha.dll RE'd disasm) and our kernel
    # ATTN_K_CHUNK macro. Per-col attention reads its 1 kv_head slice from
    # BO3 via this fifo, streaming chunk-by-chunk.
    K_CHUNK = 32
    kv_chunk_elems = K_CHUNK * hd
    L1_kv_chunk_ty = np.ndarray[(kv_chunk_elems,), dtype_vec]
    # Sub-chunk type for MemTile gather: split each kv_chunk into
    # ATTN_SUBCHUNKS_PER_CHUNK sub-transfers under the AIE2P shim BD
    # inner-dim cap (1023). 2048 / 4 = 512 elements per sub-chunk.
    ATTN_SUBCHUNKS_PER_CHUNK = 4
    SUBCHUNK_ELEMS = kv_chunk_elems // ATTN_SUBCHUNKS_PER_CHUNK
    assert SUBCHUNK_ELEMS <= 1023, (
        f"sub-chunk {SUBCHUNK_ELEMS} must fit shim BD inner-dim cap"
    )
    L1_kv_subchunk_ty = np.ndarray[(SUBCHUNK_ELEMS,), dtype_vec]
    # Q-head slice: HEAD_DIM bf16 elements per head. 4 q_heads/col (GQA
    # mapping for n_q=32, cols=8). Worker body iterates 4 heads per outer
    # iter, acquiring one q_head per inner iter.
    L1_q_head_ty = np.ndarray[(hd,), dtype_vec]

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
    # Real attention compute spike — drop-in replacement for noop_kv_fn.
    # Same (kv_chunk, ctx_out, n) signature, so attn body + rt.sequence
    # stay unchanged. Internally chains qk_score → softmax → av_ctx via
    # per-tile L1 static scratch (q_head, scores, ctx_head).
    attn_compute_fn = Kernel(
        f"{func_prefix}attn_compute_chunk_bf16",
        f"{func_prefix}layer_fused_{e}_{h}_g{g}.o",
        [L1_kv_chunk_ty, L1_kv_chunk_ty, np.int32],
    )
    # Step-2 spike: q_head supplied via dedicated fifo (real q_rot data).
    # Worker drives the per-head loop (4 calls per outer iter).
    attn_compute_qhead_fn = Kernel(
        f"{func_prefix}attn_compute_with_qhead_bf16",
        f"{func_prefix}layer_fused_{e}_{h}_g{g}.o",
        [L1_q_head_ty, L1_kv_chunk_ty, L1_kv_chunk_ty, np.int32],
    )
    # Step-3 spike: streaming attention over multiple KV chunks. Worker
    # body iterates ATTN_N_CHUNKS K acquires + softmax + ATTN_N_CHUNKS V
    # acquires + drain. Each kernel below performs one step.
    attn_qk_score_at_fn = Kernel(
        f"{func_prefix}attn_qk_score_at_bf16",
        f"{func_prefix}layer_fused_{e}_{h}_g{g}.o",
        [L1_kv_chunk_ty, np.int32, np.int32],
    )
    attn_softmax_full_fn = Kernel(
        f"{func_prefix}attn_softmax_full_bf16",
        f"{func_prefix}layer_fused_{e}_{h}_g{g}.o",
        [np.int32],
    )
    attn_av_ctx_at_fn = Kernel(
        f"{func_prefix}attn_av_ctx_at_bf16",
        f"{func_prefix}layer_fused_{e}_{h}_g{g}.o",
        [L1_kv_chunk_ty, np.int32, np.int32],
    )
    attn_drain_ctx_fn = Kernel(
        f"{func_prefix}attn_drain_ctx_bf16",
        f"{func_prefix}layer_fused_{e}_{h}_g{g}.o",
        [L1_kv_chunk_ty, np.int32],
    )
    # Step-4 spike: 4-head dense compute on a single KV chunk. Drop-in
    # replacement for attn_compute_fn — same (kv_chunk, ctx_out, n)
    # signature, 4× the per-call compute density (4 q_heads in one
    # kernel invocation). Fits within shim 1-input + 1-output flow.
    attn_compute_4heads_fn = Kernel(
        f"{func_prefix}attn_compute_4heads_bf16",
        f"{func_prefix}layer_fused_{e}_{h}_g{g}.o",
        [L1_kv_chunk_ty, L1_kv_chunk_ty, np.int32],
    )
    # Step-5 spike: MemTile sub-chunk gather. Shim sends sub-chunks
    # (≤1023 elem each) → MemTile relays into compute tile L1. Compute
    # tile body acquires N_SUBCHUNKS subs (loaded into static via
    # attn_subchunk_load_bf16), then runs the full 4-head attention
    # (attn_compute_from_static_bf16). Same shape per kernel — kernel
    # arg type is the SUB type, not the full chunk.
    attn_subchunk_load_fn = Kernel(
        f"{func_prefix}attn_subchunk_load_bf16",
        f"{func_prefix}layer_fused_{e}_{h}_g{g}.o",
        [L1_kv_subchunk_ty, np.int32, np.int32],
    )
    attn_compute_from_static_fn = Kernel(
        f"{func_prefix}attn_compute_from_static_bf16",
        f"{func_prefix}layer_fused_{e}_{h}_g{g}.o",
        [L1_kv_chunk_ty, np.int32],
    )
    attn_qhead_load_fn = Kernel(
        f"{func_prefix}attn_qhead_load_bf16",
        f"{func_prefix}layer_fused_{e}_{h}_g{g}.o",
        [L1_q_head_ty, np.int32, np.int32],
    )
    attn_compute_real_qhead_acc_fn = Kernel(
        f"{func_prefix}attn_compute_real_qhead_acc_bf16",
        f"{func_prefix}layer_fused_{e}_{h}_g{g}.o",
        [np.int32],
    )
    attn_drain_real_ctx_fn = Kernel(
        f"{func_prefix}attn_drain_real_ctx_bf16",
        f"{func_prefix}layer_fused_{e}_{h}_g{g}.o",
        [L1_kv_chunk_ty, np.int32],
    )
    # Streaming variant: compute on the static buffer WITHOUT a drain
    # (ctx state stays in static for the next chunk). Last body chunk
    # calls attn_compute_from_static_fn (with drain) instead.
    attn_compute_static_acc_fn = Kernel(
        f"{func_prefix}attn_compute_from_static_acc_bf16",
        f"{func_prefix}layer_fused_{e}_{h}_g{g}.o",
        [np.int32],
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
    #
    # FFLM-style QKV weight split: 2 shim sources → MemTile split → 8 sub-fifos.
    # Frees 6 shim S2MM vs the old per-col approach (was 8, now 2).
    # MemTile placement [0,4] avoids overlap with O_proj splits [1,5].
    aqkv_mem_cols = [0, 4]
    aqkv_src_fifos = [ObjectFifo(Aqkv_src_ty, name=f"Aqkv_src_{s}", depth=2)
                      for s in range(n_shim_qkv)]
    Aqkv_sub = [None] * cols
    for s in range(n_shim_qkv):
        subs = aqkv_src_fifos[s].cons().split(
            [j * packed_qkv_tile_bytes for j in range(cols_per_shim_qkv)],
            obj_types=[L1_Aq_ty] * cols_per_shim_qkv,
            names=[f"Aqkv_{s * cols_per_shim_qkv + j}" for j in range(cols_per_shim_qkv)],
            placement=Tile(col=aqkv_mem_cols[s], row=1),
        )
        for j in range(cols_per_shim_qkv):
            Aqkv_sub[s * cols_per_shim_qkv + j] = subs[j]

    bq_l3l2_fifo = ObjectFifo(L1_Bq_ty, name="bq_L3L2", depth=1)
    # bq broadcast staged through MemTile col 2 (avoids overlap with QKV
    # split cols [0,4] and O_proj split cols [1,5]).
    bq_mem_fifo  = bq_l3l2_fifo.cons().forward(
        name="bq_mem", depth=1, placement=Tile(col=2, row=1))
    # FFLM-style: ONE Cqkv_joined[col] fifo per col (not Q+K+V separately).
    # Q/K/V phases share the same shim MM2S channel via time-multiplexed BD
    # chain in rt.inline_ops (npu_dma_memcpy_nd × 3 → cmds2seq, like FFLM).
    # 8 cols × 1 MM2S = 8 shim MM2S total for QKV outputs.
    Cqkv_joined = [ObjectFifo(L1_Cq_ty, name=f"Cqkv_{i}", depth=2)
                   for i in range(cols)]

    # ── Worker bodies ────────────────────────────────────────────────────────
    # Col 0: 4-phase body (pre-RMS → Q → K → V). Pre-RMS writes to BOTH the
    # static L1 buffer (consumed by qkv_gemv_static below) AND to the bq
    # fifo prod slot (broadcast). QKV phases use the static-input variant
    # so col 0's tile stays at 2 input channels (rms_in + Aqkv).
    # FFLM-style: ONE Cqkv output fifo per col, drained by 3 BD chain
    # (Q/K/V) in inline_ops. Worker writes m_input_qkv-row tiles back-to-back;
    # rt.inline_ops chain reorders them into BO4 q_rot/k_rot/v regions.
    def col0_body(rms_in, bq_prod, Aqkv, Cqkv, pre_rms_fn, qkv_static_fn):
        for _ in range_(0xFFFFFFFF):
            # Pre-RMS phase
            sub = rms_in.acquire(2)
            x_in  = sub[0]
            gain  = sub[1]
            bq_out = bq_prod.acquire(1)
            pre_rms_fn(x_in, gain, bq_out, e)
            rms_in.release(2)
            bq_prod.release(1)
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

    def colN_body(A, B, C, fn):
        for _ in range_(0xFFFFFFFF):
            b = B.acquire(1)
            for _ in range_(tiles_per_col_q):
                a = A.acquire(1); c = C.acquire(1)
                fn(m_input_qkv, 0, a, b, c)
                A.release(1); C.release(1)
            for _ in range_(tiles_per_col_kv):
                a = A.acquire(1); c = C.acquire(1)
                fn(m_input_qkv, 0, a, b, c)
                A.release(1); C.release(1)
            for _ in range_(tiles_per_col_kv):
                a = A.acquire(1); c = C.acquire(1)
                fn(m_input_qkv, 0, a, b, c)
                A.release(1); C.release(1)
            B.release(1)

    col0_worker = Worker(
        col0_body,
        [rms_in_fifo.cons(), bq_l3l2_fifo.prod(),
         Aqkv_sub[0].cons(), Cqkv_joined[0].prod(),
         pre_rms_col0_fn, qkv_gemv_static_fn],
    )
    qkv_workers_rest = [
        Worker(colN_body,
               [Aqkv_sub[i].cons(), bq_mem_fifo.cons(),
                Cqkv_joined[i].prod(), qkv_gemv_fn])
        for i in range(1, cols)
    ]
    qkv_workers = [col0_worker] + qkv_workers_rest

    # ── O_proj stage (P0.4-R2-F1): FFLM-style MemTile weight distribution ──────
    # Weights are NOT fed per-column-shim (that made col 0 hit 3 S2MM > cap).
    # Instead n_shim_o shim sources (on columns with room) each fill a small
    # L2 buffer of `cols_per_shim` tiles, MemTile-`split` routing one tile to
    # each of its columns' O_proj compute tiles (row 3). attn_out broadcast via
    # forward; o_out drained to BO4 (DDR baseline; R2-F2 chains on-chip).
    m_input_o       = 2   # 2 output rows/tile → drain = 4 bytes (DMA BD align)
    rows_per_col_o  = e // cols                       # 256
    tiles_per_col_o = rows_per_col_o // m_input_o     # 128
    packed_o_tile   = m_input_o * e // 2 + m_input_o * groups_e * 2  # 2304 B
    bytes_col_o     = tiles_per_col_o * packed_o_tile
    n_shim_o        = 2                                # 2 shim sources
    cols_per_shim   = cols // n_shim_o                 # 4 cols per source
    L1_Ao_ty   = np.ndarray[(packed_o_tile,), dtype_packed]            # 1 tile
    Ao_src_ty  = np.ndarray[(cols_per_shim * packed_o_tile,), dtype_packed]
    L1_Bo_ty   = np.ndarray[(e,), dtype_vec]
    L1_Co_ty   = np.ndarray[(m_input_o,), dtype_vec]
    o_proj_fn = Kernel(
        f"{func_prefix}layer_fused_o_proj_bf16",
        f"{func_prefix}layer_fused_{e}_{h}_g{g}.o",
        [np.int32, np.int32, L1_Ao_ty, L1_Bo_ty, L1_Co_ty],
    )
    # Weight sources + MemTile split → per-col weight sub-fifos.
    ao_mem_cols     = [1, 5]                          # MemTile cols for Ao split
    ao_src_fifos = [ObjectFifo(Ao_src_ty, name=f"Ao_src_{s}", depth=2)
                    for s in range(n_shim_o)]
    Ao_sub = [None] * cols
    for s in range(n_shim_o):
        subs = ao_src_fifos[s].cons().split(
            [j * packed_o_tile for j in range(cols_per_shim)],
            obj_types=[L1_Ao_ty] * cols_per_shim,
            names=[f"Ao_{s*cols_per_shim + j}" for j in range(cols_per_shim)],
            placement=Tile(col=ao_mem_cols[s], row=1))
        for j in range(cols_per_shim):
            Ao_sub[s * cols_per_shim + j] = subs[j]
    # attn_out broadcast (E bf16) from BO4 → O_proj workers: 1 shim S2MM.
    bo_l3l2_fifo = ObjectFifo(L1_Bo_ty, name="bo_L3L2", depth=1)
    bo_mem_fifo  = bo_l3l2_fifo.cons().forward(
        name="bo_mem", depth=1, placement=Tile(col=6, row=1))
    # ffn_in broadcast: ANM worker writes directly into this fifo (no shim fill).
    # gate/up workers consume via ffi_mem_fifo forward. 0 shim S2MM, 0 MM2S.
    ffi_l3l2_fifo = ObjectFifo(L1_Bo_ty, name="ffi_L3L2", depth=1)
    ffi_mem_fifo  = ffi_l3l2_fifo.cons().forward(
        name="ffi_mem", depth=1, placement=Tile(col=6, row=1))
    # P0.4-R2-F2b: O_proj outputs joined ON-CHIP at a MemTile (gather 8 cols
    # → one E vector) instead of 8 separate DDR drains. Each join "round"
    # assembles one tile (m_input_o rows) from each of the 8 cols; the joined
    # stream drains to BO4 with ONE MM2S (vs 8) via a reorder TAP. This frees
    # 7 MM2S — required to fit the global shim budget when ANM is added.
    # A MemTile has limited input DMA channels (<8), so an 8-way join on one
    # MemTile overflows. Gather in n_join groups (4 cols each) on separate
    # MemTiles → n_join drains (still frees 8-n_join MM2S vs per-col drains).
    n_join        = 2
    cols_per_join = cols // n_join                       # 4
    join_mem_cols = [1, 5]                               # free MemTile columns
    o_out_round_ty = np.ndarray[(cols_per_join * m_input_o,), dtype_vec]  # 8 bf16
    o_out_joined = [ObjectFifo(o_out_round_ty, name=f"o_out_joined_{jg}", depth=2)
                    for jg in range(n_join)]
    Co_sub = [None] * cols
    for jg in range(n_join):                 # NOT 'g' — would shadow group_size
        subs = o_out_joined[jg].prod().join(
            [j * m_input_o for j in range(cols_per_join)],
            obj_types=[L1_Co_ty] * cols_per_join,
            names=[f"Co_{jg*cols_per_join + j}" for j in range(cols_per_join)],
            placement=Tile(col=join_mem_cols[jg], row=1))
        for j in range(cols_per_join):
            Co_sub[jg * cols_per_join + j] = subs[j]

    # FFLM-style on-chip O_proj→ANM: o_out_joined[jg] is consumed directly by ANM
    # compute tile (col 0 row 4) via 2 .cons() ports. No DDR drain → saves 2 shim MM2S.

    def o_proj_body(Ao, Bo, Co, fn):
        for _ in range_(0xFFFFFFFF):
            b = Bo.acquire(1)
            for _ in range_(tiles_per_col_o):
                a = Ao.acquire(1)
                c = Co.acquire(1)
                fn(m_input_o, 0, a, b, c)
                Ao.release(1)
                Co.release(1)
            Bo.release(1)

    o_proj_workers = [
        Worker(o_proj_body,
               [Ao_sub[i].cons(), bo_mem_fifo.cons(),
                Co_sub[i].prod(), o_proj_fn],
               placement=Tile(col=i, row=3))
        for i in range(cols)
    ]

    # ── ANM leader (P0.4-R2-F2c): ADD + post-RMS, single tile ──────────────────
    # inpFF = o_out + inpL (into a LOCAL Buffer, not drained — final residual
    # handled in R2-F4); ffn_in = rms_norm(inpFF) * W_norm2, drained to BO4.
    # 1 input fifo (anm_in: 3 fills o_out/inpL/gain) + 1 output (ffn_in) keeps
    # the leader column at 2 S2MM + 2 MM2S (with its QKV Aqkv/Cqkv).
    add_fn = Kernel(
        f"{func_prefix}layer_fused_add_bf16",
        f"{func_prefix}layer_fused_{e}_{h}_g{g}.o",
        [L1_E_ty, L1_E_ty, L1_E_ty, np.int32],
    )
    rms2_fn = Kernel(
        f"{func_prefix}layer_fused_rms_norm2_bf16",
        f"{func_prefix}layer_fused_{e}_{h}_g{g}.o",
        [L1_E_ty, L1_E_ty, L1_E_ty, np.int32],
    )
    # FFLM-style: o_out arrives via DDR round-trip (2 S2MM cap).
    # anm_in_fifo carries o_out + inpL + gain (3 sequential fills).
    anm_in_fifo  = ObjectFifo(L1_E_ty, name="anm_in",  depth=3)
    # Staged through MemTile col 1 to free col 0's shim MM2S.
    anm_mem_fifo = anm_in_fifo.cons().forward(
        name="anm_mem", depth=3, placement=Tile(col=1, row=1))
    # ANM writes ffn_in directly into ffi_l3l2_fifo (no shim drain).
    # gate/up workers consume via ffi_mem_fifo forward. Saves 1 MM2S.
    anm_inpff_buf = Buffer(type=L1_E_ty, name="anm_inpff_buf")
    # Local buffer where ANM assembles full E o_out from on-chip forward fifos
    # (o_out_anm_0/1 each carry round-major chunks of cols_per_join*m_input_o
    # bf16 elements; assemble copies them into col-major position).
    anm_oout_buf  = Buffer(type=L1_E_ty, name="anm_oout_buf")

    o_out_assemble_fn = Kernel(
        f"{func_prefix}layer_fused_o_out_assemble_bf16",
        f"{func_prefix}layer_fused_{e}_{h}_g{g}.o",
        [np.ndarray[(cols_per_join * m_input_o,), dtype_vec],
         L1_E_ty,
         np.int32, np.int32, np.int32, np.int32, np.int32],
    )

    def anm_body(in_fifo, ffi_out, inpff, af, nf):
        for _ in range_(0xFFFFFFFF):
            sub = in_fifo.acquire(3)        # [o_out, inpL, gain]
            s, l, gn = sub[0], sub[1], sub[2]
            ffi = ffi_out.acquire(1)
            af(s, l, inpff, e)              # inpFF = o_out + inpL  (local buf)
            nf(inpff, gn, ffi, e)           # ffn_in = rms(inpFF) * gain
            in_fifo.release(3)
            ffi_out.release(1)

    # anm_worker FOLDED into gate_up col-0 worker (R2-F3): the layer needs
    # 33 core tiles (qkv8 + o_proj8 + anm1 + gate_up8 + down8) but AIE2P has
    # only 32 (4 core rows × 8 cols). The ANM leader's ADD+RMS runs ONCE per
    # dispatch and is sequenced (tg_anm) strictly before gate/up (tg_gu), so
    # it temporally shares col-0's gate_up tile — the FFLM tile-reuse idiom.
    # The ANM phase drains ffn_in to BO4; tg_gu then broadcasts it back. The
    # task_group barrier guarantees ANM finishes before the broadcast fill.

    # ── SwiGLU stage (P0.4-R2-F3): gate/up dual GEMV + silu*mul + down ──────────
    # Mirrors the PROVEN post_attn_fused convention (cols=4 reference), adapted
    # to cols=8 via MemTile weight split (shim S2MM cap). Per tile: gate GEMV
    # (phase 0 → kernel static lf_left_buf), up GEMV (phase 1 → lf_right_buf),
    # then silu_mul(left,right) → Cgu slice. ffn_in broadcast via forward;
    # silu_out drained PER-COLUMN to BO4 (8 MM2S ≤ 16 per-task-group cap, so no
    # join needed). down: K=H, silu_out broadcast, ffn_out per-col drain.
    #
    # BO2 gate/up layout (xdna_pack_layer_fused_w_ffn): per col, tiles are
    # interleaved [gate_t0|up_t0|gate_t1|up_t1|...]. So each col's weight
    # stream = 2*tiles_per_col_gu tiles of packed_gu_tile bytes. The MemTile
    # split must route 2 consecutive tiles (gate,up of round t) — handled by
    # the worker acquiring twice from its per-col sub-fifo.
    rows_per_col_gu  = h // cols                       # 1024
    tiles_per_col_gu = rows_per_col_gu // m_input_gu   # 256
    packed_gu_tile   = m_input_gu * e // 2 + m_input_gu * groups_e * 2
    bytes_col_gu     = 2 * tiles_per_col_gu * packed_gu_tile  # gate+up
    n_shim_gu        = 2
    cols_per_shim_gu = cols // n_shim_gu               # 4 (split 2→4, ≤6 MM2S per MemTile)

    rows_per_col_d   = e // cols                       # 256
    tiles_per_col_d  = rows_per_col_d // m_input_d     # 128
    packed_d_tile    = m_input_d * h // 2 + m_input_d * groups_h * 2
    bytes_col_d      = tiles_per_col_d * packed_d_tile
    n_shim_d         = 2
    cols_per_shim_d  = cols // n_shim_d                # 4

    # ── decomp-B (on-chip FFLM): down with K = INTER_DIM_PER_COL = H/cols ──────
    # Each column reduces ONLY its 1024-elem silu slice (kept on-chip via
    # inter_fifo, NO BO4 bounce) against W_down[:, c*K:(c+1)*K], producing a
    # PARTIAL E vector. Host sums the cols partials → ffn_out. silu_mul writes
    # the FULL per-col slice (1024) into inter_fifo in one shot.
    inter_dim_per_col = h // cols                       # 1024 = K for down-B
    # down-B weight tile: m_input_dp rows × (K/2 INT4 + K/g*2 scale) bytes.
    m_input_dp        = 2                               # DMA BD align (4 B drain)
    rows_per_col_dp   = e                               # 2048 (FULL E partial)
    tiles_per_col_dp  = rows_per_col_dp // m_input_dp   # 1024
    groups_inter      = inter_dim_per_col // g          # 32
    packed_dp_tile    = m_input_dp * inter_dim_per_col // 2 + m_input_dp * groups_inter * 2
    bytes_col_dp      = tiles_per_col_dp * packed_dp_tile
    n_shim_dp         = 2
    cols_per_shim_dp  = cols // n_shim_dp               # 4 (split 2→4, ≤6 MM2S per MemTile)

    L1_Agu_ty   = np.ndarray[(packed_gu_tile,), dtype_packed]            # 1 tile
    Agu_src_ty  = np.ndarray[(cols_per_shim_gu * packed_gu_tile,), dtype_packed]  # 1 round
    L1_Bgu_ty   = np.ndarray[(e,), dtype_vec]
    # silu slice carried on-chip to down-B (full per-col 1024 elements).
    L1_inter_ty = np.ndarray[(inter_dim_per_col,), dtype_vec]
    # down-B: weight tile, silu input (held = inter), partial-E output slice.
    L1_Adp_ty   = np.ndarray[(packed_dp_tile,), dtype_packed]
    Adp_src_ty  = np.ndarray[(cols_per_shim_dp * packed_dp_tile,), dtype_packed]
    L1_Cdp_ty   = np.ndarray[(m_input_dp,), dtype_vec]

    gate_up_fn = Kernel(
        f"{func_prefix}layer_fused_gate_up_bf16",
        f"{func_prefix}layer_fused_{e}_{h}_g{g}.o",
        [np.int32, np.int32, L1_Agu_ty, L1_Bgu_ty, np.int32],
    )
    silu_mul_fn = Kernel(
        f"{func_prefix}layer_fused_silu_mul_bf16",
        f"{func_prefix}layer_fused_{e}_{h}_g{g}.o",
        [L1_inter_ty, np.int32],
    )
    down_fn = Kernel(
        f"{func_prefix}layer_fused_down_partial_bf16",
        f"{func_prefix}layer_fused_{e}_{h}_g{g}.o",
        [np.int32, np.int32, L1_Adp_ty, L1_inter_ty, L1_Cdp_ty],
    )

    # gate/up weights split (mirrors the proven O_proj split): the shim
    # source L2 buffer holds ONE ROUND = cols_per_shim_gu tiles; split routes
    # tile j to col j's compute tile. The agu_src_taps TAP streams
    # 2*tiles_per_col_gu rounds (the interleaved [g0|u0|g1|u1...] BO2 layout),
    # so each col's worker sees gate,up,gate,up,... on successive acquires.
    agu_mem_cols     = [2, 6]                            # cols 2,6 — each gets 4 MM2S
    agu_src_fifos = [ObjectFifo(Agu_src_ty, name=f"Agu_src_{s}", depth=2)
                     for s in range(n_shim_gu)]
    Agu_sub = [None] * cols
    for s in range(n_shim_gu):
        subs = agu_src_fifos[s].cons().split(
            [j * packed_gu_tile for j in range(cols_per_shim_gu)],
            obj_types=[L1_Agu_ty] * cols_per_shim_gu,
            names=[f"Agu_{s*cols_per_shim_gu + j}" for j in range(cols_per_shim_gu)],
            placement=Tile(col=agu_mem_cols[s], row=1))
        for j in range(cols_per_shim_gu):
            Agu_sub[s * cols_per_shim_gu + j] = subs[j]

    # ffn_in broadcast (E bf16) from BO4
    # ffi_l3l2_fifo declared above; ANM worker writes ffn_in directly into it.

    # ── inter_fifos: gate/up silu → down, ON-CHIP (decomp-B, no DDR bounce) ────
    # prod() on the gate/up worker (col i), cons() on the down worker (col i).
    # IRON routes this compute-tile → compute-tile stream WITHOUT a MemTile
    # placement or BO4 drain — the donor swiglu_fused_decode idiom (F5 lever).
    # depth=2 so gate/up can start the next token's slice while down consumes.
    inter_fifos = [ObjectFifo(L1_inter_ty, name=f"inter_{i}", depth=2)
                   for i in range(cols)]

    # gate/up body (decomp-B): accumulate ALL gate tiles → lf_left_buf,
    # ALL up tiles → lf_right_buf (per-tile m_input_gu rows at row_offset),
    # then ONE silu_mul over the full 1024-elem slice → inter_fifo.
    def gate_up_body(Agu, Bgu, Inter, gu_fn, sm_fn):
        for _ in range_(0xFFFFFFFF):
            b = Bgu.acquire(1)
            # interleaved stream [g0|u0|g1|u1...]: gate phase writes left_buf
            # at advancing row_offset, up phase writes right_buf.
            ro = 0
            for _ in range_(tiles_per_col_gu):
                a_g = Agu.acquire(1)
                gu_fn(m_input_gu, ro, a_g, b, 0)   # gate -> lf_left_buf[ro:]
                Agu.release(1)
                a_u = Agu.acquire(1)
                gu_fn(m_input_gu, ro, a_u, b, 1)   # up   -> lf_right_buf[ro:]
                Agu.release(1)
                ro += m_input_gu
            inter = Inter.acquire(1)
            sm_fn(inter, inter_dim_per_col)        # silu(left)*right -> inter
            Inter.release(1)
            Bgu.release(1)

    # col-0 variant: folded ANM (ADD+post-RMS) FIRST (sequenced by tg_anm
    # before tg_gu), then the gate/up decomp-B loop. Frees the standalone ANM
    # tile so the layer fits 32 core tiles (qkv8 + o_proj8 + gate_up8 + down8).
    # FFLM-style on-chip o_out: assemble from o_out_anm forward fifos (no DDR).
    def gate_up_anm_body(anm_in, anm_ffi, inpff, af, nf,
                         Agu, Inter, gu_fn, sm_fn):
        for _ in range_(0xFFFFFFFF):
            # ── ANM (one shot per dispatch) ──
            sub = anm_in.acquire(3)        # [o_out, inpL, gain]
            o_out, inpL, gn = sub[0], sub[1], sub[2]
            ffi = anm_ffi.acquire(1)
            af(o_out, inpL, inpff, e)      # inpFF = o_out + inpL
            nf(inpff, gn, ffi, e)          # ffn_in = rms(inpFF) * gain
            anm_in.release(3)
            # ── gate/up decomp-B (uses local ffi directly, bypassing Bgu) ──
            ro = 0
            for _ in range_(tiles_per_col_gu):
                a_g = Agu.acquire(1)
                gu_fn(m_input_gu, ro, a_g, ffi, 0)
                Agu.release(1)
                a_u = Agu.acquire(1)
                gu_fn(m_input_gu, ro, a_u, ffi, 1)
                Agu.release(1)
                ro += m_input_gu
            inter = Inter.acquire(1)
            sm_fn(inter, inter_dim_per_col)
            Inter.release(1)
            anm_ffi.release(1)

    gate_up_workers = [
        Worker(gate_up_anm_body,
               [anm_mem_fifo.cons(), ffi_l3l2_fifo.prod(),
                anm_inpff_buf, add_fn, rms2_fn,
                Agu_sub[0].cons(),
                inter_fifos[0].prod(), gate_up_fn, silu_mul_fn],
               placement=Tile(col=0, row=4))
    ] + [
        Worker(gate_up_body,
               [Agu_sub[i].cons(), ffi_mem_fifo.cons(),
                inter_fifos[i].prod(), gate_up_fn, silu_mul_fn],
               placement=Tile(col=i, row=4))
        for i in range(1, cols)
    ]

    # down-B weights split: W_down[:, c*K:(c+1)*K] per col, K=inter_dim_per_col.
    # 2 shim sources (n_shim_dp=2), each split → 4 cols on its MemTile (4 MM2S).
    adp_mem_cols     = [3, 7]                            # MemTile cols for Adp split
    adp_src_fifos = [ObjectFifo(Adp_src_ty, name=f"Adp_src_{s}", depth=2)
                     for s in range(n_shim_dp)]
    Adp_sub = [None] * cols
    for s in range(n_shim_dp):
        subs = adp_src_fifos[s].cons().split(
            [j * packed_dp_tile for j in range(cols_per_shim_dp)],
            obj_types=[L1_Adp_ty] * cols_per_shim_dp,
            names=[f"Adp_{s*cols_per_shim_dp + j}" for j in range(cols_per_shim_dp)],
            placement=Tile(col=adp_mem_cols[s], row=1))
        for j in range(cols_per_shim_dp):
            Adp_sub[s * cols_per_shim_dp + j] = subs[j]

    # FFLM-style on-chip Down partials join to save 6 global S2MM channels:
    # Join Down partials on MemTiles (gather 8 cols -> 2 groups -> 2 drains)
    n_join_dp        = 2
    cols_per_join_dp = cols // n_join_dp                       # 4
    join_mem_cols_dp = [3, 7]                                 # free MemTile columns
    ffn_out_round_ty = np.ndarray[(cols_per_join_dp * m_input_dp,), dtype_vec]
    ffn_out_joined = [ObjectFifo(ffn_out_round_ty, name=f"ffn_out_joined_{jg}", depth=2)
                      for jg in range(n_join_dp)]
    Cdp_sub = [None] * cols
    for jg in range(n_join_dp):
        subs = ffn_out_joined[jg].prod().join(
            [j * m_input_dp for j in range(cols_per_join_dp)],
            obj_types=[L1_Cdp_ty] * cols_per_join_dp,
            names=[f"Cdp_{jg*cols_per_join_dp + j}" for j in range(cols_per_join_dp)],
            placement=Tile(col=join_mem_cols_dp[jg], row=1))
        for j in range(cols_per_join_dp):
            Cdp_sub[jg * cols_per_join_dp + j] = subs[j]

    # down-B body: hold the col's silu slice, GEMV against W_down → partial E.
    def down_body(Adp, Inter, Cdp, fn):
        for _ in range_(0xFFFFFFFF):
            b = Inter.acquire(1)
            for _ in range_(tiles_per_col_dp):
                a = Adp.acquire(1)
                c = Cdp.acquire(1)
                fn(m_input_dp, 0, a, b, c)
                Adp.release(1)
                Cdp.release(1)
            Inter.release(1)

    down_workers = [
        Worker(down_body,
               [Adp_sub[i].cons(), inter_fifos[i].cons(),
                Cdp_sub[i].prod(), down_fn],
               placement=Tile(col=i, row=5))
        for i in range(cols)
    ]

    # ── Skeleton passthroughs DROPPED for attention budget ───────────────────
    # Tile budget: 0 pre_rms + 8 qkv + 8 attn = 16 of 16 (full). BO1
    # (w_o) and BO2 (w_ffn) lose their compute-tile placeholder at this
    # checkpoint — their DDR_PATCH ops will be re-introduced in stage 3
    # when O_proj and SwiGLU workers consume them. The runtime sequence
    # at this checkpoint does NOT touch BO1/BO2; XRT accepts unused arg
    # slots without complaint.
    skel_workers = []  # no compute tiles consumed

    # ── Attention workers (7 cols 1-7, col 0 SKIPPED for shim-channel cap) ──
    # AIE2P shim has 2 S2MM + 2 MM2S channels per col. Col 0 is already at
    # With QKV MemTile-split: col 0 no longer has per-col Aqkv shim S2MM.
    # Col 0 shim = rms_in S2MM + Cqkv MM2S = 1+1 (room for more).
    #
    # For this checkpoint we wire attention on cols 1-7 only (7 of 8 kv
    # heads). Col 0's kv_head (head 0) does NOT get attention compute on
    # NPU — it falls back to host CPU for that single head, contributing
    # ~12% of attention work back to CPU. Acceptable for v0; fix in next
    # step by swapping col 0's QKV worker to a smaller variant or by
    # rebalancing pre-RMS off col 0.
    # R2a (P0.4): attention spike REMOVED. FFLM keeps attention in a
    # SEPARATE attn.xclbin (CDO-proven) — it does NOT belong in the layer
    # kernel. attn_out is fed from a BO region by a separate dispatch.
    # Setting attn_cols=[] makes every attn fifo/worker/tap comprehension
    # below empty and every tg_attn loop a no-op; the attn_gather_body
    # defs remain but are never instantiated.
    attn_cols = []  # was list(range(1, cols))
    n_attn = len(attn_cols)
    n_q_per_attn = num_heads // cols  # 4 q_heads per attn col (GQA)
    # MemTile gather + multi-chunk streaming. Per body iter:
    #   N_CHUNKS = 2 full kv_chunks (each = ATTN_SUBCHUNKS_PER_CHUNK
    #   sub-chunks). Total sub acquires per body iter = N_CHUNKS *
    #   ATTN_SUBCHUNKS_PER_CHUNK = 8.
    # ctx is overwritten each chunk (last chunk wins) — semantics
    # spike-only; correctness lands in A1.3.5 with online softmax.
    ATTN_N_CHUNKS = 16
    SUBS_PER_BODY = ATTN_N_CHUNKS * ATTN_SUBCHUNKS_PER_CHUNK
    # Number of REAL q_heads streamed per body iter via the q_rot fifo
    # (rest of the heads_per_tile use kv_chunk seed). Spike — keeping
    # the fifo small to bound shim BD count.
    N_QHEADS_LOADED = 4
    kv_l3l2_fifos = [ObjectFifo(L1_kv_subchunk_ty, name=f"kv_L3L2_{c}",
                                depth=ATTN_SUBCHUNKS_PER_CHUNK)
                     for c in attn_cols]
    kv_mem_fifos  = [kv_l3l2_fifos[i].cons().forward(
                        name=f"kv_mem_{c}",
                        depth=ATTN_SUBCHUNKS_PER_CHUNK,
                        placement=Tile(col=c, row=1))
                     for i, c in enumerate(attn_cols)]
    attn_drain_fifos = [ObjectFifo(L1_kv_chunk_ty, name=f"attn_drain_{c}",
                                   depth=1) for c in attn_cols]

    # NOTE: q_rot via MemTile gather attempted but shim 2-S2MM cap is
    # was exhausted on EVERY col before QKV MemTile-split. Now that Aqkv
    # goes via split (not per-col shim), cols 1-7 have room. Dropping in a
    # entry fails placement.
    #
    # Real fix needs producer-side rerouting: e.g. drop the dedicated
    # rms_in shim path and have col 0's qkv worker read the activation
    # from BO4 via a different channel, freeing col 0 shim S2MM for
    # the q_rot producer. That's an L1.3.x rearchitecture, not a spike
    # tweak. Kernels left in `.cc` for that step.

    # NOTE: a real q_rot input fifo would push attn cols 1-7 to 3 shim
    # S2MM channels (Aqkv + kv_l3l2 + q_rot), exceeding the AIE2P 2-S2MM
    # cap. The MemTile gather here only solves the inner-dim cap; the
    # q_rot blocker still needs a Cqkv→MemTile→attn cross-col reroute
    # (separate step).

    def attn_gather_body(kv_in, drain_out, load_fn, compute_fn, acc_fn):
        # Multi-chunk streaming via MemTile gather:
        #   for each chunk c in [0, ATTN_N_CHUNKS):
        #     for each sub s in [0, ATTN_SUBCHUNKS_PER_CHUNK):
        #       acquire sub, load_fn(sub, s, SUBCHUNK_ELEMS)
        #     if last chunk: compute_fn drains ctx_out
        #     else:          acc_fn accumulates into static ctx
        for _ in range_(0xFFFFFFFF):
            for _c in range(ATTN_N_CHUNKS):
                for k in range(ATTN_SUBCHUNKS_PER_CHUNK):
                    sub = kv_in.acquire(1)
                    load_fn(sub, k, SUBCHUNK_ELEMS)
                    kv_in.release(1)
                if _c == ATTN_N_CHUNKS - 1:
                    o = drain_out.acquire(1)
                    compute_fn(o, kv_chunk_elems)
                    drain_out.release(1)
                else:
                    # zero_first=1 only on the very first chunk of the iter
                    zf = 1 if _c == 0 else 0
                    acc_fn(zf)

    # Col-1 body with REAL q_rot input via MemTile gather. Per body
    # iter: load N_QHEADS_LOADED q_heads (each HEAD_DIM bf16) from the
    # q_rot fifo into static, then run the streaming attention chain.
    def attn_gather_body_q(kv_in, q_in, drain_out,
                           qload_fn, kload_fn, real_acc_fn, drain_real_fn):
        for _ in range_(0xFFFFFFFF):
            # Load q_heads once per body iter (real q_rot data).
            for h in range(N_QHEADS_LOADED):
                q = q_in.acquire(1)
                qload_fn(q, h, hd)
                q_in.release(1)
            # Stream kv chunks; last chunk drains.
            for _c in range(ATTN_N_CHUNKS):
                for k in range(ATTN_SUBCHUNKS_PER_CHUNK):
                    sub = kv_in.acquire(1)
                    kload_fn(sub, k, SUBCHUNK_ELEMS)
                    kv_in.release(1)
                zf = 1 if _c == 0 else 0
                real_acc_fn(zf)
            o = drain_out.acquire(1)
            drain_real_fn(o, kv_chunk_elems)
            drain_out.release(1)

    # Build per-col attn workers. All cols use the streaming kv-seed
    # variant; q_rot wiring is blocked by the shim 2-S2MM cap (see
    # NOTE above). Col-1 q_rot variant kept in commented form for the
    # rearchitecture step.
    attn_workers = [
        Worker(attn_gather_body,
               [kv_mem_fifos[i].cons(), attn_drain_fifos[i].prod(),
                attn_subchunk_load_fn, attn_compute_from_static_fn,
                attn_compute_static_acc_fn])
        for i in range(n_attn)
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

    # Attention spike (MemTile gather + multi-chunk streaming): per-col
    # TAP sweeps ATTN_N_CHUNKS × ATTN_SUBCHUNKS_PER_CHUNK contiguous
    # sub-chunks (each SUBCHUNK_ELEMS=512 bf16 ≤ shim cap 1023) starting
    # at the col's kv_head offset. Net transfer = N_CHUNKS * 2048 bf16.
    attn_kv_taps = [
        TensorAccessPattern(
            tensor_dims=(1, bo3_bytes // 2),
            offset=c * (mx * hd),
            sizes=[1, 1, SUBS_PER_BODY, SUBCHUNK_ELEMS],
            strides=[0, 0, SUBCHUNK_ELEMS, 1],
        )
        for c in attn_cols
    ]
    # All cols drain into BO4 attn_out (overlapping; spike-only).
    attn_drain_taps = [
        TensorAccessPattern(
            tensor_dims=(1, bo4_elems),
            offset=bo4_off_attn_out,
            sizes=[1, 1, 1, kv_chunk_elems],
            strides=[0, 0, 0, 1],
        )
        for _ in attn_cols
    ]

    # ── Q GEMV TAPs ──────────────────────────────────────────────────────────
    # FFLM-style QKV weight TAPs for MemTile-split sources.
    # Each aqkv_src_tap[phase][s] assembles one round of tiles from
    # cols_per_shim_qkv columns, reading BO0's per-col-contiguous layout
    # with a column stride. Mirrors O_proj split TAPs (ao_src_taps).
    # Phase 0=Q, 1=K, 2=V. Each source s covers cols [s*cps .. (s+1)*cps).
    Aqkv_src_q_taps = [
        TensorAccessPattern(
            tensor_dims=(1, bo0_bytes // 2),
            offset=bo0_off_W_q + s * cols_per_shim_qkv * (bytes_col_q // 2),
            sizes=[1, tiles_per_col_q, cols_per_shim_qkv, packed_qkv_tile_bytes // 2],
            strides=[0, packed_qkv_tile_bytes // 2, bytes_col_q // 2, 1],
        )
        for s in range(n_shim_qkv)
    ]
    Aqkv_src_k_taps = [
        TensorAccessPattern(
            tensor_dims=(1, bo0_bytes // 2),
            offset=bo0_off_W_k + s * cols_per_shim_qkv * (bytes_col_kv // 2),
            sizes=[1, tiles_per_col_kv, cols_per_shim_qkv, packed_qkv_tile_bytes // 2],
            strides=[0, packed_qkv_tile_bytes // 2, bytes_col_kv // 2, 1],
        )
        for s in range(n_shim_qkv)
    ]
    Aqkv_src_v_taps = [
        TensorAccessPattern(
            tensor_dims=(1, bo0_bytes // 2),
            offset=bo0_off_W_v + s * cols_per_shim_qkv * (bytes_col_kv // 2),
            sizes=[1, tiles_per_col_kv, cols_per_shim_qkv, packed_qkv_tile_bytes // 2],
            strides=[0, packed_qkv_tile_bytes // 2, bytes_col_kv // 2, 1],
        )
        for s in range(n_shim_qkv)
    ]
    # Bq is normed (E bf16) in BO4 — broadcast via MemTile.
    bq_tap = TensorAccessPattern(
        tensor_dims=(1, bo4_elems),
        offset=bo4_off_normed,
        sizes=[1, 1, 1, e],
        strides=[0, 0, 0, 1],
    )
    # Per-col offsets (BO4 element units). inline_ops chain uses these as
    # shim_dma_bd offsets — Q/K/V/down all share one Cqkv/Cdp shim channel.
    Cq_offsets = [bo4_off_q_rot + i * rows_per_col_q for i in range(cols)]
    Ck_offsets = [bo4_off_k_rot + i * rows_per_col_kv for i in range(cols)]
    Cv_offsets = [bo4_off_v     + i * rows_per_col_kv for i in range(cols)]

    # ── O_proj TAPs (P0.4-R2-F1) ─────────────────────────────────────────────
    # ao_src[s] fill: assemble each streamed object as [col0_tile_j | col1_.. |
    # ..colN_tile_j] for this shim's cols_per_shim columns, reading BO1's
    # per-col-contiguous layout with a column stride of bytes_col_o.
    ao_src_taps = [
        TensorAccessPattern(
            tensor_dims=(1, bo1_bytes // 2),
            offset=s * cols_per_shim * (bytes_col_o // 2),
            sizes=[16, 64, 18, 32],
            strides=[(packed_o_tile // 2) * 32, (packed_o_tile // 2) // 2, 32, 1],
        )
        for s in range(n_shim_o)
    ]
    attn_out_tap = TensorAccessPattern(
        tensor_dims=(1, bo4_elems),
        offset=bo4_off_attn_out,
        sizes=[1, 1, 1, e],
        strides=[0, 0, 0, 1],
    )
    # ── ANM TAPs (P0.4-R2-F2c) ───────────────────────────────────────────────
    def _bo4_lin(off):
        return TensorAccessPattern(tensor_dims=(1, bo4_elems), offset=off,
                                   sizes=[1, 1, 1, e], strides=[0, 0, 0, 1])
    anm_oout_tap = _bo4_lin(bo4_off_o_out)    # o_out  (BO4, written by join)
    anm_inpL_tap = _bo4_lin(bo4_off_x)        # inpL   (BO4)
    anm_gain_tap = TensorAccessPattern(tensor_dims=(1, bo2_bytes // 2),
                                       offset=bo2_off_W_norm2,
                                       sizes=[1, 1, 1, e], strides=[0, 0, 0, 1])
    anm_ffi_tap  = _bo4_lin(bo4_off_ffn_in)   # ffn_in drain

    # ── SwiGLU TAPs (P0.4-R2-F3, decomp-B on-chip) ───────────────────────────
    # gate/up weights: BO2 interleaved [g0|u0|g1|u1...] per col, stride
    # bytes_col_gu between cols. The split L2 source holds ONE round
    # (cols_per_shim_gu tiles); this TAP streams 2*tiles_per_col_gu rounds —
    # round r reads tile r of each of this shim's cols (gate when r even, up
    # when r odd, matching the interleave). Mirrors the O_proj split TAP.
    agu_src_taps = [
        TensorAccessPattern(
            tensor_dims=(1, bo2_bytes // 2),
            offset=bo2_off_gate + s * cols_per_shim_gu * (bytes_col_gu // 2),
            sizes=[32, 64, 72, 32],
            strides=[packed_gu_tile // 2, (packed_gu_tile // 2) * 32, 32, 1],
        )
        for s in range(n_shim_gu)
    ]
    ffn_in_tap = _bo4_lin(bo4_off_ffn_in)

    # down-B weights: W_down packed per-col (K=inter_dim_per_col slice). Split
    # L2 source holds ONE round (cols_per_shim_dp tiles); TAP streams
    # tiles_per_col_dp rounds.
    adp_src_taps = [
        TensorAccessPattern(
            tensor_dims=(1, bo2_bytes // 2),
            offset=bo2_off_down + s * cols_per_shim_dp * (bytes_col_dp // 2),
            sizes=[2 * cols_per_shim_dp, tiles_per_col_dp // 2, 18, 32],
            strides=[(tiles_per_col_dp // 2) * (packed_dp_tile // 2), packed_dp_tile // 2, 32, 1],
        )
        for s in range(n_shim_dp)
    ]

    # Per-col ffn_out partial drain offsets (each col writes a full-E partial
    # at offset i*E into BO4 ffn_part region).
    ffn_part_offsets = [bo4_off_ffn_part + i * e for i in range(cols)]

    # o_out_joined[g] drains round-major [round r][col-in-group][row] → BO4
    # col-major o_out[(g*4 + col)*256 + r*m_input_o + row]. Mirror of split.
    o_out_join_taps = [
        TensorAccessPattern(
            tensor_dims=(1, bo4_elems),
            offset=bo4_off_o_out + jg * cols_per_join * rows_per_col_o,
            sizes=[1, tiles_per_col_o, cols_per_join, m_input_o],
            strides=[0, m_input_o, rows_per_col_o, 1],
        )
        for jg in range(n_join)
    ]

    ffn_out_join_taps = [
        TensorAccessPattern(
            tensor_dims=(1, bo4_elems),
            offset=bo4_off_ffn_part + jg * cols_per_join_dp * rows_per_col_dp,
            sizes=[2, tiles_per_col_dp // 2, cols_per_join_dp, m_input_dp],
            strides=[(tiles_per_col_dp // 2) * m_input_dp, m_input_dp, rows_per_col_dp, 1],
        )
        for jg in range(n_join_dp)
    ]

    rt = Runtime()
    with rt.sequence(
        L3_w_qkv, L3_w_o, L3_w_ffn, L3_kv, L3_act,
    ) as (w_qkv, w_o, w_ffn, kv, act):
        rt.start(*qkv_workers, *skel_workers, *attn_workers, *o_proj_workers,
                 *gate_up_workers, *down_workers)
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

        # ── Phase 2-4: QKV GEMV — FFLM-style time-multiplex BD chain ──────────
        # Per col: ONE shim MM2S channel carries Q→K→V via 3 chained BDs
        # (npu_dma_memcpy_nd × 3 → cmds2seq, exactly the FFLM mechanism).
        # Weights load via MemTile-split sources (2 S2MM, not 8 per-col).
        # Each source fills Q→K→V tiles in order; the split routes each
        # round's sub-tile to the correct col's compute tile.
        tg_qkv = rt.task_group()
        # Q phase: weight fills via split sources
        for s in range(n_shim_qkv):
            rt.fill(aqkv_src_fifos[s].prod(), w_qkv, Aqkv_src_q_taps[s], task_group=tg_qkv)
        for i in range(cols):
            # Dummy drain to register the shim cons endpoint (creates the
            # shim_dma_allocation that the inline_ops chain then re-uses).
            tap_dummy = TensorAccessPattern(
                tensor_dims=(1, bo4_elems),
                offset=Cq_offsets[i],
                sizes=[1, 1, 1, m_input_qkv],
                strides=[0, 0, 0, 1],
            )
            rt.drain(Cqkv_joined[i].cons(), act, tap_dummy, task_group=tg_qkv)
        # K phase weights via split sources
        for s in range(n_shim_qkv):
            rt.fill(aqkv_src_fifos[s].prod(), w_qkv, Aqkv_src_k_taps[s], task_group=tg_qkv)
        # V phase weights via split sources
        for s in range(n_shim_qkv):
            rt.fill(aqkv_src_fifos[s].prod(), w_qkv, Aqkv_src_v_taps[s], task_group=tg_qkv)

        # FFLM time-multiplex chain: per col, configure ONE task with 3 BDs
        # (Q→K→V) on the same shim MM2S channel. This is the npu_dma_memcpy_nd
        # × 3 pattern that gen_layer_seq composes via cmds2seq.
        def qkv_drain_chain(act_rt):
            act_op = act_rt.op
            for i in range(cols):
                task = dma_configure_task_for(f"Cqkv_{i}", issue_token=True)
                with bds(task) as bd:
                    with bd[0]:  # Q
                        shim_dma_bd(act_op,
                                    offset=Cq_offsets[i],
                                    sizes=[1, 1, 1, rows_per_col_q - m_input_qkv],
                                    strides=[0, 0, 0, 1])
                        EndOp()
                    with bd[1]:  # K
                        shim_dma_bd(act_op,
                                    offset=Ck_offsets[i],
                                    sizes=[1, 1, 1, rows_per_col_kv],
                                    strides=[0, 0, 0, 1])
                        EndOp()
                    with bd[2]:  # V
                        shim_dma_bd(act_op,
                                    offset=Cv_offsets[i],
                                    sizes=[1, 1, 1, rows_per_col_kv],
                                    strides=[0, 0, 0, 1])
                        EndOp()
                dma_start_task(task)
                dma_await_task(task)

        rt.inline_ops(qkv_drain_chain, [act])
        rt.finish_task_group(tg_qkv)

        # ── Phase 5: O_proj GEMV (P0.4-R2-F1) ───────────────────────────────
        # FFLM-style: weights via MemTile split from n_shim_o sources,
        # attn_out broadcast via forward. o_out joined and drained to DDR.
        tg_o = rt.task_group()
        for s in range(n_shim_o):
            rt.fill(ao_src_fifos[s].prod(), w_o, ao_src_taps[s], task_group=tg_o)
        rt.fill(bo_l3l2_fifo.prod(), act, attn_out_tap, task_group=tg_o)
        for jg in range(n_join):
            rt.drain(o_out_joined[jg].cons(), act, o_out_join_taps[jg], task_group=tg_o)
        rt.finish_task_group(tg_o)

        # ── Phase 6: ANM (ADD + post-RMS), o_out → ffn_in (DDR round-trip) ──
        # ANM worker writes ffn_in directly into ffi_l3l2_fifo (no DDR drain).
        # gate/up workers consume via ffi_mem_fifo forward in tg_ffn.
        tg_anm = rt.task_group()
        rt.fill(anm_in_fifo.prod(), act,   anm_oout_tap, task_group=tg_anm)
        rt.fill(anm_in_fifo.prod(), act,   anm_inpL_tap, task_group=tg_anm)
        rt.fill(anm_in_fifo.prod(), w_ffn, anm_gain_tap, task_group=tg_anm)
        rt.finish_task_group(tg_anm)

        # ── Phase 7: SwiGLU gate/up + silu → down (decomp-B, ON-CHIP) ────────
        # silu_out is NOT drained to DDR — it flows gate/up → down via the
        # per-col inter_fifo (compute-tile → compute-tile stream, F5 lever).
        # gate/up and down run in ONE task_group (the inter_fifo couples them);
        # only the FINAL ffn_out partials drain. Host sums the cols partials.
        tg_ffn = rt.task_group()
        for s in range(n_shim_gu):
            rt.fill(agu_src_fifos[s].prod(), w_ffn, agu_src_taps[s], task_group=tg_ffn)
        # ffn_in already in ffi_l3l2_fifo — written by ANM worker directly.
        for s in range(n_shim_dp):
            rt.fill(adp_src_fifos[s].prod(), w_ffn, adp_src_taps[s], task_group=tg_ffn)
        # Drain the joined ffn_out partials to DDR
        for jg in range(n_join_dp):
            rt.drain(ffn_out_joined[jg].cons(), act, ffn_out_join_taps[jg], task_group=tg_ffn, wait=(jg == n_join_dp - 1))
        rt.finish_task_group(tg_ffn)

        # ATTN phase deferred — will land in next step with MemTile gather.

        # ── BO1/BO2 not touched at this checkpoint ───────────────────────────
        # DDR_PATCH ops for w_o/w_ffn re-emerge in stage 3 once O_proj and
        # SwiGLU workers consume them. XRT runtime accepts unused arg
        # slots, so for the spike we just leave them dormant.

        # ── Attention spike (7-col probe, cols 1-7 row 3) ────────────────────
        # Col 0 skipped due to shim-channel cap (3 S2MM > 2). Per-col fill
        # of 1 KV chunk + drain to BO4 attn_out. Last drain has wait=True.
        tg_attn = rt.task_group()
        for i in range(n_attn):
            rt.fill (kv_l3l2_fifos[i].prod(), kv, attn_kv_taps[i],
                     task_group=tg_attn)
        for i in range(n_attn):
            rt.drain(attn_drain_fifos[i].cons(), act, attn_drain_taps[i],
                     task_group=tg_attn,
                     wait=(i == n_attn - 1))
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
