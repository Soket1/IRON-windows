# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused post-attention layer design v2 (corrected FIFO layout).

Four separate worker groups, each with exactly 3 FIFOs (≤ DMA channel limit):
  1. o_proj_workers   (8): O_proj GEMV,  K=embed→N=embed
  2. anm_worker       (1): ADD+RMSNorm+MUL (leader tile, col 0)
  3. gate_up_workers  (8): SwiGLU gate+up+silu_mul, K=embed→N=hidden
  4. down_workers     (8): SwiGLU down, K=hidden→N=embed

DDR layout (in order as rt.sequence args):
  0: w_o_proj    — INT4 O_proj weights:        cols × tiles_o × packed_tile_o
  1: kqv_out     — attention output (embed bf16)
  2: inpL        — residual (embed bf16)
  3: gain_weight — pre-FFN gain (embed bf16)
  4: w_gate_up   — INT4 gate+up weights:  cols × 2 × tiles_gu × packed_tile_gu
  5: w_down      — INT4 down weights:     cols × tiles_d × packed_tile_d
  6: scratch     — scratch buf (embed bf16, reused between phases)
  7: silu_buf    — silu_out buf (hidden bf16)
  8: ffn_out     — final output (embed bf16)

Data flow:
  O_proj(kqv_out, w_o) → scratch
  ANM(scratch, inpL, gain) → scratch          [scratch reused for ffn_input]
  SwiGLU_gate_up(scratch, w_gu) → silu_buf
  SwiGLU_down(silu_buf, w_d) → ffn_out
"""

import numpy as np
from pathlib import Path
from ml_dtypes import bfloat16
import argparse

import aie.dialects.index as index
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.scf import _for as range_
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2


def my_post_attn_fused(dev, cols, embed_dim, hidden_dim, group_size=32):
    """Build the fused post-attention layer IRON design."""

    assert embed_dim % cols == 0
    assert hidden_dim % cols == 0
    assert embed_dim  % group_size == 0
    assert hidden_dim % group_size == 0

    dev_ty = NPU1() if dev == "npu" else NPU2()
    dtype_packed = np.dtype[np.uint8]
    dtype_vec    = np.dtype[bfloat16]

    # ── Tiling parameters ─────────────────────────────────────────────────────
    # m_input_* MUST be chosen so tiles_per_col_* <= 1023 (NPU shim dma_bd
    # outer-dim limit on AIE2P). For cols=2, embed=2048, hidden=8192:
    #   rows_per_col_o  = 1024 -> m_input_o  >= 2 (tiles = 512)
    #   rows_per_col_d  = 1024 -> m_input_d  >= 2 (tiles = 512)
    #   rows_per_col_gu = 4096 -> m_input_gu >= 5 (use 16 = VEC so silu_mul
    #                             vectorizes cleanly with chunks=1).
    # O_proj: K=embed, N=embed
    m_input_o   = 2
    rows_per_col_o = embed_dim // cols
    tiles_per_col_o = rows_per_col_o // m_input_o
    groups_o    = embed_dim // group_size
    packed_tile_o = m_input_o * embed_dim // 2 + m_input_o * groups_o * 2
    bytes_col_o = tiles_per_col_o * packed_tile_o

    # SwiGLU gate/up: K=embed, N=hidden
    m_input_gu  = 8                        # packed_tile = 9216 B fits one L1 bank
    rows_per_col_gu = hidden_dim // cols
    tiles_per_col_gu = rows_per_col_gu // m_input_gu
    m_output_gu = m_input_gu
    packed_tile_gu = m_input_gu * embed_dim // 2 + m_input_gu * groups_o * 2
    bytes_col_gu   = tiles_per_col_gu * packed_tile_gu  # per gate OR up weight

    # SwiGLU down: K=hidden, N=embed
    m_input_d   = 2
    rows_per_col_d = embed_dim // cols
    tiles_per_col_d = rows_per_col_d // m_input_d
    groups_d    = hidden_dim // group_size
    packed_tile_d = m_input_d * hidden_dim // 2 + m_input_d * groups_d * 2
    bytes_col_d = tiles_per_col_d * packed_tile_d

    # Divisibility sanity (must be exact to avoid edge-case DMA)
    assert rows_per_col_o  % m_input_o  == 0, "rows_per_col_o must be divisible by m_input_o"
    assert rows_per_col_gu % m_input_gu == 0, "rows_per_col_gu must be divisible by m_input_gu"
    assert rows_per_col_d  % m_input_d  == 0, "rows_per_col_d must be divisible by m_input_d"
    assert tiles_per_col_o  <= 1023, f"tiles_per_col_o={tiles_per_col_o} exceeds DMA BD dim limit"
    assert tiles_per_col_gu <= 1023, f"tiles_per_col_gu={tiles_per_col_gu} exceeds DMA BD dim limit"
    assert tiles_per_col_d  <= 1023, f"tiles_per_col_d={tiles_per_col_d} exceeds DMA BD dim limit"

    # ── L1 buffer types (used by individual tile kernels) ─────────────────────
    # Output L1 types are per-tile slices (m_input_*), NOT full rows_per_col:
    # the kernel writes c_out[0..m_input-1] each iteration, drain advances
    # the DDR offset by m_input per tile.
    L1_Ao_ty  = np.ndarray[(packed_tile_o,),   dtype_packed]   # O_proj weight tile
    L1_Bo_ty  = np.ndarray[(embed_dim,),       dtype_vec]      # activation for O_proj
    L1_Co_ty  = np.ndarray[(m_input_o,),       dtype_vec]      # O_proj output slice

    L1_anm_ty = np.ndarray[(embed_dim,),       dtype_vec]      # ADD/NORM/MUL I/O

    L1_Agu_ty = np.ndarray[(packed_tile_gu,),  dtype_packed]   # gate/up weight tile
    L1_Bgu_ty = np.ndarray[(embed_dim,),       dtype_vec]      # ffn_input for SwiGLU
    L1_Cgu_ty = np.ndarray[(m_output_gu,),     dtype_vec]      # silu_out slice

    L1_Ad_ty  = np.ndarray[(packed_tile_d,),   dtype_packed]   # down weight tile
    L1_Bd_ty  = np.ndarray[(hidden_dim,),      dtype_vec]      # full silu_out (broadcast)
    L1_Cd_ty  = np.ndarray[(m_input_d,),       dtype_vec]      # ffn_out slice

    # ── L3 (DDR) buffer types ─────────────────────────────────────────────────
    total_o_bytes  = cols * bytes_col_o
    total_gu_bytes = cols * 2 * bytes_col_gu   # gate + up per col, interleaved
    total_d_bytes  = cols * bytes_col_d

    # Bundled L3 buffers (5 sequence args, fits XRT MLIR_AIE 8-group_id cap):
    #   input_bundle = [kqv (embed) | inpL (embed) | gain (embed)]
    #   io_bundle    = [scratch (embed) | silu_buf (hidden) | ffn_out (embed)]
    # Each TAP reads/writes the appropriate sub-range via `offset`; kernel-side
    # L1 ObjectFifos still see embed_dim or hidden_dim slices, unchanged.
    input_bundle_elems = 3 * embed_dim
    io_bundle_elems    = 2 * embed_dim + hidden_dim
    scratch_offset_in_io  = 0
    silu_offset_in_io     = embed_dim
    ffn_out_offset_in_io  = embed_dim + hidden_dim
    kqv_offset_in_input   = 0
    inpL_offset_in_input  = embed_dim
    gain_offset_in_input  = 2 * embed_dim

    L3_w_o     = np.ndarray[(total_o_bytes,),       dtype_packed]
    L3_w_gu    = np.ndarray[(total_gu_bytes,),      dtype_packed]
    L3_w_d     = np.ndarray[(total_d_bytes,),       dtype_packed]
    L3_inputs  = np.ndarray[(input_bundle_elems,),  dtype_vec]
    L3_io      = np.ndarray[(io_bundle_elems,),     dtype_vec]

    # ── L1 tile-to-tile intermediate type (split SwiGLU workers) ──────────────
    # gate_worker -> silu_mul_worker  and  up_worker -> silu_mul_worker
    # pass one m_input_gu-sized slice per tile iteration. No statics needed.
    L1_inter_ty = np.ndarray[(m_input_gu,), dtype_vec]

    # ── Kernel objects ─────────────────────────────────────────────────────────
    kobj = f"post_attn_fused_{embed_dim}k_g{group_size}.o"

    o_proj_fn   = Kernel("fused_dequant_matvec_v2_bf16", kobj,
                         [np.int32, np.int32, L1_Ao_ty, L1_Bo_ty, L1_Co_ty])
    add_fn      = Kernel("post_attn_add_bf16",      kobj,
                         [L1_anm_ty, L1_anm_ty, L1_anm_ty, np.int32])
    norm_mul_fn = Kernel("post_attn_rms_norm_bf16", kobj,
                         [L1_anm_ty, L1_anm_ty, L1_anm_ty, np.int32])
    # Split-worker SwiGLU: gate_worker and up_worker each call swiglu_gemv_bf16
    # (no phase, no statics, output to a passed L1 buffer); silu_mul_worker
    # then takes (gate, up) -> out.
    swiglu_gemv_fn = Kernel("swiglu_gemv_bf16", kobj,
                            [np.int32, L1_Agu_ty, L1_Bgu_ty, L1_inter_ty])
    silu_mul_fn    = Kernel("silu_mul_v_bf16", kobj,
                            [L1_inter_ty, L1_inter_ty, L1_Cgu_ty, np.int32])
    down_fn     = Kernel("fused_dequant_matvec_down_bf16", kobj,
                         [np.int32, np.int32, L1_Ad_ty, L1_Bd_ty, L1_Cd_ty])

    embed_i32  = embed_dim
    hidden_i32 = hidden_dim

    # ── ObjectFIFOs — exactly 3 per main worker ────────────────────────────────
    # O_proj phase FIFOs (per column)
    Ao_fifos  = [ObjectFifo(L1_Ao_ty,  name=f"Ao_{i}",  depth=2) for i in range(cols)]
    Bo_fifos  = [ObjectFifo(L1_Bo_ty,  name=f"Bo_{i}",  depth=1) for i in range(cols)]
    Co_fifos  = [ObjectFifo(L1_Co_ty,  name=f"Co_{i}",  depth=2) for i in range(cols)]

    # ANM phase FIFOs (leader, 1 worker).
    # Single input FIFO of depth=3 (fits AIE2P's 2-input-DMA tile limit):
    # runtime fills 3 entries in order [o_proj_out, inpL, gain_weight];
    # worker acquires(3) and indexes sub[0], sub[1], sub[2].
    anm_in_fifo = ObjectFifo(L1_anm_ty, name="anm_in", depth=3)
    anm_o_fifo  = ObjectFifo(L1_anm_ty, name="anm_o",  depth=1)

    # SwiGLU gate/up phase FIFOs (per column).
    # Split-worker layout: gate_worker and up_worker each have their own
    # weight FIFO (Agu_gate, Agu_up) so the shim BD outer dim per fill
    # is tiles_per_col_gu = 512 (instead of 2*512 = 1024 in the monolithic
    # design). Bgu is shared via broadcast — one ObjectFifo with two
    # consumer endpoints.
    Agu_gate_fifos = [ObjectFifo(L1_Agu_ty, name=f"Agu_gate_{i}", depth=1) for i in range(cols)]
    Agu_up_fifos   = [ObjectFifo(L1_Agu_ty, name=f"Agu_up_{i}",   depth=1) for i in range(cols)]
    Bgu_fifos      = [ObjectFifo(L1_Bgu_ty, name=f"Bgu_{i}",      depth=1) for i in range(cols)]
    # Tile-to-tile FIFOs: gate_worker -> silu_mul_worker, up_worker -> silu_mul_worker
    gate_out_fifos = [ObjectFifo(L1_inter_ty, name=f"gate_out_{i}", depth=2) for i in range(cols)]
    up_out_fifos   = [ObjectFifo(L1_inter_ty, name=f"up_out_{i}",   depth=2) for i in range(cols)]
    Cgu_fifos      = [ObjectFifo(L1_Cgu_ty, name=f"Cgu_{i}", depth=2) for i in range(cols)]

    # SwiGLU down phase FIFOs (per column).
    # Ad depth=1 for the same L1-budget reason (packed_tile_d ~9 KB).
    Ad_fifos  = [ObjectFifo(L1_Ad_ty,  name=f"Ad_{i}",  depth=1) for i in range(cols)]
    Bd_fifos  = [ObjectFifo(L1_Bd_ty,  name=f"Bd_{i}",  depth=1) for i in range(cols)]
    Cd_fifos  = [ObjectFifo(L1_Cd_ty,  name=f"Cd_{i}",  depth=2) for i in range(cols)]

    # ── Core bodies ────────────────────────────────────────────────────────────
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

    def anm_body(in_fifo, o_out, af, nf):
        for _ in range_(0xFFFFFFFF):
            # Acquire 3 input slots at once: [o_proj_out, inpL, gain_weight].
            sub = in_fifo.acquire(3)
            s = sub[0]
            l = sub[1]
            g = sub[2]
            o = o_out.acquire(1)
            # ADD: o = s + l
            af(s, l, o, embed_i32)
            # RMSNorm+gain MUL: o = rms_norm(o) * g  (in-place via 3rd arg)
            nf(o, g, o, embed_i32)
            in_fifo.release(3)
            o_out.release(1)

    # Gate worker: dequant-GEMV of gate weights, streams m_input_gu bf16 per
    # tile to silu_mul via a tile-to-tile FIFO. Bgu is shared (broadcast).
    def gate_body(Agu_g, Bgu, gate_out, gemv_fn):
        for _ in range_(0xFFFFFFFF):
            b = Bgu.acquire(1)
            for _ in range_(tiles_per_col_gu):
                a = Agu_g.acquire(1)
                g = gate_out.acquire(1)
                gemv_fn(m_input_gu, a, b, g)
                Agu_g.release(1)
                gate_out.release(1)
            Bgu.release(1)

    def up_body(Agu_u, Bgu, up_out, gemv_fn):
        for _ in range_(0xFFFFFFFF):
            b = Bgu.acquire(1)
            for _ in range_(tiles_per_col_gu):
                a = Agu_u.acquire(1)
                u = up_out.acquire(1)
                gemv_fn(m_input_gu, a, b, u)
                Agu_u.release(1)
                up_out.release(1)
            Bgu.release(1)

    # Silu-mul worker: receives gate+up tile slices from upstream workers,
    # writes silu(gate)*up to Cgu output FIFO.
    def silu_mul_body(gate_in, up_in, Cgu, sm_fn):
        for _ in range_(0xFFFFFFFF):
            for _ in range_(tiles_per_col_gu):
                g = gate_in.acquire(1)
                u = up_in.acquire(1)
                c = Cgu.acquire(1)
                sm_fn(g, u, c, m_output_gu)
                gate_in.release(1)
                up_in.release(1)
                Cgu.release(1)

    def down_body(Ad, Bd, Cd, fn):
        for _ in range_(0xFFFFFFFF):
            b = Bd.acquire(1)
            for _ in range_(tiles_per_col_d):
                a = Ad.acquire(1)
                c = Cd.acquire(1)
                fn(m_input_d, 0, a, b, c)
                Ad.release(1)
                Cd.release(1)
            Bd.release(1)

    # ── Workers ────────────────────────────────────────────────────────────────
    o_proj_workers = [
        Worker(o_proj_body,
               [Ao_fifos[i].cons(), Bo_fifos[i].cons(), Co_fifos[i].prod(), o_proj_fn])
        for i in range(cols)
    ]

    anm_worker = Worker(anm_body,
                        [anm_in_fifo.cons(), anm_o_fifo.prod(),
                         add_fn, norm_mul_fn])

    # SwiGLU split workers (3 per col: gate, up, silu_mul).
    # Bgu_fifos[i].cons() is called TWICE — once for gate_worker, once for
    # up_worker — which IRON ObjectFifo treats as a broadcast: a single shim
    # S2MM channel feeds both consumer tiles via on-chip routing.
    gate_workers = [
        Worker(gate_body,
               [Agu_gate_fifos[i].cons(), Bgu_fifos[i].cons(),
                gate_out_fifos[i].prod(), swiglu_gemv_fn])
        for i in range(cols)
    ]
    up_workers = [
        Worker(up_body,
               [Agu_up_fifos[i].cons(), Bgu_fifos[i].cons(),
                up_out_fifos[i].prod(), swiglu_gemv_fn])
        for i in range(cols)
    ]
    silu_mul_workers = [
        Worker(silu_mul_body,
               [gate_out_fifos[i].cons(), up_out_fifos[i].cons(),
                Cgu_fifos[i].prod(), silu_mul_fn])
        for i in range(cols)
    ]

    down_workers = [
        Worker(down_body,
               [Ad_fifos[i].cons(), Bd_fifos[i].cons(), Cd_fifos[i].prod(), down_fn])
        for i in range(cols)
    ]

    # ── TensorAccessPatterns ───────────────────────────────────────────────────
    # All A_*_taps use a single-linear-chunk layout sizes=[1,1,1,N] so that
    # the MLIR-AIE BD generator emits a fully-linearized descriptor (matching
    # the production v2 GEMV layout). Multi-dim TAPs with inner < packed_tile
    # trigger a partial-merge bug in MLIR-AIE that fails BD validation.
    Ao_taps = [
        TensorAccessPattern(tensor_dims=(1, total_o_bytes),
            offset=i * bytes_col_o,
            sizes=[1, 1, 1, bytes_col_o],
            strides=[0, 0, 0, 1])
        for i in range(cols)
    ]
    # Per-buffer TAPs into the two bundled DDR arrays. Each TAP carries
    # tensor_dims = full bundle size, offset = where the buffer lives.
    # Layout reminder:
    #   input_bundle: [kqv (E) | inpL (E) | gain (E)]
    #   io_bundle:    [scratch (E) | silu_buf (H) | ffn_out (E)]
    kqv_tap = TensorAccessPattern(tensor_dims=(1, input_bundle_elems),
                offset=kqv_offset_in_input,
                sizes=[1, 1, 1, embed_dim],  strides=[0, 0, 0, 1])
    inpL_tap = TensorAccessPattern(tensor_dims=(1, input_bundle_elems),
                offset=inpL_offset_in_input,
                sizes=[1, 1, 1, embed_dim],  strides=[0, 0, 0, 1])
    gain_tap = TensorAccessPattern(tensor_dims=(1, input_bundle_elems),
                offset=gain_offset_in_input,
                sizes=[1, 1, 1, embed_dim],  strides=[0, 0, 0, 1])
    scratch_in_tap = TensorAccessPattern(tensor_dims=(1, io_bundle_elems),
                offset=scratch_offset_in_io,
                sizes=[1, 1, 1, embed_dim],  strides=[0, 0, 0, 1])
    scratch_out_tap = scratch_in_tap   # same offset/size, drain side
    silu_full_tap = TensorAccessPattern(tensor_dims=(1, io_bundle_elems),
                offset=silu_offset_in_io,
                sizes=[1, 1, 1, hidden_dim], strides=[0, 0, 0, 1])
    # O_proj output: each col drains its rows_per_col_o slice into the
    # scratch region of io_bundle (offset 0).
    Co_taps = [
        TensorAccessPattern(tensor_dims=(1, io_bundle_elems),
            offset=scratch_offset_in_io + i * rows_per_col_o,
            sizes=[1, 1, tiles_per_col_o, rows_per_col_o // tiles_per_col_o],
            strides=[0, 0, rows_per_col_o // tiles_per_col_o, 1])
        for i in range(cols)
    ]

    # SwiGLU gate+up weights — linear DMA per fill (matches production v2 layout).
    # Each fill reads bytes_col_gu contiguous bytes from DDR; the compiler
    # collapses sizes=[1,1,1,N] directly without partial-merge, avoiding the
    # MLIR-AIE BD validator's per-dim 1023 limit (the validator rejects when
    # the compiler does a partial merge to inner > 1023).
    Agu_gate_taps = [
        TensorAccessPattern(tensor_dims=(1, total_gu_bytes),
            offset=i * 2 * bytes_col_gu,
            sizes=[1, 1, 1, bytes_col_gu],
            strides=[0, 0, 0, 1])
        for i in range(cols)
    ]
    Agu_up_taps = [
        TensorAccessPattern(tensor_dims=(1, total_gu_bytes),
            offset=i * 2 * bytes_col_gu + bytes_col_gu,
            sizes=[1, 1, 1, bytes_col_gu],
            strides=[0, 0, 0, 1])
        for i in range(cols)
    ]
    # SwiGLU output: silu_out slices, drain into io_bundle silu region.
    Cgu_taps = [
        TensorAccessPattern(tensor_dims=(1, io_bundle_elems),
            offset=silu_offset_in_io + i * rows_per_col_gu,
            sizes=[1, 1, tiles_per_col_gu, rows_per_col_gu // tiles_per_col_gu],
            strides=[0, 0, rows_per_col_gu // tiles_per_col_gu, 1])
        for i in range(cols)
    ]

    # SwiGLU down weights — same linear-DMA layout as Agu.
    Ad_taps = [
        TensorAccessPattern(tensor_dims=(1, total_d_bytes),
            offset=i * bytes_col_d,
            sizes=[1, 1, 1, bytes_col_d],
            strides=[0, 0, 0, 1])
        for i in range(cols)
    ]
    # Down input: FULL hidden_dim broadcast to each tile, reads from
    # io_bundle silu region.
    Bd_tap = TensorAccessPattern(tensor_dims=(1, io_bundle_elems),
                offset=silu_offset_in_io,
                sizes=[1, 1, 1, hidden_dim], strides=[0, 0, 0, 1])
    # Down output: embed_dim slices into io_bundle ffn_out region.
    Cd_taps = [
        TensorAccessPattern(tensor_dims=(1, io_bundle_elems),
            offset=ffn_out_offset_in_io + i * rows_per_col_d,
            sizes=[1, 1, tiles_per_col_d, rows_per_col_d // tiles_per_col_d],
            strides=[0, 0, rows_per_col_d // tiles_per_col_d, 1])
        for i in range(cols)
    ]

    # ── Runtime sequence ───────────────────────────────────────────────────────
    # Five sequence args (fits XRT MLIR_AIE 8-group_id kernel cap = 3 internal
    # + 5 buffer args). The kqv/inpL/gain trio is read from input_bundle at
    # contiguous offsets; the scratch/silu_buf/ffn_out trio reads/writes the
    # io_bundle at contiguous offsets. Host pre-packs input_bundle and reads
    # ffn_out + scratch (inpFF) from io_bundle after the dispatch.
    rt = Runtime()
    with rt.sequence(
        L3_w_o, L3_w_gu, L3_w_d, L3_inputs, L3_io,
    ) as (w_o, w_gu, w_d, inputs, io):

        rt.start(*o_proj_workers, anm_worker,
                 *gate_workers, *up_workers, *silu_mul_workers,
                 *down_workers)

        # ── Phase 1: O_proj ───────────────────────────────────────────────────
        # Reads kqv from input_bundle[0..E], writes scratch to io_bundle[0..E].
        tg_o = rt.task_group()
        for i in range(cols):
            rt.fill(Ao_fifos[i].prod(), w_o,    Ao_taps[i], task_group=tg_o)
            rt.fill(Bo_fifos[i].prod(), inputs, kqv_tap,    task_group=tg_o)
        for i in range(cols):
            rt.drain(Co_fifos[i].cons(), io, Co_taps[i],
                     task_group=tg_o, wait=True)
        rt.finish_task_group(tg_o)
        # io[0..E] now holds assembled o_proj_out

        # ── Phase 2: ADD + RMSNorm + MUL (ANM worker, 1 tile) ─────────────────
        # Three sequential fills into anm_in_fifo (depth=3): o_proj_out from
        # io, inpL/gain from input_bundle. Worker indexes sub[0..2]. ANM output
        # drains back into io[0..E] (overwriting o_proj_out with ffn_input).
        tg_anm = rt.task_group()
        rt.fill(anm_in_fifo.prod(), io,     scratch_in_tap, task_group=tg_anm)
        rt.fill(anm_in_fifo.prod(), inputs, inpL_tap,       task_group=tg_anm)
        rt.fill(anm_in_fifo.prod(), inputs, gain_tap,       task_group=tg_anm)
        rt.drain(anm_o_fifo.cons(), io, scratch_out_tap,
                 task_group=tg_anm, wait=True)
        rt.finish_task_group(tg_anm)
        # io[0..E] now holds ffn_input

        # ── Phase 3: SwiGLU gate+up+silu_mul (split workers) ──────────────────
        tg_gu = rt.task_group()
        for i in range(cols):
            rt.fill(Agu_gate_fifos[i].prod(), w_gu, Agu_gate_taps[i], task_group=tg_gu)
            rt.fill(Agu_up_fifos[i].prod(),   w_gu, Agu_up_taps[i],   task_group=tg_gu)
            rt.fill(Bgu_fifos[i].prod(),      io,   scratch_in_tap,   task_group=tg_gu)
        for i in range(cols):
            rt.drain(Cgu_fifos[i].cons(), io, Cgu_taps[i],
                     task_group=tg_gu, wait=True)
        rt.finish_task_group(tg_gu)
        # io[E..E+H] now holds silu_out (silu region of bundle)

        # ── Phase 4: SwiGLU down ──────────────────────────────────────────────
        tg_d = rt.task_group()
        for i in range(cols):
            rt.fill(Ad_fifos[i].prod(), w_d, Ad_taps[i], task_group=tg_d)
            rt.fill(Bd_fifos[i].prod(), io,  Bd_tap,     task_group=tg_d)
        for i in range(cols):
            rt.drain(Cd_fifos[i].cons(), io, Cd_taps[i],
                     task_group=tg_d, wait=True)
        rt.finish_task_group(tg_d)
        # io[E+H..2E+H] now holds ffn_out; io[0..E] still holds inpFF
        # (untouched by SwiGLU input read); host reads both for the post-FFN
        # residual ADD.

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Post-attention fused design v2")
    p.add_argument("--dev", default="npu2", choices=["npu", "npu2"])
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--embed-dim", type=int, default=2048)
    p.add_argument("--hidden-dim", type=int, default=8192)
    p.add_argument("--group-size", type=int, default=32)
    p.add_argument("-o", "--output-file-path", type=str)
    a = p.parse_args()
    module = my_post_attn_fused(a.dev, a.cols, a.embed_dim, a.hidden_dim, a.group_size)
    if a.output_file_path:
        with open(a.output_file_path, "w") as f:
            f.write(str(module))
