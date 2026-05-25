# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused post-attention layer design (Phase B).

Chains in ONE xclbin and ONE xrt::execute():
  O_proj GEMV      (cols workers, K=embed, N=embed)
  ADD_attn         (worker 0 only: o_proj_out + inpL → inpFF)
  RMSNorm + MUL    (worker 0 only: inpFF * gain → ffn_input)
  SwiGLU gate+up   (cols workers, K=embed, N=hidden)
  SwiGLU down      (cols workers, K=hidden, N=embed)

Result: 1 dispatch per layer instead of 2 (O_proj + SwiGLU).
Saves: 1 × 350µs per layer × 16 layers = 5.6ms per token.

Worker 0 is the "leader" and handles the ADD/NORM/MUL phases.
Workers 1..cols-1 are "followers" that block on their SwiGLU input
FIFOs while the leader processes ADD+NORM+MUL.

The intermediate o_proj_out assembled from all workers is stored in
a shared DDR buffer (size=embed_dim bf16). This introduces a DDR
round-trip (~2µs) but avoids inter-tile communication for the global
RMSNorm reduction.

DDR argument layout (in order):
  0: w_o_proj       — O_proj INT4 weights: cols × tiles_o × packed_tile
  1: kqv_out        — attention output (O_proj input), embed_dim bf16
  2: inpL           — pre-attention residual, embed_dim bf16
  3: gain_weight    — pre-FFN norm gain, embed_dim bf16
  4: w_gate_up      — SwiGLU gate+up INT4 weights (interleaved), cols × 2 × tiles_gu × packed_tile
  5: w_down         — SwiGLU down INT4 weights, cols × tiles_d × packed_tile
  6: o_proj_scratch — DDR scratch for assembled O_proj output, embed_dim bf16
  7: ffn_out        — final layer output, embed_dim bf16
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
    """Build the fused post-attention IRON design."""

    # ── Shape validation ──────────────────────────────────────────────────────
    assert embed_dim % cols == 0,  "embed_dim must be divisible by cols"
    assert hidden_dim % cols == 0, "hidden_dim must be divisible by cols"
    assert embed_dim % group_size == 0
    assert hidden_dim % group_size == 0
    assert group_size % 32 == 0

    dev_ty = NPU1() if dev == "npu" else NPU2()

    dtype_packed = np.dtype[np.uint8]
    dtype_vec    = np.dtype[bfloat16]

    # ── Tile shapes ───────────────────────────────────────────────────────────
    # O_proj: K=embed, N=embed, m_input=1 row per tile
    m_input_o = 1
    rows_per_col_o = embed_dim // cols            # e.g. 256
    tiles_per_col_o = rows_per_col_o // m_input_o # e.g. 256
    groups_per_row_o = embed_dim // group_size
    packed_tile_o = m_input_o * embed_dim // 2 + m_input_o * groups_per_row_o * 2
    bytes_per_col_o = tiles_per_col_o * packed_tile_o

    # SwiGLU gate/up: K=embed, N=hidden, m_input=4 rows per tile
    m_input_gu = 4
    rows_per_col_gu = hidden_dim // cols           # e.g. 1024
    tiles_per_col_gu = rows_per_col_gu // m_input_gu  # e.g. 256
    m_output_gu = m_input_gu
    packed_tile_gu = m_input_gu * embed_dim // 2 + m_input_gu * groups_per_row_o * 2
    bytes_per_col_gu = tiles_per_col_gu * packed_tile_gu   # per weight (gate or up)

    # SwiGLU down: K=hidden, N=embed, m_input=1 row per tile
    m_input_d = 1
    rows_per_col_d = embed_dim // cols
    tiles_per_col_d = rows_per_col_d // m_input_d
    groups_per_row_d = hidden_dim // group_size
    packed_tile_d = m_input_d * hidden_dim // 2 + m_input_d * groups_per_row_d * 2
    bytes_per_col_d = tiles_per_col_d * packed_tile_d

    # ── L1 buffer types ───────────────────────────────────────────────────────
    L1_o_packed  = np.ndarray[(packed_tile_o,),  dtype_packed]  # O_proj weight tile
    L1_o_in      = np.ndarray[(embed_dim,),       dtype_vec]     # O_proj activation
    L1_o_out     = np.ndarray[(rows_per_col_o,),  dtype_vec]     # O_proj output slice

    L1_embed     = np.ndarray[(embed_dim,),       dtype_vec]     # ADD/NORM/MUL I/O
    L1_embed_b   = np.ndarray[(embed_dim,),       dtype_vec]     # second input for ADD

    L1_gu_packed = np.ndarray[(packed_tile_gu,),  dtype_packed]  # gate/up weight tile
    L1_gu_in     = np.ndarray[(embed_dim,),       dtype_vec]     # SwiGLU ffn_input
    L1_gu_out    = np.ndarray[(m_output_gu,),     dtype_vec]     # SwiGLU gate/up output

    L1_d_packed  = np.ndarray[(packed_tile_d,),   dtype_packed]  # down weight tile
    L1_d_in      = np.ndarray[(hidden_dim // cols,), dtype_vec]  # down input slice
    L1_d_out     = np.ndarray[(rows_per_col_d,),  dtype_vec]     # down output slice

    # ── L3 (DDR) buffer types ─────────────────────────────────────────────────
    total_o_bytes   = cols * bytes_per_col_o
    total_gu_bytes  = cols * 2 * bytes_per_col_gu   # gate + up interleaved per col
    total_d_bytes   = cols * bytes_per_col_d

    L3_w_o    = np.ndarray[(total_o_bytes,),            dtype_packed]
    L3_kqv    = np.ndarray[(embed_dim,),                dtype_vec]
    L3_inpL   = np.ndarray[(embed_dim,),                dtype_vec]
    L3_gain   = np.ndarray[(embed_dim,),                dtype_vec]
    L3_w_gu   = np.ndarray[(total_gu_bytes,),           dtype_packed]
    L3_w_d    = np.ndarray[(total_d_bytes,),            dtype_packed]
    L3_scratch= np.ndarray[(embed_dim,),                dtype_vec]  # o_proj_out assembled
    L3_out    = np.ndarray[(embed_dim,),                dtype_vec]

    # ── Kernel objects ────────────────────────────────────────────────────────
    kobj = f"post_attn_fused_{embed_dim}k_g{group_size}.o"

    o_proj_fn = Kernel(
        "fused_dequant_matvec_v2_bf16", kobj,
        [np.int32, np.int32, L1_o_packed, L1_o_in, L1_o_out],
    )
    add_fn = Kernel(
        "post_attn_add_bf16", kobj,
        [L1_embed, L1_embed_b, L1_embed, np.int32],
    )
    norm_fn = Kernel(
        "post_attn_rms_norm_bf16", kobj,
        [L1_embed, L1_embed, L1_embed, np.int32],
    )
    mul_fn = Kernel(
        "post_attn_gain_mul_bf16", kobj,
        [L1_embed, L1_embed, L1_embed, np.int32],
    )
    gate_up_fn = Kernel(
        "dual_fused_dequant_gemv_bf16", kobj,
        [np.int32, np.int32, L1_gu_packed, L1_gu_in, np.int32],
    )
    silu_mul_fn = Kernel(
        "dual_fused_dequant_gemv_silu_mul_bf16", kobj,
        [L1_gu_out, np.int32],
    )
    down_fn = Kernel(
        "fused_dequant_matvec_v2_bf16", kobj,
        [np.int32, np.int32, L1_d_packed, L1_d_in, L1_d_out],
    )

    # ── ObjectFIFOs ───────────────────────────────────────────────────────────
    # Per-column FIFOs for O_proj phase
    o_w_fifos  = [ObjectFifo(L1_o_packed, name=f"ow_{i}",  depth=2) for i in range(cols)]
    o_in_fifos = [ObjectFifo(L1_o_in,     name=f"oin_{i}", depth=1) for i in range(cols)]
    o_out_fifos= [ObjectFifo(L1_o_out,    name=f"oout_{i}",depth=2) for i in range(cols)]

    # Leader-only FIFOs for ADD+NORM+MUL phase (col 0 only)
    add_a_fifo  = ObjectFifo(L1_embed,   name="add_a",  depth=1)  # o_proj assembled
    add_b_fifo  = ObjectFifo(L1_embed_b, name="add_b",  depth=1)  # inpL
    gain_fifo   = ObjectFifo(L1_embed,   name="gain",   depth=1)  # gain weight
    ffn_in_fifo = ObjectFifo(L1_embed,   name="ffn_in", depth=1)  # ffn_input (output to DDR)

    # Per-column FIFOs for SwiGLU gate/up phase
    gu_w_fifos  = [ObjectFifo(L1_gu_packed,name=f"guw_{i}", depth=2) for i in range(cols)]
    gu_in_fifos = [ObjectFifo(L1_gu_in,    name=f"guin_{i}",depth=1) for i in range(cols)]
    gu_out_fifos= [ObjectFifo(L1_gu_out,   name=f"guout_{i}",depth=2)for i in range(cols)]

    # Per-column FIFOs for SwiGLU down phase
    d_w_fifos   = [ObjectFifo(L1_d_packed, name=f"dw_{i}",  depth=2) for i in range(cols)]
    d_in_fifos  = [ObjectFifo(L1_d_in,     name=f"din_{i}", depth=1) for i in range(cols)]
    d_out_fifos = [ObjectFifo(L1_d_out,    name=f"dout_{i}",depth=2) for i in range(cols)]

    embed_i32 = embed_dim

    # ── Core body for LEADER (col 0): all phases ──────────────────────────────
    def leader_body(
        ow, oin, oout,               # O_proj FIFOs
        aa, ab, gn, fi,              # ADD+NORM+MUL FIFOs
        guw, guin, guout,            # SwiGLU gate/up FIFOs
        dw, din, dout,               # SwiGLU down FIFOs
        of, af, nf, mf, guf, smf, df  # kernel functions
    ):
        for _ in range_(0xFFFFFFFF):
            # Phase 1: O_proj GEMV
            b = oin.acquire(1)
            for _ in range_(tiles_per_col_o):
                a = ow.acquire(1)
                c = oout.acquire(1)
                of(m_input_o, 0, a, b, c)
                ow.release(1)
                oout.release(1)
            oin.release(1)

            # Phase 2: ADD (o_proj_assembled + inpL → inpFF, in L1)
            a_o = aa.acquire(1)
            a_i = ab.acquire(1)
            # Use oout slot as scratch for inpFF
            c_ff = oout.acquire(1)   # reuse slot
            af(a_o, a_i, c_ff, embed_i32)
            aa.release(1)
            ab.release(1)

            # Phase 3: RMSNorm + MUL gain
            g_w  = gn.acquire(1)
            c_fi = fi.acquire(1)
            nf(c_ff, g_w, c_fi, embed_i32)
            mf(c_fi, g_w, c_fi, embed_i32)
            gn.release(1)
            oout.release(1)   # release c_ff

            # Phase 4: SwiGLU gate/up (tiles_per_col_gu iterations, m_output_gu rows)
            b_gu = guin.acquire(1)
            for _ in range_(tiles_per_col_gu):
                a_gu = guw.acquire(1)
                guf(m_input_gu, 0, a_gu, b_gu, 0)
                guf(m_input_gu, 0, a_gu, b_gu, 1)
                guw.release(1)
                c_gu = guout.acquire(1)
                smf(c_gu, m_output_gu)
                guout.release(1)
            guin.release(1)
            fi.release(1)

            # Phase 5: SwiGLU down
            b_d = din.acquire(1)
            for _ in range_(tiles_per_col_d):
                a_d = dw.acquire(1)
                c_d = dout.acquire(1)
                df(m_input_d, 0, a_d, b_d, c_d)
                dw.release(1)
                dout.release(1)
            din.release(1)

    # ── Core body for FOLLOWERS (cols 1..n-1): skip ADD+NORM+MUL ─────────────
    def follower_body(
        ow, oin, oout,               # O_proj FIFOs
        guw, guin, guout,            # SwiGLU gate/up FIFOs
        dw, din, dout,               # SwiGLU down FIFOs
        of, guf, smf, df
    ):
        for _ in range_(0xFFFFFFFF):
            # Phase 1: O_proj GEMV
            b = oin.acquire(1)
            for _ in range_(tiles_per_col_o):
                a = ow.acquire(1)
                c = oout.acquire(1)
                of(m_input_o, 0, a, b, c)
                ow.release(1)
                oout.release(1)
            oin.release(1)
            # [Followers block here until guin is filled by ctrlcode]

            # Phase 4: SwiGLU gate/up
            b_gu = guin.acquire(1)
            for _ in range_(tiles_per_col_gu):
                a_gu = guw.acquire(1)
                guf(m_input_gu, 0, a_gu, b_gu, 0)
                guf(m_input_gu, 0, a_gu, b_gu, 1)
                guw.release(1)
                c_gu = guout.acquire(1)
                smf(c_gu, m_output_gu)
                guout.release(1)
            guin.release(1)

            # Phase 5: SwiGLU down
            b_d = din.acquire(1)
            for _ in range_(tiles_per_col_d):
                a_d = dw.acquire(1)
                c_d = dout.acquire(1)
                df(m_input_d, 0, a_d, b_d, c_d)
                dw.release(1)
                dout.release(1)
            din.release(1)

    # ── Workers ───────────────────────────────────────────────────────────────
    leader_worker = Worker(
        leader_body, [
            o_w_fifos[0].cons(), o_in_fifos[0].cons(), o_out_fifos[0].prod(),
            add_a_fifo.cons(), add_b_fifo.cons(), gain_fifo.cons(), ffn_in_fifo.prod(),
            gu_w_fifos[0].cons(), gu_in_fifos[0].cons(), gu_out_fifos[0].prod(),
            d_w_fifos[0].cons(), d_in_fifos[0].cons(), d_out_fifos[0].prod(),
            o_proj_fn, add_fn, norm_fn, mul_fn, gate_up_fn, silu_mul_fn, down_fn,
        ],
    )

    follower_workers = [
        Worker(
            follower_body, [
                o_w_fifos[i].cons(), o_in_fifos[i].cons(), o_out_fifos[i].prod(),
                gu_w_fifos[i].cons(), gu_in_fifos[i].cons(), gu_out_fifos[i].prod(),
                d_w_fifos[i].cons(), d_in_fifos[i].cons(), d_out_fifos[i].prod(),
                o_proj_fn, gate_up_fn, silu_mul_fn, down_fn,
            ],
        )
        for i in range(1, cols)
    ]

    # ── TensorAccessPatterns ──────────────────────────────────────────────────
    # O_proj weights: cols × bytes_per_col_o, one slice per column
    o_w_taps = [
        TensorAccessPattern(
            tensor_dims=(1, total_o_bytes),
            offset=col * bytes_per_col_o,
            sizes=[1, 1, tiles_per_col_o, packed_tile_o],
            strides=[0, 0, packed_tile_o, 1],
        ) for col in range(cols)
    ]
    # O_proj activation: same embed_dim vector for all cols
    o_in_tap = TensorAccessPattern(
        tensor_dims=(1, embed_dim),
        offset=0, sizes=[1, 1, 1, embed_dim], strides=[0, 0, 0, 1],
    )
    # O_proj output: each col drains rows_per_col_o elements to its slice
    o_out_taps = [
        TensorAccessPattern(
            tensor_dims=(1, embed_dim),
            offset=col * rows_per_col_o,
            sizes=[1, 1, tiles_per_col_o, rows_per_col_o // tiles_per_col_o],
            strides=[0, 0, rows_per_col_o // tiles_per_col_o, 1],
        ) for col in range(cols)
    ]
    # ADD inputs (leader only): full embed_dim
    add_a_tap = TensorAccessPattern(
        tensor_dims=(1, embed_dim), offset=0,
        sizes=[1, 1, 1, embed_dim], strides=[0, 0, 0, 1],
    )
    add_b_tap = add_a_tap  # same shape for inpL
    gain_tap  = add_a_tap  # same shape for gain
    ffn_out_tap = add_a_tap  # ffn_input drained to scratch

    # SwiGLU gate+up weights: interleaved [gate_col0, up_col0, gate_col1, up_col1, ...]
    gu_w_taps = [
        TensorAccessPattern(
            tensor_dims=(1, total_gu_bytes),
            offset=col * 2 * bytes_per_col_gu,
            sizes=[1, 1, tiles_per_col_gu * 2, packed_tile_gu],
            strides=[0, 0, packed_tile_gu, 1],
        ) for col in range(cols)
    ]
    # SwiGLU gate/up input: full embed_dim for all cols
    gu_in_tap = o_in_tap
    # SwiGLU gate/up output per col
    gu_out_taps = [
        TensorAccessPattern(
            tensor_dims=(1, hidden_dim),
            offset=col * rows_per_col_gu,
            sizes=[1, 1, tiles_per_col_gu, m_output_gu],
            strides=[0, 0, m_output_gu, 1],
        ) for col in range(cols)
    ]

    # SwiGLU down weights
    d_w_taps = [
        TensorAccessPattern(
            tensor_dims=(1, total_d_bytes),
            offset=col * bytes_per_col_d,
            sizes=[1, 1, tiles_per_col_d, packed_tile_d],
            strides=[0, 0, packed_tile_d, 1],
        ) for col in range(cols)
    ]
    # SwiGLU down input: each col reads its hidden_dim//cols slice
    d_in_taps = [
        TensorAccessPattern(
            tensor_dims=(1, hidden_dim),
            offset=col * (hidden_dim // cols),
            sizes=[1, 1, 1, hidden_dim // cols],
            strides=[0, 0, 0, 1],
        ) for col in range(cols)
    ]
    # SwiGLU down output
    d_out_taps = [
        TensorAccessPattern(
            tensor_dims=(1, embed_dim),
            offset=col * rows_per_col_d,
            sizes=[1, 1, tiles_per_col_d, rows_per_col_d // tiles_per_col_d],
            strides=[0, 0, rows_per_col_d // tiles_per_col_d, 1],
        ) for col in range(cols)
    ]

    # ── Runtime Sequence ──────────────────────────────────────────────────────
    rt = Runtime()
    with rt.sequence(
        L3_w_o, L3_kqv, L3_inpL, L3_gain,
        L3_w_gu, L3_w_d,
        L3_scratch,  # o_proj_out assembled here
        L3_out,
    ) as (W_o, kqv, inpL, gain, W_gu, W_d, scratch, out):

        rt.start(leader_worker, *follower_workers)

        # ── Phase 1: O_proj for all cols ──────────────────────────────────────
        tg_o = rt.task_group()
        for i in range(cols):
            rt.fill(o_w_fifos[i].prod(), W_o, o_w_taps[i], task_group=tg_o)
            rt.fill(o_in_fifos[i].prod(), kqv, o_in_tap,   task_group=tg_o)
        for i in range(cols):
            rt.drain(o_out_fifos[i].cons(), scratch, o_out_taps[i],
                     task_group=tg_o, wait=True)
        rt.finish_task_group(tg_o)

        # ── Phase 2: ADD+NORM+MUL on leader (col 0 only) ─────────────────────
        tg_anm = rt.task_group()
        rt.fill(add_a_fifo.prod(), scratch, add_a_tap, task_group=tg_anm)  # assembled o_proj
        rt.fill(add_b_fifo.prod(), inpL,    add_b_tap, task_group=tg_anm)  # residual
        rt.fill(gain_fifo.prod(),  gain,    gain_tap,  task_group=tg_anm)  # gain weight
        rt.drain(ffn_in_fifo.cons(), scratch, ffn_out_tap,
                 task_group=tg_anm, wait=True)
        rt.finish_task_group(tg_anm)

        # ── Phase 3+4: SwiGLU gate+up then down for all cols ─────────────────
        tg_gu = rt.task_group()
        for i in range(cols):
            rt.fill(gu_w_fifos[i].prod(), W_gu,    gu_w_taps[i], task_group=tg_gu)
            rt.fill(gu_in_fifos[i].prod(), scratch, gu_in_tap,   task_group=tg_gu)
        for i in range(cols):
            rt.drain(gu_out_fifos[i].cons(), out, gu_out_taps[i],
                     task_group=tg_gu, wait=True)
        rt.finish_task_group(tg_gu)

        tg_d = rt.task_group()
        for i in range(cols):
            rt.fill(d_w_fifos[i].prod(), W_d, d_w_taps[i], task_group=tg_d)
            rt.fill(d_in_fifos[i].prod(), out, d_in_taps[i], task_group=tg_d)
        for i in range(cols):
            rt.drain(d_out_fifos[i].cons(), out, d_out_taps[i],
                     task_group=tg_d, wait=True)
        rt.finish_task_group(tg_d)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Post-attention fused layer design")
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
