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
    m_input_qkv = 1
    packed_q   = m_input_qkv * e // 2 + m_input_qkv * groups_e * 2
    total_q    = cols * (e // cols) * packed_q
    total_kv_one = cols * (kv_e // cols) * packed_q
    bo0_bytes  = total_q + 2 * total_kv_one

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
    bo2_bytes  = total_gu + total_d

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

    # ── L3 (DDR / runtime sequence) types ────────────────────────────────────
    L3_w_qkv  = np.ndarray[(bo0_bytes // 2,), dtype_vec]
    L3_w_o    = np.ndarray[(bo1_bytes // 2,), dtype_vec]
    L3_w_ffn  = np.ndarray[(bo2_bytes // 2,), dtype_vec]
    L3_kv     = np.ndarray[(bo3_bytes // 2,), dtype_vec]
    L3_act    = np.ndarray[(bo4_elems,),       dtype_vec]

    # ── Skeleton compute: 1 leader tile, 1 noop kernel per BO ────────────────
    # We slice off a tiny prefix (chunk_elems bf16 elements = 2*chunk_elems
    # bytes) of each BO and run it through a 1-tile passthrough Worker.
    # The passthrough Worker just acquires/releases — no data transform.
    # This is enough to:
    #   - exercise shim-DMA channels into row 1/2 (creates DDR_PATCH ops)
    #   - mark each BO as touched in the partition (raises tile-utilisation)
    # without needing real compute kernels yet.
    chunk_elems = 32  # tiny slice, 64 B
    L1_chunk = np.ndarray[(chunk_elems,), dtype_vec]

    noop_fn = Kernel(
        f"{func_prefix}layer_fused_noop_bf16",
        f"{func_prefix}layer_fused_{e}_{h}_g{g}.o",
        [L1_chunk, L1_chunk, np.int32],
    )

    # One ObjectFifo per BO (in + out, depth=2). For BO3/BO4 (inout) we
    # reuse the same fifo for the read; the drain writes back to the same
    # offset so the BO content is unchanged.
    fifos_in  = [ObjectFifo(L1_chunk, name=f"in_{i}",  depth=2) for i in range(5)]
    fifos_out = [ObjectFifo(L1_chunk, name=f"out_{i}", depth=2) for i in range(5)]

    def passthrough_body(in_fifo, out_fifo, fn):
        for _ in range_(0xFFFFFFFF):
            i = in_fifo.acquire(1)
            o = out_fifo.acquire(1)
            fn(i, o, chunk_elems)
            in_fifo.release(1)
            out_fifo.release(1)

    workers = [
        Worker(
            passthrough_body,
            [fifos_in[i].cons(), fifos_out[i].prod(), noop_fn],
        )
        for i in range(5)
    ]

    # ── TensorAccessPatterns: linear chunk_elems prefix of each BO ───────────
    def chunk_tap(total_elems):
        return TensorAccessPattern(
            tensor_dims=(1, total_elems),
            offset=0,
            sizes=[1, 1, 1, chunk_elems],
            strides=[0, 0, 0, 1],
        )

    tap_qkv = chunk_tap(bo0_bytes // 2)
    tap_o   = chunk_tap(bo1_bytes // 2)
    tap_ffn = chunk_tap(bo2_bytes // 2)
    tap_kv  = chunk_tap(bo3_bytes // 2)
    tap_act = chunk_tap(bo4_elems)

    rt = Runtime()
    with rt.sequence(
        L3_w_qkv, L3_w_o, L3_w_ffn, L3_kv, L3_act,
    ) as (w_qkv, w_o, w_ffn, kv, act):
        rt.start(*workers)
        tg = rt.task_group()
        rt.fill (fifos_in [0].prod(), w_qkv, tap_qkv, task_group=tg)
        rt.fill (fifos_in [1].prod(), w_o,   tap_o,   task_group=tg)
        rt.fill (fifos_in [2].prod(), w_ffn, tap_ffn, task_group=tg)
        rt.fill (fifos_in [3].prod(), kv,    tap_kv,  task_group=tg)
        rt.fill (fifos_in [4].prod(), act,   tap_act, task_group=tg)
        # Drains all go back to act (inout) so we don't need 5 distinct
        # writeable bundles. For weights (in-only) the drain into a
        # writeable BO is fine — XRT just won't consider it a real output.
        rt.drain(fifos_out[0].cons(), act, tap_act, task_group=tg)
        rt.drain(fifos_out[1].cons(), act, tap_act, task_group=tg)
        rt.drain(fifos_out[2].cons(), act, tap_act, task_group=tg)
        rt.drain(fifos_out[3].cons(), kv,  tap_kv,  task_group=tg)
        rt.drain(fifos_out[4].cons(), act, tap_act, task_group=tg, wait=True)
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
