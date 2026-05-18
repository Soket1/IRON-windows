# Minimal DMA echo test for XDNA debugging.
# Four versions:
#   v1: single ObjectFifo (bo_in → tile → bo_out)
#   v2: dual ObjectFifo (bo_a + bo_b → tile concat → bo_out)
#   v3: FlowKV-like split path (bo_k → tile0 → inter FIFO → tile1, bo_v → tile1 → bo_out)
#   v4: FlowKV-like Q/K/V arg order and TAP offsets
#   v5: FlowKV-like multi-chunk Q/K/inter/V sequencing
#
# Usage: python design.py --dev npu2 --version 1 -o echo_v1.mlir
#        python design.py --dev npu2 --version 2 -o echo_v2.mlir
#        python design.py --dev npu2 --version 3 -o echo_v3.mlir
#        python design.py --dev npu2 --version 4 -o echo_v4.mlir
#        python design.py --dev npu2 --version 5 -o echo_v5.mlir

import numpy as np
from pathlib import Path
from ml_dtypes import bfloat16
import argparse

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.scf import _for as range_
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2
from aie.helpers.taplib import TensorAccessPattern


def echo_v1(dev, N=256):
    """Version 1: single ObjectFifo. DDR → tile memcpy → DDR."""
    dtype_in = np.dtype[bfloat16]
    dev_ty = NPU1() if dev == "npu" else NPU2()

    L1_ty = np.ndarray[(N,), dtype_in]
    L3_ty = np.ndarray[(N,), dtype_in]

    # Simple kernel: copy from in to out
    copy_fn = Kernel("echo_copy_bf16", "echo.o", [L1_ty, L1_ty, np.int32])

    in_fifo = ObjectFifo(L1_ty, name="in_fifo", depth=2)
    out_fifo = ObjectFifo(L1_ty, name="out_fifo", depth=2)

    def core_body(ifo, ofo, fn):
        for _ in range_(0xFFFFFFFF):
            inp = ifo.acquire(1)
            out = ofo.acquire(1)
            fn(inp, out, N)
            ifo.release(1)
            ofo.release(1)

    worker = Worker(core_body, [in_fifo.cons(), out_fifo.prod(), copy_fn])

    rt = Runtime()
    with rt.sequence(L3_ty, L3_ty) as (inp, outp):
        rt.start(worker)
        rt.fill(in_fifo.prod(), inp,
                TensorAccessPattern((N,), 0, [1, 1, 1, N], [0, 0, 0, 1]))
        rt.drain(out_fifo.cons(), outp,
                 TensorAccessPattern((N,), 0, [1, 1, 1, N], [0, 0, 0, 1]),
                 wait=True)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())


def echo_v2(dev, N=128):
    """Version 2: dual ObjectFifo. bo_a + bo_b → tile concat → bo_out.
    Simulates FlowKV's K+V → output pattern."""
    dtype_in = np.dtype[bfloat16]
    dev_ty = NPU1() if dev == "npu" else NPU2()

    L1_ty = np.ndarray[(N,), dtype_in]
    L1_full = np.ndarray[(2 * N,), dtype_in]
    L3_half = np.ndarray[(N,), dtype_in]   # one input
    L3_full = np.ndarray[(2 * N,), dtype_in]  # combined output

    concat_fn = Kernel("echo_concat_bf16", "echo.o",
                       [L1_ty, L1_ty, L1_full, np.int32])

    a_fifo = ObjectFifo(L1_ty, name="a_fifo", depth=2)
    b_fifo = ObjectFifo(L1_ty, name="b_fifo", depth=2)
    out_fifo = ObjectFifo(L1_full, name="out_fifo", depth=2)

    def core_body(af, bf, ofo, fn):
        for _ in range_(0xFFFFFFFF):
            a = af.acquire(1)
            b = bf.acquire(1)
            out = ofo.acquire(1)
            fn(a, b, out, N)
            af.release(1)
            bf.release(1)
            ofo.release(1)

    worker = Worker(core_body, [a_fifo.cons(), b_fifo.cons(),
                                out_fifo.prod(), concat_fn])

    rt = Runtime()
    with rt.sequence(L3_half, L3_half, L3_full) as (bo_a, bo_b, bo_out):
        rt.start(worker)
        tg = rt.task_group()
        rt.fill(a_fifo.prod(), bo_a,
                TensorAccessPattern((N,), 0, [1, 1, 1, N], [0, 0, 0, 1]),
                task_group=tg)
        rt.fill(b_fifo.prod(), bo_b,
                TensorAccessPattern((N,), 0, [1, 1, 1, N], [0, 0, 0, 1]),
                task_group=tg)
        tg_out = rt.task_group()
        rt.drain(out_fifo.cons(), bo_out,
                 TensorAccessPattern((2 * N,), 0, [1, 1, 1, 2 * N], [0, 0, 0, 1]),
                 task_group=tg_out, wait=True)
        rt.finish_task_group(tg)
        rt.finish_task_group(tg_out)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())


def echo_v3(dev, N=128):
    """Version 3: FlowKV-like two-worker path.
    bo_k -> k_fifo -> score tile -> inter_fifo -> value tile
    bo_v -> v_fifo ---------------------------> value tile -> out_fifo -> bo_out
    """
    dtype_in = np.dtype[bfloat16]
    dev_ty = NPU1() if dev == "npu" else NPU2()

    L1_ty = np.ndarray[(N,), dtype_in]
    L1_full = np.ndarray[(2 * N,), dtype_in]
    L3_half = np.ndarray[(N,), dtype_in]
    L3_full = np.ndarray[(2 * N,), dtype_in]

    score_fn = Kernel("echo_score_bf16", "echo.o", [L1_ty, L1_ty, np.int32])
    value_fn = Kernel("echo_value_bf16", "echo.o", [L1_ty, L1_ty, L1_full, np.int32])

    k_fifo = ObjectFifo(L1_ty, name="k_fifo", depth=2)
    v_fifo = ObjectFifo(L1_ty, name="v_fifo", depth=2)
    inter_fifo = ObjectFifo(L1_ty, name="inter_fifo", depth=2)
    out_fifo = ObjectFifo(L1_full, name="out_fifo", depth=2)

    def score_body(kf, inter, fn):
        for _ in range_(0xFFFFFFFF):
            k = kf.acquire(1)
            i = inter.acquire(1)
            fn(k, i, N)
            kf.release(1)
            inter.release(1)

    def value_body(inter, vf, ofo, fn):
        for _ in range_(0xFFFFFFFF):
            i = inter.acquire(1)
            v = vf.acquire(1)
            out = ofo.acquire(1)
            fn(i, v, out, N)
            inter.release(1)
            vf.release(1)
            ofo.release(1)

    score_worker = Worker(score_body, [k_fifo.cons(), inter_fifo.prod(), score_fn])
    value_worker = Worker(value_body, [inter_fifo.cons(), v_fifo.cons(), out_fifo.prod(), value_fn])

    rt = Runtime()
    with rt.sequence(L3_half, L3_half, L3_full) as (bo_k, bo_v, bo_out):
        rt.start(score_worker)
        rt.start(value_worker)
        tg = rt.task_group()
        rt.fill(k_fifo.prod(), bo_k,
                TensorAccessPattern((N,), 0, [1, 1, 1, N], [0, 0, 0, 1]),
                task_group=tg)
        rt.fill(v_fifo.prod(), bo_v,
                TensorAccessPattern((N,), 0, [1, 1, 1, N], [0, 0, 0, 1]),
                task_group=tg)
        tg_out = rt.task_group()
        rt.drain(out_fifo.cons(), bo_out,
                 TensorAccessPattern((2 * N,), 0, [1, 1, 1, 2 * N], [0, 0, 0, 1]),
                 task_group=tg_out, wait=True)
        rt.finish_task_group(tg)
        rt.finish_task_group(tg_out)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())


def echo_v4(dev, N=64, seq_len=2, q_stride=130):
    """Version 4: FlowKV-like Q/K/V arg order and TAP layout.
    rt.sequence args are K, V, Q, O like FlowKV. The score tile consumes Q and K,
    writes [K, Q] through inter_fifo, then the value tile consumes inter + V and
    writes [K, Q, V] to O. V TAP reads from offset N to mimic FlowKV's V region.
    """
    dtype_in = np.dtype[bfloat16]
    dev_ty = NPU1() if dev == "npu" else NPU2()

    L1_ty = np.ndarray[(N,), dtype_in]
    L1_inter = np.ndarray[(2 * N,), dtype_in]
    L1_out = np.ndarray[(3 * N,), dtype_in]
    L3_K_ty = np.ndarray[(seq_len * N,), dtype_in]
    L3_V_ty = np.ndarray[(2 * seq_len * N,), dtype_in]
    L3_Q_ty = np.ndarray[(q_stride,), dtype_in]
    L3_O_ty = np.ndarray[(3 * N,), dtype_in]

    score_fn = Kernel("echo_score_qk_bf16", "echo.o", [L1_ty, L1_ty, L1_inter, np.int32])
    value_fn = Kernel("echo_value_qkv_bf16", "echo.o", [L1_inter, L1_ty, L1_out, np.int32])

    q_fifo = ObjectFifo(L1_ty, name="q_fifo", depth=1)
    k_fifo = ObjectFifo(L1_ty, name="k_fifo", depth=2)
    v_fifo = ObjectFifo(L1_ty, name="v_fifo", depth=2)
    inter_fifo = ObjectFifo(L1_inter, name="inter_fifo", depth=2)
    out_fifo = ObjectFifo(L1_out, name="out_fifo", depth=2)

    def score_body(qf, kf, inter, fn):
        for _ in range_(0xFFFFFFFF):
            q = qf.acquire(1)
            k = kf.acquire(1)
            i = inter.acquire(1)
            fn(q, k, i, N)
            qf.release(1)
            kf.release(1)
            inter.release(1)

    def value_body(inter, vf, ofo, fn):
        for _ in range_(0xFFFFFFFF):
            i = inter.acquire(1)
            v = vf.acquire(1)
            out = ofo.acquire(1)
            fn(i, v, out, N)
            inter.release(1)
            vf.release(1)
            ofo.release(1)

    score_worker = Worker(score_body, [q_fifo.cons(), k_fifo.cons(), inter_fifo.prod(), score_fn])
    value_worker = Worker(value_body, [inter_fifo.cons(), v_fifo.cons(), out_fifo.prod(), value_fn])

    rt = Runtime()
    with rt.sequence(L3_K_ty, L3_V_ty, L3_Q_ty, L3_O_ty) as (K, V, Q, O):
        rt.start(score_worker)
        rt.start(value_worker)
        tg_k = rt.task_group()
        tg_v = rt.task_group()

        rt.fill(q_fifo.prod(), Q,
                TensorAccessPattern((q_stride,), 0, [1, 1, 1, N], [0, 0, 0, 1]),
                task_group=tg_k)
        rt.fill(k_fifo.prod(), K,
                TensorAccessPattern((seq_len * N,), 0, [1, 1, 1, N], [0, 0, 0, 1]),
                task_group=tg_k)
        rt.finish_task_group(tg_k)

        rt.fill(v_fifo.prod(), V,
                TensorAccessPattern((2 * seq_len * N,), N, [1, 1, 1, N], [0, 0, 0, 1]),
                task_group=tg_v)

        tg_out = rt.task_group()
        rt.drain(out_fifo.cons(), O,
                 TensorAccessPattern((3 * N,), 0, [1, 1, 1, 3 * N], [0, 0, 0, 1]),
                 task_group=tg_out, wait=True)
        rt.finish_task_group(tg_v)
        rt.finish_task_group(tg_out)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())


def echo_v5(dev, N=32, num_chunks=2, q_stride=66):
    """Version 5: FlowKV-like multi-chunk sequencing without attention math.
    Q is acquired once and held across chunks on the score tile. K/inter/V are
    acquired and released per chunk. The value tile writes final O only after all
    chunks have been consumed, mirroring FlowKV's value-normalize phase.
    """
    dtype_in = np.dtype[bfloat16]
    dev_ty = NPU1() if dev == "npu" else NPU2()

    L1_ty = np.ndarray[(N,), dtype_in]
    L1_inter = np.ndarray[(2 * N,), dtype_in]
    L1_out = np.ndarray[((3 + num_chunks) * N,), dtype_in]
    L3_K_ty = np.ndarray[(num_chunks * N,), dtype_in]
    L3_V_ty = np.ndarray[(2 * num_chunks * N,), dtype_in]
    L3_Q_ty = np.ndarray[(q_stride,), dtype_in]
    L3_O_ty = np.ndarray[((3 + num_chunks) * N,), dtype_in]

    score_fn = Kernel("echo_score_qk_bf16", "echo.o", [L1_ty, L1_ty, L1_inter, np.int32])
    value0_fn = Kernel("echo_value_chunk0_bf16", "echo.o", [L1_inter, L1_ty, L1_out, np.int32])
    value1_fn = Kernel("echo_value_chunk1_bf16", "echo.o", [L1_inter, L1_ty, L1_out, np.int32])

    q_fifo = ObjectFifo(L1_ty, name="q_fifo", depth=1)
    k_fifo = ObjectFifo(L1_ty, name="k_fifo", depth=2)
    v_fifo = ObjectFifo(L1_ty, name="v_fifo", depth=2)
    inter_fifo = ObjectFifo(L1_inter, name="inter_fifo", depth=2)
    out_fifo = ObjectFifo(L1_out, name="out_fifo", depth=2)

    def score_body(qf, kf, inter, fn):
        for _ in range_(0xFFFFFFFF):
            q = qf.acquire(1)
            for _ in range_(num_chunks):
                k = kf.acquire(1)
                i = inter.acquire(1)
                fn(q, k, i, N)
                kf.release(1)
                inter.release(1)
            qf.release(1)

    def value_body(inter, vf, ofo, fn0, fn1):
        for _ in range_(0xFFFFFFFF):
            out = ofo.acquire(1)

            i0 = inter.acquire(1)
            v0 = vf.acquire(1)
            fn0(i0, v0, out, N)
            inter.release(1)
            vf.release(1)

            i1 = inter.acquire(1)
            v1 = vf.acquire(1)
            fn1(i1, v1, out, N)
            inter.release(1)
            vf.release(1)

            ofo.release(1)

    score_worker = Worker(score_body, [q_fifo.cons(), k_fifo.cons(), inter_fifo.prod(), score_fn])
    value_worker = Worker(value_body, [inter_fifo.cons(), v_fifo.cons(), out_fifo.prod(), value0_fn, value1_fn])

    rt = Runtime()
    with rt.sequence(L3_K_ty, L3_V_ty, L3_Q_ty, L3_O_ty) as (K, V, Q, O):
        rt.start(score_worker)
        rt.start(value_worker)
        tg_k = rt.task_group()
        tg_v = rt.task_group()

        rt.fill(q_fifo.prod(), Q,
                TensorAccessPattern((q_stride,), 0, [1, 1, 1, N], [0, 0, 0, 1]),
                task_group=tg_k)
        for chunk in range(num_chunks):
            rt.fill(k_fifo.prod(), K,
                    TensorAccessPattern((num_chunks * N,), chunk * N, [1, 1, 1, N], [0, 0, 0, 1]),
                    task_group=tg_k)
        rt.finish_task_group(tg_k)

        for chunk in range(num_chunks):
            rt.fill(v_fifo.prod(), V,
                    TensorAccessPattern((2 * num_chunks * N,), (num_chunks + chunk) * N, [1, 1, 1, N], [0, 0, 0, 1]),
                    task_group=tg_v)

        tg_out = rt.task_group()
        rt.drain(out_fifo.cons(), O,
                 TensorAccessPattern(((3 + num_chunks) * N,), 0, [1, 1, 1, (3 + num_chunks) * N], [0, 0, 0, 1]),
                 task_group=tg_out, wait=True)
        rt.finish_task_group(tg_v)
        rt.finish_task_group(tg_out)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())


def echo_v6(dev, chunk_size=16, group_size=4, num_chunks=2, head_dim=64):
    """Version 6: FlowKV packed inter layout diagnostic.
    Mirrors the real [F_c | C_c | l] packed inter contract without attention math.
    """
    dtype_in = np.dtype[bfloat16]
    dev_ty = NPU1() if dev == "npu" else NPU2()

    scores_size = chunk_size * group_size
    packed_inter_size = scores_size + 2 * group_size
    v_size = chunk_size * head_dim
    out_block_size = packed_inter_size + v_size

    L1_kv_ty = np.ndarray[(v_size,), dtype_in]
    L1_q_ty = np.ndarray[(group_size * head_dim + head_dim + 2,), dtype_in]
    L1_inter_ty = np.ndarray[(packed_inter_size,), dtype_in]
    L1_out_ty = np.ndarray[(num_chunks * out_block_size,), dtype_in]
    L3_K_ty = np.ndarray[(num_chunks * v_size,), dtype_in]
    L3_V_ty = np.ndarray[(2 * num_chunks * v_size,), dtype_in]
    L3_Q_ty = np.ndarray[(group_size * head_dim + head_dim + 2,), dtype_in]
    L3_O_ty = np.ndarray[(num_chunks * out_block_size,), dtype_in]

    pack_fn = Kernel("echo_pack_inter_v6_bf16", "echo.o", [L1_kv_ty, L1_inter_ty, np.int32, np.int32, np.int32])
    value0_fn = Kernel("echo_value_v6_chunk0_bf16", "echo.o", [L1_inter_ty, L1_kv_ty, L1_out_ty, np.int32, np.int32, np.int32])
    value1_fn = Kernel("echo_value_v6_chunk1_bf16", "echo.o", [L1_inter_ty, L1_kv_ty, L1_out_ty, np.int32, np.int32, np.int32])

    q_fifo = ObjectFifo(L1_q_ty, name="q_fifo", depth=1)
    k_fifo = ObjectFifo(L1_kv_ty, name="k_fifo", depth=2)
    v_fifo = ObjectFifo(L1_kv_ty, name="v_fifo", depth=2)
    inter_fifo = ObjectFifo(L1_inter_ty, name="inter_fifo", depth=2)
    out_fifo = ObjectFifo(L1_out_ty, name="out_fifo", depth=1)

    def score_body(qf, kf, inter, fn):
        for _ in range_(0xFFFFFFFF):
            q = qf.acquire(1)
            for _ in range_(num_chunks):
                k = kf.acquire(1)
                p = inter.acquire(1)
                fn(k, p, chunk_size, group_size, head_dim)
                kf.release(1)
                inter.release(1)
            qf.release(1)

    def value_body(inter, vf, ofo, fn0, fn1):
        for _ in range_(0xFFFFFFFF):
            out = ofo.acquire(1)

            p0 = inter.acquire(1)
            v0 = vf.acquire(1)
            fn0(p0, v0, out, chunk_size, group_size, head_dim)
            inter.release(1)
            vf.release(1)

            p1 = inter.acquire(1)
            v1 = vf.acquire(1)
            fn1(p1, v1, out, chunk_size, group_size, head_dim)
            inter.release(1)
            vf.release(1)

            ofo.release(1)

    score_worker = Worker(score_body, [q_fifo.cons(), k_fifo.cons(), inter_fifo.prod(), pack_fn])
    value_worker = Worker(value_body, [inter_fifo.cons(), v_fifo.cons(), out_fifo.prod(), value0_fn, value1_fn])

    rt = Runtime()
    with rt.sequence(L3_K_ty, L3_V_ty, L3_Q_ty, L3_O_ty) as (K, V, Q, O):
        rt.start(score_worker)
        rt.start(value_worker)
        tg_k = rt.task_group()
        tg_v = rt.task_group()

        rt.fill(q_fifo.prod(), Q,
                TensorAccessPattern((group_size * head_dim + head_dim + 2,), 0, [1, 1, 1, group_size * head_dim + head_dim + 2], [0, 0, 0, 1]),
                task_group=tg_k)
        for chunk in range(num_chunks):
            rt.fill(k_fifo.prod(), K,
                    TensorAccessPattern((num_chunks * v_size,), chunk * v_size, [1, 1, 1, v_size], [0, 0, 0, 1]),
                    task_group=tg_k)
        rt.finish_task_group(tg_k)

        for chunk in range(num_chunks):
            rt.fill(v_fifo.prod(), V,
                    TensorAccessPattern((2 * num_chunks * v_size,), (num_chunks + chunk) * v_size, [1, 1, 1, v_size], [0, 0, 0, 1]),
                    task_group=tg_v)

        tg_out = rt.task_group()
        rt.drain(out_fifo.cons(), O,
                 TensorAccessPattern((num_chunks * out_block_size,), 0, [1, 1, 1, num_chunks * out_block_size], [0, 0, 0, 1]),
                 task_group=tg_out, wait=True)
        rt.finish_task_group(tg_v)
        rt.finish_task_group(tg_out)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())


def echo_v7(dev, chunk_size=16, group_size=4, num_chunks=2, head_dim=64):
    """Version 7: FlowKV score math diagnostic.
    Runs real score-side online-softmax math and drains packed [F_c | C_c | l].
    """
    dtype_in = np.dtype[bfloat16]
    dev_ty = NPU1() if dev == "npu" else NPU2()

    q_stride = group_size * head_dim + head_dim + 2
    kv_size = chunk_size * head_dim
    packed_inter_size = chunk_size * group_size + 2 * group_size

    L1_q_ty = np.ndarray[(q_stride,), dtype_in]
    L1_k_ty = np.ndarray[(kv_size,), dtype_in]
    L1_inter_ty = np.ndarray[(packed_inter_size,), dtype_in]
    L1_out_ty = np.ndarray[(num_chunks * packed_inter_size,), dtype_in]
    L3_K_ty = np.ndarray[(num_chunks * kv_size,), dtype_in]
    L3_Q_ty = np.ndarray[(q_stride,), dtype_in]
    L3_O_ty = np.ndarray[(num_chunks * packed_inter_size,), dtype_in]

    score_init = Kernel("echo_v7_score_init_bf16", "echo.o", [np.int32])
    score_rope_q = Kernel("echo_v7_score_rope_q_bf16", "echo.o", [L1_q_ty, np.int32, np.int32])
    score_chunk = Kernel("echo_v7_score_chunk_bf16", "echo.o", [L1_q_ty, L1_k_ty, L1_inter_ty, np.int32, np.int32, np.int32])
    copy0_fn = Kernel("echo_v7_copy_pack_chunk0_bf16", "echo.o", [L1_inter_ty, L1_out_ty, np.int32])
    copy1_fn = Kernel("echo_v7_copy_pack_chunk1_bf16", "echo.o", [L1_inter_ty, L1_out_ty, np.int32])

    q_fifo = ObjectFifo(L1_q_ty, name="q_fifo", depth=1)
    k_fifo = ObjectFifo(L1_k_ty, name="k_fifo", depth=2)
    inter_fifo = ObjectFifo(L1_inter_ty, name="inter_fifo", depth=2)
    out_fifo = ObjectFifo(L1_out_ty, name="out_fifo", depth=1)

    def score_body(qf, kf, inter, init_fn, rope_fn, chunk_fn):
        for _ in range_(0xFFFFFFFF):
            init_fn(group_size)
            q = qf.acquire(1)
            rope_fn(q, group_size, head_dim)
            for _ in range_(num_chunks):
                k = kf.acquire(1)
                p = inter.acquire(1)
                chunk_fn(q, k, p, group_size, head_dim, chunk_size)
                kf.release(1)
                inter.release(1)
            qf.release(1)

    def drain_body(inter, ofo, fn0, fn1):
        for _ in range_(0xFFFFFFFF):
            out = ofo.acquire(1)

            p0 = inter.acquire(1)
            fn0(p0, out, packed_inter_size)
            inter.release(1)

            p1 = inter.acquire(1)
            fn1(p1, out, packed_inter_size)
            inter.release(1)

            ofo.release(1)

    score_worker = Worker(score_body, [q_fifo.cons(), k_fifo.cons(), inter_fifo.prod(), score_init, score_rope_q, score_chunk])
    drain_worker = Worker(drain_body, [inter_fifo.cons(), out_fifo.prod(), copy0_fn, copy1_fn])

    rt = Runtime()
    with rt.sequence(L3_K_ty, L3_Q_ty, L3_O_ty) as (K, Q, O):
        rt.start(score_worker)
        rt.start(drain_worker)
        tg_k = rt.task_group()

        rt.fill(q_fifo.prod(), Q,
                TensorAccessPattern((q_stride,), 0, [1, 1, 1, q_stride], [0, 0, 0, 1]),
                task_group=tg_k)
        for chunk in range(num_chunks):
            rt.fill(k_fifo.prod(), K,
                    TensorAccessPattern((num_chunks * kv_size,), chunk * kv_size, [1, 1, 1, kv_size], [0, 0, 0, 1]),
                    task_group=tg_k)
        rt.finish_task_group(tg_k)

        tg_out = rt.task_group()
        rt.drain(out_fifo.cons(), O,
                 TensorAccessPattern((num_chunks * packed_inter_size,), 0, [1, 1, 1, num_chunks * packed_inter_size], [0, 0, 0, 1]),
                 task_group=tg_out, wait=True)
        rt.finish_task_group(tg_out)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="npu2", choices=["npu", "npu2"])
    ap.add_argument("--version", type=int, choices=[1, 2, 3, 4, 5, 6, 7], required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("-o", "--output-file-path", required=True)
    args = ap.parse_args()

    if args.version == 1:
        module = echo_v1(args.dev, args.n)
    elif args.version == 2:
        module = echo_v2(args.dev, args.n)
    elif args.version == 3:
        module = echo_v3(args.dev, args.n)
    elif args.version == 4:
        module = echo_v4(args.dev)
    elif args.version == 5:
        module = echo_v5(args.dev)
    elif args.version == 6:
        module = echo_v6(args.dev)
    else:
        module = echo_v7(args.dev)

    Path(args.output_file_path).write_text(str(module))
    print(f"Wrote {args.output_file_path}")
