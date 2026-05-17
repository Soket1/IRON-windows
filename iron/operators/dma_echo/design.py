# Minimal DMA echo test for XDNA debugging.
# Two versions:
#   v1: single ObjectFifo (bo_in → tile → bo_out)
#   v2: dual ObjectFifo (bo_a + bo_b → tile concat → bo_out)
#
# Usage: python design.py --dev npu2 --version 1 -o echo_v1.mlir
#        python design.py --dev npu2 --version 2 -o echo_v2.mlir

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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="npu2", choices=["npu", "npu2"])
    ap.add_argument("--version", type=int, choices=[1, 2], required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("-o", "--output-file-path", required=True)
    args = ap.parse_args()

    if args.version == 1:
        module = echo_v1(args.dev, args.n)
    else:
        module = echo_v2(args.dev, args.n)

    Path(args.output_file_path).write_text(str(module))
    print(f"Wrote {args.output_file_path}")
