#!/usr/bin/env python3
"""Minimal DMA sanity test for XDNA2.

Tests if DMA can correctly read/write DDR buffers.
Generates a minimal MLIR with 2 tiles: one reads from DDR, one writes back.
If output matches input, DMA works correctly at the IRON level.

Usage:
    python test_dma_sanity.py --dev npu2 --buf-size 1024
    python test_dma_sanity.py --dev npu2 --buf-size 1024 --output-file sanity.mlir
"""

import numpy as np
from ml_dtypes import bfloat16
from pathlib import Path
import argparse

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.scf import _for as range_
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2


def create_dma_sanity_design(dev="npu2", buf_size=1024):
    """Create a minimal DMA pass-through design.

    Architecture:
        DDR input → ObjectFifo → Score tile (copies) → ObjectFifo → DDR output

    The score tile reads buf_size bf16 values from DDR into tile memory,
    then writes them back to DDR. If output matches input, DMA works.
    """
    dtype = np.dtype[bfloat16]
    dev_ty = NPU1() if dev == "npu" else NPU2()

    L1_ty = np.ndarray[(buf_size,), dtype]

    # ObjectFifos: DDR ↔ tile
    in_fifo = ObjectFifo(L1_ty, name="in", depth=1)
    out_fifo = ObjectFifo(L1_ty, name="out", depth=1)

    def core_body():
        for _ in range_(0xFFFFFFFF):
            inp = in_fifo.acquire(1)
            out = out_fifo.acquire(1)
            # The DMA already moved data into tile memory.
            # We just need to trigger the pipeline by acquiring and releasing.
            # The actual data movement is: DDR → tile (via in_fifo acquire),
            # then tile → DDR (via out_fifo release).
            # For a pure DMA test, the core doesn't need to touch the data.
            out_fifo.release(1)
            in_fifo.release(1)

    worker = Worker(
        core_body,
        [in_fifo.cons(), out_fifo.prod()],
    )

    L3_ty = np.ndarray[(buf_size,), dtype]

    rt = Runtime()
    with rt.sequence(L3_ty, L3_ty) as (inp, out):
        rt.start(worker)
        rt.fill(in_fifo.prod(), inp)
        rt.drain(out_fifo.cons(), out, wait=True)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        prog="AIE DMA Sanity Test",
        description="Minimal test to verify DMA read/write works on XDNA2.",
    )
    argparser.add_argument("--dev", type=str, choices=["npu", "npu2"], default="npu2")
    argparser.add_argument("--buf-size", type=int, default=1024,
                           help="Buffer size in bf16 elements (default: 1024)")
    argparser.add_argument("--output-file", "-o", type=str, default=None,
                           help="Output MLIR file path")
    args = argparser.parse_args()

    module = create_dma_sanity_design(args.dev, args.buf_size)

    if args.output_file:
        output_path = Path(args.output_file)
    else:
        output_path = Path("dma_sanity.mlir")

    with open(output_path, "w") as f:
        f.write(str(module))

    print(f"MLIR written to {output_path}")
    print(f"Buffer size: {args.buf_size} bf16 elements ({args.buf_size * 2} bytes)")
