# SPDX-License-Identifier: Apache-2.0
"""Cross-column attn_out gather probe — final fused-layer prerequisite (#17).

After flowkv, each of the 4 central columns holds its own E/4=512 slice of the
attention output. O_proj needs the FULL E=2048 vector. This probe proves the
on-chip gather: 4 column workers each produce their 512-elem slice into a
per-column fifo; ObjectFifo.join gathers them at a MemTile into one E-elem
buffer, which is drained to DDR. Validates that join assembles the 4 slices into
the correct full-E vector (the R2-F2b MemTile-join mechanism) on cols 2-5,
isolated from O_proj.

Each column worker here just copies a DDR-fed slice (stand-in for the value
stage output) so the gather assembly is what's under test.
"""
import numpy as np
from ml_dtypes import bfloat16

from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2, Tile


def my_gather(dev, embed_dim=2048, num_cols=4, col_offset=2):
    dev_ty = NPU1() if dev == "npu" else NPU2()
    bf = np.dtype[bfloat16]
    E = embed_dim
    slice_n = E // num_cols                         # 512 per column

    L1_S_ty = np.ndarray[(slice_n,), bf]            # per-column slice
    L1_E_ty = np.ndarray[(E,), bf]                  # full assembled vector

    # passthrough copy kernel (stand-in for value-stage output)
    copy = Kernel("layer_fused_add_bf16", "layer_fused_gather.o",
                  [L1_S_ty, L1_S_ty, L1_S_ty, np.int32])

    # Per-column input slices (fed from DDR), and per-column producer fifos that
    # carry each column's 512 slice toward the join.
    In_f = [ObjectFifo(L1_S_ty, name=f"In_{c}", depth=2) for c in range(num_cols)]
    Z_f  = [ObjectFifo(L1_S_ty, name=f"Z_{c}", depth=1) for c in range(num_cols)]  # zero add operand

    # The assembled full-E fifo: producer side is a join of N column sub-fifos at
    # a MemTile (offsets place each 512 slice into the E vector). Workers write
    # into the sub-fifos (parts); Full.cons() yields the assembled E vector.
    Full = ObjectFifo(L1_E_ty, name="Full", depth=2)
    parts = Full.prod().join(
        offsets=[c * slice_n for c in range(num_cols)],
        placement=Tile(col=col_offset, row=1),       # MemTile in central band
        obj_types=[L1_S_ty for _ in range(num_cols)],
    )

    workers = []
    for c in range(num_cols):
        pc = c + col_offset

        def body(inf, zf, partf, copy_fn):
            for _ in range_(0xFFFFFFFF):
                a = inf.acquire(1)
                z = zf.acquire(1)
                o = partf.acquire(1)
                copy_fn(a, z, o, slice_n)            # o = in + 0 = in (copy)
                inf.release(1); zf.release(1); partf.release(1)

        workers.append(Worker(
            body, [In_f[c].cons(), Z_f[c].cons(), parts[c].prod(), copy],
            placement=Tile(col=pc, row=2)))

    def in_tap(c):
        return TensorAccessPattern(tensor_dims=(1, E), offset=c * slice_n,
            sizes=[1, 1, 1, slice_n], strides=[0, 0, 0, 1])
    z_tap = TensorAccessPattern(tensor_dims=(1, slice_n), offset=0,
        sizes=[1, 1, 1, slice_n], strides=[0, 0, 0, 1])

    L3_E = np.ndarray[(E,), bf]
    L3_Z = np.ndarray[(slice_n,), bf]
    rt = Runtime()
    with rt.sequence(L3_E, L3_E, L3_Z) as (out, inp, zero):
        rt.start(*workers)
        tg = rt.task_group()
        for c in range(num_cols):
            rt.fill(In_f[c].prod(), inp, in_tap(c), task_group=tg)
            rt.fill(Z_f[c].prod(), zero, z_tap, task_group=tg)
        rt.drain(Full.cons(), out, task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())
