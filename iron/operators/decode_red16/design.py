# SPDX-License-Identifier: Apache-2.0
"""decode_red16 — D2.2a dataflow SKELETON: broadcast-to-16 + 2-level reduce.

Isolates the NEW D2 topology risk (D1 only broadcasts/reduces 4-way, one row)
from the compute. 16 tiles (cols 2-5 × rows 2-5) each do a TRIVIAL identity copy
of a broadcast input; their 16 partials reduce 2-level to one output:
  - broadcast: input → 4 per-column forwards (MemTile(c,1) → its 4 row-tiles).
  - reduce L1: per column, 4 partials join → MemTile(c,1) → reduce4 core (col 1).
  - reduce L2: 4 column-partials join → MemTile(6,1) → reduce4 core (col 6) → out.
Expected out = 16 × input (fp32 accum → bf16). If this compiles + runs + matches,
the topology is proven and the real FFN body (gate/up/silu/down) drops in (D2.2a).

DMA budget (all ≤ caps): copy tile 1 S2MM + 1 MM2S; MemTile(c,1) 5 S2MM (4 join +
1 bcast-src) + 5 MM2S (4 bcast-out + 1 PF→reduce); col-reduce/final-reduce cores
1+1; MemTile(6,1) 4 S2MM + 1 MM2S. 5-BO ABI (out=bo0, in=bo2, bo1/3/4 dummy).
See dev_notes/track_a_build/D2_design_spike.md.
"""
import numpy as np
from ml_dtypes import bfloat16

from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2, Tile


def my_decode_red16(dev, embed_dim=2048, num_cols=4, col_offset=2):
    dev_ty = NPU1() if dev == "npu" else NPU2()
    bf = np.dtype[bfloat16]
    E = embed_dim
    R = 4                                  # rows 2..5 per column
    NC = num_cols

    L1_E_ty  = np.ndarray[(E,), bf]
    L1_4E_ty = np.ndarray[(4 * E,), bf]
    L3_E_ty  = np.ndarray[(E,), bf]
    L3_D_ty  = np.ndarray[(64,), bf]       # dummy bo1/bo3/bo4 (5-BO ABI pad)

    add = Kernel("layer_fused_add_bf16", "layer_fused_relay.o",
                 [L1_E_ty, L1_E_ty, L1_E_ty, np.int32])
    reduce4 = Kernel("layer_fused_reduce4_bf16", "layer_fused_relay.o",
                     [L1_4E_ty, L1_E_ty, L1_E_ty, np.int32])

    # A Buffer binds to ONE tile, so each worker needs its own zero buffer.
    _zn = [0]
    def mk_zero():
        b = Buffer(type=L1_E_ty, initial_value=np.zeros(E, dtype=bfloat16),
                   name=f"red16_zero_{_zn[0]}")
        _zn[0] += 1
        return b

    # --- broadcast: one input fill per column → MemTile(c,1) forward to 4 rows ---
    Bsrc = [ObjectFifo(L1_E_ty, name=f"Bsrc_{c}", depth=1) for c in range(NC)]
    Bc = [Bsrc[c].cons().forward(name=f"Bc_{c}", depth=2,
                                 placement=Tile(col=col_offset + c, row=1))
          for c in range(NC)]

    # --- L1 join: per column, 4 row partials → 4E on MemTile(c,1) ---
    PF = []
    PF_parts = []
    for c in range(NC):
        pf = ObjectFifo(L1_4E_ty, name=f"PF_{c}", depth=1)
        parts = pf.prod().join(
            offsets=[r * E for r in range(R)],
            placement=Tile(col=col_offset + c, row=1),
            obj_types=[L1_E_ty for _ in range(R)])
        PF.append(pf)
        PF_parts.append(parts)

    # --- L2 join: 4 column-partials → 4E on MemTile(6,1) ---
    FinalParts = ObjectFifo(L1_4E_ty, name="FinalParts", depth=1)
    FP_parts = FinalParts.prod().join(
        offsets=[c * E for c in range(NC)],
        placement=Tile(col=6, row=1),
        obj_types=[L1_E_ty for _ in range(NC)])

    Cout = ObjectFifo(L1_E_ty, name="Cout", depth=2)   # final output → drain

    workers = []

    # 16 copy tiles
    def copy_body(bc, pp, zero, add_fn):
        for _ in range_(0xFFFFFFFF):
            a = bc.acquire(1)
            o = pp.acquire(1)
            add_fn(a, zero, o, E)
            bc.release(1)
            pp.release(1)

    for c in range(NC):
        for r in range(R):
            workers.append(Worker(
                copy_body,
                [Bc[c].cons(), PF_parts[c][r].prod(), mk_zero(), add],
                placement=Tile(col=col_offset + c, row=2 + r)))

    # 4 column-reduce cores (col 1) : PF[c] (4E) + 0 → FP_parts[c]
    def colred_body(pf, cp, zero, red_fn):
        for _ in range_(0xFFFFFFFF):
            p = pf.acquire(1)
            o = cp.acquire(1)
            red_fn(p, zero, o, E)
            pf.release(1)
            cp.release(1)

    for c in range(NC):
        workers.append(Worker(
            colred_body,
            [PF[c].cons(), FP_parts[c].prod(), mk_zero(), reduce4],
            placement=Tile(col=1, row=2 + c)))

    # final reduce core (col 6) : FinalParts (4E) + 0 → Cout
    def finalred_body(fp, out, zero, red_fn):
        for _ in range_(0xFFFFFFFF):
            p = fp.acquire(1)
            o = out.acquire(1)
            red_fn(p, zero, o, E)
            fp.release(1)
            out.release(1)

    workers.append(Worker(
        finalred_body,
        [FinalParts.cons(), Cout.prod(), mk_zero(), reduce4],
        placement=Tile(col=6, row=2)))

    e_tap = TensorAccessPattern(tensor_dims=(1, E), offset=0,
                                sizes=[1, 1, 1, E], strides=[0, 0, 0, 1])

    rt = Runtime()
    # 5-BO ABI: bo0=Cout out, bo1=dummy, bo2=input, bo3=dummy, bo4=dummy.
    with rt.sequence(L3_E_ty, L3_D_ty, L3_E_ty, L3_D_ty, L3_D_ty) as (o, _d1, x, _d3, _d4):
        rt.start(*workers)
        tg = rt.task_group()
        for c in range(NC):
            rt.fill(Bsrc[c].prod(), x, e_tap, task_group=tg)
        rt.drain(Cout.cons(), o, e_tap, task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())
