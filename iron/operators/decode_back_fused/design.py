# SPDX-License-Identifier: Apache-2.0
"""decode_back_fused — #11.1a: 16-tile back-half (O->ANM->FFN) in ONE dispatch.

SKELETON (copy bodies, DBG_SKEL=1 default here): proves the full fused-back-half
dataflow before real GEMV. Two phases time-muxed on the same 16 center tiles
(cols 2-5 x rows 2-5), with TWO on-chip hubs between/after:
  attn(DDR) -bcast-> 16 tiles(O) -2lvl gather-> ANM-HUB(+h_in) -bcast(on-chip)->
  16 tiles(FFN) -2lvl gather-> RESID-HUB(+inpff) -> out(DDR)

Reuse to stay within caps (IRON = 1 DMA ch/fifo, no temporal packing):
  - ONE per-column join PF[c] reused both phases (else 8 S2MM > MemTile 6).
  - ONE L2 join FinalParts reused both phases.
  - MemTile(c,1): Bcol-fill + FfnCol-fill + PF[c] join(4) = 6 S2MM (= cap).
center 2 S2MM(bcol,ffncol)+1 MM2S(pf). ANM-hub 2+2. resid-hub 2+1. (probe0/probe1
proved the hub round-trip + on-chip broadcast-from-hub.) Skeleton predictable:
  O=16*attn ; inpff=O+h_in ; ffn_in=inpff ; FFN=16*ffn_in ; out=FFN+inpff=17*inpff.
Swap real bodies (gemv/gate_up/silu/down + rms ANM) at DBG_SKEL=0 next (#11.1b).
"""
import numpy as np
from ml_dtypes import bfloat16

from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2, Tile


def my_decode_back_fused(dev, embed_dim=2048, num_cols=4, col_offset=2):
    dev_ty = NPU1() if dev == "npu" else NPU2()
    bf = np.dtype[bfloat16]
    E = embed_dim
    NC, R = num_cols, 4
    L1_E_ty = np.ndarray[(E,), bf]
    L1_4E_ty = np.ndarray[(R * E,), bf]
    L3_ty = np.ndarray[(E,), bf]

    add = Kernel("layer_fused_add_bf16", "layer_fused_relay.o",
                 [L1_E_ty, L1_E_ty, L1_E_ty, np.int32])
    reduce4 = Kernel("layer_fused_reduce4_bf16", "layer_fused_relay.o",
                     [L1_4E_ty, L1_E_ty, L1_E_ty, np.int32])

    _zn = [0]
    def mk_zero():
        b = Buffer(type=L1_E_ty, initial_value=np.zeros(E, dtype=bfloat16),
                   name=f"bf_zero_{_zn[0]}"); _zn[0] += 1
        return b

    # ---- broadcast forwards (per column): phase1 attn (DDR), phase2 ffn_in (hub) ----
    Bsrc = [ObjectFifo(L1_E_ty, name=f"Bsrc_{c}", depth=1) for c in range(NC)]
    Bcol = [Bsrc[c].cons().forward(name=f"Bcol_{c}", depth=2,
                                   placement=Tile(col=col_offset + c, row=1))
            for c in range(NC)]
    Ffn = ObjectFifo(L1_E_ty, name="Ffn", depth=1)                  # ANM-hub -> MemTile(1,1)
    # ONE forward (one link); MemTile(1,1) broadcasts ffn_in to all 16 center tiles.
    FfnB = Ffn.cons().forward(name="FfnB", depth=2, placement=Tile(col=1, row=1))

    # ---- per-column join PF[c] (REUSED both phases): 4 partials -> 4E @ MemTile(c,1) ----
    PF, PF_parts = [], []
    for c in range(NC):
        pf = ObjectFifo(L1_4E_ty, name=f"PF_{c}", depth=1)
        parts = pf.prod().join(offsets=[r * E for r in range(R)],
                               placement=Tile(col=col_offset + c, row=1),
                               obj_types=[L1_E_ty for _ in range(R)])
        PF.append(pf); PF_parts.append(parts)

    # ---- L2 join FinalParts (REUSED both phases): 4 col-partials -> 4E @ MemTile(6,1) ----
    FinalParts = ObjectFifo(L1_4E_ty, name="FinalParts", depth=1)
    FP_parts = FinalParts.prod().join(offsets=[c * E for c in range(NC)],
                                      placement=Tile(col=6, row=1),
                                      obj_types=[L1_E_ty for _ in range(NC)])
    Ofifo = ObjectFifo(L1_E_ty, name="Ofifo", depth=2)              # finalred p1 -> ANM-hub
    FFNfifo = ObjectFifo(L1_E_ty, name="FFNfifo", depth=2)          # finalred p2 -> resid-hub

    Hin = ObjectFifo(L1_E_ty, name="Hin", depth=1)                  # residual (DDR) -> ANM-hub
    InpffRelay = ObjectFifo(L1_E_ty, name="InpffRelay", depth=2)    # ANM-hub -> resid-hub
    Out = ObjectFifo(L1_E_ty, name="Out", depth=2)

    workers = []

    def center_body(bcol, ffncol, pf_part, zero, add_fn):
        for _ in range_(0xFFFFFFFF):
            b = bcol.acquire(1); p = pf_part.acquire(1)
            add_fn(b, zero, p, E)                                   # phase1 O: attn -> partial
            bcol.release(1); pf_part.release(1)
            a = ffncol.acquire(1); p = pf_part.acquire(1)
            add_fn(a, zero, p, E)                                   # phase2 FFN: ffn_in -> partial
            ffncol.release(1); pf_part.release(1)

    for c in range(NC):
        for r in range(R):
            workers.append(Worker(
                center_body,
                [Bcol[c].cons(), FfnB.cons(), PF_parts[c][r].prod(), mk_zero(), add],
                placement=Tile(col=col_offset + c, row=2 + r)))

    def colred_body(pf, fp, zero, red_fn):
        for _ in range_(0xFFFFFFFF):
            f = pf.acquire(1); o = fp.acquire(1)
            red_fn(f, zero, o, E); pf.release(1); fp.release(1)     # phase1 -> L2 slot
            f = pf.acquire(1); o = fp.acquire(1)
            red_fn(f, zero, o, E); pf.release(1); fp.release(1)     # phase2 -> L2 slot

    for c in range(NC):
        workers.append(Worker(colred_body, [PF[c].cons(), FP_parts[c].prod(), mk_zero(), reduce4],
                              placement=Tile(col=1, row=2 + c)))

    def finalred_body(fp, ofi, ffni, zero, red_fn):
        for _ in range_(0xFFFFFFFF):
            f = fp.acquire(1); o = ofi.acquire(1)
            red_fn(f, zero, o, E); fp.release(1); ofi.release(1)    # phase1 -> O
            f = fp.acquire(1); o = ffni.acquire(1)
            red_fn(f, zero, o, E); fp.release(1); ffni.release(1)   # phase2 -> FFN

    workers.append(Worker(finalred_body, [FinalParts.cons(), Ofifo.prod(), FFNfifo.prod(),
                                          mk_zero(), reduce4], placement=Tile(col=6, row=2)))

    def anm_hub_body(ofi, hin, ffn_out, inpff_out, add_fn):
        for _ in range_(0xFFFFFFFF):
            o = ofi.acquire(1); h = hin.acquire(1)
            ir = inpff_out.acquire(1); add_fn(o, h, ir, E)         # inpff = O + h_in -> relay
            fi = ffn_out.acquire(1); add_fn(o, h, fi, E)           # ffn_in = inpff (skeleton)
            ofi.release(1); hin.release(1); inpff_out.release(1); ffn_out.release(1)

    workers.append(Worker(anm_hub_body, [Ofifo.cons(), Hin.cons(), Ffn.prod(), InpffRelay.prod(), add],
                          placement=Tile(col=0, row=2)))

    def resid_hub_body(ffni, inpff_in, out, add_fn):
        for _ in range_(0xFFFFFFFF):
            f = ffni.acquire(1); ir = inpff_in.acquire(1); o = out.acquire(1)
            add_fn(f, ir, o, E)                                    # out = FFN + inpff
            ffni.release(1); inpff_in.release(1); out.release(1)

    workers.append(Worker(resid_hub_body, [FFNfifo.cons(), InpffRelay.cons(), Out.prod(), add],
                          placement=Tile(col=0, row=3)))

    tap = TensorAccessPattern(tensor_dims=(1, E), offset=0,
                              sizes=[1, 1, 1, E], strides=[0, 0, 0, 1])
    rt = Runtime()
    # bo0=out, bo1=attn(O-input), bo2=h_in(residual)
    with rt.sequence(L3_ty, L3_ty, L3_ty) as (o, attn, hin):
        rt.start(*workers)
        tg = rt.task_group()
        for c in range(NC):
            rt.fill(Bsrc[c].prod(), attn, tap, task_group=tg)
        rt.fill(Hin.prod(), hin, tap, task_group=tg)
        rt.drain(Out.cons(), o, tap, task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())
