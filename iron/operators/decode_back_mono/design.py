# SPDX-License-Identifier: Apache-2.0
"""decode_back_mono — #11.1b/#12: fused back-half with the register-resident
MONOLITHIC FFN kernel (layer_fused_ffn_mono_bf16). Same proven choreography as
decode_back_real (AHUB Tile1,2 merges attn+ffn_in into one broadcast; padded-O
shares the FFN reduce join; finalred folds h_in; dual-hub) but the FFN phase is
ONE mono call per hidden-chunk that accumulates the down partial in down_acc,
with gate/up/silu in registers and NO lf_left/lf_right/lf_silu L1 statics —
removing exactly the statics involved in decode_back_real's down-0.13 residual.

Weight element = GUD (uniform): phase1 O uses 32 GUD elements (o_scatter reads
the first PACKED, rest padding); phase2 FFN uses gu_tiles GUD elements (mono
reads the full gud = [gate PK][up PK][down_nib m*E/2][down_scl E]).
"""
import numpy as np
from ml_dtypes import bfloat16
import os as _os

import aie.dialects.index as index
from aie.dialects.aie import T
from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2, Tile


def my_decode_back_mono(dev, embed_dim=2048, hidden_dim=8192, group_size=32,
                        m_input=4, num_cols=4, col_offset=2):
    dev_ty = NPU1() if dev == "npu" else NPU2()
    bf = np.dtype[bfloat16]
    u8 = np.dtype[np.uint8]
    E, H, g, m = embed_dim, hidden_dim, group_size, m_input
    NC, R = num_cols, 4
    NT = NC * R                                    # 16
    Hc16 = H // NT                                 # 512
    gu_tiles = Hc16 // m                           # 128 (FFN mono calls)
    SL_O = E // NT                                 # 128
    o_tiles = SL_O // m                            # 32 (O scatter calls)
    PACKED = m * E // 2 + m * (E // g) * 2         # 4608 (gate/up/o tile)
    GUD = 2 * PACKED + m * E // 2 + E * 2          # mono weight element
    WT_PER_TILE = o_tiles + gu_tiles              # 160 GUD elements/tile

    L1_E_ty = np.ndarray[(E,), bf]
    L1_4E_ty = np.ndarray[(R * E,), bf]
    L1_W_ty = np.ndarray[(GUD,), u8]
    L1_W4_ty = np.ndarray[(R * GUD,), u8]
    L3_E_ty = np.ndarray[(E,), bf]
    L3_W_ty = np.ndarray[(NC * WT_PER_TILE * R * GUD,), u8]

    oscat = Kernel("layer_fused_o_scatter_bf16", "layer_fused_relay.o",
                   [np.int32, np.int32, L1_W_ty, L1_E_ty, L1_E_ty])
    mono = Kernel("layer_fused_ffn_mono_bf16", "layer_fused_relay.o",
                  [np.int32, L1_W_ty, L1_E_ty, L1_E_ty])
    reduce4 = Kernel("layer_fused_reduce4_bf16", "layer_fused_relay.o",
                     [L1_4E_ty, L1_E_ty, L1_E_ty, np.int32])
    add = Kernel("layer_fused_add_bf16", "layer_fused_relay.o",
                 [L1_E_ty, L1_E_ty, L1_E_ty, np.int32])
    rms = Kernel("layer_fused_rms_norm_bf16", "layer_fused_relay.o",
                 [L1_E_ty, L1_E_ty, L1_E_ty, np.int32])

    _zn = [0]
    def mk_zero():
        b = Buffer(type=L1_E_ty, initial_value=np.zeros(E, dtype=bfloat16),
                   name=f"bm_zero_{_zn[0]}"); _zn[0] += 1
        return b
    gain_buf = Buffer(type=L1_E_ty, initial_value=np.ones(E, dtype=bfloat16), name="bm_gain")

    Attn = ObjectFifo(L1_E_ty, name="Attn", depth=1)
    Inpff = ObjectFifo(L1_E_ty, name="Inpff", depth=2)
    Bc = ObjectFifo(L1_E_ty, name="Bc", depth=1)
    BcB = Bc.cons().forward(name="BcB", depth=2, placement=Tile(col=1, row=1))

    Wsrc = [ObjectFifo(L1_W4_ty, name=f"Wsrc_{c}", depth=2) for c in range(NC)]
    Wf = [Wsrc[c].cons().split(offsets=[r * GUD for r in range(R)],
                               placement=Tile(col=col_offset + c, row=1),
                               obj_types=[L1_W_ty for _ in range(R)])
          for c in range(NC)]

    PF, PF_parts = [], []
    for c in range(NC):
        pf = ObjectFifo(L1_4E_ty, name=f"PF_{c}", depth=1)
        parts = pf.prod().join(offsets=[r * E for r in range(R)],
                               placement=Tile(col=col_offset + c, row=1),
                               obj_types=[L1_E_ty for _ in range(R)])
        PF.append(pf); PF_parts.append(parts)

    FinalParts = ObjectFifo(L1_4E_ty, name="FinalParts", depth=1)
    FP_parts = FinalParts.prod().join(offsets=[c * E for c in range(NC)],
                                      placement=Tile(col=6, row=1),
                                      obj_types=[L1_E_ty for _ in range(NC)])
    Hin = ObjectFifo(L1_E_ty, name="Hin", depth=1)
    FFNfifo = ObjectFifo(L1_E_ty, name="FFNfifo", depth=2)
    Out = ObjectFifo(L1_E_ty, name="Out", depth=2)

    workers = []

    def ahub_body(attn_in, inpff_in, bc_out, gain, zero, add_fn, rms_fn):
        for _ in range_(0xFFFFFFFF):
            a = attn_in.acquire(1); o = bc_out.acquire(1)
            add_fn(a, zero, o, E); attn_in.release(1); bc_out.release(1)
            ip = inpff_in.acquire(1); o = bc_out.acquire(1)
            rms_fn(ip, gain, o, E); inpff_in.release(1); bc_out.release(1)

    workers.append(Worker(ahub_body, [Attn.cons(), Inpff.cons(), Bc.prod(),
                                      gain_buf, mk_zero(), add, rms],
                          placement=Tile(col=1, row=2)))

    def center_body(bc, wf, pp, zero, add_fn, oscat_fn, mono_fn, scatter):
        _dbg_noo = bool(_os.environ.get("DBG_BM_NOO"))
        for _ in range_(0xFFFFFFFF):
            # phase1: O padded-reduce
            b = bc.acquire(1); p = pp.acquire(1)
            add_fn(zero, zero, p, E)
            for j in range_(o_tiles):
                w = wf.acquire(1)
                if not _dbg_noo:
                    oscat_fn(m, index.casts(T.i32(), j) * m + scatter, w, b, p)
                wf.release(1)
            bc.release(1); pp.release(1)
            # phase2: FFN (monolithic, accumulate into down_acc)
            b2 = bc.acquire(1); p2 = pp.acquire(1)
            add_fn(zero, zero, p2, E)                 # zero down_acc
            for j in range_(gu_tiles):
                w = wf.acquire(1)
                mono_fn(m, w, b2, p2)
                wf.release(1)
            bc.release(1); pp.release(1)

    for c in range(NC):
        for r in range(R):
            scatter = (c * R + r) * SL_O
            workers.append(Worker(
                center_body,
                [BcB.cons(), Wf[c][r].cons(), PF_parts[c][r].prod(),
                 mk_zero(), add, oscat, mono, scatter],
                placement=Tile(col=col_offset + c, row=2 + r)))

    def colred_body(pf, fp, zero, red_fn):
        for _ in range_(0xFFFFFFFF):
            f = pf.acquire(1); o = fp.acquire(1)
            red_fn(f, zero, o, E); pf.release(1); fp.release(1)
            f = pf.acquire(1); o = fp.acquire(1)
            red_fn(f, zero, o, E); pf.release(1); fp.release(1)

    for c in range(NC):
        workers.append(Worker(colred_body, [PF[c].cons(), FP_parts[c].prod(), mk_zero(), reduce4],
                              placement=Tile(col=0, row=2 + c)))

    def finalred_body(fp, hin, inpff_out, ffn_out, zero, red_fn):
        for _ in range_(0xFFFFFFFF):
            f = fp.acquire(1); h = hin.acquire(1); o = inpff_out.acquire(1)
            red_fn(f, h, o, E); fp.release(1); hin.release(1); inpff_out.release(1)
            f = fp.acquire(1); o = ffn_out.acquire(1)
            red_fn(f, zero, o, E); fp.release(1); ffn_out.release(1)

    workers.append(Worker(finalred_body, [FinalParts.cons(), Hin.cons(), Inpff.prod(),
                                          FFNfifo.prod(), mk_zero(), reduce4],
                          placement=Tile(col=6, row=2)))

    def resid_body(ffni, inpff_in, out, zero, add_fn):
        _dbg_inpff = bool(_os.environ.get("DBG_BM_INPFF"))
        for _ in range_(0xFFFFFFFF):
            f = ffni.acquire(1); ip = inpff_in.acquire(1); o = out.acquire(1)
            if _dbg_inpff:
                add_fn(ip, zero, o, E)
            else:
                add_fn(f, ip, o, E)
            ffni.release(1); inpff_in.release(1); out.release(1)

    workers.append(Worker(resid_body, [FFNfifo.cons(), Inpff.cons(), Out.prod(), mk_zero(), add],
                          placement=Tile(col=6, row=3)))

    tap = TensorAccessPattern(tensor_dims=(1, E), offset=0,
                              sizes=[1, 1, 1, E], strides=[0, 0, 0, 1])
    wcol = WT_PER_TILE * R * GUD

    def w_tap(c):
        return TensorAccessPattern(tensor_dims=(1, NC * wcol), offset=c * wcol,
                                   sizes=[1, 1, 1, wcol], strides=[0, 0, 0, 1])

    rt = Runtime()
    with rt.sequence(L3_E_ty, L3_W_ty, L3_E_ty, L3_E_ty) as (o, w, attn, hin):
        rt.start(*workers)
        tg = rt.task_group()
        for c in range(NC):
            rt.fill(Wsrc[c].prod(), w, w_tap(c), task_group=tg)
        rt.fill(Attn.prod(), attn, tap, task_group=tg)
        rt.fill(Hin.prod(), hin, tap, task_group=tg)
        rt.drain(Out.cons(), o, tap, task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())
