# SPDX-License-Identifier: Apache-2.0
"""decode_ffn4d — D2.5 DIAGNOSTIC: 4-tile FFN with DIRECT per-tile weight fifos
(NO MemTile split). Isolates whether the lockstep MemTile split is what blocks
compute/DMA overlap in decode_ffn16. 4 tiles (col_offset, rows 2-5), each a
direct shim→tile weight fifo (depth 2 ping-pong); broadcast activation; 4 down
partials → reduce4 → drain. If this overlaps (warm ≈ max(DMA,compute)) while the
split version does not, the split is the culprit.

DMA budget: 4 weight shim fills + 1 activation forward + 1 output drain = 6 shim
S2MM/MM2S (fits). Per tile: 2 S2MM (weight + act) + 1 MM2S (partial).
DBG_FFN4D_COPY=1 → copy body (drain weights, trivial compute) for the DMA baseline.
"""
import numpy as np
from ml_dtypes import bfloat16

import aie.dialects.index as index
from aie.dialects.aie import T
from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2, Tile


def my_decode_ffn4d(dev, embed_dim=2048, hidden_dim=8192, group_size=32,
                    m_input=4, col_offset=2):
    dev_ty = NPU1() if dev == "npu" else NPU2()
    bf = np.dtype[bfloat16]
    u8 = np.dtype[np.uint8]
    E, H, g, m = embed_dim, hidden_dim, group_size, m_input
    R = 4
    NT = 16                                        # slice as if 16-way (matches ffn16)
    Hc16 = H // NT                                 # 512
    gu_tiles = Hc16 // m                           # 128
    dn_tiles = E // m                              # 512
    PACKED = m * E // 2 + m * (E // g) * 2         # 4608
    DN_PACKED = m * Hc16 // 2 + m * (Hc16 // g) * 2  # 1152
    DN_SUB = PACKED // DN_PACKED                   # 4
    dn_elems = dn_tiles // DN_SUB                  # 128
    WT_PER_TILE = gu_tiles + gu_tiles + dn_elems   # 384

    L1_E_ty  = np.ndarray[(E,), bf]
    L1_4E_ty = np.ndarray[(4 * E,), bf]
    L1_W_ty  = np.ndarray[(PACKED,), u8]

    L3_O_ty = np.ndarray[(E,), bf]
    L3_W_ty = np.ndarray[(R * WT_PER_TILE * PACKED,), u8]   # 4 tiles, contiguous
    L3_E_ty = np.ndarray[(E,), bf]
    L3_D_ty = np.ndarray[(64,), bf]

    gate_up = Kernel("layer_fused_gate_up_bf16", "layer_fused_relay.o",
                     [np.int32, np.int32, L1_W_ty, L1_E_ty, np.int32])
    silu = Kernel("layer_fused_silu_mul_static_bf16", "layer_fused_relay.o", [np.int32])
    down = Kernel("layer_fused_down_v2_x4_bf16", "layer_fused_relay.o",
                  [np.int32, np.int32, np.int32, L1_W_ty, L1_E_ty])
    reduce4 = Kernel("layer_fused_reduce4_bf16", "layer_fused_relay.o",
                     [L1_4E_ty, L1_E_ty, L1_E_ty, np.int32])
    add = Kernel("layer_fused_add_bf16", "layer_fused_relay.o",
                 [L1_E_ty, L1_E_ty, L1_E_ty, np.int32])
    import os as _os
    DBG_COPY = bool(_os.environ.get("DBG_FFN4D_COPY"))

    _zn = [0]
    def mk_zero():
        b = Buffer(type=L1_E_ty, initial_value=np.zeros(E, dtype=bfloat16),
                   name=f"ffn4d_zero_{_zn[0]}"); _zn[0] += 1
        return b

    # broadcast activation to the 4 tiles (1 column), MemTile(1,1) forward
    Bsrc = ObjectFifo(L1_E_ty, name="Bsrc", depth=1)
    Bcol = Bsrc.cons().forward(name="Bcol", depth=2, placement=Tile(col=1, row=1))

    # DIRECT per-tile weight fifos (no split): shim → tile, depth 2 ping-pong
    Wd = [ObjectFifo(L1_W_ty, name=f"Wd_{r}", depth=2) for r in range(R)]

    # 4 partials → reduce4 @ MemTile(6,1)
    PF = ObjectFifo(L1_4E_ty, name="PF", depth=1)
    PF_parts = PF.prod().join(
        offsets=[r * E for r in range(R)], placement=Tile(col=6, row=1),
        obj_types=[L1_E_ty for _ in range(R)])
    Cout = ObjectFifo(L1_E_ty, name="Cout", depth=2)

    workers = []

    def ffn_body(wf, bc, pp, gate_up_fn, silu_fn, down_fn):
        for _ in range_(0xFFFFFFFF):
            b = bc.acquire(1)
            for j in range_(gu_tiles):
                w = wf.acquire(1); gate_up_fn(m, index.casts(T.i32(), j)*m, w, b, 0); wf.release(1)
            for j in range_(gu_tiles):
                w = wf.acquire(1); gate_up_fn(m, index.casts(T.i32(), j)*m, w, b, 1); wf.release(1)
            bc.release(1)
            silu_fn(Hc16)
            o = pp.acquire(1)
            for j in range_(dn_elems):
                w = wf.acquire(1)
                down_fn(m, index.casts(T.i32(), j)*DN_SUB*m, DN_SUB, w, o)
                wf.release(1)
            pp.release(1)

    def copy_body(wf, bc, pp, add_fn, zero):
        for _ in range_(0xFFFFFFFF):
            b = bc.acquire(1)
            for _j in range_(2*gu_tiles + dn_elems):
                w = wf.acquire(1); wf.release(1)
            o = pp.acquire(1); add_fn(b, zero, o, E); bc.release(1); pp.release(1)

    for r in range(R):
        if DBG_COPY:
            workers.append(Worker(copy_body,
                [Wd[r].cons(), Bcol.cons(), PF_parts[r].prod(), add, mk_zero()],
                placement=Tile(col=col_offset, row=2 + r)))
        else:
            workers.append(Worker(ffn_body,
                [Wd[r].cons(), Bcol.cons(), PF_parts[r].prod(), gate_up, silu, down],
                placement=Tile(col=col_offset, row=2 + r)))

    def finalred_body(fp, out, zero, red_fn):
        for _ in range_(0xFFFFFFFF):
            p = fp.acquire(1); o = out.acquire(1); red_fn(p, zero, o, E)
            fp.release(1); out.release(1)
    workers.append(Worker(finalred_body, [PF.cons(), Cout.prod(), mk_zero(), reduce4],
                          placement=Tile(col=6, row=2)))

    e_tap = TensorAccessPattern(tensor_dims=(1, E), offset=0,
                                sizes=[1, 1, 1, E], strides=[0, 0, 0, 1])
    wtile = WT_PER_TILE * PACKED

    def w_tap(r):
        return TensorAccessPattern(tensor_dims=(1, R * wtile), offset=r * wtile,
                                   sizes=[1, 1, 1, wtile], strides=[0, 0, 0, 1])

    rt = Runtime()
    with rt.sequence(L3_O_ty, L3_W_ty, L3_E_ty, L3_D_ty, L3_D_ty) as (o, w, x, _d3, _d4):
        rt.start(*workers)
        tg = rt.task_group()
        for r in range(R):
            rt.fill(Wd[r].prod(), w, w_tap(r), task_group=tg)   # DIRECT shim→tile
        rt.fill(Bsrc.prod(), x, e_tap, task_group=tg)
        rt.drain(Cout.cons(), o, e_tap, task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())
