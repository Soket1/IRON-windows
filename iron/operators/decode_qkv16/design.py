# SPDX-License-Identifier: Apache-2.0
"""decode_qkv16 — D2.3 block: 16-tile parallel QKV GEMV + 2-level concat-join.

All 16 tiles (cols 2-5 × rows 2-5) compute an output-slice of the QKV projection
(3072 outputs split 16-way = 192/tile), reading the SAME x[E] (broadcast). The
16 slices are CONCATENATED (not reduced) 2-level → QKV[3072]:
  - per column: 4 tile-slices (192) join → 768 @ MemTile(c,1) → copy-relay (col 1).
  - final: 4 column-chunks (768) join → 3072 @ MemTile(6,1) → copy-relay (col 6).
Mirrors decode_ffn16 (broadcast + weight-split + 2-level gather) but the gather is
a concat-JOIN (offsets) with copy relays instead of a reduce. Validates the
concat-join half of the full-layer assembly. v2 GEMV (no -8 bias), K=E.
See dev_notes/track_a_build/D2_design_spike.md.
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


def my_decode_qkv16(dev, embed_dim=2048, qkv_dim=3072, group_size=32,
                    m_input=4, num_cols=4, col_offset=2):
    dev_ty = NPU1() if dev == "npu" else NPU2()
    bf = np.dtype[bfloat16]
    u8 = np.dtype[np.uint8]
    E, QD, g, m = embed_dim, qkv_dim, group_size, m_input
    NC, R = num_cols, 4
    NT = NC * R                                   # 16

    SL = QD // NT                                 # 192 (out slice / tile)
    assert SL % m == 0
    qkv_tiles = SL // m                           # 48
    COL = R * SL                                  # 768 (per-column concat)
    PACKED = m * E // 2 + m * (E // g) * 2        # 4608 (K=E)
    WT_PER_TILE = qkv_tiles                       # 48

    L1_E_ty   = np.ndarray[(E,), bf]              # x activation (broadcast)
    L1_S_ty   = np.ndarray[(SL,), bf]             # tile output slice
    L1_COL_ty = np.ndarray[(COL,), bf]            # per-column concat
    L1_OUT_ty = np.ndarray[(QD,), bf]             # full QKV
    L1_W_ty   = np.ndarray[(PACKED,), u8]
    L1_W4_ty  = np.ndarray[(R * PACKED,), u8]

    L3_O_ty = np.ndarray[(QD,), bf]
    L3_W_ty = np.ndarray[(NC * WT_PER_TILE * R * PACKED,), u8]
    L3_E_ty = np.ndarray[(E,), bf]
    L3_D_ty = np.ndarray[(64,), bf]

    gemv = Kernel("fused_dequant_matvec_v2_bf16",
                  f"fused_dequant_gemv_v2_{E}k_g{g}.o",
                  [np.int32, np.int32, L1_W_ty, L1_E_ty, L1_S_ty])
    # ONE add symbol only (declaring it twice → AIECC redefinition). Column relay
    # copies the 768 concat; the final 3072 join-target is drained directly.
    cp_col = Kernel("layer_fused_add_bf16", "layer_fused_relay.o",
                    [L1_COL_ty, L1_COL_ty, L1_COL_ty, np.int32])

    _zn = [0]
    def mk_zero(ty, n):
        b = Buffer(type=ty, initial_value=np.zeros(n, dtype=bfloat16),
                   name=f"qkv16_zero_{_zn[0]}")
        _zn[0] += 1
        return b

    # broadcast x per column (skeleton-proven)
    Bsrc = [ObjectFifo(L1_E_ty, name=f"Bsrc_{c}", depth=1) for c in range(NC)]
    Bcol = [Bsrc[c].cons().forward(name=f"Bcol_{c}", depth=2,
                                   placement=Tile(col=col_offset + c, row=1))
            for c in range(NC)]

    # weights: per-column interleaved stream → split (North) to 4 per-tile fifos
    Wsrc = [ObjectFifo(L1_W4_ty, name=f"Wsrc_{c}", depth=2) for c in range(NC)]
    Wf = [Wsrc[c].cons().split(
            offsets=[r * PACKED for r in range(R)],
            placement=Tile(col=col_offset + c, row=1),
            obj_types=[L1_W_ty for _ in range(R)])
          for c in range(NC)]

    # L1 concat-join: per column, 4 slices (192) → 768 @ MemTile(c,1)
    QF, QF_parts = [], []
    for c in range(NC):
        qf = ObjectFifo(L1_COL_ty, name=f"QF_{c}", depth=2)
        parts = qf.prod().join(
            offsets=[r * SL for r in range(R)],
            placement=Tile(col=col_offset + c, row=1),
            obj_types=[L1_S_ty for _ in range(R)])
        QF.append(qf); QF_parts.append(parts)

    # L2 concat-join: 4 column-chunks (768) → 3072 @ MemTile(6,1), drained directly
    QKVfull = ObjectFifo(L1_OUT_ty, name="QKVfull", depth=2)
    FP_parts = QKVfull.prod().join(
        offsets=[c * COL for c in range(NC)],
        placement=Tile(col=6, row=1),
        obj_types=[L1_COL_ty for _ in range(NC)])

    workers = []

    def qkv_body(wf, bc, pp, gemv_fn):
        for _ in range_(0xFFFFFFFF):
            x = bc.acquire(1)
            o = pp.acquire(1)
            for j in range_(qkv_tiles):
                w = wf.acquire(1)
                gemv_fn(m, index.casts(T.i32(), j) * m, w, x, o)
                wf.release(1)
            bc.release(1)
            pp.release(1)

    for c in range(NC):
        for r in range(R):
            workers.append(Worker(
                qkv_body,
                [Wf[c][r].cons(), Bcol[c].cons(), QF_parts[c][r].prod(), gemv],
                placement=Tile(col=col_offset + c, row=2 + r)))

    # column copy-relay: QF_c (768) → final-join slot (768), on col 1
    def colcp_body(qf, fp, zero, cp_fn):
        for _ in range_(0xFFFFFFFF):
            a = qf.acquire(1); o = fp.acquire(1)
            cp_fn(a, zero, o, COL)
            qf.release(1); fp.release(1)

    for c in range(NC):
        workers.append(Worker(
            colcp_body,
            [QF[c].cons(), FP_parts[c].prod(), mk_zero(L1_COL_ty, COL), cp_col],
            placement=Tile(col=1, row=2 + c)))

    e_tap = TensorAccessPattern(tensor_dims=(1, E), offset=0,
                                sizes=[1, 1, 1, E], strides=[0, 0, 0, 1])
    out_tap = TensorAccessPattern(tensor_dims=(1, QD), offset=0,
                                  sizes=[1, 1, 1, QD], strides=[0, 0, 0, 1])
    wcol = WT_PER_TILE * R * PACKED

    def w_tap(c):
        return TensorAccessPattern(
            tensor_dims=(1, NC * wcol), offset=c * wcol,
            sizes=[1, 1, 1, wcol], strides=[0, 0, 0, 1])

    rt = Runtime()
    # 5-BO: bo0=out(QD), bo1=W, bo2=x(E), bo3/4=dummy
    with rt.sequence(L3_O_ty, L3_W_ty, L3_E_ty, L3_D_ty, L3_D_ty) as (o, w, x, _d3, _d4):
        rt.start(*workers)
        tg = rt.task_group()
        for c in range(NC):
            rt.fill(Wsrc[c].prod(), w, w_tap(c), task_group=tg)
        for c in range(NC):
            rt.fill(Bsrc[c].prod(), x, e_tap, task_group=tg)
        rt.drain(QKVfull.cons(), o, out_tap, task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())
