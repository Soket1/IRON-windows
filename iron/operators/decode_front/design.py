# SPDX-License-Identifier: Apache-2.0
"""decode_front — FFLM-style single-column gemv→attention chain.

Three tiles in one central column, one column only (one KV head group):

  tile (col, 2): GEMV + RoPE → Q_int (on-chip ObjectFifo)
  tile (col, 3): score_init + score_rope_q + score_chunk → I_int (on-chip)
  tile (col, 4): value_init + value_accum + value_normalize → O (DDR drain)

Each tile's static buffers (rotated_q, score_running_max/sum) are per-tile.
score_rope_q MUST run on the score tile (copies Q into that tile's static
rotated_q, which score_chunk reads). RoPE rotation (rope.cc TWO_HALVES) runs
on the gemv tile BEFORE Q is handed off — so when score_rope_q copies it into
the score tile's static buffer, the Q is already post-RoPE.
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


def my_decode_front(dev, embed_dim=2048, K_gemv=2048, head_dim=64, group_size=32,
                    attn_group=4, seq_len=32, chunk_size=None,
                    m_input=2, col=2):
    if chunk_size is None:
        chunk_size = seq_len
    dev_ty = NPU1() if dev == "npu" else NPU2()
    bf = np.dtype[bfloat16]
    u8 = np.dtype[np.uint8]

    q_rows = attn_group * head_dim
    q_buf_elems = q_rows + head_dim + 2
    groups = K_gemv // group_size
    packed_tile = m_input * K_gemv // 2 + m_input * groups * 2
    gemv_tiles = q_rows // m_input
    assert q_rows % m_input == 0
    num_chunks = seq_len // chunk_size
    inter_size = chunk_size * attn_group + 2 * attn_group

    L1_A_ty   = np.ndarray[(packed_tile,), u8]
    L1_B_ty   = np.ndarray[(K_gemv,), bf]
    L1_Q_ty   = np.ndarray[(q_buf_elems,), bf]
    L1_LUT_ty = np.ndarray[(head_dim,), bf]
    L1_K_ty   = np.ndarray[(chunk_size * head_dim,), bf]
    L1_I_ty   = np.ndarray[(inter_size,), bf]
    L1_V_ty   = np.ndarray[(chunk_size * head_dim,), bf]
    L1_O_ty   = np.ndarray[(attn_group * head_dim,), bf]

    DTYPE_SIZE = 2
    raw_head = seq_len * head_dim * DTYPE_SIZE
    ahs = int((raw_head + 63) / 64) * 64
    ahs_elems = ahs // DTYPE_SIZE

    L3_W_ty = np.ndarray[(gemv_tiles * packed_tile,), u8]
    L3_X_ty = np.ndarray[(K_gemv,), bf]
    L3_K_ty = np.ndarray[(ahs_elems,), bf]
    L3_V_ty = np.ndarray[(ahs_elems,), bf]
    L3_O_ty = np.ndarray[(attn_group * head_dim,), bf]

    # -------------------------------------------------------------------
    # Kernels
    # -------------------------------------------------------------------
    gemv = Kernel(
        "fused_dequant_matvec_v2_bf16",
        f"fused_dequant_gemv_v2_{K_gemv}k_g{group_size}.o",
        [np.int32, np.int32, L1_A_ty, L1_B_ty, L1_Q_ty],
    )
    rope = Kernel("rope", "rope_th.o", [L1_Q_ty, L1_LUT_ty, L1_Q_ty, np.int32])

    fkv = f"flowkv_{head_dim}d.o"
    score_init  = Kernel("flowkv_score_init_bf16",       fkv, [np.int32])
    score_rope  = Kernel("flowkv_score_rope_q_bf16",     fkv,
                         [L1_Q_ty, np.int32, np.int32])
    score_chunk = Kernel("flowkv_score_chunk_bf16",      fkv,
                         [L1_Q_ty, L1_K_ty, L1_I_ty, np.int32, np.int32, np.int32])
    val_init    = Kernel("flowkv_value_init_bf16",       fkv,
                         [np.int32, np.int32])
    val_accum   = Kernel("flowkv_value_accum_bf16",      fkv,
                         [L1_I_ty, L1_V_ty, np.int32, np.int32, np.int32])
    val_norm    = Kernel("flowkv_value_normalize_bf16",  fkv,
                         [L1_O_ty, np.int32, np.int32])

    # -------------------------------------------------------------------
    # ObjectFifos + passive LUT
    # -------------------------------------------------------------------
    A     = ObjectFifo(L1_A_ty, name="A", depth=2)
    B     = ObjectFifo(L1_B_ty, name="B", depth=1)
    K_fifo = ObjectFifo(L1_K_ty, name="K", depth=2)
    V_fifo = ObjectFifo(L1_V_ty, name="V", depth=2)
    Q_int  = ObjectFifo(L1_Q_ty, name="Q_int", depth=2)
    I_int  = ObjectFifo(L1_I_ty, name="I_int", depth=2)
    O      = ObjectFifo(L1_O_ty, name="O", depth=2)

    rope_lut_data = np.zeros(head_dim, dtype=bfloat16)
    rope_lut_data[0::2] = bfloat16(1.0); rope_lut_data[1::2] = bfloat16(0.0)
    lut = Buffer(type=L1_LUT_ty, initial_value=rope_lut_data, name="rope_lut")

    # -------------------------------------------------------------------
    # Worker 1 — tile (col,2): GEMV + RoPE → Q_int
    # -------------------------------------------------------------------
    def gemv_body(af, bf, qf, lut, gemv_fn, rope_fn):
        for _ in range_(0xFFFFFFFF):
            b = bf.acquire(1)
            q = qf.acquire(1)
            for j in range_(gemv_tiles):
                a = af.acquire(1)
                ro = index.casts(T.i32(), j) * m_input
                gemv_fn(m_input, ro, a, b, q)
                af.release(1)
            rope_fn(q, lut, q, head_dim)
            qf.release(1)
            bf.release(1)

    gemv_worker = Worker(
        gemv_body, [A.cons(), B.cons(), Q_int.prod(), lut, gemv, rope],
        placement=Tile(col=col, row=2),
    )

    # -------------------------------------------------------------------
    # Worker 2 — tile (col,3): score_init + score_rope_q + score_chunk
    # -------------------------------------------------------------------
    def score_body(kf, qf, inf, init_fn, rope_fn, chunk_fn):
        for _ in range_(0xFFFFFFFF):
            init_fn(attn_group)
            q = qf.acquire(1)
            # score_rope_q copies Q into THIS tile's static rotated_q + reads seq_len
            rope_fn(q, attn_group, head_dim)
            for _ in range_(num_chunks):
                k = kf.acquire(1)
                it = inf.acquire(1)
                chunk_fn(q, k, it, attn_group, head_dim, chunk_size)
                kf.release(1)
                inf.release(1)
            qf.release(1)

    score_worker = Worker(
        score_body, [K_fifo.cons(), Q_int.cons(), I_int.prod(),
                     score_init, score_rope, score_chunk],
        placement=Tile(col=col, row=3),
    )

    # -------------------------------------------------------------------
    # Worker 3 — tile (col,4): value_init + value_accum + value_normalize
    # -------------------------------------------------------------------
    def value_body(vf, inf, of, init_fn, accum_fn, norm_fn):
        for _ in range_(0xFFFFFFFF):
            init_fn(attn_group, head_dim)
            for _ in range_(num_chunks):
                it = inf.acquire(1)
                v = vf.acquire(1)
                accum_fn(it, v, attn_group, head_dim, chunk_size)
                inf.release(1)
                vf.release(1)
            o = of.acquire(1)
            norm_fn(o, attn_group, head_dim)
            of.release(1)

    value_worker = Worker(
        value_body, [V_fifo.cons(), I_int.cons(), O.prod(),
                     val_init, val_accum, val_norm],
        placement=Tile(col=col, row=4),
    )

    # -------------------------------------------------------------------
    # Runtime: time-multiplex DMA on one shim column via task_groups
    # -------------------------------------------------------------------
    A_tap = TensorAccessPattern(
        tensor_dims=(1, gemv_tiles * packed_tile),
        offset=0, sizes=[1, 1, 1, gemv_tiles * packed_tile],
        strides=[0, 0, 0, 1],
    )
    K_tap = TensorAccessPattern(
        tensor_dims=(ahs_elems,),
        offset=0, sizes=[1, 1, 1, seq_len * head_dim],
        strides=[0, 0, 0, 1],
    )
    V_tap = TensorAccessPattern(
        tensor_dims=(ahs_elems,),
        offset=0, sizes=[1, 1, 1, seq_len * head_dim],
        strides=[0, 0, 0, 1],
    )

    rt = Runtime()
    with rt.sequence(L3_W_ty, L3_X_ty, L3_K_ty, L3_V_ty, L3_O_ty) as (w, x, k, v, o):
        rt.start(gemv_worker, score_worker, value_worker)
        tg1 = rt.task_group()
        rt.fill(A.prod(), w, A_tap, task_group=tg1)
        rt.fill(B.prod(), x, task_group=tg1)
        rt.finish_task_group(tg1)
        tg2 = rt.task_group()
        rt.fill(K_fifo.prod(), k, K_tap, task_group=tg2)
        rt.fill(V_fifo.prod(), v, V_tap, task_group=tg2)
        rt.finish_task_group(tg2)
        tg3 = rt.task_group()
        rt.drain(O.cons(), o, task_group=tg3, wait=True)
        rt.finish_task_group(tg3)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())
