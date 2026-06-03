# SPDX-License-Identifier: Apache-2.0
"""decode_layer — fused decode layer, increment A: QKV→RoPE→flowkv→join→O_proj.

Builds on the proven 4-column decode_front chain (QKV+RoPE+flowkv on rows 2-4 of
central columns 2-5) and adds the post-attention O_proj:

  rows 2-4 (per col): GEMV+RoPE → score → value  → O_slice[c] (E/cols = 512)
  MemTile join:       4 O_slice[c] → full attn_out (E=2048)
  MemTile forward:    broadcast full attn_out to all O_proj tiles
  row 5 (per col):    O_proj INT4 GEMV (K=E, v2 kernel) → o_out_slice[c] → DDR

attn_group=8 so per-col attn slice = 8*64 = 512, ×4 = 2048 = E (O_proj K=E).
O_proj uses the v2 GEMV kernel (c_out += row_offset, no -8 bias) — same as the
oproj_probe. attn_out gather = ObjectFifo.join (gather_probe-proven). Each tile's
RMS/RoPE statics are per-tile. DMA time-multiplexed via task_group phases.
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


def my_decode_layer(dev, embed_dim=2048, head_dim=64, group_size=32,
                    attn_group=8, seq_len=32, chunk_size=None,
                    m_input=2, num_cols=4, col_offset=2):
    if chunk_size is None:
        chunk_size = seq_len
    dev_ty = NPU1() if dev == "npu" else NPU2()
    bf = np.dtype[bfloat16]; u8 = np.dtype[np.uint8]
    E = embed_dim
    K_gemv = E
    assert col_offset + num_cols <= 8
    assert attn_group * head_dim * num_cols == E, "per-col attn slice must tile E"

    q_rows = attn_group * head_dim                  # 512 (QKV out per col)
    q_buf_elems = q_rows + head_dim + 2
    groups = K_gemv // group_size
    packed_tile = m_input * K_gemv // 2 + m_input * groups * 2
    gemv_tiles = q_rows // m_input
    num_chunks = seq_len // chunk_size
    inter_size = chunk_size * attn_group + 2 * attn_group
    o_slice = E // num_cols                          # 512 (O_proj out per col)
    o_groups = E // group_size
    o_packed = m_input * E // 2 + m_input * o_groups * 2
    o_tiles = o_slice // m_input

    L1_A_ty   = np.ndarray[(packed_tile,), u8]
    L1_B_ty   = np.ndarray[(K_gemv,), bf]
    L1_Q_ty   = np.ndarray[(q_buf_elems,), bf]
    L1_LUT_ty = np.ndarray[(head_dim,), bf]
    L1_K_ty   = np.ndarray[(chunk_size * head_dim,), bf]
    L1_I_ty   = np.ndarray[(inter_size,), bf]
    L1_V_ty   = np.ndarray[(chunk_size * head_dim,), bf]
    L1_O_ty   = np.ndarray[(q_rows,), bf]            # attn out slice (512)
    L1_E_ty   = np.ndarray[(E,), bf]                 # full attn_out
    L1_OW_ty  = np.ndarray[(o_packed,), u8]          # O_proj weight tile (= packed_tile, K=E)
    # O_proj reuses the QKV v2-gemv kernel symbol → its C arg must match L1_Q_ty
    # (578). O_proj writes o_slice=512 rows into it (first 512 used). o_packed
    # == packed_tile and L1_E_ty == L1_B_ty (both K=E=2048), so all 3 args match.
    L1_OC_ty  = L1_Q_ty                               # o_out buffer (write first o_slice)

    DTYPE_SIZE = 2
    ahs = int((seq_len * head_dim * DTYPE_SIZE + 63) / 64) * 64
    ahs_elems = ahs // DTYPE_SIZE

    # FFLM-style ONE weight BO (like FFLM's bo1 = all layer weights together):
    # QKV/A region then O_proj region, contiguous in a single DDR buffer. This
    # keeps the kernel at 5 data BOs (FFLM's MLIR_AIE ABI: bo0-4), with the
    # weight BO holding all GEMV weights. Layout: [W (num_cols*a_per_col) |
    # OW (num_cols*ow_per_col)] — A_src reads at 0, OW_src at the W-region end.
    w_region  = num_cols * gemv_tiles * packed_tile          # all QKV weights
    ow_region = num_cols * o_tiles * o_packed                # all O weights
    ow_base   = w_region                                     # OW offset in the BO
    L3_WT_ty = np.ndarray[(w_region + ow_region,), u8]       # single weight BO
    L3_X_ty  = np.ndarray[(K_gemv,), bf]
    L3_K_ty  = np.ndarray[(num_cols * ahs_elems,), bf]
    L3_V_ty  = np.ndarray[(num_cols * ahs_elems,), bf]
    L3_O_ty  = np.ndarray[(E,), bf]                  # final o_out (E)

    # Kernels (declared once)
    gemv = Kernel("fused_dequant_matvec_v2_bf16",
                  f"fused_dequant_gemv_v2_{K_gemv}k_g{group_size}.o",
                  [np.int32, np.int32, L1_A_ty, L1_B_ty, L1_Q_ty])
    rope = Kernel("rope", "rope_th.o", [L1_Q_ty, L1_LUT_ty, L1_Q_ty, np.int32])
    fkv = f"flowkv_{head_dim}d.o"
    s_init  = Kernel("flowkv_score_init_bf16",      fkv, [np.int32])
    s_rope  = Kernel("flowkv_score_rope_q_bf16",    fkv, [L1_Q_ty, np.int32, np.int32])
    s_chunk = Kernel("flowkv_score_chunk_bf16",     fkv,
                     [L1_Q_ty, L1_K_ty, L1_I_ty, np.int32, np.int32, np.int32])
    v_init  = Kernel("flowkv_value_init_bf16",      fkv, [np.int32, np.int32])
    v_accum = Kernel("flowkv_value_accum_bf16",     fkv,
                     [L1_I_ty, L1_V_ty, np.int32, np.int32, np.int32])
    v_norm  = Kernel("flowkv_value_normalize_bf16", fkv, [L1_O_ty, np.int32, np.int32])
    # O_proj is the SAME v2 GEMV kernel (K=E=K_gemv) as QKV — reuse the `gemv`
    # Kernel object (declaring it twice redefines the symbol). The kernel's
    # buffer args are generic memrefs; OW/L1_E/OC match A/B/Q element types
    # closely enough (uint8 weights, bf16 in, bf16 out) for the same symbol.
    oproj = gemv

    # FFLM-style MemTile weight/input delivery: each stream is ONE shim source
    # → split (weights, fan-out to N tiles) or forward (broadcast B) at a MemTile
    # on a FREE edge column, instead of N per-column shim fills (which exhaust the
    # 16-S2MM shim fabric). Cuts shim S2MM from ~20 to ~5.
    # Source-fifo element = each column's FULL per-stage region; split gives N
    # per-column sub-fifos whose element = one worker-acquire tile. The worker
    # acquires its sub-fifo tile-by-tile; the MemTile streams the column slice.
    # Source-fifo element = ONE tile-row across all columns (num_cols * tile),
    # NOT the whole weight buffer (that would need MBs of MemTile memory). The
    # split fans the N column slices of one tile-row to N sub-fifos; the source
    # streams gemv_tiles/o_tiles rows via the DDR tap. Sub-fifo element = one
    # per-column tile, acquired tile-by-tile in the worker.
    kv_col  = chunk_size * head_dim                     # K/V per column (1 chunk)

    A_full = ObjectFifo(np.ndarray[(num_cols * packed_tile,), u8], name="A_src", depth=2)
    A_f = A_full.cons().split(
        offsets=[c * packed_tile for c in range(num_cols)], placement=Tile(col=0, row=1),
        obj_types=[L1_A_ty for _ in range(num_cols)])
    OW_full = ObjectFifo(np.ndarray[(num_cols * o_packed,), u8], name="OW_src", depth=2)
    OW_f = OW_full.cons().split(
        offsets=[c * o_packed for c in range(num_cols)], placement=Tile(col=1, row=1),
        obj_types=[L1_OW_ty for _ in range(num_cols)])
    K_full = ObjectFifo(np.ndarray[(num_cols * kv_col,), bf], name="K_src", depth=2)
    K_f = K_full.cons().split(
        offsets=[c * kv_col for c in range(num_cols)], placement=Tile(col=6, row=1),
        obj_types=[L1_K_ty for _ in range(num_cols)])
    V_full = ObjectFifo(np.ndarray[(num_cols * kv_col,), bf], name="V_src", depth=2)
    V_f = V_full.cons().split(
        offsets=[c * kv_col for c in range(num_cols)], placement=Tile(col=7, row=1),
        obj_types=[L1_V_ty for _ in range(num_cols)])
    # input vector B: one shim source, broadcast (forward) to all columns
    B_src = ObjectFifo(L1_B_ty, name="B_src", depth=1)
    B_bcast = B_src.cons().forward(name="B_bcast", depth=1, placement=Tile(col=3, row=1))

    Qi  = [ObjectFifo(L1_Q_ty, name=f"Qi_{c}", depth=2) for c in range(num_cols)]
    Ii  = [ObjectFifo(L1_I_ty, name=f"Ii_{c}", depth=2) for c in range(num_cols)]
    OC_f = [ObjectFifo(L1_OC_ty, name=f"OC_{c}", depth=2) for c in range(num_cols)]

    # attn_out join: 4 value-stage O slices → full-E fifo at MemTile col_offset.
    # Then a SEPARATE forward on a DIFFERENT MemTile broadcasts the assembled E
    # vector to the 4 O_proj workers — layer_fused keeps join and broadcast on
    # distinct MemTile columns so neither over-subscribes the 6+6 channel budget.
    # Place join + broadcast MemTiles on the FREE edge columns (1 and 6), NOT on
    # the central compute columns 2-5 — layer_fused does exactly this (join col1,
    # broadcast col6) so MemTiles don't collide with the compute-column tiles.
    # MemTile usage so far: A-split col0, OW-split col1, K-split col6, V-split
    # col7, B-forward col3. Use the remaining free MemTile cols 4,5 for the
    # attn_out join and broadcast (each MemTile is independent; compute tiles on
    # cols 4,5 are on rows 2-5, the MemTile is row 1).
    # join gathers 4 O slices → AttnFull (one ObjectFifoLinkOp). The 4 O_proj
    # workers read AttnFull.cons() directly — auto-broadcast through the MemTile.
    # (A separate .forward() would put AttnFull in TWO link ops, which AIECC
    # rejects: 'objectfifo cannot be in more than one ObjectFifoLinkOp'. The
    # gather_probe proved direct multi-consumer broadcast works.)
    AttnFull = ObjectFifo(L1_E_ty, name="AttnFull", depth=2)
    O_parts = AttnFull.prod().join(
        offsets=[c * q_rows for c in range(num_cols)],
        placement=Tile(col=4, row=1),
        obj_types=[L1_O_ty for _ in range(num_cols)])

    rope_lut_data = np.zeros(head_dim, dtype=bfloat16)
    rope_lut_data[0::2] = bfloat16(1.0); rope_lut_data[1::2] = bfloat16(0.0)
    luts = [Buffer(type=L1_LUT_ty, initial_value=rope_lut_data, name=f"rope_lut_{c}")
            for c in range(num_cols)]

    workers = []
    for c in range(num_cols):
        pc = c + col_offset

        def gemv_body(af, bf_, qf, lut, gemv_fn, rope_fn):
            for _ in range_(0xFFFFFFFF):
                b = bf_.acquire(1); q = qf.acquire(1)
                for j in range_(gemv_tiles):
                    a = af.acquire(1)
                    gemv_fn(m_input, index.casts(T.i32(), j) * m_input, a, b, q)
                    af.release(1)
                rope_fn(q, lut, q, head_dim)
                qf.release(1); bf_.release(1)
        workers.append(Worker(gemv_body,
            [A_f[c].cons(), B_bcast.cons(), Qi[c].prod(), luts[c], gemv, rope],
            placement=Tile(col=pc, row=2)))

        def score_body(kf, qf, inf, init_fn, rope_fn, chunk_fn):
            for _ in range_(0xFFFFFFFF):
                init_fn(attn_group)
                q = qf.acquire(1)
                rope_fn(q, attn_group, head_dim)
                for _ in range_(num_chunks):
                    k = kf.acquire(1); it = inf.acquire(1)
                    chunk_fn(q, k, it, attn_group, head_dim, chunk_size)
                    kf.release(1); inf.release(1)
                qf.release(1)
        workers.append(Worker(score_body,
            [K_f[c].cons(), Qi[c].cons(), Ii[c].prod(), s_init, s_rope, s_chunk],
            placement=Tile(col=pc, row=3)))
        # (K_f[c]/V_f[c] are split sub-fifos; .cons() is their read endpoint)

        def value_body(vf, inf, of, init_fn, accum_fn, norm_fn):
            for _ in range_(0xFFFFFFFF):
                init_fn(attn_group, head_dim)
                for _ in range_(num_chunks):
                    it = inf.acquire(1); v = vf.acquire(1)
                    accum_fn(it, v, attn_group, head_dim, chunk_size)
                    inf.release(1); vf.release(1)
                o = of.acquire(1)
                norm_fn(o, attn_group, head_dim)
                of.release(1)
        workers.append(Worker(value_body,
            [V_f[c].cons(), Ii[c].cons(), O_parts[c].prod(),
             v_init, v_accum, v_norm],
            placement=Tile(col=pc, row=4)))

        def oproj_body(ow, ab, oc, oproj_fn):
            for _ in range_(0xFFFFFFFF):
                a_full = ab.acquire(1)            # full attn_out (broadcast)
                o = oc.acquire(1)
                for j in range_(o_tiles):
                    w = ow.acquire(1)
                    oproj_fn(m_input, index.casts(T.i32(), j) * m_input,
                             w, a_full, o)
                    ow.release(1)
                ab.release(1); oc.release(1)
        workers.append(Worker(oproj_body,
            [OW_f[c].cons(), AttnFull.cons(), OC_f[c].prod(), oproj],
            placement=Tile(col=pc, row=5)))

    # Weight/input source fills use inline_ops BD chains (layer_fused mechanism),
    # NOT a single multi-dim rt.fill tap — a single tap cannot express the per-
    # tile column-major split streaming within the AIE DMA-BD structural limits
    # (innermost dim <=1023, OUTER dims <=64, 4-byte-multiple transfers). The
    # shim_dma_bd tap factors the per-column weight region into 4 dims, all sized
    # to fit. Sizes are in ELEMENTS (uint8 -> bytes, bf16 -> bytes//2).
    from aie.dialects.aie import EndOp
    from aie.dialects.aiex import (
        dma_configure_task_for, bds, shim_dma_bd, dma_start_task, dma_await_task,
    )

    def _factor4(n):
        """Factor a contiguous region of n elements into 4 row-major dims
        [d0,d1,d2,d3] with d0..d2 <= 64 (outer BD wrap) and d3 <= 1023 (inner),
        product == n. Mirrors layer_fused's [16,64,18,32] factorization. Returns
        (sizes, strides) for a row-major contiguous blob."""
        # greedily peel inner dim <=1023, then up to 3 outer dims <=64
        dims = []
        rem = n
        # inner d3: largest divisor <=1023 (prefer 4-byte-friendly)
        for d in range(min(rem, 1020), 0, -1):
            if rem % d == 0:
                dims.append(d); rem //= d; break
        # outer dims <=64
        while rem > 1 and len(dims) < 4:
            placed = False
            for d in range(min(rem, 64), 1, -1):
                if rem % d == 0:
                    dims.append(d); rem //= d; placed = True; break
            if not placed:
                break
        if rem != 1:
            dims.append(rem)
        dims = (dims + [1, 1, 1, 1])[:4]
        sizes = list(reversed(dims))              # [d0..d3], d3 inner
        strides, acc = [0, 0, 0, 0], 1
        for i in range(3, -1, -1):
            strides[i] = acc; acc *= sizes[i]
        return sizes, strides

    a_per_col  = gemv_tiles * packed_tile          # A weights/col (bytes, u8)
    ow_per_col = o_tiles * o_packed                 # O weights/col (bytes, u8)
    a_sz,  a_st  = _factor4(a_per_col)
    ow_sz, ow_st = _factor4(ow_per_col)
    kv_sz, kv_st = _factor4(kv_col)                 # bf16 chunk/col
    xn, xf = (K_gemv // 64, 64) if K_gemv % 64 == 0 else (K_gemv, 1)
    x_tap  = TensorAccessPattern(tensor_dims=(1, K_gemv), offset=0,
        sizes=[1, 1, xf, xn], strides=[0, 0, xn, 1])

    def oc_tap(c):
        return TensorAccessPattern(tensor_dims=(1, E), offset=c * o_slice,
            sizes=[1, 1, o_tiles, m_input], strides=[0, 0, m_input, 1])

    rt = Runtime()
    with rt.sequence(L3_WT_ty, L3_X_ty, L3_K_ty, L3_V_ty, L3_O_ty) as \
            (wt, x, k, v, o):
        rt.start(*workers)

        # Weight/chunk sources filled via inline_ops BD chains. A and OW weights
        # come from the SAME weight BO (wt) at offsets 0 and ow_base — FFLM packs
        # all layer weights into one BO. Each source: num_cols BDs (per column).
        def fill_chain(wt_rt, k_rt, v_rt):
            wt_op = wt_rt.op
            streams = [
                ("A_src",  wt_op,    a_per_col,  0,       a_sz,  a_st),
                ("OW_src", wt_op,    ow_per_col, ow_base, ow_sz, ow_st),
                ("K_src",  k_rt.op,  ahs_elems,  0,       kv_sz, kv_st),
                ("V_src",  v_rt.op,  ahs_elems,  0,       kv_sz, kv_st),
            ]
            for name, op, col_stride, base, sz, st in streams:
                task = dma_configure_task_for(name, issue_token=True)
                with bds(task) as bd:
                    for c in range(num_cols):
                        with bd[c]:
                            shim_dma_bd(op, offset=base + c * col_stride,
                                        sizes=sz, strides=st)
                            EndOp()
                dma_start_task(task)
                dma_await_task(task)
        # layer_fused pattern: a DUMMY rt.fill (simple 1D tap) registers each
        # source's producer endpoint + its shim S2MM channel; the inline_ops BD
        # chain then carries the REAL factored transfer on that same channel.
        def _dummy(n):
            return TensorAccessPattern(tensor_dims=(1, n), offset=0,
                sizes=[1, 1, 1, min(n, 1020)], strides=[0, 0, 0, 1])
        tg_w = rt.task_group()
        rt.fill(A_full.prod(),  wt, _dummy(num_cols * a_per_col),  task_group=tg_w)
        rt.fill(OW_full.prod(), wt, _dummy(num_cols * ow_per_col), task_group=tg_w)
        rt.fill(K_full.prod(),  k,  _dummy(num_cols * ahs_elems),  task_group=tg_w)
        rt.fill(V_full.prod(),  v,  _dummy(num_cols * ahs_elems),  task_group=tg_w)
        rt.finish_task_group(tg_w)
        rt.inline_ops(fill_chain, [wt, k, v])

        # B vector + o_out drains in a task_group
        tg = rt.task_group()
        rt.fill(B_src.prod(), x, x_tap, task_group=tg)
        for c in range(num_cols):
            rt.drain(OC_f[c].cons(), o, oc_tap(c), task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())
