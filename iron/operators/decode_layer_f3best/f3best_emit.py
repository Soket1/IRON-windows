#!/usr/bin/env python3
"""Parametric-ish raw-AIE emitter for the F3-best O-fold decode layer (P6.4d-1).

Extracted VERBATIM from build_ofold8_f3best.py (dims block + MLIR generation) so
emit_mlir() returns byte-identical MLIR to the build harness. Generation runs at
import (1B-fixed); OFold8F3BestEmitter validates the requested shape and is the
seam an external build driver / runtime wires against. True dim-driven generation
(the f-string templates carry output indentation) is post-P6.4d work.
"""
import numpy as np
from ml_dtypes import bfloat16

# --- dims (Llama-3.2-1B), verbatim from build_ofold8_f3best.py ---
NH, E, G, M, HD, AG = 8, 2048, 32, 4, 64, 4
H8 = 1024
GROUPS = E // G
PACKED = M * E // 2 + M * GROUPS * 2
GEMV_T, GU_T = 256 // M, H8 // M
OPROJ_T = 256 // M                                  # Wo per-head rows = 256/M = 64 (O-fold phase2, m=4)
KV_T = 64 // M                                      # K/V per-head rows = 64/M = 16 (one KV head: head_dim=64)
KV_M = KV_T * M                                     # K/V outputs per head = 64 bf16
DN_SUB = 2
DN_T = (E // M) // DN_SUB
# MEASUREMENT KNOB (default 1 = production, byte-identical MLIR). Divides only the
# FFN part of the weight stream: the three FFN loop bounds and WT_BYTES shrink
# together while every phase boundary, relay hop and lock protocol stays exactly
# as it is. Sweeping it separates a FIXED per-boundary cost (the fitted intercept)
# from raw stream bandwidth (the slope) -- the two explanations for the layer floor
# sitting above its byte budget. The layer computes a WRONG answer under it, so it
# is only ever run together with the compute stubs, as a measurement.
_FFN_DIV = int(__import__("os").environ.get('F3BEST_FFN_DIV', '1'))
assert GU_T % _FFN_DIV == 0 and DN_T % _FFN_DIV == 0, 'F3BEST_FFN_DIV must divide 256'
GU_T //= _FFN_DIV
DN_T //= _FFN_DIV
assert GU_T == DN_T, 'emitter shares one %cF loop bound for gate/up/down'
WT_TILES = GEMV_T + 2 * KV_T + OPROJ_T + 2 * GU_T + DN_T  # 64+32+64+512+256 = 928 (Wq|Wk|Wv|Wo|gate|up|down)
WT_BYTES = WT_TILES * PACKED
O_TILES = E // M
WO_BYTES = O_TILES * PACKED                         # full Wo BO = UNUSED arg3 placeholder (O now in A)
XB = E + 256 + 16
QI = 256
SEQ = 256; KVN = SEQ * HD                           # 16384 per head (f3best SEQ=256)
IT_SZ = SEQ * AG + 2 * AG                           # packed score FIFO: scores(SEQ*AG)+corr(AG)+denom(AG)
# MULTI-CHUNK attention: chunk_size=256 single-chunk corrupts the 4th q-head (SEQ-sweep root-cause);
# chunk<=128 is clean. Loop NCHUNK chunks over CHUNK-sized K/V/It buffers (also shrinks L1 32KB->16KB).
CHUNK = 128
assert SEQ % CHUNK == 0
NCHUNK = SEQ // CHUNK
KCH = CHUNK * HD                                    # per-chunk K/V buffer elems
ITC = CHUNK * AG + 2 * AG                           # per-chunk packed score FIFO
POS = 5; INV = np.float32(0.125); L2E = np.float32(1.4453125)

# P6 MemTile weight relay: which center columns route weights shim->MemTile(col,1)->core
# instead of shim->core direct. "" = off (baseline), "0" = col2 only, "0,1,2,3" = all four.
# Column c owns MemTile ['jA','jB','oJ','oK'][c] at (c+2, 1) and feeds centers t{c} (row2,
# via S2MM4/MM2S1) and t{c+4} (row3, via S2MM5/MM2S2). Shim assignment is UNCHANGED --
# @A{h} stays on sh{h} because every shim MM2S1 is already taken by X/R/K/V.
import os as _os
_rc = _os.environ.get('F3BEST_MT_RELAY', '').strip()
RELAY_COLS = sorted({int(x) for x in _rc.split(',') if x.strip() != ''}) if _rc else []
assert all(0 <= c <= 3 for c in RELAY_COLS), 'F3BEST_MT_RELAY entries must be 0..3'
# Slots per relay path = the MemTile lookahead depth. depth=2 is bare ping-pong (no
# lookahead, just an extra store-and-forward hop); the whole point of P6 is a DEEP
# queue so the core pulls from SRAM while DDR refills behind it. FFLM budgets ~128KB
# per column for this. Cost per relay MemTile: 2*D slots x 4608B buffer, 4*D locks
# (cap 64), 4*D BDs (cap 48) on top of the join's 8 locks / 8 BDs.
RELAY_DEPTH = int(_os.environ.get('F3BEST_MT_DEPTH', '2'))
assert 2 <= RELAY_DEPTH <= 10, 'F3BEST_MT_DEPTH must be 2..10 (4*D locks <= 64, 4*D+8 BDs <= 48)'
# #138: MemTile decoupling on edge columns. Adds D-slot weight relay on edge
# MemTiles (Klo/Khi/Vlo/Vhi) WITHOUT removing K/V split. Weight relay uses
# S2MM4 (fill) and MM2S4 (drain) — no channel conflict with K/V split
# (S2MM0, MM2S0-3). Locks 8+ for weight relay (split_memtile uses 0-7).
DECOUPLE = _os.environ.get('F3BEST_MT_DECOUPLE', '').strip() != ''
# #139b: remove relay (rl) bottleneck. Connect jA/jB directly to mx via packet_flow
# instead of jA+jB→rl→mx. mx S2MM1 receives 3 packet sources: jA(pkt 0), nm(pkt 1), jB(pkt 2).
RL_FIX = _os.environ.get('F3BEST_RL_FIX', '').strip() != ''
# #144: triple B-buffer on center tiles. B0(x_bundle), B1(attn_out), B2(ffn_in)
# instead of single shared B. DMA pre-fills B1/B2 while core in earlier phases.
TRIPLE_B = _os.environ.get('F3BEST_TRIPLE_B', '1').strip() != ''
# #135: ppbase-density hand-asm GEMV kernel (1.5 bundles/vmac vs current 3.5)
HANDASM_RR = _os.environ.get('F3BEST_HANDASM_RR', '').strip() != ''

# --- MLIR generation, verbatim from build_ofold8_f3best.py (returns MLIRTXT) ---
CENTER_COLS = [(2, 2), (3, 2), (4, 2), (5, 2), (2, 3), (3, 3), (4, 3), (5, 3)]
SCORE_COLS = [(0, 2), (1, 2), (6, 2), (7, 2), (0, 3), (1, 3), (6, 3), (7, 3)]
VALUE_COLS = [(0, 4), (1, 4), (6, 4), (7, 4), (0, 5), (1, 5), (6, 5), (7, 5)]


def center(h):
    p = f"c{h}"
    if TRIPLE_B:
        return _center_triple_b(h, p)
    return _center_single_b(h, p)


def _center_triple_b(h, p):
    """Triple B-buffer: B0(x_bundle), B1(attn_out), B2(ffn_in). DMA pre-fills B1/B2 while core in earlier phases."""
    return f"""
    %{p}_A0 = aie.buffer(%t{h}) {{sym_name = "{p}_A0"}} : memref<4608xi8>
    %{p}_A1 = aie.buffer(%t{h}) {{sym_name = "{p}_A1"}} : memref<4608xi8>
    %{p}_B0 = aie.buffer(%t{h}) {{sym_name = "{p}_B0"}} : memref<2320xbf16>
    %{p}_B1 = aie.buffer(%t{h}) {{sym_name = "{p}_B1"}} : memref<2320xbf16>
    %{p}_B2 = aie.buffer(%t{h}) {{sym_name = "{p}_B2"}} : memref<2320xbf16>
    %{p}_Q = aie.buffer(%t{h}) {{sym_name = "{p}_Q"}} : memref<322xbf16>
    %{p}_K = aie.buffer(%t{h}) {{sym_name = "{p}_K"}} : memref<130xbf16>
    %{p}_V = aie.buffer(%t{h}) {{sym_name = "{p}_V"}} : memref<130xbf16>
    %{p}_O = aie.buffer(%t{h}) {{sym_name = "{p}_O"}} : memref<2048xbf16>
    %{p}_P = aie.buffer(%t{h}) {{sym_name = "{p}_P"}} : memref<2320xbf16>
    %{p}_A0p = aie.lock(%t{h}, 0) {{init = 1 : i32, sym_name = "{p}_A0p"}}
    %{p}_A0c = aie.lock(%t{h}, 1) {{init = 0 : i32, sym_name = "{p}_A0c"}}
    %{p}_B0p = aie.lock(%t{h}, 2) {{init = 1 : i32, sym_name = "{p}_B0p"}}
    %{p}_B0c = aie.lock(%t{h}, 3) {{init = 0 : i32, sym_name = "{p}_B0c"}}
    %{p}_Qp = aie.lock(%t{h}, 4) {{init = 1 : i32, sym_name = "{p}_Qp"}}
    %{p}_Qc = aie.lock(%t{h}, 5) {{init = 0 : i32, sym_name = "{p}_Qc"}}
    %{p}_Op = aie.lock(%t{h}, 6) {{init = 1 : i32, sym_name = "{p}_Op"}}
    %{p}_Oc = aie.lock(%t{h}, 7) {{init = 0 : i32, sym_name = "{p}_Oc"}}
    %{p}_Pp = aie.lock(%t{h}, 8) {{init = 1 : i32, sym_name = "{p}_Pp"}}
    %{p}_Pc = aie.lock(%t{h}, 9) {{init = 0 : i32, sym_name = "{p}_Pc"}}
    %{p}_A1p = aie.lock(%t{h}, 10) {{init = 1 : i32, sym_name = "{p}_A1p"}}
    %{p}_A1c = aie.lock(%t{h}, 11) {{init = 0 : i32, sym_name = "{p}_A1c"}}
    %{p}_B1p = aie.lock(%t{h}, 12) {{init = 1 : i32, sym_name = "{p}_B1p"}}
    %{p}_B1c = aie.lock(%t{h}, 13) {{init = 0 : i32, sym_name = "{p}_B1c"}}
    %{p}_Kp = aie.lock(%t{h}, 16) {{init = 1 : i32, sym_name = "{p}_Kp"}}
    %{p}_Kc = aie.lock(%t{h}, 17) {{init = 0 : i32, sym_name = "{p}_Kc"}}
    %{p}_Vp = aie.lock(%t{h}, 18) {{init = 1 : i32, sym_name = "{p}_Vp"}}
    %{p}_Vc = aie.lock(%t{h}, 19) {{init = 0 : i32, sym_name = "{p}_Vc"}}
    %{p}_gate = aie.buffer(%t{h}) {{sym_name = "{p}_gate"}} : memref<1024xbf16>
    %{p}_up   = aie.buffer(%t{h}) {{sym_name = "{p}_up"}}   : memref<1024xbf16>
    %{p}_silu = aie.buffer(%t{h}) {{sym_name = "{p}_silu"}} : memref<1024xbf16>
    %{p}_uni_partial = aie.buffer(%t{h}) {{sym_name = "{p}_uni_partial"}} : memref<128xi8>
    %core{h} = aie.core(%t{h}) {{
      %c0 = arith.constant 0 : index
      %cN = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %cKV = arith.constant {KV_T*2} : index
      %cF = arith.constant {GU_T} : index
      %c4 = arith.constant 4 : i32
      %c8 = arith.constant 8 : i32
      %qr = arith.constant 256 : i32
      %kr = arith.constant 64 : i32
      %hc = arith.constant 1024 : i32
      scf.for %tok = %c0 to %cN step %c1 {{
        func.call @_ha_noop() : () -> ()
        // ---- phase 1: Q-GEMV + rope + K-GEMV + K-RoPE + V-GEMV (B0 = x_bundle, nchunk=8) ----
        aie.use_lock(%{p}_B0c, AcquireGreaterEqual, 1)
        aie.use_lock(%{p}_Qp, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %c64 step %c2 {{
          aie.use_lock(%{p}_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_q(%ji0, %{p}_A0, %{p}_B0, %{p}_uni_partial, %c8, %{p}_Q) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<322xbf16>) -> ()
          aie.use_lock(%{p}_A0p, Release, 1)
          aie.use_lock(%{p}_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_q(%ji1, %{p}_A1, %{p}_B0, %{p}_uni_partial, %c8, %{p}_Q) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<322xbf16>) -> ()
          aie.use_lock(%{p}_A1p, Release, 1)
        }}
        func.call @rope_bundled(%{p}_Q, %{p}_B0, %{p}_Q, %qr) : (memref<322xbf16>, memref<2320xbf16>, memref<322xbf16>, i32) -> ()
        aie.use_lock(%{p}_Qc, Release, 1)
        // ---- phase 1b: K-GEMV + K-RoPE (B0 = x_bundle, output -> K_buf) ----
        aie.use_lock(%{p}_Kp, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %cKV step %c2 {{
          aie.use_lock(%{p}_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_kv(%ji0, %{p}_A0, %{p}_B0, %{p}_uni_partial, %c8, %{p}_K) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<130xbf16>) -> ()
          aie.use_lock(%{p}_A0p, Release, 1)
          aie.use_lock(%{p}_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_kv(%ji1, %{p}_A1, %{p}_B0, %{p}_uni_partial, %c8, %{p}_K) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<130xbf16>) -> ()
          aie.use_lock(%{p}_A1p, Release, 1)
        }}
        func.call @rope_kv_bundled(%{p}_K, %{p}_B0, %{p}_K, %kr) : (memref<130xbf16>, memref<2320xbf16>, memref<130xbf16>, i32) -> ()
        aie.use_lock(%{p}_Kc, Release, 1)
        // ---- phase 1c: V-GEMV (B0 = x_bundle, output -> V_buf) ----
        aie.use_lock(%{p}_Vp, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %cKV step %c2 {{
          aie.use_lock(%{p}_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_kv(%ji0, %{p}_A0, %{p}_B0, %{p}_uni_partial, %c8, %{p}_V) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<130xbf16>) -> ()
          aie.use_lock(%{p}_A0p, Release, 1)
          aie.use_lock(%{p}_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_kv(%ji1, %{p}_A1, %{p}_B0, %{p}_uni_partial, %c8, %{p}_V) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<130xbf16>) -> ()
          aie.use_lock(%{p}_A1p, Release, 1)
        }}
        aie.use_lock(%{p}_Vc, Release, 1)
        aie.use_lock(%{p}_B0p, Release, 1)
        // ---- phase 2: O-proj (B1 = attn_out, nchunk=8) ----
        aie.use_lock(%{p}_B1c, AcquireGreaterEqual, 1)
        aie.use_lock(%{p}_Op, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %c64 step %c2 {{
          aie.use_lock(%{p}_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_o(%ji0, %{p}_A0, %{p}_B1, %{p}_uni_partial, %c8, %{p}_O) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<2048xbf16>) -> ()
          aie.use_lock(%{p}_A0p, Release, 1)
          aie.use_lock(%{p}_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_o(%ji1, %{p}_A1, %{p}_B1, %{p}_uni_partial, %c8, %{p}_O) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<2048xbf16>) -> ()
          aie.use_lock(%{p}_A1p, Release, 1)
        }}
        aie.use_lock(%{p}_B1p, Release, 1)
        aie.use_lock(%{p}_Oc, Release, 1)
        // ---- phase 3: FFN (B2 = ffn_in) ----
        // 3a: gate GEMV (nchunk=8, output -> gate_buf)
        aie.use_lock(%{p}_B2c, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %cF step %c2 {{
          aie.use_lock(%{p}_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_g(%ji0, %{p}_A0, %{p}_B2, %{p}_uni_partial, %c8, %{p}_gate) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
          aie.use_lock(%{p}_A0p, Release, 1)
          aie.use_lock(%{p}_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_g(%ji1, %{p}_A1, %{p}_B2, %{p}_uni_partial, %c8, %{p}_gate) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
          aie.use_lock(%{p}_A1p, Release, 1)
        }}
        // 3b: up GEMV (nchunk=8, output -> up_buf)
        scf.for %j = %c0 to %cF step %c2 {{
          aie.use_lock(%{p}_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_g(%ji0, %{p}_A0, %{p}_B2, %{p}_uni_partial, %c8, %{p}_up) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
          aie.use_lock(%{p}_A0p, Release, 1)
          aie.use_lock(%{p}_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_g(%ji1, %{p}_A1, %{p}_B2, %{p}_uni_partial, %c8, %{p}_up) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
          aie.use_lock(%{p}_A1p, Release, 1)
        }}
        aie.use_lock(%{p}_B2p, Release, 1)
        // 3c: SiLU(gate) * up -> silu_buf (explicit pointers)
        func.call @layer_fused_silu_mul_explicit_bf16(%{p}_gate, %{p}_up, %{p}_silu, %hc) : (memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>, i32) -> ()
        // 3d: down GEMV (nchunk=4, activation = silu_buf)
        aie.use_lock(%{p}_Pp, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %cF step %c2 {{
          aie.use_lock(%{p}_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_d(%ji0, %{p}_A0, %{p}_silu, %{p}_uni_partial, %c4, %{p}_P) : (i32, memref<4608xi8>, memref<1024xbf16>, memref<128xi8>, i32, memref<2320xbf16>) -> ()
          aie.use_lock(%{p}_A0p, Release, 1)
          aie.use_lock(%{p}_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_d(%ji1, %{p}_A1, %{p}_silu, %{p}_uni_partial, %c4, %{p}_P) : (i32, memref<4608xi8>, memref<1024xbf16>, memref<128xi8>, i32, memref<2320xbf16>) -> ()
          aie.use_lock(%{p}_A1p, Release, 1)
        }}
        aie.use_lock(%{p}_Pc, Release, 1)
      }}
      aie.end
    }}
    %mem_t{h} = aie.mem(%t{h}) {{
      %s0 = aie.dma_start(S2MM, 0, ^a0{h}, ^bs{h})
    ^a0{h}:
      aie.use_lock(%{p}_A0p, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_A0 : memref<4608xi8>, 0, 4608)
      aie.use_lock(%{p}_A0c, Release, 1)
      aie.next_bd ^a1{h}
    ^a1{h}:
      aie.use_lock(%{p}_A1p, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_A1 : memref<4608xi8>, 0, 4608)
      aie.use_lock(%{p}_A1c, Release, 1)
      aie.next_bd ^a0{h}
    ^bs{h}:
      %s1 = aie.dma_start(S2MM, 1, ^b0{h}, ^m0{h})
    ^b0{h}:
      aie.use_lock(%{p}_B0p, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_B0 : memref<2320xbf16>, 0, 2320)
      aie.use_lock(%{p}_B0c, Release, 1)
      aie.next_bd ^b1{h}
    ^b1{h}:
      aie.use_lock(%{p}_B1p, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_B1 : memref<2320xbf16>, 0, 2320)
      aie.use_lock(%{p}_B1c, Release, 1)
      aie.next_bd ^b2{h}
    ^b2{h}:
      aie.use_lock(%{p}_B2p, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_B2 : memref<2320xbf16>, 0, 2320)
      aie.use_lock(%{p}_B2c, Release, 1)
      aie.next_bd ^b0{h}
    ^m0{h}:
      %m0 = aie.dma_start(MM2S, 0, ^pf{h}, ^qo{h})
    ^pf{h}:
      aie.use_lock(%{p}_Pc, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_P : memref<2320xbf16>, 0, 2048)
      aie.use_lock(%{p}_Pp, Release, 1)
      aie.next_bd ^kf{h}
    ^kf{h}:
      aie.use_lock(%{p}_Kc, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_K : memref<130xbf16>, 0, 64)
      aie.use_lock(%{p}_Kp, Release, 1)
      aie.next_bd ^vf{h}
    ^vf{h}:
      aie.use_lock(%{p}_Vc, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_V : memref<130xbf16>, 0, 64)
      aie.use_lock(%{p}_Vp, Release, 1)
      aie.next_bd ^pf{h}
    ^qo{h}:
      %m1 = aie.dma_start(MM2S, 1, ^qi{h}, ^e{h})
    ^qi{h}:
      aie.use_lock(%{p}_Qc, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_Q : memref<322xbf16>, 0, 322)
      aie.use_lock(%{p}_Qp, Release, 1)
      aie.next_bd ^oh{h}
    ^oh{h}:
      aie.use_lock(%{p}_Oc, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_O : memref<2048xbf16>, 0, 256)
      aie.use_lock(%{p}_Op, Release, 1)
      aie.next_bd ^qi{h}
    ^e{h}:
      aie.end
    }}"""


def _center_single_b(h, p):
    # ALL-CIRCUIT outputs (no packet): MM2S0 = Pf -> shim ; MM2S1 = 2-BD [Qi, O_h] -> score (single
    # dest). score relays O_h onward to oJ/oK, so the center never demuxes -> no packet congestion.
    return f"""
    %{p}_A0 = aie.buffer(%t{h}) {{sym_name = "{p}_A0"}} : memref<4608xi8>
    %{p}_A1 = aie.buffer(%t{h}) {{sym_name = "{p}_A1"}} : memref<4608xi8>
    %{p}_B = aie.buffer(%t{h}) {{sym_name = "{p}_B"}} : memref<2320xbf16>
    %{p}_Q = aie.buffer(%t{h}) {{sym_name = "{p}_Q"}} : memref<322xbf16>
    %{p}_K = aie.buffer(%t{h}) {{sym_name = "{p}_K"}} : memref<130xbf16>
    %{p}_V = aie.buffer(%t{h}) {{sym_name = "{p}_V"}} : memref<130xbf16>
    %{p}_O = aie.buffer(%t{h}) {{sym_name = "{p}_O"}} : memref<2048xbf16>
    %{p}_P = aie.buffer(%t{h}) {{sym_name = "{p}_P"}} : memref<2320xbf16>
    %{p}_A0p = aie.lock(%t{h}, 0) {{init = 1 : i32, sym_name = "{p}_A0p"}}
    %{p}_A0c = aie.lock(%t{h}, 1) {{init = 0 : i32, sym_name = "{p}_A0c"}}
    %{p}_Bp = aie.lock(%t{h}, 2) {{init = 1 : i32, sym_name = "{p}_Bp"}}
    %{p}_Bc = aie.lock(%t{h}, 3) {{init = 0 : i32, sym_name = "{p}_Bc"}}
    %{p}_Qp = aie.lock(%t{h}, 4) {{init = 1 : i32, sym_name = "{p}_Qp"}}
    %{p}_Qc = aie.lock(%t{h}, 5) {{init = 0 : i32, sym_name = "{p}_Qc"}}
    %{p}_Op = aie.lock(%t{h}, 6) {{init = 1 : i32, sym_name = "{p}_Op"}}
    %{p}_Oc = aie.lock(%t{h}, 7) {{init = 0 : i32, sym_name = "{p}_Oc"}}
    %{p}_Pp = aie.lock(%t{h}, 8) {{init = 1 : i32, sym_name = "{p}_Pp"}}
    %{p}_Pc = aie.lock(%t{h}, 9) {{init = 0 : i32, sym_name = "{p}_Pc"}}
    %{p}_A1p = aie.lock(%t{h}, 10) {{init = 1 : i32, sym_name = "{p}_A1p"}}
    %{p}_A1c = aie.lock(%t{h}, 11) {{init = 0 : i32, sym_name = "{p}_A1c"}}
    %{p}_Kp = aie.lock(%t{h}, 12) {{init = 1 : i32, sym_name = "{p}_Kp"}}
    %{p}_Kc = aie.lock(%t{h}, 13) {{init = 0 : i32, sym_name = "{p}_Kc"}}
    %{p}_Vp = aie.lock(%t{h}, 14) {{init = 1 : i32, sym_name = "{p}_Vp"}}
    %{p}_Vc = aie.lock(%t{h}, 15) {{init = 0 : i32, sym_name = "{p}_Vc"}}
    %{p}_gate = aie.buffer(%t{h}) {{sym_name = "{p}_gate"}} : memref<1024xbf16>
    %{p}_up   = aie.buffer(%t{h}) {{sym_name = "{p}_up"}}   : memref<1024xbf16>
    %{p}_silu = aie.buffer(%t{h}) {{sym_name = "{p}_silu"}} : memref<1024xbf16>
    %{p}_uni_partial = aie.buffer(%t{h}) {{sym_name = "{p}_uni_partial"}} : memref<128xi8>
    %core{h} = aie.core(%t{h}) {{
      %c0 = arith.constant 0 : index
      %cN = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %cKV = arith.constant {KV_T*2} : index
      %cF = arith.constant {GU_T} : index
      %c4 = arith.constant 4 : i32
      %c8 = arith.constant 8 : i32
      %qr = arith.constant 256 : i32
      %kr = arith.constant 64 : i32
      %hc = arith.constant 1024 : i32
      scf.for %tok = %c0 to %cN step %c1 {{
        // #131: no-op call to force aiecc to link kc256.o (called by C++ wrapper on this core)
        func.call @_ha_noop() : () -> ()
        // ---- phase 1: Q-GEMV + rope + K-GEMV + K-RoPE + V-GEMV (B = x_bundle, nchunk=8) ----
        aie.use_lock(%{p}_Bc, AcquireGreaterEqual, 1)
        aie.use_lock(%{p}_Qp, AcquireGreaterEqual, 1)
        // #131B: Q-GEMV broadcast (column-major, j encodes block+chunk)
        scf.for %j = %c0 to %c64 step %c2 {{
          aie.use_lock(%{p}_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_q(%ji0, %{p}_A0, %{p}_B, %{p}_uni_partial, %c8, %{p}_Q) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<322xbf16>) -> ()
          aie.use_lock(%{p}_A0p, Release, 1)
          aie.use_lock(%{p}_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_q(%ji1, %{p}_A1, %{p}_B, %{p}_uni_partial, %c8, %{p}_Q) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<322xbf16>) -> ()
          aie.use_lock(%{p}_A1p, Release, 1)
        }}
        func.call @rope_bundled(%{p}_Q, %{p}_B, %{p}_Q, %qr) : (memref<322xbf16>, memref<2320xbf16>, memref<322xbf16>, i32) -> ()
        aie.use_lock(%{p}_Qc, Release, 1)
        // ---- phase 1b: K-GEMV + K-RoPE (B = x_bundle, output -> K_buf) ----
        aie.use_lock(%{p}_Kp, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %cKV step %c2 {{
          aie.use_lock(%{p}_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_kv(%ji0, %{p}_A0, %{p}_B, %{p}_uni_partial, %c8, %{p}_K) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<130xbf16>) -> ()
          aie.use_lock(%{p}_A0p, Release, 1)
          aie.use_lock(%{p}_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_kv(%ji1, %{p}_A1, %{p}_B, %{p}_uni_partial, %c8, %{p}_K) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<130xbf16>) -> ()
          aie.use_lock(%{p}_A1p, Release, 1)
        }}
        func.call @rope_kv_bundled(%{p}_K, %{p}_B, %{p}_K, %kr) : (memref<130xbf16>, memref<2320xbf16>, memref<130xbf16>, i32) -> ()
        aie.use_lock(%{p}_Kc, Release, 1)
        // ---- phase 1c: V-GEMV (B = x_bundle, output -> V_buf) ----
        aie.use_lock(%{p}_Vp, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %cKV step %c2 {{
          aie.use_lock(%{p}_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_kv(%ji0, %{p}_A0, %{p}_B, %{p}_uni_partial, %c8, %{p}_V) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<130xbf16>) -> ()
          aie.use_lock(%{p}_A0p, Release, 1)
          aie.use_lock(%{p}_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_kv(%ji1, %{p}_A1, %{p}_B, %{p}_uni_partial, %c8, %{p}_V) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<130xbf16>) -> ()
          aie.use_lock(%{p}_A1p, Release, 1)
        }}
        aie.use_lock(%{p}_Vc, Release, 1)
        aie.use_lock(%{p}_Bp, Release, 1)
        // ---- phase 2: O-proj broadcast (B = attn_out, nchunk=8) ----
        aie.use_lock(%{p}_Bc, AcquireGreaterEqual, 1)
        aie.use_lock(%{p}_Op, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %c64 step %c2 {{
          aie.use_lock(%{p}_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_o(%ji0, %{p}_A0, %{p}_B, %{p}_uni_partial, %c8, %{p}_O) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<2048xbf16>) -> ()
          aie.use_lock(%{p}_A0p, Release, 1)
          aie.use_lock(%{p}_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_o(%ji1, %{p}_A1, %{p}_B, %{p}_uni_partial, %c8, %{p}_O) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<2048xbf16>) -> ()
          aie.use_lock(%{p}_A1p, Release, 1)
        }}
        aie.use_lock(%{p}_Bp, Release, 1)
        aie.use_lock(%{p}_Oc, Release, 1)
        // ---- phase 3: FFN (B = ffn_in) ----
        // 3a: gate GEMV (nchunk=8, output -> gate_buf)
        aie.use_lock(%{p}_Bc, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %cF step %c2 {{
          aie.use_lock(%{p}_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_g(%ji0, %{p}_A0, %{p}_B, %{p}_uni_partial, %c8, %{p}_gate) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
          aie.use_lock(%{p}_A0p, Release, 1)
          aie.use_lock(%{p}_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_g(%ji1, %{p}_A1, %{p}_B, %{p}_uni_partial, %c8, %{p}_gate) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
          aie.use_lock(%{p}_A1p, Release, 1)
        }}
        // 3b: up GEMV (nchunk=8, output -> up_buf)
        scf.for %j = %c0 to %cF step %c2 {{
          aie.use_lock(%{p}_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_g(%ji0, %{p}_A0, %{p}_B, %{p}_uni_partial, %c8, %{p}_up) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
          aie.use_lock(%{p}_A0p, Release, 1)
          aie.use_lock(%{p}_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_g(%ji1, %{p}_A1, %{p}_B, %{p}_uni_partial, %c8, %{p}_up) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
          aie.use_lock(%{p}_A1p, Release, 1)
        }}
        aie.use_lock(%{p}_Bp, Release, 1)
        // 3c: SiLU(gate) * up -> silu_buf (explicit buffers)
        func.call @layer_fused_silu_mul_explicit_bf16(%{p}_gate, %{p}_up, %{p}_silu, %hc) : (memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>, i32) -> ()
        // 3d: down GEMV (nchunk=4, activation = silu_buf)
        aie.use_lock(%{p}_Pp, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %cF step %c2 {{
          aie.use_lock(%{p}_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_d(%ji0, %{p}_A0, %{p}_silu, %{p}_uni_partial, %c4, %{p}_P) : (i32, memref<4608xi8>, memref<1024xbf16>, memref<128xi8>, i32, memref<2320xbf16>) -> ()
          aie.use_lock(%{p}_A0p, Release, 1)
          aie.use_lock(%{p}_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_d(%ji1, %{p}_A1, %{p}_silu, %{p}_uni_partial, %c4, %{p}_P) : (i32, memref<4608xi8>, memref<1024xbf16>, memref<128xi8>, i32, memref<2320xbf16>) -> ()
          aie.use_lock(%{p}_A1p, Release, 1)
        }}
        aie.use_lock(%{p}_Pc, Release, 1)
      }}
      aie.end
    }}
    %mem{h} = aie.mem(%t{h}) {{
      %s0 = aie.dma_start(S2MM, 0, ^a0{h}, ^bs{h})
    ^a0{h}:
      aie.use_lock(%{p}_A0p, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_A0 : memref<4608xi8>, 0, 4608)
      aie.use_lock(%{p}_A0c, Release, 1)
      aie.next_bd ^a1{h}
    ^a1{h}:
      aie.use_lock(%{p}_A1p, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_A1 : memref<4608xi8>, 0, 4608)
      aie.use_lock(%{p}_A1c, Release, 1)
      aie.next_bd ^a0{h}
    ^bs{h}:
      %s1 = aie.dma_start(S2MM, 1, ^b{h}, ^m0{h})
    ^b{h}:
      aie.use_lock(%{p}_Bp, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_B : memref<2320xbf16>, 0, 2320)
      aie.use_lock(%{p}_Bc, Release, 1)
      aie.next_bd ^b{h}
    ^m0{h}:
      %m0 = aie.dma_start(MM2S, 0, ^pf{h}, ^m1{h})
    ^pf{h}:
      aie.use_lock(%{p}_Pc, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_P : memref<2320xbf16>, 0, 2048)
      aie.use_lock(%{p}_Pp, Release, 1)
      aie.next_bd ^kf{h}
    ^kf{h}:
      aie.use_lock(%{p}_Kc, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_K : memref<130xbf16>, 0, 64)
      aie.use_lock(%{p}_Kp, Release, 1)
      aie.next_bd ^vf{h}
    ^vf{h}:
      aie.use_lock(%{p}_Vc, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_V : memref<130xbf16>, 0, 64)
      aie.use_lock(%{p}_Vp, Release, 1)
      aie.next_bd ^pf{h}
    ^m1{h}:
      %m1 = aie.dma_start(MM2S, 1, ^qo{h}, ^e{h})
    ^qo{h}:
      aie.use_lock(%{p}_Qc, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_Q : memref<322xbf16>, 0, 322)
      aie.use_lock(%{p}_Qp, Release, 1)
      aie.next_bd ^oh{h}
    ^oh{h}:
      aie.use_lock(%{p}_Oc, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_O : memref<2048xbf16>, 0, 256)
      aie.use_lock(%{p}_Op, Release, 1)
      aie.next_bd ^qo{h}
    ^e{h}:
      aie.end
    }}"""


def score(h):
    p = f"sc{h}"
    return f"""
    %{p}_Qs = aie.buffer(%sc{h}) {{sym_name = "{p}_Qs"}} : memref<322xbf16>
    %{p}_K  = aie.buffer(%sc{h}) {{sym_name = "{p}_K"}}  : memref<{KCH}xbf16>
    %{p}_It = aie.buffer(%sc{h}) {{sym_name = "{p}_It"}} : memref<{ITC}xbf16>
    %{p}_Oh = aie.buffer(%sc{h}) {{sym_name = "{p}_Oh"}} : memref<256xbf16>
    %{p}_Qp = aie.lock(%sc{h}, 0) {{init = 1 : i32, sym_name = "{p}_Qp"}}
    %{p}_Qc = aie.lock(%sc{h}, 1) {{init = 0 : i32, sym_name = "{p}_Qc"}}
    %{p}_Kp = aie.lock(%sc{h}, 2) {{init = 1 : i32, sym_name = "{p}_Kp"}}
    %{p}_Kc = aie.lock(%sc{h}, 3) {{init = 0 : i32, sym_name = "{p}_Kc"}}
    %{p}_Ip = aie.lock(%sc{h}, 4) {{init = 1 : i32, sym_name = "{p}_Ip"}}
    %{p}_Ic = aie.lock(%sc{h}, 5) {{init = 0 : i32, sym_name = "{p}_Ic"}}
    %{p}_Ohp = aie.lock(%sc{h}, 6) {{init = 1 : i32, sym_name = "{p}_Ohp"}}
    %{p}_Ohc = aie.lock(%sc{h}, 7) {{init = 0 : i32, sym_name = "{p}_Ohc"}}
    %{p}_Ohdp = aie.lock(%sc{h}, 8) {{init = 1 : i32, sym_name = "{p}_Ohdp"}}
    %{p}_Ohdc = aie.lock(%sc{h}, 9) {{init = 0 : i32, sym_name = "{p}_Ohdc"}}
    %core_sc{h} = aie.core(%sc{h}) {{
      %c0 = arith.constant 0 : index
      %cN = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %a4 = arith.constant 4 : i32
      %h64 = arith.constant 64 : i32
      %cnc = arith.constant {NCHUNK} : index
      %sC = arith.constant {CHUNK} : i32
      scf.for %tok = %c0 to %cN step %c1 {{
        // ph1: scoring — multi-chunk online softmax (chunk={CHUNK}, nchunk={NCHUNK})
        func.call @flowkv_score_init_bf16(%a4) : (i32) -> ()
        aie.use_lock(%{p}_Qc, AcquireGreaterEqual, 1)
        func.call @flowkv_score_rope_q_bf16(%{p}_Qs, %a4, %h64) : (memref<322xbf16>, i32, i32) -> ()
        scf.for %ci = %c0 to %cnc step %c1 {{
          aie.use_lock(%{p}_Kc, AcquireGreaterEqual, 1)
          aie.use_lock(%{p}_Ip, AcquireGreaterEqual, 1)
          func.call @flowkv_score_chunk_bf16(%{p}_Qs, %{p}_K, %{p}_It, %a4, %h64, %sC) : (memref<322xbf16>, memref<{KCH}xbf16>, memref<{ITC}xbf16>, i32, i32, i32) -> ()
          aie.use_lock(%{p}_Kp, Release, 1)
          aie.use_lock(%{p}_Ic, Release, 1)
        }}
        aie.use_lock(%{p}_Qp, Release, 1)
        // ph2: relay O_h (S2MM0-BD1 -> MM2S1, no compute, rl-style lock-dance)
        aie.use_lock(%{p}_Ohc, AcquireGreaterEqual, 1)
        aie.use_lock(%{p}_Ohdp, AcquireGreaterEqual, 1)
        aie.use_lock(%{p}_Ohp, Release, 1)
        aie.use_lock(%{p}_Ohdc, Release, 1)
      }}
      aie.end
    }}
    %mem_sc{h} = aie.mem(%sc{h}) {{
      %s0 = aie.dma_start(S2MM, 0, ^q{h}, ^ks{h})
    ^q{h}:
      aie.use_lock(%{p}_Qp, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_Qs : memref<322xbf16>, 0, 322)
      aie.use_lock(%{p}_Qc, Release, 1)
      aie.next_bd ^oh{h}
    ^oh{h}:
      aie.use_lock(%{p}_Ohp, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_Oh : memref<256xbf16>, 0, 256)
      aie.use_lock(%{p}_Ohc, Release, 1)
      aie.next_bd ^q{h}
    ^ks{h}:
      %s1 = aie.dma_start(S2MM, 1, ^k{h}, ^im{h})
    ^k{h}:
      aie.use_lock(%{p}_Kp, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_K : memref<{KCH}xbf16>, 0, {KCH})
      aie.use_lock(%{p}_Kc, Release, 1)
      aie.next_bd ^k{h}
    ^im{h}:
      %m0 = aie.dma_start(MM2S, 0, ^io{h}, ^ohm{h})
    ^io{h}:
      aie.use_lock(%{p}_Ic, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_It : memref<{ITC}xbf16>, 0, {ITC})
      aie.use_lock(%{p}_Ip, Release, 1)
      aie.next_bd ^io{h}
    ^ohm{h}:
      %m1 = aie.dma_start(MM2S, 1, ^ohf{h}, ^e{h})
    ^ohf{h}:
      aie.use_lock(%{p}_Ohdc, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_Oh : memref<256xbf16>, 0, 256)
      aie.use_lock(%{p}_Ohdp, Release, 1)
      aie.next_bd ^ohf{h}
    ^e{h}:
      aie.end
    }}"""


def value(h):
    p = f"va{h}"
    return f"""
    %{p}_Iv = aie.buffer(%va{h}) {{sym_name = "{p}_Iv"}} : memref<{ITC}xbf16>
    %{p}_V  = aie.buffer(%va{h}) {{sym_name = "{p}_V"}}  : memref<{KCH}xbf16>
    %{p}_Of = aie.buffer(%va{h}) {{sym_name = "{p}_Of"}} : memref<256xbf16>
    %{p}_Ip = aie.lock(%va{h}, 0) {{init = 1 : i32, sym_name = "{p}_Ip"}}
    %{p}_Ic = aie.lock(%va{h}, 1) {{init = 0 : i32, sym_name = "{p}_Ic"}}
    %{p}_Vp = aie.lock(%va{h}, 2) {{init = 1 : i32, sym_name = "{p}_Vp"}}
    %{p}_Vc = aie.lock(%va{h}, 3) {{init = 0 : i32, sym_name = "{p}_Vc"}}
    %{p}_Op = aie.lock(%va{h}, 4) {{init = 1 : i32, sym_name = "{p}_Op"}}
    %{p}_Oc = aie.lock(%va{h}, 5) {{init = 0 : i32, sym_name = "{p}_Oc"}}
    %core_va{h} = aie.core(%va{h}) {{
      %c0 = arith.constant 0 : index
      %cN = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %a4 = arith.constant 4 : i32
      %h64 = arith.constant 64 : i32
      %cnc = arith.constant {NCHUNK} : index
      %sC = arith.constant {CHUNK} : i32
      scf.for %tok = %c0 to %cN step %c1 {{
        func.call @flowkv_value_init_bf16(%a4, %h64) : (i32, i32) -> ()
        scf.for %ci = %c0 to %cnc step %c1 {{
          aie.use_lock(%{p}_Ic, AcquireGreaterEqual, 1)
          aie.use_lock(%{p}_Vc, AcquireGreaterEqual, 1)
          func.call @flowkv_value_accum_bf16(%{p}_Iv, %{p}_V, %a4, %h64, %sC) : (memref<{ITC}xbf16>, memref<{KCH}xbf16>, i32, i32, i32) -> ()
          aie.use_lock(%{p}_Ip, Release, 1)
          aie.use_lock(%{p}_Vp, Release, 1)
        }}
        aie.use_lock(%{p}_Op, AcquireGreaterEqual, 1)
        func.call @flowkv_value_normalize_bf16(%{p}_Of, %a4, %h64) : (memref<256xbf16>, i32, i32) -> ()
        aie.use_lock(%{p}_Oc, Release, 1)
      }}
      aie.end
    }}
    %mem_va{h} = aie.mem(%va{h}) {{
      %s0 = aie.dma_start(S2MM, 0, ^iv{h}, ^vs{h})
    ^iv{h}:
      aie.use_lock(%{p}_Ip, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_Iv : memref<{ITC}xbf16>, 0, {ITC})
      aie.use_lock(%{p}_Ic, Release, 1)
      aie.next_bd ^iv{h}
    ^vs{h}:
      %s1 = aie.dma_start(S2MM, 1, ^v{h}, ^om{h})
    ^v{h}:
      aie.use_lock(%{p}_Vp, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_V : memref<{KCH}xbf16>, 0, {KCH})
      aie.use_lock(%{p}_Vc, Release, 1)
      aie.next_bd ^v{h}
    ^om{h}:
      %m0 = aie.dma_start(MM2S, 0, ^oo{h}, ^e{h})
    ^oo{h}:
      aie.use_lock(%{p}_Oc, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_Of : memref<256xbf16>, 0, 256)
      aie.use_lock(%{p}_Op, Release, 1)
      aie.next_bd ^oo{h}
    ^e{h}:
      aie.end
    }}"""


def join_memtile(name, tile, with_weight_relay=False, packet_id=None):
    """Join MemTile: 4 S2MM gather -> 1 MM2S output.

    When with_weight_relay=True (cols 2-5), also adds weight relay channels:
    - S2MM4-5: receive weight chunks from 2 shim tiles
    - MM2S1-2: forward weight chunks to 2 core tiles
    Uses lock indices 8-15 for weight relay (4 lock pairs × 2 paths).

    When packet_id is set (#139b), MM2S0 emits a packet header on the output
    (jA→mx pkt_id=0, jB→mx pkt_id=2).
    """
    SL = 4608                     # per-weight-chunk bytes (one GEMV tile i8)
    s = f"""
    %{name}_buf = aie.buffer(%{tile}) {{sym_name = "{name}_buf"}} : memref<2048xbf16>"""
    for k in range(4):
        s += f"""
    %{name}_p{k} = aie.lock(%{tile}, {2*k}) {{init = 1 : i32, sym_name = "{name}_p{k}"}}
    %{name}_c{k} = aie.lock(%{tile}, {2*k+1}) {{init = 0 : i32, sym_name = "{name}_c{k}"}}"""
    D = RELAY_DEPTH
    if with_weight_relay:
        # Weight relay buffer: 2 paths x D slots x SL bytes. Deeper D = more lookahead
        # held in MemTile SRAM while DDR refills behind the core.
        s += f"""
    %{name}_wt = aie.buffer(%{tile}) {{sym_name = "{name}_wt"}} : memref<{2*D*SL}xi8>"""
        # Locks 8.. : per path p in {A,B}, per slot k, a producer/consumer pair.
        for p in range(2):
            for k in range(D):
                base = 8 + 2 * (p * D + k)
                s += f"""
    %{name}_w{p}s{k}p = aie.lock(%{tile}, {base}) {{init = 1 : i32, sym_name = "{name}_w{p}s{k}p"}}
    %{name}_w{p}s{k}c = aie.lock(%{tile}, {base+1}) {{init = 0 : i32, sym_name = "{name}_w{p}s{k}c"}}"""
    s += f"""
    %{name}_dma = aie.memtile_dma(%{tile}) {{"""
    # S2MM0-3: join receivers — each subsequent channel needs a label for the previous
    # channel's next-channel pointer.
    for ch in range(4):
        if ch == 3:
            nxt = f"^{name}w0" if with_weight_relay else f"^{name}m0"
        else:
            nxt = f"^{name}s{ch+1}"
        if ch > 0:
            s += f"""
    ^{name}s{ch}:"""
        s += f"""
      %s{ch} = aie.dma_start(S2MM, {ch}, ^{name}g{ch}, {nxt})
    ^{name}g{ch}:
      aie.use_lock(%{name}_p{ch}, AcquireGreaterEqual, 1)
      aie.dma_bd(%{name}_buf : memref<2048xbf16>, {ch*QI}, {QI})
      aie.use_lock(%{name}_c{ch}, Release, 1)
      aie.next_bd ^{name}g{ch}"""
    if with_weight_relay:
        # S2MM4/5: fill path A (-> center row2) and path B (-> center row3), D-slot
        # cyclic BD chain each. Deeper chain = more weight chunks buffered ahead.
        for p in range(2):
            nxt = f"^{name}w1" if p == 0 else f"^{name}m0"
            s += f"""
    ^{name}w{p}:
      %w{p} = aie.dma_start(S2MM, {4+p}, ^{name}w{p}s0, {nxt})"""
            for k in range(D):
                s += f"""
    ^{name}w{p}s{k}:
      aie.use_lock(%{name}_w{p}s{k}p, AcquireGreaterEqual, 1)
      aie.dma_bd(%{name}_wt : memref<{2*D*SL}xi8>, {(p*D+k)*SL}, {SL})
      aie.use_lock(%{name}_w{p}s{k}c, Release, 1)
      aie.next_bd ^{name}w{p}s{(k+1) % D}"""
    nxt_mm2s0 = f"^{name}x0" if with_weight_relay else f"^{name}e"
    # MM2S0: join output
    s += f"""
    ^{name}m0:
      %m0 = aie.dma_start(MM2S, 0, ^{name}o0, {nxt_mm2s0})
    ^{name}o0:"""
    if packet_id is not None:
        s += f"""
      aie.dma_bd_packet(0, {packet_id})"""
    s += f"""
      aie.use_lock(%{name}_c0, AcquireGreaterEqual, 1)
      aie.dma_bd(%{name}_buf : memref<2048xbf16>, 0, {QI})
      aie.use_lock(%{name}_p0, Release, 1)
      aie.next_bd ^{name}o1
    ^{name}o1:
      aie.use_lock(%{name}_c1, AcquireGreaterEqual, 1)
      aie.dma_bd(%{name}_buf : memref<2048xbf16>, {QI}, {QI})
      aie.use_lock(%{name}_p1, Release, 1)
      aie.next_bd ^{name}o2
    ^{name}o2:
      aie.use_lock(%{name}_c2, AcquireGreaterEqual, 1)
      aie.dma_bd(%{name}_buf : memref<2048xbf16>, {2*QI}, {QI})
      aie.use_lock(%{name}_p2, Release, 1)
      aie.next_bd ^{name}o3
    ^{name}o3:
      aie.use_lock(%{name}_c3, AcquireGreaterEqual, 1)
      aie.dma_bd(%{name}_buf : memref<2048xbf16>, {3*QI}, {QI})
      aie.use_lock(%{name}_p3, Release, 1)
      aie.next_bd ^{name}o0"""
    if with_weight_relay:
        # MM2S1/2: drain path A -> center row2, path B -> center row3 (mirrors the fill chain).
        for p in range(2):
            nxt = f"^{name}x1" if p == 0 else f"^{name}e"
            s += f"""
    ^{name}x{p}:
      %x{p} = aie.dma_start(MM2S, {1+p}, ^{name}x{p}s0, {nxt})"""
            for k in range(D):
                s += f"""
    ^{name}x{p}s{k}:
      aie.use_lock(%{name}_w{p}s{k}c, AcquireGreaterEqual, 1)
      aie.dma_bd(%{name}_wt : memref<{2*D*SL}xi8>, {(p*D+k)*SL}, {SL})
      aie.use_lock(%{name}_w{p}s{k}p, Release, 1)
      aie.next_bd ^{name}x{p}s{(k+1) % D}"""
    s += f"""
    ^{name}e:
      aie.end
    }}"""
    return s


def split_memtile(name, tile, with_weight_relay=False):
    """1 shim stream (4*KVN) -> 4 MM2S slices (KVN each). F3split per-slice locks + 4-BD fill.

    When with_weight_relay=True (#138), also adds weight relay on S2MM4/MM2S4
    with D-slot cyclic BD chain. Uses locks 8+ (split uses 0-7).
    """
    SL = KVN
    s = f"""
    %{name}_buf = aie.buffer(%{tile}) {{sym_name = "{name}_buf"}} : memref<{4*KVN}xbf16>"""
    for k in range(4):
        s += f"""
    %{name}_p{k} = aie.lock(%{tile}, {2*k}) {{init = 1 : i32, sym_name = "{name}_p{k}"}}
    %{name}_c{k} = aie.lock(%{tile}, {2*k+1}) {{init = 0 : i32, sym_name = "{name}_c{k}"}}"""
    if with_weight_relay:
        D = RELAY_DEPTH
        WS = 4608  # per-weight-chunk bytes
        s += f"""
    %{name}_wt = aie.buffer(%{tile}) {{sym_name = "{name}_wt"}} : memref<{D*WS}xi8>"""
        for k in range(D):
            s += f"""
    %{name}_wsp{k} = aie.lock(%{tile}, {8+2*k}) {{init = 1 : i32, sym_name = "{name}_wsp{k}"}}
    %{name}_wsc{k} = aie.lock(%{tile}, {8+2*k+1}) {{init = 0 : i32, sym_name = "{name}_wsc{k}"}}"""
    # Next label after fill chain: m0 (=MM2S0 start) if no weight relay, else w0 (=S2MM4 start)
    fill_nxt = f"^{name}w0" if with_weight_relay else f"^{name}m0"
    s += f"""
    %{name}_dma = aie.memtile_dma(%{tile}) {{
      %s0 = aie.dma_start(S2MM, 0, ^{name}f0, {fill_nxt})"""
    for k in range(4):
        s += f"""
    ^{name}f{k}:
      aie.use_lock(%{name}_p{k}, AcquireGreaterEqual, 1)
      aie.dma_bd(%{name}_buf : memref<{4*KVN}xbf16>, {k*SL}, {SL})
      aie.use_lock(%{name}_c{k}, Release, 1)
      aie.next_bd ^{name}f{(k+1) % 4}"""
    if with_weight_relay:
        # S2MM4: weight fill, D-slot cyclic
        nxt_mm2s = f"^{name}m0"
        s += f"""
    ^{name}w0:
      %w = aie.dma_start(S2MM, 4, ^{name}ws0, {nxt_mm2s})"""
        for k in range(D):
            s += f"""
    ^{name}ws{k}:
      aie.use_lock(%{name}_wsp{k}, AcquireGreaterEqual, 1)
      aie.dma_bd(%{name}_wt : memref<{D*WS}xi8>, {k*WS}, {WS})
      aie.use_lock(%{name}_wsc{k}, Release, 1)
      aie.next_bd ^{name}ws{(k+1) % D}"""
    # MM2S0..3: K/V drain (4 slices, each cyclic)
    for ch in range(4):
        nxt = f"^{name}m{ch+1}" if ch < 3 else (f"^{name}wrel" if with_weight_relay else f"^{name}e")
        s += f"""
    ^{name}m{ch}:
      %m{ch} = aie.dma_start(MM2S, {ch}, ^{name}o{ch}, {nxt})
    ^{name}o{ch}:
      aie.use_lock(%{name}_c{ch}, AcquireGreaterEqual, 1)
      aie.dma_bd(%{name}_buf : memref<{4*KVN}xbf16>, {ch*SL}, {SL})
      aie.use_lock(%{name}_p{ch}, Release, 1)
      aie.next_bd ^{name}o{ch}"""
    if with_weight_relay:
        # MM2S4: weight drain, D-slot cyclic
        s += f"""
    ^{name}wrel:
      %x = aie.dma_start(MM2S, 4, ^{name}xs0, ^{name}e)"""
        for k in range(D):
            s += f"""
    ^{name}xs{k}:
      aie.use_lock(%{name}_wsc{k}, AcquireGreaterEqual, 1)
      aie.dma_bd(%{name}_wt : memref<{D*WS}xi8>, {k*WS}, {WS})
      aie.use_lock(%{name}_wsp{k}, Release, 1)
      aie.next_bd ^{name}xs{(k+1) % D}"""
    s += f"""
    ^{name}e:
      aie.end
    }}"""
    return s


tile_decls = "".join(f"    %t{h} = aie.tile({CENTER_COLS[h][0]}, {CENTER_COLS[h][1]})\n" for h in range(NH))
tile_decls += "".join(f"    %sc{h} = aie.tile({SCORE_COLS[h][0]}, {SCORE_COLS[h][1]})\n" for h in range(NH))
tile_decls += "".join(f"    %va{h} = aie.tile({VALUE_COLS[h][0]}, {VALUE_COLS[h][1]})\n" for h in range(NH))
# Attn-join stays at working attn8 positions (cols 2,3). Put O-join on cols 4,5 to avoid the
# score-right -> far-left congestion; move Klo/Khi to the now-free cols 0,1 (placement-probe PASS).
tile_decls += "    %jA = aie.tile(2, 1)\n    %jB = aie.tile(3, 1)\n"
tile_decls += "    %Klo = aie.tile(0, 1)\n    %Khi = aie.tile(1, 1)\n    %Vlo = aie.tile(6, 1)\n    %Vhi = aie.tile(7, 1)\n"
tile_decls += "    %oJ = aie.tile(4, 1)\n    %oK = aie.tile(5, 1)\n"
if RL_FIX:
    tile_decls += "    %op = aie.tile(3, 4)\n    %nm = aie.tile(4, 4)\n    %mx = aie.tile(5, 4)\n"
else:
    tile_decls += "    %rl = aie.tile(2, 4)\n    %op = aie.tile(3, 4)\n    %nm = aie.tile(4, 4)\n    %mx = aie.tile(5, 4)\n"
# no extra pre-merge tile: all 32 compute tiles are already occupied (router failed with 36th tile).
shim_decls = "".join(f"    %sh{h} = aie.tile({h}, 0)\n" for h in range(NH))

_ha_link = 'layer_fused_bcast_kc256_rr.o' if HANDASM_RR else 'layer_fused_bcast_kc256.o'
funcs = (
    f'    func.func private @_ha_noop() -> () attributes {{link_with = "{_ha_link}"}}\n'
    '    func.func private @fused_dequant_matvec_v2_bf16(i32, i32, memref<4608xi8>, memref<2320xbf16>, memref<322xbf16>) attributes {link_with = "fused_dequant_gemv_v2_signed_2048k_g32.o"}\n'
    '    func.func private @rope_bundled(memref<322xbf16>, memref<2320xbf16>, memref<322xbf16>, i32) attributes {link_with = "rope_il.o"}\n'
    '    func.func private @rope_kv_bundled(memref<130xbf16>, memref<2320xbf16>, memref<130xbf16>, i32) attributes {link_with = "rope_il.o"}\n'
    '    func.func private @layer_fused_gate_up_bf16(i32, i32, memref<4608xi8>, memref<2320xbf16>, i32) attributes {link_with = "layer_fused_relay.o"}\n'
    '    func.func private @generic_bcast_gemv_bf16_q(i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<322xbf16>) attributes {link_with = "layer_fused_unified_bcast.o"}\n'
    '    func.func private @generic_bcast_gemv_bf16_o(i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<2048xbf16>) attributes {link_with = "layer_fused_unified_bcast.o"}\n'
    '    func.func private @generic_bcast_gemv_bf16_g(i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) attributes {link_with = "layer_fused_unified_bcast.o"}\n'
    '    func.func private @generic_bcast_gemv_bf16_kv(i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<130xbf16>) attributes {link_with = "layer_fused_unified_bcast.o"}\n'
    '    func.func private @generic_bcast_gemv_bf16_d(i32, memref<4608xi8>, memref<1024xbf16>, memref<128xi8>, i32, memref<2320xbf16>) attributes {link_with = "layer_fused_unified_bcast.o"}\n'
    '    func.func private @layer_fused_silu_mul_explicit_bf16(memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>, i32) attributes {link_with = "layer_fused_relay.o"}\n'
    '    func.func private @layer_fused_down_v2_x4_bf16(i32, i32, i32, memref<4608xi8>, memref<2320xbf16>) attributes {link_with = "layer_fused_relay.o"}\n'
    '    func.func private @attn_copy_bf16(memref<2320xbf16>, memref<2320xbf16>, i32) attributes {link_with = "attn_concat.o"}\n'
    '    func.func private @oproj_matvec_v2_bf16(i32, i32, memref<4608xi8>, memref<2320xbf16>, memref<2048xbf16>) attributes {link_with = "fused_dequant_gemv_v2_oproj_signed_2048k_g32.o"}\n'
    '    func.func private @layer_fused_add_bf16(memref<2048xbf16>, memref<2048xbf16>, memref<2320xbf16>, i32) attributes {link_with = "layer_fused_relay.o"}\n'
    '    func.func private @layer_fused_rms_norm2_bf16(memref<2320xbf16>, memref<2048xbf16>, memref<2320xbf16>, i32) attributes {link_with = "layer_fused_relay.o"}\n'
    '    func.func private @flowkv_score_init_bf16(i32) attributes {link_with = "flowkv_64d_h4_c256.o"}\n'
    '    func.func private @flowkv_score_rope_q_bf16(memref<322xbf16>, i32, i32) attributes {link_with = "flowkv_64d_h4_c256.o"}\n'
    f'    func.func private @flowkv_score_chunk_bf16(memref<322xbf16>, memref<{KCH}xbf16>, memref<{ITC}xbf16>, i32, i32, i32) attributes {{link_with = "flowkv_64d_h4_c256.o"}}\n'
    '    func.func private @flowkv_value_init_bf16(i32, i32) attributes {link_with = "flowkv_64d_h4_c256.o"}\n'
    f'    func.func private @flowkv_value_accum_bf16(memref<{ITC}xbf16>, memref<{KCH}xbf16>, i32, i32, i32) attributes {{link_with = "flowkv_64d_h4_c256.o"}}\n'
    '    func.func private @flowkv_value_normalize_bf16(memref<256xbf16>, i32, i32) attributes {link_with = "flowkv_64d_h4_c256.o"}\n')

flows = []
flows.append("    aie.flow(%sh0, DMA : 1, %mx, DMA : 0)   // x_bundle -> mux S2MM0")
# mux S2MM1 needs two producers (attn_out, ffn_in) and no extra compute tile is available.
# Use F3e-1 packet arbitration: rl pkt0 -> mx.mxa, nm pkt1 -> mx.mxf.
if RL_FIX:
    flows.append("    aie.packet_flow(0) { aie.packet_source<%jA, DMA : 0> aie.packet_dest<%mx, DMA : 1> }   // attnA -> mux S2MM1")
    flows.append("    aie.packet_flow(1) { aie.packet_source<%nm, DMA : 0> aie.packet_dest<%mx, DMA : 1> }   // ffn_in -> mux S2MM1")
    flows.append("    aie.packet_flow(2) { aie.packet_source<%jB, DMA : 0> aie.packet_dest<%mx, DMA : 1> }   // attnB -> mux S2MM1")
else:
    flows.append("    aie.packet_flow(0) { aie.packet_source<%rl, DMA : 0> aie.packet_dest<%mx, DMA : 1> }   // attn_out -> mux S2MM1")
    flows.append("    aie.packet_flow(1) { aie.packet_source<%nm, DMA : 0> aie.packet_dest<%mx, DMA : 1> }   // ffn_in -> mux S2MM1")
# Edge weight relay mapping (DECOUPLE). Edge columns 0,1,6,7 relay weights through
# Klo/Khi/Vlo/Vhi MemTiles alongside K/V split. Weight relay uses S2MM4 + MM2S4
# (no conflict with K/V split S2MM0 + MM2S0-3). Uses locks 8+ (split uses 0-7).
EDGE_RELAY = {0: 'Klo', 1: 'Khi', 6: 'Vlo', 7: 'Vhi'} if DECOUPLE else {}
for h in range(NH):
    oj = 'oJ' if h < 4 else 'oK'
    osl = h if h < 4 else h - 4
    # Weight delivery. Relay columns go shim -> MemTile(col,1) -> core (P6 decoupling);
    # the rest keep the direct shim -> core flow.
    rc = h % 4 if h % 4 in RELAY_COLS else None
    edge_mt = EDGE_RELAY.get(h)  # edge weight relay MemTile name or None
    if edge_mt is not None:
        # Edge weight relay: shim -> edge MemTile S2MM4 -> MM2S4 -> center core
        # (MM2S4 avoids conflict with K/V split MM2S0-3)
        flows.append(f"    aie.flow(%sh{h}, DMA : 0, %{edge_mt}, DMA : 4)   // A{h} shim -> {edge_mt} (edge relay)")
        flows.append(f"    aie.flow(%{edge_mt}, DMA : 4, %t{h}, DMA : 0)   // A{h} {edge_mt} -> center")
    elif rc is not None:
        mt_name = ['jA', 'jB', 'oJ', 'oK'][rc]   # jA(2,1) jB(3,1) oJ(4,1) oK(5,1)
        s2mm_ch = 4 if h < 4 else 5               # S2MM4/MM2S1 -> row2, S2MM5/MM2S2 -> row3
        mm2s_ch = 1 if h < 4 else 2
        flows.append(f"    aie.flow(%sh{h}, DMA : 0, %{mt_name}, DMA : {s2mm_ch})   // A{h} shim -> {mt_name}")
        flows.append(f"    aie.flow(%{mt_name}, DMA : {mm2s_ch}, %t{h}, DMA : 0)   // A{h} {mt_name} -> center")
    else:
        flows.append(f"    aie.flow(%sh{h}, DMA : 0, %t{h}, DMA : 0)   // A{h}")
    flows.append(f"    aie.flow(%mx, DMA : 0, %t{h}, DMA : 1)   // mux 3-bcast -> center {h}")
    flows.append(f"    aie.flow(%t{h}, DMA : 0, %sh{h}, DMA : 0)   // Pf{h} -> drain (circuit)")
    flows.append(f"    aie.flow(%t{h}, DMA : 1, %sc{h}, DMA : 0)   // [Qi,O_h]{h} 2-BD -> score{h} (circuit)")
    flows.append(f"    aie.flow(%sc{h}, DMA : 0, %va{h}, DMA : 0)   // inter{h} -> value{h}")
    flows.append(f"    aie.flow(%sc{h}, DMA : 1, %{oj}, DMA : {osl})   // O_h{h} relayed by score -> {oj}[{osl}]")
# K/V split delivery: Klo->sc0-3, Khi->sc4-7, Vlo->va0-3, Vhi->va4-7
flows.append("    aie.flow(%sh3, DMA : 1, %Klo, DMA : 0)   // K_lo -> Klo split")
flows.append("    aie.flow(%sh4, DMA : 1, %Khi, DMA : 0)   // K_hi -> Khi split")
flows.append("    aie.flow(%sh5, DMA : 1, %Vlo, DMA : 0)   // V_lo -> Vlo split")
flows.append("    aie.flow(%sh6, DMA : 1, %Vhi, DMA : 0)   // V_hi -> Vhi split")
for k in range(4):
    flows.append(f"    aie.flow(%Klo, DMA : {k}, %sc{k}, DMA : 1)   // K[{k}] -> score{k}")
    flows.append(f"    aie.flow(%Khi, DMA : {k}, %sc{k+4}, DMA : 1)   // K[{k+4}] -> score{k+4}")
    flows.append(f"    aie.flow(%Vlo, DMA : {k}, %va{k}, DMA : 1)   // V[{k}] -> value{k}")
    flows.append(f"    aie.flow(%Vhi, DMA : {k}, %va{k+4}, DMA : 1)   // V[{k+4}] -> value{k+4}")
# attn_out join: va0-3 -> jA slots, va4-7 -> jB slots (circuit)
for k in range(4):
    flows.append(f"    aie.flow(%va{k}, DMA : 0, %jA, DMA : {k})   // Of{k} -> joinA")
    flows.append(f"    aie.flow(%va{k+4}, DMA : 0, %jB, DMA : {k})   // Of{k+4} -> joinB")
if not RL_FIX:
    flows.append("    aie.flow(%jA, DMA : 0, %rl, DMA : 0)   // attn Half0 -> relay")
    flows.append("    aie.flow(%jB, DMA : 0, %rl, DMA : 1)   // attn Half1 -> relay")
# rl drains attn_out -> mux via packet_flow(0) above (no circuit flow to op)
# O-join: oJ/oK (center O_h) -> op (O-relay) -> nm (ANM)
flows.append("    aie.flow(%oJ, DMA : 0, %op, DMA : 0)   // O Half0 -> orelay")
flows.append("    aie.flow(%oK, DMA : 0, %op, DMA : 1)   // O Half1 -> orelay")
flows.append("    aie.flow(%op, DMA : 0, %nm, DMA : 0)   // O -> ANM")
flows.append("    aie.flow(%sh2, DMA : 1, %nm, DMA : 1)   // resid+gain -> ANM (2-BD on S2MM1)")
flows.append("    aie.flow(%nm, DMA : 1, %sh4, DMA : 1)   // s = O+resid (attn-residual) -> arg0 tail")
# nm drains ffn_in -> mux via packet_flow(17) above (no circuit flow to mux)
flows_txt = "\n".join(flows) + "\n"

relay = """
    %rl_A = aie.buffer(%rl) {sym_name = "rl_A"} : memref<2048xbf16>
    %rl_p0 = aie.lock(%rl, 0) {init = 1 : i32, sym_name = "rl_p0"}
    %rl_c0 = aie.lock(%rl, 1) {init = 0 : i32, sym_name = "rl_c0"}
    %rl_p1 = aie.lock(%rl, 2) {init = 1 : i32, sym_name = "rl_p1"}
    %rl_c1 = aie.lock(%rl, 3) {init = 0 : i32, sym_name = "rl_c1"}
    %rl_op = aie.lock(%rl, 4) {init = 1 : i32, sym_name = "rl_op"}
    %rl_oc = aie.lock(%rl, 5) {init = 0 : i32, sym_name = "rl_oc"}
    %core_rl = aie.core(%rl) {
      %z = arith.constant 0 : index
      %N = arith.constant 9223372036854775807 : index
      %one = arith.constant 1 : index
      scf.for %it = %z to %N step %one {
        aie.use_lock(%rl_c0, AcquireGreaterEqual, 1)
        aie.use_lock(%rl_c1, AcquireGreaterEqual, 1)
        aie.use_lock(%rl_op, AcquireGreaterEqual, 1)
        aie.use_lock(%rl_p0, Release, 1)
        aie.use_lock(%rl_p1, Release, 1)
        aie.use_lock(%rl_oc, Release, 1)
      }
      aie.end
    }
    %mem_rl = aie.mem(%rl) {
      %s0 = aie.dma_start(S2MM, 0, ^rh0, ^rs1)
    ^rh0:
      aie.use_lock(%rl_p0, AcquireGreaterEqual, 1)
      aie.dma_bd(%rl_A : memref<2048xbf16>, 0, 1024)
      aie.use_lock(%rl_c0, Release, 1)
      aie.next_bd ^rh0
    ^rs1:
      %s1 = aie.dma_start(S2MM, 1, ^rh1, ^rm0)
    ^rh1:
      aie.use_lock(%rl_p1, AcquireGreaterEqual, 1)
      aie.dma_bd(%rl_A : memref<2048xbf16>, 1024, 1024)
      aie.use_lock(%rl_c1, Release, 1)
      aie.next_bd ^rh1
    ^rm0:
      %m0 = aie.dma_start(MM2S, 0, ^ro, ^re)
    ^ro:
      aie.use_lock(%rl_oc, AcquireGreaterEqual, 1)
      aie.dma_bd_packet(0, 0)
      aie.dma_bd(%rl_A : memref<2048xbf16>, 0, 2048)
      aie.use_lock(%rl_op, Release, 1)
      aie.next_bd ^ro
    ^re:
      aie.end
    }"""

op = """
    %op_O  = aie.buffer(%op) {sym_name = "op_O"}  : memref<2048xbf16>
    %op_p0 = aie.lock(%op, 0) {init = 1 : i32, sym_name = "op_p0"}
    %op_c0 = aie.lock(%op, 1) {init = 0 : i32, sym_name = "op_c0"}
    %op_p1 = aie.lock(%op, 2) {init = 1 : i32, sym_name = "op_p1"}
    %op_c1 = aie.lock(%op, 3) {init = 0 : i32, sym_name = "op_c1"}
    %op_op = aie.lock(%op, 4) {init = 1 : i32, sym_name = "op_op"}
    %op_oc = aie.lock(%op, 5) {init = 0 : i32, sym_name = "op_oc"}
    %core_op = aie.core(%op) {
      %z = arith.constant 0 : index
      %N = arith.constant 9223372036854775807 : index
      %one = arith.constant 1 : index
      scf.for %it = %z to %N step %one {
        aie.use_lock(%op_c0, AcquireGreaterEqual, 1)
        aie.use_lock(%op_c1, AcquireGreaterEqual, 1)
        aie.use_lock(%op_op, AcquireGreaterEqual, 1)
        aie.use_lock(%op_p0, Release, 1)
        aie.use_lock(%op_p1, Release, 1)
        aie.use_lock(%op_oc, Release, 1)
      }
      aie.end
    }
    %mem_op = aie.mem(%op) {
      %s0 = aie.dma_start(S2MM, 0, ^oh0, ^os1)
    ^oh0:
      aie.use_lock(%op_p0, AcquireGreaterEqual, 1)
      aie.dma_bd(%op_O : memref<2048xbf16>, 0, 1024)
      aie.use_lock(%op_c0, Release, 1)
      aie.next_bd ^oh0
    ^os1:
      %s1 = aie.dma_start(S2MM, 1, ^oh1, ^om0)
    ^oh1:
      aie.use_lock(%op_p1, AcquireGreaterEqual, 1)
      aie.dma_bd(%op_O : memref<2048xbf16>, 1024, 1024)
      aie.use_lock(%op_c1, Release, 1)
      aie.next_bd ^oh1
    ^om0:
      %m0 = aie.dma_start(MM2S, 0, ^oo, ^oe)
    ^oo:
      aie.use_lock(%op_oc, AcquireGreaterEqual, 1)
      aie.dma_bd(%op_O : memref<2048xbf16>, 0, 2048)
      aie.use_lock(%op_op, Release, 1)
      aie.next_bd ^oo
    ^oe:
      aie.end
    }"""

nm = """
    %nm_O = aie.buffer(%nm) {sym_name = "nm_O"} : memref<2048xbf16>
    %nm_R = aie.buffer(%nm) {sym_name = "nm_R"} : memref<2048xbf16>
    %nm_F = aie.buffer(%nm) {sym_name = "nm_F"} : memref<2320xbf16>
    %nm_FN = aie.buffer(%nm) {sym_name = "nm_FN"} : memref<2320xbf16>
    %nm_gain = aie.buffer(%nm) {sym_name = "nm_gain"} : memref<2048xbf16>
    %nm_Op = aie.lock(%nm, 0) {init = 1 : i32, sym_name = "nm_Op"}
    %nm_Oc = aie.lock(%nm, 1) {init = 0 : i32, sym_name = "nm_Oc"}
    %nm_Rp = aie.lock(%nm, 2) {init = 1 : i32, sym_name = "nm_Rp"}
    %nm_Rc = aie.lock(%nm, 3) {init = 0 : i32, sym_name = "nm_Rc"}
    %nm_Fp = aie.lock(%nm, 4) {init = 1 : i32, sym_name = "nm_Fp"}
    %nm_Fc = aie.lock(%nm, 5) {init = 0 : i32, sym_name = "nm_Fc"}
    %nm_Gp = aie.lock(%nm, 6) {init = 1 : i32, sym_name = "nm_Gp"}
    %nm_Gc = aie.lock(%nm, 7) {init = 0 : i32, sym_name = "nm_Gc"}
    %nm_FNp = aie.lock(%nm, 8) {init = 1 : i32, sym_name = "nm_FNp"}
    %nm_FNc = aie.lock(%nm, 9) {init = 0 : i32, sym_name = "nm_FNc"}
    %core_nm = aie.core(%nm) {
      %z = arith.constant 0 : index
      %N = arith.constant 9223372036854775807 : index
      %one = arith.constant 1 : index
      %ne = arith.constant 2048 : i32
      scf.for %it = %z to %N step %one {
        aie.use_lock(%nm_Oc, AcquireGreaterEqual, 1)
        aie.use_lock(%nm_Rc, AcquireGreaterEqual, 1)
        aie.use_lock(%nm_Gc, AcquireGreaterEqual, 1)
        aie.use_lock(%nm_Fp, AcquireGreaterEqual, 1)
        aie.use_lock(%nm_FNp, AcquireGreaterEqual, 1)
        func.call @layer_fused_add_bf16(%nm_O, %nm_R, %nm_F, %ne) : (memref<2048xbf16>, memref<2048xbf16>, memref<2320xbf16>, i32) -> ()
        func.call @layer_fused_rms_norm2_bf16(%nm_F, %nm_gain, %nm_FN, %ne) : (memref<2320xbf16>, memref<2048xbf16>, memref<2320xbf16>, i32) -> ()
        aie.use_lock(%nm_Op, Release, 1)
        aie.use_lock(%nm_Rp, Release, 1)
        aie.use_lock(%nm_Gp, Release, 1)
        aie.use_lock(%nm_Fc, Release, 1)
        aie.use_lock(%nm_FNc, Release, 1)
      }
      aie.end
    }
    %mem_nm = aie.mem(%nm) {
      %s0 = aie.dma_start(S2MM, 0, ^no, ^nrs)
    ^no:
      aie.use_lock(%nm_Op, AcquireGreaterEqual, 1)
      aie.dma_bd(%nm_O : memref<2048xbf16>, 0, 2048)
      aie.use_lock(%nm_Oc, Release, 1)
      aie.next_bd ^no
    ^nrs:
      %s1 = aie.dma_start(S2MM, 1, ^nr, ^nm0)
    ^nr:
      aie.use_lock(%nm_Rp, AcquireGreaterEqual, 1)
      aie.dma_bd(%nm_R : memref<2048xbf16>, 0, 2048)
      aie.use_lock(%nm_Rc, Release, 1)
      aie.next_bd ^ng
    ^ng:
      aie.use_lock(%nm_Gp, AcquireGreaterEqual, 1)
      aie.dma_bd(%nm_gain : memref<2048xbf16>, 0, 2048)
      aie.use_lock(%nm_Gc, Release, 1)
      aie.next_bd ^nr
    ^nm0:
      %m0 = aie.dma_start(MM2S, 0, ^nf, ^nm1)
    ^nf:
      aie.use_lock(%nm_FNc, AcquireGreaterEqual, 1)
      aie.dma_bd_packet(0, 1)
      aie.dma_bd(%nm_FN : memref<2320xbf16>, 0, 2048)
      aie.use_lock(%nm_FNp, Release, 1)
      aie.next_bd ^nf
    ^nm1:
      %m1 = aie.dma_start(MM2S, 1, ^ns, ^nme)
    ^ns:
      aie.use_lock(%nm_Fc, AcquireGreaterEqual, 1)
      aie.dma_bd(%nm_F : memref<2320xbf16>, 0, 2048)
      aie.use_lock(%nm_Fp, Release, 1)
      aie.next_bd ^ns
    ^nme:
      aie.end
    }"""

mux = ""
if RL_FIX:
    # #139b: no relay. mx receives jA(pkt 0, 1024 bf16) + nm(pkt 1, 2048) + jB(pkt 2, 1024 bf16)
    # on S2MM1. Core concatenates a0+a1 into full attn_out then forwards.
    mux = """
    %mx_x = aie.buffer(%mx) {sym_name = "mx_x"} : memref<2320xbf16>
    %mx_a = aie.buffer(%mx) {sym_name = "mx_a"} : memref<2320xbf16>
    %mx_f = aie.buffer(%mx) {sym_name = "mx_f"} : memref<2320xbf16>
    %mx_o = aie.buffer(%mx) {sym_name = "mx_o"} : memref<2320xbf16>
    %mx_xp = aie.lock(%mx, 0) {init = 1 : i32, sym_name = "mx_xp"}
    %mx_xc = aie.lock(%mx, 1) {init = 0 : i32, sym_name = "mx_xc"}
    %mx_a0p = aie.lock(%mx, 2) {init = 1 : i32, sym_name = "mx_a0p"}
    %mx_a0c = aie.lock(%mx, 3) {init = 0 : i32, sym_name = "mx_a0c"}
    %mx_fp = aie.lock(%mx, 4) {init = 1 : i32, sym_name = "mx_fp"}
    %mx_fc = aie.lock(%mx, 5) {init = 0 : i32, sym_name = "mx_fc"}
    %mx_a1p = aie.lock(%mx, 6) {init = 1 : i32, sym_name = "mx_a1p"}
    %mx_a1c = aie.lock(%mx, 7) {init = 0 : i32, sym_name = "mx_a1c"}
    %mx_op = aie.lock(%mx, 8) {init = 1 : i32, sym_name = "mx_op"}
    %mx_oc = aie.lock(%mx, 9) {init = 0 : i32, sym_name = "mx_oc"}
    %core_mx = aie.core(%mx) {
      %z = arith.constant 0 : index
      %N = arith.constant 9223372036854775807 : index
      %one = arith.constant 1 : index
      %nx = arith.constant 2320 : i32
      %ne = arith.constant 2048 : i32
      scf.for %it = %z to %N step %one {
        // phase 1: x_bundle (circuit S2MM0) -> mx_o
        aie.use_lock(%mx_xc, AcquireGreaterEqual, 1)
        aie.use_lock(%mx_op, AcquireGreaterEqual, 1)
        func.call @attn_copy_bf16(%mx_x, %mx_o, %nx) : (memref<2320xbf16>, memref<2320xbf16>, i32) -> ()
        aie.use_lock(%mx_xp, Release, 1)
        aie.use_lock(%mx_oc, Release, 1)
        // phase 2: attn_out — wait BOTH halves (jA pkt0 + jB pkt2) then copy full
        aie.use_lock(%mx_a0c, AcquireGreaterEqual, 1)
        aie.use_lock(%mx_a1c, AcquireGreaterEqual, 1)
        aie.use_lock(%mx_op, AcquireGreaterEqual, 1)
        func.call @attn_copy_bf16(%mx_a, %mx_o, %ne) : (memref<2320xbf16>, memref<2320xbf16>, i32) -> ()
        aie.use_lock(%mx_a0p, Release, 1)
        aie.use_lock(%mx_a1p, Release, 1)
        aie.use_lock(%mx_oc, Release, 1)
        // phase 3: ffn_in (pkt 1, S2MM1) -> mx_o
        aie.use_lock(%mx_fc, AcquireGreaterEqual, 1)
        aie.use_lock(%mx_op, AcquireGreaterEqual, 1)
        func.call @attn_copy_bf16(%mx_f, %mx_o, %ne) : (memref<2320xbf16>, memref<2320xbf16>, i32) -> ()
        aie.use_lock(%mx_fp, Release, 1)
        aie.use_lock(%mx_oc, Release, 1)
      }
      aie.end
    }
    %mem_mx = aie.mem(%mx) {
      %s0 = aie.dma_start(S2MM, 0, ^mxi, ^mxs1)
    ^mxi:
      aie.use_lock(%mx_xp, AcquireGreaterEqual, 1)
      aie.dma_bd(%mx_x : memref<2320xbf16>, 0, 2320)
      aie.use_lock(%mx_xc, Release, 1)
      aie.next_bd ^mxi
    ^mxs1:
      %s1 = aie.dma_start(S2MM, 1, ^mxa0, ^mxm0)
    ^mxa0:
      aie.use_lock(%mx_a0p, AcquireGreaterEqual, 1)
      aie.dma_bd_packet(0, 0)
      aie.dma_bd(%mx_a : memref<2320xbf16>, 0, 1024)
      aie.use_lock(%mx_a0c, Release, 1)
      aie.next_bd ^mxf
    ^mxf:
      aie.use_lock(%mx_fp, AcquireGreaterEqual, 1)
      aie.dma_bd_packet(0, 1)
      aie.dma_bd(%mx_f : memref<2320xbf16>, 0, 2048)
      aie.use_lock(%mx_fc, Release, 1)
      aie.next_bd ^mxa1
    ^mxa1:
      aie.use_lock(%mx_a1p, AcquireGreaterEqual, 1)
      aie.dma_bd_packet(0, 2)
      aie.dma_bd(%mx_a : memref<2320xbf16>, 1024, 1024)
      aie.use_lock(%mx_a1c, Release, 1)
      aie.next_bd ^mxa0
    ^mxm0:
      %m0 = aie.dma_start(MM2S, 0, ^mxo, ^mxe)
    ^mxo:
      aie.use_lock(%mx_oc, AcquireGreaterEqual, 1)
      aie.dma_bd(%mx_o : memref<2320xbf16>, 0, 2320)
      aie.use_lock(%mx_op, Release, 1)
      aie.next_bd ^mxo
    ^mxe:
      aie.end
    }"""
else:
    mux = """
    %mx_x = aie.buffer(%mx) {sym_name = "mx_x"} : memref<2320xbf16>
    %mx_a = aie.buffer(%mx) {sym_name = "mx_a"} : memref<2320xbf16>
    %mx_f = aie.buffer(%mx) {sym_name = "mx_f"} : memref<2320xbf16>
    %mx_o = aie.buffer(%mx) {sym_name = "mx_o"} : memref<2320xbf16>
    %mx_xp = aie.lock(%mx, 0) {init = 1 : i32, sym_name = "mx_xp"}
    %mx_xc = aie.lock(%mx, 1) {init = 0 : i32, sym_name = "mx_xc"}
    %mx_ap = aie.lock(%mx, 2) {init = 1 : i32, sym_name = "mx_ap"}
    %mx_ac = aie.lock(%mx, 3) {init = 0 : i32, sym_name = "mx_ac"}
    %mx_fp = aie.lock(%mx, 4) {init = 1 : i32, sym_name = "mx_fp"}
    %mx_fc = aie.lock(%mx, 5) {init = 0 : i32, sym_name = "mx_fc"}
    %mx_op = aie.lock(%mx, 6) {init = 1 : i32, sym_name = "mx_op"}
    %mx_oc = aie.lock(%mx, 7) {init = 0 : i32, sym_name = "mx_oc"}
    %core_mx = aie.core(%mx) {
      %z = arith.constant 0 : index
      %N = arith.constant 9223372036854775807 : index
      %one = arith.constant 1 : index
      %nx = arith.constant 2320 : i32
      %ne = arith.constant 2048 : i32
      scf.for %it = %z to %N step %one {
        // phase1 bcast: x (2320)
        aie.use_lock(%mx_xc, AcquireGreaterEqual, 1)
        aie.use_lock(%mx_op, AcquireGreaterEqual, 1)
        func.call @attn_copy_bf16(%mx_x, %mx_o, %nx) : (memref<2320xbf16>, memref<2320xbf16>, i32) -> ()
        aie.use_lock(%mx_xp, Release, 1)
        aie.use_lock(%mx_oc, Release, 1)
        // phase2 bcast: attn_out (2048, tail keeps x LUT)
        aie.use_lock(%mx_ac, AcquireGreaterEqual, 1)
        aie.use_lock(%mx_op, AcquireGreaterEqual, 1)
        func.call @attn_copy_bf16(%mx_a, %mx_o, %ne) : (memref<2320xbf16>, memref<2320xbf16>, i32) -> ()
        aie.use_lock(%mx_ap, Release, 1)
        aie.use_lock(%mx_oc, Release, 1)
        // phase3 bcast: ffn_in (2048)
        aie.use_lock(%mx_fc, AcquireGreaterEqual, 1)
        aie.use_lock(%mx_op, AcquireGreaterEqual, 1)
        func.call @attn_copy_bf16(%mx_f, %mx_o, %ne) : (memref<2320xbf16>, memref<2320xbf16>, i32) -> ()
        aie.use_lock(%mx_fp, Release, 1)
        aie.use_lock(%mx_oc, Release, 1)
      }
      aie.end
    }
    %mem_mx = aie.mem(%mx) {
      %s0 = aie.dma_start(S2MM, 0, ^mxi, ^mxs1)
    ^mxi:
      aie.use_lock(%mx_xp, AcquireGreaterEqual, 1)
      aie.dma_bd(%mx_x : memref<2320xbf16>, 0, 2320)
      aie.use_lock(%mx_xc, Release, 1)
      aie.next_bd ^mxi
    ^mxs1:
      %s1 = aie.dma_start(S2MM, 1, ^mxa, ^mxm0)
    ^mxa:
      aie.use_lock(%mx_ap, AcquireGreaterEqual, 1)
      aie.dma_bd_packet(0, 0)
      aie.dma_bd(%mx_a : memref<2320xbf16>, 0, 2048)
      aie.use_lock(%mx_ac, Release, 1)
      aie.next_bd ^mxf
    ^mxf:
      aie.use_lock(%mx_fp, AcquireGreaterEqual, 1)
      aie.dma_bd_packet(0, 1)
      aie.dma_bd(%mx_f : memref<2320xbf16>, 0, 2048)
      aie.use_lock(%mx_fc, Release, 1)
      aie.next_bd ^mxa
    ^mxm0:
      %m0 = aie.dma_start(MM2S, 0, ^mxo, ^mxe)
    ^mxo:
      aie.use_lock(%mx_oc, AcquireGreaterEqual, 1)
      aie.dma_bd(%mx_o : memref<2320xbf16>, 0, 2320)
      aie.use_lock(%mx_op, Release, 1)
      aie.next_bd ^mxo
    ^mxe:
      aie.end
    }"""

prea = """
    %pa_a = aie.buffer(%pa) {sym_name = "pa_a"} : memref<2320xbf16>
    %pa_f = aie.buffer(%pa) {sym_name = "pa_f"} : memref<2320xbf16>
    %pa_o = aie.buffer(%pa) {sym_name = "pa_o"} : memref<2320xbf16>
    %pa_ap = aie.lock(%pa, 0) {init = 1 : i32, sym_name = "pa_ap"}
    %pa_ac = aie.lock(%pa, 1) {init = 0 : i32, sym_name = "pa_ac"}
    %pa_fp = aie.lock(%pa, 2) {init = 1 : i32, sym_name = "pa_fp"}
    %pa_fc = aie.lock(%pa, 3) {init = 0 : i32, sym_name = "pa_fc"}
    %pa_op = aie.lock(%pa, 4) {init = 1 : i32, sym_name = "pa_op"}
    %pa_oc = aie.lock(%pa, 5) {init = 0 : i32, sym_name = "pa_oc"}
    %core_pa = aie.core(%pa) {
      %z = arith.constant 0 : index
      %N = arith.constant 9223372036854775807 : index
      %one = arith.constant 1 : index
      %ne = arith.constant 2048 : i32
      scf.for %it = %z to %N step %one {
        aie.use_lock(%pa_ac, AcquireGreaterEqual, 1)
        aie.use_lock(%pa_op, AcquireGreaterEqual, 1)
        func.call @attn_copy_bf16(%pa_a, %pa_o, %ne) : (memref<2320xbf16>, memref<2320xbf16>, i32) -> ()
        aie.use_lock(%pa_ap, Release, 1)
        aie.use_lock(%pa_oc, Release, 1)
        aie.use_lock(%pa_fc, AcquireGreaterEqual, 1)
        aie.use_lock(%pa_op, AcquireGreaterEqual, 1)
        func.call @attn_copy_bf16(%pa_f, %pa_o, %ne) : (memref<2320xbf16>, memref<2320xbf16>, i32) -> ()
        aie.use_lock(%pa_fp, Release, 1)
        aie.use_lock(%pa_oc, Release, 1)
      }
      aie.end
    }
    %mem_pa = aie.mem(%pa) {
      %s0 = aie.dma_start(S2MM, 0, ^pai, ^pas1)
    ^pai:
      aie.use_lock(%pa_ap, AcquireGreaterEqual, 1)
      aie.dma_bd(%pa_a : memref<2320xbf16>, 0, 2048)
      aie.use_lock(%pa_ac, Release, 1)
      aie.next_bd ^pai
    ^pas1:
      %s1 = aie.dma_start(S2MM, 1, ^paf, ^pam0)
    ^paf:
      aie.use_lock(%pa_fp, AcquireGreaterEqual, 1)
      aie.dma_bd(%pa_f : memref<2320xbf16>, 0, 2048)
      aie.use_lock(%pa_fc, Release, 1)
      aie.next_bd ^paf
    ^pam0:
      %m0 = aie.dma_start(MM2S, 0, ^pao, ^pae)
    ^pao:
      aie.use_lock(%pa_oc, AcquireGreaterEqual, 1)
      aie.dma_bd(%pa_o : memref<2320xbf16>, 0, 2048)
      aie.use_lock(%pa_op, Release, 1)
      aie.next_bd ^pao
    ^pae:
      aie.end
    }"""

shim_allocs = "    aie.shim_dma_allocation @X_alloc(%sh0, MM2S, 1)\n"
shim_allocs += "    aie.shim_dma_allocation @R_alloc(%sh2, MM2S, 1)\n"
shim_allocs += "    aie.shim_dma_allocation @Klo_alloc(%sh3, MM2S, 1)\n"
shim_allocs += "    aie.shim_dma_allocation @Khi_alloc(%sh4, MM2S, 1)\n"
shim_allocs += "    aie.shim_dma_allocation @Vlo_alloc(%sh5, MM2S, 1)\n"
shim_allocs += "    aie.shim_dma_allocation @Vhi_alloc(%sh6, MM2S, 1)\n"
shim_allocs += "    aie.shim_dma_allocation @S_alloc(%sh4, S2MM, 1)\n"
shim_allocs += "".join(
    f'    aie.shim_dma_allocation @A{h}(%sh{h}, MM2S, 0)\n'
    f'    aie.shim_dma_allocation @P{h}(%sh{h}, S2MM, 0)\n' for h in range(NH))

WT_TY = f"{NH*WT_BYTES}xi8"; P_TY = f"{NH*(E+2*KV_M)+E}xbf16"; KV_TY = f"{2*NH*KVN}xbf16"
PH_STRIDE = E + 2*KV_M  # per-head output stride: [Pg, K, V] = 2048+64+64 = 2176 bf16
HALF_KV = 4 * KVN
rt = []
rt.append(f"""      %tx = aiex.dma_configure_task_for @X_alloc {{
        aie.dma_bd(%arg1 : memref<6416xbf16>, 0, 2320, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2320, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%tx)
      %tr = aiex.dma_configure_task_for @R_alloc {{
        aie.dma_bd(%arg1 : memref<6416xbf16>, 2320, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%tr)
      %tklo = aiex.dma_configure_task_for @Klo_alloc {{
        aie.dma_bd(%arg4 : memref<{KV_TY}>, 0, {HALF_KV}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {HALF_KV}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%tklo)
      %tkhi = aiex.dma_configure_task_for @Khi_alloc {{
        aie.dma_bd(%arg4 : memref<{KV_TY}>, {HALF_KV}, {HALF_KV}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {HALF_KV}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%tkhi)
      %tvlo = aiex.dma_configure_task_for @Vlo_alloc {{
        aie.dma_bd(%arg4 : memref<{KV_TY}>, {2*HALF_KV}, {HALF_KV}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {HALF_KV}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%tvlo)
      %tvhi = aiex.dma_configure_task_for @Vhi_alloc {{
        aie.dma_bd(%arg4 : memref<{KV_TY}>, {3*HALF_KV}, {HALF_KV}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {HALF_KV}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%tvhi)""")
for h in range(NH):
    rt.append(f"""      %ta{h} = aiex.dma_configure_task_for @A{h} {{
        aie.dma_bd(%arg2 : memref<{WT_TY}>, {h*WT_BYTES}, {WT_BYTES}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {WT_BYTES}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%ta{h})""")
    rt.append(f"""      %tp{h} = aiex.dma_configure_task_for @P{h} {{
        aie.dma_bd(%arg0 : memref<{P_TY}>, {h*PH_STRIDE}, {PH_STRIDE}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {PH_STRIDE}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%tp{h})""")
rt.append(f"""      %ts = aiex.dma_configure_task_for @S_alloc {{
        aie.dma_bd(%arg0 : memref<{P_TY}>, {NH*PH_STRIDE}, {E}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {E}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%ts)""")
rt.append("".join(f"      aiex.dma_await_task(%tp{h})\n" for h in range(NH)).rstrip()
          + "\n      aiex.dma_await_task(%ts)")
rt_body = "\n".join(rt) + "\n"
rt_args = (f"%arg0: memref<{P_TY}>, %arg1: memref<6416xbf16>, %arg2: memref<{WT_TY}>, "
           f"%arg3: memref<2359296xi8>, %arg4: memref<{KV_TY}>")

centers = "".join(center(h) for h in range(NH))
scores = "".join(score(h) for h in range(NH))
values = "".join(value(h) for h in range(NH))
if RL_FIX:
    joins = (join_memtile('jA', 'jA', with_weight_relay=(0 in RELAY_COLS), packet_id=0)
           + join_memtile('jB', 'jB', with_weight_relay=(1 in RELAY_COLS), packet_id=2)
           + join_memtile('oJ', 'oJ', with_weight_relay=(2 in RELAY_COLS))
           + join_memtile('oK', 'oK', with_weight_relay=(3 in RELAY_COLS)))
else:
    joins = "".join(join_memtile(n, n, with_weight_relay=(c in RELAY_COLS))
                    for c, n in enumerate(['jA', 'jB', 'oJ', 'oK']))
if DECOUPLE:
    ksplit = split_memtile('Klo', 'Klo', with_weight_relay=True) \
           + split_memtile('Khi', 'Khi', with_weight_relay=True) \
           + split_memtile('Vlo', 'Vlo', with_weight_relay=True) \
           + split_memtile('Vhi', 'Vhi', with_weight_relay=True)
else:
    ksplit = split_memtile('Klo', 'Klo') + split_memtile('Khi', 'Khi') + split_memtile('Vlo', 'Vlo') + split_memtile('Vhi', 'Vhi')
_relay_block = "" if RL_FIX else relay
MLIRTXT = (f"module {{\n  aie.device(npu2) {{\n{tile_decls}{shim_decls}\n{funcs}\n"
           f"{flows_txt}\n{centers}\n{scores}\n{values}\n{joins}\n{ksplit}\n{_relay_block}\n{op}\n{nm}\n{mux}\n"
           f"{shim_allocs}\n    aie.runtime_sequence({rt_args}) {{\n{rt_body}    }}\n  }}\n}}\n")


class OFold8F3BestEmitter:
    """Thin validating wrapper around the 1B F3-best MLIR generation."""

    def __init__(self, NH=8, E=2048, G=32, M=4, HD=64, AG=4, SEQ=32, POS=5):
        self.NH, self.E, self.G, self.M = NH, E, G, M
        self.HD, self.AG, self.SEQ, self.POS = HD, AG, SEQ, POS
        self.validate()

    def validate(self):
        if (self.NH, self.E, self.G, self.M, self.HD, self.AG, self.SEQ) != (8, 2048, 32, 4, 64, 4, SEQ):
            raise ValueError('P6.4d-1 emitter is behavior-equivalent for Llama-3.2-1B shape only; '
                             'generalization comes next')

    def emit_mlir(self):
        return MLIRTXT


def emit_mlir(**kwargs):
    return OFold8F3BestEmitter(**kwargs).emit_mlir()
