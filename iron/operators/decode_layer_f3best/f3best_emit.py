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
DN_SUB = 2
DN_T = (E // M) // DN_SUB
WT_TILES = GEMV_T + OPROJ_T + 2 * GU_T + DN_T       # 64+64+512+256 = 896 (Wq|Wo|gate|up|down)
WT_BYTES = WT_TILES * PACKED
O_TILES = E // M
WO_BYTES = O_TILES * PACKED                         # full Wo BO = UNUSED arg3 placeholder (O now in A)
XB = E + 256 + 16
QI = 256
SEQ = 32; KVN = SEQ * HD                            # 2048 per head
POS = 5; INV = np.float32(0.125); L2E = np.float32(1.4453125)

# --- MLIR generation, verbatim from build_ofold8_f3best.py (returns MLIRTXT) ---
CENTER_COLS = [(2, 2), (3, 2), (4, 2), (5, 2), (2, 3), (3, 3), (4, 3), (5, 3)]
SCORE_COLS = [(0, 2), (1, 2), (6, 2), (7, 2), (0, 3), (1, 3), (6, 3), (7, 3)]
VALUE_COLS = [(0, 4), (1, 4), (6, 4), (7, 4), (0, 5), (1, 5), (6, 5), (7, 5)]


def center(h):
    p = f"c{h}"
    # ALL-CIRCUIT outputs (no packet): MM2S0 = Pf -> shim ; MM2S1 = 2-BD [Qi, O_h] -> score (single
    # dest). score relays O_h onward to oJ/oK, so the center never demuxes -> no packet congestion.
    return f"""
    %{p}_A0 = aie.buffer(%t{h}) {{sym_name = "{p}_A0"}} : memref<4608xi8>
    %{p}_A1 = aie.buffer(%t{h}) {{sym_name = "{p}_A1"}} : memref<4608xi8>
    %{p}_B = aie.buffer(%t{h}) {{sym_name = "{p}_B"}} : memref<2320xbf16>
    %{p}_Q = aie.buffer(%t{h}) {{sym_name = "{p}_Q"}} : memref<322xbf16>
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
    %core{h} = aie.core(%t{h}) {{
      %c0 = arith.constant 0 : index
      %cN = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %m4 = arith.constant 4 : i32
      %qr = arith.constant 256 : i32
      %hc = arith.constant 1024 : i32
      %g0 = arith.constant 0 : i32
      %g1 = arith.constant 1 : i32
      %ds = arith.constant 2 : i32
      scf.for %tok = %c0 to %cN step %c1 {{
        // ---- phase 1: Q-GEMV + rope (B = x_bundle, A = Wq 64 tiles) ----
        aie.use_lock(%{p}_Bc, AcquireGreaterEqual, 1)
        aie.use_lock(%{p}_Qp, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %c64 step %c2 {{
          aie.use_lock(%{p}_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          %ro0 = arith.muli %ji0, %m4 : i32
          func.call @fused_dequant_matvec_v2_bf16(%m4, %ro0, %{p}_A0, %{p}_B, %{p}_Q) : (i32, i32, memref<4608xi8>, memref<2320xbf16>, memref<322xbf16>) -> ()
          aie.use_lock(%{p}_A0p, Release, 1)
          aie.use_lock(%{p}_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          %ro1 = arith.muli %ji1, %m4 : i32
          func.call @fused_dequant_matvec_v2_bf16(%m4, %ro1, %{p}_A1, %{p}_B, %{p}_Q) : (i32, i32, memref<4608xi8>, memref<2320xbf16>, memref<322xbf16>) -> ()
          aie.use_lock(%{p}_A1p, Release, 1)
        }}
        func.call @rope_bundled(%{p}_Q, %{p}_B, %{p}_Q, %qr) : (memref<322xbf16>, memref<2320xbf16>, memref<322xbf16>, i32) -> ()
        aie.use_lock(%{p}_Qc, Release, 1)
        aie.use_lock(%{p}_Bp, Release, 1)
        // ---- phase 2: O-proj (B = attn_out 2048, A = Wo 64 tiles -> O[256]) ----
        aie.use_lock(%{p}_Bc, AcquireGreaterEqual, 1)
        aie.use_lock(%{p}_Op, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %c64 step %c2 {{
          aie.use_lock(%{p}_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          %ro0 = arith.muli %ji0, %m4 : i32
          func.call @oproj_matvec_v2_bf16(%m4, %ro0, %{p}_A0, %{p}_B, %{p}_O) : (i32, i32, memref<4608xi8>, memref<2320xbf16>, memref<2048xbf16>) -> ()
          aie.use_lock(%{p}_A0p, Release, 1)
          aie.use_lock(%{p}_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          %ro1 = arith.muli %ji1, %m4 : i32
          func.call @oproj_matvec_v2_bf16(%m4, %ro1, %{p}_A1, %{p}_B, %{p}_O) : (i32, i32, memref<4608xi8>, memref<2320xbf16>, memref<2048xbf16>) -> ()
          aie.use_lock(%{p}_A1p, Release, 1)
        }}
        aie.use_lock(%{p}_Bp, Release, 1)
        aie.use_lock(%{p}_Oc, Release, 1)
        // ---- phase 3: FFN (B = ffn_in, A = gate|up|down) ----
        aie.use_lock(%{p}_Bc, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %c256 step %c2 {{
          aie.use_lock(%{p}_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          %ro0 = arith.muli %ji0, %m4 : i32
          func.call @layer_fused_gate_up_bcast_bf16(%ji0, %g0, %{p}_A0, %{p}_B, %g0) : (i32, i32, memref<4608xi8>, memref<2320xbf16>, i32) -> ()
          aie.use_lock(%{p}_A0p, Release, 1)
          aie.use_lock(%{p}_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          %ro1 = arith.muli %ji1, %m4 : i32
          func.call @layer_fused_gate_up_bcast_bf16(%ji1, %g0, %{p}_A1, %{p}_B, %g0) : (i32, i32, memref<4608xi8>, memref<2320xbf16>, i32) -> ()
          aie.use_lock(%{p}_A1p, Release, 1)
        }}
        scf.for %j = %c0 to %c256 step %c2 {{
          aie.use_lock(%{p}_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          %ro0 = arith.muli %ji0, %m4 : i32
          func.call @layer_fused_gate_up_bcast_bf16(%ji0, %g0, %{p}_A0, %{p}_B, %g1) : (i32, i32, memref<4608xi8>, memref<2320xbf16>, i32) -> ()
          aie.use_lock(%{p}_A0p, Release, 1)
          aie.use_lock(%{p}_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          %ro1 = arith.muli %ji1, %m4 : i32
          func.call @layer_fused_gate_up_bcast_bf16(%ji1, %g0, %{p}_A1, %{p}_B, %g1) : (i32, i32, memref<4608xi8>, memref<2320xbf16>, i32) -> ()
          aie.use_lock(%{p}_A1p, Release, 1)
        }}
        aie.use_lock(%{p}_Bp, Release, 1)
        func.call @layer_fused_silu_mul_static_bf16(%hc) : (i32) -> ()
        aie.use_lock(%{p}_Pp, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %c256 step %c2 {{
          aie.use_lock(%{p}_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @layer_fused_down_bcast_bf16(%ji0, %g0, %{p}_A0, %{p}_P) : (i32, i32, memref<4608xi8>, memref<2320xbf16>) -> ()
          aie.use_lock(%{p}_A0p, Release, 1)
          aie.use_lock(%{p}_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @layer_fused_down_bcast_bf16(%ji1, %g0, %{p}_A1, %{p}_P) : (i32, i32, memref<4608xi8>, memref<2320xbf16>) -> ()
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
    %{p}_K  = aie.buffer(%sc{h}) {{sym_name = "{p}_K"}}  : memref<2048xbf16>
    %{p}_It = aie.buffer(%sc{h}) {{sym_name = "{p}_It"}} : memref<136xbf16>
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
      %s32 = arith.constant 32 : i32
      scf.for %tok = %c0 to %cN step %c1 {{
        // ph1: scoring (Qi from S2MM0-BD0)
        func.call @flowkv_score_init_bf16(%a4) : (i32) -> ()
        aie.use_lock(%{p}_Qc, AcquireGreaterEqual, 1)
        func.call @flowkv_score_rope_q_bf16(%{p}_Qs, %a4, %h64) : (memref<322xbf16>, i32, i32) -> ()
        aie.use_lock(%{p}_Kc, AcquireGreaterEqual, 1)
        aie.use_lock(%{p}_Ip, AcquireGreaterEqual, 1)
        func.call @flowkv_score_chunk_bf16(%{p}_Qs, %{p}_K, %{p}_It, %a4, %h64, %s32) : (memref<322xbf16>, memref<2048xbf16>, memref<136xbf16>, i32, i32, i32) -> ()
        aie.use_lock(%{p}_Kp, Release, 1)
        aie.use_lock(%{p}_Ic, Release, 1)
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
      aie.dma_bd(%{p}_K : memref<2048xbf16>, 0, 2048)
      aie.use_lock(%{p}_Kc, Release, 1)
      aie.next_bd ^k{h}
    ^im{h}:
      %m0 = aie.dma_start(MM2S, 0, ^io{h}, ^ohm{h})
    ^io{h}:
      aie.use_lock(%{p}_Ic, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_It : memref<136xbf16>, 0, 136)
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
    %{p}_Iv = aie.buffer(%va{h}) {{sym_name = "{p}_Iv"}} : memref<136xbf16>
    %{p}_V  = aie.buffer(%va{h}) {{sym_name = "{p}_V"}}  : memref<2048xbf16>
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
      %s32 = arith.constant 32 : i32
      scf.for %tok = %c0 to %cN step %c1 {{
        func.call @flowkv_value_init_bf16(%a4, %h64) : (i32, i32) -> ()
        aie.use_lock(%{p}_Ic, AcquireGreaterEqual, 1)
        aie.use_lock(%{p}_Vc, AcquireGreaterEqual, 1)
        func.call @flowkv_value_accum_bf16(%{p}_Iv, %{p}_V, %a4, %h64, %s32) : (memref<136xbf16>, memref<2048xbf16>, i32, i32, i32) -> ()
        aie.use_lock(%{p}_Ip, Release, 1)
        aie.use_lock(%{p}_Vp, Release, 1)
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
      aie.dma_bd(%{p}_Iv : memref<136xbf16>, 0, 136)
      aie.use_lock(%{p}_Ic, Release, 1)
      aie.next_bd ^iv{h}
    ^vs{h}:
      %s1 = aie.dma_start(S2MM, 1, ^v{h}, ^om{h})
    ^v{h}:
      aie.use_lock(%{p}_Vp, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_V : memref<2048xbf16>, 0, 2048)
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


def join_memtile(name, tile):
    s = f"""
    %{name}_buf = aie.buffer(%{tile}) {{sym_name = "{name}_buf"}} : memref<2048xbf16>"""
    for k in range(4):
        s += f"""
    %{name}_p{k} = aie.lock(%{tile}, {2*k}) {{init = 1 : i32, sym_name = "{name}_p{k}"}}
    %{name}_c{k} = aie.lock(%{tile}, {2*k+1}) {{init = 0 : i32, sym_name = "{name}_c{k}"}}"""
    s += f"""
    %{name}_dma = aie.memtile_dma(%{tile}) {{
      %s0 = aie.dma_start(S2MM, 0, ^{name}g0, ^{name}s1)
    ^{name}g0:
      aie.use_lock(%{name}_p0, AcquireGreaterEqual, 1)
      aie.dma_bd(%{name}_buf : memref<2048xbf16>, 0, {QI})
      aie.use_lock(%{name}_c0, Release, 1)
      aie.next_bd ^{name}g0
    ^{name}s1:
      %s1 = aie.dma_start(S2MM, 1, ^{name}g1, ^{name}s2)
    ^{name}g1:
      aie.use_lock(%{name}_p1, AcquireGreaterEqual, 1)
      aie.dma_bd(%{name}_buf : memref<2048xbf16>, {QI}, {QI})
      aie.use_lock(%{name}_c1, Release, 1)
      aie.next_bd ^{name}g1
    ^{name}s2:
      %s2 = aie.dma_start(S2MM, 2, ^{name}g2, ^{name}s3)
    ^{name}g2:
      aie.use_lock(%{name}_p2, AcquireGreaterEqual, 1)
      aie.dma_bd(%{name}_buf : memref<2048xbf16>, {2*QI}, {QI})
      aie.use_lock(%{name}_c2, Release, 1)
      aie.next_bd ^{name}g2
    ^{name}s3:
      %s3 = aie.dma_start(S2MM, 3, ^{name}g3, ^{name}m0)
    ^{name}g3:
      aie.use_lock(%{name}_p3, AcquireGreaterEqual, 1)
      aie.dma_bd(%{name}_buf : memref<2048xbf16>, {3*QI}, {QI})
      aie.use_lock(%{name}_c3, Release, 1)
      aie.next_bd ^{name}g3
    ^{name}m0:
      %m0 = aie.dma_start(MM2S, 0, ^{name}o0, ^{name}e)
    ^{name}o0:
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
      aie.next_bd ^{name}o0
    ^{name}e:
      aie.end
    }}"""
    return s


def split_memtile(name, tile):
    """1 shim stream (4*2048) -> 4 MM2S slices (2048 each). F3split per-slice locks + 4-BD fill."""
    SL = KVN
    s = f"""
    %{name}_buf = aie.buffer(%{tile}) {{sym_name = "{name}_buf"}} : memref<8192xbf16>"""
    for k in range(4):
        s += f"""
    %{name}_p{k} = aie.lock(%{tile}, {2*k}) {{init = 1 : i32, sym_name = "{name}_p{k}"}}
    %{name}_c{k} = aie.lock(%{tile}, {2*k+1}) {{init = 0 : i32, sym_name = "{name}_c{k}"}}"""
    s += f"""
    %{name}_dma = aie.memtile_dma(%{tile}) {{
      %s0 = aie.dma_start(S2MM, 0, ^{name}f0, ^{name}m0)
    ^{name}f0:
      aie.use_lock(%{name}_p0, AcquireGreaterEqual, 1)
      aie.dma_bd(%{name}_buf : memref<8192xbf16>, 0, {SL})
      aie.use_lock(%{name}_c0, Release, 1)
      aie.next_bd ^{name}f1
    ^{name}f1:
      aie.use_lock(%{name}_p1, AcquireGreaterEqual, 1)
      aie.dma_bd(%{name}_buf : memref<8192xbf16>, {SL}, {SL})
      aie.use_lock(%{name}_c1, Release, 1)
      aie.next_bd ^{name}f2
    ^{name}f2:
      aie.use_lock(%{name}_p2, AcquireGreaterEqual, 1)
      aie.dma_bd(%{name}_buf : memref<8192xbf16>, {2*SL}, {SL})
      aie.use_lock(%{name}_c2, Release, 1)
      aie.next_bd ^{name}f3
    ^{name}f3:
      aie.use_lock(%{name}_p3, AcquireGreaterEqual, 1)
      aie.dma_bd(%{name}_buf : memref<8192xbf16>, {3*SL}, {SL})
      aie.use_lock(%{name}_c3, Release, 1)
      aie.next_bd ^{name}f0
    ^{name}m0:
      %m0 = aie.dma_start(MM2S, 0, ^{name}o0, ^{name}m1)
    ^{name}o0:
      aie.use_lock(%{name}_c0, AcquireGreaterEqual, 1)
      aie.dma_bd(%{name}_buf : memref<8192xbf16>, 0, {SL})
      aie.use_lock(%{name}_p0, Release, 1)
      aie.next_bd ^{name}o0
    ^{name}m1:
      %m1 = aie.dma_start(MM2S, 1, ^{name}o1, ^{name}m2)
    ^{name}o1:
      aie.use_lock(%{name}_c1, AcquireGreaterEqual, 1)
      aie.dma_bd(%{name}_buf : memref<8192xbf16>, {SL}, {SL})
      aie.use_lock(%{name}_p1, Release, 1)
      aie.next_bd ^{name}o1
    ^{name}m2:
      %m2 = aie.dma_start(MM2S, 2, ^{name}o2, ^{name}m3)
    ^{name}o2:
      aie.use_lock(%{name}_c2, AcquireGreaterEqual, 1)
      aie.dma_bd(%{name}_buf : memref<8192xbf16>, {2*SL}, {SL})
      aie.use_lock(%{name}_p2, Release, 1)
      aie.next_bd ^{name}o2
    ^{name}m3:
      %m3 = aie.dma_start(MM2S, 3, ^{name}o3, ^{name}e)
    ^{name}o3:
      aie.use_lock(%{name}_c3, AcquireGreaterEqual, 1)
      aie.dma_bd(%{name}_buf : memref<8192xbf16>, {3*SL}, {SL})
      aie.use_lock(%{name}_p3, Release, 1)
      aie.next_bd ^{name}o3
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
tile_decls += "    %rl = aie.tile(2, 4)\n    %op = aie.tile(3, 4)\n    %nm = aie.tile(4, 4)\n    %mx = aie.tile(5, 4)\n"
# no extra pre-merge tile: all 32 compute tiles are already occupied (router failed with 36th tile).
shim_decls = "".join(f"    %sh{h} = aie.tile({h}, 0)\n" for h in range(NH))

funcs = (
    '    func.func private @fused_dequant_matvec_v2_bf16(i32, i32, memref<4608xi8>, memref<2320xbf16>, memref<322xbf16>) attributes {link_with = "fused_dequant_gemv_v2_signed_2048k_g32.o"}\n'
    '    func.func private @rope_bundled(memref<322xbf16>, memref<2320xbf16>, memref<322xbf16>, i32) attributes {link_with = "rope_il.o"}\n'
    '    func.func private @layer_fused_gate_up_bf16(i32, i32, memref<4608xi8>, memref<2320xbf16>, i32) attributes {link_with = "layer_fused_relay.o"}\n'
    '    func.func private @layer_fused_gate_up_bcast_bf16(i32, i32, memref<4608xi8>, memref<2320xbf16>, i32) attributes {link_with = "layer_fused_relay.o"}\n'
    '    func.func private @layer_fused_silu_mul_static_bf16(i32) attributes {link_with = "layer_fused_relay.o"}\n'
    '    func.func private @layer_fused_down_v2_x4_bf16(i32, i32, i32, memref<4608xi8>, memref<2320xbf16>) attributes {link_with = "layer_fused_relay.o"}\n'
    '    func.func private @layer_fused_down_bcast_bf16(i32, i32, memref<4608xi8>, memref<2320xbf16>) attributes {link_with = "layer_fused_relay.o"}\n'
    '    func.func private @attn_copy_bf16(memref<2320xbf16>, memref<2320xbf16>, i32) attributes {link_with = "attn_concat.o"}\n'
    '    func.func private @oproj_matvec_v2_bf16(i32, i32, memref<4608xi8>, memref<2320xbf16>, memref<2048xbf16>) attributes {link_with = "fused_dequant_gemv_v2_oproj_signed_2048k_g32.o"}\n'
    '    func.func private @layer_fused_add_bf16(memref<2048xbf16>, memref<2048xbf16>, memref<2320xbf16>, i32) attributes {link_with = "layer_fused_relay.o"}\n'
    '    func.func private @layer_fused_rms_norm2_bf16(memref<2320xbf16>, memref<2048xbf16>, memref<2320xbf16>, i32) attributes {link_with = "layer_fused_relay.o"}\n'
    '    func.func private @flowkv_score_init_bf16(i32) attributes {link_with = "flowkv_64d_h4.o"}\n'
    '    func.func private @flowkv_score_rope_q_bf16(memref<322xbf16>, i32, i32) attributes {link_with = "flowkv_64d_h4.o"}\n'
    '    func.func private @flowkv_score_chunk_bf16(memref<322xbf16>, memref<2048xbf16>, memref<136xbf16>, i32, i32, i32) attributes {link_with = "flowkv_64d_h4.o"}\n'
    '    func.func private @flowkv_value_init_bf16(i32, i32) attributes {link_with = "flowkv_64d_h4.o"}\n'
    '    func.func private @flowkv_value_accum_bf16(memref<136xbf16>, memref<2048xbf16>, i32, i32, i32) attributes {link_with = "flowkv_64d_h4.o"}\n'
    '    func.func private @flowkv_value_normalize_bf16(memref<256xbf16>, i32, i32) attributes {link_with = "flowkv_64d_h4.o"}\n')

flows = []
flows.append("    aie.flow(%sh0, DMA : 1, %mx, DMA : 0)   // x_bundle -> mux S2MM0")
# mux S2MM1 needs two producers (attn_out, ffn_in) and no extra compute tile is available.
# Use F3e-1 packet arbitration: rl pkt0 -> mx.mxa, nm pkt1 -> mx.mxf.
flows.append("    aie.packet_flow(0) { aie.packet_source<%rl, DMA : 0> aie.packet_dest<%mx, DMA : 1> }   // attn_out -> mux S2MM1")
flows.append("    aie.packet_flow(1) { aie.packet_source<%nm, DMA : 0> aie.packet_dest<%mx, DMA : 1> }   // ffn_in -> mux S2MM1")
for h in range(NH):
    oj = 'oJ' if h < 4 else 'oK'
    osl = h if h < 4 else h - 4
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
flows.append("    aie.flow(%jA, DMA : 0, %rl, DMA : 0)   // attn Half0 -> relay")
flows.append("    aie.flow(%jB, DMA : 0, %rl, DMA : 1)   // attn Half1 -> relay")
# rl drains attn_out -> mux via packet_flow(16) above (no circuit flow to op)
# O-join: oJ/oK (center O_h) -> op (O-relay) -> nm (ANM)
flows.append("    aie.flow(%oJ, DMA : 0, %op, DMA : 0)   // O Half0 -> orelay")
flows.append("    aie.flow(%oK, DMA : 0, %op, DMA : 1)   // O Half1 -> orelay")
flows.append("    aie.flow(%op, DMA : 0, %nm, DMA : 0)   // O -> ANM")
flows.append("    aie.flow(%sh2, DMA : 1, %nm, DMA : 1)   // resid -> ANM")
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
    %nm_gain = aie.buffer(%nm) {sym_name = "nm_gain"} : memref<2048xbf16> = dense<1.000000e+00>
    %nm_Op = aie.lock(%nm, 0) {init = 1 : i32, sym_name = "nm_Op"}
    %nm_Oc = aie.lock(%nm, 1) {init = 0 : i32, sym_name = "nm_Oc"}
    %nm_Rp = aie.lock(%nm, 2) {init = 1 : i32, sym_name = "nm_Rp"}
    %nm_Rc = aie.lock(%nm, 3) {init = 0 : i32, sym_name = "nm_Rc"}
    %nm_Fp = aie.lock(%nm, 4) {init = 1 : i32, sym_name = "nm_Fp"}
    %nm_Fc = aie.lock(%nm, 5) {init = 0 : i32, sym_name = "nm_Fc"}
    %core_nm = aie.core(%nm) {
      %z = arith.constant 0 : index
      %N = arith.constant 9223372036854775807 : index
      %one = arith.constant 1 : index
      %ne = arith.constant 2048 : i32
      scf.for %it = %z to %N step %one {
        aie.use_lock(%nm_Oc, AcquireGreaterEqual, 1)
        aie.use_lock(%nm_Rc, AcquireGreaterEqual, 1)
        aie.use_lock(%nm_Fp, AcquireGreaterEqual, 1)
        func.call @layer_fused_add_bf16(%nm_O, %nm_R, %nm_F, %ne) : (memref<2048xbf16>, memref<2048xbf16>, memref<2320xbf16>, i32) -> ()
        func.call @layer_fused_rms_norm2_bf16(%nm_F, %nm_gain, %nm_F, %ne) : (memref<2320xbf16>, memref<2048xbf16>, memref<2320xbf16>, i32) -> ()
        aie.use_lock(%nm_Op, Release, 1)
        aie.use_lock(%nm_Rp, Release, 1)
        aie.use_lock(%nm_Fc, Release, 1)
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
      aie.next_bd ^nr
    ^nm0:
      %m0 = aie.dma_start(MM2S, 0, ^nf, ^nme)
    ^nf:
      aie.use_lock(%nm_Fc, AcquireGreaterEqual, 1)
      aie.dma_bd_packet(0, 1)
      aie.dma_bd(%nm_F : memref<2320xbf16>, 0, 2048)
      aie.use_lock(%nm_Fp, Release, 1)
      aie.next_bd ^nf
    ^nme:
      aie.end
    }"""

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
      aie.dma_bd(%mx_a : memref<2320xbf16>, 0, 2048)
      aie.use_lock(%mx_ac, Release, 1)
      aie.next_bd ^mxf
    ^mxf:
      aie.use_lock(%mx_fp, AcquireGreaterEqual, 1)
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
shim_allocs += "".join(
    f'    aie.shim_dma_allocation @A{h}(%sh{h}, MM2S, 0)\n'
    f'    aie.shim_dma_allocation @P{h}(%sh{h}, S2MM, 0)\n' for h in range(NH))

WT_TY = f"{NH*WT_BYTES}xi8"; P_TY = f"{NH*E}xbf16"; KV_TY = f"{2*NH*KVN}xbf16"
HALF_KV = 4 * KVN
rt = []
rt.append(f"""      %tx = aiex.dma_configure_task_for @X_alloc {{
        aie.dma_bd(%arg1 : memref<4368xbf16>, 0, 2320, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2320, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%tx)
      %tr = aiex.dma_configure_task_for @R_alloc {{
        aie.dma_bd(%arg1 : memref<4368xbf16>, 2320, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {{burst_length = 0 : i32}}
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
        aie.dma_bd(%arg0 : memref<{P_TY}>, {h*E}, {E}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {E}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%tp{h})""")
rt.append("".join(f"      aiex.dma_await_task(%tp{h})\n" for h in range(NH)).rstrip())
rt_body = "\n".join(rt) + "\n"
rt_args = (f"%arg0: memref<{P_TY}>, %arg1: memref<4368xbf16>, %arg2: memref<{WT_TY}>, "
           f"%arg3: memref<2359296xi8>, %arg4: memref<{KV_TY}>")

centers = "".join(center(h) for h in range(NH))
scores = "".join(score(h) for h in range(NH))
values = "".join(value(h) for h in range(NH))
joins = (join_memtile('jA', 'jA') + join_memtile('jB', 'jB')
         + join_memtile('oJ', 'oJ') + join_memtile('oK', 'oK'))
ksplit = split_memtile('Klo', 'Klo') + split_memtile('Khi', 'Khi') + split_memtile('Vlo', 'Vlo') + split_memtile('Vhi', 'Vhi')
MLIRTXT = (f"module {{\n  aie.device(npu2) {{\n{tile_decls}{shim_decls}\n{funcs}\n"
           f"{flows_txt}\n{centers}\n{scores}\n{values}\n{joins}\n{ksplit}\n{relay}\n{op}\n{nm}\n{mux}\n"
           f"{shim_allocs}\n    aie.runtime_sequence({rt_args}) {{\n{rt_body}    }}\n  }}\n}}\n")


class OFold8F3BestEmitter:
    """Thin validating wrapper around the 1B F3-best MLIR generation."""

    def __init__(self, NH=8, E=2048, G=32, M=4, HD=64, AG=4, SEQ=32, POS=5):
        self.NH, self.E, self.G, self.M = NH, E, G, M
        self.HD, self.AG, self.SEQ, self.POS = HD, AG, SEQ, POS
        self.validate()

    def validate(self):
        if (self.NH, self.E, self.G, self.M, self.HD, self.AG, self.SEQ) != (8, 2048, 32, 4, 64, 4, 32):
            raise ValueError('P6.4d-1 emitter is behavior-equivalent for Llama-3.2-1B shape only; '
                             'generalization comes next')

    def emit_mlir(self):
        return MLIRTXT


def emit_mlir(**kwargs):
    return OFold8F3BestEmitter(**kwargs).emit_mlir()
