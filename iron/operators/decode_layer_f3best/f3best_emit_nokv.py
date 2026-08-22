#!/usr/bin/env python3
"""Parametric raw-AIE emitter for the F3-best O-fold decode layer (P6.4d-1).

NH=8 tile topology is fixed (4 center columns x 2 rows + 8 score + 8 value + 8
MemTile + 4 hub). All buffer sizes, loop bounds, and weight tile counts are computed
from (E, H, G, M, HD, AG, NH) so the emitter works for any llama-like model whose
E and H are divisible by NH.
"""
import numpy as np
from ml_dtypes import bfloat16
import os as _os


# ── Tile coordinates (fixed NPU2 grid) ──
CENTER_COLS = [(2, 2), (3, 2), (4, 2), (5, 2), (2, 3), (3, 3), (4, 3), (5, 3)]
SCORE_COLS  = [(0, 2), (1, 2), (6, 2), (7, 2), (0, 3), (1, 3), (6, 3), (7, 3)]
VALUE_COLS  = [(0, 4), (1, 4), (6, 4), (7, 4), (0, 5), (1, 5), (6, 5), (7, 5)]


# ═══════════════════════════════════════════════════════════════════════════
#  Auto-allocators for locks and DMA channels (#154)
# ═══════════════════════════════════════════════════════════════════════════

class _LockAlloc:
    """Sequential per-tile lock allocator with overflow validation.

    Usage in a tile-method:
        L = _LockAlloc('core')
        a0_p, a0_c = L.pair()   # -> (0, 1)
        b0_p, b0_c = L.pair()   # -> (2, 3)
        ...
        L.mlir_decl(tref, name, p_id, c_id)  # -> "aie.lock(%t0, 0) {...}..."
    """
    def __init__(self, tile_kind='core'):
        self._next = 0
        self._max = {'core': 16, 'memtile': 64, 'shim': 16}.get(tile_kind, 16)
        self._kind = tile_kind
        self._pairs = {}   # name -> (p_id, c_id)

    def pair(self, name):
        """Allocate lock pair (producer, consumer). Returns (p_id, c_id)."""
        p, c = self._next, self._next + 1
        self._next += 2
        if self._next > self._max:
            raise OverflowError(
                f"Lock overflow on {self._kind} tile: {self._next} > {self._max}"
            )
        self._pairs[name] = (p, c)
        return p, c

    @staticmethod
    def mlir_decl(tile_ref, prefix, name, p_id, c_id):
        """Generate aie.lock MLIR declarations for a buffer's lock pair.

        tile_ref: MLIR tile SSA value (e.g. 't0', 'jA')
        prefix: SSA name prefix (e.g. 'c0', 'sc3', 'jA')
        """
        return (
            f"%{prefix}_{name}p = aie.lock(%{tile_ref}, {p_id}) "
            f'{{init = 1 : i32, sym_name = "{prefix}_{name}p"}}\n'
            f"%{prefix}_{name}c = aie.lock(%{tile_ref}, {c_id}) "
            f'{{init = 0 : i32, sym_name = "{prefix}_{name}c"}}'
        )

    @property
    def used(self):
        return self._next


class _DmaAlloc:
    """Sequential per-tile DMA channel allocator.

    Usage:
        D = _DmaAlloc('core')
        w_ch = D.s2mm()     # -> 0
        b_ch = D.s2mm()     # -> 1
        p_ch = D.mm2s()     # -> 0
    """
    def __init__(self, tile_kind='core'):
        max_ch = {'core': (2, 2), 'memtile': (6, 6), 'shim': (2, 2)}.get(tile_kind, (2, 2))
        self._s2mm_max, self._mm2s_max = max_ch
        self._next_s2mm = 0
        self._next_mm2s = 0
        self._kind = tile_kind
        # track assigned channels
        self.s2mm_chs = {}
        self.mm2s_chs = {}

    def s2mm(self, name=''):
        """Allocate next S2MM channel. Returns channel index."""
        ch = self._next_s2mm
        self._next_s2mm += 1
        if self._next_s2mm > self._s2mm_max:
            raise OverflowError(
                f"S2MM overflow on {self._kind} tile: {self._next_s2mm} > {self._s2mm_max}"
            )
        if name:
            self.s2mm_chs[name] = ch
        return ch

    def mm2s(self, name=''):
        """Allocate next MM2S channel. Returns channel index."""
        ch = self._next_mm2s
        self._next_mm2s += 1
        if self._next_mm2s > self._mm2s_max:
            raise OverflowError(
                f"MM2S overflow on {self._kind} tile: {self._next_mm2s} > {self._mm2s_max}"
            )
        if name:
            self.mm2s_chs[name] = ch
        return ch


class OFold8F3BestEmitter:
    """Parametric validating wrapper around the F3-best MLIR generation."""

    def __init__(self, NH=8, E=2048, H=8192, G=32, M=4, HD=64, AG=4, SEQ=256,
                 POS=5, flowkv_obj_name=None, rope_obj_name=None, relay_obj_name=None):
        # ── primary dimensions ──
        self.NH, self.E, self.H, self.G, self.M = NH, E, H, G, M
        self.HD, self.AG, self.SEQ, self.POS = HD, AG, SEQ, POS

        # ── derived dimensions ──
        self.H8 = H // NH                                 # hidden per tile (1B: 1024)
        self.GROUPS = E // G                              # scale groups per input row (1B: 64)
        self.PER_TILE = E // NH                           # outputs per center tile (1B: 256)

        # ── weight packing ──
        self.PACKED = M * E // 2 + M * self.GROUPS * 2    # bytes per weight tile (1B: 4608)

        # ── per-tile weight tile counts ──
        # Derives from pack_bcast in f3best_pack.h:
        #   tiles = (n_rows/N) * (K_pack/KC), N=32, KC=256.
        # Q/O: n_rows=PER_TILE=E/NH, K_pack=E  -> Q_T = (PER_TILE/32)*(E/256)
        # GU:  n_rows=H8=H/NH,        K_pack=E  -> GU_T = (H8/32)*(E/256)
        # DN:  n_rows=E,              K_pack=H8 -> DN_T = (E/32)*(H8/256)
        _N = 32; _KC = 256
        self.Q_T = (self.PER_TILE // _N) * (self.E // _KC)  # Q/O weight tiles per tile (1B: 64, 3B: 144)
        self.GEMV_T = self.Q_T
        self.OPROJ_T = self.Q_T
        self.GU_T = (self.H8 // _N) * (self.E // _KC)        # gate/up weight tiles (1B: 256, 3B: 384)
        self.DN_T = (self.E // _N) * (self.H8 // _KC)        # down weight tiles (1B: 256, 3B: 384)

        # F3BEST_FFN_DIV probe (divides FFN weight stream)
        _FFN_DIV = int(_os.environ.get('F3BEST_FFN_DIV', '1'))
        assert self.GU_T % _FFN_DIV == 0, 'F3BEST_FFN_DIV must divide GU_T'
        self.GU_T //= _FFN_DIV
        self._ffn_div = _FFN_DIV

        self.WT_TILES = self.GEMV_T + self.OPROJ_T + 2 * self.GU_T + self.DN_T  # (1B no-KV: 896, 3B no-KV: 1440)
        self.WT_BYTES = self.WT_TILES * self.PACKED
        self.O_TILES = E // M
        self.WO_BYTES = self.O_TILES * self.PACKED         # unused arg3 placeholder

        # ── buffer sizes (bf16 elements unless noted) ──
        self.XB = E + AG * HD + 16                         # x_bundle: input + rope LUT (q_rows=AG*HD) + seq meta (1B: 2320, 3B: 2448)
        self.QI = self.PER_TILE                            # quadrant size (1B: 256)
        self.Q_SZ = self.PER_TILE + HD + 2                 # Q buffer (1B: 322)
        # #235: allocation size padded to 32 bf16 (64 B). Q_SZ/ITC are odd
        # (3B: 514/390 elems = 1028/780 B), and when aiecc cannot satisfy
        # bank-aware L1 placement it packs the tile's buffers contiguously --
        # then those odd sizes cascade and push the FOLLOWING buffers off the
        # 32-byte grid. On 3B that landed va*_Of at addr%32==12 while
        # flowkv_value_normalize_bf16 writes it with aie::store_v of a
        # 16xbf16 (32 B) vector, i.e. an unaligned vector store.
        # 1B never showed it only because its K/V buffers are 16 KB, so
        # bank-aware alloc succeeded and every buffer sat on a bank boundary.
        # Padding the ALLOCATION keeps every buffer 64 B-aligned in either
        # placement mode; DMA lengths below stay at the logical Q_SZ/ITC.
        self.Q_SZ_ALLOC = (self.Q_SZ + 31) // 32 * 32      # 1B: 352, 3B: 544
        self.O_SZ = E                                      # O buffer (1B: 2048)
        self.P_SZ = self.XB                                # P buffer = XB (1B: 2320)
        self.HPER = self.H8                                # gate/up/silu buffer (1B: 1024)
        # uni_partial: the wrapper's 128-byte float32 accumulator, written as a
        # PAIR of 64-byte vector ops (vst bmll0/bmlh0). #206/#207: a 192-byte
        # guard band was tried to break adjacency with gate and 64-align the
        # base - it changed the whole L1 map yet left the corruption bit-for-bit
        # identical, so buffer adjacency/alignment is NOT the cause.
        self.UNI_SZ = int(_os.environ.get('F3BEST_UNI_SZ', '128'))

        # ── attention chunking ──
        self.CHUNK = 128
        self.NCHUNK = SEQ // self.CHUNK
        self.KCH = self.CHUNK * HD                         # per-chunk K/V buffer elems (1B: 8192)
        self.ITC = self.CHUNK * AG + 2 * AG                # per-chunk score packet (1B: 520)
        self.ITC_ALLOC = (self.ITC + 31) // 32 * 32        # 1B: 544, 3B: 416 (see Q_SZ_ALLOC)
        self.KVN = SEQ * HD                                # per-head KV buffer (1B: 16384)
        self.FLOWKV_LIB = (
            flowkv_obj_name
            if flowkv_obj_name is not None
            else f"flowkv_{HD}d_h{AG}_c{SEQ}.o"
        )
        self.ROPE_LIB = rope_obj_name if rope_obj_name is not None else "rope_il.o"
        self.RELAY_LIB = (
            relay_obj_name if relay_obj_name is not None else "layer_fused_relay.o"
        )

        # ── XR bundle ──
        self.XR_ELEMS = self.XB + E + E                    # x_bundle + resid + gain (1B: 6416)

        # ── output BO ──
        KV_M = 0                                           # no-KV ABI
        self.OUT_ELEMS = (self.NH + 1) * E                  # NH*P|s (1B: 18432)
        self.PH_STRIDE = E                                  # stride between heads in output (1B: 2048)

        # ── K/V split ──
        self.HALF_KV = 4 * self.KVN                         # 4 heads worth of K/V (1B: 65536)

        # ── env flags ──
        self._read_env_flags()

        # ── L1 budget + auto single/triple-B ──
        self._compute_l1_budget()

        # ── validate ──
        self.validate()

    def _read_env_flags(self):
        """Read environment flags (same semantics as the old module-level code)."""
        _rc = _os.environ.get('F3BEST_MT_RELAY', '').strip()
        self.RELAY_COLS = sorted({int(x) for x in _rc.split(',') if x.strip() != ''}) if _rc else []
        assert all(0 <= c <= 3 for c in self.RELAY_COLS), 'F3BEST_MT_RELAY entries must be 0..3'

        self.RELAY_DEPTH = int(_os.environ.get('F3BEST_MT_DEPTH', '2'))
        assert 2 <= self.RELAY_DEPTH <= 10, 'F3BEST_MT_DEPTH must be 2..10'

        self.DECOUPLE = _os.environ.get('F3BEST_MT_DECOUPLE', '').strip() != ''
        self.RL_FIX = _os.environ.get('F3BEST_RL_FIX', '').strip() != ''
        # #188: hand-asm .s removed (baseline NaN, RR ~500x blow-up); pure C++ bcast path
        self.HANDASM_RR = False

        # triple-B: env overrides auto-detection. '0' = force off; '1' or unset = auto.
        _tb_env = _os.environ.get('F3BEST_TRIPLE_B', '1').strip()
        self._tb_env_forced_off = (_tb_env == '0')

        # #188 Ш21 Direct-NPU Q dump: when set, each score tile copies its received
        # roped-Q (PT elems) into a dedicated Qd buffer and the O_h relay MM2S1 is
        # re-pointed at Qd, so the per-head Q lands in the host-visible s-region of
        # the output BO (the O output is displaced for this diagnostic build).
        self.QDUMP = _os.environ.get('F3BEST_QDUMP', '').strip() != ''
        # #259 SDUMP: dump per-position Q·K scores from the score tile's %It
        # buffer to host via the SAME proven @S_alloc S2MM shim that QDUMP uses
        # (single-shot host task — zero AIE2P runtime risk, unlike #245 tap).
        # After flowkv_score_chunk_bf16 fills %It, copy It→Sd; the ^ohm S2MM BD
        # (which QDUMP re-points at %Qd) is here re-pointed at %Sd. The value
        # tile still receives %It via the ^io MM2S BD — attention computes
        # normally AND scores reach host. Diagnostic-only build (cache key
        # _sdump suffix). Reuses layer_fused_dump_q_bf16 (generic memcpy).
        self.SDUMP = _os.environ.get('F3BEST_SDUMP', '').strip() != ''
        # #211 B0 dump: after phase1 (B0=x_bundle consumed), memcpy B0 -> P and
        # drain P via the existing Pf0 MM2S to the shim s-region. Lets the host
        # read the CONTENTS of B0 as delivered by the mux->center flow, to confirm
        # whether B0 holds x_bundle or stale/wrong-phase data. Only valid for
        # triple-B (single-B has no B0 buffer).
        self.B0DUMP = _os.environ.get('F3BEST_B0DUMP', '').strip() != ''
        # #245 tapped debug-build: when set, the relay (rl) tile copies its
        # attn_out buffer (rl_A, the full E-element pre-O-proj attention output)
        # into a host-visible debug BO via the rl tile's FREE MM2S channel 1 and a
        # free shim S2MM (sh2 ch0). Numerically identical to the production path
        # (rl_A is unchanged; this only ADDS a second MM2S BD copying it host-side).
        # Default off → MLIR identical to production (zero risk to 1B).
        self.ATTN_DUMP = _os.environ.get('F3BEST_ATTN_DUMP', '').strip() != ''
        # #260 AIE2P hardware trace: emit aie.trace ops on score tiles to capture
        # DMA/lock/stream-stall events, routed to a trace shim→DDR BO. Use with
        # F3BEST_SDUMP to debug why the ^ohf→%nm→@S_alloc path stalls (s_blk=0).
        # Parse: python -m aie.utils.trace.parse --input <trace.bin> --mlir <mlir>.
        self.AIE_TRACE = _os.environ.get('F3BEST_AIE_TRACE', '').strip() != ''

    def _score_trace_ops(self, h):
        """Emit aie.trace ops for one score tile %sc{h} (8 DMA/lock/stall events).
        AIE2P Core Tile events (see AIE2P event tables):
          core_dma_mm2s_1_go(47)/stalled(51) — ^ohf MM2S ch1 (_mm2s_oh)
          core_dma_s2mm_0_stalled(48) — ^q S2MM input
          stream_stall_ms0(34)/ms1(35) — output stream backpressure
          core_stall_lock(16) — lock wait. Broadcast start=15/stop=14."""
        if not self.AIE_TRACE:
            return ''
        p = f'sc{h}'
        return f"""
    aie.trace @trace_{p}(%{p}) {{
      aie.trace.mode "Event-Time"
      aie.trace.packet id={3+h} type=core
      aie.trace.event<"STREAM_STALL">
      aie.trace.event<"LOCK_STALL">
      aie.trace.event<"MEMORY_STALL">
      aie.trace.event<"PORT_STALLED_0">
      aie.trace.event<"PORT_STALLED_1">
      aie.trace.event<"PORT_STALLED_2">
      aie.trace.event<"INSTR_LOCK_ACQUIRE_REQ">
      aie.trace.event<"NONE">
      aie.trace.start event = <"TRUE">
      aie.trace.stop event = <"NONE">
    }}"""

    def _compute_l1_budget(self):
        """Compute L1 usage and auto-select single-B vs triple-B (64KB limit)."""
        a_bytes = 2 * self.PACKED
        b_bytes = self.XB * 2                              # bf16 → bytes
        q_bytes = self.Q_SZ_ALLOC * 2
        o_bytes = self.O_SZ * 2
        p_bytes = self.P_SZ * 2
        g_bytes = self.HPER * 2
        u_bytes = self.HPER * 2
        s_bytes = self.HPER * 2
        uni_bytes = self.UNI_SZ

        self._single_l1 = (a_bytes + b_bytes + q_bytes + o_bytes + p_bytes
                          + g_bytes + u_bytes + s_bytes + uni_bytes)
        self._triple_l1 = (a_bytes + 3 * b_bytes + q_bytes + o_bytes + p_bytes
                          + g_bytes + u_bytes + s_bytes + uni_bytes)

        # Auto-select: triple-B only if it fits (60KB margin) and env didn't force off
        if self._tb_env_forced_off:
            self.TRIPLE_B = False
        elif self._triple_l1 <= 60 * 1024:
            self.TRIPLE_B = True
        else:
            self.TRIPLE_B = False

    def validate(self):
        """Validate dimensions for the NH=8 topology."""
        if self.E % self.NH != 0:
            raise ValueError(f'E ({self.E}) must be divisible by NH ({self.NH})')
        if self.H % self.NH != 0:
            raise ValueError(f'H ({self.H}) must be divisible by NH ({self.NH})')
        if self.H8 % self.M != 0:
            raise ValueError(f'H/NH ({self.H8}) must be divisible by M ({self.M})')
        if self.PER_TILE % self.M != 0:
            raise ValueError(f'E/NH ({self.PER_TILE}) must be divisible by M ({self.M})')
        if self._single_l1 > 64 * 1024:
            raise ValueError(f'Single-B L1 ({self._single_l1}B) exceeds 64KB tile limit')
        if self._ffn_div != 1 and self.GU_T != self.DN_T:
            raise ValueError('FFN_DIV must produce GU_T == DN_T')

    # ═══════════════════════════════════════════════════════════════════════
    #  MLIR generators (methods)
    # ═══════════════════════════════════════════════════════════════════════

    def _center(self, h):
        p = f"c{h}"
        if self.TRIPLE_B:
            return self._center_triple_b(h, p)
        return self._center_single_b(h, p)

    def _center_triple_b(self, h, p):
        """Triple B-buffer: B0(x_bundle), B1(attn_out), B2(ffn_in)."""
        # QSZ/ITC name the memref TYPE (= L1 allocation, 64 B-aligned); the
        # QSZL/ITCL suffix is the logical DMA length. See Q_SZ_ALLOC.
        E = self.E; PK = self.PACKED; XB = self.XB; QSZ = self.Q_SZ_ALLOC
        QSZL = self.Q_SZ
        OSZ = self.O_SZ; PSZ = self.P_SZ; HP = self.HPER; GT = self.GU_T
        PT = self.PER_TILE; UNI = self.UNI_SZ

        # Auto-allocate locks and DMA channels (#154)
        L = _LockAlloc('core')
        D = _DmaAlloc('core')
        _a0 = L.pair('A0'); _b = L.pair('B')  # single B lock pair shared across B0/B1/B2 (sequential phases)
        # // #271: separate Q lock pairs for three sync paths:
        #   _q_c2m: center core -> center mem (Q ready for DMA to score)
        #   _q_c2s: center core -> score core (Q ready for score tile)
        _q_c2m = L.pair('Qc2m'); _q_c2s = L.pair('Qc2s')
        _o = L.pair('O');   _p = L.pair('P');   _a1 = L.pair('A1')
        _s2mm_w  = D.s2mm('weight')   # A0/A1 ping-pong
        _s2mm_b  = D.s2mm('bcast')    # B0/B1/B2 triple
        _mm2s_p  = D.mm2s('drain')    # P drain
        _mm2s_qo = D.mm2s('qo')       # Q+O drain

        # Lock declarations (auto-allocated)
        _ld = lambda name, ids: _LockAlloc.mlir_decl(f't{h}', f'{p}', name, *ids)
        _lock_decls = '\n'.join([
            _ld('A0', _a0), _ld('B', _b), _ld('Qc2m', _q_c2m), _ld('Qc2s', _q_c2s),
            _ld('O', _o), _ld('P', _p), _ld('A1', _a1),
        ])

        return f"""
    %{p}_A0 = aie.buffer(%t{h}) {{sym_name = "{p}_A0"}} : memref<{PK}xi8>
    %{p}_A1 = aie.buffer(%t{h}) {{sym_name = "{p}_A1"}} : memref<{PK}xi8>
    %{p}_B0 = aie.buffer(%t{h}) {{sym_name = "{p}_B0"}} : memref<{XB}xbf16>
    %{p}_B1 = aie.buffer(%t{h}) {{sym_name = "{p}_B1"}} : memref<{XB}xbf16>
    %{p}_B2 = aie.buffer(%t{h}) {{sym_name = "{p}_B2"}} : memref<{XB}xbf16>
    %{p}_Q = aie.buffer(%t{h}) {{sym_name = "{p}_Q"}} : memref<{QSZ}xbf16>
    %{p}_O = aie.buffer(%t{h}) {{sym_name = "{p}_O"}} : memref<{OSZ}xbf16>
    %{p}_P = aie.buffer(%t{h}) {{sym_name = "{p}_P"}} : memref<{PSZ}xbf16>
    {_lock_decls}
    %core{h} = aie.core(%t{h}) {{
      %c0 = arith.constant 0 : index
      %cN = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant {self.GEMV_T} : index
      %cF = arith.constant {GT} : index
      %c4 = arith.constant 4 : i32
      %c8 = arith.constant 8 : i32
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %qr = arith.constant {PT} : i32
      %hc = arith.constant {HP} : i32
      %xb_i32 = arith.constant {XB} : i32
      scf.for %tok = %c0 to %cN step %c1 {{
        // ---- phase 1: Q-GEMV + rope (B0 = x_bundle, nchunk=8) ----
        aie.use_lock(%{p}_Bc, AcquireGreaterEqual, 1)
        // #271: wait for score tile to finish reading previous layer's Q (Qc2s consumer)
        aie.use_lock(%{p}_Qc2sc, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %c64 step %c2 {{
          aie.use_lock(%{p}_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @layer_fused_qkv_bcast_bf16(%ji0, %c0_i32, %{p}_A0, %{p}_B0, %{p}_Q) : (i32, i32, memref<{PK}xi8>, memref<{XB}xbf16>, memref<{QSZ}xbf16>) -> ()
          aie.use_lock(%{p}_A0p, Release, 1)
          aie.use_lock(%{p}_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @layer_fused_qkv_bcast_bf16(%ji1, %c0_i32, %{p}_A1, %{p}_B0, %{p}_Q) : (i32, i32, memref<{PK}xi8>, memref<{XB}xbf16>, memref<{QSZ}xbf16>) -> ()
          aie.use_lock(%{p}_A1p, Release, 1)
        }}
        func.call @rope_bundled(%{p}_Q, %{p}_B0, %{p}_Q, %qr) : (memref<{QSZ}xbf16>, memref<{XB}xbf16>, memref<{QSZ}xbf16>, i32) -> ()
        // #271: release Q for BOTH center mem (DMA, Qc2m producer) AND score tile (processing, Qc2s producer)
        aie.use_lock(%{p}_Qc2mp, Release, 1)
        aie.use_lock(%{p}_Qc2sp, Release, 1)
        aie.use_lock(%{p}_Bp, Release, 1)
        // #211 B0 dump: snapshot B0 (x_bundle as delivered by mux->center) into P,
        // then drain P via the existing Pf0 MM2S to the shim s-region. The host
        // reads P and compares TB=0 vs TB=1 B0 contents. Reuses attn_copy_bf16
        // (already linked via attn_concat.o) and the P buffer/Pf0 drain - no new
        // kernel, DMA channel, or lock pair. Pp/Pc are recycled: the dump fills P
        // (Pc=1), Pf0 drains it (releases Pp), and phase3d later re-acquires Pp.
        """ + (f"""
        aie.use_lock(%{p}_Pp, AcquireGreaterEqual, 1)
        func.call @attn_copy_bf16(%{p}_B0, %{p}_P, %xb_i32) : (memref<{XB}xbf16>, memref<{XB}xbf16>, i32) -> ()
        aie.use_lock(%{p}_Pc, Release, 1)
        """ if self.B0DUMP else "") + f"""
        // ---- phase 2: O-proj (B1 = attn_out, nchunk=8) ----
        aie.use_lock(%{p}_Bc, AcquireGreaterEqual, 1)
        aie.use_lock(%{p}_Op, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %c64 step %c2 {{
          aie.use_lock(%{p}_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @layer_fused_oproj_bcast_bf16(%ji0, %c0_i32, %{p}_A0, %{p}_B1, %{p}_O) : (i32, i32, memref<{PK}xi8>, memref<{XB}xbf16>, memref<{OSZ}xbf16>) -> ()
          aie.use_lock(%{p}_A0p, Release, 1)
          aie.use_lock(%{p}_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @layer_fused_oproj_bcast_bf16(%ji1, %c0_i32, %{p}_A1, %{p}_B1, %{p}_O) : (i32, i32, memref<{PK}xi8>, memref<{XB}xbf16>, memref<{OSZ}xbf16>) -> ()
          aie.use_lock(%{p}_A1p, Release, 1)
        }}
        aie.use_lock(%{p}_Bp, Release, 1)
        aie.use_lock(%{p}_Oc, Release, 1)
        // ---- phase 3: FFN (B2 = ffn_in) ----
        // 3a: gate GEMV (nchunk=8, output -> gate_buf)
        aie.use_lock(%{p}_Bc, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %cF step %c2 {{
          aie.use_lock(%{p}_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @layer_fused_gate_up_bcast_bf16(%ji0, %c0_i32, %{p}_A0, %{p}_B2, %c0_i32) : (i32, i32, memref<{PK}xi8>, memref<{XB}xbf16>, i32) -> ()
          aie.use_lock(%{p}_A0p, Release, 1)
          aie.use_lock(%{p}_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @layer_fused_gate_up_bcast_bf16(%ji1, %c0_i32, %{p}_A1, %{p}_B2, %c0_i32) : (i32, i32, memref<{PK}xi8>, memref<{XB}xbf16>, i32) -> ()
          aie.use_lock(%{p}_A1p, Release, 1)
        }}
        // 3b: up GEMV (nchunk=8, output -> up_buf)
        scf.for %j = %c0 to %cF step %c2 {{
          aie.use_lock(%{p}_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @layer_fused_gate_up_bcast_bf16(%ji0, %c0_i32, %{p}_A0, %{p}_B2, %c1_i32) : (i32, i32, memref<{PK}xi8>, memref<{XB}xbf16>, i32) -> ()
          aie.use_lock(%{p}_A0p, Release, 1)
          aie.use_lock(%{p}_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @layer_fused_gate_up_bcast_bf16(%ji1, %c0_i32, %{p}_A1, %{p}_B2, %c1_i32) : (i32, i32, memref<{PK}xi8>, memref<{XB}xbf16>, i32) -> ()
          aie.use_lock(%{p}_A1p, Release, 1)
        }}
        aie.use_lock(%{p}_Bp, Release, 1)
        // 3c: SiLU(gate) * up -> silu_buf.
        // The C++ bcast wrappers keep gate/up/silu in the file-scope statics
        // lf_left_buf / lf_right_buf / lf_silu_buf: gate_up writes the statics
        // and down reads lf_silu_buf. Calling the explicit-pointer SiLU on the
        // IRON buffers would read gate/up nobody ever wrote and leave down on
        // the raw (un-SiLU'd) gate - so use the static variant that matches.
        func.call @layer_fused_silu_mul_static_bf16(%hc) : (i32) -> ()
        // 3d: down GEMV (nchunk=4, activation = silu_buf)
        aie.use_lock(%{p}_Pp, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %cF step %c2 {{
          aie.use_lock(%{p}_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @layer_fused_down_bcast_bf16(%ji0, %c0_i32, %{p}_A0, %{p}_P) : (i32, i32, memref<{PK}xi8>, memref<{PSZ}xbf16>) -> ()
          aie.use_lock(%{p}_A0p, Release, 1)
          aie.use_lock(%{p}_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @layer_fused_down_bcast_bf16(%ji1, %c0_i32, %{p}_A1, %{p}_P) : (i32, i32, memref<{PK}xi8>, memref<{PSZ}xbf16>) -> ()
          aie.use_lock(%{p}_A1p, Release, 1)
        }}
        aie.use_lock(%{p}_Pc, Release, 1)
      }}
      aie.end
    }}
    %mem_t{h} = aie.mem(%t{h}) {{
      %s0 = aie.dma_start(S2MM, {_s2mm_w}, ^a0{h}, ^bs{h})
    ^a0{h}:
      aie.use_lock(%{p}_A0p, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_A0 : memref<{PK}xi8>, 0, {PK})
      aie.use_lock(%{p}_A0c, Release, 1)
      aie.next_bd ^a1{h}
    ^a1{h}:
      aie.use_lock(%{p}_A1p, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_A1 : memref<{PK}xi8>, 0, {PK})
      aie.use_lock(%{p}_A1c, Release, 1)
      aie.next_bd ^a0{h}
    ^bs{h}:
      %s1 = aie.dma_start(S2MM, {_s2mm_b}, ^b0{h}, ^m0{h})
    ^b0{h}:
      aie.use_lock(%{p}_Bp, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_B0 : memref<{XB}xbf16>, 0, {XB})
      aie.use_lock(%{p}_Bc, Release, 1)
      aie.next_bd ^b1{h}
    ^b1{h}:
      aie.use_lock(%{p}_Bp, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_B1 : memref<{XB}xbf16>, 0, {XB})
      aie.use_lock(%{p}_Bc, Release, 1)
      aie.next_bd ^b2{h}
    ^b2{h}:
      aie.use_lock(%{p}_Bp, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_B2 : memref<{XB}xbf16>, 0, {XB})
      aie.use_lock(%{p}_Bc, Release, 1)
      aie.next_bd ^b0{h}
    ^m0{h}:
      %m0 = aie.dma_start(MM2S, {_mm2s_p}, ^pf{h}, ^qo{h})
    ^pf{h}:
      aie.use_lock(%{p}_Pc, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_P : memref<{PSZ}xbf16>, 0, {E})
      aie.use_lock(%{p}_Pp, Release, 1)
      aie.next_bd ^pf{h}
    ^qo{h}:
      %m1 = aie.dma_start(MM2S, {_mm2s_qo}, ^qi{h}, ^e{h})
    ^qi{h}:
      // #271: wait for center core to signal Q ready for DMA (Qc2m consumer)
      aie.use_lock(%{p}_Qc2mc, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_Q : memref<{QSZ}xbf16>, 0, {QSZL})
      // #271: signal center core that DMA is done (Qc2m producer)
      aie.use_lock(%{p}_Qc2mp, Release, 1)
      aie.next_bd ^oh{h}
    ^oh{h}:
      aie.use_lock(%{p}_Oc, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_O : memref<{OSZ}xbf16>, 0, {PT})
      aie.use_lock(%{p}_Op, Release, 1)
      aie.next_bd ^qi{h}
    ^e{h}:
      aie.end
    }}"""

    def _center_single_b(self, h, p):
        """Single B-buffer: B = shared activation (x_bundle / attn_out / ffn_in)."""
        E = self.E; PK = self.PACKED; XB = self.XB; QSZ = self.Q_SZ_ALLOC
        QSZL = self.Q_SZ
        OSZ = self.O_SZ; PSZ = self.P_SZ; HP = self.HPER; GT = self.GU_T
        PT = self.PER_TILE; UNI = self.UNI_SZ

        # Auto-allocate locks and DMA channels (#154)
        L = _LockAlloc('core')
        D = _DmaAlloc('core')
        _a0 = L.pair('A0'); _b = L.pair('B')
        _o = L.pair('O');   _p = L.pair('P'); _a1 = L.pair('A1')
        _s2mm_w  = D.s2mm('weight')
        _s2mm_b  = D.s2mm('bcast')
        _mm2s_p  = D.mm2s('drain')
        _mm2s_qo = D.mm2s('qo')

        # // #271: separate Q lock pairs for center->mem and center->score
        _q_c2m = L.pair('Qc2m'); _q_c2s = L.pair('Qc2s')

        _ld = lambda name, ids: _LockAlloc.mlir_decl(f't{h}', f'{p}', name, *ids)
        _lock_decls = '\n'.join([
            _ld('A0', _a0), _ld('B', _b), _ld('Qc2m', _q_c2m), _ld('Qc2s', _q_c2s),
            _ld('O', _o), _ld('P', _p), _ld('A1', _a1),
        ])

        return f"""
    %{p}_A0 = aie.buffer(%t{h}) {{sym_name = "{p}_A0"}} : memref<{PK}xi8>
    %{p}_A1 = aie.buffer(%t{h}) {{sym_name = "{p}_A1"}} : memref<{PK}xi8>
    %{p}_B = aie.buffer(%t{h}) {{sym_name = "{p}_B"}} : memref<{XB}xbf16>
    %{p}_Q = aie.buffer(%t{h}) {{sym_name = "{p}_Q"}} : memref<{QSZ}xbf16>
    %{p}_O = aie.buffer(%t{h}) {{sym_name = "{p}_O"}} : memref<{OSZ}xbf16>
    %{p}_P = aie.buffer(%t{h}) {{sym_name = "{p}_P"}} : memref<{PSZ}xbf16>
    {_lock_decls}
    %core{h} = aie.core(%t{h}) {{
      %c0 = arith.constant 0 : index
      %cN = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant {self.GEMV_T} : index
      %cF = arith.constant {GT} : index
      %c4 = arith.constant 4 : i32
      %c8 = arith.constant 8 : i32
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %qr = arith.constant {PT} : i32
      %hc = arith.constant {HP} : i32
      %xb_i32 = arith.constant {XB} : i32
      scf.for %tok = %c0 to %cN step %c1 {{
        // ---- phase 1: Q-GEMV + rope (B = x_bundle, nchunk=8) ----
        aie.use_lock(%{p}_Bc, AcquireGreaterEqual, 1)
        // #271: wait for score tile to finish reading previous layer's Q (Qc2s consumer)
        aie.use_lock(%{p}_Qc2sc, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %c64 step %c2 {{
          aie.use_lock(%{p}_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @layer_fused_qkv_bcast_bf16(%ji0, %c0_i32, %{p}_A0, %{p}_B, %{p}_Q) : (i32, i32, memref<{PK}xi8>, memref<{XB}xbf16>, memref<{QSZ}xbf16>) -> ()
          aie.use_lock(%{p}_A0p, Release, 1)
          aie.use_lock(%{p}_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @layer_fused_qkv_bcast_bf16(%ji1, %c0_i32, %{p}_A1, %{p}_B, %{p}_Q) : (i32, i32, memref<{PK}xi8>, memref<{XB}xbf16>, memref<{QSZ}xbf16>) -> ()
          aie.use_lock(%{p}_A1p, Release, 1)
        }}
        func.call @rope_bundled(%{p}_Q, %{p}_B, %{p}_Q, %qr) : (memref<{QSZ}xbf16>, memref<{XB}xbf16>, memref<{QSZ}xbf16>, i32) -> ()
        // #271: release Q for BOTH center mem (DMA, Qc2m producer) AND score tile (processing, Qc2s producer)
        aie.use_lock(%{p}_Qc2mp, Release, 1)
        aie.use_lock(%{p}_Qc2sp, Release, 1)
        aie.use_lock(%{p}_Bp, Release, 1)
        // #211 B0 dump (single-B): snapshot B (x_bundle as delivered by mux->center)
        // into P, then drain P via Pf0 MM2S. Mirrors the triple-B dump so the host
        // can compare TB=0 vs TB=1 B0 contents from the same code path.
        """ + (f"""
        aie.use_lock(%{p}_Pp, AcquireGreaterEqual, 1)
        func.call @attn_copy_bf16(%{p}_B, %{p}_P, %xb_i32) : (memref<{XB}xbf16>, memref<{XB}xbf16>, i32) -> ()
        aie.use_lock(%{p}_Pc, Release, 1)
        """ if self.B0DUMP else "") + f"""
        // ---- phase 2: O-proj broadcast (B = attn_out, nchunk=8) ----
        aie.use_lock(%{p}_Bc, AcquireGreaterEqual, 1)
        aie.use_lock(%{p}_Op, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %c64 step %c2 {{
          aie.use_lock(%{p}_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @layer_fused_oproj_bcast_bf16(%ji0, %c0_i32, %{p}_A0, %{p}_B, %{p}_O) : (i32, i32, memref<{PK}xi8>, memref<{XB}xbf16>, memref<{OSZ}xbf16>) -> ()
          aie.use_lock(%{p}_A0p, Release, 1)
          aie.use_lock(%{p}_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @layer_fused_oproj_bcast_bf16(%ji1, %c0_i32, %{p}_A1, %{p}_B, %{p}_O) : (i32, i32, memref<{PK}xi8>, memref<{XB}xbf16>, memref<{OSZ}xbf16>) -> ()
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
          func.call @layer_fused_gate_up_bcast_bf16(%ji0, %c0_i32, %{p}_A0, %{p}_B, %c0_i32) : (i32, i32, memref<{PK}xi8>, memref<{XB}xbf16>, i32) -> ()
          aie.use_lock(%{p}_A0p, Release, 1)
          aie.use_lock(%{p}_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @layer_fused_gate_up_bcast_bf16(%ji1, %c0_i32, %{p}_A1, %{p}_B, %c0_i32) : (i32, i32, memref<{PK}xi8>, memref<{XB}xbf16>, i32) -> ()
          aie.use_lock(%{p}_A1p, Release, 1)
        }}
        // 3b: up GEMV (nchunk=8, output -> up_buf)
        scf.for %j = %c0 to %cF step %c2 {{
          aie.use_lock(%{p}_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @layer_fused_gate_up_bcast_bf16(%ji0, %c0_i32, %{p}_A0, %{p}_B, %c1_i32) : (i32, i32, memref<{PK}xi8>, memref<{XB}xbf16>, i32) -> ()
          aie.use_lock(%{p}_A0p, Release, 1)
          aie.use_lock(%{p}_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @layer_fused_gate_up_bcast_bf16(%ji1, %c0_i32, %{p}_A1, %{p}_B, %c1_i32) : (i32, i32, memref<{PK}xi8>, memref<{XB}xbf16>, i32) -> ()
          aie.use_lock(%{p}_A1p, Release, 1)
        }}
        aie.use_lock(%{p}_Bp, Release, 1)
        // 3c: SiLU(gate) * up -> silu_buf (statics - see the triple-B center).
        func.call @layer_fused_silu_mul_static_bf16(%hc) : (i32) -> ()
        // 3d: down GEMV (nchunk=4, activation = silu_buf)
        aie.use_lock(%{p}_Pp, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %cF step %c2 {{
          aie.use_lock(%{p}_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @layer_fused_down_bcast_bf16(%ji0, %c0_i32, %{p}_A0, %{p}_P) : (i32, i32, memref<{PK}xi8>, memref<{PSZ}xbf16>) -> ()
          aie.use_lock(%{p}_A0p, Release, 1)
          aie.use_lock(%{p}_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @layer_fused_down_bcast_bf16(%ji1, %c0_i32, %{p}_A1, %{p}_P) : (i32, i32, memref<{PK}xi8>, memref<{PSZ}xbf16>) -> ()
          aie.use_lock(%{p}_A1p, Release, 1)
        }}
        aie.use_lock(%{p}_Pc, Release, 1)
      }}
      aie.end
    }}
    %mem{h} = aie.mem(%t{h}) {{
      %s0 = aie.dma_start(S2MM, {_s2mm_w}, ^a0{h}, ^bs{h})
    ^a0{h}:
      aie.use_lock(%{p}_A0p, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_A0 : memref<{PK}xi8>, 0, {PK})
      aie.use_lock(%{p}_A0c, Release, 1)
      aie.next_bd ^a1{h}
    ^a1{h}:
      aie.use_lock(%{p}_A1p, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_A1 : memref<{PK}xi8>, 0, {PK})
      aie.use_lock(%{p}_A1c, Release, 1)
      aie.next_bd ^a0{h}
    ^bs{h}:
      %s1 = aie.dma_start(S2MM, {_s2mm_b}, ^b{h}, ^m0{h})
    ^b{h}:
      aie.use_lock(%{p}_Bp, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_B : memref<{XB}xbf16>, 0, {XB})
      aie.use_lock(%{p}_Bc, Release, 1)
      aie.next_bd ^b{h}
    ^m0{h}:
      %m0 = aie.dma_start(MM2S, {_mm2s_p}, ^pf{h}, ^m1{h})
    ^pf{h}:
      aie.use_lock(%{p}_Pc, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_P : memref<{PSZ}xbf16>, 0, {E})
      aie.use_lock(%{p}_Pp, Release, 1)
      aie.next_bd ^pf{h}
    ^m1{h}:
      %m1 = aie.dma_start(MM2S, {_mm2s_qo}, ^qo{h}, ^e{h})
    ^qo{h}:
      // #271: wait for center core to signal Q ready for DMA (Qc2m consumer)
      aie.use_lock(%{p}_Qc2mc, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_Q : memref<{QSZ}xbf16>, 0, {QSZL})
      // #271: signal center core that DMA is done (Qc2m producer)
      aie.use_lock(%{p}_Qc2mp, Release, 1)
      aie.next_bd ^oh{h}
    ^oh{h}:
      aie.use_lock(%{p}_Oc, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_O : memref<{OSZ}xbf16>, 0, {PT})
      aie.use_lock(%{p}_Op, Release, 1)
      aie.next_bd ^qo{h}
    ^e{h}:
      aie.end
    }}"""

    def _score(self, h):
        p = f"sc{h}"
        KCH = self.KCH; ITC = self.ITC_ALLOC; QSZ = self.Q_SZ_ALLOC
        QSZL = self.Q_SZ; ITCL = self.ITC
        NC = self.NCHUNK; CH = self.CHUNK; PT = self.PER_TILE

        # Auto-allocate locks and DMA channels (#154)
        L = _LockAlloc('core'); D = _DmaAlloc('core')
        # // #271: separate Q lock pairs for center->score and score->mem
        _q_c2s = L.pair('Qc2s'); _q_s2m = L.pair('Qs2m')
        _k = L.pair('K'); _i = L.pair('I')
        _oh = L.pair('Oh'); _ohd = L.pair('Ohd')
        _s2mm_in  = D.s2mm('in')   # Q + Oh relay (2-BD chain)
        _s2mm_kv  = D.s2mm('kv')   # K
        _mm2s_out = D.mm2s('out')  # inter (I) to value
        _mm2s_oh  = D.mm2s('oh')   # Oh relay to join

        _ld = lambda name, ids: _LockAlloc.mlir_decl(f'sc{h}', f'{p}', name, *ids)
        if self.SDUMP:
            # #259 score dump: one extra lock pair gates the Sd relay buffer (the
            # score tile has 16/16 locks with 5 pairs used -> 3 pairs headroom; no
            # extra DMA channel — the ^ohm S2MM BD is re-pointed at Sd, like QDUMP
            # re-points it at Qd). @S_alloc proven single-shot, zero runtime risk.
            _sd = L.pair('Sd')
            _lock_decls = '\n'.join([
                _ld('Qc2s', _q_c2s), _ld('Qs2m', _q_s2m), _ld('K', _k), _ld('I', _i),
                _ld('Oh', _oh), _ld('Ohd', _ohd), _ld('Sd', _sd),
            ])
        elif self.QDUMP:
            # #188 Q dump: one extra lock pair gates the Qd relay buffer (the score
            # tile has 16/16 locks with 5 pairs used -> 3 pairs headroom; no extra
            # DMA channel is allocated - the O_h relay MM2S1 is re-pointed at Qd).
            _qd = L.pair('Qd')
            _lock_decls = '\n'.join([
                _ld('Qc2s', _q_c2s), _ld('Qs2m', _q_s2m), _ld('K', _k), _ld('I', _i),
                _ld('Oh', _oh), _ld('Ohd', _ohd), _ld('Qd', _qd),
            ])
        else:
            _lock_decls = '\n'.join([
                _ld('Qc2s', _q_c2s), _ld('Qs2m', _q_s2m), _ld('K', _k), _ld('I', _i),
                _ld('Oh', _oh), _ld('Ohd', _ohd),
            ])

        if self.SDUMP:
            # #259 score-dump variant: after each score_chunk fills %It, copy It→Sd
            # and gate the ^ohm S2MM relay on Sd (instead of Oh). The value tile
            # still receives %It via the ^io MM2S BD — attention computes normally
            # AND per-chunk Q·K scores reach host via @S_alloc. Ohc/Ohp cycled so
            # S2MM0-BD1 (core O) never stalls; its data is discarded.
            return f"""
    %{p}_Qs = aie.buffer(%sc{h}) {{sym_name = "{p}_Qs"}} : memref<{QSZ}xbf16>
    %{p}_K  = aie.buffer(%sc{h}) {{sym_name = "{p}_K"}}  : memref<{KCH}xbf16>
    %{p}_It = aie.buffer(%sc{h}) {{sym_name = "{p}_It"}} : memref<{ITC}xbf16>
    %{p}_Oh = aie.buffer(%sc{h}) {{sym_name = "{p}_Oh"}} : memref<{PT}xbf16>
    %{p}_Sd = aie.buffer(%sc{h}) {{sym_name = "{p}_Sd"}} : memref<{ITC}xbf16>
    {_lock_decls}
    %core_sc{h} = aie.core(%sc{h}) {{
      %c0 = arith.constant 0 : index
      %cN = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %ag = arith.constant {self.AG} : i32
      %hd = arith.constant {self.HD} : i32
      %cnc = arith.constant {NC} : index
      %sC = arith.constant {CH} : i32
      %citc = arith.constant {ITC} : i32
      scf.for %tok = %c0 to %cN step %c1 {{
        func.call @flowkv_score_init_bf16(%ag) : (i32) -> ()
        // #271: wait for center core to signal Q ready (Qc2s consumer)
        aie.use_lock(%{p}_Qc2sc, AcquireGreaterEqual, 1)
        // #271: wait for score mem S2MM DMA to fill Qs (Qs2m consumer)
        aie.use_lock(%{p}_Qs2mc, AcquireGreaterEqual, 1)
        func.call @flowkv_score_rope_q_bf16(%{p}_Qs, %ag, %hd) : (memref<{QSZ}xbf16>, i32, i32) -> ()
        scf.for %ci = %c0 to %cnc step %c1 {{
          aie.use_lock(%{p}_Kc, AcquireGreaterEqual, 1)
          aie.use_lock(%{p}_Ip, AcquireGreaterEqual, 1)
          func.call @flowkv_score_chunk_bf16(%{p}_Qs, %{p}_K, %{p}_It, %ag, %hd, %sC) : (memref<{QSZ}xbf16>, memref<{KCH}xbf16>, memref<{ITC}xbf16>, i32, i32, i32) -> ()
          // #259: relay the per-chunk Q·K scores (It) to host via @S_alloc (the
          // ^ohm S2MM BD re-pointed at Sd). Compute is the producer of Sd.
          aie.use_lock(%{p}_Sdp, AcquireGreaterEqual, 1)
          func.call @layer_fused_dump_q_bf16(%{p}_It, %{p}_Sd, %citc) : (memref<{ITC}xbf16>, memref<{ITC}xbf16>, i32) -> ()
          aie.use_lock(%{p}_Sdc, Release, 1)
          aie.use_lock(%{p}_Kp, Release, 1)
          aie.use_lock(%{p}_Ic, Release, 1)
        }}
        // #271: signal center core that score tile is done reading Q (Qc2s producer)
        aie.use_lock(%{p}_Qc2sp, Release, 1)
        // cycle the dead O_h buffer so the incoming core O (S2MM0-BD1) never stalls
        aie.use_lock(%{p}_Ohc, AcquireGreaterEqual, 1)
        aie.use_lock(%{p}_Ohp, Release, 1)
      }}
      aie.end
    }}
    %mem_sc{h} = aie.mem(%sc{h}) {{
      %s0 = aie.dma_start(S2MM, {_s2mm_in}, ^q{h}, ^ks{h})
    ^q{h}:
      // #271: wait for center mem MM2S stream (Qs2m consumer)
      aie.use_lock(%{p}_Qs2mc, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_Qs : memref<{QSZ}xbf16>, 0, {QSZL})
      // #271: signal score core that Qs DMA is done (Qs2m producer)
      aie.use_lock(%{p}_Qs2mp, Release, 1)
      aie.next_bd ^oh{h}
    ^oh{h}:
      aie.use_lock(%{p}_Ohp, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_Oh : memref<{PT}xbf16>, 0, {PT})
      aie.use_lock(%{p}_Ohc, Release, 1)
      aie.next_bd ^q{h}
    ^ks{h}:
      %s1 = aie.dma_start(S2MM, {_s2mm_kv}, ^k{h}, ^im{h})
    ^k{h}:
      aie.use_lock(%{p}_Kp, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_K : memref<{KCH}xbf16>, 0, {KCH})
      aie.use_lock(%{p}_Kc, Release, 1)
      aie.next_bd ^k{h}
    ^im{h}:
      %m0 = aie.dma_start(MM2S, {_mm2s_out}, ^io{h}, ^ohm{h})
    ^io{h}:
      aie.use_lock(%{p}_Ic, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_It : memref<{ITC}xbf16>, 0, {ITCL})
      aie.use_lock(%{p}_Ip, Release, 1)
      aie.next_bd ^io{h}
    ^ohm{h}:
      %m1 = aie.dma_start(MM2S, {_mm2s_oh}, ^ohf{h}, ^e{h})
    ^ohf{h}:
      aie.use_lock(%{p}_Sdc, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_Sd : memref<{ITC}xbf16>, 0, {ITCL})
      aie.use_lock(%{p}_Sdp, Release, 1)
      aie.next_bd ^ohf{h}
    ^e{h}:
      aie.end
    }}"""

        if self.QDUMP:
            # #188 Q-dump variant: after rope, copy the received roped-Q (PT) into
            # Qd and gate the relay on Qd (instead of O_h). The O_h relay buffer is
            # still filled by S2MM0-BD1 (core sends O via mm2s_qo BD1) - we cycle
            # Ohc/Ohp so that S2MM0 BD1 never spins - but its data is discarded.
            # The O_hd pair (Ohd) is left idle (no code references it -> no deadlock).
            return f"""
    %{p}_Qs = aie.buffer(%sc{h}) {{sym_name = "{p}_Qs"}} : memref<{QSZ}xbf16>
    %{p}_K  = aie.buffer(%sc{h}) {{sym_name = "{p}_K"}}  : memref<{KCH}xbf16>
    %{p}_It = aie.buffer(%sc{h}) {{sym_name = "{p}_It"}} : memref<{ITC}xbf16>
    %{p}_Oh = aie.buffer(%sc{h}) {{sym_name = "{p}_Oh"}} : memref<{PT}xbf16>
    %{p}_Qd = aie.buffer(%sc{h}) {{sym_name = "{p}_Qd"}} : memref<{PT}xbf16>
    {_lock_decls}
    %core_sc{h} = aie.core(%sc{h}) {{
      %c0 = arith.constant 0 : index
      %cN = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %ag = arith.constant {self.AG} : i32
      %hd = arith.constant {self.HD} : i32
      %cnc = arith.constant {NC} : index
      %sC = arith.constant {CH} : i32
      %cd = arith.constant {PT} : i32
      scf.for %tok = %c0 to %cN step %c1 {{
        func.call @flowkv_score_init_bf16(%ag) : (i32) -> ()
        // #271: wait for center core to signal Q ready (Qc2s consumer)
        aie.use_lock(%{p}_Qc2sc, AcquireGreaterEqual, 1)
        // #271: wait for score mem S2MM DMA to fill Qs (Qs2m consumer)
        aie.use_lock(%{p}_Qs2mc, AcquireGreaterEqual, 1)
        func.call @flowkv_score_rope_q_bf16(%{p}_Qs, %ag, %hd) : (memref<{QSZ}xbf16>, i32, i32) -> ()
        // #188 Q dump: relay the roped-Q slice off to the host s-region via the O_h
        // relay (MM2S1, re-pointed at Qd). Compute is the producer of Qd.
        aie.use_lock(%{p}_Qdp, AcquireGreaterEqual, 1)
        func.call @layer_fused_dump_q_bf16(%{p}_Qs, %{p}_Qd, %cd) : (memref<{QSZ}xbf16>, memref<{PT}xbf16>, i32) -> ()
        aie.use_lock(%{p}_Qdc, Release, 1)
        scf.for %ci = %c0 to %cnc step %c1 {{
          aie.use_lock(%{p}_Kc, AcquireGreaterEqual, 1)
          aie.use_lock(%{p}_Ip, AcquireGreaterEqual, 1)
          func.call @flowkv_score_chunk_bf16(%{p}_Qs, %{p}_K, %{p}_It, %ag, %hd, %sC) : (memref<{QSZ}xbf16>, memref<{KCH}xbf16>, memref<{ITC}xbf16>, i32, i32, i32) -> ()
          aie.use_lock(%{p}_Kp, Release, 1)
          aie.use_lock(%{p}_Ic, Release, 1)
        }}
        // #271: signal center core that score tile is done reading Q (Qc2s producer)
        aie.use_lock(%{p}_Qc2sp, Release, 1)
        // cycle the dead O_h buffer so the incoming core O (S2MM0-BD1) never stalls
        aie.use_lock(%{p}_Ohc, AcquireGreaterEqual, 1)
        aie.use_lock(%{p}_Ohp, Release, 1)
      }}
      aie.end
    }}
    %mem_sc{h} = aie.mem(%sc{h}) {{
      %s0 = aie.dma_start(S2MM, {_s2mm_in}, ^q{h}, ^ks{h})
    ^q{h}:
      // #271: wait for center mem MM2S stream (Qs2m consumer)
      aie.use_lock(%{p}_Qs2mc, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_Qs : memref<{QSZ}xbf16>, 0, {QSZL})
      // #271: signal score core that Qs DMA is done (Qs2m producer)
      aie.use_lock(%{p}_Qs2mp, Release, 1)
      aie.next_bd ^oh{h}
    ^oh{h}:
      aie.use_lock(%{p}_Ohp, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_Oh : memref<{PT}xbf16>, 0, {PT})
      aie.use_lock(%{p}_Ohc, Release, 1)
      aie.next_bd ^q{h}
    ^ks{h}:
      %s1 = aie.dma_start(S2MM, {_s2mm_kv}, ^k{h}, ^im{h})
    ^k{h}:
      aie.use_lock(%{p}_Kp, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_K : memref<{KCH}xbf16>, 0, {KCH})
      aie.use_lock(%{p}_Kc, Release, 1)
      aie.next_bd ^k{h}
    ^im{h}:
      %m0 = aie.dma_start(MM2S, {_mm2s_out}, ^io{h}, ^ohm{h})
    ^io{h}:
      aie.use_lock(%{p}_Ic, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_It : memref<{ITC}xbf16>, 0, {ITCL})
      aie.use_lock(%{p}_Ip, Release, 1)
      aie.next_bd ^io{h}
    ^ohm{h}:
      %m1 = aie.dma_start(MM2S, {_mm2s_oh}, ^ohf{h}, ^e{h})
    ^ohf{h}:
      aie.use_lock(%{p}_Qdc, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_Qd : memref<{PT}xbf16>, 0, {PT})
      aie.use_lock(%{p}_Qdp, Release, 1)
      aie.next_bd ^ohf{h}
    ^e{h}:
      aie.end
    }}"""

        return f"""
    %{p}_Qs = aie.buffer(%sc{h}) {{sym_name = "{p}_Qs"}} : memref<{QSZ}xbf16>
    %{p}_K  = aie.buffer(%sc{h}) {{sym_name = "{p}_K"}}  : memref<{KCH}xbf16>
    %{p}_It = aie.buffer(%sc{h}) {{sym_name = "{p}_It"}} : memref<{ITC}xbf16>
    %{p}_Oh = aie.buffer(%sc{h}) {{sym_name = "{p}_Oh"}} : memref<{PT}xbf16>
    {_lock_decls}
    %core_sc{h} = aie.core(%sc{h}) {{
      %c0 = arith.constant 0 : index
      %cN = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %ag = arith.constant {self.AG} : i32
      %hd = arith.constant {self.HD} : i32
      %cnc = arith.constant {NC} : index
      %sC = arith.constant {CH} : i32
      scf.for %tok = %c0 to %cN step %c1 {{
        func.call @flowkv_score_init_bf16(%ag) : (i32) -> ()
        // #271: wait for center core to signal Q ready (Qc2s consumer)
        aie.use_lock(%{p}_Qc2sc, AcquireGreaterEqual, 1)
        // #271: wait for score mem S2MM DMA to fill Qs (Qs2m consumer)
        aie.use_lock(%{p}_Qs2mc, AcquireGreaterEqual, 1)
        func.call @flowkv_score_rope_q_bf16(%{p}_Qs, %ag, %hd) : (memref<{QSZ}xbf16>, i32, i32) -> ()
        scf.for %ci = %c0 to %cnc step %c1 {{
          aie.use_lock(%{p}_Kc, AcquireGreaterEqual, 1)
          aie.use_lock(%{p}_Ip, AcquireGreaterEqual, 1)
          func.call @flowkv_score_chunk_bf16(%{p}_Qs, %{p}_K, %{p}_It, %ag, %hd, %sC) : (memref<{QSZ}xbf16>, memref<{KCH}xbf16>, memref<{ITC}xbf16>, i32, i32, i32) -> ()
          aie.use_lock(%{p}_Kp, Release, 1)
          aie.use_lock(%{p}_Ic, Release, 1)
        }}
        // #271: signal center core that score tile is done reading Q (Qc2s producer)
        aie.use_lock(%{p}_Qc2sp, Release, 1)
        // ph2: relay O_h (S2MM0-BD1 -> MM2S1, no compute, rl-style lock-dance)
        aie.use_lock(%{p}_Ohc, AcquireGreaterEqual, 1)
        aie.use_lock(%{p}_Ohdp, AcquireGreaterEqual, 1)
        aie.use_lock(%{p}_Ohp, Release, 1)
        aie.use_lock(%{p}_Ohdc, Release, 1)
      }}
      aie.end
    }}
    %mem_sc{h} = aie.mem(%sc{h}) {{
      %s0 = aie.dma_start(S2MM, {_s2mm_in}, ^q{h}, ^ks{h})
    ^q{h}:
      // #271: wait for center mem MM2S stream (Qs2m consumer)
      aie.use_lock(%{p}_Qs2mc, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_Qs : memref<{QSZ}xbf16>, 0, {QSZL})
      // #271: signal score core that Qs DMA is done (Qs2m producer)
      aie.use_lock(%{p}_Qs2mp, Release, 1)
      aie.next_bd ^oh{h}
    ^oh{h}:
      aie.use_lock(%{p}_Ohp, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_Oh : memref<{PT}xbf16>, 0, {PT})
      aie.use_lock(%{p}_Ohc, Release, 1)
      aie.next_bd ^q{h}
    ^ks{h}:
      %s1 = aie.dma_start(S2MM, {_s2mm_kv}, ^k{h}, ^im{h})
    ^k{h}:
      aie.use_lock(%{p}_Kp, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_K : memref<{KCH}xbf16>, 0, {KCH})
      aie.use_lock(%{p}_Kc, Release, 1)
      aie.next_bd ^k{h}
    ^im{h}:
      %m0 = aie.dma_start(MM2S, {_mm2s_out}, ^io{h}, ^ohm{h})
    ^io{h}:
      aie.use_lock(%{p}_Ic, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_It : memref<{ITC}xbf16>, 0, {ITCL})
      aie.use_lock(%{p}_Ip, Release, 1)
      aie.next_bd ^io{h}
    ^ohm{h}:
      %m1 = aie.dma_start(MM2S, {_mm2s_oh}, ^ohf{h}, ^e{h})
    ^ohf{h}:
      aie.use_lock(%{p}_Ohdc, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_Oh : memref<{PT}xbf16>, 0, {PT})
      aie.use_lock(%{p}_Ohdp, Release, 1)
      aie.next_bd ^ohf{h}
    ^e{h}:
      aie.end
    }}"""

    def _value(self, h):
        p = f"va{h}"
        KCH = self.KCH; ITC = self.ITC_ALLOC; PT = self.PER_TILE
        ITCL = self.ITC
        NC = self.NCHUNK; CH = self.CHUNK

        # Auto-allocate locks and DMA channels (#154)
        L = _LockAlloc('core'); D = _DmaAlloc('core')
        _i = L.pair('I'); _v = L.pair('V'); _o = L.pair('O')
        _s2mm_in = D.s2mm('in')   # inter
        _s2mm_v  = D.s2mm('v')    # V
        _mm2s_o  = D.mm2s('out')  # Of

        _ld = lambda name, ids: _LockAlloc.mlir_decl(f'va{h}', f'{p}', name, *ids)
        _lock_decls = '\n'.join([_ld('I', _i), _ld('V', _v), _ld('O', _o)])

        return f"""
    %{p}_Iv = aie.buffer(%va{h}) {{sym_name = "{p}_Iv"}} : memref<{ITC}xbf16>
    %{p}_V  = aie.buffer(%va{h}) {{sym_name = "{p}_V"}}  : memref<{KCH}xbf16>
    %{p}_Of = aie.buffer(%va{h}) {{sym_name = "{p}_Of"}} : memref<{PT}xbf16>
    {_lock_decls}
    %core_va{h} = aie.core(%va{h}) {{
      %c0 = arith.constant 0 : index
      %cN = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %ag = arith.constant {self.AG} : i32
      %hd = arith.constant {self.HD} : i32
      %cnc = arith.constant {NC} : index
      %sC = arith.constant {CH} : i32
      scf.for %tok = %c0 to %cN step %c1 {{
        func.call @flowkv_value_init_bf16(%ag, %hd) : (i32, i32) -> ()
        scf.for %ci = %c0 to %cnc step %c1 {{
          aie.use_lock(%{p}_Ic, AcquireGreaterEqual, 1)
          aie.use_lock(%{p}_Vc, AcquireGreaterEqual, 1)
          func.call @flowkv_value_accum_bf16(%{p}_Iv, %{p}_V, %ag, %hd, %sC) : (memref<{ITC}xbf16>, memref<{KCH}xbf16>, i32, i32, i32) -> ()
          aie.use_lock(%{p}_Ip, Release, 1)
          aie.use_lock(%{p}_Vp, Release, 1)
        }}
        aie.use_lock(%{p}_Op, AcquireGreaterEqual, 1)
        func.call @flowkv_value_normalize_bf16(%{p}_Of, %ag, %hd) : (memref<{PT}xbf16>, i32, i32) -> ()
        aie.use_lock(%{p}_Oc, Release, 1)
      }}
      aie.end
    }}
    %mem_va{h} = aie.mem(%va{h}) {{
      %s0 = aie.dma_start(S2MM, {_s2mm_in}, ^iv{h}, ^vs{h})
    ^iv{h}:
      aie.use_lock(%{p}_Ip, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_Iv : memref<{ITC}xbf16>, 0, {ITCL})
      aie.use_lock(%{p}_Ic, Release, 1)
      aie.next_bd ^iv{h}
    ^vs{h}:
      %s1 = aie.dma_start(S2MM, {_s2mm_v}, ^v{h}, ^om{h})
    ^v{h}:
      aie.use_lock(%{p}_Vp, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_V : memref<{KCH}xbf16>, 0, {KCH})
      aie.use_lock(%{p}_Vc, Release, 1)
      aie.next_bd ^v{h}
    ^om{h}:
      %m0 = aie.dma_start(MM2S, {_mm2s_o}, ^oo{h}, ^e{h})
    ^oo{h}:
      aie.use_lock(%{p}_Oc, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_Of : memref<{PT}xbf16>, 0, {PT})
      aie.use_lock(%{p}_Op, Release, 1)
      aie.next_bd ^oo{h}
    ^e{h}:
      aie.end
    }}"""

    def _join_memtile(self, name, tile, with_weight_relay=False, packet_id=None):
        """Join MemTile: 4 S2MM gather -> 1 MM2S output."""
        SL = self.PACKED; QI = self.PER_TILE; E = self.E; D = self.RELAY_DEPTH
        # Auto-allocate locks (#154)
        L = _LockAlloc('memtile')
        base_locks = [L.pair(f'p{k}') for k in range(4)]
        # weight relay locks (auto-continue from base pairs)
        wt_locks = []
        if with_weight_relay:
            for p in range(2):
                for k in range(D):
                    wt_locks.append(L.pair(f'w{p}s{k}'))

        s = f"""
    %{name}_buf = aie.buffer(%{tile}) {{sym_name = "{name}_buf"}} : memref<{E}xbf16>"""
        for k, (lp, lc) in enumerate(base_locks):
            s += f"""
    %{name}_p{k} = aie.lock(%{tile}, {lp}) {{init = 1 : i32, sym_name = "{name}_p{k}"}}
    %{name}_c{k} = aie.lock(%{tile}, {lc}) {{init = 0 : i32, sym_name = "{name}_c{k}"}}"""
        if with_weight_relay:
            s += f"""
    %{name}_wt = aie.buffer(%{tile}) {{sym_name = "{name}_wt"}} : memref<{2*D*SL}xi8>"""
            for idx, (lp, lc) in enumerate(wt_locks):
                p = idx // D; k = idx % D
                s += f"""
    %{name}_w{p}s{k}p = aie.lock(%{tile}, {lp}) {{init = 1 : i32, sym_name = "{name}_w{p}s{k}p"}}
    %{name}_w{p}s{k}c = aie.lock(%{tile}, {lc}) {{init = 0 : i32, sym_name = "{name}_w{p}s{k}c"}}"""
        s += f"""
    %{name}_dma = aie.memtile_dma(%{tile}) {{"""
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
      aie.dma_bd(%{name}_buf : memref<{E}xbf16>, {ch*QI}, {QI})
      aie.use_lock(%{name}_c{ch}, Release, 1)
      aie.next_bd ^{name}g{ch}"""
        if with_weight_relay:
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
        s += f"""
    ^{name}m0:
      %m0 = aie.dma_start(MM2S, 0, ^{name}o0, {nxt_mm2s0})
    ^{name}o0:"""
        if packet_id is not None:
            s += f"""
      aie.dma_bd_packet(0, {packet_id})"""
        s += f"""
      aie.use_lock(%{name}_c0, AcquireGreaterEqual, 1)
      aie.dma_bd(%{name}_buf : memref<{E}xbf16>, 0, {QI})
      aie.use_lock(%{name}_p0, Release, 1)
      aie.next_bd ^{name}o1
    ^{name}o1:
      aie.use_lock(%{name}_c1, AcquireGreaterEqual, 1)
      aie.dma_bd(%{name}_buf : memref<{E}xbf16>, {QI}, {QI})
      aie.use_lock(%{name}_p1, Release, 1)
      aie.next_bd ^{name}o2
    ^{name}o2:
      aie.use_lock(%{name}_c2, AcquireGreaterEqual, 1)
      aie.dma_bd(%{name}_buf : memref<{E}xbf16>, {2*QI}, {QI})
      aie.use_lock(%{name}_p2, Release, 1)
      aie.next_bd ^{name}o3
    ^{name}o3:
      aie.use_lock(%{name}_c3, AcquireGreaterEqual, 1)
      aie.dma_bd(%{name}_buf : memref<{E}xbf16>, {3*QI}, {QI})
      aie.use_lock(%{name}_p3, Release, 1)
      aie.next_bd ^{name}o0"""
        if with_weight_relay:
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

    def _split_memtile(self, name, tile, with_weight_relay=False):
        """1 shim stream (4*KVN) -> 4 MM2S slices (KVN each)."""
        SL_KV = self.KVN; KVN = self.KVN; D = self.RELAY_DEPTH; WS = self.PACKED
        # Auto-allocate locks (#154)
        L = _LockAlloc('memtile')
        base_locks = [L.pair(f'p{k}') for k in range(4)]
        wt_locks = []
        if with_weight_relay:
            for k in range(D):
                wt_locks.append(L.pair(f'ws{k}'))

        s = f"""
    %{name}_buf = aie.buffer(%{tile}) {{sym_name = "{name}_buf"}} : memref<{4*KVN}xbf16>"""
        for k, (lp, lc) in enumerate(base_locks):
            s += f"""
    %{name}_p{k} = aie.lock(%{tile}, {lp}) {{init = 1 : i32, sym_name = "{name}_p{k}"}}
    %{name}_c{k} = aie.lock(%{tile}, {lc}) {{init = 0 : i32, sym_name = "{name}_c{k}"}}"""
        if with_weight_relay:
            s += f"""
    %{name}_wt = aie.buffer(%{tile}) {{sym_name = "{name}_wt"}} : memref<{D*WS}xi8>"""
            for k, (lp, lc) in enumerate(wt_locks):
                s += f"""
    %{name}_wsp{k} = aie.lock(%{tile}, {lp}) {{init = 1 : i32, sym_name = "{name}_wsp{k}"}}
    %{name}_wsc{k} = aie.lock(%{tile}, {lc}) {{init = 0 : i32, sym_name = "{name}_wsc{k}"}}"""
        fill_nxt = f"^{name}w0" if with_weight_relay else f"^{name}m0"
        s += f"""
    %{name}_dma = aie.memtile_dma(%{tile}) {{
      %s0 = aie.dma_start(S2MM, 0, ^{name}f0, {fill_nxt})"""
        for k in range(4):
            s += f"""
    ^{name}f{k}:
      aie.use_lock(%{name}_p{k}, AcquireGreaterEqual, 1)
      aie.dma_bd(%{name}_buf : memref<{4*KVN}xbf16>, {k*SL_KV}, {SL_KV})
      aie.use_lock(%{name}_c{k}, Release, 1)
      aie.next_bd ^{name}f{(k+1) % 4}"""
        if with_weight_relay:
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
        for ch in range(4):
            nxt = f"^{name}m{ch+1}" if ch < 3 else (f"^{name}wrel" if with_weight_relay else f"^{name}e")
            s += f"""
    ^{name}m{ch}:
      %m{ch} = aie.dma_start(MM2S, {ch}, ^{name}o{ch}, {nxt})
    ^{name}o{ch}:
      aie.use_lock(%{name}_c{ch}, AcquireGreaterEqual, 1)
      aie.dma_bd(%{name}_buf : memref<{4*KVN}xbf16>, {ch*SL_KV}, {SL_KV})
      aie.use_lock(%{name}_p{ch}, Release, 1)
      aie.next_bd ^{name}o{ch}"""
        if with_weight_relay:
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

    def _relay_block(self):
        """rl tile: relay attn_out half0+half1 -> mx (packet).

        #245 ATTN_DUMP: a second MM2S (free channel 1) copies rl_A into a host-
        visible debug BO via a free shim S2MM (sh2 ch0), so the host can read the
        full pre-O-proj attention output per q-head. Numerically identical to the
        production path (rl_A is read-only here). Gated by F3BEST_ATTN_DUMP."""
        E = self.E; E2 = E // 2
        L = _LockAlloc('core'); D = _DmaAlloc('core')
        _h0 = L.pair('p0'); _h1 = L.pair('p1'); _oo = L.pair('op')
        _s2mm_0 = D.s2mm(); _s2mm_1 = D.s2mm(); _mm2s = D.mm2s()
        _ld = lambda name, ids: _LockAlloc.mlir_decl('rl', 'rl', name, *ids)
        if self.ATTN_DUMP:
            # Dedicated lock pair for the dump MM2S (core signals "data ready" via
            # dp_consumer release; dump MM2S acquires it, copies rl_A, releases
            # dp_producer; next-iteration core acquires dp_producer = dump done).
            _dp = L.pair('dp')
            _mm2s_dbg = D.mm2s('attn_dump')   # free channel 1
            _lock_decls = '\n'.join([_ld('p0', _h0), _ld('p1', _h1), _ld('op', _oo), _ld('dp', _dp)])
            # #245 core dump handshake: acquire dpp (previous dump done) BEFORE
            # releasing opc/dpc (trigger new drains), so both MM2S channels are
            # guaranteed idle before rl_A is overwritten. Release dpc (data ready
            # for tap) alongside opc (data ready for production drain).
            _dump_core_wait = "        aie.use_lock(%rl_dpp, AcquireGreaterEqual, 1)\n"
            _dump_core_fire = "        aie.use_lock(%rl_dpc, Release, 1)"
            _dump_mem = (
                f"""    ^rm0:
      %m0 = aie.dma_start(MM2S, {_mm2s}, ^ro, ^rd)
    ^ro:
      aie.use_lock(%rl_opc, AcquireGreaterEqual, 1)
      aie.dma_bd_packet(0, 0)
      aie.dma_bd(%rl_A : memref<{E}xbf16>, 0, {E})
      aie.use_lock(%rl_opp, Release, 1)
      aie.next_bd ^ro
    ^rd:
      %m1 = aie.dma_start(MM2S, {_mm2s_dbg}, ^rdo, ^re)
    ^rdo:
      aie.use_lock(%rl_dpc, AcquireGreaterEqual, 1)
      aie.dma_bd(%rl_A : memref<{E}xbf16>, 0, {E})
      aie.use_lock(%rl_dpp, Release, 1)
      aie.next_bd ^rdo
    ^re:
      aie.end
    }}""")
        else:
            _lock_decls = '\n'.join([_ld('p0', _h0), _ld('p1', _h1), _ld('op', _oo)])
            _dump_core_wait = ""
            _dump_core_fire = ""
            _dump_mem = (
                f"""    ^rm0:
      %m0 = aie.dma_start(MM2S, {_mm2s}, ^ro, ^re)
    ^ro:
      aie.use_lock(%rl_opc, AcquireGreaterEqual, 1)
      aie.dma_bd_packet(0, 0)
      aie.dma_bd(%rl_A : memref<{E}xbf16>, 0, {E})
      aie.use_lock(%rl_opp, Release, 1)
      aie.next_bd ^ro
    ^re:
      aie.end
    }}""")
        return f"""
    %rl_A = aie.buffer(%rl) {{sym_name = "rl_A"}} : memref<{E}xbf16>
    {_lock_decls}
    %core_rl = aie.core(%rl) {{
      %z = arith.constant 0 : index
      %N = arith.constant 9223372036854775807 : index
      %one = arith.constant 1 : index
      scf.for %it = %z to %N step %one {{
        aie.use_lock(%rl_p0c, AcquireGreaterEqual, 1)
        aie.use_lock(%rl_p1c, AcquireGreaterEqual, 1)
        aie.use_lock(%rl_opp, AcquireGreaterEqual, 1)
{_dump_core_wait}
        aie.use_lock(%rl_p0p, Release, 1)
        aie.use_lock(%rl_p1p, Release, 1)
        aie.use_lock(%rl_opc, Release, 1)
{_dump_core_fire}
      }}
      aie.end
    }}
    %mem_rl = aie.mem(%rl) {{
      %s0 = aie.dma_start(S2MM, {_s2mm_0}, ^rh0, ^rs1)
    ^rh0:
      aie.use_lock(%rl_p0p, AcquireGreaterEqual, 1)
      aie.dma_bd(%rl_A : memref<{E}xbf16>, 0, {E2})
      aie.use_lock(%rl_p0c, Release, 1)
      aie.next_bd ^rh0
    ^rs1:
      %s1 = aie.dma_start(S2MM, {_s2mm_1}, ^rh1, ^rm0)
    ^rh1:
      aie.use_lock(%rl_p1p, AcquireGreaterEqual, 1)
      aie.dma_bd(%rl_A : memref<{E}xbf16>, {E2}, {E2})
      aie.use_lock(%rl_p1c, Release, 1)
      aie.next_bd ^rh1
{_dump_mem}"""

    def _op_block(self):
        """op tile: O-relay - concat O_h0..3 + O_h4..7 -> nm."""
        E = self.E; E2 = E // 2
        L = _LockAlloc('core'); D = _DmaAlloc('core')
        _h0 = L.pair('p0'); _h1 = L.pair('p1'); _oo = L.pair('op')
        _s2mm_0 = D.s2mm(); _s2mm_1 = D.s2mm(); _mm2s = D.mm2s()
        _ld = lambda name, ids: _LockAlloc.mlir_decl('op', 'op', name, *ids)
        _lock_decls = '\n'.join([_ld('p0', _h0), _ld('p1', _h1), _ld('op', _oo)])
        return f"""
    %op_O  = aie.buffer(%op) {{sym_name = "op_O"}}  : memref<{E}xbf16>
    {_lock_decls}
    %core_op = aie.core(%op) {{
      %z = arith.constant 0 : index
      %N = arith.constant 9223372036854775807 : index
      %one = arith.constant 1 : index
      scf.for %it = %z to %N step %one {{
        aie.use_lock(%op_p0c, AcquireGreaterEqual, 1)
        aie.use_lock(%op_p1c, AcquireGreaterEqual, 1)
        aie.use_lock(%op_opp, AcquireGreaterEqual, 1)
        aie.use_lock(%op_p0p, Release, 1)
        aie.use_lock(%op_p1p, Release, 1)
        aie.use_lock(%op_opc, Release, 1)
      }}
      aie.end
    }}
    %mem_op = aie.mem(%op) {{
      %s0 = aie.dma_start(S2MM, {_s2mm_0}, ^oh0, ^os1)
    ^oh0:
      aie.use_lock(%op_p0p, AcquireGreaterEqual, 1)
      aie.dma_bd(%op_O : memref<{E}xbf16>, 0, {E2})
      aie.use_lock(%op_p0c, Release, 1)
      aie.next_bd ^oh0
    ^os1:
      %s1 = aie.dma_start(S2MM, {_s2mm_1}, ^oh1, ^om0)
    ^oh1:
      aie.use_lock(%op_p1p, AcquireGreaterEqual, 1)
      aie.dma_bd(%op_O : memref<{E}xbf16>, {E2}, {E2})
      aie.use_lock(%op_p1c, Release, 1)
      aie.next_bd ^oh1
    ^om0:
      %m0 = aie.dma_start(MM2S, {_mm2s}, ^oo, ^oe)
    ^oo:
      aie.use_lock(%op_opc, AcquireGreaterEqual, 1)
      aie.dma_bd(%op_O : memref<{E}xbf16>, 0, {E})
      aie.use_lock(%op_opp, Release, 1)
      aie.next_bd ^oo
    ^oe:
      aie.end
    }}"""

    def _nm_block(self):
        """nm tile: add O+resid, RMSNorm(ffn_in, gain)."""
        E = self.E; XB = self.XB

        # Auto-allocate locks and DMA channels (#154)
        L = _LockAlloc('core'); D = _DmaAlloc('core')
        _o = L.pair('O'); _r = L.pair('R'); _f = L.pair('F')
        _g = L.pair('G'); _fn = L.pair('FN')
        _s2mm_0 = D.s2mm(); _s2mm_1 = D.s2mm(); _mm2s_0 = D.mm2s(); _mm2s_1 = D.mm2s()

        _ld = lambda name, ids: _LockAlloc.mlir_decl('nm', 'nm', name, *ids)
        _lock_decls = '\n'.join([
            _ld('O', _o), _ld('R', _r), _ld('F', _f),
            _ld('G', _g), _ld('FN', _fn),
        ])

        return f"""
    %nm_O = aie.buffer(%nm) {{sym_name = "nm_O"}} : memref<{E}xbf16>
    %nm_R = aie.buffer(%nm) {{sym_name = "nm_R"}} : memref<{E}xbf16>
    %nm_F = aie.buffer(%nm) {{sym_name = "nm_F"}} : memref<{XB}xbf16>
    %nm_FN = aie.buffer(%nm) {{sym_name = "nm_FN"}} : memref<{XB}xbf16>
    %nm_gain = aie.buffer(%nm) {{sym_name = "nm_gain"}} : memref<{E}xbf16>
    {_lock_decls}
    %core_nm = aie.core(%nm) {{
      %z = arith.constant 0 : index
      %N = arith.constant 9223372036854775807 : index
      %one = arith.constant 1 : index
      %ne = arith.constant {E} : i32
      scf.for %it = %z to %N step %one {{
        aie.use_lock(%nm_Oc, AcquireGreaterEqual, 1)
        aie.use_lock(%nm_Rc, AcquireGreaterEqual, 1)
        aie.use_lock(%nm_Gc, AcquireGreaterEqual, 1)
        aie.use_lock(%nm_Fp, AcquireGreaterEqual, 1)
        aie.use_lock(%nm_FNp, AcquireGreaterEqual, 1)
        // #242: in QDUMP mode, add_bf16 still runs but resid (nm_R) is host-zeroed,
        // so F = O(Q) + 0 = Q (clean, no resid corruption). The host zeros nm_R via
        // the XR bundle resid region when QDUMP is active. This keeps the s-region
        // = Q exactly (bf16, but without the bf16(Q+R)-minus-R precision loss).
        func.call @layer_fused_add_bf16(%nm_O, %nm_R, %nm_F, %ne) : (memref<{E}xbf16>, memref<{E}xbf16>, memref<{XB}xbf16>, i32) -> ()
        func.call @layer_fused_rms_norm2_bf16(%nm_F, %nm_gain, %nm_FN, %ne) : (memref<{XB}xbf16>, memref<{E}xbf16>, memref<{XB}xbf16>, i32) -> ()
        aie.use_lock(%nm_Op, Release, 1)
        aie.use_lock(%nm_Rp, Release, 1)
        aie.use_lock(%nm_Gp, Release, 1)
        aie.use_lock(%nm_Fc, Release, 1)
        aie.use_lock(%nm_FNc, Release, 1)
      }}
      aie.end
    }}
    %mem_nm = aie.mem(%nm) {{
      %s0 = aie.dma_start(S2MM, {_s2mm_0}, ^no, ^nrs)
    ^no:
      aie.use_lock(%nm_Op, AcquireGreaterEqual, 1)
      aie.dma_bd(%nm_O : memref<{E}xbf16>, 0, {E})
      aie.use_lock(%nm_Oc, Release, 1)
      aie.next_bd ^no
    ^nrs:
      %s1 = aie.dma_start(S2MM, {_s2mm_1}, ^nr, ^nm0)
    ^nr:
      aie.use_lock(%nm_Rp, AcquireGreaterEqual, 1)
      aie.dma_bd(%nm_R : memref<{E}xbf16>, 0, {E})
      aie.use_lock(%nm_Rc, Release, 1)
      aie.next_bd ^ng
    ^ng:
      aie.use_lock(%nm_Gp, AcquireGreaterEqual, 1)
      aie.dma_bd(%nm_gain : memref<{E}xbf16>, 0, {E})
      aie.use_lock(%nm_Gc, Release, 1)
      aie.next_bd ^nr
    ^nm0:
      %m0 = aie.dma_start(MM2S, {_mm2s_0}, ^nf, ^nm1)
    ^nf:
      aie.use_lock(%nm_FNc, AcquireGreaterEqual, 1)
      aie.dma_bd_packet(0, 1)
      aie.dma_bd(%nm_FN : memref<{XB}xbf16>, 0, {E})
      aie.use_lock(%nm_FNp, Release, 1)
      aie.next_bd ^nf
    ^nm1:
      %m1 = aie.dma_start(MM2S, {_mm2s_1}, ^ns, ^nme)
    ^ns:
      aie.use_lock(%nm_Fc, AcquireGreaterEqual, 1)
      aie.dma_bd(%nm_F : memref<{XB}xbf16>, 0, {E})
      aie.use_lock(%nm_Fp, Release, 1)
      aie.next_bd ^ns
    ^nme:
      aie.end
    }}"""

    def _mux_block(self):
        """mx tile: 3-phase broadcast mux (x / attn_out / ffn_in -> center tiles)."""
        E = self.E; XB = self.XB
        L = _LockAlloc('core'); D = _DmaAlloc('core')
        _s2mm_0 = D.s2mm(); _s2mm_1 = D.s2mm(); _mm2s = D.mm2s()
        _ld = lambda name, ids: _LockAlloc.mlir_decl('mx', 'mx', name, *ids)
        if self.RL_FIX:
            _x = L.pair('x'); _a0 = L.pair('a0'); _f = L.pair('f')
            _a1 = L.pair('a1'); _o = L.pair('o')
            _lock_decls = '\n'.join([_ld('x', _x), _ld('a0', _a0), _ld('f', _f),
                                     _ld('a1', _a1), _ld('o', _o)])
            return f"""
    %mx_x = aie.buffer(%mx) {{sym_name = "mx_x"}} : memref<{XB}xbf16>
    %mx_a = aie.buffer(%mx) {{sym_name = "mx_a"}} : memref<{XB}xbf16>
    %mx_f = aie.buffer(%mx) {{sym_name = "mx_f"}} : memref<{XB}xbf16>
    %mx_o = aie.buffer(%mx) {{sym_name = "mx_o"}} : memref<{XB}xbf16>
    {_lock_decls}
    %core_mx = aie.core(%mx) {{
      %z = arith.constant 0 : index
      %N = arith.constant 9223372036854775807 : index
      %one = arith.constant 1 : index
      %nx = arith.constant {XB} : i32
      %ne = arith.constant {E} : i32
      scf.for %it = %z to %N step %one {{
        // NOTE: F3BEST_RL_FIX is OFF in production - this RL_FIX branch is dead.
        // The non-RL_FIX mux below (L1340+) is what ships; its lock protocol is
        // already a clean acquire mx_op / release mx_oc ping-pong on all 3 phases.
        aie.use_lock(%mx_xc, AcquireGreaterEqual, 1)
        aie.use_lock(%mx_op, AcquireGreaterEqual, 1)
        func.call @attn_copy_bf16(%mx_x, %mx_o, %nx) : (memref<{XB}xbf16>, memref<{XB}xbf16>, i32) -> ()
        aie.use_lock(%mx_xp, Release, 1)
        aie.use_lock(%mx_oc, Release, 1)
        aie.use_lock(%mx_a0c, AcquireGreaterEqual, 1)
        aie.use_lock(%mx_a1c, AcquireGreaterEqual, 1)
        aie.use_lock(%mx_op, AcquireGreaterEqual, 1)
        func.call @attn_copy_bf16(%mx_a, %mx_o, %ne) : (memref<{XB}xbf16>, memref<{XB}xbf16>, i32) -> ()
        aie.use_lock(%mx_a0p, Release, 1)
        aie.use_lock(%mx_a1p, Release, 1)
        aie.use_lock(%mx_oc, Release, 1)
        aie.use_lock(%mx_fc, AcquireGreaterEqual, 1)
        aie.use_lock(%mx_op, AcquireGreaterEqual, 1)
        func.call @attn_copy_bf16(%mx_f, %mx_o, %ne) : (memref<{XB}xbf16>, memref<{XB}xbf16>, i32) -> ()
        aie.use_lock(%mx_fp, Release, 1)
        aie.use_lock(%mx_oc, Release, 1)
      }}
      aie.end
    }}
    %mem_mx = aie.mem(%mx) {{
      %s0 = aie.dma_start(S2MM, {_s2mm_0}, ^mxi, ^mxs1)
    ^mxi:
      aie.use_lock(%mx_xp, AcquireGreaterEqual, 1)
      aie.dma_bd(%mx_x : memref<{XB}xbf16>, 0, {XB})
      aie.use_lock(%mx_xc, Release, 1)
      aie.next_bd ^mxi
    ^mxs1:
      %s1 = aie.dma_start(S2MM, {_s2mm_1}, ^mxa0, ^mxm0)
    ^mxa0:
      aie.use_lock(%mx_a0p, AcquireGreaterEqual, 1)
      aie.dma_bd_packet(0, 0)
      aie.dma_bd(%mx_a : memref<{XB}xbf16>, 0, {E//2})
      aie.use_lock(%mx_a0c, Release, 1)
      aie.next_bd ^mxf
    ^mxf:
      aie.use_lock(%mx_fp, AcquireGreaterEqual, 1)
      aie.dma_bd_packet(0, 1)
      aie.dma_bd(%mx_f : memref<{XB}xbf16>, 0, {E})
      aie.use_lock(%mx_fc, Release, 1)
      aie.next_bd ^mxa1
    ^mxa1:
      aie.use_lock(%mx_a1p, AcquireGreaterEqual, 1)
      aie.dma_bd_packet(0, 2)
      aie.dma_bd(%mx_a : memref<{XB}xbf16>, {E//2}, {E//2})
      aie.use_lock(%mx_a1c, Release, 1)
      aie.next_bd ^mxa0
    ^mxm0:
      %m0 = aie.dma_start(MM2S, {_mm2s}, ^mxo, ^mxe)
    ^mxo:
      aie.use_lock(%mx_oc, AcquireGreaterEqual, 1)
      aie.dma_bd(%mx_o : memref<{XB}xbf16>, 0, {XB})
      aie.use_lock(%mx_op, Release, 1)
      aie.next_bd ^mxo
    ^mxe:
      aie.end
    }}"""
        else:
            _x = L.pair('x'); _a = L.pair('a'); _f = L.pair('f'); _o = L.pair('o')
            _lock_decls = '\n'.join([_ld('x', _x), _ld('a', _a), _ld('f', _f), _ld('o', _o)])
            return f"""
    %mx_x = aie.buffer(%mx) {{sym_name = "mx_x"}} : memref<{XB}xbf16>
    %mx_a = aie.buffer(%mx) {{sym_name = "mx_a"}} : memref<{XB}xbf16>
    %mx_f = aie.buffer(%mx) {{sym_name = "mx_f"}} : memref<{XB}xbf16>
    %mx_o = aie.buffer(%mx) {{sym_name = "mx_o"}} : memref<{XB}xbf16>
    {_lock_decls}
    %core_mx = aie.core(%mx) {{
      %z = arith.constant 0 : index
      %N = arith.constant 9223372036854775807 : index
      %one = arith.constant 1 : index
      %nx = arith.constant {XB} : i32
      %ne = arith.constant {E} : i32
      scf.for %it = %z to %N step %one {{
        aie.use_lock(%mx_xc, AcquireGreaterEqual, 1)
        aie.use_lock(%mx_op, AcquireGreaterEqual, 1)
        func.call @attn_copy_bf16(%mx_x, %mx_o, %nx) : (memref<{XB}xbf16>, memref<{XB}xbf16>, i32) -> ()
        aie.use_lock(%mx_xp, Release, 1)
        aie.use_lock(%mx_oc, Release, 1)
        aie.use_lock(%mx_ac, AcquireGreaterEqual, 1)
        aie.use_lock(%mx_op, AcquireGreaterEqual, 1)
        func.call @attn_copy_bf16(%mx_a, %mx_o, %ne) : (memref<{XB}xbf16>, memref<{XB}xbf16>, i32) -> ()
        aie.use_lock(%mx_ap, Release, 1)
        aie.use_lock(%mx_oc, Release, 1)
        aie.use_lock(%mx_fc, AcquireGreaterEqual, 1)
        aie.use_lock(%mx_op, AcquireGreaterEqual, 1)
        func.call @attn_copy_bf16(%mx_f, %mx_o, %ne) : (memref<{XB}xbf16>, memref<{XB}xbf16>, i32) -> ()
        aie.use_lock(%mx_fp, Release, 1)
        aie.use_lock(%mx_oc, Release, 1)
      }}
      aie.end
    }}
    %mem_mx = aie.mem(%mx) {{
      %s0 = aie.dma_start(S2MM, {_s2mm_0}, ^mxi, ^mxs1)
    ^mxi:
      aie.use_lock(%mx_xp, AcquireGreaterEqual, 1)
      aie.dma_bd(%mx_x : memref<{XB}xbf16>, 0, {XB})
      aie.use_lock(%mx_xc, Release, 1)
      aie.next_bd ^mxi
    ^mxs1:
      %s1 = aie.dma_start(S2MM, {_s2mm_1}, ^mxa, ^mxm0)
    ^mxa:
      aie.use_lock(%mx_ap, AcquireGreaterEqual, 1)
      aie.dma_bd_packet(0, 0)
      aie.dma_bd(%mx_a : memref<{XB}xbf16>, 0, {E})
      aie.use_lock(%mx_ac, Release, 1)
      aie.next_bd ^mxf
    ^mxf:
      aie.use_lock(%mx_fp, AcquireGreaterEqual, 1)
      aie.dma_bd_packet(0, 1)
      aie.dma_bd(%mx_f : memref<{XB}xbf16>, 0, {E})
      aie.use_lock(%mx_fc, Release, 1)
      aie.next_bd ^mxa
    ^mxm0:
      %m0 = aie.dma_start(MM2S, {_mm2s}, ^mxo, ^mxe)
    ^mxo:
      aie.use_lock(%mx_oc, AcquireGreaterEqual, 1)
      aie.dma_bd(%mx_o : memref<{XB}xbf16>, 0, {XB})
      aie.use_lock(%mx_op, Release, 1)
      aie.next_bd ^mxo
    ^mxe:
      aie.end
    }}"""

    # ═══════════════════════════════════════════════════════════════════════
    #  emit_mlir - assemble complete MLIR text
    # ═══════════════════════════════════════════════════════════════════════

    def emit_mlir(self):
        NH = self.NH; E = self.E; WT_BYTES = self.WT_BYTES
        KVN = self.KVN; XB = self.XB

        # tile declarations
        tile_decls = "".join(f"    %t{h} = aie.tile({CENTER_COLS[h][0]}, {CENTER_COLS[h][1]})\n" for h in range(NH))
        tile_decls += "".join(f"    %sc{h} = aie.tile({SCORE_COLS[h][0]}, {SCORE_COLS[h][1]})\n" for h in range(NH))
        tile_decls += "".join(f"    %va{h} = aie.tile({VALUE_COLS[h][0]}, {VALUE_COLS[h][1]})\n" for h in range(NH))
        tile_decls += "    %jA = aie.tile(2, 1)\n    %jB = aie.tile(3, 1)\n"
        tile_decls += "    %Klo = aie.tile(0, 1)\n    %Khi = aie.tile(1, 1)\n    %Vlo = aie.tile(6, 1)\n    %Vhi = aie.tile(7, 1)\n"
        tile_decls += "    %oJ = aie.tile(4, 1)\n    %oK = aie.tile(5, 1)\n"
        if self.RL_FIX:
            tile_decls += "    %op = aie.tile(3, 4)\n    %nm = aie.tile(4, 4)\n    %mx = aie.tile(5, 4)\n"
        else:
            tile_decls += "    %rl = aie.tile(2, 4)\n    %op = aie.tile(3, 4)\n    %nm = aie.tile(4, 4)\n    %mx = aie.tile(5, 4)\n"
        shim_decls = "".join(f"    %sh{h} = aie.tile({h}, 0)\n" for h in range(NH))

        # kernel function declarations (parametric buffer sizes)
        # Types must match the buffer ALLOCATION, so use the padded sizes.
        PK = self.PACKED; XB = self.XB; QSZ = self.Q_SZ_ALLOC; OSZ = self.O_SZ
        PSZ = self.P_SZ; HP = self.HPER; UNI = self.UNI_SZ; E = self.E
        KCH = self.KCH; ITC = self.ITC_ALLOC; PT = self.PER_TILE
        funcs = (
            f'    func.func private @fused_dequant_matvec_v2_bf16(i32, i32, memref<{PK}xi8>, memref<{XB}xbf16>, memref<{QSZ}xbf16>) attributes {{link_with = "fused_dequant_gemv_v2_signed_{self.E}k_g32.o"}}\n'
            f'    func.func private @rope_bundled(memref<{QSZ}xbf16>, memref<{XB}xbf16>, memref<{QSZ}xbf16>, i32) attributes {{link_with = "{self.ROPE_LIB}"}}\n'
            f'    func.func private @layer_fused_gate_up_bf16(i32, i32, memref<{PK}xi8>, memref<{XB}xbf16>, i32) attributes {{link_with = "{self.RELAY_LIB}"}}\n'
            f'    func.func private @layer_fused_qkv_bcast_bf16(i32, i32, memref<{PK}xi8>, memref<{XB}xbf16>, memref<{QSZ}xbf16>) attributes {{link_with = "{self.RELAY_LIB}"}}\n'
            f'    func.func private @layer_fused_oproj_bcast_bf16(i32, i32, memref<{PK}xi8>, memref<{XB}xbf16>, memref<{OSZ}xbf16>) attributes {{link_with = "{self.RELAY_LIB}"}}\n'
            f'    func.func private @layer_fused_gate_up_bcast_bf16(i32, i32, memref<{PK}xi8>, memref<{XB}xbf16>, i32) attributes {{link_with = "{self.RELAY_LIB}"}}\n'
            f'    func.func private @layer_fused_down_bcast_bf16(i32, i32, memref<{PK}xi8>, memref<{PSZ}xbf16>) attributes {{link_with = "{self.RELAY_LIB}"}}\n'
            f'    func.func private @layer_fused_silu_mul_static_bf16(i32) attributes {{link_with = "{self.RELAY_LIB}"}}\n'
            f'    func.func private @layer_fused_down_v2_x4_bf16(i32, i32, i32, memref<{PK}xi8>, memref<{PSZ}xbf16>) attributes {{link_with = "{self.RELAY_LIB}"}}\n'
            f'    func.func private @attn_copy_bf16(memref<{XB}xbf16>, memref<{XB}xbf16>, i32) attributes {{link_with = "attn_concat.o"}}\n'
            f'    func.func private @oproj_matvec_v2_bf16(i32, i32, memref<{PK}xi8>, memref<{XB}xbf16>, memref<{OSZ}xbf16>) attributes {{link_with = "fused_dequant_gemv_v2_oproj_signed_{self.E}k_g32.o"}}\n'
            f'    func.func private @layer_fused_add_bf16(memref<{E}xbf16>, memref<{E}xbf16>, memref<{XB}xbf16>, i32) attributes {{link_with = "{self.RELAY_LIB}"}}\n'
            f'    func.func private @layer_fused_rms_norm2_bf16(memref<{XB}xbf16>, memref<{E}xbf16>, memref<{XB}xbf16>, i32) attributes {{link_with = "{self.RELAY_LIB}"}}\n'
            f'    func.func private @flowkv_score_init_bf16(i32) attributes {{link_with = "{self.FLOWKV_LIB}"}}\n'
            f'    func.func private @flowkv_score_rope_q_bf16(memref<{QSZ}xbf16>, i32, i32) attributes {{link_with = "{self.FLOWKV_LIB}"}}\n'
            f'    func.func private @flowkv_score_chunk_bf16(memref<{QSZ}xbf16>, memref<{KCH}xbf16>, memref<{ITC}xbf16>, i32, i32, i32) attributes {{link_with = "{self.FLOWKV_LIB}"}}\n'
            f'    func.func private @flowkv_value_init_bf16(i32, i32) attributes {{link_with = "{self.FLOWKV_LIB}"}}\n'
            f'    func.func private @flowkv_value_accum_bf16(memref<{ITC}xbf16>, memref<{KCH}xbf16>, i32, i32, i32) attributes {{link_with = "{self.FLOWKV_LIB}"}}\n'
            f'    func.func private @flowkv_value_normalize_bf16(memref<{PT}xbf16>, i32, i32) attributes {{link_with = "{self.FLOWKV_LIB}"}}\n'
            + (f'    func.func private @layer_fused_dump_q_bf16(memref<{ITC}xbf16>, memref<{ITC}xbf16>, i32) attributes {{link_with = "{self.RELAY_LIB}"}}\n'
               if self.SDUMP else
               (f'    func.func private @layer_fused_dump_q_bf16(memref<{QSZ}xbf16>, memref<{PT}xbf16>, i32) attributes {{link_with = "{self.RELAY_LIB}"}}\n'
                if self.QDUMP else ''))
        )

        # flows
        flows = []
        flows.append("    aie.flow(%sh0, DMA : 1, %mx, DMA : 0)   // x_bundle -> mux S2MM0")
        if self.AIE_TRACE:
            # #260: cascade trace through score tiles to ONE shim (avoids 8
            # parallel trace lines exhausting shim port-slots). Windows x86
            # simulator doesn't model stalls, so this must work on hardware —
            # route sc0 Trace:0 -> sc1 (neighbour) -> sh7 DMA (free shim with
            # East/West transits available per North:0/South:2 only occupied).
            flows.append(f'    aie.flow(%sc1, Trace : 0, %sh7, DMA : 1)   // trace sc1 -> sh7 S2MM1')
        if self.RL_FIX:
            flows.append("    aie.packet_flow(0) { aie.packet_source<%jA, DMA : 0> aie.packet_dest<%mx, DMA : 1> }   // attnA -> mux S2MM1")
            flows.append("    aie.packet_flow(1) { aie.packet_source<%nm, DMA : 0> aie.packet_dest<%mx, DMA : 1> }   // ffn_in -> mux S2MM1")
            flows.append("    aie.packet_flow(2) { aie.packet_source<%jB, DMA : 0> aie.packet_dest<%mx, DMA : 1> }   // attnB -> mux S2MM1")
        else:
            flows.append("    aie.packet_flow(0) { aie.packet_source<%rl, DMA : 0> aie.packet_dest<%mx, DMA : 1> }   // attn_out -> mux S2MM1")
            flows.append("    aie.packet_flow(1) { aie.packet_source<%nm, DMA : 0> aie.packet_dest<%mx, DMA : 1> }   // ffn_in -> mux S2MM1")

        EDGE_RELAY = {0: 'Klo', 1: 'Khi', 6: 'Vlo', 7: 'Vhi'} if self.DECOUPLE else {}
        for h in range(NH):
            oj = 'oJ' if h < 4 else 'oK'
            osl = h if h < 4 else h - 4
            rc = h % 4 if h % 4 in self.RELAY_COLS else None
            edge_mt = EDGE_RELAY.get(h)
            if edge_mt is not None:
                flows.append(f"    aie.flow(%sh{h}, DMA : 0, %{edge_mt}, DMA : 4)   // A{h} shim -> {edge_mt} (edge relay)")
                flows.append(f"    aie.flow(%{edge_mt}, DMA : 4, %t{h}, DMA : 0)   // A{h} {edge_mt} -> center")
            elif rc is not None:
                mt_name = ['jA', 'jB', 'oJ', 'oK'][rc]
                s2mm_ch = 4 if h < 4 else 5
                mm2s_ch = 1 if h < 4 else 2
                flows.append(f"    aie.flow(%sh{h}, DMA : 0, %{mt_name}, DMA : {s2mm_ch})   // A{h} shim -> {mt_name}")
                flows.append(f"    aie.flow(%{mt_name}, DMA : {mm2s_ch}, %t{h}, DMA : 0)   // A{h} {mt_name} -> center")
            else:
                flows.append(f"    aie.flow(%sh{h}, DMA : 0, %t{h}, DMA : 0)   // A{h}")
            flows.append(f"    aie.flow(%mx, DMA : 0, %t{h}, DMA : 1)   // mux 3-bcast -> center {h}")
            flows.append(f"    aie.flow(%t{h}, DMA : 0, %sh{h}, DMA : 0)   // Pf{h} -> drain (circuit)")
            flows.append(f"    aie.flow(%t{h}, DMA : 1, %sc{h}, DMA : 0)   // [Qi,O_h]{h} 2-BD -> score{h} (circuit)")
            flows.append(f"    aie.flow(%sc{h}, DMA : 0, %va{h}, DMA : 0)   // inter{h} -> value{h}")
            if not self.SDUMP:
                flows.append(f"    aie.flow(%sc{h}, DMA : 1, %{oj}, DMA : {osl})   // O_h{h} relayed by score -> {oj}[{osl}]")
        flows.append("    aie.flow(%sh3, DMA : 1, %Klo, DMA : 0)   // K_lo -> Klo split")
        flows.append("    aie.flow(%sh4, DMA : 1, %Khi, DMA : 0)   // K_hi -> Khi split")
        flows.append("    aie.flow(%sh5, DMA : 1, %Vlo, DMA : 0)   // V_lo -> Vlo split")
        flows.append("    aie.flow(%sh6, DMA : 1, %Vhi, DMA : 0)   // V_hi -> Vhi split")
        for k in range(4):
            flows.append(f"    aie.flow(%Klo, DMA : {k}, %sc{k}, DMA : 1)   // K[{k}] -> score{k}")
            flows.append(f"    aie.flow(%Khi, DMA : {k}, %sc{k+4}, DMA : 1)   // K[{k+4}] -> score{k+4}")
            flows.append(f"    aie.flow(%Vlo, DMA : {k}, %va{k}, DMA : 1)   // V[{k}] -> value{k}")
            flows.append(f"    aie.flow(%Vhi, DMA : {k}, %va{k+4}, DMA : 1)   // V[{k+4}] -> value{k+4}")
        for k in range(4):
            flows.append(f"    aie.flow(%va{k}, DMA : 0, %jA, DMA : {k})   // Of{k} -> joinA")
            flows.append(f"    aie.flow(%va{k+4}, DMA : 0, %jB, DMA : {k})   // Of{k+4} -> joinB")
        if not self.RL_FIX:
            flows.append("    aie.flow(%jA, DMA : 0, %rl, DMA : 0)   // attn Half0 -> relay")
            flows.append("    aie.flow(%jB, DMA : 0, %rl, DMA : 1)   // attn Half1 -> relay")
        flows.append("    aie.flow(%oJ, DMA : 0, %op, DMA : 0)   // O Half0 -> orelay")
        flows.append("    aie.flow(%oK, DMA : 0, %op, DMA : 1)   // O Half1 -> orelay")
        flows.append("    aie.flow(%op, DMA : 0, %nm, DMA : 0)   // O -> ANM")
        flows.append("    aie.flow(%sh2, DMA : 1, %nm, DMA : 1)   // resid+gain -> ANM (2-BD on S2MM1)")
        if self.SDUMP:
            flows.append("    aie.flow(%sc0, DMA : 1, %sh4, DMA : 1)   // SDUMP: sc0 scores -> @S_alloc (host reads scores)")
        else:
            flows.append("    aie.flow(%nm, DMA : 1, %sh4, DMA : 1)   // s = O+resid (attn-residual) -> arg0 tail")
        if self.ATTN_DUMP:
            flows.append("    aie.flow(%rl, DMA : 1, %sh7, DMA : 1)   // attn_out tap (ATTN_DUMP) -> sh7 S2MM1")
        flows_txt = "\n".join(flows) + "\n"

        # shim allocations
        shim_allocs = "    aie.shim_dma_allocation @X_alloc(%sh0, MM2S, 1)\n"
        shim_allocs += "    aie.shim_dma_allocation @R_alloc(%sh2, MM2S, 1)\n"
        shim_allocs += "    aie.shim_dma_allocation @Klo_alloc(%sh3, MM2S, 1)\n"
        shim_allocs += "    aie.shim_dma_allocation @Khi_alloc(%sh4, MM2S, 1)\n"
        shim_allocs += "    aie.shim_dma_allocation @Vlo_alloc(%sh5, MM2S, 1)\n"
        shim_allocs += "    aie.shim_dma_allocation @Vhi_alloc(%sh6, MM2S, 1)\n"
        shim_allocs += "    aie.shim_dma_allocation @S_alloc(%sh4, S2MM, 1)\n"
        if self.AIE_TRACE:
            # #260: dedicated trace-destination shim. sh7 S2MM ch1 is free
            # (ATTN_DUMP uses it only when enabled; production SDUMP doesn't).
            # sh7 DMA:1 is the working route — DMA:0 collides (South:2 busy).
            shim_allocs += "    aie.shim_dma_allocation @T_alloc(%sh7, S2MM, 1)\n"
        if self.ATTN_DUMP:
            shim_allocs += "    aie.shim_dma_allocation @ATN_alloc(%sh7, S2MM, 1)\n"
        shim_allocs += "".join(
            f'    aie.shim_dma_allocation @A{h}(%sh{h}, MM2S, 0)\n'
            f'    aie.shim_dma_allocation @P{h}(%sh{h}, S2MM, 0)\n' for h in range(NH))

        # #245: grow P_TY by E in ATTN_DUMP to hold the tap region at [(NH+1)*E, (NH+2)*E).
        P_TY = f"{(NH+2)*E}xbf16" if self.ATTN_DUMP else f"{(NH+1)*E}xbf16"
        WT_TY = f"{NH*WT_BYTES}xi8"; KV_TY = f"{2*NH*KVN}xbf16"
        HALF_KV = self.HALF_KV

        # runtime sequence
        rt = []
        if self.AIE_TRACE:
            rt.append(f"""      aie.trace.host_config buffer_size=65536 arg_idx=6
      aie.trace.start_config @trace_sc1
""")
        rt.append(f"""      %tx = aiex.dma_configure_task_for @X_alloc {{
        aie.dma_bd(%arg1 : memref<{self.XR_ELEMS}xbf16>, 0, {XB}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {XB}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%tx)
      %tr = aiex.dma_configure_task_for @R_alloc {{
        aie.dma_bd(%arg1 : memref<{self.XR_ELEMS}xbf16>, {XB}, {2*E}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {2*E}, stride = 1>]) {{burst_length = 0 : i32}}
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
        rt.append(f"""      %ts = aiex.dma_configure_task_for @S_alloc {{
        aie.dma_bd(%arg0 : memref<{P_TY}>, {NH*E}, {E}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {E}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%ts)""")
        if self.AIE_TRACE:
            rt.append(f"""      %ttr = aiex.dma_configure_task_for @T_alloc {{
        aie.dma_bd(%arg3 : memref<65536xi8>, 0, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 65536, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%ttr)""")
        if self.ATTN_DUMP:
            # #245: single-shot shim S2MM task capturing rl_A (pre-O-proj attn_out)
            # from sh7 into arg0 (output BO) at offset (NH+1)*E. One dispatch = one
            # layer iteration, so this captures exactly the current layer's frame
            # (same model as @S_alloc). P_TY grew by E to hold this region.
            rt.append(f"""      %tatn = aiex.dma_configure_task_for @ATN_alloc {{
        aie.dma_bd(%arg0 : memref<{P_TY}>, {(NH+1)*E}, {E}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {E}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%tatn)""")
        _await_extra = "\n      aiex.dma_await_task(%tatn)" if self.ATTN_DUMP else ""
        rt.append("".join(f"      aiex.dma_await_task(%tp{h})\n" for h in range(NH)).rstrip()
                  + "\n      aiex.dma_await_task(%ts)" + _await_extra)
        rt_body = "\n".join(rt) + "\n"
        _wo_ty = "65536xi8" if self.AIE_TRACE else f"{self.WO_BYTES}xi8"
        rt_args = (f"%arg0: memref<{P_TY}>, %arg1: memref<{self.XR_ELEMS}xbf16>, %arg2: memref<{WT_TY}>, "
                   f"%arg3: memref<{_wo_ty}>, %arg4: memref<{KV_TY}>")

        # Build MLIR sections
        centers = "".join(self._center(h) for h in range(NH))
        scores = "".join(self._score(h) + (self._score_trace_ops(h) if h == 1 else '') for h in range(NH))
        values = "".join(self._value(h) for h in range(NH))
        if self.RL_FIX:
            joins = (self._join_memtile('jA', 'jA', with_weight_relay=(0 in self.RELAY_COLS), packet_id=0)
                   + self._join_memtile('jB', 'jB', with_weight_relay=(1 in self.RELAY_COLS), packet_id=2)
                   + self._join_memtile('oJ', 'oJ', with_weight_relay=(2 in self.RELAY_COLS))
                   + self._join_memtile('oK', 'oK', with_weight_relay=(3 in self.RELAY_COLS)))
        else:
            joins = "".join(self._join_memtile(n, n, with_weight_relay=(c in self.RELAY_COLS))
                            for c, n in enumerate(['jA', 'jB', 'oJ', 'oK']))
        if self.DECOUPLE:
            ksplit = (self._split_memtile('Klo', 'Klo', with_weight_relay=True)
                    + self._split_memtile('Khi', 'Khi', with_weight_relay=True)
                    + self._split_memtile('Vlo', 'Vlo', with_weight_relay=True)
                    + self._split_memtile('Vhi', 'Vhi', with_weight_relay=True))
        else:
            ksplit = (self._split_memtile('Klo', 'Klo') + self._split_memtile('Khi', 'Khi')
                    + self._split_memtile('Vlo', 'Vlo') + self._split_memtile('Vhi', 'Vhi'))
        _relay_block = "" if self.RL_FIX else self._relay_block()

        MLIRTXT = (f"module {{\n  aie.device(npu2) {{\n{tile_decls}{shim_decls}\n{funcs}\n"
                   f"{flows_txt}\n{centers}\n{scores}\n{values}\n{joins}\n{ksplit}\n{_relay_block}\n{self._op_block()}\n{self._nm_block()}\n{self._mux_block()}\n"
                   f"{shim_allocs}\n    aie.runtime_sequence({rt_args}) {{\n{rt_body}    }}\n  }}\n}}\n")
        return MLIRTXT


def emit_mlir(**kwargs):
    return OFold8F3BestEmitter(**kwargs).emit_mlir()
