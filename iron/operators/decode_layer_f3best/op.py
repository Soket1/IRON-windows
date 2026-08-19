# SPDX-License-Identifier: Apache-2.0
"""decode_layer_f3best operator — full fused decode layer in ONE dispatch.

8 phase-blind center tiles (cols 2-5 x rows 2-3) time-mux Q-GEMV+RoPE -> O-proj ->
FFN on one weight stream A=[Wq|Wo|gate|up|down]; 8 score + 8 value edge tiles run
spatial-8 attention; one continuous BD-chain. The MLIR is authored as raw aie-dialect
(below IRON ObjectFifo) by design.my_decode_layer_f3best -> f3best_emit; aiecc is the
same backend the IRON operators use.

Kernel objects are the same five as decode_attn_oproj (signed int4 GEMV + its O-proj
rename + interleaved RoPE + flowkv + attn concat) PLUS the fused relay kernel built
from layer_fused_f3best.cc. Shape is fixed to Llama-3.2-1B (the emitter validates);
the BO layout matches the standalone replay (bo0 out, bo1 X|resid, bo2 weights A,
bo3 unused Wo placeholder, bo4 KV cache)."""
import numpy as np
from ml_dtypes import bfloat16
from pathlib import Path

from iron.common import (
    AIEOperatorBase,
    XclbinArtifact,
    InstsBinArtifact,
    KernelObjectArtifact,
    SourceArtifact,
    PythonGeneratedMLIRArtifact,
)
from iron.operators.flowkv_decode.contract import (
    flowkv_geometry_flags,
    flowkv_kernel_object_name,
)


class AIEDecodeLayerF3Best(AIEOperatorBase):
    # f3best center layout is fixed: 8 center tiles, npu2.
    NH = 8
    WO_BYTES = 2359296          # unused arg3 placeholder (Wo lives inside A)

    def __init__(self, embed_dim=2048, hidden_dim=8192, K_gemv=2048, head_dim=64,
                 group_size=32, attn_group=4, num_kv_heads=8, m_input=4, seq_len=256,
                 num_q_heads=32, with_npu_kv=False, context=None):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.K = K_gemv
        self.head_dim = head_dim
        self.group_size = group_size
        self.attn_group = attn_group
        self.num_kv_heads = num_kv_heads
        self.m_input = m_input
        self.seq_len = seq_len
        self.num_q_heads = num_q_heads
        self.with_npu_kv = with_npu_kv
        # Compute WT_TILES from the same pack_bcast tile-count formulas the emitter
        # uses (tiles = (n_rows/32)*(K_pack/256)). For 1B this gives 896 (no-KV) /
        # 960 (KV); for 3B it gives 1440 (no-KV) / 1504 (KV). Hardcoding was wrong
        # for any E != 2048.
        _N = 32; _KC = 256
        _per_tile = embed_dim // self.NH
        _h8 = hidden_dim // self.NH
        _q_t = (_per_tile // _N) * (embed_dim // _KC)
        _gu_t = (_h8 // _N) * (embed_dim // _KC)
        _dn_t = (embed_dim // _N) * (_h8 // _KC)
        _kv_t = 128 // m_input if with_npu_kv else 0   # KV_M=128, same as emitter
        self.WT_TILES = 2 * _q_t + 2 * _gu_t + _dn_t + 2 * _kv_t
        self.xclbin_artifact = None
        self.insts_artifact = None
        AIEOperatorBase.__init__(self, context=context)

    def get_artifacts(self, prefix="decode_layer_f3best_"):
        operator_dir = Path(__file__).parent
        E, H, g = self.embed_dim, self.hidden_dim, self.group_size
        import os as _os
        _ffn_div = _os.environ.get('F3BEST_FFN_DIV', '1')
        _div_suffix = f'_d{_ffn_div}' if _ffn_div != '1' else ''
        _decouple = '_decouple' if _os.environ.get('F3BEST_MT_DECOUPLE', '').strip() != '' else ''
        # triple-B: the emitter treats ONLY '0' as force-off ('1'/unset = auto), so the
        # key must use the same test. Testing for "non-empty" made F3BEST_TRIPLE_B=0
        # emit a single-B design under the `_tb` name, silently colliding with the real
        # triple-B artifact in the shared cache directory.
        # #211/#212: triple-B (F3BEST_TRIPLE_B=1) collapses in attention from layer 0.
        # Root = ARCHITECTURAL: circuit-flow mux->center multicast has NO per-consumer
        # back-pressure; the center S2MM1 round-robin (b0->b1->b2) and the center core
        # phase index are two independent state machines that drift because the 8 centers
        # finish phases at different times (weight-stream contention). A slow center
        # writes attn_out into B0 while its core expects x -> permanent 1-phase lag, no
        # resync. The mux lock protocol (non-RL_FIX, the production branch) was ALREADY
        # balanced (all 3 phases acquire mx_op / release mx_oc) — an earlier lock fix
        # landed in the DEAD RL_FIX branch and had zero effect. Fix paths (a)-(g)
        # exhausted (per-buffer S2MM needs 3 S2MM, cap is 2; cross-tile locks N/A;
        # packet-tagged doesn't route to different buffers). Default = single-B
        # (rock-solid, 47.30 t/s, 1 passed/0 failed in harness); triple-B opt-in.
        _triple_b = _os.environ.get('F3BEST_TRIPLE_B', '0').strip()
        _triple_b = '_tb' if _triple_b == '1' else ''
        _cpp_bcast = '_cpp'  # #188: pure C++ bcast path (layer_fused_*_bcast_bf16 in layer_fused_relay.o); hand-asm .s dropped
        _qdump = '_qdump7' if _os.environ.get('F3BEST_QDUMP', '').strip() != '' else ''  # #188 Ш21 Direct-NPU Q dump (v6 = fixed hand-asm kc256.s + output-width store)
        _sdump = '_sdump' if _os.environ.get('F3BEST_SDUMP', '').strip() != '' else ''  # #259: score-tile dump — per-position Q·K scores from %It via @S_alloc (like QDUMP)
        _b0dump = '_b0dump' if _os.environ.get('F3BEST_B0DUMP', '').strip() != '' else ''  # #211: dump B0 contents via Pf0 drain (snapshot x_bundle delivery)
        _attnout = '_attnout' if _os.environ.get('F3BEST_ATTN_DUMP', '').strip() != '' else ''  # #245: tapped debug-build — rl tile dumps pre-O-proj attn_out into a host-visible debug BO
        _kv_abi = '_kv' if self.with_npu_kv else '_nokv'
        # uni_partial size (#206/#207): participates in the L1 layout, so it
        # must participate in the cache key.
        _uni = _os.environ.get('F3BEST_UNI_SZ', '128')
        _uni_sfx = '' if _uni == '128' else f'_u{_uni}'
        flowkv_tuning = (
            "-DFLOWKV_PRESCALE_Q=1",
            "-DFLOWKV_VEC_EXP=1",
            "-DFLOWKV_VALUE_AMAC=1",
        )
        flowkv_obj_name = flowkv_kernel_object_name(
            self.head_dim, self.attn_group, self.seq_len, flowkv_tuning
        )
        # The object filename is the complete fixed FlowKV specialization: cache
        # the outer xclbin under a tag derived from it rather than raw environment.
        flowkv_obj_tag = f"_fkobj_{flowkv_obj_name.removesuffix('.o')}"
        flowkv_flags = flowkv_geometry_flags(
            self.head_dim, self.attn_group, self.seq_len
        )
        # KernelObjectArtifact freshness does not include extra_flags. Keep every
        # flag-specialized object in its own filename namespace so a shared build
        # directory can never reuse a 1B object for a 3B xclbin.
        rope_obj_name = f"rope_il_k{self.K}_d{self.head_dim}.o"
        relay_obj_name = (
            f"layer_fused_relay_e{E}_h{H}_g{g}_d{self.head_dim}"
            f"_qh{self.num_q_heads}_kvh{self.num_kv_heads}_s2048_c8_m1024.o"
        )
        base = (f"{prefix}{E}x{H}_d{self.head_dim}_g{g}_s{self.seq_len}"
                f"_a{self.attn_group}_kv{self.num_kv_heads}_mc_preq_vexp_vreg_dq8_qp_mxp_ub_amac_ug2{_kv_abi}{_div_suffix}{_decouple}{_triple_b}{_cpp_bcast}{_qdump}{_sdump}{_b0dump}{_attnout}{_uni_sfx}{flowkv_obj_tag}_fkfix2_silu2_mxpp_objid2_abi3_al64")  # _al64 = #235: L1 buffer ALLOCATIONS (Q_SZ_ALLOC/ITC_ALLOC) padded to 32 bf16 so every attention buffer stays 64 B-aligned even when aiecc falls back from bank-aware to contiguous placement (3B: va*_Of landed at addr%32==12 under an aie::store_v of a 32 B vector). _silu = #210: FFN gate/up/down all live in the C++ statics (lf_left/right/silu_buf); the MLIR was calling the explicit-pointer SiLU on IRON buffers nobody wrote, so down consumed the raw gate. _fkfix2 = #208 root: F3BEST_MT_DECOUPLE edge MemTile weight relay corrupts KV+center — drop decouple from all presets (root #208). _cpp = pure C++ bcast path (#188 fix — hand-asm .s dropped: baseline NaN, RR ~500x blow-up), _qdump = #188 Ш21 direct-NPU Q dump (routes Q into the O_h relay), _mc = multi-chunk attn fix (#70/#74), _preq/_vexp = flowkv score density cuts, _vreg = register-resident value accumulator, _dq8 = single int4->int8 unpack, _qp = 4 groups/iteration in two chains, _mxp = mx packet demux fix (#142), _ub = unified bcast GEMV (#157 L1), _amac = native bf16 MAC in value tile (#176)

        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{base}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="my_decode_layer_f3best",
            callback_args=[
                self.context.device_manager.device_type,
                E, H, g, self.head_dim, self.num_kv_heads,
                self.attn_group, self.seq_len, self.with_npu_kv, flowkv_obj_name,
                rope_obj_name, relay_obj_name,
            ],
        )

        k2p = self.context.base_dir / "aie_kernels" / "aie2p"
        gen = self.context.base_dir / "aie_kernels" / "generic"

        # SIGNED int4 GEMV (Q-proj): on-chip (nib-8)*scale, host packs (nib-8)&0xF.
        gemv_obj = KernelObjectArtifact.new(
            f"fused_dequant_gemv_v2_signed_{self.K}k_g{g}.o",
            depends=[SourceArtifact.new(k2p / "fused_dequant_gemv_v2.cc")],
            extra_flags=[f"-DDIM_K={self.K}", f"-DGROUP_SIZE={g}", "-DWEIGHT_SIGNED"],
        )
        # Same GEMV body renamed for the O-proj phase (distinct link symbol).
        oproj_obj = KernelObjectArtifact.new(
            f"fused_dequant_gemv_v2_oproj_signed_{self.K}k_g{g}.o",
            depends=[SourceArtifact.new(k2p / "fused_dequant_gemv_v2.cc")],
            extra_flags=[f"-DDIM_K={self.K}", f"-DGROUP_SIZE={g}", "-DWEIGHT_SIGNED",
                         "-Dfused_dequant_matvec_v2_bf16=oproj_matvec_v2_bf16"],
        )
        rope_obj = KernelObjectArtifact.new(
            rope_obj_name,
            depends=[SourceArtifact.new(gen / "rope.cc")],
            extra_flags=["-DINTERLEAVED", f"-DLUT_OFF={self.K}",
                         f"-DSEQ_META={self.head_dim}"],
        )
        flowkv_obj = KernelObjectArtifact.new(
            flowkv_obj_name,
            depends=[SourceArtifact.new(k2p / "flowkv.cc")],
            extra_flags=[*flowkv_flags, *flowkv_tuning],
        )
        concat_obj = KernelObjectArtifact.new(
            "attn_concat.o",
            depends=[SourceArtifact.new(k2p / "attn_concat.cc")],
        )
        # Fused relay: the phase-blind center body (Q-GEMV+RoPE / O-proj / FFN) +
        # rms hub + mux, built for the 8-center-column f3best layout.
        # #188: pure C++ bcast path (layer_fused_*_bcast_bf16); hand-asm .s dropped.
        relay_obj = KernelObjectArtifact.new(
            relay_obj_name,
            depends=[SourceArtifact.new(k2p / "layer_fused_f3best.cc")],
            extra_flags=[
                f"-DEMBED_DIM={E}", f"-DHIDDEN_DIM={H}", f"-DGROUP_SIZE={g}",
                f"-DHEAD_DIM={self.head_dim}", f"-DNUM_HEADS={self.num_q_heads}",
                f"-DNUM_KV_HEADS={self.num_kv_heads}", f"-DMAX_SEQ_LEN=2048",
                "-DNUM_AIE_COLUMNS=8", "-DM_OUTPUT_MAX=1024",
            ],
        )

        xclbin_artifact = XclbinArtifact.new(
            f"{base}.xclbin",
            depends=[mlir_artifact, gemv_obj, oproj_obj, rope_obj, flowkv_obj,
                     concat_obj, relay_obj],
        )
        insts_artifact = InstsBinArtifact.new(f"{base}.bin", depends=[mlir_artifact])
        return xclbin_artifact, insts_artifact

    def set_up_artifacts(self):
        self.xclbin_artifact, self.insts_artifact = self.get_artifacts()
        self.add_artifacts([self.xclbin_artifact, self.insts_artifact])

    def set_up_runtime(self):
        NH, E, g, m = self.NH, self.embed_dim, self.group_size, self.m_input
        PACKED = m * E // 2 + m * (E // g) * 2
        WT_BYTES = self.WT_TILES * PACKED
        KVN = self.seq_len * self.head_dim

        # no-KV ABI: [NH*P | s], 18432 bf16. KV ABI: [NH*(P|K|V) | s],
        # with KV_M=128 bf16 per head per projection, 20480 bf16 total.
        # bo1 [X(XB) | residual(E) | ffn_norm gain(E)]; bo2 weights A;
        # bo3 unused Wo placeholder; bo4 KV cache (K then V, NH heads).
        KV_M = 128 if self.with_npu_kv else 0
        self.add_buffer("output", NH * (E + 2*KV_M) + E, dtype=bfloat16)
        # #187: q_rows = attn_group*head_dim varies (1B:256, 3B:384). Was hardcoded
        # 256 (1B-only), which under-allocated XR by (q_rows-256) bf16 for 3B,
        # mis-matching host XB=E+q_rows+16 and the emitter's parametric x_bundle.
        q_rows = self.attn_group * self.head_dim
        self.add_buffer("XR", (E + q_rows + 16) + E + E, dtype=bfloat16)
        self.add_buffer("A", NH * WT_BYTES, dtype=np.uint8)
        self.add_buffer("Wo", self.WO_BYTES, dtype=np.uint8)
        self.add_buffer("KV", 2 * NH * KVN, dtype=bfloat16)
        self.add_kernel("decode_layer_f3best", self.xclbin_artifact,
                        self.xclbin_artifact.kernel_name, self.insts_artifact)
        self.add_to_runlist("decode_layer_f3best", "output", "XR", "A", "Wo", "KV")
