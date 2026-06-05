# SPDX-License-Identifier: Apache-2.0
"""Op wrapper for the fused decode layer (increment A: QKV→attn→join→O_proj).
Links v2 GEMV (QKV + O_proj), rope (TWO_HALVES), flowkv attention kernels."""
import numpy as np
from ml_dtypes import bfloat16
from pathlib import Path

from iron.common import (
    AIEOperatorBase, XclbinArtifact, InstsBinArtifact,
    KernelObjectArtifact, SourceArtifact, PythonGeneratedMLIRArtifact,
)


class AIEDecodeLayer(AIEOperatorBase):
    def __init__(self, embed_dim=2048, head_dim=64, group_size=32, attn_group=8,
                 seq_len=32, m_input=2, num_cols=4, output_first=False,
                 hidden_dim=8192, stub_ffn=False, context=None):
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.group_size = group_size
        self.attn_group = attn_group
        self.seq_len = seq_len
        self.m_input = m_input
        self.num_cols = num_cols
        self.output_first = output_first
        self.hidden_dim = hidden_dim
        self.stub_ffn = stub_ffn
        self.xclbin_artifact = None
        self.insts_artifact = None
        AIEOperatorBase.__init__(self, context=context)

    def get_artifacts(self, prefix="decode_layer_"):
        operator_dir = Path(__file__).parent
        E, g, nc = self.embed_dim, self.group_size, self.num_cols
        Hc = self.hidden_dim // nc
        of = "_ofirst" if self.output_first else ""
        sf = "" if self.stub_ffn else "_ffn"
        base = f"{prefix}e{E}_d{self.head_dim}_g{g}_s{self.seq_len}_h{self.hidden_dim}{of}{sf}"
        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{base}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="my_decode_layer",
            callback_args=[self.context.device_manager.device_type,
                           E, self.head_dim, g, self.attn_group, self.seq_len,
                           self.output_first, self.hidden_dim, self.stub_ffn],
        )
        gemv_obj = KernelObjectArtifact.new(
            f"fused_dequant_gemv_v2_{E}k_g{g}.o",
            depends=[SourceArtifact.new(
                self.context.base_dir / "aie_kernels" / "aie2p"
                / "fused_dequant_gemv_v2.cc")],
            extra_flags=[f"-DDIM_K={E}", f"-DGROUP_SIZE={g}"],
        )
        # O_proj: same source, renamed entry symbol so its C output arg can be
        # typed o_slice (512) for the join instead of the QKV q-buffer (578).
        oproj_obj = KernelObjectArtifact.new(
            f"fused_dequant_gemv_v2_oproj_{E}k_g{g}.o",
            depends=[SourceArtifact.new(
                self.context.base_dir / "aie_kernels" / "aie2p"
                / "fused_dequant_gemv_v2.cc")],
            extra_flags=[f"-DDIM_K={E}", f"-DGROUP_SIZE={g}",
                         "-Dfused_dequant_matvec_v2_bf16=oproj_matvec_v2_bf16"],
        )
        down_obj = KernelObjectArtifact.new(
            f"fused_dequant_gemv_v2_down_{Hc}k_g{g}.o",
            depends=[SourceArtifact.new(
                self.context.base_dir / "aie_kernels" / "aie2p"
                / "fused_dequant_gemv_v2.cc")],
            extra_flags=[f"-DDIM_K={Hc}", f"-DGROUP_SIZE={g}",
                         "-Dfused_dequant_matvec_v2_bf16=down_matvec_v2_bf16"],
        )
        rope_obj = KernelObjectArtifact.new(
            "rope_th.o",
            depends=[SourceArtifact.new(
                self.context.base_dir / "aie_kernels" / "generic" / "rope.cc")],
            extra_flags=["-DTWO_HALVES"],
        )
        flowkv_obj = KernelObjectArtifact.new(
            f"flowkv_{self.head_dim}d_h{self.attn_group}.o",
            depends=[SourceArtifact.new(
                self.context.base_dir / "aie_kernels" / "aie2p" / "flowkv.cc")],
            extra_flags=[f"-DHEAD_DIM={self.head_dim}",
                         f"-DMAX_Q_HEADS={self.attn_group}"],
        )
        # relay/ANM kernel (layer_fused_add) for the join→broadcast relay tile
        E = self.embed_dim
        relay_obj = KernelObjectArtifact.new(
            "layer_fused_relay.o",
            depends=[SourceArtifact.new(
                self.context.base_dir / "aie_kernels" / "aie2p" / "layer_fused.cc")],
            extra_flags=[
                f"-DEMBED_DIM={E}", f"-DHIDDEN_DIM={self.hidden_dim}",
                f"-DGROUP_SIZE={g}", f"-DHEAD_DIM={self.head_dim}",
                f"-DNUM_HEADS={self.num_heads if hasattr(self,'num_heads') else 32}",
                f"-DNUM_KV_HEADS=8", f"-DMAX_SEQ_LEN=2048",
                f"-DNUM_AIE_COLUMNS={nc}",
                f"-DM_OUTPUT_MAX={max(Hc, E)}", f"-DINTER_DIM_PER_COL={E}",
            ],
        )
        xclbin_artifact = XclbinArtifact.new(
            f"{base}.xclbin",
            depends=[mlir_artifact, gemv_obj, oproj_obj, down_obj, rope_obj,
                     flowkv_obj, relay_obj])
        insts_artifact = InstsBinArtifact.new(f"{base}.bin", depends=[mlir_artifact])
        return xclbin_artifact, insts_artifact

    def set_up_artifacts(self):
        self.xclbin_artifact, self.insts_artifact = self.get_artifacts()
        self.add_artifacts([self.xclbin_artifact, self.insts_artifact])

    def set_up_runtime(self):
        E, g, m, nc = self.embed_dim, self.group_size, self.m_input, self.num_cols
        Hc = self.hidden_dim // nc
        groups = E // g
        packed_tile = m * E // 2 + m * groups * 2
        gemv_tiles = (self.attn_group * self.head_dim) // m
        o_tiles = (E // nc) // m
        o_packed = m * E // 2 + m * groups * 2
        gu_tiles = Hc // m
        dn_tiles = E // m
        dn_groups = Hc // g
        ffnw_per_col = 2*gu_tiles*(m*E//2+m*(E//g)*2) + dn_tiles*(m*Hc//2+m*dn_groups*2)
        dts = 2
        ahs = int((self.seq_len * self.head_dim * dts + 63) / 64) * 64
        w_region = nc * gemv_tiles * packed_tile
        ow_region = nc * o_tiles * o_packed
        ffn_region = nc * ffnw_per_col
        WT_elems = w_region + ow_region + ffn_region
        self.add_buffer("WT", WT_elems, dtype=np.uint8)
        self.add_buffer("X", 2 * E, dtype=bfloat16)
        self.add_buffer("Kc", nc * (ahs // dts), dtype=bfloat16)
        self.add_buffer("Vc", nc * (ahs // dts), dtype=bfloat16)
        self.add_buffer("O", E, dtype=bfloat16)
        self.add_kernel("decode_layer", self.xclbin_artifact,
                        self.xclbin_artifact.kernel_name, self.insts_artifact)
        self.add_to_runlist("decode_layer", "WT", "X", "Kc", "Vc", "O")
