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
                 context=None):
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.group_size = group_size
        self.attn_group = attn_group
        self.seq_len = seq_len
        self.m_input = m_input
        self.num_cols = num_cols
        self.output_first = output_first
        self.xclbin_artifact = None
        self.insts_artifact = None
        AIEOperatorBase.__init__(self, context=context)

    def get_artifacts(self, prefix="decode_layer_"):
        operator_dir = Path(__file__).parent
        E, g = self.embed_dim, self.group_size
        of = "_ofirst" if self.output_first else ""
        base = f"{prefix}e{E}_d{self.head_dim}_g{g}_s{self.seq_len}{of}"
        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{base}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="my_decode_layer",
            callback_args=[self.context.device_manager.device_type,
                           E, self.head_dim, g, self.attn_group, self.seq_len,
                           self.output_first],
        )
        gemv_obj = KernelObjectArtifact.new(
            f"fused_dequant_gemv_v2_{E}k_g{g}.o",
            depends=[SourceArtifact.new(
                self.context.base_dir / "aie_kernels" / "aie2p"
                / "fused_dequant_gemv_v2.cc")],
            extra_flags=[f"-DDIM_K={E}", f"-DGROUP_SIZE={g}"],
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
                f"-DEMBED_DIM={E}", f"-DHIDDEN_DIM={E*4}",
                f"-DGROUP_SIZE={self.group_size}", f"-DHEAD_DIM={self.head_dim}",
                f"-DNUM_HEADS={self.num_heads if hasattr(self,'num_heads') else 32}",
                f"-DNUM_KV_HEADS=8", f"-DMAX_SEQ_LEN=2048",
                f"-DNUM_AIE_COLUMNS={self.num_cols}",
                f"-DM_OUTPUT_MAX={E}", f"-DINTER_DIM_PER_COL={E}",
            ],
        )
        xclbin_artifact = XclbinArtifact.new(
            f"{base}.xclbin",
            depends=[mlir_artifact, gemv_obj, rope_obj, flowkv_obj, relay_obj])
        insts_artifact = InstsBinArtifact.new(f"{base}.bin", depends=[mlir_artifact])
        return xclbin_artifact, insts_artifact

    def set_up_artifacts(self):
        self.xclbin_artifact, self.insts_artifact = self.get_artifacts()
        self.add_artifacts([self.xclbin_artifact, self.insts_artifact])

    def set_up_runtime(self):
        E, g, m = self.embed_dim, self.group_size, self.m_input
        nc = self.num_cols
        q_rows = self.attn_group * self.head_dim
        groups = E // g
        packed_tile = m * E // 2 + m * groups * 2
        gemv_tiles = q_rows // m
        o_tiles = (E // nc) // m
        o_packed = m * E // 2 + m * groups * 2
        dts = 2
        ahs = int((self.seq_len * self.head_dim * dts + 63) / 64) * 64
        w_region = nc * gemv_tiles * packed_tile
        ow_region = nc * o_tiles * o_packed
        # FFLM-style single weight BO (all GEMV weights): QKV region + O region.
        self.add_buffer("WT", w_region + ow_region, dtype=np.uint8)
        self.add_buffer("X", E, dtype=bfloat16)
        self.add_buffer("Kc", nc * (ahs // dts), dtype=bfloat16)
        self.add_buffer("Vc", nc * (ahs // dts), dtype=bfloat16)
        self.add_buffer("O", E, dtype=bfloat16)
        self.add_kernel("decode_layer", self.xclbin_artifact,
                        self.xclbin_artifact.kernel_name, self.insts_artifact)
        self.add_to_runlist("decode_layer", "WT", "X", "Kc", "Vc", "O")
