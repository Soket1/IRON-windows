# SPDX-License-Identifier: Apache-2.0
"""decode_attn_split operator — F1.2a probe: Q-GEMV on center cols, flowkv attention on
edge cols, Q relayed center→edge. Same kernels/ABI as decode_front_attn (signed v2 GEMV +
rope_il INTERLEAVED + flowkv). See design.py."""
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


class AIEDecodeAttnSplit(AIEOperatorBase):
    def __init__(self, embed_dim=2048, K_gemv=2048, head_dim=64, group_size=32,
                 attn_group=4, num_kv_heads=8, m_input=4, seq_len=32, num_cols=2,
                 center_col_offset=2, edge_col_offset=6, fuse_sv=False, context=None):
        self.embed_dim = embed_dim
        self.K = K_gemv
        self.head_dim = head_dim
        self.group_size = group_size
        self.attn_group = attn_group
        self.num_kv_heads = num_kv_heads
        self.m_input = m_input
        self.seq_len = seq_len
        self.num_cols = num_cols
        self.center_col_offset = center_col_offset
        self.edge_col_offset = edge_col_offset
        self.fuse_sv = fuse_sv
        self.xclbin_artifact = None
        self.insts_artifact = None
        AIEOperatorBase.__init__(self, context=context)

    def get_artifacts(self, prefix="decode_attn_split_"):
        operator_dir = Path(__file__).parent
        base = (f"{prefix}{self.embed_dim}x{self.K}_d{self.head_dim}"
                f"_g{self.group_size}_s{self.seq_len}_a{self.attn_group}"
                f"_kv{self.num_kv_heads}_c{self.num_cols}"
                f"_cc{self.center_col_offset}_ec{self.edge_col_offset}"
                f"{'_fsv' if self.fuse_sv else ''}")

        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{base}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="my_decode_attn_split",
            callback_args=[
                self.context.device_manager.device_type,
                self.embed_dim,
                self.K,
                self.head_dim,
                self.group_size,
                self.attn_group,
                self.num_kv_heads,
                self.seq_len,
                None,                # chunk_size -> seq_len
                self.m_input,
                self.num_cols,
                self.center_col_offset,
                self.edge_col_offset,
                self.fuse_sv,
            ],
        )

        gemv_obj = KernelObjectArtifact.new(
            f"fused_dequant_gemv_v2_signed_{self.K}k_g{self.group_size}.o",
            depends=[SourceArtifact.new(
                self.context.base_dir / "aie_kernels" / "aie2p"
                / "fused_dequant_gemv_v2.cc")],
            extra_flags=[f"-DDIM_K={self.K}", f"-DGROUP_SIZE={self.group_size}",
                         "-DWEIGHT_SIGNED"],
        )
        rope_obj = KernelObjectArtifact.new(
            "rope_il.o",
            depends=[SourceArtifact.new(
                self.context.base_dir / "aie_kernels" / "generic" / "rope.cc")],
            extra_flags=["-DINTERLEAVED", f"-DLUT_OFF={self.K}",
                         f"-DSEQ_META={self.head_dim}"],
        )
        flowkv_obj = KernelObjectArtifact.new(
            f"flowkv_{self.head_dim}d_h{self.attn_group}.o",
            depends=[SourceArtifact.new(
                self.context.base_dir / "aie_kernels" / "aie2p" / "flowkv.cc")],
            extra_flags=[f"-DHEAD_DIM={self.head_dim}",
                         f"-DMAX_Q_HEADS={self.attn_group}"],
        )

        xclbin_artifact = XclbinArtifact.new(
            f"{base}.xclbin", depends=[mlir_artifact, gemv_obj, rope_obj, flowkv_obj],
        )
        insts_artifact = InstsBinArtifact.new(f"{base}.bin", depends=[mlir_artifact])
        return xclbin_artifact, insts_artifact

    def set_up_artifacts(self):
        self.xclbin_artifact, self.insts_artifact = self.get_artifacts()
        self.add_artifacts([self.xclbin_artifact, self.insts_artifact])

    def set_up_runtime(self):
        ng = self.num_kv_heads
        groups = self.K // self.group_size
        packed_tile = self.m_input * self.K // 2 + self.m_input * groups * 2
        q_rows = self.attn_group * self.head_dim
        gemv_tiles = q_rows // self.m_input

        dts = 2
        raw_head = self.seq_len * self.head_dim * dts
        ahs = int((raw_head + 63) / 64) * 64
        ahs_elems = ahs // dts

        self.add_buffer("output", ng * self.attn_group * self.head_dim, dtype=bfloat16)
        self.add_buffer("packed_weights", ng * gemv_tiles * packed_tile, dtype=np.uint8)
        self.add_buffer("vector", self.K + q_rows + 16, dtype=bfloat16)
        self.add_buffer("K_cache", ng * ahs_elems, dtype=bfloat16)
        self.add_buffer("V_cache", ng * ahs_elems, dtype=bfloat16)
        self.add_kernel("decode_attn_split", self.xclbin_artifact,
                        self.xclbin_artifact.kernel_name, self.insts_artifact)
        self.add_to_runlist("decode_attn_split", "output", "packed_weights",
                            "vector", "K_cache", "V_cache")
