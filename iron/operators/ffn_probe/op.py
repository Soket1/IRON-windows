# SPDX-License-Identifier: Apache-2.0
"""Op wrapper for the 4-column SwiGLU FFN probe (see design.py). Compiles
layer_fused.cc (gate_up + silu_mul + add) and the v2 GEMV for down (K=Hc)."""
import numpy as np
from ml_dtypes import bfloat16
from pathlib import Path

from iron.common import (
    AIEOperatorBase, XclbinArtifact, InstsBinArtifact,
    KernelObjectArtifact, SourceArtifact, PythonGeneratedMLIRArtifact,
)


class AIEFFNProbe(AIEOperatorBase):
    def __init__(self, embed_dim=2048, hidden_dim=8192, group_size=32,
                 head_dim=64, num_heads=32, num_kv_heads=8, max_seq_len=2048,
                 num_cols=4, m_gemv=2, context=None):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.group_size = group_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.max_seq_len = max_seq_len
        self.num_cols = num_cols
        self.m_gemv = m_gemv
        self.xclbin_artifact = None
        self.insts_artifact = None
        AIEOperatorBase.__init__(self, context=context)

    def get_artifacts(self, prefix="ffn_probe_"):
        operator_dir = Path(__file__).parent
        E, g = self.embed_dim, self.group_size
        Hc = self.hidden_dim // self.num_cols
        base = f"{prefix}e{E}_h{self.hidden_dim}_g{g}_c{self.num_cols}"
        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{base}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="my_ffn",
            callback_args=[self.context.device_manager.device_type,
                           E, self.hidden_dim, g, self.m_gemv, self.num_cols],
        )
        # INTER_DIM_PER_COL = Hc (down partial K). M_OUTPUT_MAX >= max(Hc, E).
        kobj = KernelObjectArtifact.new(
            "layer_fused_ffn.o",
            depends=[SourceArtifact.new(
                self.context.base_dir / "aie_kernels" / "aie2p" / "layer_fused.cc")],
            extra_flags=[
                f"-DEMBED_DIM={E}",
                f"-DHIDDEN_DIM={self.hidden_dim}",
                f"-DGROUP_SIZE={g}",
                f"-DHEAD_DIM={self.head_dim}",
                f"-DNUM_HEADS={self.num_heads}",
                f"-DNUM_KV_HEADS={self.num_kv_heads}",
                f"-DMAX_SEQ_LEN={self.max_seq_len}",
                f"-DNUM_AIE_COLUMNS={self.num_cols}",
                f"-DM_OUTPUT_MAX={max(Hc, E)}",
                f"-DINTER_DIM_PER_COL={Hc}",
            ],
        )
        down_kobj = KernelObjectArtifact.new(
            f"fused_dequant_gemv_v2_{Hc}k_g{g}.o",
            depends=[SourceArtifact.new(
                self.context.base_dir / "aie_kernels" / "aie2p"
                / "fused_dequant_gemv_v2.cc")],
            extra_flags=[f"-DDIM_K={Hc}", f"-DGROUP_SIZE={g}"],
        )
        xclbin_artifact = XclbinArtifact.new(f"{base}.xclbin",
                                             depends=[mlir_artifact, kobj, down_kobj])
        insts_artifact = InstsBinArtifact.new(f"{base}.bin", depends=[mlir_artifact])
        return xclbin_artifact, insts_artifact

    def set_up_artifacts(self):
        self.xclbin_artifact, self.insts_artifact = self.get_artifacts()
        self.add_artifacts([self.xclbin_artifact, self.insts_artifact])
