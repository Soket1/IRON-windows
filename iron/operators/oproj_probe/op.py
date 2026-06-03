# SPDX-License-Identifier: Apache-2.0
"""Op wrapper for the standalone O_proj validation probe (see design.py).
Compiles layer_fused.cc (which contains layer_fused_o_proj_bf16) with the full
-D macro set it requires (INTER_DIM_PER_COL has no #ifndef default)."""
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


class AIEOProjProbe(AIEOperatorBase):
    def __init__(self, embed_dim=2048, hidden_dim=8192, group_size=32,
                 head_dim=64, num_heads=32, num_kv_heads=8, max_seq_len=2048,
                 num_cols=4, m_input=2, context=None):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.group_size = group_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.max_seq_len = max_seq_len
        self.num_cols = num_cols
        self.m_input = m_input
        self.xclbin_artifact = None
        self.insts_artifact = None
        AIEOperatorBase.__init__(self, context=context)

    def get_artifacts(self, prefix="oproj_probe_"):
        operator_dir = Path(__file__).parent
        E, g = self.embed_dim, self.group_size
        base = f"{prefix}e{E}_g{g}_c{self.num_cols}"
        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{base}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="my_oproj",
            callback_args=[self.context.device_manager.device_type,
                           E, g, self.num_cols],
        )
        kobj = KernelObjectArtifact.new(
            f"fused_dequant_gemv_v2_{E}k_g{g}.o",
            depends=[SourceArtifact.new(
                self.context.base_dir / "aie_kernels" / "aie2p"
                / "fused_dequant_gemv_v2.cc")],
            extra_flags=[f"-DDIM_K={E}", f"-DGROUP_SIZE={g}"],
        )
        xclbin_artifact = XclbinArtifact.new(f"{base}.xclbin",
                                             depends=[mlir_artifact, kobj])
        insts_artifact = InstsBinArtifact.new(f"{base}.bin", depends=[mlir_artifact])
        return xclbin_artifact, insts_artifact

    def set_up_artifacts(self):
        self.xclbin_artifact, self.insts_artifact = self.get_artifacts()
        self.add_artifacts([self.xclbin_artifact, self.insts_artifact])

    def set_up_runtime(self):
        E, g = self.embed_dim, self.group_size
        rows_per_col = E // self.num_cols
        groups = E // g
        packed_tile = self.m_input * E // 2 + self.m_input * groups * 2
        tiles_per_col = rows_per_col // self.m_input
        self.add_buffer("output", E, dtype=bfloat16)
        self.add_buffer("weights", self.num_cols * tiles_per_col * packed_tile,
                        dtype=np.uint8)
        self.add_buffer("attn_out", E, dtype=bfloat16)
        self.add_kernel("oproj", self.xclbin_artifact,
                        self.xclbin_artifact.kernel_name, self.insts_artifact)
        self.add_to_runlist("oproj", "output", "weights", "attn_out")
