# SPDX-License-Identifier: Apache-2.0
"""Op wrapper for single-tile FFN probe (ffn_probe1). Outputs per-column E
partials (no reduction) so the test compares each column's full FFN output."""
import numpy as np
from pathlib import Path

from iron.common import (
    AIEOperatorBase, XclbinArtifact, InstsBinArtifact,
    KernelObjectArtifact, SourceArtifact, PythonGeneratedMLIRArtifact,
)


class AIEFFNProbe1(AIEOperatorBase):
    def __init__(self, embed_dim=2048, hidden_dim=8192, group_size=32,
                 num_cols=4, m_gemv=2, context=None):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.group_size = group_size
        self.num_cols = num_cols
        self.m_gemv = m_gemv
        self.xclbin_artifact = None
        self.insts_artifact = None
        AIEOperatorBase.__init__(self, context=context)

    def get_artifacts(self, prefix="ffn_probe1_"):
        operator_dir = Path(__file__).parent
        E, g = self.embed_dim, self.group_size
        Hc = self.hidden_dim // self.num_cols
        base = f"{prefix}e{E}_h{self.hidden_dim}_g{g}_c{self.num_cols}"
        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{base}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="my_ffn1",
            callback_args=[self.context.device_manager.device_type,
                           E, self.hidden_dim, g, self.m_gemv, self.num_cols],
        )
        kobj = KernelObjectArtifact.new(
            "layer_fused_ffn1.o",
            depends=[SourceArtifact.new(
                self.context.base_dir / "aie_kernels" / "aie2p" / "layer_fused.cc")],
            extra_flags=[
                f"-DEMBED_DIM={E}", f"-DHIDDEN_DIM={self.hidden_dim}",
                f"-DGROUP_SIZE={g}", f"-DHEAD_DIM=64", f"-DNUM_HEADS=32",
                f"-DNUM_KV_HEADS=8", f"-DMAX_SEQ_LEN=2048",
                f"-DNUM_AIE_COLUMNS={self.num_cols}",
                f"-DM_OUTPUT_MAX={max(Hc, E)}", f"-DINTER_DIM_PER_COL={E}",
            ],
        )
        down_obj = KernelObjectArtifact.new(
            f"fused_dequant_gemv_v2_down_{Hc}k_g{g}.o",
            depends=[SourceArtifact.new(
                self.context.base_dir / "aie_kernels" / "aie2p"
                / "fused_dequant_gemv_v2.cc")],
            extra_flags=[f"-DDIM_K={Hc}", f"-DGROUP_SIZE={g}",
                         "-Dfused_dequant_matvec_v2_bf16=down_matvec_v2_bf16"],
        )
        xclbin_artifact = XclbinArtifact.new(f"{base}.xclbin",
                                             depends=[mlir_artifact, kobj, down_obj])
        insts_artifact = InstsBinArtifact.new(f"{base}.bin", depends=[mlir_artifact])
        return xclbin_artifact, insts_artifact

    def set_up_artifacts(self):
        self.xclbin_artifact, self.insts_artifact = self.get_artifacts()
        self.add_artifacts([self.xclbin_artifact, self.insts_artifact])
