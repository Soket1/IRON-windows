# SPDX-License-Identifier: Apache-2.0
"""Op wrapper for the SwiGLU validation probe (see design.py). Compiles
layer_fused.cc (gate_up + silu_mul + down_partial). M_OUTPUT_MAX must be >= Hc
(gate/up write lf_left/right of size hidden_per_col)."""
import numpy as np
from ml_dtypes import bfloat16
from pathlib import Path

from iron.common import (
    AIEOperatorBase, XclbinArtifact, InstsBinArtifact,
    KernelObjectArtifact, SourceArtifact, PythonGeneratedMLIRArtifact,
)


class AIESwiGLUProbe(AIEOperatorBase):
    def __init__(self, embed_dim=2048, hidden_dim=8192, hidden_per_col=512,
                 group_size=32, head_dim=64, num_heads=32, num_kv_heads=8,
                 max_seq_len=2048, num_cols=4, m_gemv=2, context=None):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.hidden_per_col = hidden_per_col
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

    def get_artifacts(self, prefix="swiglu_probe_"):
        operator_dir = Path(__file__).parent
        E, Hc, g = self.embed_dim, self.hidden_per_col, self.group_size
        base = f"{prefix}e{E}_hc{Hc}_g{g}"
        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{base}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="my_swiglu",
            callback_args=[self.context.device_manager.device_type,
                           E, Hc, g, self.m_gemv],
        )
        # INTER_DIM_PER_COL = Hc (down partial K). M_OUTPUT_MAX >= Hc.
        kobj = KernelObjectArtifact.new(
            "layer_fused_swiglu.o",
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
        xclbin_artifact = XclbinArtifact.new(f"{base}.xclbin",
                                             depends=[mlir_artifact, kobj])
        insts_artifact = InstsBinArtifact.new(f"{base}.bin", depends=[mlir_artifact])
        return xclbin_artifact, insts_artifact

    def set_up_artifacts(self):
        self.xclbin_artifact, self.insts_artifact = self.get_artifacts()
        self.add_artifacts([self.xclbin_artifact, self.insts_artifact])

    def set_up_runtime(self):
        E, Hc, g, m = self.embed_dim, self.hidden_per_col, self.group_size, self.m_gemv
        gu_groups = E // g
        gu_packed = m * E // 2 + m * gu_groups * 2
        gu_tiles = Hc // m
        dn_groups = Hc // g
        dn_packed = m * Hc // 2 + m * dn_groups * 2
        dn_tiles = E // m
        self.add_buffer("out", E, dtype=bfloat16)
        self.add_buffer("gu_w", 2 * gu_tiles * gu_packed, dtype=np.uint8)
        self.add_buffer("ffn_in", E, dtype=bfloat16)
        self.add_buffer("dn_w", dn_tiles * dn_packed, dtype=np.uint8)
        self.add_kernel("swiglu", self.xclbin_artifact,
                        self.xclbin_artifact.kernel_name, self.insts_artifact)
        self.add_to_runlist("swiglu", "out", "gu_w", "ffn_in", "dn_w")
