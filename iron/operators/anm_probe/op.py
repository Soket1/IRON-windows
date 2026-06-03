# SPDX-License-Identifier: Apache-2.0
"""Op wrapper for the ANM (ADD + RMSNorm) validation probe (see design.py).
Compiles layer_fused.cc (contains add + rms_norm2) with the required -D set."""
import numpy as np
from ml_dtypes import bfloat16
from pathlib import Path

from iron.common import (
    AIEOperatorBase, XclbinArtifact, InstsBinArtifact,
    KernelObjectArtifact, SourceArtifact, PythonGeneratedMLIRArtifact,
)


class AIEANMProbe(AIEOperatorBase):
    def __init__(self, embed_dim=2048, hidden_dim=8192, group_size=32,
                 head_dim=64, num_heads=32, num_kv_heads=8, max_seq_len=2048,
                 num_cols=4, context=None):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.group_size = group_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.max_seq_len = max_seq_len
        self.num_cols = num_cols
        self.xclbin_artifact = None
        self.insts_artifact = None
        AIEOperatorBase.__init__(self, context=context)

    def get_artifacts(self, prefix="anm_probe_"):
        operator_dir = Path(__file__).parent
        E = self.embed_dim
        base = f"{prefix}e{E}"
        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{base}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="my_anm",
            callback_args=[self.context.device_manager.device_type, E],
        )
        kobj = KernelObjectArtifact.new(
            "layer_fused_anm.o",
            depends=[SourceArtifact.new(
                self.context.base_dir / "aie_kernels" / "aie2p" / "layer_fused.cc")],
            extra_flags=[
                f"-DEMBED_DIM={E}",
                f"-DHIDDEN_DIM={self.hidden_dim}",
                f"-DGROUP_SIZE={self.group_size}",
                f"-DHEAD_DIM={self.head_dim}",
                f"-DNUM_HEADS={self.num_heads}",
                f"-DNUM_KV_HEADS={self.num_kv_heads}",
                f"-DMAX_SEQ_LEN={self.max_seq_len}",
                f"-DNUM_AIE_COLUMNS={self.num_cols}",
                f"-DM_OUTPUT_MAX={self.hidden_dim // self.num_cols}",
                f"-DINTER_DIM_PER_COL={self.hidden_dim // self.num_cols}",
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
        E = self.embed_dim
        self.add_buffer("out", E, dtype=bfloat16)
        self.add_buffer("o_out", E, dtype=bfloat16)
        self.add_buffer("inpL", E, dtype=bfloat16)
        self.add_kernel("anm", self.xclbin_artifact,
                        self.xclbin_artifact.kernel_name, self.insts_artifact)
        self.add_to_runlist("anm", "out", "o_out", "inpL")
