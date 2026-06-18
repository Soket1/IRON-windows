# SPDX-License-Identifier: Apache-2.0
"""Op wrapper for decode_qkvffn16_b1 — P6.4b baseline (one reused E-join, pure IRON)."""
import numpy as np
from ml_dtypes import bfloat16
from pathlib import Path

from iron.common import (
    AIEOperatorBase, XclbinArtifact, InstsBinArtifact,
    KernelObjectArtifact, SourceArtifact, PythonGeneratedMLIRArtifact,
)


class AIEDecodeQKVFFN16B1(AIEOperatorBase):
    def __init__(self, embed_dim=2048, qkv_dim=3072, hidden_dim=8192, group_size=32,
                 m_input=4, num_cols=4, context=None):
        self.embed_dim = embed_dim
        self.qkv_dim = qkv_dim
        self.hidden_dim = hidden_dim
        self.group_size = group_size
        self.m_input = m_input
        self.num_cols = num_cols
        self.xclbin_artifact = None
        self.insts_artifact = None
        AIEOperatorBase.__init__(self, context=context)

    def get_artifacts(self, prefix="decode_qkvffn16_b1_"):
        operator_dir = Path(__file__).parent
        E, g, H = self.embed_dim, self.group_size, self.hidden_dim
        base = f"{prefix}e{E}_q{self.qkv_dim}_h{H}"
        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{base}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="my_decode_qkvffn16_b1",
            callback_args=[self.context.device_manager.device_type,
                           E, self.qkv_dim, H, g, self.m_input, self.num_cols],
        )
        gemv_obj = KernelObjectArtifact.new(
            f"fused_dequant_gemv_v2_{E}k_g{g}.o",
            depends=[SourceArtifact.new(
                self.context.base_dir / "aie_kernels" / "aie2p"
                / "fused_dequant_gemv_v2.cc")],
            extra_flags=[f"-DDIM_K={E}", f"-DGROUP_SIZE={g}"],
        )
        NT = self.num_cols * 4
        Hc16 = H // NT
        _relay_flags = [
            f"-DEMBED_DIM={E}", f"-DHIDDEN_DIM={H}",
            f"-DGROUP_SIZE={g}", f"-DHEAD_DIM=64",
            f"-DNUM_HEADS=32", f"-DNUM_KV_HEADS=8", f"-DMAX_SEQ_LEN=2048",
            f"-DNUM_AIE_COLUMNS={NT}", f"-DM_OUTPUT_MAX={max(Hc16, 256)}",
        ]
        relay_obj = KernelObjectArtifact.new(
            "layer_fused_relay.o",
            depends=[SourceArtifact.new(
                self.context.base_dir / "aie_kernels" / "aie2p" / "layer_fused.cc")],
            extra_flags=_relay_flags,
        )
        xclbin_artifact = XclbinArtifact.new(
            f"{base}.xclbin", depends=[mlir_artifact, gemv_obj, relay_obj])
        insts_artifact = InstsBinArtifact.new(f"{base}.bin", depends=[mlir_artifact])
        return xclbin_artifact, insts_artifact

    def set_up_artifacts(self):
        self.xclbin_artifact, self.insts_artifact = self.get_artifacts()
        self.add_artifacts([self.xclbin_artifact, self.insts_artifact])

    def set_up_runtime(self):
        pass
