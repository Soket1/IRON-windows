# SPDX-License-Identifier: Apache-2.0
"""Op wrapper for decode_ffn_mono (standalone monolithic-FFN single-tile test)."""
import numpy as np
from ml_dtypes import bfloat16
from pathlib import Path

from iron.common import (
    AIEOperatorBase, XclbinArtifact, InstsBinArtifact,
    KernelObjectArtifact, SourceArtifact, PythonGeneratedMLIRArtifact,
)


class AIEDecodeFFNMono(AIEOperatorBase):
    def __init__(self, embed_dim=2048, hc16=512, group_size=32, m_input=4, context=None):
        self.embed_dim = embed_dim
        self.hc16 = hc16
        self.group_size = group_size
        self.m_input = m_input
        self.xclbin_artifact = None
        self.insts_artifact = None
        AIEOperatorBase.__init__(self, context=context)

    def get_artifacts(self, prefix="decode_ffn_mono_"):
        operator_dir = Path(__file__).parent
        E, g = self.embed_dim, self.group_size
        base = f"{prefix}e{E}_hc{self.hc16}"
        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{base}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="my_decode_ffn_mono",
            callback_args=[self.context.device_manager.device_type,
                           E, self.hc16, g, self.m_input],
        )
        relay_flags = [
            f"-DEMBED_DIM={E}", "-DHIDDEN_DIM=8192", f"-DGROUP_SIZE={g}",
            "-DHEAD_DIM=64", "-DNUM_HEADS=32", "-DNUM_KV_HEADS=8",
            "-DMAX_SEQ_LEN=2048", "-DNUM_AIE_COLUMNS=16", "-DM_OUTPUT_MAX=512",
        ]
        relay_obj = KernelObjectArtifact.new(
            "layer_fused_relay.o",
            depends=[SourceArtifact.new(
                self.context.base_dir / "aie_kernels" / "aie2p" / "layer_fused.cc")],
            extra_flags=relay_flags,
        )
        xclbin_artifact = XclbinArtifact.new(f"{base}.xclbin",
                                             depends=[mlir_artifact, relay_obj])
        insts_artifact = InstsBinArtifact.new(f"{base}.bin", depends=[mlir_artifact])
        return xclbin_artifact, insts_artifact

    def set_up_artifacts(self):
        self.xclbin_artifact, self.insts_artifact = self.get_artifacts()
        self.add_artifacts([self.xclbin_artifact, self.insts_artifact])

    def set_up_runtime(self):
        E, g, m = self.embed_dim, self.group_size, self.m_input
        Hc = self.hc16
        PK = m * E // 2 + m * (E // g) * 2
        GUD = 2 * PK + m * E // 2 + E * 2
        n_chunks = Hc // m
        self.add_buffer("O", E, dtype=bfloat16)
        self.add_buffer("W", n_chunks * GUD, dtype=np.uint8)
        self.add_buffer("X", E, dtype=bfloat16)
        self.add_kernel("decode_ffn_mono", self.xclbin_artifact,
                        self.xclbin_artifact.kernel_name, self.insts_artifact)
        self.add_to_runlist("decode_ffn_mono", "O", "W", "X")
