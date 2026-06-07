# SPDX-License-Identifier: Apache-2.0
"""Op wrapper for decode_back_fused (#11.1a — fused back-half, 16-tile)."""
import numpy as np
from ml_dtypes import bfloat16
from pathlib import Path

from iron.common import (
    AIEOperatorBase, XclbinArtifact, InstsBinArtifact,
    KernelObjectArtifact, SourceArtifact, PythonGeneratedMLIRArtifact,
)


class AIEDecodeBackFused(AIEOperatorBase):
    def __init__(self, embed_dim=2048, num_cols=4, context=None):
        self.embed_dim = embed_dim
        self.num_cols = num_cols
        self.xclbin_artifact = None
        self.insts_artifact = None
        AIEOperatorBase.__init__(self, context=context)

    def get_artifacts(self, prefix="decode_back_fused_"):
        operator_dir = Path(__file__).parent
        base = f"{prefix}e{self.embed_dim}"
        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{base}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="my_decode_back_fused",
            callback_args=[self.context.device_manager.device_type,
                           self.embed_dim, self.num_cols],
        )
        relay_flags = [
            "-DEMBED_DIM=2048", "-DHIDDEN_DIM=8192", "-DGROUP_SIZE=32",
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
        E = self.embed_dim
        self.add_buffer("O", E, dtype=bfloat16)
        self.add_buffer("ATTN", E, dtype=bfloat16)
        self.add_buffer("HIN", E, dtype=bfloat16)
        self.add_kernel("decode_back_fused", self.xclbin_artifact,
                        self.xclbin_artifact.kernel_name, self.insts_artifact)
        self.add_to_runlist("decode_back_fused", "O", "ATTN", "HIN")
