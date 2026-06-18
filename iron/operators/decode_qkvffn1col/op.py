# SPDX-License-Identifier: Apache-2.0
"""Op wrapper for decode_qkvffn1col — P6.4a de-risk vehicle. Copy bodies, layer_fused relay only."""
from pathlib import Path

from iron.common import (
    AIEOperatorBase, XclbinArtifact, InstsBinArtifact,
    KernelObjectArtifact, SourceArtifact, PythonGeneratedMLIRArtifact,
)


class AIEDecodeQKVFFN1Col(AIEOperatorBase):
    def __init__(self, embed_dim=2048, context=None):
        self.embed_dim = embed_dim
        self.xclbin_artifact = None
        self.insts_artifact = None
        AIEOperatorBase.__init__(self, context=context)

    def get_artifacts(self, prefix="decode_qkvffn1col_"):
        operator_dir = Path(__file__).parent
        E = self.embed_dim
        base = f"{prefix}e{E}"
        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{base}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="my_decode_qkvffn1col",
            callback_args=[self.context.device_manager.device_type, E],
        )
        _relay_flags = [
            f"-DEMBED_DIM={E}", f"-DHIDDEN_DIM=8192",
            f"-DGROUP_SIZE=32", f"-DHEAD_DIM=64",
            f"-DNUM_HEADS=32", f"-DNUM_KV_HEADS=8", f"-DMAX_SEQ_LEN=2048",
            f"-DNUM_AIE_COLUMNS=16", f"-DM_OUTPUT_MAX=512",
        ]
        relay_obj = KernelObjectArtifact.new(
            "layer_fused_relay.o",
            depends=[SourceArtifact.new(
                self.context.base_dir / "aie_kernels" / "aie2p" / "layer_fused.cc")],
            extra_flags=_relay_flags,
        )
        xclbin_artifact = XclbinArtifact.new(
            f"{base}.xclbin", depends=[mlir_artifact, relay_obj])
        insts_artifact = InstsBinArtifact.new(f"{base}.bin", depends=[mlir_artifact])
        return xclbin_artifact, insts_artifact

    def set_up_artifacts(self):
        self.xclbin_artifact, self.insts_artifact = self.get_artifacts()
        self.add_artifacts([self.xclbin_artifact, self.insts_artifact])

    def set_up_runtime(self):
        pass
