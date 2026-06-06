# SPDX-License-Identifier: Apache-2.0
"""Op wrapper for decode_red16 — D2.2a broadcast-to-16 + 2-level reduce skeleton.
Reuses layer_fused_relay.o (add + reduce4). See design.py."""
import numpy as np
from ml_dtypes import bfloat16
from pathlib import Path

from iron.common import (
    AIEOperatorBase, XclbinArtifact, InstsBinArtifact,
    KernelObjectArtifact, SourceArtifact, PythonGeneratedMLIRArtifact,
)


class AIEDecodeRed16(AIEOperatorBase):
    def __init__(self, embed_dim=2048, num_cols=4, context=None):
        self.embed_dim = embed_dim
        self.num_cols = num_cols
        self.xclbin_artifact = None
        self.insts_artifact = None
        AIEOperatorBase.__init__(self, context=context)

    def get_artifacts(self, prefix="decode_red16_"):
        operator_dir = Path(__file__).parent
        E = self.embed_dim
        base = f"{prefix}e{E}"
        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{base}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="my_decode_red16",
            callback_args=[self.context.device_manager.device_type,
                           E, self.num_cols],
        )
        # relay.o (add + reduce4). Hidden/inter dims irrelevant for the skeleton;
        # keep the same flags shape as decode_tile so the source compiles.
        nc = self.num_cols
        Hc = (4 * E) // nc
        _relay_flags = [
            f"-DEMBED_DIM={E}", f"-DHIDDEN_DIM={4 * E}",
            f"-DGROUP_SIZE=32", f"-DHEAD_DIM=64",
            f"-DNUM_HEADS=32", f"-DNUM_KV_HEADS=8", f"-DMAX_SEQ_LEN=2048",
            f"-DNUM_AIE_COLUMNS={nc}",
            f"-DM_OUTPUT_MAX={max(Hc, E)}", f"-DINTER_DIM_PER_COL={Hc}",
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
        E = self.embed_dim
        self.add_buffer("O", E, dtype=bfloat16)
        self.add_buffer("D1", 64, dtype=bfloat16)
        self.add_buffer("X", E, dtype=bfloat16)
        self.add_buffer("D3", 64, dtype=bfloat16)
        self.add_buffer("D4", 64, dtype=bfloat16)
        self.add_kernel("decode_red16", self.xclbin_artifact,
                        self.xclbin_artifact.kernel_name, self.insts_artifact)
        self.add_to_runlist("decode_red16", "O", "D1", "X", "D3", "D4")
