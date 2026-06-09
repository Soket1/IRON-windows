# SPDX-License-Identifier: Apache-2.0
"""Op wrapper for gemv_bcast_probe (#30 broadcast-GEMV density probe)."""
import numpy as np
from ml_dtypes import bfloat16
from pathlib import Path

from iron.common import (
    AIEOperatorBase, XclbinArtifact, InstsBinArtifact,
    KernelObjectArtifact, SourceArtifact, PythonGeneratedMLIRArtifact,
)


class AIEGemvBcastProbe(AIEOperatorBase):
    def __init__(self, N=32, K=2048, G=32, context=None):
        self.N = N; self.K = K; self.G = G
        self.xclbin_artifact = None
        self.insts_artifact = None
        AIEOperatorBase.__init__(self, context=context)

    def get_artifacts(self, prefix="gemv_bcast_"):
        operator_dir = Path(__file__).parent
        base = f"{prefix}n{self.N}_k{self.K}"
        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{base}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="my_gemv_bcast_probe",
            callback_args=[self.context.device_manager.device_type, self.N, self.K, self.G],
        )
        relay_flags = [
            f"-DEMBED_DIM={self.K}", "-DHIDDEN_DIM=8192", f"-DGROUP_SIZE={self.G}",
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
        WB = self.K * self.N // 2; SB = self.N * (self.K // self.G)
        self.add_buffer("WS", WB + SB * 2, dtype=np.uint8)
        self.add_buffer("X", self.K, dtype=bfloat16)
        self.add_buffer("O", self.N, dtype=bfloat16)
        self.add_kernel("gemv_bcast", self.xclbin_artifact,
                        self.xclbin_artifact.kernel_name, self.insts_artifact)
        self.add_to_runlist("gemv_bcast", "WS", "X", "O")
