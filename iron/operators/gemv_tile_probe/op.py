# SPDX-License-Identifier: Apache-2.0
"""Op wrapper for gemv_tile_probe (#30 (B): real per-tile compute-bound A/B).
Same I/O for both algorithms; kernel symbol selects bcast vs dot. M_OUT outputs
over K, so the kernel is compute-bound (not dispatch-floor-bound)."""
import numpy as np
from ml_dtypes import bfloat16
from pathlib import Path

from iron.common import (
    AIEOperatorBase, XclbinArtifact, InstsBinArtifact,
    KernelObjectArtifact, SourceArtifact, PythonGeneratedMLIRArtifact,
)


class AIEGemvTileProbe(AIEOperatorBase):
    def __init__(self, kernel="bcast", M_OUT=512, K=2048, G=32, context=None):
        self.kernel = kernel        # "bcast" or "dot"
        self.M_OUT = M_OUT; self.K = K; self.G = G
        self.xclbin_artifact = None
        self.insts_artifact = None
        AIEOperatorBase.__init__(self, context=context)

    @property
    def sym(self):
        return {"bcast": "layer_fused_gemv_bcast_tile_bf16",
                "dot": "layer_fused_gemv_dot_tile_bf16",
                "floor": "layer_fused_gemv_floor_tile_bf16",
                "floor_reg": "layer_fused_gemv_floor_reg_tile_bf16",
                "floor_reg2": "layer_fused_gemv_floor_reg2_tile_bf16"}[self.kernel]

    def get_artifacts(self, prefix="gemv_tile_"):
        operator_dir = Path(__file__).parent
        base = f"{prefix}{self.kernel}_m{self.M_OUT}_k{self.K}"
        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{base}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="my_gemv_tile_probe",
            callback_args=[self.context.device_manager.device_type,
                           self.sym, self.M_OUT, self.K, self.G],
        )
        relay_flags = [
            f"-DEMBED_DIM={self.K}", "-DHIDDEN_DIM=8192", f"-DGROUP_SIZE={self.G}",
            "-DHEAD_DIM=64", "-DNUM_HEADS=32", "-DNUM_KV_HEADS=8",
            "-DMAX_SEQ_LEN=2048", "-DNUM_AIE_COLUMNS=16", f"-DM_OUTPUT_MAX={self.M_OUT}",
        ]
        # #30 peano scheduling experiment: extra -mllvm flags via env (e.g.
        # GEMV_MLLVM="-enable-pipeliner -pipeliner-max-stages=4"). Each token is
        # passed as `-mllvm <token>` to the kernel clang.
        import os as _os
        for _tok in _os.environ.get("GEMV_MLLVM", "").split():
            relay_flags += ["-mllvm", _tok]
        relay_flags += _os.environ.get("GEMV_CFLAGS", "").split()
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
        WB = self.M_OUT * self.K // 2; SB = self.M_OUT * (self.K // self.G)
        self.add_buffer("WS", WB + SB * 2, dtype=np.uint8)
        self.add_buffer("X", self.K, dtype=bfloat16)
        self.add_buffer("O", self.M_OUT, dtype=bfloat16)
        self.add_kernel("gemv_tile", self.xclbin_artifact,
                        self.xclbin_artifact.kernel_name, self.insts_artifact)
        self.add_to_runlist("gemv_tile", "WS", "X", "O")
