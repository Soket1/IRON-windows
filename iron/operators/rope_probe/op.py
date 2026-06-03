# SPDX-License-Identifier: Apache-2.0
"""Op wrapper for the on-NPU RoPE validation probe (see design.py).
Compiles aie_kernels/generic/rope.cc with -DTWO_HALVES."""
import numpy as np
from ml_dtypes import bfloat16
from pathlib import Path

from iron.common import (
    AIEOperatorBase,
    XclbinArtifact,
    InstsBinArtifact,
    KernelObjectArtifact,
    SourceArtifact,
    PythonGeneratedMLIRArtifact,
)


class AIERopeProbe(AIEOperatorBase):
    def __init__(self, head_dim=64, context=None):
        self.head_dim = head_dim
        self.xclbin_artifact = None
        self.insts_artifact = None
        AIEOperatorBase.__init__(self, context=context)

    def get_artifacts(self, prefix="rope_probe_"):
        operator_dir = Path(__file__).parent
        base = f"{prefix}d{self.head_dim}_th"
        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{base}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="my_rope",
            callback_args=[self.context.device_manager.device_type, self.head_dim],
        )
        rope_obj = KernelObjectArtifact.new(
            "rope_th.o",
            depends=[SourceArtifact.new(
                self.context.base_dir / "aie_kernels" / "generic" / "rope.cc")],
            extra_flags=["-DTWO_HALVES"],
        )
        xclbin_artifact = XclbinArtifact.new(f"{base}.xclbin",
                                             depends=[mlir_artifact, rope_obj])
        insts_artifact = InstsBinArtifact.new(f"{base}.bin", depends=[mlir_artifact])
        return xclbin_artifact, insts_artifact

    def set_up_artifacts(self):
        self.xclbin_artifact, self.insts_artifact = self.get_artifacts()
        self.add_artifacts([self.xclbin_artifact, self.insts_artifact])

    def set_up_runtime(self):
        self.add_buffer("out", self.head_dim, dtype=bfloat16)
        self.add_buffer("inp", self.head_dim, dtype=bfloat16)
        self.add_buffer("lut", self.head_dim, dtype=bfloat16)
        self.add_kernel("rope_probe", self.xclbin_artifact,
                        self.xclbin_artifact.kernel_name, self.insts_artifact)
        self.add_to_runlist("rope_probe", "out", "inp", "lut")
