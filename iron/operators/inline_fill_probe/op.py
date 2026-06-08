# SPDX-License-Identifier: Apache-2.0
"""Op wrapper for inline_fill_probe (BD-chain vs ObjectFifo fill latency probe)."""
import os
import numpy as np
from ml_dtypes import bfloat16
from pathlib import Path

from iron.common import (
    AIEOperatorBase, XclbinArtifact, InstsBinArtifact,
    KernelObjectArtifact, SourceArtifact, PythonGeneratedMLIRArtifact,
)


class AIEInlineFillProbe(AIEOperatorBase):
    def __init__(self, n=2048, n_fills=8, context=None):
        self.n = n
        self.n_fills = n_fills
        self.xclbin_artifact = None
        self.insts_artifact = None
        AIEOperatorBase.__init__(self, context=context)

    def get_artifacts(self, prefix="inline_fill_probe_"):
        operator_dir = Path(__file__).parent
        mode = os.environ.get("PROBE_MODE", "fifo")
        base = f"{prefix}{mode}_n{self.n}_f{self.n_fills}"
        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{base}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="my_inline_fill_probe",
            callback_args=[self.context.device_manager.device_type, self.n, self.n_fills],
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
        self.add_buffer("W", self.n_fills * self.n, dtype=bfloat16)
        self.add_buffer("OUT", self.n, dtype=bfloat16)
        self.add_kernel("inline_fill_probe", self.xclbin_artifact,
                        self.xclbin_artifact.kernel_name, self.insts_artifact)
        self.add_to_runlist("inline_fill_probe", "W", "OUT")
