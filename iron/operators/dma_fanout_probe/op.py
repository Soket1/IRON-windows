# SPDX-License-Identifier: Apache-2.0
"""Op wrapper for dma_fanout_probe (#34) — pure weight-ingress bandwidth vs
shim column count. No compute kernel (workers pure-drain), so no relay.o dep."""
import numpy as np
from ml_dtypes import bfloat16
from pathlib import Path

from iron.common import (
    AIEOperatorBase, XclbinArtifact, InstsBinArtifact,
    PythonGeneratedMLIRArtifact,
)


class AIEDmaFanoutProbe(AIEOperatorBase):
    def __init__(self, cols=(2, 3, 4, 5), embed_dim=2048, group_size=32,
                 m_input=4, context=None):
        self.cols = tuple(cols)
        self.embed_dim = embed_dim
        self.group_size = group_size
        self.m_input = m_input
        self.xclbin_artifact = None
        self.insts_artifact = None
        AIEOperatorBase.__init__(self, context=context)

    def get_artifacts(self, prefix="dma_fanout_"):
        operator_dir = Path(__file__).parent
        E, g = self.embed_dim, self.group_size
        base = f"{prefix}n{len(self.cols)}_c{'_'.join(map(str, self.cols))}"
        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{base}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="my_dma_fanout_probe",
            callback_args=[self.context.device_manager.device_type,
                           list(self.cols), E, g, self.m_input],
        )
        xclbin_artifact = XclbinArtifact.new(f"{base}.xclbin", depends=[mlir_artifact])
        insts_artifact = InstsBinArtifact.new(f"{base}.bin", depends=[mlir_artifact])
        return xclbin_artifact, insts_artifact

    def set_up_artifacts(self):
        self.xclbin_artifact, self.insts_artifact = self.get_artifacts()
        self.add_artifacts([self.xclbin_artifact, self.insts_artifact])

    def set_up_runtime(self):
        E, g, m = self.embed_dim, self.group_size, self.m_input
        NCOL, R = len(self.cols), 4
        PACKED = m * E // 2 + m * (E // g) * 2
        WT_PER_TILE = 1536 // NCOL
        W_BYTES = NCOL * WT_PER_TILE * R * PACKED
        O_ELEMS = NCOL * R * 32
        self.add_buffer("O", O_ELEMS, dtype=bfloat16)
        self.add_buffer("W", W_BYTES, dtype=np.uint8)
        self.add_kernel("dma_fanout_probe", self.xclbin_artifact,
                        self.xclbin_artifact.kernel_name, self.insts_artifact)
        self.add_to_runlist("dma_fanout_probe", "O", "W")
