# SPDX-License-Identifier: Apache-2.0
"""Multi-.o link probe operator: one core linking INT4 GEMV (.o #1) + bf16
flowkv attention (.o #2). See design.py. Mirrors fused_dequant_gemv_v2/op.py
but declares TWO KernelObjectArtifacts so AIECC must link both objects."""
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


class AIEDecodeFront(AIEOperatorBase):
    def __init__(self, embed_dim=2048, K=2048, head_dim=64, group_size=32,
                 m_input=2, context=None):
        self.embed_dim = embed_dim
        self.K = K
        self.head_dim = head_dim
        self.group_size = group_size
        self.m_input = m_input
        self.xclbin_artifact = None
        self.insts_artifact = None
        AIEOperatorBase.__init__(self, context=context)

    def get_artifacts(self, prefix="decode_front_"):
        operator_dir = Path(__file__).parent
        base = f"{prefix}{self.embed_dim}x{self.K}_d{self.head_dim}_g{self.group_size}"

        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{base}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="my_decode_front",
            callback_args=[
                self.context.device_manager.device_type,
                self.embed_dim,
                self.K,
                self.head_dim,
                self.group_size,
            ],
        )

        gemv_obj = KernelObjectArtifact.new(
            f"fused_dequant_gemv_v2_{self.K}k_g{self.group_size}.o",
            depends=[SourceArtifact.new(
                self.context.base_dir / "aie_kernels" / "aie2p"
                / "fused_dequant_gemv_v2.cc")],
            extra_flags=[f"-DDIM_K={self.K}", f"-DGROUP_SIZE={self.group_size}"],
        )
        flowkv_obj = KernelObjectArtifact.new(
            f"flowkv_{self.head_dim}d.o",
            depends=[SourceArtifact.new(
                self.context.base_dir / "aie_kernels" / "aie2p" / "flowkv.cc")],
            extra_flags=[f"-DHEAD_DIM={self.head_dim}"],
        )

        xclbin_artifact = XclbinArtifact.new(
            f"{base}.xclbin", depends=[mlir_artifact, gemv_obj, flowkv_obj],
        )
        insts_artifact = InstsBinArtifact.new(f"{base}.bin", depends=[mlir_artifact])
        return xclbin_artifact, insts_artifact

    def set_up_artifacts(self):
        self.xclbin_artifact, self.insts_artifact = self.get_artifacts()
        self.add_artifacts([self.xclbin_artifact, self.insts_artifact])

    def set_up_runtime(self):
        groups = self.K // self.group_size
        packed_tile = self.m_input * self.K // 2 + self.m_input * groups * 2
        self.add_buffer("packed_weights", packed_tile, dtype=np.uint8)
        self.add_buffer("vector", self.K, dtype=bfloat16)
        self.add_buffer("output", self.m_input, dtype=bfloat16)
        self.add_kernel("decode_front", self.xclbin_artifact,
                        self.xclbin_artifact.kernel_name, self.insts_artifact)
        self.add_to_runlist("decode_front", "packed_weights", "vector", "output")
