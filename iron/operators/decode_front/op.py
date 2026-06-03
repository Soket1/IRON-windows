# SPDX-License-Identifier: Apache-2.0
"""Fused decode-front operator: one-column gemv→rope→score→value chain.
Links three kernel objects: INT4 GEMV + rope (TWO_HALVES) + flowkv attention.
See design.py."""
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
    def __init__(self, embed_dim=2048, K_gemv=2048, head_dim=64, group_size=32,
                 attn_group=4, m_input=2, seq_len=32, context=None):
        self.embed_dim = embed_dim
        self.K = K_gemv
        self.head_dim = head_dim
        self.group_size = group_size
        self.attn_group = attn_group
        self.m_input = m_input
        self.seq_len = seq_len
        self.xclbin_artifact = None
        self.insts_artifact = None
        AIEOperatorBase.__init__(self, context=context)

    def get_artifacts(self, prefix="decode_front_"):
        operator_dir = Path(__file__).parent
        base = (f"{prefix}{self.embed_dim}x{self.K}_d{self.head_dim}"
                f"_g{self.group_size}_s{self.seq_len}")

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
                self.attn_group,
                self.seq_len,
            ],
        )

        gemv_obj = KernelObjectArtifact.new(
            f"fused_dequant_gemv_v2_{self.K}k_g{self.group_size}.o",
            depends=[SourceArtifact.new(
                self.context.base_dir / "aie_kernels" / "aie2p"
                / "fused_dequant_gemv_v2.cc")],
            extra_flags=[f"-DDIM_K={self.K}", f"-DGROUP_SIZE={self.group_size}"],
        )
        rope_obj = KernelObjectArtifact.new(
            "rope_th.o",
            depends=[SourceArtifact.new(
                self.context.base_dir / "aie_kernels" / "generic" / "rope.cc")],
            extra_flags=["-DTWO_HALVES"],
        )
        flowkv_obj = KernelObjectArtifact.new(
            f"flowkv_{self.head_dim}d.o",
            depends=[SourceArtifact.new(
                self.context.base_dir / "aie_kernels" / "aie2p" / "flowkv.cc")],
            extra_flags=[f"-DHEAD_DIM={self.head_dim}"],
        )

        xclbin_artifact = XclbinArtifact.new(
            f"{base}.xclbin", depends=[mlir_artifact, gemv_obj, rope_obj, flowkv_obj],
        )
        insts_artifact = InstsBinArtifact.new(f"{base}.bin", depends=[mlir_artifact])
        return xclbin_artifact, insts_artifact

    def set_up_artifacts(self):
        self.xclbin_artifact, self.insts_artifact = self.get_artifacts()
        self.add_artifacts([self.xclbin_artifact, self.insts_artifact])

    def set_up_runtime(self):
        groups = self.K // self.group_size
        packed_tile = self.m_input * self.K // 2 + self.m_input * groups * 2
        q_rows = self.attn_group * self.head_dim
        tiles = q_rows // self.m_input

        dts = 2
        raw_head = self.seq_len * self.head_dim * dts
        ahs = int((raw_head + 63) / 64) * 64

        self.add_buffer("packed_weights", tiles * packed_tile, dtype=np.uint8)
        self.add_buffer("vector", self.K, dtype=bfloat16)
        self.add_buffer("K_cache", ahs // dts, dtype=bfloat16)
        self.add_buffer("V_cache", ahs // dts, dtype=bfloat16)
        self.add_buffer("output", self.attn_group * self.head_dim, dtype=bfloat16)
        self.add_kernel("decode_front", self.xclbin_artifact,
                        self.xclbin_artifact.kernel_name, self.insts_artifact)
        self.add_to_runlist("decode_front", "packed_weights", "vector",
                            "K_cache", "V_cache", "output")
