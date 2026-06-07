# SPDX-License-Identifier: Apache-2.0
"""Op wrapper for decode_back_real (#11.1b — fused back-half, real bodies)."""
import numpy as np
from ml_dtypes import bfloat16
from pathlib import Path

from iron.common import (
    AIEOperatorBase, XclbinArtifact, InstsBinArtifact,
    KernelObjectArtifact, SourceArtifact, PythonGeneratedMLIRArtifact,
)


class AIEDecodeBackReal(AIEOperatorBase):
    def __init__(self, embed_dim=2048, hidden_dim=8192, group_size=32,
                 m_input=4, num_cols=4, context=None):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.group_size = group_size
        self.m_input = m_input
        self.num_cols = num_cols
        self.xclbin_artifact = None
        self.insts_artifact = None
        AIEOperatorBase.__init__(self, context=context)

    def get_artifacts(self, prefix="decode_back_real_"):
        operator_dir = Path(__file__).parent
        E, g = self.embed_dim, self.group_size
        base = f"{prefix}e{E}_h{self.hidden_dim}"
        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{base}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="my_decode_back_real",
            callback_args=[self.context.device_manager.device_type,
                           E, self.hidden_dim, g, self.m_input, self.num_cols],
        )
        gemv_obj = None
        relay_flags = [
            f"-DEMBED_DIM={E}", f"-DHIDDEN_DIM={self.hidden_dim}",
            f"-DGROUP_SIZE={g}", "-DHEAD_DIM=64", "-DNUM_HEADS=32",
            "-DNUM_KV_HEADS=8", "-DMAX_SEQ_LEN=2048", "-DNUM_AIE_COLUMNS=16",
            "-DM_OUTPUT_MAX=512",
        ]
        relay_obj = KernelObjectArtifact.new(
            "layer_fused_relay.o",
            depends=[SourceArtifact.new(
                self.context.base_dir / "aie_kernels" / "aie2p" / "layer_fused.cc")],
            extra_flags=relay_flags,
        )
        xclbin_artifact = XclbinArtifact.new(
            f"{base}.xclbin", depends=[mlir_artifact, relay_obj])
        insts_artifact = InstsBinArtifact.new(f"{base}.bin", depends=[mlir_artifact])
        return xclbin_artifact, insts_artifact

    def set_up_artifacts(self):
        self.xclbin_artifact, self.insts_artifact = self.get_artifacts()
        self.add_artifacts([self.xclbin_artifact, self.insts_artifact])

    def set_up_runtime(self):
        E, g, m, nc = self.embed_dim, self.group_size, self.m_input, self.num_cols
        H = self.hidden_dim
        R = 4; NT = nc * R
        Hc16 = H // NT
        o_tiles = (E // NT) // m
        gu_tiles = Hc16 // m
        PACKED = m * E // 2 + m * (E // g) * 2
        DN_PACKED = m * Hc16 // 2 + m * (Hc16 // g) * 2
        DN_SUB = PACKED // DN_PACKED
        dn_elems = (E // m) // DN_SUB
        WT_PER_TILE = o_tiles + gu_tiles + gu_tiles + dn_elems
        W_BYTES = nc * WT_PER_TILE * R * PACKED
        import os as _os
        self.add_buffer("O", E, dtype=bfloat16)
        self.add_buffer("W", W_BYTES, dtype=np.uint8)
        self.add_buffer("ATTN", E, dtype=bfloat16)
        self.add_buffer("HIN", E, dtype=bfloat16)
        self.add_kernel("decode_back_real", self.xclbin_artifact,
                        self.xclbin_artifact.kernel_name, self.insts_artifact)
        if bool(_os.environ.get("DBG_BR_DUMP0")):
            self.add_buffer("DUMP", E, dtype=bfloat16)
            self.add_to_runlist("decode_back_real", "O", "W", "ATTN", "HIN", "DUMP")
        else:
            self.add_to_runlist("decode_back_real", "O", "W", "ATTN", "HIN")
