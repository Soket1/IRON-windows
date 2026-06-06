# SPDX-License-Identifier: Apache-2.0
"""Op wrapper for decode_ffn4d — D2.5 direct-delivery FFN overlap diagnostic."""
import numpy as np
from ml_dtypes import bfloat16
from pathlib import Path

from iron.common import (
    AIEOperatorBase, XclbinArtifact, InstsBinArtifact,
    KernelObjectArtifact, SourceArtifact, PythonGeneratedMLIRArtifact,
)


class AIEDecodeFFN4D(AIEOperatorBase):
    def __init__(self, embed_dim=2048, hidden_dim=8192, group_size=32,
                 m_input=4, context=None):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.group_size = group_size
        self.m_input = m_input
        self.xclbin_artifact = None
        self.insts_artifact = None
        AIEOperatorBase.__init__(self, context=context)

    def get_artifacts(self, prefix="decode_ffn4d_"):
        operator_dir = Path(__file__).parent
        E, g = self.embed_dim, self.group_size
        base = f"{prefix}e{E}_h{self.hidden_dim}"
        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{base}.mlir", import_path=operator_dir / "design.py",
            callback_fn="my_decode_ffn4d",
            callback_args=[self.context.device_manager.device_type,
                           E, self.hidden_dim, g, self.m_input])
        N_TILES = 16
        Hc16 = self.hidden_dim // N_TILES
        relay_obj = KernelObjectArtifact.new(
            "layer_fused_relay.o",
            depends=[SourceArtifact.new(
                self.context.base_dir / "aie_kernels" / "aie2p" / "layer_fused.cc")],
            extra_flags=[
                f"-DEMBED_DIM={E}", f"-DHIDDEN_DIM={self.hidden_dim}",
                f"-DGROUP_SIZE={g}", f"-DHEAD_DIM=64", f"-DNUM_HEADS=32",
                f"-DNUM_KV_HEADS=8", f"-DMAX_SEQ_LEN=2048",
                f"-DNUM_AIE_COLUMNS={N_TILES}", f"-DM_OUTPUT_MAX={max(Hc16,256)}"])
        xclbin_artifact = XclbinArtifact.new(f"{base}.xclbin",
                                             depends=[mlir_artifact, relay_obj])
        insts_artifact = InstsBinArtifact.new(f"{base}.bin", depends=[mlir_artifact])
        return xclbin_artifact, insts_artifact

    def set_up_artifacts(self):
        self.xclbin_artifact, self.insts_artifact = self.get_artifacts()
        self.add_artifacts([self.xclbin_artifact, self.insts_artifact])

    def set_up_runtime(self):
        E, g, m = self.embed_dim, self.group_size, self.m_input
        Hc16 = self.hidden_dim // 16
        gu_tiles = Hc16 // m
        dn_tiles = E // m
        PACKED = m * E // 2 + m * (E // g) * 2
        DN_PACKED = m * Hc16 // 2 + m * (Hc16 // g) * 2
        dn_elems = dn_tiles // (PACKED // DN_PACKED)
        wt_per_tile = gu_tiles + gu_tiles + dn_elems
        W_BYTES = 4 * wt_per_tile * PACKED
        self.add_buffer("O", E, dtype=bfloat16)
        self.add_buffer("W", W_BYTES, dtype=np.uint8)
        self.add_buffer("X", E, dtype=bfloat16)
        self.add_buffer("D3", 64, dtype=bfloat16)
        self.add_buffer("D4", 64, dtype=bfloat16)
        self.add_kernel("decode_ffn4d", self.xclbin_artifact,
                        self.xclbin_artifact.kernel_name, self.insts_artifact)
        self.add_to_runlist("decode_ffn4d", "O", "W", "X", "D3", "D4")
