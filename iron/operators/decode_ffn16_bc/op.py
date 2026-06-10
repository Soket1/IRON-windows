# SPDX-License-Identifier: Apache-2.0
"""Op wrapper for decode_ffn16 — D2.2b 16-tile parallel FFN. See design.py.
relay.o recompiled with INTER_DIM_PER_COL=Hc16=512 (down K) and M_OUTPUT_MAX=512
(gate/up per-tile output). Reuses layer_fused gate_up/silu/down + reduce4."""
import numpy as np
from ml_dtypes import bfloat16
from pathlib import Path

from iron.common import (
    AIEOperatorBase, XclbinArtifact, InstsBinArtifact,
    KernelObjectArtifact, SourceArtifact, PythonGeneratedMLIRArtifact,
)


class AIEDecodeFFN16(AIEOperatorBase):
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

    def get_artifacts(self, prefix="decode_ffn16bc_"):
        operator_dir = Path(__file__).parent
        E, g = self.embed_dim, self.group_size
        base = f"{prefix}e{E}_h{self.hidden_dim}"
        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{base}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="my_decode_ffn16",
            callback_args=[self.context.device_manager.device_type,
                           E, self.hidden_dim, g, self.m_input, self.num_cols],
        )
        Hc16 = self.hidden_dim // (self.num_cols * 4)   # 512
        # INTER_DIM_PER_COL is computed inside layer_fused.cc as
        # HIDDEN_DIM/NUM_AIE_COLUMNS (unconditional #define — a -D flag is
        # IGNORED). For the 16-way split down K must = Hc16, so drive it via
        # NUM_AIE_COLUMNS = total tiles (16): 8192/16 = 512.
        N_TILES = self.num_cols * 4
        _relay_flags = [
            f"-DEMBED_DIM={E}", f"-DHIDDEN_DIM={self.hidden_dim}",
            f"-DGROUP_SIZE={g}", f"-DHEAD_DIM=64",
            f"-DNUM_HEADS=32", f"-DNUM_KV_HEADS=8", f"-DMAX_SEQ_LEN=2048",
            f"-DNUM_AIE_COLUMNS={N_TILES}",
            f"-DM_OUTPUT_MAX={max(Hc16, 256)}",
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
        E, g, m, nc = self.embed_dim, self.group_size, self.m_input, self.num_cols
        Hc16 = self.hidden_dim // (nc * 4)
        gu_tiles = Hc16 // m
        dn_tiles = E // m
        PACKED = m * E // 2 + m * (E // g) * 2
        DN_PACKED = m * Hc16 // 2 + m * (Hc16 // g) * 2
        dn_elems = dn_tiles // (PACKED // DN_PACKED)     # unpadded: 4 down/elem
        wt_per_tile = gu_tiles + gu_tiles + dn_elems     # 384
        W_BYTES = nc * wt_per_tile * 4 * PACKED
        self.add_buffer("O", E, dtype=bfloat16)
        self.add_buffer("W", W_BYTES, dtype=np.uint8)
        self.add_buffer("X", E, dtype=bfloat16)
        self.add_buffer("D3", 64, dtype=bfloat16)
        self.add_buffer("D4", 64, dtype=bfloat16)
        self.add_kernel("decode_ffn16", self.xclbin_artifact,
                        self.xclbin_artifact.kernel_name, self.insts_artifact)
        self.add_to_runlist("decode_ffn16", "O", "W", "X", "D3", "D4")
