# SPDX-License-Identifier: Apache-2.0
"""Op wrapper for ffn_overlap_probe — 2-phase fused FFN, weight delivery via
ObjectFifo (WMODE=fifo) vs raw-aiex BD-chain (WMODE=chain). Same relay.o/kernels
as decode_ffn16_2mm. The fifo-vs-chain latency delta = the overlap lever. See
design.py."""
import os
import numpy as np
from ml_dtypes import bfloat16
from pathlib import Path

from iron.common import (
    AIEOperatorBase, XclbinArtifact, InstsBinArtifact,
    KernelObjectArtifact, SourceArtifact, PythonGeneratedMLIRArtifact,
)


class AIEFFNOverlapProbe(AIEOperatorBase):
    NPHASE = 2

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

    def get_artifacts(self, prefix="ffn_overlap_probe_"):
        operator_dir = Path(__file__).parent
        E, g = self.embed_dim, self.group_size
        wmode = os.environ.get("WMODE", "fifo")
        # WMODE in artifact name so fifo/chain don't collide in the build cache.
        base = f"{prefix}{wmode}_e{E}_h{self.hidden_dim}"
        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{base}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="my_ffn_overlap_probe",
            callback_args=[self.context.device_manager.device_type,
                           E, self.hidden_dim, g, self.m_input, self.num_cols],
        )
        Hc16 = self.hidden_dim // (self.num_cols * 4)
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
        dn_elems = dn_tiles // (PACKED // DN_PACKED)
        wt_per_tile = gu_tiles + gu_tiles + dn_elems
        # NPHASE phases x (nc cols x 4 tiles) x wt_per_tile rounds x PACKED bytes
        W_BYTES = self.NPHASE * nc * wt_per_tile * 4 * PACKED
        self.add_buffer("O", E, dtype=bfloat16)
        self.add_buffer("W", W_BYTES, dtype=np.uint8)
        self.add_buffer("X", E, dtype=bfloat16)
        self.add_kernel("ffn_overlap_probe", self.xclbin_artifact,
                        self.xclbin_artifact.kernel_name, self.insts_artifact)
        self.add_to_runlist("ffn_overlap_probe", "O", "W", "X")
