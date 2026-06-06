# SPDX-License-Identifier: Apache-2.0
"""Op wrapper for decode_tile — D2.1 isolated single-tile time-mux GEMV chain.
Reuses decode_layer's kernels (v2 GEMV qkv+oproj, layer_fused gate_up/silu/down).
Buffers: output first (bo0) so xclbin_replay dumps it. See design.py."""
import numpy as np
from ml_dtypes import bfloat16
from pathlib import Path

from iron.common import (
    AIEOperatorBase, XclbinArtifact, InstsBinArtifact,
    KernelObjectArtifact, SourceArtifact, PythonGeneratedMLIRArtifact,
)


class AIEDecodeTile(AIEOperatorBase):
    def __init__(self, embed_dim=2048, head_dim=64, group_size=32, attn_group=8,
                 hidden_dim=8192, m_input=4, num_cols=4, context=None):
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.group_size = group_size
        self.attn_group = attn_group
        self.hidden_dim = hidden_dim
        self.m_input = m_input
        self.num_cols = num_cols
        self.xclbin_artifact = None
        self.insts_artifact = None
        AIEOperatorBase.__init__(self, context=context)

    def get_artifacts(self, prefix="decode_tile_"):
        operator_dir = Path(__file__).parent
        E, g = self.embed_dim, self.group_size
        base = f"{prefix}e{E}_d{self.head_dim}_g{g}_h{self.hidden_dim}"
        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{base}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="my_decode_tile",
            callback_args=[self.context.device_manager.device_type,
                           E, self.head_dim, g, self.attn_group,
                           self.hidden_dim, self.m_input, self.num_cols],
        )
        gemv_obj = KernelObjectArtifact.new(
            f"fused_dequant_gemv_v2_{E}k_g{g}.o",
            depends=[SourceArtifact.new(
                self.context.base_dir / "aie_kernels" / "aie2p"
                / "fused_dequant_gemv_v2.cc")],
            extra_flags=[f"-DDIM_K={E}", f"-DGROUP_SIZE={g}"],
        )
        oproj_obj = KernelObjectArtifact.new(
            f"fused_dequant_gemv_v2_oproj_{E}k_g{g}.o",
            depends=[SourceArtifact.new(
                self.context.base_dir / "aie_kernels" / "aie2p"
                / "fused_dequant_gemv_v2.cc")],
            extra_flags=[f"-DDIM_K={E}", f"-DGROUP_SIZE={g}",
                         "-Dfused_dequant_matvec_v2_bf16=oproj_matvec_v2_bf16"],
        )
        nc = self.num_cols
        Hc = self.hidden_dim // nc
        _relay_flags = [
            f"-DEMBED_DIM={E}", f"-DHIDDEN_DIM={self.hidden_dim}",
            f"-DGROUP_SIZE={g}", f"-DHEAD_DIM={self.head_dim}",
            f"-DNUM_HEADS=32", f"-DNUM_KV_HEADS=8", f"-DMAX_SEQ_LEN=2048",
            f"-DNUM_AIE_COLUMNS={nc}",
            f"-DM_OUTPUT_MAX={max(Hc, E)}", f"-DINTER_DIM_PER_COL={Hc}",
        ]
        relay_obj = KernelObjectArtifact.new(
            "layer_fused_relay.o",
            depends=[SourceArtifact.new(
                self.context.base_dir / "aie_kernels" / "aie2p" / "layer_fused.cc")],
            extra_flags=_relay_flags,
        )
        xclbin_artifact = XclbinArtifact.new(
            f"{base}.xclbin",
            depends=[mlir_artifact, gemv_obj, oproj_obj, relay_obj])
        insts_artifact = InstsBinArtifact.new(f"{base}.bin", depends=[mlir_artifact])
        return xclbin_artifact, insts_artifact

    def set_up_artifacts(self):
        self.xclbin_artifact, self.insts_artifact = self.get_artifacts()
        self.add_artifacts([self.xclbin_artifact, self.insts_artifact])

    def set_up_runtime(self):
        E, g, m, nc = self.embed_dim, self.group_size, self.m_input, self.num_cols
        Hc = self.hidden_dim // nc
        groups = E // g
        packed_tile = m * E // 2 + m * groups * 2
        q_rows = self.attn_group * self.head_dim
        o_slice = E // nc
        qkv_tiles = q_rows // m
        o_tiles = o_slice // m
        gu_tiles = Hc // m
        dn_tiles = E // m
        n_wtiles = qkv_tiles + o_tiles + 2 * gu_tiles + dn_tiles
        # 5-BO ABI: O out (bo0), W (bo1), X/attn/ffn activations (bo2/3/4)
        self.add_buffer("O", 3 * E, dtype=bfloat16)
        self.add_buffer("W", n_wtiles * packed_tile, dtype=np.uint8)
        self.add_buffer("X", E, dtype=bfloat16)
        self.add_buffer("A", E, dtype=bfloat16)
        self.add_buffer("F", E, dtype=bfloat16)
        self.add_kernel("decode_tile", self.xclbin_artifact,
                        self.xclbin_artifact.kernel_name, self.insts_artifact)
        self.add_to_runlist("decode_tile", "O", "W", "X", "A", "F")
