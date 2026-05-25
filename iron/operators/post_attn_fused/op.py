# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operator class for the fused post-attention layer (Phase B).

Single xclbin: O_proj GEMV → ADD(attn_res) → RMSNorm → MUL(gain) → SwiGLU.
"""

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


class PostAttnFused(AIEOperatorBase):
    """Fused post-attention layer: O_proj + ADD + RMSNorm + MUL + SwiGLU.

    One xclbin, one xrt::execute per layer (replaces separate O_proj + SwiGLU).
    """

    def __init__(self, embed_dim=2048, hidden_dim=8192,
                 num_aie_columns=8, group_size=32, context=None):
        cols = num_aie_columns
        assert embed_dim % cols == 0
        assert hidden_dim % cols == 0
        assert embed_dim % group_size == 0
        assert hidden_dim % group_size == 0

        self.embed_dim      = embed_dim
        self.hidden_dim     = hidden_dim
        self.num_aie_columns = cols
        self.group_size     = group_size

        self.xclbin_artifact = None
        self.insts_artifact  = None

        AIEOperatorBase.__init__(self, context=context)

    def get_artifacts(self):
        operator_dir = Path(__file__).parent
        e, h, c, g = self.embed_dim, self.hidden_dim, self.num_aie_columns, self.group_size
        # v7 = v6 PLUS monolithic gate_up_worker (one worker per col
        # instead of the gate/up/silu_mul split). Targets cols=4 where
        # 2 * tiles_per_col_gu = 512 fits the shim BD outer cap.
        name = f"post_attn_fused_v7_e{e}_h{h}_c{c}_g{g}"

        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{name}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="my_post_attn_fused",
            callback_args=[
                self.context.device_manager.device_type,
                c, e, h, g,
            ],
        )

        kobj_name = f"post_attn_fused_{e}k_g{g}.o"
        xclbin_artifact = XclbinArtifact.new(
            f"{name}.xclbin",
            depends=[
                mlir_artifact,
                KernelObjectArtifact.new(
                    kobj_name,
                    depends=[
                        SourceArtifact.new(
                            self.context.base_dir / "aie_kernels" / "aie2p"
                            / "post_attn_fused.cc"
                        )
                    ],
                    extra_flags=[
                        f"-DDIM_K={e}",
                        f"-DDIM_K_DOWN={h}",
                        f"-DGROUP_SIZE={g}",
                        # m_input_gu=8 in design.py; left/right L1 bufs only
                        # need that many slots, plus a little headroom.
                        f"-DM_OUTPUT_MAX=32",
                        f"-DEMBED_DIM={e}",
                    ],
                ),
            ],
        )

        insts_artifact = InstsBinArtifact.new(f"{name}.bin", depends=[mlir_artifact])
        return xclbin_artifact, insts_artifact

    def set_up_artifacts(self):
        xclbin_artifact, insts_artifact = self.get_artifacts()
        self.xclbin_artifact = xclbin_artifact
        self.insts_artifact  = insts_artifact
        self.add_artifacts([xclbin_artifact, insts_artifact])

    def set_up_runtime(self):
        e, h, c, g = self.embed_dim, self.hidden_dim, self.num_aie_columns, self.group_size
        groups_o  = e // g
        packed_o  = 1 * e // 2 + 1 * groups_o * 2
        total_o   = c * (e // c) * packed_o

        packed_gu = 4 * e // 2 + 4 * groups_o * 2
        total_gu  = c * 2 * (h // c // 4) * packed_gu

        groups_d  = h // g
        packed_d  = 1 * h // 2 + 1 * groups_d * 2
        total_d   = c * (e // c) * packed_d

        self.add_buffer("w_o_proj",   total_o,  dtype=np.uint8)
        self.add_buffer("kqv_out",    e,         dtype=bfloat16)
        self.add_buffer("inpL",       e,         dtype=bfloat16)
        self.add_buffer("gain_weight",e,         dtype=bfloat16)
        self.add_buffer("w_gate_up",  total_gu,  dtype=np.uint8)
        self.add_buffer("w_down",     total_d,   dtype=np.uint8)
        self.add_buffer("scratch",    e,         dtype=bfloat16)
        self.add_buffer("ffn_out",    e,         dtype=bfloat16)
        self.add_kernel("post_attn_fused", self.xclbin_artifact, self.insts_artifact)
