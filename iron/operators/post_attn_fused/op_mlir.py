# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MLIROperator-derived variant of PostAttnFused.

Same compute graph (design.py / post_attn_fused.cc), but exposed through
the MLIROperator interface so it can be wrapped in FusedMLIROperator and
compiled via FullElfArtifact (naked N-BO kernel signature, no opcode/
instr/ninstr prefix). Used by Phase β step 2d to emit FFLM-style ELFs.
"""

from dataclasses import dataclass, field
from typing import ClassVar, Dict
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16

import aie.utils as aie_utils
from iron.common.base import MLIROperator, AIERuntimeArgSpec
from iron.common.compilation import (
    PythonGeneratedMLIRArtifact,
    KernelObjectArtifact,
    SourceArtifact,
    DesignGenerator,
)
from iron.common.device_utils import get_kernel_dir


@dataclass
class PostAttnFusedMLIR(MLIROperator):
    """MLIROperator wrapper of post_attn_fused/design.py.

    Sequence args (matching design.py rt.sequence order):
      0: w_o          uint8  (packed INT4)
      1: w_gu         uint8  (packed INT4)
      2: w_d          uint8  (packed INT4)
      3: input_bundle bf16   [kqv|inpL|gain]   3*embed
      4: io_bundle    bf16   [scratch|silu_buf|ffn_out|inpff_save]  3*embed+hidden
    """

    embed_dim: int = 2048
    hidden_dim: int = 8192
    num_aie_columns: int = 4
    group_size: int = 32
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases,
        "embed_dim": "e",
        "hidden_dim": "h",
        "group_size": "g",
    }

    def __post_init__(self):
        assert self.embed_dim % self.num_aie_columns == 0
        assert self.hidden_dim % self.num_aie_columns == 0
        assert self.embed_dim % self.group_size == 0
        assert self.hidden_dim % self.group_size == 0
        MLIROperator.__init__(self, context=self.context)

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "my_post_attn_fused",
                (
                    aie_utils.get_current_device(),
                    self.num_aie_columns,
                    self.embed_dim,
                    self.hidden_dim,
                    self.group_size,
                ),
            ),
        )

    def get_kernel_artifacts(self):
        arch_dir = get_kernel_dir()
        e, h, g = self.embed_dim, self.hidden_dim, self.group_size
        kobj_name = f"post_attn_fused_{e}k_g{g}.o"
        return [
            KernelObjectArtifact(
                kobj_name,
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir / "aie_kernels" / arch_dir
                        / "post_attn_fused.cc"
                    )
                ],
                extra_flags=[
                    f"-DDIM_K={e}",
                    f"-DDIM_K_DOWN={h}",
                    f"-DGROUP_SIZE={g}",
                    f"-DM_OUTPUT_MAX=32",
                    f"-DEMBED_DIM={e}",
                ],
            ),
        ]

    def get_arg_spec(self):
        e, h, c, g = (
            self.embed_dim, self.hidden_dim, self.num_aie_columns, self.group_size,
        )
        # Weight buffers are byte-packed INT4 but declared as bf16-element
        # halved-shape on the L3 (runtime sequence) side — see design.py
        # comment. This is required for FusedMLIROperator's bf16-only
        # consolidator path. Byte count is unchanged.
        groups_o  = e // g
        packed_o  = 1 * e // 2 + 1 * groups_o * 2
        total_o   = c * (e // c) * packed_o

        packed_gu = 4 * e // 2 + 4 * groups_o * 2
        total_gu  = c * 2 * (h // c // 4) * packed_gu

        groups_d  = h // g
        packed_d  = 1 * h // 2 + 1 * groups_d * 2
        total_d   = c * (e // c) * packed_d

        assert total_o  % 2 == 0
        assert total_gu % 2 == 0
        assert total_d  % 2 == 0
        bf = np.dtype(bfloat16)
        return [
            AIERuntimeArgSpec("in",    (total_o  // 2,),  dtype=bf),  # w_o
            AIERuntimeArgSpec("in",    (total_gu // 2,),  dtype=bf),  # w_gu
            AIERuntimeArgSpec("in",    (total_d  // 2,),  dtype=bf),  # w_d
            AIERuntimeArgSpec("inout", (3 * e,),          dtype=bf),  # input_bundle
            AIERuntimeArgSpec("inout", (3 * e + h,),      dtype=bf),  # io_bundle
        ]
