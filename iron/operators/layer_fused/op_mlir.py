# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LayerFused — monolithic transformer-layer MLIROperator.

Fuses one full transformer layer (pre-RMS + Q/K/V + RoPE + KV slot +
GQA attention + O_proj + residual ADD + post-RMS + MUL + SwiGLU
gate/up/down + final ADD) into a single aie.device, single cores
device, single xclbin via the legacy_xclbin path.

Exists alongside post_attn_fused (which only covers the post-attn
half of a layer); replaces it once layer_fused reaches parity.

Sequence args (5 BOs, fits XRT 8-arg cap = 3 internal + 5 BO):
  0: w_qkv         uint8  (packed INT4) — Q/K/V weights, all heads
  1: w_o           uint8  (packed INT4) — O_proj weights
  2: w_ffn         uint8  (packed INT4) — gate + up + down weights
  3: kv_pair       bf16              — K_cache | V_cache (per-layer,
                                          patched by host per dispatch)
  4: activations   bf16              — x | rope_lut | scratch | etc.

All weight buffers are declared on the L3/runtime side as bf16-element
halved-shape — required for FusedMLIROperator's bf16-only consolidator.
Byte count is unchanged. See post_attn_fused/op_mlir.py:104-107 for the
established precedent.

Initial skeleton (A1.3.1): minimal passthrough rt.sequence so the
operator compiles end-to-end via legacy_xclbin and the BO connectivity
is validated. Compute stages are added in A1.3.2-A1.3.4.
"""

from dataclasses import dataclass, field
from typing import ClassVar, Dict
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
class LayerFusedMLIR(MLIROperator):
    """MLIROperator wrapping layer_fused/design.py.

    Llama-3.2-1B defaults are baked in; parameterise per model later.
    """

    embed_dim: int = 2048
    hidden_dim: int = 8192
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 64
    max_seq_len: int = 2048
    num_aie_columns: int = 8
    group_size: int = 32
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases,
        "embed_dim": "e",
        "hidden_dim": "h",
        "num_heads": "nh",
        "num_kv_heads": "nkv",
        "head_dim": "hd",
        "max_seq_len": "mx",
        "group_size": "g",
    }

    def __post_init__(self):
        e, h, c, g = (
            self.embed_dim, self.hidden_dim,
            self.num_aie_columns, self.group_size,
        )
        assert e % c == 0, f"embed_dim {e} must divide num_aie_columns {c}"
        assert h % c == 0, f"hidden_dim {h} must divide num_aie_columns {c}"
        assert e % g == 0, f"embed_dim {e} must divide group_size {g}"
        assert h % g == 0, f"hidden_dim {h} must divide group_size {g}"
        assert self.num_heads * self.head_dim == e, (
            f"num_heads*head_dim {self.num_heads*self.head_dim} != embed_dim {e}"
        )
        assert self.num_heads % self.num_kv_heads == 0, (
            "num_heads must be a multiple of num_kv_heads"
        )
        MLIROperator.__init__(self, context=self.context)

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "my_layer_fused",
                (
                    aie_utils.get_current_device(),
                    self.num_aie_columns,
                    self.embed_dim,
                    self.hidden_dim,
                    self.num_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    self.max_seq_len,
                    self.group_size,
                ),
            ),
        )

    def get_kernel_artifacts(self):
        arch_dir = get_kernel_dir()
        e, h, g = self.embed_dim, self.hidden_dim, self.group_size
        kobj_name = f"layer_fused_{e}_{h}_g{g}.o"
        return [
            KernelObjectArtifact(
                kobj_name,
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir / "aie_kernels" / arch_dir
                        / "layer_fused.cc"
                    )
                ],
                extra_flags=[
                    f"-DEMBED_DIM={e}",
                    f"-DHIDDEN_DIM={h}",
                    f"-DGROUP_SIZE={g}",
                    f"-DHEAD_DIM={self.head_dim}",
                    f"-DNUM_HEADS={self.num_heads}",
                    f"-DNUM_KV_HEADS={self.num_kv_heads}",
                    f"-DMAX_SEQ_LEN={self.max_seq_len}",
                    f"-DNUM_AIE_COLUMNS={self.num_aie_columns}",
                    f"-DM_OUTPUT_MAX={self.hidden_dim // self.num_aie_columns}",
                ],
            ),
        ]

    def _bundle_byte_sizes(self):
        """Compute byte sizes for the 5 BO bundles. Single source of
        truth for both the runtime arg spec (here) and design.py.

        Layout (bf16-element-counted byte regions, all even-sized):
          BO0 (in)    = W_norm1 [E] | W_q [E×E INT4] | W_k [E×kvE INT4]
                                    | W_v [E×kvE INT4]
          BO1 (in)    = W_o    [E×E INT4]
          BO2 (in)    = W_norm2 [E] | W_gate [E×H INT4] | W_up [E×H INT4]
                                    | W_down [H×E INT4]
          BO3 (inout) = K_cache | V_cache  (per-layer, DDR_PATCH'd)
          BO4 (inout) = activations (x, scratch, ..., outL)
        """
        e, h, g = self.embed_dim, self.hidden_dim, self.group_size
        c = self.num_aie_columns
        nkv, hd = self.num_kv_heads, self.head_dim
        mx = self.max_seq_len
        kv_e = nkv * hd  # 512 for Llama-3.2-1B

        groups_e = e // g
        groups_h = h // g

        # Per-weight packed-tile byte budgets (mirror post_attn_fused
        # design.py:65-86 layout: m_input rows of E packed INT4 nibbles
        # plus m_input scales per group_size group, two bytes each).
        # m_input_qkv=2 because the AIE2P shim DMA requires transfer
        # lengths that are multiples of 4 bytes — m_input=1 gives a
        # 2-byte output drain (1 bf16) which fails resource allocation.
        m_input_qkv = 2
        packed_q = m_input_qkv * e // 2 + m_input_qkv * groups_e * 2
        # packed_q is bytes for ONE tile = m_input_qkv rows. Per-col total =
        # (rows_per_col / m_input_qkv) tiles × packed_q. Without the
        # `// m_input_qkv` divisor the Q/K/V regions are sized 2× and the
        # K/V TAPs read from past the packer-written bytes.
        total_q = c * (e // c // m_input_qkv) * packed_q
        # K/V output dims are kv_e (= n_kv * head_dim), not e
        total_kv_one = c * (kv_e // c // m_input_qkv) * packed_q  # K alone (or V alone)

        m_input_o = 1
        packed_o = m_input_o * e // 2 + m_input_o * groups_e * 2
        total_o = c * (e // c) * packed_o

        # SwiGLU gate+up packed together (each row m_input_gu)
        m_input_gu = 4
        packed_gu = m_input_gu * e // 2 + m_input_gu * groups_e * 2
        # 2 * for gate AND up
        total_gu = c * 2 * (h // c // m_input_gu) * packed_gu

        m_input_d = 1
        packed_d = m_input_d * h // 2 + m_input_d * groups_h * 2
        total_d = c * (e // c) * packed_d

        # Norm gain weights (bf16)
        norm_bytes = e * 2

        # BO0: W_norm1 (bf16) + Q + K + V weights
        bo0_bytes = norm_bytes + total_q + 2 * total_kv_one
        # BO1: O_proj weights
        bo1_bytes = total_o
        # BO2: W_norm2 (bf16) + gate+up+down weights
        bo2_bytes = norm_bytes + total_gu + total_d
        # BO3: K_cache | V_cache, both bf16
        kv_one_bytes = nkv * mx * hd * 2
        bo3_bytes = 2 * kv_one_bytes
        # BO4: activations bundle (bf16)
        # layout: x [E] | rope_lut [2*MAX*head_dim] | scratch [E] |
        #         q_rot [E] | k_rot [kv_e] | v [kv_e] |
        #         attn_out [E] | o_proj_out [E] | inpFF [E] |
        #         normed [E] | ffn_in [E] | silu_out [H] | ffn_out [E] |
        #         outL [E] | ffn_out_partials [cols*E]
        bo4_elems = (
            e                            # x
            + 2 * mx * hd                # rope_lut (sin, cos per pos × head_dim)
            + e                          # scratch
            + e                          # q_rot
            + kv_e                       # k_rot
            + kv_e                       # v
            + e                          # attn_out
            + e                          # o_proj_out
            + e                          # inpFF
            + e                          # normed
            + e                          # ffn_in
            + h                          # silu_out
            + e                          # ffn_out
            + e                          # outL
            + c * e                      # ffn_out_partials (decomp-B: cols partial-E vectors)
        )
        bo4_bytes = bo4_elems * 2
        return bo0_bytes, bo1_bytes, bo2_bytes, bo3_bytes, bo4_bytes, bo4_elems

    def get_arg_spec(self):
        bo0_b, bo1_b, bo2_b, bo3_b, bo4_b, bo4_elems = self._bundle_byte_sizes()
        # All weight bundles declared as bf16-element halved-shape so
        # FusedMLIROperator's bf16 consolidator accepts them. Byte count
        # is what XRT cares about.
        for nm, b in (("bo0", bo0_b), ("bo1", bo1_b),
                      ("bo2", bo2_b), ("bo3", bo3_b)):
            assert b % 2 == 0, f"{nm} byte size {b} must be even"
        bf = np.dtype(bfloat16)
        return [
            AIERuntimeArgSpec("in",    (bo0_b // 2,), dtype=bf),  # w_qkv
            AIERuntimeArgSpec("in",    (bo1_b // 2,), dtype=bf),  # w_o
            AIERuntimeArgSpec("in",    (bo2_b // 2,), dtype=bf),  # w_ffn
            AIERuntimeArgSpec("inout", (bo3_b // 2,), dtype=bf),  # kv_pair
            AIERuntimeArgSpec("inout", (bo4_elems,),  dtype=bf),  # activations
        ]
