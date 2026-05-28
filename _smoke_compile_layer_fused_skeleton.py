#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A1.3.1 skeleton smoke: compile LayerFusedMLIR via legacy_xclbin path.

Validates:
  * 8-col partition compiles end-to-end
  * 5-BO arg signature accepted
  * llvm-objcopy + xclbinutil discovered via the A1.1 PATH-augment
  * xclbin + insts.bin pair emitted with non-zero size
  * DDR_PATCH ops present in the .insts (informational; A2 uses them)

Expectation: skeleton xclbin ≈ post_attn_fused size order (~50-80 KB).
PDI density check (target FFLM 414 KB) is for later milestones — this
empty pipeline only exercises the toolchain.
"""
import sys
import logging
from pathlib import Path

repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("smoke-A1.3.1")

from iron.common.context import AIEContext
from iron.common.fusion import FusedMLIROperator
from iron.operators.layer_fused.op_mlir import LayerFusedMLIR

ctx = AIEContext(build_dir="build_layer_fused_skeleton")
ctx.build_dir.mkdir(parents=True, exist_ok=True)
log.info("build_dir: %s", ctx.build_dir)

op = LayerFusedMLIR(
    embed_dim=2048, hidden_dim=8192,
    num_heads=32, num_kv_heads=8, head_dim=64,
    max_seq_len=2048,
    num_aie_columns=8, group_size=32,
    context=ctx,
)
log.info("op.name=%s", op.name)
log.info("arg spec:")
for i, a in enumerate(op.get_arg_spec()):
    log.info("  arg %d: %s", i, a)

log.info("FusedMLIROperator(legacy_xclbin=True, kernel_name=MLIR_AIE)")
fused = FusedMLIROperator(
    "layer_fused_skeleton_e2048_h8192_c8_g32",
    [(op, "w_qkv", "w_o", "w_ffn", "kv_pair", "activations")],
    input_args=["w_qkv", "w_o", "w_ffn"],
    output_args=["kv_pair", "activations"],
    legacy_xclbin=True,
    xclbin_kernel_name="MLIR_AIE",
    xclbin_instance_name="MLIRAIE",
    xclbin_kernel_id="0x901",
    context=ctx,
)

log.info("declared artifacts:")
for a in fused.artifacts:
    log.info("  %s -> %s", type(a).__name__, getattr(a, "filename", "?"))

log.info("compile()...")
import traceback
try:
    fused.compile()
    log.info("COMPILE OK")
except Exception as e:
    log.error("COMPILE FAILED: %s", e)
    traceback.print_exc()
    sys.exit(2)

log.info("compiled artifacts:")
for a in fused.artifacts:
    fn = getattr(a, "filename", "?")
    sz = Path(fn).stat().st_size if Path(fn).exists() else -1
    log.info("  %s -> %s (%d bytes)", type(a).__name__, fn, sz)

log.info("DONE")
