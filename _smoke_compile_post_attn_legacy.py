#!/usr/bin/env python3
"""Phase A1.1 smoke-test: FusedMLIROperator(legacy_xclbin=True) on
PostAttnFusedMLIR. Validates that fusion.py patch produces a working
xclbin + insts.bin pair via AieccXclbinInstsCompilationRule.

Expectation: aiecc.exe accepts a FusedMLIRSource artifact in the same
way it accepts a single-op MLIR source, and emits combined.xclbin +
post_attn_fused_legacy_*.insts.
"""
import sys
from pathlib import Path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("smoke-legacy")

from iron.common.context import AIEContext
from iron.common.fusion import FusedMLIROperator
from iron.operators.post_attn_fused.op_mlir import PostAttnFusedMLIR

ctx = AIEContext(build_dir="build_legacy_xclbin")
ctx.build_dir.mkdir(parents=True, exist_ok=True)
log.info("build_dir: %s", ctx.build_dir)

op = PostAttnFusedMLIR(
    embed_dim=2048, hidden_dim=8192,
    num_aie_columns=4, group_size=32,
    context=ctx,
)
log.info("op.name=%s", op.name)

log.info("FusedMLIROperator(legacy_xclbin=True, kernel_name=MLIR_AIE)")
fused = FusedMLIROperator(
    "post_attn_fused_legacy_e2048_h8192_c4_g32",
    [(op, "w_o", "w_gu", "w_d", "input_bundle", "io_bundle")],
    input_args=["w_o", "w_gu", "w_d", "input_bundle"],
    output_args=["io_bundle"],
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
