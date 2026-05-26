#!/usr/bin/env python3
"""Step 2d-β-1: try compiling PostAttnFusedMLIR (the MLIROperator variant)
wrapped as a 1-entry FusedMLIROperator. Goal: get a FullElfArtifact ELF
for our actual production op (cols=4, e=2048, h=8192, g=32).
"""
import sys
from pathlib import Path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("smoke-postattn")

from iron.common.context import AIEContext
from iron.common.fusion import FusedMLIROperator
from iron.operators.post_attn_fused.op_mlir import PostAttnFusedMLIR

ctx = AIEContext()
ctx.build_dir.mkdir(parents=True, exist_ok=True)
log.info("build_dir: %s", ctx.build_dir)

log.info("instantiate PostAttnFusedMLIR (e=2048 h=8192 cols=4 g=32)")
op = PostAttnFusedMLIR(
    embed_dim=2048, hidden_dim=8192,
    num_aie_columns=4, group_size=32,
    context=ctx,
)
log.info("  name=%s", op.name)
log.info("  arg_spec: %s", op.get_arg_spec())

log.info("FusedMLIROperator(...) with 1 entry (no real fusion)")
fused = FusedMLIROperator(
    "post_attn_fused_fullelf_e2048_h8192_c4_g32",
    [(op, "w_o", "w_gu", "w_d", "input_bundle", "io_bundle")],
    input_args=["w_o", "w_gu", "w_d", "input_bundle"],
    output_args=["io_bundle"],
    context=ctx,
)

log.info("compile()...")
import traceback
try:
    fused.compile()
    log.info("COMPILE OK")
except Exception as e:
    log.error("COMPILE FAILED: %s", e)
    traceback.print_exc()
    sys.exit(2)

log.info("artifacts:")
for a in fused.artifacts:
    log.info("  %s -> %s", type(a).__name__, getattr(a, 'filename', '?'))

log.info("DONE")
