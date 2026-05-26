#!/usr/bin/env python3
"""Step 2: try compiling a minimal FusedMLIROperator (just two ops fused).
This exercises the FullElfArtifact pipeline — the path that produces a
"naked" 5-BO kernel signature (FFLM-style).
"""
import sys
from pathlib import Path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "iron" / "applications" / "llama_3.2_1b"))

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("smoke-fused")

from iron.operators.rms_norm.op import RMSNorm
from iron.operators.elementwise_mul.op import ElementwiseMul
from iron.common.context import AIEContext
from iron.common.fusion import FusedMLIROperator

ctx = AIEContext()
ctx.build_dir.mkdir(parents=True, exist_ok=True)
log.info("build_dir: %s", ctx.build_dir)

# Minimal two-op runlist: RMSNorm -> ElementwiseMul (small sizes)
size = 2048
log.info("create child ops (size=%d)", size)
rms = RMSNorm(
    size=size, num_aie_columns=8, num_channels=1,
    tile_size=size // 8, weighted=True, context=ctx,
)
mul = ElementwiseMul(
    size=size, num_aie_columns=8, tile_size=size // 8, context=ctx,
)

log.info("build runlist (RMSNorm: input=x,W,output=x ; Mul: x,W2 -> y)")
runlist = [
    (rms, "x", "W_norm", "x"),
    (mul, "x", "W_mul", "y"),
]
log.info("FusedMLIROperator construction...")
fused = FusedMLIROperator(
    "smoke_fused",
    runlist,
    input_args=["x"],
    output_args=["y"],
    # Sizes are in BYTES (bf16=2 bytes/elem). RMSNorm weight = size of one
    # tile (size/num_aie_columns). Mul weight = full tensor.
    buffer_sizes={"W_norm": (size // 8) * 2, "W_mul": size * 2},
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
