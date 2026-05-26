#!/usr/bin/env python3
"""Smoke-test that llama_npu.py's compile pipeline (RMSNorm only, prefill)
starts without weights. Goal: see if FusedMLIROperator / FullElfArtifact
can actually invoke aiecc on our Windows+AIE2P setup.
"""
import sys
from pathlib import Path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "iron" / "applications" / "llama_3.2_1b"))

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("smoke")

class MockConfig:
    vocab_size  = 128256
    emb_dim     = 2048
    n_layers    = 16
    n_heads     = 32
    n_kv_groups = 8
    head_dim    = 64
    hidden_dim  = 8192
    rope_base   = 500000.0
    context_length = 131072

prompt_len = 2048

log.info("step 1: import IRON RMSNorm")
from iron.operators.rms_norm.op import RMSNorm
from iron.common.context import AIEContext

log.info("step 2: build context")
ctx = AIEContext()
ctx.build_dir.mkdir(parents=True, exist_ok=True)
log.info("  build_dir: %s", ctx.build_dir)

log.info("step 3: instantiate RMSNorm (standalone, not yet fused)")
rms = RMSNorm(
    size=prompt_len * MockConfig.emb_dim,
    num_aie_columns=8,
    num_channels=1,
    tile_size=MockConfig.emb_dim,
    weighted=True,
    context=ctx,
)
log.info("  RMSNorm created: %s", rms)

log.info("step 4: compile RMSNorm (this invokes aiecc)")
import traceback
try:
    rms.compile()
    log.info("  COMPILE OK")
except Exception as e:
    log.error("  COMPILE FAILED: %s", e)
    traceback.print_exc()
    sys.exit(2)

log.info("step 5: artifacts")
for a in rms.artifacts:
    log.info("  %s -> %s", type(a).__name__, getattr(a, 'filename', '?'))

log.info("DONE")
