#!/usr/bin/env python3
import sys
import logging
from pathlib import Path

repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("smoke-cols8")

from iron.common.context import AIEContext
from iron.common.fusion import FusedMLIROperator
from iron.operators.layer_fused.op_mlir import LayerFusedMLIR

ctx = AIEContext(build_dir="build_layer_fused_skeleton_cols8")
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

fused = FusedMLIROperator(
    "layer_fused_skeleton_e2048_h8192_c8_g32_cols8",
    [(op, "w_qkv", "w_o", "w_ffn", "kv_pair", "activations")],
    input_args=["w_qkv", "w_o", "w_ffn"],
    output_args=["kv_pair", "activations"],
    legacy_xclbin=True,
    xclbin_kernel_name="MLIR_AIE",
    xclbin_instance_name="MLIRAIE",
    xclbin_kernel_id="0x901",
    context=ctx,
)

log.info("compile()...")
try:
    fused.compile()
    log.info("COMPILE OK")
except Exception as e:
    log.error("COMPILE FAILED: %s", e)
    sys.exit(2)
log.info("DONE")
