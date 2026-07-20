#!/usr/bin/env python3
"""Smoke compile: layer_dense via FusedMLIROperator → xclbin.

Tests the FULL pipeline: resolve_program → AIECC → xclbin packaging.
"""
import sys
import os
import logging
from pathlib import Path

repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

# DLL paths for XRT
driver_dir = r"C:\Windows\System32\DriverStore\FileRepository\kipudrv.inf_amd64_1a1aa059597c4810"
if os.path.exists(driver_dir):
    os.add_dll_directory(driver_dir)
xrt_sdk_dlls = r"C:\Users\Kuhnya\Downloads\xrt_windows_sdk\xrt_sdk\xrt\bin\condautils"
if os.path.exists(xrt_sdk_dlls):
    os.add_dll_directory(xrt_sdk_dlls)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("smoke-layer-dense")

from iron.common.context import AIEContext
from iron.common.fusion import FusedMLIROperator
from iron.operators.layer_dense.op_mlir import LayerFusedMLIR

ctx = AIEContext(build_dir="build_layer_dense_smoke")
ctx.build_dir.mkdir(parents=True, exist_ok=True)
log.info("build_dir: %s", ctx.build_dir)

op = LayerFusedMLIR(
    embed_dim=2048, hidden_dim=8192,
    num_heads=32, num_kv_heads=8, head_dim=64,
    max_seq_len=2048,
    num_aie_columns=4, group_size=32,
    context=ctx,
)
log.info("op.name=%s", op.name)

log.info("FusedMLIROperator(legacy_xclbin=True, kernel_name=MLIR_AIE)")
fused = FusedMLIROperator(
    "layer_dense_e2048_h8192_c8_g32",
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
