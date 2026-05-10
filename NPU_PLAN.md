# IRON-windows NPU Optimization Plan

> Last updated: 2026-05-10, based on actual profiling data

## Current Status: **5.8 t/s** (Llama-3.2-1B-BF16, STX NPU2)

| Component | Status | NPU? | Per-dispatch | Per token (×12) |
|---|---|---|---|---|
| QKV projection | ✅ Working | NPU | 1.17 ms | 14.0 ms |
| SwiGLU (gate+up+silu+down) | ✅ Working | NPU | 3.62 ms | 43.4 ms |
| Output projection (decode batch) | ✅ Working | NPU | 1.06 ms | 12.7 ms |
| **NPU subtotal** | | | | **70 ms (41%)** |
| Attention (Q@K^T, softmax, scores@V) | ❌ CPU | CPU | — | ~60-70 ms |
| RMSNorm + MUL (gain) | ❌ CPU | CPU | — | ~15-20 ms |
| Residual ADD, RoPE, KV cache | ❌ CPU | CPU | — | ~10-15 ms |
| **CPU subtotal** | | | | **~102 ms (59%)** |
| **Total per token** | | | | **172 ms → 5.8 t/s** |

## Per-dispatch profiling (decode M=1, 115 samples)

| Operation | rl_build | rl_exec | rl_wait (NPU) | Total |
|---|---|---|---|---|
| SwiGLU | 54 µs | 51 µs | 3486 µs | 3620 µs |
| QKV | — | — | — | 1165 µs |
| decode_batch | — | — | — | 1055 µs |

## Dispatch pattern per layer

```
graph_compute n_nodes=1   → SOFT_MAX (CPU, separate call)
graph_compute n_nodes=32  → Main layer:
  RMS_NORM      → CPU (tile_size=32 bug)
  MUL (gain)    → CPU
  MUL_MAT ×3    → NPU via QKV dispatch (1.17 ms)
  ROPE ×2       → CPU
  SET_ROWS ×2   → CPU (KV cache writes)
  VIEW/RESHAPE/PERMUTE → skipped
  MUL_MAT       → NPU via decode_batch (output proj, 1.06 ms)
  ADD           → CPU (residual)
  RMS_NORM      → CPU
  MUL (gain)    → CPU
  GLU/SwiGLU    → NPU (3.62 ms)
  ADD           → CPU (residual)
```

## What DOESN'T work (lessons learned)

### ❌ Weighted RMSNorm patch (reverted, commit a584e70)
- `src[1]` is always NULL — weight applied via separate `GGML_OP_MUL`
- Dead code

### ❌ RMSNorm on NPU (tile_size=32 bug)
- Kernel normalizes by `tile_size=32`, not by full row (2048)
- Test passes by coincidence (test tensor dim = 32)
- Fix needs two-pass with reduction — not worth it (~1% of time)

### ❌ Transformer block fusion for decode (`tblock_match=0`)
- Early-reject: `ne[1] < 32` blocks decode (ne[1]=1)
- Gate: `seq_len >= 256` blocks decode (seq_len=1)
- Matcher requires `FLASH_ATTN_EXT` (decode uses expanded pattern)
- Even if fixed, impact is small: RMSNorm can't be included

### ❌ Decode batch efficiency
- Plans 4 batchable GEMVs but only captures 1 per flush
- CPU ops between GEMVs force flush after each one

## Priorities

### Priority 1: Attention → NPU (save ~60-70 ms/token)

**Expected gain: 5.8 → ~10-12 t/s**

Structure for decode attention on NPU:
- Q@K^T: GEMV — [n_heads, 1, head_dim] × [n_heads, head_dim, seq_len]
- Softmax: needs reduction over seq_len (like RMSNorm — same infra)
- scores@V: GEMV — [n_heads, 1, seq_len] × [n_heads, seq_len, head_dim]

**Blockers:**
- Softmax needs global reduction (same problem as RMSNorm)
- For short seq_len (≤64): single tile works, no reduction needed
- For long seq_len: two-pass needed

**Shortcut:** Start with seq_len ≤ 64 support (covers most decode scenarios).

### Priority 2: Merge QKV + decode_batch (save ~5 ms/token)

Currently: QKV (1.17 ms) + decode_batch (1.06 ms) = 2 dispatches per layer.
Target: single dispatch = ~1.5 ms → saves 0.7 ms/layer × 12 = ~8 ms/token.

### Priority 3: RMSNorm → NPU (save ~10-15 ms/token)

Only when:
- Reduction infra built for softmax (reusable for RMSNorm)
- All other ops on NPU

## Testing

### Pure CPU baseline (needed)
```bat
set XDNA_ENABLE_GEMV=0
set XDNA_ENABLE_SWIGLU=0
set XDNA_ENABLE_QKV=0
set XDNA_ENABLE_RMS_NORM=0
set XDNA_ENABLE_DECODE_BATCH=0
```

### Current NPU config
```bat
set XDNA_ENABLE_GEMV=1
set XDNA_ENABLE_SWIGLU=1
set XDNA_ENABLE_QKV=1
set XDNA_ENABLE_RMS_NORM=0
set XDNA_ENABLE_SWIGLU_PREFILL=0
set XDNA_ENABLE_DECODE_BATCH=1
set XDNA_ENABLE_TRANSFORMER_BLOCK=1
set XDNA_DEBUG=1
set GGML_XDNA_NUM_COLS=8
```

## Milestones

| Milestone | Current | Target |
|---|---|---|
| Steady-state | **5.8 t/s** | — |
| + Attention on NPU | — | ~10-12 t/s |
| + Merge QKV+batch | — | ~12-13 t/s |
| + RMSNorm on NPU | — | ~13-14 t/s |
| Full model on NPU | — | ~15+ t/s |

## Environment

```
XRT_SDK:     C:\Users\Kuhnya\Downloads\xrt_windows_sdk\xrt_sdk\xrt
PEANO:       C:\ProgramData\miniforge3\envs\ryzen-ai-1.7.1\Lib\site-packages\win64.o\tools\peano
MLIR_AIE:    C:\ProgramData\miniforge3\envs\ryzen-ai-1.7.1\Lib\site-packages\mlir_aie\bin
DRIVER:      C:\Windows\System32\DriverStore\FileRepository\kipudrv.inf_amd64_*
PYTHON:      C:\Python313\python.exe
MODEL:       models\llama-3.2-1b-instruct-BF16.gguf
REPO:        https://github.com/Soket1/IRON-windows (branch: devel)
```
