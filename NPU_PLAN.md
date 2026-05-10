# IRON-windows NPU Optimization Plan

> Last updated: 2026-05-10, based on actual profiling data from `test_short1.bat`

## Status Summary

| Component | Status | NPU? | Notes |
|---|---|---|---|
| QKV projection | ✅ Working | NPU | ~1.16 ms/dispatch |
| SwiGLU (gate+up+silu+down) | ✅ Working | NPU | ~3.59 ms/dispatch |
| Output projection (decode batch) | ✅ Working | NPU | ~1.7 ms/dispatch |
| Attention (Q@K^T, softmax, scores@V) | ❌ CPU | CPU | Separate graph_compute calls |
| RMSNorm | ❌ Broken | CPU | tile_size=32 bug |
| Residual ADD | ❌ CPU | CPU | |
| Transformer block fusion | ❌ Not matching | — | `tblock_match=0` for decode |

## Profiling Data (Llama-3.2-1B-BF16, decode M=1)

### Per-dispatch timing (average, 115 samples)

| Operation | rl_build | rl_exec | rl_wait (NPU compute) | Total |
|---|---|---|---|---|
| SwiGLU decode | 54 µs | 51 µs | **3486 µs** | **3595 µs** |
| QKV | — | — | — | **1157 µs** |
| decode_batch flush | — | — | — | **~1700 µs** |

### Per-token budget (12 layers)

| Component | Per layer | × 12 layers | % of 1000 ms |
|---|---|---|---|
| SwiGLU NPU | 3.59 ms | **43.1 ms** | 4.3% |
| QKV NPU | 1.16 ms | **13.9 ms** | 1.4% |
| decode_batch NPU | ~1.7 ms | **~20.4 ms** | 2.0% |
| **NPU subtotal** | | **~77 ms** | **7.7%** |
| CPU ops + overhead | | **~923 ms** | **92.3%** |
| **Total** | | **~1000 ms** | **→ 1.0 t/s** |

### Observed dispatch pattern per layer

```
graph_compute n_nodes=1   → SOFT_MAX (CPU, separate call)
graph_compute n_nodes=32  → Main layer:
  RMS_NORM    → CPU
  MUL (gain)  → CPU
  MUL_MAT ×3  → NPU via QKV dispatch (1.16 ms)
  ROPE ×2     → CPU
  SET_ROWS ×2 → CPU (KV cache writes)
  ... (VIEW/RESHAPE/PERMUTE skipped) ...
  MUL_MAT     → NPU via decode_batch (output proj, ~1.7 ms)
  ADD         → CPU (residual)
  RMS_NORM    → CPU
  MUL (gain)  → CPU
  GLU/SwiGLU  → NPU (3.59 ms)
  ADD         → CPU (residual)
```

### Token count discrepancy

- Requested: 32 tokens
- QKV M=1 dispatches: 112 ÷ 12 layers = **~9 decode tokens**
- QKV M=2 dispatches: 32 ÷ 12 = **~3 warmup tokens**
- SwiGLU M=2: 30, M=41: 15 (different code path)
- **Only ~12 tokens actually generated** — not 32. Possible causes:
  - Conversation mode (`-cnv`) generates fewer tokens
  - Process exits early
  - Compilation time included in 1.0 t/s measurement

## Critical unknowns (need profiling)

1. **Where is the 923 ms?** CPU ops can't explain 923 ms for a 1B model. Possible:
   - Compilation time included in measurement
   - CPU↔NPU context switch overhead per dispatch (driver, DMA sync)
   - CPU attention (Q@K^T, softmax, scores@V) is slower than expected
   - Graph dispatch overhead (pattern matching per graph_compute call)

2. **Actual per-token steady-state time?** Need timing WITHOUT compilation.

3. **CPU attention cost?** Need isolated measurement of Q@K^T + softmax + scores@V per layer.

## What DOESN'T work (lessons learned)

### ❌ Weighted RMSNorm patch (reverted)
- `src[1]` is always NULL in this ggml version
- Weight applied via separate `GGML_OP_MUL`, not inside `RMS_NORM`
- Dead code, reverted in commit `a584e70`

### ❌ RMSNorm on NPU (tile_size bug)
- Kernel normalizes by `tile_size=32`, not by full row (2048)
- Test passes because test tensor is (64, 32) — matches tile_size
- Real model: hidden_dim=2048 >> 32 → wrong normalization
- Fix needs two-pass with reduction — expensive, not worth it (~1% of time)

### ❌ Transformer block fusion for decode
- `tblock_match=0` because:
  - Early-reject: `ne[1] < 32` blocks decode tokens (ne[1]=1)
  - Gate: `seq_len >= 256` blocks decode (seq_len=1)
  - Attention matcher requires `FLASH_ATTN_EXT` (decode uses expanded pattern)
- **Even if fixed, impact is small**: RMSNorm can't be included (tile_size bug), so fusion only saves ~2 dispatches/layer (~3 ms), not 20

### ❌ Decode batch efficiency
- Plans 4 batchable GEMVs but only captures 1 per flush
- CPU ops between GEMVs force flush after each one
- Same root cause: no block-level fusion

## Revised priorities

### Priority 0: Profiling (MUST DO FIRST)

Before optimizing, need to understand where 923 ms goes.

**Action items:**
1. Run with `XDNA_DEBUG=1` and add wall-clock timestamps to `ggml_backend_xdna_graph_compute`
2. Measure per-graph_compute wall time
3. Measure CPU attention ops isolation (disable NPU, pure CPU baseline)
4. Measure steady-state per-token time (skip first 2 tokens as warmup)
5. Check if 1.0 t/s includes compilation time

**Env vars to add:**
```
# Add to test_short1.bat for profiling
set XDNA_PROFILE=1       # if exists, enables wall-clock per-dispatch timing
```

### Priority 1: Attention → NPU (if CPU attention is the bottleneck)

**Only if profiling confirms CPU attention > 500 ms/token.**

Expected structure for decode attention on NPU:
- Q@K^T: GEMV — [n_heads, 1, head_dim] × [n_heads, head_dim, seq_len]
- Softmax: needs reduction over seq_len (like RMSNorm — same infra needed)
- scores@V: GEMV — [n_heads, 1, seq_len] × [n_heads, seq_len, head_dim]

**Blockers:**
- Softmax needs global reduction (same problem as RMSNorm)
- For short seq_len (≤32): single tile works
- For long seq_len: two-pass needed

**Possible shortcut:** If seq_len is small (≤64 during generation), softmax can fit in one tile. Start with seq_len ≤ 64 support, skip long-sequence for now.

### Priority 2: Reduce dispatch overhead (if context switching is the bottleneck)

**If profiling shows high per-dispatch overhead (>5 ms each):**

Options:
- Combine QKV + output_proj into single dispatch (save 1 dispatch/layer)
- Combine SwiGLU dispatches (already done — fused gate+up+down)
- Use `xrt::runlist` for back-to-back dispatches (already done for decode_batch)

### Priority 3: RMSNorm → NPU (deferred)

Only when:
- Attention is on NPU (eliminates the 923 ms mystery)
- Reduction infra built for softmax (reusable for RMSNorm)
- All other ops on NPU — RMSNorm is last mile

## Testing protocol

### Quick baseline test (no NPU)
```bat
set XDNA_ENABLE_GEMV=0
set XDNA_ENABLE_SWIGLU=0
set XDNA_ENABLE_QKV=0
set XDNA_ENABLE_RMS_NORM=0
set XDNA_ENABLE_DECODE_BATCH=0
```
→ Pure CPU baseline. Compare with NPU enabled.

### Steady-state timing test
```bat
set XDNA_ENABLE_GEMV=1
set XDNA_ENABLE_SWIGLU=1
set XDNA_ENABLE_QKV=1
set XDNA_ENABLE_RMS_NORM=0
set XDNA_ENABLE_DECODE_BATCH=1
set XDNA_DEBUG=1
```
→ Look at last 5 tokens' dispatch times (skip warmup).

### Isolate NPU vs CPU time
Add wall-clock timestamps in `ggml_backend_xdna_graph_compute`:
- Before/after each `xdna_delegate_range` (CPU range)
- Before/after each QKV dispatch
- Before/after each SwiGLU dispatch
- Before/after each decode_batch flush

## Existing infrastructure

### Kernels (reuse as-is)
| Kernel | File | Status |
|---|---|---|
| `rms_norm_bf16_vector` | `aie_kernels/aie2/rms_norm.cc` | Works per-tile (broken for full row) |
| QKV | compiled xclbin | Working |
| SwiGLU decode | compiled xclbin | Working |
| SwiGLU prefill | compiled xclbin | Working |
| decode_batch GEMV | via SwiGLU infra | Working (but only 1 per flush) |

### Env vars
```
XDNA_ENABLE_GEMV=1            # GEMV on NPU
XDNA_ENABLE_SWIGLU=1          # SiLU on NPU
XDNA_ENABLE_QKV=1             # QKV on NPU
XDNA_ENABLE_RMS_NORM=0        # CPU (broken on NPU)
XDNA_ENABLE_SWIGLU_PREFILL=0  # prefill SiLU off
XDNA_ENABLE_DECODE_BATCH=1    # batch GEMVs
XDNA_ENABLE_TRANSFORMER_BLOCK=1  # tblock fusion (not matching for decode)
XDNA_DEBUG=1                  # debug logs
GGML_XDNA_NUM_COLS=8          # NPU2, 8 columns
```

## Milestones (revised)

| Milestone | Condition | Expected t/s |
|---|---|---|
| Baseline (all CPU) | Need measurement | ? |
| Current (QKV+SwiGLU on NPU) | Done | ~1.0 (includes compilation) |
| Steady-state measurement | Profiling | ~2-5? |
| Attention on NPU | If CPU attention is bottleneck | ~5-8 |
| Full model on NPU | All ops fused | ~10-15 |

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

---

## Profiling results (2026-05-10, test_timing.bat, cached kernels)

**Steady-state: 5.8 t/s** (1.0 t/s was including compilation)

### Per-dispatch timing (decode M=1)

| Operation | Count | Avg total | rl_wait (NPU compute) |
|---|---|---|---|
| QKV | 112 | 1165 µs | — |
| SwiGLU | 115 | 3620 µs | ~3500 µs |
| decode_batch | 112 | 1055 µs | — |

### Per-token budget (12 layers, 172 ms/token at 5.8 t/s)

| Component | Per layer | × 12 | Total | % |
|---|---|---|---|---|
| SwiGLU NPU | 3.62 ms | 12 | 43.4 ms | 25% |
| QKV NPU | 1.17 ms | 12 | 14.0 ms | 8% |
| decode_batch NPU | 1.06 ms | 12 | 12.7 ms | 7% |
| **NPU subtotal** | | | **70 ms** | **41%** |
| **CPU + overhead** | | | **102 ms** | **59%** |

### Key findings

1. **Steady-state is 5.8 t/s**, not 1.0 t/s (compilation was included in first measurement)
2. **NPU = 70 ms/token (41%)** — SwiGLU dominates (43 ms)
3. **CPU = 102 ms/token (59%)** — attention (Q@K^T, softmax, scores@V) + RMS_NORM + residual
4. **tblock_match=0** still holds — decode transformer block not matching
5. **decode_batch only captures 1 GEMV per flush** — CPU ops between GEMVs force flush

### Revised priorities

**Priority 1: Reduce CPU time (102 ms → target 30 ms)**
- Attention on NPU: Q@K^T + softmax + scores@V (~60-70 ms savings)
- RMSNorm on NPU: needs two-pass reduction (~10-15 ms savings)

**Priority 2: Reduce NPU time (70 ms → target 40 ms)**
- SwiGLU is 43 ms — already optimized (fused gate+up+down)
- QKV + decode_batch = 27 ms — could merge into single dispatch (~5 ms savings)

**Target: 172 ms → 70 ms → ~14 t/s**
