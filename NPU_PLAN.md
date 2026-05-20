# IRON-windows NPU Optimization Plan

> Last updated: 2026-05-20

## Current Status: **FlowKV decode WORKING, batch mode operational**

FlowKV decode attention is integrated and produces correct output on STX NPU2.
Debug flowkv.bat: Step 2 (no FlowKV) = "The capital of France is Paris." ✅
                    Step 3 (with FlowKV) = "The capital of France is Paris." ✅

Benchmark (run_flowkv_bench.bat, 32 tokens):

| Config | Prompt | Generation |
|--------|--------|------------|
| Baseline (no FlowKV) | 123.7 t/s | **6.0 t/s** |
| FlowKV num_cols=1 | 127.6 t/s | **4.8 t/s** (-20%) |
| FlowKV num_cols=4 (batch) | 122.7 t/s | **5.3 t/s** (-12%) |

Batch mode (num_cols=4) reduces dispatch count from 96 to 24 per token.
Root cause of batch failures: `num_heads` parameter was `q_heads_per_kv` (4) instead of `q_heads_per_kv * num_cols` (16), causing `group_size=1` in compiled xclbin.

## Two bugs found and fixed (2026-05-19)

### Bug 1 (v10): K DMA routing
IRON compiler reads K FIFO from arg0 (bo_k), not arg1 (bo_v) as assumed.
Host wrote K only to bo_v[0] → tile saw zeros.
Fix: mirror K data into bo_k via memcpy after writing to bo_v.

### Bug 2 (v11): CONT node skip
`continue` skipped CONT node after FlowKV dispatch. MUL_MAT read from
CONT output buffer (empty), not from kqv_out->data (CONT input).
Fix: removed `continue`, let CONT execute in CPU range.

Both bugs masked each other. After both fixes, model produces correct output.

## Component Status

| Component | Status | NPU? | Per-dispatch | Per token (×12) |
|---|---|---|---|---|
| QKV projection | ✅ Working | NPU | 1.17 ms | 14.0 ms |
| SwiGLU (gate+up+silu+down) | ✅ Working | NPU | 3.62 ms | 43.4 ms |
| Output projection (decode batch) | ✅ Working | NPU | 1.06 ms | 12.7 ms |
| **Attention (FlowKV decode)** | ✅ Working | NPU | ~2 ms × 8 | ~16 ms |
| RMSNorm + MUL (gain) | ❌ CPU | CPU | — | ~15-20 ms |
| Residual ADD, RoPE, KV cache | ❌ CPU | CPU | — | ~10-15 ms |

## Dispatch pattern per layer

```
graph_compute n_nodes=1   → SOFT_MAX (skipped when FlowKV enabled)
graph_compute n_nodes=N   → Main layer:
  RMS_NORM      → CPU
  MUL (gain)    → CPU
  MUL_MAT ×3    → NPU via QKV dispatch (1.17 ms)
  ROPE ×2       → CPU
  KV cache ops  → CPU
  CONT + attn   → FlowKV POC dispatch (8 × ~2 ms = ~16 ms)
  ADD           → CPU (residual)
  RMS_NORM      → CPU
  MUL (gain)    → CPU
  GLU/SwiGLU    → NPU (3.62 ms)
  ADD           → CPU (residual)
```

## Per-dispatch profiling (decode M=1)

| Operation | Total |
|---|---|
| SwiGLU | 3620 µs |
| QKV | 1165 µs |
| decode_batch | 1055 µs |
| FlowKV (per KV head) | ~2000 µs × 8 dispatches |

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
- Even if fixed, impact is small: RMSNorm can't be included

### ❌ Decode batch efficiency
- Plans 4 batchable GEMVs but only captures 1 per flush
- CPU ops between GEMVs force flush after each one

### ✅ FlowKV batch num_cols=1→4 (resolved 2026-05-20)
- Attempt 1: Wrong V layout (interleaved). Garbage.
- Attempt 2: Correct contiguous V layout. Still garbage.
- Attempt 3: Persistent xrt::run. Worse (3.9 vs 4.8 t/s).
- Attempt 4: Aligned stride (PDF fix). Garbage — didn't help for S256 (already 64-byte aligned).
- **Root cause**: `num_heads` param to `get_or_load_flowkv_kernel()` was `q_heads_per_kv` (4) instead of `q_heads_per_kv * num_cols` (16). Kernel compiled with `group_size=1`, host prepared data for `group_size=4`.
- **Fix**: Pass `q_heads_per_kv * num_cols` as `num_heads`. Cache key changes to `flowkv_H16_KV4_...`.
- Result: **5.3 t/s** (from 4.8 t/s with num_cols=1).

### ❌ FlowKV host-side optimizations
- Remove memset of bo_v: no effect (4.8 t/s). Kernel only reads actual_seq_len.
- Persistent xrt::run (num_cols=1): worse (3.9 t/s). XRT doesn't optimize run reuse.
- Persistent xrt::run (num_cols=4): no improvement (5.0 vs 5.3 t/s). Overhead is elsewhere.
- num_cols=8: IRON SequentialPlacer fails — XRT runtime limits context to 4 columns (other 4 reserved for Windows Studio Effects).

## Priorities

### Priority 1: Attention → NPU ✅ DONE

FlowKV decode integrated and working (2026-05-19).
- Per-KV-head dispatch (Option 3)
- 8 dispatches for Llama 3.2 1B (group_size=4, 8 KV heads)
- Gate: `XDNA_ENABLE_FLOWKV_DECODE=1`
- Identity RoPE angles (Q already rotated by graph)
- Two bugs fixed: K DMA routing + CONT node skip

### Priority 2: Merge QKV + decode_batch (save ~5 ms/token)

Currently: QKV (1.17 ms) + decode_batch (1.06 ms) = 2 dispatches per layer.
Target: single dispatch = ~1.5 ms → saves 0.7 ms/layer × 12 = ~8 ms/token.

### Priority 3: RMSNorm → NPU (save ~10-15 ms/token)

Only when:
- Reduction infra built for softmax (reusable for RMSNorm)
- All other ops on NPU

### Priority 4: Batch KV heads per dispatch ✅ DONE (2026-05-20)

**Problem:** FlowKV dispatches 8 times per layer (1 per KV head) = 96 dispatches per token.
Dispatch overhead (xrt::run.wait hardware time) dominates.

**Solution:** Batch 4 KV heads into a single xrt::run call.
- Dispatch count: 8 → 2 per layer (24 per token instead of 96)
- IRON design uses 4 columns, runtime sequence fills 4 KV heads in parallel
- Cache key: `flowkv_H16_KV4_d64_S256_C32_4col`

**Root cause of previous failures:**
`num_heads` param passed to `get_or_load_flowkv_kernel()` was `q_heads_per_kv` (4) instead of `q_heads_per_kv * num_cols` (16). This compiled xclbin with `group_size=1` while host prepared data for `group_size=4`, causing DMA misalignment and garbage output.

**Aligned stride (PDF fix):**
For S256, `raw_head_bytes = 256*64*2 = 32768` — already 64-byte aligned. The PDF fix only matters when `raw_head_bytes % 64 != 0`. Still applied for correctness with arbitrary seq_len.

### Priority 5: Multi-column parallelism (BLOCKED)

XDNA 2 physically has 8 compute columns, but XRT runtime allocates only 4 per user context.
Remaining 4 are reserved for OS background AI tasks (Windows Studio Effects, noise cancellation).
SequentialPlacer fails with "Failed to find a tile matching column 2: tried until column 8".
**num_cols=4 is the hard limit** for user XRT contexts on client Ryzen AI processors.

### Priority 6: Fuse FlowKV + output projection (→ +1-2 t/s)

Combine FlowKV output → attn_output.weight MUL_MAT in single dispatch.
Remove CONT + MUL_MAT from CPU path.

## Testing

### FlowKV verification

```bat
debug_flowkv.bat
```

Runs Step 2 (baseline without FlowKV) and Step 3 (with FlowKV).
Both should output "The capital of France is Paris."

### Diagnostic tools

- `XDNA_FLOWKV_REAL_PROBE=1` — v10: data transport probe
- `XDNA_FLOWKV_MATH_DIAG=1` — v11: CPU reference comparison

### Current NPU config

```bat
set XDNA_ENABLE_GEMV=1
set XDNA_ENABLE_SWIGLU=1
set XDNA_ENABLE_QKV=1
set XDNA_ENABLE_RMS_NORM=0
set XDNA_ENABLE_SWIGLU_PREFILL=0
set XDNA_ENABLE_DECODE_BATCH=1
set XDNA_ENABLE_TRANSFORMER_BLOCK=1
set XDNA_ENABLE_FLOWKV_DECODE=1
set GGML_XDNA_NUM_COLS=8
```

## Milestones

| Milestone | Current | Target |
|---|---|---|
| Steady-state (no FlowKV) | **5.8 t/s** | — |
| + Attention on NPU (FlowKV num_cols=1) | **4.8 t/s** | — |
| + Batch 4 KV heads (num_cols=4) | **5.3 t/s** | — |
| + Multi-column parallelism (8 cols) | BLOCKED (XRT 4 col limit) | ~8-10 t/s |
| + Merge QKV+batch | — | ~12-13 t/s |
| + RMSNorm on NPU | — | ~13-14 t/s |
| Full model on NPU | — | ~15+ t/s |

## Architecture

### FlowKV Decode Attention

2-tile streaming pipeline per KV head group:
- **Score tile (CT0)**: Q*K^T/sqrt(d) + online softmax → packed [F_c | C_c | l]
- **Value tile (CT1)**: weighted V accumulation + normalize → output

### DDR buffer layout

- **KV cache**: `[K_all | V_all]` combined in bo_v, K mirrored to bo_k
- **Q**: `[Q_group | angles | actual_seq_len | probe_magic]` per KV group
- **Output**: `(num_heads * head_dim)` bf16

### Cache key

`flowkv_H16_KV4_d64_S256_C32_4col` (batch mode, num_heads=16, num_kv_heads=4, num_cols=4)

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
