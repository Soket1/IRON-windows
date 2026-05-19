# IRON-windows NPU Optimization Plan

> Last updated: 2026-05-19

## Current Status: **FlowKV decode WORKING**

FlowKV decode attention is integrated and produces correct output on STX NPU2.
Debug flowkv.bat: Step 2 (no FlowKV) = "The capital of France is Paris." ✅
                    Step 3 (with FlowKV) = "The capital of France is Paris." ✅

Benchmark (run_flowkv_bench.bat, 32 tokens):

| Config | Prompt | Generation |
|--------|--------|------------|
| Baseline (no FlowKV) | 123.7 t/s | **6.0 t/s** |
| With FlowKV | 127.6 t/s | **4.8 t/s** (-20%) |

FlowKV is slower due to 8 separate dispatches per layer (96 per token).
Dispatch overhead (xrt::run setup + wait) dominates. Priority 4: batch KV heads.

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

### Priority 4: Batch KV heads per dispatch (4.4 → ~5.5-6.0 t/s)

**Problem:** FlowKV dispatches 8 times per layer (1 per KV head) = 96 dispatches per token.
Dispatch overhead (xrt::run setup + wait) dominates, making FlowKV SLOWER than baseline.

**Solution:** Batch 4 KV heads into a single xrt::run call.
- Dispatch count: 8 → 2 per layer (24 per token instead of 96)
- Kernel stays the same, called 4 times inside one dispatch
- Expected gain: 4.4 → ~5.5-6.0 t/s

**Implementation:**
1. Host: write K/V/Q data for 4 KV heads into combined buffers
2. IRON design: runtime sequence with 4× K/V/Q/O buffers
3. Kernel: loop over 4 KV heads within single dispatch
4. Scatter: write 4× output back to correct head positions

### Priority 5: Multi-column parallelism (→ ~8-10 t/s)

Distribute 8 KV heads across 8 NPU columns (STX NPU2 has 8 columns).
Each KV head on its own column, all in parallel.

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
| + Attention on NPU (FlowKV) | ✅ Working | ~8-10 t/s |
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

`flowkv_H4_KV1_d64_S256_C32_1col`

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
