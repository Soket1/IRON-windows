# IRON-windows NPU Optimization Plan

## Status Summary

| Component | Status | NPU? | Notes |
|---|---|---|---|
| RMSNorm | ❌ Broken | CPU (XDNA_ENABLE_RMS_NORM=0) | tile_size=32 normalizes per-chunk, not per-row |
| Attention | 🔲 Not started | CPU | Priority #1 — 85% of decode latency |
| FFN (gate/up/down GEMV) | 🔲 Not started | CPU | Priority #2 |
| RoPE | 🔲 Not started | CPU | Small cost, low priority |
| Softmax | 🔲 Not started | CPU | Needs reduction (like RMSNorm) |
| QKV projection | 🔲 Not started | CPU | GEMV — similar to FFN |

## Current Performance

- **Decode speed**: ~2.5 t/s (llama-3.2-1b-bf16, NPU2/STX)
- **Target**: 5-12 t/s
- **Bottleneck**: Attention and FFN on CPU

---

## Phase 1: Attention → NPU (Priority #1)

**Expected gain**: 3-5× decode speed (2.5 → 8-12 t/s)

### What is attention?

For each decode step:
```
Q = x @ Wq          # [1, 2048] @ [2048, 2048] → [1, 2048]
K = x @ Wk          # same
V = x @ Wv          # same

# Multi-head (32 heads × 64 dim each)
scores = Q @ K^T / sqrt(64)    # [32, 1, seq_len]
attn = softmax(scores)          # [32, 1, seq_len]
out = attn @ V                  # [32, 1, 64]
out = concat(heads) @ Wo        # [1, 2048]
```

### Sub-tasks

#### 1a. QKV GEMV on NPU
- `x @ Wq`, `x @ Wk`, `x @ Wv` — three matrix-vector products
- Each: [1, 2048] × [2048, 2048]
- Similar to FFN GEMV — reuse existing gemv_int8 infrastructure
- **Challenge**: 3 separate GEMVs, need to batch or pipeline

#### 1b. RoPE on NPU
- Rotary position embedding on Q and K
- Element-wise multiply with sin/cos tables
- Simple kernel, low priority

#### 1c. K cache management
- KV cache: store K and V for all past tokens
- For decode: append new K,V to cache, read full cache for attention
- DMA pattern: write new KV, read full K cache

#### 1d. Attention scores: Q @ K^T
- Q: [n_heads, 1, head_dim] — single query per head
- K: [n_heads, seq_len, head_dim] — full cache
- Result: [n_heads, 1, seq_len] — scores per head
- **This is a matrix-matrix multiply** (not GEMV) when seq_len > 1
- For short seq_len (prefill): GEMV-like
- For long seq_len: tiled GEMM on NPU

#### 1e. Softmax on NPU
- Input: [n_heads, 1, seq_len] — scores
- Requires: max subtraction, exp, sum, divide
- **Challenge**: reduction over seq_len dimension (like RMSNorm problem)
- For short seq_len (≤32): single tile, works
- For long seq_len: needs two-pass with reduction

#### 1f. Attention: scores @ V
- scores: [n_heads, 1, seq_len]
- V: [n_heads, seq_len, head_dim]
- Result: [n_heads, 1, head_dim]
- Another matrix-matrix multiply

#### 1g. Output projection
- concat(heads) @ Wo — [1, 2048] × [2048, 2048]
- Same as QKV GEMV

### Attention implementation order

1. **QKV GEMV** (1a) — biggest bang, reuse existing infra
2. **Output projection GEMV** (1g) — same pattern
3. **Q@K^T** (1d) — attention scores
4. **scores@V** (1f) — attention output
5. **Softmax** (1e) — needs reduction infra
6. **RoPE** (1b) — low priority
7. **KV cache DMA** (1c) — optimization

### Key decision: fused vs staged

**Option A — Fused attention kernel** (one NPU dispatch for all of attention):
- Pros: minimal DMA round-trips, best latency
- Cons: massive kernel, hard to debug, L1 memory pressure

**Option B — Staged** (separate dispatch for each sub-op):
- Pros: modular, debuggable, reuse components
- Cons: more DMA overhead (5-6 round-trips per layer)

**Recommendation**: Start with Option B (staged), fuse later when profiling shows DMA overhead matters.

---

## Phase 2: FFN → NPU (Priority #2)

**Expected gain**: 2-3× on FFN portion (~40% of decode)

### Architecture (Llama-3.2-1B)
```
gate = x @ W_gate     # [1, 2048] → [1, 8192]
up   = x @ W_up       # [1, 2048] → [1, 8192]
h    = silu(gate) * up # element-wise
out  = h @ W_down     # [1, 8192] → [1, 2048]
```

### Sub-tasks

#### 2a. Gate/Up GEMV (fused)
- Two GEMVs: [1, 2048] × [2048, 8192] each
- Can fuse into single dispatch (already have fused gate/up infra in codebase)
- Check if existing `fused_gemv_int8` works for this shape

#### 2b. SiLU on NPU
- Element-wise activation: silu(x) = x * sigmoid(x)
- Already exists as `silu` kernel — verify it works for [1, 8192]

#### 2c. Element-wise multiply
- `silu(gate) * up` — simple element-wise
- Already exists as `eltwise_mul` kernel

#### 2d. Down GEMV
- [1, 8192] × [8192, 2048]
- Standard GEMV

### FFN implementation order

1. **Fused gate/up GEMV** (2a) — biggest gain
2. **SiLU** (2b) — small kernel
3. **Eltwise mul** (2c) — small kernel
4. **Down GEMV** (2d) — same pattern as gate/up

---

## Phase 3: RMSNorm → NPU (Low Priority)

**Expected gain**: <1% — not worth it until Phase 1+2 done

### The bug
Current kernel: `rms = sum(x²) / tile_size` where `tile_size=32`
Correct: `rms = sum(x²) / full_row_size` (e.g., 2048)

### Fix: Two-pass RMSNorm

#### Pass 1: Partial sums
```
DMA → core0: sum(x²) over elements [0..255]
DMA → core1: sum(x²) over elements [256..511]
...
DMA → core7: sum(x²) over elements [1792..2047]
→ partial_sums[8] on host
```

#### Reduction
```
global_sum = sum(partial_sums)  // on host or single core
```

#### Pass 2: Normalize
```
DMA(global_sum) → all cores
DMA(input) → all cores
each core: output = input / sqrt(global_sum/2048 + eps)
DMA ← output
```

### New files needed
- `aie_kernels/aie2/rms_norm_pass1.cc` — partial sum kernel
- `aie_kernels/aie2/rms_norm_pass2.cc` — normalize kernel
- `iron/operators/rms_norm/design_two_pass.py` — two-pass DMA flow
- `ggml-xdna.cpp` — two-pass dispatch logic with host-side reduction

### Dependencies
- Reduction infrastructure (may be shared with softmax reduction)
- Only worth building when attention + FFN are on NPU

---

## Phase 4: Remaining ops

| Op | Strategy | Priority |
|---|---|---|
| RoPE | Simple element-wise, low cost | Low |
| Softmax | Needs reduction (reuse RMSNorm infra) | Medium (needed for attention) |
| Embedding lookup | Table lookup, DMA-bound | Low |
| Argmax / sampling | Trivial, CPU is fine | None |

---

## Infrastructure Needed

### Reduction primitive (needed for RMSNorm + Softmax)

Both RMSNorm and Softmax need global reduction across cores. Build once, reuse:

```python
# Generic reduction pattern
def reduce_sum_across_cores(partial_values, num_cores):
    # Option A: host-side (simple, adds 1 DMA round-trip)
    # Option B: dedicated reduction core (faster, more complex)
    # Option C: tree reduction via stream switches (fastest, hardest)
```

**Recommendation**: Start with host-side (Option A), optimize later.

### Existing kernels to reuse

| Kernel | File | Status |
|---|---|---|
| `rms_norm_bf16_vector` | `aie_kernels/aie2/rms_norm.cc` | Works per-tile |
| `weighted_rms_norm` | same | Works per-tile |
| `gemv_int8` | `aie_kernels/aie2/gemv_int8.cc` | Working |
| `silu` | `aie_kernels/aie2/silu.cc` | Need to verify |
| `eltwise_mul` | `aie_kernels/aie2/eltwise_mul.cc` | Need to verify |
| `rope` | `aie_kernels/aie2/rope.cc` | Need to verify |

---

## Milestones

| Milestone | Target | Expected t/s |
|---|---|---|
| Baseline (all CPU) | Done | ~2.5 |
| QKV GEMV on NPU | Phase 1a | ~4-5 |
| Full attention on NPU | Phase 1 | ~6-8 |
| FFN on NPU | Phase 2 | ~10-12 |
| RMSNorm on NPU | Phase 3 | ~12-13 |
| Full model on NPU | Phase 4 | ~15+ |

---

## Environment Notes

### Windows build paths
```
XRT_SDK:     C:\Users\Kuhnya\Downloads\xrt_windows_sdk\xrt_sdk\xrt
PEANO:       C:\ProgramData\miniforge3\envs\ryzen-ai-1.7.1\Lib\site-packages\win64.o\tools\peano
MLIR_AIE:    C:\ProgramData\miniforge3\envs\ryzen-ai-1.7.1\Lib\site-packages\mlir_aie\bin
DRIVER:      C:\Windows\System32\DriverStore\FileRepository\kipudrv.inf_amd64_*
PYTHON:      C:\Python313\python.exe
MODEL:       models\llama-3.2-1b-instruct-BF16.gguf
```

### Key env vars
```
XDNA_ENABLE_RMS_NORM=0       # CPU (broken on NPU)
XDNA_ENABLE_GEMV=1           # GEMV on NPU
XDNA_ENABLE_SWIGLU=1         # SiLU on NPU
XDNA_ENABLE_QKV=1            # QKV on NPU
XDNA_ENABLE_SWIGLU_PREFILL=0 # prefill SiLU off
XDNA_DEBUG=1                 # debug logs
GGML_XDNA_NUM_COLS=8         # use all 8 columns (NPU2)
```

### Test script
`logs/test_short1.bat` — current test harness
