# IRON-windows NPU Optimization Plan

> Last updated: 2026-05-20

## Current Status: **FlowKV + RMSNorm on NPU, 5.9 t/s (near baseline 6.0 t/s)**

FlowKV decode attention is integrated and produces correct output on STX NPU2.
Debug flowkv.bat: Step 2 (no FlowKV) = "The capital of France is Paris." ✅
                    Step 3 (with FlowKV) = "The capital of France is Paris." ✅

Benchmark (run_flowkv_bench.bat, 32 tokens):

| Config | Prompt | Generation |
|--------|--------|------------|
| Baseline (no FlowKV) | 123.7 t/s | **6.0 t/s** |
| FlowKV num_cols=1 | 127.6 t/s | **4.8 t/s** (-20%) |
| FlowKV num_cols=4 (batch) | 126.8 t/s | **5.5 t/s** (-8%) |
| FlowKV batch + RMSNorm NPU | 96.7 t/s | **5.9 t/s** (-2%) |

Batch mode (num_cols=4) reduces dispatch count from 96 to 24 per token.
Root cause of batch failures: `num_heads` parameter was `q_heads_per_kv` (4) instead of `q_heads_per_kv * num_cols` (16), causing `group_size=1` in compiled xclbin.

### FlowKV dispatch profiling breakdown (per token, 12 layers × 2 dispatches)

| Phase | ms/dispatch | ms/token (×24) |
|-------|-------------|----------------|
| KV+Q memcpy+sync | 0.07 | 1.7 |
| run_create | 0.00 | 0.0 |
| **NPU exec** | **0.56 avg** | **13.4** |
| output_sync+scatter | 0.00 | 0.0 |
| **Total dispatch** | | **~15 ms** |

**Conclusion: 83% of overhead is NPU execution time, not host-side.**
memcpy optimization won't help. Bottleneck is NPU compute for 256-position attention.
Baseline 6.0 t/s = 167 ms, FlowKV 5.4 t/s = 185 ms. Difference = 18 ms, dispatch = 15 ms.

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

### ✅ Decode batch efficiency (not a bug)
- `xdna_plan_decode_batch()` finds 4 GEMVs: O_proj, gate_proj, up_proj, down_proj
- SwiGLU matcher (line 11025) consumes gate_proj+up_proj+GLU+down_proj as fused dispatch
- Only O_proj reaches the batcher — this is correct behavior, not a bug
- Result: 2 dispatches/layer (O_proj batch + SwiGLU fused) = 24 dispatches/token
- Residual ADD between O_proj and SwiGLU forces O_proj flush — cannot batch further

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
- Host memcpy optimization: **won't help** — profiling shows memcpy+sync = 1.7 ms/token vs NPU exec = 13.4 ms/token. 83% of overhead is NPU execution.

## Priorities

### Priority 1: Attention → NPU ✅ DONE

FlowKV decode integrated and working (2026-05-19).
- Per-KV-head dispatch (Option 3)
- 8 dispatches for Llama 3.2 1B (group_size=4, 8 KV heads)
- Gate: `XDNA_ENABLE_FLOWKV_DECODE=1`
- Identity RoPE angles (Q already rotated by graph)
- Two bugs fixed: K DMA routing + CONT node skip

### Priority 2: Merge QKV + O_proj (NOT FEASIBLE)

QKV accepts input activation as src[1]. O_proj accepts attention output as src[1].
Different tensors — cannot concatenate into one GEMV. Different xclbins → different
hw_context → XRT runlist throws on add(). QKV already 1 submission for decode (M=1).

### ✅ Priority 3: RMSNorm → NPU (DONE)

RMSNorm runs on NPU with full-row tile_size (2048). Single AIE core processes
entire row — avoids per-tile independent normalization bug (tile_size=32).

Fix: `xdna_select_rms_norm_params()` now uses `tile_size = size` (not hardcoded 32).
Result: **5.9 t/s** (from 5.5 t/s). FlowKV overhead reduced to 2%.
Note: weighted variant (gain) still applied via separate CPU MUL.

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

### Priority 5: Fuse FlowKV + Output Projection (future, high complexity)

**Idea:** Fuse FlowKV attention + O_proj GEMV into a single xrt::run dispatch per layer.

**Current flow:**
```
FlowKV dispatch (0.56ms) → bo_out → host DMA → CONT → O_proj dispatch (0.56ms)
```

**Fused flow:**
```
FlowKV+O_proj dispatch (est. 0.7ms) → bo_final → host
```

**Savings analysis:**
- Eliminate 12 O_proj dispatches: 12 × 0.56ms = 6.7 ms/token
- Fused dispatch slightly slower: ~0.7ms × 12 = 8.4ms (vs 6.7ms FlowKV-only)
- Net savings: ~5 ms/token
- Result: ~5.6 t/s (from 5.4 t/s)

**Challenges:**
- O_proj weight = 2048×2048 bf16 = 8MB. Tile memory = 32KB. Needs tiling/streaming.
- Cross-column dependency: each column computes 4 Q heads, but O_proj needs all 8 KV heads' outputs.
- Would need new IRON design with fused attention + GEMV workers.
- IRON fusion framework (`FusedMLIROperator`) exists but is designed for simple chains, not complex attention patterns.

**Verdict:** ROI is low (large complexity for ~0.2 t/s gain). Main bottleneck is NPU compute time (13.4 ms/token = 83% of overhead), which fused kernel doesn't reduce.

### Priority 6: Multi-column parallelism (BLOCKED)

XDNA 2 physically has 8 compute columns, but XRT runtime allocates only 4 per user context.
Remaining 4 are reserved for OS background AI tasks (Windows Studio Effects, noise cancellation).
SequentialPlacer fails with "Failed to find a tile matching column 2: tried until column 8".
**num_cols=4 is the hard limit** for user XRT contexts on client Ryzen AI processors.

### Priority 7: Speculative decoding (from albiol2004, ggml-org/llama.cpp#21725)

albiol2004 notes: "XDNA NPUs are not designed for LLMs — decode is memory-bound,
NPU compute underutilized. Speculative decoding can still squeeze a lot of performance."

Speculative decoding drafts 4-8 tokens in parallel, then verifies in one batch.
This transforms decode from memory-bound (1 token) to compute-bound (N tokens),
充分利用 NPU parallelism. NPU has 4 columns × 4 rows = 16 tiles — ideal for
batch verification.

Estimated gain: 3-5x throughput if speculative draft is fast (small model on CPU).

### Priority 8: INT4 quantization (from albiol2004)

albiol2004: "FLM appears to work only at INT4. Want to support both INT8 and INT4
so that virtually every GGUF runs on it."

INT4 reduces weight memory 4x vs bf16,大幅 reducing DMA transfer time.
Current bottleneck is NPU execution (13.4 ms/token) which is dominated by
weight streaming from DDR. INT4 would cut this proportionally.

Challenge: need INT4 dequantization kernel on AIE tiles, or host-side dequant.

### Priority 9: 64K SMMU alignment (from ic, ggml-org/llama.cpp#21725)

ic reports: "IRON contact with 64K alignment matching SMMU page got 41 t/s"
(vs 4 t/s without alignment, vs 61 t/s with FLM).

Current implementation uses 64-byte alignment. 64K alignment may reduce
SMMU page table overhead for DMA transfers. Worth benchmarking.

### Priority 10: Async NPU dispatch (from XDNA_OPTIMIZATION_PLAN.md)

Submit NPU run, immediately continue with CPU ops (residual add, RMSNorm).
Wait for NPU only when result is needed. Overlap NPU compute with CPU-side ops.

Currently we block on `rl.wait()` every time. If we overlap NPU compute with
CPU residual add + RMSNorm — estimated 10-15% improvement from pipelining.

### Priority 11: KV cache persistent on NPU (from XDNA_OPTIMIZATION_PLAN.md)

Pre-allocate KV cache BO at model load for max seq_len. Only DMA the new K/V
row each token, not the full cache. NPU reads cache directly from device memory.

Currently FlowKV re-uploads full K/V cache (256 × 128 bytes) every dispatch.
Persistent KV cache would save ~1.7 ms/token (the memcpy+sync overhead).

### Priority 12: Quick wins (from XDNA_OPTIMIZATION_PLAN.md)

1. **Weight BO lookup O(1)** — replace `unordered_map<void*, xrt::bo>` with
   `std::vector<xrt::bo>` indexed by layer+slot. O(1) instead of hash lookup.

2. **Prefetch next layer** — while NPU processes layer N, CPU prepares BO args
   for layer N+1. Double-buffering: zero setup latency between layers.

3. **Remove mutex on hot path** — `weights_mutex` lock on every dispatch.
   Decode is single-threaded — no contention. Use atomics or thread-local state.

### Future work

- **Multi-layer packing** — pack N consecutive transformer blocks into one ELF.
  Reduces host dispatch by Nx. Requires major IRON compiler changes.
- **Compile-time tile profiling** — benchmark tile_m/tile_k/tile_n combinations,
  build lookup table for optimal tiles per shape. Useful for new xclbin shapes.

### Known limitation: FlowKV identity RoPE (low priority)

Kernel uses identity RoPE angles (cos=1.0, sin=0.0). Q and K are already
post-RoPE from ggml, so the angles buffer is unused by the kernel. This
is not a bug — the kernel correctly computes attention over pre-rotated Q/K.

### Known bug: FlowKV garbage on multi-query sessions (FIXED)

Fixed by using RoPE position tensor for actual_seq_len instead of binary
search on K data zeros. Binary search was unreliable when KV cache contained
stale data from previous queries.

RoPE node at graph index i+2 (Q MUL_MAT → Q RESHAPE → Q ROPE). Position
tensor src[1] contains exact position for each token. actual_seq_len = max_pos + 1.

Multi-question test works: "What is the capital of France? And 2+2? And 3+3?"
→ "The capital of France is Paris. Two plus two is four." ✅

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
set XDNA_ENABLE_RMS_NORM=1
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
| + Batch 4 KV heads (num_cols=4) | **5.5 t/s** | — |
| + RMSNorm on NPU | **5.9 t/s** | — |
| + Multi-column parallelism (8 cols) | BLOCKED (XRT 4 col limit) | ~8-10 t/s |
| + Merge QKV+batch | — | ~12-13 t/s |
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
