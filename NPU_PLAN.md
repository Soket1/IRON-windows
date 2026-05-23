# IRON-windows NPU Optimization Plan

> Last updated: 2026-05-22

## Current Status: **5.7 t/s INT4 / 5.3 t/s bf16 on STX NPU2** (Llama 3.2 1B)

After Priority 8 INT4 work (v2 kernel from amd/IRON PR #101 + Q4_K
support) landed 2026-05-22, INT4 finally beats bf16 on NPU:

| Config (Llama 3.2 1B) | decode t/s |
|---|---:|
| CPU Q4_0 (reference) | 11.1 |
| **NPU INT4 v2 (production default)** | **5.7** |
| NPU bf16 | 5.3 |
| NPU INT4 v1 (old) | 3.4 |

Q4_K_M models route through the same v2 kernel via host repack
(Phase 8.4). See `NPU_PLAN_PRIORITY_8.md` for the full Priority 8
writeup.

## Supported model architectures

- ✅ **Llama 3.2 1B** (and other head_dim=64 Llama-arch models). All
  fusion paths (FlowKV, QKV, decode_batch, transformer_block, SwiGLU)
  active by default.
- ⚠️ **Non-Llama architectures** (Qwen, Mistral, Gemma, etc with
  head_dim ≠ 64, M-RoPE, or SWA): use the `npu_int4_gemv_only` preset.
  Attention runs on CPU; only the pure matmul INT4 GEMV path
  accelerates FFN/QKV-proj matmuls. Tested on Qwen3.5-9B-Q4_0:
  3.5 t/s decode (+9% over CPU 3.2 t/s), 737 chars exact match vs
  CPU baseline before normal bf16 drift.
- **Why**: `head_dim != 64` is hardcoded in 9 NPU matchers
  (ggml-xdna.cpp lines 5546, 7184, 7551, 8855, 9030, 10097, 10166,
  10297, 11447). M-RoPE / SWA are also Llama-MHA-specific.
  Lifting these is a future Phase 8.5 (3-5 days FlowKV + matcher
  rework) — not yet on the roadmap.

## Older context

- **Single-query** (`--single-turn` or one prompt per process): all NPU
  operators safe, including RMS_NORM → **5.9 t/s**.
- **Chat-mode multi-query** (`-cnv` with multiple turns in one process):
  RMS_NORM⊕QKV interference breaks Q2+ output. Workaround: set
  `XDNA_ENABLE_RMS_NORM=0`. RMS_NORM falls back to CPU → **5.5 t/s**.
  See `## Known bug: multi-query garbage` below for details and the
  proposed long-term fix (multi-column RMS_NORM with reduction).

Baseline (no FlowKV, no RMS_NORM on NPU) = **6.0 t/s** generation.

`debug_flowkv.bat` Step 2 (no FlowKV) = "The capital of France is Paris." ✅
                    Step 3 (with FlowKV) = "The capital of France is Paris." ✅

Benchmark (`run_flowkv_bench.bat`, 32 tokens, single-query):

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

These historical bugs have been resolved. Details of the discovery and fixes are documented in the [Appendix: Debugging History Log](#two-bugs-found-and-fixed-2026-05-19).

## Component Status

| Component | Status | NPU? | Per-dispatch | Per token |
|---|---|---|---|---|
| QKV projection | ✅ Working | NPU (8col) | 1.17 ms | 14.0 ms (×12 layers) |
| SwiGLU (gate+up+silu+down) | ✅ Working | NPU (8col) | 3.62 ms | 43.4 ms |
| Output projection (decode batch) | ✅ Working | NPU (8col) | 1.06 ms | 12.7 ms |
| **Attention (FlowKV decode, batched)** | ✅ Working | NPU (4col, num_cols=4) | ~0.56 ms × 2 batches | ~13.4 ms (×16 layers, 32 dispatches total) |
| RMSNorm (full-row tile, no gain) | ⚠ Conditional NPU (1col) | NPU single-query / CPU chat-mode | ~0.05 ms | ~1.6 ms (when on NPU) |
| MUL (gain), Residual ADD, RoPE, KV cache | ❌ CPU | CPU | — | ~10-20 ms total |

**Important asymmetry**: RMS_NORM is the only **1-col** operator. It works
correctly in isolation and in single-query mode, but its 1-col hw_context
conflicts with the 8-col QKV context in chat-mode multi-query (root cause
documented in `## Known bug` section). Until a multi-column RMS_NORM
design lands, chat-mode users should set `XDNA_ENABLE_RMS_NORM=0`.

**Historical FlowKV dispatch counts**: earlier text mentioned "8 dispatches
× ~2 ms" for FlowKV (per-KV-head, num_cols=1). That predates the batched
path. Current production path is `num_cols=4` batching → **2 dispatches
per layer** = **32 per token** for Llama-3.2-1B (16 layers × 2 batches of
4 KV heads each), each ~0.56 ms NPU exec. The legacy per-head dispatch
function (`ggml_backend_xdna_flowkv_per_head`) is dead code — kept in
the source but `xdna_plan_flowkv()` returns empty in the current
cgraph layout, so it never fires. Scheduled for removal.

## Dispatch pattern per layer (current production path)

```
graph_compute n_nodes=1   → SOFT_MAX (skipped when FlowKV enabled)
graph_compute n_nodes=N   → Main layer:
  RMS_NORM      → NPU 1-col (single-query) | CPU (chat-mode w/ workaround)
  MUL (gain)    → CPU (one weighted-RMSNorm fusion attempt failed, see lessons)
  MUL_MAT ×3    → NPU via QKV dispatch (8col, 1.17 ms)
  ROPE ×2       → CPU
  KV cache ops  → CPU
  CONT + attn   → FlowKV POC dispatch (4col, 2 batches × ~0.56 ms = ~1.12 ms NPU,
                  + ~16 ms total per-token across 16 layers including host overhead)
  ADD           → CPU (residual)
  RMS_NORM      → as above
  MUL (gain)    → CPU
  GLU/SwiGLU    → NPU (8col, 3.62 ms)
  ADD           → CPU (residual)
```

## Per-dispatch profiling (decode M=1)

| Operation | NPU exec time |
|---|---|
| SwiGLU | 3620 µs |
| QKV | 1165 µs |
| O_proj (decode_batch) | 1055 µs |
| **FlowKV (batched, ~0.56ms × 2)** | **~1120 µs per layer = ~13.4 ms/token** |
| RMSNorm (when on NPU) | ~50 µs |

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

### Priority 1: Attention → NPU ✅ DONE [TASK-P1]

FlowKV decode integrated and working (2026-05-19).
- **[TASK-P1.1]** Per-KV-head dispatch (Option 3)
- **[TASK-P1.2]** 8 dispatches for Llama 3.2 1B (group_size=4, 8 KV heads)
- **[TASK-P1.3]** Gate: `XDNA_ENABLE_FLOWKV_DECODE=1`
- **[TASK-P1.4]** Identity RoPE angles (Q already rotated by graph)
- **[TASK-P1.5]** Two bugs fixed: K DMA routing + CONT node skip

### Priority 2: Merge QKV + O_proj (NOT FEASIBLE) [TASK-P2]

QKV accepts input activation as src[1]. O_proj accepts attention output as src[1].
Different tensors — cannot concatenate into one GEMV. Different xclbins → different
hw_context → XRT runlist throws on add(). QKV already 1 submission for decode (M=1).

### ✅ Priority 3: RMSNorm → NPU (DONE) [TASK-P3]

RMSNorm runs on NPU with full-row tile_size (2048). Single AIE core processes
entire row — avoids per-tile independent normalization bug (tile_size=32).

**[TASK-P3.1]** Fix: `xdna_select_rms_norm_params()` now uses `tile_size = size` (not hardcoded 32).
**[TASK-P3.2]** Result: **5.9 t/s** (from 5.5 t/s). FlowKV overhead reduced to 2%.
**[TASK-P3.3]** Note: weighted variant (gain) still applied via separate CPU MUL.

### Priority 4: Batch KV heads per dispatch ✅ DONE (2026-05-20) [TASK-P4]

**Problem:** FlowKV dispatches 8 times per layer (1 per KV head) = 96 dispatches per token.
Dispatch overhead (xrt::run.wait hardware time) dominates.

**Solution:** Batch 4 KV heads into a single xrt::run call.
- **[TASK-P4.1]** Dispatch count: 8 → 2 per layer (24 per token instead of 96)
- **[TASK-P4.2]** IRON design uses 4 columns, runtime sequence fills 4 KV heads in parallel
- **[TASK-P4.3]** Cache key: `flowkv_H16_KV4_d64_S256_C32_4col`

**[TASK-P4.4]** **Root cause of previous failures:**
`num_heads` param passed to `get_or_load_flowkv_kernel()` was `q_heads_per_kv` (4) instead of `q_heads_per_kv * num_cols` (16). This compiled xclbin with `group_size=1` while host prepared data for `group_size=4`, causing DMA misalignment and garbage output.

**[TASK-P4.5]** **Aligned stride (PDF fix):**
For S256, `raw_head_bytes = 256*64*2 = 32768` — already 64-byte aligned. The PDF fix only matters when `raw_head_bytes % 64 != 0`. Still applied for correctness with arbitrary seq_len.

### Priority 5: Fuse FlowKV + Output Projection (future, high complexity) [TASK-P5]

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
- **[TASK-P5.1]** Eliminate 12 O_proj dispatches: 12 × 0.56ms = 6.7 ms/token
- **[TASK-P5.2]** Fused dispatch slightly slower: ~0.7ms × 12 = 8.4ms (vs 6.7ms FlowKV-only)
- **[TASK-P5.3]** Net savings: ~5 ms/token
- **[TASK-P5.4]** Result: ~5.6 t/s (from 5.4 t/s)

**Challenges:**
- **[TASK-P5.5]** O_proj weight = 2048×2048 bf16 = 8MB. Tile memory = 32KB. Needs tiling/streaming.
- **[TASK-P5.6]** Cross-column dependency: each column computes 4 Q heads, but O_proj needs all 8 KV heads' outputs.
- **[TASK-P5.7]** Would need new IRON design with fused attention + GEMV workers.
- **[TASK-P5.8]** IRON fusion framework (`FusedMLIROperator`) exists but is designed for simple chains, not complex attention patterns.

**[TASK-P5.9]** **Verdict:** ROI is low (large complexity for ~0.2 t/s gain). Main bottleneck is NPU compute time (13.4 ms/token = 83% of overhead), which fused kernel doesn't reduce.

### Priority 6: Multi-column parallelism (BLOCKED) [TASK-P6]

XDNA 2 physically has 8 compute columns, but XRT runtime allocates only 4 per user context.
Remaining 4 are reserved for OS background AI tasks (Windows Studio Effects, noise cancellation).
SequentialPlacer fails with "Failed to find a tile matching column 2: tried until column 8".
**[TASK-P6.1]** **num_cols=4 is the hard limit** for user XRT contexts on client Ryzen AI processors.

### Priority 7: Speculative decoding (from albiol2004, ggml-org/llama.cpp#21725) [TASK-P7]

albiol2004 notes: "XDNA NPUs are not designed for LLMs — decode is memory-bound,
NPU compute underutilized. Speculative decoding can still squeeze a lot of performance."

**[TASK-P7.1]** Speculative decoding drafts 4-8 tokens in parallel, then verifies in one batch.
**[TASK-P7.2]** This transforms decode from memory-bound (1 token) to compute-bound (N tokens),
充分利用 NPU parallelism. NPU has 4 columns × 4 rows = 16 tiles — ideal for
batch verification.

**[TASK-P7.3]** Estimated gain: 3-5x throughput if speculative draft is fast (small model on CPU).

### Priority 8: INT4 quantization (from albiol2004) [TASK-P8]

albiol2004: "FLM appears to work only at INT4. Want to support both INT8 and INT4
so that virtually every GGUF runs on it."

**[TASK-P8.1]** INT4 reduces weight memory 4x vs bf16,大幅 reducing DMA transfer time.
**[TASK-P8.2]** Current bottleneck is NPU execution (13.4 ms/token) which is dominated by
weight streaming from DDR. INT4 would cut this proportionally.

**[TASK-P8.3]** Challenge: need INT4 dequantization kernel on AIE tiles, or host-side dequant.

### Priority 9: 64K SMMU alignment (from ic, ggml-org/llama.cpp#21725) [TASK-P9]

ic reports: "IRON contact with 64K alignment matching SMMU page got 41 t/s"
(vs 4 t/s without alignment, vs 61 t/s with FLM).

**[TASK-P9.1]** Current implementation uses 64-byte alignment. 64K alignment may reduce
SMMU page table overhead for DMA transfers. Worth benchmarking.

### Priority 10: Async NPU dispatch (from XDNA_OPTIMIZATION_PLAN.md) [TASK-P10]

Submit NPU run, immediately continue with CPU ops (residual add, RMSNorm).
Wait for NPU only when result is needed. Overlap NPU compute with CPU-side ops.

**[TASK-P10.1]** Currently we block on `rl.wait()` every time. If we overlap NPU compute with
CPU residual add + RMSNorm — estimated 10-15% improvement from pipelining.

### Priority 11: KV cache persistent on NPU (from XDNA_OPTIMIZATION_PLAN.md) [TASK-P11]

Pre-allocate KV cache BO at model load for max seq_len. Only DMA the new K/V
row each token, not the full cache. NPU reads cache directly from device memory.

**[TASK-P11.1]** Currently FlowKV re-uploads full K/V cache (256 × 128 bytes) every dispatch.
**[TASK-P11.2]** Persistent KV cache would save ~1.7 ms/token (the memcpy+sync overhead).

### Priority 12: Quick wins (from XDNA_OPTIMIZATION_PLAN.md) [TASK-P12]

1. **[TASK-P12.1]** **Weight BO lookup O(1)** — replace `unordered_map<void*, xrt::bo>` with
   `std::vector<xrt::bo>` indexed by layer+slot. O(1) instead of hash lookup.

2. **[TASK-P12.2]** **Prefetch next layer** — while NPU processes layer N, CPU prepares BO args
   for layer N+1. Double-buffering: zero setup latency between layers.

3. **[TASK-P12.3]** **Remove mutex on hot path** — `weights_mutex` lock on every dispatch.
   Decode is single-threaded — no contention. Use atomics or thread-local state.

### Future work [TASK-PFUTURE]

- **[TASK-PFUTURE.1]** **Multi-layer packing** — pack N consecutive transformer blocks into one ELF.
  Reduces host dispatch by Nx. Requires major IRON compiler changes.
- **[TASK-PFUTURE.2]** **Compile-time tile profiling** — benchmark tile_m/tile_k/tile_n combinations,
  build lookup table for optimal tiles per shape. Useful for new xclbin shapes.
- **[TASK-PFUTURE.3]** **Multi-column RMS_NORM with cross-core reduction** — unblocks
  `XDNA_ENABLE_RMS_NORM=1` in chat-mode (currently breaks multi-query
  via RMS_NORM⊕QKV interference; see hypothesis #2 below). Needs a new
  IRON design with a two-pass or shared-memory reduction step so the
  kernel can run on `num_cols × channels × tile_size = size` with
  `cols > 1` without falling back to per-tile means. Existing 1-col
  full-row design works correctly but its 1-col hw_context conflicts
  with the 8-col QKV/SwiGLU contexts when both are alive. Expected
  payoff: +0.4 t/s (5.5 → 5.9) in chat-mode, no other gains since
  RMS_NORM is ~1% of token time. Low ROI unless chat-mode performance
  is the explicit goal — single-query workflows aren't affected.

### Known limitation: FlowKV identity RoPE (low priority)

Kernel uses identity RoPE angles (cos=1.0, sin=0.0). Q and K are already
post-RoPE from ggml, so the angles buffer is unused by the kernel. This
is not a bug — the kernel correctly computes attention over pre-rotated Q/K.

### Known bug: multi-query garbage — RMS_NORM⊕QKV (RESOLVED ROOT CAUSE 2026-05-21)

**Update 2026-05-21:** the multi-query garbage is **NOT a FlowKV bug**.
Bisect via toggling `XDNA_ENABLE_*` shows it is interference between
`XDNA_ENABLE_RMS_NORM=1` and `XDNA_ENABLE_QKV=1`. Either alone works in
chat-mode. Both together: Q2 garbage. FlowKV ON or OFF makes no difference.
See "Multi-query garbage — ROOT CAUSE" section below for the bisect matrix.

**Workaround:** disable either `XDNA_ENABLE_RMS_NORM` (back to 5.5 t/s)
or `XDNA_ENABLE_QKV` (slower QKV on CPU). Other NPU operators are safe.

The historical FlowKV-related text below is preserved for context but
is misleading — the bug was misattributed to FlowKV because FlowKV was
the latest addition. RMS_NORM⊕QKV interference is the real fault.

### Historical: FlowKV garbage on multi-query sessions (now reclassified)

FlowKV works correctly for single queries (short and long prompts, 5.4-5.8 t/s).
In interactive chat sessions (multiple queries in one process), later queries
produce garbage.

Attempted fixes (all reverted — didn't solve the issue):
- Binary search on K data zeros — unreliable with stale KV cache
- RoPE position tensor — n_past is correct, but output still garbage
- Staleness check (Q data pointer) — didn't help
- Seq_len change detection — didn't help
- Three-invariant fix (a3ef22209 + 50f826233) — didn't solve garbage (see below)

Root cause is deeper than actual_seq_len detection. Likely KV cache management
or attention computation with contaminated cache from previous queries.

**Workaround:** use separate llama-cli processes per query.

**Potential fix:** pass actual_seq_len as kernel argument (requires IRON design
changes + kernel signature modification). See user's suggestion about
kernel-side position_offset mask.

### Historical Diagnostics (2026-05-21)

All diagnostic sessions and investigations relating to multi-query garbage, `MATH_DIAG` breakthroughs, and `RMS_NORM ⊕ QKV` interference have been moved to the [Appendix: Debugging History Log](#appendix-debugging-history-log) at the end of the file.

## Testing

### Correctness harness (recommended for every NPU change)

```powershell
python C:\llama.cpp-xdna\ggml\src\ggml-xdna\tools\correctness_test.py            # all tests (~3-5 min)
python C:\llama.cpp-xdna\ggml\src\ggml-xdna\tools\correctness_test.py paris_short  # one test (~20 s)
python C:\llama.cpp-xdna\ggml\src\ggml-xdna\tools\correctness_test.py --list       # show all tests
```

Python stdlib-only script (no deps). Runs each test against three operator
presets — `cpu_baseline`, `npu_chat_safe` (production config), `npu_full`
(everything incl. RMS_NORM) — with greedy sampling (`--temp 0 --seed 42`)
for determinism, then compares the generated text token-for-token.

Current test cases:
| Test | What it probes |
|---|---|
| `paris_short` | Sanity. Short prompt, 16 tokens. |
| `paris_drift_128` | bf16 numerical drift over 128 tokens. |
| `multiquery_basic` | RMS_NORM+QKV regression (`npu_full` expected fail). |
| `multiquery_chatsafe` | 3-query chat with the production workaround. |
| `seq_len_short` | Tiny seq case (V-PERMUTE matcher gating). |
| `seq_len_near_chunk32` | FlowKV chunk_size=32 boundary crossing. |

Exit code 0 if all pass, 1 if any fail. `expected_fail` for known issues
flips logic: an accidental fix surfaces as `?? UNEXPECTED PASS` (also a fail).
Add new test cases by appending to `TESTS` in the script.

### FlowKV verification (legacy quick check)

```bat
debug_flowkv.bat
```

Runs Step 2 (baseline without FlowKV) and Step 3 (with FlowKV).
Both should output "The capital of France is Paris." Subsumed by the
correctness harness above; kept for quick smoke checks.

### Diagnostic tools

- `XDNA_DEBUG=1` — enables `FlowKV-DIAG` + `BO ADDRESS DIAGNOSTIC` + `poc_dbg`
  blocks (very verbose, ~750 lines of stderr per token)
- `XDNA_FLOWKV_BO_PROBE=1` — dumps source vs BO bytes (Q/K/V at pos 0, mid, last)
- `XDNA_FLOWKV_MATH_DIAG=1` — NPU output vs CPU reference comparison (now fixed
  for num_cols>1, see commit 250eed45b)
- `XDNA_FLOWKV_REAL_PROBE=1` — v10 data transport probe
- `XDNA_FLOWKV_PER_HEAD_LEGACY=1` — re-enable the legacy per-head FlowKV path
  (default disabled, see commit ea435baa2)

### Current NPU config

```bat
set XDNA_ENABLE_GEMV=1
set XDNA_ENABLE_SWIGLU=1
set XDNA_ENABLE_QKV=1
set XDNA_ENABLE_RMS_NORM=0        :: 1 for single-query (+0.4 t/s), 0 for chat-mode
set XDNA_ENABLE_SWIGLU_PREFILL=0
set XDNA_ENABLE_DECODE_BATCH=1
set XDNA_ENABLE_TRANSFORMER_BLOCK=1
set XDNA_ENABLE_FLOWKV_DECODE=1
set GGML_XDNA_NUM_COLS=8
```

**Note on `GGML_XDNA_NUM_COLS=8`:** Earlier text in this document (and
NPU_PLAN's Priority 6) claimed XRT "hard-limits" user contexts to 4
columns. That turned out to be operator-specific: SequentialPlacer
fails for a single 8-col IRON design, but QKV/SwiGLU (compiled as
8col) actually do dispatch and run on this STX NPU2 alongside
FlowKV (4col). Concretely the current cache has:
- `qkv_K2048_Nq2048_Nk512_Nv512_8col` ← runs
- `swiglu_*_8col` ← runs
- `flowkv_*_4col` ← runs
- `rms_norm_S2048_bf16_1c1ch_t2048` ← runs but conflicts with QKV in chat-mode

So `GGML_XDNA_NUM_COLS=8` is the right value for QKV/SwiGLU; FlowKV
hardcodes its own `num_cols=4` internally; RMS_NORM hardcodes 1. The
plan's earlier "4 col hard limit" wording is outdated and should be
read as "mixed col counts coexist except the 1-vs-8 case".

## Milestones

| Milestone | Current | Target |
|---|---|---|
| Steady-state (no FlowKV) | **5.8–6.0 t/s** | — |
| + Attention on NPU (FlowKV num_cols=1) | **4.8 t/s** | — |
| + Batch 4 KV heads (num_cols=4) | **5.5 t/s** | — |
| + RMSNorm on NPU (single-query only) | **5.9 t/s** | — |
| + RMSNorm on NPU also in chat-mode | BLOCKED (RMS⊕QKV) | 5.9 t/s |
| + Multi-row FlowKV parallelism (4×4 split per KV group) | — | ~7-8 t/s |
| + INT4 weights for QKV/SwiGLU/O_proj | — | ~9-12 t/s |
| + Speculative decoding | — | ~15-25 t/s |
| Full model on NPU + all of the above | — | aspirational |

(Removed earlier "Merge QKV+O_proj" milestone — Priority 2 below
explicitly marks that fusion as NOT FEASIBLE due to incompatible
src[1] tensors, so it can't appear as a roadmap target.)

## Architecture

### Authoritative FlowKV ABI (host ↔ kernel contract, as of 2026-05-21)

**This section is the single source of truth.** If `design.py` /
`flowkv.cc` / host code disagree, the host code (ggml-xdna.cpp POC
dispatch block, lines ~11100-11800) is what actually runs and should
be matched.

**Kernel arguments** (xrt::run.set_arg index → semantics):

| Arg # | Name in xclbin | group_id | Host BO | Purpose |
|-------|---------------|----------|---------|---------|
| 0 | opcode | — | inline u32 = 3 | xrt instruction opcode |
| 1 | insts | — | `fk_entry->insts_bo` | NPU microcode buffer |
| 2 | insts_size | — | inline u32 | size of insts in bytes |
| 3 | DDR_buf_0 | 3 | `fk_entry->bo_k` | K-only mirror (see below) |
| 4 | DDR_buf_1 | 4 | `fk_entry->bo_v` | **K+V combined** in one BO |
| 5 | DDR_buf_2 | 5 | `fk_entry->bo_q` | Q + angles + actual_seq_len + magic |
| 6 | DDR_buf_3 | 6 | `fk_entry->bo_out` | output (kqv_out for the batch) |
| 7 | (unknown) | — | inline u32 = 0 | reserved |

**Why arg3 (bo_k) duplicates K from arg4 (bo_v):** the IRON compiler
emits DMA descriptors that read the K stream from arg0 (which we route
to bo_k via group_id 3). It reads the V stream from arg1 (bo_v) at
offset `aligned_v_region_offset_bytes`. So the K data lives in **two
host BOs** (kernel reads it once via the K-stream DMA from bo_k), and
the V data lives only in bo_v after the K region. The host writes K
into bo_v[0 : kv_region_size) first, then `memcpy`s the same K bytes
into bo_k. Cost is ~32 KB per dispatch, negligible.

**Layout inside `bo_v`** (the K+V combined buffer, total
`num_cols * 2 * aligned_head_stride_bytes`):

```
offset 0
  ┌─────────────────────────────────────────────┐
  │ K region (num_cols × aligned_head_stride)   │  ← K_col0, K_col1, …, K_col(N-1)
  ├─────────────────────────────────────────────┤  ← aligned_v_region_offset_bytes
  │ V region (num_cols × aligned_head_stride)   │  ← V_col0, V_col1, …, V_col(N-1)
  └─────────────────────────────────────────────┘
```

Each `K_col_i` and `V_col_i` is laid out as `[seq_len][head_dim]` bf16
row-major (positions [0..actual_seq_len) populated, the rest zeroed by
a `memset(bo_v_ptr, 0, …)` at the start of each batch).

`aligned_head_stride_bytes = ceil(seq_len * head_dim * 2 / 64) * 64`
(Shim DMA needs 64-byte alignment for parallel channels).

`aligned_v_region_offset_bytes =
   ceil(num_cols * aligned_head_stride_bytes / 64) * 64`.

**Layout inside `bo_q`** (per kv-head-group, repeated `num_cols`
times):

```
offset col * q_group_stride
  ┌───────────────────────────────────────────────────────┐
  │ Q group: q_heads_per_kv × head_dim bf16              │
  ├───────────────────────────────────────────────────────┤  ← angles_off
  │ angles[head_dim]: cos/sin pairs (currently identity:  │
  │   cos=1.0=0x3F80, sin=0.0=0x0000 — kernel ignores)    │
  ├───────────────────────────────────────────────────────┤
  │ actual_seq_len: 1 bf16 (host writes (uint16)(f32_to_bf16(int_to_f32))) │
  ├───────────────────────────────────────────────────────┤
  │ probe_magic: 1 bf16 (diagnostic, kernel may ignore)   │
  └───────────────────────────────────────────────────────┘
```

`q_group_stride = ceil((q_heads_per_kv * head_dim + head_dim + 2) * 2 / 64) * 64`.

`angles_off = q_heads_per_kv * head_dim * 2` bytes from group start.

`actual_seq_len` lives at `angles_off + head_dim * 2` bytes.

**Layout inside `bo_out`**: `num_cols * aligned_out_stride_bytes`
where `aligned_out_stride_bytes = ceil(q_heads_per_kv * head_dim * 2 / 64) * 64`.
Each column-slot holds `q_heads_per_kv * head_dim` bf16 outputs.
Host scatters columns back into the model's `kqv_out` tensor in
contiguous q_head order.

**RoPE status (clarification):** the kernel reads the angles buffer
but currently treats it as identity (cos=1.0, sin=0.0). Q and K
delivered to the kernel are **already RoPE-rotated** by the ggml ROPE
op upstream (CPU). So the angles buffer is functionally dead but kept
in the ABI for future re-enabling of in-kernel RoPE. Do not document
this as "fused RoPE" — the kernel does no rotation work.

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

## Appendix: Debugging History Log

This appendix consolidates all historical debugging logs, diagnostic sessions, and breakthroughs for reference.

### Two bugs found and fixed (2026-05-19)

#### Bug 1 (v10): K DMA routing
IRON compiler reads K FIFO from arg0 (bo_k), not arg1 (bo_v) as assumed.
Host wrote K only to bo_v[0] → tile saw zeros.
Fix: mirror K data into bo_k via memcpy after writing to bo_v.

#### Bug 2 (v11): CONT node skip
`continue` skipped CONT node after FlowKV dispatch. MUL_MAT read from
CONT output buffer (empty), not from kqv_out->data (CONT input).
Fix: removed `continue`, let CONT execute in CPU range.

Both bugs masked each other. After both fixes, model produces correct output.

### Diagnostic session (2026-05-21): reproduced multi-query garbage

Build `b8953-50f826233` (head of `ggml-xdna`, three-invariant fix applied).

**Setup:** Test driven from MSYS bash on Windows. Configuration matches
`debug_flowkv.bat` (XDNA_ENABLE_FLOWKV_DECODE=1, num_cols=8 cache dir).

**PowerShell stderr capture gotcha:** PS5.1 `2>` redirect silently dropped
all native-process stderr from llama-cli.exe in our reproducer — stderr.log
ended up 0 bytes despite the binary printing ~2.6 MB of diagnostics. Same
binary under MSYS bash captured everything. **For FlowKV debugging on
Windows, use `cmd /c` or `Start-Process -RedirectStandardError` — not PS `2>`.**

**Tests:**
| Mode | Result |
|------|--------|
| Single-query (`--single-turn`) | "The capital of France is Paris." ✅ 4.2 t/s, 224 POC dispatches |
| Single-query + `GGML_SCHED_KV_OFFLOAD=1` | "The capital of France is Paris." ✅ 5.6 t/s |
| Chat-mode (`-cnv`, stdin-piped 2 queries) | Q1 "Paris." ✅, Q2 "Hellofnfnf...fawfahav" ❌ |

`GGML_SCHED_KV_OFFLOAD=1` did not break single-query — earlier suspicion ruled out.

**stderr analysis (chat mode, 2.6 MB):**
- 2 prefill rounds at lines 8 and 410 — **both happen before any M=1 decode**
- 71 decode tokens (1136 layer-events / 16 layers), all M=1, all from line 746+
- 2272 POC dispatches (32 per token = 16 layers × 2 num_cols=4 batches)
- BO addresses constant across all dispatches (no DMA layout drift):
  `bo_k=0x3F2D000  bo_v=0x3F4D000  bo_q=0x3F1C000  bo_out=0x3F1D000`
- V-K delta = 131072 bytes (exactly `k_size`) — no driver metadata insertion
- Both single-query and chat-mode use **identical** POC mechanics: same BO sizes,
  group_ids, addresses. POC fires for every layer of every decode token in both.

**Implication:** the transition from "correct" (~first 10 decodes = Q1 response)
to "garbage" (~next 60 decodes = Q2 response) happens DURING one continuous
M=1 decode stream. POC fires the same way for both halves — but data fed into
the BOs (sourced via `flowkv_poc_*_perm->data` pointers) is correct for Q1's
decodes and corrupted for Q2's.

**Why the three-invariant fix doesn't help:**
Inv #1 invalidates `flowkv_poc_valid` only when prefill (M>1) is detected in
the first 30 nodes of a graph_compute. In chat-mode both prefills fire BEFORE
any decode — so the chat-boundary between Q1's EOT and Q2's first decode token
sees no prefill in any segment. POC pointers from Q1 leak across the boundary
into Q2's decodes. Three-invariant Inv #2 (PERMUTE shape matching) is correct
but irrelevant — POC stays valid through the boundary regardless.

**Side finding:** Inv #2's V-PERMUTE matcher `nd->ne[0] > hd && nd->ne[1] == hd`
silently fails when `seq_len ≤ 64` (early generation, short prompt) because
V's `ne[0] = seq_len ≤ 64` doesn't satisfy `> hd`. POC silently disabled in
that regime. Not the cause of current bug, but a hidden gating issue.

**Next minimal fix tried — did NOT work:** Invalidate POC across cgraph
segments where QKV is not refreshed. Tracked last cgraph pointer; invalidated
when `cgraph_changed && !has_decode_qkv`. This dropped POC dispatch count
to **zero** even on single-query test — confirming a3ef22209's reasoning
that QKV and CONT really do live in different graph_compute calls. The
CONT segment (no QKV in first 30 nodes) was being invalidated right before
it could consume POC. Reverted.

**actual_seq_len heuristic is NOT the bug.** With `XDNA_DEBUG=1`, the binary
search returns expected values throughout chat (Q1: 43→49 token-by-token,
jump to 61 at Q2 prefill boundary, Q2: 61→123). Monotonic increments confirm
the K-zero scan is working correctly for this case (cache is contiguous,
no gaps). `actual_seq_len` is correctly encoded into Q BO at angles[head_dim].

**V11 MATH_DIAG (NPU vs CPU reference) is broken for num_cols>1.** The CPU
reference at lines 11806-11807 hard-codes `k=bo_v[0]`, `v=bo_v[seq_len*row_bytes]`
— that's the num_cols=1 layout. With num_cols=4, V actually lives at
`aligned_v_region_offset_bytes` (~128 KB), not `seq_len*row_bytes` (32 KB).
So MATH_DIAG produces FAIL for every dispatch regardless of correctness
(including known-good Q1 dispatches that produce "Paris."). The diagnostic
needs to be updated before it can isolate the bug. Until then, isolating
"is the NPU computing right?" requires either:
1. Fix V11 MATH_DIAG for num_cols=4 layout (read V from offset
   `aligned_v_region_offset_bytes`, not `seq_len*row_bytes`).
2. Force `num_cols=1` to use the MATH_DIAG's existing layout assumption.
3. Probe kqv_out delta (NPU result minus CPU result-before-overwrite) and
   compare across "correct" Q1 decodes and "garbage" Q2 decodes.

**Other observations from this session:**
- POC mechanics are byte-identical between known-good Q1 decodes and garbage
  Q2 decodes: same BO addresses (`0x3F2D000` / `0x3F4D000` / `0x3F1C000` /
  `0x3F1D000`), same sizes, same kernel group_ids, same V-K delta (131072).
- Q2's first decoded token is often coherent ("Bonjour"/"Hello"/"It") before
  text degenerates — suggesting attention is partially right then drifts.
- Decode rate ~5.3-5.7 t/s during both correct (Q1) and garbage (Q2) phases
  — no slowdown indicating the dispatch path is unchanged.

### MATH_DIAG breakthrough (after num_cols>1 fix in commit 250eed45b)

With the V offset fixed, V11 MATH_DIAG now produces meaningful numbers:
- **max_abs stable at 0.0006-0.014 across ALL ~1120 dispatches** (Q1 correct
  AND Q2 garbage). No jump at the chat boundary, no drift over time.
- `FAIL` verdict only because rel_threshold=0.10 trips on near-zero values
  (tiny abs error becomes large relative). Absolute error is well within
  bf16 precision (~0.004).
- **NPU output matches CPU reference computed over the same BO data.**

**This sharply changes the diagnosis.** The NPU computes attention correctly
in absolute terms. Both NPU and CPU agree on the result of the (Q, K, V)
data sitting in the BOs. So either:

1. **Host-side BO data is wrong** for Q2 — `flowkv_poc_*_perm->data` reads
   stale/corrupt source data, both NPU and CPU agree because they read the
   same (wrong) BO. This is consistent with the stale-Q-pointer or
   memory-pool-reuse hypothesis. Next probe: dump first 32 bytes of Q/K/V
   BO contents for first decode of Q1 vs first decode of Q2, compare with
   a CPU pre-image scan of `cache_k`/`cache_v`/`Qcur` source tensors.

2. **Downstream of POC is wrong** — POC correctly overwrites `kqv_out`,
   but a later op reads from somewhere it shouldn't. Less likely since
   single-query works with the same downstream sequence.

POC mechanics, kernel computation, and `actual_seq_len` are now ruled out
as direct causes. Focus shifts to **what `flowkv_poc_*_perm->data` actually
points to** at the Q1→Q2 boundary.

### Multi-query garbage — ROOT CAUSE: RMS_NORM ⊕ QKV interference (2026-05-21)

After exhausting FlowKV-side hypotheses (NPU computation, actual_seq_len,
POC pointer staleness, host data prep), did a bisect by toggling
`XDNA_ENABLE_*` env vars while keeping FlowKV OFF:

| Config | Q2 output |
|---|---|
| All NPU OFF | ✅ "Hello. Is there something I can help you with..." |
| GEMV only | ✅ "Hello again. It's nice to meet you..." |
| GEMV + SWIGLU + QKV | ✅ "It's nice to meet you. Is there something..." |
| GEMV + SWIGLU + QKV + DECODE_BATCH | ✅ "Hello. How can I assist you today?" |
| + TRANSFORMER_BLOCK (no RMS_NORM) | ✅ "Bonjour! How can I assist you today?" |
| + RMS_NORM (= full original config minus FlowKV) | ❌ "Hellodies Hajivalido..." |
| RMS_NORM **only** (everything else OFF) | ✅ "Hello again. What would you like..." |
| **RMS_NORM + QKV** (clean pair) | ❌ "Bonjourïnaïdalectinearadvi..." |
| RMS_NORM + GEMV | ✅ |
| RMS_NORM + GEMV + SWIGLU | ✅ |

**Verdict: Multi-query garbage is NOT a FlowKV bug.** It is an interference
between `XDNA_ENABLE_RMS_NORM=1` and `XDNA_ENABLE_QKV=1`. Either one alone
works correctly in chat-mode. Both together: Q1 ok, Q2 garbage.

Possible mechanisms (untested):
1. **Shared static state.** Both RMS_NORM and QKV cache kernel entries
   and BOs via `static` containers in graph_compute. If the cache key
   collides or one path invalidates the other's BO, Q2's dispatch reads
   stale data.
2. **XRT context column-set conflict.** RMS_NORM is single-column;
   QKV uses 8 columns (`GGML_XDNA_NUM_COLS=8`). Switching between
   contexts mid-graph may leak state if the previous context's columns
   aren't fully released.
3. **In-place output overwrite.** RMS_NORM writes to its output tensor;
   QKV consumes that as `src[1]`. If RMS_NORM's NPU path doesn't fully
   flush before QKV reads (DMA pipeline races), the first decode of Q2
   may see partially-updated activation.

#### Hypothesis verdicts (2026-05-21, code-reading only — no runtime test)

**#1 Shared static state: RULED OUT.**
- Each operator owns its own context-level cache (`rms_norm_cache`,
  `qkv_cache`, `swiglu_cache`, …) keyed by distinct cache_key strings.
  Per-weight BO caches inside each entry are keyed by `data` pointer —
  no cross-operator collision possible.
- No shared static state in the hot-path dispatch functions.

**#3 DMA pipeline race: RULED OUT.**
- `ggml_backend_xdna_rms_norm` does `out_bo->sync(XCL_BO_SYNC_BO_FROM_DEVICE)`
  + CPU memcpy/convert to `node->data` BEFORE returning. Subsequent ops
  see fresh data.
- Between RMS_NORM and QKV the cgraph has a CPU MUL (gain weight),
  which reads RMS_NORM's output and writes its own output entirely on
  the host side. QKV reads `src1_input->data` from host memory, not
  from any device BO of RMS_NORM. No NPU-host pipeline straddles the
  data path.

**#2 XRT column-set conflict: leading candidate.**
- Precompiled kernel cache shows three column counts in use:
  `flowkv: 4col`, `qkv/swiglu: 8col`, `rms_norm: 1c1ch (= 1col)`.
- FlowKV (4col) and QKV (8col) coexist without garbage. The bug only
  appears when the **1-col** RMS_NORM is added — and RMS_NORM is the
  only 1-col operator in the pipeline.
- Each kernel creates its own `xrt::hw_context(device, uuid)`, and the
  XRT runtime time-multiplexes these contexts as different operators
  dispatch. A 1-col hw_ctx and an 8-col hw_ctx must overlap on physical
  column 0; if XRT's context-switch doesn't fully reset tile memory
  state, the 1-col path can leave residue that corrupts the 8-col
  path's read on the next dispatch — visible only on the second query
  because the first query's tile state is still "clean enough".

#### Concrete next experiment

Recompile RMS_NORM xclbin for 8 columns (the kernel logically uses 1
core but allocates all 8 columns to match QKV's footprint, eliminating
the 1-vs-8 switch). Requires running the Python compile pipeline
(MLIR-AIE + Peano) to produce
`rms_norm_S2048_bf16_8c1ch_t2048/combined.xclbin`, then updating
`xdna_select_rms_norm_params()` to return `*out_cols = ctx->num_cols`
instead of hard-coded 1.

If 8-col RMS_NORM coexists with QKV in chat-mode without garbage →
hypothesis #2 confirmed and the right long-term fix is to keep all
operators at the same col count.

If garbage still appears with 8-col RMS_NORM → root cause is elsewhere
(maybe deeper in the kernel itself, IRON design, or XRT runtime).

#### Result of the 8-col compile attempt (2026-05-21): BLOCKED

Tried changing `xdna_select_rms_norm_params()` to return
`*out_cols = max_cols` (8) and triggering the auto-compile path. `compile.py`
rejects the parameter set:

```
ValueError: size (2048) must be a multiple of
  num_aie_columns * num_channels * tile_size (16384)
```

The RMS_NORM IRON design enforces `size = cols × channels × tile_size`.
With `cols=8` and `size=2048`, the only legal tile_size is `2048 / 8 = 256`
— which reintroduces the **per-tile mean bug** (each of the 8 cores
computes mean over its own 256-element shard rather than the full 2048-
element row; RMS_NORM by definition needs the full-row mean to normalize
correctly).

The existing kernel has no cross-core reduction step, so cannot operate
correctly with tile_size < size. NPU_PLAN already noted this earlier
under "❌ RMSNorm on NPU (tile_size=32 bug)" — fix would need a two-pass
or shared-memory reduction design, which isn't a small change.

**Conclusion on hypothesis #2 confirmation:** cannot validate by simply
reformatting the existing kernel as 8-col. Confirming or refuting #2
requires either:
1. A new IRON design with multi-column reduction (substantial kernel work)
2. Some runtime trick to force the XRT placer to put the 1-col
   RMS_NORM kernel on a column that's not used by QKV (currently the
   hw_context allocator's decision, not directly steerable from host)
3. A 1-col kernel using `cols=1 channels=8 tile_size=256` — still 1-col
   hw_context footprint, doesn't validate column-conflict hypothesis,
   and would need cross-channel reduction anyway

**Current status:** `XDNA_ENABLE_RMS_NORM=0` remains the only known
workaround for chat-mode. Loss is ~0.4 t/s (5.9 → 5.5 single-query).

Pre-fix verification (FlowKV OFF, RMS_NORM ON, QKV ON) gives garbage but
~6 t/s, vs full config 5.9 t/s — so the regression isn't from FlowKV
overhead, it's the RMS_NORM⊕QKV pair.

The prior "FlowKV multi-query bug" framing was misleading: garbage was
present even when FlowKV was disabled. NPU_PLAN earlier sections that
claim "FlowKV works correctly for single queries" remain technically
true, but the multi-query problem is unrelated to FlowKV.

Next steps to localize the actual fault:
- Compare BO addresses and cache keys touched by RMS_NORM vs QKV
  between Q1 first decode and Q2 first decode. If any address collision
  or cache-key reuse appears across operators, that is the smoking gun.
- Audit `static` containers in `xdna_select_rms_norm_params`,
  `get_or_load_rms_norm_kernel`, and `ggml_backend_xdna_mul_mat_qkv`
  for shared state that is not partitioned per-operator.
- Test with `GGML_XDNA_NUM_COLS=4` to rule out the column-set hypothesis.
- Force a `XCL_BO_SYNC_BO_FROM_DEVICE` on RMS_NORM output before any
  subsequent NPU op reads it, to rule out DMA pipeline races.

Added a host-side probe that dumps 8 bf16 values from each of (Q_src, K_src,
V_src, Q_bo, K_bo, V_bo) at positions 0, mid, and last_active, gated on
`kv_h==0` so it fires once per layer per decode. Result for first decode
of Q1 (actual_seq=43, layer 0) vs first decode of Q2 (actual_seq=61, layer 0):

| Probe | Q1 first decode | Q2 first decode | Verdict |
|-------|-----------------|-----------------|---------|
| K_src@0 (BOS K, dim 0..7) | `31E2 2D0B 3146 ...` | `31E2 2D0B 3146 ...` | identical (correct — same BOS) |
| V_src@0 (BOS V, seq 0..7) | `8A86 2E4E 2AE8 ...` | `8A86 2E4E 2AE8 ...` | identical (correct) |
| K_bo@0 / V_bo@0 | identical to Q1 | identical to Q1 | host write preserved data |
| K_src@last (pos 42 / pos 60) | legitimate non-zero | legitimate non-zero | new K written correctly by QKV |
| V_src@last | nonzero then zeros (correct — only positions ≤actual_seq filled) | same pattern | layout consistent |

Cache types: Q=F32, K=F16, V=F16. Strides: nb_K=[2,1024,128], nb_V=[2,1024,65536]
(reflects cache_v underlying max_seq=512 from `-c 512`, viewed as 256-window).

**Three host-data-prep hypotheses are now ruled out:**
1. Stale Q pointer — Q_src looks fresh, Q_bo = bf16-truncate(Q_src) byte-exact
2. Wrong K/V stride/offset — host write matches what's in the source tensor
3. KV cache contamination — position 0 identical Q1/Q2 (correct BOS),
   newly-appended positions are plausible non-zero values

Combined with MATH_DIAG showing NPU output ≈ CPU reference within bf16
precision: the kernel computes correct attention over correct inputs.
**Yet the model produces garbage from Q2.**

Possible remaining causes:
- POC's `kqv_out->data` overwrite lands in the right address but downstream
  CONT (CPU range) doesn't read from there at the chat boundary (state in
  the segment delegation / cpu_run_start tracking)
- Some other op in the cgraph between POC and the next layer reads from a
  different buffer than POC wrote to
- Sampling-stage state corruption (much less likely)

Next probe: dump first 8 bf16 of `kqv_out->data` immediately AFTER POC
finishes the scatter (line ~11797), AND inspect what downstream MUL_MAT
(blk.N.attn_output) reads at its `src[1]` data pointer. If those two
differ, the bug is in the buffer routing between POC and O_proj.
```
```

### Priority 9 SMMU recon (2026-05-23)

Probed current BO alignment on Llama 3.2 1B Q4_0 via `XDNA_DEBUG=1`
+ printf in `mul_mat_gemv_int4` weight cache warm path. Recorded
`weight_bo->address()` and BO size for each unique shape, computed
mod 64 K (SMMU page size on Strix Point).

**Default (`xrt::bo::flags::host_only`):**

| Shape | Weight BO addr mod 64K | Size mod 64K |
|---|---|---:|
| K=2048 N=2048 | **0x03000** | 0 (size = 36 × 64K) |
| K=2048 N=512  | **0x07000** | 0 (size = 9 × 64K)  |
| K=2048 N=8192 | varies     | 0                   |
| K=8192 N=2048 | varies     | 0                   |

Size is always a multiple of 64 K (driver rounds up), but **base
addresses sit on arbitrary 4 K boundaries within the 64 K page**.
So every large weight BO straddles 2 SMMU pages worth of
4 K small-page mappings, plus partial pages at edges. The NPU
DMA fetches across an extra SMMU page boundary on every weight
read, paying TLB pressure.

**Trial: `xrt::bo::flags::cacheable` instead of `host_only`:**

| Shape | Weight BO addr mod 64K | Size mod 64K |
|---|---|---:|
| K=2048 N=2048 | **0x00000** | 0 |
| K=2048 N=512  | 0x08000 (32K) | 0 |

Cacheable flag drops the large BO to **64 K-aligned base** for the
shapes whose total size is exact multiples of 64 K. (Smaller shapes
land on 32 K boundaries — likely an artifact of the cacheable pool's
sub-allocator.)

**But correctness broke immediately.** `paris_short_q4_0_int4_v2`
output went from "The capital of France is Paris." to
"SoundsGGGGGGGG" -- the classic GGGG pattern that means NPU is
reading stale / wrong weight data. Root cause: `cacheable` BO lives
in CPU-cacheable host memory; CPU memcpy lands in CPU cache; our
existing `xrt::bo::sync(XCL_BO_SYNC_BO_TO_DEVICE)` evidently does
not invalidate the relevant cache lines for the cacheable flag (or
does, but the timing relative to NPU read is off).

**Conclusion so far:**

- The 64K-alignment win **does exist** -- `cacheable` flag proves
  the driver can land BOs on 64K boundaries when asked.
- Just flipping the flag isn't safe -- coherence breaks.
- To make the win usable we need one of:
  - explicit CPU cache flush before `sync(TO_DEVICE)` on cacheable
    BOs (need to find the right XRT/AMD API call);
  - a different allocation path that gives 64K alignment without
    cacheable semantics (e.g. `xclAllocUserPtrBO` over a host
    `_aligned_malloc(size, 65536)` buffer, then BO wraps it);
  - or a different flag combination if XRT supports it.

**Code reverted** to keep the tree green. The recon code (SMMU-P9
printf block) was diagnostic-only and removed with the revert.
This is a real lever (potentially 1.5-2× on weight-DMA-bound
shapes; not the 10× the ic ggml-org/llama.cpp#21725 report claimed,
that figure probably bundled multiple optimisations) but needs
more careful XRT-API work to land safely.

**Open question for next session:** find the right XRT idiom for
"give me a 64K-aligned coherent BO" on Windows XDNA -- check
`xrt_bo.h` for additional flags / constructors not used here, or
the IRON-windows reference for how their bench achieves it.

### Priority 9 SMMU follow-up: UserPtrBO doesn't move the device VA (2026-05-23)

Tried path **B** from the SMMU recon section: wrap the weight BO around
a host buffer pre-allocated with `_aligned_malloc(size, 65536)` via
`xrt::bo(device, userptr, size, host_only, group_id)`. Cacheable broke
coherence -- UserPtrBO does NOT, so paris_short_q4_0_int4_v2 stays
byte-exact in both default and `XDNA_BO_ALIGNED_64K=1` modes.

But bench numbers were identical:

| Config | default | XDNA_BO_ALIGNED_64K=1 |
|---|---:|---:|
| NPU INT4 v2 sync | 5.6 | 5.5 |
| NPU INT4 v2 async (Phase 9) | 6.2 | 6.1 |
| NPU INT4 +SwiGLU sync | 5.1 | 5.1 |
| NPU INT4 +SwiGLU async | 5.4 | 5.3 |

All within ±0.1 t/s noise. Investigated by dumping host vs device
addresses for the weight BOs:

```
host_userptr=0x000001F20F3D0000 (64K-aligned, last 16 bits = 0)
dev_addr   =0x0000000000013000 (4K-aligned only, mod 64K = 0x3000)
```

The user-side allocation IS 64K-aligned, but `xrt::bo::address()`
returns the **device-side VA** the NPU sees through SMMU, and it's
still on the same 4K-aligned boundary as in the default
`host_only` path. The XRT/AMD driver re-maps the user buffer
through its own SMMU page tables independently of the host
alignment we provided. Same device VA whether or not we
hand-aligned the host pointer.

**Implication.** True 64K SMMU-page alignment on AMD XDNA Windows
is a **driver-level decision**, not user-controllable from the
xrt::bo API surface we have:

- `host_only` flag: default, 4K-aligned device VA
- `cacheable` flag: 64K-aligned device VA (proven) but breaks
  CPU/NPU cache coherence with our existing sync calls
- User-pointer wrap: zero effect on device VA

To unlock the lever we'd need:

1. A specific XCL ioctl / extension that requests 64K-aligned
   device VA. Public xrt::bo API doesn't expose one. Worth
   checking AMD/Xilinx forums for an undocumented flag.
2. Or use `cacheable` + figure out the right cache-flush API
   (Windows-specific; ic's IRON might document it).
3. Or kernel-side patch to the XDNA driver to allocate from
   a huge-pages-backed pool.

All three are bigger projects than user-mode work. **The lever is
real but out of reach without driver / IRON-level investigation.**
ic's reported 41 t/s likely included this driver-level fix
combined with other IRON-side optimisations -- the ratio is too
large to be alignment alone on our setup.

Code reverted; UserPtrBO experiment removed. The recon and
verify printfs are diagnostic-only and not in the tree. Updated
roadmap: deprioritise Priority 9 until someone identifies the
correct XRT idiom (or upstream IRON publishes their setup).
