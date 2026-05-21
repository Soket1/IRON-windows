# Priority 8: INT4 Quantization — Detailed Plan

> Подплан к `NPU_PLAN.md` Priority 8. Цель: запустить Q4_0/Q4_K GGUF на NPU
> с decode ≥9–12 t/s (от текущих 5.9 t/s; FLM с INT4+fused даёт ~61 t/s).

## TL;DR

В `IRON-windows` уже лежит готовое к работе ядро `fused_dequant_gemv` (INT4×bf16→bf16,
4× DDR-bandwidth reduction). На стороне `llama.cpp-xdna` его никто не вызывает,
а `supports_op` не claim'ит Q4_0/Q4_K. Работа сводится к **host-side repack +
compile.py + dispatch путь**, схема 1:1 копируется с существующего W8A16
для Q8_0 (`xdna_repack_q8_0_to_gemv_int8`, `XDNA_OP_SWIGLU_DECODE_INT8`).

## Что уже есть

| Артефакт | Местоположение | Готовность |
|---|---|---|
| INT4 fused dequant+GEMV kernel | [[file:IRON-windows/aie_kernels/aie2p/fused_dequant_gemv.cc]] | ✅ Тест проходит на 2048×2048×4col×g32 |
| IRON-обёртка `AIEFusedDequantGEMV` | [[file:IRON-windows/iron/operators/fused_dequant_gemv/op.py]] | ✅ MLIR design + arg spec готовы |
| Reference / golden | [[file:IRON-windows/iron/operators/fused_dequant_gemv/reference.py]] | ✅ Описывает точный DDR layout |
| Standalone INT4→bf16 expand | [[file:IRON-windows/iron/operators/dequant]] | ✅ Не нужен для горячего пути, но есть |
| W8A16 host-side repack pattern | [[file:llama.cpp-xdna/ggml/src/ggml-xdna/ggml-xdna.cpp:892-950]] | ✅ Образец для INT4 repack |
| decode_batch (xrt::runlist по shape) | [[file:llama.cpp-xdna/ggml/src/ggml-xdna/ggml-xdna.cpp:4833-5056]] | ✅ Подхватит INT4 без правок |

## Чего нет

1. `compile.py`: `compile_gemv()` отбрасывает всё кроме bf16 ([[file:llama.cpp-xdna/ggml/src/ggml-xdna/compile.py:781-782]]).
2. C++: нет `xdna_repack_q4_0_to_fused_int4()`, нет ветки INT4 в `mul_mat_gemv`/`decode_batcher`.
3. `supports_op`: Q4_0/Q4_K даже не claim'ятся ([[file:llama.cpp-xdna/ggml/src/ggml-xdna/ggml-xdna.cpp:12173-12214]]).
4. IRON: нет `swiglu_decode_int4` (для Phase 8.2).
5. Q4_K: нет ядра с двухуровневыми scales (только Q4_0-совместимый per-group bf16 scale).

## Формат, который ожидает kernel

Из [[file:IRON-windows/aie_kernels/aie2p/fused_dequant_gemv.cc:9-13]] и
`quantize_and_pack` ([[file:IRON-windows/iron/operators/fused_dequant_gemv/reference.py:86-107]]):

- **Значения**: `uint4`, диапазон `[0, 15]`, zero-point фиксирован в 0.
- **Упаковка**: `byte[k] = elem[2k] | (elem[2k+1] << 4)` — соседние элементы.
- **Scales**: один `bfloat16` на группу из `group_size` (кратно 32) элементов.
- **Tile layout** (всё в одном непрерывном uint8 буфере):
  ```
  [m_input * K / 2 bytes]            packed uint4 weights
  [m_input * (K / group_size) * 2 bytes] bf16 scales
  ```
- **Tile order в DDR**: column 0 tiles 0..T-1, потом column 1, ... — ровно как
  в `xdna_repack_q8_0_to_gemv_int8`.

Соответствие GGML формату:

| Параметр | GGML Q4_0 | IRON kernel |
|---|---|---|
| Знак | signed `[-8..7]` | unsigned `[0..15]` |
| Packing | byte j = (elem[j] & 0xF) \| ((elem[j+16] & 0xF) << 4) — **интерливленный** | byte k = elem[2k] \| (elem[2k+1] << 4) — **линейный** |
| Scale | `ggml_fp16_t d`, один на блок из 32 | `bfloat16 sf`, один на группу (≥32) |
| Zero-point | implicit `−8` после распаковки | implicit `0` |

→ Host repack должен: (1) развернуть нибблы из интерливленного в линейный
порядок, (2) прибавить `+8` к каждому нибблу (компенсация знака), (3)
вычислить bias-term для GEMV: `−8·scale·Σactivation_per_group`, либо изменить
kernel (subtract-after-unpack).

## Phase-разбивка

### Phase 8.0 — Валидация (0.5 дня)

**Цель**: убедиться что готовое ядро работает на текущем железе (STX NPU2) и
оценить latency.

1. `source /opt/xilinx/xrt/setup.sh && source ironenv/bin/activate`
2. `pytest iron/operators/fused_dequant_gemv/ --iterations 1` — extensive=false (2048×2048, 4col, g32).
3. Снять `Latency (us)` из метрик теста, сравнить с bf16 GEMV того же размера
   (`pytest iron/operators/gemv/` 2048×2048×4col).
4. **Acceptance**: kernel проходит с `rel_tol=0.07`, INT4 latency ≤ 0.5×bf16
   latency (ожидание 0.25× при memory-bound, 0.5× если есть compute хвост).

### Phase 8.1 — Q4_0 GEMV на NPU, W4A16 (3–5 дней)

**Status (2026-05-21): ✅ DONE.** End-to-end INT4 dispatch path lands
correctly and matches `cpu_baseline` byte-exact on three regression tests
(see `correctness_test.py`).

| Sub-step | Status | Where |
|---|---|---|
| 8.1.1 `compile.py` `fused-dequant-gemv` subcommand + cache key | ✅ DONE | `compile.py` lines ~97-112, ~825-870, ~1222-1237 |
| 8.1.2 `XDNA_OP_GEMV_INT4 = 10` enum entry | ✅ DONE | `ggml-xdna.cpp` line ~93 |
| 8.1.3 `xdna_repack_q4_0_to_fused_int4()` host-side repack | ✅ DONE | `ggml-xdna.cpp` lines ~1023-1091 |
| 8.1.4 Weight BO caching (keyed by `src0->data`) | ✅ DONE | `mul_mat_gemv_int4` lazy alloc |
| 8.1.5 Dispatch path in `mul_mat_gemv` | ✅ DONE | helper `mul_mat_gemv_int4` (~180 LOC) |
| 8.1.6 `supports_op` claim for Q4_0 under `XDNA_ENABLE_GEMV_INT4` | ✅ DONE | gated default-off |
| 8.1.7 `decode_batch` integration | ✅ DONE (excluded) | Q4_0 skipped, routes through bare `mul_mat_gemv` |
| 8.1.8 End-to-end byte-exact match vs CPU baseline | ✅ DONE | 3 tests pass, see below |
| Host-side bias compensation math | ✅ DONE | `bias[i] = 8 * sum_g sf[i,g] * S[g]` |
| `select_gemv_tiles` mirror in C++ (`tile_in` may be 1, 2, 4, or 8) | ✅ DONE | `xdna_select_gemv_tiles_int4()` |
| Kernel-side `aie::sub(8)` (review note N2) | ⏳ FUTURE | optimization; would remove the host bias step |

**Root-cause of the original "GGGG" output bug:** `mul_mat_gemv_int4`
hardcoded `m_input = 1` in the repack and dispatch sizing, but
`compile.py:select_gemv_tiles` picks `tile_in` based on the 64 KB L1
budget (typically 4, sometimes 1 for the K=8192 shape, capped at 8).
The packed buffer layout the kernel reads (rows-per-tile interleaved
with their scales) did not match what we wrote, so the kernel produced
inf/nan outputs. Fix: mirror `select_gemv_tiles` in C++ via
`xdna_select_gemv_tiles_int4()`, use the selected `tile_in` everywhere
(repack, BO sizing, bias scale lookup).

**N3 acceptance — correctness_test.py integration:** the harness now
ships a `npu_int4` preset (chat-safe + `XDNA_ENABLE_GEMV_INT4=1`) and
three INT4 tests:

| Test | Mode | n_predict | Result vs `cpu_baseline` |
|---|---|---|---|
| `paris_short_q4_0_int4` | single-turn | 12 | byte-exact |
| `paris_drift_64_q4_0_int4` | single-turn | 64 | byte-exact (256+ chars) — implicitly stresses the `tile_in=1` ffn_down path via ~1000 dispatches per run |
| `multiquery_q4_0_int4` | chat (2 turns) | 24 | byte-exact on both turns — covers INT4 + FlowKV decode + chat-mode composition |

**Pyxrt correction:** the pyxrt ABI mismatch that blocks the IRON Python
pytest does **NOT** block ggml-xdna's compile pipeline. ggml-xdna invokes
`compile.py` via `system()` / `cmd.exe`, which uses a different DLL
search path than the conda Python interpreter loading pyxrt directly.

**Behavior when enabling `XDNA_ENABLE_GEMV_INT4=1`:**

| Aspect | Status |
|---|---|
| Build | ✅ clean |
| Kernel compile (compile.py) | ✅ produces xclbins per (K, N) shape, e.g. `fused_dequant_gemv_8192x2048_4tsi_1024tso_8col_g32.xclbin` |
| xclbin load + hw_ctx + xrt::kernel | ✅ no errors |
| INT4 weight repack | ✅ runs on first-touch per src0->data, caches |
| BO allocation / DMA sync | ✅ no errors |
| Kernel dispatch (xrt::run + wait) | ✅ completes without crash |
| Output values vs CPU | ✅ byte-exact (3/3 tests) |

**Throughput measurement (2026-05-22, llama-3.2-1B-Instruct, n_predict=64,
`--bench` mode of `correctness_test.py`, median of 2 runs):**

| Config | single-turn | chat (-cnv) |
|---|---:|---:|
| CPU Q4_0 (`cpu_baseline`) | **10.70** | 10.30 |
| NPU bf16 (`npu_chat_safe`, BF16 model) | 5.20 | 5.20 |
| NPU INT4 (`npu_int4`, Q4_0 model) | **3.30** | **3.40** |

(decode t/s; prompt-eval is roughly 1.3-1.6× of decode and tracks the same pattern.)

Single-turn vs chat-mode results are within ±0.2 t/s noise — the
fusion-bypass effect is mode-independent (per-token compute is the same).

**Critical finding: Phase 8.1 alone is a net regression.** NPU INT4 is
~36 % slower than NPU bf16 and ~3 × slower than CPU Q4_0. The root cause
is that turning on `GEMV_INT4` disables the existing NPU fusion path for
Q4_0 weights — they bypass `decode_batch`, SwiGLU, and the
transformer-block xclbin (those operators are bf16-only today), so each
Q4_0 matmul becomes an individual `xrt::run` dispatch with full DMA
round-trip overhead. The 4× DDR-bandwidth saving from INT4 packing
cannot compensate for losing batched dispatch + kernel fusion.

**Implication for the roadmap:** Phase 8.2 (SwiGLU INT4) and/or 8.3 (QKV
INT4) are now **required** for INT4 to pay off, not "nice to have".
Until then, `XDNA_ENABLE_GEMV_INT4=1` should remain default-off.

**Open items (deferred to future phases):**

1. **Phase 8.2 SwiGLU INT4 (HIGH PRIORITY)** — restores fusion on the
   FFN path; without this the INT4 dispatch path is a regression.
2. **N2 kernel-side `aie::sub(8)`** — replace the host bias compensation
   with an in-kernel offset. Removes ~50 µs/matmul of CPU work.
3. **Q4_K_M re-quantize → W4A16 path** (Phase 8.4).

**Original plan text below preserved for the file-by-file detail:**

---

(original 8.1 plan continues from here ↓)

### Phase 8.1 (original detail) — Q4_0 GEMV на NPU, W4A16 (3–5 дней)

**Цель**: dispatch Q4_0 `mul_mat` с M=1 на NPU через fused_dequant_gemv.
Эффект: O_proj decode (12×0.56ms) + bare-MM decode → ~3× быстрее на этих
узлах, ожидаемо `5.9 → 7.0 t/s`.

#### 8.1.1 `compile.py`

- Добавить `FUSED_DEQUANT_GEMV_KERNEL = ("fused_dequant_gemv",)` (1-kernel xclbin).
- Добавить `fused_dequant_gemv_cache_key(M, K, num_aie_columns, group_size)`.
- Добавить `compile_fused_dequant_gemv(M, K, num_aie_columns, group_size, output_path)`
  — зеркало `compile_gemv` но с `from iron.operators.fused_dequant_gemv.op
  import AIEFusedDequantGEMV` и `tile_size_input=1`, `tile_size_output=M//cols`.
- Добавить CLI subcommand `fused_dequant_gemv` в `__main__`.
- Использовать существующий select для `tile_size_*` (есть assert в `AIEFusedDequantGEMV.__init__`).

#### 8.1.2 `ensure_compiled` / `get_or_load_kernel`

- Новый `XDNA_OP_GEMV_INT4 = 6` в enum.
- Кэш-ключ: `gemv_int4_K{K}_N{N}_{cols}col_g{group_size}`.
- В `make_cache_key` ветка с `op=XDNA_OP_GEMV_INT4`.

#### 8.1.3 Host-side repack Q4_0

Новая функция в `ggml-xdna.cpp` после `xdna_repack_q8_0_to_gemv_int8`
([[file:llama.cpp-xdna/ggml/src/ggml-xdna/ggml-xdna.cpp:892]]):

```cpp
static void xdna_repack_q4_0_to_fused_int4(
        const uint8_t * q4_0,          // GGML buffer: N rows × (K/32) × 18 bytes
        int64_t M, int64_t K,
        int m_input, int cols, int group_size,
        uint8_t * packed_out) {
    GGML_ASSERT(group_size == 32);     // Q4_0 фиксирует blocks по 32
    // ...
    // Per row:
    //   for each 32-elem block (16 bytes qs):
    //     unpack 32 signed nibbles -> sub(-8) -> +8 makes [0..15]
    //     repack into 16 bytes IRON-style: byte[k] = elem[2k] | (elem[2k+1]<<4)
    //   convert fp16 d -> bf16 (truncate low 16 bits of fp32(d))
    //   write to packed_out at the tile_offset that fused_dequant_gemv expects
}
```

Псевдокод репака одного 32-элементного блока (Q4_0 `qs[16]` → IRON 16 bytes):

```cpp
// Развернуть нибблы из (j, j+16) в (2k, 2k+1)
uint8_t e[32];
for (int j = 0; j < 16; j++) {
    e[j]      = (qs[j] & 0x0F);          // [-8..7] стало [0..15] без shift,
    e[j + 16] = (qs[j] >>   4) & 0x0F;   // эквивалентно сложить +8 от подписанного
}
// Записать линейно
for (int k = 0; k < 16; k++)
    iron_byte[k] = e[2*k] | (e[2*k + 1] << 4);
```

Q4_0 хранит `int4 = (nibble - 8)`. Сделав репак **без вычитания −8** мы
автоматически получаем `uint4 = signed + 8 ∈ [0..15]`. Значит, GEMV вычислит:
```
ŷ[i] = Σ_g scale[g] · Σ_k (q_int4[g,k] + 8) · x[k_in_g]
     = true_ŷ[i] + Σ_g scale[g] · 8 · Σ_k x[k_in_g]
```
Compensation term (per output row, **тот же для всех row** только если scales
не зависят от row — а они зависят, поэтому считаем per row):
```
bias[i] = 8 · Σ_g (scale[i,g] · Σ_{k ∈ g} x[k])
```
- Считается на CPU после GEMV (или внутри fallback пути дешёвой ALU).
- `Σ_{k ∈ g} x[k]` — посчитать один раз за токен для всех rows одного weight'а.
  Это `K/group_size` сумм по 32 элемента = 64 FMA для K=2048. Бесплатно.
- Затем для каждого row: `bias[i] = 8 · Σ_g sf[i,g] · S[g]`. Стоит `(K/G)·M
  ≈ 64·N` FMA = 130 K FMA для N=2048; на CPU ~50 µs. Дешевле чем DMA-туда-обратно.

**Альтернатива** (рекомендация если хочется чистого решения): добавить
`-DQ4_SIGNED=1` flag в kernel compile и в `fused_dequant_matvec` после `aie::unpack(I0)`
вставить:
```cpp
#ifdef Q4_SIGNED
aie::vector<int8, block_size> as_int8s =
    aie::sub(as_int8.cast_to<int8>(), aie::broadcast<int8, block_size>(8));
#endif
```
Один вектор-subtract на блок (32 lanes за такт). Bias-compensation на хосте
тогда не нужен. Ставлю это в **рекомендацию для production**; вариант с host
bias оставляем как PoC шаг 1.

#### 8.1.4 Weight BO кэширование

Зеркало `swiglu_warm_weight_int8` ([[file:llama.cpp-xdna/ggml/src/ggml-xdna/ggml-xdna.cpp:2390-2447]]):

- `entry->b_bo_cache` keyed по `src0->data` (immutable после model load).
- На first-touch: allocate `_packed_buffer_size()` байт `xrt::bo`, вызвать
  `xdna_repack_q4_0_to_fused_int4` прямо в `bo.map<void*>()`, `bo.sync(TO_DEVICE)`.
- На следующий dispatch: переиспользовать.
- Память: `M·K/2 + M·(K/32)·2 = 0.5·M·K + M·K/16 = 0.5625·M·K` против `2·M·K` для bf16. Для O_proj (2048²): **2.25 MB вместо 8 MB**.

#### 8.1.5 Dispatch путь

Расширить `ggml_backend_xdna_mul_mat_gemv` ([[file:llama.cpp-xdna/ggml/src/ggml-xdna/ggml-xdna.cpp:1405]]):

```cpp
const bool use_int4 = (src0->type == GGML_TYPE_Q4_0);
if (use_int4) {
    int group_size = 32;
    cache_key = make_cache_key(XDNA_OP_GEMV_INT4, 1, K, N, "uint4", num_cols, group_size);
    ensure_compiled(...);  // → compile_fused_dequant_gemv
    entry = get_or_load_kernel(...);

    // a_bo: packed_weights uint8 (per-data-ptr cached)
    // b_bo: vector bf16 (per-call)
    // c_bo: output bf16 (per-call)
    // arg order: kernel(opcode=3, insts, n_insts, packed_weights, vector, output)
    // → AIEFusedDequantGEMV.set_up_runtime() кладёт arg_id 3,4,5 в этом порядке
}
```

`opcode = 3` тот же что для GEMV (см. [[file:llama.cpp-xdna/ggml/src/ggml-xdna/ggml-xdna.cpp:1482]]).

#### 8.1.6 `supports_op`

В `ggml_backend_xdna_device_supports_op` ([[file:llama.cpp-xdna/ggml/src/ggml-xdna/ggml-xdna.cpp:12173]]):

```cpp
static const bool int4_ok = xdna_env_enabled("XDNA_ENABLE_GEMV_INT4");
if (int4_ok && src0->type == GGML_TYPE_Q4_0) return true;
```

Под gate'ом — поведение симметрично `int8_ok` чтобы не ломать графовую сегментацию.

#### 8.1.7 `decode_batch` integration

В `xdna_plan_decode_batch` ([[file:llama.cpp-xdna/ggml/src/ggml-xdna/ggml-xdna.cpp:5061]]) добавить:
- Включить `GGML_TYPE_Q4_0` weight в eligible set, если `int4_ok`.
- Сгруппировать по `(K, N, src0->type)` — INT4 и bf16 имеют разные xclbin → разные `hw_ctx`, нельзя в одном runlist.

В `decode_batcher::flush` ([[file:llama.cpp-xdna/ggml/src/ggml-xdna/ggml-xdna.cpp:4855]]) при batch_n≥2 INT4 GEMV'ов одной формы:
- Сделать один `xrt::runlist`.
- Per-item `in_bos` (bf16 vector), `out_bos` (bf16 output).
- `w_ptrs` тянутся из кэша `entry->b_bo_cache` (уже repacked).
- Те же 6 args что для bf16 GEMV.

#### 8.1.8 Тестирование

1. **Unit-тест repack**: пройтись по 100 случайным rows × 32 элементов, сравнить
   результат GEMV на NPU с `dequantize_row_q4_0 + ref GEMV` на CPU, `abs_tol=0.05`.
2. **End-to-end**: `llama-cli -m models/llama-3.2-1b-instruct-q4_0.gguf -p "The capital of France is"`,
   ожидаемый ответ "Paris". На двух конфигах:
   - `XDNA_ENABLE_GEMV_INT4=0` (baseline через CPU dequant) — sanity.
   - `XDNA_ENABLE_GEMV_INT4=1` (NPU) — accuracy.
3. **Бенчмарк**: 32 токена decode, ожидание ≥7 t/s (1.2× от 5.9 t/s).

### Phase 8.2 — INT4 SwiGLU FFN (3–4 дня)

**Цель**: вынести `swiglu_decode` на INT4, забрать 30+ ms/token. Это самая
дорогая операция (3.62 ms × 12 = 43 ms/token).

#### 8.2.1 IRON: `swiglu_decode_int4`

Создать `iron/operators/swiglu_decode_int4/{op.py, design.py, reference.py, test.py}`
по шаблону `swiglu_decode/` но с `AIEFusedDequantGEMV` вместо GEMV:

- 4-kernel xclbin: `fused_dequant_gemv_1 → silu → eltwise_mul → fused_dequant_gemv_2`.
- Использовать `chain_swiglu_artifacts` ([[file:IRON-windows/iron/operators/swiglu_base.py:10]]).
- Промежуточные bf16 буфера в L2.

#### 8.2.2 `compile.py`

- `compile_swiglu_decode_int4(embedding_dim, hidden_dim, num_aie_columns,
  output_dir, group_size=32)` — зеркало `compile_swiglu_decode_int8` но c
  IRON `AIESwiGLUDecodeInt4` и `validate_swiglu_decode_int4_shapes`.

#### 8.2.3 Host-side: `ggml_backend_xdna_mul_mat_swiglu_int4`

Зеркало `_swiglu_int8` ([[file:llama.cpp-xdna/ggml/src/ggml-xdna/ggml-xdna.cpp:2458]]):
- Три INT4 weight BO: gate, up, down. Repack один раз через `xdna_repack_q4_0_to_fused_int4`.
- В качестве `m_input` для каждого GEMV: clamp по L1 budget ([[file:llama.cpp-xdna/ggml/src/ggml-xdna/ggml-xdna.cpp:2492-2495]]).
- Gate: `XDNA_ENABLE_SWIGLU_INT4` (новый env var).

#### 8.2.4 Matcher

В `ggml_backend_xdna_swiglu_match` ([[file:llama.cpp-xdna/ggml/src/ggml-xdna/ggml-xdna.cpp:5210]]) расширить:
- `all_q4_0 = (gate_w->type == GGML_TYPE_Q4_0) && (up_w->type == GGML_TYPE_Q4_0) && (down_w->type == GGML_TYPE_Q4_0)`
- `allow_int4 = (int4_swiglu_enabled) && all_q4_0 && shape_dispatchable`
- Disjoint от int8 пути (mutually exclusive).

#### 8.2.5 Acceptance

- 12 слоёв × ~1 ms = **12 ms/token вместо 43 ms/token**.
- Ожидаемо: `5.9 → 9–10 t/s` (предполагая что не упёрлись в L1).

### Phase 8.3 — INT4 QKV (1–2 дня, опционально)

QKV — три GEMV с общим input, разные output. На текущем bf16 пути это один
fused xclbin ([[file:llama.cpp-xdna/ggml/src/ggml-xdna/ggml-xdna.cpp:5955]]) с
1.17 ms на dispatch. Переход на INT4:

- Либо: переиспользовать тот же matcher что в `mul_mat_swiglu_int4`, собрать 3
  отдельных INT4 GEMV.
- Либо: новый `fused_qkv_int4` xclbin (3 параллельных `fused_dequant_gemv` worker'а).

ROI оценить **после Phase 8.2** замером. Если QKV доминирует на профайле —
строить. Иначе не трогать.

### Phase 8.4 — Q4_K support (3–5 дней)

Q4_K сложнее: super-block 256 элементов, 8 sub-blocks по 32, 6-битные scales
и mins ([[file:llama.cpp-xdna/ggml/src/ggml-common.h:317-328]]):

```
[fp16 d][fp16 dmin][12 bytes scales/mins][128 bytes qs]
```

Каждый элемент: `q · sub_scale · d − sub_min · dmin` где sub_scale, sub_min — 6-битные.

**Вариант A (рекомендация)**: host-side full dequant → re-quantize в W4A16
per-group-32:
1. `dequantize_row_q4_K(src0->data, fp32_buf, row_size)`.
2. Для каждой группы 32 элементов: compute `scale = max(|x|) / 7.5` (или MSE),
   re-quantize в `uint4 = round(x / scale + 8)`, clamp `[0..15]`.
3. Сохранить в IRON-формате, scale в bf16.

Потеря качества <0.5% perplexity (то же делает Quark когда конвертирует W4A16
checkpoint в W4A8, [см. блог про Kimi K2.5](https://rocm.blogs.amd.com/artificial-intelligence/kimi-k2.5-w4a8/)).
Прохождение по весам один раз при first-touch — амортизируется.

**Вариант B**: native Q4_K kernel с двухуровневыми scales. Существенно больше
работы, не критично для PoC. Пропустить до Phase 8.5+.

### Phase 8.5 — W4A8 (опционально, 5+ дней)

Если W4A16 упёрлось в bf16 MAC throughput (а не в memory): перейти на
INT4-веса × INT8-активации, INT8 MFMA (см. ROCm Kimi K2.5 блог).

- Новое ядро: `fused_dequant_gemv_int8.cc` — unpack uint4→int8 (не →bf16),
  затем `aie::mac` int8×int8→i32, scale-apply в эпилоге.
- Активации квантуются per-token (уже есть `xdna_quantize_bf16_to_int8` на
  [[file:llama.cpp-xdna/ggml/src/ggml-xdna/ggml-xdna.cpp:1033]]).
- Theoretical: 2× от W4A16 если compute-bound.

**Гейт**: запускать только если Phase 8.2 даёт <10 t/s и профайл показывает
compute-bound.

## Риски и митигации

| Риск | Митигация |
|---|---|
| Repack ошибка (nibble ordering Q4_0) | Unit-тест на 100 rows vs `dequantize_row_q4_0`, abs_tol=1e-3 |
| Bias compensation overhead > выигрыш | Перейти на kernel вариант с `aie::sub(8)` если профайл показывает >5% CPU bias-cost |
| L1 budget overflow для SwiGLU INT4 | Те же tile_size_input clamp правила что и в W8A16 (уже отлажены) |
| Cache collision bf16/INT4 | Разные cache keys → разные xclbin → разные `b_bo_cache` per entry (изоляция гарантирована) |
| `supports_op` claim ломает сегментацию | Поведение зеркалит Q8_0 (gated via env) — побочных эффектов на bf16 пути нет |
| GGUF Q4_K layout edge cases | Phase 8.4 идёт **через CPU dequant→reroute → W4A16 repack**, обходит всю Q4_K специфику |
| AIE2/AIE2P ядра разные | В `fused_dequant_gemv` есть оба: aie2/aie2p/`fused_dequant_gemv.cc` |

## Изменения в файлах (Phase 8.1 + 8.2)

| Файл | Тип | Объём |
|---|---|---|
| `ggml/src/ggml-xdna/ggml-xdna.cpp` | + repack Q4_0, + mul_mat_gemv ветка INT4, + supports_op, + swiglu_int4 dispatch, + decode_batch INT4 | ~400 строк |
| `ggml/src/ggml-xdna/compile.py` | + `compile_fused_dequant_gemv`, + `compile_swiglu_decode_int4`, + cache key functions | ~150 строк |
| `iron/operators/swiglu_decode_int4/` | новый каталог: `op.py`, `design.py`, `reference.py`, `test.py` | ~400 строк |
| `aie_kernels/aie2p/fused_dequant_gemv.cc` | опционально: +`-DQ4_SIGNED=1` ветка с `aie::sub(8)` | ~10 строк |
| `ggml/src/ggml-xdna/tests/test_gemv_int4.py` | новый: золотой тест Q4_0 GEMV против CPU | ~100 строк |

## Roadmap

| Phase | Дни P50 | Дни P90 | Выход | t/s | Status |
|---|---|---|---|---|---|
| 8.0 Validation | 0.5 | 1 | INT4 kernel works on STX | 5.9 (baseline) | ⚪ partial (pyxrt blocked for pytest; compile.py via system() works) |
| 8.1 Q4_0 GEMV scaffolding (enum + supports_op + repack + compile.py) | landed | landed | safe default-off scaffolding | 5.9 (unchanged) | ✅ commit 35ae3fee1 |
| 8.1 Q4_0 GEMV dispatch path (BO + kernel + bias + tile_in selector) | landed | landed | NPU dispatch for bare Q4_0 mul_mat, byte-exact vs CPU on 3 regression tests | **decode 3.40 t/s** measured (regression vs NPU bf16 5.30 t/s -- fusion lost) | ✅ DONE (2026-05-21) |
| 8.2 Q4_0 SwiGLU FFN | 3–4 | 8–10 | FFN on INT4, restores fusion -- needed to net-positive on 8.1 | target ≥9–10 | **HIGH PRIORITY** -- 8.1 alone regresses |
| 8.3 QKV INT4 | 1–2 (or skip) | 3 | only if 8.2 profile shows QKV dominates | ~10–11 | measurement-gated |
| 8.4 Q4_K (via W4A16 repack) | 3–5 | 7 | support for Q4_K GGUF | no perf delta | not started |
| 8.5 W4A8 | 5+ | 10+ | INT8 MFMA if bf16 MAC bound | maybe 12–15 | not started |
| **Realistic total to target** | **2 weeks** | **3–4 weeks** | Q4_0 + Q4_K on NPU | **5.9 → 9–10 t/s** | |

## Что делегировать Build агенту

Естественно стэкуется (Phase 8.1 трогает те же файлы что 8.2):

1. **Task 1** (parallel-safe): Phase 8.0 валидация + Phase 8.1 wire-up (compile.py + repack + gemv dispatch + supports_op + unit test). PR в `llama.cpp-xdna`.
2. **Task 2** (stacked on PR 1): Phase 8.2 SwiGLU INT4 — новый IRON op в `IRON-windows`, host swiglu dispatch в `llama.cpp-xdna`. Двух-репный PR.
3. Phase 8.3/8.4/8.5 — отдельные таски после замера на 8.2.

## Prerequisites validation log (2026-05-21)

Status of the pre-flight steps before any Phase 8.x coding can start.

### Q4_0 / Q4_K_M GGUF — DONE

```powershell
cd C:\llama.cpp-xdna
.\build\bin\Release\llama-quantize.exe `
    models\llama-3.2-1b-instruct-BF16.gguf `
    models\llama-3.2-1b-instruct-Q4_0.gguf q4_0
.\build\bin\Release\llama-quantize.exe `
    models\llama-3.2-1b-instruct-BF16.gguf `
    models\llama-3.2-1b-instruct-Q4_K_M.gguf q4_k_m
```

| File | Size | BPW | Sanity (CPU-only, --temp 0 -s 42) |
|---|---|---|---|
| llama-3.2-1b-instruct-BF16.gguf | 2.4 GB | 16.00 | "The capital of France is Paris." |
| llama-3.2-1b-instruct-Q4_0.gguf | 736 MB | 4.94 | "The capital of France is Paris." (13.1 t/s) |
| llama-3.2-1b-instruct-Q4_K_M.gguf | 771 MB | 5.18 | "The capital of France is Paris." |

All three produce byte-identical responses on the Paris test, so the
re-quantizations are correct and ready for NPU acceleration comparison.

### IRON pytest on Windows — BLOCKED

Attempted N1 (run `pytest iron/operators/fused_dequant_gemv/` on Windows
to validate the kernel before any wire-up). Result:

```
ImportError: DLL load failed while importing pyxrt
```

`pyxrt.pyd` at
`C:\Users\Kuhnya\Downloads\xrt_windows_sdk\xrt_sdk\xrt\python\pyxrt.pyd`
fails to load under the conda env's Python 3.12. Adding PATH to driver
directories (`kipudrv.inf_amd64_*`, `System32`) and `os.add_dll_directory`
did not help — the binding is built against a different Python ABI
(or has unresolved external DLL deps we don't have here).

Pre-built XRT Python binding is only present at one location on this
system; there is no Python 3.12-compatible version. Source is available
at `XRT-202610.2.23.0_Canonical/src/python/pybind11/`, but rebuilding
requires the full MSVC + CMake + Python dev headers toolchain and
takes ~30-60 min of one-time setup. Out of scope for this session.

**Implication for Phase 8.0:**

The kernel `fused_dequant_gemv` is verified by IRON contributors elsewhere
(they ship the test). We cannot independently re-verify on this Windows
box without the pyxrt fix. Two paths forward:

1. **Trust the upstream test.** The kernel is already present at
   `aie_kernels/aie2p/fused_dequant_gemv.cc` and tested in CI by IRON
   maintainers. Proceed to Phase 8.1 wire-up without local validation;
   treat any post-landing dispatch failure as a kernel/host integration
   issue, not a kernel correctness issue.
2. **Run validation on a Linux/WSL box** with a working `pyxrt`.
   Recommended if available — gives a clean confirmation before
   spending 3-5 days on Phase 8.1.

For this codebase's day-to-day workflow, path 1 is fine — the existing
W8A16 (Q8_0) integration was wired up the same way without local IRON
pytest, and it works.

### Updated Phase 8.0 status

| Step | Status | Notes |
|---|---|---|
| Q4_0 GGUF generated | ✅ | `models/llama-3.2-1b-instruct-Q4_0.gguf` |
| Q4_K_M GGUF generated | ✅ | `models/llama-3.2-1b-instruct-Q4_K_M.gguf` |
| llama-quantize.exe present | ✅ | already built |
| IRON pytest runs | ❌ | pyxrt blocker; trust upstream OR run on Linux |
| Compile-only check (kernel xclbin produces) | not attempted | requires Python entry-point that doesn't import iron.common |

**Recommendation**: skip the IRON pytest step, proceed directly to
Phase 8.1 wire-up using `correctness_test.py` for end-to-end validation
on the Windows host (NPU-on vs CPU-baseline comparison on the Q4_0
model, using a min_prefix_match tolerance for bf16 vs INT4 drift).

---

## Review notes (added 2026-05-21)

External review of this plan after verification of all file:line claims
against the actual code. Plan is grounded in reality (all referenced
artifacts exist and code locations are accurate within a few lines), but
the following points need attention before execution:

### N1. Phase 8.0 instructions are Linux-only

```bash
source /opt/xilinx/xrt/setup.sh && source ironenv/bin/activate
pytest iron/operators/fused_dequant_gemv/ --iterations 1
```

Target machine is Windows STX NPU2. `/opt/xilinx/xrt/setup.sh` does not
exist there. Two options:

- **A**: run the validation step in WSL/Linux on a separate XDNA-capable
  box, just to confirm the kernel compiles and passes its test. The actual
  Phase 8.1 wire-up still happens on Windows.
- **B**: replicate the test on Windows via the existing `ryzen-ai-1.7.1`
  conda env with `mlir_aie` python bindings already installed. Requires
  Win-equivalent of `setup.sh` (just `set XRT_BIN_DIR=...` etc., we already
  do this in `correctness_test.py`'s `BASE_ENV`).

**Action**: rewrite section 8.0 with Windows-first instructions, drop
the Linux `source` lines or move them to a "Linux dev box (optional)"
sub-section.

### N2. Bias-compensation overhead understated

Plan says "host bias ~50 µs/matmul on CPU, dешевле чем DMA-туда-обратно".
Correct per single matmul, but doesn't multiply over the full token:

- 16 layers × (3 QKV + 1 O_proj + 3 SwiGLU matmuls) = ~80 matmuls/token
- 80 × 50 µs = ~4 ms/token CPU overhead
- 4 ms / 170 ms total token time = ~2.3% — acceptable, but not free

The kernel-side `aie::sub(8)` after `aie::unpack` is the correct
long-term solution (one vector subtract per block = essentially free on
AIE). Plan currently recommends host bias as "PoC step 1" and kernel
fix as "production recommendation". This ordering risks landing the
host-bias PoC and never coming back to the kernel fix.

**Action**: do the kernel `aie::sub(8)` from day 1. Skip the host-bias
intermediate step entirely unless the kernel change is blocked by IRON
compile issues. Update Phase 8.1.3 to land both `xdna_repack_q4_0` AND
the kernel `-DQ4_SIGNED=1` flag together.

### N3. No correctness harness integration

We landed `ggml/src/ggml-xdna/tools/correctness_test.py` (commit 7b2231eeb)
which provides regression testing for NPU vs CPU output. The plan
should integrate INT4 cases into it:

- **Phase 8.1 acceptance**: add test `paris_short_q4_0` that loads the
  Q4_0 model. Expected: identical to bf16 baseline within
  `min_prefix_match` tolerance. Mark `expected_fail` for npu_chat_safe
  while INT4 path is not landed yet; flip to expected_pass when 8.1
  ships.
- **Phase 8.2 acceptance**: add `multiquery_q4_0_chatsafe` that exercises
  INT4 SwiGLU through chat-mode multi-query (to catch any new
  RMS-NORM-like interference between INT4 paths and others).

**Action**: append a step to each phase's acceptance criteria:
"Add test case to `correctness_test.py`; expected_fail before landing,
flip on landing."

### N4. Q4_K perplexity claim is unsupported

> "Потеря качества <0.5% perplexity (то же делает Quark когда конвертирует W4A16
> checkpoint в W4A8)"

The Kimi K2.5 ROCm blog discusses **W4A8 serving on GPU**, not Q4_K→W4A16
re-quantize on NPU. The "<0.5%" figure is borrowed, not measured.

**Action**: Phase 8.4 acceptance must include an empirical measurement:

```
llama-perplexity -m model.q4_K.gguf [CPU baseline]
llama-perplexity -m model.q4_K.gguf [NPU with W4A16 repack]
diff |should be <1% on wikitext-2|
```

If delta is >1%, fall back to native Q4_K kernel (Option B), or skip Q4_K
support entirely.

### N5. Roadmap timing optimistic, no P90

"~2 weeks total" assumes everything works first try. Realistic estimate
should include debug cycles, especially for Phase 8.2 (new IRON op with
multi-kernel chain, MLIR-AIE compiles ~10 min/iter).

| Phase | Days P50 | Days P90 | Notes |
|---|---|---|---|
| 8.0 | 0.5 | 1 | Just running pytest; OS issue may need workaround |
| 8.1 | 3–5 | 7–8 | Host repack edge cases + cache invalidation |
| 8.2 | 3–4 | 8–10 | New IRON op, intermediate L2 buffers, MLIR debug |
| 8.3 | 1–2 | 3 (or skip) | May negative-ROI; see N8 below |
| 8.4 | 3–5 | 7 | Re-quantize math + perplexity validation |
| 8.5 | 5+ | 10+ | INT8 MFMA on AIE2P needs separate kernel design |
| **Total** | **2 weeks** | **3–4 weeks** | Realistic with debug |

**Action**: update Roadmap table with both columns.

### N6. Missing prerequisite: Q4_0 GGUF file

Phase 8.1.8 references `models/llama-3.2-1b-instruct-q4_0.gguf` for the
end-to-end test. This file does not exist in `models/`. Plan should
add an explicit prerequisite step:

```
cd C:\llama.cpp-xdna
.\build\bin\Release\llama-quantize.exe \
    models\llama-3.2-1b-instruct-BF16.gguf \
    models\llama-3.2-1b-instruct-q4_0.gguf \
    q4_0
```

Takes ~1 min, output ~750 MB. Same for Q4_K in Phase 8.4.

**Action**: add a "Prerequisites" section before Phase 8.0 with the
quantize commands.

### N7. W4A8 (Phase 8.5) cites GPU material

> "см. ROCm Kimi K2.5 блог"

ROCm = AMD GPU. AIE-2P MAC instructions are different (vector intrinsics
`aie::mac`, not MFMA). The conceptual gain (INT8 MAC > bf16 MAC for
memory-bound ops) carries over but specific implementation details
don't.

**Action**: rewrite Phase 8.5 around concrete AIE evidence. Step 1:
write a microbenchmark in `aie_kernels/aie2p/bench/` comparing
`aie::mac<int8>` vs `aie::mac<bf16>` throughput on AIE-2P. If <2× speedup,
W4A8 isn't worth the kernel complexity. Don't presume from GPU
literature.

### N8. Phase 8.3 ROI is likely negative — flag explicitly

Current QKV path is a **fused 8col xclbin** producing all of Q, K, V
in a single 1.17 ms NPU dispatch (see Authoritative FlowKV ABI section
in NPU_PLAN.md). Splitting into 3 separate INT4 GEMVs means 3 separate
dispatches + 3 weight BO syncs. For 2048×512 matrices the INT4 memory
saving (~3x) may not beat the dispatch overhead of going from 1→3 runs.

Plan says "ROI оценить после Phase 8.2" — agreed, but should be
*explicitly* marked "may be negative ROI" in the Roadmap so it doesn't
get treated as a default next step after 8.2.

**Action**: change Phase 8.3 from "опционально, 1–2 days" to "blocked
on measurement: only proceed if Phase 8.2 profiling shows QKV
dominates the residual decode time".

### N9. RMS_NORM+QKV interference is not mentioned

The chat-mode bug (RMS_NORM 1-col ⊕ QKV 8-col, see NPU_PLAN.md "Known
bug" section) means `XDNA_ENABLE_RMS_NORM=0` is the current default.
With Q4_0 QKV (Phase 8.1), the column count of the new INT4 GEMV
xclbin needs to be ≥4col to avoid recreating the 1-vs-8 interference.

`AIEFusedDequantGEMV` defaults — check `op.py` for column count
assumptions. If it inherits `num_aie_columns` from the host, ensure
the host passes `ctx->num_cols` (8) and not 1. If it has any 1-col
hard-coding internally, that's a landmine.

**Action**: add a risk row to the table:
"INT4 GEMV xclbin compiled with cols=1 → may revive RMS+QKV-class
interference. Mitigation: always compile with num_aie_columns ≥ 4."

### N10. Build-agent delegation note inconsistency

> "Task 1: ... PR в `llama.cpp-xdna`."

Task 1 (Phase 8.0 + 8.1) is mostly one-repo (main repo), correct.
Task 2 (Phase 8.2) is correctly described as two-repo.

Phase 8.4 needs **both** repos if going Option B (native Q4_K kernel),
single-repo if going Option A (re-quantize host-side). Plan currently
doesn't say which the build agent should take. Default to Option A —
agree with the plan's recommendation but make it explicit in the
delegation note.

**Action**: clarify per-task repo scope in the delegation list.

---

### Summary of recommended edits

1. Rewrite Phase 8.0 with Windows instructions (N1).
2. Skip the host-bias detour; do kernel `aie::sub(8)` from start (N2).
3. Add `correctness_test.py` cases per phase acceptance (N3).
4. Add empirical perplexity check to Phase 8.4 (N4).
5. Update Roadmap with P50/P90 columns (N5).
6. Add Prerequisites section with `llama-quantize` commands (N6).
7. Replace ROCm citation in 8.5 with AIE microbenchmark plan (N7).
8. Mark Phase 8.3 as measurement-gated (N8).
9. Add column-count risk for INT4 GEMV (N9).
10. Clarify per-task repo scope in delegation list (N10).

None of these block the overall plan — it is sound and grounded. They
just remove ambiguity that would burn an executing agent's time.
