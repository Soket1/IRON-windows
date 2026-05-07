# MLIR-AIE на Windows: полный гайд

> Руководство по программированию AMD Ryzen AI NPU через MLIR-AIE / IRON на Windows (нативно, без WSL).
> Актуально на май 2026, mlir_aie v1.3.1.

---

## Содержание

1. [Обзор](#1-обзор)
2. [Архитектура AIE](#2-архитектура-aie)
3. [Что нужно установить](#3-что-нужно-установить)
4. [Установка](#4-установка)
5. [Ключевые флаги aiecc](#5-ключевые-флаги-aiecc)
6. [Первый пример: vector_scalar_mul](#6-первый-пример-vector_scalar_mul)
7. [Готовые операторы IRON](#7-готовые-операторы-iron)
8. [Написание своего оператора](#8-написание-своего-оператора)
9. [Llama 3.2 1B на NPU](#9-llama-32-1b-на-npu)
10. [Известные проблемы и решения](#10-известные-проблемы-и-решения)
11. [Ссылки](#11-ссылки)

---

## 1. Обзор

**MLIR-AIE** — инструментарий от AMD (ранее Xilinx) для программирования AI Engine,
встроенного в процессоры Ryzen AI (NPU) и FPGA Versal.

**IRON** — Python API поверх MLIR-AIE, заточенное под close-to-metal программирование NPU.

### Что можно делать

- Запускать векторные/матричные операции на NPU (GEMM, attention, convolutions)
- Инференс LLM (Llama 3.2 1B пример включён)
- Кастомные DSP/ML операторы с векторизацией через AIE intrinsics

### Ключевые репозитории

| Репозиторий | Описание |
|---|---|
| [Xilinx/mlir-aie](https://github.com/Xilinx/mlir-aie) | Основной toolchain (MLIR dialect, aiecc, wheels) |
| [Xilinx/llvm-aie](https://github.com/Xilinx/llvm-aie) | Peano — LLVM-based компилятор для AIE core |
| [amd/IRON](https://github.com/amd/IRON) | Официальный IRON (Python API + операторы) |
| [Soket1/IRON-windows](https://github.com/Soket1/IRON-windows) | Форк IRON с Windows-адаптациями |

---

## 2. Архитектура AIE

```
┌─────────────────────────────────────────────┐
│              AIE Array (spatial)             │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐    │
│  │Tile 0│──│Tile 1│──│Tile 2│──│Tile 3│    │
│  │ Core │  │ Core │  │ Core │  │ Core │    │
│  │ L1   │  │ L1   │  │ L1   │  │ L1   │    │
│  │ DMA  │  │ DMA  │  │ DMA  │  │ DMA  │    │
│  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘    │
│     └─────────┴─────────┴─────────┘         │
│           Stream Switch Interconnect         │
└──────────────────────┬──────────────────────┘
                       │ DMA
              ┌────────┴────────┐
              │   Host (x86)    │
              │   XRT runtime   │
              └─────────────────┘
```

**Принципы:**
- Каждый тайл — векторный процессор + L1 scratchpad
- Ядро работает **только** со своими локальными данными
- Данные перемещаются через **DMA** по коммутируемому интерконнекту
- Compute и data movement — **независимы и параллельны**

---

## 3. Что нужно установить

### Обязательно

| Компонент | Зачем | Откуда |
|---|---|---|
| **Python 3.10–3.14** | IRON, mlir_aie | [python.org](https://www.python.org/downloads/) или Microsoft Store |
| **mlir_aie wheel** | MLIR Python bindings, aiecc | `pip install mlir_aie==1.3.1` |
| **llvm-aie (Peano)** | AIE core compiler (clang++) | `pip install llvm-aie` |
| **XRT (Xilinx Runtime)** | Доступ к NPU из хоста | [AMD Ryzen AI SDK](https://ryzenai.docs.amd.com/en/latest/inst/install.html) |
| **NPU Driver** | Драйвер для Ryzen AI NPU | Там же, от AMD |
| **Visual Studio Build Tools** | Компиляция host code (MSVC) | [VS Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) |
| **CMake** | Сборка host code | [cmake.org](https://cmake.org/download/) или `pip install cmake` |
| **Git** | Клонирование репозиториев | [git-scm.com](https://git-scm.com/download/win) |

### Опционально

| Компонент | Зачем |
|---|---|
| **OpenCV** | Vision examples |
| **Jupyter Notebook** | Интерактивные примеры |

### Совместимость

| Python | mlir_aie v1.3.1 | llvm-aie | XRT |
|---|---|---|---|
| 3.10 | ✅ win_amd64 | ✅ | ✅ |
| 3.11 | ✅ win_amd64 | ✅ | ✅ |
| 3.12 | ✅ win_amd64 | ✅ | ✅ |
| 3.13 | ✅ win_amd64 | ✅ | ✅ |
| 3.14 | ✅ win_amd64 | ✅ | ✅ |

---

## 4. Установка

### 4.1. NPU Driver и XRT

1. Скачать [Ryzen AI Software](https://ryzenai.docs.amd.com/en/latest/inst/install.html)
2. Установить NPU driver (`.exe` установщик)
3. Установить XRT (по умолчанию в `C:\Xilinx\XRT` или `C:\Program Files\AMD\XRT`)
4. **Перезагрузиться**

Проверить:
```powershell
& "C:\Xilinx\XRT\bin\xrt-smi.exe" examine
```

Должно показать:
```
Devices present
BDF              : Name
------------------------------------
[0000:66:00.1]   : NPU Strix
```

### 4.2. Python окружение

```powershell
# Клонировать IRON-windows (адаптированный под Windows)
git clone https://github.com/Soket1/IRON-windows.git
cd IRON-windows

# Создать venv
python -m venv ironenv
ironenv\Scripts\activate
python -m pip install --upgrade pip
```

### 4.3. Установка mlir_aie и Peano

```powershell
# MLIR-AIE (Windows wheel, v1.3.1)
pip install mlir_aie==1.3.1 -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/v1.3.1

# Peano (AIE core compiler — ставится как .exe)
pip install llvm-aie -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
```

### 4.4. Зависимости проекта

```powershell
pip install -r requirements.txt
```

### 4.5. Переменные окружения

Создайте `setup.ps1` в корне проекта:
```powershell
# setup.ps1
ironenv\Scripts\activate
call "C:\Xilinx\XRT\setup.bat"
```

Запускайте перед каждой сессией: `.\setup.ps1`

Или добавьте XRT в системный PATH:
```
Параметры системы → Переменные среды → Path → Добавить:
  C:\Xilinx\XRT\bin
```

### 4.6. Проверка

```powershell
# Быстрый тест одного оператора
pytest .\iron\operators\axpy\
```

Ожидаемый вывод: `PASSED`

---

## 5. Ключевые флаги aiecc

При ручном вызове компилятора **обязательно** использовать:

```
--aie-generate-xclbin          Генерировать xclbin (бинарник для NPU)
--aie-generate-npu-insts       Генерировать инструкции NPU
--no-xchesscc                  НЕ использовать Vitis xchesscc ← ОБЯЗАТЕЛЬНО
--no-xbridge                   НЕ использовать xbridge         ← ОБЯЗАТЕЛЬНО
--no-compile-host              Компиляцию host code — отдельно через cmake
--xclbin-name=output.xclbin    Имя выходного xclbin
--npu-insts-name=insts.bin     Имя выходного файла инструкций
```

> ⚠️ `--no-xchesscc --no-xbridge` — **всегда** на Windows. Vitis AIE Tools не поддерживают Windows.
> Peano (из llvm-aie wheel) полностью заменяет xchesscc для AIE2/AIE2P.

---

## 6. Первый пример: vector_scalar_mul

### Структура

```
programming_examples/basic/vector_scalar_mul/
├── aie_kernels/aie2/scale.cc    # C++ kernel для AIE core
├── design.py                    # IRON Python: топология + data movement
├── test.cpp                     # Host code (C++, XRT)
├── run_mlir_aie_example.py      # Скрипт сборки + запуска
└── CMakeLists.txt               # Сборка host code
```

### Сборка и запуск

```powershell
cd programming_examples\basic\vector_scalar_mul

# Шаг 1: Собрать всё (kernel → xclbin → host exe)
python .\run_mlir_aie_example.py build

# Шаг 2: Запустить на NPU
python .\run_mlir_aie_example.py run
```

### Что происходит внутри

```
1. scale.cc ──→ clang++.exe (Peano) ──→ scale.o           # Код для AIE core
2. design.py ──→ aiecc ──→ final.xclbin + insts.bin        # MLIR → xclbin
3. test.cpp ──→ cmake + MSVC ──→ test.exe                  # Host code
4. test.exe ──→ XRT ──→ NPU execution                      # Запуск
```

Ожидаемый вывод:
```
Avg NPU time: 645us.
Min NPU time: 645us.
Max NPU time: 645us.
PASS!
```

---

## 7. Готовые операторы IRON

Все операторы из IRON-windows:

| Оператор | Описание | AIE2 | AIE2P | Тест |
|---|---|---|---|---|
| `axpy` | A * X + Y | ✅ | ✅ | `pytest iron/operators/axpy/` |
| `elementwise_add` | Поэлементное сложение | ✅ | ✅ | `pytest iron/operators/elementwise_add/` |
| `elementwise_mul` | Поэлементное умножение | ✅ | ✅ | `pytest iron/operators/elementwise_mul/` |
| `gemm` | Matrix multiplication | ✅ | ✅ | `pytest iron/operators/gemm/` |
| `gemv` | Matrix-vector multiply | ✅ | ✅ | `pytest iron/operators/gemv/` |
| `mha` | Multi-Head Attention / GQA | — | ✅ | `pytest iron/operators/mha/` |
| `rms_norm` | RMS Normalization | ✅ | ✅ | `pytest iron/operators/rms_norm/` |
| `layer_norm` | Layer Normalization | ✅ | ✅ | `pytest iron/operators/layer_norm/` |
| `softmax` | Softmax | ✅ | ✅ | `pytest iron/operators/softmax/` |
| `rope` | Rotary Positional Embedding | ✅ | ✅ | `pytest iron/operators/rope/` |
| `silu` | SiLU activation | ✅ | ✅ | `pytest iron/operators/silu/` |
| `gelu` | GELU activation | ✅ | ✅ | `pytest iron/operators/gelu/` |
| `relu` | ReLU | ✅ | ✅ | `pytest iron/operators/relu/` |
| `sigmoid` | Sigmoid | ✅ | ✅ | `pytest iron/operators/sigmoid/` |
| `tanh` | Tanh | ✅ | ✅ | `pytest iron/operators/tanh/` |
| `transpose` | Transpose | ✅ | ✅ | `pytest iron/operators/transpose/` |
| `mem_copy` | Copy (passthrough) | ✅ | ✅ | `pytest iron/operators/mem_copy/` |
| `dequant` | AWQ Q4→bfloat16 | ✅ | ✅ | `pytest iron/operators/dequant/` |

### Запуск тестов

```powershell
# Все операторы (базовые)
pytest iron/operators/ -m "not extensive"

# Все операторы (полные)
pytest iron/operators/

# Конкретный
pytest iron/operators/gemm/
```

### Структура каждого оператора

```
iron/operators/gemm/
├── op.py           # Python API: как вызвать оператор
├── design.py       # MLIR-AIE: топология тайлов, ObjectFIFO, data movement
├── reference.py    # CPU reference implementation (numpy)
└── test.py         # Тест: NPU результат vs CPU reference
```

---

## 8. Написание своего оператора

### Шаблон: element-wise add

**Шаг 1: Kernel для AIE core** — `aie_kernels/generic/add.cc`

```cpp
#include <aie_api/aie.hpp>

void add_bf16(bfloat16 *a, bfloat16 *b, bfloat16 *out, int N) {
    for (int i = 0; i < N; i += 32) {
        v32bfloat16 va = *(v32bfloat16 *)(a + i);
        v32bfloat16 vb = *(v32bfloat16 *)(b + i);
        v32bfloat16 vout = va + vb;
        *(v32bfloat16 *)(out + i) = vout;
    }
}
```

**Шаг 2: Design** — `iron/operators/my_add/design.py`

```python
from aie.iron import Program, Runtime, Worker, ObjectFifo
from aie.iron.device import NPU1

dev = NPU1()

# ObjectFIFOs — абстракция передачи данных между хостом и тайлами
of_in_a = ObjectFifo(tile_ty, name="in_a")
of_in_b = ObjectFifo(tile_ty, name="in_b")
of_out  = ObjectFifo(tile_ty, name="out")

# Runtime — последовательность операций хоста
rt = Runtime()
with rt.sequence(tensor_ty, tensor_ty, tensor_ty) as (a, b, out):
    rt.fill(of_in_a.prod(), a)
    rt.fill(of_in_b.prod(), b)
    rt.dispatch(Worker(add_kernel))
    rt.drain(of_out.cons(), out)

my_program = Program(dev, rt)
print(my_program.compile())
```

**Шаг 3: Operator API** — `iron/operators/my_add/op.py`

```python
def my_add(M: int, N: int):
    """Create and return the add operator."""
    # настройка размеров, вызов design.py, сборка
```

**Шаг 4: Reference** — `iron/operators/my_add/reference.py`

```python
import numpy as np

def reference_add(a, b):
    return a + b
```

**Шаг 5: Test** — `iron/operators/my_add/test.py`

```python
def test_add():
    # создать данные, запустить на NPU, сравнить с reference
    assert np.allclose(npu_result, reference_result)
```

---

## 9. Llama 3.2 1B на NPU

IRON-windows включает end-to-end инференс Llama 3.2 1B.

### Расположение

```
iron/applications/llama_3.2_1b/
├── README.md           # Инструкции
├── model.py            # Модель
├── operators/          # Fused operators
└── weights/            # (нужно скачать)
```

### Запуск

```powershell
cd iron\applications\llama_3.2_1b

# Следовать инструкциям в README.md (скачать веса, конвертировать)
python run.py --prompt "Hello world"
```

### Что задействовано на NPU

| Оператор | Роль в LLM |
|---|---|
| **GEMM** | Матричные умножения в каждом слое |
| **MHA / GQA** | Multi-Head / Grouped Query Attention |
| **RMSNorm** | Нормализация |
| **RoPE** | Rotary Positional Embedding |
| **SiLU** | Активация в FFN |
| **Softmax** | В attention |
| **Dequant** | Деквантизация весов AWQ Q4 → bfloat16 |

Всё выполняется на NPU, хост только координирует передачу данных.

---

## 10. Известные проблемы и решения

### `ERT_CMD_STATE_ERROR` при выполнении

**Причина:** Несовместимость версии NPU driver и XRT.

**Решение:** Обновить NPU driver до последней версии с [AMD Ryzen AI SDK](https://ryzenai.docs.amd.com/en/latest/inst/install.html).

---

### Makefile примеры не работают

**Причина:** Содержат Unix-команды (`cp`, `ln`, `sh`).

**Решение:** Использовать `run_mlir_aie_example.py` вместо `make`, или IRON-windows, где пути адаптированы.

---

### `env_setup.sh` не работает

**Решение:** Создать `setup.ps1`:
```powershell
ironenv\Scripts\activate
call "C:\Xilinx\XRT\setup.bat"
```

---

### Пути с пробелами

**Решение:** Не ставить проекты в `C:\Program Files\...`. Использовать `C:\dev\...`.

---

### `aiecc` не находит Peano

**Причина:** Peano не найден в PATH.

**Решение:** mlir_aie v1.3.1 находит Peano автоматически через pip metadata. Если нет:
```powershell
# Найти путь к Peano
python -c "import pathlib, importlib; print(pathlib.Path(importlib.import_module('llvm_aie').__file__).parent / 'bin')"

# Добавить в PATH
$env:PATH += ";<путь_из_команды_выше>"
```

---

### Xclbin генерация падает с ошибкой link

**Причина:** Не хватает флагов.

**Решение:** Всегда использовать `--no-xchesscc --no-xbridge`.

---

### `test_library.cpp` не компилируется

**Причина:** Использует POSIX `<sys/mman.h>`, недоступный на Windows.

**Решение:** Не критично. Это библиотека для симуляции, не нужна для реального выполнения на NPU. `test_utils.lib` работает нормально (исправлено в PR #3024).

---

## 11. Ссылки

### Основные

- [mlir-aie GitHub](https://github.com/Xilinx/mlir-aie)
- [IRON-windows](https://github.com/Soket1/IRON-windows)
- [amd/IRON](https://github.com/amd/IRON)
- [IRON Programming Guide](https://xilinx.github.io/mlir-aie/programming_guide)
- [AIE API (C++ intrinsics)](https://xilinx.github.io/aie_api/topics.html)
- [AMD Ryzen AI SDK](https://ryzenai.docs.amd.com/en/latest/inst/install.html)

### Windows-специфичное

- [Windows Setup (official docs)](https://xilinx.github.io/mlir-aie/buildHostWin.html)
- [PR #2677: Windows wheels](https://github.com/Xilinx/mlir-aie/pull/2677)
- [PR #3024: test_utils Windows fix](https://github.com/Xilinx/mlir-aie/pull/3024)
- [v1.3.1 Release (Windows wheels)](https://github.com/Xilinx/mlir-aie/releases/tag/v1.3.1)

### Архитектура

- [AIE1 Architecture Manual (AM009)](https://docs.amd.com/r/en-US/am009-versal-ai-engine/Overview)
- [AIE2 Architecture Manual (AM020)](https://docs.amd.com/r/en-US/am020-versal-aie-ml/Overview)
- [IRON API Paper (FCCM 2025)](https://arxiv.org/abs/2504.18430)

### Сообщество

- [IRON Discord](https://discord.gg/cW99Ds85e8)
- [GitHub Issues](https://github.com/amd/iron/issues)

---

## Приложение: быстрый старт (копипаста)

```powershell
# 1. Установить NPU driver + XRT с https://ryzenai.docs.amd.com/en/latest/inst/install.html
# 2. Перезагрузиться

# 3. Клонировать
git clone https://github.com/Soket1/IRON-windows.git
cd IRON-windows

# 4. Venv
python -m venv ironenv
ironenv\Scripts\activate
pip install --upgrade pip

# 5. Установить toolchain
pip install mlir_aie==1.3.1 -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/v1.3.1
pip install llvm-aie -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly

# 6. Зависимости
pip install -r requirements.txt

# 7. XRT
call "C:\Xilinx\XRT\setup.bat"

# 8. Проверить
pytest .\iron\operators\axpy\

# 9. Собрать и запустить пример
cd programming_examples\basic\vector_scalar_mul
python .\run_mlir_aie_example.py build
python .\run_mlir_aie_example.py run
```

---

*Последнее обновление: май 2026. mlir_aie v1.3.1, IRON-windows devel.*
