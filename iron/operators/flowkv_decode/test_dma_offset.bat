@echo off
setlocal

REM ============================================================
REM DMA OFFSET TEST — verifies where K DMA actually reads from.
REM
REM Uses existing FlowKV xclbin with specific buffer patterns.
REM No new compilation needed — just git pull and run.
REM
REM Test plan:
REM   1. Fill K buffer with 0xAAAA pattern
REM   2. Fill V buffer with 0xBBBB pattern
REM   3. Dispatch FlowKV
REM   4. Check K_DIAG
REM
REM Expected results:
REM   K_DIAG = 0xAAAA → DMA reads K correctly (offset 0)
REM   K_DIAG = 0xBBBB → DMA reads V (arg mapping wrong)
REM   K_DIAG = random  → DMA reads from wrong address
REM ============================================================

set "AMD_DRIVER_DIR=C:\Windows\System32\DriverStore\FileRepository\kipudrv.inf_amd64_1a1aa059597c4810"
set "PATH=%AMD_DRIVER_DIR%;%CD%;%PATH%"

set "PYTHONPATH=C:\Users\Kuhnya\Downloads\xrt_windows_sdk\xrt_sdk\xrt\python;C:\Python313\Lib\site-packages;%PYTHONPATH%"
set "GGML_XDNA_PYTHON_CMD=C:\Python313\python.exe"
set "PEANO_INSTALL_DIR=C:\ProgramData\miniforge3\envs\ryzen-ai-1.7.1\Lib\site-packages\win64.o\tools\peano"
set "XRT_BIN_DIR=C:\Users\Kuhnya\Downloads\xrt_windows_sdk\xrt_sdk\xrt"
set "MLIR_AIE_BIN_DIR=C:\ProgramData\miniforge3\envs\ryzen-ai-1.7.1\Lib\site-packages\mlir_aie\bin"
set "PATH=%PEANO_INSTALL_DIR%\bin;%XRT_BIN_DIR%;%MLIR_AIE_BIN_DIR%;%PATH%"

set "GGML_XDNA_CACHE_DIR=C:\llama.cpp-xdna\npu_kernels_win_8col"
set "MODEL_PATH=models\llama-3.2-1b-instruct-BF16.gguf"

REM ===== TEST 1: Separate buffers (current code) =====
echo ============================================================
echo TEST 1: Separate buffers (bo_k + bo_v)
echo ============================================================
set "XDNA_ENABLE_FLOWKV_DECODE=1"
set "XNA_SCHED_DEBUG=1"
set "XDNA_DEBUG=1"
set "XDNA_FORCE_CPU="
set "XDNA_ENABLE_GEMV=1"
set "XDNA_ENABLE_SWIGLU=1"
set "XDNA_ENABLE_QKV=1"
set "XDNA_ENABLE_DECODE_BATCH=1"
set "XDNA_ENABLE_TRANSFORMER_BLOCK=1"
set GGML_XDNA_NUM_COLS=8
set GGML_XDNA_FORCE_CH1=0
set GGML_SCHED_KV_OFFLOAD=1

echo /exit | build\bin\Release\llama-cli.exe -m "%MODEL_PATH%" -p "What is the capital of France" -n 16 -c 512 -ngl 100 --no-mmap -fa off >dma_test1_output.log 2>dma_test1_xdna.log

echo.
echo --- TEST 1 RESULTS ---
findstr /C:"K_DIAG" dma_test1_xdna.log | findstr /C:"[0:8]" | head -3
findstr /C:"Host.*K\[0\]" dma_test1_xdna.log | head -3
findstr /C:"Match count" dma_test1_xdna.log | head -3
echo.

REM ===== TEST 2: Shared buffer (revert via git stash) =====
echo ============================================================
echo TEST 2: Shared buffer (original code)
echo ============================================================
echo Saving current changes...
git stash

echo Running with shared buffer...
echo /exit | build\bin\Release\llama-cli.exe -m "%MODEL_PATH%" -p "What is the capital of France" -n 16 -c 512 -ngl 100 --no-mmap -fa off >dma_test2_output.log 2>dma_test2_xdna.log

echo.
echo --- TEST 2 RESULTS ---
findstr /C:"K_DIAG" dma_test2_xdna.log | findstr /C:"[0:8]" | head -3
findstr /C:"Host.*K\[0\]" dma_test2_xdna.log | head -3
findstr /C:"V\[0\]" dma_test2_xdna.log | head -3
findstr /C:"Match count" dma_test2_xdna.log | head -3

echo Restoring changes...
git stash pop

echo.
echo ============================================================
echo COMPARISON
echo ============================================================
echo.
echo TEST 1 (separate buffers):
echo   K_DIAG should show what DMA reads from bo_k
echo.
echo TEST 2 (shared buffer):
echo   K_DIAG should show V[0] data (0xB928 0x3B4A...)
echo   This confirms DMA reads from offset 16384, not offset 0
echo.
echo If TEST 2 K_DIAG = V[0] data: DMA offset is +16384 bf16
echo   Workaround: swap K and V positions in shared buffer
echo.
echo Files:
echo   dma_test1_output.log / dma_test1_xdna.log — separate buffers
echo   dma_test2_output.log / dma_test2_xdna.log — shared buffer
echo.
pause
