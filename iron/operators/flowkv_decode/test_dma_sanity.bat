@echo off
setlocal

set "AMD_DRIVER_DIR=C:\Windows\System32\DriverStore\FileRepository\kipudrv.inf_amd64_1a1aa059597c4810"
set "PATH=%AMD_DRIVER_DIR%;%CD%;%PATH%"

set "PYTHONPATH=C:\Users\Kuhnya\Downloads\xrt_windows_sdk\xrt_sdk\xrt\python;C:\Python313\Lib\site-packages;%PYTHONPATH%"
set "GGML_XDNA_PYTHON_CMD=C:\Python313\python.exe"
set "PEANO_INSTALL_DIR=C:\ProgramData\miniforge3\envs\ryzen-ai-1.7.1\Lib\site-packages\win64.o\tools\peano"
set "XRT_BIN_DIR=C:\Users\Kuhnya\Downloads\xrt_windows_sdk\xrt_sdk\xrt"
set "MLIR_AIE_BIN_DIR=C:\ProgramData\miniforge3\envs\ryzen-ai-1.7.1\Lib\site-packages\mlir_aie\bin"
set "PATH=%PEANO_INSTALL_DIR%\bin;%XRT_BIN_DIR%;%MLIR_AIE_BIN_DIR%;%PATH%"

echo ============================================================
echo DMA SANITY TEST — minimal DMA read/write verification
echo ============================================================
echo.

REM ===== STEP 1: Generate minimal MLIR =====
echo === STEP 1: Generate minimal DMA MLIR ===
C:\Python313\python.exe -c "import sys; sys.path.insert(0, r'C:\Users\Kuhnya\Downloads\xrt_windows_sdk\xrt_sdk\xrt\python'); sys.path.insert(0, r'C:\llama.cpp-xdna'); from iron.operators.flowkv_decode.test_dma_sanity import create_dma_sanity_design; m = create_dma_sanity_design('npu2', 1024); open('dma_sanity.mlir','w').write(str(m)); print('OK: dma_sanity.mlir written')"
if errorlevel 1 (
    echo FAILED: MLIR generation failed
    pause
    exit /b 1
)
echo.

REM ===== STEP 2: Try using FlowKV xclbin with known patterns =====
echo === STEP 2: Test DMA offset with FlowKV xclbin ===
echo This test uses the existing FlowKV kernel to check if DMA
echo reads from the correct buffer offset.
echo.
echo The test will:
echo   1. Fill K buffer with 0xAAAA pattern
echo   2. Fill V buffer with 0xBBBB pattern
echo   3. Dispatch FlowKV kernel
echo   4. Check what K_DIAG reads
echo.
echo If K_DIAG = 0xAAAA: DMA reads K correctly (offset 0)
echo If K_DIAG = 0xBBBB: DMA reads from V buffer (offset bug)
echo If K_DIAG = random: DMA reads from wrong address entirely
echo.

echo === STEP 2b: Test with shared buffer (revert) ===
echo Using shared bo_kv with K at offset 0, V at offset 16384...
echo.

REM Run with XDNA_DEBUG=1 to get DIAG output
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
set "MODEL_PATH=models\llama-3.2-1b-instruct-BF16.gguf"

echo Running FlowKV with debug output...
echo /exit | build\bin\Release\llama-cli.exe -m "%MODEL_PATH%" -p "What is the capital of France" -n 16 -c 512 -ngl 100 --no-mmap -fa off >dma_sanity_output.log 2>dma_sanity_xdna.log

echo.
echo === RESULTS ===
findstr /C:"Prompt:" dma_sanity_output.log
echo.
echo --- K_DIAG values ---
findstr /C:"K_DIAG" dma_sanity_xdna.log | findstr /C:"[0:8]" | head -5
echo.
echo --- Match results ---
findstr /C:"Match count" dma_sanity_xdna.log | head -5
echo.
echo --- NPU vs CPU ---
findstr /C:"NPU=" dma_sanity_xdna.log | head -5
echo.
echo === Files ===
echo   dma_sanity_output.log — llama output
echo   dma_sanity_xdna.log — NPU debug log
echo   dma_sanity.mlir — minimal MLIR (not yet compiled)
echo.
pause
