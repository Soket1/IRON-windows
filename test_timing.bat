@echo off
setlocal

echo [1/2] Warming up kernel cache...

set "AMD_DRIVER_DIR=C:\Windows\System32\DriverStore\FileRepository\kipudrv.inf_amd64_1a1aa059597c4810"
set "PATH=%AMD_DRIVER_DIR%;%CD%;%PATH%"

set "PYTHONPATH=C:\Users\Kuhnya\Downloads\xrt_windows_sdk\xrt_sdk\xrt\python;C:\Python313\Lib\site-packages;%PYTHONPATH%"
set "GGML_XDNA_PYTHON_CMD=C:\Python313\python.exe"
set "PEANO_INSTALL_DIR=C:\ProgramData\miniforge3\envs\ryzen-ai-1.7.1\Lib\site-packages\win64.o\tools\peano"
set "XRT_BIN_DIR=C:\Users\Kuhnya\Downloads\xrt_windows_sdk\xrt_sdk\xrt"
set "MLIR_AIE_BIN_DIR=C:\ProgramData\miniforge3\envs\ryzen-ai-1.7.1\Lib\site-packages\mlir_aie\bin"
set "PATH=%PEANO_INSTALL_DIR%\bin;%XRT_BIN_DIR%;%MLIR_AIE_BIN_DIR%;%PATH%"

set "GGML_XDNA_CACHE_DIR=C:\llama.cpp-xdna\npu_kernels_win_8col"
set XDNA_ENABLE_GEMV=1
set XDNA_ENABLE_SWIGLU=1
set XDNA_ENABLE_QKV=1
set XDNA_ENABLE_RMS_NORM=0
set XDNA_ENABLE_SWIGLU_PREFILL=0
set XDNA_DEBUG=0
set GGML_XDNA_NUM_COLS=8
set GGML_XDNA_FORCE_CH1=0
set XDNA_ENABLE_TRANSFORMER_BLOCK=1
set XDNA_ENABLE_DECODE_BATCH=1

REM FlowKV decode attention on NPU
set XDNA_ENABLE_FLOWKV_DECODE=1

set "MODEL_PATH=models\llama-3.2-1b-instruct-BF16.gguf"

build\bin\Release\llama-cli.exe -m "%MODEL_PATH%" -p "Hello" -n 1 -c 512 -ngl 100 --no-mmap -fa off >nul 2>&1

echo [1/2] Warmup done.
echo.
echo [2/2] Running timing test (64 decode tokens)...

set XDNA_DEBUG=1

build\bin\Release\llama-cli.exe -m "%MODEL_PATH%" -p "What is the capital of France" -n 64 -c 512 -ngl 100 --no-mmap -fa off >llama_timing.log 2>xdna_timing.log

echo.
echo === llama-cli output ===
type llama_timing.log
echo.
echo === Last 30 lines of XDNA debug ===
powershell -Command "Get-Content xdna_timing.log -Tail 30"
echo.
echo === FlowKV dispatch times ===
powershell -Command "Get-Content xdna_timing.log | Select-String 'FlowKV' | Select-Object -Last 20"
echo.
echo === Per-token dispatch times (QKV + SwiGLU, last 20 tokens) ===
powershell -Command "Get-Content xdna_timing.log | Select-String 'qkv_prof.*M=1|swiglu_prof decode M=1' | Select-Object -Last 40"

pause
