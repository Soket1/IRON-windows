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

set "GGML_XDNA_CACHE_DIR=C:\llama.cpp-xdna\npu_kernels_win_8col"
set XDNA_ENABLE_GEMV=1
set XDNA_ENABLE_SWIGLU=1
set XDNA_ENABLE_QKV=1
set XDNA_ENABLE_RMS_NORM=0
set XDNA_ENABLE_SWIGLU_PREFILL=0
set XDNA_DEBUG=1
set GGML_XDNA_NUM_COLS=8
set GGML_XDNA_FORCE_CH1=0

set "MODEL_PATH=models\llama-3.2-1b-instruct-BF16.gguf"

build\bin\Release\llama-cli.exe -m "%MODEL_PATH%" -p "Hi" -n 4 -c 64 -ngl 100 --no-mmap -fa off -cnv 1> llama_output.log 2> xdna_internal.log & type llama_output.log & type xdna_internal.log
pause
