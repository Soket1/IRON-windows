# Short test with clean cache — FlowKV decode attention
# Usage: .\test_short1.ps1

# Clean iron compiler cache in build directory
Remove-Item -Path "C:\llama.cpp-xdna\build\*" -Include *.mlir, *.o, *.bin, *.insts, *.xclbin -Recurse -Force -ErrorAction SilentlyContinue

# Remove aiecc temporary folders
Get-ChildItem -Path "C:\llama.cpp-xdna\build\*" -Directory -Filter "*.prj" -Recurse | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# Clear final cache folder
if (Test-Path npu_kernels_win_8col) { Remove-Item -Recurse -Force npu_kernels_win_8col }

$env:AMD_DRIVER_DIR = "C:\Windows\System32\DriverStore\FileRepository\kipudrv.inf_amd64_1a1aa059597c4810"
$env:PATH = "$env:AMD_DRIVER_DIR;$env:PWD;$env:PATH"

$env:PYTHONPATH = "C:\Users\Kuhnya\Downloads\xrt_windows_sdk\xrt_sdk\xrt\python;C:\Python313\Lib\site-packages;$env:PYTHONPATH"
$env:GGML_XDNA_PYTHON_CMD = "C:\Python313\python.exe"
$env:PEANO_INSTALL_DIR = "C:\ProgramData\miniforge3\envs\ryzen-ai-1.7.1\Lib\site-packages\win64.o\tools\peano"
$env:XRT_BIN_DIR = "C:\Users\Kuhnya\Downloads\xrt_windows_sdk\xrt_sdk\xrt"
$env:MLIR_AIE_BIN_DIR = "C:\ProgramData\miniforge3\envs\ryzen-ai-1.7.1\Lib\site-packages\mlir_aie\bin"
$env:PATH = "$env:PEANO_INSTALL_DIR\bin;$env:XRT_BIN_DIR;$env:MLIR_AIE_BIN_DIR;$env:PATH"

$env:GGML_XDNA_CACHE_DIR = "C:\llama.cpp-xdna\npu_kernels_win_8col"
$env:XDNA_ENABLE_GEMV = "1"
$env:XDNA_ENABLE_SWIGLU = "1"
$env:XDNA_ENABLE_QKV = "1"
$env:XDNA_ENABLE_RMS_NORM = "0"
$env:XDNA_ENABLE_SWIGLU_PREFILL = "0"
$env:XDNA_DEBUG = "1"
$env:GGML_XDNA_NUM_COLS = "8"
$env:GGML_XDNA_FORCE_CH1 = "0"
$env:XDNA_ENABLE_TRANSFORMER_BLOCK = "1"
$env:XDNA_ENABLE_DECODE_BATCH = "1"

# FlowKV decode attention on NPU
$env:XDNA_ENABLE_FLOWKV_DECODE = "1"

$env:MODEL_PATH = "models\llama-3.2-1b-instruct-BF16.gguf"

build\bin\Release\llama-cli.exe -m $env:MODEL_PATH -p "What is the capital of France" -n 32 -c 512 -ngl 100 --no-mmap -fa off -cnv > llama_output.log 2> xdna_internal.log

Write-Host "=== llama-cli output ===" -ForegroundColor Cyan
Get-Content llama_output.log

Write-Host ""
Write-Host "=== XDNA internal log ===" -ForegroundColor Cyan
Get-Content xdna_internal.log

Read-Host "Press Enter to exit"
