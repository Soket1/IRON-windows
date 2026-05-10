# FlowKV decode attention timing test
# Usage: .\test_timing.ps1

Write-Host "[1/2] Warming up kernel cache..." -ForegroundColor Cyan

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
$env:XDNA_DEBUG = "0"
$env:GGML_XDNA_NUM_COLS = "8"
$env:GGML_XDNA_FORCE_CH1 = "0"
$env:XDNA_ENABLE_TRANSFORMER_BLOCK = "1"
$env:XDNA_ENABLE_DECODE_BATCH = "1"

# FlowKV decode attention on NPU
$env:XDNA_ENABLE_FLOWKV_DECODE = "1"

$env:MODEL_PATH = "models\llama-3.2-1b-instruct-BF16.gguf"

build\bin\Release\llama-cli.exe -m $env:MODEL_PATH -p "Hello" -n 1 -c 512 -ngl 100 --no-mmap -fa off > $null 2>&1

Write-Host "[1/2] Warmup done." -ForegroundColor Green
Write-Host ""
Write-Host "[2/2] Running timing test (64 decode tokens)..." -ForegroundColor Cyan

$env:XDNA_DEBUG = "1"

build\bin\Release\llama-cli.exe -m $env:MODEL_PATH -p "What is the capital of France" -n 64 -c 512 -ngl 100 --no-mmap -fa off > llama_timing.log 2> xdna_timing.log

Write-Host ""
Write-Host "=== llama-cli output ===" -ForegroundColor Cyan
Get-Content llama_timing.log

Write-Host ""
Write-Host "=== Last 30 lines of XDNA debug ===" -ForegroundColor Cyan
Get-Content xdna_timing.log -Tail 30

Write-Host ""
Write-Host "=== FlowKV dispatch times ===" -ForegroundColor Yellow
Select-String -Path xdna_timing.log -Pattern "FlowKV" | Select-Object -Last 20 | ForEach-Object { $_.Line }

Write-Host ""
Write-Host "=== Per-token dispatch times (QKV + SwiGLU, last 20 tokens) ===" -ForegroundColor Yellow
Select-String -Path xdna_timing.log -Pattern "qkv_prof.*M=1|swiglu_prof decode M=1" | Select-Object -Last 40 | ForEach-Object { $_.Line }

Read-Host "Press Enter to exit"
