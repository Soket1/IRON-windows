# FlowKV decode attention test script
# Usage: .\test_flowkv.ps1 -NumTokens 50 -Prompt "Hello world"

param(
    [int]$NumTokens = 50,
    [string]$Prompt = "Hello"
)

# Enable FlowKV decode attention on NPU
$env:XDNA_ENABLE_FLOWKV_DECODE = "1"

# Existing NPU flags
$env:XDNA_ENABLE_GEMV = "1"
$env:XDNA_ENABLE_SWIGLU = "1"
$env:XDNA_ENABLE_QKV = "1"
$env:XDNA_ENABLE_DECODE_BATCH = "1"
$env:XDNA_ENABLE_TRANSFORMER_BLOCK = "1"
$env:GGML_XDNA_NUM_COLS = "8"

# Debug output
$env:XDNA_DEBUG = "1"

Write-Host "=== FlowKV Decode Attention Test ===" -ForegroundColor Cyan
Write-Host "Prompt: $Prompt"
Write-Host "Tokens: $NumTokens"
Write-Host ""

# Adjust path to your llama-cli
llama-cli -m "models\llama-3.2-1b-instruct-BF16.gguf" -p $Prompt -n $NumTokens 2>flowkv_debug.log

Write-Host ""
Write-Host "=== Debug log (FlowKV lines) ===" -ForegroundColor Cyan
Select-String -Path flowkv_debug.log -Pattern "FlowKV" | ForEach-Object { $_.Line }
