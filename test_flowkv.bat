@echo off
REM FlowKV decode attention test script
REM Usage: test_flowkv.bat [num_tokens] [prompt]

set NUM_TOKENS=%1
if "%NUM_TOKENS%"=="" set NUM_TOKENS=50

set PROMPT=%2
if "%PROMPT%"=="" set PROMPT=Hello

REM Enable FlowKV decode attention on NPU
set XDNA_ENABLE_FLOWKV_DECODE=1

REM Existing NPU flags
set XDNA_ENABLE_GEMV=1
set XDNA_ENABLE_SWIGLU=1
set XDNA_ENABLE_QKV=1
set XDNA_ENABLE_DECODE_BATCH=1
set XDNA_ENABLE_TRANSFORMER_BLOCK=1
set GGML_XDNA_NUM_COLS=8

REM Debug output (set to 0 to disable)
set XDNA_DEBUG=1

echo === FlowKV Decode Attention Test ===
echo Prompt: %PROMPT%
echo Tokens: %NUM_TOKENS%
echo.

REM Adjust path to your llama-cli
llama-cli -m models\llama-3.2-1b-instruct-BF16.gguf -p "%PROMPT%" -n %NUM_TOKENS% 2>flowkv_debug.log

echo.
echo === Debug log: flowkv_debug.log ===
findstr /C:"FlowKV" flowkv_debug.log
