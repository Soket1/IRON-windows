@echo off
setlocal

REM ============================================================
REM DMA Echo Test — Build + Run using IRON test framework.
REM
REM Usage: run_echo.bat [1|2|all]
REM   1   = single ObjectFifo (v1, default)
REM   2   = dual ObjectFifo (v2)
REM   all = both tests
REM ============================================================

set "VERSION=%~1"
if "%VERSION%"=="" set "VERSION=1"

set "IRON_DIR=C:\llama.cpp-xdna\IRON-windows"
set "CONDA_PYTHON=C:\ProgramData\miniforge3\envs\ryzen-ai-1.7.1\python.exe"
set "ECHO_DIR=%IRON_DIR%\iron\operators\dma_echo"

REM Same env as debug_flowkv.bat
set "AMD_DRIVER_DIR=C:\Windows\System32\DriverStore\FileRepository\kipudrv.inf_amd64_1a1aa059597c4810"
set "PATH=%AMD_DRIVER_DIR%;%CD%;%PATH%"

set "PYTHONPATH=C:\Users\Kuhnya\Downloads\xrt_windows_sdk\xrt_sdk\xrt\python;C:\Python313\Lib\site-packages\llvm-aie;%PYTHONPATH%"
set "PEANO_INSTALL_DIR=C:\ProgramData\miniforge3\envs\ryzen-ai-1.7.1\Lib\site-packages\win64.o\tools\peano"
set "LLVM_AIE_BIN=C:\Python313\Lib\site-packages\llvm-aie\bin"
set "XRT_BIN_DIR=C:\Users\Kuhnya\Downloads\xrt_windows_sdk\xrt_sdk\xrt"
set "MLIR_AIE_BIN_DIR=C:\ProgramData\miniforge3\envs\ryzen-ai-1.7.1\Lib\site-packages\mlir_aie\bin"
set "PATH=%PEANO_INSTALL_DIR%\bin;%LLVM_AIE_BIN%;%XRT_BIN_DIR%;%MLIR_AIE_BIN_DIR%;%PATH%"

cd /d "%IRON_DIR%"

echo ============================================================
echo DMA Echo Test v%VERSION% — IRON test framework
echo ============================================================
echo.

if "%VERSION%"=="all" (
    echo Running all echo tests...
    "%CONDA_PYTHON%" -m pytest "%ECHO_DIR%\test.py" -v --iterations 1 --tb=short 2>&1
) else (
    echo Running echo v%VERSION%...
    "%CONDA_PYTHON%" -m pytest "%ECHO_DIR%\test.py" -v --iterations 1 --tb=short -k "test_echo_v%VERSION%" 2>&1
)

echo.
pause
