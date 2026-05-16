@echo off
setlocal

REM ============================================================
REM Build & Run DMA Echo Test for XDNA NPU debugging.
REM
REM Uses C:\Python313 (same as debug_flowkv.bat) with XRT SDK
REM on PYTHONPATH. Conda env only for Peano/mlir_aie tools.
REM
REM Usage: run_echo.bat [1|2]
REM   1 = single ObjectFifo (default)
REM   2 = dual ObjectFifo (K+V concat pattern)
REM ============================================================

set "VERSION=%~1"
if "%VERSION%"=="" set "VERSION=1"

set "IRON_DIR=C:\llama.cpp-xdna\IRON-windows"
set "ECHO_DIR=%IRON_DIR%\iron\operators\dma_echo"

REM Same env as debug_flowkv.bat
set "AMD_DRIVER_DIR=C:\Windows\System32\DriverStore\FileRepository\kipudrv.inf_amd64_1a1aa059597c4810"
set "PATH=%AMD_DRIVER_DIR%;%CD%;%PATH%"

set "PYTHONPATH=C:\Users\Kuhnya\Downloads\xrt_windows_sdk\xrt_sdk\xrt\python;C:\Python313\Lib\site-packages;%PYTHONPATH%"
set "PEANO_INSTALL_DIR=C:\ProgramData\miniforge3\envs\ryzen-ai-1.7.1\Lib\site-packages\win64.o\tools\peano"
set "XRT_BIN_DIR=C:\Users\Kuhnya\Downloads\xrt_windows_sdk\xrt_sdk\xrt"
set "MLIR_AIE_BIN_DIR=C:\ProgramData\miniforge3\envs\ryzen-ai-1.7.1\Lib\site-packages\mlir_aie\bin"
set "PATH=%PEANO_INSTALL_DIR%\bin;%XRT_BIN_DIR%;%MLIR_AIE_BIN_DIR%;%PATH%"

set "PYTHON=C:\Python313\python.exe"

echo ============================================================
echo DMA Echo Test v%VERSION% — Build + Run
echo ============================================================
echo.

REM ===== Compile =====
echo === STEP 1: Compile ===
"%PYTHON%" "%ECHO_DIR%\compile_echo.py" --version %VERSION%
if errorlevel 1 (
    echo.
    echo COMPILATION FAILED
    pause
    exit /b 1
)

echo.
echo === STEP 2: Test on NPU ===
"%PYTHON%" "%ECHO_DIR%\test_echo.py" --version %VERSION%
if errorlevel 1 (
    echo.
    echo TEST FAILED
) else (
    echo.
    echo TEST PASSED
)

echo.
pause
