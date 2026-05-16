@echo off
setlocal

REM ============================================================
REM Build & Run DMA Echo Test for XDNA NPU debugging.
REM
REM Uses IRON's compilation framework (aiecc.py, NOT aiecc.exe).
REM
REM Usage: run_echo.bat [1|2]
REM   1 = single ObjectFifo (default)
REM   2 = dual ObjectFifo (K+V concat pattern)
REM ============================================================

set "VERSION=%~1"
if "%VERSION%"=="" set "VERSION=1"

set "IRON_DIR=C:\llama.cpp-xdna\IRON-windows"
set "CONDA_PYTHON=C:\ProgramData\miniforge3\envs\ryzen-ai-1.7.1\python.exe"
set "ECHO_DIR=%IRON_DIR%\iron\operators\dma_echo"

echo ============================================================
echo DMA Echo Test v%VERSION% — Build + Run
echo ============================================================
echo.

REM ===== Compile =====
echo === STEP 1: Compile ===
"%CONDA_PYTHON%" "%ECHO_DIR%\compile_echo.py" --version %VERSION%
if errorlevel 1 (
    echo.
    echo COMPILATION FAILED
    echo.
    echo Troubleshooting:
    echo   1. Make sure aiecc.py works:  python "C:\ProgramData\miniforge3\envs\ryzen-ai-1.7.1\Lib\site-packages\mlir_aie\bin\aiecc.py" --help
    echo   2. If aiecc.py hangs, kill it and check Windows Defender exclusions
    echo   3. Try running aiecc.py directly with -v flag to see verbose output
    pause
    exit /b 1
)

echo.
echo === STEP 2: Test on NPU ===
"%CONDA_PYTHON%" "%ECHO_DIR%\test_echo.py" --version %VERSION%
if errorlevel 1 (
    echo.
    echo TEST FAILED — check output above
) else (
    echo.
    echo TEST PASSED
)

echo.
pause
