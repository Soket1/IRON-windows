@echo off
setlocal

REM ============================================================
REM Build DMA Echo Test for XDNA NPU debugging.
REM
REM Generates MLIR from design.py, compiles echo.cc kernel,
REM and runs aiecc.py to produce xclbin + insts.bin.
REM
REM Usage: build_echo.bat [1|2]
REM   1 = single ObjectFifo (v1, default)
REM   2 = dual ObjectFifo (v2, simulates K+V pattern)
REM ============================================================

set "VERSION=%~1"
if "%VERSION%"=="" set "VERSION=1"

echo ============================================================
echo DMA Echo Test Build — Version %VERSION%
echo ============================================================
echo.

REM ===== Environment =====
set "IRON_DIR=C:\llama.cpp-xdna\IRON-windows"
set "PEANO_DIR=C:\ProgramData\miniforge3\envs\ryzen-ai-1.7.1\Lib\site-packages\win64.o\tools\peano"
set "MLIR_AIE_DIR=C:\ProgramData\miniforge3\envs\ryzen-ai-1.7.1\Lib\site-packages\mlir_aie"
set "CONDA_PYTHON=C:\ProgramData\miniforge3\envs\ryzen-ai-1.7.1\python.exe"

set "BUILD_DIR=%IRON_DIR%\build_echo_v%VERSION%"
set "KERNEL_SRC=%IRON_DIR%\aie_kernels\aie2p\echo.cc"
set "KERNEL_OBJ=%BUILD_DIR%\echo.o"
set "MLIR_FILE=%BUILD_DIR%\echo_v%VERSION%.mlir"
set "XCLBIN_FILE=%BUILD_DIR%\echo_v%VERSION%.xclbin"
set "INSTS_FILE=%BUILD_DIR%\echo_v%VERSION%.bin"

REM ===== Derived paths =====
set "CLANG=%PEANO_DIR%\bin\clang.exe"
set "AIECC=%MLIR_AIE_DIR%\bin\aiecc.py"

REM ===== Verify tools exist =====
if not exist "%CLANG%" (
    echo ERROR: Peano clang not found: %CLANG%
    echo Install: pip install llvm-aie -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
    exit /b 1
)
if not exist "%AIECC%" (
    echo ERROR: aiecc.py not found: %AIECC%
    echo Install: pip install mlir_aie==1.3.1 -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/v1.3.1
    exit /b 1
)

REM ===== Create build dir =====
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

REM ===== STEP 1: Generate MLIR =====
echo === STEP 1: Generate MLIR (echo_v%VERSION%) ===
"%CONDA_PYTHON%" -c "import sys; sys.path.insert(0, r'%IRON_DIR%'); from iron.operators.dma_echo.design import echo_v%VERSION%; m = echo_v%VERSION%('npu2'); open(r'%MLIR_FILE%', 'w').write(str(m)); print('OK: %MLIR_FILE%')"
if errorlevel 1 (
    echo FAILED: MLIR generation
    exit /b 1
)
echo.

REM ===== STEP 2: Compile kernel with Peano =====
echo === STEP 2: Compile echo.cc kernel ===
set "TARGET=aie2p-none-unknown-elf"
set "INCLUDE=%MLIR_AIE_DIR%\include"
set "RUNTIME_LIB=%MLIR_AIE_DIR%\aie_runtime_lib\AIE2P"

REM Find llvm-aie C++ headers
set "CXX_INCLUDE="
for /d %%D in ("%CONDA_PYTHON%\..\..\Lib\site-packages\llvm-aie*") do (
    if exist "%%D\include\aie2p-none-unknown-elf\c++\v1" (
        set "CXX_INCLUDE=%%D\include\aie2p-none-unknown-elf\c++\v1"
    )
)

set "CXX_FLAGS=-isystem %INCLUDE%"
if defined CXX_INCLUDE (
    set "CXX_FLAGS=%CXX_FLAGS% -isystem %CXX_INCLUDE%"
    REM Also add generic c++/v1 headers
    for %%D in ("%CXX_INCLUDE%\..\..\..") do (
        if exist "%%D\c++\v1" (
            set "CXX_FLAGS=%CXX_FLAGS% -isystem %%D\c++\v1"
        )
    )
)

echo Target: %TARGET%
echo Kernel: %KERNEL_SRC%
echo Output: %KERNEL_OBJ%
echo.

"%CLANG%" --target=%TARGET% -O2 %CXX_FLAGS% ^
    -I"%INCLUDE%" -I"%RUNTIME_LIB%" ^
    -c "%KERNEL_SRC%" -o "%KERNEL_OBJ%"
if errorlevel 1 (
    echo FAILED: kernel compilation
    exit /b 1
)
echo OK: %KERNEL_OBJ%
echo.

REM ===== STEP 3: Strip .tctmemtab if present =====
echo === STEP 3: Clean kernel object ===
where llvm-objcopy >nul 2>&1
if not errorlevel 1 (
    llvm-objcopy --remove-section=.tctmemtab "%KERNEL_OBJ%" 2>nul
    echo Stripped .tctmemtab (if present)
) else (
    echo llvm-objcopy not found, skipping .tctmemtab strip
)
echo.

REM ===== STEP 4: Run aiecc.py =====
echo === STEP 4: Compile MLIR to xclbin + insts.bin ===
echo This may take several minutes...
echo.

REM Set PEANO_INSTALL_DIR so aiecc finds Peano tools
set "PEANO_INSTALL_DIR=%PEANO_DIR%"
set "PATH=%PEANO_DIR%\bin;%MLIR_AIE_DIR%\bin;%PATH%"

REM For v1: kernel name is "echo_copy_bf16"
REM For v2: kernel name is "echo_concat_bf16"
set "KERNEL_NAME=echo_copy_bf16"
if "%VERSION%"=="2" set "KERNEL_NAME=echo_concat_bf16"

"%CONDA_PYTHON%" "%AIECC%" ^
    -v -j1 ^
    --no-compile-host ^
    --no-xchesscc ^
    --no-xbridge ^
    --peano "%PEANO_DIR%" ^
    --dynamic-objFifos ^
    --aie-generate-xclbin ^
    --xclbin-name="%XCLBIN_FILE%" ^
    --xclbin-kernel-name=%KERNEL_NAME% ^
    --aie-generate-npu-insts ^
    --npu-insts-name="%INSTS_FILE%" ^
    "%MLIR_FILE%"
if errorlevel 1 (
    echo FAILED: aiecc compilation
    exit /b 1
)

echo.
echo ============================================================
echo BUILD SUCCESS
echo ============================================================
echo   MLIR:   %MLIR_FILE%
echo   Kernel: %KERNEL_OBJ%
echo   XCLBIN: %XCLBIN_FILE%
echo   INSTS:  %INSTS_FILE%
echo.
echo To test: run test_echo.bat %VERSION%
echo.
pause
