@echo off
setlocal
REM ============================================================
REM Build + Run Echo Test via Custom XRT Dispatch
REM Bypasses IRON framework — tests raw DMA path.
REM
REM Usage: run_echo_custom.bat [1|2|all]
REM   1   = single ObjectFifo copy (default)
REM   2   = dual ObjectFifo concat
REM ============================================================

set "VERSION=%~1"
if "%VERSION%"=="" set "VERSION=1"

set "IRON_DIR=C:\llama.cpp-xdna\IRON-windows"
set "ECHO_DIR=%IRON_DIR%\iron\operators\dma_echo"
set "BUILD_DIR=%IRON_DIR%\build\echo_custom"
set "PYTHON=C:\Python313\python.exe"

REM === Environment (same as debug_flowkv.bat) ===
set "AMD_DRIVER_DIR=C:\Windows\System32\DriverStore\FileRepository\kipudrv.inf_amd64_1a1aa059597c4810"
set "PATH=%AMD_DRIVER_DIR%;%CD%;%PATH%"
set "PYTHONPATH=C:\Users\Kuhnya\Downloads\xrt_windows_sdk\xrt_sdk\xrt\python;%PYTHONPATH%"
set "PEANO_INSTALL_DIR=C:\ProgramData\miniforge3\envs\ryzen-ai-1.7.1\Lib\site-packages\win64.o\tools\peano"
set "LLVM_AIE_BIN=C:\Python313\Lib\site-packages\llvm-aie\bin"
set "XRT_BIN_DIR=C:\Users\Kuhnya\Downloads\xrt_windows_sdk\xrt_sdk\xrt"
set "MLIR_AIE_BIN_DIR=C:\ProgramData\miniforge3\envs\ryzen-ai-1.7.1\Lib\site-packages\mlir_aie\bin"
set "PATH=%PEANO_INSTALL_DIR%\bin;%LLVM_AIE_BIN%;%XRT_BIN_DIR%;%MLIR_AIE_BIN_DIR%;%PATH%"

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

REM === Step 1: Generate MLIR ===
echo === Step 1: Generate echo v%VERSION% MLIR ===
%PYTHON% -c "import sys; sys.path.insert(0, r'%IRON_DIR%'); from iron.operators.dma_echo.design import echo_v1, echo_v2; m = echo_v1('npu2', 256) if %VERSION%==1 else echo_v2('npu2', 128); open(r'%BUILD_DIR%\echo_v%VERSION%.mlir','w').write(str(m)); print('MLIR written')"
if errorlevel 1 (
    echo FAIL: MLIR generation
    exit /b 1
)

REM === Step 2: Compile echo kernel (echo.o) ===
echo === Step 2: Compile echo.o ===
set "CC=%PEANO_INSTALL_DIR%\bin\clang.exe"
set "CC_TARGET=aie2p-none-unknown-elf"
%CC% --target=%CC_TARGET% -O2 -c "%IRON_DIR%\aie_kernels\aie2p\echo.cc" -o "%BUILD_DIR%\echo.o"
if errorlevel 1 (
    echo FAIL: echo.o compilation
    exit /b 1
)
echo echo.o compiled

REM === Step 3: Compile MLIR to xclbin + insts via aiecc ===
echo === Step 3: aiecc compilation ===
set "AIECC=%MLIR_AIE_BIN_DIR%\aiecc.exe"
set "PEANO=%PEANO_INSTALL_DIR%"

%PYTHON% "%AIECC%" -v -j1 ^
    --no-compile-host --no-xchesscc --no-xbridge ^
    --peano "%PEANO%" ^
    --dynamic-objFifos ^
    --aie-generate-xclbin --xclbin-name="%BUILD_DIR%\echo_v%VERSION%.xclbin" ^
    --xclbin-kernel-name=MLIR_AIE ^
    --aie-generate-npu-insts --npu-insts-name="%BUILD_DIR%\echo_v%VERSION%.insts" ^
    "%BUILD_DIR%\echo_v%VERSION%.mlir"
if errorlevel 1 (
    echo FAIL: aiecc compilation
    exit /b 1
)
echo xclbin + insts compiled

REM === Step 4: Compile echo_test.cpp ===
echo === Step 4: Compile echo_test.cpp ===
set "XRT_INCLUDE=C:\Users\Kuhnya\Downloads\xrt_windows_sdk\xrt_sdk\xrt\include"
set "XRT_LIB=C:\Users\Kuhnya\Downloads\xrt_windows_sdk\xrt_sdk\xrt\lib"

cl.exe /EHsc /std:c++17 /O2 ^
    /I"%XRT_INCLUDE%" ^
    "%ECHO_DIR%\echo_test.cpp" ^
    /Fe:"%BUILD_DIR%\echo_test.exe" ^
    /link "%XRT_LIB%\xrt_coreutil.lib"
if errorlevel 1 (
    echo FAIL: echo_test.cpp compilation
    echo Trying with xrt_core.lib...
    cl.exe /EHsc /std:c++17 /O2 ^
        /I"%XRT_INCLUDE%" ^
        "%ECHO_DIR%\echo_test.cpp" ^
        /Fe:"%BUILD_DIR%\echo_test.exe" ^
        /link "%XRT_LIB%\xrt_core.lib"
    if errorlevel 1 (
        echo FAIL: echo_test.cpp compilation (both libs)
        exit /b 1
    )
)
echo echo_test.exe compiled

REM === Step 5: Run ===
echo === Step 5: Run echo v%VERSION% ===
"%BUILD_DIR%\echo_test.exe" ^
    "%BUILD_DIR%\echo_v%VERSION%.xclbin" ^
    "%BUILD_DIR%\echo_v%VERSION%.insts" ^
    %VERSION%

echo.
echo === DONE ===
