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
set "MLIR_AIE_BIN_DIR=C:\Python313\Lib\site-packages\mlir_aie\bin"
set "PATH=%PEANO_INSTALL_DIR%\bin;%LLVM_AIE_BIN%;%XRT_BIN_DIR%;%MLIR_AIE_BIN_DIR%;%PATH%"
set "VCVARS64=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if exist "%VCVARS64%" call "%VCVARS64%" >nul

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
set "CC=%LLVM_AIE_BIN%\clang.exe"
set "CC_TARGET=aie2p-none-unknown-elf"
set "LLVM_AIE_INC=C:\Python313\Lib\site-packages\llvm-aie\include"
set "MLIR_AIE_INC=C:\Python313\Lib\site-packages\mlir_aie\include"
set "MLIR_AIE_RUNTIME=C:\Python313\Lib\site-packages\mlir_aie\aie_runtime_lib\AIE2P"
%CC% --target=%CC_TARGET% -O2 -std=c++20 ^
    -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body -Wno-missing-template-arg-list-after-template-kw ^
    -I"%MLIR_AIE_INC%" ^
    -I"%MLIR_AIE_RUNTIME%" ^
    -include"%IRON_DIR%\aie_kernels\aie_platform_shim.h" ^
    -isystem "%LLVM_AIE_INC%\aie2p-none-unknown-elf\c++\v1" ^
    -isystem "%LLVM_AIE_INC%\c++\v1" ^
    -c "%IRON_DIR%\aie_kernels\aie2p\echo.cc" -o "%BUILD_DIR%\echo.o"
if errorlevel 1 (
    echo FAIL: echo.o compilation
    exit /b 1
)
echo echo.o compiled
%PYTHON% -c "import sys; from pathlib import Path; sys.path.insert(0, r'%IRON_DIR%'); from iron.common.compilation.base import _elf_remove_section; p=Path(r'%BUILD_DIR%\echo.o'); [_elf_remove_section(p, s) for s in ('.tctmemtab','.tctmemtabl','.tctmemstrtab')]"
if errorlevel 1 (
    echo FAIL: echo.o section strip
    exit /b 1
)

echo echo.o stripped

REM === Step 3: Compile MLIR to xclbin + insts via aiecc ===
echo === Step 3: aiecc compilation ===
if exist "echo_v%VERSION%.mlir.prj" rmdir /s /q "echo_v%VERSION%.mlir.prj"
set "AIECC=%BUILD_DIR%\aiecc_orphan_place.exe"
if not exist "%AIECC%" %PYTHON% -c "from pathlib import Path; src=Path(r'%MLIR_AIE_BIN_DIR%\aiecc.exe'); dst=Path(r'%BUILD_DIR%\aiecc_orphan_place.exe'); b=src.read_bytes(); old=b'--orphan-handling=error'; new=b'--orphan-handling=place'; assert old in b; dst.write_bytes(b.replace(old,new))"
set "PEANO=%PEANO_INSTALL_DIR%"

"%AIECC%" -v -j1 ^
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

cl.exe /EHsc /std:c++17 /Zc:__cplusplus /O2 ^
    /I"%XRT_INCLUDE%" ^
    "%ECHO_DIR%\echo_test.cpp" ^
    /Fe:"%BUILD_DIR%\echo_test.exe" ^
    /link "%XRT_LIB%\xrt_coreutil.lib"
if errorlevel 1 (
    echo FAIL: echo_test.cpp compilation
    echo Trying with xrt_core.lib...
    cl.exe /EHsc /std:c++17 /Zc:__cplusplus /O2 ^
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
if errorlevel 1 (
    echo FAIL: echo_test.exe run
    exit /b 1
)

echo.
echo === DONE ===
