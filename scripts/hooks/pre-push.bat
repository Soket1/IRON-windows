@echo off
REM SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
REM SPDX-License-Identifier: Apache-2.0

setlocal enabledelayedexpansion

echo Running pre-push linting checks...
echo ====================================

REM Check if we're in the right directory
if not exist "scripts\clang-format-wrapper.py" (
    echo Error: Must be run from repository root
    exit /b 1
)

set FAILED=0

REM Check Licenses
echo.
echo Checking licenses with reuse...
where reuse >nul 2>nul
if %errorlevel% equ 0 (
    reuse lint
    if !errorlevel! neq 0 (
        echo ❌ License check failed
        echo    Run: reuse annotate --template ApacheAMD --copyright-prefix spdx-string-c --copyright "Advanced Micro Devices, Inc. All rights reserved." --license="Apache-2.0" --recursive --skip-unrecognised ./
        set FAILED=1
    ) else (
        echo ✅ License check passed
    )
) else (
    echo ⚠️  Warning: reuse not installed, skipping license check
    echo    Install with: pip install reuse
)

REM Format Python
echo.
echo Checking Python formatting with black...
where black >nul 2>nul
if %errorlevel% equ 0 (
    black --check .
    if !errorlevel! neq 0 (
        echo ❌ Python formatting check failed
        echo    Run: black .
        set FAILED=1
    ) else (
        echo ✅ Python formatting check passed
    )
) else (
    echo ⚠️  Warning: black not installed, skipping Python format check
    echo    Install with: pip install black
)

REM Format C++
echo.
echo Checking C++ formatting with clang-format...
where clang-format >nul 2>nul
if %errorlevel% equ 0 (
    python scripts\clang-format-wrapper.py --check
    if !errorlevel! neq 0 (
        echo ❌ C++ formatting check failed
        echo    Run: python scripts\clang-format-wrapper.py --fix
        set FAILED=1
    ) else (
        echo ✅ C++ formatting check passed
    )
) else (
    echo ⚠️  Warning: clang-format not installed, skipping C++ format check
    echo    Install with: download from LLVM releases or use Visual Studio
)

echo.
echo ====================================

if %FAILED% equ 1 (
    echo ❌ Pre-push checks FAILED
    echo.
    echo Please fix the issues above before pushing.
    echo To bypass this hook ^(not recommended^), use: git push --no-verify
    exit /b 1
) else (
    echo ✅ All pre-push checks PASSED
    echo.
)

exit /b 0
