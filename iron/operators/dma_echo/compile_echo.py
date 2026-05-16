"""Compile DMA echo test using IRON's compilation framework.

Uses aiecc.py (not aiecc.exe which hangs on Windows due to PyInstaller).

Usage:
    C:\ProgramData\miniforge3\envs\ryzen-ai-1.7.1\python.exe compile_echo.py --version 1
"""
import os
import sys
from pathlib import Path

# ===== Environment setup (MUST be before any iron imports) =====

# 1. XRT SDK — pyxrt Python bindings
XRT_PYTHON = r"C:\Users\Kuhnya\Downloads\xrt_windows_sdk\xrt_sdk\xrt\python"
if os.path.isdir(XRT_PYTHON) and XRT_PYTHON not in sys.path:
    sys.path.insert(0, XRT_PYTHON)

# 2. XRT runtime DLLs — needed for pyxrt DLL load
XRT_BIN = r"C:\Users\Kuhnya\Downloads\xrt_windows_sdk\xrt_sdk\xrt"
if os.path.isdir(XRT_BIN):
    os.environ["PATH"] = XRT_BIN + os.pathsep + os.environ.get("PATH", "")

# 3. IRON-windows — must be FIRST to beat iron_repo
IRON_DIR = str(Path(__file__).resolve().parent.parent.parent)
if IRON_DIR not in sys.path:
    sys.path.insert(0, IRON_DIR)

# 4. Kill iron_repo entries — it shadows IRON-windows
sys.path = [p for p in sys.path if "iron_repo" not in p.replace("\\", "/").lower()]

# 5. Verify
print(f"IRON_DIR = {IRON_DIR}")
print(f"XRT_PYTHON = {XRT_PYTHON}")
print(f"iron_repo filtered = {'iron_repo' not in '|'.join(sys.path).lower()}")
print()

# ===== Now import =====
import argparse
from iron.common.compilation import base as comp


def compile_echo(version: int):
    build_dir = Path(IRON_DIR) / f"build_echo_v{version}"
    build_dir.mkdir(exist_ok=True)

    conda_prefix = Path(os.environ.get(
        "CONDA_PREFIX",
        r"C:\ProgramData\miniforge3\envs\ryzen-ai-1.7.1"
    ))
    peano_dir = conda_prefix / "Lib" / "site-packages" / "win64.o" / "tools" / "peano"
    mlir_aie_dir = conda_prefix / "Lib" / "site-packages" / "mlir_aie"

    if not peano_dir.exists():
        print(f"ERROR: Peano not found at {peano_dir}")
        sys.exit(1)
    if not mlir_aie_dir.exists():
        print(f"ERROR: mlir_aie not found at {mlir_aie_dir}")
        sys.exit(1)

    print(f"=== Compiling Echo v{version} ===")
    print(f"Peano:    {peano_dir}")
    print(f"mlir_aie: {mlir_aie_dir}")
    print(f"Build:    {build_dir}")
    print()

    file_base = f"echo_v{version}"
    design_dir = Path(__file__).resolve().parent
    kernel_src = Path(IRON_DIR) / "aie_kernels" / "aie2p" / "echo.cc"

    mlir_artifact = comp.PythonGeneratedMLIRArtifact.new(
        f"{file_base}.mlir",
        import_path=design_dir / "design.py",
        callback_fn=f"echo_v{version}",
        callback_args=["npu2"],
    )

    kernel_artifact = comp.KernelObjectArtifact.new(
        "echo.o",
        depends=[comp.SourceArtifact.new(kernel_src)],
    )

    xclbin_artifact = comp.XclbinArtifact.new(
        f"{file_base}.xclbin",
        depends=[mlir_artifact, kernel_artifact],
    )

    insts_artifact = comp.InstsBinArtifact.new(
        f"{file_base}.bin",
        depends=[mlir_artifact],
    )

    graph = comp.CompilationArtifactGraph()
    graph.add(xclbin_artifact)
    graph.add(insts_artifact)

    rules = [
        comp.GenerateMLIRFromPythonCompilationRule(),
        comp.PeanoCompilationRule(peano_dir, mlir_aie_dir),
        comp.AieccXclbinInstsCompilationRule(build_dir, peano_dir, mlir_aie_dir),
    ]

    try:
        comp.compile(rules, graph, build_dir=str(build_dir))
    except RuntimeError as e:
        print(f"\nCOMPILATION FAILED: {e}")
        sys.exit(1)

    xclbin_path = build_dir / f"{file_base}.xclbin"
    insts_path = build_dir / f"{file_base}.bin"

    print()
    print("=" * 60)
    print("BUILD SUCCESS")
    print("=" * 60)
    if xclbin_path.exists():
        print(f"  XCLBIN: {xclbin_path} ({xclbin_path.stat().st_size} bytes)")
    if insts_path.exists():
        print(f"  INSTS:  {insts_path} ({insts_path.stat().st_size} bytes)")
    print()
    print(f"Test: python test_echo.py --version {version}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=int, choices=[1, 2], required=True)
    args = ap.parse_args()
    compile_echo(args.version)
