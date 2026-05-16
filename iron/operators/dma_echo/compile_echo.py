# Compile DMA echo test using IRON's compilation framework.
#
# Must be run with C:\Python313\python.exe + PYTHONPATH including XRT SDK.
# Use run_echo.bat which sets up the environment correctly.
#
# Usage (manual, from IRON-windows root):
#   set PYTHONPATH=C:\Users\Kuhnya\Downloads\xrt_windows_sdk\xrt_sdk\xrt\python;%PYTHONPATH%
#   C:\Python313\python.exe iron\operators\dma_echo\compile_echo.py --version 1
import os
import sys
from pathlib import Path

# IRON-windows root (4 levels up from this script)
IRON_DIR = str(Path(__file__).resolve().parent.parent.parent.parent)

# Nuke iron_repo from sys.path — it shadows IRON-windows
sys.path = [p for p in sys.path if "iron_repo" not in p]

# IRON-windows FIRST
if IRON_DIR not in sys.path:
    sys.path.insert(0, IRON_DIR)

# Verify iron resolves correctly
import importlib
_spec = importlib.util.find_spec("iron")
if _spec is None:
    print("ERROR: 'iron' package not found. Is IRON-windows cloned correctly?")
    sys.exit(1)
_origin = str(_spec.origin or _spec.submodule_search_locations)
if "iron_repo" in _origin:
    print(f"ERROR: iron resolves to iron_repo: {_origin}")
    print("Remove or rename C:\\llama.cpp-xdna\\iron_repo")
    sys.exit(1)
print(f"iron = {_origin}")

import argparse
from iron.common.compilation import base as comp


def compile_echo(version: int):
    build_dir = Path(IRON_DIR) / f"build_echo_v{version}"
    build_dir.mkdir(exist_ok=True)

    conda_prefix = Path(os.environ.get(
        "CONDA_PREFIX",
        r"C:\ProgramData\miniforge3\envs\ryzen-ai-1.7.1"
    ))
    # win64.o peano has AIE-targeted clang (for kernel compilation)
    peano_dir = conda_prefix / "Lib" / "site-packages" / "win64.o" / "tools" / "peano"
    mlir_aie_dir = conda_prefix / "Lib" / "site-packages" / "mlir_aie"
    # llvm-aie has opt/llc (for aiecc.py xclbin/insts compilation)
    llvm_aie_dir = Path(r"C:\Python313\Lib\site-packages\llvm-aie")

    if not peano_dir.exists():
        print(f"ERROR: Peano not found at {peano_dir}")
        sys.exit(1)
    if not mlir_aie_dir.exists():
        print(f"ERROR: mlir_aie not found at {mlir_aie_dir}")
        sys.exit(1)
    if not llvm_aie_dir.exists():
        print(f"ERROR: llvm-aie not found at {llvm_aie_dir}")
        print("Install: pip install llvm-aie -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly")
        sys.exit(1)

    print(f"=== Compiling Echo v{version} ===")
    print(f"Peano (clang):  {peano_dir}")
    print(f"LLVM-AIE (opt): {llvm_aie_dir}")
    print(f"mlir_aie:       {mlir_aie_dir}")
    print(f"Build:          {build_dir}")
    print()

    file_base = f"echo_v{version}"
    design_dir = Path(IRON_DIR) / "iron" / "operators" / "dma_echo"
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

    # Full ELF for NPU dispatch (pyxrt.elf path, not xclbin + insts.bin)
    elf_artifact = comp.FullElfArtifact(
        f"{file_base}.elf",
        mlir_input=mlir_artifact,
        dependencies=[mlir_artifact, kernel_artifact],
    )

    graph = comp.CompilationArtifactGraph()
    graph.add(elf_artifact)

    rules = [
        comp.GenerateMLIRFromPythonCompilationRule(),
        comp.PeanoCompilationRule(peano_dir, mlir_aie_dir),
        comp.AieccFullElfCompilationRule(build_dir, llvm_aie_dir, mlir_aie_dir),
    ]

    try:
        comp.compile(rules, graph, build_dir=str(build_dir))
    except RuntimeError as e:
        print(f"\nCOMPILATION FAILED: {e}")
        sys.exit(1)

    elf_path = build_dir / f"{file_base}.elf"

    print()
    print("=" * 60)
    print("BUILD SUCCESS")
    print("=" * 60)
    if elf_path.exists():
        print(f"  ELF: {elf_path} ({elf_path.stat().st_size} bytes)")
    print()
    print(f"Test: python test_echo.py --version {version}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=int, choices=[1, 2], required=True)
    args = ap.parse_args()
    compile_echo(args.version)
