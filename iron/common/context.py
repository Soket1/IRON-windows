# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar
import os
import sys

from . import compilation as comp
import aie.utils.config
import aie.utils as aie_utils


def _resolve_peano_dir() -> Path:
    """Return a Peano installation directory that contains opt/llc.

    ``aie.utils.config.peano_install_dir()`` may point to a stripped
    package (e.g. ``win64.o/tools/peano``) that ships only clang/ld.lld
    but is missing opt/llc.  This function tries several fallbacks:

    1. The path from ``aie.utils.config.peano_install_dir()`` if it has opt/llc.
    2. The ``PEANO_INSTALL_DIR`` environment variable.
    3. The ``llvm-aie`` pip-package in the current or common Python installs.
    4. The raw ``aie.utils.config`` path as last resort.
    """
    exe = ".exe" if sys.platform == "win32" else ""
    opt_name = f"opt{exe}"

    # 1. Default from mlir_aie config
    default = Path(aie.utils.config.peano_install_dir())
    if (default / "bin" / opt_name).is_file():
        return default

    # 2. Environment variable
    env_dir = os.environ.get("PEANO_INSTALL_DIR")
    if env_dir:
        p = Path(env_dir)
        if (p / "bin" / opt_name).is_file():
            return p

    # 3. Search for the llvm-aie pip package in known site-packages
    #    (current env + common Windows Python locations)
    import site, glob
    search_roots: list[str] = list(site.getsitepackages())
    try:
        search_roots.append(site.getusersitepackages())
    except Exception:
        pass
    # Common Windows Python install dirs that may not be in getsitepackages()
    # when running from a different conda/venv environment.
    for extra in glob.glob(r"C:\Python*\Lib\site-packages"):
        search_roots.append(extra)
    for extra in glob.glob(r"C:\ProgramData\miniforge3\envs\*\Lib\site-packages"):
        search_roots.append(extra)
    for sp in search_roots:
        candidate = Path(sp) / "llvm-aie"
        if (candidate / "bin" / opt_name).is_file():
            return candidate
        candidate2 = Path(sp) / "win64.o" / "tools" / "peano"
        if (candidate2 / "bin" / opt_name).is_file():
            return candidate2

    # 4. Fallback — return whatever aie gave us (will produce a clear error later)
    return default


class _DeviceManager:
    """Compatibility shim for new operators that access context.device_manager.device_type."""

    @property
    def device_type(self):
        return aie_utils.get_current_device()


def _detect_xrt_root() -> Path | None:
    """Detect XRT installation root on the current platform.

    Returns the path if found, None otherwise.
    The caller should ``source <root>/setup.sh`` (Linux) or ``<root>\\setup.bat``
    (Windows) before using XRT.
    """
    if sys.platform == "win32":
        # Common Windows locations
        candidates = [
            Path(os.environ.get("XILINX_XRT", "")),
            Path("C:/Xilinx/XRT"),
            Path("C:/Program Files/AMD/XRT"),
            Path(os.environ.get("LOCALAPPDATA", "")) / "Xilinx/XRT",
        ]
    else:
        candidates = [
            Path("/opt/xilinx/xrt"),
            Path(os.environ.get("XILINX_XRT", "")),
        ]
    for p in candidates:
        if p and p.is_dir():
            return p
    return None


@dataclass
class AIEContext:
    """Context for managing AIE operator compilation state.

    Attributes:
        base_dir: Repository root directory (three levels above this file).
        build_dir: Directory where compiled artifacts are written.
        mlir_verbose: Enable verbose MLIR output during compilation.
    """

    # Repo root: iron/common/../../.. = three levels up from this file.
    base_dir: ClassVar[Path] = Path(__file__).parent.parent.parent

    build_dir: Path = field(default_factory=lambda: Path(os.getcwd()) / "build")
    mlir_verbose: bool = False

    def __post_init__(self) -> None:
        """Normalize build_dir to a Path object."""
        self.build_dir = Path(self.build_dir)

    @property
    def device_manager(self):
        """Compatibility shim for operators that access context.device_manager.device_type."""
        return _DeviceManager()

    @property
    def compilation_rules(self):
        """Return the ordered list of compilation rules for this context.

        Returns:
            List of ``CompilationRule`` instances configured for the current
            mlir-aie and peano installation paths.
        """
        mlir_aie_dir = Path(aie.utils.config.root_path())
        peano_dir = _resolve_peano_dir()
        return [
            comp.FusePythonGeneratedMLIRCompilationRule(),
            comp.GenerateMLIRFromPythonCompilationRule(),
            comp.PeanoCompilationRule(self.build_dir, peano_dir, mlir_aie_dir),
            comp.ArchiveCompilationRule(self.build_dir, peano_dir, mlir_aie_dir),
            comp.AieccXclbinInstsCompilationRule(
                self.build_dir, peano_dir, mlir_aie_dir
            ),
            comp.AieccFullElfCompilationRule(self.build_dir, peano_dir, mlir_aie_dir),
        ]
