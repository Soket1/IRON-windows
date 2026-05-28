# IRON-windows — Windows Install Guide

Building and running IRON operators on a Windows host requires a few
external dependencies that are **not** bundled with the `llvm-aie` and
`mlir_aie` Python wheels. Without them, compilation will fail at the
`KernelObjectArtifact` rename-symbols step (`llvm-objcopy not found`) or
the final xclbin packaging step (`xclbinutil not found, skipping xclbin
generation`).

This document lists the minimum set and explains the discovery order
the build system uses, so you can either install tools to standard
locations or point at non-standard ones via environment variables.

## Required external tools

| Tool                | Why                                                       | Source                                                   |
|---------------------|-----------------------------------------------------------|----------------------------------------------------------|
| `llvm-objcopy.exe`  | Symbol prefixing/renaming on `KernelObjectArtifact`       | [LLVM standalone install](https://releases.llvm.org/)    |
| `xclbinutil.exe`    | xclbin packaging (`MEM_TOPOLOGY`, `AIE_PARTITION` add)    | [XRT SDK for Windows](https://github.com/Xilinx/XRT)     |

The `llvm-aie` wheel **does** ship `llvm-nm.exe`, `clang.exe`, `lld.exe`,
etc. — only `llvm-objcopy.exe` is missing. The `mlir_aie` wheel does
not ship `xclbinutil.exe`.

## Recommended install layout

The build system probes the following paths automatically:

**For `llvm-objcopy.exe`:**
1. `C:\Program Files\LLVM\bin\` (LLVM Windows installer default)
2. `C:\Program Files (x86)\LLVM\bin\`
3. System `PATH`

**For `xclbinutil.exe`:**
1. `%XRT_ROOT%` or `%XILINX_XRT%` (env var, if set)
2. `C:\Users\<you>\Downloads\xrt_windows_sdk\xrt_sdk\xrt\` (matches the layout of the AMD-distributed XRT SDK zip)
3. `C:\Xilinx\XRT\bin\`
4. System `PATH`

If your install lives elsewhere, the simplest fix is to set:
```
set XRT_ROOT=C:\path\to\xrt_sdk\xrt
```
or prepend the directory to `PATH` before invoking `python compile.py`.

## Setup steps

1. **Install Python dependencies** as usual:
   ```
   python -m pip install -r requirements.txt
   ```

2. **Install LLVM for Windows** (provides `llvm-objcopy.exe`):
   - Download the `LLVM-*-win64.exe` installer from
     [https://releases.llvm.org/](https://releases.llvm.org/).
   - During install, accept the default path
     (`C:\Program Files\LLVM\`); no `PATH` change is required because
     the build system probes that path directly.

3. **Install XRT SDK for Windows** (provides `xclbinutil.exe`):
   - Either:
     - extract the AMD-distributed `xrt_windows_sdk.zip` to
       `C:\Users\<you>\Downloads\xrt_windows_sdk\`, **or**
     - set `XRT_ROOT=<your-extracted-xrt-sdk-path>` (the directory that
       contains `xclbinutil.exe`).

4. **Verify** by running the smoke compile script:
   ```
   python _smoke_compile_post_attn_fullelf.py
   ```
   You should see `COMPILE OK` and a `.elf` artifact in `build/`.

## Troubleshooting

- **`llvm-objcopy not found`** at compile time, even though it is
  installed — make sure the install path matches one of the probed
  locations above, or add the directory to `PATH`.
- **`Warning: xclbinutil not found, skipping xclbin generation`** — the
  build system's `PATH` augmentation did not include your XRT install.
  Set `XRT_ROOT` and rebuild.
- The probe logic lives in `iron/common/compilation/base.py::_find_tool`
  and `AieccCompilationRule.__init__`. You can extend the candidate
  list there if your install layout is unusual.
