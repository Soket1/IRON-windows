"""Run DMA echo test on XDNA NPU.

Verifies that DMA path works correctly by sending known data through
a minimal kernel (memcpy or concat) and checking the output.

Uses full ELF path for NPU dispatch (pyxrt.elf + hw_context).

Usage:
    python test_echo.py --version 1  # single ObjectFifo (bo_in -> tile -> bo_out)
    python test_echo.py --version 2  # dual ObjectFifo (bo_a + bo_b -> tile -> bo_out)
"""
import argparse
import ctypes
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import pyxrt


def load_elf_to_pyxrt(elf_path: Path):
    """Load ELF file and create pyxrt.elf object."""
    elf_data = elf_path.read_bytes()
    elf_u8 = np.frombuffer(elf_data, dtype=np.uint8)
    ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object
    ctypes.pythonapi.PyCapsule_New.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p,
    ]
    capsule = ctypes.pythonapi.PyCapsule_New(elf_u8.ctypes.data, None, None)
    return pyxrt.elf(capsule, len(elf_data))


def run_echo_test(version: int, n: int = 256):
    build_dir = Path(__file__).resolve().parent.parent.parent.parent / f"build_echo_v{version}"
    elf_path = build_dir / f"echo_v{version}.elf"

    if not elf_path.exists():
        print(f"ERROR: ELF not found: {elf_path}")
        print(f"Run compile_echo.py --version {version} first.")
        sys.exit(1)

    print(f"=== DMA Echo Test v{version} (N={n}) ===")
    print(f"ELF: {elf_path}")
    print()

    # ===== Open device + load ELF =====
    device = pyxrt.device(0)
    xrt_elf = load_elf_to_pyxrt(elf_path)

    # ===== Create hw_context + kernel =====
    hw_ctx = pyxrt.hw_context(device, xrt_elf)
    kernel = pyxrt.ext.kernel(hw_ctx, "main:sequence")
    print(f"Kernel loaded: main:sequence")

    # ===== Allocate BOs =====
    elem_bytes = 2  # bf16

    if version == 1:
        bo_in = pyxrt.bo(device, n * elem_bytes, pyxrt.bo.normal, kernel.group_id(0))
        bo_out = pyxrt.bo(device, n * elem_bytes, pyxrt.bo.normal, kernel.group_id(1))
    elif version == 2:
        bo_a = pyxrt.bo(device, n * elem_bytes, pyxrt.bo.normal, kernel.group_id(0))
        bo_b = pyxrt.bo(device, n * elem_bytes, pyxrt.bo.normal, kernel.group_id(1))
        bo_out = pyxrt.bo(device, 2 * n * elem_bytes, pyxrt.bo.normal, kernel.group_id(2))

    # ===== Fill buffers =====
    MARKER_A = 0xDEAD
    MARKER_B = 0xBEEF
    PATTERN = 0x3E80  # bf16 ~0.25

    if version == 1:
        in_map = bo_in.map()
        in_arr = np.frombuffer(in_map, dtype=np.uint16)
        in_arr[:] = PATTERN
        in_arr[0] = 0x3E39
        in_arr[1] = 0x3D9F
        print(f"Input[0:8]  = {[hex(x) for x in in_arr[:8]]}")

        out_map = bo_out.map()
        out_arr = np.frombuffer(out_map, dtype=np.uint16)
        out_arr[:] = 0x0000

        bo_in.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, n * elem_bytes, 0)
        bo_out.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, n * elem_bytes, 0)

    elif version == 2:
        a_map = bo_a.map()
        a_arr = np.frombuffer(a_map, dtype=np.uint16)
        a_arr[:] = PATTERN
        a_arr[0] = MARKER_A

        b_map = bo_b.map()
        b_arr = np.frombuffer(b_map, dtype=np.uint16)
        b_arr[:] = PATTERN
        b_arr[0] = MARKER_B

        out_map = bo_out.map()
        out_arr = np.frombuffer(out_map, dtype=np.uint16)
        out_arr[:] = 0x0000

        print(f"A[0:8]  = {[hex(x) for x in a_arr[:8]]}")
        print(f"B[0:8]  = {[hex(x) for x in b_arr[:8]]}")

        bo_a.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, n * elem_bytes, 0)
        bo_b.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, n * elem_bytes, 0)
        bo_out.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, 2 * n * elem_bytes, 0)

    # ===== Run =====
    print()
    print("Running kernel on NPU...")
    run = pyxrt.run(kernel)
    if version == 1:
        run.set_arg(0, bo_in)
        run.set_arg(1, bo_out)
    elif version == 2:
        run.set_arg(0, bo_a)
        run.set_arg(1, bo_b)
        run.set_arg(2, bo_out)
    run.start()
    state = run.wait(10000)  # 10s timeout
    print(f"Kernel finished, state={state}")

    if state != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
        print(f"Kernel execution failed: {state}")
        return False

    # ===== Read back output =====
    out_size = (2 * n * elem_bytes) if version == 2 else (n * elem_bytes)
    bo_out.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, out_size, 0)

    out_map = bo_out.map()
    out_arr = np.frombuffer(out_map, dtype=np.uint16)

    print()
    print("=== RESULTS ===")
    print(f"Output[0:16] = {[hex(x) for x in out_arr[:16]]}")
    print()

    success = True

    if version == 1:
        match_count = np.sum(out_arr[:n] == PATTERN)
        print(f"Match with input pattern (0x{PATTERN:04X}): {match_count}/{n}")
        if out_arr[0] == 0x3E39:
            print(f"Output[0] = 0x3E39 (matches input[0])")
        else:
            print(f"Output[0] = 0x{out_arr[0]:04X} (expected 0x3E39)")

        if match_count == n:
            print("DMA ECHO v1 PASSED - output matches input!")
        elif match_count > n * 0.9:
            print("MOSTLY OK - some data corruption")
        else:
            zero_count = np.sum(out_arr[:n] == 0)
            if zero_count == n:
                print("FAILED - output is all zeros (DMA didn't write)")
            else:
                print(f"FAILED - DMA reads/writes wrong data")
                print(f"   Expected: 0x{PATTERN:04X} x {n}")
                print(f"   Got:      {match_count} matches, {n - match_count} mismatches")
            success = False

    elif version == 2:
        a_match = np.sum(out_arr[:n] == PATTERN)
        b_match = np.sum(out_arr[n:2*n] == PATTERN)
        a_marker_ok = out_arr[0] == MARKER_A
        b_marker_ok = out_arr[n] == MARKER_B

        print(f"A region match: {a_match}/{n}  marker[0]: 0x{out_arr[0]:04X} {'OK' if a_marker_ok else 'FAIL'}")
        print(f"B region match: {b_match}/{n}  marker[N]: 0x{out_arr[n]:04X} {'OK' if b_marker_ok else 'FAIL'}")

        if a_match == n and b_match == n:
            print("DMA ECHO v2 PASSED - dual ObjectFifo concat works!")
        else:
            print("FAILED - dual DMA path broken")
            success = False

    print()
    return success


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=int, choices=[1, 2], required=True)
    ap.add_argument("--n", type=int, default=256)
    args = ap.parse_args()
    ok = run_echo_test(args.version, args.n)
    sys.exit(0 if ok else 1)
