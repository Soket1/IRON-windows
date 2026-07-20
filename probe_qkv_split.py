"""
Minimal probe: QKV weights via MemTile split (FFLM-style).

Mirrors the PROVEN O_proj split pattern (layer_dense/design.py L466-478):
  2 shim sources -> MemTile split -> 8 col sub-fifos (4 per source)

Tests resolve_program only -- no hardware run.
Success = "OK" means IRON can route this split topology.
"""
import os
import sys

# -- Driver / SDK paths -----------------------------------------------
driver_dir = r"C:\Windows\System32\DriverStore\FileRepository\kipudrv.inf_amd64_1a1aa059597c4810"
if os.path.exists(driver_dir):
    os.add_dll_directory(driver_dir)
xrt_sdk_dlls = r"C:\Users\Kuhnya\Downloads\xrt_windows_sdk\xrt_sdk\xrt\bin\condautils"
if os.path.exists(xrt_sdk_dlls):
    os.add_dll_directory(xrt_sdk_dlls)

import aie.utils.config as aie_config
peano_dir = r"C:\ProgramData\miniforge3\envs\ryzen-ai-1.7.1\Lib\site-packages\win64.o\tools\peano"
if os.path.exists(peano_dir):
    aie_config.peano_install_dir = lambda: peano_dir

sys.path.insert(0, "IRON-windows")

import numpy as np
from ml_dtypes import bfloat16
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2, Tile
from aie.helpers.taplib import TensorAccessPattern
from aie.helpers.dialects.scf import _for as range_

# -- Llama-3.2-1B QKV geometry ----------------------------------------
e = 2048             # embed_dim
g = 32               # group_size
m_input_qkv = 2     # rows per tile
cols = 8
n_shim_qkv = 2      # 2 shim sources (like O_proj n_shim_o=2)
cols_per_shim = cols // n_shim_qkv  # 4 cols per source

packed_qkv_tile_bytes = m_input_qkv * e // 2 + m_input_qkv * (e // g) * 2  # 2304 B

# -- IRON type annotations (square brackets, not round!) ---------------
dtype_packed = np.dtype[np.uint8]
dtype_vec = np.dtype[bfloat16]

L1_Aq_ty = np.ndarray[(packed_qkv_tile_bytes,), dtype_packed]          # 1 tile
Aqkv_src_ty = np.ndarray[(cols_per_shim * packed_qkv_tile_bytes,), dtype_packed]  # 4 tiles
L1_Bq_ty = np.ndarray[(e,), dtype_vec]     # activation
L1_Cq_ty = np.ndarray[(m_input_qkv,), dtype_vec]  # output
BO_ty = np.ndarray[(100000,), dtype_vec]   # dummy BO

# -- NPU2 device -------------------------------------------------------
dev_ty = NPU2()

# -- QKV weight fifos: 2 shim -> MemTile split -> 8 per-col sub-fifos --
qkv_mem_cols = [0, 4]  # MemTile columns for split

aqkv_src_fifos = [
    ObjectFifo(Aqkv_src_ty, name=f"Aqkv_src_{s}", depth=2)
    for s in range(n_shim_qkv)
]

Aqkv_sub = [None] * cols
for s in range(n_shim_qkv):
    subs = aqkv_src_fifos[s].cons().split(
        [j * packed_qkv_tile_bytes for j in range(cols_per_shim)],
        obj_types=[L1_Aq_ty] * cols_per_shim,
        names=[f"Aqkv_{s * cols_per_shim + j}" for j in range(cols_per_shim)],
        placement=Tile(col=qkv_mem_cols[s], row=1),
    )
    for j in range(cols_per_shim):
        Aqkv_sub[s * cols_per_shim + j] = subs[j]

# -- Activation broadcast (bq) -----------------------------------------
bq_l3l2 = ObjectFifo(L1_Bq_ty, name="bq_L3L2", depth=1)
bq_mem = bq_l3l2.cons().forward(
    name="bq_mem", depth=1, placement=Tile(col=2, row=1)
)

# -- Output fifos (one per col) -----------------------------------------
Cqkv_fifos = [ObjectFifo(L1_Cq_ty, name=f"Cqkv_{i}", depth=2) for i in range(cols)]

# -- Worker body (minimal: acquire weight+activation, produce output) ---
def worker_body(a_in, b_in, c_out):
    for _ in range_(0, 1):
        a = a_in.acquire(1)
        b = b_in.acquire(1)
        c = c_out.acquire(1)
        c_out.release(1)
        b_in.release(1)
        a_in.release(1)

workers = [
    Worker(
        worker_body,
        [Aqkv_sub[i].cons(), bq_mem.cons(), Cqkv_fifos[i].prod()],
        placement=Tile(col=i, row=2),
    )
    for i in range(cols)
]

# -- Runtime sequence ---------------------------------------------------
rt = Runtime()
with rt.sequence(BO_ty) as bo:
    rt.start(*workers)
    # Fill QKV weights: 2 shim sources, each feeds 4 cols
    for s in range(n_shim_qkv):
        tap = TensorAccessPattern(
            tensor_dims=(1, 100000),
            offset=0,
            sizes=[1, 1, 1, cols_per_shim * packed_qkv_tile_bytes // 2],
            strides=[0, 0, 0, 1],
        )
        rt.fill(aqkv_src_fifos[s].prod(), bo, tap)
    # Fill activation broadcast
    tap_b = TensorAccessPattern(
        tensor_dims=(1, 100000),
        offset=0,
        sizes=[1, 1, 1, e],
        strides=[0, 0, 0, 1],
    )
    rt.fill(bq_l3l2.prod(), bo, tap_b)
    # Drain outputs
    for i in range(cols):
        tap_c = TensorAccessPattern(
            tensor_dims=(1, 100000),
            offset=0,
            sizes=[1, 1, 1, m_input_qkv],
            strides=[0, 0, 0, 1],
        )
        rt.drain(Cqkv_fifos[i].cons(), bo, tap_c)

# -- Resolve ------------------------------------------------------------
try:
    m = Program(dev_ty, rt).resolve_program(SequentialPlacer())
    print("OK: QKV MemTile-split resolves (2 shim -> 8 cols via split)")
except Exception as e:
    msg = str(e)
    print(f"FAIL: {msg[:300]}")
    import traceback
    traceback.print_exc()
