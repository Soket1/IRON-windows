---
name: XRT binary analysis for SMMU alignment
description: Deep RE of ipustack.sys, xrt_core.dll, xrt_coreutil.dll — found D3DKMT allocation path, carveout flag, xrt.ini config, buffer_import class, potential 64K alignment levers
type: reference
---

## XRT-MCDM Binary Deep Analysis (2026-05-24)

Reverse-engineered strings from the XRT runtime binaries to find
undocumented SMMU alignment levers for Priority 9.

### Binaries scanned

| File | Path | Size |
|---|---|---|
| ipustack.sys | `C:\Windows\System32\DriverStore\FileRepository\kipudrv.inf_amd64_1a1aa059597c4810\` | 433 KB |
| xrt_core.dll | same dir | 966 KB |
| xrt_coreutil.dll | same dir | 2.6 MB |
| xrt_bo.h | XRT SDK headers | API reference |

Driver is **not** `amdxdna.sys` — it's `ipustack.sys` (MCDM-based
IPU driver, ClassGuid `{F01A9D53-3FF6-48D2-9F97-C8A7004BE10C}` = ComputeAccelerator).

### ipustack.sys (kernel driver) findings

- `MmAllocateContiguousMemorySpecifyCache` — allocates physically
  contiguous memory with selectable cache policy. This is the
  kernel API that CAN give large-page-aligned allocations when
  the requested size is a multiple of the large page.
- `IpuKmdDevice::OpenAllocation` — allocation entry point
- `KeFlushIoBuffers` — I/O buffer flush
- `MmBuildMdlForNonPagedPool` — MDL-based memory descriptor
- **No** PageSize/HugePage/Align/SMMU/registry strings in the driver

### xrt_core.dll (XRT runtime) findings

**D3DKMT allocation stack (dxgkrnl-based):**

| API | Role |
|---|---|
| `D3DKMTCreateAllocation2` | Primary allocation path via dxgkrnl |
| `D3DKMTMapGpuVirtualAddress` | Maps into device VA space |
| `D3DKMTDestroyAllocation2` | Frees allocation |
| `D3DKMTOpenResourceFromNtHandle` | **Import via NT handle exists!** |
| `D3DKMTQueryResourceInfoFromNtHandle` | Query imported resource info |
| `D3DKMTQueryAdapterInfo` + `KMTQAITYPE_QUERYREGISTRY` | **XRT reads registry through D3DKMT** |
| `D3DKMTLock2` / `D3DKMTMakeResident` | Lock + make resident for DMA |
| `D3DKMTInvalidateCache` | Cache invalidation |

**Key strings:**

- `"internal error: CACHEABLE flag unexpected here"` — confirms
  cacheable path is broken/unexpected in this build
- `"memory manager could not page-in all of the required allocations"` —
  memory pressure related
- `"userptr is not aligned"` — alignment validation exists
- `"No host memory for imported buffer"` — import path exists

**Source path leaked:** `W:\src\sw-stack\XRT-MCDM\` — this is
XRT-MCDM, not standard XRT. Different allocator path from Linux XRT.

### xrt_coreutil.dll (XRT core utilities) findings

**INI configuration system:**

- `XRT_INI_PATH` — env var to set custom xrt.ini path
- `SDACCEL_INI_PATH` — legacy alias for the same
- `"[XRT] Failed to read xrt.ini: "` — XRT reads xrt.ini from
  CWD or XRT_INI_PATH on every launch
- INI parser: boost::property_tree::ini_parser

**Runtime.* settings found in binary:**

| Setting | Notes |
|---|---|
| `Runtime.xrt_bo` | BO management setting (unknown values) |
| `Runtime.ert_slotsize` | ERT slot size |
| `Runtime.verbosity` | Log verbosity |
| `Runtime.cpu_affinity` | CPU affinity for XRT threads |
| `Runtime.platform_repo_path` | xclbin search path |
| `Runtime.rw_shared` | Shared read/write mode |
| `Runtime.npu_sync_destroy_allocation` | NPU-specific sync destroy |
| `Runtime.hardware_context_type` | HW context type selector |
| `Runtime.kernel_channels` | Kernel channel config |
| `Runtime.runtime_log` | Runtime logging |
| `Runtime.trace_logging` | Trace logging |
| `Runtime.usage_metrics_logging` | Usage metrics |

**Debug.* settings:**

| Setting | Notes |
|---|---|
| `Debug.xrt_debug` | General XRT debug |
| `Debug.aie_debug` / `aie_trace` / `aie_profile` | AIE debug/trace |
| `Debug.host_trace` | Host-side tracing |
| `Debug.dtrace_lib_path` | Dynamic trace library |
| `Debug.dump_control_codes` | Dump control codes |
| `Debug.dump_scratchpad_mem` | Dump scratchpad |

**BO type classes found (xrt_coreutil.dll RTTI):**

| Class | Description |
|---|---|
| `buffer_hbuf` | Host buffer |
| `buffer_kbuf` | Kernel buffer |
| `buffer_ubuf` | User-pointer buffer |
| `buffer_dbuf` | Device buffer |
| `buffer_nodma` | No-DMA buffer |
| `buffer_import` | **Imported buffer** |
| `buffer_clone` | Cloned buffer |
| `buffer_sub` | Sub-buffer |
| `buffer_xbuf` | Unknown xbuf |

**`xrt::bo::flags::carveout` = `XCL_BO_FLAGS_KERNBUF` (bit 25).**
This is NOT the same as cacheable (bit 24) or host_only (bit 29).
Description in header: "reserved memory pool. For AMD Ryzen NPU
this memory is allocated from a host memory carveout pool."
Uses a DIFFERENT allocator inside ipustack.sys — possibly
`MmAllocateContiguousMemorySpecifyCache` which CAN give large-page
alignment.

**xrt::bo API surface (from xrt_bo.h):**

- `bo(hwctx, sz, flags, grp)` — main allocator
- `bo(device, export_handle)` — **import constructor exists!**
- `export_buffer()` — returns `uint64_t` handle on Windows
- `address()` — returns device VA
- `alignment()` — static method returning BO alignment requirement
- `get_flags()` — returns flags used at construction
- Async variant: `async(dir, sz, offset)` returns `async_handle`

**Export handle on Windows is `uint64_t`** (line 141 of xrt_bo.h),
on Linux it's `int32_t` (fd-based). Windows uses NT handle.

### Unexplored levers identified

1. **`xrt::bo::flags::carveout`** — different allocator in kernel
   driver, may give 64K alignment. Quick test: allocate weight BO
   with carveout, check `address() % 65536`.

2. **`xrt.ini` with `Runtime.xrt_bo`** — unknown setting, might
   control BO allocation strategy. Create xrt.ini in CWD.

3. **`Runtime.hardware_context_type`** — might select different
   allocation mode for hw_context.

4. **`D3DKMTOpenResourceFromNtHandle`** — import path confirmed
   in binary. Create D3D12 resource, export as NT handle, import
   into XRT. Requires testing if MCDM driver honors the alignment.

5. **`xrt::bo::alignment()`** — what value does this return?
   If it returns 64K, the driver already knows about large pages.

### Registry search results

- `HKLM\SYSTEM\CurrentControlSet\Services\amdxdna` — does NOT exist
- `HKLM\SYSTEM\CurrentControlSet\Services\AMDXDNA` — does NOT exist
- `HKLM\SYSTEM\CurrentControlSet\Services\kipudrv` — does NOT exist
- `HKLM\SOFTWARE\AMD` — does NOT exist on this machine
- `HKLM\SYSTEM\CurrentControlSet\Control\Class\{F01A9D53-...}` — empty

Registry has NO configurable parameters for the XDNA/IPU driver.
All tuning is either in xrt.ini or hardcoded in binaries.

### Conclusion

Priority 9 (64K SMMU alignment) is confirmed dead for the
`host_only` + `UserPtrBO` + `CLWB` paths. But `carveout` flag
and `xrt.ini Runtime.*` settings are unexplored and could provide
the lever. The D3DKMT import path (`D3DKMTOpenResourceFromNtHandle`)
also exists in the binary and is worth testing.
