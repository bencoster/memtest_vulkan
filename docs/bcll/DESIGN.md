# BCLL diagnostics extension — design

Branch `bcll/diagnostics` of the `bencoster/memtest_vulkan` fork. See REVIEW.md for the upstream analysis
and USAGE.md for commands. The report format is in REPORT_SCHEMA.md and `report.schema.json`.

## 1. Goals and non-goals

**Goals.** Deep per-card diagnostics for the BC_LocalLLM rig, producing warranty-grade evidence:
- full card identity and health
- PCIe and thermal telemetry *during* load
- VRAM, ALU, PCIe-transfer and codec functional tests
- one machine-readable JSON report that BC_LocalLLM's GPU-health section ingests

**Non-goals.**
- Changing upstream's memory-test algorithm.
- Kernel-driver/MMIO/BAR access. WinRing0 and RWEverything are out of scope for security reasons.
  On Linux the deeper path is `lspci`/`setpci`/DCGM, documented in USAGE.md.
- A job queue or card reservation. BC_LocalLLM's `gpu.vram_test` owns "hold and clear the card".
  This tool runs on a card that is already cleared.

## 2. Upstream-mergeability rules

- **Default behaviour is unchanged.** Without a `--long-flag` the program runs exactly the
  upstream interactive/positional flow.
- **Isolation.** All new code lives in `src/bcll/**` (cargo feature `diagnostics`) and in
  `src/hooks.rs` (std-only, always compiled, a no-op unless its env vars are set).
  `cargo build --no-default-features` is upstream plus the inert hooks.
- **Touch points in upstream files** are kept to a handful of lines:
  1. `main.rs`: `mod hooks; #[cfg(feature="diagnostics")] mod bcll;` and a 3-line dispatch at the top of
     `main()`.
  2. `main.rs::test_device`: four `hooks::*` calls (started / error / progress / finished) and the
     deadline check next to `close::close_requested()`.
  3. `NamedComputeDevice` gains `pci` (domain:bus:device.function) and `device_uuid` (from
     `VkPhysicalDeviceIDProperties`), filled in the existing properties2 chain.
  4. `Cargo.toml`: features and optional dependencies.

## 3. Process model

```
memtest_vulkan --gpu 0000:01:00.0 --memtest 300 --compute 60 --report-json r.json
 └─ orchestrator (parent; never touches the GPU with work)
     ├─ inventory: Vulkan enumeration (upstream fn) + NVML (by PCI id) + NVAPI (Windows, by bus)
     ├─ safety gate: display-attached? VRAM in use? → refuse unless overridden
     ├─ health snapshot BEFORE (ECC volatile/aggregate DRAM/SRAM, remapper, retired pages)
     ├─ telemetry thread (NVML + NVAPI + Linux sysfs AER), prints a line every N s
     ├─ for each selected test: spawn a WORKER child, wait with a watchdog deadline
     │    memtest      : exe <-1-vkidx> <bytes>   (upstream child protocol, unchanged)
     │                   + env MEMTEST_VULKAN_EVENTS / MEMTEST_VULKAN_STOP_AFTER_S
     │    compute      : exe --bcll-worker compute  --vk-index i --seconds s --events f
     │    transfer     : exe --bcll-worker transfer --vk-index i --seconds s --events f
     │    cuda-memtest : exe --bcll-worker cuda-memtest --uuid U --pci P --seconds s --events f
     │    codec        : ffmpeg children (no worker needed)
     ├─ health snapshot AFTER, deltas, verdict
     └─ JSON report + exit code
```

Why child processes: a GPU fault (Xid, TDR, `DEVICE_LOST`, CUDA sticky error) kills only the worker. The
orchestrator still collects telemetry, takes the AFTER snapshot (which is exactly when ECC/remap deltas
appear) and writes the report. A watchdog kills a worker that overruns `seconds + grace`, so the fork
does not inherit upstream's infinite fence wait (R5).

Workers report through a **JSON-lines events file**: one object per line, written and flushed per
event. A crash leaves a truncated, still-parseable file. The memtest child's events are written by
`hooks.rs` with hand-formatted JSON (numbers and fixed strings only), so upstream builds need no serde.

## 4. Card selection and safety

`--gpu SPEC`:

| Form | Meaning |
|---|---|
| `0`, `index:0` | Nth GPU in **PCI bus order**. That equals the nvidia-smi index and CUDA with `CUDA_DEVICE_ORDER=PCI_BUS_ID`, the BC_LocalLLM numbering. It is deliberately NOT upstream's label number (REVIEW R7). |
| `0000:81:00.0`, `81:00.0`, `pci:81:00.0` | PCI address, case-insensitive; domain defaults to 0 |
| `GPU-6727…`, `uuid:GPU-6727…` | NVML/CUDA UUID; a unique prefix is accepted |
| `vk:N` | upstream label number (for parity with upstream output) |

Identity sources:
- **PCI**: `VK_EXT_pci_bus_info`, cross-checked with NVML `pci_info`.
- **UUID**: Vulkan `deviceUUID` (equal to the CUDA/NVML UUID on NVIDIA), otherwise NVML.

Refusals (exit code 3):
- **Display-attached** (NVML `is_display_active || is_display_connected`): refused unless
  `--allow-display`. If the display state is **unknown** (no NVML, non-NVIDIA), the tool fails closed
  and still requires `--allow-display`.
- **VRAM in use** above `--max-used-mb` (default 1024), or compute processes listed by NVML: refused
  unless `--allow-busy`. This catches "the LLM is still loaded on that card".

## 5. Card details (feature `nvml`, crate `nvml-wrapper` 0.13, NVML dynamically loaded)

| Group | Fields |
|---|---|
| Identity | name, UUID, serial, board part number, VBIOS, InfoROM, PCI bus id, device/subsystem id, architecture, compute capability, cores, bus width |
| Driver | driver version, CUDA driver version, NVML version, driver model (Windows) |
| Memory | total/used/free/reserved |
| ECC | current/pending mode; volatile + aggregate corrected/uncorrected totals; per location DRAM (`Device`) and SRAM, aggregate |
| Remapper | correctable/uncorrectable remapped rows, pending, failure (raw `nvmlDeviceGetRemappedRows`) and the bank-availability histogram (raw `nvmlDeviceGetRowRemapperHistogram`) |
| Retired pages | Pre-Ampere mechanism; read and reported, normally empty on Ampere/Ada |
| Clocks | current and max graphics/SM/memory/video; GPC and memory VF offsets |
| Power | usage, management limit, enforced limit, default limit, total energy |
| Thermal | GPU core temperature, slowdown/shutdown thresholds; memory temperature via field 82 (null on GeForce) |
| Fan | speed % per fan |
| Throttle reasons | bitmask plus decoded names |
| PCIe | gen/width current and max, max link speed |
| PCIe field counters | replay, replay-rollover, L0→recovery, correctable, NAK sent/received, bad TLP/DLLP, non-fatal, fatal. Reported as null when the driver returns NotSupported, which GeForce often does. |

Every field is `Option`: one failing query never fails the snapshot.

## 6. Telemetry during tests

A thread samples every `--telemetry-ms` (default 1000). Each sample records:
- GPU and memory-junction temperature, power, SM and memory clocks
- GPU/memory/encoder/decoder utilisation
- PCIe gen/width, TX/RX KB/s, replay counter
- ECC volatile corrected/uncorrected, remapped rows
- throttle bits, fan
- Linux AER totals

The thread prints one compact line every `--telemetry-print-s` (default 10 s). The summary has:
- min/max temperature, power and clocks
- PCIe replay delta and per-counter deltas
- link changes, plus the lowest gen/width observed **while busy** (GPU util ≥ 50 %). Idle downshifts
  are normal ASPM/power management and are not flagged.
- ECC, remap and AER deltas
- throttle reasons seen (count of samples per reason)

**Memory-junction temperature** (feature `nvapi`, Windows only; owner decision 2026-10-10). NVML does not
expose GDDR6X junction temperature on GeForce. The tool calls the undocumented
`NvAPI_GPU_ThermalGetSensors` (interface 0x65FE3AAD, struct v2, 168 bytes) through `nvapi64.dll`
(`nvapi_QueryInterface`), using the mapping LibreHardwareMonitor uses:
- valid-mask probing
- value/256 °C
- hotspot = channel 1
- junction = channel 9 on Ampere, 7 on Ada, 2 on Blackwell

GPUs are matched by PCI bus (`NvAPI_GPU_GetBusId`). Any non-OK status (including
INCOMPATIBLE_STRUCT_VERSION) disables the source silently. The report marks the source
`"nvapi_undocumented"`. `--no-nvapi` turns it off. On Linux there is no supported userspace path: NVML
field 82 is datacenter-only, and the community tools read it through BAR0 MMIO, which is out of scope.
This is documented.

**Linux AER.** The tool reads `/sys/bus/pci/devices/<bdf>/aer_dev_{correctable,nonfatal,fatal}` (key/value
lines, `TOTAL_ERR_*`) per sample and before/after, and reports the deltas. Root-port AER is noted in
USAGE.md.

## 7. Tests

### 7.1 VRAM, Vulkan (upstream memtest, unchanged algorithm)
The upstream child runs with a deadline (`--memtest SECONDS`, default 300 = upstream's "standard
5-minute test"). `--memtest-bytes` caps the size.

Events:
- `memtest_started`: allocation, window size/count, tested bytes. Coverage = `window_size*count`, which
  corrects REVIEW R4's overstatement.
- `memtest_error`: absolute byte range, count *including* all-ones words (corrects R1), and the
  bit histograms.
- `memtest_progress`, `memtest_finished`.

Status comes from the exit code: `RUNTIME_ERRORS` → fail; `RUNTIME_ABORT` after `INITED_OK` →
incomplete/error. Device-lost on an ECC card plus an ECC uncorrected delta → **fail** with reason
"uncorrectable ECC".

### 7.2 Compute (Vulkan; WGSL compiled at build time like upstream)
Each invocation runs a deterministic chain of `iters` steps:
- integer (xorshift32 + LCG + rotate)
- FP32 (`fma` chain + `sqrt`)

Verification:
- **Integer**: bit-exact against a CPU reference (`compute_ref.rs`).
- **FP32 by consensus**: problem `p = gid % P` is computed by `N/P` replicas spread over every SM. The
  majority value is the reference, and a replica that disagrees is an error. A CPU f32 reference with a
  relative tolerance of 1e-3 also flags systematic error.

This avoids depending on the driver's fusing/contracting choices, which Vulkan allows. Each dispatch
lasts a few ms (TDR-safe). The seed changes per dispatch, and the test loops for `--compute SECONDS`.
Errors carry the invocation index, workgroup and xor mask.

### 7.3 PCIe transfer (Vulkan copy engine)
- The CPU fills a 256 MiB host-visible buffer with a seeded pattern.
- `vkCmdCopyBuffer` copies host→device, then device→a second host buffer, timed separately.
- The CPU compares the round trip word by word.

The test reports H2D/D2H GB/s and corrupted words. NVML replay/throughput telemetry runs alongside, so
PCIe errors are correlated with real traffic. The memtest itself creates almost no PCIe traffic.

### 7.4 VRAM, CUDA (feature `cuda`)
**Decision: implemented in Rust here, not in the Python companion.** The owner asked for driver-API
VMM coverage:
- `cuMemCreate` chunks (granularity-aligned, 512 MiB, halving down to the granularity to grab the tail)
- mapped into one `cuMemAddressReserve` range, with `cuMemSetAccess`

torch's caching allocator cannot do that or account for coverage. The tool reports `tested_bytes`,
`total_bytes`, `free_bytes_at_start` and `reserved_or_used_bytes = total - free`.

Implementation:
- **No new crate.** `nvcuda.dll` / `libcuda.so.1` is loaded with the existing `libloading`; the ~25
  driver-API symbols are declared in `cuda/ffi.rs`. cudarc was considered, but it needs a CUDA-version
  feature or build-time nvcc and pulls a large API surface. This path needs only the driver, which is
  always present.
- **Kernels.** The CUDA C source `cuda/kernels.cu` is compiled once with nvcc to **PTX**
  (`cuda/kernels.ptx`, `compute_80`, checked in). The driver JITs it at load (`cuModuleLoadData`), so no
  toolkit is needed at runtime. The regeneration command is in USAGE.md, and `ptxas -arch=sm_86/sm_89`
  validates it offline.
- **Patterns per pass:**
  - own-address (64-bit index hash)
  - moving inversions (verify X and write ~X, then verify ~X and write X)
  - walking ones
  - random (index-hash)

  A pass verifies the whole mapped range. Errors record word offset, expected and actual (first 1024),
  plus a total count and a per-bit histogram.
- The worker selects its card with `CUDA_VISIBLE_DEVICES=<UUID>` and asserts the PCI id matches.

The CPU reference of the pattern functions is unit-tested (`cuda/patterns.rs`); the same constants
appear in `kernels.cu`.

**Division of labour with BC_LocalLLM `gpu.vram_test` (Python/torch, branch feat/gpu-vram-test).** The
Python side owns the job-queue integration and card hold/clear/verify/release (`card_hold`, `session`).
Through its `runner` "external tester hook" it invokes this binary with `--report-json` and ingests the
shared schema. Its own torch pattern test stays as the no-native-binary fallback and writes the same
schema with `engine: "torch"`. Neither side duplicates the other's queue or VMM logic.

### 7.5 Codecs (ffmpeg)
The tool shells out to `--ffmpeg PATH`, or `ffmpeg` on PATH. On BC_LocalLLM the path is
`web/shared.py:FFMPEG`. For each of h264/hevc/av1:
1. Probe `-encoders`/`-decoders` for `<c>_nvenc` / `<c>_cuvid`.
2. Encode N frames of `testsrc2` 1080p on the selected GPU.
3. Decode with `<c>_cuvid` and compute PSNR against the regenerated source in one filter graph.

Pass requires average PSNR ≥ 30 dB **and** the decoded frame count equal to the encoded count. The GPU is
pinned with `CUDA_VISIBLE_DEVICES=<UUID>` plus `-gpu 0`. "No capable devices" and "unsupported" become
status `unsupported` (for example AV1 encode on Ampere), which does not affect the verdict. The NVIDIA
Video Codec SDK was rejected: a large native integration for no extra fault coverage over a round trip.

## 8. Report and verdict

Schema `bcll.gpu_diag.report` v1 (REPORT_SCHEMA.md). The verdict uses PASS/FAIL/INCOMPLETE, the same
vocabulary as `gpu_diag.record`.

**FAIL** if any of:
- a test failed
- ECC uncorrected volatile delta > 0
- uncorrectable remapped rows delta > 0, or remap failure / pending
- retired pages delta > 0
- AER fatal/non-fatal delta > 0

**Warnings** (the verdict stays PASS):
- PCIe replay delta > 0
- AER correctable delta > 0
- ECC corrected delta > 0
- link below max **while busy**
- hardware/thermal slowdown throttling
- memory junction ≥ 100 °C (GDDR6X throttles near 110 °C)

**INCOMPLETE** if, without a FAIL, any test is incomplete/error (timeout, device lost without ECC
evidence, worker crash).

Exit codes:

| Code | Meaning |
|---|---|
| 0 | PASS |
| 1 | FAIL |
| 2 | INCOMPLETE |
| 3 | refused / usage error |
| 4 | internal error |

## 9. Proposed upstream patches (not applied here, to keep the upstream path byte-identical)

| Finding | Patch |
|---|---|
| R1 | `total_errors += actual_ff` |
| R2 | `err_bitcount[min(one_bits, 31)]`, or widen the array to 33 |
| R5 | finite fence timeout (for example 10 s) → treat as device hang |
| R6 | reject non-numeric argv1 with usage text instead of entering interactive mode |
| R9 | enable validation only under `verbose` |

## 10. New dependencies

| Crate | Version | Feature | Why |
|---|---|---|---|
| `serde` (+derive) | 1.0.x | diagnostics | report/event (de)serialisation; ubiquitous |
| `serde_json` | 1.0.x | diagnostics | JSON report and event parsing; ubiquitous |
| `nvml-wrapper` | 0.13.0 | nvml | maintained safe NVML binding; loads `nvml.dll`/`libnvidia-ml.so.1` at runtime, so non-NVIDIA hosts just report "unavailable". Pulls `nvml-wrapper-sys`, `libloading 0.8`, `bitflags`, `thiserror`, `static_assertions`, `wrapcenum-derive`. |

NVAPI and CUDA use the existing `libloading` dependency and need no new crates.
