# memtest_vulkan (BCLL fork): build and use

Fork branch `bcll/diagnostics`. See REVIEW.md (upstream analysis), DESIGN.md (architecture) and SCHEMA.md
plus `report.schema.json` (JSON output).

> **Safety first.** Without `--flags` the binary is upstream memtest_vulkan. That includes its
> interactive mode, which **autoselects a GPU after 10 s and starts testing** (REVIEW R6/R7).
> On the BC_LocalLLM rig always use the long-flag CLI. Never run the bare binary, and never run upstream's
> `memtest_vulkan --help`.

## 1. Build

### Windows (MSVC toolchain)

```powershell
cd D:\BC_Tools\memtest_vulkan
cargo build --release            # target\release\memtest_vulkan.exe, default features
cargo test                       # unit tests (no GPU needed)
cargo build --release --no-default-features   # pure upstream program (+ inert hooks)
```

Runtime needs only the NVIDIA driver, which provides `vulkan-1.dll`, `nvml.dll`, `nvcuda.dll` and
`nvapi64.dll`. No Vulkan SDK or CUDA toolkit is needed: shaders are compiled into the binary and the
CUDA kernels ship as PTX.

### Linux (x86_64)

```bash
curl https://sh.rustup.rs -sSf | sh          # rust >= 1.89
sudo apt install build-essential pkg-config  # nothing GPU-specific at build time
cargo build --release
cargo test
```

Runtime requirements:
- the NVIDIA driver, which provides `libvulkan.so.1` (from `libvulkan1`) plus the NVIDIA ICD,
  `libnvidia-ml.so.1` and `libcuda.so.1`
- an ffmpeg build with nvenc/cuvid for `--codec`. Distribution ffmpeg normally has them, because
  nv-codec-headers load at runtime.

Linux behaviour differences:
- The NVAPI junction temperature is not available (see §6).
- PCIe AER counters are read from sysfs.
- Without a display there is no driver watchdog, so a hung kernel is caught by:
  - the supervisor's worker deadline (`--worker-grace-s`), which kills the worker and *confirms* its
    death or says it could not
  - the worker's own hard deadline
  - `PR_SET_PDEATHSIG` if the supervisor itself dies

### Cargo features

| Feature | Default | Adds |
|---|---|---|
| `diagnostics` | on | long-flag CLI, selection, safety gate, JSON output, Vulkan compute/transfer tests, ffmpeg codec test |
| `nvml` | on | card details, ECC/remap/PCIe counters, telemetry (crate `nvml-wrapper`, NVML loaded at runtime) |
| `nvapi` | on | Windows-only GDDR6X junction + hotspot temperature (undocumented NVAPI call) |
| `cuda` | on | CUDA driver-API VMM memory test |
| `verbose` | off | upstream verbose output |

## 2. Commands

```text
memtest_vulkan --list [--json]                    # GPUs: index, PCI, vk label, UUID, VRAM, display
memtest_vulkan --info --gpu SPEC [--json] [--report-json FILE]   # read-only details + health
memtest_vulkan --gpu SPEC [tests] [options]       # run tests (default: --memtest 300)
memtest_vulkan --help | --version
```

`--list` and `--info` are **read-only**: Vulkan enumeration (no device is created), NVML queries and one
NVAPI thermal read. They are safe on live, busy and display cards, like `nvidia-smi -q`.

**GPU spec:**

| Form | Meaning |
|---|---|
| `0`..`N`, `index:N` | the **nvidia-smi index** (read from NVML by PCI id), which is what BC_LocalLLM calls GPU N. Refused without NVML: use PCI/UUID instead. |
| `0000:81:00.0`, `81:00.0` | PCI address |
| `GPU-6727…` | UUID or unique prefix |
| `vk:N` | upstream label |

Upstream's labels do *not* match nvidia-smi. On this rig `vk:3` is GPU 1, the display 4090.

**Tests** (any combination, run in this order):

| Flag | What |
|---|---|
| `--memtest S` | upstream Vulkan VRAM test, unchanged algorithm, time-limited; `--memtest-bytes 20G` caps size |
| `--compute S` | ALU/FPU: integer chain bit-exact vs CPU, FP32 chain by cross-SM consensus + CPU tolerance |
| `--transfer S` | PCIe: 256 MiB host→device→host copies, word-exact verification, H2D/D2H GB/s |
| `--cuda-memtest S` | CUDA VMM: claims free VRAM minus `--cuda-keep-free-mb` (default 256) in 2 MiB-granular chunks; own-address, moving inversions, walking ones, random |
| `--codec` | NVENC→NVDEC round trip (h264/hevc/av1), PSNR ≥ 30 dB + frame count; `--ffmpeg PATH` |

**Safety gate** (refusal = exit 3):

| Condition | Refused unless |
|---|---|
| a display is attached | `--allow-display` |
| display state unknown (no NVML, or a display query failed) | `--allow-display` |
| more than `--max-used-mb` (default 1024) VRAM in use | `--allow-busy` |
| compute processes present | `--allow-busy` |
| VRAM usage / compute-process list unknown (query failed) | `--allow-busy` |

A refusal with `--report-json` still writes the output file, with `refusal` set and verdict null.

**Output:**
- console: upstream progress lines, plus one telemetry line every `--telemetry-print-s` (10 s), plus a
  summary with a VERDICT
- `--report-json FILE`: the full JSON. It is written atomically (temp file + rename).
- `FILE` with extension `.memtest.log`: upstream's console log of the Vulkan memtest child. Each run uses
  its own working directory (temp, or `--work-dir`), so parallel runs on different cards never share a log.

**Exit codes:**

| Code | Meaning |
|---|---|
| 0 | PASS |
| 1 | FAIL |
| 2 | INCOMPLETE |
| 3 | refused / usage (output written with `refusal`) |
| 4 | operational error, including "requested output file could not be written" (verdict still printed) |

A previous output file at the `--report-json` path is deleted before the run starts. If it cannot be
deleted, the run is refused (exit 4).

**Telemetry line:**

```text
[telemetry   120.0s] 71C hot 84C mem 96C | 342W | sm 1905 mem 9751 MHz | util 100% | PCIe gen4 x16 tx 3 rx 1 MB/s replay +0 | ECC c+0 u+0 | throttle sw_power_cap
```

`replay`/`ECC` are increases since the first sample. `hot`/`mem` come from NVAPI on Windows; they show
`-` elsewhere.

## 3. Running on the BC_LocalLLM rig (live run, on a cleared card)

Current rig map:

| GPU | Card | Notes |
|---|---|---|
| GPU 0 | 3090 @01:00.0 | |
| GPU 1 | 4090 @81:00.0 | display + LLM, ECC on |
| GPU 2 | 3090 @82:00.0 | |
| GPU 3 | 3090 @c1:00.0 | |

1. **Choose a non-display card** (GPU 0, 2 or 3).
2. **Clear it.**
   - Drain the job queue (`python scripts/queue_maintenance.py drain --timeout 120`, or
     `POST /api/jobs/maintenance/drain`).
   - Stop the services pinned to that card (imagine pool member / sidecar on it).
   - Alternatively, once it lands, use `gpu.vram_test`'s card hold.
3. **Verify it is cleared:**
   ```powershell
   memtest_vulkan --list
   nvidia-smi --query-gpu=index,memory.used,display_active --format=csv
   memtest_vulkan --info --gpu 3 --report-json D:\BC_Tools\gpu-reports\gpu3-before.json
   ```
4. **Prove the reporting path end to end** (≈2 min). Upstream's fault injector flips one bit in
   iteration 3, so this run must end with **FAIL**:
   ```powershell
   $env:MEMTEST_VULKAN_EMULATE_WRITE_BUG_ITERATION = "3"
   memtest_vulkan --gpu 3 --memtest 30 --report-json D:\BC_Tools\gpu-reports\gpu3-inject.json
   Remove-Item Env:MEMTEST_VULKAN_EMULATE_WRITE_BUG_ITERATION
   ```
5. **Smoke run** (≈5 min) of every engine:
   ```powershell
   memtest_vulkan --gpu 3 --memtest 60 --compute 30 --transfer 30 --cuda-memtest 60 --codec `
     --ffmpeg "C:\Users\Ben Coster\ffmpeg.exe" --report-json D:\BC_Tools\gpu-reports\gpu3-smoke.json
   ```
6. **Full run** (≈1.5 h):
   ```powershell
   memtest_vulkan --gpu 3 --memtest 1800 --compute 600 --transfer 300 --cuda-memtest 1800 --codec `
     --ffmpeg "C:\Users\Ben Coster\ffmpeg.exe" --report-json D:\BC_Tools\gpu-reports\gpu3-full.json
   ```
7. **Optional independent cross-check** in a second console:
   `nvidia-smi dmon -i 3 -s pucvmet -d 1`, which gives per-second power, utilisation, clocks, PCIe
   throughput, ECC and temperature.
8. Restart the services and resume the queue.

**GPU 1 (display + LLM, ECC on).**
- `--info` is safe at any time. Take snapshots before and after LLM sessions and compare ECC aggregate
  uncorrected and remapped-rows counts: a rise means the card is failing.
- Stress tests are refused by default. Run them only:
  - with the monitor moved to another card, or
  - on Linux with no display server (`systemctl isolate multi-user.target`), using `--allow-display`
    only if NVML still reports it attached
- On this card the VRAM tests see ECC-*corrected* data. The evidence is in the report's
  `health_delta` / telemetry ECC counters and the remapper histogram, not in the memtest error count
  (REVIEW §6).

## 4. Interpreting results / warranty evidence

- **FAIL with memtest/cuda errors on a 3090:** DRAM/controller fault.
  - `xor_mask` and `single_bit_errors_by_bit` show stuck or flaky bit lanes.
  - Repeated identical offsets point to a bad cell/row.
  - Compare against `temp_mem_junction_c` to judge temperature dependence.
- **FAIL from ECC uncorrected or remapped-rows deltas (4090):** uncorrectable DRAM errors were seen by
  the card itself. Attach the report together with the Windows `nvlddmkm` events or the Linux `Xid 48/94/95`
  lines.
- **Compute `fp_consensus` / `int_exact` errors:** an SM/ALU fault on one replica. The workgroup ids
  locate the scheduler slot. Compute errors without memory errors point at the core, not the VRAM.
- **Transfer mismatches or replay/AER deltas:** PCIe path (riser, slot, cable, signal integrity).
  Re-seat and retest before blaming the GPU.
- **Codec `fail`** (not `unsupported`): an NVENC/NVDEC engine fault. AV1 encode on Ampere is
  `unsupported` by design.

## 5. Linux deeper path (no extra tooling in this binary)

- **AER**:
  - The report includes `/sys/bus/pci/devices/<bdf>/aer_dev_{correctable,nonfatal,fatal}` before/after
    and per sample.
  - The upstream root port has its own counters: find it with `readlink -f /sys/bus/pci/devices/<bdf>`,
    then read the parent's `aer_*` files.
- **Link + capability dump**:
  - `sudo lspci -vvv -s 81:00.0`: `LnkCap`/`LnkSta` (speed/width, "downgraded"), `DevSta` (CorrErr/
    UnsupReq), and `UESta`/`CESta` in the AER capability.
  - `setpci -s 81:00.0 CAP_EXP+0x12.w` reads raw link status.
- **Kernel log**: `journalctl -k | grep -E "NVRM: Xid|AER|pcieport"`. Xid 48/63/64/94/95 are memory-related;
  79 = fallen off the bus.
- **DCGM**: `dcgmi diag -r 3` (NVIDIA Data Center GPU Manager) runs NVIDIA's own memory, PCIe and SM
  stress plus diagnostics. It is Linux-only and officially supported on datacenter GPUs; on GeForce some
  plugins are skipped. Run it as the second opinion on the same cleared card.
- **Persistence mode**: `nvidia-smi -pm 1` keeps NVML counters warm between runs.

## 6. Memory-junction temperature

| OS | Availability |
|---|---|
| Windows | `nvapi` feature. Uses the undocumented `NvAPI_GPU_ThermalGetSensors` (HWiNFO / GPU-Z / LibreHardwareMonitor use the same). It is marked `nvapi_undocumented` in the output and disables itself on any error. Verified on this rig: 3090 channel 9 and 4090 channel 7 read plausible idle values. `--no-nvapi` turns it off. |
| Linux | No supported userspace API: NVML's memory temperature field is datacenter-only. The community tool `gddr6` reads it through BAR0 MMIO as root. That approach is deliberately out of scope here (raw MMIO / kernel access, security). Use the GPU core temperature + throttle reasons as the proxy, or check on Windows. |

## 7. Maintenance

**Regenerating the CUDA PTX** (only after editing `src/bcll/cuda/kernels.cu`; keep `cuda/patterns.rs` in
sync, since its unit tests pin the pattern arithmetic):

```bash
nvcc -ccbin "<MSVC>/bin/Hostx64/x64" -ptx -arch=compute_80 -O3 src/bcll/cuda/kernels.cu -o src/bcll/cuda/kernels.ptx
ptxas -arch=sm_86 src/bcll/cuda/kernels.ptx -o /dev/null   # 3090
ptxas -arch=sm_89 src/bcll/cuda/kernels.ptx -o /dev/null   # 4090
```

Do not add `-lineinfo`, which embeds local paths. The PTX ISA version is 8.7 (CUDA 12.8), so a driver
≥ 570 is required to JIT it.

**Syncing with upstream:**

```bash
git fetch upstream
git rebase upstream/main     # conflicts can only be at the touch points listed in DESIGN.md section 2
cargo test && ./run-lint.sh   # upstream lint script (fmt + clippy)
```

**Lint:** `cargo fmt --check`, plus `cargo clippy --all --all-targets -- -D clippy::all -W clippy::too-many-arguments`
(upstream's flags; the single remaining warning is upstream's own `test_device`), plus
`RUSTFLAGS="-D warnings" cargo check --all-features --all-targets`.
