# memtest_vulkan — source review (BCLL fork)

Reviewed: upstream `main` @ `fd9ff59` (v0.5.0-75, Cargo version `0.6.0`), 2026-10-10.
Scope: the whole crate (`src/*.rs`, `memtest_vulkan_build/`), build/CI scripts.
Purpose: decide whether memtest_vulkan is a sound base for deep per-card diagnostics on the
BC_LocalLLM rig (3x RTX 3090 without ECC reporting, 1x RTX 4090 with ECC on and recurring
uncorrectable DRAM errors) and for warranty evidence.

Line numbers refer to `fd9ff59`.

## 1. Headlines

1. **The core test is sound and cheap to trust.** It uses an address-derived pattern that changes every
   iteration. Verification runs on the GPU, and bit-level error statistics come back in a small coherent
   buffer. It catches stuck bits, most addressing faults and retention loss in one window. On non-ECC
   cards (the 3090s) it is a good DRAM/memory-controller detector.
2. **On an ECC-enabled card (GPU 1, the 4090) it is close to blind.** Single-bit errors are corrected
   before the shader sees them. An uncorrectable error kills the context (Xid 48/94/95, then
   `VK_ERROR_DEVICE_LOST`), and the tool reports that as a "runtime error", not as a memory error. You
   need ECC and row-remapper counters to read that card. NVML provides them, and the tool reads none.
3. **There is no device identity beyond "Bus=0xBB:DD".** The tool does not read the PCI domain or
   function, the UUID, the serial number or the VBIOS version. Its 1-based numbering is "discrete GPUs
   sorted by *descending* PCI bus", which does not match the nvidia-smi / CUDA PCI_BUS_ID index. On this
   rig, upstream's `3` is the **display + LLM card** (bus 0x81).
4. **Safety foot-guns for an unattended rig.** (a) Any first argument that is not an integer, such as
   `--help`, leaves the program in *interactive* mode. It then auto-selects a device after 10 s and
   starts testing (`main.rs:1859-1867`, `1535-1547`). (b) Nothing refuses a display-attached GPU.
   (c) `wait_for_fences(.., u64::MAX)` waits forever (`main.rs:925`). On Linux, a compute-only card
   has no TDR, so a wedged GPU hangs the process.
5. **No telemetry, no machine-readable output.** Results exist only as console/log text. Error
   addresses are reported as a min..max range per window, not as a list. Temperatures, clocks,
   PCIe state and throttling are not observed. That matters because GDDR6X faults correlate with
   memory temperature.
6. **The dependencies are ageing.** `erupt` 0.23 (2022; the project is archived, and `ash` is the
   maintained alternative) and `naga` 0.10 (2022, older WGSL syntax: module-scope `let`,
   `main.rs:108-109`). Upstream is a low-activity single-maintainer project: 4 commits in 2026,
   38 in 2025, mostly build/CI work. The last commit was 2026-06-02.

## 2. Architecture

| File | Role |
|---|---|
| `src/main.rs` (2004 lines) | Everything: embedded WGSL shader, Vulkan instance/device setup, allocation, the test loop, device enumeration and selection, the CLI, result display |
| `src/close.rs` | Process status bits encoded in the **exit code** (`SIGNATURE 0b010xxxxx`, `INITED_OK`, `RUNTIME_ERRORS`, `RUNTIME_ABORT`, `QUIT_JOB_REQUESTED`), Ctrl+C / signal handlers |
| `src/input.rs` | Terminal prompt (via `mortal`) for the device index; pass/fail colour |
| `src/output.rs` | `LogDupler`: tees stdout to `memtest_vulkan.log` in the **current directory**, locked append, truncation at 50 MB |
| `src/ram.rs` | Stubs: `budget()` always returns `None`, and `virt_addr_details()` always returns `None` |
| `src/erupt_vendored_utils_loading.rs` | Loads `vulkan-1.dll` / `libvulkan.so.1` through `libloading` |
| `memtest_vulkan_build/` | Proc-macro that compiles the WGSL string to SPIR-V at build time with `naga` |

**Process model.** The interactive parent enumerates devices and prompts. It then **re-executes itself**
as a child, passing `argv1 = -1 - vk_enumeration_index` and `argv2 = max_test_bytes`
(`main.rs:1634-1638`). The child runs the test and exits with the status bits. If the child aborts during
init (`RUNTIME_ABORT` without `INITED_OK`), the parent retries with 400 MB less (`main.rs:1716-1733`). If
spawning is impossible, the test falls back to running in-process (`main.rs:1747-1751`). This isolation is
good: a driver fault kills the child, not the reporting parent.

## 3. How the memory test works

### 3.1 Allocation and coverage
- **Budget** (`try_fill_default_mem_budget`, `main.rs:538-588`). For each DEVICE_LOCAL heap:
  `free = min(size - usage, budget)`, taken from `VK_EXT_memory_budget`, minus `TEST_DATA_KEEP_FREE`
  = 400 MB (`main.rs:191`). The test therefore only covers VRAM that is *free at start*. It never
  touches the display framebuffer, the driver's reserved region, or memory used by other processes.
- **One** `vkAllocateMemory` of the whole size (`main.rs:850-861`). The memory type is DEVICE_LOCAL
  from the largest heap, preferring HOST_VISIBLE (`main.rs:787-811`). If allocation, buffer creation,
  binding or the first fill fails, the size steps down by 400 MB (`main.rs:976`). In child mode
  the process exits instead and the parent retries (`main.rs:482-490`, `965-967`).
- **Windows.** The allocation is split into `n >= 2` windows of at most 4 GiB - 128 MiB
  (`main.rs:188-190`, `866-871`). Each window is bound through the descriptor offset/range. The window
  size is rounded **down** to a 128 MiB multiple (`main.rs:870-871`), so up to 128 MiB per window is
  allocated but **never tested** (about 0.4-0.8 GB on a 24 GB card). Because the tail is silently
  dropped, the reported "test memory size" overstates coverage.
- **Physical mapping is unknown.** Offsets are within the virtual allocation. Nothing maps them to a
  physical address, FBPA, bank or row, so you cannot correlate them with the driver's
  `physAddr`/row-remap records (see §6).

### 3.2 Patterns
- `test_value_by_index(i) = rotl(a, a % 31)` with `a = 4*i + calc_param + {1,2,3,4}`
  (`main.rs:93-105`). The value is unique per address within a window, so it catches addressing,
  aliasing and decoder faults. Rotation varies the bit density.
- `calc_param = iter * 0x100107` (`main.rs:335`), plus `window * 0x81` (`main.rs:1049`, `1066`). Every
  iteration writes a new value to every cell.
- **Not present:** solid 0x00/0xFF, checkerboard, walking ones/zeros, moving inversions (write X, then
  verify and write ~X) and random data. Faults that depend on neighbour coupling and that only an
  inversion sequence exposes are covered only statistically.

### 3.3 The iteration (`main.rs:1043-1188`)
1. **Write** windows `1..n` (`main.rs:1046-1059`). Within each 128 MiB block, the write order is
   reversed relative to the thread id (`main.rs:163-166`).
2. **Read and verify** all windows `0..n` (`main.rs:1063-1111`). The read order is rotated within
   8192-vec4 groups (`main.rs:113-116`) so it is not purely sequential.
3. **Window 0 is never rewritten** after the initial fill (`main.rs:934-946`). Each iteration
   re-reads it against the iteration-1 pattern (`NEXT_RE_READ` mode, `main.rs:1064-1075`). That is a
   **retention / read-disturb test** of the first ≤4 GiB: by design, only window 0 gets it.
4. Each dispatch is synchronous: submit, then wait on the fence (`main.rs:887-929`). One dispatch covers
   ≤4 GiB, a few ms at 1 TB/s, which is well inside the Windows 2 s TDR.

### 3.4 Verification and error reporting
- The shader compares `vec4<u32>` values. On a mismatch, the slow path records the following with
  atomics into the coherent `IOBuf` (`main.rs:119-149`):
  - `err_bit1_idx[bit]`: a histogram of **single-bit** flips by bit position
  - `err_bitcount[popcount(xor) % 32]`: errors by number of flipped bits
  - `mem_bitcount[popcount(actual)]`: a histogram of the bad *values*
  - `idx_min` / `idx_max`: the error address **range**
  - `actual_min` / `actual_max` / `actual_ff`
- `check_vec_first` (`main.rs:366-380`) recomputes the expected value of element 1 on the CPU. This
  detects a shader that did not run or computed wrongly. It is a good "the GPU actually did the work"
  canary. In the same spirit, `done_iter_or_err` is set only by invocation 0 (`main.rs:152-153`).
- Output is text (`main.rs:1090-1107`): mode, count, %, address range, iteration and the stats table.
  The exit code carries the `RUNTIME_ERRORS` bit.

### 3.5 Test length
- Report cadence: immediately, then after 1 s, 5 s, and every 30 s (`main.rs:1130-1138`). After
  12 reports (≈5 min) it prints "Standard 5-minute test PASSed / fail" and continues indefinitely
  (`iter_count = 1e8`, `main.rs:1032`). Before the 12th report it sleeps for half the interval
  (`main.rs:1176`) so the hot card idles and memory clocks drop, testing clock transitions.
- No time or iteration limit can be set from the command line. The only way to stop is Ctrl+C /
  SIGTERM, so a harness has to kill it.

## 4. Correctness / robustness findings

| # | Sev | Where | Finding |
|---|---|---|---|
| R1 | Med | `main.rs:355` vs `139-148` | `total_errors = sum(mem_bitcount)` excludes the **all-ones** bad words, which are counted only in `actual_ff`. A failing region that reads back 0xFFFFFFFF (a typical dead-channel signature) is under-reported, even down to "total errors 0x0" while errors are flagged. The fix is `sum + actual_ff`. |
| R2 | Low | `main.rs:133` | `err_bitcount[one_bits % 32]`: a word with all 32 bits flipped is counted at index 0 ("0 bits toggled"). |
| R3 | Med | `main.rs:135-136`, `357-360` | Only the min/max address of the errors is kept per window, so scattered errors produce a meaningless range spanning the window. You cannot list failing addresses or count distinct rows. |
| R4 | Low | `main.rs:870-871` | Up to 128 MiB per window is allocated but untested, and the reported coverage includes it. |
| R5 | Med | `main.rs:925` | `wait_for_fences(u64::MAX)`: no timeout. On Linux (and on a TCC/compute-only card) a hung GPU hangs the tool forever instead of reporting. |
| R6 | High (ops) | `main.rs:1859-1867`, `1535-1547` | Any non-integer argument, including `--help`, keeps interactive mode. With more than one GPU, the first device is autoselected after **10 s** and testing starts. On a production rig this is unsafe: **never run the upstream binary to "see the help"**. |
| R7 | High (ops) | `main.rs:1462-1470` | The default device is "the discrete GPU with the highest PCI bus number", on the assumption that the display card has the lowest bus. On this rig the display card is bus 0x81, *not* the lowest. The default happens to be 0xC1 (GPU 3, a 3090), but **label 3 is the display/LLM 4090**. No display check exists. |
| R8 | Med | `main.rs:1504` | The label shows `bus:device` only. Domain and function are missing, and no UUID is shown, so cards cannot be identified unambiguously for an RMA report. |
| R9 | Low | `main.rs:1282-1313` | `VK_LAYER_KHRONOS_validation` is enabled whenever it is installed, in release builds too, so the Vulkan SDK on a host slows the test and adds noise. |
| R10 | Low | `main.rs:1028` | `MEMTEST_VULKAN_EMULATE_WRITE_BUG_ITERATION` injects one bit flip at `0xADBA` in the chosen iteration. It is useful for harness tests and is kept. |
| R11 | Low | `ram.rs:3-5` | `ram::budget()` always returns `None`, so integrated GPUs are not capped by free system RAM, even though the comment at `main.rs:580` says they are. |
| R12 | Low | `close.rs:83-86` | On Windows close/logoff events, the handler sleeps for 25 s and then `ExitProcess`. That is fine, but it means a killed parent leaves the child running for up to 25 s. |
| R13 | Low | `main.rs:1991-1994` | The log goes to `memtest_vulkan.log` in the **CWD**, which can be a read-only directory or a shared one with concurrent writers. The file lock helps, but runs on different cards interleave in one file. |
| R14 | Info | `main.rs:1400` | A TODO about negative-index handling. Child mode with `-1-idx` filters enumeration correctly, but the semantics are subtle (`ProcessEnv::device_label`, `main.rs:1801`). |
| R15 | Info | `main.rs:1176` | The intentional idle sleep before the 12th report is good practice; the code calls it out only in a comment. |

There is no unit-test suite (`cargo test` runs 0 tests). Lint (`run-lint.sh`) is clippy with `-D clippy::all`.

## 5. Windows vs Linux

| Aspect | Windows (WDDM) | Linux (nvidia.ko) |
|---|---|---|
| Hang protection | TDR (2 s by default; 60 s on this rig) resets the GPU, and the fence wait returns `DEVICE_LOST`. On the **display GPU** that is a black screen and every app's context is reset. | Without a display there is no watchdog, so a hang blocks forever (R5). With a display, the X/Wayland watchdog depends on the driver. |
| Residency | WDDM can **evict/demote** allocations to system memory under over-commit. The budget clamp (`main.rs:575-578`) makes that unlikely, but it is not guaranteed: an eviction would silently test host RAM over PCIe. | Allocations are pinned in VRAM. No demotion. |
| Signals | `SetConsoleCtrlHandler` (`close.rs:79-91`) | `sigaction` on INT/TERM/HUP/QUIT (`close.rs:103-121`) |
| Vulkan loader | `vulkan-1.dll` | `libvulkan.so.1`; `VK_DRIVER_FILES` is honoured (`main.rs:1331`) |
| Extras | None | AER counters in sysfs, `dmesg` NVRM Xid, DCGM (see USAGE.md) |

## 6. What it cannot detect

- **ECC-corrected errors** on cards with ECC enabled (GPU 1). The data reaches the shader corrected, so
  the counters are the only evidence.
- **Uncorrectable ECC errors** surface as `DEVICE_LOST` (a "runtime error"), not as memory errors, and the
  physical address appears only in the driver log (Windows `nvlddmkm` events / Linux Xid 48/94/95).
- **VRAM that is not free**: framebuffer, driver-reserved memory, other processes, the 400 MB keep-free
  margin and the window tail (R4). On GPU 1 that includes the LLM's ~20+ GB, so the card has to be
  cleared first.
- **The physical location** (FBPA/channel/bank/row) of an error.
- **PCIe link faults.** The test is device-local; the host only reads a ~400 B status buffer.
  Replays, link downgrades and AER errors are not observed.
- **SM/ALU/FPU faults**, except by accident (the shader uses only integer add/shift/compare).
- **Thermal correlation.** No temperatures are sampled, and GDDR6X faults are temperature-dependent.
- **Codec engines** (NVENC/NVDEC), **copy engines**, **L2 cache** (only indirectly).

## 7. Code quality

- A single 2 kLOC `main.rs` with free functions that take 10+ arguments (`test_device`, `main.rs:711-723`).
  The large `'memsize` loop nests 6 levels deep (`main.rs:843-977`).
- Errors are `Box<dyn Error>` strings. Process state lives in global atomics and is encoded in the exit
  code. That is pragmatic, but it makes in-process reuse hard: `display_this_process_result` is `-> !`.
- Good: careful exit-code protocol, the subprocess isolation, the build-time shader compile
  (no runtime shader toolchain), and clippy-clean code.
- Old dependencies (headline 6). `erupt` is archived upstream, so a future port to `ash` will be
  needed before it bit-rots against newer Vulkan headers. It is not urgent: the API surface used here
  is Vulkan 1.1 core plus two EXTs.

## 8. Upstream activity

`git log upstream/main`: 142 commits in 2022, 21 in 2023, 12 in 2024, 38 in 2025, 4 in 2026 (last
2026-06-02, "Workaround erupt UB panic while running debug build"). Releases: tags v0.3.0 to v0.5.0.
Cargo says 0.6.0 (unreleased). Activity is mostly cross-build and CI maintenance by one maintainer.
The fork should therefore keep its changes **additive and isolated** (new modules, cargo features,
new flags) so that rebasing onto upstream stays trivial; DESIGN.md lists the exact touch points.

## 9. Recommendations (implemented on `bcll/diagnostics` unless noted)

1. Long-flag CLI that never falls into the interactive autoselect path (fixes R6 for the fork).
2. Card selection by PCI bus id / UUID / nvidia-smi index; refuse display-attached GPUs unless
   `--allow-display`; unknown display state fails closed (R7, R8).
3. NVML card details plus ECC, row-remapper and retired-page snapshots before and after; NVML telemetry
   during the test (PCIe replay/throughput/link state, temperature, power, clocks, throttle reasons)
   (§6).
4. Structured error events from the memtest child (offsets, bit histograms, R1-corrected counts) and a
   JSON report (R3 partially: per-window ranges plus counts).
5. Optional Vulkan compute-verification and PCIe-transfer-verification tests, an optional CUDA
   VMM memory test with explicit coverage accounting, and an optional NVENC/NVDEC round trip.
6. Not changed (to stay upstream-mergeable): the shader, the iteration structure, R2/R4/R5 inside the
   upstream path. R1 is corrected in the *report* (structured events carry `actual_ff` separately and
   the parent adds it). Proposed upstream patches for R1/R2/R5 are listed in DESIGN.md §9.
