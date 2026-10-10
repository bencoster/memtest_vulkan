# Output schema `bcll.gpu_diag.report` v1

The machine-readable result of one diagnostics run on one GPU. Two tools write it:

| Writer | `tool.implementation` | Engines |
|---|---|---|
| `memtest_vulkan --report-json FILE` (this fork) | `rust` | `vulkan`, `cuda`, `ffmpeg` |
| BC_LocalLLM `gpu.vram_test` (`services/gpu_diag`) | `python` | `torch`; it embeds this tool's output via its external-tester hook |

The formal JSON Schema is `report.schema.json` (draft 2020-12). The unit test
`bcll::report::tests::json_schema_matches_types` fails if the Rust types and the schema drift apart.

**Compatibility rule:** within `schema_version: 1`, fields are only **added** (always optional or
nullable), never renamed, removed or re-typed. Readers must ignore unknown keys. A breaking change bumps
`schema_version`.

## Top level

| Key | Type | Meaning |
|---|---|---|
| `schema` | `"bcll.gpu_diag.report"` | constant |
| `schema_version` | `1` | constant |
| `tool` | object | `name`, `version` (e.g. `0.6.0-bcll.1`), `implementation`, `features[]` |
| `run` | object | `started_at` / `finished_at` (RFC 3339 UTC), `duration_s`, `os`, `hostname`, `command_line[]` |
| `device` | object | identity + NVML `details` (below) |
| `health_before` / `health_after` | object or null | growing counters snapshot (below) |
| `health_delta` | object or null | `{counter: increase}` for counters that increased only |
| `tests[]` | array | one entry per test (below) |
| `telemetry` | object or null | `interval_ms`, `sources[]`, `memory_junction_source`, `summary`, `samples[]` |
| `verdict` | `PASS` / `FAIL` / `INCOMPLETE` / null | null for `--info` (no test) |
| `reasons[]` | strings | why FAIL / INCOMPLETE |
| `warnings[]` | strings | non-fatal findings (replays, corrected ECC, throttling, junction ≥100 °C, unsupported tests, VRAM coverage < 85 %, telemetry unavailable) |

Verdict rules: DESIGN.md §8.

## `device`

| Key | Notes |
|---|---|
| `selector` | what was passed to `--gpu` |
| `index_pci_order` | equals the nvidia-smi index / `CUDA_DEVICE_ORDER=PCI_BUS_ID` |
| `vk_label` | upstream memtest_vulkan label (descending PCI bus) |
| `pci_bus_id` | `dddd:bb:dd.f`, lower case |
| `uuid` | `GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx` (NVML, else Vulkan deviceUUID) |
| `name`, `vendor_id`, `device_id`, `vulkan_driver_version`, `vulkan_api_version`, `device_local_bytes` | from Vulkan |
| `display_attached` | NVML `display_active OR display_connected`; null = unknown |
| `details` | NVML card details (see below); null without NVML |

### `device.details` (NVML; every leaf may be null)

- **Identity and driver**
  - `index`, `name`, `uuid`, `serial` (null on GeForce), `board_part_number` (null when the board reports an empty string), `vbios_version`, `inforom_image_version`
  - `architecture`, `compute_capability`, `cuda_cores`, `memory_bus_width_bits`
  - `pci{bus_id, domain, bus, device, pci_device_id, pci_subsystem_id}`
  - `driver_model` (Windows), `persistence_mode` (Linux)
  - `system{driver_version, nvml_version, cuda_driver_version}`
- **Memory**
  - `memory{total_bytes, used_bytes, free_bytes, reserved_bytes}`
  - `bar1{…}`
  - `ecc_mode{current, pending}`
- **Clocks, power, thermal**
  - `clocks{graphics|sm|memory|video: {current_mhz, max_mhz}, gpc_vf_offset_mhz, mem_vf_offset_mhz}`
  - `power{usage_w, limit_w, enforced_limit_w, default_limit_w, total_energy_j}`
  - `thermal{gpu_c, memory_c_nvml, slowdown_threshold_c, shutdown_threshold_c, memory_max_c, gpu_max_c}`
  - Windows NVAPI additions to `thermal`: `hotspot_c_nvapi`, `memory_junction_c_nvapi`, `nvapi_channels_c[[channel, °C]]`, `nvapi_source: "nvapi_undocumented"`, or `nvapi_error`
  - `fans_pct[]`
  - `throttle_reasons{bits, names[]}`, `performance_state`
- **PCIe and display**
  - `pcie{gen_current, gen_max, width_current, width_max, speed_mbps_current, max_link_speed, replay_counter}`
  - `display_active`, `display_connected`

## Health snapshot (`health_before`, `health_after`)

- **ECC**
  - `ecc_mode_current`, `ecc_mode_pending`
  - totals: `ecc_{volatile,aggregate}_{corrected,uncorrected}`
  - by location: `ecc_{dram,sram}_{volatile,aggregate}_{corrected,uncorrected}`
  - All are null on cards without ECC reporting (RTX 3090).
- **Row remapper**
  - `remapped_rows_correctable`, `remapped_rows_uncorrectable`, `remap_pending`, `remap_failure`
  - `remap_histogram{max, high, partial, low, none}` (bank availability)
- **Retired pages** (pre-Ampere mechanism)
  - `retired_pages_single_bit`, `retired_pages_double_bit`, `retired_pages_pending`, `retired_page_addresses[]`
- **`pcie_counters`** (NVML field values; null = unsupported)
  - `replay`, `replay_rollover`, `l0_to_recovery`, `correctable_errors`, `naks_received`, `naks_sent`, `bad_tlp`, `bad_dllp`, `non_fatal_errors`, `fatal_errors`
  - Verified readable on RTX 4090 + driver 610.88.
  - `l0_to_recovery` also counts benign link retrains on gen changes (power saving). It is reported as a delta, not judged.
- **`aer`** (Linux only)
  - `correctable{…}`, `nonfatal{…}`, `fatal{…}`: kernel counter maps including `TOTAL_ERR_*`
  - `current_link_speed`, `current_link_width`, `max_link_speed`, `max_link_width`

## Test result (`tests[]`)

| Key | Notes |
|---|---|
| `kind` | `vram` / `compute` / `pcie_transfer` / `codec` |
| `engine` | `vulkan` / `cuda` / `ffmpeg` / `torch` |
| `name` | `vulkan_memtest`, `vulkan_compute`, `vulkan_transfer`, `cuda_vmm_memtest`, `h264_nvenc_cuvid`, `hevc_nvenc_cuvid`, `av1_nvenc_cuvid` (Python: its own names) |
| `status` | `pass` / `fail` / `incomplete` / `error` / `unsupported` / `skipped` |
| `started_at`, `duration_s` | |
| `coverage` | `tested_bytes`, `total_bytes`, `free_bytes_at_start`, `reserved_or_used_bytes` (total−free), `driver_reserved_bytes` (NVML `reserved`) |
| `error_count` | exact total, even when `errors[]` is truncated |
| `errors[]` | at most 1024 records; `errors_truncated` true if more |
| `metrics` | free-form per test (below) |
| `message` | human explanation for non-pass |
| `termination` | completed / mismatch / runtime_error / device_lost / timed_out / crashed / init_failed / cancelled / unsupported |

**Error record:**

| Key | Meaning |
|---|---|
| `offset_bytes` | for single-word errors (CUDA, transfer) |
| `offset_range` | `[first, last]`, for the Vulkan memtest's per-window range |
| `expected`, `actual`, `xor_mask` | u32 words |
| `count` | errors this record stands for |
| `iteration` | iteration / round / batch |
| `detail` | engine-specific |

Offsets are **virtual** offsets within the tested allocation. No GPU API exposes physical DRAM addresses
to userspace. Use the driver log (Windows `nvlddmkm` / Linux Xid 48/94/95 `physAddr`) together with the
row-remap counters to locate a fault physically.

**Metrics by test:**

| Test | Metrics |
|---|---|
| `vulkan_memtest` | `allocation_attempts[{bytes, error}]` (OOM retries), `iterations`, `write_gbps_avg`, `check_gbps_avg`, `finish_reason`, `exit_code`, `allocation_bytes`, `window_bytes`, `window_count`. Each error's `detail` has `mode` (`INITIAL_FILL` = startup verification / `INITIAL_READ` / `NEXT_RE_READ`), `single_bit_errors_by_bit`, `errors_by_flipped_bit_count`, `bad_values_by_popcount`, `all_ones_words`, `actual_min`, `actual_max` (null when only all-ones words failed). |
| `vulkan_compute` | `batches`, `int_errors`, `fp_consensus_errors`, `fp_reference_errors`, `invocations`, `problems`, `iters`, `base_seed`. Error `detail`: `invocation`, `workgroup`, `problem`, `lane` (0 = int x, 1 = int acc, 2 = fp g, 3 = fp h), `check`, `seed`. |
| `vulkan_transfer` | `rounds`, `bytes_per_round`, `h2d_gbps{avg,min,max}`, `d2h_gbps{avg,min,max}` |
| `cuda_vmm_memtest` | `passes{own_address|moving_inversions|walking_ones|random: {runs, gbps_avg, errors}}`, `rounds_completed`, `bit_histogram[32]`, `granularity_bytes`, `chunks`, `keep_free_bytes`, `requested_bytes`, `sm_count` |
| codec | `ffmpeg`, `encode_fps`, `decode_fps`, `psnr_avg_db`, `frames_encoded`, `frames_decoded`, `encoded_bytes` |

## Telemetry

`telemetry.samples[]` holds one object per interval. Each sample has `t_s` and:
- temperatures: `temp_gpu_c`, `temp_hotspot_c`, `temp_mem_junction_c`
- power and clocks: `power_w`, `clock_sm_mhz`, `clock_mem_mhz`
- utilisation: `util_{gpu,mem,enc,dec}_pct`
- PCIe: `pcie_gen`, `pcie_width`, `pcie_tx_kbs`, `pcie_rx_kbs`, `pcie_replay`
- ECC and remapper: `ecc_volatile_{corrected,uncorrected}`, `remapped_rows_uncorrectable`
- `throttle_bits`, `fan_pct`
- Linux AER: `aer_{correctable,nonfatal,fatal}`

`telemetry.summary` has:
- `samples`, `duration_s`
- temperature maxima, `power_w_{max,avg}`
- `clock_*_min_busy` / `_max`
- `pcie_{gen,width}_min_busy`, `pcie_{gen,width}_max_seen`, `pcie_{tx,rx}_kbs_max`
- `link_changes[{t_s, from, to}]`
- counter deltas, `throttle_reasons_seen{reason: samples}`

"Busy" means GPU utilisation ≥ 50 %.

## Worker events (internal, JSON lines)

These are internal: they pass between the orchestrator and its worker children, and are not part of the
output contract. Every line has `ev` and `t_s`.

| Source | Events |
|---|---|
| upstream memtest child (`src/hooks.rs`) | `memtest_started`, `memtest_error`, `memtest_progress`, `memtest_finished`, `memtest_init_failure{message}` |
| Vulkan workers | `worker_started`, `compute_batch`, `compute_error`, `compute_summary`, `transfer_round`, `transfer_error`, `transfer_summary` |
| CUDA worker | `cuda_coverage`, `cuda_pass`, `cuda_error`, `cuda_canary_mismatch`, `cuda_summary` |
| all workers | `worker_finished`, `worker_error` (`device_lost`, `timeout`, `gpu_fault`, `ecc_uncorrectable`, `unsupported` flags) |

## Mapping for the Python `gpu.vram_test`

The Python record maps onto the same file:
- `tool.implementation: "python"`
- the torch pattern test as one `tests[]` entry (`kind: "vram"`, `engine: "torch"`)
- when it ran this binary through its external-tester hook: this tool's `tests[]` appended unchanged, and
  this tool's `telemetry` / `health_*` kept as-is (do not re-derive)
- its own verdict computed with the same rules (DESIGN.md §8), so PASS/FAIL/INCOMPLETE mean the same in both
