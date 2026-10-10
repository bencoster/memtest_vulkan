//! NVML card details, health snapshots and telemetry samples (feature `nvml`).
//!
//! NVML is loaded at runtime by nvml-wrapper (`nvml.dll` / `libnvidia-ml.so.1`);
//! every query is optional so one unsupported call (common on GeForce) never
//! fails the whole snapshot. Row-remapper calls go through the raw symbol table
//! because nvml-wrapper 0.13 does not wrap them.

use super::telemetry::{HealthSnapshot, Sample, Usage, display_from};
use nvml_wrapper::{
    Device, Nvml,
    enum_wrappers::device::{
        Clock, EccCounter, MemoryError, MemoryLocation, PcieUtilCounter, RetirementCause,
        TemperatureSensor, TemperatureThreshold,
    },
    enums::device::{SampleValue, UsedGpuMemory},
    structs::device::FieldId,
    sys_exports::field_id as fid,
};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;

pub struct NvmlCtx {
    nvml: Nvml,
}

fn put<T: serde::Serialize, E>(m: &mut Map<String, Value>, key: &str, r: Result<T, E>) {
    m.insert(
        key.to_string(),
        r.ok()
            .and_then(|v| serde_json::to_value(v).ok())
            // GeForce boards report e.g. an empty part number: treat as unknown
            .filter(|v| v.as_str().is_none_or(|s| !s.trim().is_empty()))
            .unwrap_or(Value::Null),
    );
}

fn sample_u64(v: &SampleValue) -> Option<u64> {
    match v {
        SampleValue::U32(x) => Some(*x as u64),
        SampleValue::U64(x) => Some(*x),
        SampleValue::I64(x) => u64::try_from(*x).ok(),
        SampleValue::F64(x) => (*x >= 0.0).then_some(*x as u64),
    }
}

/// NVML PCIe/field counters collected in health snapshots.
const PCIE_FIELDS: &[(&str, u32)] = &[
    ("replay", fid::NVML_FI_DEV_PCIE_REPLAY_COUNTER),
    (
        "replay_rollover",
        fid::NVML_FI_DEV_PCIE_REPLAY_ROLLOVER_COUNTER,
    ),
    (
        "l0_to_recovery",
        fid::NVML_FI_DEV_PCIE_L0_TO_RECOVERY_COUNTER,
    ),
    (
        "correctable_errors",
        fid::NVML_FI_DEV_PCIE_COUNT_CORRECTABLE_ERRORS,
    ),
    ("naks_received", fid::NVML_FI_DEV_PCIE_COUNT_NAKS_RECEIVED),
    ("naks_sent", fid::NVML_FI_DEV_PCIE_COUNT_NAKS_SENT),
    ("bad_tlp", fid::NVML_FI_DEV_PCIE_COUNT_BAD_TLP),
    ("bad_dllp", fid::NVML_FI_DEV_PCIE_COUNT_BAD_DLLP),
    (
        "non_fatal_errors",
        fid::NVML_FI_DEV_PCIE_COUNT_NON_FATAL_ERROR,
    ),
    ("fatal_errors", fid::NVML_FI_DEV_PCIE_COUNT_FATAL_ERROR),
];

#[repr(C)]
#[derive(Default)]
struct RemapHistogram {
    max: u32,
    high: u32,
    partial: u32,
    low: u32,
    none: u32,
}

impl NvmlCtx {
    pub fn open() -> Result<Self, String> {
        // load the driver's copy explicitly (System32 on Windows), not via the search path
        let lib = super::system_library(if cfg!(windows) {
            "nvml.dll"
        } else {
            "libnvidia-ml.so.1"
        })
        .map_err(|e| format!("NVML unavailable: {e}"))?;
        Nvml::builder()
            .lib_path(std::ffi::OsStr::new(&lib))
            .init()
            .map(|nvml| NvmlCtx { nvml })
            .map_err(|e| format!("NVML unavailable: {e}"))
    }

    fn device(&self, pci_bus_id: &str) -> Option<Device<'_>> {
        self.nvml.device_by_pci_bus_id(pci_bus_id).ok()
    }

    pub fn system(&self) -> Value {
        let mut m = Map::new();
        put(&mut m, "driver_version", self.nvml.sys_driver_version());
        put(&mut m, "nvml_version", self.nvml.sys_nvml_version());
        put(
            &mut m,
            "cuda_driver_version",
            self.nvml.sys_cuda_driver_version().map(|v| {
                format!(
                    "{}.{}",
                    nvml_wrapper::cuda_driver_version_major(v),
                    nvml_wrapper::cuda_driver_version_minor(v)
                )
            }),
        );
        Value::Object(m)
    }

    pub fn uuid(&self, pci_bus_id: &str) -> Option<String> {
        self.device(pci_bus_id)?.uuid().ok()
    }

    pub fn architecture(&self, pci_bus_id: &str) -> Option<String> {
        self.device(pci_bus_id)?
            .architecture()
            .ok()
            .map(|a| format!("{a:?}"))
    }

    pub fn display_attached(&self, pci_bus_id: &str) -> Option<bool> {
        let d = self.device(pci_bus_id)?;
        display_from(d.is_display_active().ok(), d.is_display_connected().ok())
    }

    /// nvidia-smi index of the device at this PCI id.
    pub fn index(&self, pci_bus_id: &str) -> Option<u32> {
        self.device(pci_bus_id)?.index().ok()
    }

    pub fn usage(&self, pci_bus_id: &str) -> Option<Usage> {
        let d = self.device(pci_bus_id)?;
        let mem = d.memory_info().ok()?;
        let compute_pids = d
            .running_compute_processes()
            .map(|v| {
                v.into_iter()
                    .filter(|p| !matches!(p.used_gpu_memory, UsedGpuMemory::Used(0)))
                    .map(|p| p.pid)
                    .collect()
            })
            // a failed process query is UNKNOWN, never "no processes" (BLOCK 89de0c26 #3)
            .ok();
        Some(Usage {
            total_bytes: mem.total,
            used_bytes: mem.used,
            free_bytes: mem.free,
            reserved_bytes: (mem.reserved > 0).then_some(mem.reserved),
            compute_pids,
        })
    }

    /// (max gen, max width) the link can train to in this system.
    pub fn max_link(&self, pci_bus_id: &str) -> Option<(u32, u32)> {
        let d = self.device(pci_bus_id)?;
        Some((d.max_pcie_link_gen().ok()?, d.max_pcie_link_width().ok()?))
    }

    fn field_u64(d: &Device, ids: &[(&str, u32)]) -> BTreeMap<String, Option<u64>> {
        let field_ids: Vec<FieldId> = ids.iter().map(|(_, id)| FieldId(*id)).collect();
        let values = d.field_values_for(&field_ids).unwrap_or_default();
        ids.iter()
            .enumerate()
            .map(|(i, (name, _))| {
                let v = values
                    .get(i)
                    .and_then(|r| r.as_ref().ok())
                    .and_then(|s| s.value.as_ref().ok())
                    .and_then(sample_u64);
                (name.to_string(), v)
            })
            .collect()
    }

    fn remapped_rows(&self, d: &Device) -> Option<(u64, u64, bool, bool)> {
        let f = self.nvml.lib().nvmlDeviceGetRemappedRows.as_ref().ok()?;
        let (mut corr, mut unc, mut pending, mut failure) = (0u32, 0u32, 0u32, 0u32);
        // SAFETY: valid device handle from this Nvml instance; out-params are plain u32.
        let rc = unsafe { f(d.handle(), &mut corr, &mut unc, &mut pending, &mut failure) };
        (rc == 0).then_some((corr as u64, unc as u64, pending != 0, failure != 0))
    }

    fn remap_histogram(&self, d: &Device) -> Option<BTreeMap<String, u64>> {
        let f = self
            .nvml
            .lib()
            .nvmlDeviceGetRowRemapperHistogram
            .as_ref()
            .ok()?;
        let mut h = RemapHistogram::default();
        // SAFETY: RemapHistogram mirrors nvmlRowRemapperHistogramValues_t (5 x c_uint).
        let rc = unsafe { f(d.handle(), &mut h as *mut RemapHistogram as *mut _) };
        (rc == 0).then(|| {
            [
                ("max", h.max),
                ("high", h.high),
                ("partial", h.partial),
                ("low", h.low),
                ("none", h.none),
            ]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v as u64))
            .collect()
        })
    }

    pub fn health(&self, pci_bus_id: &str) -> Option<HealthSnapshot> {
        let d = self.device(pci_bus_id)?;
        let ecc_total = |e, c| d.total_ecc_errors(e, c).ok();
        let ecc_loc = |e, c, l| d.memory_error_counter(e, c, l).ok();
        let ecc_mode = d.is_ecc_enabled().ok();
        let remap = self.remapped_rows(&d);
        let sbe = d
            .retired_pages(RetirementCause::MultipleSingleBitEccErrors)
            .ok();
        let dbe = d.retired_pages(RetirementCause::DoubleBitEccError).ok();
        let mut addresses: Vec<u64> = sbe
            .iter()
            .chain(dbe.iter())
            .flatten()
            .map(|p| p.address)
            .collect();
        addresses.truncate(1024);
        Some(HealthSnapshot {
            taken_at: crate::output::NowTime.to_string(),
            ecc_mode_current: ecc_mode.as_ref().map(|m| m.currently_enabled),
            ecc_mode_pending: ecc_mode.as_ref().map(|m| m.pending_enabled),
            ecc_volatile_corrected: ecc_total(MemoryError::Corrected, EccCounter::Volatile),
            ecc_volatile_uncorrected: ecc_total(MemoryError::Uncorrected, EccCounter::Volatile),
            ecc_aggregate_corrected: ecc_total(MemoryError::Corrected, EccCounter::Aggregate),
            ecc_aggregate_uncorrected: ecc_total(MemoryError::Uncorrected, EccCounter::Aggregate),
            ecc_dram_volatile_corrected: ecc_loc(
                MemoryError::Corrected,
                EccCounter::Volatile,
                MemoryLocation::Device,
            ),
            ecc_dram_volatile_uncorrected: ecc_loc(
                MemoryError::Uncorrected,
                EccCounter::Volatile,
                MemoryLocation::Device,
            ),
            ecc_dram_aggregate_corrected: ecc_loc(
                MemoryError::Corrected,
                EccCounter::Aggregate,
                MemoryLocation::Device,
            ),
            ecc_dram_aggregate_uncorrected: ecc_loc(
                MemoryError::Uncorrected,
                EccCounter::Aggregate,
                MemoryLocation::Device,
            ),
            ecc_sram_volatile_corrected: ecc_loc(
                MemoryError::Corrected,
                EccCounter::Volatile,
                MemoryLocation::SRAM,
            ),
            ecc_sram_volatile_uncorrected: ecc_loc(
                MemoryError::Uncorrected,
                EccCounter::Volatile,
                MemoryLocation::SRAM,
            ),
            ecc_sram_aggregate_corrected: ecc_loc(
                MemoryError::Corrected,
                EccCounter::Aggregate,
                MemoryLocation::SRAM,
            ),
            ecc_sram_aggregate_uncorrected: ecc_loc(
                MemoryError::Uncorrected,
                EccCounter::Aggregate,
                MemoryLocation::SRAM,
            ),
            remapped_rows_correctable: remap.map(|r| r.0),
            remapped_rows_uncorrectable: remap.map(|r| r.1),
            remap_pending: remap.map(|r| r.2),
            remap_failure: remap.map(|r| r.3),
            remap_histogram: self.remap_histogram(&d),
            retired_pages_single_bit: sbe.as_ref().map(|v| v.len() as u64),
            retired_pages_double_bit: dbe.as_ref().map(|v| v.len() as u64),
            retired_pages_pending: d.are_pages_pending_retired().ok(),
            retired_page_addresses: addresses,
            pcie_counters: Self::field_u64(&d, PCIE_FIELDS),
            aer: None,
        })
    }

    pub fn details(&self, pci_bus_id: &str) -> Option<Value> {
        let d = self.device(pci_bus_id)?;
        let mut m = Map::new();
        put(&mut m, "index", d.index());
        put(&mut m, "name", d.name());
        put(&mut m, "uuid", d.uuid());
        put(&mut m, "serial", d.serial());
        put(&mut m, "board_part_number", d.board_part_number());
        put(&mut m, "vbios_version", d.vbios_version());
        put(&mut m, "inforom_image_version", d.info_rom_image_version());
        put(
            &mut m,
            "architecture",
            d.architecture().map(|a| format!("{a:?}")),
        );
        put(
            &mut m,
            "compute_capability",
            d.cuda_compute_capability()
                .map(|c| format!("{}.{}", c.major, c.minor)),
        );
        put(&mut m, "cuda_cores", d.num_cores());
        put(&mut m, "memory_bus_width_bits", d.memory_bus_width());
        if let Ok(p) = d.pci_info() {
            m.insert(
                "pci".into(),
                json!({
                    "bus_id": p.bus_id,
                    "domain": p.domain, "bus": p.bus, "device": p.device,
                    "pci_device_id": format!("0x{:08X}", p.pci_device_id),
                    "pci_subsystem_id": p.pci_sub_system_id.map(|s| format!("0x{s:08X}")),
                }),
            );
        }
        #[cfg(target_os = "windows")]
        put(
            &mut m,
            "driver_model",
            d.driver_model().map(|s| format!("{:?}", s.current)),
        );
        put(
            &mut m,
            "memory",
            d.memory_info().map(|mi| {
                json!({
                    "total_bytes": mi.total, "used_bytes": mi.used, "free_bytes": mi.free,
                    "reserved_bytes": mi.reserved,
                })
            }),
        );
        put(
            &mut m,
            "bar1",
            d.bar1_memory_info().map(|b| {
                json!({
                    "total_bytes": b.total, "used_bytes": b.used, "free_bytes": b.free,
                })
            }),
        );
        put(
            &mut m,
            "ecc_mode",
            d.is_ecc_enabled().map(|e| {
                json!({
                    "current": e.currently_enabled, "pending": e.pending_enabled,
                })
            }),
        );
        let clocks = |c: Clock| {
            json!({
                "current_mhz": d.clock_info(c).ok(),
                "max_mhz": d.max_clock_info(c).ok(),
            })
        };
        m.insert(
            "clocks".into(),
            json!({
                "graphics": clocks(Clock::Graphics),
                "sm": clocks(Clock::SM),
                "memory": clocks(Clock::Memory),
                "video": clocks(Clock::Video),
                "gpc_vf_offset_mhz": d.gpc_clock_vf_offset().ok(),
                "mem_vf_offset_mhz": d.mem_clock_vf_offset().ok(),
            }),
        );
        m.insert("power".into(), json!({
            "usage_w": d.power_usage().ok().map(|mw| mw as f64 / 1000.0),
            "limit_w": d.power_management_limit().ok().map(|mw| mw as f64 / 1000.0),
            "enforced_limit_w": d.enforced_power_limit().ok().map(|mw| mw as f64 / 1000.0),
            "default_limit_w": d.power_management_limit_default().ok().map(|mw| mw as f64 / 1000.0),
            "total_energy_j": d.total_energy_consumption().ok().map(|mj| mj as f64 / 1000.0),
        }));
        let mem_temp = Self::field_u64(&d, &[("memory_temp", fid::NVML_FI_DEV_MEMORY_TEMP)])
            .remove("memory_temp")
            .flatten();
        m.insert("thermal".into(), json!({
            "gpu_c": d.temperature(TemperatureSensor::Gpu).ok(),
            "memory_c_nvml": mem_temp,
            "slowdown_threshold_c": d.temperature_threshold(TemperatureThreshold::Slowdown).ok(),
            "shutdown_threshold_c": d.temperature_threshold(TemperatureThreshold::Shutdown).ok(),
            "memory_max_c": d.temperature_threshold(TemperatureThreshold::MemoryMax).ok(),
            "gpu_max_c": d.temperature_threshold(TemperatureThreshold::GpuMax).ok(),
        }));
        let fans: Vec<Option<u32>> = (0..d.num_fans().unwrap_or(0))
            .map(|i| d.fan_speed(i).ok())
            .collect();
        m.insert("fans_pct".into(), json!(fans));
        put(
            &mut m,
            "throttle_reasons",
            d.current_throttle_reasons().map(|t| {
                json!({
                    "bits": t.bits(),
                    "names": super::telemetry::throttle_names(t.bits()),
                })
            }),
        );
        m.insert(
            "pcie".into(),
            json!({
                "gen_current": d.current_pcie_link_gen().ok(),
                "gen_max": d.max_pcie_link_gen().ok(),
                "width_current": d.current_pcie_link_width().ok(),
                "width_max": d.max_pcie_link_width().ok(),
                "speed_mbps_current": d.pcie_link_speed().ok(),
                "max_link_speed": d.max_pcie_link_speed().ok().map(|s| format!("{s:?}")),
                "replay_counter": d.pcie_replay_counter().ok(),
            }),
        );
        put(
            &mut m,
            "performance_state",
            d.performance_state().map(|p| format!("{p:?}")),
        );
        put(&mut m, "display_active", d.is_display_active());
        put(&mut m, "display_connected", d.is_display_connected());
        #[cfg(target_os = "linux")]
        put(&mut m, "persistence_mode", d.is_in_persistent_mode());
        Some(Value::Object(m))
    }

    /// One telemetry sample (NVML part only; NVAPI/AER are merged by the caller).
    pub fn sample(&self, pci_bus_id: &str, t_s: f64) -> Sample {
        let Some(d) = self.device(pci_bus_id) else {
            return Sample {
                t_s,
                ..Default::default()
            };
        };
        let util = d.utilization_rates().ok();
        let remap = self.remapped_rows(&d);
        let mem_temp = Self::field_u64(&d, &[("memory_temp", fid::NVML_FI_DEV_MEMORY_TEMP)])
            .remove("memory_temp")
            .flatten();
        Sample {
            t_s,
            temp_gpu_c: d.temperature(TemperatureSensor::Gpu).ok(),
            temp_hotspot_c: None,
            temp_mem_junction_c: mem_temp.filter(|t| *t > 0).map(|t| t as f32),
            power_w: d.power_usage().ok().map(|mw| mw as f64 / 1000.0),
            clock_sm_mhz: d.clock_info(Clock::SM).ok(),
            clock_mem_mhz: d.clock_info(Clock::Memory).ok(),
            util_gpu_pct: util.as_ref().map(|u| u.gpu),
            util_mem_pct: util.as_ref().map(|u| u.memory),
            util_enc_pct: d.encoder_utilization().ok().map(|u| u.utilization),
            util_dec_pct: d.decoder_utilization().ok().map(|u| u.utilization),
            pcie_gen: d.current_pcie_link_gen().ok(),
            pcie_width: d.current_pcie_link_width().ok(),
            pcie_tx_kbs: d.pcie_throughput(PcieUtilCounter::Send).ok(),
            pcie_rx_kbs: d.pcie_throughput(PcieUtilCounter::Receive).ok(),
            pcie_replay: d.pcie_replay_counter().ok().map(u64::from),
            ecc_volatile_corrected: d
                .total_ecc_errors(MemoryError::Corrected, EccCounter::Volatile)
                .ok(),
            ecc_volatile_uncorrected: d
                .total_ecc_errors(MemoryError::Uncorrected, EccCounter::Volatile)
                .ok(),
            remapped_rows_uncorrectable: remap.map(|r| r.1),
            throttle_bits: d.current_throttle_reasons().ok().map(|t| t.bits()),
            fan_pct: d.fan_speed(0).ok(),
            ..Default::default()
        }
    }
}
