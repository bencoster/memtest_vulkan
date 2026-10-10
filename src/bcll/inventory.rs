//! GPU inventory and selection. Enumeration runs INSIDE the query worker
//! (`query.rs`): the supervisor only receives the resulting JSON, so a wedged
//! driver cannot stall it (Codex BLOCK 89de0c26 #1).

use super::cli::Options;
use super::nvml::NvmlCtx;
use super::report::{self, ToolInfo};
use super::select;
use crate::PciAddress;
use erupt::vk;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::ffi::CStr;
use std::sync::atomic::Ordering::SeqCst;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GpuRecord {
    pub vk_label: usize,
    pub vk_enum_index: usize,
    pub name: String,
    pub pci: PciAddress,
    pub pci_bus_id: String,
    /// VkPhysicalDeviceIDProperties::deviceUUID (what Vulkan workers verify)
    pub vk_uuid: Option<String>,
    /// NVML UUID joined by PCI id (what CUDA / nvidia-smi use)
    pub nvml_uuid: Option<String>,
    /// nvidia-smi index joined by PCI id (None without NVML / non-NVIDIA)
    pub nvml_index: Option<u32>,
    pub vendor_id: u32,
    pub device_id: u32,
    pub driver_version: u32,
    pub api_version: String,
    pub device_local_bytes: u64,
    /// upstream's free-VRAM budget for the memtest (heap free - 400 MB)
    pub memtest_budget_bytes: i64,
    /// position in PCI-address order among Vulkan GPUs (informational only)
    pub index_pci_order: usize,
}

impl GpuRecord {
    /// The UUID to show / pin CUDA with: NVML's, else Vulkan's.
    pub fn uuid(&self) -> Option<String> {
        self.nvml_uuid.clone().or_else(|| self.vk_uuid.clone())
    }

    /// Both identities present but different: never pick by UUID silently.
    pub fn uuid_conflict(&self) -> bool {
        matches!((&self.vk_uuid, &self.nvml_uuid), (Some(a), Some(b)) if !a.eq_ignore_ascii_case(b))
    }

    pub fn candidate(&self) -> select::Candidate {
        select::Candidate {
            pci: self.pci,
            uuid: self.uuid(),
            vk_label: self.vk_label,
            nvml_index: self.nvml_index,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Inventory {
    pub gpus: Vec<GpuRecord>,
    pub nvml_error: Option<String>,
}

pub(super) fn tool_info() -> ToolInfo {
    let mut features = vec!["diagnostics".to_string()];
    for (on, name) in [
        (cfg!(feature = "nvml"), "nvml"),
        (cfg!(feature = "nvapi"), "nvapi"),
        (cfg!(feature = "cuda"), "cuda"),
    ] {
        if on {
            features.push(name.into());
        }
    }
    ToolInfo {
        name: "memtest_vulkan".into(),
        version: format!("{}-bcll.2", env!("CARGO_PKG_VERSION")),
        implementation: "rust".into(),
        features,
    }
}

pub(super) fn hostname() -> Option<String> {
    std::env::var("COMPUTERNAME")
        .or_else(|_| std::env::var("HOSTNAME"))
        .ok()
        .or_else(|| std::fs::read_to_string("/etc/hostname").ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Enumerates Vulkan GPUs through upstream's own enumeration (same labels, same budget
/// logic) and joins NVML by PCI id. Runs in the query worker process.
pub fn enumerate(nvml: Option<&NvmlCtx>) -> Result<Vec<GpuRecord>, String> {
    let env = crate::ProcessEnv::default();
    let mut log = crate::output::LogDupler::new(std::io::sink(), None, 0, "bcll inventory");
    let loaded = crate::list_devices_ordered_labaled_from_1(&env, &mut log)
        .map_err(|e| format!("Vulkan enumeration failed: {e}"))?;
    let mut out = Vec::new();
    for (i, d) in loaded.devices.iter().enumerate() {
        if d.physical_props.device_type == vk::PhysicalDeviceType::CPU {
            continue; // software rasterisers (llvmpipe) are not test targets
        }
        let budget_env = crate::ProcessEnv::default();
        crate::try_fill_default_mem_budget(d, &budget_env, &mut log);
        let device_local_bytes = (0..d.memory_props.memory_heap_count as usize)
            .filter(|h| {
                d.memory_props.memory_heaps[*h]
                    .flags
                    .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
            })
            .map(|h| d.memory_props.memory_heaps[h].size)
            .max()
            .unwrap_or(0);
        let pci_bus_id = select::format_pci(&d.pci);
        let pci_ok = select::pci_known(&d.pci);
        let props = d.physical_props;
        out.push(GpuRecord {
            vk_label: i + 1,
            vk_enum_index: d.index_in_device_iteration,
            // SAFETY: device_name is a NUL-terminated array filled by the driver.
            name: unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
                .to_string_lossy()
                .into_owned(),
            pci: d.pci,
            vk_uuid: d
                .device_uuid
                .filter(|u| u.iter().any(|b| *b != 0))
                .map(|u| select::uuid_string(&u)),
            // joining by an all-zero PCI id would attach another card's identity
            nvml_uuid: nvml.filter(|_| pci_ok).and_then(|n| n.uuid(&pci_bus_id)),
            nvml_index: nvml.filter(|_| pci_ok).and_then(|n| n.index(&pci_bus_id)),
            pci_bus_id,
            vendor_id: props.vendor_id,
            device_id: props.device_id,
            driver_version: props.driver_version,
            api_version: format!(
                "{}.{}.{}",
                vk::api_version_major(props.api_version),
                vk::api_version_minor(props.api_version),
                vk::api_version_patch(props.api_version)
            ),
            device_local_bytes,
            memtest_budget_bytes: budget_env.max_test_bytes.load(SeqCst),
            index_pci_order: 0,
        });
    }
    let cands: Vec<select::Candidate> = out.iter().map(|g| g.candidate()).collect();
    for (pos, i) in select::pci_order(&cands).into_iter().enumerate() {
        out[i].index_pci_order = pos;
    }
    out.sort_by_key(|g| g.index_pci_order);
    Ok(out)
}

pub(super) fn gib(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

pub(super) fn list(opts: &Options) -> i32 {
    let exe = match std::env::current_exe() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("memtest_vulkan: cannot locate own executable: {e}");
            return report::EXIT_INTERNAL;
        }
    };
    let inv = match super::query::inventory(&exe) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("memtest_vulkan: {e}");
            return report::EXIT_INTERNAL;
        }
    };
    // display state per card, each from its own bounded snapshot worker
    let display: Vec<Option<bool>> = inv
        .gpus
        .iter()
        .map(|g| super::query::snapshot(&exe, g, true).display_attached)
        .collect();
    if opts.json {
        let v: Vec<Value> = inv
            .gpus
            .iter()
            .zip(&display)
            .map(|(g, d)| {
                json!({
                    "nvml_index": g.nvml_index, "pci_bus_id": g.pci_bus_id, "vk_label": g.vk_label,
                    "uuid": g.uuid(), "vk_uuid": g.vk_uuid, "nvml_uuid": g.nvml_uuid,
                    "name": g.name, "device_local_bytes": g.device_local_bytes,
                    "display_attached": d,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&v).unwrap_or_default());
        return 0;
    }
    println!(
        "{:<5} {:<13} {:<4} {:<41} {:>7}  {:<8} name",
        "smi", "pci", "vk", "uuid", "VRAM", "display"
    );
    for (g, d) in inv.gpus.iter().zip(&display) {
        println!(
            "{:<5} {:<13} {:<4} {:<41} {:>6.1}G  {:<8} {}{}",
            g.nvml_index
                .map(|i| i.to_string())
                .unwrap_or_else(|| "-".into()),
            g.pci_bus_id,
            g.vk_label,
            g.uuid().as_deref().unwrap_or("-"),
            gib(g.device_local_bytes),
            match d {
                Some(true) => "YES",
                Some(false) => "no",
                None => "unknown",
            },
            g.name,
            if g.uuid_conflict() {
                "  (Vulkan/NVML UUID differ!)"
            } else {
                ""
            }
        );
    }
    if let Some(e) = &inv.nvml_error {
        println!(
            "(NVML unavailable: {e}; no nvidia-smi indices, display state unknown -> tests need --allow-display; select by PCI or UUID)"
        );
    }
    0
}

pub(super) fn select_gpu(inv: &[GpuRecord], spec_raw: &str) -> Result<GpuRecord, String> {
    let spec = select::parse_spec(spec_raw)?;
    let cands: Vec<select::Candidate> = inv.iter().map(|g| g.candidate()).collect();
    let g = select::select(&cands, &spec).map(|i| inv[i].clone())?;
    if !select::pci_known(&g.pci) {
        return Err(format!(
            "{} has no PCI identity (driver lacks VK_EXT_pci_bus_info); workers cannot verify it",
            g.name
        ));
    }
    if g.uuid_conflict() {
        return Err(format!(
            "{}: Vulkan UUID {:?} differs from NVML UUID {:?}; refusing an ambiguous identity",
            g.pci_bus_id, g.vk_uuid, g.nvml_uuid
        ));
    }
    Ok(g)
}
