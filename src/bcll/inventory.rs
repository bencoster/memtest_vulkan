//! GPU inventory and selection: upstream's Vulkan enumeration merged with NVML
//! identity, numbered in PCI bus order (DESIGN.md section 4).

use super::cli::Options;
use super::nvml::NvmlCtx;
use super::report::{self, ToolInfo};
use super::select;
use crate::PciAddress;
use erupt::vk;
use serde_json::{Value, json};
use std::ffi::CStr;
use std::sync::atomic::Ordering::SeqCst;

#[derive(Debug, Clone)]
pub struct GpuRecord {
    pub vk_label: usize,
    pub vk_enum_index: usize,
    pub name: String,
    pub pci: PciAddress,
    pub pci_bus_id: String,
    pub uuid: Option<String>,
    pub vendor_id: u32,
    pub device_id: u32,
    pub driver_version: u32,
    pub api_version: String,
    pub device_local_bytes: u64,
    /// upstream's free-VRAM budget for the memtest (heap free - 400 MB)
    pub memtest_budget_bytes: i64,
    pub index_pci_order: usize,
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
        version: format!("{}-bcll.1", env!("CARGO_PKG_VERSION")),
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

/// Enumerates Vulkan GPUs through upstream's own enumeration (same labels, same budget logic).
pub fn inventory(nvml: Option<&NvmlCtx>) -> Result<Vec<GpuRecord>, String> {
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
        let vk_uuid = d
            .device_uuid
            .filter(|u| u.iter().any(|b| *b != 0))
            .map(|u| select::uuid_string(&u));
        let nvml_uuid = nvml.and_then(|n| n.uuid(&pci_bus_id));
        let props = d.physical_props;
        out.push(GpuRecord {
            vk_label: i + 1,
            vk_enum_index: d.index_in_device_iteration,
            // SAFETY: device_name is a NUL-terminated array filled by the driver.
            name: unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
                .to_string_lossy()
                .into_owned(),
            pci: d.pci,
            pci_bus_id,
            uuid: nvml_uuid.or(vk_uuid),
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
    let cands: Vec<select::Candidate> = out
        .iter()
        .map(|g| select::Candidate {
            pci: g.pci,
            uuid: g.uuid.clone(),
            vk_label: g.vk_label,
        })
        .collect();
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
    let nvml = NvmlCtx::open().ok();
    let inv = match inventory(nvml.as_ref()) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("memtest_vulkan: {e}");
            return report::EXIT_INTERNAL;
        }
    };
    let display = |g: &GpuRecord| {
        nvml.as_ref()
            .and_then(|n| n.display_attached(&g.pci_bus_id))
    };
    if opts.json {
        let v: Vec<Value> = inv
            .iter()
            .map(|g| {
                json!({
                    "index": g.index_pci_order, "pci_bus_id": g.pci_bus_id, "vk_label": g.vk_label,
                    "uuid": g.uuid, "name": g.name, "device_local_bytes": g.device_local_bytes,
                    "display_attached": display(g),
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&v).unwrap_or_default());
        return 0;
    }
    println!(
        "{:<5} {:<13} {:<4} {:<41} {:>7}  {:<8} name",
        "index", "pci", "vk", "uuid", "VRAM", "display"
    );
    for g in &inv {
        println!(
            "{:<5} {:<13} {:<4} {:<41} {:>6.1}G  {:<8} {}",
            g.index_pci_order,
            g.pci_bus_id,
            g.vk_label,
            g.uuid.as_deref().unwrap_or("-"),
            gib(g.device_local_bytes),
            match display(g) {
                Some(true) => "YES",
                Some(false) => "no",
                None => "unknown",
            },
            g.name
        );
    }
    if nvml.is_none() {
        println!("(NVML unavailable: display state unknown; tests will require --allow-display)");
    }
    0
}

pub(super) fn select_gpu(inv: &[GpuRecord], spec_raw: &str) -> Result<GpuRecord, String> {
    let spec = select::parse_spec(spec_raw)?;
    // `select` indexes the slice it is given; inventory is already sorted by PCI order
    let cands: Vec<select::Candidate> = inv
        .iter()
        .map(|g| select::Candidate {
            pci: g.pci,
            uuid: g.uuid.clone(),
            vk_label: g.vk_label,
        })
        .collect();
    select::select(&cands, &spec).map(|i| inv[i].clone())
}
