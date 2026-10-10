//! GDDR6X memory-junction / hotspot temperature via the UNDOCUMENTED NVAPI call
//! `NvAPI_GPU_ThermalGetSensors` (Windows, feature `nvapi`; owner decision 2026-10-10).
//!
//! NVML does not expose memory-junction temperature on GeForce. HWiNFO, GPU-Z
//! and LibreHardwareMonitor read it through this NVAPI interface; the layout and
//! channel mapping below follow LibreHardwareMonitor (MPL-2.0, facts only):
//! interface id 0x65FE3AAD, struct version 2 (168 bytes: version, mask,
//! 8 reserved i32, 32 temperatures as i32 in 1/256 C), valid mask found by
//! probing bits, hotspot = channel 1, memory junction = channel 9 (Ampere),
//! 7 (Ada), 2 (Blackwell).
//!
//! Fail closed: any non-OK status (incl. INCOMPATIBLE_STRUCT_VERSION) disables
//! the reader; values outside 1..=150 C are discarded. Report source is
//! "nvapi_undocumented". Linux has no supported userspace equivalent.

// pure helpers stay compiled (and unit-tested) on builds without the Windows reader
#![cfg_attr(not(all(windows, feature = "nvapi")), allow(dead_code))]

use serde::Serialize;

#[derive(Serialize, Debug, Clone, Default, PartialEq)]
pub struct NvapiTemps {
    pub hotspot_c: Option<f32>,
    pub mem_junction_c: Option<f32>,
    /// All valid channels (index -> C), for later re-interpretation.
    pub raw_c: Vec<(usize, f32)>,
}

pub const SOURCE: &str = "nvapi_undocumented";

/// Channel holding the memory-junction temperature, by architecture / name.
pub fn junction_channel(arch: Option<&str>, name: &str) -> Option<usize> {
    let arch = arch.unwrap_or("").to_ascii_lowercase();
    let name = name.to_ascii_uppercase();
    if arch.contains("blackwell") || name.contains("RTX 50") {
        Some(2)
    } else if arch.contains("ada") || name.contains("RTX 40") {
        Some(7)
    } else if arch.contains("ampere") || name.contains("RTX 30") {
        Some(9)
    } else {
        None
    }
}

/// Hotspot channel (zeroed on Blackwell per LibreHardwareMonitor).
pub fn hotspot_channel(arch: Option<&str>, name: &str) -> Option<usize> {
    let blackwell = arch
        .unwrap_or("")
        .to_ascii_lowercase()
        .contains("blackwell")
        || name.to_ascii_uppercase().contains("RTX 50");
    (!blackwell).then_some(1)
}

fn plausible(raw: i32) -> Option<f32> {
    let c = raw as f32 / 256.0;
    (1.0..=150.0).contains(&c).then_some(c)
}

/// Mask with every channel below the first failing probe bit (LibreHardwareMonitor logic).
pub fn mask_from_probe(mut probe_ok: impl FnMut(u32) -> bool) -> Option<u32> {
    for bit in 0..32u32 {
        if !probe_ok(1u32 << bit) {
            return (bit > 0).then(|| (1u32 << bit) - 1);
        }
    }
    Some(u32::MAX)
}

pub fn interpret(
    temps: &[i32; 32],
    mask: u32,
    junction: Option<usize>,
    hotspot: Option<usize>,
) -> NvapiTemps {
    let raw_c: Vec<(usize, f32)> = (0..32)
        .filter(|i| mask & (1 << i) != 0)
        .filter_map(|i| plausible(temps[i]).map(|c| (i, c)))
        .collect();
    let pick =
        |ch: Option<usize>| ch.and_then(|i| raw_c.iter().find(|(j, _)| *j == i).map(|(_, c)| *c));
    NvapiTemps {
        hotspot_c: pick(hotspot),
        mem_junction_c: pick(junction),
        raw_c,
    }
}

#[cfg(all(windows, feature = "nvapi"))]
mod imp {
    use super::*;
    use std::ffi::c_void;

    type QueryInterface = unsafe extern "C" fn(u32) -> *mut c_void;
    type Initialize = unsafe extern "C" fn() -> i32;
    type EnumPhysicalGpus = unsafe extern "C" fn(*mut [*mut c_void; 64], *mut u32) -> i32;
    type GetBusId = unsafe extern "C" fn(*mut c_void, *mut u32) -> i32;
    type ThermalGetSensors = unsafe extern "C" fn(*mut c_void, *mut ThermalSensors) -> i32;

    const ID_INITIALIZE: u32 = 0x0150_E828;
    const ID_ENUM_PHYSICAL_GPUS: u32 = 0xE5AC_921F;
    const ID_GPU_GET_BUS_ID: u32 = 0x1BE0_B8E5;
    const ID_GPU_THERMAL_GET_SENSORS: u32 = 0x65FE_3AAD;

    #[repr(C)]
    pub struct ThermalSensors {
        version: u32,
        mask: u32,
        reserved: [i32; 8],
        temperatures: [i32; 32],
    }
    const _: () = assert!(std::mem::size_of::<ThermalSensors>() == 168);
    const THERMAL_SENSORS_VERSION: u32 = 168 | (2 << 16);

    impl ThermalSensors {
        fn new(mask: u32) -> Self {
            ThermalSensors {
                version: THERMAL_SENSORS_VERSION,
                mask,
                reserved: [0; 8],
                temperatures: [0; 32],
            }
        }
    }

    pub struct Reader {
        _lib: libloading::Library,
        get: ThermalGetSensors,
        gpu: *mut c_void,
        mask: u32,
        junction: Option<usize>,
        hotspot: Option<usize>,
        disabled: bool,
    }

    impl Reader {
        pub fn open(pci_bus: u32, arch: Option<&str>, name: &str) -> Result<Reader, String> {
            // SAFETY: loading the NVIDIA-shipped system DLL; all symbols are checked for null.
            unsafe {
                let lib = libloading::Library::new("nvapi64.dll")
                    .map_err(|e| format!("nvapi64.dll not loadable: {e}"))?;
                let qi: QueryInterface = *lib
                    .get::<QueryInterface>(b"nvapi_QueryInterface\0")
                    .map_err(|e| format!("nvapi_QueryInterface missing: {e}"))?;
                let sym = |id: u32| -> Result<*mut c_void, String> {
                    let p = qi(id);
                    if p.is_null() {
                        Err(format!("NVAPI interface 0x{id:08X} unavailable"))
                    } else {
                        Ok(p)
                    }
                };
                let init: Initialize = std::mem::transmute(sym(ID_INITIALIZE)?);
                let enum_gpus: EnumPhysicalGpus = std::mem::transmute(sym(ID_ENUM_PHYSICAL_GPUS)?);
                let bus_id: GetBusId = std::mem::transmute(sym(ID_GPU_GET_BUS_ID)?);
                let get: ThermalGetSensors = std::mem::transmute(sym(ID_GPU_THERMAL_GET_SENSORS)?);
                if init() != 0 {
                    return Err("NvAPI_Initialize failed".into());
                }
                let mut handles = [std::ptr::null_mut(); 64];
                let mut count = 0u32;
                if enum_gpus(&mut handles, &mut count) != 0 {
                    return Err("NvAPI_EnumPhysicalGPUs failed".into());
                }
                let gpu = handles[..(count as usize).min(64)]
                    .iter()
                    .copied()
                    .find(|h| {
                        let mut b = u32::MAX;
                        bus_id(*h, &mut b) == 0 && b == pci_bus
                    })
                    .ok_or_else(|| format!("no NVAPI GPU on PCI bus 0x{pci_bus:02X}"))?;
                let mask = mask_from_probe(|m| {
                    let mut s = ThermalSensors::new(m);
                    get(gpu, &mut s) == 0
                })
                .ok_or("NvAPI_GPU_ThermalGetSensors unsupported (struct version or GPU)")?;
                Ok(Reader {
                    _lib: lib,
                    get,
                    gpu,
                    mask,
                    junction: junction_channel(arch, name),
                    hotspot: hotspot_channel(arch, name),
                    disabled: false,
                })
            }
        }

        pub fn read(&mut self) -> Option<NvapiTemps> {
            if self.disabled {
                return None;
            }
            let mut s = ThermalSensors::new(self.mask);
            // SAFETY: function pointer obtained from nvapi_QueryInterface; struct is versioned.
            let rc = unsafe { (self.get)(self.gpu, &mut s) };
            if rc != 0 {
                self.disabled = true;
                return None;
            }
            Some(interpret(
                &s.temperatures,
                self.mask,
                self.junction,
                self.hotspot,
            ))
        }
    }
}

#[cfg(not(all(windows, feature = "nvapi")))]
mod imp {
    use super::*;
    pub struct Reader;
    impl Reader {
        pub fn open(_pci_bus: u32, _arch: Option<&str>, _name: &str) -> Result<Reader, String> {
            Err("NVAPI thermal reader is Windows-only (feature 'nvapi')".into())
        }
        pub fn read(&mut self) -> Option<NvapiTemps> {
            None
        }
    }
}

pub use imp::Reader;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_mapping_matches_lhm() {
        assert_eq!(
            junction_channel(Some("Ampere"), "NVIDIA GeForce RTX 3090"),
            Some(9)
        );
        assert_eq!(
            junction_channel(Some("Ada"), "NVIDIA GeForce RTX 4090"),
            Some(7)
        );
        assert_eq!(junction_channel(None, "NVIDIA GeForce RTX 4090"), Some(7));
        assert_eq!(junction_channel(None, "NVIDIA GeForce RTX 5090"), Some(2));
        assert_eq!(junction_channel(Some("Turing"), "RTX 2080"), None);
        assert_eq!(hotspot_channel(Some("Blackwell"), "RTX 5090"), None);
        assert_eq!(hotspot_channel(Some("Ampere"), "RTX 3090"), Some(1));
    }

    #[test]
    fn probe_builds_mask_below_first_failure() {
        assert_eq!(mask_from_probe(|m| m < (1 << 10)), Some(0x3FF));
        assert_eq!(mask_from_probe(|_| false), None);
        assert_eq!(mask_from_probe(|_| true), Some(u32::MAX));
    }

    #[test]
    fn interpret_scales_and_filters() {
        let mut t = [0i32; 32];
        t[0] = 60 * 256;
        t[1] = 75 * 256 + 128; // 75.5 C hotspot
        t[9] = 96 * 256; // junction on Ampere
        t[12] = 400 * 256; // implausible, outside mask anyway
        let r = interpret(&t, 0x3FF, Some(9), Some(1));
        assert_eq!(r.hotspot_c, Some(75.5));
        assert_eq!(r.mem_junction_c, Some(96.0));
        assert_eq!(r.raw_c.len(), 3);
        // junction channel outside the valid mask -> None, never a garbage value
        let r = interpret(&t, 0x3, Some(9), Some(1));
        assert_eq!(r.mem_junction_c, None);
    }
}
