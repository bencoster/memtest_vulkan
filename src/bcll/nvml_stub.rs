//! Stand-in for `nvml.rs` when the crate is built without the `nvml` feature.
//! Same API; `open()` always fails so callers take their "no NVML" paths.

use super::telemetry::{HealthSnapshot, Sample};
use serde_json::Value;

pub struct NvmlCtx;

#[derive(Debug, Clone, Default)]
pub struct Usage {
    pub total_bytes: u64,
    pub used_bytes: u64,
    pub free_bytes: u64,
    pub reserved_bytes: Option<u64>,
    pub compute_pids: Vec<u32>,
}

impl NvmlCtx {
    pub fn open() -> Result<Self, String> {
        Err("built without the 'nvml' feature".into())
    }
    pub fn system(&self) -> Value {
        Value::Null
    }
    pub fn uuid(&self, _pci: &str) -> Option<String> {
        None
    }
    pub fn architecture(&self, _pci: &str) -> Option<String> {
        None
    }
    pub fn display_attached(&self, _pci: &str) -> Option<bool> {
        None
    }
    pub fn usage(&self, _pci: &str) -> Option<Usage> {
        None
    }
    pub fn max_link(&self, _pci: &str) -> Option<(u32, u32)> {
        None
    }
    pub fn health(&self, _pci: &str) -> Option<HealthSnapshot> {
        None
    }
    pub fn details(&self, _pci: &str) -> Option<Value> {
        None
    }
    pub fn sample(&self, _pci: &str, t_s: f64) -> Sample {
        Sample {
            t_s,
            ..Default::default()
        }
    }
}
