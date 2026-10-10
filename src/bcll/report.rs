//! Report schema `bcll.gpu_diag.report` v1 (docs/bcll/SCHEMA.md).
//!
//! Shared with BC_LocalLLM's Python `gpu.vram_test` (engine "torch"); keep
//! field names stable and only ADD optional fields within a schema version.

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const SCHEMA: &str = "bcll.gpu_diag.report";
pub const SCHEMA_VERSION: u32 = 1;
/// Cap on per-test error records kept in the report (counts stay exact).
pub const MAX_ERROR_RECORDS: usize = 1024;

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "UPPERCASE")]
pub enum Verdict {
    Pass,
    Fail,
    Incomplete,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TestStatus {
    Pass,
    Fail,
    /// Started but did not run to its end (timeout, crash, device lost without ECC evidence).
    Incomplete,
    /// Could not run because of a tool/environment error.
    Error,
    /// The hardware or driver lacks the feature (e.g. AV1 NVENC on Ampere); never affects the verdict.
    Unsupported,
    Skipped,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct Report {
    pub schema: String,
    pub schema_version: u32,
    pub tool: ToolInfo,
    pub run: RunInfo,
    pub device: Option<DeviceInfo>,
    pub health_before: Option<Value>,
    pub health_after: Option<Value>,
    pub health_delta: Option<Value>,
    pub tests: Vec<TestResult>,
    pub telemetry: Option<Value>,
    pub verdict: Option<Verdict>,
    pub reasons: Vec<String>,
    pub warnings: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct ToolInfo {
    pub name: String,
    pub version: String,
    /// "rust" for this binary, "python" for gpu.vram_test.
    pub implementation: String,
    pub features: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct RunInfo {
    pub started_at: String,
    pub finished_at: Option<String>,
    pub duration_s: Option<f64>,
    pub os: String,
    pub hostname: Option<String>,
    pub command_line: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct DeviceInfo {
    /// What the user passed to --gpu.
    pub selector: Option<String>,
    /// Position in PCI bus order (= nvidia-smi index on all-NVIDIA hosts).
    pub index_pci_order: usize,
    /// Upstream memtest_vulkan label number (1-based, descending-bus order).
    pub vk_label: usize,
    pub pci_bus_id: String,
    pub uuid: Option<String>,
    pub name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub vulkan_driver_version: u32,
    pub vulkan_api_version: String,
    pub device_local_bytes: u64,
    /// true/false from NVML; null when unknown (treated as "attached" by the safety gate).
    pub display_attached: Option<bool>,
    /// NVML card details (docs/bcll/DESIGN.md section 5); null without NVML.
    pub details: Option<Value>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct Coverage {
    pub tested_bytes: u64,
    pub total_bytes: Option<u64>,
    pub free_bytes_at_start: Option<u64>,
    /// total - free at start: driver reservation, framebuffer, other processes.
    pub reserved_or_used_bytes: Option<u64>,
    /// NVML `reserved` memory (firmware/driver carve-out), when the driver reports it.
    pub driver_reserved_bytes: Option<u64>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq)]
pub struct ErrorRecord {
    /// Byte offset inside the tested allocation (virtual, not physical).
    pub offset_bytes: Option<u64>,
    /// Inclusive byte range when only a range is known (Vulkan memtest per window).
    pub offset_range: Option<[u64; 2]>,
    pub expected: Option<u32>,
    pub actual: Option<u32>,
    pub xor_mask: Option<u32>,
    pub count: u64,
    pub iteration: Option<u64>,
    pub detail: Option<Value>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TestResult {
    /// "vram" | "compute" | "pcie_transfer" | "codec"
    pub kind: String,
    /// "vulkan" | "cuda" | "ffmpeg" | "torch"
    pub engine: String,
    pub name: String,
    pub status: TestStatus,
    pub started_at: Option<String>,
    pub duration_s: Option<f64>,
    pub coverage: Option<Coverage>,
    pub error_count: u64,
    pub errors: Vec<ErrorRecord>,
    pub errors_truncated: bool,
    pub metrics: Value,
    pub message: Option<String>,
    /// completed | mismatch | runtime_error | device_lost | timed_out | crashed |
    /// init_failed | cancelled | unsupported (null when not determined)
    pub termination: Option<String>,
}

impl TestResult {
    pub fn new(kind: &str, engine: &str, name: &str) -> Self {
        TestResult {
            kind: kind.into(),
            engine: engine.into(),
            name: name.into(),
            status: TestStatus::Incomplete,
            started_at: None,
            duration_s: None,
            coverage: None,
            error_count: 0,
            errors: Vec::new(),
            errors_truncated: false,
            metrics: Value::Object(Default::default()),
            message: None,
            termination: None,
        }
    }

    /// Adds an error record, keeping the list bounded; `error_count` must be maintained by the caller.
    pub fn push_error(&mut self, record: ErrorRecord) {
        if self.errors.len() < MAX_ERROR_RECORDS {
            self.errors.push(record);
        } else {
            self.errors_truncated = true;
        }
    }

    pub fn set_metric(&mut self, key: &str, value: impl Serialize) {
        if let Value::Object(map) = &mut self.metrics
            && let Ok(v) = serde_json::to_value(value)
        {
            map.insert(key.to_string(), v);
        }
    }
}

impl Report {
    pub fn new(tool: ToolInfo, run: RunInfo) -> Self {
        Report {
            schema: SCHEMA.into(),
            schema_version: SCHEMA_VERSION,
            tool,
            run,
            ..Default::default()
        }
    }
}

/// Facts from health snapshots / telemetry that feed the verdict (docs/bcll/DESIGN.md section 8).
#[derive(Debug, Default, Clone)]
pub struct HealthFindings {
    pub ecc_uncorrected_delta: Option<u64>,
    pub ecc_corrected_delta: Option<u64>,
    pub remapped_uncorrectable_delta: Option<u64>,
    pub remap_failure: bool,
    pub remap_pending: bool,
    pub retired_pages_delta: Option<u64>,
    pub pcie_replay_delta: Option<u64>,
    pub aer_fatal_or_nonfatal_delta: Option<u64>,
    pub aer_correctable_delta: Option<u64>,
    pub link_below_max_while_busy: Option<String>,
    pub hw_slowdown_seen: bool,
    pub max_memory_junction_c: Option<f64>,
}

pub const MEMORY_JUNCTION_WARN_C: f64 = 100.0;
/// A passing VRAM test below this coverage is flagged (reduced allocation is not a full-card PASS).
pub const MIN_COVERAGE_PCT: f64 = 85.0;

/// Combines test outcomes and health deltas into the verdict, reasons and warnings.
pub fn decide(tests: &[TestResult], h: &HealthFindings) -> (Verdict, Vec<String>, Vec<String>) {
    let mut reasons = Vec::new();
    let mut warnings = Vec::new();
    let positive = |v: Option<u64>| v.filter(|d| *d > 0);

    for t in tests {
        if t.status == TestStatus::Fail {
            reasons.push(format!(
                "test {} ({}) failed: {} error(s){}",
                t.name,
                t.engine,
                t.error_count,
                t.message
                    .as_ref()
                    .map(|m| format!(" - {m}"))
                    .unwrap_or_default()
            ));
        }
    }
    if let Some(d) = positive(h.ecc_uncorrected_delta) {
        reasons.push(format!(
            "ECC uncorrected errors increased by {d} during the run"
        ));
    }
    if let Some(d) = positive(h.remapped_uncorrectable_delta) {
        reasons.push(format!("{d} new row remap(s) due to uncorrectable errors"));
    }
    if h.remap_failure {
        reasons.push("row remapper reports a remapping FAILURE (no spare rows)".into());
    }
    if h.remap_pending {
        reasons.push("row remap pending - reset/reboot required to apply".into());
    }
    if let Some(d) = positive(h.retired_pages_delta) {
        reasons.push(format!("{d} new retired page(s)"));
    }
    if let Some(d) = positive(h.aer_fatal_or_nonfatal_delta) {
        reasons.push(format!("PCIe AER fatal/non-fatal errors increased by {d}"));
    }

    if let Some(d) = positive(h.pcie_replay_delta) {
        warnings.push(format!(
            "PCIe replay counter increased by {d} (link-level retries)"
        ));
    }
    if let Some(d) = positive(h.aer_correctable_delta) {
        warnings.push(format!("PCIe AER correctable errors increased by {d}"));
    }
    if let Some(d) = positive(h.ecc_corrected_delta) {
        warnings.push(format!("ECC corrected errors increased by {d}"));
    }
    if let Some(link) = &h.link_below_max_while_busy {
        warnings.push(format!("PCIe link below maximum while busy: {link}"));
    }
    if h.hw_slowdown_seen {
        warnings.push("hardware/thermal slowdown throttling observed".into());
    }
    if let Some(t) = h
        .max_memory_junction_c
        .filter(|t| *t >= MEMORY_JUNCTION_WARN_C)
    {
        warnings.push(format!(
            "memory junction reached {t:.0} C (GDDR6X throttles near 110 C)"
        ));
    }
    for t in tests {
        if let Some(c) = &t.coverage
            && let Some(total) = c.total_bytes.filter(|t| *t > 0)
            && t.kind == "vram"
            && t.status == TestStatus::Pass
        {
            let pct = c.tested_bytes as f64 * 100.0 / total as f64;
            if pct < MIN_COVERAGE_PCT {
                warnings.push(format!(
                    "test {} covered only {pct:.1}% of VRAM ({} of {} bytes): not a full-card result",
                    t.name, c.tested_bytes, total
                ));
            }
        }
    }
    for t in tests {
        if t.status == TestStatus::Unsupported {
            warnings.push(format!("test {} unsupported on this card/driver", t.name));
        }
    }

    let verdict = if !reasons.is_empty() {
        Verdict::Fail
    } else if tests.is_empty()
        || tests
            .iter()
            .any(|t| matches!(t.status, TestStatus::Incomplete | TestStatus::Error))
    {
        for t in tests
            .iter()
            .filter(|t| matches!(t.status, TestStatus::Incomplete | TestStatus::Error))
        {
            reasons.push(format!(
                "test {} did not complete: {}",
                t.name,
                t.message.as_deref().unwrap_or("no detail")
            ));
        }
        if tests.is_empty() {
            reasons.push("no test ran".into());
        }
        Verdict::Incomplete
    } else {
        Verdict::Pass
    };
    (verdict, reasons, warnings)
}

pub fn exit_code(verdict: Verdict) -> i32 {
    match verdict {
        Verdict::Pass => 0,
        Verdict::Fail => 1,
        Verdict::Incomplete => 2,
    }
}

pub const EXIT_REFUSED: i32 = 3;
pub const EXIT_INTERNAL: i32 = 4;

#[cfg(test)]
mod tests {
    use super::*;

    fn test_with(status: TestStatus) -> TestResult {
        let mut t = TestResult::new("vram", "vulkan", "vulkan_memtest");
        t.status = status;
        t
    }

    #[test]
    fn all_pass_is_pass() {
        let (v, r, w) = decide(&[test_with(TestStatus::Pass)], &HealthFindings::default());
        assert_eq!(v, Verdict::Pass);
        assert!(r.is_empty() && w.is_empty());
    }

    #[test]
    fn failed_test_fails() {
        let mut t = test_with(TestStatus::Fail);
        t.error_count = 5;
        let (v, r, _) = decide(&[t], &HealthFindings::default());
        assert_eq!(v, Verdict::Fail);
        assert!(r[0].contains("5 error(s)"));
    }

    #[test]
    fn ecc_uncorrected_delta_fails_even_when_tests_pass() {
        let h = HealthFindings {
            ecc_uncorrected_delta: Some(2),
            ..Default::default()
        };
        let (v, r, _) = decide(&[test_with(TestStatus::Pass)], &h);
        assert_eq!(v, Verdict::Fail);
        assert!(r[0].contains("ECC uncorrected"));
    }

    #[test]
    fn device_lost_with_ecc_evidence_is_fail_not_incomplete() {
        let h = HealthFindings {
            ecc_uncorrected_delta: Some(1),
            ..Default::default()
        };
        let (v, _, _) = decide(&[test_with(TestStatus::Incomplete)], &h);
        assert_eq!(v, Verdict::Fail);
    }

    #[test]
    fn incomplete_without_evidence_is_incomplete() {
        let (v, r, _) = decide(
            &[test_with(TestStatus::Pass), test_with(TestStatus::Error)],
            &HealthFindings::default(),
        );
        assert_eq!(v, Verdict::Incomplete);
        assert!(r[0].contains("did not complete"));
    }

    #[test]
    fn unsupported_and_replays_only_warn() {
        let h = HealthFindings {
            pcie_replay_delta: Some(3),
            ecc_corrected_delta: Some(0),
            max_memory_junction_c: Some(104.0),
            ..Default::default()
        };
        let (v, r, w) = decide(
            &[
                test_with(TestStatus::Pass),
                test_with(TestStatus::Unsupported),
            ],
            &h,
        );
        assert_eq!(v, Verdict::Pass);
        assert!(r.is_empty());
        assert_eq!(w.len(), 3, "{w:?}");
    }

    #[test]
    fn low_coverage_pass_is_flagged() {
        let mut t = test_with(TestStatus::Pass);
        t.coverage = Some(Coverage {
            tested_bytes: 10,
            total_bytes: Some(24),
            ..Default::default()
        });
        let (v, _, w) = decide(&[t], &HealthFindings::default());
        assert_eq!(v, Verdict::Pass);
        assert!(w[0].contains("not a full-card result"), "{w:?}");
    }

    #[test]
    fn no_tests_is_incomplete() {
        let (v, _, _) = decide(&[], &HealthFindings::default());
        assert_eq!(v, Verdict::Incomplete);
    }

    #[test]
    fn error_records_are_capped_but_flagged() {
        let mut t = test_with(TestStatus::Fail);
        for _ in 0..MAX_ERROR_RECORDS + 5 {
            t.push_error(ErrorRecord::default());
        }
        assert_eq!(t.errors.len(), MAX_ERROR_RECORDS);
        assert!(t.errors_truncated);
    }

    #[test]
    fn report_serialises_with_stable_names() {
        let mut r = Report::new(ToolInfo::default(), RunInfo::default());
        let mut t = test_with(TestStatus::Pass);
        t.set_metric("check_gbps", 812.5);
        r.tests.push(t);
        r.verdict = Some(Verdict::Pass);
        let v: Value = serde_json::to_value(&r).unwrap();
        assert_eq!(v["schema"], SCHEMA);
        assert_eq!(v["schema_version"], 1);
        assert_eq!(v["verdict"], "PASS");
        assert_eq!(v["tests"][0]["status"], "pass");
        assert_eq!(v["tests"][0]["metrics"]["check_gbps"], 812.5);
        let back: Report = serde_json::from_value(v).unwrap();
        assert_eq!(back.tests[0].status, TestStatus::Pass);
    }

    /// docs/bcll/report.schema.json and these types must not drift apart.
    #[test]
    fn json_schema_matches_types() {
        let schema: Value =
            serde_json::from_str(include_str!("../../docs/bcll/report.schema.json")).unwrap();
        let mut r = Report::new(ToolInfo::default(), RunInfo::default());
        r.device = Some(DeviceInfo::default());
        let mut t = test_with(TestStatus::Pass);
        t.coverage = Some(Coverage::default());
        t.push_error(ErrorRecord::default());
        r.tests.push(t);
        r.verdict = Some(Verdict::Pass);
        let v = serde_json::to_value(&r).unwrap();
        let has_all = |obj: &Value, schema_obj: &Value, what: &str| {
            for key in schema_obj["required"].as_array().unwrap() {
                let k = key.as_str().unwrap();
                assert!(
                    obj.get(k).is_some(),
                    "{what}: required key '{k}' missing from serialisation"
                );
            }
            for k in schema_obj["properties"].as_object().unwrap().keys() {
                assert!(
                    obj.get(k).is_some(),
                    "{what}: schema property '{k}' not produced by the type"
                );
            }
        };
        has_all(&v, &schema, "report");
        has_all(&v["tool"], &schema["properties"]["tool"], "tool");
        has_all(&v["device"], &schema["properties"]["device"], "device");
        let test_schema = &schema["$defs"]["test"];
        has_all(&v["tests"][0], test_schema, "test");
        has_all(
            &v["tests"][0]["coverage"],
            &test_schema["properties"]["coverage"],
            "coverage",
        );
        has_all(
            &v["tests"][0]["errors"][0],
            &schema["$defs"]["error"],
            "error",
        );
        let statuses: Vec<&str> = test_schema["properties"]["status"]["enum"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s.as_str().unwrap())
            .collect();
        for st in [
            TestStatus::Pass,
            TestStatus::Fail,
            TestStatus::Incomplete,
            TestStatus::Error,
            TestStatus::Unsupported,
            TestStatus::Skipped,
        ] {
            let name = serde_json::to_value(st).unwrap();
            assert!(
                statuses.contains(&name.as_str().unwrap()),
                "{name} not in schema"
            );
        }
        assert_eq!(schema["properties"]["schema"]["const"], SCHEMA);
        assert_eq!(
            schema["properties"]["schema_version"]["const"],
            SCHEMA_VERSION
        );
    }
}
