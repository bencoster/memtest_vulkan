//! Pure judges: turn worker exit codes + JSON-lines events into test results
//! (DESIGN.md sections 7 and 8). No I/O here, so every rule is unit-tested.

use super::events::of_kind;
use super::report::{Coverage, ErrorRecord, TestResult, TestStatus};
use serde_json::{Value, json};

/// Lower-memory retries after an out-of-memory init failure (upstream retried unboundedly).
pub const MAX_ALLOC_RETRIES: usize = 8;
const OOM_MARKERS: &[&str] = &[
    "OUT_OF_DEVICE_MEMORY",
    "OUT_OF_HOST_MEMORY",
    "OUT_OF_POOL_MEMORY",
    "FRAGMENTED_POOL",
    "TOO_MANY_OBJECTS",
];

#[derive(Debug, PartialEq)]
pub enum InitOutcome {
    /// the test loop started (or the child ended for another reason): judge it
    Proceed,
    /// out-of-memory during init: retry with less memory
    RetryLower(String),
    /// init failed for a non-memory reason (e.g. DEVICE_LOST) or retries exhausted: stop
    Stop(String),
}

/// Codex audit F3: never treat DEVICE_LOST (or an unknown init failure) as memory pressure.
pub fn init_outcome(code: Option<i32>, ev: &[Value], retries_done: usize) -> InitOutcome {
    use crate::close::app_status as st;
    let init_abort = code.map(|c| c as u32 as u8) == Some(st::SIGNATURE | st::RUNTIME_ABORT);
    if !init_abort || of_kind(ev, "memtest_started").next().is_some() {
        return InitOutcome::Proceed;
    }
    let msg = of_kind(ev, "memtest_init_failure")
        .last()
        .and_then(|e| e["message"].as_str())
        .unwrap_or("init failed without a reported Vulkan error")
        .to_string();
    if !OOM_MARKERS.iter().any(|m| msg.contains(m)) {
        InitOutcome::Stop(msg)
    } else if retries_done >= MAX_ALLOC_RETRIES {
        InitOutcome::Stop(format!(
            "{msg} (gave up after {retries_done} smaller allocations)"
        ))
    } else {
        InitOutcome::RetryLower(msg)
    }
}

/// Converts the upstream memtest child's exit status + events into a test result (pure; unit-tested).
pub fn judge_memtest(
    mut t: TestResult,
    outcome_code: Option<i32>,
    timed_out: bool,
    ev: &[Value],
) -> TestResult {
    use crate::close::app_status as st;
    for e in of_kind(ev, "memtest_error") {
        let count = e["total_errors"].as_u64().unwrap_or(0);
        // only all-ones words seen: the non-FF min/max are still at their sentinels
        let empty_range = e["actual_min"].as_u64() > e["actual_max"].as_u64();
        t.error_count += count.max(1);
        let nonzero = |key: &str| -> Value {
            let m: serde_json::Map<String, Value> = e[key]
                .as_array()
                .map(|a| {
                    a.iter()
                        .enumerate()
                        .filter(|(_, v)| v.as_u64().unwrap_or(0) > 0)
                        .map(|(i, v)| (i.to_string(), v.clone()))
                        .collect()
                })
                .unwrap_or_default();
            Value::Object(m)
        };
        t.push_error(ErrorRecord {
            offset_range: match (e["addr_first"].as_u64(), e["addr_last"].as_u64()) {
                (Some(a), Some(b)) => Some([a, b]),
                _ => None,
            },
            count,
            iteration: e["iteration"].as_u64(),
            detail: Some(json!({
                "mode": e["mode"], "window_offset": e["window_offset"],
                "single_bit_errors_by_bit": nonzero("single_bit_idx"),
                "errors_by_flipped_bit_count": nonzero("toggle_count"),
                "bad_values_by_popcount": nonzero("ones_in_value"),
                "all_ones_words": e["actual_ff"],
                "actual_min": if empty_range { Value::Null } else { e["actual_min"].clone() },
                "actual_max": if empty_range { Value::Null } else { e["actual_max"].clone() },
            })),
            ..Default::default()
        });
    }
    let progress: Vec<&Value> = of_kind(ev, "memtest_progress").collect();
    let avg = |key: &str| {
        let v: Vec<f64> = progress
            .iter()
            .filter_map(|p| p[key].as_f64())
            .filter(|x| *x > 0.0)
            .collect();
        (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64)
    };
    t.set_metric(
        "iterations",
        progress.last().and_then(|p| p["iteration"].as_u64()),
    );
    t.set_metric("write_gbps_avg", avg("write_gbps"));
    t.set_metric("check_gbps_avg", avg("check_gbps"));
    let finished = of_kind(ev, "memtest_finished")
        .last()
        .and_then(|f| f["reason"].as_str())
        .map(String::from);
    t.set_metric("finish_reason", finished.clone());
    t.set_metric("exit_code", outcome_code);

    let code = outcome_code.map(|c| c as u32 as u8);
    let has_sig = code.is_some_and(|c| c & st::SIGNATURE_MASK == st::SIGNATURE);
    let bits = |b: u8| code.is_some_and(|c| has_sig && c & b != 0);
    t.status = if bits(st::RUNTIME_ERRORS) || t.error_count > 0 {
        TestStatus::Fail
    } else if timed_out {
        t.message = Some("memtest child exceeded its deadline and was killed (GPU hang?)".into());
        TestStatus::Incomplete
    } else if !has_sig {
        t.message = Some(format!(
            "memtest child crashed (exit code {outcome_code:?})"
        ));
        TestStatus::Incomplete
    } else if !bits(st::INITED_OK) {
        t.message = Some("memtest did not complete initialisation".into());
        TestStatus::Incomplete
    } else if bits(st::RUNTIME_ABORT) {
        t.message = Some("runtime error after init (VK_ERROR_DEVICE_LOST? check ECC/Xid)".into());
        TestStatus::Incomplete
    } else if finished.as_deref() == Some("duration_reached") {
        TestStatus::Pass
    } else {
        t.message = Some(format!("memtest stopped early ({finished:?})"));
        TestStatus::Incomplete
    };
    if t.status == TestStatus::Fail && t.message.is_none() {
        t.message = Some("memory errors detected by the Vulkan memtest".into());
    }
    t.termination = Some(
        match t.status {
            TestStatus::Fail => "mismatch",
            TestStatus::Pass => "completed",
            _ if timed_out => "timed_out",
            _ if !has_sig => "crashed",
            _ if !bits(st::INITED_OK) => "init_failed",
            _ if bits(st::RUNTIME_ABORT) => "runtime_error",
            _ => "cancelled",
        }
        .into(),
    );
    t
}

/// Terminal state vocabulary (Codex audit): completed, mismatch, runtime_error, device_lost,
/// timed_out, crashed, init_failed, cancelled, unsupported.
pub fn worker_termination(
    status: TestStatus,
    code: Option<i32>,
    timed_out: bool,
    ev: &[Value],
) -> &'static str {
    let err = of_kind(ev, "worker_error").last();
    let flag = |k: &str| err.is_some_and(|e| e[k] == true);
    match status {
        TestStatus::Pass => "completed",
        TestStatus::Fail if flag("ecc_uncorrectable") => "device_lost",
        TestStatus::Fail => "mismatch",
        TestStatus::Unsupported => "unsupported",
        _ if timed_out || flag("timeout") => "timed_out",
        _ if flag("device_lost") || flag("gpu_fault") => "device_lost",
        _ if !matches!(code, Some(0..=4)) => "crashed",
        _ => "runtime_error",
    }
}

pub(super) fn worker_status(
    code: Option<i32>,
    timed_out: bool,
    ev: &[Value],
) -> (TestStatus, Option<String>) {
    let err = of_kind(ev, "worker_error").last();
    let msg = err.and_then(|e| e["message"].as_str()).map(String::from);
    if timed_out {
        return (
            TestStatus::Incomplete,
            Some("worker exceeded its deadline and was killed (GPU hang?)".into()),
        );
    }
    match code {
        Some(0) => (TestStatus::Pass, None),
        Some(1) => (TestStatus::Fail, msg),
        Some(2) => (TestStatus::Incomplete, msg),
        Some(4) if err.is_some_and(|e| e["unsupported"] == true) => (TestStatus::Unsupported, msg),
        Some(4) => (TestStatus::Error, msg),
        other => (
            TestStatus::Incomplete,
            Some(format!(
                "worker crashed (exit code {other:?}){}",
                msg.map(|m| format!(": {m}")).unwrap_or_default()
            )),
        ),
    }
}

/// CUDA worker exit + events -> result (pure; unit-tested).
pub fn judge_cuda(
    mut t: TestResult,
    code: Option<i32>,
    timed_out: bool,
    ev: &[Value],
) -> TestResult {
    (t.status, t.message) = worker_status(code, timed_out, ev);
    if let Some(c) = of_kind(ev, "cuda_coverage").last() {
        t.coverage = Some(Coverage {
            tested_bytes: c["tested_bytes"].as_u64().unwrap_or(0),
            total_bytes: c["total_bytes"].as_u64(),
            free_bytes_at_start: c["free_bytes_at_start"].as_u64(),
            reserved_or_used_bytes: c["reserved_or_used_bytes"].as_u64(),
            driver_reserved_bytes: None,
        });
        for k in [
            "granularity_bytes",
            "chunks",
            "keep_free_bytes",
            "requested_bytes",
            "sm_count",
        ] {
            t.set_metric(k, c[k].clone());
        }
    }
    let passes: Vec<&Value> = of_kind(ev, "cuda_pass").collect();
    let mut by_pass: std::collections::BTreeMap<String, (u64, f64, u64)> = Default::default();
    for p in &passes {
        let e = by_pass
            .entry(p["pass"].as_str().unwrap_or("?").to_string())
            .or_default();
        e.0 += 1;
        e.1 += p["gbps"].as_f64().unwrap_or(0.0);
        e.2 += p["errors"].as_u64().unwrap_or(0);
    }
    t.set_metric(
        "passes",
        by_pass
            .iter()
            .map(|(k, (n, g, e))| {
                (
                    k.clone(),
                    json!({"runs": n, "gbps_avg": g / *n as f64, "errors": e}),
                )
            })
            .collect::<serde_json::Map<String, Value>>(),
    );
    t.error_count = passes.iter().filter_map(|p| p["errors"].as_u64()).sum();
    if let Some(s) = of_kind(ev, "cuda_summary").last() {
        t.set_metric("rounds_completed", s["rounds_completed"].clone());
        t.set_metric("bit_histogram", s["bit_histogram"].clone());
    }
    if let Some(e) = of_kind(ev, "worker_error").last()
        && e["ecc_uncorrectable"] == true
    {
        t.status = TestStatus::Fail;
        t.error_count = t.error_count.max(1);
        t.message = Some("CUDA_ERROR_ECC_UNCORRECTABLE during the test".into());
    }
    for e in of_kind(ev, "cuda_error") {
        t.push_error(ErrorRecord {
            offset_bytes: e["offset_bytes"].as_u64(),
            expected: e["expected"].as_u64().map(|v| v as u32),
            actual: e["actual"].as_u64().map(|v| v as u32),
            xor_mask: e["xor_mask"].as_u64().map(|v| v as u32),
            count: 1,
            iteration: e["round"].as_u64(),
            detail: Some(json!({"pass": e["pass"]})),
            ..Default::default()
        });
    }
    if t.error_count > 0 && t.status == TestStatus::Pass {
        t.status = TestStatus::Fail;
    }
    t.termination = Some(worker_termination(t.status, code, timed_out, ev).into());
    t
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::close::app_status as st;
    use serde_json::json;

    fn mt() -> TestResult {
        TestResult::new("vram", "vulkan", "vulkan_memtest")
    }

    fn code(bits: u8) -> Option<i32> {
        Some((st::SIGNATURE | bits) as i32)
    }

    fn parse(lines: &[String]) -> Vec<Value> {
        lines
            .iter()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    #[test]
    fn init_failure_retries_only_out_of_memory() {
        let abort = code(st::RUNTIME_ABORT);
        let oom = parse(&[crate::hooks::format_init_failure(
            0.0,
            "ERROR_OUT_OF_DEVICE_MEMORY while getting DeviceMemory in context allocate_memory",
        )]);
        assert!(matches!(
            init_outcome(abort, &oom, 0),
            InitOutcome::RetryLower(_)
        ));
        assert!(
            matches!(init_outcome(abort, &oom, MAX_ALLOC_RETRIES), InitOutcome::Stop(m) if m.contains("gave up"))
        );
        let lost = parse(&[crate::hooks::format_init_failure(
            0.0,
            "ERROR_DEVICE_LOST while getting () in context wait_for_fences",
        )]);
        assert!(
            matches!(init_outcome(abort, &lost, 0), InitOutcome::Stop(m) if m.contains("DEVICE_LOST"))
        );
        // no reason reported -> do not hammer the card
        assert!(matches!(init_outcome(abort, &[], 0), InitOutcome::Stop(_)));
        // errors latched during init (RUNTIME_ERRORS bit) are judged, never retried away
        let latched = code(st::RUNTIME_ABORT | st::RUNTIME_ERRORS);
        assert_eq!(init_outcome(latched, &oom, 0), InitOutcome::Proceed);
        // a started run is judged even if it aborted later
        let started = parse(&[crate::hooks::format_started(0.0, 1, 1, 2)]);
        assert_eq!(init_outcome(abort, &started, 0), InitOutcome::Proceed);
    }

    #[test]
    fn memtest_clean_run_passes() {
        let ev = parse(&[
            crate::hooks::format_started(0.0, 1000, 400, 2),
            crate::hooks::format_progress(1.0, 5, 10, 20, 1.0, 2.0),
            crate::hooks::format_finished(10.0, 9, "duration_reached"),
        ]);
        let t = judge_memtest(
            mt(),
            code(st::INITED_OK | st::QUIT_JOB_REQUESTED),
            false,
            &ev,
        );
        assert_eq!(t.status, TestStatus::Pass, "{:?}", t.message);
        assert_eq!(t.metrics["iterations"], 5);
    }

    #[test]
    fn memtest_errors_fail_with_ranges_and_ff_words() {
        let arrays = ([0u32; 32], [0u32; 32], [0u32; 32]);
        let r = crate::hooks::ErrorReport {
            iteration: 3,
            mode: "INITIAL_READ",
            window_offset: 4096,
            window_bytes: 1 << 20,
            idx_min: 1,
            idx_max: 2,
            single_bit_idx: &arrays.0,
            toggle_count: &arrays.1,
            ones_in_value: &arrays.2,
            actual_ff: 2, // upstream's own count would be 0 here (REVIEW R1)
            actual_min: u32::MAX,
            actual_max: 0,
        };
        let ev = parse(&[crate::hooks::format_error(1.0, &r)]);
        let t = judge_memtest(
            mt(),
            code(st::INITED_OK | st::RUNTIME_ERRORS | st::QUIT_JOB_REQUESTED),
            false,
            &ev,
        );
        assert_eq!(t.status, TestStatus::Fail);
        assert_eq!(t.error_count, 2);
        assert_eq!(t.errors[0].offset_range, Some([4100, 4107]));
        assert_eq!(t.errors[0].detail.as_ref().unwrap()["all_ones_words"], 2);
    }

    #[test]
    fn memtest_device_lost_after_init_is_incomplete() {
        let t = judge_memtest(mt(), code(st::INITED_OK | st::RUNTIME_ABORT), false, &[]);
        assert_eq!(t.status, TestStatus::Incomplete);
        assert_eq!(t.termination.as_deref(), Some("runtime_error"));
        assert!(t.message.unwrap().contains("DEVICE_LOST"));
    }

    #[test]
    fn memtest_crash_and_timeout_are_incomplete() {
        assert_eq!(
            judge_memtest(mt(), Some(-1073741819), false, &[]).status,
            TestStatus::Incomplete
        );
        assert_eq!(
            judge_memtest(mt(), None, true, &[]).status,
            TestStatus::Incomplete
        );
    }

    #[test]
    fn cuda_worker_results() {
        let ev: Vec<Value> = vec![
            json!({"ev": "cuda_coverage", "tested_bytes": 23u64 << 30, "total_bytes": 24u64 << 30,
                   "free_bytes_at_start": 23u64 << 30, "reserved_or_used_bytes": 1u64 << 30,
                   "granularity_bytes": 2 << 20, "chunks": 46}),
            json!({"ev": "cuda_pass", "pass": "own_address", "gbps": 800.0, "errors": 0}),
            json!({"ev": "cuda_pass", "pass": "random", "gbps": 700.0, "errors": 3}),
            json!({"ev": "cuda_error", "pass": "random", "offset_bytes": 1024, "expected": 1, "actual": 3, "xor_mask": 2, "round": 0}),
        ];
        let t = judge_cuda(
            TestResult::new("vram", "cuda", "cuda_vmm_memtest"),
            Some(1),
            false,
            &ev,
        );
        assert_eq!(t.status, TestStatus::Fail);
        assert_eq!(t.error_count, 3);
        assert_eq!(t.coverage.as_ref().unwrap().tested_bytes, 23u64 << 30);
        assert_eq!(t.metrics["passes"]["random"]["errors"], 3);
        assert_eq!(t.errors[0].offset_bytes, Some(1024));
    }

    #[test]
    fn cuda_ecc_uncorrectable_is_fail_not_incomplete() {
        let ev = vec![
            json!({"ev": "worker_error", "message": "x", "ecc_uncorrectable": true, "gpu_fault": true}),
        ];
        let t = judge_cuda(TestResult::new("vram", "cuda", "c"), Some(2), false, &ev);
        assert_eq!(t.status, TestStatus::Fail);
    }

    #[test]
    fn worker_exit_codes() {
        assert_eq!(worker_status(Some(0), false, &[]).0, TestStatus::Pass);
        assert_eq!(worker_status(Some(1), false, &[]).0, TestStatus::Fail);
        assert_eq!(worker_status(Some(2), false, &[]).0, TestStatus::Incomplete);
        assert_eq!(worker_status(Some(4), false, &[]).0, TestStatus::Error);
        let unsup =
            vec![json!({"ev": "worker_error", "message": "no driver", "unsupported": true})];
        assert_eq!(
            worker_status(Some(4), false, &unsup).0,
            TestStatus::Unsupported
        );
        assert_eq!(worker_status(Some(0), true, &[]).0, TestStatus::Incomplete);
        assert_eq!(worker_status(None, false, &[]).0, TestStatus::Incomplete);
    }
}
