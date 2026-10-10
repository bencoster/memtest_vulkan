//! BCLL fork: structured-event hooks called from the upstream memtest loop.
//!
//! Every function here is a no-op unless the orchestrator set one of the
//! environment variables below on the child process, so the upstream program
//! behaves exactly as before. Kept std-only (hand-formatted JSON with numbers
//! and fixed strings) so `--no-default-features` builds need no serde.
//!
//! Events are JSON lines, flushed per event so a crashed child still leaves a
//! parseable file. Schema: docs/bcll/SCHEMA.md ("worker events").

use std::{
    fmt::Write as _,
    fs::File,
    io::Write as _,
    sync::{Mutex, OnceLock},
    time::{Duration, Instant},
};

/// Path of the JSON-lines events file to append to.
pub const ENV_EVENTS: &str = "MEMTEST_VULKAN_EVENTS";
/// Stop the memtest loop this many seconds after the first test iteration starts.
pub const ENV_STOP_AFTER_S: &str = "MEMTEST_VULKAN_STOP_AFTER_S";
/// Identity the orchestrator approved; the child refuses any other device (BLOCK 89de0c26 #2).
pub const ENV_EXPECT_PCI: &str = "MEMTEST_VULKAN_EXPECT_PCI";
pub const ENV_EXPECT_UUID: &str = "MEMTEST_VULKAN_EXPECT_UUID";
/// Hard wall-clock limit for the whole child, enforced by a watchdog thread even if the
/// test loop is stuck in a driver call or the orchestrator died (BLOCK 89de0c26 #1).
pub const ENV_HARD_DEADLINE_S: &str = "MEMTEST_VULKAN_HARD_DEADLINE_S";

struct State {
    file: Option<Mutex<File>>,
    stop_after: Option<Duration>,
    start: Mutex<Option<Instant>>,
}

fn state() -> &'static State {
    static STATE: OnceLock<State> = OnceLock::new();
    STATE.get_or_init(|| {
        arm_hard_deadline();
        new_state()
    })
}

fn arm_hard_deadline() {
    let Some(limit) = std::env::var(ENV_HARD_DEADLINE_S)
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .filter(|s| *s > 0.0)
    else {
        return;
    };
    let _ = std::thread::Builder::new()
        .name("memtest-hard-deadline".into())
        .spawn(move || {
            std::thread::sleep(Duration::from_secs_f64(limit));
            emit(&format!(
                "{{\"ev\":\"memtest_hard_deadline\",\"t_s\":{:.3},\"limit_s\":{limit}}}",
                elapsed_s()
            ));
            crate::close::immediate_exit(true);
        });
}

fn new_state() -> State {
    State {
        file: std::env::var_os(ENV_EVENTS)
            .and_then(|path| {
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .ok()
            })
            .map(Mutex::new),
        stop_after: std::env::var(ENV_STOP_AFTER_S)
            .ok()
            .and_then(|s| s.trim().parse::<f64>().ok())
            .filter(|s| *s > 0.0)
            .map(Duration::from_secs_f64),
        start: Mutex::new(None),
    }
}

/// `dddd:bb:dd.f`, the PCI id format used by the orchestrator.
pub fn pci_string(domain: u32, bus: u32, device: u32, function: u32) -> String {
    format!("{domain:04x}:{bus:02x}:{device:02x}.{function:x}")
}

/// Formats a 16-byte UUID the way NVML / nvidia-smi / CUDA_VISIBLE_DEVICES do.
pub fn uuid_string(b: &[u8; 16]) -> String {
    let h: Vec<String> = b.iter().map(|x| format!("{x:02x}")).collect();
    format!(
        "GPU-{}-{}-{}-{}-{}",
        h[0..4].concat(),
        h[4..6].concat(),
        h[6..8].concat(),
        h[8..10].concat(),
        h[10..16].concat()
    )
}

/// Pure identity comparison (unit-tested): Ok when nothing is expected or everything matches.
pub fn identity_verdict(
    expected_pci: Option<&str>,
    expected_uuid: Option<&str>,
    actual_pci: &str,
    actual_uuid: Option<&str>,
) -> Result<(), String> {
    if let Some(p) = expected_pci {
        if actual_pci == "0000:00:00.0" {
            return Err(format!("device reports no PCI identity; expected {p}"));
        }
        if !actual_pci.eq_ignore_ascii_case(p) {
            return Err(format!("expected PCI {p}, selected device is {actual_pci}"));
        }
    }
    if let Some(u) = expected_uuid {
        match actual_uuid {
            Some(a) if a.eq_ignore_ascii_case(u) => {}
            Some(a) => return Err(format!("expected UUID {u}, selected device has {a}")),
            None => return Err(format!("expected UUID {u}, selected device reports none")),
        }
    }
    Ok(())
}

/// Called by upstream's selection code before any device is created. Returns false (after
/// emitting `memtest_identity_mismatch`) when the selected device is not the approved one.
pub fn identity_ok(pci: (u32, u32, u32, u32), uuid: Option<[u8; 16]>) -> bool {
    let _ = state(); // arms the hard-deadline watchdog before any device work
    let expected_pci = std::env::var(ENV_EXPECT_PCI).ok();
    let expected_uuid = std::env::var(ENV_EXPECT_UUID).ok();
    if expected_pci.is_none() && expected_uuid.is_none() {
        return true;
    }
    let actual_pci = pci_string(pci.0, pci.1, pci.2, pci.3);
    let actual_uuid = uuid
        .filter(|u| u.iter().any(|b| *b != 0))
        .map(|u| uuid_string(&u));
    match identity_verdict(
        expected_pci.as_deref(),
        expected_uuid.as_deref(),
        &actual_pci,
        actual_uuid.as_deref(),
    ) {
        Ok(()) => true,
        Err(why) => {
            emit(&format!(
                "{{\"ev\":\"memtest_identity_mismatch\",\"t_s\":{:.3},\"message\":\"{}\"}}",
                elapsed_s(),
                json_escape(&why)
            ));
            false
        }
    }
}

fn elapsed_s() -> f64 {
    match *state().start.lock().unwrap_or_else(|e| e.into_inner()) {
        Some(start) => start.elapsed().as_secs_f64(),
        None => 0.0,
    }
}

fn emit(line: &str) {
    if let Some(file) = &state().file {
        let mut file = file.lock().unwrap_or_else(|e| e.into_inner());
        let _ = writeln!(file, "{line}");
        let _ = file.flush();
    }
}

/// Error statistics of one verified window, as gathered by the read shader.
pub struct ErrorReport<'a> {
    pub iteration: u32,
    /// INITIAL_FILL (startup verification), INITIAL_READ or NEXT_RE_READ
    pub mode: &'static str,
    pub window_offset: i64,
    pub window_bytes: i64,
    pub idx_min: u32,
    pub idx_max: u32,
    pub single_bit_idx: &'a [u32; 32],
    pub toggle_count: &'a [u32; 32],
    pub ones_in_value: &'a [u32; 32],
    pub actual_ff: u32,
    pub actual_min: u32,
    pub actual_max: u32,
}

impl ErrorReport<'_> {
    /// Bad words in the window. Unlike upstream's console line this includes
    /// words that read back as 0xFFFFFFFF (counted only in `actual_ff`).
    pub fn total_errors(&self) -> u64 {
        self.ones_in_value.iter().map(|v| *v as u64).sum::<u64>() + self.actual_ff as u64
    }
}

fn push_array(out: &mut String, name: &str, values: &[u32]) {
    let _ = write!(out, ",\"{name}\":[");
    for (i, v) in values.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let _ = write!(out, "{v}");
    }
    out.push(']');
}

pub fn format_started(
    t_s: f64,
    allocation_bytes: i64,
    window_bytes: i64,
    window_count: i64,
) -> String {
    format!(
        "{{\"ev\":\"memtest_started\",\"t_s\":{t_s:.3},\"allocation_bytes\":{allocation_bytes},\"window_bytes\":{window_bytes},\"window_count\":{window_count},\"tested_bytes\":{}}}",
        window_bytes * window_count
    )
}

pub fn format_error(t_s: f64, r: &ErrorReport) -> String {
    const ELEMENT_SIZE: i64 = 4;
    let mut out = format!(
        "{{\"ev\":\"memtest_error\",\"t_s\":{t_s:.3},\"iteration\":{},\"mode\":\"{}\",\"window_offset\":{},\"window_bytes\":{},\"addr_first\":{},\"addr_last\":{},\"total_errors\":{},\"actual_ff\":{},\"actual_min\":{},\"actual_max\":{}",
        r.iteration,
        r.mode,
        r.window_offset,
        r.window_bytes,
        r.window_offset + r.idx_min as i64 * ELEMENT_SIZE,
        r.window_offset + (r.idx_max as i64 + 1) * ELEMENT_SIZE - 1,
        r.total_errors(),
        r.actual_ff,
        r.actual_min,
        r.actual_max,
    );
    push_array(&mut out, "single_bit_idx", r.single_bit_idx);
    push_array(&mut out, "toggle_count", r.toggle_count);
    push_array(&mut out, "ones_in_value", r.ones_in_value);
    out.push('}');
    out
}

pub fn format_progress(
    t_s: f64,
    iteration: u64,
    written_bytes: i64,
    read_bytes: i64,
    write_gbps: f32,
    check_gbps: f32,
) -> String {
    let finite = |v: f32| if v.is_finite() { v } else { 0.0 };
    format!(
        "{{\"ev\":\"memtest_progress\",\"t_s\":{t_s:.3},\"iteration\":{iteration},\"written_bytes\":{written_bytes},\"read_bytes\":{read_bytes},\"write_gbps\":{:.3},\"check_gbps\":{:.3}}}",
        finite(write_gbps),
        finite(check_gbps)
    )
}

pub fn format_finished(t_s: f64, iteration: u64, reason: &str) -> String {
    format!(
        "{{\"ev\":\"memtest_finished\",\"t_s\":{t_s:.3},\"iteration\":{iteration},\"reason\":\"{reason}\"}}"
    )
}

/// Minimal JSON string escaping for free-form messages.
pub fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out
}

pub fn format_init_failure(t_s: f64, message: &str) -> String {
    format!(
        "{{\"ev\":\"memtest_init_failure\",\"t_s\":{t_s:.3},\"message\":\"{}\"}}",
        json_escape(message)
    )
}

/// Init failed and the child is about to exit for a lower-memory retry; `message`
/// carries the Vulkan error so the orchestrator can tell OOM from DEVICE_LOST.
pub fn init_failure(message: &str) {
    emit(&format_init_failure(elapsed_s(), message));
}

/// Called once the allocation is verified usable; starts the duration clock.
pub fn memtest_started(allocation_bytes: i64, window_bytes: i64, window_count: i64) {
    *state().start.lock().unwrap_or_else(|e| e.into_inner()) = Some(Instant::now());
    emit(&format_started(
        0.0,
        allocation_bytes,
        window_bytes,
        window_count,
    ));
}

pub fn memtest_error(report: &ErrorReport) {
    emit(&format_error(elapsed_s(), report));
}

pub fn memtest_progress(
    iteration: u64,
    written_bytes: i64,
    read_bytes: i64,
    write_gbps: f32,
    check_gbps: f32,
) {
    emit(&format_progress(
        elapsed_s(),
        iteration,
        written_bytes,
        read_bytes,
        write_gbps,
        check_gbps,
    ));
}

/// `reason` must be a fixed identifier (it is not JSON-escaped).
pub fn memtest_finished(iteration: u64, reason: &str) {
    emit(&format_finished(elapsed_s(), iteration, reason));
}

/// True once the requested test duration has elapsed (never without the env var).
pub fn deadline_reached() -> bool {
    let state = state();
    match (
        state.stop_after,
        *state.start.lock().unwrap_or_else(|e| e.into_inner()),
    ) {
        (Some(limit), Some(start)) => start.elapsed() >= limit,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample<'a>(arrays: &'a ([u32; 32], [u32; 32], [u32; 32])) -> ErrorReport<'a> {
        ErrorReport {
            iteration: 7,
            mode: "INITIAL_READ",
            window_offset: 0x1_0000_0000,
            window_bytes: 0x8000_0000,
            idx_min: 0x10,
            idx_max: 0x20,
            single_bit_idx: &arrays.0,
            toggle_count: &arrays.1,
            ones_in_value: &arrays.2,
            actual_ff: 3,
            actual_min: 1,
            actual_max: 0xFFFF_FFFE,
        }
    }

    #[test]
    fn total_errors_counts_all_ones_words() {
        let mut arrays = ([0u32; 32], [0u32; 32], [0u32; 32]);
        arrays.2[5] = 2;
        arrays.2[31] = 1;
        let r = sample(&arrays);
        // upstream would print 3 (sum of ones_in_value); the 3 FF words make it 6
        assert_eq!(r.total_errors(), 6);
    }

    #[test]
    fn error_line_has_absolute_byte_range() {
        let mut arrays = ([0u32; 32], [0u32; 32], [0u32; 32]);
        arrays.0[22] = 1;
        let line = format_error(1.5, &sample(&arrays));
        assert!(line.starts_with("{\"ev\":\"memtest_error\""));
        assert!(line.contains("\"addr_first\":4294967360")); // 0x1_0000_0000 + 0x10*4
        assert!(line.contains("\"addr_last\":4294967427")); // + (0x20+1)*4 - 1
        assert!(line.contains("\"mode\":\"INITIAL_READ\""));
        assert!(line.ends_with("]}"));
        assert_eq!(line.matches('[').count(), 3);
    }

    #[test]
    fn identity_verdict_cases() {
        let ok = identity_verdict(
            Some("0000:c1:00.0"),
            Some("GPU-a"),
            "0000:c1:00.0",
            Some("gpu-A"),
        );
        assert!(ok.is_ok());
        assert!(identity_verdict(None, None, "0000:00:00.0", None).is_ok());
        assert!(
            identity_verdict(Some("0000:c1:00.0"), None, "0000:81:00.0", None)
                .unwrap_err()
                .contains("selected device is 0000:81:00.0")
        );
        assert!(
            identity_verdict(Some("0000:00:00.0"), None, "0000:00:00.0", None)
                .unwrap_err()
                .contains("no PCI identity")
        );
        assert!(identity_verdict(None, Some("GPU-a"), "0000:c1:00.0", Some("GPU-b")).is_err());
        assert!(identity_verdict(None, Some("GPU-a"), "0000:c1:00.0", None).is_err());
        assert_eq!(pci_string(0, 0xc1, 0, 0), "0000:c1:00.0");
    }

    #[test]
    fn init_failure_message_is_escaped() {
        let line = format_init_failure(0.5, "ERROR_DEVICE_LOST \"x\"\\y\nz\u{1}");
        assert!(
            line.contains("ERROR_DEVICE_LOST \\\"x\\\"\\\\y\\nz\\u0001"),
            "{line}"
        );
    }

    #[test]
    fn progress_never_emits_non_finite_numbers() {
        let line = format_progress(0.0, 1, 2, 3, f32::NAN, f32::INFINITY);
        assert!(line.contains("\"write_gbps\":0.000"));
        assert!(line.contains("\"check_gbps\":0.000"));
    }

    #[test]
    fn started_line_reports_tested_bytes_not_allocation() {
        let line = format_started(0.0, 1000, 300, 3);
        assert!(line.contains("\"tested_bytes\":900"));
        assert!(line.contains("\"allocation_bytes\":1000"));
    }
}
