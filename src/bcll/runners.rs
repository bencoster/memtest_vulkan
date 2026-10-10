//! Test runners: spawn one worker child per test, enforce its deadline, and
//! convert its events into a `TestResult` via `judge` (DESIGN.md sections 3, 7).

use super::cli::Options;
use super::events::{self, of_kind};
use super::inventory::GpuRecord;
use super::judge::{self, InitOutcome};
use super::nvml::NvmlCtx;
use super::report::{Coverage, ErrorRecord, TestResult, TestStatus};
use serde_json::{Value, json};
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Duration, Instant};

/// Extra wall time a memtest child gets for allocation + initial fill.
const MEMTEST_INIT_ALLOWANCE_S: f64 = 90.0;
const MIN_TEST_BYTES: i64 = 400 << 20;
const RETRY_STEP_BYTES: i64 = 400 << 20;

pub(super) struct Ctx<'a> {
    pub(super) opts: &'a Options,
    pub(super) gpu: GpuRecord,
    pub(super) exe: PathBuf,
    pub(super) work: PathBuf,
    pub(super) nvml: Option<NvmlCtx>,
}

enum WaitOutcome {
    Exited(Option<i32>),
    TimedOut,
    SpawnFailed(String),
}

fn spawn_wait(mut cmd: std::process::Command, timeout: Duration) -> WaitOutcome {
    let mut child = match cmd.stdin(Stdio::null()).spawn() {
        Ok(c) => c,
        Err(e) => return WaitOutcome::SpawnFailed(e.to_string()),
    };
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return WaitOutcome::Exited(status.code()),
            Ok(None) if start.elapsed() > timeout => {
                let _ = child.kill();
                let _ = child.wait();
                return WaitOutcome::TimedOut;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(200)),
            Err(e) => return WaitOutcome::SpawnFailed(e.to_string()),
        }
    }
}

fn events_path(ctx: &Ctx, name: &str) -> PathBuf {
    let p = ctx.work.join(format!("{name}-events.jsonl"));
    let _ = std::fs::remove_file(&p);
    p
}

fn coverage_from_nvml(ctx: &Ctx, tested: u64) -> Coverage {
    let usage = ctx.nvml.as_ref().and_then(|n| n.usage(&ctx.gpu.pci_bus_id));
    Coverage {
        tested_bytes: tested,
        total_bytes: usage
            .as_ref()
            .map(|u| u.total_bytes)
            .or(Some(ctx.gpu.device_local_bytes)),
        free_bytes_at_start: usage.as_ref().map(|u| u.free_bytes),
        reserved_or_used_bytes: usage
            .as_ref()
            .map(|u| u.total_bytes.saturating_sub(u.free_bytes)),
        driver_reserved_bytes: usage.as_ref().and_then(|u| u.reserved_bytes),
    }
}

pub(super) fn run_memtest(ctx: &Ctx, seconds: f64) -> TestResult {
    let mut t = TestResult::new("vram", "vulkan", "vulkan_memtest");
    t.started_at = Some(crate::output::NowTime.to_string());
    let start = Instant::now();
    let mut bytes = ctx.gpu.memtest_budget_bytes;
    if let Some(cap) = ctx.opts.memtest_bytes {
        bytes = bytes.min(cap as i64);
    }
    println!(
        "[vram/vulkan] {seconds:.0}s, up to {:.2} GiB (upstream memtest algorithm)",
        bytes as f64 / (1u64 << 30) as f64
    );
    let mut attempts: Vec<Value> = Vec::new();
    loop {
        if bytes < MIN_TEST_BYTES {
            t.status = TestStatus::Error;
            t.termination = Some("init_failed".into());
            t.message = Some(format!("not enough free VRAM to test ({bytes} bytes)"));
            break;
        }
        let ev_path = events_path(ctx, "memtest");
        let mut cmd = std::process::Command::new(&ctx.exe);
        // per-run working dir: upstream's memtest_vulkan.log rotation is not safe for
        // concurrent writers (Codex audit F8), so each run gets its own log file
        cmd.arg((-1 - ctx.gpu.vk_enum_index as isize).to_string())
            .arg(bytes.to_string())
            .current_dir(&ctx.work)
            .env(crate::hooks::ENV_EVENTS, &ev_path)
            .env(crate::hooks::ENV_STOP_AFTER_S, format!("{seconds}"));
        let timeout =
            Duration::from_secs_f64(seconds + MEMTEST_INIT_ALLOWANCE_S + ctx.opts.worker_grace_s);
        let (code, timed_out) = match spawn_wait(cmd, timeout) {
            WaitOutcome::Exited(c) => (c, false),
            WaitOutcome::TimedOut => (None, true),
            WaitOutcome::SpawnFailed(e) => {
                t.status = TestStatus::Error;
                t.message = Some(format!("cannot start memtest child: {e}"));
                break;
            }
        };
        let ev = events::read_events(&ev_path);
        let started = of_kind(&ev, "memtest_started").last().cloned();
        match judge::init_outcome(code, &ev, attempts.len()) {
            InitOutcome::Proceed => {}
            InitOutcome::RetryLower(msg) => {
                attempts.push(json!({"bytes": bytes, "error": msg}));
                bytes -= RETRY_STEP_BYTES;
                println!(
                    "[vram/vulkan] out of memory during init ({msg}); retrying with {:.2} GiB",
                    bytes as f64 / (1u64 << 30) as f64
                );
                continue;
            }
            InitOutcome::Stop(msg) => {
                attempts.push(json!({"bytes": bytes, "error": msg}));
                t.status = TestStatus::Incomplete;
                t.termination = Some(
                    if msg.contains("DEVICE_LOST") {
                        "device_lost"
                    } else {
                        "init_failed"
                    }
                    .into(),
                );
                t.message = Some(format!("memtest init failed, not retried: {msg}"));
                t.set_metric("allocation_attempts", &attempts);
                break;
            }
        }
        t.set_metric("allocation_attempts", &attempts);
        t = judge::judge_memtest(t, code, timed_out, &ev);
        let tested = started
            .as_ref()
            .and_then(|s| s["tested_bytes"].as_u64())
            .unwrap_or(0);
        t.coverage = Some(coverage_from_nvml(ctx, tested));
        if let Some(s) = started {
            t.set_metric("allocation_bytes", s["allocation_bytes"].clone());
            t.set_metric("window_bytes", s["window_bytes"].clone());
            t.set_metric("window_count", s["window_count"].clone());
        }
        break;
    }
    keep_memtest_log(ctx);
    t.duration_s = Some(start.elapsed().as_secs_f64());
    t
}

/// Keeps upstream's console log of this run next to the JSON output, if one was requested.
fn keep_memtest_log(ctx: &Ctx) {
    let log = ctx.work.join("memtest_vulkan.log");
    if let Some(out) = &ctx.opts.report_json
        && log.is_file()
    {
        let dest = out.with_extension("memtest.log");
        match std::fs::copy(&log, &dest) {
            Ok(_) => eprintln!("memtest log kept: {}", dest.display()),
            Err(e) => eprintln!("memtest_vulkan: cannot keep memtest log: {e}"),
        }
    }
}

pub(super) fn run_vk_worker(ctx: &Ctx, kind: &str, seconds: f64) -> TestResult {
    let (test_kind, name) = match kind {
        "compute" => ("compute", "vulkan_compute"),
        _ => ("pcie_transfer", "vulkan_transfer"),
    };
    println!("[{test_kind}/vulkan] {seconds:.0}s");
    let mut t = TestResult::new(test_kind, "vulkan", name);
    t.started_at = Some(crate::output::NowTime.to_string());
    let start = Instant::now();
    let ev_path = events_path(ctx, kind);
    let mut cmd = std::process::Command::new(&ctx.exe);
    cmd.args(["--bcll-worker", kind, "--vk-index"])
        .arg(ctx.gpu.vk_enum_index.to_string())
        .arg("--seconds")
        .arg(format!("{seconds}"))
        .arg("--events")
        .arg(&ev_path);
    let outcome = spawn_wait(
        cmd,
        Duration::from_secs_f64(seconds + ctx.opts.worker_grace_s),
    );
    let ev = events::read_events(&ev_path);
    let (code, timed_out) = match outcome {
        WaitOutcome::Exited(c) => (c, false),
        WaitOutcome::TimedOut => (None, true),
        WaitOutcome::SpawnFailed(e) => {
            t.status = TestStatus::Error;
            t.message = Some(format!("cannot start worker: {e}"));
            return t;
        }
    };
    (t.status, t.message) = judge::worker_status(code, timed_out, &ev);
    if kind == "compute" {
        let batches: Vec<&Value> = of_kind(&ev, "compute_batch").collect();
        t.set_metric("batches", batches.len());
        let sum = |k: &str| batches.iter().filter_map(|b| b[k].as_u64()).sum::<u64>();
        t.set_metric("int_errors", sum("int_errors"));
        t.set_metric("fp_consensus_errors", sum("fp_consensus_errors"));
        t.set_metric("fp_reference_errors", sum("fp_reference_errors"));
        t.error_count = sum("int_errors") + sum("fp_consensus_errors") + sum("fp_reference_errors");
        if let Some(s) = of_kind(&ev, "worker_started").last() {
            for k in ["invocations", "problems", "iters", "base_seed"] {
                t.set_metric(k, s[k].clone());
            }
        }
        for e in of_kind(&ev, "compute_error") {
            t.push_error(ErrorRecord {
                expected: e["expected"].as_u64().map(|v| v as u32),
                actual: e["actual"].as_u64().map(|v| v as u32),
                xor_mask: e["xor_mask"].as_u64().map(|v| v as u32),
                count: 1,
                iteration: e["batch"].as_u64(),
                detail: Some(json!({
                    "invocation": e["invocation"], "workgroup": e["workgroup"], "problem": e["problem"],
                    "lane": e["lane"], "check": e["check"], "seed": e["seed"],
                })),
                ..Default::default()
            });
        }
    } else {
        let rounds: Vec<&Value> = of_kind(&ev, "transfer_round").collect();
        let stat = |k: &str| {
            let v: Vec<f64> = rounds.iter().filter_map(|r| r[k].as_f64()).collect();
            (!v.is_empty()).then(|| {
                json!({
                    "avg": v.iter().sum::<f64>() / v.len() as f64,
                    "min": v.iter().cloned().fold(f64::INFINITY, f64::min),
                    "max": v.iter().cloned().fold(0.0, f64::max),
                })
            })
        };
        t.set_metric("rounds", rounds.len());
        t.set_metric(
            "bytes_per_round",
            rounds.first().map(|r| r["bytes"].clone()),
        );
        t.set_metric("h2d_gbps", stat("h2d_gbps"));
        t.set_metric("d2h_gbps", stat("d2h_gbps"));
        t.error_count = rounds
            .iter()
            .filter_map(|r| r["mismatched_words"].as_u64())
            .sum();
        for e in of_kind(&ev, "transfer_error") {
            t.push_error(ErrorRecord {
                offset_bytes: e["offset_bytes"].as_u64(),
                expected: e["expected"].as_u64().map(|v| v as u32),
                actual: e["actual"].as_u64().map(|v| v as u32),
                xor_mask: e["xor_mask"].as_u64().map(|v| v as u32),
                count: 1,
                iteration: e["round"].as_u64(),
                ..Default::default()
            });
        }
    }
    if t.error_count > 0 && t.status == TestStatus::Pass {
        t.status = TestStatus::Fail; // never trust an exit code over recorded errors
    }
    t.termination = Some(judge::worker_termination(t.status, code, timed_out, &ev).into());
    t.duration_s = Some(start.elapsed().as_secs_f64());
    t
}

pub(super) fn run_cuda(ctx: &Ctx, seconds: f64) -> TestResult {
    println!(
        "[vram/cuda] {seconds:.0}s, keep-free {} MiB",
        ctx.opts.cuda_keep_free_mb
    );
    let mut t = TestResult::new("vram", "cuda", "cuda_vmm_memtest");
    t.started_at = Some(crate::output::NowTime.to_string());
    let start = Instant::now();
    if !super::cuda::AVAILABLE {
        t.status = TestStatus::Unsupported;
        t.message = Some("built without the 'cuda' feature".into());
        return t;
    }
    let Some(uuid) = ctx.gpu.uuid.clone() else {
        t.status = TestStatus::Error;
        t.message = Some("GPU UUID unknown - cannot pin the CUDA worker to the card".into());
        return t;
    };
    let ev_path = events_path(ctx, "cuda");
    let mut cmd = std::process::Command::new(&ctx.exe);
    cmd.args([
        "--bcll-worker",
        "cuda-memtest",
        "--uuid",
        &uuid,
        "--pci",
        &ctx.gpu.pci_bus_id,
    ])
    .arg("--seconds")
    .arg(format!("{seconds}"))
    .arg("--events")
    .arg(&ev_path)
    .arg("--cuda-keep-free-mb")
    .arg(ctx.opts.cuda_keep_free_mb.to_string())
    .env("CUDA_VISIBLE_DEVICES", &uuid)
    .env("CUDA_DEVICE_ORDER", "PCI_BUS_ID");
    let outcome = spawn_wait(
        cmd,
        Duration::from_secs_f64(seconds + ctx.opts.worker_grace_s),
    );
    let ev = events::read_events(&ev_path);
    let (code, timed_out) = match outcome {
        WaitOutcome::Exited(c) => (c, false),
        WaitOutcome::TimedOut => (None, true),
        WaitOutcome::SpawnFailed(e) => {
            t.status = TestStatus::Error;
            t.message = Some(format!("cannot start worker: {e}"));
            return t;
        }
    };
    t = judge::judge_cuda(t, code, timed_out, &ev);
    if let Some(c) = t.coverage.as_mut() {
        c.driver_reserved_bytes = coverage_from_nvml(ctx, 0).driver_reserved_bytes;
    }
    t.duration_s = Some(start.elapsed().as_secs_f64());
    t
}
