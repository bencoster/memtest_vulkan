//! Orchestrator: inventory -> selection -> snapshot -> safety gate -> telemetry + worker
//! children -> snapshot -> verdict -> JSON output. The supervisor never calls a GPU driver
//! itself: inventory, snapshots and telemetry run in bounded worker processes (query.rs),
//! so a wedged driver costs a killed worker, not the run (DESIGN.md section 3).

use super::cli::{self, Command, Options, WorkerKind};
use super::codec;
use super::inventory::{GpuRecord, gib, hostname, list, select_gpu, tool_info};
use super::query;
use super::report::{self, DeviceInfo, Report, RunInfo};
use super::runners::{Ctx, run_cuda, run_memtest, run_vk_worker};
use super::telemetry::{self, HealthSnapshot, Usage};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub fn main(args: Vec<String>) -> i32 {
    let opts = match cli::parse(&args) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("memtest_vulkan: {e}");
            return report::EXIT_REFUSED;
        }
    };
    match &opts.command {
        Command::Help => {
            print!("{}", cli::HELP);
            0
        }
        Command::Version => {
            let t = tool_info();
            println!(
                "memtest_vulkan {} (features: {})",
                t.version,
                t.features.join(",")
            );
            0
        }
        Command::Worker(w) => match w.kind {
            WorkerKind::Compute | WorkerKind::Transfer => super::vk_diag::run_worker(w),
            WorkerKind::CudaMemtest => cuda_worker(w),
            WorkerKind::Query => query::run_query_worker(w),
            WorkerKind::Telemetry => query::run_telemetry_worker(w),
        },
        Command::List => list(&opts),
        Command::Info | Command::Run => run(&opts, args),
    }
}

#[cfg(feature = "cuda")]
fn cuda_worker(w: &cli::WorkerArgs) -> i32 {
    super::cuda::worker::run_worker(w)
}

#[cfg(not(feature = "cuda"))]
fn cuda_worker(w: &cli::WorkerArgs) -> i32 {
    let mut ev = super::events::EventWriter::create(&w.events);
    ev.emit(
        "worker_error",
        serde_json::json!({"message": "built without the 'cuda' feature", "unsupported": true}),
    );
    report::EXIT_INTERNAL
}

enum Abort {
    Refused(String),
    Error(String),
}

/// Ends a run that never reached testing: the reason goes into the output (if requested)
/// so a consumer never mistakes silence for a result (BLOCK 89de0c26 #8).
fn abort(rep: &mut Report, opts: &Options, started: Instant, why: Abort) -> i32 {
    let code = match why {
        Abort::Refused(m) => {
            eprintln!("memtest_vulkan: REFUSED: {m}");
            rep.refusal = Some(m);
            report::EXIT_REFUSED
        }
        Abort::Error(m) => {
            eprintln!("memtest_vulkan: {m}");
            rep.error = Some(m);
            report::EXIT_INTERNAL
        }
    };
    match finish_report(rep, opts, started) {
        Ok(()) => code,
        Err(_) => report::EXIT_INTERNAL,
    }
}

/// Unique per run, even under a caller-supplied --work-dir (no shared event/log names).
fn run_dir(opts: &Options) -> (PathBuf, bool) {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let name = format!("bcll-run-{}-{nanos}", std::process::id());
    match &opts.work_dir {
        Some(d) => (d.join(name), false),
        None => (std::env::temp_dir().join(name), true),
    }
}

fn run(opts: &Options, raw_args: Vec<String>) -> i32 {
    let started = Instant::now();
    // `--info --json` keeps stdout pure JSON
    let quiet = opts.json && opts.command == Command::Info;
    let mut rep = Report::new(
        tool_info(),
        RunInfo {
            started_at: crate::output::NowTime.to_string(),
            os: std::env::consts::OS.into(),
            hostname: hostname(),
            command_line: std::iter::once("memtest_vulkan".to_string())
                .chain(raw_args)
                .collect(),
            ..Default::default()
        },
    );
    // an old output file must never be mistaken for this run's (BLOCK 89de0c26 #8)
    if let Some(p) = &opts.report_json
        && p.exists()
        && let Err(e) = std::fs::remove_file(p)
    {
        eprintln!(
            "memtest_vulkan: cannot remove previous output {}: {e}; refusing to run",
            p.display()
        );
        return report::EXIT_INTERNAL;
    }
    let exe = match std::env::current_exe() {
        Ok(p) => p,
        Err(e) => {
            return abort(
                &mut rep,
                opts,
                started,
                Abort::Error(format!("cannot locate own executable: {e}")),
            );
        }
    };
    let inv = match query::inventory(&exe) {
        Ok(v) => v,
        Err(e) => return abort(&mut rep, opts, started, Abort::Error(e)),
    };
    if let Some(e) = &inv.nvml_error
        && !quiet
    {
        println!("note: {e}");
    }
    let spec = opts.gpu.as_deref().unwrap_or_default();
    let gpu = match select_gpu(&inv.gpus, spec) {
        Ok(g) => g,
        Err(e) => {
            return abort(
                &mut rep,
                opts,
                started,
                Abort::Refused(format!("{e} (see --list)")),
            );
        }
    };
    let before = query::snapshot(&exe, &gpu, opts.no_nvapi);
    if !quiet {
        for n in &before.notes {
            println!("note: {n}");
        }
    }
    rep.device = Some(device_info(opts, &gpu, &before));
    if !quiet {
        println!(
            "Selected GPU {} | nvidia-smi index {} | {} | {} | vk label {}",
            gpu.name,
            gpu.nvml_index
                .map(|i| i.to_string())
                .unwrap_or_else(|| "?".into()),
            gpu.pci_bus_id,
            gpu.uuid().as_deref().unwrap_or("uuid unknown"),
            gpu.vk_label
        );
    }
    let health_before = before.health.clone();
    rep.health_before = health_before
        .as_ref()
        .and_then(|h| serde_json::to_value(h).ok());

    if opts.command == Command::Info {
        rep.verdict = None;
        if !quiet {
            print_health(health_before.as_ref());
        }
        let ok = finish_report(&mut rep, opts, started).is_ok();
        if quiet && let Ok(s) = serde_json::to_string_pretty(&rep) {
            println!("{s}");
        }
        return if ok { 0 } else { report::EXIT_INTERNAL };
    }

    // ---- safety gate (DESIGN.md section 4): unknown evidence refuses ----
    if let Err(why) = safety_gate(opts, &gpu, before.display_attached, before.usage.as_ref()) {
        return abort(&mut rep, opts, started, Abort::Refused(why));
    }

    let (work, cleanup_work) = run_dir(opts);
    if let Err(e) = std::fs::create_dir_all(&work) {
        return abort(
            &mut rep,
            opts,
            started,
            Abort::Error(format!("cannot create work dir {}: {e}", work.display())),
        );
    }
    let planned_s = planned_seconds(opts);
    let tele = if opts.no_telemetry {
        None
    } else {
        match query::start_telemetry(&exe, &gpu, opts, &work, planned_s) {
            Ok(t) => Some(t),
            Err(e) => {
                rep.warnings
                    .push(format!("telemetry worker not started: {e}"));
                None
            }
        }
    };
    let ctx = Ctx {
        opts,
        gpu,
        exe,
        work: work.clone(),
        usage_before: before.usage.clone(),
    };

    if let Some(s) = opts.memtest_s {
        rep.tests.push(run_memtest(&ctx, s));
    }
    if let Some(s) = opts.compute_s {
        rep.tests.push(run_vk_worker(&ctx, "compute", s));
    }
    if let Some(s) = opts.transfer_s {
        rep.tests.push(run_vk_worker(&ctx, "transfer", s));
    }
    if let Some(s) = opts.cuda_memtest_s {
        rep.tests.push(run_cuda(&ctx, s));
    }
    if opts.codec {
        rep.tests.extend(codec::run(
            opts.ffmpeg.as_deref(),
            ctx.gpu.uuid().as_deref(),
            opts.codec_frames,
            &ctx.work,
        ));
    }

    let mut extra_warnings = Vec::new();
    let (samples, tele_meta) = match tele {
        Some(t) => {
            let (samples, meta, warn) = query::stop_telemetry(t);
            extra_warnings.extend(warn);
            (samples, Some(meta))
        }
        None => (Vec::new(), None),
    };
    let after = query::snapshot(&ctx.exe, &ctx.gpu, true);
    extra_warnings.extend(after.notes.iter().map(|n| format!("after-run {n}")));
    let health_after = after.health.clone();
    rep.health_after = health_after
        .as_ref()
        .and_then(|h| serde_json::to_value(h).ok());
    if let (Some(b), Some(a)) = (&health_before, &health_after) {
        rep.health_delta = serde_json::to_value(telemetry::health_delta(b, a)).ok();
    }
    let summary = (!samples.is_empty()).then(|| telemetry::summarize(&samples));
    if let Some(mut meta) = tele_meta {
        if let Value::Object(m) = &mut meta {
            m.insert(
                "summary".into(),
                serde_json::to_value(&summary).unwrap_or(Value::Null),
            );
            m.insert(
                "samples".into(),
                serde_json::to_value(&samples).unwrap_or(Value::Null),
            );
        }
        rep.telemetry = Some(meta);
    }
    let findings = telemetry::findings(
        health_before.as_ref(),
        health_after.as_ref(),
        summary.as_ref(),
        before.max_link,
    );
    let (verdict, reasons, warnings) = report::decide(&rep.tests, &findings);
    rep.verdict = Some(verdict);
    rep.reasons = reasons;
    rep.warnings.extend(warnings);
    rep.warnings.extend(extra_warnings);
    if opts.no_telemetry {
        rep.warnings
            .push("telemetry disabled (--no-telemetry): no thermal/PCIe/ECC correlation".into());
    } else if samples.is_empty() {
        rep.warnings
            .push("telemetry unavailable: PCIe/ECC/thermal evidence missing".into());
    }
    for t in &rep.tests {
        if t.metrics.get("worker_dead_confirmed") == Some(&Value::Bool(false)) {
            rep.warnings.push(format!(
                "worker of {} did not confirm its death after kill: the card may still be busy",
                t.name
            ));
        }
    }
    print_summary(&rep);
    let persisted = finish_report(&mut rep, opts, started);
    if cleanup_work {
        let _ = std::fs::remove_dir_all(&work);
    }
    match persisted {
        Ok(()) => report::exit_code(verdict),
        Err(_) => report::EXIT_INTERNAL, // verdict printed above; the requested file is missing
    }
}

/// Upper bound of the test schedule, for the telemetry worker's own deadline.
fn planned_seconds(opts: &Options) -> f64 {
    let tests = [
        opts.memtest_s,
        opts.compute_s,
        opts.transfer_s,
        opts.cuda_memtest_s,
    ];
    let n = tests.iter().flatten().count() as f64 + if opts.codec { 1.0 } else { 0.0 };
    let codec = if opts.codec {
        3.0 * 2.0 * (60.0 + opts.codec_frames as f64 / 10.0) + 120.0
    } else {
        0.0
    };
    tests.iter().flatten().sum::<f64>() + codec + 90.0 + n * opts.worker_grace_s
}

fn device_info(opts: &Options, gpu: &GpuRecord, snap: &query::Snapshot) -> DeviceInfo {
    DeviceInfo {
        selector: opts.gpu.clone(),
        index_pci_order: gpu.index_pci_order,
        vk_label: gpu.vk_label,
        nvml_index: gpu.nvml_index,
        pci_bus_id: gpu.pci_bus_id.clone(),
        uuid: gpu.uuid(),
        name: gpu.name.clone(),
        vendor_id: gpu.vendor_id,
        device_id: gpu.device_id,
        vulkan_driver_version: gpu.driver_version,
        vulkan_api_version: gpu.api_version.clone(),
        device_local_bytes: gpu.device_local_bytes,
        display_attached: snap.display_attached,
        details: snap.details.clone(),
    }
}

/// Refuses unless the evidence POSITIVELY shows an idle, non-display card; any unknown
/// (failed query, missing NVML) needs the matching explicit override (BLOCK 89de0c26 #3).
pub fn safety_gate(
    opts: &Options,
    gpu: &GpuRecord,
    display: Option<bool>,
    usage: Option<&Usage>,
) -> Result<(), String> {
    let who = format!("GPU {} ({})", gpu.name, gpu.pci_bus_id);
    if !opts.allow_display {
        match display {
            Some(true) => {
                return Err(format!(
                    "{who} drives a display; testing it can black-screen the desktop. Pass --allow-display to override."
                ));
            }
            None => {
                return Err(format!(
                    "display state of {who} unknown (NVML unavailable or a display query failed); pass --allow-display after checking it drives no monitor."
                ));
            }
            Some(false) => {}
        }
    }
    if !opts.allow_busy {
        let Some(u) = usage else {
            return Err(format!(
                "VRAM usage of {who} unknown (NVML query failed); pass --allow-busy only after checking the card is idle."
            ));
        };
        let used_mb = u.used_bytes >> 20;
        if used_mb > opts.max_used_mb {
            return Err(format!(
                "{who} has {used_mb} MiB VRAM in use (> --max-used-mb {}); clear the card (stop its services) or pass --allow-busy.",
                opts.max_used_mb
            ));
        }
        match &u.compute_pids {
            None => {
                return Err(format!(
                    "compute process list of {who} unavailable; pass --allow-busy only after checking the card is idle."
                ));
            }
            Some(p) if !p.is_empty() => {
                return Err(format!(
                    "{who} has compute processes {p:?}; stop them or pass --allow-busy."
                ));
            }
            Some(_) => {}
        }
    }
    Ok(())
}

fn print_health(h: Option<&HealthSnapshot>) {
    let Some(h) = h else {
        println!("health: unavailable (no NVML)");
        return;
    };
    let o = |v: Option<u64>| v.map(|x| x.to_string()).unwrap_or_else(|| "n/a".into());
    println!(
        "health: ECC mode {:?} | volatile corr {} uncorr {} | aggregate corr {} uncorr {} | remapped rows corr {} uncorr {} pending {:?} failure {:?} | PCIe replay {}",
        h.ecc_mode_current,
        o(h.ecc_volatile_corrected),
        o(h.ecc_volatile_uncorrected),
        o(h.ecc_aggregate_corrected),
        o(h.ecc_aggregate_uncorrected),
        o(h.remapped_rows_correctable),
        o(h.remapped_rows_uncorrectable),
        h.remap_pending,
        h.remap_failure,
        o(h.pcie_counters.get("replay").copied().flatten()),
    );
}

fn print_summary(rep: &Report) {
    println!();
    println!("================ BCLL diagnostics summary ================");
    for t in &rep.tests {
        let cov = t
            .coverage
            .as_ref()
            .map(|c| format!(" | tested {:.2} GiB", gib(c.tested_bytes)))
            .unwrap_or_default();
        println!(
            "  {:<22} {:<12} errors {}{}{}",
            t.name,
            format!("{:?}", t.status).to_uppercase(),
            t.error_count,
            cov,
            t.message
                .as_ref()
                .map(|m| format!(" | {m}"))
                .unwrap_or_default()
        );
    }
    for r in &rep.reasons {
        println!("  reason : {r}");
    }
    for w in &rep.warnings {
        println!("  warning: {w}");
    }
    println!(
        "VERDICT: {:?}",
        rep.verdict.unwrap_or(report::Verdict::Incomplete)
    );
}

/// Ok when no output was requested or it was written; Err (already reported) otherwise.
fn finish_report(rep: &mut Report, opts: &Options, started: Instant) -> Result<(), String> {
    rep.run.finished_at = Some(crate::output::NowTime.to_string());
    rep.run.duration_s = Some(started.elapsed().as_secs_f64());
    let Some(path) = &opts.report_json else {
        return Ok(());
    };
    match write_report(rep, path) {
        Ok(()) => {
            eprintln!("output written: {}", path.display());
            Ok(())
        }
        Err(e) => {
            let msg = format!("cannot write output {}: {e}", path.display());
            eprintln!("memtest_vulkan: {msg}");
            Err(msg)
        }
    }
}

/// Unique temp file in the destination directory, then rename; the temp file is removed
/// on failure. Concurrent writers never share a temp name.
pub fn write_report(rep: &Report, path: &Path) -> std::io::Result<()> {
    let text = serde_json::to_string_pretty(rep).map_err(std::io::Error::other)?;
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tmp = path.with_extension(format!("json.{}.{nanos}.tmp", std::process::id()));
    let result = std::fs::write(&tmp, text).and_then(|()| std::fs::rename(&tmp, path));
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PciAddress;
    use crate::bcll::cli::Options;
    use crate::bcll::report::TestResult;

    fn gpu() -> GpuRecord {
        GpuRecord {
            vk_label: 3,
            vk_enum_index: 1,
            name: "NVIDIA GeForce RTX 4090".into(),
            pci: PciAddress {
                domain: 0,
                bus: 0x81,
                device: 0,
                function: 0,
            },
            pci_bus_id: "0000:81:00.0".into(),
            vk_uuid: None,
            nvml_uuid: None,
            nvml_index: Some(1),
            vendor_id: 0x10de,
            device_id: 0x2684,
            driver_version: 0,
            api_version: "1.4.0".into(),
            device_local_bytes: 24 << 30,
            memtest_budget_bytes: 20 << 30,
            index_pci_order: 1,
        }
    }

    fn idle() -> Usage {
        Usage {
            total_bytes: 24 << 30,
            used_bytes: 300 << 20,
            free_bytes: 23 << 30,
            reserved_bytes: None,
            compute_pids: Some(vec![]),
        }
    }

    fn opts() -> Options {
        Options {
            gpu: Some("1".into()),
            ..Default::default()
        }
    }

    #[test]
    fn safety_gate_display_evidence() {
        let g = gpu();
        let u = idle();
        assert!(
            safety_gate(&opts(), &g, Some(true), Some(&u))
                .unwrap_err()
                .contains("drives a display")
        );
        assert!(
            safety_gate(&opts(), &g, None, Some(&u))
                .unwrap_err()
                .contains("unknown")
        );
        assert!(safety_gate(&opts(), &g, Some(false), Some(&u)).is_ok());
        let allow = Options {
            allow_display: true,
            ..opts()
        };
        assert!(safety_gate(&allow, &g, Some(true), Some(&u)).is_ok());
    }

    #[test]
    fn safety_gate_refuses_partial_usage_evidence() {
        let g = gpu();
        // usage query failed entirely
        assert!(
            safety_gate(&opts(), &g, Some(false), None)
                .unwrap_err()
                .contains("usage")
        );
        // memory read but the process list failed: unknown, not "no processes"
        let unknown_pids = Usage {
            compute_pids: None,
            ..idle()
        };
        assert!(
            safety_gate(&opts(), &g, Some(false), Some(&unknown_pids))
                .unwrap_err()
                .contains("process list")
        );
        let busy_pids = Usage {
            compute_pids: Some(vec![4242]),
            ..idle()
        };
        assert!(
            safety_gate(&opts(), &g, Some(false), Some(&busy_pids))
                .unwrap_err()
                .contains("4242")
        );
        let busy_mem = Usage {
            used_bytes: 20 << 30,
            ..idle()
        };
        assert!(
            safety_gate(&opts(), &g, Some(false), Some(&busy_mem))
                .unwrap_err()
                .contains("in use")
        );
        let allow = Options {
            allow_busy: true,
            ..opts()
        };
        assert!(safety_gate(&allow, &g, Some(false), None).is_ok());
    }

    #[test]
    fn nvml_display_tristate() {
        use crate::bcll::telemetry::display_from as d;
        assert_eq!(d(Some(true), None), Some(true));
        assert_eq!(d(None, Some(true)), Some(true));
        assert_eq!(d(Some(false), Some(false)), Some(false));
        // one "no" + one failed query is NOT evidence of "detached"
        assert_eq!(d(Some(false), None), None);
        assert_eq!(d(None, Some(false)), None);
        assert_eq!(d(None, None), None);
    }

    #[test]
    fn report_write_is_valid_json_and_failures_propagate() {
        let dir = std::env::temp_dir().join(format!("bcll-rep-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("r.json");
        let mut rep = Report::new(tool_info(), RunInfo::default());
        rep.tests
            .push(TestResult::new("vram", "vulkan", "vulkan_memtest"));
        write_report(&rep, &path).unwrap();
        let v: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(v["schema"], report::SCHEMA);
        assert_eq!(v["tool"]["implementation"], "rust");
        // destination directory missing: error, and no temp file left behind
        let bad = dir.join("missing-subdir").join("r.json");
        assert!(write_report(&rep, &bad).is_err());
        // finish_report turns a write failure into Err (-> non-zero exit)
        let o = Options {
            report_json: Some(bad),
            ..opts()
        };
        assert!(finish_report(&mut rep, &o, Instant::now()).is_err());
        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn refusal_is_written_with_null_verdict() {
        let dir = std::env::temp_dir().join(format!("bcll-ref-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("r.json");
        let o = Options {
            report_json: Some(path.clone()),
            ..opts()
        };
        let mut rep = Report::new(tool_info(), RunInfo::default());
        let code = abort(
            &mut rep,
            &o,
            Instant::now(),
            Abort::Refused("display".into()),
        );
        assert_eq!(code, report::EXIT_REFUSED);
        let v: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(v["refusal"], "display");
        assert!(v["verdict"].is_null());
        // refusal whose output cannot be written is an operational error
        let o = Options {
            report_json: Some(dir.join("nope").join("r.json")),
            ..opts()
        };
        let code = abort(
            &mut rep,
            &o,
            Instant::now(),
            Abort::Refused("display".into()),
        );
        assert_eq!(code, report::EXIT_INTERNAL);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_dirs_are_unique_under_a_shared_work_dir() {
        let o = Options {
            work_dir: Some(PathBuf::from("shared")),
            ..opts()
        };
        let (a, ca) = run_dir(&o);
        std::thread::sleep(std::time::Duration::from_millis(2));
        let (b, _) = run_dir(&o);
        assert_ne!(a, b);
        assert!(a.starts_with("shared") && !ca);
    }
}
