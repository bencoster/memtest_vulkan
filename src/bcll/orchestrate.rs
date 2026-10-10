//! Orchestrator: inventory -> selection -> safety gate -> health snapshot ->
//! telemetry + worker children -> health snapshot -> verdict -> JSON report.
//! The orchestrator itself never puts work on a GPU (DESIGN.md section 3).

use super::cli::{self, Command, Options, WorkerKind};
use super::inventory::{GpuRecord, gib, hostname, inventory, list, select_gpu, tool_info};
use super::nvml::NvmlCtx;
use super::report::{self, DeviceInfo, Report, RunInfo};
use super::runners::{Ctx, run_cuda, run_memtest, run_vk_worker};
use super::telemetry::{self, HealthSnapshot, Sample, Sampler};
use super::{aer, codec, nvapi};
use serde_json::{Value, json};
use std::path::Path;
use std::time::{Duration, Instant};

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
        json!({"message": "built without the 'cuda' feature", "unsupported": true}),
    );
    report::EXIT_INTERNAL
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
    let nvml = match NvmlCtx::open() {
        Ok(n) => Some(n),
        Err(e) => {
            if !quiet {
                println!("note: {e}");
            }
            None
        }
    };
    let inv = match inventory(nvml.as_ref()) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("memtest_vulkan: {e}");
            return report::EXIT_INTERNAL;
        }
    };
    let spec = opts.gpu.as_deref().unwrap_or_default();
    let gpu = match select_gpu(&inv, spec) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("memtest_vulkan: {e} (see --list)");
            return report::EXIT_REFUSED;
        }
    };
    let display = nvml
        .as_ref()
        .and_then(|n| n.display_attached(&gpu.pci_bus_id));
    let mut details = nvml.as_ref().and_then(|n| n.details(&gpu.pci_bus_id));
    if let (Some(Value::Object(m)), Some(n)) = (&mut details, &nvml) {
        m.insert("system".into(), n.system());
        if !opts.no_nvapi
            && let Some(Value::Object(thermal)) = m.get_mut("thermal")
        {
            let arch = n.architecture(&gpu.pci_bus_id);
            match nvapi::Reader::open(gpu.pci.bus, arch.as_deref(), &gpu.name).map(|mut r| r.read())
            {
                Ok(Some(t)) => {
                    thermal.insert("hotspot_c_nvapi".into(), json!(t.hotspot_c));
                    thermal.insert("memory_junction_c_nvapi".into(), json!(t.mem_junction_c));
                    thermal.insert("nvapi_channels_c".into(), json!(t.raw_c));
                    thermal.insert("nvapi_source".into(), json!(nvapi::SOURCE));
                }
                Ok(None) => {
                    thermal.insert("nvapi_error".into(), json!("thermal read failed"));
                }
                Err(e) => {
                    thermal.insert("nvapi_error".into(), json!(e));
                }
            }
        }
    }
    rep.device = Some(DeviceInfo {
        selector: opts.gpu.clone(),
        index_pci_order: gpu.index_pci_order,
        vk_label: gpu.vk_label,
        pci_bus_id: gpu.pci_bus_id.clone(),
        uuid: gpu.uuid.clone(),
        name: gpu.name.clone(),
        vendor_id: gpu.vendor_id,
        device_id: gpu.device_id,
        vulkan_driver_version: gpu.driver_version,
        vulkan_api_version: gpu.api_version.clone(),
        device_local_bytes: gpu.device_local_bytes,
        display_attached: display,
        details,
    });
    if !quiet {
        println!(
            "Selected GPU index {} | {} | {} | {} | vk label {}",
            gpu.index_pci_order,
            gpu.pci_bus_id,
            gpu.uuid.as_deref().unwrap_or("uuid unknown"),
            gpu.name,
            gpu.vk_label
        );
    }
    let health_before = snapshot(nvml.as_ref(), &gpu.pci_bus_id);
    rep.health_before = health_before
        .as_ref()
        .and_then(|h| serde_json::to_value(h).ok());

    if opts.command == Command::Info {
        rep.verdict = None;
        if !quiet {
            print_health(health_before.as_ref());
        }
        finish_report(&mut rep, opts, started);
        if quiet && let Ok(s) = serde_json::to_string_pretty(&rep) {
            println!("{s}");
        }
        return 0;
    }

    // ---- safety gate (DESIGN.md section 4) ----
    if let Err(why) = safety_gate(opts, &gpu, display, nvml.as_ref()) {
        eprintln!("memtest_vulkan: REFUSED: {why}");
        return report::EXIT_REFUSED;
    }

    let exe = match std::env::current_exe() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("memtest_vulkan: cannot locate own executable: {e}");
            return report::EXIT_INTERNAL;
        }
    };
    let (work, cleanup_work) = match &opts.work_dir {
        Some(d) => (d.clone(), false),
        None => (
            std::env::temp_dir().join(format!("bcll-memtest-{}", std::process::id())),
            true,
        ),
    };
    if let Err(e) = std::fs::create_dir_all(&work) {
        eprintln!(
            "memtest_vulkan: cannot create work dir {}: {e}",
            work.display()
        );
        return report::EXIT_INTERNAL;
    }

    let tele = (!opts.no_telemetry).then(|| start_telemetry(opts, &gpu, nvml.as_ref()));
    let ctx = Ctx {
        opts,
        gpu,
        exe,
        work: work.clone(),
        nvml,
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
            ctx.gpu.uuid.as_deref(),
            opts.codec_frames,
            &ctx.work,
        ));
    }

    let (samples, tele_meta) = match tele {
        Some((handle, meta)) => (handle.stop(), Some(meta)),
        None => (Vec::new(), None),
    };
    let health_after = snapshot(ctx.nvml.as_ref(), &ctx.gpu.pci_bus_id);
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
    let max_link = ctx
        .nvml
        .as_ref()
        .and_then(|n| n.max_link(&ctx.gpu.pci_bus_id));
    let findings = telemetry::findings(
        health_before.as_ref(),
        health_after.as_ref(),
        summary.as_ref(),
        max_link,
    );
    let (verdict, reasons, warnings) = report::decide(&rep.tests, &findings);
    rep.verdict = Some(verdict);
    rep.reasons = reasons;
    rep.warnings = warnings;
    if opts.no_telemetry {
        rep.warnings
            .push("telemetry disabled (--no-telemetry): no thermal/PCIe/ECC correlation".into());
    } else if samples.is_empty() || ctx.nvml.is_none() {
        rep.warnings
            .push("telemetry unavailable (no NVML): PCIe/ECC/thermal evidence missing".into());
    }
    print_summary(&rep);
    finish_report(&mut rep, opts, started);
    if cleanup_work {
        let _ = std::fs::remove_dir_all(&work);
    }
    report::exit_code(verdict)
}

pub fn safety_gate(
    opts: &Options,
    gpu: &GpuRecord,
    display: Option<bool>,
    nvml: Option<&NvmlCtx>,
) -> Result<(), String> {
    if !opts.allow_display {
        match display {
            Some(true) => {
                return Err(format!(
                    "GPU {} ({}) drives a display; testing it can black-screen the desktop. Pass --allow-display to override.",
                    gpu.index_pci_order, gpu.pci_bus_id
                ));
            }
            None => {
                return Err(format!(
                    "display state of GPU {} unknown (no NVML); pass --allow-display after checking it drives no monitor.",
                    gpu.index_pci_order
                ));
            }
            Some(false) => {}
        }
    }
    if !opts.allow_busy
        && let Some(u) = nvml.and_then(|n| n.usage(&gpu.pci_bus_id))
    {
        let used_mb = u.used_bytes >> 20;
        if used_mb > opts.max_used_mb {
            return Err(format!(
                "GPU {} has {used_mb} MiB VRAM in use (> --max-used-mb {}); clear the card (stop its services) or pass --allow-busy.",
                gpu.index_pci_order, opts.max_used_mb
            ));
        }
        if !u.compute_pids.is_empty() {
            return Err(format!(
                "GPU {} has compute processes {:?}; stop them or pass --allow-busy.",
                gpu.index_pci_order, u.compute_pids
            ));
        }
    }
    Ok(())
}

fn snapshot(nvml: Option<&NvmlCtx>, pci_bus_id: &str) -> Option<HealthSnapshot> {
    let aer_snap = aer::read(pci_bus_id);
    match nvml.and_then(|n| n.health(pci_bus_id)) {
        Some(mut h) => {
            h.aer = aer_snap;
            Some(h)
        }
        None => aer_snap.map(|a| HealthSnapshot {
            taken_at: crate::output::NowTime.to_string(),
            aer: Some(a),
            ..Default::default()
        }),
    }
}

struct CompositeSampler {
    nvml: Option<NvmlCtx>,
    pci_bus_id: String,
    nvapi: Option<nvapi::Reader>,
}

impl Sampler for CompositeSampler {
    fn sample(&mut self, t_s: f64) -> Sample {
        let mut s = match &self.nvml {
            Some(n) => n.sample(&self.pci_bus_id, t_s),
            None => Sample {
                t_s,
                ..Default::default()
            },
        };
        if let Some(t) = self.nvapi.as_mut().and_then(|r| r.read()) {
            s.temp_hotspot_c = t.hotspot_c;
            if t.mem_junction_c.is_some() {
                s.temp_mem_junction_c = t.mem_junction_c;
            }
        }
        if let Some(a) = aer::read(&self.pci_bus_id).filter(|a| a.has_counters()) {
            s.aer_correctable = Some(a.total_correctable());
            s.aer_nonfatal = Some(a.total_nonfatal());
            s.aer_fatal = Some(a.total_fatal());
        }
        s
    }
}

fn start_telemetry(
    opts: &Options,
    gpu: &GpuRecord,
    nvml: Option<&NvmlCtx>,
) -> (telemetry::Handle, Value) {
    let use_nvapi = cfg!(all(windows, feature = "nvapi")) && !opts.no_nvapi;
    let arch = nvml.and_then(|n| n.architecture(&gpu.pci_bus_id));
    let mut sources = Vec::new();
    if nvml.is_some() {
        sources.push("nvml");
    }
    if use_nvapi {
        sources.push(nvapi::SOURCE);
    }
    if cfg!(target_os = "linux") {
        sources.push("linux_sysfs_aer");
    }
    let meta = json!({
        "interval_ms": opts.telemetry_ms,
        "sources": sources,
        "memory_junction_source": if use_nvapi { Some(nvapi::SOURCE) } else { None },
    });
    let (pci_bus_id, bus, name) = (gpu.pci_bus_id.clone(), gpu.pci.bus, gpu.name.clone());
    let handle = telemetry::start(
        Duration::from_millis(opts.telemetry_ms),
        Duration::from_secs_f64(opts.telemetry_print_s),
        move || {
            let nvapi_reader = if use_nvapi {
                match nvapi::Reader::open(bus, arch.as_deref(), &name) {
                    Ok(r) => Some(r),
                    Err(e) => {
                        println!("note: NVAPI junction temperature unavailable: {e}");
                        None
                    }
                }
            } else {
                None
            };
            Some(Box::new(CompositeSampler {
                nvml: NvmlCtx::open().ok(),
                pci_bus_id,
                nvapi: nvapi_reader,
            }) as Box<dyn Sampler>)
        },
    );
    (handle, meta)
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

fn finish_report(rep: &mut Report, opts: &Options, started: Instant) {
    rep.run.finished_at = Some(crate::output::NowTime.to_string());
    rep.run.duration_s = Some(started.elapsed().as_secs_f64());
    if let Some(path) = &opts.report_json {
        match write_report(rep, path) {
            Ok(()) => eprintln!("report written: {}", path.display()),
            Err(e) => eprintln!(
                "memtest_vulkan: cannot write report {}: {e}",
                path.display()
            ),
        }
    }
}

/// Atomic-ish write: temp file in the same directory, then rename.
pub fn write_report(rep: &Report, path: &Path) -> std::io::Result<()> {
    let text = serde_json::to_string_pretty(rep).map_err(std::io::Error::other)?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, text)?;
    std::fs::rename(&tmp, path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PciAddress;
    use crate::bcll::cli::Options;
    use crate::bcll::report::TestResult;

    fn mt() -> TestResult {
        TestResult::new("vram", "vulkan", "vulkan_memtest")
    }

    fn rec(display: Option<bool>) -> (GpuRecord, Option<bool>) {
        (
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
                uuid: None,
                vendor_id: 0x10de,
                device_id: 0x2684,
                driver_version: 0,
                api_version: "1.4.0".into(),
                device_local_bytes: 24 << 30,
                memtest_budget_bytes: 20 << 30,
                index_pci_order: 1,
            },
            display,
        )
    }

    #[test]
    fn safety_gate_refuses_display_and_unknown() {
        let opts = Options {
            gpu: Some("1".into()),
            ..Default::default()
        };
        let (g, d) = rec(Some(true));
        assert!(
            safety_gate(&opts, &g, d, None)
                .unwrap_err()
                .contains("drives a display")
        );
        let (g, d) = rec(None);
        assert!(
            safety_gate(&opts, &g, d, None)
                .unwrap_err()
                .contains("unknown")
        );
        let (g, d) = rec(Some(false));
        assert!(safety_gate(&opts, &g, d, None).is_ok());
        let allow = Options {
            allow_display: true,
            ..opts
        };
        let (g, d) = rec(Some(true));
        assert!(safety_gate(&allow, &g, d, None).is_ok());
    }

    #[test]
    fn report_write_is_valid_json() {
        let dir = std::env::temp_dir().join(format!("bcll-rep-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("r.json");
        let mut rep = Report::new(tool_info(), RunInfo::default());
        rep.tests.push(mt());
        write_report(&rep, &path).unwrap();
        let v: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(v["schema"], report::SCHEMA);
        assert_eq!(v["tool"]["implementation"], "rust");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
