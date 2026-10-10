//! Orchestrator: inventory -> selection -> safety gate -> health snapshot ->
//! telemetry + worker children -> health snapshot -> verdict -> JSON report.
//! The orchestrator itself never puts work on a GPU (DESIGN.md section 3).

use super::cli::{self, Command, Options, WorkerKind};
use super::events::{self, of_kind};
use super::nvml::NvmlCtx;
use super::report::{
    self, Coverage, DeviceInfo, ErrorRecord, Report, RunInfo, TestResult, TestStatus, ToolInfo,
};
use super::telemetry::{self, HealthSnapshot, Sample, Sampler};
use super::{aer, codec, nvapi, select};
use crate::PciAddress;
use erupt::vk;
use serde_json::{Value, json};
use std::ffi::CStr;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::Ordering::SeqCst;
use std::time::{Duration, Instant};

/// Extra wall time a memtest child gets for allocation + initial fill.
const MEMTEST_INIT_ALLOWANCE_S: f64 = 90.0;
const MIN_TEST_BYTES: i64 = 400 << 20;
const RETRY_STEP_BYTES: i64 = 400 << 20;
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

fn tool_info() -> ToolInfo {
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

fn hostname() -> Option<String> {
    std::env::var("COMPUTERNAME")
        .or_else(|_| std::env::var("HOSTNAME"))
        .ok()
        .or_else(|| std::fs::read_to_string("/etc/hostname").ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

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
    let mut ev = events::EventWriter::create(&w.events);
    ev.emit(
        "worker_error",
        json!({"message": "built without the 'cuda' feature", "unsupported": true}),
    );
    report::EXIT_INTERNAL
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

fn gib(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

fn list(opts: &Options) -> i32 {
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

fn select_gpu(inv: &[GpuRecord], spec_raw: &str) -> Result<GpuRecord, String> {
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

struct Ctx<'a> {
    opts: &'a Options,
    gpu: GpuRecord,
    exe: PathBuf,
    work: PathBuf,
    nvml: Option<NvmlCtx>,
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

fn run_memtest(ctx: &Ctx, seconds: f64) -> TestResult {
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
        match init_outcome(code, &ev, attempts.len()) {
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
        t = judge_memtest(t, code, timed_out, &ev);
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

fn worker_status(code: Option<i32>, timed_out: bool, ev: &[Value]) -> (TestStatus, Option<String>) {
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

fn run_vk_worker(ctx: &Ctx, kind: &str, seconds: f64) -> TestResult {
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
    (t.status, t.message) = worker_status(code, timed_out, &ev);
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
    t.termination = Some(worker_termination(t.status, code, timed_out, &ev).into());
    t.duration_s = Some(start.elapsed().as_secs_f64());
    t
}

fn run_cuda(ctx: &Ctx, seconds: f64) -> TestResult {
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
    t = judge_cuda(t, code, timed_out, &ev);
    if let Some(c) = t.coverage.as_mut() {
        c.driver_reserved_bytes = coverage_from_nvml(ctx, 0).driver_reserved_bytes;
    }
    t.duration_s = Some(start.elapsed().as_secs_f64());
    t
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
    use crate::close::app_status as st;

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
