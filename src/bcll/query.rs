//! Driver-facing worker processes (Codex BLOCK 89de0c26 #1): the supervisor never calls
//! Vulkan/NVML/NVAPI itself. It spawns
//! * `--bcll-worker query --query inventory|snapshot` and reads ONE marked JSON line from
//!   its stdout, with a timeout (a wedged driver call costs one killed worker, not the run);
//! * `--bcll-worker telemetry`, which appends samples to an events file until it is
//!   stopped (bounded kill) or reaches its own hard deadline.

use super::cli::WorkerArgs;
use super::events::{self, EventWriter, of_kind};
use super::inventory::{self, GpuRecord, Inventory};
use super::nvml::NvmlCtx;
use super::proc::{self, Limits};
use super::telemetry::{self, HealthSnapshot, Sample, Sampler, Usage};
use super::{aer, nvapi, select};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

pub const RESULT_MARK: &str = "BCLL-RESULT ";
pub const QUERY_TIMEOUT: Duration = Duration::from_secs(60);
const TELEMETRY_STOP_GRACE: Duration = Duration::from_secs(10);

/// One read-only look at a card (display, occupancy, details, growing counters).
#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq)]
pub struct Snapshot {
    pub display_attached: Option<bool>,
    pub usage: Option<Usage>,
    pub details: Option<Value>,
    pub health: Option<HealthSnapshot>,
    pub max_link: Option<(u32, u32)>,
    pub architecture: Option<String>,
    /// why parts are missing (NVML unavailable, worker timed out, ...)
    pub notes: Vec<String>,
}

// ---------------------------------------------------------------- worker side

pub fn run_query_worker(w: &WorkerArgs) -> i32 {
    let limit = w
        .hard_deadline_s
        .unwrap_or(QUERY_TIMEOUT.as_secs_f64() - 5.0);
    proc::arm_self_deadline(Some(limit), 2, Path::new(""));
    let nvml = NvmlCtx::open();
    let out = match w.query.as_deref() {
        Some("inventory") => match inventory::enumerate(nvml.as_ref().ok()) {
            Ok(gpus) => json!(Inventory {
                gpus,
                nvml_error: nvml.as_ref().err().cloned(),
            }),
            Err(e) => json!({ "error": e }),
        },
        Some("snapshot") => json!(snapshot_local(
            nvml.as_ref().map_err(|e| e.as_str()),
            w.pci.as_deref().unwrap_or_default(),
            w.name.as_deref(),
            w.no_nvapi,
        )),
        other => json!({ "error": format!("unknown query {other:?}") }),
    };
    println!("{RESULT_MARK}{out}");
    0
}

fn snapshot_local(
    nvml: Result<&NvmlCtx, &str>,
    pci: &str,
    name: Option<&str>,
    no_nvapi: bool,
) -> Snapshot {
    let mut s = Snapshot::default();
    match nvml {
        Err(e) => s.notes.push(e.to_string()),
        Ok(n) => {
            s.display_attached = n.display_attached(pci);
            s.usage = n.usage(pci);
            s.architecture = n.architecture(pci);
            s.max_link = n.max_link(pci);
            s.health = n.health(pci);
            let mut details = n.details(pci);
            if let Some(Value::Object(m)) = &mut details {
                m.insert("system".into(), n.system());
                let card_name = name
                    .map(String::from)
                    .or_else(|| m.get("name").and_then(|v| v.as_str()).map(String::from))
                    .unwrap_or_default();
                if !no_nvapi
                    && let Some(p) = select::parse_pci(pci)
                    && let Some(Value::Object(thermal)) = m.get_mut("thermal")
                {
                    match nvapi::Reader::open(
                        p.bus,
                        p.device,
                        s.architecture.as_deref(),
                        &card_name,
                    )
                    .map(|mut r| r.read())
                    {
                        Ok(Some(t)) => {
                            thermal.insert("hotspot_c_nvapi".into(), json!(t.hotspot_c));
                            thermal
                                .insert("memory_junction_c_nvapi".into(), json!(t.mem_junction_c));
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
            s.details = details;
        }
    }
    let aer_snap = aer::read(pci);
    match &mut s.health {
        Some(h) => h.aer = aer_snap,
        None if aer_snap.is_some() => {
            s.health = Some(HealthSnapshot {
                taken_at: crate::output::NowTime.to_string(),
                aer: aer_snap,
                ..Default::default()
            })
        }
        None => {}
    }
    s
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

pub fn run_telemetry_worker(w: &WorkerArgs) -> i32 {
    proc::arm_self_deadline(
        w.hard_deadline_s.or(Some(w.seconds + 30.0)),
        0,
        Path::new(""),
    );
    let pci = w.pci.clone().unwrap_or_default();
    let nvml = NvmlCtx::open().ok();
    let use_nvapi = cfg!(all(windows, feature = "nvapi")) && !w.no_nvapi;
    let arch = nvml.as_ref().and_then(|n| n.architecture(&pci));
    let nvapi_reader = if use_nvapi {
        let p = select::parse_pci(&pci).unwrap_or_default();
        match nvapi::Reader::open(
            p.bus,
            p.device,
            arch.as_deref(),
            w.name.as_deref().unwrap_or(""),
        ) {
            Ok(r) => Some(r),
            Err(e) => {
                println!("note: NVAPI junction temperature unavailable: {e}");
                None
            }
        }
    } else {
        None
    };
    let mut sources = Vec::new();
    if nvml.is_some() {
        sources.push("nvml");
    }
    if nvapi_reader.is_some() {
        sources.push(nvapi::SOURCE);
    }
    if cfg!(target_os = "linux") {
        sources.push("linux_sysfs_aer");
    }
    let mut ev = EventWriter::create(&w.events);
    ev.emit(
        "telemetry_started",
        json!({
            "sources": sources,
            "memory_junction_source": nvapi_reader.as_ref().map(|_| nvapi::SOURCE),
        }),
    );
    let mut sampler = CompositeSampler {
        nvml,
        pci_bus_id: pci,
        nvapi: nvapi_reader,
    };
    telemetry::run_loop(
        &mut sampler,
        Duration::from_millis(w.interval_ms),
        Duration::from_secs_f64(w.print_s),
        Instant::now() + Duration::from_secs_f64(w.seconds),
        |s| {
            if let Ok(v) = serde_json::to_value(s) {
                ev.emit("telemetry_sample", v);
            }
        },
    );
    0
}

// ---------------------------------------------------------------- supervisor side

/// Extracts the marked result line (pure; unit-tested).
pub fn parse_result(stdout: &str) -> Result<Value, String> {
    let line = stdout
        .lines()
        .rev()
        .find_map(|l| l.trim_end().strip_prefix(RESULT_MARK))
        .ok_or("query worker produced no result")?;
    let v: Value = serde_json::from_str(line).map_err(|e| format!("bad query result: {e}"))?;
    match v.get("error").and_then(|e| e.as_str()) {
        Some(e) => Err(e.to_string()),
        None => Ok(v),
    }
}

fn query(exe: &Path, args: &[&str]) -> Result<Value, String> {
    let mut cmd = Command::new(exe);
    cmd.args(["--bcll-worker", "query"]).args(args);
    let o = proc::run(cmd, &Limits::new(QUERY_TIMEOUT), true, false)?;
    if o.timed_out {
        return Err(format!(
            "driver query timed out after {}s (wedged driver?){}",
            QUERY_TIMEOUT.as_secs(),
            o.death_note().map(|n| format!("; {n}")).unwrap_or_default()
        ));
    }
    parse_result(&o.stdout)
}

pub fn inventory(exe: &Path) -> Result<Inventory, String> {
    let v = query(exe, &["--query", "inventory"])?;
    serde_json::from_value(v).map_err(|e| format!("bad inventory: {e}"))
}

/// Never fails: an unavailable snapshot is all-unknown with a note (gates then refuse).
pub fn snapshot(exe: &Path, gpu: &GpuRecord, no_nvapi: bool) -> Snapshot {
    let mut args = vec![
        "--query",
        "snapshot",
        "--pci",
        gpu.pci_bus_id.as_str(),
        "--name",
        gpu.name.as_str(),
    ];
    if no_nvapi {
        args.push("--no-nvapi");
    }
    match query(exe, &args).and_then(|v| serde_json::from_value(v).map_err(|e| e.to_string())) {
        Ok(s) => s,
        Err(e) => Snapshot {
            notes: vec![format!("snapshot unavailable: {e}")],
            ..Default::default()
        },
    }
}

pub struct TelemetryRun {
    bg: proc::Background,
    events: PathBuf,
    interval_ms: u64,
}

pub fn start_telemetry(
    exe: &Path,
    gpu: &GpuRecord,
    opts: &super::cli::Options,
    work: &Path,
    planned_s: f64,
) -> Result<TelemetryRun, String> {
    let events = work.join("telemetry-events.jsonl");
    let mut cmd = Command::new(exe);
    cmd.args([
        "--bcll-worker",
        "telemetry",
        "--pci",
        &gpu.pci_bus_id,
        "--name",
        &gpu.name,
    ])
    .arg("--seconds")
    .arg(format!("{planned_s}"))
    .arg("--hard-deadline-s")
    .arg(format!("{}", planned_s + 60.0))
    .arg("--events")
    .arg(&events)
    .arg("--telemetry-ms")
    .arg(opts.telemetry_ms.to_string())
    .arg("--telemetry-print-s")
    .arg(format!("{}", opts.telemetry_print_s));
    if opts.no_nvapi {
        cmd.arg("--no-nvapi");
    }
    proc::Background::spawn(cmd).map(|bg| TelemetryRun {
        bg,
        events,
        interval_ms: opts.telemetry_ms,
    })
}

/// Stops the sampler (bounded) and returns its samples, report metadata and an optional warning.
pub fn stop_telemetry(t: TelemetryRun) -> (Vec<Sample>, Value, Option<String>) {
    let out = t.bg.stop(TELEMETRY_STOP_GRACE);
    let ev = events::read_events(&t.events);
    let (samples, meta) = telemetry_from_events(&ev, t.interval_ms);
    (
        samples,
        meta,
        out.death_note().map(|n| format!("telemetry worker: {n}")),
    )
}

/// Pure conversion of the telemetry worker's events (unit-tested).
pub fn telemetry_from_events(ev: &[Value], interval_ms: u64) -> (Vec<Sample>, Value) {
    let samples: Vec<Sample> = of_kind(ev, "telemetry_sample")
        .filter_map(|v| serde_json::from_value(v.clone()).ok())
        .collect();
    let started = of_kind(ev, "telemetry_started").last();
    let meta = json!({
        "interval_ms": interval_ms,
        "sources": started.map(|s| s["sources"].clone()).unwrap_or(json!([])),
        "memory_junction_source": started.map(|s| s["memory_junction_source"].clone()).unwrap_or(Value::Null),
    });
    (samples, meta)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn result_line_is_found_among_other_output() {
        let out = format!(
            "note: something\nDestroying vk instance...\n{RESULT_MARK}{{\"gpus\":[],\"nvml_error\":\"x\"}}\n"
        );
        let v = parse_result(&out).unwrap();
        assert_eq!(v["nvml_error"], "x");
        assert!(parse_result("no marker").unwrap_err().contains("no result"));
        assert_eq!(
            parse_result(&format!(
                "{RESULT_MARK}{{\"error\":\"Vulkan enumeration failed\"}}"
            ))
            .unwrap_err(),
            "Vulkan enumeration failed"
        );
        assert!(parse_result(&format!("{RESULT_MARK}{{truncated")).is_err());
    }

    #[test]
    fn snapshot_without_nvml_is_all_unknown() {
        let s = snapshot_local(Err("NVML unavailable: x"), "0000:c1:00.0", None, true);
        assert_eq!(s.display_attached, None);
        assert_eq!(s.usage, None);
        assert_eq!(s.notes, vec!["NVML unavailable: x".to_string()]);
        let back: Snapshot = serde_json::from_value(json!(s)).unwrap();
        assert_eq!(back, s);
    }

    #[test]
    fn telemetry_events_become_samples() {
        let ev = vec![
            json!({"ev": "telemetry_started", "t_s": 0.0, "sources": ["nvml"], "memory_junction_source": null}),
            json!({"ev": "telemetry_sample", "t_s": 1.0, "temp_gpu_c": 60, "pcie_replay": 3}),
            json!({"ev": "telemetry_sample", "t_s": 2.0, "temp_gpu_c": 61, "pcie_replay": 4}),
            json!({"ev": "telemetry_sample", "t_s": "garbage"}),
        ];
        let (s, meta) = telemetry_from_events(&ev, 1000);
        assert_eq!(s.len(), 2);
        assert_eq!(s[1].temp_gpu_c, Some(61));
        assert_eq!(meta["sources"][0], "nvml");
        assert_eq!(telemetry::summarize(&s).pcie_replay_delta, Some(1));
    }
}
