//! Long-flag CLI of the BCLL diagnostics mode.
//!
//! Only reached when argv[1] starts with "--" (see `bcll::run_if_requested`), so
//! upstream's positional child protocol (`<index> <bytes>`) and the interactive
//! flow are untouched. Fixes REVIEW R6 for the fork: `--help` prints help and
//! never falls into the interactive autoselect.

use std::path::PathBuf;

pub const DEFAULT_MEMTEST_S: f64 = 300.0;

#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    Help,
    Version,
    List,
    Info,
    Run,
    Worker(WorkerArgs),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerKind {
    Compute,
    Transfer,
    CudaMemtest,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WorkerArgs {
    pub kind: WorkerKind,
    pub vk_index: Option<usize>,
    pub uuid: Option<String>,
    pub pci: Option<String>,
    pub seconds: f64,
    pub events: PathBuf,
    pub keep_free_mb: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Options {
    pub command: Command,
    pub gpu: Option<String>,
    pub allow_display: bool,
    pub allow_busy: bool,
    pub max_used_mb: u64,
    pub json: bool,
    pub report_json: Option<PathBuf>,
    pub memtest_s: Option<f64>,
    pub memtest_bytes: Option<u64>,
    pub compute_s: Option<f64>,
    pub transfer_s: Option<f64>,
    pub cuda_memtest_s: Option<f64>,
    pub cuda_keep_free_mb: u64,
    pub codec: bool,
    pub codec_frames: u32,
    pub ffmpeg: Option<PathBuf>,
    pub telemetry_ms: u64,
    pub telemetry_print_s: f64,
    pub no_telemetry: bool,
    pub no_nvapi: bool,
    pub work_dir: Option<PathBuf>,
    pub worker_grace_s: f64,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            command: Command::Run,
            gpu: None,
            allow_display: false,
            allow_busy: false,
            max_used_mb: 1024,
            json: false,
            report_json: None,
            memtest_s: None,
            memtest_bytes: None,
            compute_s: None,
            transfer_s: None,
            cuda_memtest_s: None,
            cuda_keep_free_mb: 256,
            codec: false,
            codec_frames: 150,
            ffmpeg: None,
            telemetry_ms: 1000,
            telemetry_print_s: 10.0,
            no_telemetry: false,
            no_nvapi: false,
            work_dir: None,
            worker_grace_s: 120.0,
        }
    }
}

impl Options {
    pub fn any_test_selected(&self) -> bool {
        self.memtest_s.is_some()
            || self.compute_s.is_some()
            || self.transfer_s.is_some()
            || self.cuda_memtest_s.is_some()
            || self.codec
    }
}

pub const HELP: &str = "\
memtest_vulkan (BCLL diagnostics fork)

Without --flags the program behaves exactly like upstream memtest_vulkan.

USAGE
  memtest_vulkan --list [--json]
  memtest_vulkan --info --gpu SPEC [--report-json FILE]
  memtest_vulkan --gpu SPEC [TESTS] [OPTIONS]

GPU SELECTION (--gpu SPEC)
  0 | index:0                 Nth GPU in PCI bus order (= nvidia-smi index,
                              CUDA_DEVICE_ORDER=PCI_BUS_ID). NOT upstream's label.
  0000:81:00.0 | 81:00.0      PCI address (pci: prefix optional)
  GPU-6727... | uuid:GPU-...  NVML/CUDA UUID (unique prefix accepted)
  vk:N                        upstream memtest_vulkan label number

TESTS (default when none given: --memtest 300)
  --memtest SECONDS           upstream Vulkan VRAM test, time limited
  --memtest-bytes SIZE        cap the tested size (e.g. 20G, 512M)
  --compute SECONDS           Vulkan ALU/FPU verification (int exact + FP32 consensus)
  --transfer SECONDS          PCIe host<->device copy integrity + bandwidth
  --cuda-memtest SECONDS      CUDA driver-API VMM VRAM test (patterns, coverage)
  --cuda-keep-free-mb N       VRAM left unclaimed by --cuda-memtest (default 256)
  --codec                     NVENC/NVDEC round trip via ffmpeg (h264/hevc/av1)
  --ffmpeg PATH               ffmpeg executable (default: ffmpeg on PATH)
  --codec-frames N            frames per codec (default 150)

SAFETY
  --allow-display             permit a display-attached (or display-unknown) GPU
  --allow-busy                permit a GPU with VRAM in use / compute processes
  --max-used-mb N             'busy' threshold for used VRAM (default 1024)

OUTPUT / TELEMETRY
  --report-json FILE          write the bcll.gpu_diag.report v1 JSON
  --telemetry-ms N            sample interval (default 1000)
  --telemetry-print-s S       console telemetry line period (default 10)
  --no-telemetry              disable the sampler
  --no-nvapi                  do not use the undocumented NVAPI thermal call
  --work-dir DIR              where worker event files go (default: temp dir)
  --worker-grace-s S          extra time before a worker is declared hung (default 120)

EXIT CODES  0 PASS, 1 FAIL, 2 INCOMPLETE, 3 refused/usage, 4 internal error
Docs: docs/bcll/USAGE.md
";

fn parse_f64(flag: &str, v: &str) -> Result<f64, String> {
    let x: f64 = v
        .trim()
        .parse()
        .map_err(|_| format!("{flag}: expected a number, got '{v}'"))?;
    if !x.is_finite() || x <= 0.0 {
        return Err(format!("{flag}: must be > 0, got '{v}'"));
    }
    Ok(x)
}

fn parse_u64(flag: &str, v: &str) -> Result<u64, String> {
    v.trim()
        .parse()
        .map_err(|_| format!("{flag}: expected an integer, got '{v}'"))
}

/// Parses `123`, `512K`, `512M`, `20G`, `1T` (binary units, optional trailing `B`/`iB`).
pub fn parse_size(v: &str) -> Result<u64, String> {
    let t = v.trim().to_ascii_uppercase();
    let t = t
        .strip_suffix("IB")
        .or_else(|| t.strip_suffix('B'))
        .unwrap_or(&t);
    let (num, mul) = match t.chars().last() {
        Some('K') => (&t[..t.len() - 1], 1u64 << 10),
        Some('M') => (&t[..t.len() - 1], 1u64 << 20),
        Some('G') => (&t[..t.len() - 1], 1u64 << 30),
        Some('T') => (&t[..t.len() - 1], 1u64 << 40),
        _ => (t, 1u64),
    };
    let n: f64 = num
        .trim()
        .parse()
        .map_err(|_| format!("invalid size '{v}'"))?;
    if !n.is_finite() || n <= 0.0 {
        return Err(format!("invalid size '{v}'"));
    }
    Ok((n * mul as f64) as u64)
}

pub fn parse(args: &[String]) -> Result<Options, String> {
    let mut o = Options::default();
    let mut explicit_command: Option<Command> = None;
    let mut worker_kind: Option<WorkerKind> = None;
    let mut w_vk_index = None;
    let mut w_uuid = None;
    let mut w_pci = None;
    let mut w_seconds = None;
    let mut w_events = None;

    let mut i = 0;
    while i < args.len() {
        let raw = &args[i];
        let (flag, inline_value) = match raw.split_once('=') {
            Some((f, v)) if raw.starts_with("--") => (f.to_string(), Some(v.to_string())),
            _ => (raw.clone(), None),
        };
        let mut take = |name: &str| -> Result<String, String> {
            if let Some(v) = inline_value.clone() {
                return Ok(v);
            }
            i += 1;
            args.get(i)
                .cloned()
                .ok_or_else(|| format!("{name} requires a value"))
        };
        match flag.as_str() {
            "--help" | "-h" => explicit_command = Some(Command::Help),
            "--version" | "-V" => explicit_command = Some(Command::Version),
            "--list" => explicit_command = Some(Command::List),
            "--info" => explicit_command = Some(Command::Info),
            "--json" => o.json = true,
            "--gpu" => o.gpu = Some(take("--gpu")?),
            "--allow-display" => o.allow_display = true,
            "--allow-busy" => o.allow_busy = true,
            "--max-used-mb" => o.max_used_mb = parse_u64(&flag, &take(&flag)?)?,
            "--report-json" => o.report_json = Some(PathBuf::from(take(&flag)?)),
            "--memtest" => o.memtest_s = Some(parse_f64(&flag, &take(&flag)?)?),
            "--memtest-bytes" => o.memtest_bytes = Some(parse_size(&take(&flag)?)?),
            "--compute" => o.compute_s = Some(parse_f64(&flag, &take(&flag)?)?),
            "--transfer" => o.transfer_s = Some(parse_f64(&flag, &take(&flag)?)?),
            "--cuda-memtest" => o.cuda_memtest_s = Some(parse_f64(&flag, &take(&flag)?)?),
            "--cuda-keep-free-mb" => o.cuda_keep_free_mb = parse_u64(&flag, &take(&flag)?)?,
            "--codec" => o.codec = true,
            "--codec-frames" => {
                let n = parse_u64(&flag, &take(&flag)?)?;
                if !(10..=10_000).contains(&n) {
                    return Err("--codec-frames must be within 10..=10000".into());
                }
                o.codec_frames = n as u32;
            }
            "--ffmpeg" => o.ffmpeg = Some(PathBuf::from(take(&flag)?)),
            "--telemetry-ms" => {
                let n = parse_u64(&flag, &take(&flag)?)?;
                if n < 100 {
                    return Err("--telemetry-ms must be >= 100".into());
                }
                o.telemetry_ms = n;
            }
            "--telemetry-print-s" => o.telemetry_print_s = parse_f64(&flag, &take(&flag)?)?,
            "--no-telemetry" => o.no_telemetry = true,
            "--no-nvapi" => o.no_nvapi = true,
            "--work-dir" => o.work_dir = Some(PathBuf::from(take(&flag)?)),
            "--worker-grace-s" => o.worker_grace_s = parse_f64(&flag, &take(&flag)?)?,
            // internal worker protocol (spawned by the orchestrator)
            "--bcll-worker" => {
                worker_kind = Some(match take(&flag)?.as_str() {
                    "compute" => WorkerKind::Compute,
                    "transfer" => WorkerKind::Transfer,
                    "cuda-memtest" => WorkerKind::CudaMemtest,
                    other => return Err(format!("unknown worker '{other}'")),
                })
            }
            "--vk-index" => w_vk_index = Some(parse_u64(&flag, &take(&flag)?)? as usize),
            "--uuid" => w_uuid = Some(take(&flag)?),
            "--pci" => w_pci = Some(take(&flag)?),
            "--seconds" => w_seconds = Some(parse_f64(&flag, &take(&flag)?)?),
            "--events" => w_events = Some(PathBuf::from(take(&flag)?)),
            other => return Err(format!("unknown argument '{other}' (see --help)")),
        }
        i += 1;
    }

    if let Some(kind) = worker_kind {
        o.command = Command::Worker(WorkerArgs {
            kind,
            vk_index: w_vk_index,
            uuid: w_uuid,
            pci: w_pci,
            seconds: w_seconds.ok_or("--bcll-worker requires --seconds")?,
            events: w_events.ok_or("--bcll-worker requires --events")?,
            keep_free_mb: o.cuda_keep_free_mb,
        });
        return Ok(o);
    }
    o.command = explicit_command.unwrap_or(Command::Run);
    if o.command == Command::Run {
        if o.gpu.is_none() {
            return Err(
                "--gpu is required to run tests (use --list to see GPUs; there is no autoselect)"
                    .into(),
            );
        }
        if !o.any_test_selected() {
            o.memtest_s = Some(DEFAULT_MEMTEST_S);
        }
    }
    if o.command == Command::Info && o.gpu.is_none() {
        return Err("--info requires --gpu".into());
    }
    Ok(o)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
    }

    #[test]
    fn help_never_runs_a_test() {
        assert_eq!(parse(&args("--help")).unwrap().command, Command::Help);
        assert_eq!(
            parse(&args("--gpu 1 --help")).unwrap().command,
            Command::Help
        );
    }

    #[test]
    fn run_requires_explicit_gpu() {
        let e = parse(&args("--memtest 10")).unwrap_err();
        assert!(e.contains("--gpu is required"));
    }

    #[test]
    fn default_test_is_standard_memtest() {
        let o = parse(&args("--gpu 0")).unwrap();
        assert_eq!(o.command, Command::Run);
        assert_eq!(o.memtest_s, Some(DEFAULT_MEMTEST_S));
    }

    #[test]
    fn explicit_tests_disable_default_memtest() {
        let o = parse(&args("--gpu 0 --compute 30 --codec")).unwrap();
        assert_eq!(o.memtest_s, None);
        assert_eq!(o.compute_s, Some(30.0));
        assert!(o.codec);
    }

    #[test]
    fn inline_values_and_sizes() {
        let o = parse(&args(
            "--gpu=81:00.0 --memtest=60 --memtest-bytes=20G --report-json=r.json",
        ))
        .unwrap();
        assert_eq!(o.gpu.as_deref(), Some("81:00.0"));
        assert_eq!(o.memtest_bytes, Some(20 << 30));
        assert_eq!(o.report_json, Some(PathBuf::from("r.json")));
    }

    #[test]
    fn size_parsing() {
        assert_eq!(parse_size("512M").unwrap(), 512 << 20);
        assert_eq!(parse_size("1.5G").unwrap(), 3 << 29);
        assert_eq!(parse_size("4096").unwrap(), 4096);
        assert_eq!(parse_size("2GiB").unwrap(), 2 << 30);
        assert!(parse_size("-1G").is_err());
        assert!(parse_size("lots").is_err());
    }

    #[test]
    fn rejects_unknown_and_bad_values() {
        assert!(parse(&args("--gpu 0 --frobnicate")).is_err());
        assert!(parse(&args("--gpu 0 --memtest 0")).is_err());
        assert!(parse(&args("--gpu 0 --memtest")).is_err());
        assert!(parse(&args("--gpu 0 --telemetry-ms 5")).is_err());
        assert!(parse(&args("--info")).is_err());
    }

    #[test]
    fn worker_protocol() {
        let o = parse(&args(
            "--bcll-worker cuda-memtest --uuid GPU-1 --pci 0000:01:00.0 --seconds 5 --events e.jsonl --cuda-keep-free-mb 64",
        ))
        .unwrap();
        match o.command {
            Command::Worker(w) => {
                assert_eq!(w.kind, WorkerKind::CudaMemtest);
                assert_eq!(w.uuid.as_deref(), Some("GPU-1"));
                assert_eq!(w.keep_free_mb, 64);
                assert_eq!(w.seconds, 5.0);
            }
            other => panic!("{other:?}"),
        }
        assert!(parse(&args("--bcll-worker compute --seconds 5")).is_err());
    }
}
