//! NVENC/NVDEC functional test through ffmpeg (DESIGN.md section 7.5).
//!
//! Per codec: encode N frames of `testsrc2` 1080p30 with `<codec>_nvenc` on the
//! selected GPU, decode with `<codec>_cuvid` and compare against the regenerated
//! source with the `psnr` filter in one pass. Pass = average PSNR >= 30 dB and
//! decoded frame count == N. The GPU is pinned with CUDA_VISIBLE_DEVICES=<UUID>
//! plus `-gpu 0`, independent of enumeration order.

use super::proc::{self, Limits};
use super::report::{TestResult, TestStatus};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

pub const MIN_PSNR_DB: f64 = 30.0;
const DISCOVERY_TIMEOUT: Duration = Duration::from_secs(30);
pub const CODECS: &[(&str, &str, &str)] = &[
    ("h264", "h264_nvenc", "h264_cuvid"),
    ("hevc", "hevc_nvenc", "hevc_cuvid"),
    ("av1", "av1_nvenc", "av1_cuvid"),
];
const SOURCE: &str = "testsrc2=size=1920x1080:rate=30";

pub fn find_ffmpeg(explicit: Option<&Path>) -> Option<PathBuf> {
    if let Some(p) = explicit {
        return p.is_file().then(|| p.to_path_buf());
    }
    let exe = if cfg!(windows) {
        "ffmpeg.exe"
    } else {
        "ffmpeg"
    };
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|d| d.join(exe))
            .find(|p| p.is_file())
    })
}

pub fn encode_args(encoder: &str, frames: u32, out: &Path) -> Vec<String> {
    [
        "-hide_banner",
        "-nostdin",
        "-y",
        "-f",
        "lavfi",
        "-i",
        SOURCE,
        "-frames:v",
        &frames.to_string(),
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        encoder,
        "-gpu",
        "0",
        "-preset",
        "p4",
        "-rc",
        "vbr",
        "-b:v",
        "20M",
        "-maxrate",
        "30M",
        "-bufsize",
        "40M",
    ]
    .iter()
    .map(|s| s.to_string())
    .chain(std::iter::once(out.to_string_lossy().into_owned()))
    .collect()
}

pub fn decode_psnr_args(decoder: &str, frames: u32, input: &Path) -> Vec<String> {
    let graph = format!(
        "[0:v]format=yuv420p[a];[1:v]trim=end_frame={frames},format=yuv420p[b];[a][b]psnr=shortest=1"
    );
    vec![
        "-hide_banner".into(),
        "-nostdin".into(),
        "-c:v".into(),
        decoder.into(),
        "-gpu".into(),
        "0".into(),
        "-i".into(),
        input.to_string_lossy().into_owned(),
        "-f".into(),
        "lavfi".into(),
        "-i".into(),
        SOURCE.into(),
        "-filter_complex".into(),
        graph,
        "-f".into(),
        "null".into(),
        "-".into(),
    ]
}

/// `average:41.23` from the psnr filter summary; `inf` (identical frames) maps to 100 dB.
pub fn parse_psnr(stderr: &str) -> Option<f64> {
    let line = stderr
        .lines()
        .rev()
        .find(|l| l.contains("PSNR") && l.contains("average:"))?;
    let v = line.split("average:").nth(1)?.split_whitespace().next()?;
    if v.eq_ignore_ascii_case("inf") {
        Some(100.0)
    } else {
        v.parse().ok()
    }
}

/// Last `frame= N` of ffmpeg's progress output (progress lines are \r-separated).
pub fn parse_frames(stderr: &str) -> Option<u64> {
    stderr
        .rsplit("frame=")
        .next()
        .filter(|_| stderr.contains("frame="))?
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

pub fn parse_fps(stderr: &str) -> Option<f64> {
    stderr
        .rsplit("fps=")
        .next()
        .filter(|_| stderr.contains("fps="))?
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

/// True when the failure means "this card/driver/ffmpeg cannot do it", not a fault.
pub fn is_unsupported(stderr: &str) -> bool {
    const MARKERS: &[&str] = &[
        "No capable devices found",
        "Codec not supported",
        "not supported by the device",
        "Unknown encoder",
        "Unknown decoder",
        "Driver does not support the required nvenc API version",
        "Cannot load nvcuda",
        "Cannot load libcuda",
        "Cannot load libnvidia-encode",
        "Cannot load nvEncodeAPI",
        "Unsupported codec",
    ];
    MARKERS.iter().any(|m| stderr.contains(m))
}

fn tail(s: &str, n: usize) -> String {
    let lines: Vec<&str> = s.lines().filter(|l| !l.trim().is_empty()).collect();
    lines[lines.len().saturating_sub(n)..].join(" | ")
}

/// Runs a child under `proc` (bounded wait, kill and pipe drain; contained), capturing
/// stderr. Ok((success, stderr)); Err on spawn failure or timeout (BLOCK 89de0c26 #1).
pub fn run_with_timeout(cmd: Command, timeout: Duration) -> Result<(bool, String), String> {
    let o = proc::run(cmd, &Limits::new(timeout), false, true)?;
    if o.timed_out {
        return Err(format!(
            "timed out after {}s: {}{}",
            timeout.as_secs(),
            tail(&o.stderr, 3),
            o.death_note().map(|n| format!("; {n}")).unwrap_or_default()
        ));
    }
    Ok((o.exit_code == Some(0), o.stderr))
}

fn ffmpeg_cmd(ffmpeg: &Path, uuid: &str, args: &[String]) -> Command {
    let mut c = Command::new(ffmpeg);
    c.args(args)
        .env("CUDA_VISIBLE_DEVICES", uuid)
        .env("CUDA_DEVICE_ORDER", "PCI_BUS_ID");
    c
}

/// Capability discovery; a failing/hanging ffmpeg is an ERROR, not "no codecs" (BLOCK #6).
fn list_codecs(ffmpeg: &Path, flag: &str) -> Result<String, String> {
    let mut c = Command::new(ffmpeg);
    c.args(["-hide_banner", flag]);
    let o = proc::run(c, &Limits::new(DISCOVERY_TIMEOUT), true, true)?;
    discovery_result(&o, flag)
}

/// Pure judgement of a discovery run (unit-tested).
pub fn discovery_result(o: &proc::Outcome, flag: &str) -> Result<String, String> {
    if o.timed_out {
        return Err(format!("ffmpeg {flag} timed out"));
    }
    if o.exit_code != Some(0) {
        return Err(format!(
            "ffmpeg {flag} failed (exit {:?}): {}",
            o.exit_code,
            tail(&o.stderr, 2)
        ));
    }
    if !o.stdout.contains("------") {
        return Err(format!("ffmpeg {flag} produced no codec table"));
    }
    Ok(o.stdout.clone())
}

fn has_codec(listing: &str, name: &str) -> bool {
    listing.split_whitespace().any(|w| w == name)
}

/// Evaluates one codec from the outputs of its encode and decode steps (pure; unit-tested).
fn judge_inner(
    mut t: TestResult,
    frames: u32,
    encode: Result<(bool, String), String>,
    decode: impl FnOnce() -> Result<(bool, String), String>,
) -> TestResult {
    match encode {
        Err(e) => {
            t.status = TestStatus::Incomplete;
            t.message = Some(format!("encode: {e}"));
            return t;
        }
        Ok((false, stderr)) => {
            t.status = if is_unsupported(&stderr) {
                TestStatus::Unsupported
            } else {
                TestStatus::Fail
            };
            t.error_count = (t.status == TestStatus::Fail) as u64;
            t.message = Some(format!("encode failed: {}", tail(&stderr, 3)));
            return t;
        }
        Ok((true, stderr)) => {
            if let Some(fps) = parse_fps(&stderr) {
                t.set_metric("encode_fps", fps);
            }
        }
    }
    match decode() {
        Err(e) => {
            t.status = TestStatus::Incomplete;
            t.message = Some(format!("decode: {e}"));
        }
        Ok((false, stderr)) => {
            t.status = if is_unsupported(&stderr) {
                TestStatus::Unsupported
            } else {
                TestStatus::Fail
            };
            t.error_count = (t.status == TestStatus::Fail) as u64;
            t.message = Some(format!("decode failed: {}", tail(&stderr, 3)));
        }
        Ok((true, stderr)) => {
            let psnr = parse_psnr(&stderr);
            let decoded = parse_frames(&stderr);
            t.set_metric("psnr_avg_db", psnr);
            t.set_metric("frames_encoded", frames);
            t.set_metric("frames_decoded", decoded);
            if let Some(fps) = parse_fps(&stderr) {
                t.set_metric("decode_fps", fps);
            }
            let psnr_ok = psnr.is_some_and(|p| p >= MIN_PSNR_DB);
            let frames_ok = decoded == Some(frames as u64);
            t.status = if psnr_ok && frames_ok {
                TestStatus::Pass
            } else {
                TestStatus::Fail
            };
            if t.status == TestStatus::Fail {
                t.error_count = 1;
                t.message = Some(format!(
                    "round trip mismatch: psnr {:?} dB (min {MIN_PSNR_DB}), frames {:?}/{frames}",
                    psnr, decoded
                ));
            }
        }
    }
    t
}

/// Evaluates one codec from its encode and decode steps and sets the terminal state.
pub fn judge(
    t: TestResult,
    frames: u32,
    encode: Result<(bool, String), String>,
    decode: impl FnOnce() -> Result<(bool, String), String>,
) -> TestResult {
    let mut t = judge_inner(t, frames, encode, decode);
    t.termination = Some(
        match t.status {
            TestStatus::Pass => "completed",
            TestStatus::Fail => "mismatch",
            TestStatus::Unsupported => "unsupported",
            _ if t
                .message
                .as_deref()
                .is_some_and(|m| m.contains("timed out")) =>
            {
                "timed_out"
            }
            _ => "runtime_error",
        }
        .into(),
    );
    t
}

pub fn run(
    ffmpeg: Option<&Path>,
    uuid: Option<&str>,
    frames: u32,
    work_dir: &Path,
) -> Vec<TestResult> {
    let mut results = Vec::new();
    let new = |codec: &str| {
        let mut t = TestResult::new("codec", "ffmpeg", &format!("{codec}_nvenc_cuvid"));
        t.started_at = Some(crate::output::NowTime.to_string());
        t
    };
    let Some(ffmpeg) = find_ffmpeg(ffmpeg) else {
        for (codec, _, _) in CODECS {
            let mut t = new(codec);
            t.status = TestStatus::Error;
            t.message = Some("ffmpeg not found (use --ffmpeg PATH)".into());
            results.push(t);
        }
        return results;
    };
    let Some(uuid) = uuid else {
        for (codec, _, _) in CODECS {
            let mut t = new(codec);
            t.status = TestStatus::Error;
            t.message = Some("GPU UUID unknown (needs NVML or Vulkan deviceUUID) - cannot pin ffmpeg to the card".into());
            results.push(t);
        }
        return results;
    };
    let listings = list_codecs(&ffmpeg, "-encoders")
        .and_then(|e| list_codecs(&ffmpeg, "-decoders").map(|d| (e, d)));
    let (encoders, decoders) = match listings {
        Ok(v) => v,
        Err(e) => {
            for (codec, _, _) in CODECS {
                let mut t = new(codec);
                t.status = TestStatus::Error;
                t.termination = Some("runtime_error".into());
                t.message = Some(format!("capability discovery failed: {e}"));
                results.push(t);
            }
            return results;
        }
    };
    let timeout = Duration::from_secs(60 + frames as u64 / 10);
    for (codec, enc, dec) in CODECS {
        let mut t = new(codec);
        t.set_metric("ffmpeg", ffmpeg.to_string_lossy());
        if !has_codec(&encoders, enc) || !has_codec(&decoders, dec) {
            t.status = TestStatus::Unsupported;
            t.message = Some(format!("this ffmpeg build lacks {enc} or {dec}"));
            results.push(t);
            continue;
        }
        let start = Instant::now();
        let out = work_dir.join(format!("bcll-codec-{codec}-{}.mkv", std::process::id()));
        let encode = run_with_timeout(
            ffmpeg_cmd(&ffmpeg, uuid, &encode_args(enc, frames, &out)),
            timeout,
        );
        let mut t = judge(t, frames, encode, || {
            run_with_timeout(
                ffmpeg_cmd(&ffmpeg, uuid, &decode_psnr_args(dec, frames, &out)),
                timeout,
            )
        });
        if let Ok(meta) = std::fs::metadata(&out) {
            t.set_metric("encoded_bytes", meta.len());
        }
        let _ = std::fs::remove_file(&out);
        t.duration_s = Some(start.elapsed().as_secs_f64());
        println!(
            "[codec] {codec}: {:?}{}",
            t.status,
            t.message
                .as_ref()
                .map(|m| format!(" - {m}"))
                .unwrap_or_default()
        );
        results.push(t);
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    const DECODE_OK: &str = "Input #0, matroska\nframe=  150 fps=410 q=-0.0 Lsize=N/A time=00:00:05.00\r\
[Parsed_psnr_4 @ 000001] PSNR y:43.21 u:47.10 v:46.88 average:44.12 min:41.03 max:47.55\n";

    fn t() -> TestResult {
        TestResult::new("codec", "ffmpeg", "h264_nvenc_cuvid")
    }

    #[test]
    fn parses_psnr_frames_fps() {
        assert_eq!(parse_psnr(DECODE_OK), Some(44.12));
        assert_eq!(parse_frames(DECODE_OK), Some(150));
        assert_eq!(parse_fps(DECODE_OK), Some(410.0));
        assert_eq!(
            parse_psnr("PSNR y:inf u:inf v:inf average:inf min:inf max:inf"),
            Some(100.0)
        );
        assert_eq!(parse_psnr("no summary"), None);
        assert_eq!(parse_frames("nothing"), None);
    }

    #[test]
    fn progress_with_carriage_returns_uses_last_frame() {
        let s = "frame=   10 fps=0.0\rframe=   90 fps=200\rframe=  149 fps=300\n";
        assert_eq!(parse_frames(s), Some(149));
    }

    #[test]
    fn pass_needs_psnr_and_all_frames() {
        let ok = judge(t(), 150, Ok((true, "frame=150 fps=900".into())), || {
            Ok((true, DECODE_OK.into()))
        });
        assert_eq!(ok.status, TestStatus::Pass);
        assert_eq!(ok.metrics["psnr_avg_db"], 44.12);

        let short = judge(t(), 151, Ok((true, String::new())), || {
            Ok((true, DECODE_OK.into()))
        });
        assert_eq!(short.status, TestStatus::Fail);

        let garbled = DECODE_OK.replace("average:44.12", "average:12.50");
        let bad = judge(t(), 150, Ok((true, String::new())), || Ok((true, garbled)));
        assert_eq!(bad.status, TestStatus::Fail);
        assert!(bad.message.unwrap().contains("psnr"));
    }

    #[test]
    fn unsupported_encode_is_not_a_failure() {
        let r = judge(
            t(),
            150,
            Ok((
                false,
                "[av1_nvenc @ 0x1] No capable devices found\nError".into(),
            )),
            || unreachable!(),
        );
        assert_eq!(r.status, TestStatus::Unsupported);
        assert_eq!(r.error_count, 0);
        assert_eq!(r.termination.as_deref(), Some("unsupported"));
    }

    #[test]
    fn unexplained_encode_error_fails_and_timeout_is_incomplete() {
        let r = judge(
            t(),
            150,
            Ok((false, "Segmentation fault".into())),
            || unreachable!(),
        );
        assert_eq!(r.status, TestStatus::Fail);
        let r = judge(
            t(),
            150,
            Err("timed out after 75s".into()),
            || unreachable!(),
        );
        assert_eq!(r.status, TestStatus::Incomplete);
    }

    #[test]
    fn args_pin_gpu_zero_and_bound_the_reference() {
        let e = encode_args("hevc_nvenc", 150, Path::new("o.mkv"));
        assert!(e.windows(2).any(|w| w[0] == "-gpu" && w[1] == "0"));
        assert!(e.windows(2).any(|w| w[0] == "-frames:v" && w[1] == "150"));
        assert_eq!(e.last().unwrap(), "o.mkv");
        let d = decode_psnr_args("hevc_cuvid", 150, Path::new("o.mkv"));
        let graph = &d[d.iter().position(|a| a == "-filter_complex").unwrap() + 1];
        assert!(graph.contains("trim=end_frame=150"));
        assert!(graph.contains("psnr=shortest=1"));
    }

    #[test]
    fn discovery_failure_is_distinguished_from_unsupported() {
        let ok = proc::Outcome {
            exit_code: Some(0),
            stdout: " ------\n V....D h264_nvenc NVIDIA NVENC\n".into(),
            dead_confirmed: true,
            pipes_complete: true,
            ..Default::default()
        };
        assert!(discovery_result(&ok, "-encoders").is_ok());
        let failed = proc::Outcome {
            exit_code: Some(1),
            stderr: "dll missing".into(),
            ..ok.clone()
        };
        assert!(
            discovery_result(&failed, "-encoders")
                .unwrap_err()
                .contains("failed")
        );
        let hung = proc::Outcome {
            timed_out: true,
            exit_code: None,
            ..ok.clone()
        };
        assert!(
            discovery_result(&hung, "-encoders")
                .unwrap_err()
                .contains("timed out")
        );
        let empty = proc::Outcome {
            stdout: String::new(),
            ..ok
        };
        assert!(
            discovery_result(&empty, "-encoders")
                .unwrap_err()
                .contains("no codec table")
        );
    }

    #[test]
    fn codec_listing_match_is_whole_word() {
        let listing = " V....D h264_nvenc           NVIDIA NVENC H.264 encoder\n";
        assert!(has_codec(listing, "h264_nvenc"));
        assert!(!has_codec(listing, "h264_nv"));
    }
}
