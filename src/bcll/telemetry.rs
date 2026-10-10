//! Telemetry samples, health snapshots, and the pure logic turning them into
//! report summaries and verdict inputs (docs/bcll/DESIGN.md sections 5, 6, 8).
//! The sampling thread is generic over `Sampler` so this file needs no NVML.

use super::aer::{self, AerSnapshot};
use super::report::HealthFindings;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

pub const MAX_SAMPLES: usize = 100_000;
pub const MAX_LINK_CHANGES: usize = 200;
pub const BUSY_UTIL_PCT: u32 = 50;

/// Counters that only grow; compared before vs after a run.
#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq)]
pub struct HealthSnapshot {
    pub taken_at: String,
    pub ecc_mode_current: Option<bool>,
    pub ecc_mode_pending: Option<bool>,
    pub ecc_volatile_corrected: Option<u64>,
    pub ecc_volatile_uncorrected: Option<u64>,
    pub ecc_aggregate_corrected: Option<u64>,
    pub ecc_aggregate_uncorrected: Option<u64>,
    pub ecc_dram_volatile_corrected: Option<u64>,
    pub ecc_dram_volatile_uncorrected: Option<u64>,
    pub ecc_dram_aggregate_corrected: Option<u64>,
    pub ecc_dram_aggregate_uncorrected: Option<u64>,
    pub ecc_sram_volatile_corrected: Option<u64>,
    pub ecc_sram_volatile_uncorrected: Option<u64>,
    pub ecc_sram_aggregate_corrected: Option<u64>,
    pub ecc_sram_aggregate_uncorrected: Option<u64>,
    pub remapped_rows_correctable: Option<u64>,
    pub remapped_rows_uncorrectable: Option<u64>,
    pub remap_pending: Option<bool>,
    pub remap_failure: Option<bool>,
    /// Bank availability histogram: max/high/partial/low/none.
    pub remap_histogram: Option<BTreeMap<String, u64>>,
    pub retired_pages_single_bit: Option<u64>,
    pub retired_pages_double_bit: Option<u64>,
    pub retired_pages_pending: Option<bool>,
    pub retired_page_addresses: Vec<u64>,
    /// NVML PCIe field counters (replay, L0->recovery, NAK, bad TLP/DLLP, ...); null = unsupported.
    pub pcie_counters: BTreeMap<String, Option<u64>>,
    pub aer: Option<AerSnapshot>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq)]
pub struct Sample {
    pub t_s: f64,
    pub temp_gpu_c: Option<u32>,
    /// NVAPI (undocumented source), Windows only.
    pub temp_hotspot_c: Option<f32>,
    /// NVAPI (undocumented source) on GeForce; NVML field 82 where supported.
    pub temp_mem_junction_c: Option<f32>,
    pub power_w: Option<f64>,
    pub clock_sm_mhz: Option<u32>,
    pub clock_mem_mhz: Option<u32>,
    pub util_gpu_pct: Option<u32>,
    pub util_mem_pct: Option<u32>,
    pub util_enc_pct: Option<u32>,
    pub util_dec_pct: Option<u32>,
    pub pcie_gen: Option<u32>,
    pub pcie_width: Option<u32>,
    pub pcie_tx_kbs: Option<u32>,
    pub pcie_rx_kbs: Option<u32>,
    pub pcie_replay: Option<u64>,
    pub ecc_volatile_corrected: Option<u64>,
    pub ecc_volatile_uncorrected: Option<u64>,
    pub remapped_rows_uncorrectable: Option<u64>,
    pub throttle_bits: Option<u64>,
    pub fan_pct: Option<u32>,
    pub aer_correctable: Option<u64>,
    pub aer_nonfatal: Option<u64>,
    pub aer_fatal: Option<u64>,
}

/// NVML clocks-event (throttle) reason bits.
pub const THROTTLE_NAMES: &[(u64, &str)] = &[
    (0x1, "gpu_idle"),
    (0x2, "applications_clocks_setting"),
    (0x4, "sw_power_cap"),
    (0x8, "hw_slowdown"),
    (0x10, "sync_boost"),
    (0x20, "sw_thermal_slowdown"),
    (0x40, "hw_thermal_slowdown"),
    (0x80, "hw_power_brake_slowdown"),
    (0x100, "display_clock_setting"),
];
pub const HW_SLOWDOWN_MASK: u64 = 0x8 | 0x40 | 0x80;

pub fn throttle_names(bits: u64) -> Vec<&'static str> {
    THROTTLE_NAMES
        .iter()
        .filter(|(b, _)| bits & b != 0)
        .map(|(_, n)| *n)
        .collect()
}

#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct LinkChange {
    pub t_s: f64,
    pub from: String,
    pub to: String,
}

#[derive(Serialize, Debug, Clone, Default, PartialEq)]
pub struct Summary {
    pub samples: usize,
    pub duration_s: f64,
    pub temp_gpu_max_c: Option<u32>,
    pub temp_hotspot_max_c: Option<f32>,
    pub temp_mem_junction_max_c: Option<f32>,
    pub power_w_max: Option<f64>,
    pub power_w_avg: Option<f64>,
    pub clock_sm_mhz_min_busy: Option<u32>,
    pub clock_sm_mhz_max: Option<u32>,
    pub clock_mem_mhz_min_busy: Option<u32>,
    pub clock_mem_mhz_max: Option<u32>,
    pub pcie_gen_min_busy: Option<u32>,
    pub pcie_width_min_busy: Option<u32>,
    pub pcie_gen_max_seen: Option<u32>,
    pub pcie_width_max_seen: Option<u32>,
    pub pcie_tx_kbs_max: Option<u32>,
    pub pcie_rx_kbs_max: Option<u32>,
    pub link_changes: Vec<LinkChange>,
    pub pcie_replay_delta: Option<u64>,
    pub ecc_corrected_delta: Option<u64>,
    pub ecc_uncorrected_delta: Option<u64>,
    pub remapped_uncorrectable_delta: Option<u64>,
    pub aer_correctable_delta: Option<u64>,
    pub aer_nonfatal_delta: Option<u64>,
    pub aer_fatal_delta: Option<u64>,
    /// reason -> number of samples it was active in
    pub throttle_reasons_seen: BTreeMap<String, usize>,
}

fn max_opt<T: PartialOrd + Copy>(it: impl Iterator<Item = Option<T>>) -> Option<T> {
    it.flatten().fold(None, |acc, v| match acc {
        Some(a) if a >= v => Some(a),
        _ => Some(v),
    })
}

fn min_opt<T: PartialOrd + Copy>(it: impl Iterator<Item = Option<T>>) -> Option<T> {
    it.flatten().fold(None, |acc, v| match acc {
        Some(a) if a <= v => Some(a),
        _ => Some(v),
    })
}

/// Last minus first of a monotonically growing counter (saturating; None without two readings).
fn counter_delta(values: impl Iterator<Item = Option<u64>>) -> Option<u64> {
    let present: Vec<u64> = values.flatten().collect();
    match (present.first(), present.last()) {
        (Some(a), Some(b)) if present.len() >= 2 => Some(b.saturating_sub(*a)),
        _ => None,
    }
}

fn is_busy(s: &Sample) -> bool {
    s.util_gpu_pct.is_some_and(|u| u >= BUSY_UTIL_PCT)
}

pub fn summarize(samples: &[Sample]) -> Summary {
    let busy: Vec<&Sample> = samples.iter().filter(|s| is_busy(s)).collect();
    let powers: Vec<f64> = samples.iter().filter_map(|s| s.power_w).collect();
    let mut link_changes = Vec::new();
    let mut prev: Option<(u32, u32)> = None;
    for s in samples {
        if let (Some(g), Some(w)) = (s.pcie_gen, s.pcie_width) {
            if let Some((pg, pw)) = prev
                && (pg, pw) != (g, w)
                && link_changes.len() < MAX_LINK_CHANGES
            {
                link_changes.push(LinkChange {
                    t_s: s.t_s,
                    from: format!("gen{pg} x{pw}"),
                    to: format!("gen{g} x{w}"),
                });
            }
            prev = Some((g, w));
        }
    }
    let mut throttle_reasons_seen = BTreeMap::new();
    for bits in samples.iter().filter_map(|s| s.throttle_bits) {
        for name in throttle_names(bits) {
            *throttle_reasons_seen.entry(name.to_string()).or_insert(0) += 1;
        }
    }
    Summary {
        samples: samples.len(),
        duration_s: match (samples.first(), samples.last()) {
            (Some(a), Some(b)) => b.t_s - a.t_s,
            _ => 0.0,
        },
        temp_gpu_max_c: max_opt(samples.iter().map(|s| s.temp_gpu_c)),
        temp_hotspot_max_c: max_opt(samples.iter().map(|s| s.temp_hotspot_c)),
        temp_mem_junction_max_c: max_opt(samples.iter().map(|s| s.temp_mem_junction_c)),
        power_w_max: max_opt(samples.iter().map(|s| s.power_w)),
        power_w_avg: (!powers.is_empty()).then(|| powers.iter().sum::<f64>() / powers.len() as f64),
        clock_sm_mhz_min_busy: min_opt(busy.iter().map(|s| s.clock_sm_mhz)),
        clock_sm_mhz_max: max_opt(samples.iter().map(|s| s.clock_sm_mhz)),
        clock_mem_mhz_min_busy: min_opt(busy.iter().map(|s| s.clock_mem_mhz)),
        clock_mem_mhz_max: max_opt(samples.iter().map(|s| s.clock_mem_mhz)),
        pcie_gen_min_busy: min_opt(busy.iter().map(|s| s.pcie_gen)),
        pcie_width_min_busy: min_opt(busy.iter().map(|s| s.pcie_width)),
        pcie_gen_max_seen: max_opt(samples.iter().map(|s| s.pcie_gen)),
        pcie_width_max_seen: max_opt(samples.iter().map(|s| s.pcie_width)),
        pcie_tx_kbs_max: max_opt(samples.iter().map(|s| s.pcie_tx_kbs)),
        pcie_rx_kbs_max: max_opt(samples.iter().map(|s| s.pcie_rx_kbs)),
        link_changes,
        pcie_replay_delta: counter_delta(samples.iter().map(|s| s.pcie_replay)),
        ecc_corrected_delta: counter_delta(samples.iter().map(|s| s.ecc_volatile_corrected)),
        ecc_uncorrected_delta: counter_delta(samples.iter().map(|s| s.ecc_volatile_uncorrected)),
        remapped_uncorrectable_delta: counter_delta(
            samples.iter().map(|s| s.remapped_rows_uncorrectable),
        ),
        aer_correctable_delta: counter_delta(samples.iter().map(|s| s.aer_correctable)),
        aer_nonfatal_delta: counter_delta(samples.iter().map(|s| s.aer_nonfatal)),
        aer_fatal_delta: counter_delta(samples.iter().map(|s| s.aer_fatal)),
        throttle_reasons_seen,
    }
}

fn delta_opt(before: Option<u64>, after: Option<u64>) -> Option<u64> {
    match (before, after) {
        (Some(b), Some(a)) => Some(a.saturating_sub(b)),
        _ => None,
    }
}

/// Field-by-field increase of the growing counters between two snapshots.
pub fn health_delta(before: &HealthSnapshot, after: &HealthSnapshot) -> BTreeMap<String, u64> {
    let mut out = BTreeMap::new();
    let pairs: [(&str, Option<u64>, Option<u64>); 14] = [
        (
            "ecc_volatile_corrected",
            before.ecc_volatile_corrected,
            after.ecc_volatile_corrected,
        ),
        (
            "ecc_volatile_uncorrected",
            before.ecc_volatile_uncorrected,
            after.ecc_volatile_uncorrected,
        ),
        (
            "ecc_aggregate_corrected",
            before.ecc_aggregate_corrected,
            after.ecc_aggregate_corrected,
        ),
        (
            "ecc_aggregate_uncorrected",
            before.ecc_aggregate_uncorrected,
            after.ecc_aggregate_uncorrected,
        ),
        (
            "ecc_dram_volatile_corrected",
            before.ecc_dram_volatile_corrected,
            after.ecc_dram_volatile_corrected,
        ),
        (
            "ecc_dram_volatile_uncorrected",
            before.ecc_dram_volatile_uncorrected,
            after.ecc_dram_volatile_uncorrected,
        ),
        (
            "ecc_sram_volatile_corrected",
            before.ecc_sram_volatile_corrected,
            after.ecc_sram_volatile_corrected,
        ),
        (
            "ecc_sram_volatile_uncorrected",
            before.ecc_sram_volatile_uncorrected,
            after.ecc_sram_volatile_uncorrected,
        ),
        (
            "remapped_rows_correctable",
            before.remapped_rows_correctable,
            after.remapped_rows_correctable,
        ),
        (
            "remapped_rows_uncorrectable",
            before.remapped_rows_uncorrectable,
            after.remapped_rows_uncorrectable,
        ),
        (
            "retired_pages_single_bit",
            before.retired_pages_single_bit,
            after.retired_pages_single_bit,
        ),
        (
            "retired_pages_double_bit",
            before.retired_pages_double_bit,
            after.retired_pages_double_bit,
        ),
        (
            "aer_correctable",
            before.aer.as_ref().map(|a| a.total_correctable()),
            after.aer.as_ref().map(|a| a.total_correctable()),
        ),
        (
            "aer_nonfatal_fatal",
            before
                .aer
                .as_ref()
                .map(|a| a.total_nonfatal() + a.total_fatal()),
            after
                .aer
                .as_ref()
                .map(|a| a.total_nonfatal() + a.total_fatal()),
        ),
    ];
    for (k, b, a) in pairs {
        if let Some(d) = delta_opt(b, a).filter(|d| *d > 0) {
            out.insert(k.to_string(), d);
        }
    }
    for (k, a) in &after.pcie_counters {
        let b = before.pcie_counters.get(k).copied().flatten();
        if let Some(d) = delta_opt(b, *a).filter(|d| *d > 0) {
            out.insert(format!("pcie.{k}"), d);
        }
    }
    if let (Some(b), Some(a)) = (&before.aer, &after.aer) {
        for (k, v) in aer::delta(b, a) {
            out.insert(format!("aer.{k}"), v);
        }
    }
    out
}

/// Verdict inputs: snapshot deltas take precedence; telemetry deltas fill gaps.
pub fn findings(
    before: Option<&HealthSnapshot>,
    after: Option<&HealthSnapshot>,
    summary: Option<&Summary>,
    max_link: Option<(u32, u32)>,
) -> HealthFindings {
    let snap = |f: fn(&HealthSnapshot) -> Option<u64>| match (before, after) {
        (Some(b), Some(a)) => delta_opt(f(b), f(a)),
        _ => None,
    };
    let either = |a: Option<u64>, b: Option<u64>| match (a, b) {
        (Some(x), Some(y)) => Some(x.max(y)),
        (x, y) => x.or(y),
    };
    let mut h = HealthFindings {
        ecc_uncorrected_delta: either(
            snap(|s| s.ecc_volatile_uncorrected),
            summary.and_then(|s| s.ecc_uncorrected_delta),
        ),
        ecc_corrected_delta: either(
            snap(|s| s.ecc_volatile_corrected),
            summary.and_then(|s| s.ecc_corrected_delta),
        ),
        remapped_uncorrectable_delta: either(
            snap(|s| s.remapped_rows_uncorrectable),
            summary.and_then(|s| s.remapped_uncorrectable_delta),
        ),
        retired_pages_delta: either(
            snap(|s| s.retired_pages_single_bit),
            snap(|s| s.retired_pages_double_bit),
        ),
        pcie_replay_delta: either(
            snap(|s| s.pcie_counters.get("replay").copied().flatten()),
            summary.and_then(|s| s.pcie_replay_delta),
        ),
        aer_fatal_or_nonfatal_delta: either(
            snap(|s| s.aer.as_ref().map(|a| a.total_nonfatal() + a.total_fatal())),
            summary.and_then(|s| match (s.aer_nonfatal_delta, s.aer_fatal_delta) {
                (None, None) => None,
                (a, b) => Some(a.unwrap_or(0) + b.unwrap_or(0)),
            }),
        ),
        aer_correctable_delta: either(
            snap(|s| s.aer.as_ref().map(|a| a.total_correctable())),
            summary.and_then(|s| s.aer_correctable_delta),
        ),
        ..Default::default()
    };
    if let Some(a) = after {
        h.remap_failure = a.remap_failure == Some(true);
        h.remap_pending = a.remap_pending == Some(true);
    }
    if let Some(s) = summary {
        h.hw_slowdown_seen = THROTTLE_NAMES
            .iter()
            .filter(|(bit, _)| bit & HW_SLOWDOWN_MASK != 0)
            .any(|(_, name)| s.throttle_reasons_seen.contains_key(*name));
        h.max_memory_junction_c = s.temp_mem_junction_max_c.map(f64::from);
        if let (Some((max_gen, max_width)), Some(g), Some(w)) =
            (max_link, s.pcie_gen_min_busy, s.pcie_width_min_busy)
            && (g < max_gen || w < max_width)
        {
            h.link_below_max_while_busy =
                Some(format!("gen{g} x{w} (max gen{max_gen} x{max_width})"));
        }
    }
    h
}

/// One compact console line; counters shown as deltas against `first`.
pub fn format_line(s: &Sample, first: &Sample) -> String {
    let o = |v: Option<u32>| v.map(|x| x.to_string()).unwrap_or_else(|| "-".into());
    let of = |v: Option<f32>| v.map(|x| format!("{x:.0}")).unwrap_or_else(|| "-".into());
    let d = |a: Option<u64>, b: Option<u64>| match (a, b) {
        (Some(a), Some(b)) => format!("+{}", a.saturating_sub(b)),
        _ => "-".into(),
    };
    format!(
        "[telemetry {:>7.1}s] {}C hot {}C mem {}C | {}W | sm {} mem {} MHz | util {}% | PCIe gen{} x{} tx {} rx {} MB/s replay {} | ECC c{} u{} | throttle {}",
        s.t_s,
        o(s.temp_gpu_c),
        of(s.temp_hotspot_c),
        of(s.temp_mem_junction_c),
        s.power_w
            .map(|p| format!("{p:.0}"))
            .unwrap_or_else(|| "-".into()),
        o(s.clock_sm_mhz),
        o(s.clock_mem_mhz),
        o(s.util_gpu_pct),
        o(s.pcie_gen),
        o(s.pcie_width),
        o(s.pcie_tx_kbs.map(|k| k / 1024)),
        o(s.pcie_rx_kbs.map(|k| k / 1024)),
        d(s.pcie_replay, first.pcie_replay),
        d(s.ecc_volatile_corrected, first.ecc_volatile_corrected),
        d(s.ecc_volatile_uncorrected, first.ecc_volatile_uncorrected),
        s.throttle_bits
            .map(|b| {
                let n = throttle_names(b & !0x1);
                if n.is_empty() {
                    "none".to_string()
                } else {
                    n.join(",")
                }
            })
            .unwrap_or_else(|| "-".into()),
    )
}

pub trait Sampler {
    fn sample(&mut self, t_s: f64) -> Sample;
}

pub struct Handle {
    stop: Arc<AtomicBool>,
    samples: Arc<Mutex<Vec<Sample>>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Handle {
    pub fn snapshot(&self) -> Vec<Sample> {
        self.samples
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    pub fn stop(mut self) -> Vec<Sample> {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        self.snapshot()
    }
}

/// Starts the sampling thread. `make` runs on that thread (NVML/NVAPI handles stay thread-local).
pub fn start<F>(interval: Duration, print_every: Duration, make: F) -> Handle
where
    F: FnOnce() -> Option<Box<dyn Sampler>> + Send + 'static,
{
    let stop = Arc::new(AtomicBool::new(false));
    let samples = Arc::new(Mutex::new(Vec::new()));
    let (stop_t, samples_t) = (stop.clone(), samples.clone());
    let thread = std::thread::Builder::new()
        .name("bcll-telemetry".into())
        .spawn(move || {
            let Some(mut sampler) = make() else { return };
            let start = Instant::now();
            let mut last_print: Option<Instant> = None;
            let mut first: Option<Sample> = None;
            while !stop_t.load(Ordering::SeqCst) {
                let tick = Instant::now();
                let s = sampler.sample(start.elapsed().as_secs_f64());
                let first_ref = first.get_or_insert_with(|| s.clone()).clone();
                if last_print.is_none_or(|p| p.elapsed() >= print_every) {
                    println!("{}", format_line(&s, &first_ref));
                    last_print = Some(Instant::now());
                }
                {
                    let mut v = samples_t.lock().unwrap_or_else(|e| e.into_inner());
                    if v.len() < MAX_SAMPLES {
                        v.push(s);
                    }
                }
                // sleep in small steps so stop() returns promptly
                while tick.elapsed() < interval && !stop_t.load(Ordering::SeqCst) {
                    std::thread::sleep(Duration::from_millis(50).min(interval));
                }
            }
        })
        .ok();
    Handle {
        stop,
        samples,
        thread,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(t: f64, util: u32, gen_: u32, width: u32) -> Sample {
        Sample {
            t_s: t,
            util_gpu_pct: Some(util),
            pcie_gen: Some(gen_),
            pcie_width: Some(width),
            ..Default::default()
        }
    }

    #[test]
    fn idle_link_downshift_is_not_a_busy_downgrade() {
        // idle at gen1 (power saving) then busy at gen4 x16
        let v = vec![s(0.0, 0, 1, 16), s(1.0, 99, 4, 16), s(2.0, 98, 4, 16)];
        let sum = summarize(&v);
        assert_eq!(sum.pcie_gen_min_busy, Some(4));
        assert_eq!(sum.link_changes.len(), 1);
        let h = findings(None, None, Some(&sum), Some((4, 16)));
        assert!(h.link_below_max_while_busy.is_none());
    }

    #[test]
    fn busy_width_drop_is_flagged() {
        let v = vec![s(0.0, 99, 4, 16), s(1.0, 99, 4, 8)];
        let sum = summarize(&v);
        let h = findings(None, None, Some(&sum), Some((4, 16)));
        assert_eq!(
            h.link_below_max_while_busy.as_deref(),
            Some("gen4 x8 (max gen4 x16)")
        );
    }

    #[test]
    fn counter_deltas_and_throttle_counts() {
        let mut a = s(0.0, 99, 4, 16);
        a.pcie_replay = Some(10);
        a.ecc_volatile_uncorrected = Some(0);
        a.throttle_bits = Some(0x4);
        let mut b = s(1.0, 99, 4, 16);
        b.pcie_replay = Some(13);
        b.ecc_volatile_uncorrected = Some(1);
        b.throttle_bits = Some(0x4 | 0x40);
        let sum = summarize(&[a, b]);
        assert_eq!(sum.pcie_replay_delta, Some(3));
        assert_eq!(sum.ecc_uncorrected_delta, Some(1));
        assert_eq!(sum.throttle_reasons_seen["sw_power_cap"], 2);
        assert_eq!(sum.throttle_reasons_seen["hw_thermal_slowdown"], 1);
        let h = findings(None, None, Some(&sum), None);
        assert!(h.hw_slowdown_seen);
        assert_eq!(h.ecc_uncorrected_delta, Some(1));
    }

    #[test]
    fn single_reading_gives_no_delta() {
        let mut a = s(0.0, 0, 1, 16);
        a.pcie_replay = Some(5);
        assert_eq!(summarize(&[a]).pcie_replay_delta, None);
    }

    #[test]
    fn snapshot_delta_wins_and_fills_findings() {
        let before = HealthSnapshot {
            ecc_volatile_uncorrected: Some(8),
            remapped_rows_uncorrectable: Some(7),
            pcie_counters: [("replay".to_string(), Some(100))].into_iter().collect(),
            ..Default::default()
        };
        let after = HealthSnapshot {
            ecc_volatile_uncorrected: Some(9),
            remapped_rows_uncorrectable: Some(8),
            remap_pending: Some(true),
            pcie_counters: [("replay".to_string(), Some(104))].into_iter().collect(),
            ..Default::default()
        };
        let d = health_delta(&before, &after);
        assert_eq!(d["ecc_volatile_uncorrected"], 1);
        assert_eq!(d["remapped_rows_uncorrectable"], 1);
        assert_eq!(d["pcie.replay"], 4);
        let h = findings(Some(&before), Some(&after), None, None);
        assert_eq!(h.ecc_uncorrected_delta, Some(1));
        assert_eq!(h.remapped_uncorrectable_delta, Some(1));
        assert_eq!(h.pcie_replay_delta, Some(4));
        assert!(h.remap_pending);
    }

    #[test]
    fn line_formats_without_data() {
        let line = format_line(&Sample::default(), &Sample::default());
        assert!(line.starts_with("[telemetry"));
        assert!(line.contains("throttle -"));
    }

    struct Fake(u32);
    impl Sampler for Fake {
        fn sample(&mut self, t_s: f64) -> Sample {
            self.0 += 1;
            Sample {
                t_s,
                pcie_replay: Some(self.0 as u64),
                ..Default::default()
            }
        }
    }

    #[test]
    fn thread_collects_and_stops() {
        let h = start(
            Duration::from_millis(100),
            Duration::from_secs(3600),
            || Some(Box::new(Fake(0)) as Box<dyn Sampler>),
        );
        std::thread::sleep(Duration::from_millis(350));
        let v = h.stop();
        assert!(v.len() >= 2, "{}", v.len());
        assert!(v.windows(2).all(|w| w[1].t_s >= w[0].t_s));
    }
}
