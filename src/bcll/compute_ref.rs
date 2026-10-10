//! CPU references and verifiers for the Vulkan compute and PCIe-transfer tests
//! (docs/bcll/DESIGN.md sections 7.2, 7.3).
//!
//! The compute kernel (`vk_diag.rs`, WGSL) must stay in lock-step with
//! `int_chain` / `fp_chain` below. Every invocation `gid` solves problem
//! `p = gid % problems`, so each problem is computed by many invocations spread
//! over all SMs:
//! * integer lanes (x, acc) are checked bit-exactly against the CPU;
//! * FP32 lanes (g, h) are checked by exact consensus among replicas (immune to
//!   the driver's FMA contraction / sqrt precision choices, which Vulkan allows)
//!   and against the CPU with a relative tolerance (catches systematic error).
//!   The FP recurrence is a contraction (factor 0.999 / 0.5) so rounding
//!   differences do not grow chaotically.

use std::collections::HashMap;

pub const FP_REL_TOLERANCE: f32 = 1e-3;

pub fn int_seed(p: u32, seed: u32) -> u32 {
    (p.wrapping_mul(0x9E37_79B9).wrapping_add(seed)) | 1
}

/// Returns (x, acc) after `iters` steps. Mirrors WGSL `stress()`.
pub fn int_chain(p: u32, seed: u32, iters: u32) -> (u32, u32) {
    let mut x = int_seed(p, seed);
    let mut acc = seed ^ 0x85EB_CA6B;
    for i in 0..iters {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        acc = acc.wrapping_mul(1_664_525).wrapping_add(x ^ i);
        acc = acc.rotate_left(7);
    }
    (x, acc)
}

/// Returns (g, h) after `iters` steps. Mirrors WGSL `stress()`.
pub fn fp_chain(p: u32, seed: u32, iters: u32) -> (f32, f32) {
    let mut x = int_seed(p, seed);
    let mut g = 0.0f32;
    let mut h = 0.0f32;
    for _ in 0..iters {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        let u = (x >> 8) as f32 * (1.0 / 16_777_216.0);
        g = g.mul_add(0.999, u.sqrt());
        h = u.mul_add(u, h * 0.5);
    }
    (g, h)
}

#[derive(Debug, Clone, PartialEq)]
pub struct ComputeError {
    pub invocation: u32,
    pub problem: u32,
    /// 0 = x, 1 = acc, 2 = g (f32 bits), 3 = h (f32 bits)
    pub lane: u32,
    pub expected: u32,
    pub actual: u32,
    /// "int_exact" | "fp_consensus" | "fp_no_consensus" | "fp_reference"
    pub check: &'static str,
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct ComputeOutcome {
    pub int_errors: u64,
    pub fp_consensus_errors: u64,
    pub fp_reference_errors: u64,
    pub records: Vec<ComputeError>,
}

impl ComputeOutcome {
    pub fn total(&self) -> u64 {
        self.int_errors + self.fp_consensus_errors + self.fp_reference_errors
    }
    fn record(&mut self, e: ComputeError, max: usize) {
        if self.records.len() < max {
            self.records.push(e);
        }
    }
}

fn rel_close(a: f32, b: f32) -> bool {
    if !a.is_finite() || !b.is_finite() {
        return false;
    }
    (a - b).abs() <= FP_REL_TOLERANCE * a.abs().max(b.abs()).max(1e-3)
}

/// Verifies one dispatch output (`out[gid] = [x, acc, g_bits, h_bits]`).
pub fn verify(
    out: &[[u32; 4]],
    problems: u32,
    seed: u32,
    iters: u32,
    max_records: usize,
) -> ComputeOutcome {
    let mut res = ComputeOutcome::default();
    let p_count = problems.max(1) as usize;
    let int_ref: Vec<(u32, u32)> = (0..p_count as u32)
        .map(|p| int_chain(p, seed, iters))
        .collect();

    // integer lanes: exact
    for (gid, v) in out.iter().enumerate() {
        let p = gid % p_count;
        for (lane, exp) in [(0u32, int_ref[p].0), (1, int_ref[p].1)] {
            let act = v[lane as usize];
            if act != exp {
                res.int_errors += 1;
                res.record(
                    ComputeError {
                        invocation: gid as u32,
                        problem: p as u32,
                        lane,
                        expected: exp,
                        actual: act,
                        check: "int_exact",
                    },
                    max_records,
                );
            }
        }
    }

    // FP lanes: consensus per problem, then the majority vs CPU with tolerance
    for lane in [2usize, 3] {
        let mut per_problem: Vec<HashMap<u32, u32>> = vec![HashMap::new(); p_count];
        for (gid, v) in out.iter().enumerate() {
            *per_problem[gid % p_count].entry(v[lane]).or_insert(0) += 1;
        }
        let mut majority: Vec<Option<u32>> = Vec::with_capacity(p_count);
        for (p, counts) in per_problem.iter().enumerate() {
            let replicas: u32 = counts.values().sum();
            let best = counts
                .iter()
                .max_by_key(|(bits, n)| (**n, std::cmp::Reverse(**bits)));
            let m = best
                .filter(|(_, n)| **n * 2 > replicas)
                .map(|(bits, _)| *bits);
            if let Some(bits) = m {
                let (g, h) = fp_chain(p as u32, seed, iters);
                let cpu = if lane == 2 { g } else { h };
                if !rel_close(f32::from_bits(bits), cpu) {
                    res.fp_reference_errors += 1;
                    res.record(
                        ComputeError {
                            invocation: p as u32,
                            problem: p as u32,
                            lane: lane as u32,
                            expected: cpu.to_bits(),
                            actual: bits,
                            check: "fp_reference",
                        },
                        max_records,
                    );
                }
            }
            majority.push(m);
        }
        for (gid, v) in out.iter().enumerate() {
            let p = gid % p_count;
            match majority[p] {
                Some(bits) if v[lane] == bits => {}
                Some(bits) => {
                    res.fp_consensus_errors += 1;
                    res.record(
                        ComputeError {
                            invocation: gid as u32,
                            problem: p as u32,
                            lane: lane as u32,
                            expected: bits,
                            actual: v[lane],
                            check: "fp_consensus",
                        },
                        max_records,
                    );
                }
                None => {
                    res.fp_consensus_errors += 1;
                    res.record(
                        ComputeError {
                            invocation: gid as u32,
                            problem: p as u32,
                            lane: lane as u32,
                            expected: 0,
                            actual: v[lane],
                            check: "fp_no_consensus",
                        },
                        max_records,
                    );
                }
            }
        }
    }
    res
}

/// Index-addressed 32-bit pattern for the transfer test (any word verifiable alone).
pub fn transfer_word(index: u64, seed: u64) -> u32 {
    // splitmix64 finaliser
    let mut z = index.wrapping_add(seed).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (z ^ (z >> 31)) as u32
}

pub fn fill_transfer(buf: &mut [u32], seed: u64) {
    for (i, w) in buf.iter_mut().enumerate() {
        *w = transfer_word(i as u64, seed);
    }
}

/// Returns (mismatch count, first mismatches as (word index, expected, actual)).
pub fn verify_transfer(buf: &[u32], seed: u64, max_records: usize) -> (u64, Vec<(u64, u32, u32)>) {
    let mut count = 0u64;
    let mut recs = Vec::new();
    for (i, w) in buf.iter().enumerate() {
        let e = transfer_word(i as u64, seed);
        if *w != e {
            count += 1;
            if recs.len() < max_records {
                recs.push((i as u64, e, *w));
            }
        }
    }
    (count, recs)
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: u32 = 64;
    const ITERS: u32 = 256;

    fn golden(n: usize, seed: u32) -> Vec<[u32; 4]> {
        (0..n)
            .map(|gid| {
                let p = gid as u32 % P;
                let (x, acc) = int_chain(p, seed, ITERS);
                let (g, h) = fp_chain(p, seed, ITERS);
                [x, acc, g.to_bits(), h.to_bits()]
            })
            .collect()
    }

    #[test]
    fn clean_output_verifies() {
        let out = golden(P as usize * 8, 42);
        let r = verify(&out, P, 42, ITERS, 16);
        assert_eq!(r.total(), 0, "{r:?}");
    }

    #[test]
    fn chains_are_seed_and_problem_sensitive() {
        assert_ne!(int_chain(1, 1, 64), int_chain(2, 1, 64));
        assert_ne!(int_chain(1, 1, 64), int_chain(1, 2, 64));
        let (g, h) = fp_chain(3, 9, 4096);
        assert!(g.is_finite() && h.is_finite());
        assert!(g > 100.0 && g < 1000.0, "g={g}"); // ~ E[sqrt u]/(1-0.999) = 666
        assert!(h > 0.3 && h < 1.0, "h={h}"); // ~ E[u^2]/(1-0.5) = 0.667
    }

    #[test]
    fn injected_int_fault_is_caught_with_location() {
        let mut out = golden(P as usize * 8, 7);
        out[77][1] ^= 0x0000_0400;
        let r = verify(&out, P, 7, ITERS, 16);
        assert_eq!(r.int_errors, 1);
        assert_eq!(r.records[0].invocation, 77);
        assert_eq!(r.records[0].lane, 1);
        assert_eq!(r.records[0].expected ^ r.records[0].actual, 0x400);
    }

    #[test]
    fn single_replica_fp_fault_breaks_consensus() {
        let mut out = golden(P as usize * 8, 7);
        out[200][2] ^= 1; // one ULP in g on one invocation
        let r = verify(&out, P, 7, ITERS, 16);
        assert_eq!(r.fp_consensus_errors, 1);
        assert_eq!(r.fp_reference_errors, 0);
        assert_eq!(r.records[0].check, "fp_consensus");
        assert_eq!(r.records[0].invocation, 200);
    }

    #[test]
    fn rounding_differences_within_tolerance_are_accepted() {
        // simulate a GPU that rounds differently but consistently on every SM
        let mut out = golden(P as usize * 4, 3);
        for v in out.iter_mut() {
            v[2] = (f32::from_bits(v[2]) * (1.0 + 1e-6)).to_bits();
        }
        let r = verify(&out, P, 3, ITERS, 16);
        assert_eq!(r.total(), 0, "{r:?}");
    }

    #[test]
    fn systematic_fp_error_fails_reference_check() {
        let mut out = golden(P as usize * 4, 3);
        for v in out.iter_mut().filter(|v| v[3] != 0) {
            v[3] = (f32::from_bits(v[3]) * 1.5).to_bits();
        }
        let r = verify(&out, P, 3, ITERS, 1000);
        assert_eq!(r.fp_reference_errors, P as u64);
        assert_eq!(r.fp_consensus_errors, 0);
    }

    #[test]
    fn no_majority_is_reported() {
        let mut out = golden(P as usize * 2, 5); // 2 replicas per problem
        out[P as usize][2] ^= 4; // problem 0: 1 vs 1 -> no majority
        let r = verify(&out, P, 5, ITERS, 16);
        assert_eq!(r.fp_consensus_errors, 2);
        assert!(r.records.iter().all(|e| e.check == "fp_no_consensus"));
    }

    #[test]
    fn transfer_pattern_roundtrip_and_fault() {
        let mut buf = vec![0u32; 4096];
        fill_transfer(&mut buf, 11);
        assert_eq!(verify_transfer(&buf, 11, 4).0, 0);
        assert_ne!(verify_transfer(&buf, 12, 4).0, 0);
        buf[1000] ^= 0x8000_0000;
        let (n, recs) = verify_transfer(&buf, 11, 4);
        assert_eq!(n, 1);
        assert_eq!(recs[0].0, 1000);
    }
}
