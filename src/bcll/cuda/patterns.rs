//! CPU reference of the CUDA kernels' patterns and the per-round pass schedule.
//! `pattern` MUST stay identical to `pattern()` in kernels.cu.

pub const MODE_ADDRESS: u32 = 0;
pub const MODE_CHECKER: u32 = 1;
pub const MODE_WALKING: u32 = 2;
pub const MODE_RANDOM: u32 = 3;

fn fmix32(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x85EB_CA6B);
    h ^= h >> 13;
    h = h.wrapping_mul(0xC2B2_AE35);
    h ^= h >> 16;
    h
}

pub fn pattern(mode: u32, i: u64, seed: u32) -> u32 {
    let lo = i as u32;
    let hi = (i >> 32) as u32;
    match mode {
        MODE_ADDRESS => lo ^ hi.wrapping_mul(0x9E37_79B9) ^ seed,
        MODE_CHECKER => {
            if lo & 1 != 0 {
                !seed
            } else {
                seed
            }
        }
        MODE_WALKING => 1u32 << ((i.wrapping_add(seed as u64) as u32) & 31),
        _ => fmix32(lo ^ hi.wrapping_mul(0x85EB_CA6B) ^ seed),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    /// base[i] = pattern ^ xor_out
    Fill { mode: u32, seed: u32, xor_out: u32 },
    /// check base[i] == pattern ^ xor_in; if rewrite, base[i] = expected ^ xor_out
    Verify {
        mode: u32,
        seed: u32,
        xor_in: u32,
        rewrite: bool,
        xor_out: u32,
        reverse: bool,
    },
}

impl Op {
    /// Bytes moved over the memory bus for `words` words (bandwidth accounting).
    pub fn bytes(&self, words: u64) -> u64 {
        match self {
            Op::Fill { .. } => words * 4,
            Op::Verify { rewrite, .. } => words * 4 * if *rewrite { 2 } else { 1 },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pass {
    pub name: &'static str,
    pub ops: Vec<Op>,
}

const CHECKER_SEEDS: [u32; 4] = [0x5555_5555, 0x3333_3333, 0x0F0F_0F0F, 0x00FF_00FF];

/// The passes of one round; seeds vary per round so every round writes new data.
pub fn round_passes(round: u32) -> Vec<Pass> {
    let seed = fmix32(round.wrapping_add(0x1234_5678));
    let checker = CHECKER_SEEDS[(round % 4) as usize];
    let v = |mode, seed, xor_in, rewrite, xor_out, reverse| Op::Verify {
        mode,
        seed,
        xor_in,
        rewrite,
        xor_out,
        reverse,
    };
    vec![
        Pass {
            name: "own_address",
            ops: vec![
                Op::Fill {
                    mode: MODE_ADDRESS,
                    seed,
                    xor_out: 0,
                },
                v(MODE_ADDRESS, seed, 0, false, 0, true),
            ],
        },
        Pass {
            name: "moving_inversions",
            ops: vec![
                Op::Fill {
                    mode: MODE_CHECKER,
                    seed: checker,
                    xor_out: 0,
                },
                // verify X, write ~X (ascending), verify ~X, write X (descending), verify X
                v(MODE_CHECKER, checker, 0, true, u32::MAX, false),
                v(MODE_CHECKER, checker, u32::MAX, true, u32::MAX, true),
                v(MODE_CHECKER, checker, 0, false, 0, false),
            ],
        },
        Pass {
            name: "walking_ones",
            ops: vec![
                Op::Fill {
                    mode: MODE_WALKING,
                    seed: round,
                    xor_out: 0,
                },
                v(MODE_WALKING, round, 0, true, u32::MAX, false),
                v(MODE_WALKING, round, u32::MAX, false, 0, true),
            ],
        },
        Pass {
            name: "random",
            ops: vec![
                Op::Fill {
                    mode: MODE_RANDOM,
                    seed,
                    xor_out: 0,
                },
                v(MODE_RANDOM, seed, 0, false, 0, false),
            ],
        },
    ]
}

#[cfg(test)]
/// CPU model of the kernels, for testing the schedule. Returns the error count of each Verify.
pub fn simulate(mem: &mut [u32], ops: &[Op], mut fault: impl FnMut(usize, u32) -> u32) -> Vec<u64> {
    let mut results = Vec::new();
    for op in ops {
        match *op {
            Op::Fill {
                mode,
                seed,
                xor_out,
            } => {
                for (i, w) in mem.iter_mut().enumerate() {
                    *w = fault(i, pattern(mode, i as u64, seed) ^ xor_out);
                }
            }
            Op::Verify {
                mode,
                seed,
                xor_in,
                rewrite,
                xor_out,
                reverse,
            } => {
                let n = mem.len();
                let mut errors = 0u64;
                for k in 0..n {
                    let i = if reverse { n - 1 - k } else { k };
                    let expected = pattern(mode, i as u64, seed) ^ xor_in;
                    if mem[i] != expected {
                        errors += 1;
                    }
                    if rewrite {
                        mem[i] = fault(i, expected ^ xor_out);
                    }
                }
                results.push(errors);
            }
        }
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_vectors() {
        // pinned values; kernels.cu implements the same arithmetic
        assert_eq!(pattern(MODE_ADDRESS, 0, 0), 0);
        assert_eq!(pattern(MODE_ADDRESS, 1 << 32, 0), 0x9E37_79B9);
        assert_eq!(pattern(MODE_CHECKER, 0, 0x5555_5555), 0x5555_5555);
        assert_eq!(pattern(MODE_CHECKER, 1, 0x5555_5555), 0xAAAA_AAAA);
        assert_eq!(pattern(MODE_WALKING, 3, 0), 8);
        assert_eq!(pattern(MODE_WALKING, 31, 1), 1);
        assert_eq!(pattern(MODE_RANDOM, 0, 0), 0);
        assert_eq!(pattern(MODE_RANDOM, 1, 0), fmix32(1));
    }

    #[test]
    fn address_pattern_unique_across_4g_words() {
        // words 2^32 apart must differ (catches high-address aliasing)
        for i in [0u64, 7, 123_456] {
            assert_ne!(
                pattern(MODE_ADDRESS, i, 9),
                pattern(MODE_ADDRESS, i + (1 << 32), 9)
            );
        }
    }

    #[test]
    fn clean_memory_passes_every_round() {
        for round in 0..4 {
            let mut mem = vec![0u32; 513];
            for pass in round_passes(round) {
                let errs = simulate(&mut mem, &pass.ops, |_, v| v);
                assert!(
                    errs.iter().all(|e| *e == 0),
                    "round {round} {}: {errs:?}",
                    pass.name
                );
            }
        }
    }

    fn detected(pass: &Pass, fault: impl FnMut(usize, u32) -> u32) -> bool {
        let mut mem = vec![0u32; 256];
        simulate(&mut mem, &pass.ops, fault).iter().sum::<u64>() > 0
    }

    #[test]
    fn stuck_bits_are_always_caught_by_inversion_and_walking_passes() {
        // address/random patterns catch a stuck bit only when the pattern holds the
        // opposite value there (~50%); the inversion and walking passes guarantee it.
        for round in 0..8 {
            for bit in [0u32, 7, 31] {
                for stuck_one in [true, false] {
                    let fault = |i: usize, v: u32| match (i == 100, stuck_one) {
                        (true, true) => v | (1 << bit),
                        (true, false) => v & !(1 << bit),
                        _ => v,
                    };
                    for pass in round_passes(round) {
                        let guaranteed = pass.name == "moving_inversions"
                            || (pass.name == "walking_ones" && stuck_one);
                        if guaranteed {
                            assert!(
                                detected(&pass, fault),
                                "round {round} bit {bit} one={stuck_one}: {} missed",
                                pass.name
                            );
                        }
                    }
                    let round_hit = round_passes(round).iter().any(|p| detected(p, fault));
                    assert!(
                        round_hit,
                        "round {round} missed bit {bit} stuck {stuck_one}"
                    );
                }
            }
        }
    }

    #[test]
    fn stuck_at_zero_caught_by_inversions() {
        let pass = &round_passes(0)[1];
        assert_eq!(pass.name, "moving_inversions");
        let mut mem = vec![0u32; 64];
        let errs = simulate(&mut mem, &pass.ops, |i, v| if i == 10 { v & !1 } else { v });
        // bit 0 of word 10 is 1 in either X or ~X, so at least one verify must fail
        assert!(errs.iter().sum::<u64>() >= 1, "{errs:?}");
    }

    #[test]
    fn bandwidth_accounting() {
        let f = Op::Fill {
            mode: 0,
            seed: 0,
            xor_out: 0,
        };
        let v = Op::Verify {
            mode: 0,
            seed: 0,
            xor_in: 0,
            rewrite: true,
            xor_out: 0,
            reverse: false,
        };
        assert_eq!(f.bytes(10), 40);
        assert_eq!(v.bytes(10), 80);
    }
}
