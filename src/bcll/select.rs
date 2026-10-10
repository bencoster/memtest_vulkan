//! `--gpu SPEC` parsing and matching (docs/bcll/DESIGN.md section 4).

use crate::PciAddress;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Spec {
    /// Position in PCI bus order (= nvidia-smi index on all-NVIDIA hosts).
    Index(usize),
    Pci(PciAddress),
    /// Upper-cased UUID or unique prefix, always starting with "GPU-".
    Uuid(String),
    /// Upstream memtest_vulkan label (1-based).
    VkLabel(usize),
}

/// What selection needs to know about one enumerated GPU.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    pub pci: PciAddress,
    pub uuid: Option<String>,
    pub vk_label: usize,
    /// NVML index (= nvidia-smi index), joined by PCI id; None without NVML / non-NVIDIA.
    pub nvml_index: Option<u32>,
}

pub fn format_pci(p: &PciAddress) -> String {
    format!(
        "{:04x}:{:02x}:{:02x}.{:x}",
        p.domain, p.bus, p.device, p.function
    )
}

/// Accepts `dddd:bb:dd.f`, `dddddddd:bb:dd.f` (NVML style), `bb:dd.f`, `bb:dd`; hex, any case.
pub fn parse_pci(s: &str) -> Option<PciAddress> {
    let s = s.trim();
    let (head, function) = match s.rsplit_once('.') {
        Some((h, f)) => (h, u32::from_str_radix(f, 16).ok()?),
        None => (s, 0),
    };
    let parts: Vec<&str> = head.split(':').collect();
    let hex = |t: &str, max_len: usize| -> Option<u32> {
        if t.is_empty() || t.len() > max_len {
            return None;
        }
        u32::from_str_radix(t, 16).ok()
    };
    let (domain, bus, device) = match parts.as_slice() {
        [d, b, dev] => (hex(d, 8)?, hex(b, 2)?, hex(dev, 2)?),
        [b, dev] => (0, hex(b, 2)?, hex(dev, 2)?),
        _ => return None,
    };
    if device > 0x1f || function > 7 {
        return None;
    }
    Some(PciAddress {
        domain,
        bus,
        device,
        function,
    })
}

/// Formats a 16-byte UUID the way NVML / nvidia-smi / CUDA_VISIBLE_DEVICES do.
pub fn uuid_string(b: &[u8; 16]) -> String {
    crate::hooks::uuid_string(b)
}

/// An all-zero address means the driver did not report PCI info: never an identity.
pub fn pci_known(p: &PciAddress) -> bool {
    (p.domain, p.bus, p.device, p.function) != (0, 0, 0, 0)
}

/// Re-resolves the approved card inside a worker process (Codex BLOCK 89de0c26 #2):
/// `cands` = (pci id, uuid) of the devices this process enumerated, in any order.
/// Exactly one must match the expected full PCI id, and its UUID must match when one
/// is expected. Missing, ambiguous or mismatching identity is refused.
pub fn pick_by_identity(
    cands: &[(String, Option<String>)],
    expected_pci: &str,
    expected_uuid: Option<&str>,
) -> Result<usize, String> {
    let want = parse_pci(expected_pci)
        .filter(pci_known)
        .ok_or_else(|| format!("expected PCI id '{expected_pci}' is not a usable identity"))?;
    let want_s = format_pci(&want);
    let hits: Vec<usize> = (0..cands.len()).filter(|i| cands[*i].0 == want_s).collect();
    let i = match hits.as_slice() {
        [one] => *one,
        [] => {
            return Err(format!(
                "approved GPU {want_s} is not present in this process's enumeration"
            ));
        }
        _ => {
            return Err(format!(
                "PCI id {want_s} is ambiguous in this process's enumeration"
            ));
        }
    };
    if let Some(u) = expected_uuid {
        match &cands[i].1 {
            Some(actual) if actual.eq_ignore_ascii_case(u) => {}
            Some(actual) => {
                return Err(format!(
                    "identity mismatch at {want_s}: expected {u}, found {actual}"
                ));
            }
            None => return Err(format!("device at {want_s} reports no UUID; expected {u}")),
        }
    }
    Ok(i)
}

pub fn parse_spec(raw: &str) -> Result<Spec, String> {
    let s = raw.trim();
    let lower = s.to_ascii_lowercase();
    if let Some(rest) = lower.strip_prefix("index:") {
        return rest
            .parse()
            .map(Spec::Index)
            .map_err(|_| format!("bad index in '{raw}'"));
    }
    if let Some(rest) = lower.strip_prefix("vk:") {
        return match rest.parse::<usize>() {
            Ok(n) if n >= 1 => Ok(Spec::VkLabel(n)),
            _ => Err(format!("bad vk label in '{raw}' (1-based)")),
        };
    }
    if let Some(rest) = lower.strip_prefix("pci:") {
        return parse_pci(rest)
            .map(Spec::Pci)
            .ok_or_else(|| format!("bad PCI address in '{raw}'"));
    }
    if let Some(rest) = lower.strip_prefix("uuid:") {
        return uuid_spec(rest, raw);
    }
    if lower.starts_with("gpu-") {
        return uuid_spec(&lower, raw);
    }
    if !s.is_empty() && s.chars().all(|c| c.is_ascii_digit()) {
        return s
            .parse()
            .map(Spec::Index)
            .map_err(|_| format!("bad index '{raw}'"));
    }
    parse_pci(s).map(Spec::Pci).ok_or_else(|| {
        format!("unrecognised GPU spec '{raw}' (index, PCI address, GPU-UUID or vk:N)")
    })
}

fn uuid_spec(lower: &str, raw: &str) -> Result<Spec, String> {
    let u = if lower.starts_with("gpu-") {
        lower.to_string()
    } else {
        format!("gpu-{lower}")
    };
    if u.len() < 8 {
        return Err(format!(
            "UUID prefix too short in '{raw}' (need at least 4 hex digits)"
        ));
    }
    Ok(Spec::Uuid(u.to_ascii_uppercase()))
}

/// Indices of `cands` sorted by PCI address (the "index" numbering).
pub fn pci_order(cands: &[Candidate]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..cands.len()).collect();
    idx.sort_by_key(|i| (cands[*i].pci, cands[*i].vk_label));
    idx
}

/// Returns the position in `cands` matching `spec`.
pub fn select(cands: &[Candidate], spec: &Spec) -> Result<usize, String> {
    match spec {
        // A bare index is the nvidia-smi / NVML index, resolved by stable identity (Codex
        // BLOCK 89de0c26 #7) - never a position in some other enumeration.
        Spec::Index(n) => {
            if cands.iter().all(|c| c.nvml_index.is_none()) {
                return Err(format!(
                    "GPU index {n} means the nvidia-smi index, which needs NVML (unavailable); select by PCI address or UUID"
                ));
            }
            cands
                .iter()
                .position(|c| c.nvml_index == Some(*n as u32))
                .ok_or_else(|| format!("no GPU with nvidia-smi index {n}"))
        }
        Spec::VkLabel(n) => cands
            .iter()
            .position(|c| c.vk_label == *n)
            .ok_or_else(|| format!("no GPU with vk label {n}")),
        Spec::Pci(p) => {
            let hits: Vec<usize> = (0..cands.len())
                .filter(|i| {
                    let c = &cands[*i].pci;
                    (c.domain, c.bus, c.device, c.function)
                        == (p.domain, p.bus, p.device, p.function)
                })
                .collect();
            match hits.as_slice() {
                [one] => Ok(*one),
                [] => Err(format!("no GPU at PCI {}", format_pci(p))),
                _ => Err(format!(
                    "PCI {} is ambiguous (driver did not report PCI info?) - select by UUID",
                    format_pci(p)
                )),
            }
        }
        Spec::Uuid(prefix) => {
            let hits: Vec<usize> = (0..cands.len())
                .filter(|i| {
                    cands[*i]
                        .uuid
                        .as_ref()
                        .is_some_and(|u| u.to_ascii_uppercase().starts_with(prefix))
                })
                .collect();
            match hits.as_slice() {
                [one] => Ok(*one),
                [] => Err(format!("no GPU with UUID {prefix}")),
                _ => Err(format!("UUID prefix {prefix} matches several GPUs")),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pci(bus: u32) -> PciAddress {
        PciAddress {
            domain: 0,
            bus,
            device: 0,
            function: 0,
        }
    }

    /// The BC_LocalLLM rig as upstream enumerates it (vk labels by descending bus).
    fn rig() -> Vec<Candidate> {
        vec![
            Candidate {
                pci: pci(0xC1),
                uuid: Some("GPU-514405d5-9dda-22a9-2f0c-52e4e26adb0f".into()),
                vk_label: 1,
                nvml_index: Some(3),
            },
            Candidate {
                pci: pci(0x82),
                uuid: Some("GPU-5d837ee0-189d-961a-8ea2-253fa75c1554".into()),
                vk_label: 2,
                nvml_index: Some(2),
            },
            Candidate {
                pci: pci(0x81),
                uuid: Some("GPU-67271087-a085-99c3-04af-175497ab1f95".into()),
                vk_label: 3,
                nvml_index: Some(1),
            },
            Candidate {
                pci: pci(0x01),
                uuid: Some("GPU-3836e883-2ccf-da1f-e38b-8b2d9d0a6ab3".into()),
                vk_label: 4,
                nvml_index: Some(0),
            },
        ]
    }

    #[test]
    fn index_follows_pci_order_like_nvidia_smi() {
        let c = rig();
        // nvidia-smi: 0=01:00.0, 1=81:00.0 (display 4090), 2=82:00.0, 3=C1:00.0
        assert_eq!(
            c[select(&c, &parse_spec("0").unwrap()).unwrap()].pci.bus,
            0x01
        );
        assert_eq!(
            c[select(&c, &parse_spec("1").unwrap()).unwrap()].pci.bus,
            0x81
        );
        assert_eq!(
            c[select(&c, &parse_spec("index:3").unwrap()).unwrap()]
                .pci
                .bus,
            0xC1
        );
        assert!(select(&c, &Spec::Index(4)).is_err());
    }

    #[test]
    fn index_is_the_nvml_index_not_a_position() {
        let mut c = rig();
        // a driver may number cards differently from PCI order: the NVML index wins
        c[0].nvml_index = Some(0); // C1 claims index 0
        c[3].nvml_index = Some(3);
        assert_eq!(
            c[select(&c, &parse_spec("0").unwrap()).unwrap()].pci.bus,
            0xC1
        );
        assert!(
            select(&c, &Spec::Index(7))
                .unwrap_err()
                .contains("nvidia-smi index 7")
        );
        // without NVML a bare index is refused instead of guessed
        for x in c.iter_mut() {
            x.nvml_index = None;
        }
        assert!(
            select(&c, &Spec::Index(0))
                .unwrap_err()
                .contains("needs NVML")
        );
        // PCI / UUID selection still works without NVML
        assert!(select(&c, &parse_spec("81:00.0").unwrap()).is_ok());
    }

    fn enumerated(order: &[(u32, &str)]) -> Vec<(String, Option<String>)> {
        order
            .iter()
            .map(|(bus, u)| (format_pci(&pci(*bus)), Some(u.to_string())))
            .collect()
    }

    #[test]
    fn workers_reresolve_identity_in_any_enumeration_order() {
        let a = enumerated(&[
            (0xC1, "GPU-c1"),
            (0x82, "GPU-82"),
            (0x81, "GPU-81"),
            (0x01, "GPU-01"),
        ]);
        let reordered = enumerated(&[
            (0x81, "GPU-81"),
            (0x01, "GPU-01"),
            (0xC1, "GPU-c1"),
            (0x82, "GPU-82"),
        ]);
        assert_eq!(pick_by_identity(&a, "0000:c1:00.0", Some("GPU-c1")), Ok(0));
        assert_eq!(
            pick_by_identity(&reordered, "0000:c1:00.0", Some("GPU-C1")),
            Ok(2)
        );
    }

    #[test]
    fn workers_refuse_missing_ambiguous_or_mismatching_identity() {
        // card disappeared: an ordinal handoff would have picked another card (maybe the display)
        let reduced = enumerated(&[(0x81, "GPU-81"), (0x01, "GPU-01")]);
        assert!(
            pick_by_identity(&reduced, "0000:c1:00.0", Some("GPU-c1"))
                .unwrap_err()
                .contains("not present")
        );
        // same PCI slot, different board (swap) -> UUID mismatch
        let swapped = enumerated(&[(0xC1, "GPU-other")]);
        assert!(
            pick_by_identity(&swapped, "0000:c1:00.0", Some("GPU-c1"))
                .unwrap_err()
                .contains("mismatch")
        );
        // driver without PCI info reports zeros for every device
        let zeros = vec![
            ("0000:00:00.0".to_string(), None),
            ("0000:00:00.0".to_string(), None),
        ];
        assert!(
            pick_by_identity(&zeros, "0000:00:00.0", None)
                .unwrap_err()
                .contains("not a usable identity")
        );
        let dup = enumerated(&[(0xC1, "GPU-a"), (0xC1, "GPU-b")]);
        assert!(
            pick_by_identity(&dup, "c1:00.0", None)
                .unwrap_err()
                .contains("ambiguous")
        );
        let no_uuid = vec![(format_pci(&pci(0xC1)), None)];
        assert!(
            pick_by_identity(&no_uuid, "c1:00.0", Some("GPU-c1"))
                .unwrap_err()
                .contains("no UUID")
        );
    }

    #[test]
    fn vk_label_is_upstream_numbering() {
        let c = rig();
        // upstream label 3 is the display card on this rig (REVIEW R7)
        assert_eq!(
            c[select(&c, &parse_spec("vk:3").unwrap()).unwrap()].pci.bus,
            0x81
        );
    }

    #[test]
    fn pci_forms() {
        let c = rig();
        for s in [
            "81:00.0",
            "0000:81:00.0",
            "00000000:81:00.0",
            "pci:81:00",
            "PCI:0000:81:00.0",
        ] {
            let spec = parse_spec(s).unwrap();
            assert_eq!(c[select(&c, &spec).unwrap()].vk_label, 3, "{s}");
        }
        assert!(select(&c, &parse_spec("02:00.0").unwrap()).is_err());
        assert!(parse_pci("81:20.0").is_none());
        assert!(parse_pci("81:00.8").is_none());
        assert!(parse_pci("zz:00.0").is_none());
    }

    #[test]
    fn uuid_forms_and_prefixes() {
        let c = rig();
        let full = "GPU-67271087-a085-99c3-04af-175497ab1f95";
        assert_eq!(
            c[select(&c, &parse_spec(full).unwrap()).unwrap()].vk_label,
            3
        );
        assert_eq!(
            c[select(&c, &parse_spec("uuid:67271087").unwrap()).unwrap()].vk_label,
            3
        );
        assert_eq!(
            c[select(&c, &parse_spec("gpu-5d83").unwrap()).unwrap()].vk_label,
            2
        );
        assert!(parse_spec("GPU-5").is_err());
        let mut dup = rig();
        dup[0].uuid = Some("GPU-5d837ee0-0000-0000-0000-000000000000".into());
        assert!(
            select(&dup, &parse_spec("GPU-5d83").unwrap())
                .unwrap_err()
                .contains("several")
        );
    }

    #[test]
    fn ambiguous_pci_when_driver_reports_zeros() {
        let mut c = rig();
        for x in c.iter_mut() {
            x.pci = PciAddress::default();
        }
        assert!(
            select(&c, &parse_spec("00:00.0").unwrap())
                .unwrap_err()
                .contains("ambiguous")
        );
    }

    #[test]
    fn formats() {
        assert_eq!(format_pci(&pci(0x81)), "0000:81:00.0");
        let b = [
            0x67, 0x27, 0x10, 0x87, 0xa0, 0x85, 0x99, 0xc3, 0x04, 0xaf, 0x17, 0x54, 0x97, 0xab,
            0x1f, 0x95,
        ];
        assert_eq!(uuid_string(&b), "GPU-67271087-a085-99c3-04af-175497ab1f95");
    }

    #[test]
    fn garbage_rejected() {
        assert!(parse_spec("").is_err());
        assert!(parse_spec("vk:0").is_err());
        assert!(parse_spec("the big one").is_err());
    }
}
