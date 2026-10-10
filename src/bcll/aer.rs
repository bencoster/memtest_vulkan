//! Linux PCIe AER counters and link state from sysfs.
//!
//! `/sys/bus/pci/devices/<dddd:bb:dd.f>/aer_dev_{correctable,nonfatal,fatal}`
//! hold `NAME COUNT` lines ending in `TOTAL_ERR_COR` / `TOTAL_ERR_NONFATAL` /
//! `TOTAL_ERR_FATAL`. The parser is platform-independent (unit-tested on any
//! host); only `read()` touches sysfs and returns None elsewhere.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq)]
pub struct AerSnapshot {
    pub correctable: BTreeMap<String, u64>,
    pub nonfatal: BTreeMap<String, u64>,
    pub fatal: BTreeMap<String, u64>,
    pub current_link_speed: Option<String>,
    pub current_link_width: Option<String>,
    pub max_link_speed: Option<String>,
    pub max_link_width: Option<String>,
}

#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub fn parse_counters(text: &str) -> BTreeMap<String, u64> {
    text.lines()
        .filter_map(|line| {
            let mut it = line.split_whitespace();
            let name = it.next()?;
            let value = it.next()?.parse::<u64>().ok()?;
            Some((name.to_string(), value))
        })
        .collect()
}

fn total(map: &BTreeMap<String, u64>, key: &str) -> u64 {
    map.get(key).copied().unwrap_or_else(|| {
        map.iter()
            .filter(|(k, _)| !k.starts_with("TOTAL_"))
            .map(|(_, v)| v)
            .sum()
    })
}

impl AerSnapshot {
    pub fn total_correctable(&self) -> u64 {
        total(&self.correctable, "TOTAL_ERR_COR")
    }
    pub fn total_nonfatal(&self) -> u64 {
        total(&self.nonfatal, "TOTAL_ERR_NONFATAL")
    }
    pub fn total_fatal(&self) -> u64 {
        total(&self.fatal, "TOTAL_ERR_FATAL")
    }
    pub fn has_counters(&self) -> bool {
        !(self.correctable.is_empty() && self.nonfatal.is_empty() && self.fatal.is_empty())
    }
}

/// Per-counter increase from `before` to `after` (only counters that increased).
pub fn delta(before: &AerSnapshot, after: &AerSnapshot) -> BTreeMap<String, u64> {
    let mut out = BTreeMap::new();
    for (class, b, a) in [
        ("correctable", &before.correctable, &after.correctable),
        ("nonfatal", &before.nonfatal, &after.nonfatal),
        ("fatal", &before.fatal, &after.fatal),
    ] {
        for (k, av) in a {
            let bv = b.get(k).copied().unwrap_or(0);
            if *av > bv {
                out.insert(format!("{class}.{k}"), av - bv);
            }
        }
    }
    out
}

#[cfg(target_os = "linux")]
pub fn read(pci_bus_id: &str) -> Option<AerSnapshot> {
    let dir = std::path::Path::new("/sys/bus/pci/devices").join(pci_bus_id);
    if !dir.exists() {
        return None;
    }
    let file = |name: &str| std::fs::read_to_string(dir.join(name)).ok();
    let line = |name: &str| file(name).map(|s| s.trim().to_string());
    Some(AerSnapshot {
        correctable: file("aer_dev_correctable")
            .map(|t| parse_counters(&t))
            .unwrap_or_default(),
        nonfatal: file("aer_dev_nonfatal")
            .map(|t| parse_counters(&t))
            .unwrap_or_default(),
        fatal: file("aer_dev_fatal")
            .map(|t| parse_counters(&t))
            .unwrap_or_default(),
        current_link_speed: line("current_link_speed"),
        current_link_width: line("current_link_width"),
        max_link_speed: line("max_link_speed"),
        max_link_width: line("max_link_width"),
    })
}

#[cfg(not(target_os = "linux"))]
pub fn read(_pci_bus_id: &str) -> Option<AerSnapshot> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const COR: &str = "RxErr 0\nBadTLP 2\nBadDLLP 1\nRollover 0\nTimeout 0\nNonFatalErr 0\nCorrIntErr 0\nHeaderOF 0\nTOTAL_ERR_COR 3\n";
    const NONFATAL: &str =
        "Undefined 0\nDLP 0\nSDES 0\nTLP 0\nFCP 0\nCmpltTO 1\nTOTAL_ERR_NONFATAL 1\n";

    #[test]
    fn parses_kernel_format() {
        let m = parse_counters(COR);
        assert_eq!(m["BadTLP"], 2);
        assert_eq!(m["TOTAL_ERR_COR"], 3);
        assert_eq!(m.len(), 9);
    }

    #[test]
    fn ignores_garbage_lines() {
        let m = parse_counters("foo\nBar x\n\nBaz 7 extra\n");
        assert_eq!(m.len(), 1);
        assert_eq!(m["Baz"], 7);
    }

    #[test]
    fn totals_fall_back_to_sum_when_kernel_lacks_total_line() {
        let s = AerSnapshot {
            correctable: parse_counters("RxErr 1\nBadTLP 2\n"),
            nonfatal: parse_counters(NONFATAL),
            ..Default::default()
        };
        assert_eq!(s.total_correctable(), 3);
        assert_eq!(s.total_nonfatal(), 1);
        assert_eq!(s.total_fatal(), 0);
    }

    #[test]
    fn delta_reports_only_increases() {
        let before = AerSnapshot {
            correctable: parse_counters(COR),
            ..Default::default()
        };
        let mut after = before.clone();
        *after.correctable.get_mut("BadDLLP").unwrap() += 4;
        *after.correctable.get_mut("TOTAL_ERR_COR").unwrap() += 4;
        after.fatal.insert("TOTAL_ERR_FATAL".into(), 1);
        let d = delta(&before, &after);
        assert_eq!(d["correctable.BadDLLP"], 4);
        assert_eq!(d["fatal.TOTAL_ERR_FATAL"], 1);
        assert_eq!(d.len(), 3);
    }
}
