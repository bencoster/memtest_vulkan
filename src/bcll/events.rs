//! JSON-lines event files between worker children and the orchestrator.
//! One object per line, flushed per event; readers tolerate a truncated tail
//! (a crashed worker) by skipping unparsable lines.

use serde_json::Value;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

pub struct EventWriter {
    file: Option<File>,
    start: std::time::Instant,
}

impl EventWriter {
    pub fn create(path: &Path) -> EventWriter {
        EventWriter {
            file: std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .ok(),
            start: std::time::Instant::now(),
        }
    }

    /// Writes `value` with `ev` and `t_s` added.
    pub fn emit(&mut self, ev: &str, mut value: Value) {
        if let Value::Object(map) = &mut value {
            map.insert("ev".into(), Value::from(ev));
            map.insert(
                "t_s".into(),
                Value::from((self.start.elapsed().as_secs_f64() * 1000.0).round() / 1000.0),
            );
        }
        if let Some(f) = &mut self.file {
            let _ = writeln!(f, "{value}");
            let _ = f.flush();
        }
    }
}

pub fn read_events(path: &Path) -> Vec<Value> {
    let Ok(file) = File::open(path) else {
        return Vec::new();
    };
    BufReader::new(file)
        .lines()
        .map_while(Result::ok)
        .filter_map(|l| serde_json::from_str::<Value>(l.trim()).ok())
        .filter(|v| v.get("ev").is_some())
        .collect()
}

pub fn of_kind<'a>(events: &'a [Value], ev: &'a str) -> impl Iterator<Item = &'a Value> + 'a {
    events.iter().filter(move |e| e["ev"] == ev)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn roundtrip_and_truncated_tail() {
        let dir = std::env::temp_dir().join(format!("bcll-ev-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("e.jsonl");
        let _ = std::fs::remove_file(&path);
        let mut w = EventWriter::create(&path);
        w.emit("a", json!({"n": 1}));
        w.emit("b", json!({"n": 2}));
        drop(w);
        // simulate a crash mid-line plus an upstream hooks line
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        writeln!(
            f,
            "{}",
            crate::hooks::format_finished(1.0, 3, "duration_reached")
        )
        .unwrap();
        write!(f, "{{\"ev\":\"c\",\"n\":").unwrap();
        drop(f);
        let ev = read_events(&path);
        assert_eq!(ev.len(), 3);
        assert_eq!(ev[1]["n"], 2);
        assert!(ev[0]["t_s"].is_number());
        assert_eq!(of_kind(&ev, "memtest_finished").count(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn init_failure_roundtrips_any_message() {
        let msg = "ERROR_DEVICE_LOST \"q\" \\ path\nnext\u{7}";
        let v: Value = serde_json::from_str(&crate::hooks::format_init_failure(1.0, msg)).unwrap();
        assert_eq!(v["message"], msg);
        assert_eq!(v["ev"], "memtest_init_failure");
    }

    #[test]
    fn hooks_error_line_is_valid_json() {
        let arrays = ([1u32; 32], [0u32; 32], [2u32; 32]);
        let r = crate::hooks::ErrorReport {
            iteration: 1,
            mode: "NEXT_RE_READ",
            window_offset: 0,
            window_bytes: 1 << 30,
            idx_min: 0,
            idx_max: 9,
            single_bit_idx: &arrays.0,
            toggle_count: &arrays.1,
            ones_in_value: &arrays.2,
            actual_ff: 1,
            actual_min: 0,
            actual_max: u32::MAX,
        };
        let v: Value = serde_json::from_str(&crate::hooks::format_error(2.0, &r)).unwrap();
        assert_eq!(v["total_errors"], 65);
        assert_eq!(v["mode"], "NEXT_RE_READ");
        assert_eq!(v["single_bit_idx"].as_array().unwrap().len(), 32);
    }
}
