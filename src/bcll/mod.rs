//! BCLL diagnostics extension (cargo feature `diagnostics`).
//! Design: docs/bcll/DESIGN.md. Usage: docs/bcll/USAGE.md.

pub mod aer;
pub mod cli;
pub mod codec;
pub mod compute_ref;
pub mod cuda;
pub mod events;
pub mod inventory;
pub mod judge;
pub mod nvapi;
#[cfg(feature = "nvml")]
#[path = "nvml.rs"]
pub mod nvml;
#[cfg(not(feature = "nvml"))]
#[path = "nvml_stub.rs"]
pub mod nvml;
pub mod orchestrate;
pub mod proc;
pub mod query;
pub mod report;
pub mod runners;
pub mod select;
pub mod shaders;
pub mod telemetry;
pub mod vk_diag;
pub mod vk_gpu;

/// Absolute path of a driver-shipped system library (Codex BLOCK 5497b81c #2). Windows:
/// ALWAYS `%SystemRoot%\System32\<name>`, even if absent (then loading fails) - never a bare
/// name that the default DLL search path could resolve to a planted copy. Other OSes: the
/// soname, resolved by the dynamic loader's trusted paths.
#[cfg_attr(
    not(any(feature = "nvml", feature = "cuda", all(windows, feature = "nvapi"))),
    allow(dead_code)
)]
pub fn system_library(name: &str) -> Result<String, String> {
    system_library_in(std::env::var_os("SystemRoot"), name)
}

/// `system_library` with the root injected (unit-tested without touching any driver).
pub fn system_library_in(root: Option<std::ffi::OsString>, name: &str) -> Result<String, String> {
    if cfg!(windows) {
        let root =
            root.ok_or("SystemRoot is not set: refusing to load driver DLLs by search path")?;
        let p = std::path::Path::new(&root).join("System32").join(name);
        if !p.is_absolute() {
            return Err(format!("SystemRoot {:?} is not an absolute path", root));
        }
        Ok(p.to_string_lossy().into_owned())
    } else {
        Ok(name.to_string())
    }
}

/// Entry point from `main()`: handles long-flag invocations, returns the exit
/// code; `None` means "not ours" (upstream positional/interactive flow).
pub fn run_if_requested() -> Option<i32> {
    let args: Vec<String> = std::env::args_os()
        .skip(1)
        .map(|a| a.to_string_lossy().into_owned())
        .collect();
    let first = args.first()?;
    if !first.starts_with('-') || first.parse::<isize>().is_ok() {
        return None;
    }
    crate::QUIET_INSTANCE_DROP.store(true, std::sync::atomic::Ordering::SeqCst);
    // supervisors (not workers) contain themselves before starting any child; a failure is
    // reported by the first spawn, which then refuses (fail closed)
    if !args.iter().any(|a| a == "--bcll-worker")
        && let Err(e) = proc::ensure_contained()
    {
        eprintln!("memtest_vulkan: parent-death containment unavailable: {e}");
    }
    Some(orchestrate::main(args))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(windows)]
    fn system_library_never_falls_back_to_a_bare_name() {
        assert!(
            system_library_in(None, "nvml.dll")
                .unwrap_err()
                .contains("SystemRoot")
        );
        // absent file: still the absolute System32 path (load fails), never "nvml.dll"
        let p = system_library_in(Some("C:\\NoSuchWindows".into()), "nvml.dll").unwrap();
        assert_eq!(p, "C:\\NoSuchWindows\\System32\\nvml.dll");
        assert!(system_library_in(Some("relative".into()), "nvml.dll").is_err());
        let real = system_library("nvapi64.dll").unwrap();
        assert!(std::path::Path::new(&real).is_absolute());
    }

    #[test]
    #[cfg(not(windows))]
    fn system_library_uses_soname_elsewhere() {
        assert_eq!(
            system_library_in(None, "libcuda.so.1").unwrap(),
            "libcuda.so.1"
        );
    }
}
