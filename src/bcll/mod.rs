//! BCLL diagnostics extension (cargo feature `diagnostics`).
//! Design: docs/bcll/DESIGN.md. Usage: docs/bcll/USAGE.md.

pub mod aer;
pub mod cli;
pub mod codec;
pub mod compute_ref;
pub mod cuda;
pub mod events;
pub mod nvapi;
#[cfg(feature = "nvml")]
#[path = "nvml.rs"]
pub mod nvml;
#[cfg(not(feature = "nvml"))]
#[path = "nvml_stub.rs"]
pub mod nvml;
pub mod orchestrate;
pub mod report;
pub mod select;
pub mod shaders;
pub mod telemetry;
pub mod vk_diag;

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
    Some(orchestrate::main(args))
}
