//! `--bcll-worker cuda-memtest`: claim (nearly) all free VRAM with the CUDA
//! virtual-memory-management API and run the pattern passes over it.
//!
//! Allocation: cuMemCreate chunks (512 MiB, halving down to the recommended
//! granularity to take the tail) mapped back-to-back into ONE reserved VA range,
//! so the kernels see a contiguous buffer and coverage is exact.

use super::ffi::*;
use super::patterns::{self, Op};
use crate::bcll::cli::WorkerArgs;
use crate::bcll::events::EventWriter;
use crate::bcll::select;
use serde_json::json;
use std::ffi::c_void;
use std::time::{Duration, Instant};

const PTX: &str = include_str!("kernels.ptx");
const CHUNK_BYTES: usize = 512 << 20;
const LOG_ENTRIES: usize = 1024;
const MAX_ERROR_EVENTS: usize = 4096;
const THREADS: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy)]
struct ErrEntry {
    index: u64,
    expected: u32,
    actual: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ErrLog {
    count: u64,
    bit_hist: [u32; 32],
    logged: u32,
    pad: u32,
    entries: [ErrEntry; LOG_ENTRIES],
}

const _: () = assert!(std::mem::size_of::<ErrLog>() == 144 + 16 * LOG_ENTRIES);

pub const EXIT_PASS: i32 = 0;
pub const EXIT_FAIL: i32 = 1;
pub const EXIT_INCOMPLETE: i32 = 2;
pub const EXIT_INTERNAL: i32 = 4;

/// Word indices spot-checked after each fill: both ends, the middle, and every 4 GiB boundary.
pub fn canary_indices(words: u64) -> Vec<u64> {
    if words == 0 {
        return Vec::new();
    }
    let mut v = vec![0, 1, words / 2, words - 1];
    let mut b = 1u64 << 30; // 4 GiB in 32-bit words
    while b < words {
        v.push(b - 1);
        v.push(b);
        b += 1 << 30;
    }
    v.retain(|i| *i < words);
    v.sort_unstable();
    v.dedup();
    v
}

/// Rounds `x` down to a multiple of `g` (g > 0).
pub fn round_down(x: usize, g: usize) -> usize {
    x - x % g
}

/// The chunk sizes the allocator will try, largest first, all multiples of `gran`.
pub fn chunk_ladder(gran: usize) -> Vec<usize> {
    let mut v = Vec::new();
    let mut c = round_down(CHUNK_BYTES.max(gran), gran);
    while c >= gran {
        v.push(c);
        if c == gran {
            break;
        }
        c = round_down(c / 2, gran).max(gran);
    }
    v
}

struct Claim {
    base: CUdeviceptr,
    reserved: usize,
    chunks: Vec<(CUmemGenericAllocationHandle, usize)>,
    mapped: usize,
}

fn mem_prop(device: CUdevice) -> CUmemAllocationProp {
    CUmemAllocationProp {
        type_: CU_MEM_ALLOCATION_TYPE_PINNED,
        requestedHandleTypes: CU_MEM_HANDLE_TYPE_NONE,
        location: CUmemLocation {
            type_: CU_MEM_LOCATION_TYPE_DEVICE,
            id: device,
        },
        win32HandleMetaData: std::ptr::null_mut(),
        allocFlags: CUmemAllocFlags::default(),
    }
}

fn claim(api: &Api, device: CUdevice, target: usize, gran: usize) -> Result<Claim, CudaError> {
    let prop = mem_prop(device);
    let mut base: CUdeviceptr = 0;
    // SAFETY: plain out-param calls into the driver.
    api.check(
        unsafe { (api.cuMemAddressReserve)(&mut base, target, 0, 0, 0) },
        "cuMemAddressReserve",
    )?;
    let mut c = Claim {
        base,
        reserved: target,
        chunks: Vec::new(),
        mapped: 0,
    };
    for size in chunk_ladder(gran) {
        while c.mapped + size <= target {
            let mut h: CUmemGenericAllocationHandle = 0;
            let rc = unsafe { (api.cuMemCreate)(&mut h, size, &prop, 0) };
            if rc == CUDA_ERROR_OUT_OF_MEMORY {
                break; // try the next smaller chunk size
            }
            api.check(rc, "cuMemCreate")?;
            let rc = unsafe { (api.cuMemMap)(base + c.mapped as u64, size, 0, h, 0) };
            if rc != CUDA_SUCCESS {
                unsafe { (api.cuMemRelease)(h) };
                api.check(rc, "cuMemMap")?;
            }
            c.chunks.push((h, size));
            c.mapped += size;
        }
    }
    if c.mapped > 0 {
        let desc = CUmemAccessDesc {
            location: prop.location,
            flags: CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        };
        api.check(
            unsafe { (api.cuMemSetAccess)(base, c.mapped, &desc, 1) },
            "cuMemSetAccess",
        )?;
    }
    Ok(c)
}

fn release(api: &Api, c: &Claim) {
    unsafe {
        let mut off = 0usize;
        for (h, size) in &c.chunks {
            (api.cuMemUnmap)(c.base + off as u64, *size);
            (api.cuMemRelease)(*h);
            off += size;
        }
        (api.cuMemAddressFree)(c.base, c.reserved);
    }
}

fn find_device(api: &Api, uuid: Option<&str>, pci: Option<&str>) -> Result<CUdevice, String> {
    let mut count = 0i32;
    api.check(
        unsafe { (api.cuDeviceGetCount)(&mut count) },
        "cuDeviceGetCount",
    )
    .map_err(|e| e.message)?;
    let want_pci = pci.and_then(select::parse_pci);
    for ordinal in 0..count {
        let mut dev: CUdevice = 0;
        if unsafe { (api.cuDeviceGet)(&mut dev, ordinal) } != CUDA_SUCCESS {
            continue;
        }
        let mut raw = [0u8; 16];
        let dev_uuid = (unsafe { (api.cuDeviceGetUuid)(&mut raw, dev) } == CUDA_SUCCESS)
            .then(|| select::uuid_string(&raw));
        let attr = |a: i32| {
            let mut v = 0i32;
            (unsafe { (api.cuDeviceGetAttribute)(&mut v, a, dev) } == CUDA_SUCCESS)
                .then_some(v as u32)
        };
        let dev_pci = (
            attr(ATTR_PCI_DOMAIN_ID),
            attr(ATTR_PCI_BUS_ID),
            attr(ATTR_PCI_DEVICE_ID),
        );
        let uuid_ok = uuid.is_none_or(|u| {
            dev_uuid
                .as_deref()
                .is_some_and(|d| d.eq_ignore_ascii_case(u))
        });
        let pci_ok =
            want_pci.is_none_or(|p| dev_pci == (Some(p.domain), Some(p.bus), Some(p.device)));
        if uuid_ok && pci_ok {
            return Ok(dev);
        }
    }
    Err(format!(
        "no CUDA device matches uuid={} pci={} ({count} visible)",
        uuid.unwrap_or("-"),
        pci.unwrap_or("-")
    ))
}

pub fn run_worker(args: &WorkerArgs) -> i32 {
    crate::bcll::proc::arm_self_deadline(args.hard_deadline_s, EXIT_INCOMPLETE, &args.events);
    let mut ev = EventWriter::create(&args.events);
    let api = match Api::load() {
        Ok(a) => a,
        Err(e) => {
            ev.emit("worker_error", json!({"message": e, "unsupported": true}));
            return EXIT_INTERNAL;
        }
    };
    match run(&api, args, &mut ev) {
        Ok(0) => {
            ev.emit("worker_finished", json!({"errors": 0}));
            EXIT_PASS
        }
        Ok(n) => {
            ev.emit("worker_finished", json!({"errors": n}));
            EXIT_FAIL
        }
        Err(e) => {
            let fault = e.is_fatal_fault();
            ev.emit(
                "worker_error",
                json!({"message": e.message, "cuda_error": e.code, "gpu_fault": fault,
                       "ecc_uncorrectable": e.code == CUDA_ERROR_ECC_UNCORRECTABLE}),
            );
            if fault {
                // sticky context error: skip teardown, it can block or fail
                std::process::exit(EXIT_INCOMPLETE);
            }
            EXIT_INTERNAL
        }
    }
}

fn err(message: String) -> CudaError {
    CudaError { code: -1, message }
}

fn run(api: &Api, args: &WorkerArgs, ev: &mut EventWriter) -> Result<u64, CudaError> {
    api.check(unsafe { (api.cuInit)(0) }, "cuInit")?;
    let device = find_device(api, args.uuid.as_deref(), args.pci.as_deref()).map_err(err)?;
    let attr = |a: i32| {
        let mut v = 0i32;
        api.check(
            unsafe { (api.cuDeviceGetAttribute)(&mut v, a, device) },
            "cuDeviceGetAttribute",
        )
        .map(|_| v)
    };
    if attr(ATTR_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED)? == 0 {
        return Err(err(
            "device does not support CUDA virtual memory management".into(),
        ));
    }
    let sm_count = attr(ATTR_MULTIPROCESSOR_COUNT)?.max(1) as u32;
    let mut name_buf = [0 as std::ffi::c_char; 256];
    unsafe { (api.cuDeviceGetName)(name_buf.as_mut_ptr(), 255, device) };
    let name = unsafe { std::ffi::CStr::from_ptr(name_buf.as_ptr()) }
        .to_string_lossy()
        .into_owned();

    let mut ctx: CUcontext = std::ptr::null_mut();
    api.check(
        unsafe { (api.cuDevicePrimaryCtxRetain)(&mut ctx, device) },
        "cuDevicePrimaryCtxRetain",
    )?;
    api.check(unsafe { (api.cuCtxSetCurrent)(ctx) }, "cuCtxSetCurrent")?;

    // small allocations BEFORE the big claim, so the claim can take the rest
    let ptx = std::ffi::CString::new(PTX).map_err(|_| err("PTX contains NUL".into()))?;
    let mut module: CUmodule = std::ptr::null_mut();
    api.check(
        unsafe { (api.cuModuleLoadData)(&mut module, ptx.as_ptr() as *const c_void) },
        "cuModuleLoadData (PTX JIT)",
    )?;
    let func = |n: &str| -> Result<CUfunction, CudaError> {
        let c = std::ffi::CString::new(n).unwrap();
        let mut f: CUfunction = std::ptr::null_mut();
        api.check(
            unsafe { (api.cuModuleGetFunction)(&mut f, module, c.as_ptr()) },
            "cuModuleGetFunction",
        )?;
        Ok(f)
    };
    let (k_fill, k_verify) = (func("bcll_fill")?, func("bcll_verify")?);
    let log_size = std::mem::size_of::<ErrLog>();
    let mut log_dev: CUdeviceptr = 0;
    api.check(
        unsafe { (api.cuMemAlloc_v2)(&mut log_dev, log_size) },
        "cuMemAlloc(error log)",
    )?;
    api.check(
        unsafe { (api.cuMemsetD8_v2)(log_dev, 0, log_size) },
        "cuMemsetD8",
    )?;

    let (mut free, mut total) = (0usize, 0usize);
    api.check(
        unsafe { (api.cuMemGetInfo_v2)(&mut free, &mut total) },
        "cuMemGetInfo",
    )?;
    let prop = mem_prop(device);
    let mut gran = 0usize;
    api.check(
        unsafe {
            (api.cuMemGetAllocationGranularity)(
                &mut gran,
                &prop,
                CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
            )
        },
        "cuMemGetAllocationGranularity",
    )?;
    let gran = gran.max(2 << 20);
    let keep_free = (args.keep_free_mb as usize) << 20;
    let target = round_down(free.saturating_sub(keep_free), gran);
    if target < gran {
        return Err(err(format!(
            "only {free} bytes free, keep-free {keep_free}: nothing to test"
        )));
    }
    let c = claim(api, device, target, gran)?;
    let words = (c.mapped / 4) as u64;
    ev.emit(
        "cuda_coverage",
        json!({
            "device": name, "total_bytes": total, "free_bytes_at_start": free,
            "reserved_or_used_bytes": total - free, "keep_free_bytes": keep_free,
            "granularity_bytes": gran, "requested_bytes": target, "tested_bytes": c.mapped,
            "chunks": c.chunks.len(), "sm_count": sm_count,
        }),
    );
    if c.mapped == 0 {
        release(api, &c);
        return Err(err("cuMemCreate could not claim any memory".into()));
    }

    let blocks = sm_count * 16;
    let launch = |f: CUfunction, params: &mut [*mut c_void]| -> Result<(), CudaError> {
        api.check(
            unsafe {
                (api.cuLaunchKernel)(
                    f,
                    blocks,
                    1,
                    1,
                    THREADS,
                    1,
                    1,
                    0,
                    std::ptr::null_mut(),
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        )?;
        api.check(unsafe { (api.cuCtxSynchronize)() }, "cuCtxSynchronize")
    };
    let run_op = |op: &Op| -> Result<(), CudaError> {
        let mut base = c.base;
        let mut n = words;
        match *op {
            Op::Fill {
                mode,
                seed,
                xor_out,
            } => {
                let (mut mode, mut seed, mut xor_out) = (mode, seed, xor_out);
                let mut p: [*mut c_void; 5] = [
                    &mut base as *mut _ as _,
                    &mut n as *mut _ as _,
                    &mut mode as *mut _ as _,
                    &mut seed as *mut _ as _,
                    &mut xor_out as *mut _ as _,
                ];
                launch(k_fill, &mut p)
            }
            Op::Verify {
                mode,
                seed,
                xor_in,
                rewrite,
                xor_out,
                reverse,
            } => {
                let (mut mode, mut seed, mut xor_in, mut xor_out) = (mode, seed, xor_in, xor_out);
                let (mut rw, mut rev) = (rewrite as u32, reverse as u32);
                let mut log = log_dev;
                let mut p: [*mut c_void; 9] = [
                    &mut base as *mut _ as _,
                    &mut n as *mut _ as _,
                    &mut mode as *mut _ as _,
                    &mut seed as *mut _ as _,
                    &mut xor_in as *mut _ as _,
                    &mut rw as *mut _ as _,
                    &mut xor_out as *mut _ as _,
                    &mut rev as *mut _ as _,
                    &mut log as *mut _ as _,
                ];
                launch(k_verify, &mut p)
            }
        }
    };

    // canary: after each fill, read back a few words and compare with the CPU
    // reference (proves PTX == patterns.rs at runtime; a mismatch is an error too)
    let canary = |op: &Op| -> Result<u64, CudaError> {
        let Op::Fill {
            mode,
            seed,
            xor_out,
        } = *op
        else {
            return Ok(0);
        };
        let mut bad = 0u64;
        for idx in canary_indices(words) {
            let mut v = 0u32;
            api.check(
                unsafe {
                    (api.cuMemcpyDtoH_v2)(&mut v as *mut u32 as *mut c_void, c.base + idx * 4, 4)
                },
                "cuMemcpyDtoH(canary)",
            )?;
            if v != patterns::pattern(mode, idx, seed) ^ xor_out {
                bad += 1;
            }
        }
        Ok(bad)
    };

    let deadline = Instant::now() + Duration::from_secs_f64(args.seconds);
    let mut total_errors = 0u64;
    let mut error_events = 0usize;
    let mut bit_hist = [0u64; 32];
    let mut round = 0u32;
    let outcome: Result<(), CudaError> = (|| {
        'rounds: loop {
            for pass in patterns::round_passes(round) {
                if Instant::now() >= deadline {
                    break 'rounds;
                }
                let t = Instant::now();
                let mut pass_errors = 0u64;
                let mut bytes = 0u64;
                for op in &pass.ops {
                    run_op(op)?;
                    bytes += op.bytes(words);
                    let canary_bad = canary(op)?;
                    if canary_bad > 0 {
                        pass_errors += canary_bad;
                        ev.emit("cuda_canary_mismatch", json!({
                            "round": round, "pass": pass.name, "mismatched_words": canary_bad,
                            "note": "fill readback differs from the CPU reference: kernel/PTX divergence or a memory fault",
                        }));
                    }
                    if matches!(op, Op::Verify { .. }) {
                        let mut host: ErrLog = unsafe { std::mem::zeroed() };
                        api.check(
                            unsafe {
                                (api.cuMemcpyDtoH_v2)(
                                    &mut host as *mut ErrLog as *mut c_void,
                                    log_dev,
                                    log_size,
                                )
                            },
                            "cuMemcpyDtoH(error log)",
                        )?;
                        if host.count > 0 {
                            pass_errors += host.count;
                            // exact count per op, so an unfinished pass still counts (BLOCK #5)
                            ev.emit(
                                "cuda_op_errors",
                                json!({
                                    "round": round, "pass": pass.name, "errors": host.count,
                                }),
                            );
                            for (b, n) in host.bit_hist.iter().enumerate() {
                                bit_hist[b] += *n as u64;
                            }
                            for e in &host.entries[..(host.logged as usize).min(LOG_ENTRIES)] {
                                if error_events >= MAX_ERROR_EVENTS {
                                    break;
                                }
                                error_events += 1;
                                ev.emit("cuda_error", json!({
                                    "round": round, "pass": pass.name, "offset_bytes": e.index * 4,
                                    "expected": e.expected, "actual": e.actual, "xor_mask": e.expected ^ e.actual,
                                }));
                            }
                            api.check(
                                unsafe { (api.cuMemsetD8_v2)(log_dev, 0, log_size) },
                                "cuMemsetD8",
                            )?;
                        }
                    }
                }
                total_errors += pass_errors;
                let secs = t.elapsed().as_secs_f64();
                ev.emit("cuda_pass", json!({
                    "round": round, "pass": pass.name, "seconds": secs,
                    "gbps": bytes as f64 / (1u64 << 30) as f64 / secs.max(1e-9), "errors": pass_errors,
                }));
            }
            round += 1;
        }
        Ok(())
    })();
    outcome?;
    ev.emit(
        "cuda_summary",
        json!({"rounds_completed": round, "errors": total_errors, "bit_histogram": bit_hist}),
    );
    release(api, &c);
    unsafe {
        (api.cuMemFree_v2)(log_dev);
        (api.cuModuleUnload)(module);
        (api.cuDevicePrimaryCtxRelease_v2)(device);
    }
    Ok(total_errors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ladder_is_granular_and_descending() {
        let g = 2 << 20;
        let l = chunk_ladder(g);
        assert_eq!(l.first(), Some(&(512 << 20)));
        assert_eq!(l.last(), Some(&g));
        assert!(l.windows(2).all(|w| w[0] > w[1]));
        assert!(l.iter().all(|c| c % g == 0));
        // odd granularity still ends at exactly one granule
        let l = chunk_ladder(6 << 20);
        assert!(l.iter().all(|c| c % (6 << 20) == 0));
        assert_eq!(l.last(), Some(&(6 << 20)));
    }

    #[test]
    fn canary_covers_ends_and_4g_boundaries() {
        assert!(canary_indices(0).is_empty());
        assert_eq!(canary_indices(1), vec![0]);
        let v = canary_indices(3 << 30); // 12 GiB
        for idx in [
            0,
            (1u64 << 30) - 1,
            1 << 30,
            (2u64 << 30) - 1,
            2 << 30,
            (3u64 << 30) - 1,
        ] {
            assert!(v.contains(&idx), "{idx} missing");
        }
        assert!(v.iter().all(|i| *i < 3 << 30));
    }

    #[test]
    fn rounding() {
        assert_eq!(round_down(10, 4), 8);
        assert_eq!(round_down(8, 4), 8);
    }
}
