//! Vulkan worker processes: ALU/FPU verification (`compute`) and PCIe copy
//! integrity/bandwidth (`transfer`). Spawned by the orchestrator as
//! `--bcll-worker compute|transfer --vk-index N --seconds S --events FILE`.
//!
//! Unlike upstream's infinite fence wait (REVIEW R5) every wait here has a
//! finite timeout; a hang is reported as an incomplete test, not a stuck tool.
use super::cli::{WorkerArgs, WorkerKind};
use super::compute_ref;
use super::events::EventWriter;
use super::shaders;
use super::vk_gpu::{Buf, Gpu, GpuFault, host_read_barrier, time_seed, transfer_barrier, vkr};
use byte_strings::c_str;
use erupt::vk;
use serde_json::json;
use std::time::{Duration, Instant};

pub const EXIT_PASS: i32 = 0;
pub const EXIT_FAIL: i32 = 1;
pub const EXIT_INCOMPLETE: i32 = 2;
pub const EXIT_INTERNAL: i32 = 4;
const MAX_RECORDS_PER_BATCH: usize = 64;
const MAX_ERROR_EVENTS: usize = 1024;

// compute test geometry: 1 Mi invocations, 1024 problems -> 1024 replicas each
const INVOCATIONS: u32 = 1 << 20;
const PROBLEMS: u32 = 1024;
const ITERS: u32 = 8192;

const TRANSFER_BYTES: u64 = 256 << 20;
const TRANSFER_ROUNDS_PER_SEED: u32 = 8;

pub fn run_worker(args: &WorkerArgs) -> i32 {
    let mut ev = EventWriter::create(&args.events);
    super::proc::arm_self_deadline(args.hard_deadline_s, EXIT_INCOMPLETE, &args.events);
    let pci = args.pci.as_deref().unwrap_or_default();
    let gpu = match Gpu::open(pci, args.uuid.as_deref()) {
        Ok(g) => g,
        Err(e) => {
            let identity = matches!(e, GpuFault::Identity(_));
            ev.emit(
                "worker_error",
                json!({"message": e.to_string(), "identity_mismatch": identity}),
            );
            return EXIT_INTERNAL;
        }
    };
    let deadline = Instant::now() + Duration::from_secs_f64(args.seconds);
    let result = match args.kind {
        WorkerKind::Compute => compute(&gpu, &mut ev, deadline),
        WorkerKind::Transfer => transfer(&gpu, &mut ev, deadline),
        _ => unreachable!("only compute/transfer are routed here"),
    };
    match result {
        Ok(errors) => {
            ev.emit("worker_finished", json!({"errors": errors}));
            if errors == 0 { EXIT_PASS } else { EXIT_FAIL }
        }
        Err(fault) => {
            let lost = matches!(fault, GpuFault::DeviceLost(_));
            let hang = matches!(fault, GpuFault::Timeout(_));
            ev.emit(
                "worker_error",
                json!({"message": fault.to_string(), "device_lost": lost, "timeout": hang}),
            );
            if lost || hang {
                // the device is unusable; skip Vulkan teardown (it can block) and exit now
                std::process::exit(EXIT_INCOMPLETE);
            }
            EXIT_INTERNAL
        }
    }
}

struct ComputeSlot {
    params: Buf,
    out: Buf,
    set: vk::DescriptorSet,
    seed: u32,
    submitted_at: Instant,
}

fn compute(gpu: &Gpu, ev: &mut EventWriter, deadline: Instant) -> Result<u64, GpuFault> {
    let d = gpu.device();
    let bindings = [0, 1].map(|b| {
        vk::DescriptorSetLayoutBindingBuilder::new()
            .binding(b)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
    });
    let dsl_info = vk::DescriptorSetLayoutCreateInfoBuilder::new().bindings(&bindings);
    let dsl = [vkr(
        unsafe { d.create_descriptor_set_layout(&dsl_info, None) },
        "descriptor_set_layout",
    )?];
    let sizes = [vk::DescriptorPoolSizeBuilder::new()
        ._type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(4)];
    let pool_info = vk::DescriptorPoolCreateInfoBuilder::new()
        .pool_sizes(&sizes)
        .max_sets(2);
    let dpool = vkr(
        unsafe { d.create_descriptor_pool(&pool_info, None) },
        "descriptor_pool",
    )?;
    let layouts = [dsl[0], dsl[0]];
    let set_info = vk::DescriptorSetAllocateInfoBuilder::new()
        .descriptor_pool(dpool)
        .set_layouts(&layouts);
    let sets = vkr(
        unsafe { d.allocate_descriptor_sets(&set_info) },
        "allocate_descriptor_sets",
    )?;
    let pl_info = vk::PipelineLayoutCreateInfoBuilder::new().set_layouts(&dsl);
    let playout = vkr(
        unsafe { d.create_pipeline_layout(&pl_info, None) },
        "pipeline_layout",
    )?;
    let code = Vec::from(shaders::STRESS_SHADER);
    let sm_info = vk::ShaderModuleCreateInfoBuilder::new().code(&code);
    let module = vkr(
        unsafe { d.create_shader_module(&sm_info, None) },
        "shader_module",
    )?;
    let stage = vk::PipelineShaderStageCreateInfoBuilder::new()
        .stage(vk::ShaderStageFlagBits::COMPUTE)
        .module(module)
        .name(c_str!("stress"));
    let cp_info = [vk::ComputePipelineCreateInfoBuilder::new()
        .layout(playout)
        .stage(*stage)];
    let pipeline = vkr(
        unsafe { d.create_compute_pipelines(Default::default(), &cp_info, None) },
        "compute_pipeline",
    )?[0];

    let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let mut slots = Vec::new();
    for set in sets.iter().copied() {
        let params = gpu.create_buf(
            16,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            host,
            vk::MemoryPropertyFlags::empty(),
            vk::MemoryPropertyFlags::empty(),
        )?;
        let out = gpu.create_buf(
            INVOCATIONS as u64 * 16,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            host,
            vk::MemoryPropertyFlags::HOST_CACHED,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let infos0 = [vk::DescriptorBufferInfoBuilder::new()
            .buffer(params.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let infos1 = [vk::DescriptorBufferInfoBuilder::new()
            .buffer(out.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let writes = [
            vk::WriteDescriptorSetBuilder::new()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&infos0),
            vk::WriteDescriptorSetBuilder::new()
                .dst_set(set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&infos1),
        ];
        unsafe { d.update_descriptor_sets(&writes, &[]) };
        slots.push(ComputeSlot {
            params,
            out,
            set,
            seed: 0,
            submitted_at: Instant::now(),
        });
    }

    let base_seed = time_seed() as u32;
    ev.emit(
        "worker_started",
        json!({
            "kind": "compute", "device": gpu.label, "invocations": INVOCATIONS,
            "problems": PROBLEMS, "iters": ITERS, "base_seed": base_seed,
        }),
    );

    let submit_slot = |slot: usize, s: &mut ComputeSlot, seed: u32| -> Result<(), GpuFault> {
        s.seed = seed;
        // SAFETY: params buffer is 16 host-coherent bytes and not in use by the GPU (its fence was waited).
        unsafe {
            std::ptr::copy_nonoverlapping(
                [seed, ITERS, PROBLEMS, INVOCATIONS].as_ptr(),
                s.params.mapped as *mut u32,
                4,
            );
            std::ptr::write_bytes(s.out.mapped, 0, s.out.size as usize);
        }
        s.submitted_at = Instant::now();
        let set = [s.set];
        gpu.submit(slot, "compute dispatch", |d, cmd| unsafe {
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            d.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, playout, 0, &set, &[]);
            d.cmd_dispatch(cmd, INVOCATIONS / shaders::STRESS_WG_SIZE, 1, 1);
            host_read_barrier(
                d,
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::AccessFlags::SHADER_WRITE,
            );
        })
    };

    let mut total_errors = 0u64;
    let mut error_events = 0usize;
    let mut batches = 0u64;
    let mut next_seed = base_seed;
    let mut in_flight = [false, false];
    let result: Result<(), GpuFault> = (|| {
        for (i, s) in slots.iter_mut().enumerate() {
            next_seed = next_seed.wrapping_add(0x9E37_79B9);
            submit_slot(i, s, next_seed)?;
            in_flight[i] = true;
        }
        let mut slot = 0usize;
        while in_flight.iter().any(|f| *f) {
            if !in_flight[slot] {
                slot ^= 1;
                continue;
            }
            gpu.wait(slot, "compute dispatch")?;
            in_flight[slot] = false;
            let s = &mut slots[slot];
            let gpu_ms = s.submitted_at.elapsed().as_secs_f64() * 1000.0;
            // SAFETY: fence waited + host-read barrier recorded; buffer holds INVOCATIONS vec4<u32>.
            let out: &[[u32; 4]] = unsafe {
                std::slice::from_raw_parts(s.out.mapped as *const [u32; 4], INVOCATIONS as usize)
            };
            let outcome = compute_ref::verify(out, PROBLEMS, s.seed, ITERS, MAX_RECORDS_PER_BATCH);
            batches += 1;
            total_errors += outcome.total();
            ev.emit(
                "compute_batch",
                json!({
                    "batch": batches, "seed": s.seed, "gpu_ms": gpu_ms,
                    "int_errors": outcome.int_errors,
                    "fp_consensus_errors": outcome.fp_consensus_errors,
                    "fp_reference_errors": outcome.fp_reference_errors,
                }),
            );
            for e in outcome.records {
                if error_events >= MAX_ERROR_EVENTS {
                    break;
                }
                error_events += 1;
                ev.emit(
                    "compute_error",
                    json!({
                        "batch": batches, "seed": s.seed, "invocation": e.invocation,
                        "workgroup": e.invocation / shaders::STRESS_WG_SIZE, "problem": e.problem,
                        "lane": e.lane, "expected": e.expected, "actual": e.actual,
                        "xor_mask": e.expected ^ e.actual, "check": e.check,
                    }),
                );
            }
            if Instant::now() < deadline {
                next_seed = next_seed.wrapping_add(0x9E37_79B9);
                submit_slot(slot, s, next_seed)?;
                in_flight[slot] = true;
            }
            slot ^= 1;
        }
        Ok(())
    })();
    result?;
    for s in slots {
        gpu.destroy_buf(s.params);
        gpu.destroy_buf(s.out);
    }
    unsafe {
        d.destroy_pipeline(pipeline, None);
        d.destroy_shader_module(module, None);
        d.destroy_pipeline_layout(playout, None);
        d.destroy_descriptor_pool(dpool, None);
        d.destroy_descriptor_set_layout(dsl[0], None);
    }
    ev.emit(
        "compute_summary",
        json!({"batches": batches, "errors": total_errors}),
    );
    Ok(total_errors)
}

fn transfer(gpu: &Gpu, ev: &mut EventWriter, deadline: Instant) -> Result<u64, GpuFault> {
    let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let none = vk::MemoryPropertyFlags::empty();
    let src = gpu.create_buf(
        TRANSFER_BYTES,
        vk::BufferUsageFlags::TRANSFER_SRC,
        host,
        none,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    let dst = gpu.create_buf(
        TRANSFER_BYTES,
        vk::BufferUsageFlags::TRANSFER_DST,
        host,
        vk::MemoryPropertyFlags::HOST_CACHED,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    let dev = gpu.create_buf(
        TRANSFER_BYTES,
        vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        none,
        vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;
    let words = (TRANSFER_BYTES / 4) as usize;
    let base_seed = time_seed();
    ev.emit("worker_started", json!({"kind": "transfer", "device": gpu.label, "bytes": TRANSFER_BYTES, "base_seed": base_seed}));
    let region = [vk::BufferCopyBuilder::new()
        .src_offset(0)
        .dst_offset(0)
        .size(TRANSFER_BYTES)];
    let mut total_errors = 0u64;
    let mut error_events = 0usize;
    let mut round = 0u64;
    let gb = TRANSFER_BYTES as f64 / (1u64 << 30) as f64;
    let result: Result<(), GpuFault> = (|| {
        let mut seed = base_seed;
        while Instant::now() < deadline {
            seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            // SAFETY: src is host-coherent, TRANSFER_BYTES long, idle (previous submits waited).
            let src_words =
                unsafe { std::slice::from_raw_parts_mut(src.mapped as *mut u32, words) };
            compute_ref::fill_transfer(src_words, seed);
            for _ in 0..TRANSFER_ROUNDS_PER_SEED {
                if Instant::now() >= deadline {
                    break;
                }
                round += 1;
                // poison the device copy so a silently skipped H2D cannot pass
                gpu.submit_wait("fill device buffer", |d, cmd| unsafe {
                    transfer_barrier(d, cmd); // prior D2H read of `dev` before overwriting it
                    d.cmd_fill_buffer(cmd, dev.buffer, 0, vk::WHOLE_SIZE, 0xDEAD_BEEF);
                })?;
                let h2d = gpu.submit_wait("copy host->device", |d, cmd| unsafe {
                    transfer_barrier(d, cmd); // fill write (WAW) + host writes of `src`
                    d.cmd_copy_buffer(cmd, src.buffer, dev.buffer, &region);
                })?;
                unsafe { std::ptr::write_bytes(dst.mapped, 0, TRANSFER_BYTES as usize) };
                let d2h = gpu.submit_wait("copy device->host", |d, cmd| unsafe {
                    transfer_barrier(d, cmd); // H2D write of `dev` -> this read; host zeroing of `dst`
                    d.cmd_copy_buffer(cmd, dev.buffer, dst.buffer, &region);
                    host_read_barrier(
                        d,
                        cmd,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::AccessFlags::TRANSFER_WRITE,
                    );
                })?;
                // SAFETY: dst idle after the waited submit + host-read barrier.
                let dst_words =
                    unsafe { std::slice::from_raw_parts(dst.mapped as *const u32, words) };
                let (bad, recs) = if dst_words == &src_words[..] {
                    (0, Vec::new())
                } else {
                    compute_ref::verify_transfer(dst_words, seed, MAX_RECORDS_PER_BATCH)
                };
                total_errors += bad;
                ev.emit(
                    "transfer_round",
                    json!({
                        "round": round, "bytes": TRANSFER_BYTES,
                        "h2d_gbps": gb / h2d.as_secs_f64(), "d2h_gbps": gb / d2h.as_secs_f64(),
                        "mismatched_words": bad,
                    }),
                );
                for (idx, exp, act) in recs {
                    if error_events >= MAX_ERROR_EVENTS {
                        break;
                    }
                    error_events += 1;
                    ev.emit("transfer_error", json!({
                        "round": round, "offset_bytes": idx * 4, "expected": exp, "actual": act, "xor_mask": exp ^ act,
                    }));
                }
            }
        }
        Ok(())
    })();
    result?;
    gpu.destroy_buf(src);
    gpu.destroy_buf(dst);
    gpu.destroy_buf(dev);
    ev.emit(
        "transfer_summary",
        json!({"rounds": round, "errors": total_errors}),
    );
    Ok(total_errors)
}
