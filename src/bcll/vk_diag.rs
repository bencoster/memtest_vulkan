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
use byte_strings::c_str;
use erupt::{DeviceLoader, vk};
use serde_json::json;
use std::sync::atomic::Ordering::SeqCst;
use std::time::{Duration, Instant};

pub const EXIT_PASS: i32 = 0;
pub const EXIT_FAIL: i32 = 1;
pub const EXIT_INCOMPLETE: i32 = 2;
pub const EXIT_INTERNAL: i32 = 4;

const FENCE_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_RECORDS_PER_BATCH: usize = 64;
const MAX_ERROR_EVENTS: usize = 1024;

// compute test geometry: 1 Mi invocations, 1024 problems -> 1024 replicas each
const INVOCATIONS: u32 = 1 << 20;
const PROBLEMS: u32 = 1024;
const ITERS: u32 = 8192;

const TRANSFER_BYTES: u64 = 256 << 20;
const TRANSFER_ROUNDS_PER_SEED: u32 = 8;

#[derive(Debug)]
pub enum GpuFault {
    Timeout(&'static str),
    DeviceLost(&'static str),
    Other(String),
}

impl std::fmt::Display for GpuFault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuFault::Timeout(w) => write!(
                f,
                "GPU did not finish {w} within {}s (hang?)",
                FENCE_TIMEOUT.as_secs()
            ),
            GpuFault::DeviceLost(w) => {
                write!(f, "VK_ERROR_DEVICE_LOST during {w} (driver reset / Xid)")
            }
            GpuFault::Other(m) => f.write_str(m),
        }
    }
}

fn vkr<T>(r: erupt::utils::VulkanResult<T>, what: &'static str) -> Result<T, GpuFault> {
    if r.raw == vk::Result::ERROR_DEVICE_LOST {
        return Err(GpuFault::DeviceLost(what));
    }
    r.result()
        .map_err(|e| GpuFault::Other(format!("{what}: {e}")))
}

pub struct Buf {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: u64,
    pub mapped: *mut u8,
}

/// One logical device on the selected physical device, with 2 command slots.
pub struct Gpu {
    device: DeviceLoader,
    queue: vk::Queue,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    pool: vk::CommandPool,
    cmds: Vec<vk::CommandBuffer>,
    fences: Vec<vk::Fence>,
    pub label: String,
    // dropped after `device` is destroyed in Drop (instance outlives device)
    _loaded: crate::LoadedDevices,
}

impl Gpu {
    pub fn open(vk_index: usize) -> Result<Gpu, GpuFault> {
        let env = crate::ProcessEnv::default();
        env.device_label.store(-1 - vk_index as isize, SeqCst);
        let mut log = crate::output::LogDupler::new(std::io::sink(), None, 0, "bcll worker");
        let loaded = crate::list_devices_ordered_labaled_from_1(&env, &mut log)
            .map_err(|e| GpuFault::Other(format!("Vulkan enumeration failed: {e}")))?;
        let [dev] = loaded.devices.as_slice() else {
            return Err(GpuFault::Other(format!(
                "Vulkan device with enumeration index {vk_index} not found"
            )));
        };
        let (phys, family, mem_props, label) = (
            dev.physical_device,
            dev.queue_family_index,
            dev.memory_props,
            dev.label.clone(),
        );
        let qci = [vk::DeviceQueueCreateInfoBuilder::new()
            .queue_family_index(family)
            .queue_priorities(&[1.0])];
        let dci = vk::DeviceCreateInfoBuilder::new()
            .enabled_extension_names(&*crate::NO_EXTENSIONS_ARRAY)
            .queue_create_infos(&qci);
        // SAFETY: physical device from this instance; create info outlives the call.
        let device = unsafe { DeviceLoader::new(loaded.instance(), phys, &dci) }
            .map_err(|e| GpuFault::Other(format!("vkCreateDevice: {e}")))?;
        // SAFETY: valid device; queue family 0 index exists (requested above).
        let queue = unsafe { device.get_device_queue(family, 0) };
        let pool_info = vk::CommandPoolCreateInfoBuilder::new()
            .queue_family_index(family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let pool = vkr(
            unsafe { device.create_command_pool(&pool_info, None) },
            "create_command_pool",
        )?;
        let alloc = vk::CommandBufferAllocateInfoBuilder::new()
            .command_pool(pool)
            .command_buffer_count(2)
            .level(vk::CommandBufferLevel::PRIMARY);
        let cmds = vkr(
            unsafe { device.allocate_command_buffers(&alloc) },
            "allocate_command_buffers",
        )?
        .to_vec();
        let mut fences = Vec::new();
        for _ in 0..2 {
            fences.push(vkr(
                unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) },
                "create_fence",
            )?);
        }
        Ok(Gpu {
            device,
            queue,
            mem_props,
            pool,
            cmds,
            fences,
            label,
            _loaded: loaded,
        })
    }

    pub fn device(&self) -> &DeviceLoader {
        &self.device
    }

    /// Picks a memory type containing `required`, maximising `preferred` bits and minimising `avoid` bits.
    fn memory_type(
        &self,
        type_bits: u32,
        required: vk::MemoryPropertyFlags,
        preferred: vk::MemoryPropertyFlags,
        avoid: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        (0..self.mem_props.memory_type_count)
            .filter(|i| type_bits & (1 << i) != 0)
            .filter(|i| {
                self.mem_props.memory_types[*i as usize]
                    .property_flags
                    .contains(required)
            })
            .max_by_key(|i| {
                let f = self.mem_props.memory_types[*i as usize].property_flags;
                (
                    (f & preferred).bits().count_ones(),
                    std::cmp::Reverse((f & avoid).bits().count_ones()),
                    std::cmp::Reverse(*i),
                )
            })
    }

    pub fn create_buf(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
        required: vk::MemoryPropertyFlags,
        preferred: vk::MemoryPropertyFlags,
        avoid: vk::MemoryPropertyFlags,
    ) -> Result<Buf, GpuFault> {
        let d = &self.device;
        let info = vk::BufferCreateInfoBuilder::new()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = vkr(unsafe { d.create_buffer(&info, None) }, "create_buffer")?;
        let req = unsafe { d.get_buffer_memory_requirements(buffer) };
        let Some(type_index) = self.memory_type(req.memory_type_bits, required, preferred, avoid)
        else {
            unsafe { d.destroy_buffer(buffer, None) };
            return Err(GpuFault::Other(format!("no memory type with {required:?}")));
        };
        let ai = vk::MemoryAllocateInfoBuilder::new()
            .allocation_size(req.size)
            .memory_type_index(type_index);
        let memory = match vkr(unsafe { d.allocate_memory(&ai, None) }, "allocate_memory") {
            Ok(m) => m,
            Err(e) => {
                unsafe { d.destroy_buffer(buffer, None) };
                return Err(e);
            }
        };
        vkr(
            unsafe { d.bind_buffer_memory(buffer, memory, 0) },
            "bind_buffer_memory",
        )?;
        let mapped = if required.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
            vkr(
                unsafe { d.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::default()) },
                "map_memory",
            )? as *mut u8
        } else {
            std::ptr::null_mut()
        };
        Ok(Buf {
            buffer,
            memory,
            size,
            mapped,
        })
    }

    pub fn destroy_buf(&self, b: Buf) {
        unsafe {
            if !b.mapped.is_null() {
                self.device.unmap_memory(b.memory);
            }
            self.device.destroy_buffer(b.buffer, None);
            self.device.free_memory(b.memory, None);
        }
    }

    /// Records into slot `slot`'s command buffer and submits it with that slot's fence.
    pub fn submit(
        &self,
        slot: usize,
        what: &'static str,
        record: impl FnOnce(&DeviceLoader, vk::CommandBuffer),
    ) -> Result<(), GpuFault> {
        let (d, cmd) = (&self.device, self.cmds[slot]);
        unsafe {
            vkr(
                d.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()),
                what,
            )?;
            vkr(
                d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()),
                what,
            )?;
            record(d, cmd);
            vkr(d.end_command_buffer(cmd), what)?;
            let cbs = [cmd];
            let submit = [vk::SubmitInfoBuilder::new().command_buffers(&cbs)];
            vkr(d.queue_submit(self.queue, &submit, self.fences[slot]), what)
        }
    }

    /// Waits for slot `slot` with a finite timeout, then resets its fence.
    pub fn wait(&self, slot: usize, what: &'static str) -> Result<(), GpuFault> {
        let f = [self.fences[slot]];
        let r = unsafe {
            self.device
                .wait_for_fences(&f, true, FENCE_TIMEOUT.as_nanos() as u64)
        };
        if r.raw == vk::Result::TIMEOUT {
            return Err(GpuFault::Timeout(what));
        }
        vkr(r, what)?;
        vkr(unsafe { self.device.reset_fences(&f) }, what)
    }

    pub fn submit_wait(
        &self,
        what: &'static str,
        record: impl FnOnce(&DeviceLoader, vk::CommandBuffer),
    ) -> Result<Duration, GpuFault> {
        let t = Instant::now();
        self.submit(0, what, record)?;
        self.wait(0, what)?;
        Ok(t.elapsed())
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            for f in &self.fences {
                self.device.destroy_fence(*f, None);
            }
            self.device.destroy_command_pool(self.pool, None);
            self.device.destroy_device(None);
        }
    }
}

fn host_read_barrier(
    d: &DeviceLoader,
    cmd: vk::CommandBuffer,
    src_stage: vk::PipelineStageFlags,
    src_access: vk::AccessFlags,
) {
    let barrier = [vk::MemoryBarrierBuilder::new()
        .src_access_mask(src_access)
        .dst_access_mask(vk::AccessFlags::HOST_READ)];
    unsafe {
        d.cmd_pipeline_barrier(
            cmd,
            src_stage,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &barrier,
            &[],
            &[],
        );
    }
}

fn time_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x1234_5678)
}

pub fn run_worker(args: &WorkerArgs) -> i32 {
    let mut ev = EventWriter::create(&args.events);
    let Some(vk_index) = args.vk_index else {
        ev.emit("worker_error", json!({"message": "--vk-index required"}));
        return EXIT_INTERNAL;
    };
    let gpu = match Gpu::open(vk_index) {
        Ok(g) => g,
        Err(e) => {
            ev.emit("worker_error", json!({"message": e.to_string()}));
            return EXIT_INTERNAL;
        }
    };
    let deadline = Instant::now() + Duration::from_secs_f64(args.seconds);
    let result = match args.kind {
        WorkerKind::Compute => compute(&gpu, &mut ev, deadline),
        WorkerKind::Transfer => transfer(&gpu, &mut ev, deadline),
        WorkerKind::CudaMemtest => unreachable!("routed to the cuda worker"),
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
                    d.cmd_fill_buffer(cmd, dev.buffer, 0, vk::WHOLE_SIZE, 0xDEAD_BEEF);
                })?;
                let h2d = gpu.submit_wait("copy host->device", |d, cmd| unsafe {
                    d.cmd_copy_buffer(cmd, src.buffer, dev.buffer, &region);
                })?;
                unsafe { std::ptr::write_bytes(dst.mapped, 0, TRANSFER_BYTES as usize) };
                let d2h = gpu.submit_wait("copy device->host", |d, cmd| unsafe {
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
