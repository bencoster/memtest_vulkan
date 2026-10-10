//! Minimal Vulkan device wrapper for the BCLL workers: one logical device on the
//! selected physical device (found through upstream's enumeration), two command
//! slots with fences, buffer helpers, and FINITE fence waits (REVIEW R5).

use erupt::{DeviceLoader, vk};
use std::sync::atomic::Ordering::SeqCst;
use std::time::{Duration, Instant};

pub const FENCE_TIMEOUT: Duration = Duration::from_secs(30);

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

pub fn vkr<T>(r: erupt::utils::VulkanResult<T>, what: &'static str) -> Result<T, GpuFault> {
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

pub fn host_read_barrier(
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

pub fn time_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x1234_5678)
}
