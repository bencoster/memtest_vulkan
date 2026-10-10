//! Minimal CUDA driver-API binding, loaded at runtime with `libloading`
//! (nvcuda.dll / libcuda.so.1 ship with the driver; no toolkit needed).
//! Signatures and constants checked against CUDA 12.8 `cuda.h`.

#![allow(non_snake_case, non_camel_case_types)]

use std::ffi::{c_char, c_void};

pub type CUresult = i32;
pub type CUdevice = i32;
pub type CUdeviceptr = u64;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUmemGenericAllocationHandle = u64;

pub const CUDA_SUCCESS: CUresult = 0;
pub const CUDA_ERROR_OUT_OF_MEMORY: CUresult = 2;
pub const CUDA_ERROR_ECC_UNCORRECTABLE: CUresult = 214;
pub const CUDA_ERROR_ILLEGAL_ADDRESS: CUresult = 700;
pub const CUDA_ERROR_LAUNCH_TIMEOUT: CUresult = 702;

pub const ATTR_MULTIPROCESSOR_COUNT: i32 = 16;
pub const ATTR_PCI_BUS_ID: i32 = 33;
pub const ATTR_PCI_DEVICE_ID: i32 = 34;
pub const ATTR_PCI_DOMAIN_ID: i32 = 50;
pub const ATTR_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED: i32 = 102;

pub const CU_MEM_ALLOCATION_TYPE_PINNED: u32 = 1;
pub const CU_MEM_HANDLE_TYPE_NONE: u32 = 0;
pub const CU_MEM_LOCATION_TYPE_DEVICE: u32 = 1;
pub const CU_MEM_ACCESS_FLAGS_PROT_READWRITE: u32 = 3;
pub const CU_MEM_ALLOC_GRANULARITY_RECOMMENDED: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CUmemLocation {
    pub type_: u32,
    pub id: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CUmemAllocFlags {
    pub compressionType: u8,
    pub gpuDirectRDMACapable: u8,
    pub usage: u16,
    pub reserved: [u8; 4],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CUmemAllocationProp {
    pub type_: u32,
    pub requestedHandleTypes: u32,
    pub location: CUmemLocation,
    pub win32HandleMetaData: *mut c_void,
    pub allocFlags: CUmemAllocFlags,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CUmemAccessDesc {
    pub location: CUmemLocation,
    pub flags: u32,
}

const _: () = assert!(std::mem::size_of::<CUmemAllocationProp>() == 32);
const _: () = assert!(std::mem::size_of::<CUmemAccessDesc>() == 12);

macro_rules! cuda_api {
    ($($name:ident : fn($($arg:ty),*) $(,)?;)*) => {
        pub struct Api {
            _lib: libloading::Library,
            $(pub $name: unsafe extern "system" fn($($arg),*) -> CUresult,)*
        }
        impl Api {
            pub fn load() -> Result<Api, String> {
                // SAFETY: loading the NVIDIA driver library; symbols are resolved by name and type
                // matches cuda.h (CUDAAPI = __stdcall on 32-bit Windows, C ABI on x64/Linux).
                unsafe {
                    let lib = libloading::Library::new(crate::bcll::system_library(LIB_NAME))
                        .map_err(|e| format!("{LIB_NAME} not loadable (no NVIDIA driver?): {e}"))?;
                    $(
                        let $name = *lib
                            .get::<unsafe extern "system" fn($($arg),*) -> CUresult>(
                                concat!(stringify!($name), "\0").as_bytes(),
                            )
                            .map_err(|e| format!("{} missing in {LIB_NAME}: {e}", stringify!($name)))?;
                    )*
                    Ok(Api { _lib: lib, $($name,)* })
                }
            }
        }
    };
}

#[cfg(windows)]
pub const LIB_NAME: &str = "nvcuda.dll";
#[cfg(not(windows))]
pub const LIB_NAME: &str = "libcuda.so.1";

cuda_api! {
    cuInit: fn(u32);
    cuDeviceGetCount: fn(*mut i32);
    cuDeviceGet: fn(*mut CUdevice, i32);
    cuDeviceGetName: fn(*mut c_char, i32, CUdevice);
    cuDeviceGetAttribute: fn(*mut i32, i32, CUdevice);
    cuDeviceGetUuid: fn(*mut [u8; 16], CUdevice);
    cuDevicePrimaryCtxRetain: fn(*mut CUcontext, CUdevice);
    cuDevicePrimaryCtxRelease_v2: fn(CUdevice);
    cuCtxSetCurrent: fn(CUcontext);
    cuCtxSynchronize: fn();
    cuMemGetInfo_v2: fn(*mut usize, *mut usize);
    cuMemGetAllocationGranularity: fn(*mut usize, *const CUmemAllocationProp, u32);
    cuMemCreate: fn(*mut CUmemGenericAllocationHandle, usize, *const CUmemAllocationProp, u64);
    cuMemRelease: fn(CUmemGenericAllocationHandle);
    cuMemAddressReserve: fn(*mut CUdeviceptr, usize, usize, CUdeviceptr, u64);
    cuMemAddressFree: fn(CUdeviceptr, usize);
    cuMemMap: fn(CUdeviceptr, usize, usize, CUmemGenericAllocationHandle, u64);
    cuMemUnmap: fn(CUdeviceptr, usize);
    cuMemSetAccess: fn(CUdeviceptr, usize, *const CUmemAccessDesc, usize);
    cuMemAlloc_v2: fn(*mut CUdeviceptr, usize);
    cuMemFree_v2: fn(CUdeviceptr);
    cuMemcpyDtoH_v2: fn(*mut c_void, CUdeviceptr, usize);
    cuMemsetD8_v2: fn(CUdeviceptr, u8, usize);
    cuModuleLoadData: fn(*mut CUmodule, *const c_void);
    cuModuleUnload: fn(CUmodule);
    cuModuleGetFunction: fn(*mut CUfunction, CUmodule, *const c_char);
    cuLaunchKernel: fn(CUfunction, u32, u32, u32, u32, u32, u32, u32, *mut c_void, *mut *mut c_void, *mut *mut c_void);
    cuGetErrorName: fn(CUresult, *mut *const c_char);
}

impl Api {
    pub fn error_name(&self, rc: CUresult) -> String {
        let mut p: *const c_char = std::ptr::null();
        // SAFETY: cuGetErrorName writes a pointer to a static string (or leaves it null).
        unsafe {
            if (self.cuGetErrorName)(rc, &mut p) == CUDA_SUCCESS && !p.is_null() {
                return std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned();
            }
        }
        format!("CUDA error {rc}")
    }

    pub fn check(&self, rc: CUresult, what: &str) -> Result<(), CudaError> {
        if rc == CUDA_SUCCESS {
            Ok(())
        } else {
            Err(CudaError {
                code: rc,
                message: format!("{what}: {}", self.error_name(rc)),
            })
        }
    }
}

#[derive(Debug, Clone)]
pub struct CudaError {
    pub code: CUresult,
    pub message: String,
}

impl CudaError {
    /// Errors after which the context is unusable (sticky) and indicate a GPU fault.
    pub fn is_fatal_fault(&self) -> bool {
        matches!(
            self.code,
            CUDA_ERROR_ECC_UNCORRECTABLE
                | CUDA_ERROR_ILLEGAL_ADDRESS
                | CUDA_ERROR_LAUNCH_TIMEOUT
                | 999
        )
    }
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}
