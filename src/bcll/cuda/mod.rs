//! CUDA driver-API VRAM test (DESIGN.md section 7.4). The pattern reference is
//! always compiled (unit tests); the FFI and worker need feature `cuda`.

#[cfg(feature = "cuda")]
pub mod ffi;
#[cfg(any(feature = "cuda", test))]
pub mod patterns;
#[cfg(feature = "cuda")]
pub mod worker;

#[cfg(feature = "cuda")]
pub const AVAILABLE: bool = true;
#[cfg(not(feature = "cuda"))]
pub const AVAILABLE: bool = false;
