//! WGSL compute kernels of the BCLL tests, compiled to SPIR-V at build time by
//! the upstream `memtest_vulkan_build` proc-macro (naga 0.10 WGSL dialect:
//! module-scope constants use `let`).
//!
//! `stress` MUST mirror `compute_ref::int_chain` / `compute_ref::fp_chain`.

pub const STRESS_WG_SIZE: u32 = 64;

pub const STRESS_SHADER: &[u32] = memtest_vulkan_build::compiled_vk_compute_spirv!(
    r#"
struct Params {
    seed: u32,
    iters: u32,
    problems: u32,
    invocations: u32,
}

@group(0) @binding(0) var<storage, read_write> params: Params;
@group(0) @binding(1) var<storage, read_write> out: array<vec4<u32>>;

@compute @workgroup_size(64, 1, 1)
fn stress(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx: u32 = gid.x;
    if idx >= params.invocations {
        return;
    }
    let p: u32 = idx % params.problems;
    var x: u32 = (p * 0x9E3779B9u + params.seed) | 1u;
    var acc: u32 = params.seed ^ 0x85EBCA6Bu;
    var g: f32 = 0.0;
    var h: f32 = 0.0;
    for (var i: u32 = 0u; i < params.iters; i++) {
        x = x ^ (x << 13u);
        x = x ^ (x >> 17u);
        x = x ^ (x << 5u);
        acc = acc * 1664525u + (x ^ i);
        acc = (acc << 7u) | (acc >> 25u);
        let u: f32 = f32(x >> 8u) * 0.000000059604644775390625;
        g = fma(g, 0.999, sqrt(u));
        h = fma(u, u, h * 0.5);
    }
    out[idx] = vec4<u32>(x, acc, bitcast<u32>(g), bitcast<u32>(h));
}
"#
);
