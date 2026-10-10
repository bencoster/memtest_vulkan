// BCLL memtest_vulkan fork: CUDA VRAM test kernels.
//
// Compiled ONCE to PTX and checked in as kernels.ptx (the driver JITs it at
// runtime, so no CUDA toolkit is needed to run the tool). Regenerate with the
// command in docs/bcll/USAGE.md ("Regenerating the CUDA PTX").
//
// pattern() MUST stay identical to src/bcll/cuda/patterns.rs::pattern
// (unit-tested there against fixed vectors that this file must also satisfy).

#define MODE_ADDRESS 0u
#define MODE_CHECKER 1u
#define MODE_WALKING 2u
#define MODE_RANDOM 3u

#define LOG_ENTRIES 1024u

struct ErrEntry {
    unsigned long long index;  // 32-bit word index within the mapped range
    unsigned int expected;
    unsigned int actual;
};

struct ErrLog {
    unsigned long long count;
    unsigned int bit_hist[32];
    unsigned int logged;
    unsigned int pad;
    ErrEntry entries[LOG_ENTRIES];
};

__device__ __forceinline__ unsigned int fmix32(unsigned int h) {
    h ^= h >> 16;
    h *= 0x85EBCA6Bu;
    h ^= h >> 13;
    h *= 0xC2B2AE35u;
    h ^= h >> 16;
    return h;
}

__device__ __forceinline__ unsigned int pattern(unsigned int mode, unsigned long long i, unsigned int seed) {
    unsigned int lo = (unsigned int)i;
    unsigned int hi = (unsigned int)(i >> 32);
    switch (mode) {
    case MODE_ADDRESS:
        return lo ^ (hi * 0x9E3779B9u) ^ seed;
    case MODE_CHECKER:
        return (lo & 1u) ? ~seed : seed;
    case MODE_WALKING:
        return 1u << ((unsigned int)(i + (unsigned long long)seed) & 31u);
    default:
        return fmix32(lo ^ (hi * 0x85EBCA6Bu) ^ seed);
    }
}

extern "C" __global__ void bcll_fill(unsigned int* base, unsigned long long n,
                                     unsigned int mode, unsigned int seed, unsigned int xor_out) {
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        base[i] = pattern(mode, i, seed) ^ xor_out;
    }
}

extern "C" __global__ void bcll_verify(unsigned int* base, unsigned long long n,
                                       unsigned int mode, unsigned int seed, unsigned int xor_in,
                                       unsigned int rewrite, unsigned int xor_out, unsigned int reverse,
                                       ErrLog* log) {
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long k = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; k < n; k += stride) {
        unsigned long long i = reverse ? (n - 1ull - k) : k;
        unsigned int expected = pattern(mode, i, seed) ^ xor_in;
        unsigned int actual = base[i];
        if (actual != expected) {
            atomicAdd(&log->count, 1ull);
            unsigned int diff = actual ^ expected;
            while (diff) {
                int b = __ffs(diff) - 1;
                atomicAdd(&log->bit_hist[b], 1u);
                diff &= diff - 1u;
            }
            unsigned int slot = atomicAdd(&log->logged, 1u);
            if (slot < LOG_ENTRIES) {
                log->entries[slot].index = i;
                log->entries[slot].expected = expected;
                log->entries[slot].actual = actual;
            }
        }
        if (rewrite) {
            base[i] = expected ^ xor_out;
        }
    }
}
