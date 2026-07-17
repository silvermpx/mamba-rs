// Global-norm gradient clipping support.
//
// Deterministic sum-of-squares partial reduction over the flat grad arena:
// FIXED grid (GCLIP_BLOCKS x GCLIP_THREADS) with a fixed-stride grid loop,
// per-thread f64 accumulation, fixed shared-memory tree reduce, one f64
// partial per block. The host performs the final ordered sum of the
// GCLIP_BLOCKS partials — no atomics anywhere, so the norm is bit-stable
// across runs (an atomicAdd norm would be the crate's first determinism
// regression). The actual clip scaling reuses the existing scale_grads_f32
// elementwise kernel.
//
// Section-local geometry constants, #undef'd at end of section (the 0.4.0
// ambient-defines lesson: kernel sections must own their geometry).

#define GCLIP_THREADS 256
#define GCLIP_BLOCKS 512

// partials: [GCLIP_BLOCKS] f64, one per block.
// Launch geometry MUST be exactly (GCLIP_BLOCKS, GCLIP_THREADS) — the
// fixed-stride loop and the partial count depend on it.
extern "C" __global__ void grad_sumsq_partial_f32(
    double* __restrict__ partials,
    const float* __restrict__ g,
    int n
) {
    __shared__ double smem[GCLIP_THREADS];
    double acc = 0.0;
    const int stride = GCLIP_BLOCKS * GCLIP_THREADS;
    for (int i = blockIdx.x * GCLIP_THREADS + threadIdx.x; i < n; i += stride) {
        double v = (double)g[i];
        acc += v * v;
    }
    smem[threadIdx.x] = acc;
    __syncthreads();
    for (int s = GCLIP_THREADS / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] += smem[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partials[blockIdx.x] = smem[0];
    }
}

#undef GCLIP_THREADS
#undef GCLIP_BLOCKS
