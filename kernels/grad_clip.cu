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

// Device-side clip coefficient: folds the ordered f64 partial sum, the
// sqrt and the PyTorch clip_grad_norm_ coefficient into ONE single-thread
// kernel so the host never has to drain the stream between the norm and
// the scaling pass. The fold MUST stay serial and ascending - it is the
// same association the host loop performed, and the norm is part of the
// determinism contract.
//
// Bit contract vs the host path: identical f64 adds in identical order,
// IEEE sqrt, the same f64 divide and the same `coef < 1.0` gate. When the
// gate does not fire the coefficient is exactly 1.0f and the scaling pass
// is a bitwise no-op (x * 1.0f == x for every finite, infinite and NaN
// bit pattern), so always scaling matches the old conditional skip.
// A non-finite norm yields coef = 1.0 (NaN < 1.0 is false), leaving the
// gradients untouched exactly as the host early-return did before the
// caller raises the error.
extern "C" __global__ void grad_clip_coef_f32(
    float* __restrict__ coef_out,        // [1] clip coefficient
    float* __restrict__ norm_out,        // [1] PRE-clip global L2 norm
    const double* __restrict__ partials, // [GCLIP_BLOCKS]
    int n_partials,
    float max_norm
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    double sum = 0.0;
    for (int i = 0; i < n_partials; i++) {
        sum += partials[i];
    }
    double norm = sqrt(sum);
    norm_out[0] = (float)norm;
    double coef = (double)max_norm / (norm + 1e-6);
    coef_out[0] = coef < 1.0 ? (float)coef : 1.0f;
}

#undef GCLIP_THREADS
#undef GCLIP_BLOCKS
