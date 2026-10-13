// Mamba activation kernels: SiLU + softplus (forward + backward).
//
// Templated over activation dtype via extern "C" wrappers with suffixes:
//   NAME_f32, NAME_bf16, NAME_f16
// Math in f32, storage in T_IN (upcast on load, downcast on store).
// Backward kernels remain f32-only (training path is f32).

#include "_typed_prelude.cuh"

// The element index of this thread and the grid's stride. The element
// kernels below take 64-bit counts and stride over them, so a window past
// 2^31 elements runs whole; below that every element keeps its thread.
#ifndef ELEM_LOOP
#define ELEM_LOOP(I, N)                                                         \
    for (long long I = (long long)blockIdx.x * blockDim.x + threadIdx.x;       \
         I < (N); I += (long long)gridDim.x * blockDim.x)
#endif

// ===================== SiLU forward (templated) =====================

// ===================== Softplus forward (templated) =====================

#define DEFINE_SOFTPLUS_FWD(SUFFIX, T, FROM_F)                            \
extern "C" __global__ void softplus_forward_##SUFFIX(T* x, long long n) { \
    ELEM_LOOP(i, n) {                                                     \
        float v = to_f(x[i]);                                             \
        x[i] = FROM_F(v > 20.0f ? v : log1pf(exp2f(v * LOG2E)));          \
    }                                                                     \
}

DEFINE_SOFTPLUS_FWD(f32,  float,         from_f_f32)
DEFINE_SOFTPLUS_FWD(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_SOFTPLUS_FWD(f16,  __half,        from_f_f16)

// The same softplus from row-major rows [batch, T, d] into t-major rows
// [batch, d, tstride], for the parallel scan, which reads one channel's
// times in a row. A 32 x 32 tile goes through shared memory, so the loads
// run along d and the stores along t. Each value is the formula above on
// the same input.
#define SOFTPLUS_TILE 32
#define DEFINE_SOFTPLUS_FWD_TMAJOR(SUFFIX, T, FROM_F)                     \
extern "C" __global__ void softplus_forward_tmajor_##SUFFIX(              \
    T* __restrict__ dst, const T* __restrict__ src,                       \
    int T_len, int d, int tstride,                                        \
    int y_first, /* this run's first y block */                           \
    int z_first /* this run's first z block */                            \
) {                                                                       \
    __shared__ T tile[SOFTPLUS_TILE][SOFTPLUS_TILE + 1];                  \
    long long b = z_first + blockIdx.z;                                   \
    int t0 = (y_first + blockIdx.y) * SOFTPLUS_TILE;                      \
    int d0 = blockIdx.x * SOFTPLUS_TILE;                                  \
    for (int r = threadIdx.y; r < SOFTPLUS_TILE; r += blockDim.y) {       \
        int t = t0 + r, c = d0 + threadIdx.x;                             \
        if (t < T_len && c < d) {                                         \
            float v = to_f(src[((long long)b * T_len + t) * d + c]);      \
            tile[r][threadIdx.x] =                                        \
                FROM_F(v > 20.0f ? v : log1pf(exp2f(v * LOG2E)));         \
        }                                                                 \
    }                                                                     \
    __syncthreads();                                                      \
    for (int r = threadIdx.y; r < SOFTPLUS_TILE; r += blockDim.y) {       \
        int c = d0 + r, t = t0 + threadIdx.x;                             \
        if (c < d && t < T_len)                                           \
            dst[((long long)b * d + c) * tstride + t] = tile[threadIdx.x][r]; \
    }                                                                     \
}

DEFINE_SOFTPLUS_FWD_TMAJOR(f32,  float,         from_f_f32)
DEFINE_SOFTPLUS_FWD_TMAJOR(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_SOFTPLUS_FWD_TMAJOR(f16,  __half,        from_f_f16)

// The scan output from its t-major rows [batch, d, tstride] back into
// row-major rows [batch, T, d], with the gate the scan applies on the
// row-major route: y * SiLU(g) with g the gate half of the in_proj row
// (gate_stride > 0), or y as it is (gate_stride = 0). The value stored is
// rounded as the scan's own store rounds it: the output rounded once, the
// SiLU rounded once, their product rounded once, which on f32 is the plain
// product. The tile runs the loads along t and the stores along d.
#define DEFINE_GATE_FROM_TMAJOR(SUFFIX, T, FROM_F)                         \
extern "C" __global__ void gate_from_tmajor_##SUFFIX(                     \
    T* __restrict__ out, const T* __restrict__ y_t,                       \
    const T* __restrict__ proj_gate,                                      \
    int T_len, int d, int tstride, int gate_stride,                       \
    int y_first, /* this run's first y block */                           \
    int z_first /* this run's first z block */                            \
) {                                                                       \
    __shared__ T tile[SOFTPLUS_TILE][SOFTPLUS_TILE + 1];                  \
    long long b = z_first + blockIdx.z;                                   \
    int t0 = (y_first + blockIdx.y) * SOFTPLUS_TILE;                      \
    int d0 = blockIdx.x * SOFTPLUS_TILE;                                  \
    for (int r = threadIdx.y; r < SOFTPLUS_TILE; r += blockDim.y) {       \
        int c = d0 + r, t = t0 + threadIdx.x;                             \
        if (c < d && t < T_len)                                           \
            tile[r][threadIdx.x] = y_t[((long long)b * d + c) * tstride + t]; \
    }                                                                     \
    __syncthreads();                                                      \
    for (int r = threadIdx.y; r < SOFTPLUS_TILE; r += blockDim.y) {       \
        int t = t0 + r, c = d0 + threadIdx.x;                             \
        if (t < T_len && c < d) {                                         \
            T ty = tile[threadIdx.x][r];                                  \
            if (gate_stride > 0) {                                        \
                float g = to_f(proj_gate[((long long)b * T_len + t)       \
                                         * gate_stride + d + c]);         \
                T tg = FROM_F(g / (1.0f + exp2f(-g * 1.4426950408889634f))); \
                ty = FROM_F(to_f(ty) * to_f(tg));                         \
            }                                                             \
            out[((long long)b * T_len + t) * d + c] = ty;                 \
        }                                                                 \
    }                                                                     \
}

DEFINE_GATE_FROM_TMAJOR(f32,  float,         from_f_f32)
DEFINE_GATE_FROM_TMAJOR(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_GATE_FROM_TMAJOR(f16,  __half,        from_f_f16)

// Untyped f32 entry: the training and prefill paths load it as softplus_fwd.
extern "C" __global__ void softplus_forward(float* x, long long n) {
    ELEM_LOOP(i, n) {
        float v = x[i];
        x[i] = v > 20.0f ? v : log1pf(exp2f(v * LOG2E));
    }
}

// ===================== Backward (legacy f32 + typed variants) =====================
// Training path stays f32 (atomicAdd on bf16 is sm_90+ only).
// Typed variants exist for future use and parity testing.

extern "C" __global__ void softplus_backward(
    float* dx, const float* x_saved, const float* dy, long long n
) {
    ELEM_LOOP(i, n) dx[i] = dy[i] / (1.0f + exp2f(-x_saved[i] * LOG2E));
}

#define DEFINE_SOFTPLUS_BWD(SUFFIX, T, FROM_F)                            \
extern "C" __global__ void softplus_backward_##SUFFIX(                    \
    T* dx, const T* x_saved, const T* dy, long long n                     \
) {                                                                       \
    ELEM_LOOP(i, n) {                                                     \
        float xs = to_f(x_saved[i]);                                      \
        dx[i] = FROM_F(to_f(dy[i]) / (1.0f + exp2f(-xs * LOG2E)));        \
    }                                                                     \
}

DEFINE_SOFTPLUS_BWD(f32,  float,         from_f_f32)
DEFINE_SOFTPLUS_BWD(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_SOFTPLUS_BWD(f16,  __half,        from_f_f16)
