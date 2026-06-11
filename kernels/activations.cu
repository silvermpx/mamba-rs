// Mamba activation kernels: SiLU + softplus (forward + backward).
//
// Templated over activation dtype via extern "C" wrappers with suffixes:
//   NAME_f32, NAME_bf16, NAME_f16
// Math in f32, storage in T_IN (upcast on load, downcast on store).
// Backward kernels remain f32-only (training path is f32).

#include "_typed_prelude.cuh"

// ===================== SiLU forward (templated) =====================

#define DEFINE_SILU_FWD(SUFFIX, T, FROM_F)                                \
extern "C" __global__ void silu_forward_##SUFFIX(T* x, int n) {           \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                        \
    if (i >= n) return;                                                   \
    float v = to_f(x[i]);                                                 \
    x[i] = FROM_F(v / (1.0f + exp2f(-v * LOG2E)));                        \
}

DEFINE_SILU_FWD(f32,  float,         from_f_f32)
DEFINE_SILU_FWD(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_SILU_FWD(f16,  __half,        from_f_f16)

// Legacy alias — existing code calls `silu_forward` without suffix.
extern "C" __global__ void silu_forward(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    x[i] = v / (1.0f + exp2f(-v * LOG2E));
}

// ===================== Softplus forward (templated) =====================

#define DEFINE_SOFTPLUS_FWD(SUFFIX, T, FROM_F)                            \
extern "C" __global__ void softplus_forward_##SUFFIX(T* x, int n) {       \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                        \
    if (i >= n) return;                                                   \
    float v = to_f(x[i]);                                                 \
    x[i] = FROM_F(v > 20.0f ? v : log1pf(exp2f(v * LOG2E)));         \
}

DEFINE_SOFTPLUS_FWD(f32,  float,         from_f_f32)
DEFINE_SOFTPLUS_FWD(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_SOFTPLUS_FWD(f16,  __half,        from_f_f16)

// Legacy alias
extern "C" __global__ void softplus_forward(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    x[i] = v > 20.0f ? v : log1pf(exp2f(v * LOG2E));
}

// ===================== Backward (legacy f32 + typed variants) =====================
// Training path stays f32 (atomicAdd on bf16 is sm_90+ only).
// Typed variants exist for future use and parity testing.

extern "C" __global__ void silu_backward(
    float* dx, const float* x_saved, const float* dy, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float xi = x_saved[i];
    float sigma = 1.0f / (1.0f + exp2f(-xi * LOG2E));
    dx[i] = dy[i] * sigma * (1.0f + xi * (1.0f - sigma));
}

extern "C" __global__ void softplus_backward(
    float* dx, const float* x_saved, const float* dy, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dx[i] = dy[i] / (1.0f + exp2f(-x_saved[i] * LOG2E));
}

// Typed backward — dx/dy in T_IN, x_saved in T_IN. All math in f32.
#define DEFINE_SILU_BWD(SUFFIX, T, FROM_F)                                \
extern "C" __global__ void silu_backward_##SUFFIX(                        \
    T* dx, const T* x_saved, const T* dy, int n                           \
) {                                                                       \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                        \
    if (i >= n) return;                                                   \
    float xi = to_f(x_saved[i]);                                          \
    float sigma = 1.0f / (1.0f + exp2f(-xi * LOG2E));                     \
    dx[i] = FROM_F(to_f(dy[i]) * sigma * (1.0f + xi * (1.0f - sigma)));   \
}

DEFINE_SILU_BWD(f32,  float,         from_f_f32)
DEFINE_SILU_BWD(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_SILU_BWD(f16,  __half,        from_f_f16)

#define DEFINE_SOFTPLUS_BWD(SUFFIX, T, FROM_F)                            \
extern "C" __global__ void softplus_backward_##SUFFIX(                    \
    T* dx, const T* x_saved, const T* dy, int n                           \
) {                                                                       \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                        \
    if (i >= n) return;                                                   \
    float xs = to_f(x_saved[i]);                                          \
    dx[i] = FROM_F(to_f(dy[i]) / (1.0f + exp2f(-xs * LOG2E)));            \
}

DEFINE_SOFTPLUS_BWD(f32,  float,         from_f_f32)
DEFINE_SOFTPLUS_BWD(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_SOFTPLUS_BWD(f16,  __half,        from_f_f16)
