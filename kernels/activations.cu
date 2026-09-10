// Mamba activation kernels: SiLU + softplus (forward + backward).
//
// Templated over activation dtype via extern "C" wrappers with suffixes:
//   NAME_f32, NAME_bf16, NAME_f16
// Math in f32, storage in T_IN (upcast on load, downcast on store).
// Backward kernels remain f32-only (training path is f32).

#include "_typed_prelude.cuh"

// ===================== SiLU forward (templated) =====================

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

// Untyped f32 entry: the training and prefill paths load it as softplus_fwd.
extern "C" __global__ void softplus_forward(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    x[i] = v > 20.0f ? v : log1pf(exp2f(v * LOG2E));
}

// ===================== Backward (legacy f32 + typed variants) =====================
// Training path stays f32 (atomicAdd on bf16 is sm_90+ only).
// Typed variants exist for future use and parity testing.

extern "C" __global__ void softplus_backward(
    float* dx, const float* x_saved, const float* dy, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dx[i] = dy[i] / (1.0f + exp2f(-x_saved[i] * LOG2E));
}

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
