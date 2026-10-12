// The gated MLP's activation (mamba_ssm/modules/mlp.py): each row of the
// up projection holds the value columns, then the gate columns, and the
// activation is value * silu(gate). The projections themselves are GEMMs.
//
// Composed into the Mamba-3 SISO, Mamba-3 MIMO and Mamba-2 modules, which
// run its two kernels for the MLP after the mixer.

#include "_typed_prelude.cuh"

#ifndef LOG2E
#define LOG2E 1.4426950408889634f
#endif

// act[r, j] = up[r, j] * silu(up[r, hidden + j]) for rows r < rows and
// columns j < hidden. One thread per output element, 64-bit indices.
// Grid: 1-D over rows * hidden.
#define DEFINE_MLP_GATE_FWD(NAME, TY, FROM_F)                                  \
extern "C" __global__ void NAME(                                               \
    TY* __restrict__ act,                                                      \
    const TY* __restrict__ up,                                                 \
    int rows, int hidden                                                       \
) {                                                                            \
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;            \
    long long n = (long long)rows * hidden;                                    \
    if (i >= n) return;                                                        \
    long long r = i / hidden;                                                  \
    long long j = i - r * hidden;                                              \
    float v = to_f(up[r * 2 * hidden + j]);                                    \
    float g = to_f(up[r * 2 * hidden + hidden + j]);                           \
    act[i] = FROM_F(v * (g / (1.0f + exp2f(-g * LOG2E))));                     \
}

DEFINE_MLP_GATE_FWD(mlp_gate_fwd, float, from_f_f32)
DEFINE_MLP_GATE_FWD(mlp_gate_fwd_bf16, __nv_bfloat16, from_f_bf16)
DEFINE_MLP_GATE_FWD(mlp_gate_fwd_f16, __half, from_f_f16)

// The gate's backward: from d act, the value and gate gradients in place
// of the up projection's row, d value = d act * silu(gate) and d gate =
// d act * value * silu'(gate). Every rounding is spelled out, so the
// kernel gives the same bits in every module that composes it.
// Grid: 1-D over rows * hidden.
__device__ __forceinline__ float mlp_exp(float x) { return exp2f(__fmul_rn(x, LOG2E)); }

__device__ __forceinline__ float mlp_sigmoid(float x) {
    return __fdiv_rn(1.0f, __fadd_rn(1.0f, mlp_exp(-x)));
}

__device__ __forceinline__ float mlp_silu_grad(float v) {
    const float sg = mlp_sigmoid(v);
    return __fmul_rn(sg, __fmaf_rn(v, __fsub_rn(1.0f, sg), 1.0f));
}

#define DEFINE_MLP_GATE_BWD(NAME, TY, FROM_F)                                  \
extern "C" __global__ void NAME(                                               \
    TY* __restrict__ d_up, const TY* __restrict__ d_act,                       \
    const TY* __restrict__ up, int rows, int hidden                            \
) {                                                                            \
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;      \
    const long long n = (long long)rows * hidden;                              \
    if (i >= n) return;                                                        \
    const long long r = i / hidden;                                            \
    const long long j = i - r * hidden;                                        \
    const float d = to_f(d_act[i]);                                            \
    const float v = to_f(up[r * 2 * hidden + j]);                              \
    const float g = to_f(up[r * 2 * hidden + hidden + j]);                     \
    const float silu = __fdiv_rn(g, __fadd_rn(1.0f, mlp_exp(-g)));            \
    d_up[r * 2 * hidden + j] = FROM_F(__fmul_rn(d, silu));                     \
    d_up[r * 2 * hidden + hidden + j] =                                        \
        FROM_F(__fmul_rn(__fmul_rn(d, v), mlp_silu_grad(g)));                 \
}

DEFINE_MLP_GATE_BWD(mlp_gate_bwd, float, from_f_f32)
DEFINE_MLP_GATE_BWD(mlp_gate_bwd_bf16, __nv_bfloat16, from_f_bf16)
DEFINE_MLP_GATE_BWD(mlp_gate_bwd_f16, __half, from_f_f16)
