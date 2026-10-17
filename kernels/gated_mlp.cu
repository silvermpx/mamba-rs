// The gated MLP's activation (mamba_ssm/modules/mlp.py): each row of the
// up projection holds the value columns, then the gate columns, and the
// activation is value * silu(gate). The projections themselves are GEMMs.
//
// Composed into the Mamba-1, Mamba-2, Mamba-3 SISO and Mamba-3 MIMO
// modules, which run its two kernels for the MLP after the mixer.

#include "_typed_prelude.cuh"

#ifndef LOG2E
#define LOG2E 1.4426950408889634f
#endif

// act[r, j] = up[r, j] * silu(up[r, hidden + j]) for rows r < rows and
// columns j < hidden. A block's y threads take rows, strided by the grid,
// its x threads the columns: no index is divided.
#define DEFINE_MLP_GATE_FWD(NAME, TY, FROM_F)                                  \
extern "C" __global__ void NAME(                                               \
    TY* __restrict__ act,                                                      \
    const TY* __restrict__ up,                                                 \
    int rows, int hidden                                                       \
) {                                                                            \
    for (long long r = (long long)blockIdx.x * blockDim.y + threadIdx.y;       \
         r < rows; r += (long long)gridDim.x * blockDim.y) {                   \
        const TY* u = up + r * 2 * hidden;                                     \
        for (int j = threadIdx.x; j < hidden; j += blockDim.x) {               \
            float v = to_f(u[j]);                                              \
            float g = to_f(u[hidden + j]);                                     \
            act[r * hidden + j] =                                              \
                FROM_F(v * (g / (1.0f + exp2f(-g * LOG2E))));                  \
        }                                                                      \
    }                                                                          \
}

DEFINE_MLP_GATE_FWD(mlp_gate_fwd, float, from_f_f32)
DEFINE_MLP_GATE_FWD(mlp_gate_fwd_bf16, __nv_bfloat16, from_f_bf16)
DEFINE_MLP_GATE_FWD(mlp_gate_fwd_f16, __half, from_f_f16)

// The gate's backward: from d act, the value and gate gradients in place
// of the up projection's row, d value = d act * silu(gate) and d gate =
// d act * value * silu'(gate). Every rounding is spelled out, so the
// kernel gives the same bits in every module that composes it. Rows and
// columns go to the threads as in the forward.
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
    for (long long r = (long long)blockIdx.x * blockDim.y + threadIdx.y;       \
         r < rows; r += (long long)gridDim.x * blockDim.y) {                   \
        const TY* u = up + r * 2 * hidden;                                     \
        TY* du = d_up + r * 2 * hidden;                                        \
        for (int j = threadIdx.x; j < hidden; j += blockDim.x) {               \
            const float d = to_f(d_act[r * hidden + j]);                       \
            const float v = to_f(u[j]);                                        \
            const float g = to_f(u[hidden + j]);                               \
            const float silu = __fdiv_rn(g, __fadd_rn(1.0f, mlp_exp(-g)));     \
            du[j] = FROM_F(__fmul_rn(d, silu));                                \
            du[hidden + j] =                                                   \
                FROM_F(__fmul_rn(__fmul_rn(d, v), mlp_silu_grad(g)));          \
        }                                                                      \
    }                                                                          \
}

DEFINE_MLP_GATE_BWD(mlp_gate_bwd, float, from_f_f32)
DEFINE_MLP_GATE_BWD(mlp_gate_bwd_bf16, __nv_bfloat16, from_f_bf16)
DEFINE_MLP_GATE_BWD(mlp_gate_bwd_f16, __half, from_f_f16)
