// The gated MLP's activation (mamba_ssm/modules/mlp.py): each row of the
// up projection holds the value columns, then the gate columns, and the
// activation is value * silu(gate). The projections themselves are GEMMs.
//
// Composed after the Mamba-3 head kernels in the Mamba-3 module.

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
