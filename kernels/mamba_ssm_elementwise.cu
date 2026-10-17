// Element-wise kernels only the Mamba-1 block launches: the a_log
// exponent, the x_proj column gathers and the d_xdbl assembly, the SiLU
// gating forward and backward, softplus, and the cast, concat and scatter
// staging of the mixed backward. They compile into the Mamba1 module, so a
// change here moves no other family's kernels.
//
// Typed kernels carry the suffixes _f32, _bf16 and _f16: math in f32,
// storage in the activation type.

#include "_typed_prelude.cuh"

// The element index of this thread and the grid's stride. The element
// kernels below take 64-bit counts and stride over them, so a window past
// 2^31 elements runs whole; below that every element keeps its thread.
#ifndef ELEM_LOOP
#define ELEM_LOOP(I, N)                                                         \
    for (long long I = (long long)blockIdx.x * blockDim.x + threadIdx.x;       \
         I < (N); I += (long long)gridDim.x * blockDim.x)
#endif

extern "C" __global__ void exp_negate(
    float* y, const float* x, long long n
) {
    ELEM_LOOP(i, n) y[i] = -exp2f(x[i] * 1.4426950408889634f);
}

// Two-destination exp_negate: the trainer refreshes BOTH a_neg mirrors
// (backward-side and forward-side) from the same master a_log every
// step — one kernel stores the same register twice instead of two
// launches recomputing the same exp.
extern "C" __global__ void exp_negate2(
    float* y0, float* y1, const float* x, long long n
) {
    ELEM_LOOP(i, n) {
        float v = -exp2f(x[i] * 1.4426950408889634f);
        y0[i] = v;
        y1[i] = v;
    }
}

extern "C" __global__ void gather_cols(
    float* dst, const float* src,
    int batch, int src_stride, int dst_dim, int offset
) {
    ELEM_LOOP(idx, (long long)batch * dst_dim) {
        long long b = idx / dst_dim;
        int d = (int)(idx % dst_dim);
        dst[b * dst_dim + d] = src[b * src_stride + offset + d];
    }
}

// Gating forward with the SiLU recomputed from the gate. The SiLU form
// is the DIVISION `g / (1 + exp2(-g*log2e))` the split kernel used - not
// the algebraically equal `g * sigma`, which rounds differently. The gate
// is read through a (row stride, column offset) pair: the gate half of the
// in_proj output in production, so nothing splits it out first.
extern "C" __global__ void gate_mul_silu(
    float* __restrict__ gated,
    const float* __restrict__ y,
    const float* __restrict__ gate_pre,
    long long n, int d_inner, int gate_stride, int gate_offset
) {
    ELEM_LOOP(i, n) {
        long long row = i / d_inner;
        int col = (int)(i - row * d_inner);
        float g = gate_pre[row * gate_stride + gate_offset + col];
        gated[i] = y[i] * (g / (1.0f + exp2f(-g * 1.4426950408889634f)));
    }
}

extern "C" __global__ void gating_backward(
    float* d_y,            // [n] gradient w.r.t. SSM output
    float* d_gate_pre,     // [n] gradient w.r.t. gate pre-SiLU
    const float* d_gated,  // [n] incoming gradient
    const float* y,        // [n] SSM output (saved)
    const float* gate_pre, // gate pre-SiLU, read through the same geometry
    long long n, int d_inner,
    // Row stride and column offset of the gate half: the gate is read from
    // the in_proj output and its gradient lands directly in the d_proj
    // [bt, 2*d_inner] half, so neither a split nor a concat pass runs.
    int out_stride, int out_offset
) {
    ELEM_LOOP(i, n) {
        long long row = i / d_inner;
        int col = (int)(i - row * d_inner);
        float dg = d_gated[i];
        float x = gate_pre[row * out_stride + out_offset + col];
        // Post-SiLU recomputed in the split kernel's exact form.
        d_y[i] = dg * (x / (1.0f + exp2f(-x * 1.4426950408889634f)));
        // SiLU derivative: sigma * (1 + x * (1 - sigma))
        float sigma = 1.0f / (1.0f + exp2f(-x * 1.4426950408889634f));
        d_gate_pre[row * out_stride + out_offset + col] =
            dg * y[i] * sigma * (1.0f + x * (1.0f - sigma));
    }
}

// gating_backward typed (bf16/f16/f32) for mixed-precision training.
// `y = ssm_out * gate_silu(z)` where z = gate_pre and gate_silu = z*sigma(z).
// All math in f32, activations T_IN (typed). Outputs typed.
// Reference math identical to f32 above; matches state-spaces/mamba
// selective_scan_bwd_kernel.cuh z/gate-branch backward.
#define DEFINE_GATING_BWD(SUFFIX, T, FROM_F)                                   \
extern "C" __global__ void gating_backward_##SUFFIX(                           \
    T* d_y, T* d_gate_pre,                                                     \
    const T* d_gated, const T* y,                                              \
    const T* gate_pre,                                                         \
    long long n, int d_inner, int out_stride, int out_offset                   \
) {                                                                            \
    ELEM_LOOP(i, n) {                                                          \
        long long row = i / d_inner;                                           \
        int col = (int)(i - row * d_inner);                                    \
        float dg = to_f(d_gated[i]);                                           \
        float xv = to_f(gate_pre[row * out_stride + out_offset + col]);        \
        /* Post-SiLU recomputed through the SAME store rounding the        \
           deleted activation carried: FROM_F of the split kernel's        \
           division form, then back to f32 - the exact bits                \
           `gate_post[i]` held. */                                         \
        float gp = to_f(FROM_F(xv / (1.0f + exp2f(-xv * LOG2E))));             \
        d_y[i] = FROM_F(dg * gp);                                              \
        float sigma = 1.0f / (1.0f + exp2f(-xv * 1.4426950408889634f));        \
        d_gate_pre[row * out_stride + out_offset + col] =                      \
            FROM_F(dg * to_f(y[i]) * sigma * (1.0f + xv * (1.0f - sigma)));    \
    }                                                                          \
}

DEFINE_GATING_BWD(f32,  float,         from_f_f32)
DEFINE_GATING_BWD(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_GATING_BWD(f16,  __half,        from_f_f16)

#define DEFINE_GATHER_COLS(SUFFIX, T)                                         \
extern "C" __global__ void gather_cols_##SUFFIX(                              \
    T* dst, const T* src,                                                     \
    int batch, int src_stride, int dst_dim, int offset                        \
) {                                                                           \
    ELEM_LOOP(idx, (long long)batch * dst_dim) {                              \
        long long b = idx / dst_dim;                                          \
        int d = (int)(idx % dst_dim);                                         \
        dst[b * dst_dim + d] = src[b * src_stride + offset + d];              \
    }                                                                         \
}

DEFINE_GATHER_COLS(f32,  float)
DEFINE_GATHER_COLS(bf16, __nv_bfloat16)
DEFINE_GATHER_COLS(f16,  __half)

#define DEFINE_GATHER_BC(SUFFIX, T)                                           \
extern "C" __global__ void gather_bc_cols_##SUFFIX(                           \
    T* dst_b, T* dst_c, const T* src,                                         \
    int batch, int src_stride, int ds, int b_offset, int c_offset             \
) {                                                                           \
    ELEM_LOOP(idx, (long long)batch * ds) {                                   \
        long long b = idx / ds;                                               \
        int d = (int)(idx % ds);                                              \
        long long row = b * src_stride;                                       \
        dst_b[b * ds + d] = src[row + b_offset + d];                          \
        dst_c[b * ds + d] = src[row + c_offset + d];                          \
    }                                                                         \
}

DEFINE_GATHER_BC(f32,  float)
DEFINE_GATHER_BC(bf16, __nv_bfloat16)
DEFINE_GATHER_BC(f16,  __half)

// The B and C columns of the x projection rows, t-major ([b][n][t]) for
// the parallel scan, which reads them per (d, n) lane over consecutive t.
// A block stages GBC_TILE_T tokens of GBC_SLICE_N state indices through
// shared memory and writes t-contiguous runs per n, where a thread an
// element would scatter its writes one 32-byte sector apart. A copy, so
// every value keeps its bits.
//
// Grid (ceil(T / GBC_TILE_T), batch, ceil(ds / GBC_SLICE_N)). Dynamic
// shared memory: 2 min(ds, GBC_SLICE_N) (GBC_TILE_T + 1) elements, rows
// padded by one so the transposed reads fall in distinct banks; 16.5 KB in
// f32 at most, whatever d_state.
#define GBC_TILE_T 32
#define GBC_SLICE_N 64
#define DEFINE_GATHER_BC_TMAJOR_TILED(SUFFIX, T_ACT)                          \
extern "C" __global__ void gather_bc_cols_tmajor_tiled_##SUFFIX(              \
    T_ACT* dst_b, T_ACT* dst_c, const T_ACT* src,                             \
    int T, int src_stride, int ds, int b_offset, int c_offset,                \
    int y_first /* this run's first y block */                                \
) {                                                                           \
    extern __shared__ unsigned char gbc_sm_raw[];                             \
    const int n0 = blockIdx.z * GBC_SLICE_N;                                  \
    const int ns = min(GBC_SLICE_N, ds - n0);                                 \
    T_ACT* smb = (T_ACT*)gbc_sm_raw;                                          \
    T_ACT* smc = smb + ns * (GBC_TILE_T + 1);                                 \
    long long b = y_first + blockIdx.y;                                       \
    int t0 = blockIdx.x * GBC_TILE_T;                                         \
    int nt = min(GBC_TILE_T, T - t0);                                         \
    int tile_elems = nt * ns;                                                 \
    for (int idx = threadIdx.x; idx < tile_elems; idx += blockDim.x) {        \
        int tt = idx / ns;                                                    \
        int d = idx % ns;                                                     \
        long long row = (b * T + t0 + tt) * src_stride + n0 + d;              \
        smb[d * (GBC_TILE_T + 1) + tt] = src[row + b_offset];                 \
        smc[d * (GBC_TILE_T + 1) + tt] = src[row + c_offset];                 \
    }                                                                         \
    __syncthreads();                                                          \
    for (int idx = threadIdx.x; idx < tile_elems; idx += blockDim.x) {        \
        int d = idx / nt;                                                     \
        int tt = idx % nt;                                                    \
        long long col = (b * ds + n0 + d) * T + t0 + tt;                      \
        dst_b[col] = smb[d * (GBC_TILE_T + 1) + tt];                          \
        dst_c[col] = smc[d * (GBC_TILE_T + 1) + tt];                          \
    }                                                                         \
}

DEFINE_GATHER_BC_TMAJOR_TILED(f32,  float)
DEFINE_GATHER_BC_TMAJOR_TILED(bf16, __nv_bfloat16)
DEFINE_GATHER_BC_TMAJOR_TILED(f16,  __half)

/* Typed gating forward with the SiLU recomputed through its store
 * rounding - reproduces `elementwise_mul(y, gate_post)` exactly. */
#define DEFINE_GATE_MUL_SILU(SUFFIX, T, FROM_F)                               \
extern "C" __global__ void gate_mul_silu_##SUFFIX(                            \
    T* __restrict__ gated, const T* __restrict__ y,                           \
    const T* __restrict__ gate_pre,                                           \
    long long n, int d_inner, int gate_stride, int gate_offset                \
) {                                                                           \
    ELEM_LOOP(i, n) {                                                         \
        long long row = i / d_inner;                                          \
        int col = (int)(i - row * d_inner);                                   \
        float g = to_f(gate_pre[row * gate_stride + gate_offset + col]);      \
        float gp = to_f(FROM_F(g / (1.0f + exp2f(-g * LOG2E))));               \
        gated[i] = FROM_F(to_f(y[i]) * gp);                                   \
    }                                                                         \
}

DEFINE_GATE_MUL_SILU(f32,  float,         from_f_f32)
DEFINE_GATE_MUL_SILU(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_GATE_MUL_SILU(f16,  __half,        from_f_f16)

// 16-byte vectorized twin of gate_mul_silu: one uint4 per operand and the
// same per-element arithmetic in the same order, so every output bit is
// unchanged (the class is instruction-bound on 2-byte accesses, see
// elementwise_mul_v in elementwise.cu). The launcher routes here only when
// the element count divides the vector width and every operand pointer is
// 16-byte aligned; otherwise the scalar kernel runs.
#define DEFINE_GATE_MUL_SILU_V(SUFFIX, T, FROM_F)                             \
extern "C" __global__ void gate_mul_silu_v_##SUFFIX(                          \
    T* __restrict__ gated, const T* __restrict__ y,                           \
    const T* __restrict__ gate_pre,                                           \
    long long n_vec, int d_inner, int gate_stride, int gate_offset            \
) {                                                                           \
    const int NPV = 16 / (int)sizeof(T);                                      \
    ELEM_LOOP(i, n_vec) {                                                     \
        /* One vector never straddles a row: the launcher requires the        \
           vector width to divide d_inner. */                                 \
        long long e = i * NPV;                                                \
        long long row = e / d_inner;                                          \
        int col = (int)(e - row * d_inner);                                   \
        const T* grow = gate_pre + row * gate_stride + gate_offset + col;     \
        uint4 yv = reinterpret_cast<const uint4*>(y)[i];                      \
        uint4 gv = *reinterpret_cast<const uint4*>(grow);                     \
        uint4 ov;                                                             \
        const T* yp = reinterpret_cast<const T*>(&yv);                        \
        const T* gp = reinterpret_cast<const T*>(&gv);                        \
        T* op = reinterpret_cast<T*>(&ov);                                    \
        for (int k = 0; k < NPV; k++) {                                       \
            float g = to_f(gp[k]);                                            \
            float sl = to_f(FROM_F(g / (1.0f + exp2f(-g * LOG2E))));          \
            op[k] = FROM_F(to_f(yp[k]) * sl);                                 \
        }                                                                     \
        reinterpret_cast<uint4*>(gated)[i] = ov;                              \
    }                                                                         \
}

DEFINE_GATE_MUL_SILU_V(f32,  float,         from_f_f32)
DEFINE_GATE_MUL_SILU_V(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_GATE_MUL_SILU_V(f16,  __half,        from_f_f16)

#define DEFINE_SOFTPLUS_COPY(SUFFIX, T, FROM_F)                               \
extern "C" __global__ void softplus_copy_##SUFFIX(                            \
    T* dst, const T* src, long long n                                         \
) {                                                                           \
    ELEM_LOOP(i, n) {                                                         \
        float x = to_f(src[i]);                                               \
        dst[i] = FROM_F(x > 20.0f ? x : log1pf(exp2f(x * LOG2E)));            \
    }                                                                         \
}

DEFINE_SOFTPLUS_COPY(f32,  float,         from_f_f32)
DEFINE_SOFTPLUS_COPY(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_SOFTPLUS_COPY(f16,  __half,        from_f_f16)

// Typed cast-from-f32 replacing the zero() + vec_add_inplace staging
// idiom (dst = FROM_F(0 + src)). The `0.0f +` is DELIBERATE and must
// stay: it reproduces today's add-on-zeroed-destination bits exactly —
// a bare FROM_F(src[i]) differs at src = -0.0 (+0.0 vs -0.0).
#define DEFINE_VEC_CAST_ZPLUS(SUFFIX, TY, FROM_F)                             \
extern "C" __global__ void vec_cast_zplus_##SUFFIX(                           \
    TY* dst, const float* src, long long n                                    \
) {                                                                           \
    ELEM_LOOP(i, n) dst[i] = FROM_F(0.0f + src[i]);                           \
}

DEFINE_VEC_CAST_ZPLUS(f32,  float,         from_f_f32)
DEFINE_VEC_CAST_ZPLUS(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_VEC_CAST_ZPLUS(f16,  __half,        from_f_f16)

// Typed concat_halves — mirrors f32 `concat_halves` for the mixed backward
// wiring where both `first_half` and `second_half` are typed gradient
// scratches (d_x_branch, d_gate) and the output `proj` feeds a typed dW
// GEMM via in_proj backward. Pure-load/store op, no arithmetic.
#define DEFINE_CONCAT_HALVES(SUFFIX, TY)                                      \
extern "C" __global__ void concat_halves_##SUFFIX(                            \
    TY* proj,                                                                 \
    const TY* first_half,                                                     \
    const TY* second_half,                                                    \
    int batch, int d_inner                                                    \
) {                                                                           \
    ELEM_LOOP(idx, (long long)batch * d_inner) {                              \
        long long b = idx / d_inner;                                          \
        int d = (int)(idx % d_inner);                                         \
        long long proj_off = b * 2 * d_inner;                                 \
        proj[proj_off + d] = first_half[idx];                                 \
        proj[proj_off + d_inner + d] = second_half[idx];                      \
    }                                                                         \
}

DEFINE_CONCAT_HALVES(f32,  float)
DEFINE_CONCAT_HALVES(bf16, __nv_bfloat16)
DEFINE_CONCAT_HALVES(f16,  __half)

// Typed scatter_add_cols — for mixed backward: d_xdbl (typed) accumulates
// d_delta_raw / d_B / d_C slices (typed). Accumulate in f32 then downcast
// to avoid compounded bf16 round-off during successive scatters.
#define DEFINE_SCATTER_ADD_COLS(SUFFIX, TY, FROM_F)                           \
extern "C" __global__ void scatter_add_cols_##SUFFIX(                         \
    TY* dst, const TY* src,                                                   \
    int batch, int dst_stride, int src_dim, int offset                        \
) {                                                                           \
    ELEM_LOOP(idx, (long long)batch * src_dim) {                              \
        long long b = idx / src_dim;                                          \
        int d = (int)(idx % src_dim);                                         \
        long long di = b * dst_stride + offset + d;                           \
        dst[di] = FROM_F(to_f(dst[di]) + to_f(src[idx]));                     \
    }                                                                         \
}

DEFINE_SCATTER_ADD_COLS(f32,  float,         from_f_f32)
DEFINE_SCATTER_ADD_COLS(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_SCATTER_ADD_COLS(f16,  __half,        from_f_f16)

// d_xdbl assembly in ONE kernel: the three column ranges [0..dt_rank),
// [dt_rank..dt_rank+ds), [dt_rank+ds..dt_rank+2ds) exactly tile the
// xdbl row, so the old zero + per-range scatter staging (f32 lane:
// 1 memset + 3 scatter_add; mixed lane: 2 memsets + 2 casts + 3
// scatter_add) collapses into a single full-domain store. The dt source
// is already the compute dtype (d_dt_input), B/C sources are the f32
// reduce outputs. Every store keeps the FROM_F(0.0f + x) form —
// bit-identical to the old zero+add chains including the -0.0 class,
// and identical to the old double-round (rounding an already-rounded
// value is the identity).
#define DEFINE_PACK_XDBL_COLS(SUFFIX, TY, FROM_F)                             \
extern "C" __global__ void pack_xdbl_cols_##SUFFIX(                           \
    TY* dst,               /* [batch * (dt_rank + 2*d_state)] */              \
    const TY* dt_src,      /* [batch * dt_rank] */                            \
    const float* b_src,    /* [batch * d_state] */                            \
    const float* c_src,    /* [batch * d_state] */                            \
    int batch, int dt_rank, int d_state                                      \
) {                                                                           \
    int w = dt_rank + 2 * d_state;                                            \
    ELEM_LOOP(idx, (long long)batch * w) {                                    \
        long long b = idx / w;                                                \
        int c = (int)(idx % w);                                               \
        float v;                                                              \
        if (c < dt_rank) {                                                    \
            v = to_f(dt_src[b * dt_rank + c]);                                \
        } else if (c < dt_rank + d_state) {                                   \
            v = b_src[b * d_state + (c - dt_rank)];                           \
        } else {                                                              \
            v = c_src[b * d_state + (c - dt_rank - d_state)];                 \
        }                                                                     \
        dst[idx] = FROM_F(0.0f + v);                                          \
    }                                                                         \
}

DEFINE_PACK_XDBL_COLS(f32,  float,         from_f_f32)
DEFINE_PACK_XDBL_COLS(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_PACK_XDBL_COLS(f16,  __half,        from_f_f16)

// FalconMamba's mixer norms: the dt, B and C parts of every x_proj row,
// of widths dt_rank, ds and ds, each divided by its root mean square,
// v / sqrt(mean(v^2) + eps), in place. One block of three warps per row,
// warp p on part p. Lane j < 16 sums the squares of elements j, j + 16,
// ... with fused multiply-adds in ascending order, and the sixteen lane
// sums fold halves, quarters, pairs and the last pair: the host's
// sixteen-lane order, so the scale is the host's bit for bit. `rstd`
// (rows x 3) keeps the scales for the backward, or is null.
__device__ __forceinline__ float bcdt_fold16(float acc) {
    acc = __fadd_rn(acc, __shfl_down_sync(0xffffffffu, acc, 8));
    acc = __fadd_rn(acc, __shfl_down_sync(0xffffffffu, acc, 4));
    acc = __fadd_rn(acc, __shfl_down_sync(0xffffffffu, acc, 2));
    acc = __fadd_rn(acc, __shfl_down_sync(0xffffffffu, acc, 1));
    return __shfl_sync(0xffffffffu, acc, 0);
}

#define DEFINE_BCDT_NORM_FWD(SUFFIX, T, FROM_F)                               \
extern "C" __global__ void bcdt_norm_fwd_##SUFFIX(                            \
    T* __restrict__ xdbl, float* __restrict__ rstd,                           \
    long long rows, int stride, int dt_rank, int ds, float eps                \
) {                                                                           \
    const int part = threadIdx.x >> 5;                                        \
    const int lane = threadIdx.x & 31;                                        \
    const int lo = part == 0 ? 0 : (part == 1 ? dt_rank : dt_rank + ds);      \
    const int n = part == 0 ? dt_rank : ds;                                   \
    for (long long row = blockIdx.x; row < rows; row += gridDim.x) {          \
        T* v = xdbl + row * (long long)stride + lo;                           \
        float acc = 0.0f;                                                     \
        if (lane < 16) {                                                      \
            for (int i = lane; i < n; i += 16) {                              \
                const float x = to_f(v[i]);                                   \
                acc = __fmaf_rn(x, x, acc);                                   \
            }                                                                 \
        }                                                                     \
        const float sum = bcdt_fold16(acc);                                   \
        const float r = __fdiv_rn(1.0f,                                       \
            __fsqrt_rn(__fadd_rn(__fdiv_rn(sum, (float)n), eps)));            \
        for (int i = lane; i < n; i += 32) {                                  \
            v[i] = FROM_F(__fmul_rn(to_f(v[i]), r));                          \
        }                                                                     \
        if (rstd != nullptr && lane == 0) rstd[row * 3 + part] = r;           \
    }                                                                         \
}

DEFINE_BCDT_NORM_FWD(f32,  float,         from_f_f32)
DEFINE_BCDT_NORM_FWD(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_BCDT_NORM_FWD(f16,  __half,        from_f_f16)

// The d_xdbl assembly of a FalconMamba model: pack_xdbl_cols's dt | B | C
// gradient of the normalized row, taken back through the norms, stored
// once. The gradient dy of a part with normalized values y (the forward's
// stored row) and scale r becomes r (dy - y mean(dy y)), the mean's dot in
// the sixteen-lane order. Block and warp layout as bcdt_norm_fwd.
#define DEFINE_PACK_XDBL_NORM_BWD(SUFFIX, TY, FROM_F)                         \
extern "C" __global__ void pack_xdbl_norm_bwd_##SUFFIX(                       \
    TY* __restrict__ dst,                                                     \
    const TY* __restrict__ dt_src,                                            \
    const float* __restrict__ b_src,                                          \
    const float* __restrict__ c_src,                                          \
    const TY* __restrict__ y,                                                 \
    const float* __restrict__ rstd,                                           \
    long long rows, int dt_rank, int ds                                       \
) {                                                                           \
    const int part = threadIdx.x >> 5;                                        \
    const int lane = threadIdx.x & 31;                                        \
    const int lo = part == 0 ? 0 : (part == 1 ? dt_rank : dt_rank + ds);      \
    const int n = part == 0 ? dt_rank : ds;                                   \
    const long long w = (long long)dt_rank + 2LL * ds;                        \
    for (long long row = blockIdx.x; row < rows; row += gridDim.x) {          \
        const TY* yr = y + row * w + lo;                                      \
        const float r = rstd[row * 3 + part];                                 \
        float acc = 0.0f;                                                     \
        if (lane < 16) {                                                      \
            for (int i = lane; i < n; i += 16) {                              \
                const float g = part == 0                                     \
                    ? __fadd_rn(0.0f, to_f(dt_src[row * dt_rank + i]))        \
                    : __fadd_rn(0.0f, (part == 1 ? b_src : c_src)[row * ds + i]); \
                acc = __fmaf_rn(g, to_f(yr[i]), acc);                         \
            }                                                                 \
        }                                                                     \
        const float m = __fdiv_rn(bcdt_fold16(acc), (float)n);                \
        for (int i = lane; i < n; i += 32) {                                  \
            const float g = part == 0                                         \
                ? __fadd_rn(0.0f, to_f(dt_src[row * dt_rank + i]))            \
                : __fadd_rn(0.0f, (part == 1 ? b_src : c_src)[row * ds + i]); \
            const float d = __fsub_rn(g, __fmul_rn(to_f(yr[i]), m));          \
            dst[row * w + lo + i] = FROM_F(__fmul_rn(r, d));                  \
        }                                                                     \
    }                                                                         \
}

DEFINE_PACK_XDBL_NORM_BWD(f32,  float,         from_f_f32)
DEFINE_PACK_XDBL_NORM_BWD(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_PACK_XDBL_NORM_BWD(f16,  __half,        from_f_f16)
