// Element-wise CUDA kernels the model families share.
//
// Bias broadcast and reductions, residual add, element-wise products, the
// last-timestep gather. All kernels: 1D grid, 256 threads/block.
//
// Activation-touching kernels are templated via extern "C" wrappers with
// suffixes (_f32, _bf16, _f16). Math in f32, storage in T_IN.

#include "_typed_prelude.cuh"

extern "C" __global__ void bias_broadcast(
    float* y, const float* bias,
    int batch, int n_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * n_out;
    if (idx >= total) return;
    int j = idx % n_out;
    y[idx] = bias[j];
}

// Column sum over the rows. Each column's sum is one serial chain of adds
// in ascending row order held by one owning thread, so the bits are those
// of the plain per-column walk; what changed is how the rows reach it. A
// block covers COLSUM_COLS columns with COLSUM_THREADS threads that stage
// a tile of COLSUM_ROWS rows through shared memory, four rows per thread
// with the next tile's loads in flight while the owners add the current
// one. The walk no longer waits on one load at a time, and a launch of a
// few hundred columns fills a hundred blocks instead of three.
#define COLSUM_COLS 8
#define COLSUM_THREADS 256
#define COLSUM_ROWS 128
extern "C" __global__ __launch_bounds__(COLSUM_THREADS)
void colsum_accumulate(
    float* db, const float* dy,
    int batch, int n_out
) {
    constexpr int TROWS = COLSUM_THREADS / COLSUM_COLS;   // rows a tile step covers
    constexpr int PER_THREAD = COLSUM_ROWS / TROWS;       // rows per thread per tile
    __shared__ float tile[2][COLSUM_ROWS][COLSUM_COLS];
    const int tc = threadIdx.x % COLSUM_COLS;
    const int tr = threadIdx.x / COLSUM_COLS;
    const int j = blockIdx.x * COLSUM_COLS + tc;
    const bool live = j < n_out;
    const int n_tiles = (batch + COLSUM_ROWS - 1) / COLSUM_ROWS;
    float sum = 0.0f;
    _Pragma("unroll")
    for (int q = 0; q < PER_THREAD; q++) {
        int r = tr + q * TROWS;
        tile[0][r][tc] = (live && r < batch)
            ? dy[(long long)r * n_out + j] : 0.0f;
    }
    __syncthreads();
    for (int k = 0; k < n_tiles; k++) {
        const int buf = k & 1;
        // The next tile's rows are requested before this tile's adds and
        // stored after them, into the buffer the adds do not read.
        float next[PER_THREAD];
        _Pragma("unroll")
        for (int q = 0; q < PER_THREAD; q++) {
            int r = (k + 1) * COLSUM_ROWS + tr + q * TROWS;
            next[q] = (k + 1 < n_tiles && live && r < batch)
                ? dy[(long long)r * n_out + j] : 0.0f;
        }
        if (tr == 0) {
            const int rows = min(COLSUM_ROWS, batch - k * COLSUM_ROWS);
            _Pragma("unroll 16")
            for (int r = 0; r < rows; r++) {
                sum += tile[buf][r][tc];
            }
        }
        _Pragma("unroll")
        for (int q = 0; q < PER_THREAD; q++) {
            int r = tr + q * TROWS;
            tile[buf ^ 1][r][tc] = next[q];
        }
        __syncthreads();
    }
    if (tr == 0 && live) db[j] += sum;
}

// Segmented column sum: out[s][j] = sum_t src[s][t][j].
//
// The batched twin of colsum_accumulate for the prefill's pooled route:
// each segment (sample) is summed over its OWN seg_len rows, so samples
// never mix. Per column the accumulation is the identical contract to
// the batch=1 path - ascending t, pure f32 adds seeded at 0.0 - so a
// sample's output is bit-identical whether it rode alone or inside a
// batch. Grid: (ceil(n_out / block), segments).
extern "C" __global__ void colsum_segments(
    float* __restrict__ out,         // [segments * n_out]
    const float* __restrict__ src,   // [segments * seg_len * n_out]
    int segments, int seg_len, int n_out
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y;
    if (j >= n_out || s >= segments) return;
    const float* base = src + (size_t)s * (size_t)seg_len * (size_t)n_out + j;
    size_t stride = (size_t)n_out;
    float sum = 0.0f;
    /* Same ascending serial chain as colsum_accumulate, loads eight deep. */
    int t = 0;
    for (; t + 8 <= seg_len; t += 8) {
        float s0 = base[(size_t)(t + 0) * stride];
        float s1 = base[(size_t)(t + 1) * stride];
        float s2 = base[(size_t)(t + 2) * stride];
        float s3 = base[(size_t)(t + 3) * stride];
        float s4 = base[(size_t)(t + 4) * stride];
        float s5 = base[(size_t)(t + 5) * stride];
        float s6 = base[(size_t)(t + 6) * stride];
        float s7 = base[(size_t)(t + 7) * stride];
        sum += s0;
        sum += s1;
        sum += s2;
        sum += s3;
        sum += s4;
        sum += s5;
        sum += s6;
        sum += s7;
    }
    for (; t < seg_len; t++) {
        sum += base[(size_t)t * stride];
    }
    out[(size_t)s * (size_t)n_out + j] = sum;
}

// Generic 2D reduce-along-axis-0: out[d] = sum_b(partials[b * dim + d]).
// Used as the stage-2 finalizer after Rule-B per-sample partials writes,
// replacing atomicAdd accumulators with a deterministic tree reduction.
// Grid: dim blocks (one per output column). Block: next_pow2(batch).clamp(32, 256).
// Shared memory: block_dim * sizeof(float).
// Deterministic across runs: tree reduce in fixed order, single-thread write.
extern "C" __global__ __launch_bounds__(256, 4)
void reduce_sum_axis0(
    float* __restrict__ out,            // [dim]
    const float* __restrict__ partials, // [batch * dim]
    int batch, int dim,
    int accumulate                      // 0 = overwrite, 1 = +=
) {
    int d = blockIdx.x;
    if (d >= dim) return;
    int tid = threadIdx.x;
    extern __shared__ float sdata[];

    float sum = 0.0f;
    for (int b = tid; b < batch; b += blockDim.x) {
        sum += partials[b * dim + d];
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        out[d] = accumulate ? out[d] + sdata[0] : sdata[0];
    }
}

extern "C" __global__ void fill_scalar(
    float* dst, float val, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = val;
}

extern "C" __global__ void vec_add_inplace(
    float* a, const float* b, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] += b[i];
}

extern "C" __global__ void gather_last_timestep(
    float* __restrict__ dst,      // [B * D]
    const float* __restrict__ src, // [B * T * D]
    int B, int T, int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * D) return;
    int b = idx / D;
    int d = idx % D;
    dst[idx] = src[(b * T + (T - 1)) * D + d];
}

// Templated gather_last_timestep — T_out may differ from T_in (e.g., f32 dst
// from bf16 src for mixed prefill when the downstream lm_head expects f32).
// When dst/src share the same dtype, this is a simple typed copy.
#define DEFINE_GATHER_LAST_TIMESTEP(SUFFIX, T)                                \
extern "C" __global__ void gather_last_timestep_##SUFFIX(                     \
    T* __restrict__ dst,                                                      \
    const T* __restrict__ src,                                                \
    int B, int Tlen, int D                                                    \
) {                                                                           \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                          \
    if (idx >= B * D) return;                                                 \
    int b = idx / D;                                                          \
    int d = idx % D;                                                          \
    dst[idx] = src[(b * Tlen + (Tlen - 1)) * D + d];                          \
}

DEFINE_GATHER_LAST_TIMESTEP(f32,  float)
DEFINE_GATHER_LAST_TIMESTEP(bf16, __nv_bfloat16)
DEFINE_GATHER_LAST_TIMESTEP(f16,  __half)

extern "C" __global__ void residual_add(
    float* dst, const float* a, const float* b, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = a[i] + b[i];
}

// ===========================================================================
// Templated variants for activation-touching kernels.
// Suffix _f32/_bf16/_f16 — Rust dispatch selects by ctx.activation_dtype.
// Bias remains f32 (biases are always f32 in production LLMs).
// ===========================================================================

#define DEFINE_BIAS_BROADCAST(SUFFIX, T, FROM_F)                              \
extern "C" __global__ void bias_broadcast_##SUFFIX(                           \
    T* y, const float* bias, int batch, int n_out                             \
) {                                                                           \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                          \
    int total = batch * n_out;                                                \
    if (idx >= total) return;                                                 \
    int j = idx % n_out;                                                      \
    y[idx] = FROM_F(bias[j]);                                                 \
}

DEFINE_BIAS_BROADCAST(f32,  float,         from_f_f32)
DEFINE_BIAS_BROADCAST(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_BIAS_BROADCAST(f16,  __half,        from_f_f16)

extern "C" __global__ void elementwise_mul(
    float* y, const float* a, const float* b, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = a[i] * b[i];
}

#define DEFINE_ELEMENTWISE_MUL(SUFFIX, T, FROM_F)                             \
extern "C" __global__ void elementwise_mul_##SUFFIX(                          \
    T* y, const T* a, const T* b, int n                                       \
) {                                                                           \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                            \
    if (i >= n) return;                                                       \
    y[i] = FROM_F(to_f(a[i]) * to_f(b[i]));                                   \
}

DEFINE_ELEMENTWISE_MUL(f32,  float,         from_f_f32)
DEFINE_ELEMENTWISE_MUL(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_ELEMENTWISE_MUL(f16,  __half,        from_f_f16)

#define DEFINE_RESIDUAL_ADD(SUFFIX, T, FROM_F)                                \
extern "C" __global__ void residual_add_##SUFFIX(                             \
    T* dst, const T* a, const T* b, int n                                     \
) {                                                                           \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                            \
    if (i >= n) return;                                                       \
    dst[i] = FROM_F(to_f(a[i]) + to_f(b[i]));                                 \
}

DEFINE_RESIDUAL_ADD(f32,  float,         from_f_f32)
DEFINE_RESIDUAL_ADD(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_RESIDUAL_ADD(f16,  __half,        from_f_f16)

// Mixed residual add: f32 residual accumulator + bf16/f16 branch output
// → writes f32 (replaces residual in place or into a new f32 dst).
// Used in end-to-end bf16 inference where `residual_in_fp32` keeps the
// cross-layer residual stream f32 while per-layer branch outputs are bf16.
#define DEFINE_RESIDUAL_ADD_F32_T(SUFFIX, T_IN)                               \
extern "C" __global__ void residual_add_f32_##SUFFIX(                         \
    float* dst, const float* a, const T_IN* b, int n                          \
) {                                                                           \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                            \
    if (i >= n) return;                                                       \
    dst[i] = a[i] + to_f(b[i]);                                               \
}

DEFINE_RESIDUAL_ADD_F32_T(bf16, __nv_bfloat16)
DEFINE_RESIDUAL_ADD_F32_T(f16,  __half)

// ---------------------------------------------------------------------------
// 16-byte vectorized twin of the hot elementwise product.
//
// The class runs at the scalar-copy rate on sm_120 (~2.1 TB/s measured)
// while a uint4-shaped copy of the same bytes reaches ~5.5 TB/s: the
// kernels are instruction-bound on 2-byte accesses, not bandwidth-bound.
// Each thread moves ONE uint4 per operand - 8 bf16/f16 elements or 4
// f32 - and performs the SAME per-element arithmetic in the SAME order,
// so every output bit is unchanged. The launcher routes here only when
// the element count divides the vector width and every operand pointer is
// 16-byte aligned; otherwise the scalar kernel runs unchanged.
// ---------------------------------------------------------------------------

#define DEFINE_ELEMENTWISE_MUL_V(SUFFIX, T, FROM_F)                           \
extern "C" __global__ void elementwise_mul_v_##SUFFIX(                        \
    T* __restrict__ y, const T* __restrict__ a,                               \
    const T* __restrict__ b, int n_vec                                        \
) {                                                                           \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                            \
    if (i >= n_vec) return;                                                   \
    const int NPV = 16 / (int)sizeof(T);                                      \
    uint4 av = reinterpret_cast<const uint4*>(a)[i];                          \
    uint4 bv = reinterpret_cast<const uint4*>(b)[i];                          \
    uint4 ov;                                                                 \
    const T* ap = reinterpret_cast<const T*>(&av);                            \
    const T* bp = reinterpret_cast<const T*>(&bv);                            \
    T* op = reinterpret_cast<T*>(&ov);                                        \
    for (int k = 0; k < NPV; k++) {                                           \
        op[k] = FROM_F(to_f(ap[k]) * to_f(bp[k]));                            \
    }                                                                         \
    reinterpret_cast<uint4*>(y)[i] = ov;                                      \
}

DEFINE_ELEMENTWISE_MUL_V(f32,  float,         from_f_f32)
DEFINE_ELEMENTWISE_MUL_V(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_ELEMENTWISE_MUL_V(f16,  __half,        from_f_f16)

// Typed bias reduction: `d_bias[i] += sum over (b, t) of dy[b, t, i]` where
// `dy` is typed and `d_bias` is f32 master grad. Used by mixed dt_proj
// backward to accumulate the bias gradient. One block per bias index i; one
// warp per block handles the B*T reduction. Each block writes to a distinct
// index `i`, so the final `d_bias[i] += sdata[0]` write has no race —
// deterministic without atomics.
#define DEFINE_REDUCE_BIAS(SUFFIX, TY)                                        \
extern "C" __global__ __launch_bounds__(256, 4)                               \
void reduce_bias_##SUFFIX(                                                    \
    float* __restrict__ d_bias, const TY* __restrict__ dy, int bt, int dim    \
) {                                                                           \
    int i = blockIdx.x;                                                       \
    if (i >= dim) return;                                                     \
    float sum = 0.0f;                                                         \
    for (int r = threadIdx.x; r < bt; r += blockDim.x) {                      \
        sum += to_f(dy[r * dim + i]);                                         \
    }                                                                         \
    /* Warp + block reduction via shared memory. */                           \
    extern __shared__ float sdata[];                                          \
    sdata[threadIdx.x] = sum;                                                 \
    __syncthreads();                                                          \
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {                       \
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];    \
        __syncthreads();                                                      \
    }                                                                         \
    /* No race: each block handles a distinct bias index i. */                \
    if (threadIdx.x == 0) d_bias[i] = d_bias[i] + sdata[0];                   \
}

DEFINE_REDUCE_BIAS(f32,  float)
DEFINE_REDUCE_BIAS(bf16, __nv_bfloat16)
DEFINE_REDUCE_BIAS(f16,  __half)
