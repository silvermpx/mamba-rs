// Mamba-2's conv weight and bias gradients as they were released: a chain
// a (channel, tap) over the rows ascending, the rows staged in shared
// memory four loads a thread at a time and each step loading its operands
// as it ran. The kernel that loads a batch of rows ahead of the chain and
// stages the rows sixteen loads deep is held to it bit for bit. The typed
// prelude and the packed-row helpers precede this fixture; the staging
// helper below is the module's own as it was released.

// Stages `count` values into shared memory through load(i) and put(i, v),
// thread `tid` of `nthr` taking four indices at a time with every load of
// the four issued before any store, so a thread waits one round trip a
// batch rather than one a value. The values are load's, unchanged.
template <class LF, class PF>
__device__ __forceinline__ void m2_stage4(int count, int tid, int nthr, LF load, PF put) {
    for (int i0 = tid; i0 < count; i0 += 4 * nthr) {
        float v[4];
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const int i = i0 + k * nthr;
            v[k] = i < count ? load(i) : 0.0f;
        }
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const int i = i0 + k * nthr;
            if (i < count) put(i, v[k]);
        }
    }
}

// The conv weight and bias gradients, one thread per (c, k), k == d_conv
// standing for the bias: rows ascending (sequence, then token). A block
// takes 32 channels (a lane each, a warp per k) and streams the rows in
// tiles of M2B_DW_ROWS through shared memory: dv [rows][32] and the conv
// inputs [rows + d_conv - 1][32] from the row d_conv - 1 before the tile
// (a token's taps inside its sequence are those rows; the taps before a
// sequence's first token come from state0). Grid (ceil(conv_dim / 32),
// ceil((d_conv + 1) / warps)), block (32, warps) with warps at most 32:
// a block takes the taps blockIdx.y * warps onward and stages the same
// rows as the others. A tap of another sequence of a packed row reads 0,
// as the forward. `g_b` may be null.
#define M2B_DW_ROWS 128
template <typename T>
__device__ __forceinline__ void m2b_conv_dw_body(
    float* __restrict__ g_w, float* __restrict__ g_b,
    const float* __restrict__ dv,
    const float* __restrict__ state0,
    const T* __restrict__ proj, long long ld_proj, int xbc_col,
    int batch, int seq, int conv_dim, int d_conv,
    const int* __restrict__ seq_idx
) {
    extern __shared__ float m2bc_sm[];
    constexpr int CH = 32;
    const int halo = d_conv - 1;
    float* dvs = m2bc_sm;                    // [M2B_DW_ROWS][CH]
    float* xs = dvs + M2B_DW_ROWS * CH;      // [M2B_DW_ROWS + halo][CH]
    const int c0 = blockIdx.x * CH;
    const int lane = threadIdx.x, k = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = c0 + lane;
    const int tid = threadIdx.y * CH + lane, nthr = CH * blockDim.y;
    const long long rows = (long long)batch * seq;
    const bool live = c < conv_dim && (k < d_conv || (k == d_conv && g_b != nullptr));
    float acc = 0.0f;
    long long b = 0;
    int t = 0;
    for (long long r0 = 0; r0 < rows; r0 += M2B_DW_ROWS) {
        const int rn = (int)min((long long)M2B_DW_ROWS, rows - r0);
        __syncthreads();
        m2_stage4((rn + halo) * CH, tid, nthr,
                  [&](int e) {
                      const int rr = e / CH, ch = c0 + e % CH;
                      const long long xrow = r0 - halo + rr;
                      return xrow >= 0 && ch < conv_dim
                          ? to_f(proj[xrow * ld_proj + xbc_col + ch]) : 0.0f;
                  },
                  [&](int e, float v) { xs[e] = v; });
        m2_stage4(rn * CH, tid, nthr,
                  [&](int e) {
                      const int rr = e / CH, ch = c0 + e % CH;
                      return ch < conv_dim ? dv[(r0 + rr) * conv_dim + ch] : 0.0f;
                  },
                  [&](int e, float v) { dvs[e] = v; });
        __syncthreads();
        if (live) {
            for (int rr = 0; rr < rn; rr++) {
                const float g = dvs[rr * CH + lane];
                if (k == d_conv) {
                    acc = __fadd_rn(acc, g);
                } else {
                    const int src = t - halo + k;
                    float xv = src >= 0
                        ? xs[(rr + k) * CH + lane]
                        : state0[(b * conv_dim + c) * d_conv + d_conv + src];
                    if (vl_id(seq_idx, b, seq, src) != vl_id(seq_idx, b, seq, t)) xv = 0.0f;
                    acc = __fmaf_rn(xv, g, acc);
                }
                if (++t == seq) {
                    t = 0;
                    b++;
                }
            }
        }
    }
    if (!live) return;
    if (k == d_conv) {
        g_b[c] = __fadd_rn(g_b[c], acc);
    } else {
        g_w[(long long)c * d_conv + k] = __fadd_rn(g_w[(long long)c * d_conv + k], acc);
    }
}

#define DEFINE_M2B_CONV_DW(NAME, T)                                             \
extern "C" __global__ void NAME(                                                \
    float* __restrict__ g_w,                                                    \
    float* __restrict__ g_b,                                                    \
    const float* __restrict__ dv,                                               \
    const float* __restrict__ state0,                                           \
    const T* __restrict__ proj,                                                 \
    long long ld_proj,                                                          \
    int xbc_col,                                                                \
    int batch,                                                                  \
    int seq,                                                                    \
    int conv_dim,                                                               \
    int d_conv,                                                                 \
    const int* __restrict__ seq_idx                                             \
) {                                                                             \
    m2b_conv_dw_body<T>(g_w, g_b, dv, state0, proj, ld_proj, xbc_col,           \
        batch, seq, conv_dim, d_conv, seq_idx);                                 \
}
DEFINE_M2B_CONV_DW(m2b_reference_conv_dw, float)
DEFINE_M2B_CONV_DW(m2b_reference_conv_dw_bf16, __nv_bfloat16)
DEFINE_M2B_CONV_DW(m2b_reference_conv_dw_f16, __half)
