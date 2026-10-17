// The MIMO chunked backward's angle walk as it was released before the
// batched walk replaced it: a token at a time, two barriers a token and
// lane 0's sum of the token's products. The batched walk is held to it bit
// for bit. The typed prelude and the packed-row helpers precede this
// fixture.

template <typename TI>
__device__ __forceinline__ float mimo_tanh_pi(const TI* pr, int ang_col, int a) {
    return __fmul_rn(tanhf(to_f(pr[ang_col + a])), 3.14159265358979f);
}

// The angles over the window, walked back per (sequence, head): the
// gradient of every angle increment is the sum of the angle gradients from
// its token on, to the end of its packed sequence (the angles restart at
// the next one's first token); d tanh_pi = that sum dt into d_tanh_h (the
// angle projection's backward input), and its share of ddt (sum dt-side, a
// ascending) into ddt_ang [rows, H]. Grid (B, H), block 32: one lane per
// angle, lane 0 sums.
template <typename T>
__device__ __forceinline__ void m3mb_reference_angles_body(
    float* __restrict__ d_tanh_h, float* __restrict__ ddt_ang,
    const float* __restrict__ dth, const float* __restrict__ coef,
    const T* __restrict__ proj, long long ld_proj, int ang_col,
    int seq, int nh, int na,
    int y_first, /* this run's first y block */
    const int* __restrict__ seq_idx
) {
    extern __shared__ float m3mb_sm[];
    float* prod = m3mb_sm;  // [na]
    const int b = blockIdx.x;
    const int h = y_first + blockIdx.y;
    float cum[8];
    const int per = (na + (int)blockDim.x - 1) / (int)blockDim.x;
    for (int k = 0; k < per && k < 8; k++) cum[k] = 0.0f;
    for (int t = seq - 1; t >= 0; t--) {
        const long long row = (long long)b * seq + t;
        const float dt = coef[(row * nh + h) * 4];
        const T* pr = proj + row * ld_proj;
        for (int k = 0; k < per && k < 8; k++) {
            const int a = threadIdx.x + k * blockDim.x;
            if (a < na) {
                cum[k] = __fadd_rn(cum[k], dth[(row * nh + h) * na + a]);
                d_tanh_h[(row * nh + h) * na + a] = __fmul_rn(cum[k], dt);
                prod[a] = __fmul_rn(cum[k], mimo_tanh_pi(pr, ang_col, a));
            }
        }
        if (vl_starts(seq_idx, b, seq, t)) {
            for (int k = 0; k < per && k < 8; k++) cum[k] = 0.0f;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float s = 0.0f;
            for (int a = 0; a < na; a++) s = __fadd_rn(s, prod[a]);
            ddt_ang[row * nh + h] = s;
        }
        __syncthreads();
    }
}

#define DEFINE_M3MB_ANGLES(NAME, T)                                            \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ d_tanh_h, float* __restrict__ ddt_ang,                 \
    const float* __restrict__ dth, const float* __restrict__ coef,             \
    const T* __restrict__ proj, long long ld_proj, int ang_col,                \
    int seq, int nh, int na,                                                   \
    int y_first, /* this run's first y block */                                \
    const int* __restrict__ seq_idx                                            \
) {                                                                            \
    m3mb_reference_angles_body<T>(d_tanh_h, ddt_ang, dth, coef, proj, ld_proj, ang_col,  \
        seq, nh, na, y_first, seq_idx);                                        \
}
DEFINE_M3MB_ANGLES(m3mb_reference_angles, float)
DEFINE_M3MB_ANGLES(m3mb_reference_angles_bf16, __nv_bfloat16)
DEFINE_M3MB_ANGLES(m3mb_reference_angles_f16, __half)
