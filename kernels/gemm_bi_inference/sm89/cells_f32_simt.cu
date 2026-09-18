// Exact-F32 NN GEMM family on scalar FMA chains.
//
// Every output element is one ascending-k chain of __fmaf_rn starting from
// +0, then alpha, the post-dot bias add and beta, exactly as the retained
// exact kernels do it, so any tile geometry below reproduces their bits.
// The geometry is where the speed comes from: a thread owns TM rows and four
// columns, reads its A values four k at a time (LDS.128 along k) and its B
// values as one LDS.128 along n, so the shared-memory instruction count per
// FMA is small; A rows are padded by four floats so the two row groups a
// warp touches never share a bank line; a multi-stage cp.async ring keeps
// STAGES-1 slabs in flight.
//


namespace sm89_cell_f32_simt {

template <int BM, int BN, int BK, int STAGES, int THREADS, int TN>
struct Geometry {
    static constexpr int GX = BN / TN;
    static constexpr int GY = THREADS / GX;
    static constexpr int TM = BM / GY;
    static_assert(TN == 4 || TN == 6 || TN == 8, "thread columns must be 4, 6 or 8");
    static_assert(GX * TN == BN, "thread columns must tile the tile columns");
    static_assert(THREADS % GX == 0, "thread grid must tile the columns");
    static_assert(GY * TM == BM, "thread grid must tile the rows");
    static constexpr int A_STRIDE = BK + 4;
    static constexpr int A_STAGE = BM * A_STRIDE;
    static constexpr int B_STAGE = BK * BN;
    static constexpr int A_CHUNKS = BM * (BK / 4);
    static constexpr int B_CHUNKS = BK * (BN / 4);
    static constexpr int SHARED_BYTES = STAGES * (A_STAGE + B_STAGE) * 4;
    static_assert(THREADS % (BK / 4) == 0, "copy rows must advance uniformly");
    static_assert(THREADS % (BN / 4) == 0, "copy rows must advance uniformly");
    static constexpr int A_ROWS_PER_PASS = THREADS / (BK / 4);
    static constexpr int B_ROWS_PER_PASS = THREADS / (BN / 4);
    static constexpr int A_PASSES = (A_CHUNKS + THREADS - 1) / THREADS;
    static constexpr int B_PASSES = (B_CHUNKS + THREADS - 1) / THREADS;
};

template <int BM, int BN, int BK, int STAGES, int THREADS, int TN>
__device__ __forceinline__ void kernel(
    float* __restrict__ c, const float* __restrict__ a, const float* __restrict__ b,
    const float* __restrict__ bias, FixedSm89HalfParams p) {
    using G = Geometry<BM, BN, BK, STAGES, THREADS, TN>;
    extern __shared__ __align__(16) float sm89_cell_f32_shared[];
    float* a_stages = sm89_cell_f32_shared;
    float* b_stages = sm89_cell_f32_shared + STAGES * G::A_STAGE;

    const int tiles_n = (p.n + BN - 1) / BN;
    const int tile_m = (int)blockIdx.x / tiles_n;
    const int tile_n = (int)blockIdx.x % tiles_n;
    const int row0 = tile_m * BM;
    const int col0 = tile_n * BN;
    const int tid = (int)threadIdx.x;
    const int tx = tid % G::GX;
    const int ty = tid / G::GX;

    float acc[G::TM][TN];
#pragma unroll
    for (int i = 0; i < G::TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;
    }

    // Copy plan: this thread's A chunks sit at rows a_row + pass * rows-per-pass
    // and k chunk a_kc; its B chunks at k rows b_row + pass * rows-per-pass and
    // column chunk b_cc. Sources advance by BK per slab; lengths clamp to the
    // remaining reduction so the last slab zero-fills.
    const int a_row = tid / (BK / 4);
    const int a_kc = (tid % (BK / 4)) * 4;
    const int b_row = tid / (BN / 4);
    const int b_cc = (tid % (BN / 4)) * 4;
    const int b_col = col0 + b_cc;
    const int b_col_bytes = b_col < p.n ? (p.n - b_col >= 4 ? 16 : (p.n - b_col) * 4) : 0;
    const unsigned a_shared = sm89_cell_smem_addr(a_stages);
    const unsigned b_shared = sm89_cell_smem_addr(b_stages);
    const int num_k_tiles = (p.k + BK - 1) / BK;

    auto issue = [&](int stage, int k0) {
        const unsigned a_dst = a_shared + (unsigned)(stage * G::A_STAGE * 4);
        const unsigned b_dst = b_shared + (unsigned)(stage * G::B_STAGE * 4);
        const int a_k = k0 + a_kc;
        const int a_k_bytes = a_k < p.k ? (p.k - a_k >= 4 ? 16 : (p.k - a_k) * 4) : 0;
#pragma unroll
        for (int pass = 0; pass < G::A_PASSES; ++pass) {
            const int row = a_row + pass * G::A_ROWS_PER_PASS;
            if (row < BM) {
                const int grow = row0 + row;
                const int bytes = grow < p.m ? a_k_bytes : 0;
                const float* src = bytes > 0 ? a + (long long)grow * p.lda + a_k : a;
                sm89_cell_cp_async_16(a_dst + (unsigned)((row * G::A_STRIDE + a_kc) * 4), src, bytes);
            }
        }
#pragma unroll
        for (int pass = 0; pass < G::B_PASSES; ++pass) {
            const int row = b_row + pass * G::B_ROWS_PER_PASS;
            if (row < BK) {
                const int gk = k0 + row;
                const int bytes = gk < p.k ? b_col_bytes : 0;
                const float* src = bytes > 0 ? b + (long long)gk * p.ldb + b_col : b;
                sm89_cell_cp_async_16(b_dst + (unsigned)((row * BN + b_cc) * 4), src, bytes);
            }
        }
    };

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < num_k_tiles) issue(s, s * BK);
        sm89_cell_cp_commit();
    }

    for (int kt = 0; kt < num_k_tiles; ++kt) {
        sm89_cell_cp_wait<STAGES - 2>();
        __syncthreads();
        const int next = kt + STAGES - 1;
        if (next < num_k_tiles) issue(next % STAGES, next * BK);
        sm89_cell_cp_commit();

        const float* a_stage = a_stages + (kt % STAGES) * G::A_STAGE;
        const float* b_stage = b_stages + (kt % STAGES) * G::B_STAGE;
#pragma unroll
        for (int k4 = 0; k4 < BK / 4; ++k4) {
            float4 a_reg[G::TM];
#pragma unroll
            for (int i = 0; i < G::TM; ++i) {
                a_reg[i] = *reinterpret_cast<const float4*>(
                    a_stage + (ty + G::GY * i) * G::A_STRIDE + k4 * 4);
            }
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const float* b_row = b_stage + (k4 * 4 + q) * BN + tx * TN;
                float b_reg[TN];
                if constexpr (TN == 6) {
                    // Six-float spans start on 8-byte boundaries only, so
                    // they are read as three float2.
                    const float2 x0 = *reinterpret_cast<const float2*>(b_row);
                    const float2 x1 = *reinterpret_cast<const float2*>(b_row + 2);
                    const float2 x2 = *reinterpret_cast<const float2*>(b_row + 4);
                    b_reg[0] = x0.x; b_reg[1] = x0.y; b_reg[2] = x1.x;
                    b_reg[3] = x1.y; b_reg[4] = x2.x; b_reg[5] = x2.y;
                } else {
                    const float4 lo = *reinterpret_cast<const float4*>(b_row);
                    b_reg[0] = lo.x; b_reg[1] = lo.y; b_reg[2] = lo.z; b_reg[3] = lo.w;
                    if constexpr (TN == 8) {
                        const float4 hi = *reinterpret_cast<const float4*>(b_row + 4);
                        b_reg[4] = hi.x; b_reg[5] = hi.y; b_reg[6] = hi.z; b_reg[7] = hi.w;
                    }
                }
#pragma unroll
                for (int i = 0; i < G::TM; ++i) {
                    const float av = q == 0 ? a_reg[i].x : (q == 1 ? a_reg[i].y : (q == 2 ? a_reg[i].z : a_reg[i].w));
#pragma unroll
                    for (int j = 0; j < TN; ++j) acc[i][j] = __fmaf_rn(av, b_reg[j], acc[i][j]);
                }
            }
        }
    }

    const int col = col0 + tx * TN;
    // Vector stores need the whole thread column span inside n, an 8-byte
    // aligned row base (16-byte when TN is a multiple of four) and the same
    // alignment on the bias vector; otherwise the scalar path stores each
    // element with the identical arithmetic.
    const bool vector_store = col + TN <= p.n && (p.ldc & 1) == 0
        && (reinterpret_cast<unsigned long long>(c) & 7ULL) == 0
        && (TN == 6 || ((p.ldc & 3) == 0 && (reinterpret_cast<unsigned long long>(c) & 15ULL) == 0));
#pragma unroll
    for (int i = 0; i < G::TM; ++i) {
        const int row = row0 + ty + G::GY * i;
        if (row >= p.m) continue;
        float v[TN];
#pragma unroll
        for (int j = 0; j < TN; ++j) v[j] = __fmul_rn(p.alpha, acc[i][j]);
        if (vector_store) {
            if (bias != nullptr) {
#pragma unroll
                for (int j = 0; j < TN; ++j) v[j] = __fadd_rn(v[j], bias[col + j]);
            }
            float* dst = c + (long long)row * p.ldc + col;
            if (p.beta != 0.0f) {
#pragma unroll
                for (int j = 0; j < TN; ++j) v[j] = __fmaf_rn(p.beta, dst[j], v[j]);
            }
            if constexpr (TN == 6) {
                *reinterpret_cast<float2*>(dst) = make_float2(v[0], v[1]);
                *reinterpret_cast<float2*>(dst + 2) = make_float2(v[2], v[3]);
                *reinterpret_cast<float2*>(dst + 4) = make_float2(v[4], v[5]);
            } else {
#pragma unroll
                for (int j = 0; j < TN; j += 4) {
                    *reinterpret_cast<float4*>(dst + j) = make_float4(v[j], v[j + 1], v[j + 2], v[j + 3]);
                }
            }
        } else {
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int column = col + j;
                if (column >= p.n) continue;
                float value = v[j];
                if (bias != nullptr) value = __fadd_rn(value, bias[column]);
                float* dst = c + (long long)row * p.ldc + column;
                if (p.beta != 0.0f) value = __fmaf_rn(p.beta, *dst, value);
                *dst = value;
            }
        }
    }
}

} // namespace sm89_cell_f32_simt

#define SM89_CELL_F32_SIMT(NAME, BM, BN, BK, STAGES, THREADS, MIN_BLOCKS)                 \
extern "C" __global__ __launch_bounds__(THREADS, MIN_BLOCKS)                                \
void NAME(float* __restrict__ c, const float* __restrict__ a, const float* __restrict__ b, \
          const float* __restrict__ bias, FixedSm89HalfParams p) {                         \
    sm89_cell_f32_simt::kernel<BM, BN, BK, STAGES, THREADS, 4>(c, a, b, bias, p);           \
}

static_assert(sm89_cell_f32_simt::Geometry<112, 128, 32, 3, 256, 4>::SHARED_BYTES == 97536,
              "the hot_c exact cell's shared-memory pin moved");

// 4621 x 1928 x 384, exact F32: 42 x 3 tiles of 112 x 128, two CTAs per SM.
SM89_CELL_F32_SIMT(nn_sm89_m112n128_bk32_s3_f32, 112, 128, 32, 3, 256, 1)

#undef SM89_CELL_F32_SIMT
