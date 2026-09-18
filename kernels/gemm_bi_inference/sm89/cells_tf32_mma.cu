// TF32 NN GEMM family on mma.sync m16n8k8 with explicit cvt.rna operands.
//
// Numerically this is the retained wide RNA kernel: f32 operands staged as
// is, rounded with cvt.rna.tf32.f32 when they enter the fragments, bias
// seeded into the accumulator, one m16n8k8 per ascending 8-wide k step,
// alpha=1 stores the accumulator untouched. The tile geometry is chosen per
// shape: 112-row tiles put 126 or 252 tiles on 142 SMs for M=4621 where
// 128-row tiles leave 22 percent idle; 128 x 144 gives 2048 x 2304 exactly
// 256 two-per-SM tiles.
//
// A rows are padded by four floats (ldmatrix phases hit distinct bank
// groups); B rows are padded so the row stride is 32 bytes modulo 128,
// which makes the four k-rows of a B fragment load land in four distinct
// bank quarters (one wavefront per LDS.32).

namespace sm89_cell_tf32_mma {

template <int BM, int BN, int BK, int STAGES, int WM, int WN>
struct Geometry {
    static constexpr int NT = WM * WN * 32;
    static constexpr int WTM = BM / WM;
    static constexpr int WTN = BN / WN;
    static constexpr int MA = WTM / 16;
    static constexpr int NA = WTN / 8;
    static_assert(WTM % 16 == 0 && WTN % 8 == 0, "warp tile must be whole atoms");
    static_assert(BK % 8 == 0, "slab must hold whole k8 steps");
    static constexpr int A_STRIDE = BK + 4;
    static constexpr int B_PAD = ((160 - (BN * 4) % 128) % 128) / 4;
    static constexpr int B_STRIDE = BN + B_PAD;
    static_assert((B_STRIDE * 4) % 128 == 32, "B row stride must be 32 bytes modulo 128");
    static constexpr int A_STAGE = BM * A_STRIDE;
    static constexpr int B_STAGE = BK * B_STRIDE;
    static constexpr int A_CPR = BK / 4;
    static constexpr int B_CPR = BN / 4;
    static constexpr int A_CHUNKS = BM * A_CPR;
    static constexpr int B_CHUNKS = BK * B_CPR;
    static constexpr int A_PASSES = (A_CHUNKS + NT - 1) / NT;
    static constexpr int B_PASSES = (B_CHUNKS + NT - 1) / NT;
    static constexpr int SHARED_BYTES = STAGES * (A_STAGE + B_STAGE) * 4;
};

template <int BM, int BN, int BK, int STAGES, int WM, int WN, bool Prefetch = false>
__device__ __forceinline__ void kernel(
    float* __restrict__ c, const float* __restrict__ a, const float* __restrict__ b,
    const float* __restrict__ bias, FixedSm89HalfParams p) {
    using G = Geometry<BM, BN, BK, STAGES, WM, WN>;
    extern __shared__ __align__(16) float sm89_cell_tf32_shared[];
    float* a_stages = sm89_cell_tf32_shared;
    float* b_stages = sm89_cell_tf32_shared + STAGES * G::A_STAGE;

    const int tiles_n = (p.n + BN - 1) / BN;
    const int tile_m = (int)blockIdx.x / tiles_n;
    const int tile_n = (int)blockIdx.x % tiles_n;
    const int row0 = tile_m * BM;
    const int col0 = tile_n * BN;
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int warp_m = (warp / WN) * G::WTM;
    const int warp_n = (warp % WN) * G::WTN;
    const int g = lane >> 2;
    const int t = lane & 3;

    float acc[G::MA][G::NA][4];
#pragma unroll
    for (int i = 0; i < G::MA; ++i) {
#pragma unroll
        for (int j = 0; j < G::NA; ++j) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int row = row0 + warp_m + i * 16 + g + (e >= 2 ? 8 : 0);
                const int column = col0 + warp_n + j * 8 + 2 * t + (e & 1);
                acc[i][j][e] = row < p.m && column < p.n && bias != nullptr ? bias[column] : 0.0f;
            }
        }
    }

    int a_off[G::A_PASSES];
    unsigned a_dst[G::A_PASSES];
    int a_kc[G::A_PASSES];
    bool a_ok[G::A_PASSES];
#pragma unroll
    for (int pass = 0; pass < G::A_PASSES; ++pass) {
        const int chunk = tid + pass * G::NT;
        const int row = chunk / G::A_CPR;
        const int kc = (chunk % G::A_CPR) * 4;
        const int grow = row0 + row;
        a_ok[pass] = chunk < G::A_CHUNKS && grow < p.m;
        a_off[pass] = a_ok[pass] ? grow * p.lda + kc : 0;
        a_kc[pass] = kc;
        a_dst[pass] = (unsigned)((row * G::A_STRIDE + kc) * 4);
    }
    int b_off[G::B_PASSES];
    unsigned b_dst[G::B_PASSES];
    int b_row[G::B_PASSES];
    int b_bytes[G::B_PASSES];
#pragma unroll
    for (int pass = 0; pass < G::B_PASSES; ++pass) {
        const int chunk = tid + pass * G::NT;
        const int row = chunk / G::B_CPR;
        const int nc = (chunk % G::B_CPR) * 4;
        const int gcol = col0 + nc;
        const int remaining = p.n - gcol;
        b_bytes[pass] = chunk < G::B_CHUNKS && remaining > 0 ? (remaining >= 4 ? 16 : remaining * 4) : 0;
        b_off[pass] = b_bytes[pass] > 0 ? row * p.ldb + gcol : 0;
        b_row[pass] = row;
        b_dst[pass] = (unsigned)((row * G::B_STRIDE + nc) * 4);
    }
    const unsigned a_shared = sm89_cell_smem_addr(a_stages);
    const unsigned b_shared = sm89_cell_smem_addr(b_stages);
    const int num_k_tiles = (p.k + BK - 1) / BK;

    auto issue = [&](int stage, int k0) {
        const unsigned a_base = a_shared + (unsigned)(stage * G::A_STAGE * 4);
        const unsigned b_base = b_shared + (unsigned)(stage * G::B_STAGE * 4);
#pragma unroll
        for (int pass = 0; pass < G::A_PASSES; ++pass) {
            const int remaining = p.k - k0 - a_kc[pass];
            const int bytes = a_ok[pass] && remaining > 0 ? (remaining >= 4 ? 16 : remaining * 4) : 0;
            const float* src = bytes > 0 ? a + a_off[pass] + k0 : a;
            if (G::A_CHUNKS % G::NT == 0 || tid + pass * G::NT < G::A_CHUNKS) {
                sm89_cell_cp_async_16(a_base + a_dst[pass], src, bytes);
            }
        }
#pragma unroll
        for (int pass = 0; pass < G::B_PASSES; ++pass) {
            const int gk = k0 + b_row[pass];
            const int bytes = gk < p.k ? b_bytes[pass] : 0;
            const float* src = bytes > 0 ? b + b_off[pass] + (long long)k0 * p.ldb : b;
            if (G::B_CHUNKS % G::NT == 0 || tid + pass * G::NT < G::B_CHUNKS) {
                sm89_cell_cp_async_16(b_base + b_dst[pass], src, bytes);
            }
        }
    };

    // A fragments through ldmatrix.x4 on the f32 tile: lanes 0-7 rows 0-7 at
    // k 0-3, 8-15 rows 8-15 at k 0-3, 16-23 rows 0-7 at k 4-7, 24-31 rows 8-15
    // at k 4-7, so the four registers are a0..a3 of the m16n8k8 fragment.
    unsigned a_frag_off[G::MA];
#pragma unroll
    for (int i = 0; i < G::MA; ++i) {
        a_frag_off[i] = (unsigned)(((warp_m + i * 16 + (lane & 15)) * G::A_STRIDE + (lane >> 4) * 4) * 4);
    }
    int b_frag_off[G::NA];
#pragma unroll
    for (int j = 0; j < G::NA; ++j) {
        b_frag_off[j] = t * G::B_STRIDE + warp_n + j * 8 + g;
    }

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

        const unsigned a_read = a_shared + (unsigned)((kt % STAGES) * G::A_STAGE * 4);
        const float* b_read = b_stages + (kt % STAGES) * G::B_STAGE;
        // The operands of one k8 step, converted; two sets when the step's
        // loads are issued one step ahead of the MMA chain that consumes them.
        struct Step {
            unsigned a[G::MA][4];
            unsigned b[G::NA][2];
        };
        auto load_step = [&](int ks, Step& step) {
#pragma unroll
            for (int i = 0; i < G::MA; ++i) {
                unsigned raw[4];
                sm89_cell_ldmatrix_x4(a_read + a_frag_off[i] + (unsigned)(ks * 32), raw);
                step.a[i][0] = sm89_cell_tf32(raw[0]);
                step.a[i][1] = sm89_cell_tf32(raw[1]);
                step.a[i][2] = sm89_cell_tf32(raw[2]);
                step.a[i][3] = sm89_cell_tf32(raw[3]);
            }
            const float* b_step = b_read + ks * 8 * G::B_STRIDE;
#pragma unroll
            for (int j = 0; j < G::NA; ++j) {
                step.b[j][0] = sm89_cell_tf32(__float_as_uint(b_step[b_frag_off[j]]));
                step.b[j][1] = sm89_cell_tf32(__float_as_uint(b_step[b_frag_off[j] + 4 * G::B_STRIDE]));
            }
        };
        auto consume = [&](const Step& step) {
#pragma unroll
            for (int i = 0; i < G::MA; ++i) {
#pragma unroll
                for (int j = 0; j < G::NA; ++j) {
                    sm89_cell_mma_tf32(acc[i][j], step.a[i], step.b[j]);
                }
            }
        };
        if constexpr (Prefetch) {
            Step steps[2];
            load_step(0, steps[0]);
#pragma unroll
            for (int ks = 0; ks < BK / 8; ++ks) {
                if (ks + 1 < BK / 8) load_step(ks + 1, steps[(ks + 1) & 1]);
                consume(steps[ks & 1]);
            }
        } else {
#pragma unroll
            for (int ks = 0; ks < BK / 8; ++ks) {
                Step step;
                load_step(ks, step);
                consume(step);
            }
        }
    }

    const bool scale = p.alpha != 1.0f;
    const bool blend = p.beta != 0.0f;
    const bool pair_ok = (p.ldc & 1) == 0 && (reinterpret_cast<unsigned long long>(c) & 7ULL) == 0;
#pragma unroll
    for (int i = 0; i < G::MA; ++i) {
#pragma unroll
        for (int j = 0; j < G::NA; ++j) {
            const int c0 = col0 + warp_n + j * 8 + 2 * t;
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int row = row0 + warp_m + i * 16 + g + half * 8;
                if (row >= p.m) continue;
                float v0 = acc[i][j][2 * half];
                float v1 = acc[i][j][2 * half + 1];
                if (scale) {
                    v0 = __fmul_rn(p.alpha, v0);
                    v1 = __fmul_rn(p.alpha, v1);
                }
                float* dst = c + (long long)row * p.ldc + c0;
                if (c0 + 1 < p.n && pair_ok) {
                    if (blend) {
                        const float2 old = *reinterpret_cast<const float2*>(dst);
                        v0 = __fmaf_rn(p.beta, old.x, v0);
                        v1 = __fmaf_rn(p.beta, old.y, v1);
                    }
                    *reinterpret_cast<float2*>(dst) = make_float2(v0, v1);
                } else {
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        const int column = c0 + e;
                        if (column >= p.n) continue;
                        float value = e ? v1 : v0;
                        if (blend) value = __fmaf_rn(p.beta, dst[e], value);
                        dst[e] = value;
                    }
                }
            }
        }
    }
}

} // namespace sm89_cell_tf32_mma

#define SM89_CELL_TF32_MMA(NAME, BM, BN, BK, STAGES, WM, WN, MIN_BLOCKS)                      \
extern "C" __global__ __launch_bounds__((WM) * (WN) * 32, MIN_BLOCKS)                          \
void NAME(float* __restrict__ c, const float* __restrict__ a, const float* __restrict__ b,    \
          const float* __restrict__ bias, FixedSm89HalfParams p) {                             \
    sm89_cell_tf32_mma::kernel<BM, BN, BK, STAGES, WM, WN>(c, a, b, bias, p);                  \
}

static_assert(sm89_cell_tf32_mma::Geometry<64, 288, 16, 2, 1, 4>::SHARED_BYTES == 48128,
              "the hot_d TF32 cell's shared-memory pin moved");
static_assert(sm89_cell_tf32_mma::Geometry<64, 96, 32, 2, 2, 2>::SHARED_BYTES == 45056,
              "the hot_e TF32 cell's shared-memory pin moved");

// 2048 x 768 x 2304, TF32: 32 x 8 tiles of 64 x 288, two CTAs per SM.
SM89_CELL_TF32_MMA(nn_sm89_m64n288_bk16_s2_tf32, 64, 288, 16, 2, 1, 4, 2)
// 2048 x 2304 x 768 with a bias, TF32: 32 x 8 tiles of 64 x 96.
SM89_CELL_TF32_MMA(nn_sm89_m64n96_bk32_s2_tf32, 64, 96, 32, 2, 2, 2, 2)

#undef SM89_CELL_TF32_MMA
