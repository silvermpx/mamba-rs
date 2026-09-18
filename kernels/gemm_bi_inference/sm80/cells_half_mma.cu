// Half-input NN GEMM family on mma.sync m16n8k16 with f32 accumulators.
//
// The reduction each output sees is the retained ladder's: bias seeded into
// the accumulator, then one m16n8k16 per ascending 16-wide k step across
// ascending 64-wide (or 32-wide) slabs, zero-filled past K. Only the tile
// geometry, the staging ring and the epilogue transport differ, which is
// what lets a shape pick its tile by SM count instead of by habit: 2048 x
// 2304 splits into 128 x 144 or 64 x 288 tiles that put exactly 256 CTAs on
// 284 two-per-SM slots, where 128 x 128 tiles need 2.03 waves.
//
// Shared memory rows are padded by eight halves so the eight rows of every
// ldmatrix phase land in distinct bank groups; cp.async stages every slab
// but one ahead; the epilogue stores straight from the fragments (float2 for
// f32 output, packed pairs for half output), the same values the incumbent's
// shared-memory transport writes.

namespace sm89_cell_half_mma {

template <int BM, int BN, int BK, int STAGES, int WM, int WN>
struct Geometry {
    static constexpr int NT = WM * WN * 32;
    static constexpr int WTM = BM / WM;
    static constexpr int WTN = BN / WN;
    static constexpr int MA = WTM / 16;
    static constexpr int NA = WTN / 8;
    static_assert(WTM % 16 == 0 && WTN % 8 == 0, "warp tile must be whole atoms");
    static_assert(BK % 16 == 0, "slab must hold whole k16 steps");
    static constexpr int A_STRIDE = BK + 8;
    static constexpr int B_STRIDE = BN + 8;
    static constexpr int A_STAGE = BM * A_STRIDE;
    static constexpr int B_STAGE = BK * B_STRIDE;
    static constexpr int A_CPR = BK / 8;
    static constexpr int B_CPR = BN / 8;
    static constexpr int A_CHUNKS = BM * A_CPR;
    static constexpr int B_CHUNKS = BK * B_CPR;
    static constexpr int A_PASSES = (A_CHUNKS + NT - 1) / NT;
    static constexpr int B_PASSES = (B_CHUNKS + NT - 1) / NT;
    static constexpr int SHARED_BYTES = STAGES * (A_STAGE + B_STAGE) * 2;
};

template <typename TOut>
struct Store;

template <>
struct Store<float> {
    static __device__ __forceinline__ void pair(float* dst, float lo, float hi, bool aligned) {
        if (aligned) {
            *reinterpret_cast<float2*>(dst) = make_float2(lo, hi);
        } else {
            dst[0] = lo;
            dst[1] = hi;
        }
    }
    static __device__ __forceinline__ void one(float* dst, float v) { *dst = v; }
    static __device__ __forceinline__ float load(const float* src) { return *src; }
};

template <>
struct Store<__nv_bfloat16> {
    static __device__ __forceinline__ void pair(__nv_bfloat16* dst, float lo, float hi, bool aligned) {
        if (aligned) {
            *reinterpret_cast<unsigned*>(dst) = Sm89CellHalf<__nv_bfloat16>::pack(lo, hi);
        } else {
            dst[0] = __float2bfloat16_rn(lo);
            dst[1] = __float2bfloat16_rn(hi);
        }
    }
    static __device__ __forceinline__ void one(__nv_bfloat16* dst, float v) { *dst = __float2bfloat16_rn(v); }
    static __device__ __forceinline__ float load(const __nv_bfloat16* src) { return __bfloat162float(*src); }
};

template <>
struct Store<__half> {
    static __device__ __forceinline__ void pair(__half* dst, float lo, float hi, bool aligned) {
        if (aligned) {
            *reinterpret_cast<unsigned*>(dst) = Sm89CellHalf<__half>::pack(lo, hi);
        } else {
            dst[0] = __float2half_rn(lo);
            dst[1] = __float2half_rn(hi);
        }
    }
    static __device__ __forceinline__ void one(__half* dst, float v) { *dst = __float2half_rn(v); }
    static __device__ __forceinline__ float load(const __half* src) { return __half2float(*src); }
};

template <typename T, typename TOut, int BM, int BN, int BK, int STAGES, int WM, int WN,
          bool VectorStore = false, bool Prefetch = false>
__device__ __forceinline__ void kernel(
    TOut* __restrict__ c, const T* __restrict__ a, const T* __restrict__ b,
    const float* __restrict__ bias, FixedSm89HalfParams p) {
    using G = Geometry<BM, BN, BK, STAGES, WM, WN>;
    extern __shared__ __align__(16) unsigned char sm89_cell_half_shared[];
    T* a_stages = reinterpret_cast<T*>(sm89_cell_half_shared);
    T* b_stages = a_stages + STAGES * G::A_STAGE;

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
            const int column = col0 + warp_n + j * 8 + 2 * t;
            const float first = bias != nullptr && column < p.n ? bias[column] : 0.0f;
            const float second = bias != nullptr && column + 1 < p.n ? bias[column + 1] : 0.0f;
            acc[i][j][0] = first;
            acc[i][j][1] = second;
            acc[i][j][2] = first;
            acc[i][j][3] = second;
        }
    }

    // Copy plan: chunk c of this thread's pass sits at (row, chunk column);
    // the element offsets are relative to the operand base and advance with
    // the slab. Rows past M and columns past N copy zero bytes from the base.
    int a_off[G::A_PASSES];
    unsigned a_dst[G::A_PASSES];
    int a_kc[G::A_PASSES];
    bool a_ok[G::A_PASSES];
#pragma unroll
    for (int pass = 0; pass < G::A_PASSES; ++pass) {
        const int chunk = tid + pass * G::NT;
        const int row = chunk / G::A_CPR;
        const int kc = (chunk % G::A_CPR) * 8;
        const int grow = row0 + row;
        a_ok[pass] = chunk < G::A_CHUNKS && grow < p.m;
        a_off[pass] = a_ok[pass] ? grow * p.lda + kc : 0;
        a_kc[pass] = kc;
        a_dst[pass] = (unsigned)((row * G::A_STRIDE + kc) * 2);
    }
    int b_off[G::B_PASSES];
    unsigned b_dst[G::B_PASSES];
    int b_row[G::B_PASSES];
    int b_bytes[G::B_PASSES];
#pragma unroll
    for (int pass = 0; pass < G::B_PASSES; ++pass) {
        const int chunk = tid + pass * G::NT;
        const int row = chunk / G::B_CPR;
        const int nc = (chunk % G::B_CPR) * 8;
        const int gcol = col0 + nc;
        const int remaining = p.n - gcol;
        b_bytes[pass] = chunk < G::B_CHUNKS && remaining > 0 ? (remaining >= 8 ? 16 : remaining * 2) : 0;
        b_off[pass] = b_bytes[pass] > 0 ? row * p.ldb + gcol : 0;
        b_row[pass] = row;
        b_dst[pass] = (unsigned)((row * G::B_STRIDE + nc) * 2);
    }
    const unsigned a_shared = sm89_cell_smem_addr(a_stages);
    const unsigned b_shared = sm89_cell_smem_addr(b_stages);
    const int num_k_tiles = (p.k + BK - 1) / BK;

    auto issue = [&](int stage, int k0) {
        const unsigned a_base = a_shared + (unsigned)(stage * G::A_STAGE * 2);
        const unsigned b_base = b_shared + (unsigned)(stage * G::B_STAGE * 2);
#pragma unroll
        for (int pass = 0; pass < G::A_PASSES; ++pass) {
            const int remaining = p.k - k0 - a_kc[pass];
            const int bytes = a_ok[pass] && remaining > 0 ? (remaining >= 8 ? 16 : remaining * 2) : 0;
            const T* src = bytes > 0 ? a + a_off[pass] + k0 : a;
            if (G::A_CHUNKS % G::NT == 0 || tid + pass * G::NT < G::A_CHUNKS) {
                sm89_cell_cp_async_16(a_base + a_dst[pass], src, bytes);
            }
        }
#pragma unroll
        for (int pass = 0; pass < G::B_PASSES; ++pass) {
            const int gk = k0 + b_row[pass];
            const int bytes = gk < p.k ? b_bytes[pass] : 0;
            const T* src = bytes > 0 ? b + b_off[pass] + (long long)k0 * p.ldb : b;
            if (G::B_CHUNKS % G::NT == 0 || tid + pass * G::NT < G::B_CHUNKS) {
                sm89_cell_cp_async_16(b_base + b_dst[pass], src, bytes);
            }
        }
    };

    // Fragment addresses inside a stage: A rows (lane & 15) of each m-atom at
    // k chunk (lane >> 4); B k-rows (lane & 15) at each n-atom's column.
    unsigned a_frag_off[G::MA];
#pragma unroll
    for (int i = 0; i < G::MA; ++i) {
        a_frag_off[i] = (unsigned)(((warp_m + i * 16 + (lane & 15)) * G::A_STRIDE + (lane >> 4) * 8) * 2);
    }
    unsigned b_frag_off[G::NA];
#pragma unroll
    for (int j = 0; j < G::NA; ++j) {
        b_frag_off[j] = (unsigned)(((lane & 15) * G::B_STRIDE + warp_n + j * 8) * 2);
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

        const unsigned a_read = a_shared + (unsigned)((kt % STAGES) * G::A_STAGE * 2);
        const unsigned b_read = b_shared + (unsigned)((kt % STAGES) * G::B_STAGE * 2);
        struct Step {
            unsigned a[G::MA][4];
            unsigned b[G::NA][2];
        };
        auto load_step = [&](int ks, Step& step) {
#pragma unroll
            for (int i = 0; i < G::MA; ++i) {
                sm89_cell_ldmatrix_x4(a_read + a_frag_off[i] + (unsigned)(ks * 32), step.a[i]);
            }
#pragma unroll
            for (int j = 0; j < G::NA; ++j) {
                sm89_cell_ldmatrix_x2_trans(b_read + b_frag_off[j] + (unsigned)(ks * 16 * G::B_STRIDE * 2), step.b[j]);
            }
        };
        auto consume = [&](const Step& step) {
#pragma unroll
            for (int i = 0; i < G::MA; ++i) {
#pragma unroll
                for (int j = 0; j < G::NA; ++j) {
                    Sm89CellHalf<T>::mma(acc[i][j], step.a[i], step.b[j]);
                }
            }
        };
        if constexpr (Prefetch) {
            Step steps[2];
            load_step(0, steps[0]);
#pragma unroll
            for (int ks = 0; ks < BK / 16; ++ks) {
                if (ks + 1 < BK / 16) load_step(ks + 1, steps[(ks + 1) & 1]);
                consume(steps[ks & 1]);
            }
        } else {
#pragma unroll
            for (int ks = 0; ks < BK / 16; ++ks) {
                Step step;
                load_step(ks, step);
                consume(step);
            }
        }
    }

    // Half output through shared memory: each thread's four accumulator
    // values land in a row-major scratch tile, then every thread carries
    // eight columns of one row out in a single 16-byte store. The values
    // are the ones the per-fragment path would store, computed the same way.
    if constexpr (VectorStore && sizeof(TOut) == 2) {
        constexpr int TILE_STRIDE = BN + 8;
        if (p.beta == 0.0f && (p.ldc & 7) == 0
            && (reinterpret_cast<unsigned long long>(c) & 15ULL) == 0
            && col0 + BN <= p.n) {
            __syncthreads();
            float* tile = reinterpret_cast<float*>(sm89_cell_half_shared);
#pragma unroll
            for (int i = 0; i < G::MA; ++i) {
#pragma unroll
                for (int j = 0; j < G::NA; ++j) {
#pragma unroll
                    for (int half = 0; half < 2; ++half) {
                        const int row = warp_m + i * 16 + g + half * 8;
                        const int column = warp_n + j * 8 + 2 * t;
                        *reinterpret_cast<float2*>(tile + row * TILE_STRIDE + column) =
                            make_float2(acc[i][j][2 * half], acc[i][j][2 * half + 1]);
                    }
                }
            }
            __syncthreads();
            for (int linear = tid; linear < BM * (BN / 8); linear += G::NT) {
                const int local_row = linear / (BN / 8);
                const int row = row0 + local_row;
                if (row >= p.m) continue;
                const int column = (linear % (BN / 8)) * 8;
                const float* source = tile + local_row * TILE_STRIDE + column;
                const float4 first = *reinterpret_cast<const float4*>(source);
                const float4 second = *reinterpret_cast<const float4*>(source + 4);
                uint4 packed;
                packed.x = Sm89CellHalf<T>::pack(__fmul_rn(p.alpha, first.x), __fmul_rn(p.alpha, first.y));
                packed.y = Sm89CellHalf<T>::pack(__fmul_rn(p.alpha, first.z), __fmul_rn(p.alpha, first.w));
                packed.z = Sm89CellHalf<T>::pack(__fmul_rn(p.alpha, second.x), __fmul_rn(p.alpha, second.y));
                packed.w = Sm89CellHalf<T>::pack(__fmul_rn(p.alpha, second.z), __fmul_rn(p.alpha, second.w));
                *reinterpret_cast<uint4*>(c + (long long)row * p.ldc + col0 + column) = packed;
            }
            return;
        }
    }
    const bool pair_ok = (p.ldc & 1) == 0 && (reinterpret_cast<unsigned long long>(c) & 3ULL) == 0;
#pragma unroll
    for (int i = 0; i < G::MA; ++i) {
#pragma unroll
        for (int j = 0; j < G::NA; ++j) {
            const int c0 = col0 + warp_n + j * 8 + 2 * t;
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int row = row0 + warp_m + i * 16 + g + half * 8;
                if (row >= p.m) continue;
                const float v0 = __fmul_rn(p.alpha, acc[i][j][2 * half]);
                const float v1 = __fmul_rn(p.alpha, acc[i][j][2 * half + 1]);
                TOut* dst = c + (long long)row * p.ldc + c0;
                if (p.beta == 0.0f && c0 + 1 < p.n && pair_ok) {
                    Store<TOut>::pair(dst, v0, v1, sizeof(TOut) == 2 || ((reinterpret_cast<unsigned long long>(dst) & 7ULL) == 0));
                } else {
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        const int column = c0 + e;
                        if (column >= p.n) continue;
                        float value = e ? v1 : v0;
                        if (p.beta != 0.0f) value = __fmaf_rn(p.beta, Store<TOut>::load(dst + e), value);
                        Store<TOut>::one(dst + e, value);
                    }
                }
            }
        }
    }
}

} // namespace sm89_cell_half_mma

#define SM89_CELL_HALF_MMA(NAME, T, TOUT, BM, BN, BK, STAGES, WM, WN, MIN_BLOCKS)             \
extern "C" __global__ __launch_bounds__((WM) * (WN) * 32, MIN_BLOCKS)                          \
void NAME(TOUT* __restrict__ c, const T* __restrict__ a, const T* __restrict__ b,             \
          const float* __restrict__ bias, FixedSm89HalfParams p) {                             \
    sm89_cell_half_mma::kernel<T, TOUT, BM, BN, BK, STAGES, WM, WN>(c, a, b, bias, p);         \
}
#define SM89_CELL_HALF_MMA_VEC(NAME, T, BM, BN, BK, STAGES, WM, WN, MIN_BLOCKS)               \
extern "C" __global__ __launch_bounds__((WM) * (WN) * 32, MIN_BLOCKS)                          \
void NAME(T* __restrict__ c, const T* __restrict__ a, const T* __restrict__ b,                \
          const float* __restrict__ bias, FixedSm89HalfParams p) {                             \
    sm89_cell_half_mma::kernel<T, T, BM, BN, BK, STAGES, WM, WN, true>(c, a, b, bias, p);      \
}

static_assert(sm89_cell_half_mma::Geometry<128, 144, 32, 2, 2, 2>::SHARED_BYTES == 39936,
              "the hot_d mixed cell's shared-memory pin moved");
static_assert(sm89_cell_half_mma::Geometry<128, 96, 64, 2, 2, 4>::SHARED_BYTES == 63488,
              "the hot_e half cell's shared-memory pin moved");
static_assert(sm89_cell_half_mma::Geometry<128, 144, 32, 2, 4, 2>::SHARED_BYTES == 39936,
              "the hot_d bf16 cell's shared-memory pin moved");

// 2048 x 768 x 2304, half inputs and f32 output: 16 x 16 tiles of 128 x 144.
SM89_CELL_HALF_MMA(nn_sm89_m128n144_bk32_s2_f32out_bf16, __nv_bfloat16, float, 128, 144, 32, 2, 2, 2, 2)
SM89_CELL_HALF_MMA(nn_sm89_m128n144_bk32_s2_f32out_f16, __half, float, 128, 144, 32, 2, 2, 2, 2)
// 2048 x 2304 x 768, homogeneous half: 16 x 8 tiles of 128 x 96 with packed stores.
SM89_CELL_HALF_MMA_VEC(nn_sm89_m128n96_bk64_s2_vec_bf16, __nv_bfloat16, 128, 96, 64, 2, 2, 4, 1)
SM89_CELL_HALF_MMA_VEC(nn_sm89_m128n96_bk64_s2_vec_f16, __half, 128, 96, 64, 2, 2, 4, 1)
// 2048 x 768 x 2304, bf16 only: the f16 twin loses to the f16 pipeline tile.
SM89_CELL_HALF_MMA_VEC(nn_sm89_m128n144_bk32_s2_vec_bf16, __nv_bfloat16, 128, 144, 32, 2, 4, 2, 1)

#undef SM89_CELL_HALF_MMA_VEC
#undef SM89_CELL_HALF_MMA
