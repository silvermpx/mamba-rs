#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900

#if __CUDACC_VER_MAJOR__ >= 13
struct alignas(128) CUtensorMap {
#else
struct alignas(64) CUtensorMap {
#endif
    unsigned long long opaque[16];
};
static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap size changed");
#if __CUDACC_VER_MAJOR__ >= 13
static_assert(alignof(CUtensorMap) == 128, "CUtensorMap alignment changed");
#else
static_assert(alignof(CUtensorMap) == 64, "CUtensorMap alignment changed");
#endif

#define SM90A_TILE_M 64
#define SM90A_TILE_N 128
#define SM90A_TILE_K 64
#define SM90A_STAGES 3
#define SM90A_STAGE_BYTES 24576
#define SM90A_DYNAMIC_SHARED_BYTES 73984

enum Sm90aOp {
    Sm90aNn = 0,
    Sm90aTn = 1,
    Sm90aNt = 2,
};

static __device__ __forceinline__ unsigned long long sm90a_desc(
    const void* pointer, unsigned leading_offset, unsigned stride_offset) {
    unsigned long long address =
        static_cast<unsigned long long>(__cvta_generic_to_shared(pointer));
    unsigned long long descriptor = (address >> 4) & 0x3fffULL;
    descriptor |= static_cast<unsigned long long>(leading_offset & 0x3fffU) << 16;
    descriptor |= static_cast<unsigned long long>(stride_offset & 0x3fffU) << 32;
    descriptor |= 1ULL << 62;
    return descriptor;
}

static __device__ __forceinline__ void sm90a_init_barrier(
    unsigned barrier, unsigned arrivals) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                 :: "r"(barrier), "r"(arrivals) : "memory");
}

static __device__ __forceinline__ void sm90a_wait_barrier(
    unsigned barrier, unsigned phase) {
    unsigned ready;
    do {
        asm volatile(
            "{ .reg .pred p; "
            "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2; "
            "selp.b32 %0, 1, 0, p; }"
            : "=r"(ready) : "r"(barrier), "r"(phase) : "memory");
    } while (!ready);
}

static __device__ __forceinline__ void sm90a_arrive_empty(unsigned barrier) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
                 :: "r"(barrier) : "memory");
}

static __device__ __forceinline__ void sm90a_expect_transaction(
    unsigned barrier) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(barrier), "r"(SM90A_STAGE_BYTES) : "memory");
}

static __device__ __forceinline__ void sm90a_tma_copy(
    unsigned destination, unsigned long long map, int x, int y,
    unsigned barrier) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];"
        :: "r"(destination), "l"(map), "r"(x), "r"(y), "r"(barrier)
        : "memory");
}

template <int Op>
static __device__ __forceinline__ void sm90a_produce_stage(
    const CUtensorMap& a_map, const CUtensorMap& b_map, unsigned storage,
    unsigned full_barrier, int tile, int output_row, int output_col) {
    unsigned stage = storage + (tile % SM90A_STAGES) * SM90A_STAGE_BYTES;
    unsigned a_destination = stage;
    unsigned b_destination = stage + 8192;
    unsigned long long a_descriptor =
        reinterpret_cast<unsigned long long>(&a_map);
    unsigned long long b_descriptor =
        reinterpret_cast<unsigned long long>(&b_map);
    int reduction = tile * SM90A_TILE_K;

    sm90a_expect_transaction(full_barrier);
    if (Op == Sm90aNn) {
        sm90a_tma_copy(a_destination, a_descriptor, reduction, output_row,
                       full_barrier);
        // B arrives in two halves because SW128 cannot cover 256 contiguous bytes.
        sm90a_tma_copy(b_destination, b_descriptor, output_col, reduction,
                       full_barrier);
        sm90a_tma_copy(b_destination + 8192, b_descriptor,
                       output_col + 64, reduction, full_barrier);
    } else if (Op == Sm90aTn) {
        sm90a_tma_copy(a_destination, a_descriptor, output_row, reduction,
                       full_barrier);
        sm90a_tma_copy(b_destination, b_descriptor, output_col, reduction,
                       full_barrier);
        sm90a_tma_copy(b_destination + 8192, b_descriptor,
                       output_col + 64, reduction, full_barrier);
    } else {
        sm90a_tma_copy(a_destination, a_descriptor, reduction, output_row,
                       full_barrier);
        sm90a_tma_copy(b_destination, b_descriptor, reduction, output_col,
                       full_barrier);
    }
}

#define SM90A_WGMMA_REGISTERS                                                \
    "%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"              \
    "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"   \
    "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"   \
    "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63"

#define SM90A_WGMMA_OUTPUTS                                                   \
    "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3]),              \
    "+f"(acc[4]), "+f"(acc[5]), "+f"(acc[6]), "+f"(acc[7]),              \
    "+f"(acc[8]), "+f"(acc[9]), "+f"(acc[10]), "+f"(acc[11]),            \
    "+f"(acc[12]), "+f"(acc[13]), "+f"(acc[14]), "+f"(acc[15]),          \
    "+f"(acc[16]), "+f"(acc[17]), "+f"(acc[18]), "+f"(acc[19]),          \
    "+f"(acc[20]), "+f"(acc[21]), "+f"(acc[22]), "+f"(acc[23]),          \
    "+f"(acc[24]), "+f"(acc[25]), "+f"(acc[26]), "+f"(acc[27]),          \
    "+f"(acc[28]), "+f"(acc[29]), "+f"(acc[30]), "+f"(acc[31]),          \
    "+f"(acc[32]), "+f"(acc[33]), "+f"(acc[34]), "+f"(acc[35]),          \
    "+f"(acc[36]), "+f"(acc[37]), "+f"(acc[38]), "+f"(acc[39]),          \
    "+f"(acc[40]), "+f"(acc[41]), "+f"(acc[42]), "+f"(acc[43]),          \
    "+f"(acc[44]), "+f"(acc[45]), "+f"(acc[46]), "+f"(acc[47]),          \
    "+f"(acc[48]), "+f"(acc[49]), "+f"(acc[50]), "+f"(acc[51]),          \
    "+f"(acc[52]), "+f"(acc[53]), "+f"(acc[54]), "+f"(acc[55]),          \
    "+f"(acc[56]), "+f"(acc[57]), "+f"(acc[58]), "+f"(acc[59]),          \
    "+f"(acc[60]), "+f"(acc[61]), "+f"(acc[62]), "+f"(acc[63])

#define SM90A_DEFINE_WGMMA(NAME, TYPE, TRANS_A, TRANS_B)                      \
static __device__ __forceinline__ void NAME(                                  \
    float (&acc)[64], unsigned long long a, unsigned long long b,             \
    int scale_d) {                                                            \
    asm volatile(                                                             \
        "{ .reg .pred p; setp.ne.b32 p, %66, 0; "                            \
        "wgmma.mma_async.sync.aligned.m64n128k16.f32." TYPE "." TYPE        \
        " {" SM90A_WGMMA_REGISTERS "}, %64, %65, p, 1, 1, "                \
        #TRANS_A ", " #TRANS_B "; }"                                        \
        : SM90A_WGMMA_OUTPUTS                                                 \
        : "l"(a), "l"(b), "r"(scale_d));                                    \
}

SM90A_DEFINE_WGMMA(sm90a_wgmma_nn_bf16, "bf16", 0, 1)
SM90A_DEFINE_WGMMA(sm90a_wgmma_tn_bf16, "bf16", 1, 1)
SM90A_DEFINE_WGMMA(sm90a_wgmma_nt_bf16, "bf16", 0, 0)
SM90A_DEFINE_WGMMA(sm90a_wgmma_nn_f16, "f16", 0, 1)
SM90A_DEFINE_WGMMA(sm90a_wgmma_tn_f16, "f16", 1, 1)
SM90A_DEFINE_WGMMA(sm90a_wgmma_nt_f16, "f16", 0, 0)

template <typename T, int Op>
struct Sm90aWgmma;

#define SM90A_BIND_WGMMA(TYPE, OP, FUNCTION)                                 \
template <>                                                                  \
struct Sm90aWgmma<TYPE, OP> {                                                \
    static __device__ __forceinline__ void issue(                            \
        float (&acc)[64], unsigned long long a, unsigned long long b,        \
        int scale_d) {                                                       \
        FUNCTION(acc, a, b, scale_d);                                        \
    }                                                                        \
};

SM90A_BIND_WGMMA(__nv_bfloat16, Sm90aNn, sm90a_wgmma_nn_bf16)
SM90A_BIND_WGMMA(__nv_bfloat16, Sm90aTn, sm90a_wgmma_tn_bf16)
SM90A_BIND_WGMMA(__nv_bfloat16, Sm90aNt, sm90a_wgmma_nt_bf16)
SM90A_BIND_WGMMA(__half, Sm90aNn, sm90a_wgmma_nn_f16)
SM90A_BIND_WGMMA(__half, Sm90aTn, sm90a_wgmma_tn_f16)
SM90A_BIND_WGMMA(__half, Sm90aNt, sm90a_wgmma_nt_f16)

template <int Op>
static __device__ __forceinline__ void sm90a_stage_descriptors(
    unsigned char* storage, int tile, unsigned long long& a,
    unsigned long long& b) {
    unsigned char* stage = storage +
        (tile % SM90A_STAGES) * SM90A_STAGE_BYTES;
    if (Op == Sm90aTn) {
        a = sm90a_desc(stage, 0, 64);
    } else {
        a = sm90a_desc(stage, 1, 64);
    }
    if (Op == Sm90aNt) {
        b = sm90a_desc(stage + 8192, 1, 64);
    } else {
        b = sm90a_desc(stage + 8192, 512, 64);
    }
}

template <typename T, int Op>
static __device__ __forceinline__ void sm90a_issue_tile(
    float (&acc)[64], unsigned char* storage, int tile, int& first_scale) {
    unsigned long long a;
    unsigned long long b;
    sm90a_stage_descriptors<Op>(storage, tile, a, b);
    asm volatile("" : SM90A_WGMMA_OUTPUTS :: "memory");
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#pragma unroll
    for (int slab = 0; slab < 4; ++slab) {
        unsigned a_step = Op == Sm90aTn ? 128U : 2U;
        unsigned b_step = Op == Sm90aNt ? 2U : 128U;
        Sm90aWgmma<T, Op>::issue(acc, a + slab * a_step,
                                 b + slab * b_step, first_scale);
        first_scale = 1;
    }
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
}

template <typename T>
static __device__ __forceinline__ T sm90a_from_float(float value);

template <>
__device__ __forceinline__ __nv_bfloat16 sm90a_from_float(float value) {
    return from_f_bf16(value);
}

template <>
__device__ __forceinline__ __half sm90a_from_float(float value) {
    return from_f_f16(value);
}

template <typename T, int Op>
static __device__ __forceinline__ void sm90a_store_output(
    void* output, const float* bias, float alpha, float beta,
    int output_rows, int output_cols, int output_stride, int row_tile,
    int col_tile, const float (&acc)[64]) {
    int q = threadIdx.x & 3;
    int row8 = (threadIdx.x >> 2) & 7;
    int warp = threadIdx.x >> 5;
#pragma unroll
    for (int reg = 0; reg < 64; reg += 2) {
        int row_half = (reg >> 1) & 1;
        int n_group = reg >> 2;
        int row = row_tile + row8 + 16 * warp + 8 * row_half;
        int col = col_tile + 2 * q + 8 * n_group;
        if (row >= output_rows || col >= output_cols) continue;

        if (Op == Sm90aTn) {
            float* destination = static_cast<float*>(output) +
                static_cast<long long>(row) * output_stride + col;
            float v0 = __fmaf_rn(alpha, acc[reg], destination[0]);
            bool paired = col + 1 < output_cols &&
                (reinterpret_cast<unsigned long long>(destination) & 7ULL) == 0;
            if (paired) {
                float v1 = __fmaf_rn(alpha, acc[reg + 1], destination[1]);
                float2 pair = {v0, v1};
                *reinterpret_cast<float2*>(destination) = pair;
            } else {
                destination[0] = v0;
                if (col + 1 < output_cols) {
                    destination[1] =
                        __fmaf_rn(alpha, acc[reg + 1], destination[1]);
                }
            }
        } else {
            T* destination = static_cast<T*>(output) +
                static_cast<long long>(row) * output_stride + col;
            float v0 = alpha == 1.0f ? acc[reg] : __fmul_rn(alpha, acc[reg]);
            float v1 = alpha == 1.0f ? acc[reg + 1] :
                                       __fmul_rn(alpha, acc[reg + 1]);
            if (Op == Sm90aNn) {
                if (bias != nullptr) {
                    v0 = __fadd_rn(v0, bias[col]);
                    if (col + 1 < output_cols)
                        v1 = __fadd_rn(v1, bias[col + 1]);
                }
                if (beta != 0.0f) {
                    v0 = __fmaf_rn(beta, to_f(destination[0]), v0);
                    if (col + 1 < output_cols)
                        v1 = __fmaf_rn(beta, to_f(destination[1]), v1);
                }
            }
            bool paired = col + 1 < output_cols &&
                (reinterpret_cast<unsigned long long>(destination) & 3ULL) == 0;
            if (paired) {
                sgb_store_pair_rne(destination, v0, v1);
            } else {
                destination[0] = sm90a_from_float<T>(v0);
                if (col + 1 < output_cols)
                    destination[1] = sm90a_from_float<T>(v1);
            }
        }
    }
}

template <int Op>
static __device__ __noinline__ void sm90a_wg2_producer(
    const CUtensorMap& a_map, const CUtensorMap& b_map, int m, int k, int n) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;" ::: "memory");
    __syncthreads();
    if (threadIdx.x == 128) {
        extern __shared__ __align__(1024) unsigned char storage[];
        unsigned shared =
            static_cast<unsigned>(__cvta_generic_to_shared(storage));
        unsigned full_base = shared + SM90A_STAGES * SM90A_STAGE_BYTES;
        unsigned empty_base = full_base + SM90A_STAGES * 8;
        int output_cols = Op == Sm90aNt ? k : n;
        int reduction = Op == Sm90aNn ? k : (Op == Sm90aTn ? m : n);
        int col_tiles = (output_cols - 1) / SM90A_TILE_N + 1;
        int row_tile = (blockIdx.x / col_tiles) * SM90A_TILE_M;
        int col_tile = (blockIdx.x % col_tiles) * SM90A_TILE_N;
        int tile_count = (reduction - 1) / SM90A_TILE_K + 1;
        for (int tile = 0; tile < tile_count; ++tile) {
            int stage = tile % SM90A_STAGES;
            unsigned empty_phase = 1U ^ ((tile / SM90A_STAGES) & 1U);
            sm90a_wait_barrier(empty_base + stage * 8, empty_phase);
            sm90a_produce_stage<Op>(a_map, b_map, shared,
                full_base + stage * 8, tile, row_tile, col_tile);
        }
        int used_stages = tile_count < SM90A_STAGES
            ? tile_count
            : SM90A_STAGES;
        for (int stage = 0; stage < used_stages; ++stage) {
            int last_tile = tile_count - 1 -
                ((tile_count - 1 - stage) % SM90A_STAGES);
            int next_tile = last_tile + SM90A_STAGES;
            unsigned empty_phase =
                1U ^ ((next_tile / SM90A_STAGES) & 1U);
            sm90a_wait_barrier(empty_base + stage * 8, empty_phase);
        }
    }
    __syncthreads();
}

template <typename T, int Op, bool Wg2>
static __device__ __forceinline__ void sm90a_kernel(
    void* output, const CUtensorMap& a_map, const CUtensorMap& b_map,
    const float* bias, float alpha, float beta, int m, int k, int n,
    int output_stride) {
    extern __shared__ __align__(1024) unsigned char storage[];
    unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(storage));
    unsigned full_base = shared + SM90A_STAGES * SM90A_STAGE_BYTES;
    unsigned empty_base = full_base + SM90A_STAGES * 8;
    int issuer = Wg2 ? 128 : 0;
    bool consumer = threadIdx.x < 128;
    int output_rows = Op == Sm90aTn ? k : m;
    int output_cols = Op == Sm90aNt ? k : n;
    int reduction = Op == Sm90aNn ? k : (Op == Sm90aTn ? m : n);
    int col_tiles = (output_cols - 1) / SM90A_TILE_N + 1;
    int row_tile = (blockIdx.x / col_tiles) * SM90A_TILE_M;
    int col_tile = (blockIdx.x % col_tiles) * SM90A_TILE_N;
    int tile_count = (reduction - 1) / SM90A_TILE_K + 1;

    if (threadIdx.x == issuer) {
        for (int stage = 0; stage < SM90A_STAGES; ++stage) {
            sm90a_init_barrier(full_base + stage * 8, 1);
            sm90a_init_barrier(empty_base + stage * 8, 128);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    if (Wg2 && threadIdx.x >= 128) {
        sm90a_wg2_producer<Op>(a_map, b_map, m, k, n);
        return;
    }
    if (Wg2) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 128;" ::: "memory");
        __syncthreads();
    }

    if (!Wg2 && threadIdx.x == issuer) {
        int prologue = tile_count < SM90A_STAGES ? tile_count : SM90A_STAGES;
        for (int tile = 0; tile < prologue; ++tile) {
            int stage = tile % SM90A_STAGES;
            sm90a_wait_barrier(empty_base + stage * 8, 1);
            sm90a_produce_stage<Op>(a_map, b_map, shared,
                full_base + stage * 8, tile, row_tile, col_tile);
        }
    }

    if (consumer) {
        float acc[64] = {};
        int first_scale = 0;
        for (int tile = 0; tile < tile_count; ++tile) {
            int stage = tile % SM90A_STAGES;
            unsigned full_phase = (tile / SM90A_STAGES) & 1U;
            sm90a_wait_barrier(full_base + stage * 8, full_phase);
            sm90a_issue_tile<T, Op>(acc, storage, tile, first_scale);
            if (tile > 0) {
                asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
                int released = (tile - 1) % SM90A_STAGES;
                sm90a_arrive_empty(empty_base + released * 8);
                if (!Wg2 && threadIdx.x == issuer && tile + 2 < tile_count) {
                    int refill = tile + 2;
                    unsigned empty_phase =
                        1U ^ ((refill / SM90A_STAGES) & 1U);
                    sm90a_wait_barrier(empty_base + released * 8, empty_phase);
                    sm90a_produce_stage<Op>(a_map, b_map, shared,
                        full_base + released * 8, refill, row_tile, col_tile);
                }
            }
        }
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
        asm volatile("" : SM90A_WGMMA_OUTPUTS :: "memory");
        if (tile_count > 0) {
            int final_stage = (tile_count - 1) % SM90A_STAGES;
            sm90a_arrive_empty(empty_base + final_stage * 8);
        }
        __syncthreads();
        sm90a_store_output<T, Op>(output, bias, alpha, beta, output_rows,
                                  output_cols, output_stride, row_tile,
                                  col_tile, acc);
    }
}

#define SM90A_DEFINE_WG1(NAME, TYPE, OP)                                     \
extern "C" __global__ __launch_bounds__(128, 3)                             \
void NAME(void* output, const __grid_constant__ CUtensorMap a_map,           \
          const __grid_constant__ CUtensorMap b_map, const float* bias,      \
          float alpha, float beta, int m, int k, int n, int ldc) {           \
    sm90a_kernel<TYPE, OP, false>(output, a_map, b_map, bias, alpha, beta,   \
                                  m, k, n, ldc);                              \
}

#define SM90A_DEFINE_WG2(NAME, TYPE, OP)                                     \
extern "C" __global__ __maxnreg__(128)                                      \
void NAME(void* output, const __grid_constant__ CUtensorMap a_map,           \
          const __grid_constant__ CUtensorMap b_map, const float* bias,      \
          float alpha, float beta, int m, int k, int n, int ldc) {           \
    sm90a_kernel<TYPE, OP, true>(output, a_map, b_map, bias, alpha, beta,    \
                                 m, k, n, ldc);                               \
}

SM90A_DEFINE_WG1(sgemm_bi_nn_sm90a_wgmma_wg1_bf16, __nv_bfloat16, Sm90aNn)
SM90A_DEFINE_WG1(sgemm_bi_nn_sm90a_wgmma_wg1_f16, __half, Sm90aNn)
SM90A_DEFINE_WG1(sgemm_bi_tn_sm90a_wgmma_wg1_bf16, __nv_bfloat16, Sm90aTn)
SM90A_DEFINE_WG1(sgemm_bi_tn_sm90a_wgmma_wg1_f16, __half, Sm90aTn)
SM90A_DEFINE_WG1(sgemm_bi_nt_sm90a_wgmma_wg1_bf16, __nv_bfloat16, Sm90aNt)
SM90A_DEFINE_WG1(sgemm_bi_nt_sm90a_wgmma_wg1_f16, __half, Sm90aNt)
SM90A_DEFINE_WG2(sgemm_bi_nn_sm90a_wgmma_wg2_bf16, __nv_bfloat16, Sm90aNn)
SM90A_DEFINE_WG2(sgemm_bi_nn_sm90a_wgmma_wg2_f16, __half, Sm90aNn)
SM90A_DEFINE_WG2(sgemm_bi_tn_sm90a_wgmma_wg2_bf16, __nv_bfloat16, Sm90aTn)
SM90A_DEFINE_WG2(sgemm_bi_tn_sm90a_wgmma_wg2_f16, __half, Sm90aTn)
SM90A_DEFINE_WG2(sgemm_bi_nt_sm90a_wgmma_wg2_bf16, __nv_bfloat16, Sm90aNt)
SM90A_DEFINE_WG2(sgemm_bi_nt_sm90a_wgmma_wg2_f16, __half, Sm90aNt)

#undef SM90A_DEFINE_WG2
#undef SM90A_DEFINE_WG1
#undef SM90A_BIND_WGMMA
#undef SM90A_DEFINE_WGMMA
#undef SM90A_WGMMA_OUTPUTS
#undef SM90A_WGMMA_REGISTERS
#undef SM90A_DYNAMIC_SHARED_BYTES
#undef SM90A_STAGE_BYTES
#undef SM90A_STAGES
#undef SM90A_TILE_K
#undef SM90A_TILE_N
#undef SM90A_TILE_M

#endif
