// Production Fixed SM89 homogeneous-half packed/XOR staging twin.
// Only staging addresses change; exact ascending k16 chain, copy issue
// schedule, bias seed, alpha/beta, conversion and 136-float output stride stay.
// Dynamic shared is 69,632 bytes: packed S2 inputs use 65,536, then the
// unchanged vector epilogue aliases 128*136*4 bytes. Threads=256.
// Every scalar and cp.async staging path uses the SAME tested layout helper.
// Actual GPU output, sanitizer and speed qualification remain required.
// Composed only in the Fixed/sm_89 suffix; every other target remains byte-identical.
// Exports: gemm_bi_nn_fixed_sm89_tc128_swizzle_v1_bf16 and
// gemm_bi_nn_fixed_sm89_tc128_swizzle_v1_f16.
#include "sm89_half_swizzle_layout.cuh"

struct FixedSm89HalfSwizzleParams {
    float alpha;
    float beta;
    int m;
    int n;
    int k;
    int lda;
    int ldb;
    int ldc;
};
static_assert(sizeof(FixedSm89HalfSwizzleParams) == 32, "Fixed SM89 swizzle parameter size");
static_assert(alignof(FixedSm89HalfSwizzleParams) == 4, "Fixed SM89 swizzle parameter alignment");
static_assert(__is_standard_layout(FixedSm89HalfSwizzleParams), "Fixed SM89 swizzle standard layout");
static_assert(sizeof(float) == 4 && sizeof(int) == 4, "Fixed SM89 swizzle field widths");

namespace sm89_fixed_half_swizzle {

namespace layout = sm89_fixed_half_swizzle_layout;
static constexpr int kSharedBytes = layout::kSharedBytes;
static constexpr int kOutputStride = 136;
static_assert(2 * (128 * 64 + 64 * 128) * 2 <= kSharedBytes, "packed S2 shared ABI");
static_assert(128 * kOutputStride * 4 <= kSharedBytes, "epilogue aliases S2 storage");
static_assert(sizeof(uint4) == 16, "vector store width");

template <typename T> struct HalfOps;

#define SM89_FHS_OPS(TYPE, FROM, MMA_TYPE)                                    \
template <> struct HalfOps<TYPE> {                                         \
    static __device__ __forceinline__ TYPE from_float(float value) {        \
        return FROM(value);                                                \
    }                                                                      \
    static __device__ __forceinline__ void mma(                             \
        float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {      \
        asm volatile(                                                      \
            "mma.sync.aligned.m16n8k16.row.col.f32." MMA_TYPE "."            \
            MMA_TYPE ".f32 "                                               \
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"       \
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])               \
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),                  \
              "r"(b[0]), "r"(b[1]));                                       \
    }                                                                      \
};

SM89_FHS_OPS(__nv_bfloat16, from_f_bf16, "bf16")
SM89_FHS_OPS(__half, from_f_f16, "f16")
#undef SM89_FHS_OPS

template <typename T>
static __device__ __forceinline__ void stage_scalar(
    T* a_stages, T* b_stages, const T* A, const T* B,
    int stage, int k_base, int pid_m, int pid_n,
    int M, int N, int K, int lda, int ldb) {
    T* a_stage = a_stages + stage * 128 * 64;
    T* b_stage = b_stages + stage * 64 * 128;
    for (int linear = (int)threadIdx.x; linear < 128 * 64; linear += 256) {
        int row = linear / 64;
        int k = linear % 64;
        int global_row = pid_m * 128 + row;
        int global_k = k_base + k;
        a_stage[layout::a_index(row, k)] = global_row < M && global_k < K
            ? A[(long long)global_row * lda + global_k] : HalfOps<T>::from_float(0.0f);
    }
    for (int linear = (int)threadIdx.x; linear < 64 * 128; linear += 256) {
        int k = linear / 128;
        int column = linear % 128;
        int global_k = k_base + k;
        int global_column = pid_n * 128 + column;
        b_stage[layout::b_index(k, column)] = global_k < K && global_column < N
            ? B[(long long)global_k * ldb + global_column] : HalfOps<T>::from_float(0.0f);
    }
}

struct FragmentOffsets {
    unsigned a[4];
    unsigned b[4];
};

struct Fragments {
    unsigned a[4][4];
    unsigned b[4][2];
};

static __device__ __forceinline__ FragmentOffsets fragment_offsets(int warpM, int warpN) {
    FragmentOffsets offsets;
    int lane = (int)threadIdx.x & 31;
#pragma unroll
    for (int atom = 0; atom < 4; ++atom) {
        offsets.a[atom] = layout::a_fragment_base(warpM, atom, lane);
        offsets.b[atom] = layout::b_fragment_base(warpN, atom, lane);
    }
    return offsets;
}

static __device__ __forceinline__ void load_fragments(
    unsigned a_stage, unsigned b_stage, int issue,
    const FragmentOffsets& offsets, Fragments& fragments) {
#pragma unroll
    for (int atom = 0; atom < 4; ++atom) {
        unsigned address = a_stage + layout::a_fragment_issue(offsets.a[atom], issue);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(fragments.a[atom][0]), "=r"(fragments.a[atom][1]),
              "=r"(fragments.a[atom][2]), "=r"(fragments.a[atom][3]) : "r"(address));
    }
#pragma unroll
    for (int atom = 0; atom < 4; ++atom) {
        unsigned address = b_stage + layout::b_fragment_issue(offsets.b[atom], issue);
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
            : "=r"(fragments.b[atom][0]), "=r"(fragments.b[atom][1]) : "r"(address));
    }
}

template <typename T>
static __device__ __forceinline__ void consume_fragments(
    const Fragments& fragments, float (&acc)[4][4][4]) {
#pragma unroll
    for (int fm = 0; fm < 4; ++fm) {
#pragma unroll
        for (int fn = 0; fn < 4; ++fn) {
            HalfOps<T>::mma(acc[fm][fn], fragments.a[fm], fragments.b[fn]);
        }
    }
}

struct CopyPlan {
    long long a_offset[4];
    long long b_offset[4];
    unsigned a_destination[4];
    unsigned b_destination[4];
    bool a_row_valid[4];
    int a_k;
    int b_k;
    int b_column_bytes;
};

static __device__ __forceinline__ CopyPlan copy_plan(
    unsigned a_shared, unsigned b_shared, int pid_m, int pid_n,
    int M, int N, int lda, int ldb) {
    CopyPlan plan;
    int thread = (int)threadIdx.x;
    plan.a_k = (thread & 7) * 8;
    plan.b_k = thread >> 4;
    int local_column = (thread & 15) * 8;
    int global_column = pid_n * 128 + local_column;
    int remaining = N - global_column;
    plan.b_column_bytes = remaining >= 8 ? 16 : (remaining > 0 ? remaining * 2 : 0);
#pragma unroll
    for (int slice = 0; slice < 4; ++slice) {
        int local_row = (thread >> 3) + slice * 32;
        int global_row = pid_m * 128 + local_row;
        plan.a_row_valid[slice] = global_row < M;
        plan.a_offset[slice] = (long long)(global_row < M ? global_row : 0) * lda + plan.a_k;
        plan.b_offset[slice] = (long long)(plan.b_k + slice * 16) * ldb
            + (remaining > 0 ? global_column : 0);
        plan.a_destination[slice] = a_shared + layout::a_copy_offset(thread, slice);
        plan.b_destination[slice] = b_shared + layout::b_copy_offset(thread, slice);
    }
    return plan;
}

template <typename T>
static __device__ __forceinline__ void copy_slice(
    const CopyPlan& plan, const T* A, const T* B,
    int stage, int k_base, long long b_reduction_offset, int K, int slice) {
    int remaining = K - k_base - plan.a_k;
    int a_bytes = plan.a_row_valid[slice]
        ? (remaining >= 8 ? 16 : (remaining > 0 ? remaining * 2 : 0)) : 0;
    int b_bytes = k_base + plan.b_k + slice * 16 < K ? plan.b_column_bytes : 0;
    // Only form an in-object source when at least one element is read.
    const void* a_source = a_bytes > 0 ? (const void*)(A + plan.a_offset[slice] + k_base) : (const void*)A;
    const void* b_source = b_bytes > 0 ? (const void*)(B + plan.b_offset[slice] + b_reduction_offset) : (const void*)B;
    unsigned a_destination = plan.a_destination[slice] + (unsigned)(stage * 128 * 64 * 2);
    unsigned b_destination = plan.b_destination[slice] + (unsigned)(stage * 64 * 128 * 2);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
        :: "r"(a_destination), "l"(a_source), "r"(a_bytes));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
        :: "r"(b_destination), "l"(b_source), "r"(b_bytes));
}

template <typename T>
static __device__ __forceinline__ void scalar_epilogue(
    T* C, float (&acc)[4][4][4], float alpha, float beta,
    int M, int N, int ldc, int pid_m, int pid_n, int warpM, int warpN) {
    int lane = (int)threadIdx.x & 31;
    int g = lane >> 2;
    int t = lane & 3;
#pragma unroll
    for (int fm = 0; fm < 4; ++fm) {
#pragma unroll
        for (int fn = 0; fn < 4; ++fn) {
            int r0 = pid_m * 128 + warpM + fm * 16 + g;
            int c0 = pid_n * 128 + warpN + fn * 8 + 2 * t;
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                int row = r0 + half * 8;
                if (row >= M) continue;
                float first = __fmul_rn(alpha, acc[fm][fn][2 * half]);
                float second = __fmul_rn(alpha, acc[fm][fn][2 * half + 1]);
                T* destination = c0 < N ? C + (long long)row * ldc + c0 : (T*)0;
                if (beta == 0.0f && (ldc & 1) == 0 && c0 + 1 < N && gbf_aligned4(destination)) {
                    gbf_store_pair_rne(destination, first, second);
                } else {
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        int column = c0 + e;
                        if (column >= N) continue;
                        float value = e ? second : first;
                        T* output = C + (long long)row * ldc + column;
                        if (beta != 0.0f) value = __fmaf_rn(beta, to_f(*output), value);
                        *output = HalfOps<T>::from_float(value);
                    }
                }
            }
        }
    }
}

template <typename T>
static __device__ __forceinline__ void vector_epilogue(
    T* C, float* output_tile, float (&acc)[4][4][4], float alpha,
    int M, int ldc, int pid_m, int pid_n, int warpM, int warpN) {
    int lane = (int)threadIdx.x & 31;
    int g = lane >> 2;
    int t = lane & 3;
    // Every warp must finish its last ldmatrix reads before aliasing As/Bs.
    __syncthreads();
#pragma unroll
    for (int fm = 0; fm < 4; ++fm) {
#pragma unroll
        for (int fn = 0; fn < 4; ++fn) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                int row = warpM + fm * 16 + g + half * 8;
                int column = warpN + fn * 8 + 2 * t;
                *reinterpret_cast<float2*>(output_tile + row * kOutputStride + column) =
                    make_float2(acc[fm][fn][2 * half], acc[fm][fn][2 * half + 1]);
            }
        }
    }
    __syncthreads();
    // Each warp writes two adjacent rows in 16-byte chunks. The shared row
    // padding preserves 16-byte addresses for both float4 reads.
#pragma unroll
    for (int linear = (int)threadIdx.x; linear < 128 * 16; linear += 256) {
        int local_row = linear >> 4;
        int row = pid_m * 128 + local_row;
        if (row >= M) continue;
        int column = (linear & 15) * 8;
        const float* source = output_tile + local_row * kOutputStride + column;
        float4 first = *reinterpret_cast<const float4*>(source);
        float4 second = *reinterpret_cast<const float4*>(source + 4);
        uint4 packed;
        gbf_store_pair_rne(reinterpret_cast<T*>(&packed.x), __fmul_rn(alpha, first.x), __fmul_rn(alpha, first.y));
        gbf_store_pair_rne(reinterpret_cast<T*>(&packed.y), __fmul_rn(alpha, first.z), __fmul_rn(alpha, first.w));
        gbf_store_pair_rne(reinterpret_cast<T*>(&packed.z), __fmul_rn(alpha, second.x), __fmul_rn(alpha, second.y));
        gbf_store_pair_rne(reinterpret_cast<T*>(&packed.w), __fmul_rn(alpha, second.z), __fmul_rn(alpha, second.w));
        *reinterpret_cast<uint4*>(C + (long long)row * ldc + pid_n * 128 + column) = packed;
    }
}

template <typename T>
static __device__ __forceinline__ void kernel(
    T* C, const T* A, const T* B, const float* bias,
    float alpha, float beta, int M, int N, int K, int lda, int ldb, int ldc) {
    static_assert(sizeof(T) == 2, "homogeneous half pipeline only");
    assert(alpha == 1.0f || bias == nullptr);
    extern __shared__ __align__(16) unsigned char sm89_fhs_shared[];
    T (*As)[128][64] = reinterpret_cast<T (*)[128][64]>(sm89_fhs_shared);
    T (*Bs)[64][128] = reinterpret_cast<T (*)[64][128]>(
        sm89_fhs_shared + 2 * 128 * 64 * (int)sizeof(T));
    int num_pid_n = (N + 127) / 128;
    int pid_m = (int)blockIdx.x / num_pid_n;
    int pid_n = (int)blockIdx.x % num_pid_n;
    int warp = (int)threadIdx.x >> 5;
    int warpM = (warp >> 2) * 64;
    int warpN = (warp & 3) * 32;
    int t = (int)threadIdx.x & 3;
    unsigned As_sbase = (unsigned)__cvta_generic_to_shared(&As[0][0][0]);
    unsigned Bs_sbase = (unsigned)__cvta_generic_to_shared(&Bs[0][0][0]);
    bool fast_stage = (lda & 7) == 0 && (ldb & 7) == 0 && gbf_aligned16(A) && gbf_aligned16(B);
    float acc[4][4][4];
#pragma unroll
    for (int fm = 0; fm < 4; ++fm) {
#pragma unroll
        for (int fn = 0; fn < 4; ++fn) {
            int column = pid_n * 128 + warpN + fn * 8 + 2 * t;
            float first = bias != nullptr && column < N ? bias[column] : 0.0f;
            float second = bias != nullptr && column + 1 < N ? bias[column + 1] : 0.0f;
            acc[fm][fn][0] = first;
            acc[fm][fn][1] = second;
            acc[fm][fn][2] = first;
            acc[fm][fn][3] = second;
        }
    }
    FragmentOffsets offsets = fragment_offsets(warpM, warpN);
    CopyPlan plan;
    if (fast_stage) plan = copy_plan(As_sbase, Bs_sbase, pid_m, pid_n, M, N, lda, ldb);
    int num_k_tiles = (K + 63) / 64;
    if (num_k_tiles > 0) {
        if (fast_stage) {
#pragma unroll
            for (int slice = 0; slice < 4; ++slice) copy_slice(plan, A, B, 0, 0, 0, K, slice);
            asm volatile("cp.async.commit_group;\n" ::);
        } else {
            stage_scalar(&As[0][0][0], &Bs[0][0][0], A, B, 0, 0, pid_m, pid_n, M, N, K, lda, ldb);
        }
    }
    int read_buf = 0;
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        if (fast_stage) asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        bool next = kt + 1 < num_k_tiles;
        int next_k = (kt + 1) * 64;
        if (next) {
            if (!fast_stage) {
                stage_scalar(&As[0][0][0], &Bs[0][0][0], A, B, read_buf ^ 1, next_k,
                    pid_m, pid_n, M, N, K, lda, ldb);
            }
        }
        unsigned a_read = As_sbase + (unsigned)(read_buf * 128 * 64 * 2);
        unsigned b_read = Bs_sbase + (unsigned)(read_buf * 64 * 128 * 2);
        {
            Fragments fragments[2];
            load_fragments(a_read, b_read, 0, offsets, fragments[0]);
            long long next_b = (long long)next_k * ldb;
#pragma unroll
            for (int issue = 0; issue < 4; ++issue) {
                if (fast_stage && next) copy_slice(plan, A, B, read_buf ^ 1, next_k, next_b, K, issue);
                if (issue < 3) load_fragments(a_read, b_read, issue + 1, offsets, fragments[(issue + 1) & 1]);
                consume_fragments<T>(fragments[issue & 1], acc);
            }
            if (fast_stage && next) asm volatile("cp.async.commit_group;\n" ::);
        }
        read_buf ^= 1;
    }
    {
        bool vector_output = beta == 0.0f && (ldc & 7) == 0 && gbf_aligned16(C)
            && pid_n * 128 + 128 <= N;
        if (vector_output) {
            vector_epilogue(C, reinterpret_cast<float*>(sm89_fhs_shared), acc, alpha,
                M, ldc, pid_m, pid_n, warpM, warpN);
            return;
        }
    }
    scalar_epilogue(C, acc, alpha, beta, M, N, ldc, pid_m, pid_n, warpM, warpN);
}

} // namespace sm89_fixed_half_swizzle

#define SM89_FHS_EXPORT(TYPE, SUFFIX)                                      \
extern "C" __global__ __launch_bounds__(256, 1)                            \
void gemm_bi_nn_fixed_sm89_tc128_swizzle_v1_##SUFFIX(                      \
    TYPE* __restrict__ C, const TYPE* __restrict__ A,                       \
    const TYPE* __restrict__ B, const float* __restrict__ bias,             \
    FixedSm89HalfSwizzleParams params) {                                    \
    sm89_fixed_half_swizzle::kernel<TYPE>(C, A, B, bias, params.alpha,      \
        params.beta, params.m, params.n, params.k, params.lda, params.ldb,   \
        params.ldc);                                                         \
}
SM89_FHS_EXPORT(__nv_bfloat16, bf16)
SM89_FHS_EXPORT(__half, f16)
#undef SM89_FHS_EXPORT
