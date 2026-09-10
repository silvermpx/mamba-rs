// Ada homogeneous-half CTA128x128/BK64/S3. The two-stage provider above
// supplies unchanged layout, fragment, arithmetic and epilogue helpers.
namespace sm89_fixed_half_s3 {
static constexpr int kSharedBytes = 98304;
static_assert(3 * (128 * 64 + 64 * 128) * 2 == kSharedBytes, "S3 shared ABI");
template <typename T>
static __device__ __forceinline__ void kernel(
    T* C, const T* A, const T* B, const float* bias,
    float alpha, float beta, int M, int N, int K, int lda, int ldb, int ldc) {
    static_assert(sizeof(T) == 2, "homogeneous half pipeline only");
    assert(alpha == 1.0f || bias == nullptr);
    extern __shared__ __align__(16) unsigned char sm89_fhs_shared[];
    T (*As)[128][64] = reinterpret_cast<T (*)[128][64]>(sm89_fhs_shared);
    T (*Bs)[64][128] = reinterpret_cast<T (*)[64][128]>(
        sm89_fhs_shared + 3 * 128 * 64 * (int)sizeof(T));
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
    sm89_fixed_half_swizzle::FragmentOffsets offsets = sm89_fixed_half_swizzle::fragment_offsets(warpM, warpN);
    sm89_fixed_half_swizzle::CopyPlan plan;
    if (fast_stage) plan = sm89_fixed_half_swizzle::copy_plan(As_sbase, Bs_sbase, pid_m, pid_n, M, N, lda, ldb);
    int num_k_tiles = (K + 63) / 64;
    if (num_k_tiles > 0) {
        if (fast_stage) {
#pragma unroll
            for (int slice = 0; slice < 4; ++slice) sm89_fixed_half_swizzle::copy_slice(plan, A, B, 0, 0, 0, K, slice);
            asm volatile("cp.async.commit_group;\n" ::);
            if (num_k_tiles > 1) {
#pragma unroll
                for (int slice = 0; slice < 4; ++slice)
                    sm89_fixed_half_swizzle::copy_slice(plan, A, B, 1, 64, (long long)64 * ldb, K, slice);
                asm volatile("cp.async.commit_group;\n" ::);
            }
        } else {
            sm89_fixed_half_swizzle::stage_scalar(&As[0][0][0], &Bs[0][0][0], A, B, 0, 0, pid_m, pid_n, M, N, K, lda, ldb);
        }
    }
    int read_buf = 0;
    if (fast_stage) {
        sm89_fixed_half_swizzle::Fragments fragments[2];
        if (num_k_tiles > 0) {
            if (num_k_tiles > 1) asm volatile("cp.async.wait_group 1;\n" ::);
            else asm volatile("cp.async.wait_group 0;\n" ::);
            __syncthreads();
            sm89_fixed_half_swizzle::load_fragments(As_sbase, Bs_sbase, 0, offsets, fragments[0]);
        }
        int write_buf = 2;
        for (int kt = 0; kt < num_k_tiles; ++kt) {
            bool next = kt + 1 < num_k_tiles;
            bool refill = kt + 2 < num_k_tiles;
            int next_k = (kt + 2) * 64;
            long long next_b = (long long)next_k * ldb;
            unsigned a_read = As_sbase + (unsigned)(read_buf * 128 * 64 * 2);
            unsigned b_read = Bs_sbase + (unsigned)(read_buf * 64 * 128 * 2);

            if (refill) sm89_fixed_half_swizzle::copy_slice(plan, A, B, write_buf, next_k, next_b, K, 0);
            sm89_fixed_half_swizzle::load_fragments(a_read, b_read, 1, offsets, fragments[1]);
            sm89_fixed_half_swizzle::consume_fragments<T>(fragments[0], acc);

            if (refill) sm89_fixed_half_swizzle::copy_slice(plan, A, B, write_buf, next_k, next_b, K, 1);
            sm89_fixed_half_swizzle::load_fragments(a_read, b_read, 2, offsets, fragments[0]);
            sm89_fixed_half_swizzle::consume_fragments<T>(fragments[1], acc);

            if (refill) {
                sm89_fixed_half_swizzle::copy_slice(plan, A, B, write_buf, next_k, next_b, K, 2);
                sm89_fixed_half_swizzle::copy_slice(plan, A, B, write_buf, next_k, next_b, K, 3);
            }
            sm89_fixed_half_swizzle::load_fragments(a_read, b_read, 3, offsets, fragments[1]);
            sm89_fixed_half_swizzle::consume_fragments<T>(fragments[0], acc);

            if (refill) asm volatile("cp.async.commit_group;\n" ::);
            if (next) {
                if (refill) asm volatile("cp.async.wait_group 1;\n" ::);
                else asm volatile("cp.async.wait_group 0;\n" ::);
                __syncthreads();
                read_buf = read_buf == 2 ? 0 : read_buf + 1;
                write_buf = write_buf == 2 ? 0 : write_buf + 1;
                unsigned next_a_read = As_sbase + (unsigned)(read_buf * 128 * 64 * 2);
                unsigned next_b_read = Bs_sbase + (unsigned)(read_buf * 64 * 128 * 2);
                sm89_fixed_half_swizzle::load_fragments(next_a_read, next_b_read, 0, offsets, fragments[0]);
            }
            sm89_fixed_half_swizzle::consume_fragments<T>(fragments[1], acc);
        }
    } else {
        for (int kt = 0; kt < num_k_tiles; ++kt) {
            __syncthreads();
            bool next = kt + 1 < num_k_tiles;
            int next_k = (kt + 1) * 64;
            if (next) {
                sm89_fixed_half_swizzle::stage_scalar(&As[0][0][0], &Bs[0][0][0], A, B, read_buf ^ 1, next_k,
                    pid_m, pid_n, M, N, K, lda, ldb);
            }
            unsigned a_read = As_sbase + (unsigned)(read_buf * 128 * 64 * 2);
            unsigned b_read = Bs_sbase + (unsigned)(read_buf * 64 * 128 * 2);
            sm89_fixed_half_swizzle::Fragments fragments[2];
            sm89_fixed_half_swizzle::load_fragments(a_read, b_read, 0, offsets, fragments[0]);
#pragma unroll
            for (int issue = 0; issue < 4; ++issue) {
                if (issue < 3)
                    sm89_fixed_half_swizzle::load_fragments(a_read, b_read, issue + 1, offsets, fragments[(issue + 1) & 1]);
                sm89_fixed_half_swizzle::consume_fragments<T>(fragments[issue & 1], acc);
            }
            read_buf ^= 1;
        }
    }
    {
        bool vector_output = beta == 0.0f && (ldc & 7) == 0 && gbf_aligned16(C)
            && pid_n * 128 + 128 <= N;
        if (vector_output) {
            sm89_fixed_half_swizzle::vector_epilogue(C, reinterpret_cast<float*>(sm89_fhs_shared), acc, alpha,
                M, ldc, pid_m, pid_n, warpM, warpN);
            return;
        }
    }
    sm89_fixed_half_swizzle::scalar_epilogue(C, acc, alpha, beta, M, N, ldc, pid_m, pid_n, warpM, warpN);
}

} // namespace sm89_fixed_half_s3

extern "C" __global__ __launch_bounds__(256, 1)
void gemm_bi_nn_fixed_sm89_tc128_s3_v1_bf16(
    __nv_bfloat16* __restrict__ C, const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B, const float* __restrict__ bias,
    FixedSm89HalfSwizzleParams params) {
    sm89_fixed_half_s3::kernel(C, A, B, bias, params.alpha, params.beta,
        params.m, params.n, params.k, params.lda, params.ldb, params.ldc);
}
extern "C" __global__ __launch_bounds__(256, 1)
void gemm_bi_nn_fixed_sm89_tc128_s3_v1_f16(
    __half* __restrict__ C, const __half* __restrict__ A,
    const __half* __restrict__ B, const float* __restrict__ bias,
    FixedSm89HalfSwizzleParams params) {
    sm89_fixed_half_s3::kernel(C, A, B, bias, params.alpha, params.beta,
        params.m, params.n, params.k, params.lda, params.ldb, params.ldc);
}
