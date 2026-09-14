namespace sm89_fixed_half_swizzle {
static __device__ __forceinline__ void f32_scalar_epilogue(
    float* C, float (&acc)[4][4][4], float alpha,
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
                float* destination = c0 < N ? C + (long long)row * ldc + c0 : (float*)0;
                if (c0 + 1 < N && (reinterpret_cast<unsigned long long>(destination) & 7ull) == 0ull) {
                    *reinterpret_cast<float2*>(destination) = make_float2(first, second);
                } else {
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        int column = c0 + e;
                        if (column < N) destination[e] = e ? second : first;
                    }
                }
            }
        }
    }
}

static __device__ __forceinline__ void f32_vector_epilogue(
    float* C, float* output_tile, float (&acc)[4][4][4], float alpha,
    int M, int ldc, int pid_m, int pid_n, int warpM, int warpN) {
    int lane = (int)threadIdx.x & 31;
    int g = lane >> 2;
    int t = lane & 3;
    __syncthreads();
#pragma unroll
    for (int fm = 0; fm < 4; ++fm) {
#pragma unroll
        for (int fn = 0; fn < 4; ++fn) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                int row = warpM + fm * 16 + g + half * 8;
                int column = warpN + fn * 8 + 2 * t;
                float first = __fmul_rn(alpha, acc[fm][fn][2 * half]);
                float second = __fmul_rn(alpha, acc[fm][fn][2 * half + 1]);
                *reinterpret_cast<float2*>(output_tile + row * kOutputStride + column) =
                    make_float2(first, second);
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int linear = (int)threadIdx.x; linear < 128 * 32; linear += 256) {
        int local_row = linear >> 5;
        int row = pid_m * 128 + local_row;
        if (row >= M) continue;
        int column = (linear & 31) * 4;
        const float* source = output_tile + local_row * kOutputStride + column;
        float4 values = *reinterpret_cast<const float4*>(source);
        *reinterpret_cast<float4*>(C + (long long)row * ldc + pid_n * 128 + column) = values;
    }
}

} // namespace sm89_fixed_half_swizzle

// This F32-output variant reuses the retained Ada CTA128x128/BK64/S3 mainloop
// so matching half inputs keep its qualified staging and MMA order.
namespace sm89_inference_half_s3_f32out {
static constexpr int kSharedBytes = 98304;
static_assert(3 * (128 * 64 + 64 * 128) * 2 == kSharedBytes, "S3 shared ABI");
template <typename T, bool VectorStore>
static __device__ __forceinline__ void kernel(
    float* C, const T* A, const T* B, const float* bias,
    float alpha, float beta, int M, int N, int K, int lda, int ldb, int ldc) {
    static_assert(sizeof(T) == 2, "half input pipeline only");
    assert(alpha == 1.0f && beta == 0.0f);
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
    bool vector_output = VectorStore && (ldc & 3) == 0 && gbf_aligned16(C)
        && pid_n * 128 + 128 <= N;
    if (vector_output) {
        sm89_fixed_half_swizzle::f32_vector_epilogue(C, reinterpret_cast<float*>(sm89_fhs_shared), acc, alpha,
            M, ldc, pid_m, pid_n, warpM, warpN);
        return;
    }
    sm89_fixed_half_swizzle::f32_scalar_epilogue(C, acc, alpha, M, N, ldc, pid_m, pid_n, warpM, warpN);
}

} // namespace sm89_inference_half_s3_f32out

extern "C" __global__ __launch_bounds__(256, 1)
void gemm_bi_nn_inference_sm89_tc128_f32out_s3_v1_bf16(
    float* __restrict__ C, const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B, const float* __restrict__ bias,
    FixedSm89HalfSwizzleParams params) {
    sm89_inference_half_s3_f32out::kernel<__nv_bfloat16, true>(C, A, B, bias,
        params.alpha, params.beta, params.m, params.n, params.k,
        params.lda, params.ldb, params.ldc);
}
extern "C" __global__ __launch_bounds__(256, 1)
void gemm_bi_nn_inference_sm89_tc128_f32out_s3_v1_f16(
    float* __restrict__ C, const __half* __restrict__ A,
    const __half* __restrict__ B, const float* __restrict__ bias,
    FixedSm89HalfSwizzleParams params) {
    sm89_inference_half_s3_f32out::kernel<__half, true>(C, A, B, bias,
        params.alpha, params.beta, params.m, params.n, params.k,
        params.lda, params.ldb, params.ldc);
}
