// Ada Fixed F16 N64 finalists. Optional forced routes until qualified.

namespace sm89_fixed_half_n64 {

template <int BM, int STAGES> struct RectTraits {
    static constexpr int kBn = 64;
    static constexpr int kBk = 64;
    static constexpr int kThreads = 128;
    static constexpr int kMAtoms = BM / 32;
    static constexpr int kASlices = BM * kBk / (kThreads * 8);
    static constexpr int kSharedBytes = STAGES * (BM * kBk + kBk * kBn) * 2;
    static constexpr int kOutputStride = 72;
};

template <int BM>
static __device__ __forceinline__ int rect_a_index(int row, int k) {
    return row * 64 + (k ^ ((row & 7) * 8));
}

static __device__ __forceinline__ int rect_b_index(int k, int column) {
    return k * 64 + (column ^ ((k & 7) * 8));
}

template <int BM, int STAGES, bool M_TAIL, typename T>
static __device__ __forceinline__ void rect_copy_issue(
    unsigned a_base, unsigned b_base, const T* A, const T* B,
    int stage, int tile, int pid_m, int pid_n, int M, int lda, int ldb, int issue) {
    using R = RectTraits<BM, STAGES>;
    int thread = (int)threadIdx.x;
#pragma unroll
    for (int q = 0; q < R::kASlices / 4; ++q) {
        int slice = issue + q * 4;
        int linear = thread + slice * R::kThreads;
        int row = linear >> 3;
        int k = (linear & 7) * 8;
        int global_row = pid_m * BM + row;
        int source_row = M_TAIL && global_row >= M ? 0 : global_row;
        const T* source = A + (long long)source_row * lda + tile * 64 + k;
        unsigned destination = a_base + (unsigned)(stage * BM * 64 * 2)
            + (unsigned)(2 * rect_a_index<BM>(row, k));
        if constexpr (M_TAIL) {
            int source_bytes = global_row < M ? 16 : 0;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                :: "r"(destination), "l"(source), "r"(source_bytes));
        } else {
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                :: "r"(destination), "l"(source));
        }
    }
    int linear = thread + issue * R::kThreads;
    int k = linear >> 3;
    int column = (linear & 7) * 8;
    const T* source = B + (long long)(tile * 64 + k) * ldb + pid_n * 64 + column;
    unsigned destination = b_base + (unsigned)(stage * 64 * 64 * 2)
        + (unsigned)(2 * rect_b_index(k, column));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(destination), "l"(source));
}

template <int MATOMS> struct RectFragments {
    unsigned a[MATOMS][4];
    unsigned b[4][2];
};

template <int BM, int STAGES>
static __device__ __forceinline__ void rect_load(
    unsigned a_stage, unsigned b_stage, int warp_m, int warp_n, int issue,
    RectFragments<RectTraits<BM, STAGES>::kMAtoms>& fragments) {
    using R = RectTraits<BM, STAGES>;
    int lane = (int)threadIdx.x & 31;
#pragma unroll
    for (int atom = 0; atom < R::kMAtoms; ++atom) {
        unsigned address = a_stage + ((unsigned)(2 * rect_a_index<BM>(
            warp_m + atom * 16 + (lane & 15), (lane & 16) ? 8 : 0))
            ^ (unsigned)(issue * 32));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(fragments.a[atom][0]), "=r"(fragments.a[atom][1]),
              "=r"(fragments.a[atom][2]), "=r"(fragments.a[atom][3]) : "r"(address));
    }
#pragma unroll
    for (int atom = 0; atom < 4; ++atom) {
        unsigned address = b_stage + (unsigned)(2 * rect_b_index(
            lane & 15, warp_n + atom * 8)) + (unsigned)(issue * 16 * 64 * 2);
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
            : "=r"(fragments.b[atom][0]), "=r"(fragments.b[atom][1]) : "r"(address));
    }
}

template <int BM, int STAGES, typename T>
static __device__ __forceinline__ void rect_consume(
    const RectFragments<RectTraits<BM, STAGES>::kMAtoms>& fragments,
    float (&acc)[RectTraits<BM, STAGES>::kMAtoms][4][4]) {
    using R = RectTraits<BM, STAGES>;
#pragma unroll
    for (int fm = 0; fm < R::kMAtoms; ++fm)
#pragma unroll
        for (int fn = 0; fn < 4; ++fn)
            sm89_fixed_half_swizzle::HalfOps<T>::mma(acc[fm][fn], fragments.a[fm], fragments.b[fn]);
}

template <int BM, int STAGES, typename T>
static __device__ __forceinline__ void rect_epilogue(
    T* C, unsigned char* shared, float (&acc)[RectTraits<BM, STAGES>::kMAtoms][4][4],
    int M, int N, int ldc, int pid_m, int pid_n, int warp_m, int warp_n) {
    using R = RectTraits<BM, STAGES>;
    float* scratch = reinterpret_cast<float*>(shared);
    int lane = (int)threadIdx.x & 31;
    int group = lane >> 2;
    int thread = lane & 3;
    __syncthreads();
#pragma unroll
    for (int fm = 0; fm < R::kMAtoms; ++fm)
#pragma unroll
        for (int fn = 0; fn < 4; ++fn)
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                int row = warp_m + fm * 16 + group + half * 8;
                int column = warp_n + fn * 8 + 2 * thread;
                *reinterpret_cast<float2*>(scratch + row * R::kOutputStride + column) =
                    make_float2(acc[fm][fn][2 * half], acc[fm][fn][2 * half + 1]);
            }
    __syncthreads();
    for (int linear = (int)threadIdx.x; linear < BM * 8; linear += R::kThreads) {
        int local_row = linear >> 3;
        int row = pid_m * BM + local_row;
        if (row >= M) continue;
        int column = (linear & 7) * 8;
        const float* source = scratch + local_row * R::kOutputStride + column;
        float4 first = *reinterpret_cast<const float4*>(source);
        float4 second = *reinterpret_cast<const float4*>(source + 4);
        uint4 packed;
        gbf_store_pair_rne(reinterpret_cast<T*>(&packed.x), first.x, first.y);
        gbf_store_pair_rne(reinterpret_cast<T*>(&packed.y), first.z, first.w);
        gbf_store_pair_rne(reinterpret_cast<T*>(&packed.z), second.x, second.y);
        gbf_store_pair_rne(reinterpret_cast<T*>(&packed.w), second.z, second.w);
        *reinterpret_cast<uint4*>(C + (long long)row * ldc + pid_n * 64 + column) = packed;
    }
}

template <int BM, int STAGES, bool S3, bool M_TAIL, typename T>
static __device__ __forceinline__ void rect_kernel(
    T* C, const T* A, const T* B, int M, int N, int K, int lda, int ldb, int ldc) {
    using R = RectTraits<BM, STAGES>;
    static_assert(R::kSharedBytes == 49152, "two-CTA shared budget");
    static_assert(BM * R::kOutputStride * 4 <= R::kSharedBytes, "rectangular epilogue fits");
    extern __shared__ __align__(16) unsigned char half_batch_shared[];
    int num_pid_n = (N + 63) / 64;
    int pid_m = (int)blockIdx.x / num_pid_n;
    int pid_n = (int)blockIdx.x % num_pid_n;
    int warp = (int)threadIdx.x >> 5;
    int warp_m = (warp >> 1) * (BM / 2);
    int warp_n = (warp & 1) * 32;
    unsigned a_base = (unsigned)__cvta_generic_to_shared(half_batch_shared);
    unsigned b_base = a_base + (unsigned)(STAGES * BM * 64 * 2);
    float acc[R::kMAtoms][4][4] = {};
    int tiles = K / 64;

    if constexpr (S3) {
#pragma unroll
        for (int tile = 0; tile < 2; ++tile) {
#pragma unroll
            for (int issue = 0; issue < 4; ++issue)
                rect_copy_issue<BM, STAGES, M_TAIL>(a_base, b_base, A, B, tile, tile,
                    pid_m, pid_n, M, lda, ldb, issue);
            asm volatile("cp.async.commit_group;\n" ::);
        }
        asm volatile("cp.async.wait_group 1;\n" ::);
        __syncthreads();
        RectFragments<R::kMAtoms> fragments[2];
        rect_load<BM, STAGES>(a_base, b_base, warp_m, warp_n, 0, fragments[0]);
        int read_buf = 0;
        int write_buf = 2;
        for (int tile = 0; tile < tiles; ++tile) {
            bool refill = tile + 2 < tiles;
            bool next = tile + 1 < tiles;
            unsigned a_read = a_base + (unsigned)(read_buf * BM * 64 * 2);
            unsigned b_read = b_base + (unsigned)(read_buf * 64 * 64 * 2);
            if (refill) rect_copy_issue<BM, STAGES, M_TAIL>(a_base, b_base, A, B, write_buf,
                tile + 2, pid_m, pid_n, M, lda, ldb, 0);
            rect_load<BM, STAGES>(a_read, b_read, warp_m, warp_n, 1, fragments[1]);
            rect_consume<BM, STAGES, T>(fragments[0], acc);
            if (refill) rect_copy_issue<BM, STAGES, M_TAIL>(a_base, b_base, A, B, write_buf,
                tile + 2, pid_m, pid_n, M, lda, ldb, 1);
            rect_load<BM, STAGES>(a_read, b_read, warp_m, warp_n, 2, fragments[0]);
            rect_consume<BM, STAGES, T>(fragments[1], acc);
            if (refill) {
                rect_copy_issue<BM, STAGES, M_TAIL>(a_base, b_base, A, B, write_buf,
                    tile + 2, pid_m, pid_n, M, lda, ldb, 2);
                rect_copy_issue<BM, STAGES, M_TAIL>(a_base, b_base, A, B, write_buf,
                    tile + 2, pid_m, pid_n, M, lda, ldb, 3);
            }
            rect_load<BM, STAGES>(a_read, b_read, warp_m, warp_n, 3, fragments[1]);
            rect_consume<BM, STAGES, T>(fragments[0], acc);
            if (refill) asm volatile("cp.async.commit_group;\n" ::);
            if (next) {
                if (refill) asm volatile("cp.async.wait_group 1;\n" ::);
                else asm volatile("cp.async.wait_group 0;\n" ::);
                __syncthreads();
                read_buf = read_buf == 2 ? 0 : read_buf + 1;
                write_buf = write_buf == 2 ? 0 : write_buf + 1;
                rect_load<BM, STAGES>(a_base + (unsigned)(read_buf * BM * 64 * 2),
                    b_base + (unsigned)(read_buf * 64 * 64 * 2), warp_m, warp_n, 0,
                    fragments[0]);
            }
            rect_consume<BM, STAGES, T>(fragments[1], acc);
        }
    } else {
#pragma unroll
        for (int issue = 0; issue < 4; ++issue)
            rect_copy_issue<BM, STAGES, M_TAIL>(a_base, b_base, A, B, 0, 0,
                pid_m, pid_n, M, lda, ldb, issue);
        asm volatile("cp.async.commit_group;\n" ::);
        int read_buf = 0;
        for (int tile = 0; tile < tiles; ++tile) {
            asm volatile("cp.async.wait_group 0;\n" ::);
            __syncthreads();
            bool next = tile + 1 < tiles;
            unsigned a_read = a_base + (unsigned)(read_buf * BM * 64 * 2);
            unsigned b_read = b_base + (unsigned)(read_buf * 64 * 64 * 2);
            RectFragments<R::kMAtoms> fragments[2];
            rect_load<BM, STAGES>(a_read, b_read, warp_m, warp_n, 0, fragments[0]);
#pragma unroll
            for (int issue = 0; issue < 4; ++issue) {
                if (next) rect_copy_issue<BM, STAGES, M_TAIL>(a_base, b_base, A, B, read_buf ^ 1,
                    tile + 1, pid_m, pid_n, M, lda, ldb, issue);
                if (issue < 3) rect_load<BM, STAGES>(a_read, b_read, warp_m, warp_n,
                    issue + 1, fragments[(issue + 1) & 1]);
                rect_consume<BM, STAGES, T>(fragments[issue & 1], acc);
            }
            if (next) asm volatile("cp.async.commit_group;\n" ::);
            read_buf ^= 1;
        }
    }
    rect_epilogue<BM, STAGES, T>(C, half_batch_shared, acc,
        M, N, ldc, pid_m, pid_n, warp_m, warp_n);
}

} // namespace sm89_fixed_half_n64

extern "C" __global__ __launch_bounds__(128, 2)
void gemm_bi_nn_fixed_sm89_m64n64_bk64_s3_v1_f16(
    __half* C, const __half* A, const __half* B, const float* bias,
    FixedSm89HalfSwizzleParams params) {
    if (bias != nullptr || params.alpha != 1.0f || params.beta != 0.0f
        || params.m < 1 || params.m > 2048
        || params.k != 768 || params.n != 2304
        || params.lda != 768 || params.ldb != 2304 || params.ldc != 2304
        || !gbf_aligned16(A) || !gbf_aligned16(B) || !gbf_aligned16(C)) return;
    if (params.m == 2048) {
        sm89_fixed_half_n64::rect_kernel<64, 3, true, false>(C, A, B,
            2048, 2304, 768, 768, 2304, 2304);
    } else {
        sm89_fixed_half_n64::rect_kernel<64, 3, true, true>(C, A, B,
            params.m, 2304, 768, 768, 2304, 2304);
    }
}

extern "C" __global__ __launch_bounds__(128, 2)
void gemm_bi_nn_fixed_sm89_m128n64_bk64_s2_v1_f16(
    __half* C, const __half* A, const __half* B, const float* bias,
    FixedSm89HalfSwizzleParams params) {
    if (bias != nullptr || params.alpha != 1.0f || params.beta != 0.0f
        || params.m < 1 || params.m > 2048
        || params.k != 2304 || params.n != 768
        || params.lda != 2304 || params.ldb != 768 || params.ldc != 768
        || !gbf_aligned16(A) || !gbf_aligned16(B) || !gbf_aligned16(C)) return;
    if (params.m == 2048) {
        sm89_fixed_half_n64::rect_kernel<128, 2, false, false>(C, A, B,
            2048, 768, 2304, 2304, 768, 768);
    } else {
        sm89_fixed_half_n64::rect_kernel<128, 2, false, true>(C, A, B,
            params.m, 768, 2304, 2304, 768, 768);
    }
}
