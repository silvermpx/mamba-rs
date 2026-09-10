pub const SYMBOL: &str = "gemm_bi_nn_test_fixed_sm89_m128n96_bk64_s3_f16";
pub const RETAINED_SYMBOL: &str = "gemm_bi_nn_fixed_sm89_tc128_s3_v1_f16";
pub const TARGET: (usize, usize, usize) = (2_048, 768, 3_072);
pub const TARGET_GRID: u32 = 128;
pub const BLOCK_THREADS: u32 = 256;
pub const DYNAMIC_SHARED_BYTES: usize = 86_016;
pub const REQUIRED_OCCUPANCY: u32 = 1;
pub const MAX_REGISTERS: i32 = 188;
pub const SCOUT_THRESHOLD: f64 = 0.99;
pub const FAST_THRESHOLD: f64 = 1.0;
pub const PARENT_LDSM_PER_K16_CTA: usize = 64;
pub const CANDIDATE_LDSM_PER_K16_CTA: usize = 56;
const EXPECTED_RETAINED_FNV64: u64 = 0x723b_b1f9_e4ef_26e7;

const CANDIDATE_CUDA: &str = r#"
namespace sm89_fixed_half_n96_s3 {
static constexpr int kTileM = 128;
static constexpr int kTileN = 96;
static constexpr int kTileK = 64;
static constexpr int kStages = 3;
static constexpr int kOutputStride = 104;
static constexpr int kSharedBytes = 86016;
static_assert(kStages * (kTileM * kTileK + kTileK * kTileN) * 2 == kSharedBytes,
    "N96 S3 shared ABI");
static_assert(kTileM * kOutputStride * 4 <= kSharedBytes,
    "N96 vector epilogue aliases input ring");

static __device__ __forceinline__ int a_index(int row, int k) {
    return row * 64 + (k ^ ((row & 7) * 8));
}

static __device__ __forceinline__ int b_index(int k, int column) {
    int chunk4 = (column >> 2) ^ ((k & 3) << 1);
    return k * 96 + chunk4 * 4 + (column & 3);
}

struct FragmentOffsets {
    unsigned a[4];
    unsigned b[3];
};

struct Fragments {
    unsigned a[4][4];
    unsigned b[3][2];
};

static __device__ __forceinline__ FragmentOffsets fragment_offsets(
    int warp_m, int warp_n) {
    FragmentOffsets offsets;
    int lane = (int)threadIdx.x & 31;
#pragma unroll
    for (int atom = 0; atom < 4; ++atom) {
        offsets.a[atom] =
            (unsigned)(2 * a_index(warp_m + atom * 16 + (lane & 15),
                (lane & 16) ? 8 : 0));
    }
#pragma unroll
    for (int atom = 0; atom < 3; ++atom) {
        offsets.b[atom] =
            (unsigned)(2 * b_index(lane & 15, warp_n + atom * 8));
    }
    return offsets;
}

static __device__ __forceinline__ void load_fragments(
    unsigned a_stage, unsigned b_stage, int issue,
    const FragmentOffsets& offsets, Fragments& fragments) {
#pragma unroll
    for (int atom = 0; atom < 4; ++atom) {
        unsigned address = a_stage + (offsets.a[atom] ^ (unsigned)(issue * 32));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(fragments.a[atom][0]), "=r"(fragments.a[atom][1]),
              "=r"(fragments.a[atom][2]), "=r"(fragments.a[atom][3])
            : "r"(address));
    }
#pragma unroll
    for (int atom = 0; atom < 3; ++atom) {
        unsigned address = b_stage + offsets.b[atom]
            + (unsigned)(issue * 16 * 96 * 2);
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
            : "=r"(fragments.b[atom][0]), "=r"(fragments.b[atom][1])
            : "r"(address));
    }
}

template <typename T>
static __device__ __forceinline__ void consume_fragments(
    const Fragments& fragments, float (&acc)[4][3][4]) {
#pragma unroll
    for (int fm = 0; fm < 4; ++fm) {
#pragma unroll
        for (int fn = 0; fn < 3; ++fn) {
            sm89_fixed_half_swizzle::HalfOps<T>::mma(
                acc[fm][fn], fragments.a[fm], fragments.b[fn]);
        }
    }
}

struct CopyPlan {
    long long a_offset[4];
    long long b_offset[3];
    unsigned a_destination[4];
    unsigned b_destination[3];
    bool a_valid[4];
    int a_k[4];
    int b_k[3];
    int b_bytes[3];
};

static __device__ __forceinline__ CopyPlan copy_plan(
    unsigned a_shared, unsigned b_shared, int pid_m, int pid_n,
    int M, int N, int lda, int ldb) {
    CopyPlan plan;
    int thread = (int)threadIdx.x;
#pragma unroll
    for (int slice = 0; slice < 4; ++slice) {
        int linear = thread + slice * 256;
        int row = linear >> 3;
        int local_k = (linear & 7) * 8;
        int global_row = pid_m * 128 + row;
        plan.a_valid[slice] = global_row < M;
        plan.a_k[slice] = local_k;
        plan.a_offset[slice] =
            (long long)(global_row < M ? global_row : 0) * lda + local_k;
        plan.a_destination[slice] =
            a_shared + (unsigned)(2 * a_index(row, local_k));
    }
#pragma unroll
    for (int slice = 0; slice < 3; ++slice) {
        int linear = thread + slice * 256;
        int local_k = linear / 12;
        int local_column = (linear % 12) * 8;
        int global_column = pid_n * 96 + local_column;
        int remaining = N - global_column;
        plan.b_k[slice] = local_k;
        plan.b_bytes[slice] =
            remaining >= 8 ? 16 : (remaining > 0 ? remaining * 2 : 0);
        plan.b_offset[slice] = (long long)local_k * ldb
            + (remaining > 0 ? global_column : 0);
        plan.b_destination[slice] =
            b_shared + (unsigned)(2 * b_index(local_k, local_column));
    }
    return plan;
}

template <typename T>
static __device__ __forceinline__ void copy_slice(
    const CopyPlan& plan, const T* A, const T* B,
    int stage, int k_base, long long b_reduction_offset, int K, int slice) {
    int remaining = K - k_base - plan.a_k[slice];
    int a_bytes = plan.a_valid[slice]
        ? (remaining >= 8 ? 16 : (remaining > 0 ? remaining * 2 : 0)) : 0;
    const void* a_source = a_bytes > 0
        ? (const void*)(A + plan.a_offset[slice] + k_base)
        : (const void*)A;
    unsigned a_destination = plan.a_destination[slice]
        + (unsigned)(stage * 128 * 64 * 2);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
        :: "r"(a_destination), "l"(a_source), "r"(a_bytes));
    if (slice < 3) {
        int b_bytes = k_base + plan.b_k[slice] < K ? plan.b_bytes[slice] : 0;
        const void* b_source = b_bytes > 0
            ? (const void*)(B + plan.b_offset[slice] + b_reduction_offset)
            : (const void*)B;
        unsigned b_destination = plan.b_destination[slice]
            + (unsigned)(stage * 64 * 96 * 2);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
            :: "r"(b_destination), "l"(b_source), "r"(b_bytes));
    }
}

template <typename T>
static __device__ __forceinline__ void stage_scalar(
    T* a_stages, T* b_stages, const T* A, const T* B,
    int stage, int k_base, int pid_m, int pid_n,
    int M, int N, int K, int lda, int ldb) {
    T* a_stage = a_stages + stage * 128 * 64;
    T* b_stage = b_stages + stage * 64 * 96;
    for (int linear = (int)threadIdx.x; linear < 128 * 64; linear += 256) {
        int row = linear / 64;
        int k = linear % 64;
        int global_row = pid_m * 128 + row;
        int global_k = k_base + k;
        a_stage[a_index(row, k)] =
            global_row < M && global_k < K
            ? A[(long long)global_row * lda + global_k]
            : sm89_fixed_half_swizzle::HalfOps<T>::from_float(0.0f);
    }
    for (int linear = (int)threadIdx.x; linear < 64 * 96; linear += 256) {
        int k = linear / 96;
        int column = linear % 96;
        int global_k = k_base + k;
        int global_column = pid_n * 96 + column;
        b_stage[b_index(k, column)] =
            global_k < K && global_column < N
            ? B[(long long)global_k * ldb + global_column]
            : sm89_fixed_half_swizzle::HalfOps<T>::from_float(0.0f);
    }
}

template <typename T>
static __device__ __forceinline__ void scalar_epilogue(
    T* C, float (&acc)[4][3][4], float alpha, float beta,
    int M, int N, int ldc, int pid_m, int pid_n, int warp_m, int warp_n) {
    int lane = (int)threadIdx.x & 31;
    int g = lane >> 2;
    int t = lane & 3;
#pragma unroll
    for (int fm = 0; fm < 4; ++fm) {
#pragma unroll
        for (int fn = 0; fn < 3; ++fn) {
            int r0 = pid_m * 128 + warp_m + fm * 16 + g;
            int c0 = pid_n * 96 + warp_n + fn * 8 + 2 * t;
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                int row = r0 + half * 8;
                if (row >= M) continue;
                float first = __fmul_rn(alpha, acc[fm][fn][2 * half]);
                float second = __fmul_rn(alpha, acc[fm][fn][2 * half + 1]);
                T* destination =
                    c0 < N ? C + (long long)row * ldc + c0 : (T*)0;
                if (beta == 0.0f && (ldc & 1) == 0
                    && c0 + 1 < N && gbf_aligned4(destination)) {
                    gbf_store_pair_rne(destination, first, second);
                } else {
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        int column = c0 + e;
                        if (column >= N) continue;
                        float value = e ? second : first;
                        T* output = C + (long long)row * ldc + column;
                        if (beta != 0.0f)
                            value = __fmaf_rn(beta, to_f(*output), value);
                        *output =
                            sm89_fixed_half_swizzle::HalfOps<T>::from_float(value);
                    }
                }
            }
        }
    }
}

template <typename T>
static __device__ __forceinline__ void vector_epilogue(
    T* C, float* output_tile, float (&acc)[4][3][4], float alpha,
    int M, int ldc, int pid_m, int pid_n, int warp_m, int warp_n) {
    int lane = (int)threadIdx.x & 31;
    int g = lane >> 2;
    int t = lane & 3;
    __syncthreads();
#pragma unroll
    for (int fm = 0; fm < 4; ++fm) {
#pragma unroll
        for (int fn = 0; fn < 3; ++fn) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                int row = warp_m + fm * 16 + g + half * 8;
                int column = warp_n + fn * 8 + 2 * t;
                *reinterpret_cast<float2*>(
                    output_tile + row * kOutputStride + column) =
                    make_float2(acc[fm][fn][2 * half],
                        acc[fm][fn][2 * half + 1]);
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int linear = (int)threadIdx.x; linear < 128 * 12; linear += 256) {
        int local_row = linear / 12;
        int row = pid_m * 128 + local_row;
        if (row >= M) continue;
        int column = (linear % 12) * 8;
        const float* source =
            output_tile + local_row * kOutputStride + column;
        float4 first = *reinterpret_cast<const float4*>(source);
        float4 second = *reinterpret_cast<const float4*>(source + 4);
        uint4 packed;
        gbf_store_pair_rne(reinterpret_cast<T*>(&packed.x),
            __fmul_rn(alpha, first.x), __fmul_rn(alpha, first.y));
        gbf_store_pair_rne(reinterpret_cast<T*>(&packed.y),
            __fmul_rn(alpha, first.z), __fmul_rn(alpha, first.w));
        gbf_store_pair_rne(reinterpret_cast<T*>(&packed.z),
            __fmul_rn(alpha, second.x), __fmul_rn(alpha, second.y));
        gbf_store_pair_rne(reinterpret_cast<T*>(&packed.w),
            __fmul_rn(alpha, second.z), __fmul_rn(alpha, second.w));
        *reinterpret_cast<uint4*>(
            C + (long long)row * ldc + pid_n * 96 + column) = packed;
    }
}

template <typename T>
static __device__ __forceinline__ void kernel(
    T* C, const T* A, const T* B, const float* bias,
    float alpha, float beta, int M, int N, int K,
    int lda, int ldb, int ldc) {
    static_assert(sizeof(T) == 2, "homogeneous half pipeline only");
    assert(alpha == 1.0f || bias == nullptr);
    extern __shared__ __align__(16) unsigned char shared[];
    T* a_stages = reinterpret_cast<T*>(shared);
    T* b_stages = a_stages + 3 * 128 * 64;
    int num_pid_n = (N + 95) / 96;
    int pid_m = (int)blockIdx.x / num_pid_n;
    int pid_n = (int)blockIdx.x % num_pid_n;
    int warp = (int)threadIdx.x >> 5;
    int warp_m = (warp >> 2) * 64;
    int warp_n = (warp & 3) * 24;
    int t = (int)threadIdx.x & 3;
    unsigned a_base = (unsigned)__cvta_generic_to_shared(a_stages);
    unsigned b_base = (unsigned)__cvta_generic_to_shared(b_stages);
    bool fast_stage = (lda & 7) == 0 && (ldb & 7) == 0
        && gbf_aligned16(A) && gbf_aligned16(B);
    float acc[4][3][4];
#pragma unroll
    for (int fm = 0; fm < 4; ++fm) {
#pragma unroll
        for (int fn = 0; fn < 3; ++fn) {
            int column = pid_n * 96 + warp_n + fn * 8 + 2 * t;
            float first = bias != nullptr && column < N ? bias[column] : 0.0f;
            float second =
                bias != nullptr && column + 1 < N ? bias[column + 1] : 0.0f;
            acc[fm][fn][0] = first;
            acc[fm][fn][1] = second;
            acc[fm][fn][2] = first;
            acc[fm][fn][3] = second;
        }
    }
    FragmentOffsets offsets = fragment_offsets(warp_m, warp_n);
    CopyPlan plan;
    if (fast_stage)
        plan = copy_plan(a_base, b_base, pid_m, pid_n, M, N, lda, ldb);
    int tiles = (K + 63) / 64;
    if (tiles > 0) {
        if (fast_stage) {
#pragma unroll
            for (int slice = 0; slice < 4; ++slice)
                copy_slice(plan, A, B, 0, 0, 0, K, slice);
            asm volatile("cp.async.commit_group;\n" ::);
            if (tiles > 1) {
#pragma unroll
                for (int slice = 0; slice < 4; ++slice)
                    copy_slice(plan, A, B, 1, 64,
                        (long long)64 * ldb, K, slice);
                asm volatile("cp.async.commit_group;\n" ::);
            }
        } else {
            stage_scalar(a_stages, b_stages, A, B, 0, 0,
                pid_m, pid_n, M, N, K, lda, ldb);
        }
    }
    int read_buf = 0;
    if (fast_stage) {
        Fragments fragments[2];
        if (tiles > 0) {
            if (tiles > 1) asm volatile("cp.async.wait_group 1;\n" ::);
            else asm volatile("cp.async.wait_group 0;\n" ::);
            __syncthreads();
            load_fragments(a_base, b_base, 0, offsets, fragments[0]);
        }
        int write_buf = 2;
        for (int tile = 0; tile < tiles; ++tile) {
            bool next = tile + 1 < tiles;
            bool refill = tile + 2 < tiles;
            int next_k = (tile + 2) * 64;
            long long next_b = (long long)next_k * ldb;
            unsigned a_read =
                a_base + (unsigned)(read_buf * 128 * 64 * 2);
            unsigned b_read =
                b_base + (unsigned)(read_buf * 64 * 96 * 2);
            if (refill)
                copy_slice(plan, A, B, write_buf, next_k, next_b, K, 0);
            load_fragments(a_read, b_read, 1, offsets, fragments[1]);
            consume_fragments<T>(fragments[0], acc);
            if (refill)
                copy_slice(plan, A, B, write_buf, next_k, next_b, K, 1);
            load_fragments(a_read, b_read, 2, offsets, fragments[0]);
            consume_fragments<T>(fragments[1], acc);
            if (refill) {
                copy_slice(plan, A, B, write_buf, next_k, next_b, K, 2);
                copy_slice(plan, A, B, write_buf, next_k, next_b, K, 3);
            }
            load_fragments(a_read, b_read, 3, offsets, fragments[1]);
            consume_fragments<T>(fragments[0], acc);
            if (refill) asm volatile("cp.async.commit_group;\n" ::);
            if (next) {
                if (refill) asm volatile("cp.async.wait_group 1;\n" ::);
                else asm volatile("cp.async.wait_group 0;\n" ::);
                __syncthreads();
                read_buf = read_buf == 2 ? 0 : read_buf + 1;
                write_buf = write_buf == 2 ? 0 : write_buf + 1;
                unsigned next_a =
                    a_base + (unsigned)(read_buf * 128 * 64 * 2);
                unsigned next_b_stage =
                    b_base + (unsigned)(read_buf * 64 * 96 * 2);
                load_fragments(next_a, next_b_stage, 0, offsets, fragments[0]);
            }
            consume_fragments<T>(fragments[1], acc);
        }
    } else {
        for (int tile = 0; tile < tiles; ++tile) {
            __syncthreads();
            bool next = tile + 1 < tiles;
            int next_k = (tile + 1) * 64;
            if (next)
                stage_scalar(a_stages, b_stages, A, B, read_buf ^ 1, next_k,
                    pid_m, pid_n, M, N, K, lda, ldb);
            unsigned a_read =
                a_base + (unsigned)(read_buf * 128 * 64 * 2);
            unsigned b_read =
                b_base + (unsigned)(read_buf * 64 * 96 * 2);
            Fragments fragments[2];
            load_fragments(a_read, b_read, 0, offsets, fragments[0]);
#pragma unroll
            for (int issue = 0; issue < 4; ++issue) {
                if (issue < 3)
                    load_fragments(a_read, b_read, issue + 1,
                        offsets, fragments[(issue + 1) & 1]);
                consume_fragments<T>(fragments[issue & 1], acc);
            }
            read_buf ^= 1;
        }
    }
    bool vector_output = beta == 0.0f && (ldc & 7) == 0
        && gbf_aligned16(C) && pid_n * 96 + 96 <= N;
    if (vector_output) {
        vector_epilogue(C, reinterpret_cast<float*>(shared), acc, alpha,
            M, ldc, pid_m, pid_n, warp_m, warp_n);
        return;
    }
    scalar_epilogue(C, acc, alpha, beta,
        M, N, ldc, pid_m, pid_n, warp_m, warp_n);
}
} // namespace sm89_fixed_half_n96_s3

extern "C" __global__ __launch_bounds__(256, 1)
void gemm_bi_nn_test_fixed_sm89_m128n96_bk64_s3_f16(
    __half* __restrict__ C, const __half* __restrict__ A,
    const __half* __restrict__ B, const float* __restrict__ bias,
    FixedSm89HalfSwizzleParams params) {
    sm89_fixed_half_n96_s3::kernel(C, A, B, bias,
        params.alpha, params.beta, params.m, params.n, params.k,
        params.lda, params.ldb, params.ldc);
}
"#;

pub const fn a_index(row: usize, k: usize) -> usize {
    row * 64 + (k ^ ((row & 7) * 8))
}

pub const fn b_index(k: usize, column: usize) -> usize {
    let chunk4 = (column >> 2) ^ ((k & 3) << 1);
    k * 96 + chunk4 * 4 + (column & 3)
}

pub const fn output_coordinate(
    warp: usize,
    lane: usize,
    fm: usize,
    fn_: usize,
    half: usize,
    e: usize,
) -> (usize, usize) {
    let warp_m = (warp >> 2) * 64;
    let warp_n = (warp & 3) * 24;
    let group = lane >> 2;
    let thread = lane & 3;
    (
        warp_m + fm * 16 + group + half * 8,
        warp_n + fn_ * 8 + 2 * thread + e,
    )
}

pub fn retained_source(swizzle: &str, s3: &str, layout: &str) -> Result<String, String> {
    let include = "#include \"sm89_half_swizzle_layout.cuh\"";
    if swizzle.matches(include).count() != 1 {
        return Err("Fixed S3 swizzle layout include seam changed".into());
    }
    let source = format!("{}\n{s3}", swizzle.replacen(include, layout, 1));
    require_retained_hash(&source)?;
    Ok(source)
}

pub fn candidate_source(swizzle: &str, s3: &str, layout: &str) -> Result<String, String> {
    let retained = retained_source(swizzle, s3, layout)?;
    Ok(format!("{retained}\n{CANDIDATE_CUDA}"))
}

pub fn restore_retained_source(candidate: &str) -> Result<String, String> {
    let suffix = format!("\n{CANDIDATE_CUDA}");
    let retained = candidate
        .strip_suffix(&suffix)
        .ok_or_else(|| "half-NN N96 candidate suffix seam changed".to_owned())?
        .to_owned();
    require_retained_hash(&retained)?;
    Ok(retained)
}

pub fn all_strata_below(strata: &[[f64; 2]], threshold: f64) -> bool {
    strata.len() == 4
        && threshold.is_finite()
        && threshold > 0.0
        && strata.iter().all(|[p50, p95]| {
            p50.is_finite()
                && p95.is_finite()
                && *p50 > 0.0
                && *p95 > 0.0
                && *p50 < threshold
                && *p95 < threshold
        })
}

fn require_retained_hash(source: &str) -> Result<(), String> {
    let observed = fnv64(source.as_bytes());
    if observed != EXPECTED_RETAINED_FNV64 {
        return Err(format!(
            "Fixed S3 retained source changed: expected {EXPECTED_RETAINED_FNV64:#018x}, observed {observed:#018x}"
        ));
    }
    Ok(())
}

fn fnv64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}
