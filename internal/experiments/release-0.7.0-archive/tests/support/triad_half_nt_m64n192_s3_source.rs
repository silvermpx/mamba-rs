#[path = "triad_half_nt_m64n128_s3_source.rs"]
mod parent;

pub const SYMBOL: &str = "gemm_bi_nt_test_fixed_s3_m64n192_f16";
pub const BLOCK_THREADS: u32 = 384;
pub const DYNAMIC_SHARED_BYTES: usize = 98_304;
pub const MAX_REGISTERS: i32 = 168;
pub const REQUIRED_OCCUPANCY: u32 = 1;

pub fn candidate_source(swizzle: &str, s3: &str) -> Result<String, String> {
    let mut source = parent::candidate_source(swizzle, s3)?;
    for (anchor, expected) in [
        ("namespace sm89_test_half_nt_m64n128_s3", 2),
        ("sm89_test_half_nt_m64n128_s3::kernel<TYPE>", 1),
        ("static constexpr int kSharedBytes = 73728;", 1),
        ("T (*Bs)[128][64]", 1),
        ("int num_pid_n = (K_out + 127) / 128;", 1),
        ("int warpM = (warp >> 2) * 32;", 1),
        ("int warpN = (warp & 3) * 32;", 1),
        ("struct NtM64N128CopyPlan {", 1),
        ("#define SM89_HALF_NT_S3_EXPORT(TYPE, SUFFIX)", 1),
    ] {
        require_count(&source, anchor, expected)?;
    }

    replace_all_exact(
        &mut source,
        "namespace sm89_test_half_nt_m64n128_s3",
        "namespace sm89_test_half_nt_m64n192_s3",
        2,
    )?;
    replace_exact(
        &mut source,
        "sm89_test_half_nt_m64n128_s3::kernel<TYPE>",
        "sm89_test_half_nt_m64n192_s3::kernel<TYPE>",
    )?;
    replace_exact(
        &mut source,
        "static constexpr int kSharedBytes = 73728;\nstatic_assert(3 * (64 * 64 + 128 * 64) * 2 == kSharedBytes, \"M64N128 S3 shared ABI\");",
        "static constexpr int kSharedBytes = 98304;\nstatic_assert(3 * (64 * 64 + 192 * 64) * 2 == kSharedBytes, \"M64N192 S3 shared ABI\");",
    )?;
    replace_exact(&mut source, "T (*Bs)[128][64]", "T (*Bs)[192][64]")?;
    replace_exact(
        &mut source,
        "reinterpret_cast<T (*)[128][64]>",
        "reinterpret_cast<T (*)[192][64]>",
    )?;
    replace_exact(
        &mut source,
        "int num_pid_n = (K_out + 127) / 128;",
        "int num_pid_n = (K_out + 191) / 192;",
    )?;
    replace_exact(
        &mut source,
        "int warpM = (warp >> 2) * 32;",
        "int warpM = (warp / 6) * 32;",
    )?;
    replace_exact(
        &mut source,
        "int warpN = (warp & 3) * 32;",
        "int warpN = (warp % 6) * 32;",
    )?;
    replace_all_exact(&mut source, "nt_m64n128_", "nt_m64n192_", 32)?;
    replace_all_exact(&mut source, "NtM64N128", "NtM64N192", 10)?;

    replace_range(
        &mut source,
        "template <typename T>\nstatic __device__ __forceinline__ void nt_m64n192_stage_scalar(",
        "struct NtM64N192FragmentOffsets {",
        M64N192_STAGE_SCALAR,
    )?;
    replace_range(
        &mut source,
        "struct NtM64N192CopyPlan {",
        "template <typename T>\nstatic __device__ __forceinline__ void nt_m64n192_scalar_epilogue(",
        M64N192_COPY_PLAN,
    )?;
    replace_all_exact(
        &mut source,
        "for (int slice = 0; slice < 4; ++slice) sm89_fixed_half_swizzle::nt_m64n192_copy_slice",
        "for (int slice = 0; slice < 6; ++slice) sm89_fixed_half_swizzle::nt_m64n192_copy_slice",
        1,
    )?;
    replace_all_exact(
        &mut source,
        "for (int slice = 0; slice < 4; ++slice)\n                    sm89_fixed_half_swizzle::nt_m64n192_copy_slice",
        "for (int slice = 0; slice < 6; ++slice)\n                    sm89_fixed_half_swizzle::nt_m64n192_copy_slice",
        1,
    )?;
    replace_exact(
        &mut source,
        "            if (refill) {\n                sm89_fixed_half_swizzle::nt_m64n192_copy_slice(plan, A, B, write_buf, next_k, N, 2);\n                sm89_fixed_half_swizzle::nt_m64n192_copy_slice(plan, A, B, write_buf, next_k, N, 3);\n            }\n            sm89_fixed_half_swizzle::nt_m64n192_load_fragments(a_read, b_read, 3, offsets, fragments[1]);\n            sm89_fixed_half_swizzle::nt_m64n192_consume_fragments<T>(fragments[0], acc);\n\n            if (refill) asm volatile(\"cp.async.commit_group;\\n\" ::);",
        "            if (refill) {\n                sm89_fixed_half_swizzle::nt_m64n192_copy_slice(plan, A, B, write_buf, next_k, N, 2);\n                sm89_fixed_half_swizzle::nt_m64n192_copy_slice(plan, A, B, write_buf, next_k, N, 3);\n            }\n            sm89_fixed_half_swizzle::nt_m64n192_load_fragments(a_read, b_read, 3, offsets, fragments[1]);\n            sm89_fixed_half_swizzle::nt_m64n192_consume_fragments<T>(fragments[0], acc);\n\n            if (refill) {\n                sm89_fixed_half_swizzle::nt_m64n192_copy_slice(plan, A, B, write_buf, next_k, N, 4);\n                sm89_fixed_half_swizzle::nt_m64n192_copy_slice(plan, A, B, write_buf, next_k, N, 5);\n                asm volatile(\"cp.async.commit_group;\\n\" ::);\n            }",
    )?;
    replace_all_exact(
        &mut source,
        "read_buf * 64 * 128 * 2",
        "read_buf * 192 * 64 * 2",
        3,
    )?;
    replace_all_exact(&mut source, "pid_n * 128", "pid_n * 192", 9)?;
    replace_exact(
        &mut source,
        "pid_n * 192 + 128 <= K_out",
        "pid_n * 192 + 192 <= K_out",
    )?;
    replace_range(
        &mut source,
        "template <typename T>\nstatic __device__ __forceinline__ void nt_m64n192_vector_epilogue(",
        "} // namespace sm89_fixed_half_swizzle\n\nnamespace sm89_test_half_nt_m64n192_s3 {",
        M64N192_VECTOR_EPILOGUE,
    )?;
    replace_range(
        &mut source,
        "#define SM89_HALF_NT_S3_EXPORT(TYPE, SUFFIX)",
        "#undef SM89_HALF_NT_S3_EXPORT",
        M64N192_EXPORT,
    )?;
    if source.contains("m64n128") || source.contains("M64N128") {
        return Err("M64N192 candidate retained an M64N128 identifier".into());
    }
    Ok(source)
}

pub fn measured_s3_source(swizzle: &str, s3: &str) -> Result<String, String> {
    parent::candidate_source(swizzle, s3)
}

pub fn all_strata_pass(strata: &[[f64; 2]]) -> bool {
    strata
        .iter()
        .all(|quantiles| quantiles[0] < 0.99 && quantiles[1] < 0.99)
}

const M64N192_STAGE_SCALAR: &str = r#"template <typename T>
static __device__ __forceinline__ void nt_m64n192_stage_scalar(
    T* a_stages, T* b_stages, const T* A, const T* B,
    int stage, int k_base, int pid_m, int pid_n,
    int M, int K_out, int N) {
    T* a_stage = a_stages + stage * 64 * 64;
    T* b_stage = b_stages + stage * 192 * 64;
    for (int linear = (int)threadIdx.x; linear < 64 * 64; linear += 384) {
        int row = linear / 64;
        int k = linear % 64;
        int global_row = pid_m * 64 + row;
        int global_k = k_base + k;
        a_stage[layout::a_index(row, k)] = global_row < M && global_k < N
            ? A[(long long)global_row * N + global_k] : HalfOps<T>::from_float(0.0f);
    }
    for (int linear = (int)threadIdx.x; linear < 192 * 64; linear += 384) {
        int row = linear / 64;
        int k = linear % 64;
        int global_row = pid_n * 192 + row;
        int global_k = k_base + k;
        b_stage[layout::a_index(row, k)] = global_row < K_out && global_k < N
            ? B[(long long)global_row * N + global_k] : HalfOps<T>::from_float(0.0f);
    }
}

struct NtM64N192FragmentOffsets {
"#;

const M64N192_COPY_PLAN: &str = r#"struct NtM64N192CopyPlan {
    long long offset[6];
    unsigned destination[6];
    bool from_a[6];
    bool row_valid[6];
    bool in_bounds[6];
    int k[6];
};

static __device__ __forceinline__ NtM64N192CopyPlan nt_m64n192_copy_plan(
    unsigned a_shared, unsigned b_shared, int pid_m, int pid_n,
    int M, int K_out, int N) {
    NtM64N192CopyPlan plan;
    int thread = (int)threadIdx.x;
#pragma unroll
    for (int slice = 0; slice < 6; ++slice) {
        int linear = thread * 8 + slice * 384 * 8;
        plan.in_bounds[slice] = linear < (64 + 192) * 64;
        bool from_a = linear < 64 * 64;
        int operand_linear = from_a ? linear : linear - 64 * 64;
        int row = operand_linear / 64;
        int k = operand_linear % 64;
        int global_row = from_a ? pid_m * 64 + row : pid_n * 192 + row;
        bool row_valid = from_a ? global_row < M : global_row < K_out;
        plan.from_a[slice] = from_a;
        plan.row_valid[slice] = plan.in_bounds[slice] && row_valid;
        plan.k[slice] = k;
        plan.offset[slice] = (long long)(row_valid ? global_row : 0) * N + k;
        plan.destination[slice] = (from_a ? a_shared : b_shared)
            + 2u * (unsigned)layout::a_index(row, k);
    }
    return plan;
}

template <typename T>
static __device__ __forceinline__ void nt_m64n192_copy_slice(
    const NtM64N192CopyPlan& plan, const T* A, const T* B,
    int stage, int k_base, int N, int slice) {
    if (!plan.in_bounds[slice]) return;
    int remaining = N - k_base - plan.k[slice];
    int bytes = plan.row_valid[slice]
        ? (remaining >= 8 ? 16 : (remaining > 0 ? remaining * 2 : 0)) : 0;
    const T* operand = plan.from_a[slice] ? A : B;
    const void* source = bytes > 0
        ? (const void*)(operand + plan.offset[slice] + k_base) : (const void*)operand;
    unsigned stage_stride = plan.from_a[slice] ? 64u * 64u * 2u : 192u * 64u * 2u;
    unsigned destination = plan.destination[slice] + (unsigned)stage * stage_stride;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
        :: "r"(destination), "l"(source), "r"(bytes));
}

template <typename T>
static __device__ __forceinline__ void nt_m64n192_scalar_epilogue("#;

const M64N192_EXPORT: &str = r#"extern "C" __global__ __launch_bounds__(384, 1)
void gemm_bi_nt_test_fixed_s3_m64n192_f16(
    __half* __restrict__ C, const __half* __restrict__ A,
    const __half* __restrict__ B, float alpha, int M, int N, int K_out) {
    sm89_test_half_nt_m64n192_s3::kernel<__half>(C, A, B, alpha, M, N, K_out);
}
"#;

const M64N192_VECTOR_EPILOGUE: &str = r#"static constexpr int kM64N192OutputStride = 200;

template <typename T>
static __device__ __forceinline__ void nt_m64n192_vector_epilogue(
    T* C, float* output_tile, float (&acc)[2][4][4], float alpha,
    int M, int ldc, int pid_m, int pid_n, int warpM, int warpN) {
    int lane = (int)threadIdx.x & 31;
    int g = lane >> 2;
    int t = lane & 3;
    __syncthreads();
#pragma unroll
    for (int fm = 0; fm < 2; ++fm)
#pragma unroll
        for (int fn = 0; fn < 4; ++fn)
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                int row = warpM + fm * 16 + g + half * 8;
                int column = warpN + fn * 8 + 2 * t;
                *reinterpret_cast<float2*>(output_tile + row * kM64N192OutputStride + column) =
                    make_float2(acc[fm][fn][2 * half], acc[fm][fn][2 * half + 1]);
            }
    __syncthreads();
#pragma unroll
    for (int linear = (int)threadIdx.x; linear < 64 * 24; linear += 384) {
        int local_row = linear / 24;
        int row = pid_m * 64 + local_row;
        if (row >= M) continue;
        int column = (linear % 24) * 8;
        const float* source = output_tile + local_row * kM64N192OutputStride + column;
        float4 first = *reinterpret_cast<const float4*>(source);
        float4 second = *reinterpret_cast<const float4*>(source + 4);
        uint4 packed;
        gbf_store_pair_rne(reinterpret_cast<T*>(&packed.x), __fmul_rn(alpha, first.x), __fmul_rn(alpha, first.y));
        gbf_store_pair_rne(reinterpret_cast<T*>(&packed.y), __fmul_rn(alpha, first.z), __fmul_rn(alpha, first.w));
        gbf_store_pair_rne(reinterpret_cast<T*>(&packed.z), __fmul_rn(alpha, second.x), __fmul_rn(alpha, second.y));
        gbf_store_pair_rne(reinterpret_cast<T*>(&packed.w), __fmul_rn(alpha, second.z), __fmul_rn(alpha, second.w));
        *reinterpret_cast<uint4*>(C + (long long)row * ldc + pid_n * 192 + column) = packed;
    }
}

} // namespace sm89_fixed_half_swizzle

namespace sm89_test_half_nt_m64n192_s3 {"#;

fn replace_exact(source: &mut String, before: &str, after: &str) -> Result<(), String> {
    require_count(source, before, 1)?;
    *source = source.replacen(before, after, 1);
    Ok(())
}

fn replace_all_exact(
    source: &mut String,
    before: &str,
    after: &str,
    expected: usize,
) -> Result<(), String> {
    require_count(source, before, expected)?;
    *source = source.replace(before, after);
    Ok(())
}

fn replace_range(
    source: &mut String,
    start: &str,
    end: &str,
    replacement: &str,
) -> Result<(), String> {
    require_count(source, start, 1)?;
    require_count(source, end, 1)?;
    let begin = source.find(start).unwrap();
    let finish = source.find(end).unwrap() + end.len();
    if begin >= finish {
        return Err("M64N192 replacement boundaries are reversed".into());
    }
    source.replace_range(begin..finish, replacement);
    Ok(())
}

fn require_count(source: &str, needle: &str, expected: usize) -> Result<(), String> {
    let actual = source.matches(needle).count();
    if actual != expected {
        return Err(format!(
            "M64N192 source anchor {needle:?}: expected {expected}, observed {actual}"
        ));
    }
    Ok(())
}
