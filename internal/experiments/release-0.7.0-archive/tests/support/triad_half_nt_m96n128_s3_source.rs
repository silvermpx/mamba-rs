#[path = "triad_half_nt_m64n192_s3_source.rs"]
mod parent;

pub const SYMBOL: &str = "gemm_bi_nt_test_fixed_s3_m96n128_f16";
pub const RETAINED_SYMBOL: &str = parent::SYMBOL;
pub const TARGET: (usize, usize, usize) = (2_048, 1_536, 768);
pub const TARGET_GRID: u32 = 264;
pub const BLOCK_THREADS: u32 = 384;
pub const DYNAMIC_SHARED_BYTES: usize = 86_016;
pub const MAX_REGISTERS: i32 = 168;
pub const REQUIRED_OCCUPANCY: u32 = 1;
pub const EXPECTED_HMMA: usize = 64;
pub const EXPECTED_LDGSTS: usize = 15;
pub const RETAINED_BLOCK_THREADS: u32 = parent::BLOCK_THREADS;
pub const RETAINED_DYNAMIC_SHARED_BYTES: usize = parent::DYNAMIC_SHARED_BYTES;
pub const RETAINED_MAX_REGISTERS: i32 = parent::MAX_REGISTERS;
pub const RETAINED_REQUIRED_OCCUPANCY: u32 = parent::REQUIRED_OCCUPANCY;
pub const RETAINED_EXPECTED_HMMA: usize = 64;
pub const RETAINED_EXPECTED_LDGSTS: usize = 18;

pub fn retained_source(swizzle: &str, s3: &str) -> Result<String, String> {
    parent::candidate_source(swizzle, s3)
}

pub fn candidate_source(swizzle: &str, s3: &str) -> Result<String, String> {
    let mut source = retained_source(swizzle, s3)?;
    for anchor in [
        "static constexpr int kSharedBytes = 98304;",
        "T (*As)[64][64]",
        "T (*Bs)[192][64]",
        "int num_pid_n = (K_out + 191) / 192;",
        "int warpM = (warp / 6) * 32;",
        "int warpN = (warp % 6) * 32;",
        "struct NtM64N192CopyPlan {",
        "static constexpr int kM64N192OutputStride = 200;",
        RETAINED_SYMBOL,
    ] {
        require_nonzero(&source, anchor)?;
    }

    replace_exact(
        &mut source,
        "static constexpr int kSharedBytes = 98304;\nstatic_assert(3 * (64 * 64 + 192 * 64) * 2 == kSharedBytes, \"M64N192 S3 shared ABI\");",
        "static constexpr int kSharedBytes = 86016;\nstatic_assert(3 * (96 * 64 + 128 * 64) * 2 == kSharedBytes, \"M96N128 S3 shared ABI\");",
    )?;
    replace_exact(&mut source, "T (*As)[64][64]", "T (*As)[96][64]")?;
    replace_exact(
        &mut source,
        "reinterpret_cast<T (*)[64][64]>",
        "reinterpret_cast<T (*)[96][64]>",
    )?;
    replace_exact(&mut source, "T (*Bs)[192][64]", "T (*Bs)[128][64]")?;
    replace_exact(
        &mut source,
        "reinterpret_cast<T (*)[192][64]>",
        "reinterpret_cast<T (*)[128][64]>",
    )?;
    replace_exact(
        &mut source,
        "3 * 64 * 64 * (int)sizeof(T)",
        "3 * 96 * 64 * (int)sizeof(T)",
    )?;
    replace_exact(
        &mut source,
        "int num_pid_n = (K_out + 191) / 192;",
        "int num_pid_n = (K_out + 127) / 128;",
    )?;
    replace_exact(
        &mut source,
        "int warpM = (warp / 6) * 32;",
        "int warpM = (warp >> 2) * 32;",
    )?;
    replace_exact(
        &mut source,
        "int warpN = (warp % 6) * 32;",
        "int warpN = (warp & 3) * 32;",
    )?;
    replace_range(
        &mut source,
        "template <typename T>\nstatic __device__ __forceinline__ void nt_m64n192_stage_scalar(",
        "struct NtM64N192FragmentOffsets {",
        M96N128_STAGE_SCALAR,
    )?;
    replace_range(
        &mut source,
        "struct NtM64N192CopyPlan {",
        "template <typename T>\nstatic __device__ __forceinline__ void nt_m64n192_scalar_epilogue(",
        M96N128_COPY_PLAN,
    )?;
    replace_all_exact(
        &mut source,
        "for (int slice = 0; slice < 6; ++slice)",
        "for (int slice = 0; slice < 5; ++slice)",
        2,
    )?;
    replace_exact(
        &mut source,
        "                sm89_fixed_half_swizzle::nt_m64n192_copy_slice(plan, A, B, write_buf, next_k, N, 4);\n                sm89_fixed_half_swizzle::nt_m64n192_copy_slice(plan, A, B, write_buf, next_k, N, 5);\n                asm volatile(\"cp.async.commit_group;\\n\" ::);",
        "                sm89_fixed_half_swizzle::nt_m64n192_copy_slice(plan, A, B, write_buf, next_k, N, 4);\n                asm volatile(\"cp.async.commit_group;\\n\" ::);",
    )?;
    replace_all_exact(
        &mut source,
        "read_buf * 64 * 64 * 2",
        "read_buf * 96 * 64 * 2",
        3,
    )?;
    replace_all_exact(
        &mut source,
        "read_buf * 192 * 64 * 2",
        "read_buf * 128 * 64 * 2",
        3,
    )?;
    replace_range(
        &mut source,
        "static constexpr int kM64N192OutputStride = 200;",
        "} // namespace sm89_fixed_half_swizzle\n\nnamespace sm89_test_half_nt_m64n192_s3 {",
        M96N128_VECTOR_EPILOGUE,
    )?;
    replace_exact(&mut source, M64N192_EXPORT, M96N128_EXPORT)?;

    replace_all_nonzero(&mut source, "pid_m * 64", "pid_m * 96")?;
    replace_all_nonzero(&mut source, "pid_n * 192", "pid_n * 128")?;
    replace_all_nonzero(&mut source, "nt_m64n192", "nt_m96n128")?;
    replace_all_nonzero(&mut source, "NtM64N192", "NtM96N128")?;

    if source.contains("m64n192") || source.contains("M64N192") || !has_m96n128_geometry(&source) {
        return Err("M96N128 transform retained stale geometry or missed a frozen anchor".into());
    }
    Ok(source)
}

pub fn restore_retained_source(candidate: &str, swizzle: &str, s3: &str) -> Result<String, String> {
    let expected = candidate_source(swizzle, s3)?;
    if candidate != expected {
        return Err("M96N128 candidate differs from the fail-closed transform".into());
    }
    retained_source(swizzle, s3)
}

pub fn has_m96n128_geometry(source: &str) -> bool {
    [
        "namespace sm89_test_half_nt_m96n128_s3",
        "static constexpr int kSharedBytes = 86016;",
        "T (*As)[96][64]",
        "T (*Bs)[128][64]",
        "int num_pid_n = (K_out + 127) / 128;",
        "int warpM = (warp >> 2) * 32;",
        "int warpN = (warp & 3) * 32;",
        "for (int slice = 0; slice < 5; ++slice)",
        "static constexpr int kM96N128OutputStride = 136;",
        SYMBOL,
    ]
    .iter()
    .all(|anchor| source.contains(anchor))
}

pub const fn target_grid() -> u32 {
    TARGET.0.div_ceil(96) as u32 * TARGET.1.div_ceil(128) as u32
}

pub const fn staged_half_elements_per_k_tile_m96n128() -> usize {
    TARGET.0.div_ceil(96) * TARGET.1.div_ceil(128) * (96 + 128) * 64
}

pub const fn staged_half_elements_per_k_tile_m64n192() -> usize {
    TARGET.0.div_ceil(64) * TARGET.1.div_ceil(192) * (64 + 192) * 64
}

pub fn all_strata_pass(strata: &[[f64; 2]]) -> bool {
    strata.len() == 4
        && strata.iter().all(|[p50, p95]| {
            p50.is_finite()
                && p95.is_finite()
                && *p50 > 0.0
                && *p95 > 0.0
                && *p50 < 0.99
                && *p95 < 0.99
        })
}

fn replace_exact(source: &mut String, before: &str, after: &str) -> Result<(), String> {
    replace_all_exact(source, before, after, 1)
}

fn replace_all_exact(
    source: &mut String,
    before: &str,
    after: &str,
    expected: usize,
) -> Result<(), String> {
    let actual = source.matches(before).count();
    if actual != expected {
        return Err(format!(
            "M96N128 source anchor expected {expected}, observed {actual}: {before:?}"
        ));
    }
    *source = source.replace(before, after);
    Ok(())
}

fn replace_all_nonzero(source: &mut String, before: &str, after: &str) -> Result<(), String> {
    require_nonzero(source, before)?;
    *source = source.replace(before, after);
    Ok(())
}

fn replace_range(
    source: &mut String,
    start: &str,
    end: &str,
    replacement: &str,
) -> Result<(), String> {
    replace_all_exact(source, start, start, 1)?;
    replace_all_exact(source, end, end, 1)?;
    let begin = source.find(start).unwrap();
    let finish = source[begin..]
        .find(end)
        .map(|offset| begin + offset + end.len())
        .ok_or_else(|| "M96N128 replacement end precedes start".to_owned())?;
    source.replace_range(begin..finish, replacement);
    Ok(())
}

fn require_nonzero(source: &str, anchor: &str) -> Result<(), String> {
    if source.contains(anchor) {
        Ok(())
    } else {
        Err(format!("M96N128 source missing anchor: {anchor:?}"))
    }
}

const M96N128_STAGE_SCALAR: &str = r#"template <typename T>
static __device__ __forceinline__ void nt_m96n128_stage_scalar(
    T* a_stages, T* b_stages, const T* A, const T* B,
    int stage, int k_base, int pid_m, int pid_n,
    int M, int K_out, int N) {
    T* a_stage = a_stages + stage * 96 * 64;
    T* b_stage = b_stages + stage * 128 * 64;
    for (int linear = (int)threadIdx.x; linear < 96 * 64; linear += 384) {
        int row = linear / 64;
        int k = linear % 64;
        int global_row = pid_m * 96 + row;
        int global_k = k_base + k;
        a_stage[layout::a_index(row, k)] = global_row < M && global_k < N
            ? A[(long long)global_row * N + global_k] : HalfOps<T>::from_float(0.0f);
    }
    for (int linear = (int)threadIdx.x; linear < 128 * 64; linear += 384) {
        int row = linear / 64;
        int k = linear % 64;
        int global_row = pid_n * 128 + row;
        int global_k = k_base + k;
        b_stage[layout::a_index(row, k)] = global_row < K_out && global_k < N
            ? B[(long long)global_row * N + global_k] : HalfOps<T>::from_float(0.0f);
    }
}

struct NtM96N128FragmentOffsets {"#;

const M96N128_COPY_PLAN: &str = r#"struct NtM96N128CopyPlan {
    long long offset[5];
    unsigned destination[5];
    bool row_valid[5];
    bool in_bounds[5];
    int k[5];
};

static __device__ __forceinline__ NtM96N128CopyPlan nt_m96n128_copy_plan(
    unsigned a_shared, unsigned b_shared, int pid_m, int pid_n,
    int M, int K_out, int N) {
    NtM96N128CopyPlan plan;
    int thread = (int)threadIdx.x;
#pragma unroll
    for (int slice = 0; slice < 5; ++slice) {
        int linear = thread * 8 + slice * 384 * 8;
        bool from_a = slice < 2;
        plan.in_bounds[slice] = linear < (96 + 128) * 64;
        int operand_linear = from_a ? linear : linear - 96 * 64;
        int row = operand_linear / 64;
        int k = operand_linear % 64;
        int global_row = from_a ? pid_m * 96 + row : pid_n * 128 + row;
        bool row_valid = from_a ? global_row < M : global_row < K_out;
        plan.row_valid[slice] = plan.in_bounds[slice] && row_valid;
        plan.k[slice] = k;
        plan.offset[slice] = (long long)(row_valid ? global_row : 0) * N + k;
        plan.destination[slice] = (from_a ? a_shared : b_shared)
            + 2u * (unsigned)layout::a_index(row, k);
    }
    return plan;
}

template <typename T>
static __device__ __forceinline__ void nt_m96n128_copy_slice(
    const NtM96N128CopyPlan& plan, const T* A, const T* B,
    int stage, int k_base, int N, int slice) {
    if (!plan.in_bounds[slice]) return;
    bool from_a = slice < 2;
    int remaining = N - k_base - plan.k[slice];
    int bytes = plan.row_valid[slice]
        ? (remaining >= 8 ? 16 : (remaining > 0 ? remaining * 2 : 0)) : 0;
    const T* operand = from_a ? A : B;
    const void* source = bytes > 0
        ? (const void*)(operand + plan.offset[slice] + k_base) : (const void*)operand;
    unsigned stage_stride = from_a ? 96u * 64u * 2u : 128u * 64u * 2u;
    unsigned destination = plan.destination[slice] + (unsigned)stage * stage_stride;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
        :: "r"(destination), "l"(source), "r"(bytes));
}

template <typename T>
static __device__ __forceinline__ void nt_m96n128_scalar_epilogue("#;

const M96N128_VECTOR_EPILOGUE: &str = r#"static constexpr int kM96N128OutputStride = 136;
static_assert(96 * kM96N128OutputStride * 4 <= 86016, "M96N128 output scratch fits");

template <typename T>
static __device__ __forceinline__ void nt_m96n128_vector_epilogue(
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
                *reinterpret_cast<float2*>(output_tile + row * kM96N128OutputStride + column) =
                    make_float2(acc[fm][fn][2 * half], acc[fm][fn][2 * half + 1]);
            }
    __syncthreads();
#pragma unroll
    for (int linear = (int)threadIdx.x; linear < 96 * 16; linear += 384) {
        int local_row = linear >> 4;
        int row = pid_m * 96 + local_row;
        if (row >= M) continue;
        int column = (linear & 15) * 8;
        const float* source = output_tile + local_row * kM96N128OutputStride + column;
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

} // namespace sm89_fixed_half_swizzle

namespace sm89_test_half_nt_m96n128_s3 {"#;

const M96N128_EXPORT: &str = r#"extern "C" __global__ __launch_bounds__(384, 1)
void gemm_bi_nt_test_fixed_s3_m96n128_f16(
    __half* __restrict__ C, const __half* __restrict__ A,
    const __half* __restrict__ B, float alpha, int M, int N, int K_out) {
    sm89_test_half_nt_m96n128_s3::kernel<__half>(C, A, B, alpha, M, N, K_out);
}"#;

const M64N192_EXPORT: &str = r#"extern "C" __global__ __launch_bounds__(384, 1)
void gemm_bi_nt_test_fixed_s3_m64n192_f16(
    __half* __restrict__ C, const __half* __restrict__ A,
    const __half* __restrict__ B, float alpha, int M, int N, int K_out) {
    sm89_test_half_nt_m64n192_s3::kernel<__half>(C, A, B, alpha, M, N, K_out);
}"#;

#[cfg(test)]
mod tests {
    use super::*;

    const SWIZZLE: &str = include_str!("../../kernels/gemm_bi_inference/sm89_half_swizzle.cu");
    const S3: &str = include_str!("../../kernels/gemm_bi_inference/sm89_half_s3.cu");

    #[test]
    fn transform_is_reversible_and_changes_physical_geometry() {
        let retained = retained_source(SWIZZLE, S3).unwrap();
        let candidate = candidate_source(SWIZZLE, S3).unwrap();
        assert!(has_m96n128_geometry(&candidate));
        assert_eq!(
            restore_retained_source(&candidate, SWIZZLE, S3).unwrap(),
            retained
        );
        assert!(!candidate.contains("m64n192"));
        assert!(!candidate.contains("M64N192"));
    }

    #[test]
    fn twelve_warps_own_every_m96n128_output_once() {
        let mut owners = vec![0u8; 96 * 128];
        for warp in 0..12 {
            let warp_m = (warp >> 2) * 32;
            let warp_n = (warp & 3) * 32;
            for row in warp_m..warp_m + 32 {
                for column in warp_n..warp_n + 32 {
                    owners[row * 128 + column] += 1;
                }
            }
        }
        assert!(owners.into_iter().all(|count| count == 1));
    }

    #[test]
    fn five_copy_slices_own_every_stage_segment_once() {
        let segments = (96 + 128) * 64 / 8;
        let mut owners = vec![0u8; segments];
        for slice in 0..5 {
            for thread in 0..384 {
                let linear = thread * 8 + slice * 384 * 8;
                if linear < (96 + 128) * 64 {
                    owners[linear / 8] += 1;
                }
            }
        }
        assert_eq!(owners.len(), 1_792);
        assert!(owners.into_iter().all(|count| count == 1));
        assert_eq!(96 * 64, 2 * 384 * 8);
    }

    #[test]
    fn target_keeps_two_waves_and_reduces_staged_traffic() {
        assert_eq!(target_grid(), TARGET_GRID);
        assert_eq!(TARGET_GRID, 264);
        let candidate = staged_half_elements_per_k_tile_m96n128();
        let retained = staged_half_elements_per_k_tile_m64n192();
        assert_eq!(candidate, 3_784_704);
        assert_eq!(retained, 4_194_304);
        assert_eq!(candidate * 1_000_000 / retained, 902_343);
    }

    #[test]
    fn malformed_parent_fails_closed() {
        assert!(candidate_source("", S3).is_err());
        assert!(candidate_source(SWIZZLE, "").is_err());
    }
}
