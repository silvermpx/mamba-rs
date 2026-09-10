#[path = "triad_half_nt_fixed_s3_source.rs"]
mod nt_fixed_s3;

pub const SYMBOL_PREFIX: &str = "gemm_bi_tn_test_fixed_s3_bxor_";

pub fn candidate_source(swizzle: &str, s3: &str) -> Result<String, String> {
    let source = nt_fixed_s3::candidate_source(swizzle, s3)?;
    const KERNEL_START: &str = "namespace sm89_test_half_nt_s3 {";
    const KERNEL_END: &str = "} // namespace sm89_test_half_nt_s3";
    const SUPPORT_END: &str = "\n} // namespace sm89_fixed_half_swizzle\n\n";
    for anchor in [KERNEL_START, KERNEL_END, SUPPORT_END] {
        require_count(&source, anchor, 1)?;
    }
    let kernel_start = source.find(KERNEL_START).unwrap();
    let kernel_end = source.find(KERNEL_END).unwrap() + KERNEL_END.len();
    let mut support = source[..kernel_start].to_owned();
    replace_exact(
        &mut support,
        SUPPORT_END,
        &format!("\n{TN_SUPPORT}{SUPPORT_END}"),
    )?;

    let mut kernel = source[kernel_start..kernel_end].to_owned();
    replace_exact(
        &mut kernel,
        KERNEL_START,
        "namespace sm89_test_half_tn_s3 {",
    )?;
    replace_exact(
        &mut kernel,
        KERNEL_END,
        "} // namespace sm89_test_half_tn_s3",
    )?;
    replace_exact(
        &mut kernel,
        "    T* C, const T* A, const T* B,\n    float alpha, int M, int N, int K_out)",
        "    float* C, const T* A, const T* B,\n    float alpha, int M_red, int K_out, int N)",
    )?;
    replace_exact(
        &mut kernel,
        "    T (*As)[128][64] = reinterpret_cast<T (*)[128][64]>(sm89_fhs_shared);",
        "    T (*As)[64][128] = reinterpret_cast<T (*)[64][128]>(sm89_fhs_shared);",
    )?;
    replace_exact(
        &mut kernel,
        "    T (*Bs)[128][64] = reinterpret_cast<T (*)[128][64]>(\n        sm89_fhs_shared + 3 * 128 * 64 * (int)sizeof(T));",
        "    T (*Bs)[64][128] = reinterpret_cast<T (*)[64][128]>(\n        sm89_fhs_shared + 3 * 64 * 128 * (int)sizeof(T));",
    )?;
    replace_exact(
        &mut kernel,
        "    int num_pid_n = (K_out + 127) / 128;",
        "    int num_pid_n = (N + 127) / 128;",
    )?;
    replace_exact(
        &mut kernel,
        "    bool fast_stage = (N & 7) == 0 && gbf_aligned16(A) && gbf_aligned16(B);",
        "    bool fast_stage = (K_out & 7) == 0 && (N & 7) == 0\n        && gbf_aligned16(A) && gbf_aligned16(B);",
    )?;
    replace_exact(
        &mut kernel,
        "sm89_fixed_half_swizzle::nt_fragment_offsets(warpM, warpN)",
        "sm89_fixed_half_swizzle::tn_fragment_offsets(warpM, warpN)",
    )?;
    replace_exact(
        &mut kernel,
        "    sm89_fixed_half_swizzle::NtCopyPlan plan;\n    if (fast_stage) plan = sm89_fixed_half_swizzle::nt_copy_plan(As_sbase, Bs_sbase, pid_m, pid_n, M, K_out, N);",
        "    sm89_fixed_half_swizzle::TnCopyPlan plan;\n    if (fast_stage) plan = sm89_fixed_half_swizzle::tn_copy_plan(As_sbase, Bs_sbase, pid_m, pid_n, K_out, N);",
    )?;
    replace_exact(
        &mut kernel,
        "    int num_k_tiles = (N + 63) / 64;",
        "    int num_k_tiles = (M_red + 63) / 64;",
    )?;
    for (before, after) in [
        (
            "sm89_fixed_half_swizzle::nt_copy_slice(plan, A, B, 0, 0, N, slice)",
            "sm89_fixed_half_swizzle::tn_copy_slice(plan, A, B, 0, 0, M_red, slice)",
        ),
        (
            "sm89_fixed_half_swizzle::nt_copy_slice(plan, A, B, 1, 64, N, slice)",
            "sm89_fixed_half_swizzle::tn_copy_slice(plan, A, B, 1, 64, M_red, slice)",
        ),
        (
            "sm89_fixed_half_swizzle::nt_stage_scalar(&As[0][0][0], &Bs[0][0][0], A, B, 0, 0, pid_m, pid_n, M, K_out, N)",
            "sm89_fixed_half_swizzle::tn_stage_scalar(&As[0][0][0], &Bs[0][0][0], A, B, 0, 0, pid_m, pid_n, M_red, K_out, N)",
        ),
        (
            "sm89_fixed_half_swizzle::nt_stage_scalar(&As[0][0][0], &Bs[0][0][0], A, B, read_buf ^ 1, next_k,\n                    pid_m, pid_n, M, K_out, N)",
            "sm89_fixed_half_swizzle::tn_stage_scalar(&As[0][0][0], &Bs[0][0][0], A, B, read_buf ^ 1, next_k,\n                    pid_m, pid_n, M_red, K_out, N)",
        ),
    ] {
        replace_exact(&mut kernel, before, after)?;
    }
    for slice in ["0", "1", "2", "3"] {
        replace_exact(
            &mut kernel,
            &format!(
                "sm89_fixed_half_swizzle::nt_copy_slice(plan, A, B, write_buf, next_k, N, {slice})"
            ),
            &format!(
                "sm89_fixed_half_swizzle::tn_copy_slice(plan, A, B, write_buf, next_k, M_red, {slice})"
            ),
        )?;
    }
    replace_all_exact(
        &mut kernel,
        "sm89_fixed_half_swizzle::nt_load_fragments",
        "sm89_fixed_half_swizzle::tn_load_fragments",
        7,
    )?;
    replace_range(
        &mut kernel,
        "    {\n        bool vector_output",
        "    sm89_fixed_half_swizzle::scalar_epilogue(C, acc, alpha, 0.0f, M, K_out, K_out,\n        pid_m, pid_n, warpM, warpN);",
        TN_EPILOGUE,
    )?;
    for stale in ["nt_copy_slice", "nt_stage_scalar", "nt_load_fragments"] {
        if kernel.contains(stale) {
            return Err(format!(
                "half TN Fixed S3 retained stale kernel token {stale}"
            ));
        }
    }
    Ok(format!("{support}{kernel}\n{TN_EXPORTS}"))
}

const TN_EPILOGUE: &str = r#"    int lane = (int)threadIdx.x & 31;
    int group = lane >> 2;
    int thread = lane & 3;
#pragma unroll
    for (int fm = 0; fm < 4; ++fm) {
#pragma unroll
        for (int fn = 0; fn < 4; ++fn) {
            int row0 = pid_m * 128 + warpM + fm * 16 + group;
            int column0 = pid_n * 128 + warpN + fn * 8 + 2 * thread;
#pragma unroll
            for (int element = 0; element < 4; ++element) {
                int row = row0 + (element >= 2 ? 8 : 0);
                int column = column0 + (element & 1);
                if (row < K_out && column < N) {
                    C[(long long)row * N + column] += alpha * acc[fm][fn][element];
                }
            }
        }
    }"#;

const TN_EXPORTS: &str = r#"
#define SM89_HALF_TN_S3_EXPORT(TYPE, SUFFIX)                                 \
extern "C" __global__ __launch_bounds__(256, 1)                              \
void gemm_bi_tn_test_fixed_s3_bxor_##SUFFIX(                                 \
    float* __restrict__ C, const TYPE* __restrict__ A,                       \
    const TYPE* __restrict__ B, float alpha,                                 \
    int M_red, int K_out, int N) {                                           \
    sm89_test_half_tn_s3::kernel<TYPE>(C, A, B, alpha, M_red, K_out, N);      \
}
SM89_HALF_TN_S3_EXPORT(__nv_bfloat16, bf16)
SM89_HALF_TN_S3_EXPORT(__half, f16)
#undef SM89_HALF_TN_S3_EXPORT
"#;

const TN_SUPPORT: &str = r#"
template <typename T>
static __device__ __forceinline__ void tn_stage_scalar(
    T* a_stages, T* b_stages, const T* A, const T* B,
    int stage, int reduction_base, int pid_m, int pid_n,
    int M_red, int K_out, int N) {
    T* a_stage = a_stages + stage * 64 * 128;
    T* b_stage = b_stages + stage * 64 * 128;
    for (int linear = (int)threadIdx.x; linear < 64 * 128; linear += 256) {
        int reduction = linear / 128;
        int column = linear % 128;
        int global_reduction = reduction_base + reduction;
        int global_column = pid_m * 128 + column;
        a_stage[layout::b_index(reduction, column)] =
            global_reduction < M_red && global_column < K_out
            ? A[(long long)global_reduction * K_out + global_column]
            : HalfOps<T>::from_float(0.0f);
    }
    for (int linear = (int)threadIdx.x; linear < 64 * 128; linear += 256) {
        int reduction = linear / 128;
        int column = linear % 128;
        int global_reduction = reduction_base + reduction;
        int global_column = pid_n * 128 + column;
        b_stage[layout::b_index(reduction, column)] =
            global_reduction < M_red && global_column < N
            ? B[(long long)global_reduction * N + global_column]
            : HalfOps<T>::from_float(0.0f);
    }
}

static __device__ __forceinline__ FragmentOffsets tn_fragment_offsets(int warpM, int warpN) {
    FragmentOffsets offsets;
    int lane = (int)threadIdx.x & 31;
    int row = lane & 7;
    int quadrant = lane >> 3;
#pragma unroll
    for (int atom = 0; atom < 4; ++atom) {
        offsets.a[atom] = 2u * (unsigned)layout::b_index(
            row + ((quadrant & 2) ? 8 : 0),
            warpM + atom * 16 + ((quadrant & 1) ? 8 : 0));
        offsets.b[atom] = layout::b_fragment_base(warpN, atom, lane);
    }
    return offsets;
}

static __device__ __forceinline__ void tn_load_fragments(
    unsigned a_stage, unsigned b_stage, int issue,
    const FragmentOffsets& offsets, Fragments& fragments) {
#pragma unroll
    for (int atom = 0; atom < 4; ++atom) {
        unsigned address = a_stage + offsets.a[atom] + (unsigned)(issue * 16 * 128 * 2);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
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

struct TnCopyPlan {
    long long a_offset[4];
    long long b_offset[4];
    unsigned a_destination[4];
    unsigned b_destination[4];
    int reduction[4];
    int a_column_bytes;
    int b_column_bytes;
    int a_stride;
    int b_stride;
};

static __device__ __forceinline__ TnCopyPlan tn_copy_plan(
    unsigned a_shared, unsigned b_shared, int pid_m, int pid_n,
    int K_out, int N) {
    TnCopyPlan plan;
    int thread = (int)threadIdx.x;
    int local_column = (thread & 15) * 8;
    int a_column = pid_m * 128 + local_column;
    int b_column = pid_n * 128 + local_column;
    int a_remaining = K_out - a_column;
    int b_remaining = N - b_column;
    plan.a_column_bytes = a_remaining >= 8 ? 16 : (a_remaining > 0 ? a_remaining * 2 : 0);
    plan.b_column_bytes = b_remaining >= 8 ? 16 : (b_remaining > 0 ? b_remaining * 2 : 0);
    plan.a_stride = K_out;
    plan.b_stride = N;
#pragma unroll
    for (int slice = 0; slice < 4; ++slice) {
        plan.reduction[slice] = (thread >> 4) + slice * 16;
        plan.a_offset[slice] = (long long)plan.reduction[slice] * K_out
            + (a_remaining > 0 ? a_column : 0);
        plan.b_offset[slice] = (long long)plan.reduction[slice] * N
            + (b_remaining > 0 ? b_column : 0);
        plan.a_destination[slice] = a_shared + layout::b_copy_offset(thread, slice);
        plan.b_destination[slice] = b_shared + layout::b_copy_offset(thread, slice);
    }
    return plan;
}

template <typename T>
static __device__ __forceinline__ void tn_copy_slice(
    const TnCopyPlan& plan, const T* A, const T* B,
    int stage, int reduction_base, int M_red, int slice) {
    bool valid_row = reduction_base + plan.reduction[slice] < M_red;
    int a_bytes = valid_row ? plan.a_column_bytes : 0;
    int b_bytes = valid_row ? plan.b_column_bytes : 0;
    const void* a_source = a_bytes > 0
        ? (const void*)(A + plan.a_offset[slice] + (long long)reduction_base * plan.a_stride)
        : (const void*)A;
    const void* b_source = b_bytes > 0
        ? (const void*)(B + plan.b_offset[slice] + (long long)reduction_base * plan.b_stride)
        : (const void*)B;
    unsigned a_destination = plan.a_destination[slice] + (unsigned)(stage * 64 * 128 * 2);
    unsigned b_destination = plan.b_destination[slice] + (unsigned)(stage * 64 * 128 * 2);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
        :: "r"(a_destination), "l"(a_source), "r"(a_bytes));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
        :: "r"(b_destination), "l"(b_source), "r"(b_bytes));
}
"#;

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
        return Err("half TN Fixed S3 replacement boundaries reversed".into());
    }
    source.replace_range(begin..finish, replacement);
    Ok(())
}

fn require_count(source: &str, anchor: &str, expected: usize) -> Result<(), String> {
    let actual = source.matches(anchor).count();
    if actual == expected {
        Ok(())
    } else {
        Err(format!(
            "half TN Fixed S3 anchor {anchor:?}: expected {expected}, observed {actual}"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SWIZZLE: &str = include_str!("../../kernels/gemm_bi_inference/sm89_half_swizzle.cu");
    const S3: &str = include_str!("../../kernels/gemm_bi_inference/sm89_half_s3.cu");

    fn b_index(reduction: usize, column: usize) -> usize {
        reduction * 128 + (column ^ ((reduction & 7) * 8))
    }

    #[test]
    fn both_operands_have_exact_aligned_copy_ownership() {
        for _operand in ["A", "B"] {
            let mut owners = vec![0u8; 64 * 128];
            for thread in 0..256 {
                for slice in 0..4 {
                    let reduction = (thread >> 4) + slice * 16;
                    let column = (thread & 15) * 8;
                    assert_eq!(2 * b_index(reduction, column) % 16, 0);
                    for element in 0..8 {
                        owners[b_index(reduction, column + element)] += 1;
                    }
                }
            }
            assert!(owners.into_iter().all(|owner| owner == 1));
        }
    }

    #[test]
    fn transposed_a_and_b_fragments_match_current_tn_coordinates() {
        for warp in 0..8 {
            let warp_m = (warp >> 2) * 64;
            let warp_n = (warp & 3) * 32;
            for lane in 0..32 {
                let row = lane & 7;
                let quadrant = lane >> 3;
                for issue in 0..4 {
                    for atom in 0..4 {
                        let reduction_a = issue * 16 + row + ((quadrant & 2) * 4);
                        let column_a = warp_m + atom * 16 + ((quadrant & 1) * 8);
                        let a_address =
                            b_index(row + ((quadrant & 2) * 4), column_a) + issue * 16 * 128;
                        assert_eq!(a_address, b_index(reduction_a, column_a));
                        assert_eq!(2 * a_address % 16, 0);

                        let reduction_b = issue * 16 + (lane & 15);
                        let column_b = warp_n + atom * 8;
                        let b_address = b_index(lane & 15, column_b) + issue * 16 * 128;
                        assert_eq!(b_address, b_index(reduction_b, column_b));
                        assert_eq!(2 * b_address % 16, 0);
                    }
                }
            }
        }
    }

    #[test]
    fn eight_warps_own_each_output_once() {
        let mut owners = vec![0u8; 128 * 128];
        for warp in 0..8 {
            let warp_m = (warp >> 2) * 64;
            let warp_n = (warp & 3) * 32;
            for lane in 0..32 {
                let group = lane >> 2;
                let thread = lane & 3;
                for fm in 0..4 {
                    for f_n in 0..4 {
                        for element in 0..4 {
                            let row = warp_m + fm * 16 + group + usize::from(element >= 2) * 8;
                            let column = warp_n + f_n * 8 + 2 * thread + (element & 1);
                            owners[row * 128 + column] += 1;
                        }
                    }
                }
            }
        }
        assert!(owners.into_iter().all(|owner| owner == 1));
        assert_eq!(3 * (64 * 128 + 64 * 128) * 2, 98_304);
    }

    #[test]
    fn composed_source_is_tn_only_and_keeps_s3_order_and_epilogue() {
        let source = candidate_source(SWIZZLE, S3).unwrap();
        assert!(source.contains("void gemm_bi_tn_test_fixed_s3_bxor_##SUFFIX"));
        assert!(!source.contains("void gemm_bi_nt_test_fixed_s3_bxor_##SUFFIX"));
        assert!(source.contains("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"));
        assert!(source.contains("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16"));
        assert!(source.contains("for (int issue = 0; issue < 4; ++issue)"));
        assert!(source.contains("consume_fragments<T>(fragments[0], acc);"));
        assert!(source.contains("C[(long long)row * N + column] += alpha * acc[fm][fn][element];"));
        let plan = source
            .split("static __device__ __forceinline__ TnCopyPlan tn_copy_plan(")
            .nth(1)
            .unwrap()
            .split("\n}\n")
            .next()
            .unwrap();
        assert_eq!(plan.matches("    return plan;").count(), 1);
        let copy = source
            .split("static __device__ __forceinline__ void tn_copy_slice(")
            .nth(1)
            .unwrap()
            .split("\n}\n")
            .next()
            .unwrap();
        assert!(!copy.contains(" K_out"));
        assert!(!copy.contains(" N"));
        assert!(copy.contains("plan.a_stride"));
        assert!(copy.contains("plan.b_stride"));
        let kernel = source
            .split("namespace sm89_test_half_tn_s3 {")
            .nth(1)
            .unwrap()
            .split("} // namespace sm89_test_half_tn_s3")
            .next()
            .unwrap();
        assert!(!kernel.contains("HalfOps<T>::from_float(value)"));
    }

    #[test]
    fn missing_or_duplicated_parent_anchor_is_rejected() {
        assert!(candidate_source("", S3).is_err());
        assert!(candidate_source(SWIZZLE, "").is_err());
        let duplicate = format!("{S3}\nnamespace sm89_fixed_half_s3 {{");
        assert!(candidate_source(SWIZZLE, &duplicate).is_err());
    }
}
