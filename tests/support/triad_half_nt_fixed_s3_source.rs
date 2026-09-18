const LAYOUT: &str = include_str!("../../kernels/gemm_bi_inference/sm80/half_swizzle_layout.cuh");

pub fn candidate_source(swizzle: &str, s3: &str) -> Result<String, String> {
    const SUPPORT_END: &str =
        "template <typename T>\nstatic __device__ __forceinline__ void kernel(";
    for anchor in [
        "#include \"sm89_half_swizzle_layout.cuh\"",
        "namespace sm89_fixed_half_swizzle {",
        SUPPORT_END,
    ] {
        require_count(swizzle, anchor, 1)?;
    }
    for anchor in [
        "namespace sm89_fixed_half_s3 {",
        "} // namespace sm89_fixed_half_s3",
        "void nn_sm89_tc128_s3_bf16(",
        "void nn_sm89_tc128_s3_f16(",
    ] {
        require_count(s3, anchor, 1)?;
    }

    let support_end = swizzle.find(SUPPORT_END).unwrap();
    let mut support = swizzle[..support_end].to_owned();
    replace_exact(
        &mut support,
        "#include \"sm89_half_swizzle_layout.cuh\"",
        LAYOUT,
    )?;
    support.push_str(NT_SUPPORT);
    support.push_str("\n} // namespace sm89_fixed_half_swizzle\n");

    let kernel_start = s3.find("namespace sm89_fixed_half_s3 {").unwrap();
    let kernel_end_marker = "} // namespace sm89_fixed_half_s3";
    let kernel_end = s3.find(kernel_end_marker).unwrap() + kernel_end_marker.len();
    let mut kernel = s3[kernel_start..kernel_end].to_owned();
    replace_exact(
        &mut kernel,
        "namespace sm89_fixed_half_s3 {",
        "namespace sm89_test_half_nt_s3 {",
    )?;
    replace_exact(
        &mut kernel,
        "} // namespace sm89_fixed_half_s3",
        "} // namespace sm89_test_half_nt_s3",
    )?;
    replace_exact(
        &mut kernel,
        "    T* C, const T* A, const T* B, const float* bias,\n    float alpha, float beta, int M, int N, int K, int lda, int ldb, int ldc)",
        "    T* C, const T* A, const T* B,\n    float alpha, int M, int N, int K_out)",
    )?;
    replace_exact(
        &mut kernel,
        "    assert(alpha == 1.0f || bias == nullptr);\n",
        "",
    )?;
    replace_exact(
        &mut kernel,
        "    T (*Bs)[64][128] = reinterpret_cast<T (*)[64][128]>(\n        sm89_fhs_shared + 3 * 128 * 64 * (int)sizeof(T));",
        "    T (*Bs)[128][64] = reinterpret_cast<T (*)[128][64]>(\n        sm89_fhs_shared + 3 * 128 * 64 * (int)sizeof(T));",
    )?;
    replace_exact(
        &mut kernel,
        "    int num_pid_n = (N + 127) / 128;",
        "    int num_pid_n = (K_out + 127) / 128;",
    )?;
    replace_exact(
        &mut kernel,
        "    bool fast_stage = (lda & 7) == 0 && (ldb & 7) == 0 && gbf_aligned16(A) && gbf_aligned16(B);",
        "    bool fast_stage = (N & 7) == 0 && gbf_aligned16(A) && gbf_aligned16(B);",
    )?;
    replace_range(
        &mut kernel,
        "    float acc[4][4][4];",
        "    sm89_fixed_half_swizzle::FragmentOffsets offsets = sm89_fixed_half_swizzle::fragment_offsets(warpM, warpN);",
        NT_ACCUMULATORS,
    )?;
    replace_exact(
        &mut kernel,
        "    sm89_fixed_half_swizzle::CopyPlan plan;\n    if (fast_stage) plan = sm89_fixed_half_swizzle::copy_plan(As_sbase, Bs_sbase, pid_m, pid_n, M, N, lda, ldb);",
        "    sm89_fixed_half_swizzle::NtCopyPlan plan;\n    if (fast_stage) plan = sm89_fixed_half_swizzle::nt_copy_plan(As_sbase, Bs_sbase, pid_m, pid_n, M, K_out, N);",
    )?;
    replace_exact(
        &mut kernel,
        "    int num_k_tiles = (K + 63) / 64;",
        "    int num_k_tiles = (N + 63) / 64;",
    )?;
    for (before, after, expected) in [
        (
            "sm89_fixed_half_swizzle::copy_slice(plan, A, B, 0, 0, 0, K, slice)",
            "sm89_fixed_half_swizzle::nt_copy_slice(plan, A, B, 0, 0, N, slice)",
            1,
        ),
        (
            "sm89_fixed_half_swizzle::copy_slice(plan, A, B, 1, 64, (long long)64 * ldb, K, slice)",
            "sm89_fixed_half_swizzle::nt_copy_slice(plan, A, B, 1, 64, N, slice)",
            1,
        ),
        (
            "sm89_fixed_half_swizzle::stage_scalar(&As[0][0][0], &Bs[0][0][0], A, B, 0, 0, pid_m, pid_n, M, N, K, lda, ldb)",
            "sm89_fixed_half_swizzle::nt_stage_scalar(&As[0][0][0], &Bs[0][0][0], A, B, 0, 0, pid_m, pid_n, M, K_out, N)",
            1,
        ),
        (
            "sm89_fixed_half_swizzle::load_fragments",
            "sm89_fixed_half_swizzle::nt_load_fragments",
            7,
        ),
        (
            "sm89_fixed_half_swizzle::stage_scalar(&As[0][0][0], &Bs[0][0][0], A, B, read_buf ^ 1, next_k,\n                    pid_m, pid_n, M, N, K, lda, ldb)",
            "sm89_fixed_half_swizzle::nt_stage_scalar(&As[0][0][0], &Bs[0][0][0], A, B, read_buf ^ 1, next_k,\n                    pid_m, pid_n, M, K_out, N)",
            1,
        ),
    ] {
        replace_all_exact(&mut kernel, before, after, expected)?;
    }
    replace_exact(
        &mut kernel,
        "            long long next_b = (long long)next_k * ldb;\n",
        "",
    )?;
    for slice in ["0", "1", "2", "3"] {
        replace_exact(
            &mut kernel,
            &format!(
                "sm89_fixed_half_swizzle::copy_slice(plan, A, B, write_buf, next_k, next_b, K, {slice})"
            ),
            &format!(
                "sm89_fixed_half_swizzle::nt_copy_slice(plan, A, B, write_buf, next_k, N, {slice})"
            ),
        )?;
    }
    replace_range(
        &mut kernel,
        "    {\n        bool vector_output",
        "    sm89_fixed_half_swizzle::scalar_epilogue(C, acc, alpha, beta, M, N, ldc, pid_m, pid_n, warpM, warpN);",
        NT_EPILOGUE,
    )?;

    Ok(format!("{support}\n{kernel}\n{NT_EXPORTS}"))
}

const NT_ACCUMULATORS: &str = r#"    float acc[4][4][4];
#pragma unroll
    for (int fm = 0; fm < 4; ++fm)
#pragma unroll
        for (int fn = 0; fn < 4; ++fn)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[fm][fn][e] = 0.0f;
    sm89_fixed_half_swizzle::FragmentOffsets offsets =
        sm89_fixed_half_swizzle::nt_fragment_offsets(warpM, warpN);
"#;

const NT_EPILOGUE: &str = r#"    {
        bool vector_output = (K_out & 7) == 0 && gbf_aligned16(C)
            && pid_n * 128 + 128 <= K_out;
        if (vector_output) {
            sm89_fixed_half_swizzle::vector_epilogue(C, reinterpret_cast<float*>(sm89_fhs_shared), acc, alpha,
                M, K_out, pid_m, pid_n, warpM, warpN);
            return;
        }
    }
    sm89_fixed_half_swizzle::scalar_epilogue(C, acc, alpha, 0.0f, M, K_out, K_out,
        pid_m, pid_n, warpM, warpN);"#;

const NT_EXPORTS: &str = r#"
#define SM89_HALF_NT_S3_EXPORT(TYPE, SUFFIX)                                 \
extern "C" __global__ __launch_bounds__(256, 1)                              \
void nt_test_fixed_s3_bxor_##SUFFIX(                                 \
    TYPE* __restrict__ C, const TYPE* __restrict__ A,                        \
    const TYPE* __restrict__ B, float alpha, int M, int N, int K_out) {      \
    sm89_test_half_nt_s3::kernel<TYPE>(C, A, B, alpha, M, N, K_out);         \
}
SM89_HALF_NT_S3_EXPORT(__nv_bfloat16, bf16)
SM89_HALF_NT_S3_EXPORT(__half, f16)
#undef SM89_HALF_NT_S3_EXPORT
"#;

const NT_SUPPORT: &str = r#"
template <typename T>
static __device__ __forceinline__ void nt_stage_scalar(
    T* a_stages, T* b_stages, const T* A, const T* B,
    int stage, int k_base, int pid_m, int pid_n,
    int M, int K_out, int N) {
    T* a_stage = a_stages + stage * 128 * 64;
    T* b_stage = b_stages + stage * 128 * 64;
    for (int linear = (int)threadIdx.x; linear < 128 * 64; linear += 256) {
        int row = linear / 64;
        int k = linear % 64;
        int global_row = pid_m * 128 + row;
        int global_k = k_base + k;
        a_stage[layout::a_index(row, k)] = global_row < M && global_k < N
            ? A[(long long)global_row * N + global_k] : HalfOps<T>::from_float(0.0f);
    }
    for (int linear = (int)threadIdx.x; linear < 128 * 64; linear += 256) {
        int row = linear / 64;
        int k = linear % 64;
        int global_row = pid_n * 128 + row;
        int global_k = k_base + k;
        b_stage[layout::a_index(row, k)] = global_row < K_out && global_k < N
            ? B[(long long)global_row * N + global_k] : HalfOps<T>::from_float(0.0f);
    }
}

static __device__ __forceinline__ FragmentOffsets nt_fragment_offsets(int warpM, int warpN) {
    FragmentOffsets offsets;
    int lane = (int)threadIdx.x & 31;
#pragma unroll
    for (int atom = 0; atom < 4; ++atom) {
        offsets.a[atom] = layout::a_fragment_base(warpM, atom, lane);
        offsets.b[atom] = 2u * (unsigned)layout::a_index(
            warpN + atom * 8 + (lane & 7), ((lane >> 3) & 1) * 8);
    }
    return offsets;
}

static __device__ __forceinline__ void nt_load_fragments(
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
        unsigned address = b_stage + (offsets.b[atom] ^ (unsigned)(issue * 32));
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
            : "=r"(fragments.b[atom][0]), "=r"(fragments.b[atom][1]) : "r"(address));
    }
}

struct NtCopyPlan {
    long long a_offset[4];
    long long b_offset[4];
    unsigned a_destination[4];
    unsigned b_destination[4];
    bool a_row_valid[4];
    bool b_row_valid[4];
    int k;
};

static __device__ __forceinline__ NtCopyPlan nt_copy_plan(
    unsigned a_shared, unsigned b_shared, int pid_m, int pid_n,
    int M, int K_out, int N) {
    NtCopyPlan plan;
    int thread = (int)threadIdx.x;
    plan.k = (thread & 7) * 8;
#pragma unroll
    for (int slice = 0; slice < 4; ++slice) {
        int local_row = (thread >> 3) + slice * 32;
        int a_row = pid_m * 128 + local_row;
        int b_row = pid_n * 128 + local_row;
        plan.a_row_valid[slice] = a_row < M;
        plan.b_row_valid[slice] = b_row < K_out;
        plan.a_offset[slice] = (long long)(a_row < M ? a_row : 0) * N + plan.k;
        plan.b_offset[slice] = (long long)(b_row < K_out ? b_row : 0) * N + plan.k;
        plan.a_destination[slice] = a_shared + layout::a_copy_offset(thread, slice);
        plan.b_destination[slice] = b_shared + layout::a_copy_offset(thread, slice);
    }
    return plan;
}

template <typename T>
static __device__ __forceinline__ void nt_copy_slice(
    const NtCopyPlan& plan, const T* A, const T* B,
    int stage, int k_base, int N, int slice) {
    int remaining = N - k_base - plan.k;
    int a_bytes = plan.a_row_valid[slice]
        ? (remaining >= 8 ? 16 : (remaining > 0 ? remaining * 2 : 0)) : 0;
    int b_bytes = plan.b_row_valid[slice]
        ? (remaining >= 8 ? 16 : (remaining > 0 ? remaining * 2 : 0)) : 0;
    const void* a_source = a_bytes > 0
        ? (const void*)(A + plan.a_offset[slice] + k_base) : (const void*)A;
    const void* b_source = b_bytes > 0
        ? (const void*)(B + plan.b_offset[slice] + k_base) : (const void*)B;
    unsigned a_destination = plan.a_destination[slice] + (unsigned)(stage * 128 * 64 * 2);
    unsigned b_destination = plan.b_destination[slice] + (unsigned)(stage * 128 * 64 * 2);
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
        return Err("half NT Fixed S3 replacement boundaries are reversed".into());
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
            "half NT Fixed S3 anchor {anchor:?}: expected {expected}, observed {actual}"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SWIZZLE: &str = include_str!("../../kernels/gemm_bi_inference/sm80/half_swizzle.cu");
    const S3: &str = include_str!("../../kernels/gemm_bi_inference/sm80/half_s3.cu");

    fn index(row: usize, k: usize) -> usize {
        row * 64 + (k ^ ((row & 7) * 8))
    }

    #[test]
    fn both_operands_have_exact_aligned_copy_ownership() {
        for _operand in ["A", "B"] {
            let mut owners = vec![0u8; 128 * 64];
            for thread in 0..256 {
                for slice in 0..4 {
                    let row = (thread >> 3) + slice * 32;
                    let k = (thread & 7) * 8;
                    let base = index(row, k);
                    assert_eq!(base % 8, 0);
                    for element in 0..8 {
                        owners[index(row, k + element)] += 1;
                    }
                }
            }
            assert!(owners.into_iter().all(|owner| owner == 1));
        }
    }

    #[test]
    fn fragment_coordinates_match_the_current_nt_mma_operands() {
        for warp in 0..8 {
            let warp_m = (warp >> 2) * 64;
            let warp_n = (warp & 3) * 32;
            for lane in 0..32 {
                for issue in 0..4 {
                    for atom in 0..4 {
                        let a_row = warp_m + atom * 16 + (lane & 15);
                        let a_k0 = if lane & 16 != 0 { 8 } else { 0 };
                        let a_k = issue * 16 + a_k0;
                        let a_address = index(a_row, a_k0) ^ (issue * 16);
                        assert_eq!(a_address, index(a_row, a_k));
                        assert_eq!(2 * a_address % 16, 0);

                        let b_row = warp_n + atom * 8 + (lane & 7);
                        let b_k0 = ((lane >> 3) & 1) * 8;
                        let b_k = issue * 16 + b_k0;
                        let b_address = index(b_row, b_k0) ^ (issue * 16);
                        assert_eq!(b_address, index(b_row, b_k));
                        assert_eq!(2 * b_address % 16, 0);
                    }
                }
            }
        }
    }

    #[test]
    fn eight_warps_own_every_output_once() {
        let mut owners = vec![0u8; 128 * 128];
        for warp in 0..8 {
            let warp_m = (warp >> 2) * 64;
            let warp_n = (warp & 3) * 32;
            for lane in 0..32 {
                let group = lane >> 2;
                let thread = lane & 3;
                for fm in 0..4 {
                    for f_n in 0..4 {
                        let row = warp_m + fm * 16 + group;
                        let column = warp_n + f_n * 8 + 2 * thread;
                        for element in 0..4 {
                            let output_row = row + usize::from(element >= 2) * 8;
                            let output_column = column + (element & 1);
                            owners[output_row * 128 + output_column] += 1;
                        }
                    }
                }
            }
        }
        assert!(owners.into_iter().all(|owner| owner == 1));
        assert_eq!(3 * (128 * 64 + 128 * 64) * 2, 98_304);
    }

    #[test]
    fn composed_source_is_nt_only_and_keeps_s3_issue_order() {
        let source = candidate_source(SWIZZLE, S3).unwrap();
        assert!(source.contains("void nt_test_fixed_s3_bxor_##SUFFIX"));
        let nt_load = source
            .split("static __device__ __forceinline__ void nt_load_fragments(")
            .nth(1)
            .unwrap()
            .split("struct NtCopyPlan")
            .next()
            .unwrap();
        assert!(nt_load.contains("ldmatrix.sync.aligned.m8n8.x2.shared.b16"));
        assert!(!nt_load.contains("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16"));
        assert!(source.contains("for (int issue = 0; issue < 4; ++issue)"));
        assert!(source.contains("nt_load_fragments(a_read, b_read, 1, offsets, fragments[1]);"));
        assert!(source.contains("consume_fragments<T>(fragments[0], acc);"));
        assert!(source.contains("float first = __fmul_rn(alpha, acc[fm][fn][2 * half]);"));
        assert!(source.contains("*output = HalfOps<T>::from_float(value);"));
        assert!(!source.contains("#include \"sm89_half_swizzle_layout.cuh\""));
        assert!(!source.contains(" = sm89_fixed_half_swizzle::fragment_offsets"));
        assert!(!source.contains("void nn_sm89_tc128_s3_"));
        let kernel = source
            .split("namespace sm89_test_half_nt_s3 {")
            .nth(1)
            .unwrap()
            .split("} // namespace sm89_test_half_nt_s3")
            .next()
            .unwrap();
        for stale in [" bias", " beta", " lda", " ldb", " ldc"] {
            assert!(!kernel.contains(stale), "stale NN token {stale:?}");
        }
        assert!(!kernel.contains("long long next_b ="));
        assert!(!kernel.contains(", next_b, K,"));
    }

    #[test]
    fn missing_or_duplicated_anchor_is_rejected() {
        let missing = S3.replacen("namespace sm89_fixed_half_s3 {", "", 1);
        assert!(candidate_source(SWIZZLE, &missing).is_err());
        let duplicate = format!("{S3}\nnamespace sm89_fixed_half_s3 {{\n");
        assert!(candidate_source(SWIZZLE, &duplicate).is_err());
    }
}
