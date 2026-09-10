#[path = "triad_half_nt_fixed_s3_source.rs"]
mod parent;

pub const SYMBOL_PREFIX: &str = "gemm_bi_nt_test_fixed_s3_m64n128_";

pub fn candidate_source(swizzle: &str, s3: &str) -> Result<String, String> {
    let mut source = parent::candidate_source(swizzle, s3)?;
    for anchor in [
        "} // namespace sm89_fixed_half_swizzle\n\nnamespace sm89_test_half_nt_s3 {",
        "} // namespace sm89_test_half_nt_s3",
        "void gemm_bi_nt_test_fixed_s3_bxor_##SUFFIX",
        "sm89_test_half_nt_s3::kernel<TYPE>",
    ] {
        require_count(&source, anchor, 1)?;
    }

    replace_exact(
        &mut source,
        "} // namespace sm89_fixed_half_swizzle\n\nnamespace sm89_test_half_nt_s3 {",
        &format!(
            "{M64_SUPPORT}\n}} // namespace sm89_fixed_half_swizzle\n\nnamespace sm89_test_half_nt_m64n128_s3 {{"
        ),
    )?;
    replace_exact(
        &mut source,
        "} // namespace sm89_test_half_nt_s3",
        "} // namespace sm89_test_half_nt_m64n128_s3",
    )?;
    replace_exact(
        &mut source,
        "static constexpr int kSharedBytes = 98304;\nstatic_assert(3 * (128 * 64 + 64 * 128) * 2 == kSharedBytes, \"S3 shared ABI\");",
        "static constexpr int kSharedBytes = 73728;\nstatic_assert(3 * (64 * 64 + 128 * 64) * 2 == kSharedBytes, \"M64N128 S3 shared ABI\");",
    )?;
    replace_exact(
        &mut source,
        "    T (*As)[128][64] = reinterpret_cast<T (*)[128][64]>(sm89_fhs_shared);",
        "    T (*As)[64][64] = reinterpret_cast<T (*)[64][64]>(sm89_fhs_shared);",
    )?;
    replace_exact(
        &mut source,
        "        sm89_fhs_shared + 3 * 128 * 64 * (int)sizeof(T));",
        "        sm89_fhs_shared + 3 * 64 * 64 * (int)sizeof(T));",
    )?;
    replace_exact(
        &mut source,
        "    int warpM = (warp >> 2) * 64;",
        "    int warpM = (warp >> 2) * 32;",
    )?;
    replace_range(
        &mut source,
        "    float acc[4][4][4];",
        "        sm89_fixed_half_swizzle::nt_fragment_offsets(warpM, warpN);",
        M64_ACCUMULATORS,
    )?;
    replace_exact(
        &mut source,
        "    sm89_fixed_half_swizzle::NtCopyPlan plan;\n    if (fast_stage) plan = sm89_fixed_half_swizzle::nt_copy_plan(As_sbase, Bs_sbase, pid_m, pid_n, M, K_out, N);",
        "    sm89_fixed_half_swizzle::NtM64N128CopyPlan plan;\n    if (fast_stage) plan = sm89_fixed_half_swizzle::nt_m64n128_copy_plan(As_sbase, Bs_sbase, pid_m, pid_n, M, K_out, N);",
    )?;
    replace_all_exact(
        &mut source,
        "sm89_fixed_half_swizzle::nt_copy_slice",
        "sm89_fixed_half_swizzle::nt_m64n128_copy_slice",
        6,
    )?;
    replace_all_exact(
        &mut source,
        "sm89_fixed_half_swizzle::nt_stage_scalar",
        "sm89_fixed_half_swizzle::nt_m64n128_stage_scalar",
        2,
    )?;
    replace_all_exact(
        &mut source,
        "sm89_fixed_half_swizzle::nt_load_fragments",
        "sm89_fixed_half_swizzle::nt_m64n128_load_fragments",
        7,
    )?;
    replace_all_exact(
        &mut source,
        "sm89_fixed_half_swizzle::consume_fragments<T>",
        "sm89_fixed_half_swizzle::nt_m64n128_consume_fragments<T>",
        5,
    )?;
    replace_all_exact(
        &mut source,
        "unsigned a_read = As_sbase + (unsigned)(read_buf * 128 * 64 * 2);",
        "unsigned a_read = As_sbase + (unsigned)(read_buf * 64 * 64 * 2);",
        2,
    )?;
    replace_exact(
        &mut source,
        "unsigned next_a_read = As_sbase + (unsigned)(read_buf * 128 * 64 * 2);",
        "unsigned next_a_read = As_sbase + (unsigned)(read_buf * 64 * 64 * 2);",
    )?;
    replace_exact(
        &mut source,
        "sm89_fixed_half_swizzle::vector_epilogue(C, reinterpret_cast<float*>(sm89_fhs_shared), acc, alpha,",
        "sm89_fixed_half_swizzle::nt_m64n128_vector_epilogue(C, reinterpret_cast<float*>(sm89_fhs_shared), acc, alpha,",
    )?;
    replace_exact(
        &mut source,
        "sm89_fixed_half_swizzle::scalar_epilogue(C, acc, alpha, 0.0f, M, K_out, K_out,",
        "sm89_fixed_half_swizzle::nt_m64n128_scalar_epilogue(C, acc, alpha, M, K_out,",
    )?;
    replace_exact(
        &mut source,
        "void gemm_bi_nt_test_fixed_s3_bxor_##SUFFIX",
        "void gemm_bi_nt_test_fixed_s3_m64n128_##SUFFIX",
    )?;
    replace_exact(
        &mut source,
        "sm89_test_half_nt_s3::kernel<TYPE>",
        "sm89_test_half_nt_m64n128_s3::kernel<TYPE>",
    )?;
    Ok(source)
}

const M64_ACCUMULATORS: &str = r#"    float acc[2][4][4];
#pragma unroll
    for (int fm = 0; fm < 2; ++fm)
#pragma unroll
        for (int fn = 0; fn < 4; ++fn)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[fm][fn][e] = 0.0f;
    sm89_fixed_half_swizzle::NtM64N128FragmentOffsets offsets =
        sm89_fixed_half_swizzle::nt_m64n128_fragment_offsets(warpM, warpN);
"#;

const M64_SUPPORT: &str = r#"
template <typename T>
static __device__ __forceinline__ void nt_m64n128_stage_scalar(
    T* a_stages, T* b_stages, const T* A, const T* B,
    int stage, int k_base, int pid_m, int pid_n,
    int M, int K_out, int N) {
    T* a_stage = a_stages + stage * 64 * 64;
    T* b_stage = b_stages + stage * 128 * 64;
    for (int linear = (int)threadIdx.x; linear < 64 * 64; linear += 256) {
        int row = linear / 64;
        int k = linear % 64;
        int global_row = pid_m * 64 + row;
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

struct NtM64N128FragmentOffsets {
    unsigned a[2];
    unsigned b[4];
};

static __device__ __forceinline__ NtM64N128FragmentOffsets
nt_m64n128_fragment_offsets(int warpM, int warpN) {
    NtM64N128FragmentOffsets offsets;
    int lane = (int)threadIdx.x & 31;
#pragma unroll
    for (int atom = 0; atom < 2; ++atom)
        offsets.a[atom] = layout::a_fragment_base(warpM, atom, lane);
#pragma unroll
    for (int atom = 0; atom < 4; ++atom)
        offsets.b[atom] = 2u * (unsigned)layout::a_index(
            warpN + atom * 8 + (lane & 7), ((lane >> 3) & 1) * 8);
    return offsets;
}

static __device__ __forceinline__ void nt_m64n128_load_fragments(
    unsigned a_stage, unsigned b_stage, int issue,
    const NtM64N128FragmentOffsets& offsets, Fragments& fragments) {
#pragma unroll
    for (int atom = 0; atom < 2; ++atom) {
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

template <typename T>
static __device__ __forceinline__ void nt_m64n128_consume_fragments(
    const Fragments& fragments, float (&acc)[2][4][4]) {
#pragma unroll
    for (int fm = 0; fm < 2; ++fm)
#pragma unroll
        for (int fn = 0; fn < 4; ++fn)
            HalfOps<T>::mma(acc[fm][fn], fragments.a[fm], fragments.b[fn]);
}

struct NtM64N128CopyPlan {
    long long a_offset[2];
    long long b_offset[4];
    unsigned a_destination[2];
    unsigned b_destination[4];
    bool a_row_valid[2];
    bool b_row_valid[4];
    int k;
};

static __device__ __forceinline__ NtM64N128CopyPlan nt_m64n128_copy_plan(
    unsigned a_shared, unsigned b_shared, int pid_m, int pid_n,
    int M, int K_out, int N) {
    NtM64N128CopyPlan plan;
    int thread = (int)threadIdx.x;
    plan.k = (thread & 7) * 8;
#pragma unroll
    for (int slice = 0; slice < 2; ++slice) {
        int local_row = (thread >> 3) + slice * 32;
        int a_row = pid_m * 64 + local_row;
        plan.a_row_valid[slice] = a_row < M;
        plan.a_offset[slice] = (long long)(a_row < M ? a_row : 0) * N + plan.k;
        plan.a_destination[slice] = a_shared + layout::a_copy_offset(thread, slice);
    }
#pragma unroll
    for (int slice = 0; slice < 4; ++slice) {
        int local_row = (thread >> 3) + slice * 32;
        int b_row = pid_n * 128 + local_row;
        plan.b_row_valid[slice] = b_row < K_out;
        plan.b_offset[slice] = (long long)(b_row < K_out ? b_row : 0) * N + plan.k;
        plan.b_destination[slice] = b_shared + layout::a_copy_offset(thread, slice);
    }
    return plan;
}

template <typename T>
static __device__ __forceinline__ void nt_m64n128_copy_slice(
    const NtM64N128CopyPlan& plan, const T* A, const T* B,
    int stage, int k_base, int N, int slice) {
    int remaining = N - k_base - plan.k;
    if (slice < 2) {
        int a_bytes = plan.a_row_valid[slice]
            ? (remaining >= 8 ? 16 : (remaining > 0 ? remaining * 2 : 0)) : 0;
        const void* a_source = a_bytes > 0
            ? (const void*)(A + plan.a_offset[slice] + k_base) : (const void*)A;
        unsigned a_destination = plan.a_destination[slice]
            + (unsigned)(stage * 64 * 64 * 2);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
            :: "r"(a_destination), "l"(a_source), "r"(a_bytes));
    }
    int b_bytes = plan.b_row_valid[slice]
        ? (remaining >= 8 ? 16 : (remaining > 0 ? remaining * 2 : 0)) : 0;
    const void* b_source = b_bytes > 0
        ? (const void*)(B + plan.b_offset[slice] + k_base) : (const void*)B;
    unsigned b_destination = plan.b_destination[slice]
        + (unsigned)(stage * 128 * 64 * 2);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
        :: "r"(b_destination), "l"(b_source), "r"(b_bytes));
}

template <typename T>
static __device__ __forceinline__ void nt_m64n128_scalar_epilogue(
    T* C, float (&acc)[2][4][4], float alpha,
    int M, int K_out, int pid_m, int pid_n, int warpM, int warpN) {
    int lane = (int)threadIdx.x & 31;
    int g = lane >> 2;
    int t = lane & 3;
#pragma unroll
    for (int fm = 0; fm < 2; ++fm) {
#pragma unroll
        for (int fn = 0; fn < 4; ++fn) {
            int r0 = pid_m * 64 + warpM + fm * 16 + g;
            int c0 = pid_n * 128 + warpN + fn * 8 + 2 * t;
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                int row = r0 + half * 8;
                if (row >= M) continue;
                float first = __fmul_rn(alpha, acc[fm][fn][2 * half]);
                float second = __fmul_rn(alpha, acc[fm][fn][2 * half + 1]);
                T* destination = c0 < K_out ? C + (long long)row * K_out + c0 : (T*)0;
                if ((K_out & 1) == 0 && c0 + 1 < K_out && gbf_aligned4(destination)) {
                    gbf_store_pair_rne(destination, first, second);
                } else {
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        int column = c0 + e;
                        if (column >= K_out) continue;
                        float value = e ? second : first;
                        T* output = C + (long long)row * K_out + column;
                        *output = HalfOps<T>::from_float(value);
                    }
                }
            }
        }
    }
}

template <typename T>
static __device__ __forceinline__ void nt_m64n128_vector_epilogue(
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
                *reinterpret_cast<float2*>(output_tile + row * kOutputStride + column) =
                    make_float2(acc[fm][fn][2 * half], acc[fm][fn][2 * half + 1]);
            }
    __syncthreads();
#pragma unroll
    for (int linear = (int)threadIdx.x; linear < 64 * 16; linear += 256) {
        int local_row = linear >> 4;
        int row = pid_m * 64 + local_row;
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
        return Err("half NT M64N128 S3 replacement boundaries are reversed".into());
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
            "half NT M64N128 S3 anchor {anchor:?}: expected {expected}, observed {actual}"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SWIZZLE: &str = include_str!("../../kernels/gemm_bi_inference/sm89_half_swizzle.cu");
    const S3: &str = include_str!("../../kernels/gemm_bi_inference/sm89_half_s3.cu");

    fn index(row: usize, k: usize) -> usize {
        row * 64 + (k ^ ((row & 7) * 8))
    }

    #[test]
    fn six_vector_copies_cover_the_m64_and_n128_stage_once() {
        let mut a = vec![0u8; 64 * 64];
        let mut b = vec![0u8; 128 * 64];
        for thread in 0..256 {
            let k = (thread & 7) * 8;
            for slice in 0..4 {
                let row = (thread >> 3) + slice * 32;
                if slice < 2 {
                    let base = index(row, k);
                    assert_eq!(2 * base % 16, 0);
                    for element in 0..8 {
                        a[index(row, k + element)] += 1;
                    }
                }
                let base = index(row, k);
                assert_eq!(2 * base % 16, 0);
                for element in 0..8 {
                    b[index(row, k + element)] += 1;
                }
            }
        }
        assert!(a.into_iter().all(|owners| owners == 1));
        assert!(b.into_iter().all(|owners| owners == 1));
        assert_eq!(3 * (64 * 64 + 128 * 64) * 2, 73_728);
    }

    #[test]
    fn eight_warps_own_m64_n128_outputs_and_fragment_addresses() {
        let mut outputs = vec![0u8; 64 * 128];
        for warp in 0..8 {
            let warp_m = (warp >> 2) * 32;
            let warp_n = (warp & 3) * 32;
            for lane in 0..32 {
                let group = lane >> 2;
                let thread = lane & 3;
                for issue in 0..4 {
                    for atom in 0..2 {
                        let row = warp_m + atom * 16 + (lane & 15);
                        let k0 = if lane & 16 != 0 { 8 } else { 0 };
                        let address = index(row, k0) ^ (issue * 16);
                        assert_eq!(address, index(row, k0 + issue * 16));
                        assert!(address < 64 * 64);
                        assert_eq!(2 * address % 16, 0);
                    }
                    for atom in 0..4 {
                        let row = warp_n + atom * 8 + (lane & 7);
                        let k0 = ((lane >> 3) & 1) * 8;
                        let address = index(row, k0) ^ (issue * 16);
                        assert_eq!(address, index(row, k0 + issue * 16));
                        assert!(address < 128 * 64);
                        assert_eq!(2 * address % 16, 0);
                    }
                }
                for fm in 0..2 {
                    for f_n in 0..4 {
                        let row = warp_m + fm * 16 + group;
                        let column = warp_n + f_n * 8 + 2 * thread;
                        for element in 0..4 {
                            let output_row = row + usize::from(element >= 2) * 8;
                            let output_column = column + (element & 1);
                            outputs[output_row * 128 + output_column] += 1;
                        }
                    }
                }
            }
        }
        assert!(outputs.into_iter().all(|owners| owners == 1));
    }

    fn k_loop(source: &str) -> String {
        source
            .split("    int num_k_tiles =")
            .nth(1)
            .expect("K loop start")
            .split("    {\n        bool vector_output")
            .next()
            .expect("K loop end")
            .replace("nt_m64n128_copy_slice", "nt_copy_slice")
            .replace("nt_m64n128_stage_scalar", "nt_stage_scalar")
            .replace("nt_m64n128_load_fragments", "nt_load_fragments")
            .replace("nt_m64n128_consume_fragments<T>", "consume_fragments<T>")
            .replace("read_buf * 64 * 64 * 2", "read_buf * 128 * 64 * 2")
    }

    #[test]
    fn composed_source_changes_only_the_m64_mapping_around_the_s3_loop() {
        let parent = parent::candidate_source(SWIZZLE, S3).unwrap();
        let source = candidate_source(SWIZZLE, S3).unwrap();
        assert_eq!(k_loop(&source), k_loop(&parent));
        assert!(source.contains("static_assert(3 * (64 * 64 + 128 * 64) * 2 == kSharedBytes"));
        assert!(source.contains("float acc[2][4][4];"));
        assert!(source.contains("for (int fm = 0; fm < 2; ++fm)"));
        assert!(source.contains("void gemm_bi_nt_test_fixed_s3_m64n128_##SUFFIX"));
        assert!(source.contains("gbf_store_pair_rne(destination, first, second);"));
        assert!(source.contains("*output = HalfOps<T>::from_float(value);"));
        assert!(!source.contains("void gemm_bi_nt_test_fixed_s3_bxor_##SUFFIX"));
    }

    #[test]
    fn missing_or_duplicated_parent_anchor_fails_closed() {
        let missing = S3.replacen("namespace sm89_fixed_half_s3 {", "", 1);
        assert!(candidate_source(SWIZZLE, &missing).is_err());
        let duplicate = format!("{S3}\nnamespace sm89_fixed_half_s3 {{\n");
        assert!(candidate_source(SWIZZLE, &duplicate).is_err());
    }
}
