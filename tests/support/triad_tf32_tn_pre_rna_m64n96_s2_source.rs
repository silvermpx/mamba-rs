pub const GEMM_SYMBOL: &str = "tn_test_pre_rna_m64n96_bk32_s2_prism";
pub const RETAINED_GEMM_SYMBOL: &str = "tn_sm89_tf32_pre_rna_m64n64_bk32_s3";
pub const K8_ISSUE_OFFSETS: [u32; 4] = [0, 8, 16, 24];

const EXPECTED_PRIMITIVES_FNV64: u64 = 0x0247_c603_860d_c16f;
const EXPECTED_PARENT_FNV64: u64 = 0x0327_a80f_17d3_bd55;
const SECTION_BEGIN: &str = "// BEGIN RETAINED TN_M64N64\n";
const SECTION_END: &str = "// END RETAINED TN_M64N64";

pub fn compose_candidate_source(primitives: &str, joint_source: &str) -> Result<String, String> {
    require_digest(primitives, EXPECTED_PRIMITIVES_FNV64, "joint primitives")?;
    let mut section = extract_unique_section(joint_source, SECTION_BEGIN, SECTION_END)?;
    require_digest(
        &section,
        EXPECTED_PARENT_FNV64,
        "retained TN M64N64 parent section",
    )?;
    replace_exact(
        &mut section,
        "namespace sm89_tf32_joint_tn_m64n64 {",
        "namespace sm89_tf32_test_tn_m64n96_s2 {",
        "namespace",
    )?;
    replace_exact(
        &mut section,
        "// Ada Fixed TF32 M128xN96/BK32/S3 finalist. Optional forced route until qualified.",
        "// Test-only Ada TF32 TN M64xN96/BK32/S2 candidate.",
        "description",
    )?;
    replace_exact(
        &mut section,
        RETAINED_GEMM_SYMBOL,
        GEMM_SYMBOL,
        "GEMM export",
    )?;
    replace_exact(
        &mut section,
        "extern \"C\" __global__ __launch_bounds__(256, 1)",
        "extern \"C\" __global__ __launch_bounds__(256, 2)",
        "launch bounds",
    )?;
    replace_all_exact(&mut section, "Tf32m64n64", "Tf32m64n96", 14)?;
    replace_all_exact(&mut section, "tf32m64n64", "tf32m64n96", 34)?;
    replace_exact(
        &mut section,
        "return k * 64 + chunk * 4 + (column & 3);",
        "return k * 96 + chunk * 4 + (column & 3);",
        "B shared stride",
    )?;
    replace_range(
        &mut section,
        "struct Tf32m64n96CopyPlan {",
        "struct Tf32m64n96Fragments {",
        COPY_PIPELINE,
        "copy pipeline",
    )?;
    replace_range(
        &mut section,
        "struct Tf32m64n96Fragments {",
        "__device__ __forceinline__ void tf32m64n96_store(",
        FRAGMENT_PIPELINE,
        "fragment pipeline",
    )?;
    replace_range(
        &mut section,
        "__device__ __forceinline__ void tf32m64n96_zero_reduction(",
        "__device__ __forceinline__ void tf32m64n96_kernel(",
        ZERO_REDUCTION,
        "zero-reduction geometry",
    )?;
    let export = format!("extern \"C\" __global__ __launch_bounds__(256, 2)\nvoid {GEMM_SYMBOL}(");
    replace_range(
        &mut section,
        "__device__ __forceinline__ void tf32m64n96_kernel(",
        &export,
        M64N96_S2_KERNEL,
        "M64N96/S2 kernel",
    )?;
    for (needle, expected) in [
        (GEMM_SYMBOL, 1),
        (RETAINED_GEMM_SYMBOL, 0),
        ("float acc[2][3][4];", 1),
        ("const int k8_offsets[4] = {0, 8, 16, 24};", 1),
        ("fragments.a[m_atom][0] = raw0;", 1),
        ("tf32m64n96_round(__float_as_uint(b_step", 2),
        (
            "*destination = __fmaf_rn(params.alpha, accumulator, *destination);",
            1,
        ),
    ] {
        require_count(&section, needle, expected)?;
    }
    Ok(format!("{primitives}\n{joint_source}\n{section}"))
}

fn require_digest(source: &str, expected: u64, label: &str) -> Result<(), String> {
    let observed = fnv1a64(source.as_bytes());
    if observed == expected {
        Ok(())
    } else {
        Err(format!(
            "TF32 TN M64N96/S2 {label} changed: expected {expected:#018x}, observed {observed:#018x}"
        ))
    }
}

fn extract_unique_section(source: &str, begin: &str, end: &str) -> Result<String, String> {
    require_count(source, begin, 1)?;
    require_count(source, end, 1)?;
    let body = source
        .split_once(begin)
        .ok_or("missing retained TN M64N64 begin marker")?
        .1
        .split_once(end)
        .ok_or("missing retained TN M64N64 end marker")?
        .0;
    Ok(body.to_owned())
}

fn replace_exact(
    source: &mut String,
    before: &str,
    after: &str,
    label: &str,
) -> Result<(), String> {
    require_count(source, before, 1).map_err(|error| format!("{label}: {error}"))?;
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
    label: &str,
) -> Result<(), String> {
    require_count(source, start, 1).map_err(|error| format!("{label}: {error}"))?;
    require_count(source, end, 1).map_err(|error| format!("{label}: {error}"))?;
    let begin = source.find(start).expect("validated start");
    let finish = source.find(end).expect("validated end");
    if begin >= finish {
        return Err(format!("TF32 TN M64N96/S2 {label} boundaries are reversed"));
    }
    source.replace_range(begin..finish, replacement);
    Ok(())
}

fn require_count(source: &str, needle: &str, expected: usize) -> Result<(), String> {
    let observed = source.matches(needle).count();
    if observed == expected {
        Ok(())
    } else {
        Err(format!(
            "TF32 TN M64N96/S2 anchor {needle:?}: expected {expected}, observed {observed}"
        ))
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

const COPY_PIPELINE: &str = r#"struct Tf32m64n96CopyPlan {
    const float* a_source[2];
    const float* b_source[3];
    unsigned a_destination[2];
    unsigned b_destination[3];
    int a_k_offset;
    int b_k_row[3];
    bool a_row_valid[2];
    int b_column_bytes[3];
};

__device__ __forceinline__ void tf32m64n96_copy_plan(
    float* a_stage0, float* b_stage0, const float* a, const float* b,
    const GbfTf32N96Params& params, int tile_row, int tile_column,
    Tf32m64n96CopyPlan& plan) {
    plan.a_k_offset = ((int)threadIdx.x & 7) * 4;
#pragma unroll
    for (int slice = 0; slice < 2; ++slice) {
        int linear = (int)threadIdx.x + slice * 256;
        int row = linear >> 3;
        int global_row = tile_row + row;
        plan.a_row_valid[slice] = global_row < params.m;
        plan.a_source[slice] =
            a + (long long)(plan.a_row_valid[slice] ? global_row : 0) * params.lda
            + plan.a_k_offset;
        plan.a_destination[slice] = (unsigned)__cvta_generic_to_shared(
            a_stage0 + tf32m64n96_a_index(row, plan.a_k_offset));
    }
#pragma unroll
    for (int slice = 0; slice < 3; ++slice) {
        int linear = (int)threadIdx.x + slice * 256;
        int k_row = linear / 24;
        int column = (linear % 24) * 4;
        int global_column = tile_column + column;
        int columns = params.n - global_column;
        columns = columns < 0 ? 0 : (columns > 4 ? 4 : columns);
        plan.b_k_row[slice] = k_row;
        plan.b_column_bytes[slice] = columns * 4;
        plan.b_source[slice] =
            b + (long long)k_row * params.ldb + (columns > 0 ? global_column : 0);
        plan.b_destination[slice] = (unsigned)__cvta_generic_to_shared(
            b_stage0 + tf32m64n96_b_index(k_row, column));
    }
}

__device__ __forceinline__ void tf32m64n96_stage_slice(
    const Tf32m64n96CopyPlan& plan, unsigned a_stage_bytes,
    unsigned b_stage_bytes, int k_base, int reduction, int issue) {
    if (issue < 2) {
        int remaining = reduction - k_base - plan.a_k_offset;
        remaining = remaining < 0 ? 0 : (remaining > 4 ? 4 : remaining);
        int a_bytes = plan.a_row_valid[issue] ? remaining * 4 : 0;
        gbf_tf32_copy_cg(
            plan.a_destination[issue] + a_stage_bytes,
            plan.a_source[issue], a_bytes);
    }
    if (issue < 3) {
        int b_bytes = k_base + plan.b_k_row[issue] < reduction
            ? plan.b_column_bytes[issue]
            : 0;
        gbf_tf32_copy_cg(
            plan.b_destination[issue] + b_stage_bytes,
            plan.b_source[issue], b_bytes);
    }
}

__device__ __forceinline__ void tf32m64n96_advance_plan(
    Tf32m64n96CopyPlan& plan, long long b_rows) {
#pragma unroll
    for (int slice = 0; slice < 2; ++slice) plan.a_source[slice] += 32;
#pragma unroll
    for (int slice = 0; slice < 3; ++slice) plan.b_source[slice] += b_rows;
}

__device__ __forceinline__ void tf32m64n96_stage_async(
    const Tf32m64n96CopyPlan& plan, unsigned a_stage_bytes,
    unsigned b_stage_bytes, int k_base, int reduction) {
#pragma unroll
    for (int issue = 0; issue < 4; ++issue) {
        tf32m64n96_stage_slice(
            plan, a_stage_bytes, b_stage_bytes, k_base, reduction, issue);
    }
    asm volatile("cp.async.commit_group;\n" ::);
}

"#;

const FRAGMENT_PIPELINE: &str = r#"struct Tf32m64n96Fragments {
    unsigned a[2][4];
    unsigned b[3][2];
};

struct Tf32m64n96FragmentOffsets {
    int a[2][4];
    int b[3][2];
};

__device__ __forceinline__ void tf32m64n96_fragment_offsets(
    int warp_m, int warp_n, int group, int thread, int lane,
    Tf32m64n96FragmentOffsets& offsets) {
    int a_row = warp_m + (lane & 15);
    int a_k = (lane >> 4) << 2;
#pragma unroll
    for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
        for (int step = 0; step < 4; ++step) {
            offsets.a[m_atom][step] =
                tf32m64n96_a_index(a_row + m_atom * 16, step * 8 + a_k);
        }
    }
#pragma unroll
    for (int n_atom = 0; n_atom < 3; ++n_atom) {
        int column = warp_n + n_atom * 8 + group;
        offsets.b[n_atom][0] = tf32m64n96_b_index(thread, column);
        offsets.b[n_atom][1] = tf32m64n96_b_index(thread + 4, column);
    }
}

__device__ __forceinline__ unsigned tf32m64n96_round(unsigned bits) {
    float value = __uint_as_float(bits);
    unsigned result;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(result) : "f"(value));
    return result;
}

__device__ __forceinline__ void tf32m64n96_load_fragments(
    const float* a_stage, const float* b_stage, int k8,
    const Tf32m64n96FragmentOffsets& offsets, Tf32m64n96Fragments& fragments) {
    int step = k8 >> 3;
    unsigned a_base = (unsigned)__cvta_generic_to_shared(a_stage);
#pragma unroll
    for (int m_atom = 0; m_atom < 2; ++m_atom) {
        unsigned address = a_base + (unsigned)offsets.a[m_atom][step] * 4U;
        unsigned raw0, raw1, raw2, raw3;
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(raw0), "=r"(raw1), "=r"(raw2), "=r"(raw3)
            : "r"(address));
        fragments.a[m_atom][0] = raw0;
        fragments.a[m_atom][1] = raw1;
        fragments.a[m_atom][2] = raw2;
        fragments.a[m_atom][3] = raw3;
    }
    const float* b_step = b_stage + k8 * 96;
#pragma unroll
    for (int n_atom = 0; n_atom < 3; ++n_atom) {
        fragments.b[n_atom][0] =
            tf32m64n96_round(__float_as_uint(b_step[offsets.b[n_atom][0]]));
        fragments.b[n_atom][1] =
            tf32m64n96_round(__float_as_uint(b_step[offsets.b[n_atom][1]]));
    }
}

__device__ __forceinline__ void tf32m64n96_mma(
    const Tf32m64n96Fragments& fragments, float (&acc)[2][3][4]) {
#pragma unroll
    for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 3; ++n_atom) {
            gbf_tf32_mma_m16n8k8(
                acc[m_atom][n_atom], fragments.a[m_atom], fragments.b[n_atom]);
        }
    }
}

"#;

const ZERO_REDUCTION: &str = r#"__device__ __forceinline__ void tf32m64n96_zero_reduction(
    float* output, const float* bias, const GbfTf32N96Params& params) {
    int column_tiles = (params.n + 95) / 96;
    int tile_row = (int)blockIdx.x / column_tiles * 64;
    int tile_column = (int)blockIdx.x % column_tiles * 96;
    for (int linear = (int)threadIdx.x; linear < 64 * 96;
         linear += (int)blockDim.x) {
        int row = tile_row + linear / 96;
        int column = tile_column + linear % 96;
        if (row < params.m && column < params.n) {
            float accumulator = bias == nullptr ? 0.0f : bias[column];
            tf32m64n96_store(output, row, column, accumulator, params);
        }
    }
}

"#;

const M64N96_S2_KERNEL: &str = r#"__device__ __forceinline__ void tf32m64n96_kernel(
    float* output, const float* a, const float* b, const float* bias,
    GbfTf32N96Params params) {
    extern __shared__ __align__(16) unsigned char shared_bytes[];
    float* a_stages = reinterpret_cast<float*>(shared_bytes);
    float* b_stages = a_stages + 2 * 64 * 32;
    int column_tiles = (params.n + 95) / 96;
    int tile_row = (int)blockIdx.x / column_tiles * 64;
    int tile_column = (int)blockIdx.x % column_tiles * 96;
    int warp = (int)threadIdx.x >> 5;
    int lane = (int)threadIdx.x & 31;
    int warp_m = (warp >> 2) * 32;
    int warp_n = (warp & 3) * 24;
    int group = lane >> 2;
    int thread = lane & 3;
    float acc[2][3][4];
#pragma unroll
    for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 3; ++n_atom) {
#pragma unroll
            for (int element = 0; element < 4; ++element) {
                int row = tile_row + warp_m + m_atom * 16 + group
                    + (element >= 2 ? 8 : 0);
                int column = tile_column + warp_n + n_atom * 8
                    + 2 * thread + (element & 1);
                acc[m_atom][n_atom][element] =
                    row < params.m && column < params.n && bias != nullptr
                    ? bias[column]
                    : 0.0f;
            }
        }
    }
    Tf32m64n96FragmentOffsets offsets;
    tf32m64n96_fragment_offsets(warp_m, warp_n, group, thread, lane, offsets);
    Tf32m64n96CopyPlan plan;
    tf32m64n96_copy_plan(
        a_stages, b_stages, a, b, params, tile_row, tile_column, plan);
    long long b_slab_rows = 32LL * params.ldb;
    unsigned tile_count = (static_cast<unsigned>(params.k) + 31U) / 32U;
    if (tile_count != 0) {
        tf32m64n96_stage_async(plan, 0, 0, 0, params.k);
        tf32m64n96_advance_plan(plan, b_slab_rows);
    } else {
        asm volatile("cp.async.commit_group;\n" ::);
    }
    int read_stage = 0;
    const int k8_offsets[4] = {0, 8, 16, 24};
    for (unsigned tile = 0; tile < tile_count; ++tile) {
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        unsigned next = tile + 1;
        bool has_next = next < tile_count;
        int write_stage = read_stage ^ 1;
        unsigned write_a_bytes = (unsigned)write_stage * 64U * 32U * 4U;
        unsigned write_b_bytes = (unsigned)write_stage * 32U * 96U * 4U;
        const float* a_read = a_stages + read_stage * 64 * 32;
        const float* b_read = b_stages + read_stage * 32 * 96;
        Tf32m64n96Fragments fragments[2];
        tf32m64n96_load_fragments(
            a_read, b_read, k8_offsets[0], offsets, fragments[0]);
#pragma unroll
        for (int issue = 0; issue < 4; ++issue) {
            if (has_next) {
                tf32m64n96_stage_slice(
                    plan, write_a_bytes, write_b_bytes,
                    (int)(next * 32U), params.k, issue);
            }
            if (issue < 3) {
                tf32m64n96_load_fragments(
                    a_read, b_read, k8_offsets[issue + 1], offsets,
                    fragments[(issue + 1) & 1]);
            }
            tf32m64n96_mma(fragments[issue & 1], acc);
        }
        asm volatile("cp.async.commit_group;\n" ::);
        if (has_next) tf32m64n96_advance_plan(plan, b_slab_rows);
        read_stage ^= 1;
    }
    __syncthreads();
    float* tile_output = reinterpret_cast<float*>(shared_bytes);
    bool vector_rows = tile_column + 96 <= params.n && (params.ldc & 3) == 0
        && gbf_aligned16(output);
    if (vector_rows) {
#pragma unroll
        for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
            for (int n_atom = 0; n_atom < 3; ++n_atom) {
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    int row = warp_m + m_atom * 16 + group + half * 8;
                    int column = warp_n + n_atom * 8 + 2 * thread;
                    *reinterpret_cast<float2*>(tile_output + row * 104 + column) =
                        make_float2(
                            acc[m_atom][n_atom][2 * half],
                            acc[m_atom][n_atom][2 * half + 1]);
                }
            }
        }
        __syncthreads();
#pragma unroll 4
        for (int linear = (int)threadIdx.x; linear < 64 * 24; linear += 256) {
            int row = linear / 24;
            int chunk = (linear % 24) * 4;
            int global_row = tile_row + row;
            if (global_row >= params.m) continue;
            float4 value = *reinterpret_cast<const float4*>(
                tile_output + row * 104 + chunk);
            float* destination =
                output + (long long)global_row * params.ldc + tile_column + chunk;
            float4 old = *reinterpret_cast<const float4*>(destination);
            value.x = __fmaf_rn(params.alpha, value.x, old.x);
            value.y = __fmaf_rn(params.alpha, value.y, old.y);
            value.z = __fmaf_rn(params.alpha, value.z, old.z);
            value.w = __fmaf_rn(params.alpha, value.w, old.w);
            *reinterpret_cast<float4*>(destination) = value;
        }
        return;
    }
#pragma unroll
    for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 3; ++n_atom) {
#pragma unroll
            for (int element = 0; element < 4; ++element) {
                int row = tile_row + warp_m + m_atom * 16 + group
                    + (element >= 2 ? 8 : 0);
                int column = tile_column + warp_n + n_atom * 8
                    + 2 * thread + (element & 1);
                tf32m64n96_store(
                    output, row, column, acc[m_atom][n_atom][element], params);
            }
        }
    }
}

"#;

#[cfg(test)]
mod tests {
    use super::*;

    const PRIMITIVES: &str =
        include_str!("../../kernels/gemm_bi_triad/sm80/tf32_joint_primitives.cuh");
    const JOINT_SOURCE: &str = include_str!("../../kernels/gemm_bi_triad/sm80/tf32_joint.cu");

    #[test]
    fn changed_parent_or_boundaries_fail_closed() {
        assert!(compose_candidate_source(&format!("{PRIMITIVES}\n"), JOINT_SOURCE).is_err());
        assert!(
            compose_candidate_source(
                PRIMITIVES,
                &JOINT_SOURCE.replacen(
                    "namespace sm89_tf32_joint_tn_m64n64 {",
                    "namespace changed_parent {",
                    1,
                ),
            )
            .is_err()
        );
        assert!(
            compose_candidate_source(
                PRIMITIVES,
                &JOINT_SOURCE.replacen(SECTION_END, "// END CHANGED TN_M64N64", 1),
            )
            .is_err()
        );
    }

    #[test]
    fn unrelated_joint_sections_do_not_invalidate_the_retained_parent() {
        let extended = format!(
            "{JOINT_SOURCE}\n// BEGIN UNRELATED FUTURE SECTION\n// deliberately ignored\n// END UNRELATED FUTURE SECTION\n"
        );
        assert!(compose_candidate_source(PRIMITIVES, &extended).is_ok());
    }

    #[test]
    fn candidate_is_unique_and_keeps_the_joint_arithmetic_contract() {
        let candidate = compose_candidate_source(PRIMITIVES, JOINT_SOURCE).unwrap();
        let candidate_section = candidate
            .rsplit_once("namespace sm89_tf32_test_tn_m64n96_s2 {")
            .expect("candidate namespace")
            .1;
        assert_eq!(candidate.matches(GEMM_SYMBOL).count(), 1);
        assert_eq!(candidate.matches(RETAINED_GEMM_SYMBOL).count(), 1);
        assert_eq!(K8_ISSUE_OFFSETS, [0, 8, 16, 24]);
        assert!(candidate_section.contains("fragments.a[m_atom][0] = raw0;"));
        assert_eq!(
            candidate_section
                .matches("*destination = __fmaf_rn(params.alpha, accumulator, *destination);")
                .count(),
            1
        );
    }
}
