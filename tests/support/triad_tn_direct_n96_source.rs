pub const GEMM_SYMBOL: &str = "gemm_bi_tn_test_direct_rna_n96_sm89_m128n96_bk32_s3";

const EXPECTED_TRANSPOSE_FNV64: u64 = 0xa784_a1b8_bdcf_3843;

pub const fn a_index(row: usize, k: usize) -> usize {
    k * 128 + (((row >> 2) ^ ((k & 3) << 1)) * 4) + (row & 3)
}

pub const fn direct_a_copy_coordinate(thread: usize, slice: usize) -> (usize, usize) {
    let linear = thread + slice * 256;
    (linear / 32, (linear % 32) * 4)
}

pub fn direct_a_copy_bytes(
    output_rows: usize,
    tile_row: usize,
    row: usize,
    reduction: usize,
    k_base: usize,
    k_row: usize,
) -> usize {
    if k_base.saturating_add(k_row) >= reduction {
        return 0;
    }
    output_rows
        .saturating_sub(tile_row.saturating_add(row))
        .min(4)
        * 4
}

pub const fn direct_a_fragment_coordinate(
    warp_m: usize,
    m_atom: usize,
    k8: usize,
    lane: usize,
    fragment: usize,
) -> (usize, usize) {
    let group = lane >> 2;
    let thread = lane & 3;
    let row = warp_m + m_atom * 16 + group + if fragment & 1 != 0 { 8 } else { 0 };
    let k = k8 + thread + if fragment >= 2 { 4 } else { 0 };
    (row, k)
}

pub fn candidate_source(transpose_candidate: &str) -> Result<String, String> {
    if fnv64(transpose_candidate.as_bytes()) != EXPECTED_TRANSPOSE_FNV64 {
        return Err("direct TN N96 input source digest changed".into());
    }
    let mut source = transpose_candidate.to_owned();
    replace_exact(
        &mut source,
        "gemm_bi_tn_test_transpose_rna_n96_sm89_m128n96_bk32_s3",
        GEMM_SYMBOL,
        1,
        "GEMM export",
    )?;
    replace_section(
        &mut source,
        "__device__ __forceinline__ int tf32n96_a_index",
        "\n\n__device__ __forceinline__ int tf32n96_b_index",
        DIRECT_A_INDEX,
        "A shared index",
    )?;
    replace_section(
        &mut source,
        "struct Tf32n96CopyPlan {",
        "\n\n__device__ __forceinline__ void tf32n96_copy_plan",
        DIRECT_COPY_PLAN_STRUCT,
        "copy-plan structure",
    )?;
    replace_section(
        &mut source,
        "__device__ __forceinline__ void tf32n96_copy_plan",
        "\n\n__device__ __forceinline__ void tf32n96_stage_slice",
        DIRECT_COPY_PLAN,
        "A copy plan",
    )?;
    replace_section(
        &mut source,
        "__device__ __forceinline__ void tf32n96_stage_slice",
        "\n\n__device__ __forceinline__ void tf32n96_advance_plan",
        DIRECT_STAGE_SLICE,
        "stage slice",
    )?;
    replace_section(
        &mut source,
        "__device__ __forceinline__ void tf32n96_advance_plan",
        "\n\n__device__ __forceinline__ void tf32n96_stage_async",
        DIRECT_ADVANCE_PLAN,
        "copy-plan advance",
    )?;
    replace_section(
        &mut source,
        "struct Tf32n96FragmentOffsets {",
        "\n\n__device__ __forceinline__ unsigned tf32n96_round",
        DIRECT_FRAGMENT_OFFSETS,
        "A fragment offsets",
    )?;
    replace_section(
        &mut source,
        "__device__ __forceinline__ void tf32n96_load_fragments",
        "\n\n__device__ __forceinline__ void tf32n96_mma",
        DIRECT_FRAGMENT_LOAD,
        "A fragment loads",
    )?;
    replace_exact(
        &mut source,
        "    long long b_slab_rows = 32LL * params.ldb;",
        concat!(
            "    long long a_slab_rows = 32LL * params.lda;\n",
            "    long long b_slab_rows = 32LL * params.ldb;"
        ),
        1,
        "slab advances",
    )?;
    replace_exact(
        &mut source,
        "tf32n96_advance_plan(plan, b_slab_rows);",
        "tf32n96_advance_plan(plan, a_slab_rows, b_slab_rows);",
        2,
        "advance calls",
    )?;
    replace_exact(
        &mut source,
        concat!(
            "    if (params.k == 0) {\n",
            "        tf32n96_zero_reduction(output, bias, params);\n",
            "        return;\n",
            "    }\n",
            "    tf32n96_kernel(output, a, b, bias, params);"
        ),
        concat!(
            "    if (params.k == 0) {\n",
            "        tf32n96_zero_reduction(output, bias, params);\n",
            "        return;\n",
            "    }\n",
            "    if (!gbf_aligned16(a) || !gbf_aligned16(b)\n",
            "        || (params.lda & 3) != 0 || (params.ldb & 3) != 0) return;\n",
            "    tf32n96_kernel(output, a, b, bias, params);"
        ),
        1,
        "aligned direct-TN wrapper",
    )?;
    let transpose_marker = "\n\nstruct GbfTf32TnTransposeParams {";
    if source.matches(transpose_marker).count() != 1 {
        return Err("direct TN N96 transpose appendix boundary changed".into());
    }
    source.truncate(
        source
            .find(transpose_marker)
            .ok_or("direct TN N96 transpose appendix is missing")?,
    );
    source.push('\n');
    Ok(source)
}

fn fnv64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn replace_exact(
    source: &mut String,
    from: &str,
    to: &str,
    expected: usize,
    label: &str,
) -> Result<(), String> {
    let count = source.matches(from).count();
    if count != expected {
        return Err(format!(
            "direct TN N96 {label} changed: expected {expected}, found {count}"
        ));
    }
    *source = source.replace(from, to);
    Ok(())
}

fn replace_section(
    source: &mut String,
    start: &str,
    end: &str,
    replacement: &str,
    label: &str,
) -> Result<(), String> {
    if source.matches(start).count() != 1 || source.matches(end).count() != 1 {
        return Err(format!("direct TN N96 {label} boundary changed"));
    }
    let begin = source
        .find(start)
        .ok_or_else(|| format!("missing {label} start"))?;
    let finish = source[begin..]
        .find(end)
        .map(|offset| begin + offset)
        .ok_or_else(|| format!("missing {label} end"))?;
    source.replace_range(begin..finish, replacement);
    Ok(())
}

const DIRECT_A_INDEX: &str = r#"__device__ __forceinline__ int tf32n96_a_index(int row, int k) {
    int chunk = (row >> 2) ^ ((k & 3) << 1);
    return k * 128 + chunk * 4 + (row & 3);
}"#;

const DIRECT_COPY_PLAN_STRUCT: &str = r#"struct Tf32n96CopyPlan {
    const float* a_source[4];
    const float* b_source[3];
    unsigned a_destination[4];
    unsigned b_destination[3];
    int a_k_row[4];
    int a_row_bytes[4];
    int b_k_row[3];
    int b_column_bytes[3];
};"#;

const DIRECT_COPY_PLAN: &str = r#"__device__ __forceinline__ void tf32n96_copy_plan(
    float* a_stage0, float* b_stage0, const float* a, const float* b,
    const GbfTf32N96Params& params, int tile_row, int tile_column,
    Tf32n96CopyPlan& plan) {
#pragma unroll
    for (int slice = 0; slice < 4; ++slice) {
        int linear = (int)threadIdx.x + slice * 256;
        int k_row = linear / 32;
        int row = (linear % 32) * 4;
        int global_row = tile_row + row;
        int rows = params.m - global_row;
        rows = rows < 0 ? 0 : (rows > 4 ? 4 : rows);
        plan.a_k_row[slice] = k_row;
        plan.a_row_bytes[slice] = rows * 4;
        plan.a_source[slice] =
            a + (long long)k_row * params.lda + (rows > 0 ? global_row : 0);
        plan.a_destination[slice] = (unsigned)__cvta_generic_to_shared(
            a_stage0 + tf32n96_a_index(row, k_row));
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
            b_stage0 + tf32n96_b_index(k_row, column));
    }
}"#;

const DIRECT_STAGE_SLICE: &str = r#"__device__ __forceinline__ void tf32n96_stage_slice(
    const Tf32n96CopyPlan& plan, unsigned a_stage_bytes,
    unsigned b_stage_bytes, int k_base, int reduction, int issue) {
    {
        int bytes = k_base + plan.a_k_row[issue] < reduction
            ? plan.a_row_bytes[issue]
            : 0;
        gbf_tf32_copy_cg(
            plan.a_destination[issue] + a_stage_bytes,
            plan.a_source[issue], bytes);
    }
    if (issue < 3) {
        int bytes = k_base + plan.b_k_row[issue] < reduction
            ? plan.b_column_bytes[issue]
            : 0;
        gbf_tf32_copy_cg(
            plan.b_destination[issue] + b_stage_bytes,
            plan.b_source[issue], bytes);
    }
}"#;

const DIRECT_ADVANCE_PLAN: &str = r#"__device__ __forceinline__ void tf32n96_advance_plan(
    Tf32n96CopyPlan& plan, long long a_rows, long long b_rows) {
#pragma unroll
    for (int slice = 0; slice < 4; ++slice) plan.a_source[slice] += a_rows;
#pragma unroll
    for (int slice = 0; slice < 3; ++slice) plan.b_source[slice] += b_rows;
}"#;

const DIRECT_FRAGMENT_OFFSETS: &str = r#"struct Tf32n96FragmentOffsets {
    int a[4][4][4];
    int b[3][2];
};

__device__ __forceinline__ void tf32n96_fragment_offsets(
    int warp_m, int warp_n, int group, int thread, int lane,
    Tf32n96FragmentOffsets& offsets) {
    (void)lane;
#pragma unroll
    for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
        for (int step = 0; step < 4; ++step) {
            int row = warp_m + m_atom * 16 + group;
            int k = step * 8 + thread;
            offsets.a[m_atom][step][0] = tf32n96_a_index(row, k);
            offsets.a[m_atom][step][1] = tf32n96_a_index(row + 8, k);
            offsets.a[m_atom][step][2] = tf32n96_a_index(row, k + 4);
            offsets.a[m_atom][step][3] = tf32n96_a_index(row + 8, k + 4);
        }
    }
#pragma unroll
    for (int n_atom = 0; n_atom < 3; ++n_atom) {
        int column = warp_n + n_atom * 8 + group;
        offsets.b[n_atom][0] = tf32n96_b_index(thread, column);
        offsets.b[n_atom][1] = tf32n96_b_index(thread + 4, column);
    }
}"#;

const DIRECT_FRAGMENT_LOAD: &str = r#"__device__ __forceinline__ void tf32n96_load_fragments(
    const float* a_stage, const float* b_stage, int step,
    const Tf32n96FragmentOffsets& offsets, Tf32n96Fragments& fragments) {
#pragma unroll
    for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
        for (int fragment = 0; fragment < 4; ++fragment) {
            fragments.a[m_atom][fragment] = tf32n96_round(__float_as_uint(
                a_stage[offsets.a[m_atom][step][fragment]]));
        }
    }
    const float* b_step = b_stage + step * 8 * 96;
#pragma unroll
    for (int n_atom = 0; n_atom < 3; ++n_atom) {
        fragments.b[n_atom][0] =
            tf32n96_round(__float_as_uint(b_step[offsets.b[n_atom][0]]));
        fragments.b[n_atom][1] =
            tf32n96_round(__float_as_uint(b_step[offsets.b[n_atom][1]]));
    }
}"#;

#[cfg(test)]
#[path = "triad_tn_transpose_n96_source.rs"]
mod transpose_n96_source_for_tests;

#[cfg(test)]
mod tests {
    use super::*;

    const FIXED_N96: &str = include_str!("../../kernels/gemm_bi_fixed/tf32_rna_n96.cu");

    #[test]
    fn direct_a_copy_plan_covers_each_stage_word_once_in_aligned_chunks() {
        let mut words = vec![0_u8; 32 * 128];
        for thread in 0..256 {
            for slice in 0..4 {
                let (k, row) = direct_a_copy_coordinate(thread, slice);
                assert!(k < 32 && row < 128 && row.is_multiple_of(4));
                for lane in 0..4 {
                    words[a_index(row + lane, k)] += 1;
                }
            }
        }
        assert!(words.into_iter().all(|count| count == 1));

        for lda in [36, 128, 768, 1_536] {
            for thread in 0..256 {
                for slice in 0..4 {
                    let (k, row) = direct_a_copy_coordinate(thread, slice);
                    assert!((k * lda + row).is_multiple_of(4));
                }
            }
        }

        assert_eq!(direct_a_copy_bytes(129, 128, 0, 33, 32, 0), 4);
        assert_eq!(direct_a_copy_bytes(129, 128, 4, 33, 32, 0), 0);
        assert_eq!(direct_a_copy_bytes(129, 128, 0, 33, 32, 1), 0);
        assert_eq!(direct_a_copy_bytes(128, 0, 124, 32, 0, 31), 16);
    }

    #[test]
    fn direct_a_fragment_reads_match_original_tn_coordinates_and_bank_permute() {
        for warp_m in [0, 64] {
            for m_atom in 0..4 {
                for k8 in [0, 8, 16, 24] {
                    for fragment in 0..4 {
                        let mut banks = [false; 32];
                        for lane in 0..32 {
                            let (row, k) =
                                direct_a_fragment_coordinate(warp_m, m_atom, k8, lane, fragment);
                            assert_eq!(row / 16, warp_m / 16 + m_atom);
                            assert!((k8..k8 + 8).contains(&k));
                            let bank = a_index(row, k) % 32;
                            assert!(!banks[bank]);
                            banks[bank] = true;
                        }
                        assert!(banks.into_iter().all(|seen| seen));
                    }
                }
            }
        }
    }

    #[test]
    fn source_adapter_is_fail_closed_and_preserves_n96_schedule_and_tn_epilogue() {
        let transpose = transpose_n96_source_for_tests::candidate_source(FIXED_N96).unwrap();
        let source = candidate_source(&transpose).unwrap();
        assert_eq!(source.matches(GEMM_SYMBOL).count(), 1);
        assert!(!source.contains(transpose_n96_source_for_tests::GEMM_SYMBOL));
        assert!(!source.contains(transpose_n96_source_for_tests::TRANSPOSE_SYMBOL));
        assert!(source.contains("cvt.rna.tf32.f32"));
        assert!(source.contains("Tf32n96Fragments fragments[2]"));
        assert!(source.contains("tf32n96_mma(fragments[issue & 1], acc)"));
        assert!(source.contains("__fmaf_rn(params.alpha, accumulator, *destination)"));
        assert_eq!(source.matches("__fmaf_rn(params.alpha, value.").count(), 4);
        assert!(!source.contains("ldmatrix.sync.aligned.m8n8.x4.shared.b16"));
        assert!(candidate_source("missing anchors").is_err());
    }
}
