pub const CANDIDATE_GEMM_SYMBOL: &str =
    "gemm_bi_tn_test_transpose_ab_pre_rna_n96_sm89_m128n96_bk32_s3";
pub const RETAINED_GEMM_SYMBOL: &str = "gemm_bi_tn_test_transpose_pre_rna_n96_sm89_m128n96_bk32_s3";
pub const CANDIDATE_PREPROCESS_SYMBOL: &str = "gemm_bi_tn_test_preprocess_ab_rna_u32_32x32_v1";
pub const RETAINED_PREPROCESS_SYMBOL: &str = "gemm_bi_tn_test_transpose_rna_u32_32x32_v1";

pub const fn tf32_rna_bits(bits: u32) -> u32 {
    if bits & 0x7f80_0000 == 0x7f80_0000 {
        bits
    } else {
        bits.wrapping_add(0x1000) & 0xffff_e000
    }
}

pub fn validate_rna_transposed_words(
    input: &[u32],
    rows: usize,
    columns: usize,
    stride: usize,
    output: &[u32],
) -> Result<(), String> {
    let input_len = rows
        .checked_mul(columns)
        .ok_or("TN AB-RNA input A extent overflows usize")?;
    let output_len = columns
        .checked_mul(stride)
        .ok_or("TN AB-RNA scratch A extent overflows usize")?;
    if input.len() != input_len || output.len() != output_len {
        return Err(format!(
            "TN AB-RNA A extent changed: input={}/{} output={}/{}",
            input.len(),
            input_len,
            output.len(),
            output_len
        ));
    }
    if stride < rows || !stride.is_multiple_of(4) {
        return Err(format!(
            "invalid TN AB-RNA A stride {stride} for {rows} rows"
        ));
    }
    for output_row in 0..columns {
        for output_column in 0..stride {
            let index = output_row * stride + output_column;
            let expected = if output_column < rows {
                tf32_rna_bits(input[output_column * columns + output_row])
            } else {
                0
            };
            if output[index] != expected {
                return Err(format!(
                    "TN AB-RNA A differs at ({output_row},{output_column}): actual={:#010x} expected={expected:#010x}",
                    output[index]
                ));
            }
        }
    }
    Ok(())
}

pub fn validate_rna_words(input: &[u32], output: &[u32]) -> Result<(), String> {
    if input.len() != output.len() {
        return Err(format!(
            "TN AB-RNA B extent changed: input={} output={}",
            input.len(),
            output.len()
        ));
    }
    for (index, (&input_word, &output_word)) in input.iter().zip(output).enumerate() {
        let expected = tf32_rna_bits(input_word);
        if output_word != expected {
            return Err(format!(
                "TN AB-RNA B differs at {index}: actual={output_word:#010x} expected={expected:#010x}"
            ));
        }
    }
    Ok(())
}

pub fn compose_candidate_source(retained_source: &str) -> Result<String, String> {
    let mut source = retained_source.to_owned();
    replace_exact(
        &mut source,
        RETAINED_GEMM_SYMBOL,
        CANDIDATE_GEMM_SYMBOL,
        "GEMM export",
    )?;
    replace_exact(
        &mut source,
        FRAGMENTS_WITH_B,
        FRAGMENTS_A_ONLY,
        "B fragment storage",
    )?;
    replace_exact(
        &mut source,
        B_ROUND_LOAD_LOOP,
        "",
        "B fragment preload loop",
    )?;
    replace_exact(
        &mut source,
        RETAINED_MMA,
        CANDIDATE_MMA,
        "just-in-time B MMA",
    )?;
    replace_exact(
        &mut source,
        RETAINED_MMA_CALL,
        CANDIDATE_MMA_CALL,
        "just-in-time B MMA call",
    )?;
    replace_exact(
        &mut source,
        RETAINED_PARAMS,
        CANDIDATE_PARAMS,
        "preprocess parameter ABI",
    )?;
    replace_exact(
        &mut source,
        RETAINED_SIGNATURE,
        CANDIDATE_SIGNATURE,
        "preprocess signature",
    )?;
    replace_exact(
        &mut source,
        "? input[(long long)input_row * params.columns + input_column]",
        "? input_a[(long long)input_row * params.columns + input_column]",
        "A input name",
    )?;
    replace_exact(
        &mut source,
        "output[(long long)output_row * params.output_stride + output_column] =",
        "output_a[(long long)output_row * params.output_stride + output_column] =",
        "A output name",
    )?;
    replace_exact(
        &mut source,
        PRE_SYNC,
        AB_PRE_SYNC,
        "joint B preprocess loop",
    )?;
    Ok(source)
}

pub fn restore_retained_source(candidate_source: &str) -> Result<String, String> {
    let mut source = candidate_source.to_owned();
    replace_exact(
        &mut source,
        AB_PRE_SYNC,
        PRE_SYNC,
        "restored preprocess body",
    )?;
    replace_exact(
        &mut source,
        "output_a[(long long)output_row * params.output_stride + output_column] =",
        "output[(long long)output_row * params.output_stride + output_column] =",
        "restored A output name",
    )?;
    replace_exact(
        &mut source,
        "? input_a[(long long)input_row * params.columns + input_column]",
        "? input[(long long)input_row * params.columns + input_column]",
        "restored A input name",
    )?;
    replace_exact(
        &mut source,
        CANDIDATE_SIGNATURE,
        RETAINED_SIGNATURE,
        "restored preprocess signature",
    )?;
    replace_exact(
        &mut source,
        CANDIDATE_PARAMS,
        RETAINED_PARAMS,
        "restored preprocess parameter ABI",
    )?;
    replace_exact(
        &mut source,
        CANDIDATE_MMA_CALL,
        RETAINED_MMA_CALL,
        "restored MMA call",
    )?;
    replace_exact(
        &mut source,
        CANDIDATE_MMA,
        RETAINED_MMA,
        "restored MMA body",
    )?;
    let restored_b_load =
        format!("    }}{B_ROUND_LOAD_LOOP}\n}}\n\n__device__ __forceinline__ void tf32n96_mma(");
    replace_exact(
        &mut source,
        "    }\n}\n\n__device__ __forceinline__ void tf32n96_mma(",
        &restored_b_load,
        "restored B fragment preload loop",
    )?;
    replace_exact(
        &mut source,
        FRAGMENTS_A_ONLY,
        FRAGMENTS_WITH_B,
        "restored B fragment storage",
    )?;
    replace_exact(
        &mut source,
        CANDIDATE_GEMM_SYMBOL,
        RETAINED_GEMM_SYMBOL,
        "restored GEMM export",
    )?;
    Ok(source)
}

const FRAGMENTS_WITH_B: &str = r#"struct Tf32n96Fragments {
    unsigned a[4][4];
    unsigned b[3][2];
};"#;

const FRAGMENTS_A_ONLY: &str = r#"struct Tf32n96Fragments {
    unsigned a[4][4];
};"#;

const B_ROUND_LOAD_LOOP: &str = r#"
    const float* b_step = b_stage + step * 8 * 96;
#pragma unroll
    for (int n_atom = 0; n_atom < 3; ++n_atom) {
        fragments.b[n_atom][0] =
            tf32n96_round(__float_as_uint(b_step[offsets.b[n_atom][0]]));
        fragments.b[n_atom][1] =
            tf32n96_round(__float_as_uint(b_step[offsets.b[n_atom][1]]));
    }"#;

const RETAINED_MMA: &str = r#"__device__ __forceinline__ void tf32n96_mma(
    const Tf32n96Fragments& fragments, float (&acc)[4][3][4]) {
#pragma unroll
    for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 3; ++n_atom) {
            gbf_tf32_mma_m16n8k8(
                acc[m_atom][n_atom], fragments.a[m_atom], fragments.b[n_atom]);
        }
    }
}"#;

const CANDIDATE_MMA: &str = r#"__device__ __forceinline__ void tf32n96_mma_pre_rounded_b(
    const Tf32n96Fragments& fragments, const float* b_stage, int step,
    const Tf32n96FragmentOffsets& offsets, float (&acc)[4][3][4]) {
    const float* b_step = b_stage + step * 8 * 96;
    {
        unsigned b_fragment[2] = {
            __float_as_uint(b_step[offsets.b[0][0]]),
            __float_as_uint(b_step[offsets.b[0][1]])
        };
#pragma unroll
        for (int m_atom = 0; m_atom < 4; ++m_atom) {
            gbf_tf32_mma_m16n8k8(
                acc[m_atom][0], fragments.a[m_atom], b_fragment);
        }
    }
    {
        unsigned b_fragment[2] = {
            __float_as_uint(b_step[offsets.b[1][0]]),
            __float_as_uint(b_step[offsets.b[1][1]])
        };
#pragma unroll
        for (int m_atom = 0; m_atom < 4; ++m_atom) {
            gbf_tf32_mma_m16n8k8(
                acc[m_atom][1], fragments.a[m_atom], b_fragment);
        }
    }
    {
        unsigned b_fragment[2] = {
            __float_as_uint(b_step[offsets.b[2][0]]),
            __float_as_uint(b_step[offsets.b[2][1]])
        };
#pragma unroll
        for (int m_atom = 0; m_atom < 4; ++m_atom) {
            gbf_tf32_mma_m16n8k8(
                acc[m_atom][2], fragments.a[m_atom], b_fragment);
        }
    }
}"#;

const RETAINED_MMA_CALL: &str = "            tf32n96_mma(fragments[issue & 1], acc);";
const CANDIDATE_MMA_CALL: &str = r#"            tf32n96_mma_pre_rounded_b(
                fragments[issue & 1], b_read, issue, offsets, acc);"#;

const RETAINED_PARAMS: &str = r#"struct GbfTf32TnTransposeParams {
    int rows;
    int columns;
    int output_stride;
};

static_assert(sizeof(GbfTf32TnTransposeParams) == 12, "TN transpose parameter ABI");
static_assert(alignof(GbfTf32TnTransposeParams) == 4, "TN transpose parameter alignment");"#;

const CANDIDATE_PARAMS: &str = r#"struct GbfTf32TnTransposeParams {
    int rows;
    int columns;
    int output_stride;
    int b_elements;
};

static_assert(sizeof(GbfTf32TnTransposeParams) == 16, "TN AB preprocess parameter ABI");
static_assert(alignof(GbfTf32TnTransposeParams) == 4, "TN AB preprocess parameter alignment");"#;

const RETAINED_SIGNATURE: &str = r#"void gemm_bi_tn_test_transpose_rna_u32_32x32_v1(
    const unsigned* input, unsigned* output, GbfTf32TnTransposeParams params) {"#;

const CANDIDATE_SIGNATURE: &str = r#"void gemm_bi_tn_test_preprocess_ab_rna_u32_32x32_v1(
    const unsigned* input_a, unsigned* output_a,
    const unsigned* input_b, unsigned* output_b,
    GbfTf32TnTransposeParams params) {"#;

const PRE_SYNC: &str = r#"    }
    __syncthreads();
    int output_column = (int)blockIdx.y * 32 + (int)threadIdx.x;"#;

const AB_PRE_SYNC: &str = r#"    }
    long long block_linear = (long long)blockIdx.y * gridDim.x + blockIdx.x;
    long long thread_linear = block_linear * blockDim.x * blockDim.y
        + (long long)threadIdx.y * blockDim.x + threadIdx.x;
    long long thread_count = (long long)gridDim.x * gridDim.y
        * blockDim.x * blockDim.y;
    for (long long linear = thread_linear; linear < params.b_elements;
         linear += thread_count) {
        output_b[linear] = tf32n96_round(input_b[linear]);
    }
    __syncthreads();
    int output_column = (int)blockIdx.y * 32 + (int)threadIdx.x;"#;

fn replace_exact(source: &mut String, old: &str, new: &str, label: &str) -> Result<(), String> {
    let count = source.matches(old).count();
    if count != 1 {
        return Err(format!(
            "TN joint AB-RNA N96 {label} seam changed: expected 1, observed {count}"
        ));
    }
    *source = source.replacen(old, new, 1);
    Ok(())
}
