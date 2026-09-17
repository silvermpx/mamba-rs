pub const GEMM_SYMBOL: &str = "tn_test_transpose_rna_n96_sm89_m128n96_bk32_s3";
pub const TRANSPOSE_SYMBOL: &str = "tn_test_transpose_raw_u32_32x32";

pub fn transpose_source() -> &'static str {
    TRANSPOSE_CUDA
}

pub fn padded_stride(rows: usize) -> Result<usize, String> {
    rows.checked_add(3)
        .map(|value| value & !3)
        .ok_or_else(|| "TN transpose padded stride overflows usize".into())
}

pub fn transpose_destination_index(
    rows: usize,
    columns: usize,
    stride: usize,
    source_row: usize,
    source_column: usize,
) -> Result<usize, String> {
    if stride < rows || !stride.is_multiple_of(4) {
        return Err(format!(
            "invalid TN transpose stride {stride} for {rows} rows"
        ));
    }
    if source_row >= rows || source_column >= columns {
        return Err(format!(
            "TN transpose source index ({source_row},{source_column}) exceeds ({rows},{columns})"
        ));
    }
    source_column
        .checked_mul(stride)
        .and_then(|base| base.checked_add(source_row))
        .ok_or_else(|| "TN transpose destination index overflows usize".into())
}

pub fn validate_transposed_words(
    input: &[u32],
    rows: usize,
    columns: usize,
    stride: usize,
    output: &[u32],
) -> Result<(), String> {
    let input_len = rows
        .checked_mul(columns)
        .ok_or("TN transpose input extent overflows usize")?;
    let output_len = columns
        .checked_mul(stride)
        .ok_or("TN transpose output extent overflows usize")?;
    if input.len() != input_len || output.len() != output_len {
        return Err(format!(
            "TN transpose extent changed: input={}/{} output={}/{}",
            input.len(),
            input_len,
            output.len(),
            output_len
        ));
    }
    if stride < rows || !stride.is_multiple_of(4) {
        return Err(format!(
            "invalid TN transpose stride {stride} for {rows} rows"
        ));
    }
    for output_row in 0..columns {
        for output_column in 0..stride {
            let index = output_row * stride + output_column;
            let expected = if output_column < rows {
                input[output_column * columns + output_row]
            } else {
                0
            };
            if output[index] != expected {
                return Err(format!(
                    "TN raw transpose differs at ({output_row},{output_column}): actual={:#010x} expected={expected:#010x}",
                    output[index]
                ));
            }
        }
    }
    Ok(())
}

pub fn candidate_source(fixed_n96: &str) -> Result<String, String> {
    let mut source = fixed_n96.to_owned();
    replace_exactly_once(
        &mut source,
        "nn_sm89_rna_tf32_m128n96_bk32_s3",
        GEMM_SYMBOL,
        "N96 export",
    )?;
    replace_exactly_once(
        &mut source,
        concat!(
            "    float value = params.alpha == 1.0f\n",
            "        ? accumulator\n",
            "        : __fmul_rn(params.alpha, accumulator);\n",
            "    float* destination = output + (long long)row * params.ldc + column;\n",
            "    if (params.beta != 0.0f) value = __fmaf_rn(params.beta, *destination, value);\n",
            "    *destination = value;"
        ),
        concat!(
            "    float* destination = output + (long long)row * params.ldc + column;\n",
            "    *destination = __fmaf_rn(params.alpha, accumulator, *destination);"
        ),
        "scalar TN epilogue",
    )?;
    replace_exactly_once(
        &mut source,
        concat!(
            "        bool scale = params.alpha != 1.0f;\n",
            "        bool blend = params.beta != 0.0f;\n",
            "#pragma unroll 4\n",
            "        for (int linear = (int)threadIdx.x; linear < 128 * 24; linear += 256) {\n",
            "            int row = linear / 24;\n",
            "            int chunk = (linear % 24) * 4;\n",
            "            int global_row = tile_row + row;\n",
            "            if (global_row >= params.m) continue;\n",
            "            float4 value = *reinterpret_cast<const float4*>(\n",
            "                tile_output + row * 104 + chunk);\n",
            "            float* destination =\n",
            "                output + (long long)global_row * params.ldc + tile_column + chunk;\n",
            "            if (scale) {\n",
            "                value.x = __fmul_rn(params.alpha, value.x);\n",
            "                value.y = __fmul_rn(params.alpha, value.y);\n",
            "                value.z = __fmul_rn(params.alpha, value.z);\n",
            "                value.w = __fmul_rn(params.alpha, value.w);\n",
            "            }\n",
            "            if (blend) {\n",
            "                float4 old = *reinterpret_cast<const float4*>(destination);\n",
            "                value.x = __fmaf_rn(params.beta, old.x, value.x);\n",
            "                value.y = __fmaf_rn(params.beta, old.y, value.y);\n",
            "                value.z = __fmaf_rn(params.beta, old.z, value.z);\n",
            "                value.w = __fmaf_rn(params.beta, old.w, value.w);\n",
            "            }\n",
            "            *reinterpret_cast<float4*>(destination) = value;\n",
            "        }"
        ),
        concat!(
            "#pragma unroll 4\n",
            "        for (int linear = (int)threadIdx.x; linear < 128 * 24; linear += 256) {\n",
            "            int row = linear / 24;\n",
            "            int chunk = (linear % 24) * 4;\n",
            "            int global_row = tile_row + row;\n",
            "            if (global_row >= params.m) continue;\n",
            "            float4 value = *reinterpret_cast<const float4*>(\n",
            "                tile_output + row * 104 + chunk);\n",
            "            float* destination =\n",
            "                output + (long long)global_row * params.ldc + tile_column + chunk;\n",
            "            float4 old = *reinterpret_cast<const float4*>(destination);\n",
            "            value.x = __fmaf_rn(params.alpha, value.x, old.x);\n",
            "            value.y = __fmaf_rn(params.alpha, value.y, old.y);\n",
            "            value.z = __fmaf_rn(params.alpha, value.z, old.z);\n",
            "            value.w = __fmaf_rn(params.alpha, value.w, old.w);\n",
            "            *reinterpret_cast<float4*>(destination) = value;\n",
            "        }"
        ),
        "vector TN epilogue",
    )?;
    source.push_str("\n\n");
    source.push_str(TRANSPOSE_CUDA);
    Ok(source)
}

fn replace_exactly_once(
    source: &mut String,
    from: &str,
    to: &str,
    label: &str,
) -> Result<(), String> {
    let count = source.matches(from).count();
    if count != 1 {
        return Err(format!(
            "TN transpose N96 {label} boundary changed: expected 1, found {count}"
        ));
    }
    *source = source.replacen(from, to, 1);
    Ok(())
}

const TRANSPOSE_CUDA: &str = r#"struct GbfTf32TnTransposeParams {
    int rows;
    int columns;
    int output_stride;
};

static_assert(sizeof(GbfTf32TnTransposeParams) == 12, "TN transpose parameter ABI");
static_assert(alignof(GbfTf32TnTransposeParams) == 4, "TN transpose parameter alignment");

extern "C" __global__ __launch_bounds__(256, 1)
void tn_test_transpose_raw_u32_32x32(
    const unsigned* input, unsigned* output, GbfTf32TnTransposeParams params) {
    if (params.rows < 0 || params.columns < 0 || params.output_stride < params.rows
        || (params.output_stride & 3) != 0) return;
    __shared__ unsigned tile[32][33];
    int input_column = (int)blockIdx.x * 32 + (int)threadIdx.x;
    int input_row_base = (int)blockIdx.y * 32 + (int)threadIdx.y;
#pragma unroll
    for (int offset = 0; offset < 32; offset += 8) {
        int input_row = input_row_base + offset;
        tile[(int)threadIdx.y + offset][(int)threadIdx.x] =
            input_row < params.rows && input_column < params.columns
            ? input[(long long)input_row * params.columns + input_column]
            : 0U;
    }
    __syncthreads();
    int output_column = (int)blockIdx.y * 32 + (int)threadIdx.x;
    int output_row_base = (int)blockIdx.x * 32 + (int)threadIdx.y;
#pragma unroll
    for (int offset = 0; offset < 32; offset += 8) {
        int output_row = output_row_base + offset;
        if (output_row < params.columns && output_column < params.output_stride) {
            output[(long long)output_row * params.output_stride + output_column] =
                output_column < params.rows
                ? tile[(int)threadIdx.x][(int)threadIdx.y + offset]
                : 0U;
        }
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    const FIXED_N96: &str = include_str!("../../kernels/gemm_bi_inference/sm89/tf32_rna_n96.cu");

    #[test]
    fn transpose_stride_and_indices_cover_raw_words_and_padding() {
        assert_eq!(padded_stride(0), Ok(0));
        assert_eq!(padded_stride(129), Ok(132));
        assert_eq!(padded_stride(2_048), Ok(2_048));
        assert_eq!(transpose_destination_index(129, 65, 132, 0, 0), Ok(0));
        assert_eq!(
            transpose_destination_index(129, 65, 132, 128, 64),
            Ok(64 * 132 + 128)
        );
        assert!(transpose_destination_index(129, 65, 128, 0, 0).is_err());
        assert!(transpose_destination_index(129, 65, 132, 129, 0).is_err());
        assert!(transpose_destination_index(129, 65, 132, 0, 65).is_err());
    }

    #[test]
    fn raw_transpose_validation_rejects_wrong_words_and_poisoned_padding() {
        let input = vec![10, 11, 12, 20, 21, 22];
        let expected = vec![10, 20, 0, 0, 11, 21, 0, 0, 12, 22, 0, 0];
        assert!(validate_transposed_words(&input, 2, 3, 4, &expected).is_ok());

        let mut wrong_word = expected.clone();
        wrong_word[4] ^= 1;
        assert!(validate_transposed_words(&input, 2, 3, 4, &wrong_word).is_err());

        let mut poisoned_padding = expected;
        poisoned_padding[2] = 0xdead_beef;
        assert!(validate_transposed_words(&input, 2, 3, 4, &poisoned_padding).is_err());

        assert!(validate_transposed_words(&[], 0, 3, 0, &[]).is_ok());
    }

    #[test]
    fn candidate_changes_only_symbol_and_exact_tn_epilogue_and_adds_raw_transpose() {
        let source = candidate_source(FIXED_N96).unwrap();
        assert_eq!(source.matches(GEMM_SYMBOL).count(), 1);
        assert_eq!(source.matches(TRANSPOSE_SYMBOL).count(), 1);
        assert!(source.contains("cvt.rna.tf32.f32"));
        assert!(source.contains("__fmaf_rn(params.alpha, accumulator, *destination)"));
        assert_eq!(source.matches("__fmaf_rn(params.alpha, value.").count(), 4);
        assert!(source.contains("const unsigned* input"));
        assert!(source.contains("unsigned* output"));
        assert!(source.contains("output_stride"));
        assert!(!source.contains("nn_sm89_rna_tf32_m128n96_bk32_s3"));
    }

    #[test]
    fn raw_transpose_source_is_available_without_the_tf32_candidate() {
        let source = transpose_source();
        assert_eq!(source.matches(TRANSPOSE_SYMBOL).count(), 1);
        assert!(source.contains("const unsigned* input, unsigned* output"));
        assert!(source.contains("output_column < params.rows"));
    }

    #[test]
    fn candidate_adapter_rejects_changed_n96_boundaries() {
        let changed = FIXED_N96.replacen(
            "__fmaf_rn(params.beta, *destination, value)",
            "__fadd_rn(*destination, value)",
            1,
        );
        assert!(candidate_source(&changed).is_err());
        assert!(candidate_source("").is_err());
    }
}
