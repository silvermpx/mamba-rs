pub const CANDIDATE_GEMM_SYMBOL: &str =
    "gemm_bi_tn_test_transpose_pre_rna_n96_sm89_m128n96_bk32_s3";
pub const RETAINED_GEMM_SYMBOL: &str = "gemm_bi_tn_test_transpose_rna_n96_sm89_m128n96_bk32_s3";
pub const CANDIDATE_TRANSPOSE_SYMBOL: &str = "gemm_bi_tn_test_transpose_rna_u32_32x32_v1";
pub const RETAINED_TRANSPOSE_SYMBOL: &str = "gemm_bi_tn_test_transpose_raw_u32_32x32_v1";

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
        .ok_or("TN RNA transpose input extent overflows usize")?;
    let output_len = columns
        .checked_mul(stride)
        .ok_or("TN RNA transpose output extent overflows usize")?;
    if input.len() != input_len || output.len() != output_len {
        return Err(format!(
            "TN RNA transpose extent changed: input={}/{} output={}/{}",
            input.len(),
            input_len,
            output.len(),
            output_len
        ));
    }
    if stride < rows || !stride.is_multiple_of(4) {
        return Err(format!(
            "invalid TN RNA transpose stride {stride} for {rows} rows"
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
                    "TN RNA transpose differs at ({output_row},{output_column}): actual={:#010x} expected={expected:#010x}",
                    output[index]
                ));
            }
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
        RETAINED_TRANSPOSE_SYMBOL,
        CANDIDATE_TRANSPOSE_SYMBOL,
        "transpose export",
    )?;
    replace_exact(
        &mut source,
        A_ROUND_BLOCK,
        A_PRE_ROUNDED_BLOCK,
        "A fragment RNA block",
    )?;
    replace_exact(
        &mut source,
        RAW_TRANSPOSE_VALUE,
        RNA_TRANSPOSE_VALUE,
        "RNA transpose store",
    )?;
    Ok(source)
}

pub fn restore_retained_source(candidate_source: &str) -> Result<String, String> {
    let mut source = candidate_source.to_owned();
    replace_exact(
        &mut source,
        RNA_TRANSPOSE_VALUE,
        RAW_TRANSPOSE_VALUE,
        "restored raw transpose store",
    )?;
    replace_exact(
        &mut source,
        A_PRE_ROUNDED_BLOCK,
        A_ROUND_BLOCK,
        "restored A fragment RNA block",
    )?;
    replace_exact(
        &mut source,
        CANDIDATE_TRANSPOSE_SYMBOL,
        RETAINED_TRANSPOSE_SYMBOL,
        "restored transpose export",
    )?;
    replace_exact(
        &mut source,
        CANDIDATE_GEMM_SYMBOL,
        RETAINED_GEMM_SYMBOL,
        "restored GEMM export",
    )?;
    Ok(source)
}

const A_ROUND_BLOCK: &str = r#"        fragments.a[m_atom][0] = tf32n96_round(raw0);
        fragments.a[m_atom][1] = tf32n96_round(raw1);
        fragments.a[m_atom][2] = tf32n96_round(raw2);
        fragments.a[m_atom][3] = tf32n96_round(raw3);"#;

const A_PRE_ROUNDED_BLOCK: &str = r#"        fragments.a[m_atom][0] = raw0;
        fragments.a[m_atom][1] = raw1;
        fragments.a[m_atom][2] = raw2;
        fragments.a[m_atom][3] = raw3;"#;

const RAW_TRANSPOSE_VALUE: &str = r#"                ? tile[(int)threadIdx.x][(int)threadIdx.y + offset]
                : 0U;"#;

const RNA_TRANSPOSE_VALUE: &str = r#"                ? tf32n96_round(tile[(int)threadIdx.x][(int)threadIdx.y + offset])
                : 0U;"#;

fn replace_exact(source: &mut String, old: &str, new: &str, label: &str) -> Result<(), String> {
    let count = source.matches(old).count();
    if count != 1 {
        return Err(format!(
            "TN pre-RNA transpose N96 {label} seam changed: expected 1, observed {count}"
        ));
    }
    *source = source.replacen(old, new, 1);
    Ok(())
}
