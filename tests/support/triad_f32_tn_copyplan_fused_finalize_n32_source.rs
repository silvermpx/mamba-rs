#[path = "triad_f32_tn_copyplan_fused_finalize_source.rs"]
mod fused_n64;
#[path = "triad_f32_tn_copyplan_raw_store_source.rs"]
mod raw_n64;

pub const N64_FUSED_SYMBOL: &str = fused_n64::FUSED_SYMBOL;
pub const N64_RAW_SYMBOL: &str = raw_n64::RAW_SYMBOL;
pub const N32_FUSED_SYMBOL: &str = "gemm_bi_tn_test_fixed_sm89_f32_n32_copyplan_fused_finalize_v1";
pub const N32_RAW_SYMBOL: &str = "gemm_bi_tn_test_fixed_sm89_f32_n32_copyplan_raw_v1";

fn replace_all_checked(source: &mut String, before: &str, after: &str) -> Result<(), String> {
    let count = source.matches(before).count();
    if count == 0 || source.contains(after) {
        return Err(format!(
            "N32 transform anchor {before:?}: observed{count}, destination_present={}",
            source.contains(after)
        ));
    }
    *source = source.replace(before, after);
    Ok(())
}

fn transform_to_n32(
    mut source: String,
    parent_symbol: &str,
    candidate_symbol: &str,
) -> Result<String, String> {
    replace_all_checked(&mut source, parent_symbol, candidate_symbol)?;
    replace_all_checked(&mut source, "N64", "N32")?;
    replace_all_checked(
        &mut source,
        "#define SM89_EXACT_N32_CP_BN 64",
        "#define SM89_EXACT_N32_CP_BN 32",
    )?;
    replace_all_checked(
        &mut source,
        "SM89_EXACT_N32_CP_BN == 64",
        "SM89_EXACT_N32_CP_BN == 32",
    )?;
    replace_all_checked(
        &mut source,
        "unsigned _stage = (unsigned)(BUF) * (SM89_EXACT_N32_CP_BM * SM89_EXACT_N32_CP_BK * 4);",
        "unsigned _a_stage = (unsigned)(BUF) * (SM89_EXACT_N32_CP_BM * SM89_EXACT_N32_CP_BK * 4); \\\n        unsigned _b_stage = (unsigned)(BUF) * (SM89_EXACT_N32_CP_BK * SM89_EXACT_N32_CP_BN * 4);",
    )?;
    replace_all_checked(
        &mut source,
        "a_destination + _stage",
        "a_destination + _a_stage",
    )?;
    replace_all_checked(
        &mut source,
        "b_destination + _stage",
        "b_destination + _b_stage",
    )?;
    replace_all_checked(
        &mut source,
        "for (int _i = 0; _i < 4; ++_i) {                                    \\\n            unsigned _dst = b_destination",
        "for (int _i = 0; _i < 2; ++_i) {                                    \\\n            unsigned _dst = b_destination",
    )?;
    replace_all_checked(
        &mut source,
        "const int copy_b_row = tid / (SM89_EXACT_N32_CP_BN / 4);",
        "const int copy_b_row = tid / 8;",
    )?;
    replace_all_checked(
        &mut source,
        "const int copy_b_col = col0 + (tid % (SM89_EXACT_N32_CP_BN / 4)) * 4;",
        "const int copy_b_col = col0 + (tid % 8) * 4;",
    )?;
    replace_all_checked(
        &mut source,
        "b_vector_stride = (unsigned long long)ldb * 32ULL",
        "b_vector_stride = (unsigned long long)ldb * 64ULL",
    )?;
    replace_all_checked(
        &mut source,
        "b_slab_stride = b_vector_stride * 4ULL",
        "b_slab_stride = b_vector_stride * 2ULL",
    )?;
    replace_all_checked(
        &mut source,
        "int col_base = col0 + tx * 4;",
        "int col_base = col0 + tx * 2;",
    )?;
    replace_all_checked(&mut source, "float acc[8][4];", "float acc[8][2];")?;
    replace_all_checked(&mut source, "float b_reg[4];", "float b_reg[2];")?;
    replace_all_checked(&mut source, "tx * 4 + j", "tx * 2 + j")?;
    replace_all_checked(&mut source, "j < 4", "j < 2")?;
    Ok(source)
}

fn reverse_to_n64(
    mut source: String,
    candidate_symbol: &str,
    parent_symbol: &str,
) -> Result<String, String> {
    replace_all_checked(&mut source, "j < 2", "j < 4")?;
    replace_all_checked(&mut source, "tx * 2 + j", "tx * 4 + j")?;
    replace_all_checked(&mut source, "float b_reg[2];", "float b_reg[4];")?;
    replace_all_checked(&mut source, "float acc[8][2];", "float acc[8][4];")?;
    replace_all_checked(
        &mut source,
        "int col_base = col0 + tx * 2;",
        "int col_base = col0 + tx * 4;",
    )?;
    replace_all_checked(
        &mut source,
        "b_slab_stride = b_vector_stride * 2ULL",
        "b_slab_stride = b_vector_stride * 4ULL",
    )?;
    replace_all_checked(
        &mut source,
        "b_vector_stride = (unsigned long long)ldb * 64ULL",
        "b_vector_stride = (unsigned long long)ldb * 32ULL",
    )?;
    replace_all_checked(
        &mut source,
        "const int copy_b_col = col0 + (tid % 8) * 4;",
        "const int copy_b_col = col0 + (tid % (SM89_EXACT_N32_CP_BN / 4)) * 4;",
    )?;
    replace_all_checked(
        &mut source,
        "const int copy_b_row = tid / 8;",
        "const int copy_b_row = tid / (SM89_EXACT_N32_CP_BN / 4);",
    )?;
    replace_all_checked(
        &mut source,
        "for (int _i = 0; _i < 2; ++_i) {                                    \\\n            unsigned _dst = b_destination",
        "for (int _i = 0; _i < 4; ++_i) {                                    \\\n            unsigned _dst = b_destination",
    )?;
    replace_all_checked(
        &mut source,
        "b_destination + _b_stage",
        "b_destination + _stage",
    )?;
    replace_all_checked(
        &mut source,
        "a_destination + _a_stage",
        "a_destination + _stage",
    )?;
    replace_all_checked(
        &mut source,
        "unsigned _a_stage = (unsigned)(BUF) * (SM89_EXACT_N32_CP_BM * SM89_EXACT_N32_CP_BK * 4); \\\n        unsigned _b_stage = (unsigned)(BUF) * (SM89_EXACT_N32_CP_BK * SM89_EXACT_N32_CP_BN * 4);",
        "unsigned _stage = (unsigned)(BUF) * (SM89_EXACT_N32_CP_BM * SM89_EXACT_N32_CP_BK * 4);",
    )?;
    replace_all_checked(
        &mut source,
        "SM89_EXACT_N32_CP_BN == 32",
        "SM89_EXACT_N32_CP_BN == 64",
    )?;
    replace_all_checked(
        &mut source,
        "#define SM89_EXACT_N32_CP_BN 32",
        "#define SM89_EXACT_N32_CP_BN 64",
    )?;
    replace_all_checked(&mut source, "N32", "N64")?;
    replace_all_checked(&mut source, candidate_symbol, parent_symbol)?;
    Ok(source)
}

pub fn compose_fused_source() -> Result<String, String> {
    transform_to_n32(
        fused_n64::compose_source()?,
        N64_FUSED_SYMBOL,
        N32_FUSED_SYMBOL,
    )
}

pub fn compose_raw_source() -> Result<String, String> {
    transform_to_n32(raw_n64::compose_source()?, N64_RAW_SYMBOL, N32_RAW_SYMBOL)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_n32_geometry(source: &str, symbol: &str) {
        assert_eq!(source.matches(symbol).count(), 1);
        assert!(source.contains("#define SM89_EXACT_N32_CP_BM 64"));
        assert!(source.contains("#define SM89_EXACT_N32_CP_BN 32"));
        assert!(source.contains("#define SM89_EXACT_N32_CP_BK 32"));
        assert!(source.contains("#define SM89_EXACT_N32_CP_GROUP_M 8"));
        assert!(source.contains("SM89_EXACT_N32_CP_BN == 32"));
        assert!(source.contains("float acc[8][2];"));
        assert_eq!(source.matches("float b_reg[2];").count(), 2);
        assert!(source.contains("int col_base = col0 + tx * 2;"));
    }

    fn assert_planned_producer(source: &str) {
        assert!(source.contains("const int copy_b_row = tid / 8;"));
        assert!(source.contains("const int copy_b_col = col0 + (tid % 8) * 4;"));
        assert!(source.contains("_i < 2; ++_i"));
        assert!(source.contains("_i * b_vector_stride"));
        assert!(source.contains("ldb * 64ULL"));
        assert!(source.contains("b_vector_stride * 2ULL"));
        assert!(source.contains("SM89_EXACT_N32_CP_BM * SM89_EXACT_N32_CP_BK * 4"));
        assert!(source.contains("SM89_EXACT_N32_CP_BK * SM89_EXACT_N32_CP_BN * 4"));
        assert!(source.contains("SM89_EXACT_N32_CP_BK * 4); \\\n        unsigned _b_stage"));
    }

    #[test]
    fn n32_raw_has_exact_8x2_ownership_and_24k_staging() {
        let source = compose_raw_source().unwrap();
        assert_n32_geometry(&source, N32_RAW_SYMBOL);
        assert_planned_producer(&source);
        assert_eq!(source.matches("float val = acc[i][j];").count(), 1);
        assert!(!source.contains("__fmul_rn(alpha, acc[i][j])"));
    }

    #[test]
    fn n32_fused_preserves_exact_fma_and_finalize_arithmetic() {
        let source = compose_fused_source().unwrap();
        assert_n32_geometry(&source, N32_FUSED_SYMBOL);
        assert_planned_producer(&source);
        assert_eq!(
            source
                .matches("__fmaf_rn(a_reg[i], b_reg[j], acc[i][j])")
                .count(),
            2
        );
        for required in [
            "__dadd_rn(",
            "__dmul_rn((double)alpha, sum)",
            "__double2float_rn(",
            "dW[idx] = __fadd_rn(dW[idx], update)",
        ] {
            assert!(source.contains(required), "missing {required}");
        }
    }

    #[test]
    fn reverse_transform_is_byte_identical_to_both_n64_parents() {
        assert_eq!(
            reverse_to_n64(
                compose_raw_source().unwrap(),
                N32_RAW_SYMBOL,
                N64_RAW_SYMBOL
            )
            .unwrap(),
            raw_n64::compose_source().unwrap()
        );
        assert_eq!(
            reverse_to_n64(
                compose_fused_source().unwrap(),
                N32_FUSED_SYMBOL,
                N64_FUSED_SYMBOL
            )
            .unwrap(),
            fused_n64::compose_source().unwrap()
        );
    }
}
