use std::collections::BTreeSet;

pub const SOURCE: &str = include_str!("../../../../kernels/gemm_bi_triad/sm89_tf32_joint.cu");

pub const SOURCE_SHA256: &str = "5fbb9fc1481052da3591196f73f8d372a3616e07657dc13da4f2a1d8a1e95bd3";

pub const TN_PRE_RNA_TRANSPOSE_SYMBOL: &str = "gemm_bi_tn_sm89_tf32_pre_rna_transpose_32x32_v1";
pub const TN_PRE_RNA_N96_SYMBOL: &str = "gemm_bi_tn_sm89_tf32_pre_rna_m128n96_bk32_s3_v1";
pub const TN_PRE_RNA_M64N64_SYMBOL: &str = "gemm_bi_tn_sm89_tf32_pre_rna_m64n64_bk32_s3_v1";
pub const NN_ADD_HALF_DIRECT_N96_SYMBOL: &str =
    "gemm_bi_nn_sm89_tf32_addhalf_m128n96_bk32_s3_direct_v1";

pub const SM89_TF32_JOINT_SYMBOLS: [&str; 4] = [
    NN_ADD_HALF_DIRECT_N96_SYMBOL,
    TN_PRE_RNA_N96_SYMBOL,
    TN_PRE_RNA_M64N64_SYMBOL,
    TN_PRE_RNA_TRANSPOSE_SYMBOL,
];

pub fn export_inventory(source: &str) -> Result<Vec<&str>, String> {
    let mut exports = Vec::new();
    let mut remaining = source;
    while let Some(extern_at) = remaining.find("extern \"C\"") {
        let after_extern = &remaining[extern_at + "extern \"C\"".len()..];
        let void_at = after_extern.find("void ").ok_or_else(|| {
            "SM89 TF32 joint extern-C declaration is not a void export".to_owned()
        })?;
        let declaration = &after_extern[void_at + "void ".len()..];
        let name_end = declaration
            .find('(')
            .ok_or_else(|| "SM89 TF32 joint export has no parameter list".to_owned())?;
        let name = declaration[..name_end].trim();
        if name.is_empty()
            || !name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        {
            return Err(format!("invalid SM89 TF32 joint export name {name:?}"));
        }
        exports.push(name);
        remaining = declaration;
    }
    exports.sort_unstable();
    Ok(exports)
}

pub fn validate_source_text(source: &str) -> Result<(), String> {
    for marker in ["_test_", "_exp_"] {
        if source.contains(marker) {
            return Err(format!(
                "SM89 TF32 joint source retained discovery marker {marker}"
            ));
        }
    }
    let exports = export_inventory(source)?;
    let expected = SM89_TF32_JOINT_SYMBOLS.into_iter().collect::<BTreeSet<_>>();
    let observed = exports.iter().copied().collect::<BTreeSet<_>>();
    if exports.len() != expected.len() || observed != expected {
        return Err(format!(
            "SM89 TF32 joint export inventory changed: expected {expected:?}, observed {exports:?}"
        ));
    }
    Ok(())
}

pub fn validate_source() -> Result<(), String> {
    if SOURCE_SHA256.len() != 64
        || !SOURCE_SHA256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err("SM89 TF32 joint SHA-256 freeze is not lowercase hex".into());
    }
    validate_source_text(SOURCE)
}
