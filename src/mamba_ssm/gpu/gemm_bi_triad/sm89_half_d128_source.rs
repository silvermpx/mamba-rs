//! Owner of the Ada half TN d128 fragment: the two 32x16 four-stage tiles
//! that serve the d_model-128 classifier's weight gradients in bf16 and f16.

use crate::mamba_ssm::gpu::kernel_identity::{FramedSha256, digest_hex};

pub const OWNER: &str = include_str!("../../../../kernels/gemm_bi_triad/sm89/half_d128.cu");
pub const OWNER_SHA256: &str = "feacd35ee8afe0008a6837dd3b356bcd97ba365366ee56b094406b04b3fa349a";

pub const D128_IN_BF16_SYMBOL: &str = "tn_sm89_half_d128_in_m32n16_bk64_s4_cg_bf16";
pub const D128_IN_F16_SYMBOL: &str = "tn_sm89_half_d128_in_m32n16_bk64_s4_cg_f16";
pub const D128_OUT_BF16_SYMBOL: &str = "tn_sm89_half_d128_out_m32n16_bk64_s4_cg_bf16";
pub const D128_OUT_F16_SYMBOL: &str = "tn_sm89_half_d128_out_m32n16_bk64_s4_cg_f16";
pub const SYMBOLS: [&str; 4] = [
    D128_IN_BF16_SYMBOL,
    D128_IN_F16_SYMBOL,
    D128_OUT_BF16_SYMBOL,
    D128_OUT_F16_SYMBOL,
];

const EXPORT_MACRO: &str = "TN_HALF_D128_EXPORT(";

/// The exports the fragment instantiates, each exactly once through the
/// export macro, which is the fragment's only extern "C" declaration.
pub fn export_inventory(source: &str) -> Result<Vec<&'static str>, String> {
    for symbol in SYMBOLS {
        if source.matches(&format!("{EXPORT_MACRO}{symbol},")).count() != 1 {
            return Err(format!(
                "SM89 half d128 fragment does not export {symbol} exactly once"
            ));
        }
    }
    if source.matches("extern \"C\" __global__").count() != 1
        || source.matches(EXPORT_MACRO).count() != 1 + SYMBOLS.len()
    {
        return Err("SM89 half d128 fragment contains a foreign or missing export".into());
    }
    Ok(SYMBOLS.to_vec())
}

pub fn validate_source_text(source: &str) -> Result<(), String> {
    for marker in ["_test_", "_exp_", "probe_", "PROBE_", "widen_exhaustive"] {
        if source.contains(marker) {
            return Err(format!(
                "SM89 half d128 fragment retained forbidden marker {marker}"
            ));
        }
    }
    export_inventory(source).map(drop)
}

pub fn validate_source() -> Result<(), String> {
    if digest_hex(&FramedSha256::bytes(OWNER.as_bytes())) != OWNER_SHA256 {
        return Err("SM89 half d128 CUDA owner SHA-256 changed".into());
    }
    validate_source_text(OWNER)
}

pub fn fragment() -> Result<&'static str, String> {
    validate_source()?;
    Ok(OWNER)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owner_is_sealed_and_exports_the_four_d128_tiles() {
        validate_source().unwrap();
        assert_eq!(export_inventory(OWNER).unwrap(), SYMBOLS);
        assert!(OWNER.contains("Kernel<T, KOUT, N, TM, TN, BK, STAGES, CG>::run("));
        for shape in [
            "128, 512, 32, 16, 64, 4, true, 3)",
            "256, 128, 32, 16, 64, 4, true, 3)",
        ] {
            assert_eq!(OWNER.matches(shape).count(), 2, "{shape}");
        }
    }

    #[test]
    fn validator_rejects_duplicate_missing_and_foreign_exports() {
        let duplicate = format!(
            "{OWNER}\nTN_HALF_D128_EXPORT({D128_IN_BF16_SYMBOL}, __nv_bfloat16, 128, 512, 32, 16, 64, 4, true, 3)\n"
        );
        assert!(validate_source_text(&duplicate).is_err());
        let missing = OWNER.replacen(&format!("{EXPORT_MACRO}{D128_OUT_F16_SYMBOL},"), "// ", 1);
        assert!(validate_source_text(&missing).is_err());
        let foreign = format!(
            "{OWNER}\nextern \"C\" __global__ void tn_sm89_half_d128_probe_bf16(float* output) {{ output[0] = 0.0f; }}\n"
        );
        assert!(validate_source_text(&foreign).is_err());
    }
}
