//! Owner of the Ada small-tile half fragment: the 16x64 four-stage NN and
//! NT tiles that serve the d_model-128 classifier's forward and input
//! gradient in bf16 and f16.

use crate::mamba_ssm::gpu::kernel_identity::{FramedSha256, digest_hex};

pub const OWNER: &str = include_str!("../../../../kernels/gemm_bi_triad/sm89/half_small.cu");
pub const OWNER_SHA256: &str = "50ec6b32048412ab16e0500593a811268af3128be6e9a500e1d01a8e7dd2168f";

pub const NT_BF16_SYMBOL: &str = "nt_sm89_m16n64_bk64_s4_bf16";
pub const NT_F16_SYMBOL: &str = "nt_sm89_m16n64_bk64_s4_f16";
pub const NN_BF16_SYMBOL: &str = "nn_sm89_m16n64_bk64_s4_bf16";
pub const NN_F16_SYMBOL: &str = "nn_sm89_m16n64_bk64_s4_f16";
pub const SYMBOLS: [&str; 4] = [NT_BF16_SYMBOL, NT_F16_SYMBOL, NN_BF16_SYMBOL, NN_F16_SYMBOL];

const NT_EXPORT_MACRO: &str = "SMALL_NT_EXPORT(";
const NN_EXPORT_MACRO: &str = "SMALL_NN_EXPORT(";

/// The exports the fragment instantiates, each exactly once through its
/// op's export macro; the two macros are the fragment's only extern "C"
/// declarations.
pub fn export_inventory(source: &str) -> Result<Vec<&'static str>, String> {
    for (symbol, macro_name) in [
        (NT_BF16_SYMBOL, NT_EXPORT_MACRO),
        (NT_F16_SYMBOL, NT_EXPORT_MACRO),
        (NN_BF16_SYMBOL, NN_EXPORT_MACRO),
        (NN_F16_SYMBOL, NN_EXPORT_MACRO),
    ] {
        if source.matches(&format!("{macro_name}{symbol},")).count() != 1 {
            return Err(format!(
                "SM89 half small fragment does not export {symbol} exactly once"
            ));
        }
    }
    if source.matches("extern \"C\" __global__").count() != 2
        || source.matches(NT_EXPORT_MACRO).count() != 3
        || source.matches(NN_EXPORT_MACRO).count() != 3
    {
        return Err("SM89 half small fragment contains a foreign or missing export".into());
    }
    Ok(SYMBOLS.to_vec())
}

pub fn validate_source_text(source: &str) -> Result<(), String> {
    for marker in ["_test_", "_exp_", "probe_", "PROBE_"] {
        if source.contains(marker) {
            return Err(format!(
                "SM89 half small fragment retained forbidden marker {marker}"
            ));
        }
    }
    export_inventory(source).map(drop)
}

pub fn validate_source() -> Result<(), String> {
    if digest_hex(&FramedSha256::bytes(OWNER.as_bytes())) != OWNER_SHA256 {
        return Err("SM89 half small CUDA owner SHA-256 changed".into());
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
    fn owner_is_sealed_and_exports_the_four_small_tiles() {
        validate_source().unwrap();
        assert_eq!(export_inventory(OWNER).unwrap(), SYMBOLS);
        assert!(OWNER.contains("Sm89HalfNnS3Params params"));
        assert!(OWNER.contains("float alpha, int M, int N, int K_out"));
        assert_eq!(OWNER.matches("16, 64, 1, 4, 4)").count(), 4);
    }

    #[test]
    fn validator_rejects_duplicate_missing_and_foreign_exports() {
        let duplicate =
            format!("{OWNER}\nSMALL_NT_EXPORT({NT_BF16_SYMBOL}, __nv_bfloat16, 16, 64, 1, 4, 4)\n");
        assert!(validate_source_text(&duplicate).is_err());
        let missing = OWNER.replacen(&format!("{NN_EXPORT_MACRO}{NN_F16_SYMBOL},"), "// ", 1);
        assert!(validate_source_text(&missing).is_err());
        let foreign = format!(
            "{OWNER}\nextern \"C\" __global__ void nt_sm89_m16n64_probe_f16(float* output) {{ output[0] = 0.0f; }}\n"
        );
        assert!(validate_source_text(&foreign).is_err());
    }
}
