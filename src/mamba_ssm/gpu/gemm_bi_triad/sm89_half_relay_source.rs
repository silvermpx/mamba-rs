//! Owner of the Ada half TN relay fragment: the 64x64 three-stage tile whose
//! CTAs walk a persistent range of (tile, slab) units and hand a tile's
//! accumulator chain to the next CTA untouched, so a tile count that is not a
//! multiple of the resident CTA count no longer leaves a tail wave.

use crate::mamba_ssm::gpu::kernel_identity::{FramedSha256, digest_hex};

pub const OWNER: &str = include_str!("../../../../kernels/gemm_bi_triad/sm89/half_relay.cu");
pub const OWNER_SHA256: &str = "8bfbc26ed5e65e373d7c88264f57c264882be123a08f6ff0a9d1ccb482025c9f";

pub const RELAY_BF16_SYMBOL: &str = "tn_sm89_relay_m64n64_bk64_s3_bf16";
pub const RELAY_F16_SYMBOL: &str = "tn_sm89_relay_m64n64_bk64_s3_f16";
pub const SYMBOLS: [&str; 2] = [RELAY_BF16_SYMBOL, RELAY_F16_SYMBOL];

/// Floats one CTA publishes when a tile's chain crosses its upper boundary:
/// 128 threads by the 32 accumulators of the 64x64 tile. The launcher sizes
/// the hand-off workspace from this.
pub const SLAB_FLOATS: usize = 128 * 32;

const EXPORT_MACRO: &str = "HALF_RELAY_EXPORT(";

/// The exports the fragment instantiates, each exactly once through the
/// export macro, which is the fragment's only extern "C" declaration.
pub fn export_inventory(source: &str) -> Result<Vec<&'static str>, String> {
    for symbol in SYMBOLS {
        if source.matches(&format!("{EXPORT_MACRO}{symbol},")).count() != 1 {
            return Err(format!(
                "SM89 half relay fragment does not export {symbol} exactly once"
            ));
        }
    }
    if source.matches("extern \"C\" __global__").count() != 1
        || source.matches(EXPORT_MACRO).count() != 1 + SYMBOLS.len()
    {
        return Err("SM89 half relay fragment contains a foreign or missing export".into());
    }
    Ok(SYMBOLS.to_vec())
}

pub fn validate_source_text(source: &str) -> Result<(), String> {
    for marker in ["_test_", "_exp_", "probe_", "PROBE_", "widen_exhaustive"] {
        if source.contains(marker) {
            return Err(format!(
                "SM89 half relay fragment retained forbidden marker {marker}"
            ));
        }
    }
    export_inventory(source).map(drop)
}

pub fn validate_source() -> Result<(), String> {
    if digest_hex(&FramedSha256::bytes(OWNER.as_bytes())) != OWNER_SHA256 {
        return Err("SM89 half relay CUDA owner SHA-256 changed".into());
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
    fn owner_is_sealed_and_exports_the_two_relay_tiles() {
        validate_source().unwrap();
        assert_eq!(export_inventory(OWNER).unwrap(), SYMBOLS);
        assert_eq!(OWNER.matches("RelayKernel<TYPE, STAGES>::run(").count(), 1);
        for shape in [
            "HALF_RELAY_EXPORT(tn_sm89_relay_m64n64_bk64_s3_bf16, __nv_bfloat16, 3)",
            "HALF_RELAY_EXPORT(tn_sm89_relay_m64n64_bk64_s3_f16, __half, 3)",
        ] {
            assert_eq!(OWNER.matches(shape).count(), 1, "{shape}");
        }
    }

    #[test]
    fn the_hand_off_keeps_the_tiled_reduction_order() {
        // A relayed chain is continued, never folded: the later CTA seeds its
        // accumulators from the earlier CTA's words and adds nothing to them,
        // which is why the kernel reproduces the tiled route bit for bit.
        assert!(OWNER.contains("st.global.cg.v4.f32"));
        assert!(OWNER.contains("st.release.gpu.global.u32"));
        assert!(OWNER.contains("ld.acquire.gpu.global.u32"));
        assert!(!OWNER.contains("__fadd_rn"));
        assert!(!OWNER.contains("atomicAdd"));
    }

    #[test]
    fn validator_rejects_duplicate_missing_and_foreign_exports() {
        let duplicate =
            format!("{OWNER}\nHALF_RELAY_EXPORT({RELAY_BF16_SYMBOL}, __nv_bfloat16, 3)\n");
        assert!(validate_source_text(&duplicate).is_err());
        let missing = OWNER.replacen(&format!("{EXPORT_MACRO}{RELAY_F16_SYMBOL},"), "// ", 1);
        assert!(validate_source_text(&missing).is_err());
        let foreign = format!(
            "{OWNER}\nextern \"C\" __global__ void tn_sm89_relay_probe_bf16(float* output) {{ output[0] = 0.0f; }}\n"
        );
        assert!(validate_source_text(&foreign).is_err());
    }
}
