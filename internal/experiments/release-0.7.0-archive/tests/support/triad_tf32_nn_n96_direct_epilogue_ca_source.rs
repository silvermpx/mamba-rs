pub const SYMBOL: &str =
    "gemm_bi_nn_triad_sm89_add_half_tf32_direct_epilogue_ca_exp_m128n96_bk32_s3";
pub const DIRECT_RETAINED_SYMBOL: &str =
    "gemm_bi_nn_triad_sm89_add_half_tf32_direct_epilogue_exp_m128n96_bk32_s3";
const EXPECTED_DIRECT_RETAINED_FNV64: u64 = 0x8144_785a_a923_8460;

const RETAINED_COPY: &str = "gbf_tf32_copy_cg(";
const CANDIDATE_COPY: &str = "gbf_tf32_copy_ca(";

pub fn compose_candidate_source(direct_retained_source: &str) -> Result<String, String> {
    let observed = fnv64(direct_retained_source.as_bytes());
    if observed != EXPECTED_DIRECT_RETAINED_FNV64 {
        return Err(format!(
            "direct-retained N96 source changed: expected {EXPECTED_DIRECT_RETAINED_FNV64:#018x}, observed {observed:#018x}"
        ));
    }
    let count = direct_retained_source.matches(RETAINED_COPY).count();
    if count != 2 {
        return Err(format!(
            "direct-retained N96 copy seam changed: expected 2, observed {count}"
        ));
    }
    let source = direct_retained_source.replace(RETAINED_COPY, CANDIDATE_COPY);
    replace_exact(&source, DIRECT_RETAINED_SYMBOL, SYMBOL, "export symbol")
}

pub fn restore_direct_retained_source(candidate: &str) -> Result<String, String> {
    let count = candidate.matches(CANDIDATE_COPY).count();
    if count != 2 {
        return Err(format!(
            "candidate N96 copy seam changed: expected 2, observed {count}"
        ));
    }
    let source = candidate.replace(CANDIDATE_COPY, RETAINED_COPY);
    replace_exact(
        &source,
        SYMBOL,
        DIRECT_RETAINED_SYMBOL,
        "restored export symbol",
    )
}

fn replace_exact(source: &str, old: &str, new: &str, label: &str) -> Result<String, String> {
    let count = source.matches(old).count();
    if count != 1 {
        return Err(format!(
            "N96 direct-epilogue CA {label} seam changed: expected 1, observed {count}"
        ));
    }
    Ok(source.replacen(old, new, 1))
}

fn fnv64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapter_changes_both_n96_stage_copies_and_only_the_export() {
        let direct = format!(
            "prefix {DIRECT_RETAINED_SYMBOL} middle {RETAINED_COPY} a tail {RETAINED_COPY} b"
        );
        assert_ne!(fnv64(direct.as_bytes()), EXPECTED_DIRECT_RETAINED_FNV64);
        assert!(compose_candidate_source(&direct).is_err());
    }
}
