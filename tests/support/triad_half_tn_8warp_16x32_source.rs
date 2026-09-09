#[path = "triad_half_tn_vec2_epilogue_source.rs"]
mod retained;

pub const SYMBOL_PREFIX: &str = "gemm_bi_tn_test_tc64_bk64_s2_regpipe_vec2_8warp16x32_";
pub const RETAINED_SYMBOL_PREFIX: &str = retained::SYMBOL_PREFIX;
pub const TARGET: (usize, usize, usize) = (2_048, 768, 3_072);
pub const TARGET_GRID: u32 = 576;
pub const BLOCK_THREADS: u32 = 256;
pub const STATIC_SHARED_BYTES: usize = 32_768;
pub const REGISTER_CAP: usize = 104;
pub const REQUIRED_OCCUPANCY: usize = 2;
pub const EXPECTED_HMMA: usize = 16;
pub const EXPECTED_LDSM: usize = 20;

pub fn candidate_source(production: &str) -> Result<String, String> {
    let mut source = retained_source(production)?;
    replace_exact(
        &mut source,
        "#define GEMM_BI_TC64_THREADS 128",
        "#define GEMM_BI_TC64_THREADS 256",
        1,
    )?;
    replace_exact(
        &mut source,
        "int warpM = (warp / 2) * 32;",
        "int warpM = (warp / 2) * 16;",
        1,
    )?;
    replace_exact(&mut source, "float acc[2][4][4];", "float acc[1][4][4];", 1)?;
    replace_exact(
        &mut source,
        "unsigned a_frag[2][2][4];",
        "unsigned a_frag[2][1][4];",
        1,
    )?;
    replace_exact(
        &mut source,
        "for (int fm = 0; fm < 2; fm++)",
        "for (int fm = 0; fm < 1; fm++)",
        5,
    )?;
    replace_exact(&mut source, RETAINED_SYMBOL_PREFIX, SYMBOL_PREFIX, 1)?;
    Ok(source)
}

pub fn retained_source(production: &str) -> Result<String, String> {
    retained::candidate_source(production)
}

pub fn restore_retained_source(candidate: &str) -> Result<String, String> {
    let mut source = candidate.to_owned();
    replace_exact(
        &mut source,
        "#define GEMM_BI_TC64_THREADS 256",
        "#define GEMM_BI_TC64_THREADS 128",
        1,
    )?;
    replace_exact(
        &mut source,
        "int warpM = (warp / 2) * 16;",
        "int warpM = (warp / 2) * 32;",
        1,
    )?;
    replace_exact(&mut source, "float acc[1][4][4];", "float acc[2][4][4];", 1)?;
    replace_exact(
        &mut source,
        "unsigned a_frag[2][1][4];",
        "unsigned a_frag[2][2][4];",
        1,
    )?;
    replace_exact(
        &mut source,
        "for (int fm = 0; fm < 1; fm++)",
        "for (int fm = 0; fm < 2; fm++)",
        5,
    )?;
    replace_exact(&mut source, SYMBOL_PREFIX, RETAINED_SYMBOL_PREFIX, 1)?;
    Ok(source)
}

pub const fn warp_output_origin(warp: usize) -> (usize, usize) {
    ((warp / 2) * 16, (warp % 2) * 32)
}

pub fn all_strata_below(strata: &[[f64; 2]], threshold: f64) -> bool {
    threshold.is_finite()
        && threshold > 0.0
        && strata.len() == 4
        && strata.iter().all(|quantiles| {
            quantiles
                .iter()
                .all(|&ratio| ratio.is_finite() && ratio > 0.0 && ratio < threshold)
        })
}

fn replace_exact(
    source: &mut String,
    before: &str,
    after: &str,
    expected: usize,
) -> Result<(), String> {
    let actual = source.matches(before).count();
    if actual != expected {
        return Err(format!(
            "eight-warp TN anchor {before:?}: expected {expected}, observed {actual}"
        ));
    }
    *source = source.replace(before, after);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const PRODUCTION: &str = include_str!("../../kernels/gemm_bi_triad/sm80.cu");

    #[test]
    fn eight_warps_cover_each_16x32_subtile_once() {
        let mut origins = (0..8).map(warp_output_origin).collect::<Vec<_>>();
        origins.sort_unstable();
        assert_eq!(
            origins,
            vec![
                (0, 0),
                (0, 32),
                (16, 0),
                (16, 32),
                (32, 0),
                (32, 32),
                (48, 0),
                (48, 32),
            ]
        );
    }

    #[test]
    fn adapter_is_reversible_and_preserves_real_triad_abi_and_k16_pipeline() {
        let retained = retained_source(PRODUCTION).unwrap();
        let candidate = candidate_source(PRODUCTION).unwrap();
        assert_eq!(restore_retained_source(&candidate).unwrap(), retained);
        for parameter in [
            "float* __restrict__ C",
            "const T_ACT* __restrict__ A",
            "const T_ACT* __restrict__ B",
            "float alpha",
            "int M_red, int K_out, int N",
        ] {
            assert!(
                candidate.contains(parameter),
                "missing ABI parameter: {parameter}"
            );
        }
        assert!(candidate.contains("for (int ks = 0; ks < 4; ++ks)"));
        assert!(candidate.contains("a_frag[(ks + 1) & 1][fm]"));
        assert!(candidate.contains("a_frag[ks & 1][fm]"));
        assert!(candidate.contains("gemm_bi_accumulate_float2_or_scalar("));
    }

    #[test]
    fn malformed_or_ambiguous_parent_fails_closed() {
        assert!(candidate_source("").is_err());
        let duplicate = format!("{PRODUCTION}\n#define GEMM_BI_TC64_STAGE_TN_ASYNC\n");
        assert!(candidate_source(&duplicate).is_err());
    }
}
