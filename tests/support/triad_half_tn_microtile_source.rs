pub const M32N32_SYMBOL_PREFIX: &str = "gemm_bi_tn_test_m32n32_sm80_mma_half_v1_";
pub const M16N32_SYMBOL_PREFIX: &str = "gemm_bi_tn_test_m16n32_sm80_mma_half_v1_";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Microtile {
    M32N32,
    M16N32,
}

fn replace_exact(
    source: &mut String,
    before: &str,
    after: &str,
    expected: usize,
) -> Result<(), String> {
    let observed = source.matches(before).count();
    if observed != expected {
        return Err(format!(
            "microtile source anchor {before:?}: expected {expected}, observed {observed}"
        ));
    }
    *source = source.replace(before, after);
    Ok(())
}

pub fn candidate_source(production: &str, tile: Microtile) -> Result<String, String> {
    const START: &str = "#define GEMM_BI_TN_RECT_BM 128";
    const END: &str = "#undef GEMM_BI_TN_RECT_BM";
    if production.matches(START).count() != 1 {
        return Err(format!(
            "microtile source anchor {START:?}: expected 1, observed {}",
            production.matches(START).count()
        ));
    }
    if production.matches(END).count() != 1 {
        return Err(format!(
            "microtile source anchor {END:?}: expected 1, observed {}",
            production.matches(END).count()
        ));
    }
    let start = production.find(START).unwrap();
    let end = production[start..].find(END).unwrap() + start + END.len();
    let mut section = production[start..end].to_owned();
    let (bm, threads, warp_m, warp_n, prefix) = match tile {
        Microtile::M32N32 => (
            "32",
            "128",
            "int warpM = (warp >> 1) * 16;",
            "int warpN = (warp & 1) * 16;",
            M32N32_SYMBOL_PREFIX,
        ),
        Microtile::M16N32 => (
            "16",
            "64",
            "int warpM = 0;",
            "int warpN = warp * 16;",
            M16N32_SYMBOL_PREFIX,
        ),
    };
    replace_exact(
        &mut section,
        "#define GEMM_BI_TN_RECT_BM 128",
        &format!("#define GEMM_BI_TN_RECT_BM {bm}"),
        1,
    )?;
    replace_exact(
        &mut section,
        "#define GEMM_BI_TN_RECT_BN 64",
        "#define GEMM_BI_TN_RECT_BN 32",
        1,
    )?;
    replace_exact(
        &mut section,
        "#define GEMM_BI_TN_RECT_THREADS 256",
        &format!("#define GEMM_BI_TN_RECT_THREADS {threads}"),
        1,
    )?;
    replace_exact(
        &mut section,
        "__launch_bounds__(256, 2)",
        &format!("__launch_bounds__({threads}, 1)"),
        1,
    )?;
    replace_exact(
        &mut section,
        "void gemm_bi_tn_tc128x64_##SUFFIX",
        &format!("void {prefix}##SUFFIX"),
        1,
    )?;
    replace_exact(&mut section, "int warpM = (warp >> 1) * 32;", warp_m, 1)?;
    replace_exact(&mut section, "int warpN = (warp & 1) * 32;", warp_n, 1)?;
    replace_exact(
        &mut section,
        "float acc[2][4][4];",
        "float acc[1][2][4];",
        1,
    )?;
    replace_exact(&mut section, "fm < 2", "fm < 1", 4)?;
    replace_exact(&mut section, "fn < 4", "fn < 2", 4)?;

    Ok(format!(
        "{}{}{}",
        &production[..start],
        section,
        &production[end..]
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    const PRODUCTION: &str = include_str!("../../kernels/gemm_bi_triad/sm80.cu");

    #[test]
    fn candidates_have_exact_microtile_ownership_and_symbols() {
        let m32 = candidate_source(PRODUCTION, Microtile::M32N32).unwrap();
        assert!(m32.contains("#define GEMM_BI_TN_RECT_BM 32"));
        assert!(m32.contains("#define GEMM_BI_TN_RECT_BN 32"));
        assert!(m32.contains("#define GEMM_BI_TN_RECT_THREADS 128"));
        assert!(m32.contains("int warpM = (warp >> 1) * 16;"));
        assert!(m32.contains("int warpN = (warp & 1) * 16;"));
        assert!(m32.contains("float acc[1][2][4];"));
        assert!(m32.contains(&format!("void {}##SUFFIX", M32N32_SYMBOL_PREFIX)));

        let m16 = candidate_source(PRODUCTION, Microtile::M16N32).unwrap();
        assert!(m16.contains("#define GEMM_BI_TN_RECT_BM 16"));
        assert!(m16.contains("#define GEMM_BI_TN_RECT_BN 32"));
        assert!(m16.contains("#define GEMM_BI_TN_RECT_THREADS 64"));
        assert!(m16.contains("int warpM = 0;"));
        assert!(m16.contains("int warpN = warp * 16;"));
        assert!(m16.contains("float acc[1][2][4];"));
        assert!(m16.contains(&format!("void {}##SUFFIX", M16N32_SYMBOL_PREFIX)));
    }

    #[test]
    fn candidates_keep_bk32_s3_and_tn_numeric_epilogue() {
        for tile in [Microtile::M32N32, Microtile::M16N32] {
            let source = candidate_source(PRODUCTION, tile).unwrap();
            assert!(source.contains("#define GEMM_BI_TN_RECT_BK 32"));
            assert!(source.contains("#define GEMM_BI_TN_RECT_STAGES 3"));
            assert!(source.contains("C[(long long)gr * N + gc] += alpha * acc[fm][fn][e];"));
            assert!(source.contains("for (int ks = 0; ks < (GEMM_BI_TN_RECT_BK / 16); ks++)"));
        }
    }

    #[test]
    fn output_ownership_is_exact_for_both_microtiles() {
        let cases = [
            (Microtile::M32N32, 4usize, 32usize, 32usize),
            (Microtile::M16N32, 2usize, 16usize, 32usize),
        ];
        for (tile, warps, rows, columns) in cases {
            let mut owners = vec![0u8; rows * columns];
            for warp in 0..warps {
                let (warp_m, warp_n) = match tile {
                    Microtile::M32N32 => ((warp >> 1) * 16, (warp & 1) * 16),
                    Microtile::M16N32 => (0, warp * 16),
                };
                for row in warp_m..warp_m + 16 {
                    for column in warp_n..warp_n + 16 {
                        owners[row * columns + column] += 1;
                    }
                }
            }
            assert!(owners.into_iter().all(|owners| owners == 1));
        }
    }

    #[test]
    fn bk32_tiles_preserve_the_tc64_per_output_k16_order() {
        for reduction in [1usize, 31, 32, 63, 64, 1_024] {
            let old = (0..reduction.div_ceil(64))
                .flat_map(|tile| (0..4).map(move |step| tile * 64 + step * 16))
                .take_while(|&k| k < reduction)
                .collect::<Vec<_>>();
            let new = (0..reduction.div_ceil(32))
                .flat_map(|tile| (0..2).map(move |step| tile * 32 + step * 16))
                .take_while(|&k| k < reduction)
                .collect::<Vec<_>>();
            assert_eq!(new, old, "reduction={reduction}");
        }
    }

    #[test]
    fn missing_or_duplicate_anchor_fails_closed() {
        let missing = PRODUCTION.replacen("#define GEMM_BI_TN_RECT_BM 128", "", 1);
        assert!(candidate_source(&missing, Microtile::M32N32).is_err());
        let duplicate = format!("{PRODUCTION}\n#define GEMM_BI_TN_RECT_BM 128\n");
        let error = candidate_source(&duplicate, Microtile::M16N32).unwrap_err();
        assert!(error.contains("expected 1"), "{error}");
    }
}
