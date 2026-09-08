#[path = "triad_half_tn_vec2_epilogue_source.rs"]
mod parent;

pub const SYMBOL_PREFIX: &str = "gemm_bi_tn_test_m64n128_bk64_s2_regpipe_vec2_";

pub fn candidate_source(production: &str) -> Result<String, String> {
    let mut source = parent::candidate_source(production)?;
    replace_exact(
        &mut source,
        "#define GEMM_BI_TC64_BM 64\n#define GEMM_BI_TC64_BN 64\n#define GEMM_BI_TC64_BK 64\n#define GEMM_BI_TC64_THREADS 128\n#define GEMM_BI_TC64_LDB 64\n#define GEMM_BI_HALF_TN_INDEX(row, col) ((row) * 64 + ((col) ^ (((row) & 7) * 8)))",
        "#define GEMM_BI_TC64_BM 64\n#define GEMM_BI_TC64_BN 128\n#define GEMM_BI_TC64_BK 64\n#define GEMM_BI_TC64_THREADS 128\n#define GEMM_BI_HALF_TN_X_INDEX(row, col) ((row) * 64 + ((col) ^ (((row) & 7) * 8)))\n#define GEMM_BI_HALF_TN_Y_INDEX(row, col) ((row) * 128 + ((col) ^ (((row) & 7) * 8)))",
    )?;
    for (before, after) in [
        (
            "Xs_sbase + (unsigned)((buf) * GEMM_BI_TC64_BK * GEMM_BI_TC64_LDB * 2)",
            "Xs_sbase + (unsigned)((buf) * GEMM_BI_TC64_BK * GEMM_BI_TC64_BM * 2)",
        ),
        (
            "Ys_sbase + (unsigned)((buf) * GEMM_BI_TC64_BK * GEMM_BI_TC64_LDB * 2)",
            "Ys_sbase + (unsigned)((buf) * GEMM_BI_TC64_BK * GEMM_BI_TC64_BN * 2)",
        ),
        (
            "_xs + (unsigned)((GEMM_BI_HALF_TN_INDEX(_r, _c)) * 2)",
            "_xs + (unsigned)((GEMM_BI_HALF_TN_X_INDEX(_r, _c)) * 2)",
        ),
        (
            "_ys + (unsigned)((GEMM_BI_HALF_TN_INDEX(_r, _c)) * 2)",
            "_ys + (unsigned)((GEMM_BI_HALF_TN_Y_INDEX(_r, _c)) * 2)",
        ),
        (
            "_xs[GEMM_BI_HALF_TN_INDEX(_r, _c)]",
            "_xs[GEMM_BI_HALF_TN_X_INDEX(_r, _c)]",
        ),
        (
            "_ys[GEMM_BI_HALF_TN_INDEX(_r, _c)]",
            "_ys[GEMM_BI_HALF_TN_Y_INDEX(_r, _c)]",
        ),
        (
            "T_ACT Xs[2][GEMM_BI_TC64_BK][GEMM_BI_TC64_LDB]",
            "T_ACT Xs[2][GEMM_BI_TC64_BK][GEMM_BI_TC64_BM]",
        ),
        (
            "T_ACT Ys[2][GEMM_BI_TC64_BK][GEMM_BI_TC64_LDB]",
            "T_ACT Ys[2][GEMM_BI_TC64_BK][GEMM_BI_TC64_BN]",
        ),
        ("int warpM = (warp / 2) * 32;", "int warpM = 0;"),
        ("int warpN = (warp % 2) * 32;", "int warpN = warp * 32;"),
        (
            "Xs_sbase + (unsigned)(read_buf * GEMM_BI_TC64_BK * GEMM_BI_TC64_LDB * 2)",
            "Xs_sbase + (unsigned)(read_buf * GEMM_BI_TC64_BK * GEMM_BI_TC64_BM * 2)",
        ),
        (
            "Ys_sbase + (unsigned)(read_buf * GEMM_BI_TC64_BK * GEMM_BI_TC64_LDB * 2)",
            "Ys_sbase + (unsigned)(read_buf * GEMM_BI_TC64_BK * GEMM_BI_TC64_BN * 2)",
        ),
        ("float acc[2][4][4]", "float acc[4][4][4]"),
        ("unsigned a_frag[2][2][4]", "unsigned a_frag[2][4][4]"),
        (parent::SYMBOL_PREFIX, SYMBOL_PREFIX),
        (
            "#undef GEMM_BI_HALF_TN_INDEX",
            "#undef GEMM_BI_HALF_TN_X_INDEX\n#undef GEMM_BI_HALF_TN_Y_INDEX",
        ),
    ] {
        replace_exact(&mut source, before, after)?;
    }
    replace_all_exact(
        &mut source,
        "GEMM_BI_HALF_TN_INDEX(srow, scol)",
        "GEMM_BI_HALF_TN_X_INDEX(srow, scol)",
        2,
    )?;
    replace_all_exact(
        &mut source,
        "GEMM_BI_HALF_TN_INDEX(srow, warpN + fn * 8)",
        "GEMM_BI_HALF_TN_Y_INDEX(srow, warpN + fn * 8)",
        2,
    )?;
    replace_all_exact(
        &mut source,
        "for (int fm = 0; fm < 2; fm++)",
        "for (int fm = 0; fm < 4; fm++)",
        5,
    )?;
    if source.contains("GEMM_BI_TC64_LDB") || source.contains("GEMM_BI_HALF_TN_INDEX(") {
        return Err("TN M64N128 source retained a square shared-layout anchor".into());
    }
    Ok(source)
}

fn replace_exact(source: &mut String, before: &str, after: &str) -> Result<(), String> {
    replace_all_exact(source, before, after, 1)
}

fn replace_all_exact(
    source: &mut String,
    before: &str,
    after: &str,
    expected: usize,
) -> Result<(), String> {
    let actual = source.matches(before).count();
    if actual != expected {
        return Err(format!(
            "TN M64N128 regpipe anchor {before:?}: expected {expected}, observed {actual}"
        ));
    }
    *source = source.replace(before, after);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const PRODUCTION: &str = include_str!("../../kernels/gemm_bi_triad/sm80.cu");

    fn offset(stride: usize, row: usize, column: usize) -> usize {
        row * stride + (column ^ ((row & 7) * 8))
    }

    #[test]
    fn four_warps_own_every_m64_n128_output_once() {
        let mut owners = vec![0u8; 64 * 128];
        for warp in 0..4 {
            let warp_m = 0;
            let warp_n = warp * 32;
            for lane in 0..32 {
                let group = lane >> 2;
                let thread = lane & 3;
                for m_atom in 0..4 {
                    for n_atom in 0..4 {
                        for element in 0..4 {
                            let row = warp_m + m_atom * 16 + group + usize::from(element >= 2) * 8;
                            let column = warp_n + n_atom * 8 + 2 * thread + (element & 1);
                            owners[row * 128 + column] += 1;
                        }
                    }
                }
            }
        }
        assert!(owners.into_iter().all(|count| count == 1));
    }

    #[test]
    fn two_stages_have_exact_aligned_copy_ownership_and_shared_budget() {
        let mut per_thread = [[0usize; 128]; 2];
        for stride in [64usize, 128] {
            let mut owners = vec![0u8; 2 * 64 * stride];
            for stage in 0..2 {
                for thread in 0..128 {
                    for linear in (thread..64 * (stride / 8)).step_by(128) {
                        let row = linear / (stride / 8);
                        let column = (linear % (stride / 8)) * 8;
                        let base = stage * 64 * stride + offset(stride, row, column);
                        assert_eq!(base % 8, 0);
                        for element in 0..8 {
                            owners[base + element] += 1;
                        }
                        per_thread[stage][thread] += 1;
                    }
                }
            }
            assert!(owners.into_iter().all(|count| count == 1));
        }
        assert!(per_thread.into_iter().flatten().all(|copies| copies == 12));
        assert_eq!(2 * (64 * 64 + 64 * 128) * 2, 49_152);
    }

    #[test]
    fn source_widens_only_n_ownership_around_the_existing_regpipe_vec2_body() {
        let source = candidate_source(PRODUCTION).unwrap();
        assert!(source.contains("#define GEMM_BI_TC64_BN 128"));
        assert!(source.contains("#define GEMM_BI_TC64_THREADS 128"));
        assert!(source.contains("int warpM = 0;"));
        assert!(source.contains("int warpN = warp * 32;"));
        assert!(source.contains("float acc[4][4][4];"));
        assert!(source.contains("unsigned a_frag[2][4][4];"));
        assert!(source.contains("for (int fm = 0; fm < 4; fm++)"));
        assert!(
            source.contains(
                "for (int _i = threadIdx.x; _i < GEMM_BI_TC64_BK * (GEMM_BI_TC64_BN / 8);"
            )
        );
        assert!(source.contains("_i += GEMM_BI_TC64_THREADS)"));
        assert!(source.contains("gemm_bi_accumulate_float2_or_scalar("));
        assert!(source.contains(&format!("void {SYMBOL_PREFIX}##SUFFIX")));
        assert!(source.contains("for (int ks = 0; ks < 4; ++ks)"));
        assert!(source.contains("a_frag[(ks + 1) & 1][fm]"));
        assert!(source.contains("a_frag[ks & 1][fm]"));
        assert!(!source.contains(parent::SYMBOL_PREFIX));
    }

    #[test]
    fn malformed_parent_fails_closed() {
        assert!(candidate_source("").is_err());
        let duplicate = format!("{PRODUCTION}\n#define GEMM_BI_TC64_STAGE_TN_ASYNC\n");
        assert!(candidate_source(&duplicate).is_err());
    }
}
