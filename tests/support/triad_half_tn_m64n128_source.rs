pub const SYMBOL_PREFIX: &str = "gemm_bi_tn_test_m64n128_bk64_s2_xor_";

/// Widens only the TN output-column axis of the incumbent M64N64/BK64/S2
/// body. Each warp keeps its 32x32 output fragment and ascending K16 MMA
/// order; the two operands use independent compact shared strides.
pub fn candidate_source(production: &str) -> Result<String, String> {
    const START: &str = "#define GEMM_BI_TC64_STAGE_TN_ASYNC";
    const END: &str = "#define GEMM_BI_TN_RECT_BM 128";
    for anchor in [
        START,
        END,
        "#define GEMM_BI_TC64_BK 64",
        "#define GEMM_BI_TC64_BM 64",
        "#define GEMM_BI_TC64_BN 64",
        "#define GEMM_BI_TC64_THREADS 128",
    ] {
        require_count(production, anchor, 1)?;
    }
    let start = production.find(START).unwrap();
    let end = production.find(END).unwrap();
    if start >= end {
        return Err("TN M64N128 source boundaries reversed".into());
    }
    let mut source = production[start..end].to_owned();
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
            "_xs + (unsigned)((_r * GEMM_BI_TC64_LDB + _c) * 2)",
            "_xs + (unsigned)(GEMM_BI_HALF_TN_X_INDEX(_r, _c) * 2)",
        ),
        (
            "_ys + (unsigned)((_r * GEMM_BI_TC64_LDB + _c) * 2)",
            "_ys + (unsigned)(GEMM_BI_HALF_TN_Y_INDEX(_r, _c) * 2)",
        ),
        (
            "_xs[_r * GEMM_BI_TC64_LDB + _c]",
            "_xs[GEMM_BI_HALF_TN_X_INDEX(_r, _c)]",
        ),
        (
            "_ys[_r * GEMM_BI_TC64_LDB + _c]",
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
        (
            "int warpM = (warp / 2) * 32;",
            "int warpM = (warp / 4) * 32;",
        ),
        (
            "int warpN = (warp % 2) * 32;",
            "int warpN = (warp % 4) * 32;",
        ),
        (
            "Xs_sbase + (unsigned)(read_buf * GEMM_BI_TC64_BK * GEMM_BI_TC64_LDB * 2)",
            "Xs_sbase + (unsigned)(read_buf * GEMM_BI_TC64_BK * GEMM_BI_TC64_BM * 2)",
        ),
        (
            "Ys_sbase + (unsigned)(read_buf * GEMM_BI_TC64_BK * GEMM_BI_TC64_LDB * 2)",
            "Ys_sbase + (unsigned)(read_buf * GEMM_BI_TC64_BK * GEMM_BI_TC64_BN * 2)",
        ),
        (
            "Xs_rd + (unsigned)((srow * GEMM_BI_TC64_LDB + scol) * 2)",
            "Xs_rd + (unsigned)(GEMM_BI_HALF_TN_X_INDEX(srow, scol) * 2)",
        ),
        (
            "srow * GEMM_BI_TC64_LDB + warpN + fn * 8",
            "GEMM_BI_HALF_TN_Y_INDEX(srow, warpN + fn * 8)",
        ),
        (
            "void gemm_bi_tn_tc64_##SUFFIX",
            "void gemm_bi_tn_test_m64n128_bk64_s2_xor_##SUFFIX",
        ),
    ] {
        replace_exact(&mut source, before, after)?;
    }
    if source.contains("GEMM_BI_TC64_LDB") {
        return Err("TN M64N128 source retained the square shared stride".into());
    }
    let definitions = "#define GEMM_BI_TC64_BM 64\n#define GEMM_BI_TC64_BN 128\n#define GEMM_BI_TC64_BK 64\n#define GEMM_BI_TC64_THREADS 256\n#define GEMM_BI_HALF_TN_X_INDEX(row, col) ((row) * 64 + ((col) ^ (((row) & 7) * 8)))\n#define GEMM_BI_HALF_TN_Y_INDEX(row, col) ((row) * 128 + ((col) ^ (((row) & 7) * 8)))\n";
    Ok(format!(
        "{definitions}{source}\n#undef GEMM_BI_HALF_TN_X_INDEX\n#undef GEMM_BI_HALF_TN_Y_INDEX\n"
    ))
}

fn replace_exact(source: &mut String, before: &str, after: &str) -> Result<(), String> {
    require_count(source, before, 1)?;
    *source = source.replacen(before, after, 1);
    Ok(())
}

fn require_count(source: &str, anchor: &str, expected: usize) -> Result<(), String> {
    let actual = source.matches(anchor).count();
    if actual == expected {
        Ok(())
    } else {
        Err(format!(
            "TN M64N128 anchor {anchor:?}: expected {expected}, observed {actual}"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PRODUCTION: &str = include_str!("../../kernels/gemm_bi_triad/sm80.cu");

    fn offset(stride: usize, row: usize, column: usize) -> usize {
        row * stride + (column ^ ((row & 7) * 8))
    }

    #[test]
    fn rectangular_compact_layouts_are_bijective_and_keep_six_vector_copies_per_thread() {
        for stride in [64usize, 128] {
            let mut owners = vec![0u8; 64 * stride];
            for row in 0..64 {
                for column in 0..stride {
                    owners[offset(stride, row, column)] += 1;
                }
                for column in (0..stride).step_by(8) {
                    let base = offset(stride, row, column);
                    assert_eq!(base % 8, 0);
                    for element in 0..8 {
                        assert_eq!(offset(stride, row, column + element), base + element);
                    }
                }
            }
            assert!(owners.into_iter().all(|count| count == 1));
        }
        assert_eq!((64 * 64 + 64 * 128) / 8 / 256, 6);
        assert_eq!(2 * (64 * 64 + 64 * 128) * 2, 49_152);
    }

    #[test]
    fn eight_warps_own_every_m64n128_output_once() {
        let mut owners = vec![0u8; 64 * 128];
        for warp in 0..8 {
            let warp_m = (warp >> 2) * 32;
            let warp_n = (warp & 3) * 32;
            for lane in 0..32 {
                let group = lane >> 2;
                let thread = lane & 3;
                for m_atom in 0..2 {
                    for n_atom in 0..4 {
                        let row = warp_m + m_atom * 16 + group;
                        let column = warp_n + n_atom * 8 + 2 * thread;
                        for element in 0..4 {
                            let output_row = row + usize::from(element >= 2) * 8;
                            let output_column = column + (element & 1);
                            owners[output_row * 128 + output_column] += 1;
                        }
                    }
                }
            }
        }
        assert!(owners.into_iter().all(|count| count == 1));
    }

    #[test]
    fn source_is_exactly_scoped_and_keeps_the_tn_math_and_float_epilogue() {
        let source = candidate_source(PRODUCTION).unwrap();
        assert!(source.contains("#define GEMM_BI_TC64_BM 64"));
        assert!(source.contains("#define GEMM_BI_TC64_BN 128"));
        assert!(source.contains("#define GEMM_BI_TC64_THREADS 256"));
        assert!(source.contains("T_ACT Xs[2][GEMM_BI_TC64_BK][GEMM_BI_TC64_BM]"));
        assert!(source.contains("T_ACT Ys[2][GEMM_BI_TC64_BK][GEMM_BI_TC64_BN]"));
        assert!(source.contains("int warpM = (warp / 4) * 32;"));
        assert!(source.contains("int warpN = (warp % 4) * 32;"));
        assert!(source.contains("void gemm_bi_tn_test_m64n128_bk64_s2_xor_##SUFFIX"));
        assert_eq!(
            source
                .matches("mma.sync.aligned.m16n8k16.row.col.f32.")
                .count(),
            1
        );
        assert!(source.contains("for (int ks = 0; ks < (GEMM_BI_TC64_BK / 16); ks++)"));
        assert!(source.contains("C[(long long)gr * N + gc] += alpha * acc[fm][fn][e];"));
        assert!(!source.contains("void gemm_bi_nn_"));
        assert!(!source.contains("void gemm_bi_nt_"));
        assert!(!source.contains("GEMM_BI_TC64_LDB"));
    }

    #[test]
    fn missing_or_duplicated_source_anchor_is_rejected() {
        let missing = PRODUCTION.replacen("#define GEMM_BI_TC64_STAGE_TN_ASYNC", "", 1);
        assert!(candidate_source(&missing).is_err());
        let duplicate = format!("{PRODUCTION}\n#define GEMM_BI_TC64_STAGE_TN_ASYNC\n");
        assert!(candidate_source(&duplicate).is_err());
    }
}
