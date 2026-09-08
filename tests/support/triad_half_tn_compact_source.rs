pub const SYMBOL_PREFIX: &str = "gemm_bi_tn_test_tc64_bk64_s2_xor_";

/// Reuses Fixed's row*64 + (column XOR ((row&7)*8)) half layout for both
/// TN operands. Only shared addresses change; incumbent BK64/S2 math remains.
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
        return Err("TN compact source boundaries reversed".into());
    }
    let mut source = production[start..end].to_owned();
    for (before, after, count) in [
        (
            "_r * GEMM_BI_TC64_LDB + _c",
            "GEMM_BI_HALF_TN_INDEX(_r, _c)",
            4,
        ),
        (
            "srow * GEMM_BI_TC64_LDB + scol",
            "GEMM_BI_HALF_TN_INDEX(srow, scol)",
            1,
        ),
        (
            "srow * GEMM_BI_TC64_LDB + warpN + fn * 8",
            "GEMM_BI_HALF_TN_INDEX(srow, warpN + fn * 8)",
            1,
        ),
        (
            "void gemm_bi_tn_tc64_##SUFFIX",
            "void gemm_bi_tn_test_tc64_bk64_s2_xor_##SUFFIX",
            1,
        ),
    ] {
        require_count(&source, before, count)?;
        source = source.replace(before, after);
    }
    let definitions = "#define GEMM_BI_TC64_BM 64\n#define GEMM_BI_TC64_BN 64\n#define GEMM_BI_TC64_BK 64\n#define GEMM_BI_TC64_THREADS 128\n#define GEMM_BI_TC64_LDB 64\n#define GEMM_BI_HALF_TN_INDEX(row, col) ((row) * 64 + ((col) ^ (((row) & 7) * 8)))\n";
    Ok(format!(
        "{definitions}{source}\n#undef GEMM_BI_HALF_TN_INDEX\n"
    ))
}

fn require_count(source: &str, anchor: &str, expected: usize) -> Result<(), String> {
    let actual = source.matches(anchor).count();
    if actual == expected {
        Ok(())
    } else {
        Err(format!(
            "TN compact anchor {anchor:?}: expected{expected} observed{actual}"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PRODUCTION: &str = include_str!("../../kernels/gemm_bi_triad/sm80.cu");
    const FIXED_LAYOUT: &str =
        include_str!("../../kernels/gemm_bi_fixed/sm89_half_swizzle_layout.cuh");

    fn offset(row: usize, column: usize) -> usize {
        row * 64 + (column ^ ((row & 7) * 8))
    }

    #[test]
    fn compact_tn_layout_is_bijective_and_preserves_every_vector_chunk() {
        assert!(FIXED_LAYOUT.contains("row * 64 + (k ^ ((row & 7) * 8))"));
        let mut owners = [0u8; 64 * 64];
        for row in 0..64 {
            for column in 0..64 {
                owners[offset(row, column)] += 1;
            }
            for column in (0..64).step_by(8) {
                let base = offset(row, column);
                assert_eq!(base % 8, 0);
                for element in 0..8 {
                    assert_eq!(offset(row, column + element), base + element);
                }
            }
        }
        assert!(owners.into_iter().all(|count| count == 1));
        assert_eq!(2 * 2 * 64 * 64 * 2, 32_768);
    }

    #[test]
    fn scalar_tail_staging_uses_disjoint_ring_slots_and_zero_fills_both_operands() {
        const STAGE_ELEMENTS: usize = 64 * 64;
        let mut a = vec![usize::MAX; 2 * STAGE_ELEMENTS];
        let mut b = vec![usize::MAX; 2 * STAGE_ELEMENTS];
        for stage in 0..2 {
            let reduction_rows = if stage == 0 { 61 } else { 3 };
            for row in 0..64 {
                for column in 0..64 {
                    let physical = stage * STAGE_ELEMENTS + offset(row, column);
                    a[physical] = if row < reduction_rows && column < 59 {
                        10_000 * stage + row * 64 + column
                    } else {
                        0
                    };
                    b[physical] = if row < reduction_rows && column < 57 {
                        20_000 * stage + row * 64 + column
                    } else {
                        0
                    };
                }
            }
        }
        for stage in 0..2 {
            let reduction_rows = if stage == 0 { 61 } else { 3 };
            for row in 0..64 {
                for column in 0..64 {
                    let physical = stage * STAGE_ELEMENTS + offset(row, column);
                    let expected_a = if row < reduction_rows && column < 59 {
                        10_000 * stage + row * 64 + column
                    } else {
                        0
                    };
                    let expected_b = if row < reduction_rows && column < 57 {
                        20_000 * stage + row * 64 + column
                    } else {
                        0
                    };
                    assert_eq!(a[physical], expected_a);
                    assert_eq!(b[physical], expected_b);
                }
            }
        }
    }

    #[test]
    fn source_changes_only_tn_layout_and_exports_while_preserving_math_and_epilogue() {
        let source = candidate_source(PRODUCTION).unwrap();
        assert!(source.contains("#define GEMM_BI_TC64_BK 64"));
        assert!(source.contains("#define GEMM_BI_TC64_LDB 64"));
        assert!(source.contains("Xs[2][GEMM_BI_TC64_BK][GEMM_BI_TC64_LDB]"));
        assert!(source.contains("Ys[2][GEMM_BI_TC64_BK][GEMM_BI_TC64_LDB]"));
        assert!(source.contains("void gemm_bi_tn_test_tc64_bk64_s2_xor_##SUFFIX"));
        assert_eq!(source.matches("GEMM_BI_HALF_TN_INDEX(").count(), 7);
        assert!(!source.contains("_r * GEMM_BI_TC64_LDB + _c"));
        assert!(!source.contains("srow * GEMM_BI_TC64_LDB + scol"));
        assert!(!source.contains("srow * GEMM_BI_TC64_LDB + warpN + fn * 8"));
        assert!(!source.contains("void gemm_bi_nn_"));
        assert!(!source.contains("void gemm_bi_nt_"));

        let old_section = &PRODUCTION[PRODUCTION
            .find("#define GEMM_BI_TC64_STAGE_TN_ASYNC")
            .unwrap()..];
        let old_start = old_section
            .find("mma.sync.aligned.m16n8k16.row.col.f32.")
            .unwrap();
        let new_start = source
            .find("mma.sync.aligned.m16n8k16.row.col.f32.")
            .unwrap();
        let end = "GEMM_BI_DEFINE_GEMM_BI_TN_TC64(f16,  __half,        from_f_f16,  \"f16\")";
        assert_eq!(
            &source[new_start..source.find(end).unwrap() + end.len()],
            &old_section[old_start..old_section.find(end).unwrap() + end.len()]
        );
        assert!(source.contains("cp.async.wait_group 0;"));
        assert!(source.contains("read_buf ^= 1;"));
        assert!(source.contains("C[(long long)gr * N + gc] += alpha * acc[fm][fn][e];"));
    }

    #[test]
    fn missing_or_duplicated_tn_source_anchor_is_rejected() {
        let missing = PRODUCTION.replacen("#define GEMM_BI_TC64_STAGE_TN_ASYNC", "", 1);
        assert!(candidate_source(&missing).is_err());
        let duplicate = format!("{PRODUCTION}\n#define GEMM_BI_TC64_STAGE_TN_ASYNC\n");
        assert!(candidate_source(&duplicate).is_err());
    }
}
