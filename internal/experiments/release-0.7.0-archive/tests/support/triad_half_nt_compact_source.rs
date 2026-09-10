pub const SYMBOL_PREFIX: &str = "gemm_bi_nt_test_tc64_bk64_s2_xor_";

/// Reuses Fixed's row*64 + (column XOR ((row&7)*8)) half layout for BOTH
/// NT operands. Only shared addresses change; incumbent BK64/S2 math remains.
pub fn candidate_source(production: &str) -> Result<String, String> {
    const START: &str = "#define GEMM_BI_TC64_STAGE_NT_ASYNC";
    const END: &str = "#undef GEMM_BI_DEFINE_GEMM_BI_NN_TC128";
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
        return Err("NT compact source boundaries reversed".into());
    }
    let mut source = production[start..end].to_owned();
    for (before, after, count) in [
        (
            "_m * GEMM_BI_TC64_LDA + _c",
            "GEMM_BI_HALF_NT_INDEX(_m, _c)",
            2,
        ),
        (
            "_k * GEMM_BI_TC64_LDA + _c",
            "GEMM_BI_HALF_NT_INDEX(_k, _c)",
            2,
        ),
        (
            "row * GEMM_BI_TC64_LDA + k0 + lm_col_off",
            "GEMM_BI_HALF_NT_INDEX(row, k0 + lm_col_off)",
            1,
        ),
        (
            "row * GEMM_BI_TC64_LDA + k0 + lmb_col_off",
            "GEMM_BI_HALF_NT_INDEX(row, k0 + lmb_col_off)",
            1,
        ),
        (
            "void gemm_bi_nt_tc64_##SUFFIX",
            "void gemm_bi_nt_test_tc64_bk64_s2_xor_##SUFFIX",
            1,
        ),
    ] {
        require_count(&source, before, count)?;
        source = source.replace(before, after);
    }
    let definitions = "#define GEMM_BI_TC64_BM 64\n#define GEMM_BI_TC64_BN 64\n#define GEMM_BI_TC64_BK 64\n#define GEMM_BI_TC64_THREADS 128\n#define GEMM_BI_TC64_LDA 64\n#define GEMM_BI_HALF_NT_INDEX(row, col) ((row) * 64 + ((col) ^ (((row) & 7) * 8)))\n";
    Ok(format!(
        "{definitions}{source}\n#undef GEMM_BI_HALF_NT_INDEX\n"
    ))
}

fn require_count(source: &str, anchor: &str, expected: usize) -> Result<(), String> {
    let actual = source.matches(anchor).count();
    if actual == expected {
        Ok(())
    } else {
        Err(format!(
            "NT compact anchor {anchor:?}: expected{expected} observed{actual}"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const PRODUCTION: &str = include_str!("../../kernels/gemm_bi_triad/sm80.cu");
    const FIXED_LAYOUT: &str =
        include_str!("../../kernels/gemm_bi_inference/sm89_half_swizzle_layout.cuh");

    fn offset(row: usize, column: usize) -> usize {
        row * 64 + (column ^ ((row & 7) * 8))
    }

    #[test]
    fn compact_layout_is_bijective_and_preserves_each_16_byte_copy() {
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
        assert!(owners.into_iter().all(|n| n == 1));
        assert_eq!(2 * 2 * 64 * 64 * 2, 32_768); // stages * operands * rows * columns * bytes
    }

    #[test]
    fn ldmatrix_rows_recover_original_values_and_use_distinct_bank_groups() {
        let mut shared = vec![0usize; 64 * 64];
        for row in 0..64 {
            for column in 0..64 {
                shared[offset(row, column)] = row * 64 + column;
            }
        }
        // Each eight-lane ldmatrix address group supplies eight distinct rows.
        for row_base in (0..64).step_by(8) {
            for column_base in (0..64).step_by(8) {
                let mut groups = [false; 8];
                for row in row_base..row_base + 8 {
                    let base = offset(row, column_base);
                    let group = (base * 2 / 16) % 8;
                    assert!(!groups[group]);
                    groups[group] = true;
                    for c in 0..8 {
                        assert_eq!(shared[base + c], row * 64 + column_base + c);
                    }
                }
            }
        }
    }

    #[test]
    fn source_rewrites_only_layout_and_export_preserving_math_and_pipeline() {
        let source = candidate_source(PRODUCTION).unwrap();
        assert!(source.contains("#define GEMM_BI_TC64_BK 64"));
        assert!(source.contains("#define GEMM_BI_TC64_LDA 64"));
        assert!(source.contains("Ys[2][GEMM_BI_TC64_BM][GEMM_BI_TC64_LDA]"));
        assert!(source.contains("Ws[2][GEMM_BI_TC64_BN][GEMM_BI_TC64_LDA]"));
        assert!(source.contains("void gemm_bi_nt_test_tc64_bk64_s2_xor_##SUFFIX"));
        assert_eq!(source.matches("GEMM_BI_HALF_NT_INDEX(").count(), 7);
        assert!(!source.contains("row * GEMM_BI_TC64_LDA + k0"));
        assert!(!source.contains("_m * GEMM_BI_TC64_LDA + _c"));
        assert!(!source.contains("_k * GEMM_BI_TC64_LDA + _c"));
        assert!(!source.contains("void gemm_bi_nn_"));
        assert!(!source.contains("void gemm_bi_tn_"));
        // The math and epilogue suffix must be byte-for-byte incumbent text.
        let marker = "            _Pragma(\"unroll\")";
        let math_anchor = "            for (int fm = 0; fm < 2; fm++) {";
        let old_start = PRODUCTION
            .find("#define GEMM_BI_TC64_STAGE_NT_ASYNC")
            .unwrap();
        let old = &PRODUCTION[old_start..];
        let old_start = old.find("mma.sync.aligned.m16n8k16.row.col.f32.").unwrap();
        let new_start = source
            .find("mma.sync.aligned.m16n8k16.row.col.f32.")
            .unwrap();
        let end = "GEMM_BI_DEFINE_GEMM_BI_NT_TC64(f16,  __half,        from_f_f16,  \"f16\")";
        assert_eq!(
            &source[new_start..source.find(end).unwrap() + end.len()],
            &old[old_start..old.find(end).unwrap() + end.len()]
        );
        assert!(source.contains(marker) && source.contains(math_anchor));
        assert!(source.contains("cp.async.wait_group 0;"));
        assert!(source.contains("read_buf ^= 1;"));
    }

    #[test]
    fn missing_or_duplicated_source_anchor_is_rejected() {
        let missing = PRODUCTION.replacen("#define GEMM_BI_TC64_BK 64", "", 1);
        assert!(candidate_source(&missing).is_err());
        let doubled = format!("{PRODUCTION}\n#define GEMM_BI_TC64_BK 64\n");
        assert!(candidate_source(&doubled).is_err());
    }
}
