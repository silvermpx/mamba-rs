pub const SYMBOL_PREFIX: &str = "gemm_bi_nt_test_tc64_bk32_s3_";

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
        require_once(production, anchor)?;
    }
    let start = production.find(START).unwrap();
    let end = production.find(END).unwrap();
    if start >= end {
        return Err("NT section boundaries are reversed".into());
    }
    let mut source = production[start..end].to_owned();
    for (from, to) in [
        (
            "Ys[2][GEMM_BI_TC64_BM][GEMM_BI_TC64_LDA]",
            "Ys[3][GEMM_BI_TC64_BM][GEMM_BI_TC64_LDA]",
        ),
        (
            "Ws[2][GEMM_BI_TC64_BN][GEMM_BI_TC64_LDA]",
            "Ws[3][GEMM_BI_TC64_BN][GEMM_BI_TC64_LDA]",
        ),
        ("read_buf ^= 1;", "read_buf = (read_buf + 1) % 3;"),
        (
            "void gemm_bi_nt_tc64_##SUFFIX",
            "void gemm_bi_nt_test_tc64_bk32_s3_##SUFFIX",
        ),
    ] {
        require_once(&source, from)?;
        source = source.replacen(from, to, 1);
    }
    const TILE_COUNT: &str = "    int num_n_tiles = (N + GEMM_BI_TC64_BK - 1) / GEMM_BI_TC64_BK;";
    const READ: &str = "        unsigned Ys_rd =";
    require_once(&source, TILE_COUNT)?;
    require_once(&source, READ)?;
    let begin = source[source.find(TILE_COUNT).unwrap()..]
        .find('\n')
        .unwrap()
        + source.find(TILE_COUNT).unwrap()
        + 1;
    let end = source.find(READ).unwrap();
    if begin >= end {
        return Err("NT pipeline boundaries are reversed".into());
    }
    let pipeline = r#"    for (int preload = 0; preload < 2; ++preload) {
        if (preload < num_n_tiles) {
            if (fast_stage) {
                GEMM_BI_TC64_STAGE_NT_ASYNC(preload, preload * GEMM_BI_TC64_BK);
            } else {
                GEMM_BI_TC64_STAGE_NT_SCALAR(preload, preload * GEMM_BI_TC64_BK, T_ACT, FROM_F);
            }
        } else if (fast_stage) {
            asm volatile("cp.async.commit_group;\n");
        }
    }
    int read_buf = 0;
    for (int nt = 0; nt < num_n_tiles; nt++) {
        if (fast_stage) asm volatile("cp.async.wait_group 1;\n");
        __syncthreads();
        int next = nt + 2;
        if (next < num_n_tiles) {
            if (fast_stage) {
                GEMM_BI_TC64_STAGE_NT_ASYNC(next % 3, next * GEMM_BI_TC64_BK);
            } else {
                GEMM_BI_TC64_STAGE_NT_SCALAR(next % 3, next * GEMM_BI_TC64_BK, T_ACT, FROM_F);
            }
        } else if (fast_stage) {
            asm volatile("cp.async.commit_group;\n");
        }
"#;
    let replacement = pipeline
        .lines()
        .map(|line| format!("{line} \\\n"))
        .collect::<String>();
    source.replace_range(begin..end, &replacement);
    let definitions = "#define GEMM_BI_TC64_BM 64\n#define GEMM_BI_TC64_BN 64\n#define GEMM_BI_TC64_BK 32\n#define GEMM_BI_TC64_THREADS 128\n#define GEMM_BI_TC64_LDA (GEMM_BI_TC64_BK + 8)\n";
    Ok(format!("{definitions}{source}"))
}

fn require_once(source: &str, anchor: &str) -> Result<(), String> {
    let count = source.matches(anchor).count();
    if count == 1 {
        Ok(())
    } else {
        Err(format!(
            "NT S3 anchor {anchor:?}: expected1 observed{count}"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const PRODUCTION: &str = include_str!("../../kernels/gemm_bi_triad/sm80.cu");

    #[test]
    fn source_has_only_nt_exports_and_bk32_three_stage_ring() {
        let source = candidate_source(PRODUCTION).unwrap();
        assert!(source.contains("#define GEMM_BI_TC64_BK 32"));
        assert!(source.contains("Ys[3][GEMM_BI_TC64_BM][GEMM_BI_TC64_LDA]"));
        assert!(source.contains("Ws[3][GEMM_BI_TC64_BN][GEMM_BI_TC64_LDA]"));
        assert!(source.contains("cp.async.wait_group 1;"));
        assert!(source.contains("read_buf = (read_buf + 1) % 3;"));
        assert!(source.contains("void gemm_bi_nt_test_tc64_bk32_s3_##SUFFIX"));
        assert!(!source.contains("void gemm_bi_nn_"));
        assert!(!source.contains("void gemm_bi_tn_"));
        assert!(source.contains("FROM_F(alpha * acc[fm][fn][e])"));
    }

    #[test]
    fn three_stage_copy_ring_never_overwrites_the_current_read_stage() {
        for tiles in [0usize, 1, 2, 3, 4, 24, 61] {
            let mut stages = [None; 3];
            for (slot, item) in stages.iter_mut().enumerate().take(2) {
                if slot < tiles {
                    *item = Some(slot);
                }
            }
            let mut visited = Vec::new();
            for tile in 0..tiles {
                let read = tile % 3;
                assert_eq!(stages[read], Some(tile));
                let next = tile + 2;
                let write = next % 3;
                assert_ne!(read, write);
                if next < tiles {
                    stages[write] = Some(next);
                }
                visited.extend([tile * 32, tile * 32 + 16]);
            }
            assert_eq!(visited, (0..tiles * 2).map(|i| i * 16).collect::<Vec<_>>());
        }
        assert_eq!(3 * 2 * 64 * (32 + 8) * 2, 30_720);
    }

    #[test]
    fn adapter_rejects_changed_baseline_and_ambiguous_pipeline() {
        assert!(
            candidate_source(
                &PRODUCTION.replace("#define GEMM_BI_TC64_BK 64", "#define GEMM_BI_TC64_BK 32")
            )
            .is_err()
        );
        assert!(
            candidate_source(
                &PRODUCTION.replace("        unsigned Ys_rd =", "        unsigned missing =")
            )
            .is_err()
        );
    }
}
