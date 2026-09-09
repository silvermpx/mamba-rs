#[path = "triad_half_tn_vec2_epilogue_source.rs"]
mod retained;

pub const SYMBOL_PREFIX: &str = "gemm_bi_tn_test_tc64_bk32_s3_regpipe_vec2_";
pub const RETAINED_SYMBOL_PREFIX: &str = retained::SYMBOL_PREFIX;
pub const BLOCK_THREADS: u32 = 128;
pub const STATIC_SHARED_BYTES: i32 = 24_576;
pub const REQUIRED_OCCUPANCY: u32 = 4;
pub const REGISTER_CAP: i32 = 128;
pub const RETAINED_STATIC_SHARED_BYTES: i32 = 32_768;
pub const RETAINED_REQUIRED_OCCUPANCY: u32 = 3;
pub const RETAINED_SOURCE_FNV64: u64 = 0x97e0_dd88_3dd8_96f0;
pub const TARGET: (usize, usize, usize) = (2_048, 768, 3_072);
pub const TARGET_GRID: u32 = 576;

pub fn retained_source(production: &str) -> Result<String, String> {
    retained::candidate_source(production)
}

pub fn candidate_source(production: &str) -> Result<String, String> {
    let mut source = retained_source(production)?;
    replace_exact(
        &mut source,
        "#define GEMM_BI_TC64_BK 64\n",
        "#define GEMM_BI_TC64_BK 32\n#define GEMM_BI_TC64_STAGES 3\n",
        1,
        "BK32/S3 definitions",
    )?;
    replace_exact(
        &mut source,
        "T_ACT Xs[2][GEMM_BI_TC64_BK][GEMM_BI_TC64_LDB]",
        "T_ACT Xs[GEMM_BI_TC64_STAGES][GEMM_BI_TC64_BK][GEMM_BI_TC64_LDB]",
        1,
        "Xs stage count",
    )?;
    replace_exact(
        &mut source,
        "T_ACT Ys[2][GEMM_BI_TC64_BK][GEMM_BI_TC64_LDB]",
        "T_ACT Ys[GEMM_BI_TC64_STAGES][GEMM_BI_TC64_BK][GEMM_BI_TC64_LDB]",
        1,
        "Ys stage count",
    )?;

    const SCHEDULE_START: &str =
        "    int num_m_tiles = (M_red + GEMM_BI_TC64_BK - 1) / GEMM_BI_TC64_BK;";
    const COMPUTE_START: &str = "        unsigned Xs_rd =";
    let start = unique_position(&source, SCHEDULE_START, "schedule start")?;
    let end = unique_position(&source, COMPUTE_START, "compute start")?;
    if start >= end {
        return Err("BK32/S3 schedule anchors reversed".into());
    }
    let retained_schedule = &source[start..end];
    for (needle, expected) in [
        ("GEMM_BI_TC64_STAGE_TN_ASYNC(0, 0)", 1),
        ("GEMM_BI_TC64_STAGE_TN_SCALAR(0, 0, T_ACT, FROM_F)", 1),
        ("cp.async.wait_group 0;", 1),
        ("read_buf ^ 1", 2),
        ("(mt + 1) * GEMM_BI_TC64_BK", 2),
    ] {
        require_count(retained_schedule, needle, expected, "retained schedule")?;
    }
    source.replace_range(start..end, &encode_macro_block(CANDIDATE_SCHEDULE));
    replace_exact(
        &mut source,
        "for (int ks = 0; ks < 4; ++ks)",
        "for (int ks = 0; ks < 2; ++ks)",
        1,
        "BK32 K16 loop",
    )?;
    replace_exact(
        &mut source,
        "if (ks + 1 < 4)",
        "if (ks + 1 < 2)",
        1,
        "BK32 fragment lookahead",
    )?;
    replace_exact(
        &mut source,
        "        read_buf ^= 1;                                                         \\\n",
        "",
        1,
        "remove two-slot advance",
    )?;
    replace_exact(
        &mut source,
        RETAINED_SYMBOL_PREFIX,
        SYMBOL_PREFIX,
        1,
        "candidate export",
    )?;
    replace_exact(
        &mut source,
        "\n\n#undef GEMM_BI_HALF_TN_INDEX",
        "\n#undef GEMM_BI_TC64_STAGES\n\n#undef GEMM_BI_HALF_TN_INDEX",
        1,
        "candidate stage undef",
    )?;
    Ok(source)
}

pub fn restore_retained_source(candidate: &str, production: &str) -> Result<String, String> {
    let retained = retained_source(production)?;
    let expected = candidate_source(production)?;
    if candidate != expected {
        return Err("BK32/S3 candidate differs from the fail-closed transform".into());
    }
    Ok(retained)
}

const CANDIDATE_SCHEDULE: &str = r#"    int macro64_tiles = (M_red + 63) / 64;
    int num_m_tiles = 2 * macro64_tiles;
    for (int preload = 0; preload < GEMM_BI_TC64_STAGES - 1; ++preload) {
        if (preload < num_m_tiles) {
            if (fast_stage) {
                GEMM_BI_TC64_STAGE_TN_ASYNC(
                    preload, preload * GEMM_BI_TC64_BK);
            } else {
                GEMM_BI_TC64_STAGE_TN_SCALAR(
                    preload, preload * GEMM_BI_TC64_BK, T_ACT, FROM_F);
            }
        } else if (fast_stage) {
            asm volatile("cp.async.commit_group;\n");
        }
    }
    for (int mt = 0; mt < num_m_tiles; mt++) {
        if (fast_stage) {
            asm volatile("cp.async.wait_group 1;\n");
        }
        __syncthreads();
        int next = mt + GEMM_BI_TC64_STAGES - 1;
        if (next < num_m_tiles) {
            int write_buf = next % GEMM_BI_TC64_STAGES;
            if (fast_stage) {
                GEMM_BI_TC64_STAGE_TN_ASYNC(
                    write_buf, next * GEMM_BI_TC64_BK);
            } else {
                GEMM_BI_TC64_STAGE_TN_SCALAR(
                    write_buf, next * GEMM_BI_TC64_BK, T_ACT, FROM_F);
            }
        } else if (fast_stage) {
            asm volatile("cp.async.commit_group;\n");
        }
        int read_buf = mt % GEMM_BI_TC64_STAGES;
"#;

fn encode_macro_block(decoded: &str) -> String {
    decoded.lines().map(|line| format!("{line} \\\n")).collect()
}

fn unique_position(source: &str, needle: &str, label: &str) -> Result<usize, String> {
    require_count(source, needle, 1, label)?;
    Ok(source.find(needle).unwrap())
}

fn require_count(source: &str, needle: &str, expected: usize, label: &str) -> Result<(), String> {
    let actual = source.matches(needle).count();
    if actual != expected {
        return Err(format!(
            "BK32/S3 {label} anchor expected {expected}, observed {actual}: {needle:?}"
        ));
    }
    Ok(())
}

fn replace_exact(
    source: &mut String,
    before: &str,
    after: &str,
    expected: usize,
    label: &str,
) -> Result<(), String> {
    require_count(source, before, expected, label)?;
    *source = source.replacen(before, after, expected);
    Ok(())
}

fn fnv64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

pub fn all_strata_below(strata: &[[f64; 2]], threshold: f64) -> bool {
    strata.len() == 4
        && threshold.is_finite()
        && threshold > 0.0
        && strata.iter().all(|[p50, p95]| {
            p50.is_finite()
                && p95.is_finite()
                && *p50 > 0.0
                && *p95 > 0.0
                && *p50 < threshold
                && *p95 < threshold
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    const PRODUCTION: &str = include_str!("../../kernels/gemm_bi_triad/sm80.cu");

    #[test]
    fn candidate_is_compact_bk32_s3_and_reversible_to_frozen_parent() {
        let retained = retained_source(PRODUCTION).unwrap();
        let candidate = candidate_source(PRODUCTION).unwrap();
        assert!(candidate.contains("#define GEMM_BI_TC64_BK 32"));
        assert!(candidate.contains("#define GEMM_BI_TC64_STAGES 3"));
        assert!(candidate.contains("Xs[GEMM_BI_TC64_STAGES]"));
        assert!(candidate.contains("Ys[GEMM_BI_TC64_STAGES]"));
        assert!(candidate.contains("int macro64_tiles = (M_red + 63) / 64;"));
        assert!(candidate.contains("int num_m_tiles = 2 * macro64_tiles;"));
        assert!(candidate.contains("cp.async.wait_group 1;"));
        assert!(candidate.contains("int next = mt + GEMM_BI_TC64_STAGES - 1;"));
        assert!(candidate.contains("int read_buf = mt % GEMM_BI_TC64_STAGES;"));
        assert_eq!(
            restore_retained_source(&candidate, PRODUCTION).unwrap(),
            retained
        );
    }

    #[test]
    fn bk32_preserves_the_parent_k16_reduction_order() {
        for reduction in [
            0usize, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 67, 95, 96, 127, 128, 129, 2_048,
        ] {
            let parent = (0..reduction.div_ceil(64))
                .flat_map(|tile| (0..4).map(move |issue| tile * 64 + issue * 16))
                .collect::<Vec<_>>();
            let candidate = (0..2 * reduction.div_ceil(64))
                .flat_map(|tile| (0..2).map(move |issue| tile * 32 + issue * 16))
                .collect::<Vec<_>>();
            assert_eq!(candidate, parent, "reduction={reduction}");
        }
    }

    #[test]
    fn bk32_stage_copy_ownership_and_three_slot_ring_are_disjoint() {
        let mut owners = [0_u8; 32 * 8];
        for thread in 0..128 {
            for iteration in 0..2 {
                let linear = thread + iteration * 128;
                let row = linear / 8;
                let vector = linear % 8;
                owners[row * 8 + vector] += 1;
            }
        }
        assert!(owners.into_iter().all(|count| count == 1));
        for tile in 0..128 {
            let read = tile % 3;
            let pending = (tile + 1) % 3;
            let write = (tile + 2) % 3;
            assert_ne!(read, pending);
            assert_ne!(read, write);
            assert_ne!(pending, write);
        }
        assert_eq!(3 * 2 * 32 * 64 * 2, STATIC_SHARED_BYTES);
    }

    #[test]
    fn retained_parent_hash_is_frozen() {
        let retained = retained_source(PRODUCTION).unwrap();
        assert_eq!(fnv64(retained.as_bytes()), RETAINED_SOURCE_FNV64);
    }

    #[test]
    fn source_contract_and_screen_policy_are_frozen() {
        assert_eq!(TARGET, (2_048, 768, 3_072));
        assert_eq!(TARGET_GRID, 576);
        assert_eq!(BLOCK_THREADS, 128);
        assert_eq!(STATIC_SHARED_BYTES, 24_576);
        assert_eq!(REQUIRED_OCCUPANCY, 4);
        assert_eq!(REGISTER_CAP, 128);
        assert_eq!(RETAINED_STATIC_SHARED_BYTES, 32_768);
        assert_eq!(RETAINED_REQUIRED_OCCUPANCY, 3);
        assert!(all_strata_below(&[[0.98, 0.989]; 4], 0.99));
        assert!(!all_strata_below(&[[0.98, 0.99]; 4], 0.99));
    }
}
