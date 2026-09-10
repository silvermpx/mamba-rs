#[path = "triad_half_tn_vec2_epilogue_source.rs"]
mod retained;

pub const SYMBOL_PREFIX: &str = "gemm_bi_tn_test_tc64_bk64_s2_regpipe_vec2_raster4_";
pub const RETAINED_SYMBOL_PREFIX: &str = retained::SYMBOL_PREFIX;
pub const TARGET: (usize, usize, usize) = (2_048, 768, 3_072);
pub const TARGET_GRID: (u32, u32, u32) = (4, 48, 3);
pub const TARGET_TILE_COUNT: u32 = TARGET_GRID.0 * TARGET_GRID.1 * TARGET_GRID.2;
pub const BLOCK_THREADS: u32 = 128;
pub const STATIC_SHARED_BYTES: i32 = 32_768;
pub const REQUIRED_OCCUPANCY: u32 = 3;
pub const REGISTER_CAP: i32 = 128;
pub const EXPECTED_HMMA: usize = 32;
pub const EXPECTED_LDSM: usize = 24;
pub const EXPECTED_LDGSTS: usize = 24;
pub const MAX_LDGSTS: usize = EXPECTED_LDGSTS;
pub const RETAINED_BLOCK_THREADS: u32 = BLOCK_THREADS;
pub const RETAINED_STATIC_SHARED_BYTES: i32 = STATIC_SHARED_BYTES;
pub const RETAINED_REQUIRED_OCCUPANCY: u32 = REQUIRED_OCCUPANCY;
pub const RETAINED_REGISTER_CAP: i32 = REGISTER_CAP;
pub const RETAINED_EXPECTED_HMMA: usize = EXPECTED_HMMA;
pub const RETAINED_EXPECTED_LDSM: usize = EXPECTED_LDSM;
pub const RETAINED_EXPECTED_LDGSTS: usize = EXPECTED_LDGSTS;
pub const SCOUT_WINDOWS: usize = 3;
pub const QUALIFICATION_WINDOWS: usize = 7;

const RETAINED_MAPPER: &str = r#"    int num_pid_n = (N + GEMM_BI_TC64_BN - 1) / GEMM_BI_TC64_BN;                       \
    int pid_m = blockIdx.x / num_pid_n;                                        \
    int pid_n = blockIdx.x % num_pid_n;                                        \"#;

const PACKED_MAPPER: &str = r#"    bool packed_raster = M_red == 2048 && K_out == 768 && N == 3072;           \
    int num_pid_n = (N + GEMM_BI_TC64_BN - 1) / GEMM_BI_TC64_BN;                       \
    int pid_m = packed_raster ? (int)blockIdx.x + 4 * (int)blockIdx.z           \
                              : (int)blockIdx.x / num_pid_n;                    \
    int pid_n = packed_raster ? (int)blockIdx.y                                 \
                              : (int)blockIdx.x % num_pid_n;                    \"#;

pub fn retained_source(production: &str) -> Result<String, String> {
    retained::candidate_source(production)
}

/// Adds a target-only packed 4-M-by-1-N CTA raster to the retained M64N64
/// kernel. Arithmetic, shared layout, mainloop and epilogue remain unchanged.
pub fn candidate_source(production: &str) -> Result<String, String> {
    let mut source = retained_source(production)?;
    replace_exact(
        &mut source,
        RETAINED_MAPPER,
        PACKED_MAPPER,
        1,
        "tile mapper",
    )?;
    replace_exact(
        &mut source,
        RETAINED_SYMBOL_PREFIX,
        SYMBOL_PREFIX,
        1,
        "candidate export",
    )?;
    Ok(source)
}

pub const fn packed_tile(block_x: u32, block_y: u32, block_z: u32) -> (u32, u32) {
    (block_x + 4 * block_z, block_y)
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

fn replace_exact(
    source: &mut String,
    before: &str,
    after: &str,
    expected: usize,
    label: &str,
) -> Result<(), String> {
    let actual = source.matches(before).count();
    if actual != expected {
        return Err(format!(
            "packed-raster {label} anchor expected {expected}, observed {actual}: {before:?}"
        ));
    }
    *source = source.replacen(before, after, expected);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const PRODUCTION: &str = include_str!("../../kernels/gemm_bi_triad/sm80.cu");

    #[test]
    fn packed_grid_covers_every_target_tile_exactly_once() {
        let mut owners = vec![0u8; 12 * 48];
        for z in 0..TARGET_GRID.2 {
            for y in 0..TARGET_GRID.1 {
                for x in 0..TARGET_GRID.0 {
                    let (pid_m, pid_n) = packed_tile(x, y, z);
                    assert!(pid_m < 12);
                    assert!(pid_n < 48);
                    owners[(pid_m * 48 + pid_n) as usize] += 1;
                }
            }
        }
        assert_eq!(owners.len(), TARGET_TILE_COUNT as usize);
        assert!(owners.into_iter().all(|count| count == 1));
    }

    #[test]
    fn source_uses_packed_mapper_only_for_the_exact_target() {
        let source = candidate_source(PRODUCTION).unwrap();
        assert!(source.contains("M_red == 2048 && K_out == 768 && N == 3072"));
        assert!(source.contains("blockIdx.x + 4 * (int)blockIdx.z"));
        assert!(source.contains("packed_raster ? (int)blockIdx.y"));
        assert!(source.contains(": (int)blockIdx.x / num_pid_n"));
        assert!(source.contains(": (int)blockIdx.x % num_pid_n"));
        assert!(source.contains("void gemm_bi_tn_test_tc64_bk64_s2_regpipe_vec2_raster4_##SUFFIX"));
    }

    #[test]
    fn mapper_and_export_are_the_only_source_changes() {
        let incumbent = retained_source(PRODUCTION).unwrap();
        let normalized = candidate_source(PRODUCTION)
            .unwrap()
            .replace(PACKED_MAPPER, RETAINED_MAPPER)
            .replace(SYMBOL_PREFIX, RETAINED_SYMBOL_PREFIX);
        assert_eq!(normalized, incumbent);
        assert_eq!(
            normalized
                .matches("mma.sync.aligned.m16n8k16.row.col.f32.")
                .count(),
            incumbent
                .matches("mma.sync.aligned.m16n8k16.row.col.f32.")
                .count()
        );
        assert!(normalized.contains("gemm_bi_accumulate_float2_or_scalar("));
    }

    #[test]
    fn malformed_or_ambiguous_parent_fails_closed() {
        assert!(candidate_source("").is_err());
        let mut ambiguous = format!("{RETAINED_MAPPER}\n{RETAINED_MAPPER}");
        assert!(
            replace_exact(
                &mut ambiguous,
                RETAINED_MAPPER,
                PACKED_MAPPER,
                1,
                "tile mapper",
            )
            .is_err()
        );
    }

    #[test]
    fn strict_four_stratum_gate_rejects_equality_and_non_finite_values() {
        assert!(all_strata_below(&[[0.98, 0.989]; 4], 0.99));
        assert!(!all_strata_below(&[[0.98, 0.99]; 4], 0.99));
        assert!(!all_strata_below(&[[0.98, f64::NAN]; 4], 0.99));
        assert!(!all_strata_below(&[[0.98, 0.989]; 3], 0.99));
    }
}
