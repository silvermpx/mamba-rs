#[path = "triad_half_tn_vec2_epilogue_source.rs"]
mod retained;

pub const SYMBOL_PREFIX: &str = "gemm_bi_tn_test_m96n64_bk64_s2_regpipe_vec2_";
pub const RETAINED_SYMBOL_PREFIX: &str = retained::SYMBOL_PREFIX;
pub const TARGET: (usize, usize, usize) = (2_048, 768, 3_072);
pub const TARGET_GRID: u32 = 384;
pub const BLOCK_THREADS: u32 = 192;
pub const STATIC_SHARED_BYTES: i32 = 49_152;
pub const REQUIRED_OCCUPANCY: u32 = 2;
pub const REGISTER_CAP: i32 = 128;
pub const EXPECTED_HMMA: usize = 32;
pub const EXPECTED_LDSM: usize = 24;
pub const MAX_LDGSTS: usize = 24;
pub const RETAINED_BLOCK_THREADS: u32 = 128;
pub const RETAINED_STATIC_SHARED_BYTES: i32 = 32_768;
pub const RETAINED_REQUIRED_OCCUPANCY: u32 = 3;
pub const RETAINED_REGISTER_CAP: i32 = 128;
pub const RETAINED_EXPECTED_HMMA: usize = 32;
pub const RETAINED_EXPECTED_LDSM: usize = 24;
pub const RETAINED_EXPECTED_LDGSTS: usize = 24;

pub fn retained_source(production: &str) -> Result<String, String> {
    retained::candidate_source(production)
}

pub fn candidate_source(production: &str) -> Result<String, String> {
    let mut source = retained_source(production)?;
    replace_exact(
        &mut source,
        "#define GEMM_BI_TC64_BM 64\n#define GEMM_BI_TC64_BN 64\n#define GEMM_BI_TC64_BK 64\n#define GEMM_BI_TC64_THREADS 128\n#define GEMM_BI_TC64_LDB 64\n#define GEMM_BI_HALF_TN_INDEX(row, col) ((row) * 64 + ((col) ^ (((row) & 7) * 8)))",
        "#define GEMM_BI_TC64_BM 96\n#define GEMM_BI_TC64_BN 64\n#define GEMM_BI_TC64_BK 64\n#define GEMM_BI_TC64_THREADS 192\n#define GEMM_BI_HALF_TN_X_LDB 128\n#define GEMM_BI_HALF_TN_Y_LDB 64\n#define GEMM_BI_HALF_TN_X_INDEX(row, col) ((row) * GEMM_BI_HALF_TN_X_LDB + ((col) ^ (((row) & 7) * 8)))\n#define GEMM_BI_HALF_TN_Y_INDEX(row, col) ((row) * GEMM_BI_HALF_TN_Y_LDB + ((col) ^ (((row) & 7) * 8)))",
        1,
        "geometry definitions",
    )?;
    for (before, after, label) in [
        (
            "Xs_sbase + (unsigned)((buf) * GEMM_BI_TC64_BK * GEMM_BI_TC64_LDB * 2)",
            "Xs_sbase + (unsigned)((buf) * GEMM_BI_TC64_BK * GEMM_BI_HALF_TN_X_LDB * 2)",
            "async X stage stride",
        ),
        (
            "Ys_sbase + (unsigned)((buf) * GEMM_BI_TC64_BK * GEMM_BI_TC64_LDB * 2)",
            "Ys_sbase + (unsigned)((buf) * GEMM_BI_TC64_BK * GEMM_BI_HALF_TN_Y_LDB * 2)",
            "async Y stage stride",
        ),
        (
            "_xs + (unsigned)((GEMM_BI_HALF_TN_INDEX(_r, _c)) * 2)",
            "_xs + (unsigned)((GEMM_BI_HALF_TN_X_INDEX(_r, _c)) * 2)",
            "async X destination",
        ),
        (
            "_ys + (unsigned)((GEMM_BI_HALF_TN_INDEX(_r, _c)) * 2)",
            "_ys + (unsigned)((GEMM_BI_HALF_TN_Y_INDEX(_r, _c)) * 2)",
            "async Y destination",
        ),
        (
            "_xs[GEMM_BI_HALF_TN_INDEX(_r, _c)]",
            "_xs[GEMM_BI_HALF_TN_X_INDEX(_r, _c)]",
            "scalar X destination",
        ),
        (
            "_ys[GEMM_BI_HALF_TN_INDEX(_r, _c)]",
            "_ys[GEMM_BI_HALF_TN_Y_INDEX(_r, _c)]",
            "scalar Y destination",
        ),
        (
            "T_ACT Xs[2][GEMM_BI_TC64_BK][GEMM_BI_TC64_LDB]",
            "T_ACT Xs[2][GEMM_BI_TC64_BK][GEMM_BI_HALF_TN_X_LDB]",
            "X shared allocation",
        ),
        (
            "T_ACT Ys[2][GEMM_BI_TC64_BK][GEMM_BI_TC64_LDB]",
            "T_ACT Ys[2][GEMM_BI_TC64_BK][GEMM_BI_HALF_TN_Y_LDB]",
            "Y shared allocation",
        ),
        (
            "Xs_sbase + (unsigned)(read_buf * GEMM_BI_TC64_BK * GEMM_BI_TC64_LDB * 2)",
            "Xs_sbase + (unsigned)(read_buf * GEMM_BI_TC64_BK * GEMM_BI_HALF_TN_X_LDB * 2)",
            "read X stage stride",
        ),
        (
            "Ys_sbase + (unsigned)(read_buf * GEMM_BI_TC64_BK * GEMM_BI_TC64_LDB * 2)",
            "Ys_sbase + (unsigned)(read_buf * GEMM_BI_TC64_BK * GEMM_BI_HALF_TN_Y_LDB * 2)",
            "read Y stage stride",
        ),
        (RETAINED_SYMBOL_PREFIX, SYMBOL_PREFIX, "candidate export"),
        (
            "#undef GEMM_BI_HALF_TN_INDEX",
            "#undef GEMM_BI_HALF_TN_X_INDEX\n#undef GEMM_BI_HALF_TN_Y_INDEX\n#undef GEMM_BI_HALF_TN_X_LDB\n#undef GEMM_BI_HALF_TN_Y_LDB",
            "candidate layout undef",
        ),
    ] {
        replace_exact(&mut source, before, after, 1, label)?;
    }
    replace_exact(
        &mut source,
        "GEMM_BI_HALF_TN_INDEX(srow, scol)",
        "GEMM_BI_HALF_TN_X_INDEX(srow, scol)",
        2,
        "X fragment layout",
    )?;
    replace_exact(
        &mut source,
        "GEMM_BI_HALF_TN_INDEX(srow, warpN + fn * 8)",
        "GEMM_BI_HALF_TN_Y_INDEX(srow, warpN + fn * 8)",
        2,
        "Y fragment layout",
    )?;
    if source.contains("GEMM_BI_TC64_LDB") || source.contains("GEMM_BI_HALF_TN_INDEX(") {
        return Err("M96N64 transform retained a square shared-layout anchor".into());
    }
    Ok(source)
}

pub fn restore_retained_source(candidate: &str, production: &str) -> Result<String, String> {
    let expected = candidate_source(production)?;
    if candidate != expected {
        return Err("M96N64 candidate differs from the fail-closed transform".into());
    }
    retained_source(production)
}

pub const fn target_grid() -> u32 {
    TARGET.1.div_ceil(96) as u32 * TARGET.2.div_ceil(64) as u32
}

pub const fn target_staged_half_elements_m96n64() -> usize {
    target_grid() as usize * (96 + 64) * 64
}

pub const fn target_staged_half_elements_m64n64() -> usize {
    TARGET.1.div_ceil(64) * TARGET.2.div_ceil(64) * (64 + 64) * 64
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
            "M96N64 {label} anchor expected {expected}, observed {actual}: {before:?}"
        ));
    }
    *source = source.replacen(before, after, expected);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const PRODUCTION: &str = include_str!("../../kernels/gemm_bi_triad/sm80.cu");

    const fn x_offset(row: usize, column: usize) -> usize {
        row * 128 + (column ^ ((row & 7) * 8))
    }

    const fn y_offset(row: usize, column: usize) -> usize {
        row * 64 + (column ^ ((row & 7) * 8))
    }

    #[test]
    fn six_warps_own_every_m96n64_output_once() {
        let mut owners = vec![0u8; 96 * 64];
        for warp in 0..6 {
            let warp_m = (warp / 2) * 32;
            let warp_n = (warp % 2) * 32;
            for row in warp_m..warp_m + 32 {
                for column in warp_n..warp_n + 32 {
                    owners[row * 64 + column] += 1;
                }
            }
        }
        assert!(owners.into_iter().all(|count| count == 1));
    }

    #[test]
    fn padded_x_and_compact_y_layouts_have_unique_copy_ownership() {
        for (width, stride, offset) in [
            (96usize, 128usize, x_offset as fn(usize, usize) -> usize),
            (64usize, 64usize, y_offset as fn(usize, usize) -> usize),
        ] {
            let mut owners = vec![0u8; 2 * 64 * stride];
            for stage in 0..2 {
                for thread in 0..192 {
                    for linear in (thread..64 * (width / 8)).step_by(192) {
                        let row = linear / (width / 8);
                        let column = (linear % (width / 8)) * 8;
                        let base = stage * 64 * stride + offset(row, column);
                        assert_eq!(base % 8, 0);
                        for element in 0..8 {
                            owners[base + element] += 1;
                        }
                    }
                }
            }
            assert_eq!(
                owners.iter().filter(|&&count| count == 1).count(),
                2 * 64 * width
            );
            assert!(owners.into_iter().all(|count| count <= 1));
        }
        assert_eq!(2 * 64 * (128 + 64) * 2, STATIC_SHARED_BYTES);
    }

    #[test]
    fn transform_changes_only_cta_geometry_layout_and_export() {
        let retained = retained_source(PRODUCTION).unwrap();
        let candidate = candidate_source(PRODUCTION).unwrap();
        assert!(candidate.contains("#define GEMM_BI_TC64_BM 96"));
        assert!(candidate.contains("#define GEMM_BI_TC64_BN 64"));
        assert!(candidate.contains("#define GEMM_BI_TC64_THREADS 192"));
        assert!(candidate.contains("int warpM = (warp / 2) * 32;"));
        assert!(candidate.contains("int warpN = (warp % 2) * 32;"));
        assert!(candidate.contains("float acc[2][4][4];"));
        assert!(candidate.contains("unsigned a_frag[2][2][4];"));
        assert!(candidate.contains("unsigned b_frag[2][4][2];"));
        assert!(candidate.contains("a_frag[(ks + 1) & 1][fm]"));
        assert!(candidate.contains("gemm_bi_accumulate_float2_or_scalar("));
        assert_eq!(
            restore_retained_source(&candidate, PRODUCTION).unwrap(),
            retained
        );
    }

    #[test]
    fn target_reduces_staged_traffic_without_losing_resident_warps() {
        assert_eq!(target_grid(), TARGET_GRID);
        assert_eq!(target_staged_half_elements_m96n64(), 3_932_160);
        assert_eq!(target_staged_half_elements_m64n64(), 4_718_592);
        assert_eq!(
            target_staged_half_elements_m96n64() * 6,
            target_staged_half_elements_m64n64() * 5
        );
        assert_eq!(BLOCK_THREADS / 32 * REQUIRED_OCCUPANCY, 12);
        assert_eq!(
            RETAINED_BLOCK_THREADS / 32 * RETAINED_REQUIRED_OCCUPANCY,
            12
        );
    }

    #[test]
    fn malformed_parent_fails_closed() {
        assert!(candidate_source("").is_err());
        let duplicate = format!("{PRODUCTION}\n#define GEMM_BI_TC64_STAGE_TN_ASYNC\n");
        assert!(candidate_source(&duplicate).is_err());
    }
}
