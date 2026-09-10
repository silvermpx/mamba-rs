#[path = "triad_half_tn_vec2_epilogue_source.rs"]
mod retained;

pub const SYMBOL_PREFIX: &str = "gemm_bi_tn_test_one_wave_atlas_bk64_s2_regpipe_vec2_";
pub const RETAINED_SYMBOL_PREFIX: &str = retained::SYMBOL_PREFIX;
pub const TARGET: (usize, usize, usize) = (2_048, 1_536, 768);
pub const TARGET_GRID: u32 = 142;
pub const BLOCK_THREADS: u32 = 288;
pub const STATIC_SHARED_BYTES: i32 = 0;
pub const DYNAMIC_SHARED_BYTES: u32 = 65_536;
pub const REQUIRED_OCCUPANCY: u32 = 1;
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AtlasTile {
    pub m_base: usize,
    pub m_extent: usize,
    pub n_base: usize,
}

pub const fn atlas_tile(block: u32) -> Option<AtlasTile> {
    if block >= TARGET_GRID {
        return None;
    }
    let (n_tile, local, large_ctas) = if block < 34 {
        (block / 17, block % 17, 14)
    } else {
        let rest = block - 34;
        (2 + rest / 18, rest % 18, 12)
    };
    let (m_base, m_extent) = if local < large_ctas {
        ((local * 96) as usize, 96)
    } else {
        ((large_ctas * 96 + (local - large_ctas) * 64) as usize, 64)
    };
    Some(AtlasTile {
        m_base,
        m_extent,
        n_base: (n_tile * 96) as usize,
    })
}

pub const fn target_active_warps() -> usize {
    100 * 9 + 42 * 6
}

pub const fn target_staged_half_elements() -> usize {
    100 * (96 + 96) * 64 + 42 * (64 + 96) * 64
}

pub const fn retained_staged_half_elements() -> usize {
    288 * (64 + 64) * 64
}

pub fn retained_source(production: &str) -> Result<String, String> {
    retained::candidate_source(production)
}

pub fn candidate_source(production: &str) -> Result<String, String> {
    let mut source = retained_source(production)?;
    replace_exact(
        &mut source,
        "#define GEMM_BI_TC64_BM 64\n#define GEMM_BI_TC64_BN 64\n#define GEMM_BI_TC64_BK 64\n#define GEMM_BI_TC64_THREADS 128\n#define GEMM_BI_TC64_LDB 64\n#define GEMM_BI_HALF_TN_INDEX(row, col) ((row) * 64 + ((col) ^ (((row) & 7) * 8)))",
        "#define GEMM_BI_TC64_BM 96\n#define GEMM_BI_TC64_BN 96\n#define GEMM_BI_TC64_BK 64\n#define GEMM_BI_TC64_THREADS 288\n#define GEMM_BI_HALF_TN_X_LDB 128\n#define GEMM_BI_HALF_TN_Y_LDB 128\n#define GEMM_BI_HALF_TN_X_INDEX(row, col) ((row) * GEMM_BI_HALF_TN_X_LDB + ((col) ^ (((row) & 7) * 8)))\n#define GEMM_BI_HALF_TN_Y_INDEX(row, col) ((row) * GEMM_BI_HALF_TN_Y_LDB + ((col) ^ (((row) & 7) * 8)))",
        1,
        "geometry definitions",
    )?;
    for (before, after, expected, label) in [
        (
            "Xs_sbase + (unsigned)((buf) * GEMM_BI_TC64_BK * GEMM_BI_TC64_LDB * 2)",
            "Xs_sbase + (unsigned)((buf) * GEMM_BI_TC64_BK * GEMM_BI_HALF_TN_X_LDB * 2)",
            1,
            "async X stage stride",
        ),
        (
            "Ys_sbase + (unsigned)((buf) * GEMM_BI_TC64_BK * GEMM_BI_TC64_LDB * 2)",
            "Ys_sbase + (unsigned)((buf) * GEMM_BI_TC64_BK * GEMM_BI_HALF_TN_Y_LDB * 2)",
            1,
            "async Y stage stride",
        ),
        (
            "_xs + (unsigned)((GEMM_BI_HALF_TN_INDEX(_r, _c)) * 2)",
            "_xs + (unsigned)((GEMM_BI_HALF_TN_X_INDEX(_r, _c)) * 2)",
            1,
            "async X destination",
        ),
        (
            "_ys + (unsigned)((GEMM_BI_HALF_TN_INDEX(_r, _c)) * 2)",
            "_ys + (unsigned)((GEMM_BI_HALF_TN_Y_INDEX(_r, _c)) * 2)",
            1,
            "async Y destination",
        ),
        (
            "_xs[GEMM_BI_HALF_TN_INDEX(_r, _c)]",
            "_xs[GEMM_BI_HALF_TN_X_INDEX(_r, _c)]",
            1,
            "scalar X destination",
        ),
        (
            "_ys[GEMM_BI_HALF_TN_INDEX(_r, _c)]",
            "_ys[GEMM_BI_HALF_TN_Y_INDEX(_r, _c)]",
            1,
            "scalar Y destination",
        ),
        (
            "int _gk = pid_m * GEMM_BI_TC64_BM + _c;",
            "int _gk = tile_m_base + _c;",
            2,
            "atlas X global row",
        ),
        (
            "(_gm < M_red && _gk < K_out)",
            "(_gm < M_red && _gk < K_out && _c < tile_m)",
            1,
            "scalar short-tile X zero fill",
        ),
        (
            "gemm_bi_cp_async_valid_elems(_gm < M_red, K_out, _gk)",
            "gemm_bi_cp_async_valid_elems(_gm < M_red && _c < tile_m, K_out, _gk)",
            1,
            "async short-tile X zero fill",
        ),
        (
            "int warpM = (warp / 2) * 32;",
            "int warpM = (warp / 3) * 32;",
            1,
            "three warp columns",
        ),
        (
            "int warpN = (warp % 2) * 32;",
            "int warpN = (warp % 3) * 32;",
            1,
            "three warp columns",
        ),
        (
            "Xs_sbase + (unsigned)(read_buf * GEMM_BI_TC64_BK * GEMM_BI_TC64_LDB * 2)",
            "Xs_sbase + (unsigned)(read_buf * GEMM_BI_TC64_BK * GEMM_BI_HALF_TN_X_LDB * 2)",
            1,
            "read X stage stride",
        ),
        (
            "Ys_sbase + (unsigned)(read_buf * GEMM_BI_TC64_BK * GEMM_BI_TC64_LDB * 2)",
            "Ys_sbase + (unsigned)(read_buf * GEMM_BI_TC64_BK * GEMM_BI_HALF_TN_Y_LDB * 2)",
            1,
            "read Y stage stride",
        ),
        (
            "pid_m * GEMM_BI_TC64_BM + warpM + fm * 16 + g",
            "tile_m_base + warpM + fm * 16 + g",
            1,
            "atlas epilogue row",
        ),
        (RETAINED_SYMBOL_PREFIX, SYMBOL_PREFIX, 1, "candidate export"),
        (
            "#undef GEMM_BI_HALF_TN_INDEX",
            "#undef GEMM_BI_HALF_TN_X_INDEX\n#undef GEMM_BI_HALF_TN_Y_INDEX\n#undef GEMM_BI_HALF_TN_X_LDB\n#undef GEMM_BI_HALF_TN_Y_LDB",
            1,
            "candidate layout undef",
        ),
    ] {
        replace_exact(&mut source, before, after, expected, label)?;
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
    replace_exact(
        &mut source,
        "    __shared__ __align__(16) T_ACT Xs[2][GEMM_BI_TC64_BK][GEMM_BI_TC64_LDB];           \\\n    __shared__ __align__(16) T_ACT Ys[2][GEMM_BI_TC64_BK][GEMM_BI_TC64_LDB];           \\",
        "    extern __shared__ __align__(16) T_ACT atlas_smem_##SUFFIX[];                \\\n    T_ACT (*Xs)[GEMM_BI_TC64_BK][GEMM_BI_HALF_TN_X_LDB] =                     \\\n        reinterpret_cast<T_ACT (*)[GEMM_BI_TC64_BK][GEMM_BI_HALF_TN_X_LDB]>(atlas_smem_##SUFFIX); \\\n    T_ACT (*Ys)[GEMM_BI_TC64_BK][GEMM_BI_HALF_TN_Y_LDB] =                     \\\n        reinterpret_cast<T_ACT (*)[GEMM_BI_TC64_BK][GEMM_BI_HALF_TN_Y_LDB]>(  \\\n            atlas_smem_##SUFFIX + 2 * GEMM_BI_TC64_BK * GEMM_BI_HALF_TN_X_LDB); \\",
        1,
        "dynamic shared allocation",
    )?;
    replace_exact(
        &mut source,
        "    int num_pid_n = (N + GEMM_BI_TC64_BN - 1) / GEMM_BI_TC64_BN;                       \\\n    int pid_m = blockIdx.x / num_pid_n;                                        \\\n    int pid_n = blockIdx.x % num_pid_n;                                        \\",
        ATLAS_MAPPING,
        1,
        "target atlas mapping",
    )?;
    replace_exact(
        &mut source,
        "        unsigned a_frag[2][2][4];",
        "        if (warpM < tile_m) {                                         \\\n        unsigned a_frag[2][2][4];",
        1,
        "active compute open",
    )?;
    replace_exact(
        &mut source,
        "        read_buf ^= 1;",
        "        }                                                             \\\n        read_buf ^= 1;",
        1,
        "active compute close",
    )?;
    replace_exact(
        &mut source,
        "    /* epilogue: paired f32 accumulate into dW, scalar at the N tail */",
        "    if (warpM >= tile_m) return;                                      \\\n    /* epilogue: paired f32 accumulate into dW, scalar at the N tail */",
        1,
        "inactive warp store guard",
    )?;
    if source.contains("pid_m")
        || source.contains("num_pid_n")
        || source.contains("GEMM_BI_TC64_LDB")
        || source.contains("GEMM_BI_HALF_TN_INDEX(")
    {
        return Err("one-wave atlas retained a square or linear mapping anchor".into());
    }
    Ok(source)
}

const ATLAS_MAPPING: &str = r#"    if (M_red != 2048 || K_out != 1536 || N != 768 || gridDim.x != 142) return; \
    int atlas_id = (int)blockIdx.x;                                            \
    int pid_n, atlas_local, large_ctas;                                        \
    if (atlas_id < 34) {                                                       \
        pid_n = atlas_id / 17;                                                 \
        atlas_local = atlas_id % 17;                                           \
        large_ctas = 14;                                                       \
    } else {                                                                   \
        int atlas_rest = atlas_id - 34;                                        \
        pid_n = 2 + atlas_rest / 18;                                           \
        atlas_local = atlas_rest % 18;                                         \
        large_ctas = 12;                                                       \
    }                                                                          \
    int tile_m = atlas_local < large_ctas ? 96 : 64;                           \
    int tile_m_base = atlas_local < large_ctas                                \
        ? atlas_local * 96                                                     \
        : large_ctas * 96 + (atlas_local - large_ctas) * 64;                  \"#;

pub fn restore_retained_source(candidate: &str, production: &str) -> Result<String, String> {
    let expected = candidate_source(production)?;
    if candidate != expected {
        return Err("one-wave atlas candidate differs from the fail-closed transform".into());
    }
    retained_source(production)
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
            "one-wave atlas {label} anchor expected {expected}, observed {actual}: {before:?}"
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
    fn atlas_covers_target_once_with_exact_declared_mix() {
        let mut owners = vec![0u8; TARGET.1 * TARGET.2];
        let mut large = 0;
        let mut short = 0;
        for block in 0..TARGET_GRID {
            let tile = atlas_tile(block).unwrap();
            match tile.m_extent {
                96 => large += 1,
                64 => short += 1,
                other => panic!("unexpected atlas height {other}"),
            }
            for row in tile.m_base..tile.m_base + tile.m_extent {
                for column in tile.n_base..tile.n_base + 96 {
                    owners[row * TARGET.2 + column] += 1;
                }
            }
        }
        assert_eq!((large, short), (100, 42));
        assert!(owners.into_iter().all(|count| count == 1));
        assert_eq!(atlas_tile(TARGET_GRID), None);
    }

    #[test]
    fn atlas_preserves_compute_warps_and_reduces_logical_staging() {
        assert_eq!(target_active_warps(), 1_152);
        assert_eq!(288 * 4, 1_152);
        assert_eq!(target_staged_half_elements(), 1_658_880);
        assert_eq!(retained_staged_half_elements(), 2_359_296);
        assert_eq!(
            target_staged_half_elements() * 64,
            retained_staged_half_elements() * 45
        );
    }

    #[test]
    fn source_uses_dynamic_shared_atlas_and_masks_only_short_compute_row() {
        let retained = retained_source(PRODUCTION).unwrap();
        let candidate = candidate_source(PRODUCTION).unwrap();
        assert!(candidate.contains("#define GEMM_BI_TC64_BM 96"));
        assert!(candidate.contains("#define GEMM_BI_TC64_BN 96"));
        assert!(candidate.contains("#define GEMM_BI_TC64_THREADS 288"));
        assert!(candidate.contains("extern __shared__ __align__(16) T_ACT atlas_smem_##SUFFIX[];"));
        assert!(candidate.contains("if (M_red != 2048 || K_out != 1536 || N != 768"));
        assert!(candidate.contains("if (warpM < tile_m)"));
        assert!(candidate.contains("if (warpM >= tile_m) return;"));
        assert!(candidate.contains("float acc[2][4][4];"));
        assert!(candidate.contains("unsigned a_frag[2][2][4];"));
        assert!(candidate.contains("a_frag[(ks + 1) & 1][fm]"));
        assert!(candidate.contains("gemm_bi_accumulate_float2_or_scalar("));
        assert_eq!(candidate.matches("_c < tile_m").count(), 2);
        assert_eq!(
            candidate
                .matches("mma.sync.aligned.m16n8k16.row.col.f32.")
                .count(),
            retained
                .matches("mma.sync.aligned.m16n8k16.row.col.f32.")
                .count()
        );
        assert_eq!(candidate.matches("read_buf ^= 1;").count(), 1);
        assert!(
            candidate.find("if (warpM < tile_m)").unwrap()
                < candidate.find("read_buf ^= 1;").unwrap()
        );
        assert!(
            candidate.find("read_buf ^= 1;").unwrap()
                < candidate.find("if (warpM >= tile_m) return;").unwrap()
        );
        assert_eq!(
            restore_retained_source(&candidate, PRODUCTION).unwrap(),
            retained
        );
    }

    #[test]
    fn malformed_parent_fails_closed() {
        assert!(candidate_source("").is_err());
        let duplicate = format!("{PRODUCTION}\n#define GEMM_BI_TC64_STAGE_TN_ASYNC\n");
        assert!(candidate_source(&duplicate).is_err());
    }

    #[test]
    fn bf16_and_f16_instantiations_token_paste_distinct_shared_identifiers() {
        let candidate = candidate_source(PRODUCTION).unwrap();
        assert_eq!(candidate.matches("atlas_smem_##SUFFIX").count(), 3);
        assert!(!candidate.contains(" T_ACT atlas_smem[];"));
        assert!(candidate.contains(
            "GEMM_BI_DEFINE_GEMM_BI_TN_TC64(bf16, __nv_bfloat16, from_f_bf16, \"bf16\")"
        ));
        assert!(
            candidate.contains(
                "GEMM_BI_DEFINE_GEMM_BI_TN_TC64(f16,  __half,        from_f_f16,  \"f16\")"
            )
        );
        assert_ne!("atlas_smem_bf16", "atlas_smem_f16");
    }
}
