#[path = "triad_half_tn_vec2_full_tile_stage_source.rs"]
mod parent;

pub const SYMBOL_PREFIX: &str = "gemm_bi_tn_test_tc64_bk64_s2_regpipe_vec2_full_tile_cg_";
pub const RETAINED_SYMBOL_PREFIX: &str = parent::RETAINED_SYMBOL_PREFIX;
pub const BLOCK_THREADS: u32 = parent::BLOCK_THREADS;
pub const STATIC_SHARED_BYTES: i32 = parent::STATIC_SHARED_BYTES;
pub const REQUIRED_OCCUPANCY: u32 = parent::REQUIRED_OCCUPANCY;
pub const REGISTER_CAP: i32 = parent::REGISTER_CAP;
pub const TEXT_RATIO_CAP: f64 = parent::TEXT_RATIO_CAP;
pub const EXPECTED_HMMA: usize = parent::EXPECTED_HMMA;
pub const EXPECTED_LDSM: usize = parent::EXPECTED_LDSM;
pub const EXPECTED_LDGSTS: usize = 48;
pub const TARGET: (usize, usize, usize) = parent::TARGET;
pub const TARGET_GRID: u32 = parent::TARGET_GRID;

pub fn retained_source(production: &str) -> Result<String, String> {
    parent::retained_source(production)
}

pub fn candidate_source(production: &str) -> Result<String, String> {
    let mut source = parent::candidate_source(production)?;
    let range = full_tile_stage_range(&source)?;
    let stage = source[range.clone()].to_owned();
    require_count(&stage, CP_ASYNC_CA, 2, "full-tile .ca copies")?;
    require_count(&stage, CP_ASYNC_CG, 0, "preexisting full-tile .cg copies")?;
    source.replace_range(range, &stage.replace(CP_ASYNC_CA, CP_ASYNC_CG));
    replace_exact(
        &mut source,
        parent::SYMBOL_PREFIX,
        SYMBOL_PREFIX,
        "candidate export",
    )?;
    let changed = full_tile_stage(&source)?;
    require_count(changed, CP_ASYNC_CA, 0, "remaining full-tile .ca copies")?;
    require_count(changed, CP_ASYNC_CG, 2, "full-tile .cg copies")?;
    Ok(source)
}

pub fn restore_parent_source(candidate: &str, production: &str) -> Result<String, String> {
    let expected = candidate_source(production)?;
    if candidate != expected {
        return Err("full-tile-cg candidate differs from fail-closed transform".into());
    }
    Ok(parent::candidate_source(production)?)
}

pub const fn uses_full_tile_cg(reduction: usize, k_out: usize, n: usize) -> bool {
    parent::uses_full_tile_stage(reduction, k_out, n)
}

pub fn all_strata_below(strata: &[[f64; 2]], threshold: f64) -> bool {
    parent::all_strata_below(strata, threshold)
}

const FULL_STAGE_MARKER: &str = "#define GEMM_BI_TC64_STAGE_TN_ASYNC_FULL";
const SCALAR_STAGE_MARKER: &str = "#define GEMM_BI_TC64_STAGE_TN_SCALAR";
const CP_ASYNC_CA: &str = "cp.async.ca.shared.global [%0], [%1], 16;";
const CP_ASYNC_CG: &str = "cp.async.cg.shared.global [%0], [%1], 16;";

fn full_tile_stage_range(source: &str) -> Result<std::ops::Range<usize>, String> {
    require_count(source, FULL_STAGE_MARKER, 1, "full-stage marker")?;
    require_count(source, SCALAR_STAGE_MARKER, 1, "scalar-stage marker")?;
    let start = source.find(FULL_STAGE_MARKER).unwrap();
    let end = source.find(SCALAR_STAGE_MARKER).unwrap();
    if start >= end {
        return Err("full-tile-cg stage boundaries reversed".into());
    }
    Ok(start..end)
}

fn full_tile_stage(source: &str) -> Result<&str, String> {
    Ok(&source[full_tile_stage_range(source)?])
}

fn require_count(source: &str, needle: &str, expected: usize, label: &str) -> Result<(), String> {
    let actual = source.matches(needle).count();
    if actual != expected {
        return Err(format!(
            "full-tile-cg {label} expected {expected}, observed {actual}: {needle:?}"
        ));
    }
    Ok(())
}

fn replace_exact(
    source: &mut String,
    before: &str,
    after: &str,
    label: &str,
) -> Result<(), String> {
    require_count(source, before, 1, label)?;
    *source = source.replacen(before, after, 1);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const PRODUCTION: &str = include_str!("../../kernels/gemm_bi_triad/sm80.cu");

    #[test]
    fn only_target_full_tile_copies_change_from_ca_to_cg() {
        let parent = parent::candidate_source(PRODUCTION).unwrap();
        let candidate = candidate_source(PRODUCTION).unwrap();
        let parent_stage = full_tile_stage(&parent).unwrap();
        let candidate_stage = full_tile_stage(&candidate).unwrap();
        assert_eq!(parent_stage.matches(CP_ASYNC_CA).count(), 2);
        assert_eq!(candidate_stage.matches(CP_ASYNC_CG).count(), 2);
        assert_eq!(candidate_stage.matches(CP_ASYNC_CA).count(), 0);
        assert_eq!(candidate.matches(CP_ASYNC_CG).count(), 2);
        assert_eq!(
            candidate.matches("gemm_bi_cp_async_16_zfill").count(),
            parent.matches("gemm_bi_cp_async_16_zfill").count()
        );
    }

    #[test]
    fn cache_hint_transform_preserves_addresses_math_sync_and_fallback() {
        let parent = parent::candidate_source(PRODUCTION).unwrap();
        let candidate = candidate_source(PRODUCTION).unwrap();
        assert_eq!(
            restore_parent_source(&candidate, PRODUCTION).unwrap(),
            parent
        );
        let normalized = candidate
            .replace(CP_ASYNC_CG, CP_ASYNC_CA)
            .replace(SYMBOL_PREFIX, parent::SYMBOL_PREFIX);
        assert_eq!(normalized, parent);
        for anchor in [
            "GEMM_BI_HALF_TN_INDEX(_r, _c)",
            "A + (long long)_gm * K_out + _gk",
            "B + (long long)_gm * N + _gn",
            "mma.sync.aligned.m16n8k16.row.col.f32.",
            "gemm_bi_accumulate_float2_or_scalar(",
            "cp.async.commit_group;",
            "cp.async.wait_group 0;",
            "__syncthreads();",
            "read_buf ^= 1;",
        ] {
            assert_eq!(
                candidate.matches(anchor).count(),
                parent.matches(anchor).count()
            );
        }
    }

    #[test]
    fn malformed_or_ambiguous_parent_fails_closed() {
        assert!(candidate_source("").is_err());
        let duplicate = PRODUCTION.replacen(
            SCALAR_STAGE_MARKER,
            &format!("{SCALAR_STAGE_MARKER}\n{SCALAR_STAGE_MARKER}"),
            1,
        );
        assert!(candidate_source(&duplicate).is_err());
    }
}
