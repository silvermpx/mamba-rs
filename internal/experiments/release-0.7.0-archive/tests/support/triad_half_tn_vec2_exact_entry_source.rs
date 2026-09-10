#[path = "triad_half_tn_vec2_full_domain_source.rs"]
#[allow(dead_code)]
mod full_domain;

pub const SYMBOL_PREFIX: &str = "gemm_bi_tn_test_tc64_bk64_s2_regpipe_vec2_exact_entry_";
pub const RETAINED_SYMBOL_PREFIX: &str = full_domain::RETAINED_SYMBOL_PREFIX;
pub const BLOCK_THREADS: u32 = 128;
pub const STATIC_SHARED_BYTES: i32 = 32_768;
pub const REQUIRED_OCCUPANCY: u32 = 3;
pub const REGISTER_CAP: i32 = 128;
pub const TEXT_RATIO_CAP: f64 = 1.10;
pub const EXPECTED_HMMA: usize = 32;
pub const EXPECTED_LDSM: usize = 24;
pub const EXPECTED_LDGSTS: usize = 24;
pub const TARGET: (usize, usize, usize) = (2_048, 768, 3_072);
pub const TARGET_GRID: (u32, u32, u32) = (48, 12, 1);
pub const RETAINED_TARGET_GRID: (u32, u32, u32) = (576, 1, 1);

pub fn retained_source(production: &str) -> Result<String, String> {
    full_domain::retained_source(production)
}

pub fn candidate_source(production: &str) -> Result<String, String> {
    let mut source = full_domain::candidate_source(production)?;
    let guarded_stage = stage_block(&source)?.to_owned();
    require_guarded_stage_contract(&guarded_stage)?;
    replace_exact(
        &mut source,
        &guarded_stage,
        EXACT_STAGE_MACRO,
        "guarded stage",
    )?;
    replace_exact(
        &mut source,
        FAST_STAGE_DECLARATION,
        "",
        "fast-stage declaration",
    )?;
    replace_exact(&mut source, RETAINED_PROLOGUE, EXACT_PROLOGUE, "prologue")?;
    replace_exact(&mut source, RETAINED_WAIT, EXACT_WAIT, "wait")?;
    replace_exact(&mut source, RETAINED_REFILL, EXACT_REFILL, "refill")?;
    replace_exact(
        &mut source,
        full_domain::SYMBOL_PREFIX,
        SYMBOL_PREFIX,
        "candidate export",
    )?;
    validate_generated_macros(&source)?;
    Ok(source)
}

pub fn restore_retained_source(candidate: &str, production: &str) -> Result<String, String> {
    let retained = retained_source(production)?;
    let guarded_stage = stage_block(&retained)?.to_owned();
    let mut source = candidate.to_owned();
    replace_exact(
        &mut source,
        SYMBOL_PREFIX,
        full_domain::SYMBOL_PREFIX,
        "restore export",
    )?;
    replace_exact(&mut source, EXACT_REFILL, RETAINED_REFILL, "restore refill")?;
    replace_exact(&mut source, EXACT_WAIT, RETAINED_WAIT, "restore wait")?;
    replace_exact(
        &mut source,
        EXACT_PROLOGUE,
        RETAINED_PROLOGUE,
        "restore prologue",
    )?;
    replace_exact(
        &mut source,
        ACCUMULATOR_DECLARATION,
        &format!("{FAST_STAGE_DECLARATION}{ACCUMULATOR_DECLARATION}"),
        "restore fast-stage declaration",
    )?;
    replace_exact(
        &mut source,
        EXACT_STAGE_MACRO,
        &guarded_stage,
        "restore guarded stage",
    )?;
    full_domain::restore_retained_source(&source)
}

pub const fn is_exact_target(reduction: usize, k_out: usize, n: usize) -> bool {
    reduction == TARGET.0 && k_out == TARGET.1 && n == TARGET.2
}

pub const fn tile_coordinate(block_x: u32, block_y: u32) -> (u32, u32) {
    (block_y, block_x)
}

pub fn all_strata_below(strata: &[[f64; 2]], threshold: f64) -> bool {
    full_domain::all_strata_below(strata, threshold)
}

const ASYNC_STAGE_MARKER: &str = "#define GEMM_BI_TC64_STAGE_TN_ASYNC";
const SCALAR_STAGE_MARKER: &str = "#define GEMM_BI_TC64_STAGE_TN_SCALAR";

fn stage_block(source: &str) -> Result<&str, String> {
    require_count(source, ASYNC_STAGE_MARKER, 1, "async-stage marker")?;
    require_count(source, SCALAR_STAGE_MARKER, 1, "scalar-stage marker")?;
    let start = source.find(ASYNC_STAGE_MARKER).unwrap();
    let end = source.find(SCALAR_STAGE_MARKER).unwrap();
    if start >= end {
        return Err("half TN exact-entry stage boundaries reversed".into());
    }
    Ok(&source[start..end])
}

fn require_guarded_stage_contract(stage: &str) -> Result<(), String> {
    for (anchor, expected) in [
        ("gemm_bi_cp_async_valid_elems", 2),
        ("gemm_bi_cp_async_source", 2),
        ("gemm_bi_cp_async_16_zfill", 2),
        ("GEMM_BI_HALF_TN_INDEX(_r, _c)", 2),
        ("cp.async.commit_group;", 1),
    ] {
        require_count(stage, anchor, expected, "guarded-stage contract")?;
    }
    Ok(())
}

fn validate_generated_macros(source: &str) -> Result<(), String> {
    validate_macro_continuations(
        source,
        ASYNC_STAGE_MARKER,
        SCALAR_STAGE_MARKER,
        "exact staging macro",
    )?;
    validate_macro_continuations(
        source,
        "#define GEMM_BI_DEFINE_GEMM_BI_TN_TC64",
        "GEMM_BI_DEFINE_GEMM_BI_TN_TC64(bf16",
        "outer kernel macro",
    )
}

fn validate_macro_continuations(
    source: &str,
    start_marker: &str,
    end_marker: &str,
    label: &str,
) -> Result<(), String> {
    require_count(source, start_marker, 1, label)?;
    require_count(source, end_marker, 1, label)?;
    let start = source.find(start_marker).unwrap();
    let end = source[start..]
        .find(end_marker)
        .map(|offset| start + offset)
        .ok_or_else(|| format!("half TN exact-entry {label} end precedes start"))?;
    let lines = source[start..end].lines().collect::<Vec<_>>();
    let final_line = lines
        .iter()
        .rposition(|line| !line.trim().is_empty())
        .ok_or_else(|| format!("half TN exact-entry {label} is empty"))?;
    if lines[final_line].trim_end().ends_with('\\') {
        return Err(format!(
            "half TN exact-entry {label} final line unexpectedly continues"
        ));
    }
    for (index, line) in lines[..final_line].iter().enumerate() {
        if !line.trim_end().ends_with('\\') {
            return Err(format!(
                "half TN exact-entry {label} terminates at physical line {index}: {line:?}"
            ));
        }
    }
    Ok(())
}

const FAST_STAGE_DECLARATION: &str = r#"    bool fast_stage = gemm_bi_is_aligned_16(A) && gemm_bi_is_aligned_16(B) &&         \
                      ((K_out & 7) == 0) && ((N & 7) == 0);                    \
"#;
const ACCUMULATOR_DECLARATION: &str =
    "    float acc[2][4][4];                                                        \\\n";

const RETAINED_PROLOGUE: &str = r#"    if (fast_stage) {                                                          \
        GEMM_BI_TC64_STAGE_TN_ASYNC(0, 0);                                         \
    } else {                                                                   \
        GEMM_BI_TC64_STAGE_TN_SCALAR(0, 0, T_ACT, FROM_F);                         \
    }                                                                          \"#;
const EXACT_PROLOGUE: &str =
    "    GEMM_BI_TC64_STAGE_TN_ASYNC(0, 0);                                     \\";

const RETAINED_WAIT: &str = r#"        if (fast_stage) {                                                      \
            asm volatile("cp.async.wait_group 0;\n");                          \
        }                                                                      \"#;
const EXACT_WAIT: &str =
    "        asm volatile(\"cp.async.wait_group 0;\\n\");                          \\";

const RETAINED_REFILL: &str = r#"            if (fast_stage) {                                                  \
                GEMM_BI_TC64_STAGE_TN_ASYNC(read_buf ^ 1, (mt + 1) * GEMM_BI_TC64_BK); \
            } else {                                                           \
                GEMM_BI_TC64_STAGE_TN_SCALAR(read_buf ^ 1, (mt + 1) * GEMM_BI_TC64_BK, \
                                         T_ACT, FROM_F);                       \
            }                                                                  \"#;
const EXACT_REFILL: &str = r#"            GEMM_BI_TC64_STAGE_TN_ASYNC(                                 \
                read_buf ^ 1, (mt + 1) * GEMM_BI_TC64_BK);                    \"#;

const EXACT_STAGE_MACRO: &str = r#"#define GEMM_BI_TC64_STAGE_TN_ASYNC(buf, mIdx)                                \
    do {                                                                      \
        unsigned _xs =                                                        \
            Xs_sbase + (unsigned)((buf) * GEMM_BI_TC64_BK * GEMM_BI_TC64_LDB * 2); \
        unsigned _ys =                                                        \
            Ys_sbase + (unsigned)((buf) * GEMM_BI_TC64_BK * GEMM_BI_TC64_LDB * 2); \
        for (int _i = threadIdx.x;                                             \
             _i < GEMM_BI_TC64_BK * (GEMM_BI_TC64_BM / 8);                   \
             _i += GEMM_BI_TC64_THREADS) {                                    \
            int _r = _i / (GEMM_BI_TC64_BM / 8);                             \
            int _c = (_i % (GEMM_BI_TC64_BM / 8)) * 8;                       \
            int _gm = (mIdx) + _r;                                            \
            int _gk = pid_m * GEMM_BI_TC64_BM + _c;                           \
            unsigned _dst = _xs +                                             \
                (unsigned)(GEMM_BI_HALF_TN_INDEX(_r, _c) * 2);                \
            const void* _src = A + (long long)_gm * K_out + _gk;             \
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"         \
                         :: "r"(_dst), "l"(_src));                            \
        }                                                                     \
        for (int _i = threadIdx.x;                                             \
             _i < GEMM_BI_TC64_BK * (GEMM_BI_TC64_BN / 8);                   \
             _i += GEMM_BI_TC64_THREADS) {                                    \
            int _r = _i / (GEMM_BI_TC64_BN / 8);                             \
            int _c = (_i % (GEMM_BI_TC64_BN / 8)) * 8;                       \
            int _gm = (mIdx) + _r;                                            \
            int _gn = pid_n * GEMM_BI_TC64_BN + _c;                           \
            unsigned _dst = _ys +                                             \
                (unsigned)(GEMM_BI_HALF_TN_INDEX(_r, _c) * 2);                \
            const void* _src = B + (long long)_gm * N + _gn;                 \
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"         \
                         :: "r"(_dst), "l"(_src));                            \
        }                                                                     \
        asm volatile("cp.async.commit_group;\n");                             \
    } while (0)

"#;

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

fn require_count(source: &str, anchor: &str, expected: usize, label: &str) -> Result<(), String> {
    let actual = source.matches(anchor).count();
    if actual != expected {
        return Err(format!(
            "half TN exact-entry {label} seam expected {expected}, observed {actual}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const PRODUCTION: &str = include_str!("../../kernels/gemm_bi_triad/sm80.cu");

    #[test]
    fn exact_entry_combines_2d_mapper_direct_epilogue_and_unpredicated_staging() {
        let candidate = candidate_source(PRODUCTION).unwrap();
        assert!(candidate.contains("int pid_m = blockIdx.y;"));
        assert!(candidate.contains("int pid_n = blockIdx.x;"));
        assert!(!candidate.contains("int num_pid_n = (N + GEMM_BI_TC64_BN - 1)"));
        assert!(!candidate.contains("if (gr >= K_out || gc >= N) continue;"));
        assert!(!candidate.contains("bool packed = gc + 1 < N"));
        assert_eq!(
            candidate
                .matches("cp.async.ca.shared.global [%0], [%1], 16;")
                .count(),
            2
        );
        assert!(!candidate.contains("gemm_bi_cp_async_valid_elems"));
        assert!(!candidate.contains("gemm_bi_cp_async_source"));
        assert!(!candidate.contains("gemm_bi_cp_async_16_zfill"));
        assert_eq!(candidate.matches(SYMBOL_PREFIX).count(), 1);
        assert_eq!(
            candidate
                .matches(&format!("void {RETAINED_SYMBOL_PREFIX}##SUFFIX"))
                .count(),
            0
        );
    }

    #[test]
    fn exact_entry_preserves_copy_coordinates_math_and_sync() {
        let retained = retained_source(PRODUCTION).unwrap();
        let candidate = candidate_source(PRODUCTION).unwrap();
        assert!(candidate.contains("GEMM_BI_HALF_TN_INDEX(_r, _c)"));
        assert!(candidate.contains("A + (long long)_gm * K_out + _gk"));
        assert!(candidate.contains("B + (long long)_gm * N + _gn"));
        for anchor in [
            "mma.sync.aligned.m16n8k16.row.col.f32.",
            "cp.async.commit_group;",
            "cp.async.wait_group 0;",
            "__syncthreads();",
            "read_buf ^= 1;",
        ] {
            assert_eq!(
                candidate.matches(anchor).count(),
                retained.matches(anchor).count()
            );
        }
        assert_eq!(
            restore_retained_source(&candidate, PRODUCTION).unwrap(),
            retained
        );
    }

    #[test]
    fn generated_outer_kernel_macro_continues_through_read_buffer_and_epilogue() {
        let candidate = candidate_source(PRODUCTION).unwrap();
        let start = candidate
            .find("#define GEMM_BI_DEFINE_GEMM_BI_TN_TC64")
            .unwrap();
        let end = candidate[start..]
            .find("GEMM_BI_DEFINE_GEMM_BI_TN_TC64(bf16")
            .map(|offset| start + offset)
            .unwrap();
        let lines = candidate[start..end].lines().collect::<Vec<_>>();
        let final_line = lines
            .iter()
            .rposition(|line| !line.trim().is_empty())
            .unwrap();
        assert_eq!(lines[final_line].trim(), "}");
        for (index, line) in lines[..final_line].iter().enumerate() {
            assert!(
                line.trim_end().ends_with('\\'),
                "generated outer kernel macro terminates at physical line {index}: {line:?}"
            );
        }
        assert!(candidate[start..end].contains("int read_buf = 0;"));
        assert!(candidate[start..end].contains("gemm_bi_accumulate_float2_or_scalar("));
        validate_generated_macros(&candidate).unwrap();

        let malformed =
            candidate.replacen(EXACT_PROLOGUE, EXACT_PROLOGUE.trim_end_matches('\\'), 1);
        assert!(validate_generated_macros(&malformed).is_err());
    }

    #[test]
    fn admission_is_target_only_and_rejects_every_tail_and_k0() {
        assert!(is_exact_target(2_048, 768, 3_072));
        for shape in [
            (2_047, 768, 3_072),
            (2_048, 767, 3_072),
            (2_048, 768, 3_071),
            (67, 72, 72),
            (67, 69, 71),
            (64, 64, 64),
            (0, 65, 67),
        ] {
            assert!(!is_exact_target(shape.0, shape.1, shape.2));
        }
    }

    #[test]
    fn malformed_or_ambiguous_stage_parent_fails_closed() {
        assert!(candidate_source("").is_err());
        let marker = "#define GEMM_BI_TC64_STAGE_TN_SCALAR";
        let duplicate = PRODUCTION.replacen(marker, &format!("{marker}\n{marker}"), 1);
        assert!(candidate_source(&duplicate).is_err());
    }
}
