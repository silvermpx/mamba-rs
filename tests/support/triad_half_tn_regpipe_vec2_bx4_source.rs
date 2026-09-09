#[path = "triad_half_tn_vec2_epilogue_source.rs"]
mod retained;

pub const SYMBOL_PREFIX: &str = "gemm_bi_tn_test_tc64_bk64_s2_regpipe_vec2_bx4_";
pub const RETAINED_SYMBOL_PREFIX: &str = retained::SYMBOL_PREFIX;
pub const BLOCK_THREADS: u32 = 128;
pub const STATIC_SHARED_BYTES: i32 = 32_768;
pub const REQUIRED_OCCUPANCY: u32 = 3;
pub const REGISTER_CAP: i32 = 128;
pub const TARGET: (usize, usize, usize) = (2_048, 768, 3_072);
pub const TARGET_GRID: u32 = 576;

pub fn retained_source(production: &str) -> Result<String, String> {
    retained::candidate_source(production)
}

pub fn candidate_source(production: &str) -> Result<String, String> {
    let mut source = retained_source(production)?;
    for slot in ["0", "(ks + 1) & 1"] {
        replace_exact(&mut source, &b_x2_block(slot), &b_x4_block(slot))?;
    }
    replace_exact(&mut source, RETAINED_SYMBOL_PREFIX, SYMBOL_PREFIX)?;
    Ok(source)
}

pub fn restore_retained_source(candidate: &str) -> Result<String, String> {
    let mut source = candidate.to_owned();
    for slot in ["0", "(ks + 1) & 1"] {
        replace_exact(&mut source, &b_x4_block(slot), &b_x2_block(slot))?;
    }
    replace_exact(&mut source, SYMBOL_PREFIX, RETAINED_SYMBOL_PREFIX)?;
    Ok(source)
}

fn b_x2_block(slot: &str) -> String {
    encode_macro_block(&format!(
        r#"            _Pragma("unroll")
            for (int fn = 0; fn < 4; fn++) {{
                int srow = k0 + lm_brow_off + lm_r;
                unsigned addr = Ys_rd +
                    (unsigned)((GEMM_BI_HALF_TN_INDEX(srow, warpN + fn * 8)) * 2);
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
                    "{{%0,%1}}, [%2];\n"
                    : "=r"(b_frag[{slot}][fn][0]), "=r"(b_frag[{slot}][fn][1])
                    : "r"(addr));
            }}"#
    ))
}

fn b_x4_block(slot: &str) -> String {
    encode_macro_block(&format!(
        r#"            {{
                unsigned addr = Ys_rd +
                    (unsigned)((GEMM_BI_HALF_TN_INDEX(
                        k0 + ((lm_q & 1) ? 8 : 0) + lm_r,
                        warpN + ((lm_q & 2) ? 8 : 0))) * 2);
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                    "{{%0,%1,%2,%3}}, [%4];\n"
                    : "=r"(b_frag[{slot}][0][0]),
                      "=r"(b_frag[{slot}][0][1]),
                      "=r"(b_frag[{slot}][1][0]),
                      "=r"(b_frag[{slot}][1][1])
                    : "r"(addr));
            }}
            {{
                unsigned addr = Ys_rd +
                    (unsigned)((GEMM_BI_HALF_TN_INDEX(
                        k0 + ((lm_q & 1) ? 8 : 0) + lm_r,
                        warpN + 16 + ((lm_q & 2) ? 8 : 0))) * 2);
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                    "{{%0,%1,%2,%3}}, [%4];\n"
                    : "=r"(b_frag[{slot}][2][0]),
                      "=r"(b_frag[{slot}][2][1]),
                      "=r"(b_frag[{slot}][3][0]),
                      "=r"(b_frag[{slot}][3][1])
                    : "r"(addr));
            }}"#
    ))
}

fn encode_macro_block(decoded: &str) -> String {
    decoded.lines().map(|line| format!("{line} \\\n")).collect()
}

fn replace_exact(source: &mut String, before: &str, after: &str) -> Result<(), String> {
    let actual = source.matches(before).count();
    if actual != 1 {
        return Err(format!(
            "half TN B x4 source seam expected once, observed {actual}"
        ));
    }
    *source = source.replacen(before, after, 1);
    Ok(())
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
    fn x4_lane_groups_reproduce_two_adjacent_x2_fragment_registers() {
        for pair in 0..2 {
            for lane in 0..32 {
                let q = lane >> 3;
                let row_in_matrix = lane & 7;
                let candidate_row = row_in_matrix + if q & 1 == 1 { 8 } else { 0 };
                let candidate_column = pair * 16 + if q & 2 == 2 { 8 } else { 0 };
                let candidate_register = q;

                let expected_fn = pair * 2 + if q >= 2 { 1 } else { 0 };
                let expected_half = q & 1;
                let expected_row = row_in_matrix + expected_half * 8;
                let expected_column = expected_fn * 8;
                let expected_register = (expected_fn & 1) * 2 + expected_half;

                assert_eq!(candidate_row, expected_row, "pair={pair} lane={lane}");
                assert_eq!(candidate_column, expected_column, "pair={pair} lane={lane}");
                assert_eq!(
                    candidate_register, expected_register,
                    "pair={pair} lane={lane}"
                );
            }
        }
    }

    #[test]
    fn x4_addresses_remain_naturally_aligned_in_the_retained_xor_layout() {
        for k0 in [0usize, 16, 32, 48] {
            for warp_n in [0usize, 32] {
                for pair in 0..2 {
                    for lane in 0..32 {
                        let q = lane >> 3;
                        let row = k0 + (lane & 7) + if q & 1 == 1 { 8 } else { 0 };
                        let column = warp_n + pair * 16 + if q & 2 == 2 { 8 } else { 0 };
                        let physical = row * 64 + (column ^ ((row & 7) * 8));
                        assert_eq!(physical * 2 % 16, 0, "unaligned pair={pair} lane={lane}");
                        assert!(row < 64);
                        assert!(column < 64);
                    }
                }
            }
        }
    }

    #[test]
    fn emitted_x4_loads_use_literal_destinations_and_short_lived_addresses() {
        let candidate = candidate_source(PRODUCTION).unwrap();
        assert!(!candidate.contains("for (int pair = 0; pair < 2; ++pair)"));
        assert!(!candidate.contains("pair * 2"));
        assert!(!candidate.contains("int scol = warpN + pair * 16"));
        assert_eq!(candidate.matches("warpN + ((lm_q & 2) ? 8 : 0)").count(), 2);
        assert_eq!(
            candidate
                .matches("warpN + 16 + ((lm_q & 2) ? 8 : 0)")
                .count(),
            2
        );
        for binding in [
            "b_frag[0][0][0]",
            "b_frag[0][0][1]",
            "b_frag[0][1][0]",
            "b_frag[0][1][1]",
            "b_frag[0][2][0]",
            "b_frag[0][2][1]",
            "b_frag[0][3][0]",
            "b_frag[0][3][1]",
            "b_frag[(ks + 1) & 1][0][0]",
            "b_frag[(ks + 1) & 1][0][1]",
            "b_frag[(ks + 1) & 1][1][0]",
            "b_frag[(ks + 1) & 1][1][1]",
            "b_frag[(ks + 1) & 1][2][0]",
            "b_frag[(ks + 1) & 1][2][1]",
            "b_frag[(ks + 1) & 1][3][0]",
            "b_frag[(ks + 1) & 1][3][1]",
        ] {
            assert!(candidate.contains(binding), "missing {binding}");
        }
    }

    #[test]
    fn source_transform_is_exactly_reversible_and_changes_only_b_loads_and_export() {
        let retained = retained_source(PRODUCTION).unwrap();
        let candidate = candidate_source(PRODUCTION).unwrap();
        assert_eq!(restore_retained_source(&candidate).unwrap(), retained);
        assert_eq!(
            candidate
                .matches("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16")
                .count(),
            retained
                .matches("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16")
                .count()
                + 4
        );
        assert_eq!(
            candidate
                .matches("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16")
                .count(),
            0
        );
        assert_eq!(
            candidate
                .matches("mma.sync.aligned.m16n8k16.row.col.f32.")
                .count(),
            1
        );
        for binding in [
            "b_frag[0][0][0]",
            "b_frag[0][1][1]",
            "b_frag[0][2][0]",
            "b_frag[0][3][1]",
            "b_frag[(ks + 1) & 1][0][0]",
            "b_frag[(ks + 1) & 1][1][1]",
            "b_frag[(ks + 1) & 1][2][0]",
            "b_frag[(ks + 1) & 1][3][1]",
        ] {
            assert!(candidate.contains(binding), "missing {binding}");
        }
        assert_eq!(candidate.matches(SYMBOL_PREFIX).count(), 1);
        assert!(!candidate.contains("void gemm_bi_tn_test_tc64_bk64_s2_regpipe_vec2_##SUFFIX"));
        assert!(candidate.contains("a_frag[(ks + 1) & 1][fm]"));
        assert!(candidate.contains("a_frag[ks & 1][fm]"));
        assert!(candidate.contains("gemm_bi_accumulate_float2_or_scalar("));
        assert_eq!(
            candidate.matches("cp.async.commit_group;").count(),
            retained.matches("cp.async.commit_group;").count()
        );
        assert_eq!(
            candidate.matches("cp.async.wait_group 0;").count(),
            retained.matches("cp.async.wait_group 0;").count()
        );
        assert_eq!(
            candidate.matches("__syncthreads();").count(),
            retained.matches("__syncthreads();").count()
        );
        assert_eq!(candidate.matches("read_buf ^= 1;").count(), 1);
    }

    #[test]
    fn transform_fails_closed_when_the_retained_b_load_seam_is_ambiguous() {
        assert!(candidate_source("").is_err());
        let duplicated = format!("{PRODUCTION}\n{PRODUCTION}");
        assert!(candidate_source(&duplicated).is_err());
    }

    #[test]
    fn screen_policy_requires_all_four_positive_finite_strata_below_threshold() {
        assert!(all_strata_below(&[[0.970, 0.980]; 4], 0.985));
        assert!(!all_strata_below(&[[0.970, 0.985]; 4], 0.985));
        assert!(!all_strata_below(&[[0.970, 0.980]; 3], 0.985));
        assert!(!all_strata_below(&[[f64::NAN, 0.980]; 4], 0.985));
        assert!(!all_strata_below(&[[0.0, 0.980]; 4], 0.985));
    }

    #[test]
    fn d768_in_geometry_and_resource_contract_are_frozen() {
        assert_eq!(TARGET, (2_048, 768, 3_072));
        assert_eq!(
            768usize.div_ceil(64) * 3_072usize.div_ceil(64),
            TARGET_GRID as usize
        );
        assert_eq!(BLOCK_THREADS, 128);
        assert_eq!(STATIC_SHARED_BYTES, 32_768);
        assert_eq!(REQUIRED_OCCUPANCY, 3);
        assert_eq!(REGISTER_CAP, 128);
    }
}
