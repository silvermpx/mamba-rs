#[path = "triad_half_tn_vec2_epilogue_source.rs"]
mod retained;

pub const SYMBOL_PREFIX: &str = "gemm_bi_tn_test_tc64_bk64_s2_regpipe_vec2_issue_";
pub const RETAINED_SYMBOL_PREFIX: &str = retained::SYMBOL_PREFIX;
pub const BLOCK_THREADS: u32 = 128;
pub const STATIC_SHARED_BYTES: i32 = 32_768;
pub const REQUIRED_OCCUPANCY: u32 = 3;
pub const REGISTER_CAP: i32 = 128;
pub const TARGET: (usize, usize, usize) = (2_048, 768, 3_072);
pub const TARGET_GRID: u32 = 576;

pub const fn issue_order() -> [(usize, usize); 8] {
    [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
        (0, 2),
        (1, 2),
        (0, 3),
        (1, 3),
    ]
}

pub fn retained_source(production: &str) -> Result<String, String> {
    retained::candidate_source(production)
}

pub fn candidate_source(production: &str) -> Result<String, String> {
    let mut source = retained_source(production)?;
    replace_exact(
        &mut source,
        &encode_macro_block(RETAINED_MMA_LOOP),
        &candidate_mma_block(),
        "MMA issue block",
    )?;
    replace_exact(
        &mut source,
        RETAINED_SYMBOL_PREFIX,
        SYMBOL_PREFIX,
        "candidate export",
    )?;
    Ok(source)
}

pub fn restore_retained_source(candidate: &str) -> Result<String, String> {
    let mut source = candidate.to_owned();
    replace_exact(
        &mut source,
        SYMBOL_PREFIX,
        RETAINED_SYMBOL_PREFIX,
        "restore export",
    )?;
    replace_exact(
        &mut source,
        &candidate_mma_block(),
        &encode_macro_block(RETAINED_MMA_LOOP),
        "restore MMA issue block",
    )?;
    Ok(source)
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

const RETAINED_MMA_LOOP: &str = r#"            _Pragma("unroll")
            for (int fm = 0; fm < 2; fm++) {
                _Pragma("unroll")
                for (int fn = 0; fn < 4; fn++) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32." MMA_T "."
                        MMA_T ".f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                        "{%0,%1,%2,%3};\n"
                        : "+f"(acc[fm][fn][0]), "+f"(acc[fm][fn][1]),
                          "+f"(acc[fm][fn][2]), "+f"(acc[fm][fn][3])
                        : "r"(a_frag[ks & 1][fm][0]), "r"(a_frag[ks & 1][fm][1]),
                          "r"(a_frag[ks & 1][fm][2]), "r"(a_frag[ks & 1][fm][3]),
                          "r"(b_frag[ks & 1][fn][0]), "r"(b_frag[ks & 1][fn][1]));
                }
            }"#;

fn candidate_mma_block() -> String {
    let mut decoded = String::new();
    for (fm, fn_) in issue_order() {
        decoded.push_str(&format!(
            r#"            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32." MMA_T "."
                MMA_T ".f32 "
                "{{%0,%1,%2,%3}}, {{%4,%5,%6,%7}}, {{%8,%9}}, "
                "{{%0,%1,%2,%3}};\n"
                : "+f"(acc[{fm}][{fn_}][0]), "+f"(acc[{fm}][{fn_}][1]),
                  "+f"(acc[{fm}][{fn_}][2]), "+f"(acc[{fm}][{fn_}][3])
                : "r"(a_frag[ks & 1][{fm}][0]), "r"(a_frag[ks & 1][{fm}][1]),
                  "r"(a_frag[ks & 1][{fm}][2]), "r"(a_frag[ks & 1][{fm}][3]),
                  "r"(b_frag[ks & 1][{fn_}][0]), "r"(b_frag[ks & 1][{fn_}][1]));
"#
        ));
    }
    encode_macro_block(decoded.trim_end())
}

fn encode_macro_block(decoded: &str) -> String {
    decoded.lines().map(|line| format!("{line} \\\n")).collect()
}

fn replace_exact(
    source: &mut String,
    before: &str,
    after: &str,
    label: &str,
) -> Result<(), String> {
    let actual = source.matches(before).count();
    if actual != 1 {
        return Err(format!(
            "half TN issue-order {label} seam expected 1, observed {actual}"
        ));
    }
    *source = source.replacen(before, after, 1);
    Ok(())
}
