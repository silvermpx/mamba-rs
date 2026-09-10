#[path = "triad_half_tn_compact_source.rs"]
pub mod compact;

pub const SYMBOL_PREFIX: &str = "gemm_bi_tn_test_tc64_bk64_s2_regpipe_";

pub fn candidate_source(production: &str) -> Result<String, String> {
    let source = compact::candidate_source(production)?;
    let (start, end, loads, mma) = loop_parts(&source)?;
    let buffered = |text: &str, slot: &str| {
        text.replace("a_frag[fm]", &format!("a_frag[{slot}][fm]"))
            .replace("b_frag[fn]", &format!("b_frag[{slot}][fn]"))
    };
    let body = format!(
        r#"        unsigned a_frag[2][2][4];
        unsigned b_frag[2][4][2];
        {{
            int k0 = 0;
{}        }}
        _Pragma("unroll")
        for (int ks = 0; ks < 4; ++ks) {{
            if (ks + 1 < 4) {{
                int k0 = (ks + 1) * 16;
{}            }}
{}
        }}
"#,
        buffered(&loads, "0"),
        buffered(&loads, "(ks + 1) & 1"),
        buffered(&mma, "ks & 1"),
    );
    let replacement: String = body.lines().map(|line| format!("{line} \\\n")).collect();
    let mut result = format!("{}{}{}", &source[..start], replacement, &source[end..]);
    if result.matches(compact::SYMBOL_PREFIX).count() != 1 {
        return Err("TN register pipeline export anchor changed".into());
    }
    result = result.replacen(compact::SYMBOL_PREFIX, SYMBOL_PREFIX, 1);
    Ok(result)
}

fn loop_parts(source: &str) -> Result<(usize, usize, String, String), String> {
    const LOOP: &str = "for (int ks = 0; ks < (GEMM_BI_TC64_BK / 16); ks++)";
    if source.matches(LOOP).count() != 1 || source.matches("read_buf ^= 1;").count() != 1 {
        return Err("TN register pipeline reduction-loop anchors changed".into());
    }
    let loop_at = source.find(LOOP).unwrap();
    let pragma = source[..loop_at]
        .rfind("_Pragma(\"unroll\")")
        .ok_or("TN reduction unroll anchor missing")?;
    let start = source[..pragma].rfind('\n').map_or(0, |at| at + 1);
    let read_buf = source.find("read_buf ^= 1;").unwrap();
    let end = source[..read_buf].rfind('\n').map_or(0, |at| at + 1);
    if start >= end {
        return Err("TN reduction-loop anchors reversed".into());
    }
    let decoded = decode_macro(&source[start..end]);
    const FM_LOOP: &str = "_Pragma(\"unroll\")\n            for (int fm = 0; fm < 2; fm++)";
    let fm: Vec<_> = decoded.match_indices(FM_LOOP).map(|(at, _)| at).collect();
    if fm.len() != 2 {
        return Err(format!("TN load/MMA loop anchors changed: {}", fm.len()));
    }
    let load_start = decoded[..fm[0]].rfind('\n').map_or(0, |at| at + 1);
    let mma_start = decoded[..fm[1]].rfind('\n').map_or(0, |at| at + 1);
    let mma_end = decoded
        .rfind("\n        }")
        .ok_or("TN loop closing brace missing")?;
    let loads = decoded[load_start..mma_start].to_owned();
    let mma = decoded[mma_start..mma_end].to_owned();
    for block in [&loads, &mma] {
        if block.matches("a_frag[fm]").count() != 4 || block.matches("b_frag[fn]").count() != 2 {
            return Err("TN fragment operand anchors changed".into());
        }
    }
    Ok((start, end, loads, mma))
}

fn decode_macro(source: &str) -> String {
    source
        .lines()
        .map(|line| {
            line.trim_end()
                .strip_suffix('\\')
                .unwrap_or(line)
                .trim_end()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    const PRODUCTION: &str = include_str!("../../kernels/gemm_bi_triad/sm80.cu");

    #[test]
    fn source_primes_and_alternates_fragment_slots_without_changing_the_tn_epilogue() {
        let actual = candidate_source(PRODUCTION).unwrap();
        assert!(actual.contains("void gemm_bi_tn_test_tc64_bk64_s2_regpipe_##SUFFIX"));
        assert!(actual.contains("unsigned a_frag[2][2][4];"));
        assert!(actual.contains("unsigned b_frag[2][4][2];"));
        assert!(actual.contains("a_frag[(ks + 1) & 1][fm]"));
        assert!(actual.contains("a_frag[ks & 1][fm]"));
        assert!(actual.contains("C[(long long)gr * N + gc] += alpha * acc[fm][fn][e];"));
    }

    #[test]
    fn malformed_source_is_rejected_instead_of_partially_transforming_a_kernel() {
        let missing = PRODUCTION.replacen("#define GEMM_BI_TC64_STAGE_TN_ASYNC", "", 1);
        assert!(candidate_source(&missing).is_err());
        let duplicate = format!("{PRODUCTION}\n#define GEMM_BI_TC64_STAGE_TN_ASYNC\n");
        assert!(candidate_source(&duplicate).is_err());
    }

    #[test]
    fn only_register_lifetimes_change_and_original_loads_and_mma_operands_survive() {
        let incumbent = compact::candidate_source(PRODUCTION).unwrap();
        let (start, end, loads, mma) = loop_parts(&incumbent).unwrap();
        let actual = candidate_source(PRODUCTION)
            .unwrap()
            .replace(SYMBOL_PREFIX, compact::SYMBOL_PREFIX);
        assert_eq!(&actual[..start], &incumbent[..start]);
        let suffix = &incumbent[end..];
        assert!(actual.ends_with(suffix));
        let changed = &actual[start..actual.len() - suffix.len()];
        let mut decoded = decode_macro(changed);
        for slot in ["0", "(ks + 1) & 1", "ks & 1"] {
            decoded = decoded
                .replace(&format!("a_frag[{slot}][fm]"), "a_frag[fm]")
                .replace(&format!("b_frag[{slot}][fn]"), "b_frag[fn]");
        }
        assert_eq!(decoded.matches(&loads).count(), 2);
        assert_eq!(decoded.matches(&mma).count(), 1);
        assert!(!decoded.contains("cp.async"));
        assert!(!decoded.contains("__syncthreads"));
    }

    #[test]
    fn ambiguous_reduction_loop_does_not_produce_a_partial_pipeline() {
        let incumbent = compact::candidate_source(PRODUCTION).unwrap();
        let duplicate = format!("{incumbent}\nfor (int ks = 0; ks < (GEMM_BI_TC64_BK / 16); ks++)");
        assert!(loop_parts(&duplicate).is_err());
        let missing = incumbent.replace("read_buf ^= 1;", "read_buf = 0;");
        assert!(loop_parts(&missing).is_err());
    }
}
