pub const SYMBOL: &str = "gemm_bi_tn_test_regpipe_sm80_mma_tf32_v1_m16n32_bk32_s4";
const OLD_SYMBOL: &str = "gemm_bi_tn_sm80_mma_tf32_v1_m16n32_bk32_s4";
const START: &str = "    const int k_offsets[4] = {0, 8, 16, 24};\n";
const END: &str = "\n}\n\nstruct SgbTf32ThreadPlan {";
const LOAD_START: &str = "#pragma unroll\n        for (int m_atom = 0; m_atom < MAtoms; ++m_atom) {\n            int row";
const MMA_START: &str = "#pragma unroll\n        for (int m_atom = 0; m_atom < MAtoms; ++m_atom) {\n#pragma unroll\n            for (int n_atom = 0; n_atom < NAtoms; ++n_atom) {\n                gemm_bi_tf32_mma_m16n8k8(";
const SPECIALIZE: &str =
    "    if constexpr (Op == SgbTf32Tn && BM == 16 && BN == 32 && Stages == 4) {\n";

pub fn candidate_source(production: &str) -> Result<String, String> {
    for anchor in [START, END, LOAD_START, MMA_START] {
        require_count(production, anchor, 1)?;
    }
    require_count(production, OLD_SYMBOL, 2)?;
    let start = production.find(START).unwrap();
    let end = production.find(END).unwrap();
    let load_start = production.find(LOAD_START).unwrap();
    let mma_start = production.find(MMA_START).unwrap();
    if !(start < load_start && load_start < mma_start && mma_start < end) {
        return Err("TF32 TN compute boundaries out of order".into());
    }
    let body = &production[start..end];
    if !body.ends_with("    }") {
        return Err("TF32 TN issue loop closing boundary changed".into());
    }
    let load = &production[load_start..mma_start];
    let mma = &production[mma_start..end - "    }".len()];
    let slot = |text: &str, index: &str| {
        text.replace("a_fragments[", &format!("a_fragments[{index}]["))
            .replace("b_fragments[", &format!("b_fragments[{index}]["))
    };
    let prime = slot(load, "0");
    let next = slot(load, "next_slot");
    let consume = slot(mma, "slot");
    let replacement = format!(
        concat!(
            "{SPECIALIZE}",
            "        unsigned a_fragments[2][MAtoms][4];\n",
            "        unsigned b_fragments[2][NAtoms][2];\n",
            "        {{\n            int k8 = 0;\n{prime}        }}\n",
            "#pragma unroll\n",
            "        for (int issue = 0; issue < 4; ++issue) {{\n",
            "            if (issue + 1 < 4) {{\n",
            "                int k8 = (issue + 1) * 8;\n",
            "                int next_slot = (issue + 1) & 1;\n",
            "{next}            }}\n",
            "            int slot = issue & 1;\n",
            "{consume}        }}\n",
            "    }} else {{\n{body}\n    }}"
        ),
        SPECIALIZE = SPECIALIZE,
        prime = prime,
        next = next,
        consume = consume,
        body = body,
    );
    Ok(format!(
        "{}{}{}",
        &production[..start],
        replacement,
        &production[end..]
    )
    .replace(OLD_SYMBOL, SYMBOL))
}

fn require_count(source: &str, anchor: &str, expected: usize) -> Result<(), String> {
    let count = source.matches(anchor).count();
    if count != expected {
        return Err(format!(
            "TF32 TN register pipeline anchor count: expected {expected}, found {count}: {anchor:?}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const PRODUCTION: &str = include_str!("../../kernels/gemm_bi_triad/sm80.cu");

    #[test]
    fn fragments_are_loaded_before_consumption_in_original_k8_order() {
        let mut slots = [None; 2];
        slots[0] = Some(0);
        let mut consumed = Vec::new();
        for issue in 0..4 {
            if issue + 1 < 4 {
                slots[(issue + 1) & 1] = Some(issue + 1);
            }
            assert_eq!(slots[issue & 1], Some(issue));
            consumed.push(slots[issue & 1].take().unwrap() * 8);
        }
        assert_eq!(consumed, [0, 8, 16, 24]);
    }

    #[test]
    fn candidate_only_specializes_small_tn_compute_and_renames_export() {
        let candidate = candidate_source(PRODUCTION).unwrap();
        assert_eq!(candidate.matches(SYMBOL).count(), 2);
        assert_eq!(candidate.matches(SPECIALIZE).count(), 1);
        assert!(!candidate.contains(OLD_SYMBOL));
        let start = candidate.find(SPECIALIZE).unwrap();
        let fallback = candidate[start..].find("    } else {\n").unwrap() + start;
        let end = candidate[fallback..].find(END).unwrap() + fallback;
        let original_start = PRODUCTION.find(START).unwrap();
        let original_end = PRODUCTION[original_start..].find(END).unwrap() + original_start;
        assert_eq!(
            &candidate[fallback + "    } else {\n".len()..end - "    }\n".len()],
            &PRODUCTION[original_start..original_end]
        );
        let restored = format!(
            "{}{}{}",
            &candidate[..start],
            &PRODUCTION[original_start..original_end],
            &candidate[end..]
        )
        .replace(SYMBOL, OLD_SYMBOL);
        assert_eq!(restored, PRODUCTION);
    }

    #[test]
    fn original_load_and_mma_statements_are_preserved_modulo_slot_subscript() {
        let candidate = candidate_source(PRODUCTION).unwrap();
        let special = candidate
            .split(SPECIALIZE)
            .nth(1)
            .unwrap()
            .split("    } else {\n")
            .next()
            .unwrap();
        assert!(special.contains("unsigned a_fragments[2][MAtoms][4];"));
        assert!(special.contains("unsigned b_fragments[2][NAtoms][2];"));
        let load_start = PRODUCTION.find(LOAD_START).unwrap();
        let mma_start = PRODUCTION[load_start..].find(MMA_START).unwrap() + load_start;
        let load = &PRODUCTION[load_start..mma_start];
        let mma_end = PRODUCTION[mma_start..]
            .find("    }\n}\n\nstruct SgbTf32ThreadPlan")
            .unwrap()
            + mma_start;
        let mma = &PRODUCTION[mma_start..mma_end];
        assert!(
            special.contains(
                &load
                    .replace("a_fragments[", "a_fragments[0][")
                    .replace("b_fragments[", "b_fragments[0][")
            )
        );
        assert!(
            special.contains(
                &load
                    .replace("a_fragments[", "a_fragments[next_slot][")
                    .replace("b_fragments[", "b_fragments[next_slot][")
            )
        );
        assert!(
            special.contains(
                &mma.replace("a_fragments[", "a_fragments[slot][")
                    .replace("b_fragments[", "b_fragments[slot][")
            )
        );
        assert_eq!(special.matches("gemm_bi_tf32_rna(").count(), 12);
        assert_eq!(special.matches("gemm_bi_tf32_mma_m16n8k8(").count(), 1);
    }

    #[test]
    fn modified_or_duplicated_anchors_are_rejected() {
        for anchor in [START, END, LOAD_START, MMA_START] {
            assert!(candidate_source(&PRODUCTION.replacen(anchor, "", 1)).is_err());
            assert!(candidate_source(&format!("{PRODUCTION}\n{anchor}")).is_err());
        }
        assert!(candidate_source(&PRODUCTION.replacen(OLD_SYMBOL, "missing", 1)).is_err());
        assert!(candidate_source(&format!("{PRODUCTION}\n{OLD_SYMBOL}")).is_err());
    }
}
