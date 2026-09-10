#[path = "triad_f32_tn_copyplan_fused_finalize_source.rs"]
mod group_m8;
#[path = "triad_f32_tn_copyplan_raw_store_source.rs"]
mod raw_group_m8;

pub const GROUP_M8_SYMBOL: &str = group_m8::FUSED_SYMBOL;
pub const GROUP_M12_SYMBOL: &str =
    "gemm_bi_tn_test_fixed_sm89_f32_n64_copyplan_fused_finalize_groupm12_v1";
pub const GROUP_M8_RAW_SYMBOL: &str = raw_group_m8::RAW_SYMBOL;
pub const GROUP_M12_RAW_SYMBOL: &str =
    "gemm_bi_tn_test_fixed_sm89_f32_n64_copyplan_raw_groupm12_v1";

fn replace_once(source: &mut String, before: &str, after: &str) -> Result<(), String> {
    let count = source.matches(before).count();
    if count != 1 {
        return Err(format!(
            "GROUP_M12 fused-finalize anchor {before:?}: expected1 observed{count}"
        ));
    }
    *source = source.replacen(before, after, 1);
    Ok(())
}

pub fn compose_source() -> Result<String, String> {
    let mut source = group_m8::compose_source()?;
    replace_once(&mut source, GROUP_M8_SYMBOL, GROUP_M12_SYMBOL)?;
    replace_once(
        &mut source,
        "#define SM89_EXACT_N64_CP_GROUP_M 8",
        "#define SM89_EXACT_N64_CP_GROUP_M 12",
    )?;
    Ok(source)
}

pub fn compose_raw_source() -> Result<String, String> {
    let mut source = raw_group_m8::compose_source()?;
    replace_once(&mut source, GROUP_M8_RAW_SYMBOL, GROUP_M12_RAW_SYMBOL)?;
    replace_once(
        &mut source,
        "#define SM89_EXACT_N64_CP_GROUP_M 8",
        "#define SM89_EXACT_N64_CP_GROUP_M 12",
    )?;
    Ok(source)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_m12_changes_only_symbol_and_grouping() {
        let source = compose_source().unwrap();
        assert_eq!(source.matches(GROUP_M12_SYMBOL).count(), 1);
        assert!(!source.contains(GROUP_M8_SYMBOL));
        assert_eq!(
            source
                .matches("#define SM89_EXACT_N64_CP_GROUP_M 12")
                .count(),
            1
        );
        assert!(!source.contains("#define SM89_EXACT_N64_CP_GROUP_M 8"));
        let mut reversed = source;
        replace_once(&mut reversed, GROUP_M12_SYMBOL, GROUP_M8_SYMBOL).unwrap();
        replace_once(
            &mut reversed,
            "#define SM89_EXACT_N64_CP_GROUP_M 12",
            "#define SM89_EXACT_N64_CP_GROUP_M 8",
        )
        .unwrap();
        assert_eq!(reversed, group_m8::compose_source().unwrap());
    }

    #[test]
    fn group_m12_retains_exact_fma_and_fused_reducer_arithmetic() {
        let source = compose_source().unwrap();
        assert_eq!(
            source
                .matches("__fmaf_rn(a_reg[i], b_reg[j], acc[i][j])")
                .count(),
            2
        );
        for required in [
            "__dadd_rn(",
            "__dmul_rn((double)alpha, sum)",
            "__double2float_rn(",
            "dW[idx] = __fadd_rn(dW[idx], update)",
        ] {
            assert!(source.contains(required), "missing {required}");
        }
    }

    #[test]
    fn group_m12_raw_store_exposes_the_same_grouping_without_alpha_store() {
        let source = compose_raw_source().unwrap();
        assert_eq!(source.matches(GROUP_M12_RAW_SYMBOL).count(), 1);
        assert!(source.contains("#define SM89_EXACT_N64_CP_GROUP_M 12"));
        assert!(!source.contains("#define SM89_EXACT_N64_CP_GROUP_M 8"));
        assert_eq!(source.matches("float val = acc[i][j];").count(), 1);
        assert!(!source.contains("__fmul_rn(alpha, acc[i][j])"));
        let mut reversed = source;
        replace_once(&mut reversed, GROUP_M12_RAW_SYMBOL, GROUP_M8_RAW_SYMBOL).unwrap();
        replace_once(
            &mut reversed,
            "#define SM89_EXACT_N64_CP_GROUP_M 12",
            "#define SM89_EXACT_N64_CP_GROUP_M 8",
        )
        .unwrap();
        assert_eq!(reversed, raw_group_m8::compose_source().unwrap());
    }
}
