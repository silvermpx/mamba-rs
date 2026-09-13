use super::dtype::WeightDtype;
use super::kernel_identity::{FramedSha256, digest_hex};

const OWNER_SOURCE: &str = include_str!("../../../kernels/mamba_ssm_parallel.cu");
const OWNER_SOURCE_SHA256: &str =
    "587ac281b432dac548fd8098e46dc296be778c873363d3f949e31cbde694aa9e";
const FOLD_MACRO: &str = "DEFINE_SSM_PARALLEL_SCAN_BWD_FOLD";
const SHORT_MACRO: &str = "DEFINE_SSM_PARALLEL_SCAN_BWD_FOLD_SHORT";
const STAGED_MACRO: &str = "DEFINE_SSM_PARALLEL_SCAN_BWD_FOLD_STAGED";
const STAGE_INDEX_MACRO: &str = "M1_HALF_STAGE_INDEX";
const LEGACY_INVOCATIONS: &str = concat!(
    "DEFINE_SSM_PARALLEL_SCAN_BWD_FOLD(f32,  float,         from_f_f32,  2, 1, 3)\n",
    "DEFINE_SSM_PARALLEL_SCAN_BWD_FOLD(bf16, __nv_bfloat16, from_f_bf16, 3, 0, 4)\n",
    "DEFINE_SSM_PARALLEL_SCAN_BWD_FOLD(f16,  __half,        from_f_f16,  3, 0, 4)\n",
);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FoldRoute {
    Legacy,
    Short,
    Staged,
}

pub(crate) fn compiler_admitted(
    device_cc: Option<(i32, i32)>,
    target: &str,
    state_cap: usize,
    nvrtc: (i32, i32),
) -> bool {
    device_cc == Some((8, 9))
        && target == "sm_89"
        && state_cap == 16
        && matches!(nvrtc, (12, 8) | (13, 0) | (13, 2))
}

pub(crate) fn route_for_shape(
    dtype: WeightDtype,
    batch: usize,
    time: usize,
    inner: usize,
    state: usize,
) -> FoldRoute {
    if dtype == WeightDtype::F32 || batch == 0 || time == 0 || inner == 0 || state == 0 {
        return FoldRoute::Legacy;
    }
    let short = batch <= 8
        && time <= 256
        && inner <= 768
        && inner.is_multiple_of(4)
        && matches!(state, 8 | 16);
    if short {
        return FoldRoute::Short;
    }
    let staged = batch <= 8
        && (1024..=4096).contains(&time)
        && (256..=768).contains(&inner)
        && inner.is_multiple_of(4)
        && state == 16
        && batch.checked_mul(inner).is_some_and(|grid| grid >= 3072);
    if staged {
        FoldRoute::Staged
    } else {
        FoldRoute::Legacy
    }
}

pub(crate) fn compose_fixed_source(
    composed: String,
    device_cc: Option<(i32, i32)>,
    target: &str,
    state_cap: usize,
    nvrtc: (i32, i32),
) -> Result<String, String> {
    if !compiler_admitted(device_cc, target, state_cap, nvrtc) {
        return Ok(composed);
    }
    compose_retained_fold_source(composed, OWNER_SOURCE)
}

fn compose_retained_fold_source(
    mut composed: String,
    owner_source: &str,
) -> Result<String, String> {
    let owner_digest = digest_hex(&FramedSha256::bytes(owner_source.as_bytes()));
    if owner_digest != OWNER_SOURCE_SHA256 {
        return Err(format!(
            "M1 fold transport owner source SHA-256 changed: expected {OWNER_SOURCE_SHA256}, got {owner_digest}"
        ));
    }
    let macro_anchor = format!("#define {FOLD_MACRO}(");
    let invocation_anchor = format!("\n{FOLD_MACRO}(f32,");
    let macro_start = unique_index(&composed, &macro_anchor, "fold macro")?;
    let invocation_start = macro_start
        + unique_index(
            &composed[macro_start..],
            &invocation_anchor,
            "fold invocation",
        )?;
    let owner_macro_start = unique_index(owner_source, &macro_anchor, "owner fold macro")?;
    let owner_invocation_start = owner_macro_start
        + unique_index(
            &owner_source[owner_macro_start..],
            &invocation_anchor,
            "owner fold invocation",
        )?;
    let original = &composed[macro_start..invocation_start];
    let owner_macro = &owner_source[owner_macro_start..owner_invocation_start];
    if original != owner_macro {
        return Err("M1 fold transport macro differs from its pinned owner source".into());
    }
    let short = make_short_clone(original)?;
    let staged = make_staged_clone(original)?;
    let definitions = format!(
        "\n{short}\n#define {STAGE_INDEX_MACRO}(s) ((s) ^ ((((s) >> 6) & 3) << 1))\n{staged}\n"
    );
    composed.insert_str(invocation_start, &definitions);

    let invocation_start = unique_index(&composed, LEGACY_INVOCATIONS, "legacy fold invocations")?;
    let additions = concat!(
        "DEFINE_SSM_PARALLEL_SCAN_BWD_FOLD_SHORT(short_bf16, __nv_bfloat16, from_f_bf16, 3, 0, 4)\n",
        "DEFINE_SSM_PARALLEL_SCAN_BWD_FOLD_SHORT(short_f16,  __half,        from_f_f16,  3, 0, 4)\n",
        "DEFINE_SSM_PARALLEL_SCAN_BWD_FOLD_STAGED(staged_bf16, __nv_bfloat16, from_f_bf16, 3, 0, 4)\n",
        "DEFINE_SSM_PARALLEL_SCAN_BWD_FOLD_STAGED(staged_f16,  __half,        from_f_f16,  3, 0, 4)\n",
        "#undef DEFINE_SSM_PARALLEL_SCAN_BWD_FOLD_SHORT\n",
        "#undef DEFINE_SSM_PARALLEL_SCAN_BWD_FOLD_STAGED\n",
        "#undef M1_HALF_STAGE_INDEX\n",
    );
    composed.insert_str(invocation_start + LEGACY_INVOCATIONS.len(), additions);
    Ok(composed)
}

fn make_short_clone(original: &str) -> Result<String, String> {
    let mut clone = original.to_string();
    replace_once(&mut clone, FOLD_MACRO, SHORT_MACRO, "short macro name")?;
    let store_anchor = "            /* One partial row per (n, group):";
    let epilogue_anchor = "        /* d_delta / d_u:";
    let store_start = unique_index(&clone, store_anchor, "short B/C stores")?;
    let epilogue = store_start
        + unique_index(
            &clone[store_start..],
            epilogue_anchor,
            "short fold epilogue",
        )?;
    let state_end = clone[..epilogue]
        .rfind("        }")
        .ok_or_else(|| "M1 fold transport is missing the short state-loop close".to_string())?;
    let old_stores = &clone[store_start..state_end];
    if old_stores.matches("__syncthreads();").count() != 3
        || !old_stores.contains("stage_bc[s]")
        || old_stores.contains("d_delta_raw_out")
    {
        return Err("M1 fold transport found an unexpected short B/C store body".into());
    }
    let replacement = macro_lines(&[
        "            /* Short rows need no shared store transpose; arithmetic is unchanged. */",
        "            int row_bc = ((bid * d_state + n) * (d_inner / G) + gid) * T;",
        "            _Pragma(\"unroll\")",
        "            for (int i = 0; i < NITEMS; i++) {",
        "                int t = chunk_start + threadIdx.x * NITEMS + i;",
        "                if (t < T) {",
        "                    d_B_local[row_bc + t] = FROM_F(acc_B[i]);",
        "                    d_C_local[row_bc + t] = FROM_F(acc_C[i]);",
        "                }",
        "            }",
        "            /* Finish all reads before the next state reuses reduction scratch. */",
        "            __syncthreads();",
    ]);
    clone.replace_range(store_start..state_end, &replacement);

    let stage_start = unique_index(
        &clone,
        "        /* Stage the group's delta/u/dy rows once per chunk:",
        "short input stage",
    )?;
    let stage_end = stage_start
        + unique_index(
            &clone[stage_start..],
            "        float d_u_acc[SCAN_BWD_DGROUP][NITEMS];",
            "short input stage end",
        )?;
    let mut stage = clone[stage_start..stage_end].to_string();
    replace_once(
        &mut stage,
        "for (int s = threadIdx.x; s < CHUNK_SIZE; s += NTHREADS) {",
        "for (int s = threadIdx.x; s < min(CHUNK_SIZE, T - chunk_start); s += NTHREADS) {",
        "short valid input stage loop",
    )?;
    clone.replace_range(stage_start..stage_end, &stage);
    let decay_start = unique_index(
        &clone,
        "                float da_vals[NITEMS];",
        "short decay",
    )?;
    let decay_end = decay_start
        + unique_index(
            &clone[decay_start..],
            "                /* The running postfix",
            "short decay end",
        )?;
    let mut decay = clone[decay_start..decay_end].to_string();
    replace_once(
        &mut decay,
        "float delta_v = HOLD_ROWS",
        "float delta_v = (t < T) ? (HOLD_ROWS",
        "short guarded decay start",
    )?;
    replace_once(
        &mut decay,
        ": to_f(stage_delta[gg * CHUNK_SIZE + s]);",
        ": to_f(stage_delta[gg * CHUNK_SIZE + s])) : 0.0f;",
        "short guarded decay end",
    )?;
    clone.replace_range(decay_start..decay_end, &decay);
    validate_scan_clone(&clone)?;
    Ok(clone)
}

fn make_staged_clone(original: &str) -> Result<String, String> {
    let mut clone = original.to_string();
    replace_once(&mut clone, FOLD_MACRO, STAGED_MACRO, "staged macro name")?;
    for tile in ["stage_delta", "stage_u", "stage_dy"] {
        let count = wrap_stage_subscripts(&mut clone, tile)?;
        if count != 5 {
            return Err(format!(
                "M1 fold transport expected five {tile} subscripts, found {count}"
            ));
        }
    }
    validate_scan_clone(&clone)?;
    Ok(clone)
}

fn validate_scan_clone(clone: &str) -> Result<(), String> {
    if clone.matches("block_scan_ab_with_exclusive(").count() != 1
        || clone
            .matches("block_reverse_scan_ab_with_exclusive(")
            .count()
            != 1
    {
        return Err("M1 fold transport clone changed the scan call inventory".into());
    }
    Ok(())
}

fn wrap_stage_subscripts(source: &mut String, tile: &str) -> Result<usize, String> {
    let needle = format!("{tile}[");
    let mut cursor = 0;
    let mut count = 0;
    while let Some(offset) = source[cursor..].find(&needle) {
        let start = cursor + offset;
        let index_start = start + needle.len();
        let end = index_start
            + source[index_start..].find(']').ok_or_else(|| {
                format!("M1 fold transport found an unterminated {tile} subscript")
            })?;
        let index = source[index_start..end].to_string();
        let replacement = format!("{tile}[{STAGE_INDEX_MACRO}({index})]");
        source.replace_range(start..=end, &replacement);
        cursor = start + replacement.len();
        count += 1;
    }
    Ok(count)
}

fn macro_lines(lines: &[&str]) -> String {
    let mut output = String::new();
    for line in lines {
        output.push_str(line);
        for _ in line.len()..78 {
            output.push(' ');
        }
        output.push_str("\\\n");
    }
    output
}

fn replace_once(source: &mut String, old: &str, new: &str, label: &str) -> Result<(), String> {
    let index = unique_index(source, old, label)?;
    source.replace_range(index..index + old.len(), new);
    Ok(())
}

fn unique_index(source: &str, needle: &str, label: &str) -> Result<usize, String> {
    let mut matches = source.match_indices(needle);
    let Some((index, _)) = matches.next() else {
        return Err(format!("M1 fold transport is missing the {label} anchor"));
    };
    if matches.next().is_some() {
        return Err(format!(
            "M1 fold transport found an ambiguous {label} anchor"
        ));
    }
    Ok(index)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiler_admission_is_exact() {
        for nvrtc in [(12, 8), (13, 0), (13, 2)] {
            assert!(compiler_admitted(Some((8, 9)), "sm_89", 16, nvrtc));
        }
        for (cc, target, cap, nvrtc) in [
            (Some((9, 0)), "sm_89", 16, (13, 2)),
            (Some((12, 0)), "sm_89", 16, (13, 2)),
            (Some((8, 9)), "compute_89", 16, (13, 2)),
            (Some((8, 9)), "sm_80", 16, (13, 2)),
            (Some((8, 9)), "sm_89", 16, (13, 1)),
            (Some((8, 9)), "sm_89", 16, (14, 0)),
            (Some((8, 9)), "sm_89", 32, (13, 2)),
            (Some((8, 9)), "sm_89", 64, (13, 2)),
        ] {
            assert!(!compiler_admitted(cc, target, cap, nvrtc));
        }
    }

    #[test]
    fn shape_policy_keeps_only_the_retained_half_routes() {
        for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
            for shape in [(1, 1, 4, 8), (8, 256, 768, 16)] {
                assert_eq!(
                    route_for_shape(dtype, shape.0, shape.1, shape.2, shape.3),
                    FoldRoute::Short
                );
            }
            for shape in [(4, 1024, 768, 16), (8, 4096, 768, 16)] {
                assert_eq!(
                    route_for_shape(dtype, shape.0, shape.1, shape.2, shape.3),
                    FoldRoute::Staged
                );
            }
        }
        for (dtype, shape) in [
            (WeightDtype::F32, (8, 33, 768, 16)),
            (WeightDtype::Bf16, (0, 33, 768, 16)),
            (WeightDtype::Bf16, (8, 0, 768, 16)),
            (WeightDtype::Bf16, (9, 33, 768, 16)),
            (WeightDtype::Bf16, (8, 257, 768, 16)),
            (WeightDtype::Bf16, (1, 1024, 768, 16)),
            (WeightDtype::Bf16, (8, 1024, 255, 16)),
            (WeightDtype::Bf16, (8, 1024, 767, 16)),
            (WeightDtype::Bf16, (8, 1024, 768, 8)),
            (WeightDtype::Bf16, (8, 4097, 768, 16)),
            (WeightDtype::Bf16, (8, 33, 768, 32)),
        ] {
            assert_eq!(
                route_for_shape(dtype, shape.0, shape.1, shape.2, shape.3),
                FoldRoute::Legacy
            );
        }
    }

    #[test]
    fn source_composition_is_scoped_and_retains_legacy_entries() {
        let unchanged =
            compose_fixed_source("unchanged".to_string(), Some((8, 9)), "sm_89", 64, (13, 2))
                .unwrap();
        assert_eq!(unchanged, "unchanged");

        let composed =
            compose_fixed_source(OWNER_SOURCE.to_string(), Some((8, 9)), "sm_89", 16, (13, 2))
                .unwrap();
        assert!(composed.contains(LEGACY_INVOCATIONS));
        for invocation in [
            "FOLD_SHORT(short_bf16, __nv_bfloat16, from_f_bf16, 3, 0, 4)",
            "FOLD_SHORT(short_f16,  __half,        from_f_f16,  3, 0, 4)",
            "FOLD_STAGED(staged_bf16, __nv_bfloat16, from_f_bf16, 3, 0, 4)",
            "FOLD_STAGED(staged_f16,  __half,        from_f_f16,  3, 0, 4)",
        ] {
            assert!(composed.contains(invocation));
        }
        assert_eq!(composed.matches("#undef M1_HALF_STAGE_INDEX").count(), 1);
    }

    #[test]
    fn source_composition_rejects_owner_or_anchor_drift() {
        let owner_error = compose_retained_fold_source(
            OWNER_SOURCE.to_string(),
            &OWNER_SOURCE.replacen("NTHREADS", "NTHREAD", 1),
        )
        .unwrap_err();
        assert!(owner_error.contains("owner source SHA-256 changed"));

        let anchor_error =
            compose_retained_fold_source("missing".into(), OWNER_SOURCE).unwrap_err();
        assert!(anchor_error.contains("missing the fold macro anchor"));

        let ambiguous = format!("{OWNER_SOURCE}\n{OWNER_SOURCE}");
        let ambiguous_error = compose_retained_fold_source(ambiguous, OWNER_SOURCE).unwrap_err();
        assert!(ambiguous_error.contains("ambiguous fold macro anchor"));
    }
}
