use std::collections::BTreeSet;

pub(super) const SM89_FINALIST_SYMBOL: &str =
    "gemm_bi_nt_sm89_mma_tf32_compact8_v1_m128n64_bk32_s2";

const SM80_SOURCE: &str = include_str!("../../../../kernels/gemm_bi_triad/sm80.cu");
const COMPACT_HELPER: &str = include_str!("../../../../kernels/gemm_bi_triad/sm89_nt_compact.cuh");
const PREAMBLES: [&str; 5] = [
    include_str!("../../../../kernels/_typed_prelude.cuh"),
    include_str!("../../../../kernels/gemm_bi_triad/contract.cuh"),
    include_str!("../../../../kernels/gemm_bi_triad/common.cuh"),
    include_str!("../../../../kernels/gemm_bi_triad/epilogue.cuh"),
    include_str!("../../../../kernels/gemm_bi_triad/mma16.cuh"),
];

const ORIGINAL_SYMBOL: &str = "gemm_bi_nt_sm80_mma_tf32_v1_m128n64_bk32_s2";

#[derive(Clone, Copy)]
struct Transformation {
    label: &'static str,
    from: &'static str,
    to: &'static str,
}

const TRANSFORMATIONS: [Transformation; 8] = [
    Transformation {
        label: "compact finalist storage",
        from: concat!(
            "    static constexpr int bk32 = 32;\n",
            "    static constexpr int ARows = Op == SgbTf32Tn ? bk32 : BM;\n",
            "    static constexpr int AStride = Op == SgbTf32Tn ? BM + 8 : 36;\n",
            "    static constexpr int BRows = Op == SgbTf32Nt ? BN : bk32;\n",
            "    static constexpr int BStride = Op == SgbTf32Nt ? 36\n",
            "        : (BN == 64 ? 72 : (BN == 32 ? 40 : 24));"
        ),
        to: concat!(
            "    static constexpr int bk32 = 32;\n",
            "    static constexpr bool compact_eight_warp_s2 =\n",
            "        Op == SgbTf32Nt && BM == 128 && BN == 64 && Stages == 2;\n",
            "    static constexpr int ARows = Op == SgbTf32Tn ? bk32 : BM;\n",
            "    static constexpr int AStride = Op == SgbTf32Tn ? BM + 8\n",
            "        : (compact_eight_warp_s2 ? 32 : 36);\n",
            "    static constexpr int BRows = Op == SgbTf32Nt ? BN : bk32;\n",
            "    static constexpr int BStride = Op == SgbTf32Nt\n",
            "        ? (compact_eight_warp_s2 ? 32 : 36)\n",
            "        : (BN == 64 ? 72 : (BN == 32 ? 40 : 24));"
        ),
    },
    Transformation {
        label: "compact finalist A slot",
        from: concat!(
            "    if constexpr (Op == SgbTf32Tn) {\n",
            "        return storage->a[stage][reduction][row];\n",
            "    }\n",
            "    return storage->a[stage][row][reduction];"
        ),
        to: concat!(
            "    if constexpr (Op == SgbTf32Tn) {\n",
            "        return storage->a[stage][reduction][row];\n",
            "    }\n",
            "    if constexpr (Op == SgbTf32Nt && BM == 128 && BN == 64 && Stages == 2) {\n",
            "        return storage->a[stage][row][gemm_bi_nt_compact8_xor_k(row, reduction)];\n",
            "    }\n",
            "    return storage->a[stage][row][reduction];"
        ),
    },
    Transformation {
        label: "compact finalist B slot",
        from: concat!(
            "    if constexpr (Op == SgbTf32Nt) {\n",
            "        return storage->b[stage][column][reduction];\n",
            "    }\n",
            "    return storage->b[stage][reduction][column];"
        ),
        to: concat!(
            "    if constexpr (Op == SgbTf32Nt) {\n",
            "        if constexpr (BM == 128 && BN == 64 && Stages == 2) {\n",
            "            return storage->b[stage][column][gemm_bi_nt_compact8_xor_k(column, reduction)];\n",
            "        }\n",
            "        return storage->b[stage][column][reduction];\n",
            "    }\n",
            "    return storage->b[stage][reduction][column];"
        ),
    },
    Transformation {
        label: "compact finalist storage extent",
        from: "== 55296, \"NT M128N64 s2 storage\"",
        to: "== 49152, \"NT compact-eight-warp M128N64 s2 storage\"",
    },
    Transformation {
        label: "compact finalist accumulator ownership",
        from: concat!(
            "__device__ __forceinline__ void gemm_bi_tf32_kernel(\n",
            "    float* output, const float* a, const float* b, const float* bias,\n",
            "    Sm80Tf32KernelParams params) {\n",
            "    constexpr int MAtoms = BM == 128 ? 4 : (BM == 64 ? 2 : 1);"
        ),
        to: concat!(
            "__device__ __forceinline__ void gemm_bi_tf32_kernel(\n",
            "    float* output, const float* a, const float* b, const float* bias,\n",
            "    Sm80Tf32KernelParams params) {\n",
            "    constexpr bool compact_eight_warp_s2 =\n",
            "        Op == SgbTf32Nt && BM == 128 && BN == 64 && Stages == 2;\n",
            "    constexpr int MAtoms = compact_eight_warp_s2 ? 2\n",
            "        : (BM == 128 ? 4 : (BM == 64 ? 2 : 1));"
        ),
    },
    Transformation {
        label: "compact finalist compute and row ownership",
        from: concat!(
            "    bool compute = BM != 128 || warp < 4;\n",
            "    int warp_m = BM == 128 ? (warp >> 1) * 64\n",
            "        : (BM == 64 ? (warp >> 1) * 32 : 0);"
        ),
        to: concat!(
            "    bool compute = compact_eight_warp_s2 || BM != 128 || warp < 4;\n",
            "    int warp_m = compact_eight_warp_s2 ? (warp >> 1) * 32\n",
            "        : (BM == 128 ? (warp >> 1) * 64\n",
            "        : (BM == 64 ? (warp >> 1) * 32 : 0));"
        ),
    },
    Transformation {
        label: "compact finalist target export",
        from: concat!(
            "GEMM_BI_TF32_DEFINE_KERNEL(",
            "gemm_bi_nt_sm80_mma_tf32_v1_m128n64_bk32_s2, ",
            "SgbTf32Nt, 128, 64, 2, 256, 1)"
        ),
        to: concat!(
            "GEMM_BI_TF32_DEFINE_KERNEL(",
            "gemm_bi_nt_sm89_mma_tf32_compact8_v1_m128n64_bk32_s2, ",
            "SgbTf32Nt, 128, 64, 2, 256, 1)"
        ),
    },
    Transformation {
        label: "compact finalist target signature",
        from: concat!(
            "TF32_ASSERT_KERNEL_SIGNATURE(",
            "gemm_bi_nt_sm80_mma_tf32_v1_m128n64_bk32_s2);"
        ),
        to: concat!(
            "TF32_ASSERT_KERNEL_SIGNATURE(",
            "gemm_bi_nt_sm89_mma_tf32_compact8_v1_m128n64_bk32_s2);"
        ),
    },
];

pub(super) fn compose_sm89_finalist_source() -> Result<String, String> {
    compose_from_parts(PREAMBLES, COMPACT_HELPER, SM80_SOURCE)
}

fn transform_sm80_source(source: &str) -> Result<String, String> {
    let mut transformed = source.to_owned();
    for transformation in TRANSFORMATIONS {
        replace_exact(
            &mut transformed,
            transformation.from,
            transformation.to,
            transformation.label,
        )?;
    }
    Ok(transformed)
}

fn finalist_body_from(helper: &str, source: &str) -> Result<String, String> {
    reject_quoted_include("kernels/gemm_bi_triad/sm89_nt_compact.cuh", helper)?;
    let transformed = transform_sm80_source(source)?;
    let exports = macro_names(&transformed, "GEMM_BI_TF32_DEFINE_KERNEL")?;
    let assertions = macro_names(&transformed, "TF32_ASSERT_KERNEL_SIGNATURE")?;
    if exports != assertions {
        return Err("SM89 finalist export and signature-assert sets differ".into());
    }
    if !exports.contains(SM89_FINALIST_SYMBOL) || exports.contains(ORIGINAL_SYMBOL) {
        return Err("SM89 finalist target export replacement is incomplete".into());
    }
    Ok(format!("{helper}\n{transformed}"))
}

fn replace_exact(source: &mut String, from: &str, to: &str, label: &str) -> Result<(), String> {
    let count = source.matches(from).count();
    if count != 1 {
        return Err(format!(
            "{label} anchor count changed: expected 1, observed {count}"
        ));
    }
    *source = source.replacen(from, to, 1);
    Ok(())
}

fn macro_names(source: &str, macro_name: &str) -> Result<BTreeSet<String>, String> {
    let prefix = format!("{macro_name}(");
    let mut names = BTreeSet::new();
    for line in source.lines() {
        let Some(arguments) = line.trim_start().strip_prefix(&prefix) else {
            continue;
        };
        let name = arguments
            .split([',', ')'])
            .next()
            .map(str::trim)
            .filter(|name| {
                !name.is_empty()
                    && name
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
            })
            .ok_or_else(|| format!("{macro_name} invocation has an invalid first argument"))?;
        if !names.insert(name.to_owned()) {
            return Err(format!(
                "{macro_name} contains duplicate invocation for {name}"
            ));
        }
    }
    if names.is_empty() {
        return Err(format!("{macro_name} has no concrete invocations"));
    }
    Ok(names)
}

fn reject_quoted_include(logical_name: &str, source: &str) -> Result<(), String> {
    for line in source.lines() {
        let trimmed = line.trim_start();
        let Some(rest) = trimmed.strip_prefix('#') else {
            continue;
        };
        let rest = rest.trim_start();
        let Some(target) = rest.strip_prefix("include").map(str::trim_start) else {
            continue;
        };
        if target.starts_with('"') {
            return Err(format!(
                "{logical_name} contains unexpected quoted include {target}"
            ));
        }
    }
    Ok(())
}

fn normalized_part(source: &str) -> String {
    source.lines().collect::<Vec<_>>().join("\n")
}

fn compose_from_parts(preambles: [&str; 5], helper: &str, source: &str) -> Result<String, String> {
    for (logical_name, preamble) in [
        ("kernels/_typed_prelude.cuh", preambles[0]),
        ("kernels/gemm_bi_triad/contract.cuh", preambles[1]),
        ("kernels/gemm_bi_triad/common.cuh", preambles[2]),
        ("kernels/gemm_bi_triad/epilogue.cuh", preambles[3]),
        ("kernels/gemm_bi_triad/mma16.cuh", preambles[4]),
    ] {
        reject_quoted_include(logical_name, preamble)?;
    }
    reject_quoted_include("kernels/gemm_bi_triad/sm89_nt_compact.cuh", helper)?;
    reject_quoted_include("kernels/gemm_bi_triad/sm80.cu", source)?;

    let body = finalist_body_from(helper, source)?;
    Ok(preambles
        .into_iter()
        .map(normalized_part)
        .chain(std::iter::once(normalized_part(&body)))
        .collect::<Vec<_>>()
        .join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(feature = "cuda"))]
    const TEST_HELPER_NAME: &str = "gemm_bi_nt_test_compact_xor_k";
    #[cfg(not(feature = "cuda"))]
    const TEST_SYMBOL: &str = "gemm_bi_nt_test_compact_eight_warp_sm80_mma_tf32_v1_m128n64_bk32_s2";
    #[cfg(not(feature = "cuda"))]
    const FROZEN_HELPER: &str = include_str!("../../../../tests/gemm_bi_tf32_nt_compact_xor.cu");

    #[cfg(not(feature = "cuda"))]
    mod frozen_candidate {
        include!("../../../../tests/gemm_bi_tf32_nt_compact_xor.rs");

        pub(super) fn compact_eight_warp_s2_source() -> Result<String, String> {
            compact_eight_warp_s2_candidate_source()
        }
    }

    #[cfg(not(feature = "cuda"))]
    fn normalized_frozen_candidate() -> String {
        frozen_candidate::compact_eight_warp_s2_source()
            .expect("compose frozen CompactEightWarpS2 candidate")
            .replacen(FROZEN_HELPER, COMPACT_HELPER, 1)
            .replace(TEST_HELPER_NAME, "gemm_bi_nt_compact8_xor_k")
            .replace(TEST_SYMBOL, SM89_FINALIST_SYMBOL)
    }

    #[test]
    fn missing_and_duplicate_anchors_fail_closed() {
        for transformation in TRANSFORMATIONS {
            let missing = SM80_SOURCE.replacen(transformation.from, "", 1);
            let missing_error = transform_sm80_source(&missing).unwrap_err();
            assert!(missing_error.contains(transformation.label));
            assert!(missing_error.contains("observed 0"));

            let duplicate = format!("{SM80_SOURCE}\n{}", transformation.from);
            let duplicate_error = transform_sm80_source(&duplicate).unwrap_err();
            assert!(duplicate_error.contains(transformation.label));
            assert!(duplicate_error.contains("observed 2"));
        }
    }

    #[test]
    fn reversed_transformation_restores_immutable_sm80_source() {
        let transformed = transform_sm80_source(SM80_SOURCE).unwrap();
        let restored = super::reverse_sm80_transformation_for_test(&transformed).unwrap();
        assert_eq!(restored, SM80_SOURCE);
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn production_body_matches_frozen_candidate_after_allowed_normalization() {
        let production = finalist_body_from(COMPACT_HELPER, SM80_SOURCE).unwrap();
        assert_eq!(production, normalized_frozen_candidate());
    }

    #[test]
    fn exports_and_signature_asserts_are_balanced() {
        let body = finalist_body_from(COMPACT_HELPER, SM80_SOURCE).unwrap();
        let exports = macro_names(&body, "GEMM_BI_TF32_DEFINE_KERNEL").unwrap();
        let assertions = macro_names(&body, "TF32_ASSERT_KERNEL_SIGNATURE").unwrap();
        assert_eq!(exports, assertions);
        assert!(exports.contains(SM89_FINALIST_SYMBOL));
        assert_eq!(exports.len(), 18);
    }

    #[test]
    fn composed_source_uses_exact_five_preambles_and_rejects_new_quoted_include() {
        let composed = compose_sm89_finalist_source().unwrap();
        for marker in [
            "__device__ __forceinline__ float to_f(float v)",
            "Three operand layouts for training:",
            "__device__ __forceinline__ float4 ld_global_L2_128B",
            "__device__ __forceinline__ void gemm_bi_store_pair_rne",
            "gemm_bi_cp_async_source",
        ] {
            assert!(
                composed.contains(marker),
                "missing preamble marker {marker}"
            );
        }
        assert!(composed.contains(SM89_FINALIST_SYMBOL));

        let parts = super::source_parts_for_test();
        let poisoned = format!("{}\n#include \"unexpected.cuh\"", parts[2]);
        let error = super::compose_from_parts_for_test(
            [parts[0], parts[1], &poisoned, parts[3], parts[4]],
            COMPACT_HELPER,
            SM80_SOURCE,
        )
        .unwrap_err();
        assert!(error.contains("unexpected.cuh"));
    }

    #[test]
    fn composed_source_emits_the_same_helper_it_validates() {
        let chosen = format!("// chosen helper seam\n{COMPACT_HELPER}");
        let composed = super::compose_from_parts_for_test(
            super::source_parts_for_test(),
            &chosen,
            SM80_SOURCE,
        )
        .unwrap();
        assert_eq!(composed.matches("// chosen helper seam").count(), 1);
        assert!(composed.contains(&chosen));
    }
}

#[cfg(test)]
fn reverse_sm80_transformation_for_test(source: &str) -> Result<String, String> {
    let mut restored = source.to_owned();
    for transformation in TRANSFORMATIONS.into_iter().rev() {
        replace_exact(
            &mut restored,
            transformation.to,
            transformation.from,
            transformation.label,
        )?;
    }
    Ok(restored)
}

#[cfg(test)]
fn source_parts_for_test() -> [&'static str; 5] {
    PREAMBLES
}

#[cfg(test)]
fn compose_from_parts_for_test(
    preambles: [&str; 5],
    helper: &str,
    source: &str,
) -> Result<String, String> {
    compose_from_parts(preambles, helper, source)
}
