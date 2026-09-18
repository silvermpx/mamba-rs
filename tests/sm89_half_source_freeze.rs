#[path = "support/triad_half_nt_m96n128_s3_source.rs"]
mod m96;

use m96::m64n192::m64n128::fixed_s3 as bxor;

const PRODUCTION: &str = include_str!("../kernels/gemm_bi_triad/sm80/half.cu");
const FIXED_COMMON: &str = include_str!("../kernels/gemm_bi_inference/common.cuh");
const LAYOUT: &str = include_str!("../kernels/gemm_bi_inference/sm80/half_swizzle_layout.cuh");
const SWIZZLE: &str = include_str!("../kernels/gemm_bi_inference/sm80/half_swizzle.cu");
const S3: &str = include_str!("../kernels/gemm_bi_inference/sm80/half_s3.cu");

#[test]
fn standalone_source_defines_each_fixed_common_half_helper_once() {
    for definition in [
        "static __device__ __forceinline__ bool gbf_aligned16(const void* p) {\n    return (reinterpret_cast<unsigned long long>(p) & 15ull) == 0ull;\n}",
        "static __device__ __forceinline__ bool gbf_aligned4(const void* p) {\n    return (reinterpret_cast<unsigned long long>(p) & 3ull) == 0ull;\n}",
        "static __device__ __forceinline__ void gbf_store_pair_rne(\n    __nv_bfloat16* dst, float v0, float v1) {\n    *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v0, v1);\n}",
        "static __device__ __forceinline__ void gbf_store_pair_rne(\n    __half* dst, float v0, float v1) {\n    *reinterpret_cast<__half2*>(dst) = __floats2half2_rn(v0, v1);\n}",
        "static __device__ __forceinline__ void gbf_store_pair_rne(\n    float* dst, float v0, float v1) {\n    if ((reinterpret_cast<unsigned long long>(dst) & 7ull) == 0ull) {\n        *reinterpret_cast<float2*>(dst) = make_float2(v0, v1);\n    } else {\n        dst[0] = v0;\n        dst[1] = v1;\n    }\n}",
    ] {
        assert_eq!(FIXED_COMMON.matches(definition).count(), 1);
        assert_eq!(
            PRODUCTION.matches(definition).count(),
            1,
            "standalone TriadSm89Half source must define {definition:?} exactly once",
        );
    }
}

fn callable_exports(source: &str) -> std::collections::BTreeSet<String> {
    let mut exports = std::collections::BTreeSet::new();
    let mut token_paste = std::collections::BTreeMap::new();
    let lines = source.lines().collect::<Vec<_>>();
    for (index, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        // A callable export names its operand layout first; helpers and
        // macro bodies never start a line with `void nn_`/`tn_`/`nt_`.
        if let Some(rest) = trimmed
            .strip_prefix("void ")
            .filter(|rest| ["nn_", "tn_", "nt_"].iter().any(|op| rest.starts_with(op)))
        {
            let name = rest.split('(').next().unwrap().trim();
            if let Some(prefix) = name.strip_suffix("##SUFFIX") {
                let macro_name = lines[..index]
                    .iter()
                    .rev()
                    .find_map(|candidate| candidate.trim().strip_prefix("#define "))
                    .and_then(|definition| definition.split('(').next())
                    .unwrap();
                token_paste.insert(macro_name.to_owned(), prefix.to_owned());
            } else {
                exports.insert(name.to_owned());
            }
        }
    }
    for line in lines {
        let trimmed = line.trim();
        for (macro_name, prefix) in &token_paste {
            if let Some(arguments) = trimmed
                .strip_prefix(&format!("{macro_name}("))
                .and_then(|rest| rest.strip_suffix(')'))
            {
                let suffix = arguments.rsplit(',').next().unwrap().trim();
                exports.insert(format!("{prefix}{suffix}"));
            }
        }
    }
    exports
}

#[test]
fn production_source_exposes_only_the_six_owned_callable_symbols() {
    let expected = [
        "nn_sm89_m128n128_bk64_s3_bf16",
        "nn_sm89_m128n128_bk64_s3_f16",
        "nt_sm89_m128n128_bk64_s3_bxor_bf16",
        "nt_sm89_m128n128_bk64_s3_bxor_f16",
        "nt_sm89_m96n128_bk64_s3_bf16",
        "nt_sm89_m96n128_bk64_s3_f16",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect();
    assert_eq!(callable_exports(PRODUCTION), expected);
}

fn renamed(mut source: String, replacements: &[(&str, &str)]) -> String {
    for (from, to) in replacements {
        assert!(
            source.contains(from),
            "missing measured-source anchor {from:?}"
        );
        source = source.replace(from, to);
    }
    source
}

fn without_legacy_exports(mut source: String) -> String {
    if let Some(block_start) = source.find("#define SM89_FHS_EXPORT") {
        let start = block_start.saturating_sub(1);
        let end = source[start..]
            .find("#undef SM89_FHS_EXPORT")
            .map(|offset| start + offset + "#undef SM89_FHS_EXPORT".len())
            .unwrap();
        let end = end + usize::from(source.as_bytes().get(end) == Some(&b'\n'));
        source.replace_range(start..end, "");
    }
    source.replace(
        "// Exports: nn_sm89_tc128_swizzle_bf16 and\n// nn_sm89_tc128_swizzle_f16.",
        "// Provider helpers are composed here without their legacy Fixed exports.",
    )
}

#[test]
fn production_source_is_the_measured_three_family_composition() {
    let nn = without_legacy_exports(renamed(
        format!(
            "{}\n{}",
            SWIZZLE.replace("#include \"sm89_half_swizzle_layout.cuh\"", LAYOUT),
            S3
        ),
        &[
            ("FixedSm89HalfSwizzleParams", "Sm89HalfNnS3Params"),
            ("sm89_fixed_half_swizzle_layout", "sm89_half_nn_s3_layout"),
            ("sm89_fixed_half_swizzle", "sm89_half_nn_s3_support"),
            ("sm89_fixed_half_s3", "sm89_half_nn_s3"),
            ("nn_sm89_tc128_s3_bf16", "nn_sm89_m128n128_bk64_s3_bf16"),
            ("nn_sm89_tc128_s3_f16", "nn_sm89_m128n128_bk64_s3_f16"),
        ],
    ));
    let bxor = without_legacy_exports(renamed(
        bxor::candidate_source(SWIZZLE, S3).unwrap(),
        &[
            ("FixedSm89HalfSwizzleParams", "Sm89HalfNtBxorS3Params"),
            (
                "sm89_fixed_half_swizzle_layout",
                "sm89_half_nt_bxor_s3_layout",
            ),
            ("sm89_fixed_half_swizzle", "sm89_half_nt_bxor_s3_support"),
            ("sm89_test_half_nt_s3", "sm89_half_nt_bxor_s3"),
            ("nt_test_fixed_s3_bxor_", "nt_sm89_m128n128_bk64_s3_bxor_"),
        ],
    ));
    let m96 = without_legacy_exports(renamed(
        m96::candidate_source(SWIZZLE, S3).unwrap(),
        &[
            ("FixedSm89HalfSwizzleParams", "Sm89HalfNtM96S3Params"),
            (
                "sm89_fixed_half_swizzle_layout",
                "sm89_half_nt_m96_s3_layout",
            ),
            ("sm89_fixed_half_swizzle", "sm89_half_nt_m96_s3_support"),
            ("sm89_test_half_nt_m96n128_s3", "sm89_half_nt_m96n128_s3"),
            (
                "nt_test_fixed_s3_m96n128_f16",
                "nt_sm89_m96n128_bk64_s3_f16",
            ),
        ],
    ));

    assert!(PRODUCTION.contains(&nn));
    assert!(PRODUCTION.contains(&bxor));
    assert!(PRODUCTION.contains(&m96));
    assert_eq!(callable_exports(PRODUCTION).len(), 6);
    assert!(!PRODUCTION.contains("_test_"));
}
