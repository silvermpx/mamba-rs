#[path = "support/triad_half_nt_fixed_s3_source.rs"]
mod bxor;
#[path = "support/triad_half_nt_m96n128_s3_source.rs"]
mod m96;

const PRODUCTION: &str = include_str!("../kernels/gemm_bi_triad/sm89_half.cu");
const LAYOUT: &str = include_str!("../kernels/gemm_bi_fixed/sm89_half_swizzle_layout.cuh");
const SWIZZLE: &str = include_str!("../kernels/gemm_bi_fixed/sm89_half_swizzle.cu");
const S3: &str = include_str!("../kernels/gemm_bi_fixed/sm89_half_s3.cu");

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

#[test]
fn production_source_is_the_measured_three_family_composition() {
    let nn = renamed(
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
            (
                "gemm_bi_nn_fixed_sm89_tc128_s3_v1_bf16",
                "gemm_bi_nn_sm89_m128n128_bk64_s3_v1_bf16",
            ),
            (
                "gemm_bi_nn_fixed_sm89_tc128_s3_v1_f16",
                "gemm_bi_nn_sm89_m128n128_bk64_s3_v1_f16",
            ),
        ],
    );
    let bxor = renamed(
        bxor::candidate_source(SWIZZLE, S3).unwrap(),
        &[
            ("FixedSm89HalfSwizzleParams", "Sm89HalfNtBxorS3Params"),
            (
                "sm89_fixed_half_swizzle_layout",
                "sm89_half_nt_bxor_s3_layout",
            ),
            ("sm89_fixed_half_swizzle", "sm89_half_nt_bxor_s3_support"),
            ("sm89_test_half_nt_s3", "sm89_half_nt_bxor_s3"),
            (
                "gemm_bi_nt_test_fixed_s3_bxor_",
                "gemm_bi_nt_sm89_m128n128_bk64_s3_bxor_v1_",
            ),
        ],
    );
    let m96 = renamed(
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
                "gemm_bi_nt_test_fixed_s3_m96n128_f16",
                "gemm_bi_nt_sm89_m96n128_bk64_s3_v1_f16",
            ),
        ],
    );

    assert!(PRODUCTION.contains(&nn));
    assert!(PRODUCTION.contains(&bxor));
    assert!(PRODUCTION.contains(&m96));
    assert_eq!(PRODUCTION.matches("extern \"C\" __global__").count(), 6);
    assert!(!PRODUCTION.contains("_test_"));
}
