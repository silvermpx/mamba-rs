//! Test-only reuse of Fixed's two-CTA half N64 bodies for Triad projection shapes.
pub const IN_PREFIX: &str = "gemm_bi_nn_test_fixed_n64_m64n64_s3_";
pub const OUT_PREFIX: &str = "gemm_bi_nn_test_fixed_n64_m128n64_s2_";

/// Caller prepends `_typed_prelude.cuh` and Fixed `common.cuh` (with its
/// duplicate local prelude include removed). Only these four test exports
/// are instantiated; the production bodies and HalfOps are byte-derived.
pub fn compose_source(n64: &str, swizzle: &str) -> Result<String, String> {
    let body = section(
        n64,
        "namespace sm89_fixed_half_n64 {",
        "} // namespace sm89_fixed_half_n64",
    )?;
    let ops = section(
        swizzle,
        "template <typename T> struct HalfOps;",
        "#undef SM89_FHS_OPS",
    )?;
    let mut source = format!("namespace sm89_fixed_half_swizzle {{\n{ops}\n}}\n{body}\n");
    source.push_str(
        r#"
struct TriadHalfN64Params {
    float alpha;
    float beta;
    int m;
    int n;
    int k;
    int lda;
    int ldb;
    int ldc;
};
static_assert(sizeof(TriadHalfN64Params) == 32, "test ABI size");
static_assert(alignof(TriadHalfN64Params) == 4, "test ABI alignment");
"#,
    );
    for (prefix, k, n, bm, stages, s3) in [
        (IN_PREFIX, 768, 3072, 64, 3, true),
        (OUT_PREFIX, 1536, 768, 128, 2, false),
    ] {
        let grid = (2048 / bm) * (n / 64);
        for (dtype, suffix) in [("__half", "f16"), ("__nv_bfloat16", "bf16")] {
            source.push_str(&format!(
                r#"
extern "C" __global__ __launch_bounds__(128, 2)
void {prefix}{suffix}(
    {dtype}* C, const {dtype}* A, const {dtype}* B, const float* bias,
    TriadHalfN64Params params) {{
    if (!C || !A || !B || bias != nullptr || params.alpha != 1.0f || params.beta != 0.0f
        || params.m != 2048 || params.k != {k} || params.n != {n}
        || params.lda != {k} || params.ldb != {n} || params.ldc != {n}
        || blockDim.x != 128 || blockDim.y != 1 || blockDim.z != 1
        || gridDim.x != {grid} || gridDim.y != 1 || gridDim.z != 1
        || !gbf_aligned16(A) || !gbf_aligned16(B) || !gbf_aligned16(C)) return;
    sm89_fixed_half_n64::rect_kernel<{bm}, {stages}, {s3}, false>(C, A, B,
        2048, {n}, {k}, {k}, {n}, {n});
}}
"#
            ));
        }
    }
    Ok(source)
}

fn section<'a>(source: &'a str, start: &str, end: &str) -> Result<&'a str, String> {
    if source.matches(start).count() != 1 || source.matches(end).count() != 1 {
        return Err(format!("N64 source boundary changed: {start:?} / {end:?}"));
    }
    let first = source.find(start).unwrap();
    let last = source.find(end).unwrap() + end.len();
    source
        .get(first..last)
        .ok_or_else(|| "N64 source boundaries reversed".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    const N64: &str = include_str!("../../kernels/gemm_bi_inference/sm89_half_n64.cu");
    const SWIZZLE: &str = include_str!("../../kernels/gemm_bi_inference/sm89_half_swizzle.cu");

    #[test]
    fn emits_four_exact_projection_exports_without_changing_the_compute_body() {
        let source = compose_source(N64, SWIZZLE).unwrap();
        let body = N64
            .split_once("namespace sm89_fixed_half_n64 {")
            .unwrap()
            .1
            .split_once("} // namespace sm89_fixed_half_n64")
            .unwrap()
            .0;
        assert!(source.contains(body));
        for (prefix, k, n, bm, stages, s3) in [
            (IN_PREFIX, 768, 3072, 64, 3, true),
            (OUT_PREFIX, 1536, 768, 128, 2, false),
        ] {
            for dtype in ["f16", "bf16"] {
                assert_eq!(source.matches(&format!("void {prefix}{dtype}(")).count(), 1);
            }
            assert!(source.contains(&format!("params.k != {k} || params.n != {n}")));
            assert!(source.contains(&format!("rect_kernel<{bm}, {stages}, {s3}, false>")));
        }
        assert!(!source.contains("void gemm_bi_nn_fixed_sm89_"));
        assert_eq!(source.matches("__launch_bounds__(128, 2)").count(), 4);
        assert!(source.contains("params.m != 2048"));
        assert!(source.contains("sizeof(TriadHalfN64Params) == 32"));
    }

    #[test]
    fn each_tile_has_unique_mma_output_owners_and_two_cta_shared_budget() {
        for (bm, stages, k, n) in [(64usize, 3, 768usize, 3072usize), (128, 2, 1536, 768)] {
            assert_eq!(stages * (bm * 64 + 64 * 64) * 2, 49152);
            let mut owners = vec![0; bm * 64];
            for thread in 0..128 {
                let warp = thread / 32;
                let lane = thread % 32;
                for fm in 0..bm / 32 {
                    for fn_ in 0..4 {
                        for half in 0..2 {
                            for col_half in 0..2 {
                                let row = (warp / 2) * (bm / 2) + fm * 16 + lane / 4 + half * 8;
                                let col = (warp % 2) * 32 + fn_ * 8 + 2 * (lane % 4) + col_half;
                                owners[row * 64 + col] += 1;
                            }
                        }
                    }
                }
            }
            assert!(owners.iter().all(|count| *count == 1));
            assert_eq!((2048 / bm) * (n / 64), if bm == 64 { 1536 } else { 192 });
            assert_eq!(k % 64, 0);
            assert!(k / 64 >= 2); // S3 body unconditionally prefills two slabs.
        }
    }

    #[test]
    fn rejects_missing_or_duplicated_source_boundaries() {
        assert!(compose_source("", SWIZZLE).is_err());
        assert!(compose_source(N64, "").is_err());
        assert!(compose_source(&format!("{N64}\n{N64}"), SWIZZLE).is_err());
        assert!(compose_source(N64, &format!("{SWIZZLE}\n{SWIZZLE}")).is_err());
    }
}
