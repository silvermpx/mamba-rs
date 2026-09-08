#[path = "triad_half_nn_n64_source.rs"]
mod parent;

pub const SYMBOL: &str = "gemm_bi_nn_test_fixed_n64_m128n64_s2_d768_in_f16";

pub fn candidate_source(n64: &str, swizzle: &str) -> Result<String, String> {
    let mut source = parent::compose_source(n64, swizzle)?;
    source.push_str(&format!(
        r#"
extern "C" __global__ __launch_bounds__(128, 2)
void {SYMBOL}(
    __half* C, const __half* A, const __half* B, const float* bias,
    TriadHalfN64Params params) {{
    if (!C || !A || !B || bias != nullptr || params.alpha != 1.0f || params.beta != 0.0f
        || params.m != 2048 || params.k != 768 || params.n != 3072
        || params.lda != 768 || params.ldb != 3072 || params.ldc != 3072
        || blockDim.x != 128 || blockDim.y != 1 || blockDim.z != 1
        || gridDim.x != 768 || gridDim.y != 1 || gridDim.z != 1
        || !gbf_aligned16(A) || !gbf_aligned16(B) || !gbf_aligned16(C)) return;
    sm89_fixed_half_n64::rect_kernel<128, 2, false, false>(C, A, B,
        2048, 3072, 768, 768, 3072, 3072);
}}
"#
    ));
    Ok(source)
}

#[cfg(test)]
mod tests {
    use super::*;

    const N64: &str = include_str!("../../kernels/gemm_bi_fixed/sm89_half_n64.cu");
    const SWIZZLE: &str = include_str!("../../kernels/gemm_bi_fixed/sm89_half_swizzle.cu");

    #[test]
    fn wrapper_reuses_the_exact_body_with_d768_in_geometry() {
        let base = parent::compose_source(N64, SWIZZLE).unwrap();
        let source = candidate_source(N64, SWIZZLE).unwrap();
        assert!(source.starts_with(&base));
        assert_eq!(source.matches(&format!("void {SYMBOL}(")).count(), 1);
        assert!(source.contains("params.m != 2048 || params.k != 768 || params.n != 3072"));
        assert!(source.contains("gridDim.x != 768"));
        assert!(source.contains("sm89_fixed_half_n64::rect_kernel<128, 2, false, false>(C, A, B,"));
        assert!(source.contains("sizeof(TriadHalfN64Params) == 32"));
        assert_eq!(2 * (128 * 64 + 64 * 64) * 2, 49_152);
    }

    #[test]
    fn four_warps_own_every_m128_n64_output_once() {
        let mut owners = vec![0u8; 128 * 64];
        for thread in 0..128 {
            let warp = thread / 32;
            let lane = thread % 32;
            for fm in 0..4 {
                for fn_ in 0..4 {
                    for half in 0..2 {
                        for column_half in 0..2 {
                            let row = (warp / 2) * 64 + fm * 16 + lane / 4 + half * 8;
                            let column = (warp % 2) * 32 + fn_ * 8 + 2 * (lane % 4) + column_half;
                            owners[row * 64 + column] += 1;
                        }
                    }
                }
            }
        }
        assert!(owners.into_iter().all(|count| count == 1));
    }

    #[test]
    fn malformed_parent_source_is_rejected() {
        assert!(candidate_source("", SWIZZLE).is_err());
        assert!(candidate_source(N64, "").is_err());
    }
}
