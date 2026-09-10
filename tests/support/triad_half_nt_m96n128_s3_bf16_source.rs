#[path = "triad_half_nt_m64n192_s3_bf16_source.rs"]
mod bf16_parent;
#[path = "triad_half_nt_m96n128_s3_source.rs"]
mod f16_parent;

pub const SYMBOL: &str = "gemm_bi_nt_test_fixed_s3_m96n128_bf16";
pub const RETAINED_SYMBOL: &str = bf16_parent::RETAINED_SYMBOL;
pub const TARGET: (usize, usize, usize) = f16_parent::TARGET;
pub const TARGET_GRID: u32 = f16_parent::TARGET_GRID;
pub const BLOCK_THREADS: u32 = f16_parent::BLOCK_THREADS;
pub const DYNAMIC_SHARED_BYTES: usize = f16_parent::DYNAMIC_SHARED_BYTES;
pub const MAX_REGISTERS: i32 = f16_parent::MAX_REGISTERS;
pub const REQUIRED_OCCUPANCY: u32 = f16_parent::REQUIRED_OCCUPANCY;
pub const EXPECTED_HMMA: usize = f16_parent::EXPECTED_HMMA;
pub const EXPECTED_LDGSTS: usize = f16_parent::EXPECTED_LDGSTS;
pub const RETAINED_BLOCK_THREADS: u32 = 256;
pub const RETAINED_DYNAMIC_SHARED_BYTES: usize = 73_728;
pub const RETAINED_MAX_REGISTERS: i32 = 123;
pub const RETAINED_REQUIRED_OCCUPANCY: u32 = 1;
pub const RETAINED_EXPECTED_HMMA: usize = 64;
pub const RETAINED_EXPECTED_LDGSTS: usize = 18;

pub fn retained_source(swizzle: &str, s3: &str) -> Result<String, String> {
    bf16_parent::measured_s3_source(swizzle, s3)
}

pub fn candidate_source(swizzle: &str, s3: &str) -> Result<String, String> {
    let mut source = f16_parent::candidate_source(swizzle, s3)?;
    replace_exact(&mut source, F16_EXPORT, BF16_EXPORT)?;
    require_count(&source, SYMBOL, 1)?;
    require_count(&source, f16_parent::SYMBOL, 0)?;
    if !has_m96n128_geometry(&source) {
        return Err("BF16 M96N128 transform lost frozen physical geometry".into());
    }
    Ok(source)
}

pub fn restore_f16_source(candidate: &str, swizzle: &str, s3: &str) -> Result<String, String> {
    let expected = candidate_source(swizzle, s3)?;
    if candidate != expected {
        return Err("BF16 M96N128 candidate differs from fail-closed transform".into());
    }
    let mut restored = candidate.to_owned();
    replace_exact(&mut restored, BF16_EXPORT, F16_EXPORT)?;
    Ok(restored)
}

pub fn has_m96n128_geometry(source: &str) -> bool {
    f16_parent::has_m96n128_geometry(source)
        || [
            "namespace sm89_test_half_nt_m96n128_s3",
            "static constexpr int kSharedBytes = 86016;",
            "T (*As)[96][64]",
            "T (*Bs)[128][64]",
            "int num_pid_n = (K_out + 127) / 128;",
            "int warpM = (warp >> 2) * 32;",
            "int warpN = (warp & 3) * 32;",
            "for (int slice = 0; slice < 5; ++slice)",
            "static constexpr int kM96N128OutputStride = 136;",
            SYMBOL,
        ]
        .iter()
        .all(|anchor| source.contains(anchor))
}

pub const fn target_grid() -> u32 {
    f16_parent::target_grid()
}

pub const fn staged_half_elements_per_k_tile_m96n128() -> usize {
    f16_parent::staged_half_elements_per_k_tile_m96n128()
}

pub const fn staged_half_elements_per_k_tile_m64n192() -> usize {
    f16_parent::staged_half_elements_per_k_tile_m64n192()
}

pub fn all_strata_pass(strata: &[[f64; 2]]) -> bool {
    f16_parent::all_strata_pass(strata)
}

fn replace_exact(source: &mut String, before: &str, after: &str) -> Result<(), String> {
    require_count(source, before, 1)?;
    *source = source.replacen(before, after, 1);
    Ok(())
}

fn require_count(source: &str, needle: &str, expected: usize) -> Result<(), String> {
    let actual = source.matches(needle).count();
    if actual == expected {
        Ok(())
    } else {
        Err(format!(
            "BF16 M96N128 source anchor {needle:?}: expected {expected}, observed {actual}"
        ))
    }
}

const F16_EXPORT: &str = r#"extern "C" __global__ __launch_bounds__(384, 1)
void gemm_bi_nt_test_fixed_s3_m96n128_f16(
    __half* __restrict__ C, const __half* __restrict__ A,
    const __half* __restrict__ B, float alpha, int M, int N, int K_out) {
    sm89_test_half_nt_m96n128_s3::kernel<__half>(C, A, B, alpha, M, N, K_out);
}"#;

const BF16_EXPORT: &str = r#"extern "C" __global__ __launch_bounds__(384, 1)
void gemm_bi_nt_test_fixed_s3_m96n128_bf16(
    __nv_bfloat16* __restrict__ C, const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B, float alpha, int M, int N, int K_out) {
    sm89_test_half_nt_m96n128_s3::kernel<__nv_bfloat16>(C, A, B, alpha, M, N, K_out);
}"#;

#[cfg(test)]
mod tests {
    use super::*;

    const SWIZZLE: &str = include_str!("../../kernels/gemm_bi_inference/sm89_half_swizzle.cu");
    const S3: &str = include_str!("../../kernels/gemm_bi_inference/sm89_half_s3.cu");

    #[test]
    fn adapter_changes_only_the_exported_scalar_type() {
        let f16 = f16_parent::candidate_source(SWIZZLE, S3).unwrap();
        let bf16 = candidate_source(SWIZZLE, S3).unwrap();
        assert_eq!(restore_f16_source(&bf16, SWIZZLE, S3).unwrap(), f16);
        assert!(bf16.contains("kernel<__nv_bfloat16>"));
        assert!(!bf16.contains(f16_parent::SYMBOL));
    }

    #[test]
    fn retained_comparator_is_the_measured_bf16_m64n128_s3_body() {
        let retained = retained_source(SWIZZLE, S3).unwrap();
        assert!(retained.contains("void gemm_bi_nt_test_fixed_s3_m64n128_##SUFFIX"));
        assert!(retained.contains("SM89_HALF_NT_S3_EXPORT(__nv_bfloat16, bf16)"));
        assert_eq!(RETAINED_SYMBOL, "gemm_bi_nt_test_fixed_s3_m64n128_bf16");
    }

    #[test]
    fn malformed_parent_inputs_fail_closed() {
        assert!(candidate_source("", S3).is_err());
        assert!(candidate_source(SWIZZLE, "").is_err());
    }
}
