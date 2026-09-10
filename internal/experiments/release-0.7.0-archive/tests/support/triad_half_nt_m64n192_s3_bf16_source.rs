#[path = "triad_half_nt_m64n192_s3_source.rs"]
mod parent;

pub const SYMBOL: &str = "gemm_bi_nt_test_fixed_s3_m64n192_bf16";
pub const RETAINED_SYMBOL: &str = "gemm_bi_nt_test_fixed_s3_m64n128_bf16";
pub const BLOCK_THREADS: u32 = parent::BLOCK_THREADS;
pub const DYNAMIC_SHARED_BYTES: usize = parent::DYNAMIC_SHARED_BYTES;
pub const MAX_REGISTERS: i32 = parent::MAX_REGISTERS;
pub const REQUIRED_OCCUPANCY: u32 = parent::REQUIRED_OCCUPANCY;

pub fn candidate_source(swizzle: &str, s3: &str) -> Result<String, String> {
    let mut source = parent::candidate_source(swizzle, s3)?;
    replace_exact(&mut source, F16_EXPORT, BF16_EXPORT)?;
    require_count(&source, SYMBOL, 1)?;
    require_count(&source, parent::SYMBOL, 0)?;
    Ok(source)
}

pub fn measured_s3_source(swizzle: &str, s3: &str) -> Result<String, String> {
    let source = parent::measured_s3_source(swizzle, s3)?;
    require_count(&source, "void gemm_bi_nt_test_fixed_s3_m64n128_##SUFFIX", 1)?;
    require_count(&source, "SM89_HALF_NT_S3_EXPORT(__nv_bfloat16, bf16)", 1)?;
    Ok(source)
}

pub fn all_strata_pass(strata: &[[f64; 2]]) -> bool {
    parent::all_strata_pass(strata)
}

const F16_EXPORT: &str = r#"extern "C" __global__ __launch_bounds__(384, 1)
void gemm_bi_nt_test_fixed_s3_m64n192_f16(
    __half* __restrict__ C, const __half* __restrict__ A,
    const __half* __restrict__ B, float alpha, int M, int N, int K_out) {
    sm89_test_half_nt_m64n192_s3::kernel<__half>(C, A, B, alpha, M, N, K_out);
}
"#;

const BF16_EXPORT: &str = r#"extern "C" __global__ __launch_bounds__(384, 1)
void gemm_bi_nt_test_fixed_s3_m64n192_bf16(
    __nv_bfloat16* __restrict__ C, const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B, float alpha, int M, int N, int K_out) {
    sm89_test_half_nt_m64n192_s3::kernel<__nv_bfloat16>(C, A, B, alpha, M, N, K_out);
}
"#;

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
            "BF16 M64N192 source anchor {needle:?}: expected {expected}, observed {actual}"
        ))
    }
}
