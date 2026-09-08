pub const SYMBOL: &str = "gemm_bi_tn_test_copyplan_epilogue_f32_v1";

pub fn compose_source() -> Result<String, String> {
    Ok(format!(
        r#"
extern "C" __global__ __launch_bounds__(256, 4)
void {SYMBOL}(
    float* __restrict__ output,
    const float* __restrict__ accumulator,
    float alpha,
    unsigned long long elements
) {{
    unsigned long long linear =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (linear < elements) {{
        output[linear] = __fmaf_rn(alpha, accumulator[linear], output[linear]);
    }}
}}
"#
    ))
}
