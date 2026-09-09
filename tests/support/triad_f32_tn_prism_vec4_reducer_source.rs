pub const SYMBOL: &str = "gemm_bi_tn_test_prism_sm89_f32_splitm_reduce_vec4_v1";

pub fn compose_source() -> String {
    SOURCE.to_owned()
}

const SOURCE: &str = r#"
#define PRISM_REDUCE_THREADS 128
#define PRISM_REDUCE_CHUNKS 6

extern "C" __global__ __launch_bounds__(PRISM_REDUCE_THREADS, 6)
void gemm_bi_tn_test_prism_sm89_f32_splitm_reduce_vec4_v1(
    float* __restrict__ output,
    const float* __restrict__ partial,
    float alpha,
    int K_out,
    int N,
    int chunks
) {
    const long long stride = (long long)K_out * N;
    if (chunks != PRISM_REDUCE_CHUNKS || (stride & 3LL) != 0
        || (reinterpret_cast<unsigned long long>(output) & 15ULL) != 0
        || (reinterpret_cast<unsigned long long>(partial) & 15ULL) != 0) return;
    const long long element =
        ((long long)blockIdx.x * blockDim.x + threadIdx.x) * 4LL;
    if (element >= stride) return;

    const float4 first = *reinterpret_cast<const float4*>(partial + element);
    double sum0 = (double)first.x;
    double sum1 = (double)first.y;
    double sum2 = (double)first.z;
    double sum3 = (double)first.w;
#pragma unroll
    for (int fc = 1; fc < PRISM_REDUCE_CHUNKS; ++fc) {
        const float4 value = *reinterpret_cast<const float4*>(
            partial + (long long)fc * stride + element);
        sum0 = __dadd_rn(sum0, (double)value.x);
        sum1 = __dadd_rn(sum1, (double)value.y);
        sum2 = __dadd_rn(sum2, (double)value.z);
        sum3 = __dadd_rn(sum3, (double)value.w);
    }
    const float4 previous = *reinterpret_cast<const float4*>(output + element);
    const float4 result = {
        __fadd_rn(previous.x, __double2float_rn(__dmul_rn((double)alpha, sum0))),
        __fadd_rn(previous.y, __double2float_rn(__dmul_rn((double)alpha, sum1))),
        __fadd_rn(previous.z, __double2float_rn(__dmul_rn((double)alpha, sum2))),
        __fadd_rn(previous.w, __double2float_rn(__dmul_rn((double)alpha, sum3)))
    };
    *reinterpret_cast<float4*>(output + element) = result;
}

#undef PRISM_REDUCE_THREADS
#undef PRISM_REDUCE_CHUNKS
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_pins_vec4_memory_and_six_ordered_fp64_chains() {
        let source = compose_source();
        assert_eq!(source.matches(SYMBOL).count(), 1);
        assert!(source.contains("__launch_bounds__(PRISM_REDUCE_THREADS, 6)"));
        assert!(source.contains("const float4 first"));
        assert!(source.contains("for (int fc = 1; fc < PRISM_REDUCE_CHUNKS; ++fc)"));
        for name in ["sum0", "sum1", "sum2", "sum3"] {
            assert!(source.contains(&format!("__dadd_rn({name},")));
            assert!(source.contains(&format!("__dmul_rn((double)alpha, {name})")));
        }
        assert!(source.contains("*reinterpret_cast<float4*>(output + element) = result"));
        assert!(!source.contains("atom."));
        assert!(!source.contains("mma.sync"));
    }

    #[test]
    fn source_fails_closed_outside_six_aligned_planes() {
        let source = compose_source();
        assert!(source.contains("chunks != PRISM_REDUCE_CHUNKS"));
        assert!(source.contains("(stride & 3LL) != 0"));
        assert!(source.contains("partial) & 15ULL"));
        assert!(source.contains("output) & 15ULL"));
    }
}
