pub const PRODUCTION_SYMBOL: &str = "gemm_bi_nn_fixed_sm89_f32_n64_copyplan_v1";
pub const FUSED_SYMBOL: &str = "gemm_bi_tn_test_fixed_sm89_f32_n64_copyplan_fused_finalize_v1";

const PRELUDE: &str = include_str!("../../kernels/_typed_prelude.cuh");
const PRODUCTION: &str = include_str!("../../kernels/gemm_bi_fixed/sm89_f32_n64_copyplan.cu");

fn replace_once(source: &mut String, before: &str, after: &str) -> Result<(), String> {
    let count = source.matches(before).count();
    if count != 1 {
        return Err(format!(
            "fused-finalize CopyPlan anchor expected1 observed{count}"
        ));
    }
    *source = source.replacen(before, after, 1);
    Ok(())
}

pub fn compose_source() -> Result<String, String> {
    let mut body = PRODUCTION.to_owned();
    replace_once(&mut body, PRODUCTION_SYMBOL, FUSED_SYMBOL)?;
    replace_once(
        &mut body,
        "    float* __restrict__ c,",
        "    float* __restrict__ dW,",
    )?;
    replace_once(&mut body, PRODUCTION_EPILOGUE, FUSED_EPILOGUE)?;
    Ok(format!(
        "{PRELUDE}\n#define gbf_aligned16 gemm_bi_tn_fused_aligned16\n__device__ __forceinline__ bool gemm_bi_tn_fused_aligned16(const void* p) {{ return (reinterpret_cast<unsigned long long>(p) & 15ULL) == 0ULL; }}\n{body}\n#undef gbf_aligned16\n"
    ))
}

const PRODUCTION_EPILOGUE: &str = r#"    bool pair_store_fast = row0 <= m - SM89_EXACT_N64_CP_BM
        && col0 <= n - SM89_EXACT_N64_CP_BN
        && (reinterpret_cast<unsigned long long>(c) & 7ULL) == 0
        && (ldc & 1) == 0;
    if (pair_store_fast) {
#pragma unroll
        for (int i = 0; i < 8; i++) {
            int r = row_base + i;
#pragma unroll
            for (int j = 0; j < 4; j += 2) {
                int col = col_base + j;
                float val0 = __fmul_rn(alpha, acc[i][j]);
                float val1 = __fmul_rn(alpha, acc[i][j + 1]);
                if (bias != nullptr) {
                    val0 = __fadd_rn(val0, bias[col]);
                    val1 = __fadd_rn(val1, bias[col + 1]);
                }
                if (beta != 0.0f) {
                    val0 = __fmaf_rn(beta, c[(long long)r * ldc + col], val0);
                    val1 = __fmaf_rn(beta, c[(long long)r * ldc + col + 1], val1);
                }
                float2 pair = {val0, val1};
                *reinterpret_cast<float2*>(c + (long long)r * ldc + col) = pair;
            }
        }
        return;
    }

#pragma unroll
    for (int i = 0; i < 8; i++) {
        int r = row_base + i;
        if (r >= m) continue;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int col = col_base + j;
            if (col >= n) continue;
            float val = __fmul_rn(alpha, acc[i][j]);
            if (bias != nullptr) val = __fadd_rn(val, bias[col]);
            if (beta != 0.0f)
                val = __fmaf_rn(beta, c[(long long)r * ldc + col], val);
            c[(long long)r * ldc + col] = val;
        }
    }
"#;

const FUSED_EPILOGUE: &str = r#"    const float* partial0 = bias;
    if (partial0 == nullptr || beta != 1.0f) return;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int r = row_base + i;
        if (r >= m) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int col = col_base + j;
            if (col >= n) continue;
            const int idx = r * ldc + col;
            const double sum = __dadd_rn(
                (double)partial0[idx], (double)acc[i][j]);
            const float update = __double2float_rn(
                __dmul_rn((double)alpha, sum));
            dW[idx] = __fadd_rn(dW[idx], update);
        }
    }
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fused_finalize_keeps_copyplan_mainloop_and_replaces_only_epilogue() {
        let source = compose_source().unwrap();
        let normalized = source.split_whitespace().collect::<Vec<_>>().join(" ");
        assert_eq!(source.matches(FUSED_SYMBOL).count(), 1);
        assert!(!source.contains(PRODUCTION_SYMBOL));
        assert_eq!(
            source
                .matches("__fmaf_rn(a_reg[i], b_reg[j], acc[i][j])")
                .count(),
            2
        );
        assert!(normalized.contains("__dadd_rn( (double)partial0[idx], (double)acc[i][j])"));
        assert!(source.contains("__dmul_rn((double)alpha, sum)"));
        assert!(source.contains("__double2float_rn"));
        assert!(source.contains("dW[idx] = __fadd_rn(dW[idx], update)"));
        assert!(!source.contains("__fmul_rn(alpha, acc[i][j])"));
    }

    #[test]
    fn fused_finalize_pins_partial0_and_beta_one_contract() {
        let source = compose_source().unwrap();
        assert!(source.contains("const float* partial0 = bias"));
        assert!(source.contains("if (partial0 == nullptr || beta != 1.0f) return"));
        assert!(source.contains("const int idx = r * ldc + col"));
        assert!(source.contains("static_assert(sizeof(FixedSm89ExactF32Params) == 32"));
    }
}
