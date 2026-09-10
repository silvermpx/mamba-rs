pub const DIRECT_DUAL_RAW_SYMBOL: &str = "gemm_bi_tn_test_direct_sm89_f32_n64_dual_chunk_raw_v1";
pub const DIRECT_DUAL_FUSED_SYMBOL: &str =
    "gemm_bi_tn_test_direct_sm89_f32_n64_dual_chunk_fused_finalize_v1";

const PRELUDE: &str = include_str!("../../kernels/_typed_prelude.cuh");

const SOURCE_TEMPLATE: &str = r#"
struct DirectDualChunkParams {
    float alpha;
    int m, n, k0, k1, ldx, lddy, ldc;
};
static_assert(sizeof(DirectDualChunkParams) == 32, "direct dual parameter size");
static_assert(alignof(DirectDualChunkParams) == 4, "direct dual parameter alignment");

#define DIRECT_DUAL_BM 64
#define DIRECT_DUAL_BN 64
#define DIRECT_DUAL_BK 32
#define DIRECT_DUAL_GROUP_M 8
#define DIRECT_DUAL_THREADS 128
#define DIRECT_DUAL_A_STAGE (DIRECT_DUAL_BK * DIRECT_DUAL_BM)
#define DIRECT_DUAL_B_STAGE (DIRECT_DUAL_BK * DIRECT_DUAL_BN)
static_assert(2 * (DIRECT_DUAL_A_STAGE + DIRECT_DUAL_B_STAGE) * sizeof(float)
              == 32768, "direct dual shared memory changed");

__device__ __forceinline__ bool direct_dual_aligned16(const void* pointer) {
    return (reinterpret_cast<unsigned long long>(pointer) & 15ULL) == 0ULL;
}

#define DIRECT_DUAL_ASYNC_STAGE(STAGE, K_TILE)                               \
    do {                                                                      \
        const unsigned _a_stage = a_base                                      \
            + (unsigned)(STAGE) * DIRECT_DUAL_A_STAGE * 4U;                  \
        const unsigned _b_stage = b_base                                      \
            + (unsigned)(STAGE) * DIRECT_DUAL_B_STAGE * 4U;                  \
        for (int _i = tid;                                                    \
             _i < DIRECT_DUAL_BK * (DIRECT_DUAL_BM / 4);                     \
             _i += DIRECT_DUAL_THREADS) {                                    \
            const int _kr = _i / (DIRECT_DUAL_BM / 4);                       \
            const int _mc = (_i % (DIRECT_DUAL_BM / 4)) * 4;                 \
            const int _global_r = chain_begin + (K_TILE) + _kr;              \
            const int _global_m = row0 + _mc;                                \
            const int _remaining = m - _global_m;                            \
            const int _elements = _remaining >= 4                            \
                ? 4 : (_remaining > 0 ? _remaining : 0);                     \
            const int _bytes = _global_r < chain_end ? _elements * 4 : 0;    \
            const unsigned _destination = _a_stage                           \
                + (unsigned)(_kr * DIRECT_DUAL_BM + _mc) * 4U;              \
            const float* _source = _bytes > 0                                \
                ? x + (long long)_global_r * ldx + _global_m : x;            \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" \
                         :: "r"(_destination), "l"(_source), "r"(_bytes));\
        }                                                                     \
        for (int _i = tid;                                                    \
             _i < DIRECT_DUAL_BK * (DIRECT_DUAL_BN / 4);                     \
             _i += DIRECT_DUAL_THREADS) {                                    \
            const int _kr = _i / (DIRECT_DUAL_BN / 4);                       \
            const int _nc = (_i % (DIRECT_DUAL_BN / 4)) * 4;                 \
            const int _global_r = chain_begin + (K_TILE) + _kr;              \
            const int _global_n = col0 + _nc;                                \
            const int _remaining = n - _global_n;                            \
            const int _elements = _remaining >= 4                            \
                ? 4 : (_remaining > 0 ? _remaining : 0);                     \
            const int _bytes = _global_r < chain_end ? _elements * 4 : 0;    \
            const unsigned _destination = _b_stage                           \
                + (unsigned)(_kr * DIRECT_DUAL_BN + _nc) * 4U;              \
            const float* _source = _bytes > 0                                \
                ? dy + (long long)_global_r * lddy + _global_n : dy;         \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" \
                         :: "r"(_destination), "l"(_source), "r"(_bytes));\
        }                                                                     \
        asm volatile("cp.async.commit_group;\n");                           \
    } while (0)

#define DIRECT_DUAL_SCALAR_STAGE(STAGE, K_TILE)                              \
    do {                                                                      \
        for (int _i = tid; _i < DIRECT_DUAL_A_STAGE;                         \
             _i += DIRECT_DUAL_THREADS) {                                    \
            const int _kr = _i / DIRECT_DUAL_BM;                             \
            const int _mc = _i % DIRECT_DUAL_BM;                             \
            const int _global_r = chain_begin + (K_TILE) + _kr;              \
            const int _global_m = row0 + _mc;                                \
            smem_a[(STAGE)][_i] =                                            \
                (_global_r < chain_end && _global_m < m)                     \
                    ? x[(long long)_global_r * ldx + _global_m] : 0.0f;      \
        }                                                                     \
        for (int _i = tid; _i < DIRECT_DUAL_B_STAGE;                         \
             _i += DIRECT_DUAL_THREADS) {                                    \
            const int _kr = _i / DIRECT_DUAL_BN;                             \
            const int _nc = _i % DIRECT_DUAL_BN;                             \
            const int _global_r = chain_begin + (K_TILE) + _kr;              \
            const int _global_n = col0 + _nc;                                \
            smem_b[(STAGE)][_i] =                                            \
                (_global_r < chain_end && _global_n < n)                     \
                    ? dy[(long long)_global_r * lddy + _global_n] : 0.0f;    \
        }                                                                     \
    } while (0)

#define DIRECT_DUAL_RUN_CHAIN(ACC, BEGIN, K_LEN)                             \
    do {                                                                      \
        _Pragma("unroll")                                                    \
        for (int _i = 0; _i < 8; ++_i) {                                    \
            _Pragma("unroll")                                                \
            for (int _j = 0; _j < 4; ++_j) (ACC)[_i][_j] = 0.0f;            \
        }                                                                     \
        const int chain_begin = (BEGIN);                                      \
        const int chain_k = (K_LEN);                                          \
        const int chain_end = chain_begin + chain_k;                          \
        const int chain_num_tiles =                                           \
            (chain_k + DIRECT_DUAL_BK - 1) / DIRECT_DUAL_BK;                 \
        const bool chain_async = direct_dual_aligned16(x)                     \
            && direct_dual_aligned16(dy) && (ldx & 3) == 0                   \
            && (lddy & 3) == 0;                                              \
        if (chain_num_tiles > 0) {                                            \
            if (chain_async) DIRECT_DUAL_ASYNC_STAGE(0, 0);                  \
            else DIRECT_DUAL_SCALAR_STAGE(0, 0);                             \
        }                                                                     \
        int chain_read_stage = 0;                                             \
        for (int chain_tile = 0; chain_tile < chain_num_tiles; ++chain_tile) {\
            if (chain_async) asm volatile("cp.async.wait_group 0;\n");      \
            __syncthreads();                                                  \
            if (chain_tile + 1 < chain_num_tiles) {                          \
                const int chain_next_k =                                     \
                    (chain_tile + 1) * DIRECT_DUAL_BK;                       \
                if (chain_async)                                             \
                    DIRECT_DUAL_ASYNC_STAGE(chain_read_stage ^ 1, chain_next_k);\
                else DIRECT_DUAL_SCALAR_STAGE(chain_read_stage ^ 1, chain_next_k);\
            }                                                                 \
            _Pragma("unroll")                                                \
            for (int kk = 0; kk < DIRECT_DUAL_BK; ++kk) {                   \
                float a_reg[8];                                               \
                float b_reg[4];                                               \
                _Pragma("unroll")                                            \
                for (int i = 0; i < 8; ++i)                                  \
                    a_reg[i] = smem_a[chain_read_stage]                       \
                        [kk * DIRECT_DUAL_BM + ty * 8 + i];                  \
                _Pragma("unroll")                                            \
                for (int j = 0; j < 4; ++j)                                  \
                    b_reg[j] = smem_b[chain_read_stage]                       \
                        [kk * DIRECT_DUAL_BN + tx * 4 + j];                  \
                _Pragma("unroll")                                            \
                for (int i = 0; i < 8; ++i) {                                \
                    _Pragma("unroll")                                        \
                    for (int j = 0; j < 4; ++j)                              \
                        (ACC)[i][j] = __fmaf_rn(                              \
                            a_reg[i], b_reg[j], (ACC)[i][j]);                \
                }                                                             \
            }                                                                 \
            chain_read_stage ^= 1;                                            \
        }                                                                     \
    } while (0)

extern "C" __global__ __launch_bounds__(128, 3)
void __DIRECT_DUAL_SYMBOL__(
    float* __restrict__ output,
    const float* __restrict__ x,
    const float* __restrict__ dy,
    DirectDualChunkParams params
) {
    const float alpha = params.alpha;
    const int m = params.m, n = params.n;
    const int k0 = params.k0, k1 = params.k1;
    const int ldx = params.ldx, lddy = params.lddy, ldc = params.ldc;
    __align__(16) __shared__ float smem_a[2][DIRECT_DUAL_A_STAGE];
    __align__(16) __shared__ float smem_b[2][DIRECT_DUAL_B_STAGE];

    const int num_pid_m = (m + DIRECT_DUAL_BM - 1) / DIRECT_DUAL_BM;
    const int num_pid_n = (n + DIRECT_DUAL_BN - 1) / DIRECT_DUAL_BN;
    const int num_pid_in_group = DIRECT_DUAL_GROUP_M * num_pid_n;
    const int group_id = blockIdx.x / num_pid_in_group;
    const int first_pid_m = group_id * DIRECT_DUAL_GROUP_M;
    const int group_size_m = min(num_pid_m - first_pid_m, DIRECT_DUAL_GROUP_M);
    const int pid_m =
        first_pid_m + ((blockIdx.x % num_pid_in_group) % group_size_m);
    const int pid_n = (blockIdx.x % num_pid_in_group) / group_size_m;
    const int row0 = pid_m * DIRECT_DUAL_BM;
    const int col0 = pid_n * DIRECT_DUAL_BN;
    const int tid = (int)threadIdx.x;
    const int tx = tid & 15;
    const int ty = tid >> 4;
    const int row_base = row0 + ty * 8;
    const int col_base = col0 + tx * 4;
    const unsigned a_base = (unsigned)__cvta_generic_to_shared(&smem_a[0][0]);
    const unsigned b_base = (unsigned)__cvta_generic_to_shared(&smem_b[0][0]);

    float acc0[8][4];
    float acc1[8][4];
    DIRECT_DUAL_RUN_CHAIN(acc0, 0, k0);
    __syncthreads();
    DIRECT_DUAL_RUN_CHAIN(acc1, k0, k1);

__DIRECT_DUAL_EPILOGUE__
}

#undef DIRECT_DUAL_RUN_CHAIN
#undef DIRECT_DUAL_SCALAR_STAGE
#undef DIRECT_DUAL_ASYNC_STAGE
#undef DIRECT_DUAL_THREADS
#undef DIRECT_DUAL_GROUP_M
#undef DIRECT_DUAL_B_STAGE
#undef DIRECT_DUAL_A_STAGE
#undef DIRECT_DUAL_BK
#undef DIRECT_DUAL_BN
#undef DIRECT_DUAL_BM
"#;

const RAW_EPILOGUE: &str = r#"    float* plane0 = output;
    float* plane1 = output + (long long)m * ldc;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int r = row_base + i;
        if (r >= m) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int col = col_base + j;
            if (col >= n) continue;
            const long long idx = (long long)r * ldc + col;
            plane0[idx] = acc0[i][j];
            plane1[idx] = acc1[i][j];
        }
    }"#;

const FUSED_EPILOGUE: &str = r#"#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int r = row_base + i;
        if (r >= m) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int col = col_base + j;
            if (col >= n) continue;
            const long long idx = (long long)r * ldc + col;
            const double sum = __dadd_rn((double)acc0[i][j], (double)acc1[i][j]);
            const float update = __double2float_rn(__dmul_rn((double)alpha, sum));
            output[idx] = __fadd_rn(output[idx], update);
        }
    }"#;

fn compose(symbol: &str, epilogue: &str) -> String {
    format!(
        "{PRELUDE}\n{}",
        SOURCE_TEMPLATE
            .replace("__DIRECT_DUAL_SYMBOL__", symbol)
            .replace("__DIRECT_DUAL_EPILOGUE__", epilogue)
    )
}

pub fn compose_raw_source() -> String {
    compose(DIRECT_DUAL_RAW_SYMBOL, RAW_EPILOGUE)
}

pub fn compose_fused_source() -> String {
    compose(DIRECT_DUAL_FUSED_SYMBOL, FUSED_EPILOGUE)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_common(source: &str, symbol: &str) {
        assert_eq!(source.matches(symbol).count(), 1);
        assert!(source.contains("__launch_bounds__(128, 3)"));
        assert!(source.contains("#define DIRECT_DUAL_BM 64"));
        assert!(source.contains("#define DIRECT_DUAL_BN 64"));
        assert!(source.contains("#define DIRECT_DUAL_BK 32"));
        assert!(source.contains("#define DIRECT_DUAL_GROUP_M 8"));
        assert!(source.contains("float acc0[8][4]"));
        assert!(source.contains("float acc1[8][4]"));
        assert!(source.contains("x + (long long)_global_r * ldx + _global_m"));
        assert!(source.contains("dy + (long long)_global_r * lddy + _global_n"));
        assert!(source.contains("cp.async.cg.shared.global"));
        assert!(source.contains("DIRECT_DUAL_RUN_CHAIN(acc0, 0, k0)"));
        assert!(source.contains("DIRECT_DUAL_RUN_CHAIN(acc1, k0, k1)"));
        assert_eq!(source.matches("(ACC)[i][j] = __fmaf_rn(").count(), 1);
        assert!(!source.contains("blockIdx.z"));
        assert!(!source.contains("mma.sync"));
        assert!(!source.contains("atomic"));
    }

    #[test]
    fn raw_twin_has_two_unscaled_planes_and_one_cta_per_output_tile() {
        let source = compose_raw_source();
        assert_common(&source, DIRECT_DUAL_RAW_SYMBOL);
        assert!(source.contains("float* plane0 = output"));
        assert!(source.contains("float* plane1 = output + (long long)m * ldc"));
        assert!(source.contains("plane0[idx] = acc0[i][j]"));
        assert!(source.contains("plane1[idx] = acc1[i][j]"));
    }

    #[test]
    fn fused_twin_pins_exact_fp64_finalize_order() {
        let source = compose_fused_source();
        assert_common(&source, DIRECT_DUAL_FUSED_SYMBOL);
        assert!(source.contains("__dadd_rn((double)acc0[i][j], (double)acc1[i][j])"));
        assert!(source.contains("__dmul_rn((double)alpha, sum)"));
        assert!(source.contains("__double2float_rn"));
        assert!(source.contains("output[idx] = __fadd_rn(output[idx], update)"));
    }

    #[test]
    fn compact_abi_carries_both_exact_chunk_lengths() {
        let source = compose_fused_source();
        assert!(source.contains("static_assert(sizeof(DirectDualChunkParams) == 32"));
        assert!(source.contains("int m, n, k0, k1, ldx, lddy, ldc"));
    }
}
