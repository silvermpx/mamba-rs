pub const RAW_SYMBOL: &str = "gemm_bi_tn_test_fixed_sm89_f32_m128n64_bk16_dual_chunk_raw_v1";
pub const FUSED_SYMBOL: &str = "gemm_bi_tn_test_fixed_sm89_f32_m128n64_bk16_dual_chunk_fused_v1";

const PRELUDE: &str = include_str!("../../kernels/_typed_prelude.cuh");

const TEMPLATE: &str = r#"
struct M128DualParams {
    float alpha;
    int m, n, k0, k1, lda, ldb, ldc;
};
static_assert(sizeof(M128DualParams) == 32, "M128 dual parameter size");
static_assert(alignof(M128DualParams) == 4, "M128 dual parameter alignment");

#define M128_DUAL_BM 128
#define M128_DUAL_BN 64
#define M128_DUAL_BK 16
#define M128_DUAL_THREADS 256
#define M128_DUAL_GROUP_M 6
#define M128_DUAL_A_STAGE (M128_DUAL_BM * M128_DUAL_BK)
#define M128_DUAL_B_STAGE (M128_DUAL_BK * M128_DUAL_BN)
#define M128_DUAL_SMEM_BYTES 24576
static_assert(
    2 * (M128_DUAL_A_STAGE + M128_DUAL_B_STAGE) * sizeof(float)
        == M128_DUAL_SMEM_BYTES,
    "M128 dual shared memory changed");

__device__ __forceinline__ bool m128_dual_aligned16(const void* pointer) {
    return (reinterpret_cast<unsigned long long>(pointer) & 15ULL) == 0ULL;
}

#define M128_DUAL_ASYNC_STAGE(STAGE, K_TILE) do {                              \
        const unsigned _a_stage = a_base                                      \
            + (unsigned)(STAGE) * M128_DUAL_A_STAGE * 4U;                     \
        const unsigned _b_stage = b_base                                      \
            + (unsigned)(STAGE) * M128_DUAL_B_STAGE * 4U;                     \
        for (int _i = tid;                                                     \
             _i < M128_DUAL_BM * (M128_DUAL_BK / 4);                          \
             _i += M128_DUAL_THREADS) {                                       \
            const int _mr = _i / (M128_DUAL_BK / 4);                          \
            const int _kc = (_i % (M128_DUAL_BK / 4)) * 4;                    \
            const int _global_m = row0 + _mr;                                 \
            const int _global_k = (K_TILE) + _kc;                             \
            const int _remaining = chain_k - _global_k;                       \
            const int _elements = _remaining >= 4                             \
                ? 4 : (_remaining > 0 ? _remaining : 0);                      \
            const int _bytes = _global_m < m ? _elements * 4 : 0;             \
            const unsigned _destination = _a_stage                            \
                + (unsigned)(_mr * M128_DUAL_BK + _kc) * 4U;                  \
            const float* _source = _bytes > 0                                 \
                ? chain_a + (long long)_global_m * lda + _global_k : chain_a; \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"  \
                         :: "r"(_destination), "l"(_source), "r"(_bytes)); \
        }                                                                      \
        for (int _i = tid;                                                     \
             _i < M128_DUAL_BK * (M128_DUAL_BN / 4);                          \
             _i += M128_DUAL_THREADS) {                                       \
            const int _kr = _i / (M128_DUAL_BN / 4);                          \
            const int _nc = (_i % (M128_DUAL_BN / 4)) * 4;                    \
            const int _global_k = (K_TILE) + _kr;                             \
            const int _global_n = col0 + _nc;                                 \
            const int _remaining = n - _global_n;                             \
            const int _elements = _remaining >= 4                             \
                ? 4 : (_remaining > 0 ? _remaining : 0);                      \
            const int _bytes = _global_k < chain_k ? _elements * 4 : 0;       \
            const unsigned _destination = _b_stage                            \
                + (unsigned)(_kr * M128_DUAL_BN + _nc) * 4U;                  \
            const float* _source = _bytes > 0                                 \
                ? chain_b + (long long)_global_k * ldb + _global_n : chain_b; \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"  \
                         :: "r"(_destination), "l"(_source), "r"(_bytes)); \
        }                                                                      \
        asm volatile("cp.async.commit_group;\n");                            \
    } while (0)

#define M128_DUAL_SCALAR_STAGE(STAGE, K_TILE) do {                            \
        for (int _i = tid; _i < M128_DUAL_A_STAGE;                            \
             _i += M128_DUAL_THREADS) {                                       \
            const int _mr = _i / M128_DUAL_BK;                                \
            const int _kc = _i % M128_DUAL_BK;                                \
            const int _global_m = row0 + _mr;                                 \
            const int _global_k = (K_TILE) + _kc;                             \
            smem_a[(STAGE)][_i] =                                             \
                (_global_m < m && _global_k < chain_k)                        \
                    ? chain_a[(long long)_global_m * lda + _global_k] : 0.0f; \
        }                                                                      \
        for (int _i = tid; _i < M128_DUAL_B_STAGE;                            \
             _i += M128_DUAL_THREADS) {                                       \
            const int _kr = _i / M128_DUAL_BN;                                \
            const int _nc = _i % M128_DUAL_BN;                                \
            const int _global_k = (K_TILE) + _kr;                             \
            const int _global_n = col0 + _nc;                                 \
            smem_b[(STAGE)][_i] =                                             \
                (_global_k < chain_k && _global_n < n)                        \
                    ? chain_b[(long long)_global_k * ldb + _global_n] : 0.0f; \
        }                                                                      \
    } while (0)

#define M128_DUAL_RUN_CHAIN(ACC, A_PTR, B_PTR, K_LEN) do {                    \
        _Pragma("unroll")                                                    \
        for (int _i = 0; _i < 8; ++_i) {                                     \
            _Pragma("unroll")                                                \
            for (int _j = 0; _j < 4; ++_j) (ACC)[_i][_j] = 0.0f;             \
        }                                                                      \
        const float* chain_a = (A_PTR);                                        \
        const float* chain_b = (B_PTR);                                        \
        const int chain_k = (K_LEN);                                           \
        const int chain_tiles =                                                \
            (chain_k + M128_DUAL_BK - 1) / M128_DUAL_BK;                     \
        const bool chain_async = m128_dual_aligned16(chain_a)                  \
            && m128_dual_aligned16(chain_b) && (lda & 3) == 0                 \
            && (ldb & 3) == 0;                                                \
        if (chain_tiles > 0) {                                                 \
            if (chain_async) M128_DUAL_ASYNC_STAGE(0, 0);                     \
            else M128_DUAL_SCALAR_STAGE(0, 0);                               \
        }                                                                      \
        int chain_read_stage = 0;                                              \
        for (int chain_tile = 0; chain_tile < chain_tiles; ++chain_tile) {    \
            if (chain_async) asm volatile("cp.async.wait_group 0;\n");      \
            __syncthreads();                                                   \
            const int chain_next_k = (chain_tile + 1) * M128_DUAL_BK;         \
            if (chain_tile + 1 < chain_tiles) {                               \
                if (chain_async) M128_DUAL_ASYNC_STAGE(                       \
                    chain_read_stage ^ 1, chain_next_k);                       \
                else M128_DUAL_SCALAR_STAGE(                                  \
                    chain_read_stage ^ 1, chain_next_k);                       \
            }                                                                  \
            const int valid_k = min(M128_DUAL_BK, chain_k                     \
                - chain_tile * M128_DUAL_BK);                                 \
            _Pragma("unroll")                                                \
            for (int kk = 0; kk < M128_DUAL_BK; ++kk) {                      \
                if (kk < valid_k) {                                           \
                    float a_reg[8];                                            \
                    float b_reg[4];                                            \
                    _Pragma("unroll")                                        \
                    for (int i = 0; i < 8; ++i)                              \
                        a_reg[i] = smem_a[chain_read_stage]                   \
                            [(ty * 8 + i) * M128_DUAL_BK + kk];               \
                    _Pragma("unroll")                                        \
                    for (int j = 0; j < 4; ++j)                              \
                        b_reg[j] = smem_b[chain_read_stage]                   \
                            [kk * M128_DUAL_BN + tx * 4 + j];                 \
                    _Pragma("unroll")                                        \
                    for (int i = 0; i < 8; ++i) {                            \
                        _Pragma("unroll")                                    \
                        for (int j = 0; j < 4; ++j)                          \
                            (ACC)[i][j] = __fmaf_rn(                          \
                                a_reg[i], b_reg[j], (ACC)[i][j]);             \
                    }                                                          \
                }                                                              \
            }                                                                  \
            chain_read_stage ^= 1;                                             \
        }                                                                      \
    } while (0)

extern "C" __global__ __launch_bounds__(M128_DUAL_THREADS, 2)
void __M128_DUAL_SYMBOL__(
    float* __restrict__ output,
    const float* __restrict__ a,
    const float* __restrict__ b,
    M128DualParams params
) {
    const float alpha = params.alpha;
    const int m = params.m, n = params.n;
    const int k0 = params.k0, k1 = params.k1;
    const int lda = params.lda, ldb = params.ldb, ldc = params.ldc;
    __align__(16) __shared__ float smem_a[2][M128_DUAL_A_STAGE];
    __align__(16) __shared__ float smem_b[2][M128_DUAL_B_STAGE];

    const int num_pid_m = (m + M128_DUAL_BM - 1) / M128_DUAL_BM;
    const int num_pid_n = (n + M128_DUAL_BN - 1) / M128_DUAL_BN;
    const int num_pid_in_group = M128_DUAL_GROUP_M * num_pid_n;
    const int group_id = (int)blockIdx.x / num_pid_in_group;
    const int first_pid_m = group_id * M128_DUAL_GROUP_M;
    const int group_size_m = min(num_pid_m - first_pid_m, M128_DUAL_GROUP_M);
    const int pid_m = first_pid_m
        + (((int)blockIdx.x % num_pid_in_group) % group_size_m);
    const int pid_n = ((int)blockIdx.x % num_pid_in_group) / group_size_m;
    const int row0 = pid_m * M128_DUAL_BM;
    const int col0 = pid_n * M128_DUAL_BN;
    const int tid = (int)threadIdx.x;
    const int tx = tid & 15;
    const int ty = tid >> 4;
    const int row_base = row0 + ty * 8;
    const int col_base = col0 + tx * 4;
    const unsigned a_base = (unsigned)__cvta_generic_to_shared(&smem_a[0][0]);
    const unsigned b_base = (unsigned)__cvta_generic_to_shared(&smem_b[0][0]);

    float acc0[8][4];
    float acc1[8][4];
    M128_DUAL_RUN_CHAIN(acc0, a, b, k0);
    __syncthreads();
    const float* a1 = a + k0;
    const float* b1 = b + (long long)k0 * ldb;
    M128_DUAL_RUN_CHAIN(acc1, a1, b1, k1);

__M128_DUAL_EPILOGUE__
}

#undef M128_DUAL_RUN_CHAIN
#undef M128_DUAL_SCALAR_STAGE
#undef M128_DUAL_ASYNC_STAGE
#undef M128_DUAL_SMEM_BYTES
#undef M128_DUAL_B_STAGE
#undef M128_DUAL_A_STAGE
#undef M128_DUAL_GROUP_M
#undef M128_DUAL_THREADS
#undef M128_DUAL_BK
#undef M128_DUAL_BN
#undef M128_DUAL_BM
"#;

const RAW_EPILOGUE: &str = r#"    float* plane0 = output;
    float* plane1 = output + (long long)m * ldc;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = row_base + i;
        if (row >= m) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int column = col_base + j;
            if (column >= n) continue;
            const long long index = (long long)row * ldc + column;
            plane0[index] = acc0[i][j];
            plane1[index] = acc1[i][j];
        }
    }"#;

const FUSED_EPILOGUE: &str = r#"#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = row_base + i;
        if (row >= m) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int column = col_base + j;
            if (column >= n) continue;
            const long long index = (long long)row * ldc + column;
            const double sum = __dadd_rn((double)acc0[i][j], (double)acc1[i][j]);
            const float update = __double2float_rn(__dmul_rn((double)alpha, sum));
            output[index] = __fadd_rn(output[index], update);
        }
    }"#;

fn compose(symbol: &str, epilogue: &str) -> String {
    format!(
        "{PRELUDE}\n{}",
        TEMPLATE
            .replace("__M128_DUAL_SYMBOL__", symbol)
            .replace("__M128_DUAL_EPILOGUE__", epilogue)
    )
}

pub fn compose_raw_source() -> String {
    compose(RAW_SYMBOL, RAW_EPILOGUE)
}

pub fn compose_fused_source() -> String {
    compose(FUSED_SYMBOL, FUSED_EPILOGUE)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_common(source: &str, symbol: &str) {
        assert_eq!(source.matches(symbol).count(), 1);
        for contract in [
            "M128_DUAL_BM 128",
            "M128_DUAL_BN 64",
            "M128_DUAL_BK 16",
            "M128_DUAL_THREADS 256",
            "M128_DUAL_GROUP_M 6",
            "M128_DUAL_SMEM_BYTES 24576",
            "__launch_bounds__(M128_DUAL_THREADS, 2)",
            "cp.async.cg.shared.global",
            "float acc0[8][4]",
            "float acc1[8][4]",
        ] {
            assert!(source.contains(contract), "missing {contract}");
        }
        let chain0 = source.find("M128_DUAL_RUN_CHAIN(acc0, a, b, k0);").unwrap();
        let chain1 = source
            .find("M128_DUAL_RUN_CHAIN(acc1, a1, b1, k1);")
            .unwrap();
        assert!(chain0 < chain1);
        assert!(source[chain0..chain1].contains("__syncthreads();"));
        assert!(source.contains("_Pragma(\"unroll\")"));
        assert!(source.contains("for (int kk = 0; kk < M128_DUAL_BK; ++kk)"));
        assert!(source.contains("if (kk < valid_k)"));
        assert!(!source.contains("kk < valid_k; ++kk"));
        assert!(source.contains("(ACC)[i][j] = __fmaf_rn("));
        assert!(!source.contains("mma.sync"));
        assert!(!source.contains("atomic"));
    }

    #[test]
    fn raw_twin_preserves_two_independent_ordered_planes() {
        let source = compose_raw_source();
        assert_common(&source, RAW_SYMBOL);
        assert!(source.contains("plane0[index] = acc0[i][j]"));
        assert!(source.contains("plane1[index] = acc1[i][j]"));
    }

    #[test]
    fn fused_twin_preserves_sequential_fp64_finalize() {
        let source = compose_fused_source();
        assert_common(&source, FUSED_SYMBOL);
        let normalized = source.split_whitespace().collect::<Vec<_>>().join(" ");
        assert!(
            normalized
                .contains("const double sum = __dadd_rn((double)acc0[i][j], (double)acc1[i][j]);")
        );
        assert!(source.contains("__dmul_rn((double)alpha, sum)"));
        assert!(source.contains("__double2float_rn"));
        assert!(source.contains("output[index] = __fadd_rn(output[index], update)"));
    }
}
