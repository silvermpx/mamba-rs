pub const DUAL_RAW_SYMBOL: &str = "gemm_bi_tn_test_fixed_sm89_f32_n64_dual_chunk_raw_v1";
pub const DUAL_FUSED_SYMBOL: &str =
    "gemm_bi_tn_test_fixed_sm89_f32_n64_dual_chunk_fused_finalize_v1";

const PRELUDE: &str = include_str!("../../kernels/_typed_prelude.cuh");

const SOURCE_TEMPLATE: &str = r#"
struct DualChunkParams {
    float alpha;
    int m, n, k0, k1, lda, ldb, ldc;
};
static_assert(sizeof(DualChunkParams) == 32, "dual-chunk parameter size");
static_assert(alignof(DualChunkParams) == 4, "dual-chunk parameter alignment");

#define DUAL_BM 64
#define DUAL_BN 64
#define DUAL_BK 32
#define DUAL_GROUP_M 8
#define DUAL_THREADS 128

__device__ __forceinline__ bool dual_aligned16(const void* pointer) {
    return (reinterpret_cast<unsigned long long>(pointer) & 15ULL) == 0ULL;
}

#define DUAL_GENERIC_ASYNC(BUF, K_TILE)                                      \
    do {                                                                      \
        unsigned _as =                                                        \
            (unsigned)__cvta_generic_to_shared(&smem_a[(BUF)][0]);            \
        unsigned _bs =                                                        \
            (unsigned)__cvta_generic_to_shared(&smem_b[(BUF)][0]);            \
        for (int _i = threadIdx.x; _i < DUAL_BM * (DUAL_BK / 4);             \
             _i += DUAL_THREADS) {                                            \
            int _r = _i / (DUAL_BK / 4);                                     \
            int _c = (_i % (DUAL_BK / 4)) * 4;                               \
            int _gr = row0 + _r;                                              \
            int _gc = (K_TILE) + _c;                                         \
            int _valid = (_gr < m) ? (chain_k - _gc) : 0;                    \
            int _bytes =                                                      \
                _valid >= 4 ? 16 : (_valid > 0 ? _valid * 4 : 0);            \
            unsigned _dst =                                                   \
                _as + (unsigned)((_r * DUAL_BK + _c) * 4);                   \
            const void* _src = _bytes > 0                                    \
                ? (const void*)&chain_a[(long long)_gr * lda + _gc]           \
                : (const void*)chain_a;                                       \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"  \
                         :: "r"(_dst), "l"(_src), "r"(_bytes));             \
        }                                                                     \
        for (int _i = threadIdx.x; _i < DUAL_BK * (DUAL_BN / 4);             \
             _i += DUAL_THREADS) {                                            \
            int _r = _i / (DUAL_BN / 4);                                     \
            int _c = (_i % (DUAL_BN / 4)) * 4;                               \
            int _gr = (K_TILE) + _r;                                         \
            int _gc = col0 + _c;                                              \
            int _valid = (_gr < chain_k) ? (n - _gc) : 0;                    \
            int _bytes =                                                      \
                _valid >= 4 ? 16 : (_valid > 0 ? _valid * 4 : 0);            \
            unsigned _dst =                                                   \
                _bs + (unsigned)((_r * DUAL_BN + _c) * 4);                   \
            const void* _src = _bytes > 0                                    \
                ? (const void*)&chain_b[(long long)_gr * ldb + _gc]           \
                : (const void*)chain_b;                                       \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"  \
                         :: "r"(_dst), "l"(_src), "r"(_bytes));             \
        }                                                                     \
        asm volatile("cp.async.commit_group;\n");                            \
    } while (0)

#define DUAL_GENERIC_SCALAR(BUF, K_TILE)                                     \
    do {                                                                      \
        for (int _i = threadIdx.x; _i < DUAL_BM * DUAL_BK;                   \
             _i += DUAL_THREADS) {                                            \
            int _r = _i / DUAL_BK;                                            \
            int _c = _i % DUAL_BK;                                            \
            int _gr = row0 + _r;                                              \
            int _gc = (K_TILE) + _c;                                         \
            smem_a[(BUF)][_i] = (_gr < m && _gc < chain_k)                   \
                ? chain_a[(long long)_gr * lda + _gc]                         \
                : 0.0f;                                                       \
        }                                                                     \
        for (int _i = threadIdx.x; _i < DUAL_BK * DUAL_BN;                   \
             _i += DUAL_THREADS) {                                            \
            int _r = _i / DUAL_BN;                                            \
            int _c = _i % DUAL_BN;                                            \
            int _gr = (K_TILE) + _r;                                         \
            int _gc = col0 + _c;                                              \
            smem_b[(BUF)][_i] = (_gr < chain_k && _gc < n)                   \
                ? chain_b[(long long)_gr * ldb + _gc]                         \
                : 0.0f;                                                       \
        }                                                                     \
    } while (0)

#define DUAL_PLANNED_STAGE(BUF)                                               \
    do {                                                                      \
        unsigned _stage =                                                     \
            (unsigned)(BUF) * (DUAL_BM * DUAL_BK * 4);                        \
        _Pragma("unroll")                                                     \
        for (int _i = 0; _i < 4; ++_i) {                                     \
            unsigned _dst =                                                   \
                chain_a_destination + _stage + (unsigned)(_i * 2048);         \
            unsigned long long _address = chain_a_next                        \
                + (unsigned long long)_i * chain_a_vector_stride;             \
            unsigned long long _source = chain_a_bytes[_i] > 0                \
                ? _address                                                     \
                : reinterpret_cast<unsigned long long>(chain_a);              \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"  \
                         :: "r"(_dst), "l"(_source),                         \
                            "r"(chain_a_bytes[_i]));                          \
        }                                                                     \
        _Pragma("unroll")                                                     \
        for (int _i = 0; _i < 4; ++_i) {                                     \
            unsigned _dst =                                                   \
                chain_b_destination + _stage + (unsigned)(_i * 2048);         \
            unsigned long long _address = chain_b_next                        \
                + (unsigned long long)_i * chain_b_vector_stride;             \
            unsigned long long _source = chain_b_bytes > 0                    \
                ? _address                                                     \
                : reinterpret_cast<unsigned long long>(chain_b);              \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"  \
                         :: "r"(_dst), "l"(_source), "r"(chain_b_bytes));    \
        }                                                                     \
        asm volatile("cp.async.commit_group;\n");                            \
    } while (0)

#define DUAL_RUN_CHAIN(ACC, A_PTR, B_PTR, K_LEN)                              \
    do {                                                                      \
        _Pragma("unroll")                                                     \
        for (int _i = 0; _i < 8; ++_i) {                                     \
            _Pragma("unroll")                                                 \
            for (int _j = 0; _j < 4; ++_j) (ACC)[_i][_j] = 0.0f;             \
        }                                                                     \
        const float* chain_a = (A_PTR);                                        \
        const float* chain_b = (B_PTR);                                        \
        const int chain_k = (K_LEN);                                           \
        const int chain_num_k_tiles = (chain_k + DUAL_BK - 1) / DUAL_BK;      \
        const bool chain_fast_stage = dual_aligned16(chain_a)                  \
            && dual_aligned16(chain_b) && (lda & 3) == 0 && (ldb & 3) == 0;   \
        if (chain_num_k_tiles > 0 && chain_fast_stage                          \
            && (chain_k & (DUAL_BK - 1)) == 0) {                              \
            const int chain_copy_a_row = tid / (DUAL_BK / 4);                 \
            const int chain_copy_a_col = (tid % (DUAL_BK / 4)) * 4;           \
            const int chain_copy_b_row = tid / (DUAL_BN / 4);                 \
            const int chain_copy_b_col =                                      \
                col0 + (tid % (DUAL_BN / 4)) * 4;                             \
            int chain_a_bytes[4];                                              \
            _Pragma("unroll")                                                 \
            for (int _i = 0; _i < 4; ++_i)                                   \
                chain_a_bytes[_i] =                                           \
                    row0 + chain_copy_a_row + _i * 16 < m ? 16 : 0;           \
            const int chain_remaining_b = n - chain_copy_b_col;               \
            const int chain_b_bytes = chain_remaining_b >= 4                  \
                ? 16                                                           \
                : (chain_remaining_b > 0 ? chain_remaining_b * 4 : 0);        \
            const unsigned chain_a_destination =                              \
                (unsigned)__cvta_generic_to_shared(&smem_a[0][0])             \
                + (unsigned)(tid * 16);                                       \
            const unsigned chain_b_destination =                              \
                (unsigned)__cvta_generic_to_shared(&smem_b[0][0])             \
                + (unsigned)(tid * 16);                                       \
            const unsigned long long chain_a_vector_stride =                  \
                (unsigned long long)lda * 64ULL;                              \
            const unsigned long long chain_b_vector_stride =                  \
                (unsigned long long)ldb * 32ULL;                              \
            const unsigned long long chain_b_slab_stride =                    \
                chain_b_vector_stride * 4ULL;                                 \
            unsigned long long chain_a_next =                                 \
                reinterpret_cast<unsigned long long>(chain_a)                 \
                + ((unsigned long long)(row0 + chain_copy_a_row)              \
                       * (unsigned long long)lda                               \
                   + (unsigned long long)chain_copy_a_col)                    \
                    * 4ULL;                                                    \
            unsigned long long chain_b_next =                                 \
                reinterpret_cast<unsigned long long>(chain_b)                 \
                + ((unsigned long long)chain_copy_b_row                        \
                       * (unsigned long long)ldb                               \
                   + (unsigned long long)chain_copy_b_col)                    \
                    * 4ULL;                                                    \
            DUAL_PLANNED_STAGE(0);                                             \
            chain_a_next += DUAL_BK * 4ULL;                                   \
            chain_b_next += chain_b_slab_stride;                              \
            int chain_read_buf = 0;                                            \
            for (int chain_kt = 0; chain_kt < chain_num_k_tiles; ++chain_kt) {\
                asm volatile("cp.async.wait_group 0;\n");                    \
                __syncthreads();                                               \
                if (chain_kt + 1 < chain_num_k_tiles) {                       \
                    DUAL_PLANNED_STAGE(chain_read_buf ^ 1);                    \
                    chain_a_next += DUAL_BK * 4ULL;                           \
                    chain_b_next += chain_b_slab_stride;                      \
                }                                                             \
                _Pragma("unroll")                                             \
                for (int kk = 0; kk < DUAL_BK; ++kk) {                        \
                    float a_reg[8];                                            \
                    float b_reg[4];                                            \
                    _Pragma("unroll")                                         \
                    for (int i = 0; i < 8; ++i)                               \
                        a_reg[i] = smem_a[chain_read_buf]                      \
                            [(ty * 8 + i) * DUAL_BK + kk];                     \
                    _Pragma("unroll")                                         \
                    for (int j = 0; j < 4; ++j)                               \
                        b_reg[j] = smem_b[chain_read_buf]                      \
                            [kk * DUAL_BN + tx * 4 + j];                       \
                    _Pragma("unroll")                                         \
                    for (int i = 0; i < 8; ++i) {                             \
                        _Pragma("unroll")                                     \
                        for (int j = 0; j < 4; ++j)                           \
                            (ACC)[i][j] = __fmaf_rn(                           \
                                a_reg[i], b_reg[j], (ACC)[i][j]);             \
                    }                                                         \
                }                                                             \
                chain_read_buf ^= 1;                                           \
            }                                                                 \
        } else {                                                              \
            if (chain_num_k_tiles > 0) {                                      \
                if (chain_fast_stage) DUAL_GENERIC_ASYNC(0, 0);               \
                else DUAL_GENERIC_SCALAR(0, 0);                               \
            }                                                                 \
            int chain_read_buf = 0;                                            \
            for (int chain_kt = 0; chain_kt < chain_num_k_tiles; ++chain_kt) {\
                if (chain_fast_stage)                                         \
                    asm volatile("cp.async.wait_group 0;\n");                \
                __syncthreads();                                               \
                const int chain_next_k = (chain_kt + 1) * DUAL_BK;            \
                if (chain_kt + 1 < chain_num_k_tiles) {                       \
                    if (chain_fast_stage)                                     \
                        DUAL_GENERIC_ASYNC(chain_read_buf ^ 1, chain_next_k);  \
                    else                                                      \
                        DUAL_GENERIC_SCALAR(chain_read_buf ^ 1, chain_next_k); \
                }                                                             \
                _Pragma("unroll")                                             \
                for (int kk = 0; kk < DUAL_BK; ++kk) {                        \
                    float a_reg[8];                                            \
                    float b_reg[4];                                            \
                    _Pragma("unroll")                                         \
                    for (int i = 0; i < 8; ++i)                               \
                        a_reg[i] = smem_a[chain_read_buf]                      \
                            [(ty * 8 + i) * DUAL_BK + kk];                     \
                    _Pragma("unroll")                                         \
                    for (int j = 0; j < 4; ++j)                               \
                        b_reg[j] = smem_b[chain_read_buf]                      \
                            [kk * DUAL_BN + tx * 4 + j];                       \
                    _Pragma("unroll")                                         \
                    for (int i = 0; i < 8; ++i) {                             \
                        _Pragma("unroll")                                     \
                        for (int j = 0; j < 4; ++j)                           \
                            (ACC)[i][j] = __fmaf_rn(                           \
                                a_reg[i], b_reg[j], (ACC)[i][j]);             \
                    }                                                         \
                }                                                             \
                chain_read_buf ^= 1;                                           \
            }                                                                 \
        }                                                                     \
    } while (0)

extern "C" __global__ __launch_bounds__(128, 3)
void __DUAL_SYMBOL__(
    float* __restrict__ output,
    const float* __restrict__ a,
    const float* __restrict__ b,
    DualChunkParams params
) {
    const float alpha = params.alpha;
    const int m = params.m, n = params.n;
    const int k0 = params.k0, k1 = params.k1;
    const int lda = params.lda, ldb = params.ldb, ldc = params.ldc;
    __align__(16) __shared__ float smem_a[2][DUAL_BM * DUAL_BK];
    __align__(16) __shared__ float smem_b[2][DUAL_BK * DUAL_BN];

    const int num_pid_m = (m + DUAL_BM - 1) / DUAL_BM;
    const int num_pid_n = (n + DUAL_BN - 1) / DUAL_BN;
    const int num_pid_in_group = DUAL_GROUP_M * num_pid_n;
    const int group_id = blockIdx.x / num_pid_in_group;
    const int first_pid_m = group_id * DUAL_GROUP_M;
    const int group_size_m = min(num_pid_m - first_pid_m, DUAL_GROUP_M);
    const int pid_m =
        first_pid_m + ((blockIdx.x % num_pid_in_group) % group_size_m);
    const int pid_n = (blockIdx.x % num_pid_in_group) / group_size_m;
    const int row0 = pid_m * DUAL_BM;
    const int col0 = pid_n * DUAL_BN;
    const int tid = (int)threadIdx.x;
    const int tx = tid & 15;
    const int ty = tid >> 4;
    const int row_base = row0 + ty * 8;
    const int col_base = col0 + tx * 4;

    float acc0[8][4];
    float acc1[8][4];
    DUAL_RUN_CHAIN(acc0, a, b, k0);
    __syncthreads();
    const float* a1 = a + k0;
    const float* b1 = b + (long long)k0 * ldb;
    DUAL_RUN_CHAIN(acc1, a1, b1, k1);

__DUAL_EPILOGUE__
}

#undef DUAL_RUN_CHAIN
#undef DUAL_PLANNED_STAGE
#undef DUAL_GENERIC_SCALAR
#undef DUAL_GENERIC_ASYNC
#undef DUAL_THREADS
#undef DUAL_GROUP_M
#undef DUAL_BK
#undef DUAL_BN
#undef DUAL_BM
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
            .replace("__DUAL_SYMBOL__", symbol)
            .replace("__DUAL_EPILOGUE__", epilogue)
    )
}

pub fn compose_raw_source() -> String {
    compose(DUAL_RAW_SYMBOL, RAW_EPILOGUE)
}

pub fn compose_fused_source() -> String {
    compose(DUAL_FUSED_SYMBOL, FUSED_EPILOGUE)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_common_dual_chunk_contract(source: &str, symbol: &str) {
        assert_eq!(source.matches(symbol).count(), 1);
        assert!(source.contains("__launch_bounds__(128, 3)"));
        assert!(source.contains("float acc0[8][4]"));
        assert!(source.contains("float acc1[8][4]"));
        assert_eq!(source.matches("DUAL_RUN_CHAIN(").count(), 3);
        assert!(source.contains("DUAL_RUN_CHAIN(acc0, a, b, k0);"));
        let first = source.find("DUAL_RUN_CHAIN(acc0, a, b, k0);").unwrap();
        let second = source.find("DUAL_RUN_CHAIN(acc1, a1, b1, k1);").unwrap();
        assert!(first < second);
        assert!(source[first..second].contains("__syncthreads();"));
        assert_eq!(source.matches("(ACC)[i][j] = __fmaf_rn(").count(), 2);
        assert!(!source.contains("mma.sync"));
        assert!(!source.contains("atomic"));
    }

    #[test]
    fn raw_twin_exposes_two_independent_unscaled_partial_planes() {
        let source = compose_raw_source();
        assert_common_dual_chunk_contract(&source, DUAL_RAW_SYMBOL);
        assert!(source.contains("float* plane0 = output;"));
        assert!(source.contains("float* plane1 = output + (long long)m * ldc;"));
        assert!(source.contains("plane0[idx] = acc0[i][j];"));
        assert!(source.contains("plane1[idx] = acc1[i][j];"));
        assert!(!source.contains("__fmul_rn(alpha"));
    }

    #[test]
    fn fused_twin_preserves_exact_splitm_finalize_instruction_order() {
        let source = compose_fused_source();
        assert_common_dual_chunk_contract(&source, DUAL_FUSED_SYMBOL);
        let normalized = source.split_whitespace().collect::<Vec<_>>().join(" ");
        assert!(normalized
            .contains("const double sum = __dadd_rn((double)acc0[i][j], (double)acc1[i][j]);"));
        assert!(source.contains("__dmul_rn((double)alpha, sum)"));
        assert!(source.contains("__double2float_rn"));
        assert!(source.contains("output[idx] = __fadd_rn(output[idx], update);"));
    }

    #[test]
    fn dual_params_are_fixed_32_byte_compact_abi() {
        let source = compose_fused_source();
        assert!(source.contains("static_assert(sizeof(DualChunkParams) == 32"));
        assert!(source.contains("static_assert(alignof(DualChunkParams) == 4"));
        for field in ["alpha", "m", "n", "k0", "k1", "lda", "ldb", "ldc"] {
            assert!(source.contains(field), "missing ABI field {field}");
        }
    }
}
