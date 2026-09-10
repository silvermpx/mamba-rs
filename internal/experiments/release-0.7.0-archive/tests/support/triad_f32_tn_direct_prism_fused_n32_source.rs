pub const SYMBOL: &str = "gemm_bi_tn_test_direct_prism_sm89_f32_m64n32_bk16_s2_fused_v1";

pub fn compose_source() -> String {
    SOURCE.to_owned()
}

const SOURCE: &str = r#"
#define PRISM_FUSED_TN_BM 64
#define PRISM_FUSED_TN_BN 32
#define PRISM_FUSED_TN_BK 16
#define PRISM_FUSED_TN_THREADS 128
#define PRISM_FUSED_TN_GROUP_M 6
#define PRISM_FUSED_TN_CHUNKS 6
#define PRISM_FUSED_TN_M_CHUNK 784
#define PRISM_FUSED_TN_A_STAGE (PRISM_FUSED_TN_BK * PRISM_FUSED_TN_BM)
#define PRISM_FUSED_TN_B_STAGE (PRISM_FUSED_TN_BK * PRISM_FUSED_TN_BN)
#define PRISM_FUSED_TN_SMEM_BYTES 12288
static_assert(
    2 * (PRISM_FUSED_TN_A_STAGE + PRISM_FUSED_TN_B_STAGE) * sizeof(float)
        == PRISM_FUSED_TN_SMEM_BYTES,
    "fused direct Prism TN shared memory changed");

extern "C" __global__ __launch_bounds__(PRISM_FUSED_TN_THREADS, 4)
void gemm_bi_tn_test_direct_prism_sm89_f32_m64n32_bk16_s2_fused_v1(
    float* __restrict__ output,
    const float* __restrict__ x,
    const float* __restrict__ dy,
    float alpha,
    int M_red, int K_out, int N, int M_CHUNK, int chunks
) {
    if (M_red <= 5 * PRISM_FUSED_TN_M_CHUNK
        || M_red > PRISM_FUSED_TN_CHUNKS * PRISM_FUSED_TN_M_CHUNK
        || M_CHUNK != PRISM_FUSED_TN_M_CHUNK || chunks != PRISM_FUSED_TN_CHUNKS
        || (K_out & 3) != 0 || (N & 3) != 0
        || (reinterpret_cast<unsigned long long>(output) & 15ULL) != 0
        || (reinterpret_cast<unsigned long long>(x) & 15ULL) != 0
        || (reinterpret_cast<unsigned long long>(dy) & 15ULL) != 0) return;

    __align__(16) __shared__ float smem_a[2][PRISM_FUSED_TN_A_STAGE];
    __align__(16) __shared__ float smem_b[2][PRISM_FUSED_TN_B_STAGE];

    const int num_pid_m = (K_out + PRISM_FUSED_TN_BM - 1) / PRISM_FUSED_TN_BM;
    const int num_pid_n = (N + PRISM_FUSED_TN_BN - 1) / PRISM_FUSED_TN_BN;
    const int num_pid_in_group = PRISM_FUSED_TN_GROUP_M * num_pid_n;
    const int group_id = (int)blockIdx.x / num_pid_in_group;
    const int first_pid_m = group_id * PRISM_FUSED_TN_GROUP_M;
    const int group_size_m = min(num_pid_m - first_pid_m, PRISM_FUSED_TN_GROUP_M);
    const int pid_m = first_pid_m
        + (((int)blockIdx.x % num_pid_in_group) % group_size_m);
    const int pid_n = ((int)blockIdx.x % num_pid_in_group) / group_size_m;
    const int row0 = pid_m * PRISM_FUSED_TN_BM;
    const int col0 = pid_n * PRISM_FUSED_TN_BN;
    const int tx = (int)threadIdx.x & 7;
    const int ty = (int)threadIdx.x >> 3;
    const int row_base = row0 + ty * 4;
    const int col_base = col0 + tx * 4;

    double ordered_sum[4][4];
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) ordered_sum[i][j] = 0.0;
    }

    const unsigned a_base = (unsigned)__cvta_generic_to_shared(&smem_a[0][0]);
    const unsigned b_base = (unsigned)__cvta_generic_to_shared(&smem_b[0][0]);

#define PRISM_FUSED_TN_ISSUE_STAGE(STAGE, K_TILE) do {                            \
        const unsigned _a_stage = a_base                                           \
            + (unsigned)(STAGE) * PRISM_FUSED_TN_A_STAGE * (unsigned)sizeof(float); \
        const unsigned _b_stage = b_base                                           \
            + (unsigned)(STAGE) * PRISM_FUSED_TN_B_STAGE * (unsigned)sizeof(float); \
        for (int _i = (int)threadIdx.x;                                            \
             _i < PRISM_FUSED_TN_BK * (PRISM_FUSED_TN_BM / 4);                   \
             _i += PRISM_FUSED_TN_THREADS) {                                      \
            const int _kr = _i / (PRISM_FUSED_TN_BM / 4);                        \
            const int _mc = (_i % (PRISM_FUSED_TN_BM / 4)) * 4;                  \
            const int _global_r = m_begin + (K_TILE) + _kr;                       \
            const int _global_m = row0 + _mc;                                     \
            const int _remaining = K_out - _global_m;                             \
            const int _elements = _remaining >= 4 ? 4                             \
                : (_remaining > 0 ? _remaining : 0);                              \
            const int _bytes = _global_r < m_end ? _elements * 4 : 0;             \
            const unsigned _destination = _a_stage                                \
                + (unsigned)(_kr * PRISM_FUSED_TN_BM + _mc)                       \
                    * (unsigned)sizeof(float);                                     \
            const float* _source = _bytes > 0                                     \
                ? x + (long long)_global_r * K_out + _global_m : x;               \
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"       \
                         :: "r"(_destination), "l"(_source), "r"(_bytes));       \
        }                                                                          \
        for (int _i = (int)threadIdx.x;                                            \
             _i < PRISM_FUSED_TN_BK * (PRISM_FUSED_TN_BN / 4);                   \
             _i += PRISM_FUSED_TN_THREADS) {                                      \
            const int _kr = _i / (PRISM_FUSED_TN_BN / 4);                        \
            const int _nc = (_i % (PRISM_FUSED_TN_BN / 4)) * 4;                  \
            const int _global_r = m_begin + (K_TILE) + _kr;                       \
            const int _global_n = col0 + _nc;                                     \
            const int _remaining = N - _global_n;                                 \
            const int _elements = _remaining >= 4 ? 4                             \
                : (_remaining > 0 ? _remaining : 0);                              \
            const int _bytes = _global_r < m_end ? _elements * 4 : 0;             \
            const unsigned _destination = _b_stage                                \
                + (unsigned)(_kr * PRISM_FUSED_TN_BN + _nc)                       \
                    * (unsigned)sizeof(float);                                     \
            const float* _source = _bytes > 0                                     \
                ? dy + (long long)_global_r * N + _global_n : dy;                 \
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"       \
                         :: "r"(_destination), "l"(_source), "r"(_bytes));       \
        }                                                                          \
        asm volatile("cp.async.commit_group;\n");                                 \
    } while (0)

#define PRISM_FUSED_TN_FMA_STEP(KK) do {                                          \
        float a_reg[4];                                                            \
        float b_reg[4];                                                            \
        _Pragma("unroll")                                                         \
        for (int i = 0; i < 4; ++i)                                               \
            a_reg[i] = a_read[(KK) * PRISM_FUSED_TN_BM + ty * 4 + i];             \
        _Pragma("unroll")                                                         \
        for (int j = 0; j < 4; ++j)                                               \
            b_reg[j] = b_read[(KK) * PRISM_FUSED_TN_BN + tx * 4 + j];             \
        _Pragma("unroll")                                                         \
        for (int i = 0; i < 4; ++i) {                                             \
            _Pragma("unroll")                                                     \
            for (int j = 0; j < 4; ++j)                                           \
                acc[i][j] = __fmaf_rn(a_reg[i], b_reg[j], acc[i][j]);              \
        }                                                                          \
    } while (0)

#pragma unroll 1
    for (int fc = 0; fc < PRISM_FUSED_TN_CHUNKS; ++fc) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;
        }
        const int m_begin = fc * PRISM_FUSED_TN_M_CHUNK;
        const int m_end = min(m_begin + PRISM_FUSED_TN_M_CHUNK, M_red);
        const int chunk_k = m_end - m_begin;
        const int num_k_tiles = (chunk_k + PRISM_FUSED_TN_BK - 1) / PRISM_FUSED_TN_BK;
        PRISM_FUSED_TN_ISSUE_STAGE(0, 0);
        int read_stage = 0;
        for (int tile = 0; tile < num_k_tiles; ++tile) {
            asm volatile("cp.async.wait_group 0;\n");
            __syncthreads();
            if (tile + 1 < num_k_tiles) {
                PRISM_FUSED_TN_ISSUE_STAGE(read_stage ^ 1, (tile + 1) * PRISM_FUSED_TN_BK);
            }
            const float* a_read = &smem_a[read_stage][0];
            const float* b_read = &smem_b[read_stage][0];
#pragma unroll
            for (int kk = 0; kk < PRISM_FUSED_TN_BK; ++kk) {
                PRISM_FUSED_TN_FMA_STEP(kk);
            }
            read_stage ^= 1;
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (fc == 0) ordered_sum[i][j] = (double)acc[i][j];
                else ordered_sum[i][j] = __dadd_rn(ordered_sum[i][j], (double)acc[i][j]);
            }
        }
    }
#undef PRISM_FUSED_TN_FMA_STEP
#undef PRISM_FUSED_TN_ISSUE_STAGE

    const bool vector_store = row0 <= K_out - PRISM_FUSED_TN_BM
        && col0 <= N - PRISM_FUSED_TN_BN;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = row_base + i;
        if (row >= K_out) continue;
        float values[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int column = col_base + j;
            if (column < N) {
                const float update = __double2float_rn(
                    __dmul_rn((double)alpha, ordered_sum[i][j]));
                values[j] = __fadd_rn(output[(long long)row * N + column], update);
            }
        }
        if (vector_store) {
            const float4 value = {values[0], values[1], values[2], values[3]};
            *reinterpret_cast<float4*>(output + (long long)row * N + col_base) = value;
        } else {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int column = col_base + j;
                if (column < N) output[(long long)row * N + column] = values[j];
            }
        }
    }
}

#undef PRISM_FUSED_TN_BM
#undef PRISM_FUSED_TN_BN
#undef PRISM_FUSED_TN_BK
#undef PRISM_FUSED_TN_THREADS
#undef PRISM_FUSED_TN_GROUP_M
#undef PRISM_FUSED_TN_CHUNKS
#undef PRISM_FUSED_TN_M_CHUNK
#undef PRISM_FUSED_TN_A_STAGE
#undef PRISM_FUSED_TN_B_STAGE
#undef PRISM_FUSED_TN_SMEM_BYTES
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_pins_fused_n32_geometry_and_resource_intent() {
        let source = compose_source();
        assert_eq!(source.matches(SYMBOL).count(), 1);
        assert!(source.contains("PRISM_FUSED_TN_BM 64"));
        assert!(source.contains("PRISM_FUSED_TN_BN 32"));
        assert!(source.contains("PRISM_FUSED_TN_BK 16"));
        assert!(source.contains("PRISM_FUSED_TN_SMEM_BYTES 12288"));
        assert!(source.contains("__launch_bounds__(PRISM_FUSED_TN_THREADS, 4)"));
        assert!(source.contains("double ordered_sum[4][4]"));
        assert!(source.contains("float acc[4][4]"));
    }

    #[test]
    fn source_pins_six_f32_chains_and_sequential_fp64_finalize() {
        let source = compose_source();
        assert!(source.contains("for (int fc = 0; fc < PRISM_FUSED_TN_CHUNKS; ++fc)"));
        assert!(source.contains("const int m_begin = fc * PRISM_FUSED_TN_M_CHUNK"));
        assert!(source.contains("__fmaf_rn(a_reg[i], b_reg[j], acc[i][j])"));
        assert!(source.contains("if (fc == 0) ordered_sum[i][j] = (double)acc[i][j]"));
        assert!(source.contains("__dadd_rn(ordered_sum[i][j], (double)acc[i][j])"));
        assert!(source.contains("__dmul_rn((double)alpha, ordered_sum[i][j])"));
        assert!(source.contains("__fadd_rn(output[(long long)row * N + column], update)"));
        assert!(!source.contains("partial"));
        assert!(!source.contains("atom."));
        assert!(!source.contains("mma.sync"));
    }
}
