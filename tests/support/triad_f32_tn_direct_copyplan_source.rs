pub const SYMBOL: &str = "gemm_bi_tn_test_direct_sm89_f32_m64n64_bk32_s2_raw_v1";

pub fn compose_source() -> String {
    SOURCE.to_owned()
}

const SOURCE: &str = r#"
#define DIRECT_TN_BM 64
#define DIRECT_TN_BN 64
#define DIRECT_TN_BK 32
#define DIRECT_TN_A_STRIDE 64
#define DIRECT_TN_THREADS 128
#define DIRECT_TN_GROUP_M 8
#define DIRECT_TN_A_STAGE (DIRECT_TN_BK * DIRECT_TN_A_STRIDE)
#define DIRECT_TN_B_STAGE (DIRECT_TN_BK * DIRECT_TN_BN)
#define DIRECT_TN_SMEM_BYTES 32768
static_assert(2 * (DIRECT_TN_A_STAGE + DIRECT_TN_B_STAGE) * sizeof(float)
              == DIRECT_TN_SMEM_BYTES, "direct TN shared memory changed");

extern "C" __global__ __launch_bounds__(DIRECT_TN_THREADS, 2)
void gemm_bi_tn_test_direct_sm89_f32_m64n64_bk32_s2_raw_v1(
    float* __restrict__ partial,
    const float* __restrict__ x,
    const float* __restrict__ dy,
    int M_red, int K_out, int N, int M_CHUNK
) {
    if ((K_out & 3) != 0 || (N & 3) != 0
        || (reinterpret_cast<unsigned long long>(x) & 15ULL) != 0
        || (reinterpret_cast<unsigned long long>(dy) & 15ULL) != 0) return;
    const int m = K_out;
    const int n = N;
    const int ldx = K_out;
    const int lddy = N;
    const int ldc = N;
    const int fc = (int)blockIdx.z;
    const int m_begin = fc * M_CHUNK;
    const int m_end = min(m_begin + M_CHUNK, M_red);
    if (m_begin >= M_red) return;

    __align__(16) __shared__ float smem_a[2][DIRECT_TN_A_STAGE];
    __align__(16) __shared__ float smem_b[2][DIRECT_TN_B_STAGE];

    const int num_pid_m = (m + DIRECT_TN_BM - 1) / DIRECT_TN_BM;
    const int num_pid_n = (n + DIRECT_TN_BN - 1) / DIRECT_TN_BN;
    const int num_pid_in_group = DIRECT_TN_GROUP_M * num_pid_n;
    const int group_id = (int)blockIdx.x / num_pid_in_group;
    const int first_pid_m = group_id * DIRECT_TN_GROUP_M;
    const int group_size_m = min(num_pid_m - first_pid_m, DIRECT_TN_GROUP_M);
    const int pid_m = first_pid_m
        + (((int)blockIdx.x % num_pid_in_group) % group_size_m);
    const int pid_n = ((int)blockIdx.x % num_pid_in_group) / group_size_m;
    const int row0 = pid_m * DIRECT_TN_BM;
    const int col0 = pid_n * DIRECT_TN_BN;
    const int tx = (int)threadIdx.x & 15;
    const int ty = (int)threadIdx.x >> 4;
    const int row_base = row0 + ty * 8;
    const int col_base = col0 + tx * 4;

    float acc[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;
    }

    const unsigned a_base = (unsigned)__cvta_generic_to_shared(&smem_a[0][0]);
    const unsigned b_base = (unsigned)__cvta_generic_to_shared(&smem_b[0][0]);

#define DIRECT_TN_ISSUE_STAGE(STAGE, K_TILE) do {                                  \
        const unsigned _a_stage = a_base                                            \
            + (unsigned)(STAGE) * DIRECT_TN_A_STAGE * (unsigned)sizeof(float);      \
        const unsigned _b_stage = b_base                                            \
            + (unsigned)(STAGE) * DIRECT_TN_B_STAGE * (unsigned)sizeof(float);      \
        for (int _i = (int)threadIdx.x;                                             \
             _i < DIRECT_TN_BK * (DIRECT_TN_BM / 4);                               \
             _i += DIRECT_TN_THREADS) {                                             \
            const int _kr = _i / (DIRECT_TN_BM / 4);                               \
            const int _mc = (_i % (DIRECT_TN_BM / 4)) * 4;                         \
            const int _global_r = m_begin + (K_TILE) + _kr;                        \
            const int _global_m = row0 + _mc;                                      \
            const int _remaining = m - _global_m;                                  \
            const int _elements = _remaining >= 4 ? 4                              \
                : (_remaining > 0 ? _remaining : 0);                               \
            const int _bytes = _global_r < m_end ? _elements * 4 : 0;              \
            const unsigned _destination = _a_stage                                 \
                + (unsigned)(_kr * DIRECT_TN_A_STRIDE + _mc)                       \
                    * (unsigned)sizeof(float);                                      \
            const float* _source = _bytes > 0                                      \
                ? x + (long long)_global_r * ldx + _global_m : x;                  \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"       \
                         :: "r"(_destination), "l"(_source), "r"(_bytes));      \
        }                                                                           \
        for (int _i = (int)threadIdx.x;                                             \
             _i < DIRECT_TN_BK * (DIRECT_TN_BN / 4);                               \
             _i += DIRECT_TN_THREADS) {                                             \
            const int _kr = _i / (DIRECT_TN_BN / 4);                               \
            const int _nc = (_i % (DIRECT_TN_BN / 4)) * 4;                         \
            const int _global_r = m_begin + (K_TILE) + _kr;                        \
            const int _global_n = col0 + _nc;                                      \
            const int _remaining = n - _global_n;                                  \
            const int _elements = _remaining >= 4 ? 4                              \
                : (_remaining > 0 ? _remaining : 0);                               \
            const int _bytes = _global_r < m_end ? _elements * 4 : 0;              \
            const unsigned _destination = _b_stage                                 \
                + (unsigned)(_kr * DIRECT_TN_BN + _nc)                             \
                    * (unsigned)sizeof(float);                                      \
            const float* _source = _bytes > 0                                      \
                ? dy + (long long)_global_r * lddy + _global_n : dy;               \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"       \
                         :: "r"(_destination), "l"(_source), "r"(_bytes));      \
        }                                                                           \
        asm volatile("cp.async.commit_group;\n");                                 \
    } while (0)

    const int chunk_k = m_end - m_begin;
    const int num_k_tiles = (chunk_k + DIRECT_TN_BK - 1) / DIRECT_TN_BK;
    if (num_k_tiles > 0) DIRECT_TN_ISSUE_STAGE(0, 0);
    int read_stage = 0;
    for (int tile = 0; tile < num_k_tiles; ++tile) {
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();
        if (tile + 1 < num_k_tiles) {
            DIRECT_TN_ISSUE_STAGE(read_stage ^ 1, (tile + 1) * DIRECT_TN_BK);
        }
        const float* a_read = &smem_a[read_stage][0];
        const float* b_read = &smem_b[read_stage][0];
#pragma unroll
        for (int kk = 0; kk < DIRECT_TN_BK; ++kk) {
            float a_reg[8];
            float b_reg[4];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                a_reg[i] = a_read[kk * DIRECT_TN_A_STRIDE + ty * 8 + i];
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                b_reg[j] = b_read[kk * DIRECT_TN_BN + tx * 4 + j];
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] = __fmaf_rn(a_reg[i], b_reg[j], acc[i][j]);
                }
            }
        }
        read_stage ^= 1;
    }
#undef DIRECT_TN_ISSUE_STAGE

    float* partial_chunk = partial + (long long)fc * m * ldc;
    const bool vector_store = row0 <= m - DIRECT_TN_BM
        && col0 <= n - DIRECT_TN_BN
        && (reinterpret_cast<unsigned long long>(partial_chunk) & 15ULL) == 0
        && (ldc & 3) == 0;
    if (vector_store) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float4 value = {acc[i][0], acc[i][1], acc[i][2], acc[i][3]};
            *reinterpret_cast<float4*>(
                partial_chunk + (long long)(row_base + i) * ldc + col_base) = value;
        }
        return;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = row_base + i;
        if (row >= m) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int column = col_base + j;
            if (column < n) {
                float* destination = partial_chunk + (long long)row * ldc + col_base;
                destination[j] = acc[i][j];
            }
        }
    }
}

#undef DIRECT_TN_BM
#undef DIRECT_TN_BN
#undef DIRECT_TN_BK
#undef DIRECT_TN_A_STRIDE
#undef DIRECT_TN_THREADS
#undef DIRECT_TN_GROUP_M
#undef DIRECT_TN_A_STAGE
#undef DIRECT_TN_B_STAGE
#undef DIRECT_TN_SMEM_BYTES
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_tn_source_pins_geometry_coalesced_copy_and_exact_order() {
        let source = compose_source();
        assert_eq!(source.matches(SYMBOL).count(), 1);
        assert!(source.contains("DIRECT_TN_BM 64"));
        assert!(source.contains("DIRECT_TN_BN 64"));
        assert!(source.contains("DIRECT_TN_BK 32"));
        assert!(source.contains("DIRECT_TN_A_STRIDE 64"));
        assert!(source.contains("DIRECT_TN_SMEM_BYTES 32768"));
        assert!(source.contains("cp.async.cg.shared.global"));
        assert!(source.contains("__fmaf_rn(a_reg[i], b_reg[j], acc[i][j])"));
        assert!(source.contains("for (int kk = 0; kk < DIRECT_TN_BK; ++kk)"));
        assert!(!source.contains("mma.sync"));
        assert!(!source.contains("atom."));
    }

    #[test]
    fn direct_tn_source_keeps_raw_store_and_splitm_abi() {
        let source = compose_source();
        assert!(source.contains("int M_red, int K_out, int N, int M_CHUNK"));
        assert!(source.contains("int fc = (int)blockIdx.z"));
        assert!(source.contains("int m_begin = fc * M_CHUNK"));
        assert!(source.contains("int m_end = min(m_begin + M_CHUNK, M_red)"));
        assert!(source.contains("float* partial_chunk = partial + (long long)fc * m * ldc"));
        assert!(source.contains("(K_out & 3) != 0 || (N & 3) != 0"));
        assert!(source.contains("reinterpret_cast<unsigned long long>(x) & 15ULL"));
        assert!(source.contains("reinterpret_cast<unsigned long long>(dy) & 15ULL"));
        assert!(source.contains("destination[j] = acc[i][j]"));
        assert!(!source.contains("DirectTnExactF32Params"));
    }
}
