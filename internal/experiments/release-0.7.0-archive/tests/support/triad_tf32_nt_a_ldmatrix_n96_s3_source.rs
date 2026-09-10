pub const SYMBOL: &str = "gemm_bi_nt_test_a_ldmatrix_n96_s3_sm89_v1";
pub const RETAINED_SYMBOL: &str = "gemm_bi_nt_test_scalar_a_n96_s3_sm89_v1";
pub const BLOCK_THREADS: u32 = 256;
pub const DYNAMIC_SHARED_BYTES: u32 = 86_016;
pub const TARGET_D768_IN: (usize, usize, usize) = (2_048, 768, 3_072);
pub const TARGET_PRISM: (usize, usize, usize) = (4_621, 384, 1_928);

const SM80_SOURCE: &str = include_str!("../../kernels/gemm_bi_triad/sm80.cu");

pub fn candidate_source() -> String {
    compose(SYMBOL, A_LDMATRIX_LOAD)
}

pub fn retained_source() -> String {
    compose(RETAINED_SYMBOL, SCALAR_A_LOAD)
}

fn compose(symbol: &str, a_load: &str) -> String {
    format!("{SM80_SOURCE}\n{N96_BODY}")
        .replace("NT_N96_SYMBOL", symbol)
        .replace("NT_N96_A_LOAD", a_load)
}

const SCALAR_A_LOAD: &str = r#"#pragma unroll
        for (int m_atom = 0; m_atom < 4; ++m_atom) {
            int row = warp_m + m_atom * 16 + group;
            a_fragments[m_atom][0] = gemm_bi_tf32_rna(
                nt_n96_a_slot(storage, stage, row, k8 + thread));
            a_fragments[m_atom][1] = gemm_bi_tf32_rna(
                nt_n96_a_slot(storage, stage, row + 8, k8 + thread));
            a_fragments[m_atom][2] = gemm_bi_tf32_rna(
                nt_n96_a_slot(storage, stage, row, k8 + thread + 4));
            a_fragments[m_atom][3] = gemm_bi_tf32_rna(
                nt_n96_a_slot(storage, stage, row + 8, k8 + thread + 4));
        }"#;

const A_LDMATRIX_LOAD: &str = r#"#pragma unroll
        for (int m_atom = 0; m_atom < 4; ++m_atom) {
            int row = warp_m + m_atom * 16 + (lane & 15);
            int reduction = k8 + ((lane >> 4) << 2);
            unsigned address = (unsigned)__cvta_generic_to_shared(
                &nt_n96_a_slot(storage, stage, row, reduction));
            unsigned raw0, raw1, raw2, raw3;
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(raw0), "=r"(raw1), "=r"(raw2), "=r"(raw3)
                : "r"(address));
            a_fragments[m_atom][0] = gemm_bi_tf32_rna(__uint_as_float(raw0));
            a_fragments[m_atom][1] = gemm_bi_tf32_rna(__uint_as_float(raw1));
            a_fragments[m_atom][2] = gemm_bi_tf32_rna(__uint_as_float(raw2));
            a_fragments[m_atom][3] = gemm_bi_tf32_rna(__uint_as_float(raw3));
        }"#;

const N96_BODY: &str = r#"
struct __align__(16) NtN96S3Storage {
    float a[3][128][32];
    float b[3][96][32];
};

static_assert(sizeof(NtN96S3Storage) == 86016, "NT N96 S3 storage");

__device__ __forceinline__ float& nt_n96_a_slot(
    NtN96S3Storage* storage, int stage, int row, int reduction) {
    return storage->a[stage][row][reduction ^ ((row & 7) << 2)];
}

__device__ __forceinline__ float& nt_n96_b_slot(
    NtN96S3Storage* storage, int stage, int column, int reduction) {
    return storage->b[stage][column][reduction ^ ((column & 7) << 2)];
}

__device__ __forceinline__ void nt_n96_stage_async(
    NtN96S3Storage* storage, int stage, const float* a, const float* b,
    Sm80Tf32KernelParams params, int tile_row, int tile_column, int reduction_base) {
    for (int linear = (int)threadIdx.x; linear < 128 * 8; linear += 256) {
        int row = linear >> 3;
        int reduction = (linear & 7) * 4;
        int global_row = tile_row + row;
        int global_reduction = reduction_base + reduction;
        int valid = global_row < params.m ? params.n - global_reduction : 0;
        valid = valid < 0 ? 0 : (valid > 4 ? 4 : valid);
        unsigned destination = (unsigned)__cvta_generic_to_shared(
            &nt_n96_a_slot(storage, stage, row, reduction));
        const float* source = gemm_bi_cp_async_source(
            a, valid == 0 ? 0 : (long long)global_row * params.lda + global_reduction, valid * 4);
        gemm_bi_tf32_cp_async_zfill<false, 128>(destination, source, valid * 4);
    }
    for (int linear = (int)threadIdx.x; linear < 96 * 8; linear += 256) {
        int column = linear >> 3;
        int reduction = (linear & 7) * 4;
        int global_column = tile_column + column;
        int global_reduction = reduction_base + reduction;
        int valid = global_column < params.k ? params.n - global_reduction : 0;
        valid = valid < 0 ? 0 : (valid > 4 ? 4 : valid);
        unsigned destination = (unsigned)__cvta_generic_to_shared(
            &nt_n96_b_slot(storage, stage, column, reduction));
        const float* source = gemm_bi_cp_async_source(
            b, valid == 0 ? 0 : (long long)global_column * params.ldb + global_reduction, valid * 4);
        gemm_bi_tf32_cp_async_zfill<false, 128>(destination, source, valid * 4);
    }
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void nt_n96_zero(
    float* output, Sm80Tf32KernelParams params, int tile_row, int tile_column) {
    for (int linear = (int)threadIdx.x; linear < 128 * 96; linear += 256) {
        int row = tile_row + linear / 96;
        int column = tile_column + linear % 96;
        if (row < params.m && column < params.k) {
            output[(long long)row * params.ldc + column] =
                params.alpha == 1.0f ? 0.0f : __fmul_rn(params.alpha, 0.0f);
        }
    }
}

__device__ __forceinline__ void nt_n96_s3_kernel(
    float* output, const float* a, const float* b, Sm80Tf32KernelParams params) {
    int column_tiles = (params.k + 95) / 96;
    int tile_row = (int)blockIdx.x / column_tiles * 128;
    int tile_column = (int)blockIdx.x % column_tiles * 96;
    if (params.n == 0) {
        nt_n96_zero(output, params, tile_row, tile_column);
        return;
    }
    extern __shared__ __align__(16) unsigned char shared_bytes[];
    auto* storage = reinterpret_cast<NtN96S3Storage*>(shared_bytes);
    int warp = (int)threadIdx.x >> 5;
    int lane = (int)threadIdx.x & 31;
    int warp_m = (warp >> 2) * 64;
    int warp_n = (warp & 3) * 24;
    int group = lane >> 2;
    int thread = lane & 3;
    float accumulators[4][3][4] = {};
    unsigned tile_count = (static_cast<unsigned>(params.n) + 31U) / 32U;
#pragma unroll
    for (unsigned tile = 0; tile < 2; ++tile) {
        if (tile < tile_count) {
            nt_n96_stage_async(storage, (int)tile, a, b, params, tile_row, tile_column,
                (int)(tile * 32U));
        } else {
            asm volatile("cp.async.commit_group;\n" ::);
        }
    }
    for (unsigned tile = 0; tile < tile_count; ++tile) {
        asm volatile("cp.async.wait_group 1;\n" ::);
        __syncthreads();
        unsigned next = tile + 2;
        if (next < tile_count) {
            nt_n96_stage_async(storage, (int)(next % 3U), a, b, params, tile_row, tile_column,
                (int)(next * 32U));
        } else {
            asm volatile("cp.async.commit_group;\n" ::);
        }
        int stage = (int)(tile % 3U);
        const int k_offsets[4] = {0, 8, 16, 24};
#pragma unroll
        for (int issue = 0; issue < 4; ++issue) {
            int k8 = k_offsets[issue];
            unsigned a_fragments[4][4];
            unsigned b_fragments[3][2];
            NT_N96_A_LOAD
#pragma unroll
            for (int n_atom = 0; n_atom < 3; ++n_atom) {
                int column = warp_n + n_atom * 8 + group;
                b_fragments[n_atom][0] = gemm_bi_tf32_rna(
                    nt_n96_b_slot(storage, stage, column, k8 + thread));
                b_fragments[n_atom][1] = gemm_bi_tf32_rna(
                    nt_n96_b_slot(storage, stage, column, k8 + thread + 4));
            }
#pragma unroll
            for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
                for (int n_atom = 0; n_atom < 3; ++n_atom) {
                    gemm_bi_tf32_mma_m16n8k8(
                        accumulators[m_atom][n_atom], a_fragments[m_atom], b_fragments[n_atom]);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 3; ++n_atom) {
#pragma unroll
            for (int element = 0; element < 4; ++element) {
                int row = tile_row + warp_m + m_atom * 16 + group + (element >= 2 ? 8 : 0);
                int column = tile_column + warp_n + n_atom * 8 + 2 * thread + (element & 1);
                if (row < params.m && column < params.k) {
                    float value = params.alpha == 1.0f
                        ? accumulators[m_atom][n_atom][element]
                        : __fmul_rn(params.alpha, accumulators[m_atom][n_atom][element]);
                    output[(long long)row * params.ldc + column] = value;
                }
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(256, 1)
void NT_N96_SYMBOL(
    float* output, const float* a, const float* b, const float* bias,
    Sm80Tf32KernelParams params) {
    (void)bias;
    nt_n96_s3_kernel(output, a, b, params);
}
"#;
