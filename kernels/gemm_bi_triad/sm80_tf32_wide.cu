// The wide deterministic TF32 tile: NN on 128 x 128 with eight computing
// warps (two rows by four columns of 64 x 32 warp tiles), BK 32, three
// cp.async stages in XOR-swizzled unpadded shared memory (32 KB per stage,
// 96 KB in all). The 128 x 64 tile above keeps four of its eight warps out of
// the mma loop; here every warp computes, and the census on the RTX 6000 Ada
// puts this tile at 0.72-0.87 of the best narrower tile on the training
// shapes. Per output element the mma chain is the same ascending k8 sequence
// as every other portable TF32 tile, so the output is bit-identical to them.
// Composed into the portable module for every sm80-family target except
// CC 12.x, which keeps the compute_120 module byte-identical to the one its
// TF32 cohort's portable twin was frozen against.

__device__ __forceinline__ int tf32w_a_index(int row, int k) {
    int chunk = (k >> 2) ^ (row & 7);
    return row * 32 + chunk * 4 + (k & 3);
}

__device__ __forceinline__ int tf32w_b_index(int k, int column) {
    int chunk = (column >> 2) ^ ((k & 3) << 1);
    return k * 128 + chunk * 4 + (column & 3);
}

template <int Stages>
__device__ __forceinline__ void tf32w_stage_async(
    float* a_stage, float* b_stage, const float* a, const float* b,
    const Sm80Tf32KernelParams& params, int tile_row, int tile_column, int k_base) {
    // A: 128 rows x 8 chunks of four floats.
    for (int linear = (int)threadIdx.x; linear < 128 * 8; linear += 256) {
        int row = linear >> 3;
        int k = (linear & 7) * 4;
        int global_row = tile_row + row;
        int global_k = k_base + k;
        int valid = global_row < params.m ? params.k - global_k : 0;
        valid = valid < 0 ? 0 : (valid > 4 ? 4 : valid);
        int bytes = valid * 4;
        long long offset = (long long)global_row * params.lda + global_k;
        const float* source = gemm_bi_cp_async_source(a, bytes == 0 ? 0 : offset, bytes);
        unsigned destination = (unsigned)__cvta_generic_to_shared(a_stage + tf32w_a_index(row, k));
        gemm_bi_cp_async_16_zfill_l2(destination, source, bytes);
    }
    // B: 32 reduction rows x 32 chunks of four floats.
    for (int linear = (int)threadIdx.x; linear < 32 * 32; linear += 256) {
        int k = linear >> 5;
        int column = (linear & 31) * 4;
        int global_k = k_base + k;
        int global_column = tile_column + column;
        int valid = global_k < params.k ? params.n - global_column : 0;
        valid = valid < 0 ? 0 : (valid > 4 ? 4 : valid);
        int bytes = valid * 4;
        long long offset = (long long)global_k * params.ldb + global_column;
        const float* source = gemm_bi_cp_async_source(b, bytes == 0 ? 0 : offset, bytes);
        unsigned destination = (unsigned)__cvta_generic_to_shared(b_stage + tf32w_b_index(k, column));
        gemm_bi_cp_async_16_zfill_l2(destination, source, bytes);
    }
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void tf32w_compute_stage(
    const float* a_stage, const float* b_stage, int warp_m, int warp_n,
    int group, int thread, float (&acc)[4][4][4]) {
#pragma unroll
    for (int issue = 0; issue < 4; ++issue) {
        int k8 = issue * 8;
        unsigned a_fragments[4][4];
        unsigned b_fragments[4][2];
#pragma unroll
        for (int m_atom = 0; m_atom < 4; ++m_atom) {
            int row = warp_m + m_atom * 16 + group;
            a_fragments[m_atom][0] = gemm_bi_tf32_rna(a_stage[tf32w_a_index(row, k8 + thread)]);
            a_fragments[m_atom][1] = gemm_bi_tf32_rna(a_stage[tf32w_a_index(row + 8, k8 + thread)]);
            a_fragments[m_atom][2] = gemm_bi_tf32_rna(a_stage[tf32w_a_index(row, k8 + thread + 4)]);
            a_fragments[m_atom][3] = gemm_bi_tf32_rna(a_stage[tf32w_a_index(row + 8, k8 + thread + 4)]);
        }
#pragma unroll
        for (int n_atom = 0; n_atom < 4; ++n_atom) {
            int column = warp_n + n_atom * 8 + group;
            b_fragments[n_atom][0] = gemm_bi_tf32_rna(b_stage[tf32w_b_index(k8 + thread, column)]);
            b_fragments[n_atom][1] = gemm_bi_tf32_rna(b_stage[tf32w_b_index(k8 + thread + 4, column)]);
        }
#pragma unroll
        for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
            for (int n_atom = 0; n_atom < 4; ++n_atom) {
                gemm_bi_tf32_mma_m16n8k8(acc[m_atom][n_atom], a_fragments[m_atom], b_fragments[n_atom]);
            }
        }
    }
}

template <int Stages>
__device__ __forceinline__ void tf32w_nn_kernel(
    float* output, const float* a, const float* b, const float* bias,
    Sm80Tf32KernelParams params) {
    extern __shared__ __align__(16) unsigned char tf32w_shared[];
    float* a_stages = reinterpret_cast<float*>(tf32w_shared);
    float* b_stages = a_stages + Stages * 128 * 32;
    int column_tiles = (params.n + 127) / 128;
    int tile_row = (int)blockIdx.x / column_tiles * 128;
    int tile_column = (int)blockIdx.x % column_tiles * 128;
    int warp = (int)threadIdx.x >> 5;
    int lane = (int)threadIdx.x & 31;
    int warp_m = (warp >> 2) * 64;
    int warp_n = (warp & 3) * 32;
    int group = lane >> 2;
    int thread = lane & 3;
    float acc[4][4][4];
#pragma unroll
    for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 4; ++n_atom) {
#pragma unroll
            for (int element = 0; element < 4; ++element) {
                int row = tile_row + warp_m + m_atom * 16 + group + (element >= 2 ? 8 : 0);
                int column = tile_column + warp_n + n_atom * 8 + 2 * thread + (element & 1);
                acc[m_atom][n_atom][element] =
                    row < params.m && column < params.n && bias != nullptr ? bias[column] : 0.0f;
            }
        }
    }
    unsigned tile_count = (static_cast<unsigned>(params.k) + 31U) / 32U;
#pragma unroll
    for (unsigned tile = 0; tile < Stages - 1; ++tile) {
        if (tile < tile_count) {
            tf32w_stage_async<Stages>(a_stages + tile * 128 * 32, b_stages + tile * 32 * 128,
                a, b, params, tile_row, tile_column, (int)(tile * 32U));
        } else {
            asm volatile("cp.async.commit_group;\n" ::);
        }
    }
    int read_stage = 0;
    for (unsigned tile = 0; tile < tile_count; ++tile) {
        asm volatile("cp.async.wait_group %0;\n" :: "n"(Stages - 2));
        __syncthreads();
        unsigned next = tile + Stages - 1;
        if (next < tile_count) {
            int write_stage = read_stage == 0 ? Stages - 1 : read_stage - 1;
            tf32w_stage_async<Stages>(a_stages + write_stage * 128 * 32, b_stages + write_stage * 32 * 128,
                a, b, params, tile_row, tile_column, (int)(next * 32U));
        } else {
            asm volatile("cp.async.commit_group;\n" ::);
        }
        tf32w_compute_stage(a_stages + read_stage * 128 * 32, b_stages + read_stage * 32 * 128,
            warp_m, warp_n, group, thread, acc);
        __syncthreads();
        if (++read_stage == Stages) read_stage = 0;
    }
#pragma unroll
    for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 4; ++n_atom) {
#pragma unroll
            for (int element = 0; element < 4; ++element) {
                int row = tile_row + warp_m + m_atom * 16 + group + (element >= 2 ? 8 : 0);
                int column = tile_column + warp_n + n_atom * 8 + 2 * thread + (element & 1);
                gemm_bi_tf32_store<SgbTf32Nn>(output, row, column, acc[m_atom][n_atom][element], bias, params);
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(256, 1)
void gemm_bi_nn_sm80_mma_tf32_v1_m128n128_bk32_s3(
    float* output, const float* a, const float* b, const float* bias,
    Sm80Tf32KernelParams params) {
    if (params.k == 0) {
        gemm_bi_tf32_zero_reduction_epilogue<SgbTf32Nn, 128, 128>(output, bias, params);
        return;
    }
    tf32w_nn_kernel<3>(output, a, b, bias, params);
}
