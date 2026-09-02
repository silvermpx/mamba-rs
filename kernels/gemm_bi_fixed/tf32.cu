// Deterministic TF32 inference GEMM. This is an NN-only numeric mode owned
// by the inference module; its fixed K traversal never changes with M.

struct GbfTf32Params {
    int m;
    int k;
    int n;
    int lda;
    int ldb;
    int ldc;
};

static_assert(sizeof(GbfTf32Params) == 24, "Fixed TF32 parameter ABI drift");
static_assert(alignof(GbfTf32Params) == 4, "Fixed TF32 parameter alignment drift");
static_assert(__is_standard_layout(GbfTf32Params),
              "Fixed TF32 parameters must remain standard layout");

template <int BM, int BN, int Stages, bool CompactXor = false>
struct __align__(16) GbfTf32Storage {
    float a[Stages][BM][CompactXor ? 32 : 36];
    float b[Stages][32][CompactXor ? BN : (BN == 64 ? 72 : 40)];
};

static_assert(sizeof(GbfTf32Storage<128, 64, 2>) == 55296, "M128N64 s2 storage");
static_assert(sizeof(GbfTf32Storage<128, 64, 3>) == 82944, "M128N64 s3 storage");
static_assert(sizeof(GbfTf32Storage<64, 64, 2, true>) == 32768,
              "M64N64 s2 compact XOR storage");
static_assert(sizeof(GbfTf32Storage<64, 64, 3>) == 55296, "M64N64 s3 storage");
static_assert(sizeof(GbfTf32Storage<16, 32, 4>) == 29696, "M16N32 s4 storage");

template <bool CompactXor, int BM, int BN, int Stages>
__device__ __forceinline__ float* gbf_tf32_a_slot(
    GbfTf32Storage<BM, BN, Stages, CompactXor>* storage,
    int stage, int row, int reduction) {
    if constexpr (CompactXor) {
        int chunk = (reduction >> 2) ^ (row & 7);
        return &storage->a[stage][row][chunk * 4 + (reduction & 3)];
    }
    return &storage->a[stage][row][reduction];
}

template <bool CompactXor, int BM, int BN, int Stages>
__device__ __forceinline__ float* gbf_tf32_b_slot(
    GbfTf32Storage<BM, BN, Stages, CompactXor>* storage,
    int stage, int reduction, int column) {
    if constexpr (CompactXor) {
        int chunk = (column >> 2) ^ ((reduction & 3) << 1);
        return &storage->b[stage][reduction][chunk * 4 + (column & 3)];
    }
    return &storage->b[stage][reduction][column];
}

struct GbfTf32Problem {
    float* output;
    const float* a;
    const float* b;
    const float* bias;
    GbfTf32Params params;
    int tile_row;
    int tile_column;
};

template <typename T>
__device__ __forceinline__ const T* gbf_tf32_source(
    const T* base, long long valid_offset, int valid_bytes) {
    return valid_bytes == 0 ? base : base + valid_offset;
}

__device__ __forceinline__ void gbf_tf32_copy_ca(
    unsigned shared_dst, const void* global_src, int valid_bytes) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(shared_dst), "l"(global_src), "r"(valid_bytes));
}

__device__ __forceinline__ void gbf_tf32_copy_cg(
    unsigned shared_dst, const void* global_src, int valid_bytes) {
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;\n"
                 :: "r"(shared_dst), "l"(global_src), "r"(valid_bytes));
}

__device__ __forceinline__ unsigned gbf_tf32_rna(float value) {
    unsigned result;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(result) : "f"(value));
    return result;
}

__device__ __forceinline__ void gbf_tf32_mma_m16n8k8(
    float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
}

template <bool CompactXor, int BM, int BN, int Stages>
__device__ __forceinline__ void gbf_tf32_stage_scalar(
    GbfTf32Storage<BM, BN, Stages, CompactXor>* storage, int stage,
    const GbfTf32Problem& problem, int reduction_base) {
    for (int linear = (int)threadIdx.x; linear < BM * 32; linear += (int)blockDim.x) {
        int row = linear >> 5;
        int reduction = linear & 31;
        int global_row = problem.tile_row + row;
        int global_reduction = reduction_base + reduction;
        float value = 0.0f;
        if (global_row < problem.params.m && global_reduction < problem.params.k) {
            value = problem.a[(long long)global_row * problem.params.lda + global_reduction];
        }
        *gbf_tf32_a_slot<CompactXor>(storage, stage, row, reduction) = value;
    }
    for (int linear = (int)threadIdx.x; linear < 32 * BN; linear += (int)blockDim.x) {
        int reduction = linear / BN;
        int column = linear - reduction * BN;
        int global_reduction = reduction_base + reduction;
        int global_column = problem.tile_column + column;
        float value = 0.0f;
        if (global_column < problem.params.n && global_reduction < problem.params.k) {
            value = problem.b[(long long)global_reduction * problem.params.ldb + global_column];
        }
        *gbf_tf32_b_slot<CompactXor>(storage, stage, reduction, column) = value;
    }
}

template <int BM>
__device__ __forceinline__ void gbf_tf32_copy_16(
    unsigned shared_dst, const void* global_src, int valid_bytes) {
    if constexpr (BM == 16) {
        gbf_tf32_copy_ca(shared_dst, global_src, valid_bytes);
    } else {
        gbf_tf32_copy_cg(shared_dst, global_src, valid_bytes);
    }
}

template <bool CompactXor, int BM, int BN, int Stages>
__device__ __forceinline__ void gbf_tf32_stage_async(
    GbfTf32Storage<BM, BN, Stages, CompactXor>* storage, int stage,
    const GbfTf32Problem& problem, int reduction_base) {
    for (int linear = (int)threadIdx.x; linear < BM * 8; linear += (int)blockDim.x) {
        int row = linear >> 3;
        int reduction = (linear & 7) * 4;
        int global_row = problem.tile_row + row;
        int global_reduction = reduction_base + reduction;
        int valid = global_row < problem.params.m ? problem.params.k - global_reduction : 0;
        valid = valid < 0 ? 0 : (valid > 4 ? 4 : valid);
        int valid_bytes = valid * 4;
        long long offset = (long long)global_row * problem.params.lda + global_reduction;
        const float* source = gbf_tf32_source(
            problem.a, valid_bytes == 0 ? 0 : offset, valid_bytes);
        unsigned destination = (unsigned)__cvta_generic_to_shared(
            gbf_tf32_a_slot<CompactXor>(storage, stage, row, reduction));
        gbf_tf32_copy_16<BM>(destination, source, valid_bytes);
    }
    for (int linear = (int)threadIdx.x; linear < 32 * (BN / 4);
         linear += (int)blockDim.x) {
        int reduction = linear / (BN / 4);
        int column = (linear - reduction * (BN / 4)) * 4;
        int global_reduction = reduction_base + reduction;
        int global_column = problem.tile_column + column;
        int valid = global_reduction < problem.params.k ? problem.params.n - global_column : 0;
        valid = valid < 0 ? 0 : (valid > 4 ? 4 : valid);
        int valid_bytes = valid * 4;
        long long offset = (long long)global_reduction * problem.params.ldb + global_column;
        const float* source = gbf_tf32_source(
            problem.b, valid_bytes == 0 ? 0 : offset, valid_bytes);
        unsigned destination = (unsigned)__cvta_generic_to_shared(
            gbf_tf32_b_slot<CompactXor>(storage, stage, reduction, column));
        gbf_tf32_copy_16<BM>(destination, source, valid_bytes);
    }
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int BM, int BN>
__device__ __forceinline__ void gbf_tf32_zero_reduction(
    float* output, const float* bias, const GbfTf32Params& params) {
    int column_tiles = (params.n + BN - 1) / BN;
    int tile_row = (int)blockIdx.x / column_tiles * BM;
    int tile_column = (int)blockIdx.x % column_tiles * BN;
    for (int linear = (int)threadIdx.x; linear < BM * BN; linear += (int)blockDim.x) {
        int row = tile_row + linear / BN;
        int column = tile_column + linear % BN;
        if (row < params.m && column < params.n) {
            output[(long long)row * params.ldc + column] =
                bias == nullptr ? 0.0f : bias[column];
        }
    }
}

template <bool CompactXor, int BM, int BN, int Stages, int MAtoms, int NAtoms>
__device__ __forceinline__ void gbf_tf32_compute_stage(
    GbfTf32Storage<BM, BN, Stages, CompactXor>* storage, int stage,
    int warp_m, int warp_n, int group, int thread,
    float (&accumulators)[MAtoms][NAtoms][4]) {
#pragma unroll
    for (int issue = 0; issue < 4; ++issue) {
        int k8 = issue * 8;
        unsigned a_fragments[MAtoms][4];
        unsigned b_fragments[NAtoms][2];
#pragma unroll
        for (int m_atom = 0; m_atom < MAtoms; ++m_atom) {
            int row = warp_m + m_atom * 16 + group;
            a_fragments[m_atom][0] = gbf_tf32_rna(
                *gbf_tf32_a_slot<CompactXor>(storage, stage, row, k8 + thread));
            a_fragments[m_atom][1] = gbf_tf32_rna(
                *gbf_tf32_a_slot<CompactXor>(storage, stage, row + 8, k8 + thread));
            a_fragments[m_atom][2] = gbf_tf32_rna(
                *gbf_tf32_a_slot<CompactXor>(storage, stage, row, k8 + thread + 4));
            a_fragments[m_atom][3] = gbf_tf32_rna(
                *gbf_tf32_a_slot<CompactXor>(storage, stage, row + 8, k8 + thread + 4));
        }
#pragma unroll
        for (int n_atom = 0; n_atom < NAtoms; ++n_atom) {
            int column = warp_n + n_atom * 8 + group;
            b_fragments[n_atom][0] = gbf_tf32_rna(
                *gbf_tf32_b_slot<CompactXor>(storage, stage, k8 + thread, column));
            b_fragments[n_atom][1] = gbf_tf32_rna(
                *gbf_tf32_b_slot<CompactXor>(storage, stage, k8 + thread + 4, column));
        }
#pragma unroll
        for (int m_atom = 0; m_atom < MAtoms; ++m_atom) {
#pragma unroll
            for (int n_atom = 0; n_atom < NAtoms; ++n_atom) {
                gbf_tf32_mma_m16n8k8(
                    accumulators[m_atom][n_atom], a_fragments[m_atom], b_fragments[n_atom]);
            }
        }
    }
}

template <int BM, int BN, int Stages, bool CompactXor = false>
__device__ __forceinline__ void gbf_tf32_kernel(
    float* output, const float* a, const float* b, const float* bias,
    GbfTf32Params params) {
    constexpr int MAtoms = BM == 128 ? 4 : (BM == 64 ? 2 : 1);
    constexpr int NAtoms = BM == 16 ? 1 : 4;
    int column_tiles = (params.n + BN - 1) / BN;
    GbfTf32Problem problem = {
        output, a, b, bias, params,
        (int)blockIdx.x / column_tiles * BM,
        (int)blockIdx.x % column_tiles * BN,
    };
    extern __shared__ __align__(16) unsigned char gbf_tf32_shared[];
    auto* storage = reinterpret_cast<GbfTf32Storage<BM, BN, Stages, CompactXor>*>(
        gbf_tf32_shared);

    int warp = (int)threadIdx.x >> 5;
    int lane = (int)threadIdx.x & 31;
    bool compute = BM != 128 || warp < 4;
    int warp_m = BM == 128 ? (warp >> 1) * 64 : (BM == 64 ? (warp >> 1) * 32 : 0);
    int warp_n = BM == 16 ? warp * 8 : (warp & 1) * 32;
    int group = lane >> 2;
    int thread = lane & 3;
    float accumulators[MAtoms][NAtoms][4];

#pragma unroll
    for (int m_atom = 0; m_atom < MAtoms; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < NAtoms; ++n_atom) {
#pragma unroll
            for (int element = 0; element < 4; ++element) {
                int row = problem.tile_row + warp_m + m_atom * 16
                    + group + (element >= 2 ? 8 : 0);
                int column = problem.tile_column + warp_n + n_atom * 8
                    + 2 * thread + (element & 1);
                accumulators[m_atom][n_atom][element] =
                    compute && row < params.m && column < params.n && bias != nullptr
                        ? bias[column]
                        : 0.0f;
            }
        }
    }

    unsigned tile_count = (static_cast<unsigned>(params.k) + 31U) / 32U;
    bool fast_stage = gbf_aligned16(a) && gbf_aligned16(b)
        && (params.lda & 3) == 0 && (params.ldb & 3) == 0;
    if (fast_stage) {
#pragma unroll
        for (unsigned tile = 0; tile < Stages - 1; ++tile) {
            if (tile < tile_count) {
                gbf_tf32_stage_async<CompactXor>(
                    storage, static_cast<int>(tile), problem, static_cast<int>(tile * 32U));
            } else {
                asm volatile("cp.async.commit_group;\n" ::);
            }
        }
        for (unsigned tile = 0; tile < tile_count; ++tile) {
            asm volatile("cp.async.wait_group %0;\n" :: "n"(Stages - 2));
            __syncthreads();
            unsigned next = tile + Stages - 1;
            if (next < tile_count) {
                gbf_tf32_stage_async<CompactXor>(
                    storage, static_cast<int>(next % Stages), problem,
                    static_cast<int>(next * 32U));
            } else {
                asm volatile("cp.async.commit_group;\n" ::);
            }
            if (compute) {
                gbf_tf32_compute_stage<CompactXor, BM, BN, Stages, MAtoms, NAtoms>(
                    storage, static_cast<int>(tile % Stages), warp_m, warp_n,
                    group, thread, accumulators);
            }
            __syncthreads();
        }
    } else {
        for (unsigned tile = 0; tile < tile_count; ++tile) {
            int stage = static_cast<int>(tile % Stages);
            gbf_tf32_stage_scalar<CompactXor>(
                storage, stage, problem, static_cast<int>(tile * 32U));
            __syncthreads();
            if (compute) {
                gbf_tf32_compute_stage<CompactXor, BM, BN, Stages, MAtoms, NAtoms>(
                    storage, stage, warp_m, warp_n, group, thread, accumulators);
            }
            __syncthreads();
        }
    }

    if (compute) {
#pragma unroll
        for (int m_atom = 0; m_atom < MAtoms; ++m_atom) {
#pragma unroll
            for (int n_atom = 0; n_atom < NAtoms; ++n_atom) {
#pragma unroll
                for (int element = 0; element < 4; ++element) {
                    int row = problem.tile_row + warp_m + m_atom * 16
                        + group + (element >= 2 ? 8 : 0);
                    int column = problem.tile_column + warp_n + n_atom * 8
                        + 2 * thread + (element & 1);
                    if (row < params.m && column < params.n) {
                        output[(long long)row * params.ldc + column] =
                            accumulators[m_atom][n_atom][element];
                    }
                }
            }
        }
    }
}

template <int BM, int BN, int Stages, bool CompactXor = false>
__device__ __forceinline__ void gbf_tf32_entry(
    float* output, const float* a, const float* b, const float* bias,
    GbfTf32Params params) {
    if (params.k == 0) {
        gbf_tf32_zero_reduction<BM, BN>(output, bias, params);
        return;
    }
    gbf_tf32_kernel<BM, BN, Stages, CompactXor>(output, a, b, bias, params);
}

#define GBF_TF32_KERNEL(NAME, BM, BN, STAGES, THREADS, MIN_BLOCKS)          \
extern "C" __global__ __launch_bounds__(THREADS, MIN_BLOCKS) void NAME(   \
    float* output, const float* a, const float* b, const float* bias,       \
    GbfTf32Params params) {                                                 \
    gbf_tf32_entry<BM, BN, STAGES>(output, a, b, bias, params);             \
}

GBF_TF32_KERNEL(gemm_bi_nn_tf32_v1_m128n64_bk32_s2, 128, 64, 2, 256, 1)
GBF_TF32_KERNEL(gemm_bi_nn_tf32_v1_m128n64_bk32_s3, 128, 64, 3, 256, 1)
extern "C" __global__ __launch_bounds__(128, 1)
void gemm_bi_nn_tf32_v1_m64n64_bk32_s2(
    float* output, const float* a, const float* b, const float* bias,
    GbfTf32Params params) {
    gbf_tf32_entry<64, 64, 2, true>(output, a, b, bias, params);
}
GBF_TF32_KERNEL(gemm_bi_nn_tf32_v1_m64n64_bk32_s3, 64, 64, 3, 128, 1)
GBF_TF32_KERNEL(gemm_bi_nn_tf32_v1_m16n32_bk32_s4, 16, 32, 4, 128, 3)

#undef GBF_TF32_KERNEL
