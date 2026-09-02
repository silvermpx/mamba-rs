// Test-only SM120 TF32 NN rect-wide tournament. Production registration is forbidden.
#if defined(__CUDA_ARCH__) && \
    (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)

template <int M, int StorageN, int LogicalBK, int Stages>
struct Sm120Tf32NextStorage {
    static constexpr int slabs = LogicalBK / 32;
    static constexpr int slab_bytes = (M + StorageN) * 32 * 4;
    static constexpr int stage_bytes = slabs * slab_bytes;
    static constexpr int dynamic_bytes = 128 + Stages * stage_bytes;
};

static_assert(Sm120Tf32NextStorage<80, 32, 32, 2>::dynamic_bytes == 28800);
static_assert(Sm120Tf32NextStorage<64, 64, 32, 2>::dynamic_bytes == 32896);
static_assert(Sm120Tf32NextStorage<80, 32, 32, 3>::dynamic_bytes == 43136);
static_assert(Sm120Tf32NextStorage<80, 32, 64, 2>::dynamic_bytes == 57472);

template <int M, int StorageN, int LogicalBK, int Stages>
static __device__ __forceinline__ void sm120_tf32_next_produce_stage(
    const Sm120Tf32StageContext& context, int tile) {
    constexpr int slabs = Sm120Tf32NextStorage<
        M, StorageN, LogicalBK, Stages>::slabs;
    constexpr int slab_bytes = Sm120Tf32NextStorage<
        M, StorageN, LogicalBK, Stages>::slab_bytes;
    constexpr int stage_bytes = Sm120Tf32NextStorage<
        M, StorageN, LogicalBK, Stages>::stage_bytes;
    constexpr int a_bytes = M * 32 * 4;
    constexpr int b_plane_bytes = 32 * 32 * 4;
    unsigned stage_index = (unsigned)(tile % Stages);
    unsigned stage = context.payload + stage_index * stage_bytes;
    unsigned barrier = context.full_base + stage_index * 8;
    unsigned long long a_descriptor =
        reinterpret_cast<unsigned long long>(context.a_map);
    unsigned long long b_descriptor =
        reinterpret_cast<unsigned long long>(context.b_map);
    const Sm120KernelParams& params = *context.params;
    sm120_expect_transaction<stage_bytes>(barrier);
#pragma unroll
    for (int slab = 0; slab < slabs; ++slab) {
        unsigned slab_base = stage + slab * slab_bytes;
        int reduction = tile * LogicalBK + slab * 32;
        sm120_tma_copy(slab_base, a_descriptor, reduction,
            context.output_row, params.a_x, params.a_y, barrier);
#pragma unroll
        for (int plane = 0; plane < StorageN / 32; ++plane) {
            sm120_tma_copy(slab_base + a_bytes + plane * b_plane_bytes,
                b_descriptor, context.output_column + plane * 32, reduction,
                params.b_x, params.b_y, barrier);
        }
    }
}

template <int M, int StorageN, int LogicalBK, int Stages>
static __device__ __forceinline__ float sm120_tf32_next_load_a(
    unsigned char* storage, int stage, int row, int reduction) {
    constexpr int slab_bytes = Sm120Tf32NextStorage<
        M, StorageN, LogicalBK, Stages>::slab_bytes;
    constexpr int stage_bytes = Sm120Tf32NextStorage<
        M, StorageN, LogicalBK, Stages>::stage_bytes;
    int slab = reduction / 32;
    unsigned slab_base = (unsigned)(slab * slab_bytes);
    unsigned offset = sm120_tf32_sw128_offset(
        slab_base, (unsigned)row, (unsigned)(reduction & 31));
    return *reinterpret_cast<float*>(
        storage + stage * stage_bytes + offset);
}

template <int M, int StorageN, int LogicalBK, int Stages>
static __device__ __forceinline__ float sm120_tf32_next_load_b(
    unsigned char* storage, int stage, int reduction, int column) {
    constexpr int slab_bytes = Sm120Tf32NextStorage<
        M, StorageN, LogicalBK, Stages>::slab_bytes;
    constexpr int stage_bytes = Sm120Tf32NextStorage<
        M, StorageN, LogicalBK, Stages>::stage_bytes;
    constexpr int a_bytes = M * 32 * 4;
    int slab = reduction / 32;
    unsigned slab_base = (unsigned)(slab * slab_bytes);
    unsigned plane = (unsigned)(column / 32) * 4096U;
    unsigned offset = sm120_tf32_sw128_offset(
        slab_base + a_bytes + plane, (unsigned)(reduction & 31),
        (unsigned)(column & 31));
    return *reinterpret_cast<float*>(
        storage + stage * stage_bytes + offset);
}

template <int NAtoms, int M, int StorageN, int LogicalBK, int Stages>
static __device__ __forceinline__ void sm120_tf32_next_load_issue(
    unsigned char* storage, int stage, int warp_m, int k8,
    unsigned (&a_fragment)[1][4], unsigned (&b_fragment)[NAtoms][2]) {
    int lane = (int)threadIdx.x & 31;
    int group = lane >> 2;
    int thread = lane & 3;
    int row = warp_m + group;
    a_fragment[0][0] = sm120_tf32_rna(
        sm120_tf32_next_load_a<M, StorageN, LogicalBK, Stages>(
            storage, stage, row, k8 + thread));
    a_fragment[0][1] = sm120_tf32_rna(
        sm120_tf32_next_load_a<M, StorageN, LogicalBK, Stages>(
            storage, stage, row + 8, k8 + thread));
    a_fragment[0][2] = sm120_tf32_rna(
        sm120_tf32_next_load_a<M, StorageN, LogicalBK, Stages>(
            storage, stage, row, k8 + thread + 4));
    a_fragment[0][3] = sm120_tf32_rna(
        sm120_tf32_next_load_a<M, StorageN, LogicalBK, Stages>(
            storage, stage, row + 8, k8 + thread + 4));
#pragma unroll
    for (int n_atom = 0; n_atom < NAtoms; ++n_atom) {
        int column = n_atom * 8 + group;
        b_fragment[n_atom][0] = sm120_tf32_rna(
            sm120_tf32_next_load_b<M, StorageN, LogicalBK, Stages>(
                storage, stage, k8 + thread, column));
        b_fragment[n_atom][1] = sm120_tf32_rna(
            sm120_tf32_next_load_b<M, StorageN, LogicalBK, Stages>(
                storage, stage, k8 + thread + 4, column));
    }
}

template <int NAtoms, int M, int StorageN, int LogicalBK, int Stages>
static __device__ __forceinline__ void sm120_tf32_next_issue_stage(
    unsigned char* storage, int stage, int warp_m,
    float (&accumulator)[1][NAtoms][4]) {
    unsigned a_fragments[2][1][4];
    unsigned b_fragments[2][NAtoms][2];
    sm120_tf32_next_load_issue<NAtoms, M, StorageN, LogicalBK, Stages>(
        storage, stage, warp_m, 0, a_fragments[0], b_fragments[0]);
#pragma unroll
    for (int issue = 0; issue < LogicalBK / 8; ++issue) {
        int current = issue & 1;
        int next = current ^ 1;
        if (issue + 1 < LogicalBK / 8) {
            sm120_tf32_next_load_issue<
                NAtoms, M, StorageN, LogicalBK, Stages>(storage, stage,
                warp_m, issue * 8 + 8, a_fragments[next], b_fragments[next]);
        }
#pragma unroll
        for (int n_atom = 0; n_atom < NAtoms; ++n_atom) {
            sm120_tf32_mma_m16n8k8(accumulator[0][n_atom],
                a_fragments[current][0], b_fragments[current][n_atom]);
        }
    }
}

template <int M, int N, int StorageN, int LogicalBK, int Stages,
          int ComputeWarps, bool ProducerWarp>
static __device__ __forceinline__ void sm120_tf32_next_kernel(
    void* output, const CUtensorMap& a_map, const CUtensorMap& b_map,
    const float* bias, const Sm120KernelParams& params) {
    constexpr int NAtoms = (N + 7) / 8;
    constexpr int stage_bytes = Sm120Tf32NextStorage<
        M, StorageN, LogicalBK, Stages>::stage_bytes;
    assert(params.alpha == 1.0f && params.beta == 0.0f && bias == nullptr);
    extern __shared__ __align__(1024) unsigned char storage[];
    unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(storage));
    unsigned payload = shared;
    unsigned full_base = shared + Stages * stage_bytes;
    unsigned empty_base = full_base + 64;
    int column_tiles = 1 + (params.n - 1) / N;
    int output_row = (int)blockIdx.x / column_tiles * M;
    int output_column = (int)blockIdx.x % column_tiles * N;
    int tile_count = 1 + (params.k - 1) / LogicalBK;
    const Sm120Tf32StageContext stage_context = {
        &a_map, &b_map, &params, payload, full_base, output_row, output_column};
    int warp = (int)threadIdx.x >> 5;
    int lane = (int)threadIdx.x & 31;

    if (threadIdx.x == 0) {
#pragma unroll
        for (int stage = 0; stage < Stages; ++stage) {
            sm120_init_barrier<1>(full_base + stage * 8);
            sm120_init_barrier<ComputeWarps>(empty_base + stage * 8);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    if constexpr (ProducerWarp) {
        if (warp == 0) {
            if (lane == 0) {
#pragma unroll
                for (int tile = 0; tile < Stages; ++tile) {
                    if (tile < tile_count) {
                        sm120_tf32_next_produce_stage<
                            M, StorageN, LogicalBK, Stages>(stage_context, tile);
                    }
                }
                for (int refill = Stages; refill < tile_count; ++refill) {
                    int consumed = refill - Stages;
                    int stage = consumed % Stages;
                    unsigned generation = (unsigned)(consumed / Stages);
                    sm120_wait_barrier(empty_base + stage * 8, generation & 1U);
                    sm120_tf32_next_produce_stage<
                        M, StorageN, LogicalBK, Stages>(stage_context, refill);
                }
            }
            return;
        }
    } else {
        if (warp == 0 && lane == 0) {
#pragma unroll
            for (int tile = 0; tile < Stages; ++tile) {
                if (tile < tile_count) {
                    sm120_tf32_next_produce_stage<
                        M, StorageN, LogicalBK, Stages>(stage_context, tile);
                }
            }
        }
        if constexpr (Stages == 2) sm120_sync_warp();
    }

    int compute_warp = ProducerWarp ? warp - 1 : warp;
    int warp_m = compute_warp * 16;
    int group = lane >> 2;
    int thread = lane & 3;
    float accumulator[1][NAtoms][4] = {};

    for (int tile = 0; tile < tile_count; ++tile) {
        int stage = tile % Stages;
        unsigned generation = (unsigned)(tile / Stages);
        sm120_wait_barrier(full_base + stage * 8, generation & 1U);
        sm120_tf32_next_issue_stage<
            NAtoms, M, StorageN, LogicalBK, Stages>(
                storage, stage, warp_m, accumulator);
        if (lane == 0) sm120_arrive_empty(empty_base + stage * 8);
        if constexpr (!ProducerWarp) {
            if (warp == 0 && lane == 0) {
                int refill = tile + Stages;
                if (refill < tile_count) {
                    sm120_wait_barrier(
                        empty_base + stage * 8, generation & 1U);
                    sm120_tf32_next_produce_stage<
                        M, StorageN, LogicalBK, Stages>(stage_context, refill);
                }
            }
            if constexpr (Stages == 2) sm120_sync_warp();
        }
    }

#pragma unroll
    for (int n_atom = 0; n_atom < NAtoms; ++n_atom) {
#pragma unroll
        for (int element = 0; element < 4; ++element) {
            int row = output_row + warp_m + group + (element >= 2 ? 8 : 0);
            int column = output_column + n_atom * 8
                + 2 * thread + (element & 1);
            if (column < output_column + N) {
                sm120_tf32_store<Sm120Nn>(output, row, column,
                    accumulator[0][n_atom][element], bias, params);
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(160)
void gemm_bi_nn_sm120_tma_mma_tf32_exp_m80n32_bk32_s2_v1(
    void* output, const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap b_map, const float* bias,
    const __grid_constant__ Sm120KernelParams params) {
    sm120_tf32_next_kernel<80, 32, 32, 32, 2, 5, false>(
        output, a_map, b_map, bias, params);
}

extern "C" __global__ __launch_bounds__(128)
void gemm_bi_nn_sm120_tma_mma_tf32_exp_m64n40_bk32_s2_v1(
    void* output, const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap b_map, const float* bias,
    const __grid_constant__ Sm120KernelParams params) {
    sm120_tf32_next_kernel<64, 40, 64, 32, 2, 4, false>(
        output, a_map, b_map, bias, params);
}

extern "C" __global__ __launch_bounds__(192)
void gemm_bi_nn_sm120_tma_mma_tf32_exp_m80n32_bk32_s3_producer_v1(
    void* output, const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap b_map, const float* bias,
    const __grid_constant__ Sm120KernelParams params) {
    sm120_tf32_next_kernel<80, 32, 32, 32, 3, 5, true>(
        output, a_map, b_map, bias, params);
}

extern "C" __global__ __launch_bounds__(160)
void gemm_bi_nn_sm120_tma_mma_tf32_exp_m80n32_bk64_s2_dual32_v1(
    void* output, const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap b_map, const float* bias,
    const __grid_constant__ Sm120KernelParams params) {
    sm120_tf32_next_kernel<80, 32, 32, 64, 2, 5, false>(
        output, a_map, b_map, bias, params);
}

#endif
