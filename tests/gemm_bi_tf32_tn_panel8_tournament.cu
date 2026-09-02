// Test-only SM120 TF32 TN panel tournament. Production registration is forbidden.
//
// The production TN route stages both operands as 32x32 planes with 128-byte
// rows and the TMA 128B swizzle. A TN fragment load reads four consecutive
// reduction rows at eight consecutive output indices, and under that swizzle
// two of the four rows always land on the same bank pair, so every TN fragment
// load costs two shared-memory wavefronts instead of one.
//
// The candidates below keep every instruction that touches arithmetic - the
// ascending K walk, the cvt.rna conversion at the load, the m16n8k8 issue order
// and the pair epilogue - and change only where the bytes sit in shared memory:
// each operand is staged as 8-wide panels of 32 reduction rows, so a row is 32
// bytes and the four rows of one fragment load occupy 128 contiguous bytes.
#if defined(__CUDA_ARCH__) && \
    (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)

static constexpr int SM120_PANEL_WIDTH = 8;
static constexpr int SM120_PANEL_BYTES = SM120_PANEL_WIDTH * 32 * 4;
static constexpr int SM120_PANEL_ROW_BYTES = SM120_PANEL_WIDTH * 4;

static_assert(SM120_PANEL_BYTES == 1024, "a panel holds 32 rows of 8 elements");

static __device__ __forceinline__ void sm120_tf32_panel8_tma_copy(
    unsigned destination, unsigned long long map, int reduction, int panel,
    unsigned barrier) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(destination), "l"(map), "r"(0), "r"(reduction), "r"(panel),
           "r"(barrier)
        : "memory");
}

// One TMA box per operand per stage: the box spans the whole tile in the
// output dimension (M/8 or N/8 panels) and 32 reduction rows.
template <int M, int N, int Stages>
static __device__ __forceinline__ void sm120_tf32_panel8_produce_stage(
    const Sm120Tf32StageContext& context, int tile) {
    constexpr int stage_bytes = Sm120Tf32Storage<M, N, Stages>::stage_bytes;
    unsigned stage_index = (unsigned)(tile % Stages);
    unsigned stage = context.payload + stage_index * stage_bytes;
    unsigned barrier = context.full_base + stage_index * 8;
    unsigned a_destination = stage;
    unsigned b_destination = stage + M * 32 * 4;
    unsigned long long a_descriptor =
        reinterpret_cast<unsigned long long>(context.a_map);
    unsigned long long b_descriptor =
        reinterpret_cast<unsigned long long>(context.b_map);
    int reduction = tile * 32;
    sm120_expect_transaction<stage_bytes>(barrier);
    sm120_tf32_panel8_tma_copy(a_destination, a_descriptor, reduction,
        context.output_row / SM120_PANEL_WIDTH, barrier);
    sm120_tf32_panel8_tma_copy(b_destination, b_descriptor, reduction,
        context.output_column / SM120_PANEL_WIDTH, barrier);
}

template <int M, int N, int Stages>
static __device__ __forceinline__ float sm120_tf32_panel8_load_a(
    unsigned char* storage, int stage, int row, int reduction) {
    constexpr int stage_bytes = Sm120Tf32Storage<M, N, Stages>::stage_bytes;
    unsigned offset = (unsigned)(stage * stage_bytes)
        + (unsigned)(row / SM120_PANEL_WIDTH) * SM120_PANEL_BYTES
        + (unsigned)reduction * SM120_PANEL_ROW_BYTES
        + (unsigned)(row % SM120_PANEL_WIDTH) * 4;
    return *reinterpret_cast<float*>(storage + offset);
}

template <int M, int N, int Stages>
static __device__ __forceinline__ float sm120_tf32_panel8_load_b(
    unsigned char* storage, int stage, int reduction, int column) {
    constexpr int stage_bytes = Sm120Tf32Storage<M, N, Stages>::stage_bytes;
    unsigned offset = (unsigned)(stage * stage_bytes) + M * 32 * 4
        + (unsigned)(column / SM120_PANEL_WIDTH) * SM120_PANEL_BYTES
        + (unsigned)reduction * SM120_PANEL_ROW_BYTES
        + (unsigned)(column % SM120_PANEL_WIDTH) * 4;
    return *reinterpret_cast<float*>(storage + offset);
}

// Same fragment order and the same conversion point as the production loader.
template <int M, int N, int Stages>
static __device__ __forceinline__ void sm120_tf32_panel8_load_issue(
    unsigned char* storage, int stage, int warp_m, int warp_n, int k8,
    unsigned (&a_fragments)[2][4], unsigned (&b_fragments)[4][2]) {
    int lane = (int)threadIdx.x & 31;
    int group = lane >> 2;
    int thread = lane & 3;
#pragma unroll
    for (int m_atom = 0; m_atom < 2; ++m_atom) {
        int row = warp_m + m_atom * 16 + group;
        a_fragments[m_atom][0] = sm120_tf32_rna(
            sm120_tf32_panel8_load_a<M, N, Stages>(storage, stage, row, k8 + thread));
        a_fragments[m_atom][1] = sm120_tf32_rna(
            sm120_tf32_panel8_load_a<M, N, Stages>(storage, stage, row + 8, k8 + thread));
        a_fragments[m_atom][2] = sm120_tf32_rna(
            sm120_tf32_panel8_load_a<M, N, Stages>(storage, stage, row, k8 + thread + 4));
        a_fragments[m_atom][3] = sm120_tf32_rna(
            sm120_tf32_panel8_load_a<M, N, Stages>(storage, stage, row + 8, k8 + thread + 4));
    }
#pragma unroll
    for (int n_atom = 0; n_atom < 4; ++n_atom) {
        int column = warp_n + n_atom * 8 + group;
        b_fragments[n_atom][0] = sm120_tf32_rna(
            sm120_tf32_panel8_load_b<M, N, Stages>(storage, stage, k8 + thread, column));
        b_fragments[n_atom][1] = sm120_tf32_rna(
            sm120_tf32_panel8_load_b<M, N, Stages>(storage, stage, k8 + thread + 4, column));
    }
}

template <int M, int N, int Stages>
static __device__ __forceinline__ void sm120_tf32_panel8_issue_stage(
    unsigned char* storage, int stage, int warp_m, int warp_n,
    float (&accumulator)[2][4][4]) {
    unsigned a_fragments[2][2][4];
    unsigned b_fragments[2][4][2];
    sm120_tf32_panel8_load_issue<M, N, Stages>(
        storage, stage, warp_m, warp_n, 0, a_fragments[0], b_fragments[0]);
#pragma unroll
    for (int issue = 0; issue < 4; ++issue) {
        int k8 = issue * 8;
        int current = issue & 1;
        int next = current ^ 1;
        if (issue + 1 < 4) {
            sm120_tf32_panel8_load_issue<M, N, Stages>(storage, stage,
                warp_m, warp_n, k8 + 8, a_fragments[next], b_fragments[next]);
        }
#pragma unroll
        for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
            for (int n_atom = 0; n_atom < 4; ++n_atom) {
                sm120_tf32_mma_m16n8k8(
                    accumulator[m_atom][n_atom],
                    a_fragments[current][m_atom], b_fragments[current][n_atom]);
            }
        }
    }
}

template <int M, int N, int Stages>
static __device__ __forceinline__ void sm120_tf32_tn_panel8_pair_kernel(
    void* output, const CUtensorMap& a_map, const CUtensorMap& b_map,
    const float* bias, const Sm120KernelParams& params) {
    constexpr int Op = Sm120Tn;
    assert(params.alpha == 1.0f || bias == nullptr);
    extern __shared__ __align__(1024) unsigned char storage[];
    constexpr int stage_bytes = Sm120Tf32Storage<M, N, Stages>::stage_bytes;
    constexpr int warps = Sm120Tf32Storage<M, N, Stages>::threads / 32;
    unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(storage));
    unsigned payload = shared;
    unsigned full_base = shared + Stages * stage_bytes;
    unsigned empty_base = full_base + 64;
    int rows = sm120_tf32_rows<Op>(params);
    int columns = sm120_tf32_columns<Op>(params);
    int column_tiles = 1 + (columns - 1) / N;
    int output_row = (int)blockIdx.x / column_tiles * M;
    int output_column = (int)blockIdx.x % column_tiles * N;
    int reduction = sm120_tf32_reduction<Op>(params);
    int tile_count = 1 + (reduction - 1) / 32;
    const Sm120Tf32StageContext stage_context = {
        &a_map, &b_map, &params, payload, full_base, output_row, output_column};
    int warp = (int)threadIdx.x >> 5;
    constexpr int warp_columns = N / 32;
    int warp_m = (warp / warp_columns) * 32;
    int warp_n = (warp % warp_columns) * 32;
    int lane = (int)threadIdx.x & 31;
    int group = lane >> 2;
    int thread = lane & 3;
    float accumulator[2][4][4];

#pragma unroll
    for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 4; ++n_atom) {
#pragma unroll
            for (int element = 0; element < 4; ++element) {
                accumulator[m_atom][n_atom][element] = 0.0f;
            }
        }
    }

    if (threadIdx.x == 0) {
#pragma unroll
        for (int stage = 0; stage < Stages; ++stage) {
            sm120_init_barrier<1>(full_base + stage * 8);
            sm120_init_barrier<warps>(empty_base + stage * 8);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    if (warp == 0 && lane == 0) {
#pragma unroll
        for (int tile = 0; tile < Stages; ++tile) {
            if (tile < tile_count) {
                sm120_tf32_panel8_produce_stage<M, N, Stages>(stage_context, tile);
            }
        }
    }
    if constexpr (Stages == 2) sm120_sync_warp();

    for (int tile = 0; tile < tile_count; ++tile) {
        int stage = tile % Stages;
        unsigned generation = (unsigned)(tile / Stages);
        sm120_wait_barrier(full_base + stage * 8, generation & 1U);
        sm120_tf32_panel8_issue_stage<M, N, Stages>(
            storage, stage, warp_m, warp_n, accumulator);
        if (lane == 0) {
            sm120_arrive_empty(empty_base + stage * 8);
        }
        if (warp == 0 && lane == 0) {
            int refill = tile + Stages;
            if (refill < tile_count) {
                sm120_wait_barrier(empty_base + stage * 8, generation & 1U);
                sm120_tf32_panel8_produce_stage<M, N, Stages>(
                    stage_context, refill);
            }
        }
        if constexpr (Stages == 2) sm120_sync_warp();
    }

    bool full_tile = output_row + M <= rows
        && output_column + N <= columns
        && (reinterpret_cast<unsigned long long>(output) & 7ULL) == 0ULL
        && (params.ldc & 1) == 0;
#pragma unroll
    for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 4; ++n_atom) {
#pragma unroll
            for (int element = 0; element < 4; element += 2) {
                int row = output_row + warp_m + m_atom * 16
                    + group + (element >= 2 ? 8 : 0);
                int column = output_column + warp_n + n_atom * 8
                    + 2 * thread;
                Sm120Tf32PairValue pair = {
                    accumulator[m_atom][n_atom][element],
                    accumulator[m_atom][n_atom][element + 1]};
                sm120_tf32_store_pair<Op>(
                    output, row, column, pair, bias, params, full_tile);
            }
        }
    }
}

template <int M, int N, int Stages>
static __device__ __forceinline__ void sm120_tf32_tn_panel8_pair_entry(
    void* output, const CUtensorMap& a_map, const CUtensorMap& b_map,
    const float* bias, const Sm120KernelParams& params) {
    if (sm120_tf32_reduction<Sm120Tn>(params) == 0) {
        sm120_tf32_zero_reduction_epilogue<Sm120Tn, M, N>(output, bias, params);
        return;
    }
    sm120_tf32_tn_panel8_pair_kernel<M, N, Stages>(
        output, a_map, b_map, bias, params);
}

#define SM120_DEFINE_TF32_TN_PANEL8_KERNEL(NAME, M, N, STAGES)               \
extern "C" __global__ __launch_bounds__((M * N) / 32) void NAME(            \
    void* output, const __grid_constant__ CUtensorMap a_map,                   \
    const __grid_constant__ CUtensorMap b_map, const float* bias,              \
    const __grid_constant__ Sm120KernelParams params) {                        \
    sm120_tf32_tn_panel8_pair_entry<M, N, STAGES>(                             \
        output, a_map, b_map, bias, params);                                   \
}

SM120_DEFINE_TF32_TN_PANEL8_KERNEL(
    gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s4_pair_exp_panel8_v1, 64, 128, 4)
SM120_DEFINE_TF32_TN_PANEL8_KERNEL(
    gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s3_pair_exp_panel8_v1, 64, 128, 3)
SM120_DEFINE_TF32_TN_PANEL8_KERNEL(
    gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s2_pair_exp_panel8_v1, 64, 128, 2)
SM120_DEFINE_TF32_TN_PANEL8_KERNEL(
    gemm_bi_tn_sm120_tma_mma_tf32_v1_m128n64_bk32_s3_pair_exp_panel8_v1, 128, 64, 3)
SM120_DEFINE_TF32_TN_PANEL8_KERNEL(
    gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n64_bk32_s2_pair_exp_panel8_v1, 64, 64, 2)

#endif
