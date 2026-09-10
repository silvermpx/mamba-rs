// Test-only SM120 TF32 TN split-reduction tournament. Production registration
// is forbidden.
//
// The production TN route owns one output tile per CTA and walks the whole
// reduction, so a cell with few output tiles leaves most of the 170 SMs idle:
// the prism dW cell launches 96 CTAs at one CTA per SM. These candidates cut
// the reduction into a fixed number of contiguous partitions, one CTA each,
// and combine the partitions in a fixed ascending order, so the result is
// deterministic and replay-stable but forms its own numeric family: it is not
// bit-identical to the single-chain production route and must never be
// admitted as if it were.
//
// Per CTA the mainloop is the production mainloop over the partition's tile
// range. Every partition stores its accumulators to a private scratch slab,
// the last CTA to arrive at an output tile (an integer completion counter,
// never a floating-point atomic) reads the slabs in partition order, adds them
// with __fadd_rn from partition 0 upwards, and applies the production TN
// epilogue. The counter wraps back to zero on the last arrival, so graph
// replay needs no reset.
#if defined(__CUDA_ARCH__) && \
    (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)

template <int M, int N, int Stages>
static __device__ __forceinline__ void sm120_tf32_tn_splitm_produce_stage(
    const Sm120Tf32StageContext& context, int local_tile, int absolute_tile) {
    constexpr int plane_bytes = 4096;
    constexpr int stage_bytes = Sm120Tf32Storage<M, N, Stages>::stage_bytes;
    unsigned stage_index = (unsigned)(local_tile % Stages);
    unsigned stage = context.payload + stage_index * stage_bytes;
    unsigned barrier = context.full_base + stage_index * 8;
    unsigned a_destination = stage;
    unsigned b_destination = stage + M * 32 * 4;
    unsigned long long a_descriptor =
        reinterpret_cast<unsigned long long>(context.a_map);
    unsigned long long b_descriptor =
        reinterpret_cast<unsigned long long>(context.b_map);
    const Sm120KernelParams& params = *context.params;
    int reduction = absolute_tile * 32;
    sm120_expect_transaction<stage_bytes>(barrier);
#pragma unroll
    for (int plane = 0; plane < M / 32; ++plane) {
        sm120_tma_copy(a_destination + plane * plane_bytes, a_descriptor,
            context.output_row + plane * 32, reduction,
            params.a_x, params.a_y, barrier);
    }
#pragma unroll
    for (int plane = 0; plane < N / 32; ++plane) {
        sm120_tma_copy(b_destination + plane * plane_bytes, b_descriptor,
            context.output_column + plane * 32, reduction,
            params.b_x, params.b_y, barrier);
    }
}

static __device__ __forceinline__ void sm120_tf32_splitm_store_partial(
    float* address, float2 value) {
    asm volatile("st.global.cg.v2.f32 [%0], {%1, %2};\n" ::
        "l"(address), "f"(value.x), "f"(value.y) : "memory");
}

static __device__ __forceinline__ void sm120_tf32_splitm_store_partial_one(
    float* address, float value) {
    asm volatile("st.global.cg.f32 [%0], %1;\n" ::
        "l"(address), "f"(value) : "memory");
}

static __device__ __forceinline__ float2 sm120_tf32_splitm_load_partial(
    const float* address) {
    float2 value;
    asm volatile("ld.global.cg.v2.f32 {%0, %1}, [%2];\n" :
        "=f"(value.x), "=f"(value.y) : "l"(address) : "memory");
    return value;
}

static __device__ __forceinline__ float sm120_tf32_splitm_load_partial_one(
    const float* address) {
    float value;
    asm volatile("ld.global.cg.f32 %0, [%1];\n" :
        "=f"(value) : "l"(address) : "memory");
    return value;
}

template <int M, int N, int Stages, int Partitions>
static __device__ __forceinline__ void sm120_tf32_tn_splitm_pair_kernel(
    void* output, float* partial, unsigned* counters,
    const CUtensorMap& a_map, const CUtensorMap& b_map,
    const float* bias, const Sm120KernelParams& params) {
    constexpr int Op = Sm120Tn;
    static_assert(Partitions >= 2 && Partitions <= 8, "split factor out of range");
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
    int partition = (int)blockIdx.z;
    unsigned full_tiles = (unsigned)(1 + (reduction - 1) / 32);
    unsigned tiles_per_partition = (full_tiles + Partitions - 1U) / Partitions;
    unsigned tile_begin = min((unsigned)partition * tiles_per_partition, full_tiles);
    unsigned tile_end = min(tile_begin + tiles_per_partition, full_tiles);
    int tile_count = (int)(tile_end - tile_begin);
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

    if (tile_count > 0) {
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
                    sm120_tf32_tn_splitm_produce_stage<M, N, Stages>(
                        stage_context, tile, (int)tile_begin + tile);
                }
            }
        }
        if constexpr (Stages == 2) sm120_sync_warp();

        for (int tile = 0; tile < tile_count; ++tile) {
            int stage = tile % Stages;
            unsigned generation = (unsigned)(tile / Stages);
            sm120_wait_barrier(full_base + stage * 8, generation & 1U);
            sm120_tf32_issue_stage<Op, M, N, Stages>(
                storage, stage, warp_m, warp_n, accumulator);
            if (lane == 0) {
                sm120_arrive_empty(empty_base + stage * 8);
            }
            if (warp == 0 && lane == 0) {
                int refill = tile + Stages;
                if (refill < tile_count) {
                    sm120_wait_barrier(empty_base + stage * 8, generation & 1U);
                    sm120_tf32_tn_splitm_produce_stage<M, N, Stages>(
                        stage_context, refill, (int)tile_begin + refill);
                }
            }
            if constexpr (Stages == 2) sm120_sync_warp();
        }
    }

    // Every partition, including one that owns no tiles, publishes its
    // accumulators so the reducer reads exactly Partitions slabs per tile.
    long long slab = (long long)rows * columns;
    float* mine = partial + (long long)partition * slab;
    bool full_tile = output_row + M <= rows && output_column + N <= columns;
    bool packed = full_tile && (columns & 1) == 0;
#pragma unroll
    for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 4; ++n_atom) {
#pragma unroll
            for (int element = 0; element < 4; element += 2) {
                int row = output_row + warp_m + m_atom * 16
                    + group + (element >= 2 ? 8 : 0);
                int column = output_column + warp_n + n_atom * 8 + 2 * thread;
                float* destination = mine + (long long)row * columns + column;
                if (packed) {
                    sm120_tf32_splitm_store_partial(destination, make_float2(
                        accumulator[m_atom][n_atom][element],
                        accumulator[m_atom][n_atom][element + 1]));
                } else if (row < rows) {
                    if (column < columns) {
                        sm120_tf32_splitm_store_partial_one(
                            destination, accumulator[m_atom][n_atom][element]);
                    }
                    if (column + 1 < columns) {
                        sm120_tf32_splitm_store_partial_one(
                            destination + 1, accumulator[m_atom][n_atom][element + 1]);
                    }
                }
            }
        }
    }
    // The completion flag lives in the spare tail of the barrier block, so
    // the kernel keeps the production layout with no static shared memory.
    __threadfence();
    __syncthreads();
    unsigned* last_partition = reinterpret_cast<unsigned*>(
        storage + Stages * stage_bytes + 120);
    if (threadIdx.x == 0) {
        *last_partition =
            atomicInc(counters + blockIdx.x, Partitions - 1U) == Partitions - 1U;
    }
    __syncthreads();
    if (*last_partition == 0U) return;

    // Fixed-order combine: partition 0 first, then 1, 2, ... regardless of
    // which CTA arrived last. This is the whole numeric contract of the family.
#pragma unroll
    for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 4; ++n_atom) {
#pragma unroll
            for (int element = 0; element < 4; element += 2) {
                int row = output_row + warp_m + m_atom * 16
                    + group + (element >= 2 ? 8 : 0);
                int column = output_column + warp_n + n_atom * 8 + 2 * thread;
                if (row >= rows || column >= columns) continue;
                long long index = (long long)row * columns + column;
                bool has_second = column + 1 < columns;
                bool pair = has_second && (columns & 1) == 0;
                float sum0 = 0.0f;
                float sum1 = 0.0f;
#pragma unroll
                for (int source = 0; source < Partitions; ++source) {
                    const float* slot = partial + (long long)source * slab + index;
                    float2 value;
                    if (pair) {
                        value = sm120_tf32_splitm_load_partial(slot);
                    } else {
                        value.x = sm120_tf32_splitm_load_partial_one(slot);
                        value.y = has_second
                            ? sm120_tf32_splitm_load_partial_one(slot + 1) : 0.0f;
                    }
                    if (source == 0) {
                        sum0 = value.x;
                        sum1 = value.y;
                    } else {
                        sum0 = __fadd_rn(sum0, value.x);
                        sum1 = __fadd_rn(sum1, value.y);
                    }
                }
                float* destination = static_cast<float*>(output)
                    + (long long)row * params.ldc + column;
                bool aligned = (reinterpret_cast<unsigned long long>(destination) & 7ULL) == 0ULL;
                if (has_second && aligned) {
                    float2 old = *reinterpret_cast<const float2*>(destination);
                    float first = __fmaf_rn(params.alpha, sum0, old.x);
                    float second = __fmaf_rn(params.alpha, sum1, old.y);
                    *reinterpret_cast<float2*>(destination) = make_float2(first, second);
                } else {
                    destination[0] = __fmaf_rn(params.alpha, sum0, destination[0]);
                    if (has_second) {
                        destination[1] = __fmaf_rn(params.alpha, sum1, destination[1]);
                    }
                }
            }
        }
    }
}

template <int M, int N, int Stages, int Partitions>
static __device__ __forceinline__ void sm120_tf32_tn_splitm_pair_entry(
    void* output, float* partial, unsigned* counters,
    const CUtensorMap& a_map, const CUtensorMap& b_map,
    const float* bias, const Sm120KernelParams& params) {
    if (sm120_tf32_reduction<Sm120Tn>(params) == 0) {
        if (blockIdx.z == 0) {
            sm120_tf32_zero_reduction_epilogue<Sm120Tn, M, N>(output, bias, params);
        }
        return;
    }
    sm120_tf32_tn_splitm_pair_kernel<M, N, Stages, Partitions>(
        output, partial, counters, a_map, b_map, bias, params);
}

#define SM120_DEFINE_TF32_TN_SPLITM_KERNEL(NAME, M, N, STAGES, PARTITIONS)   \
extern "C" __global__ __launch_bounds__((M * N) / 32) void NAME(            \
    void* output, float* partial, unsigned* counters,                          \
    const __grid_constant__ CUtensorMap a_map,                                 \
    const __grid_constant__ CUtensorMap b_map, const float* bias,              \
    const __grid_constant__ Sm120KernelParams params) {                        \
    sm120_tf32_tn_splitm_pair_entry<M, N, STAGES, PARTITIONS>(                 \
        output, partial, counters, a_map, b_map, bias, params);                \
}

SM120_DEFINE_TF32_TN_SPLITM_KERNEL(
    gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s4_pair_exp_splitm7_v1, 64, 128, 4, 7)
SM120_DEFINE_TF32_TN_SPLITM_KERNEL(
    gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s2_pair_exp_splitm7_v1, 64, 128, 2, 7)
SM120_DEFINE_TF32_TN_SPLITM_KERNEL(
    gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s4_pair_exp_splitm5_v1, 64, 128, 4, 5)
SM120_DEFINE_TF32_TN_SPLITM_KERNEL(
    gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s4_pair_exp_splitm4_v1, 64, 128, 4, 4)

#endif
