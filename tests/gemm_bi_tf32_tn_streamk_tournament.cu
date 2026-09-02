// Test-only SM120 TF32 TN stream-K tournament. Production registration is
// forbidden.
//
// The reduction of every output tile is cut into K-tile units and the whole
// (tile, k) space is dealt to a fixed number of persistent CTAs in equal
// contiguous ranges, so the machine stays full regardless of how many output
// tiles the cell has. A CTA whose range covers a whole tile runs the
// production epilogue directly. A tile that straddles CTAs is combined by the
// CTA holding its last unit, in ascending CTA order, with __fadd_rn: the
// partition points are a pure function of the shape and the grid, so the bits
// are deterministic and replay-stable, while forming their own numeric family.
//
// Contributors publish their accumulators in fragment order (each thread
// writes its own 32 values), fence, and raise a per-slot flag. The owner
// spins on the flags of the lower contributors, adds their slabs in order,
// adds its own, clears the flags for the next launch, and stores. Because
// every dependency points at a lower CTA and the harness admits a grid no
// larger than what the device keeps resident at once, no CTA waits on a CTA
// that cannot run.
//
// Each CTA walks its range from the end: the segment it publishes comes
// first, whole tiles next, the segment it owns last. A publisher therefore
// never waits, and every owner waits only on slabs that are already being
// produced. Walking forwards would put each owner behind the owner below it,
// whose publish comes after its own wait, and the grid would run as one
// serial chain.
#if defined(__CUDA_ARCH__) && \
    (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)

static constexpr int SM120_STREAMK_ACCUMULATORS = 32;

struct Sm120StreamKRange {
    long long first;
    long long last;
};

// Units [first, last) of CTA `cta`: the first `remainder` CTAs take one unit
// more, so every CTA differs from any other by at most one unit.
static __device__ __forceinline__ Sm120StreamKRange sm120_streamk_range(
    long long units, int grid, int cta) {
    long long base = units / grid;
    long long remainder = units % grid;
    long long first = (long long)cta * base + min((long long)cta, remainder);
    long long last = first + base + (cta < remainder ? 1 : 0);
    return {first, last};
}

// The CTA whose range contains `unit`: the inverse of the dealing formula.
static __device__ __forceinline__ int sm120_streamk_cta_of(
    long long units, int grid, long long unit) {
    long long base = units / grid;
    long long remainder = units % grid;
    long long wide = remainder * (base + 1);
    if (unit < wide) return (int)(unit / (base + 1));
    return (int)(remainder + (unit - wide) / base);
}

static __device__ __forceinline__ void sm120_streamk_store_slab(
    float* slab, const float (&accumulator)[2][4][4]) {
    float4* destination = reinterpret_cast<float4*>(
        slab + (long long)threadIdx.x * SM120_STREAMK_ACCUMULATORS);
#pragma unroll
    for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 4; ++n_atom) {
            float4 value = make_float4(
                accumulator[m_atom][n_atom][0], accumulator[m_atom][n_atom][1],
                accumulator[m_atom][n_atom][2], accumulator[m_atom][n_atom][3]);
            asm volatile("st.global.cg.v4.f32 [%0], {%1, %2, %3, %4};\n" ::
                "l"(destination + m_atom * 4 + n_atom),
                "f"(value.x), "f"(value.y), "f"(value.z), "f"(value.w)
                : "memory");
        }
    }
}

static __device__ __forceinline__ void sm120_streamk_add_slab(
    const float* slab, float (&accumulator)[2][4][4], bool first) {
    const float4* source = reinterpret_cast<const float4*>(
        slab + (long long)threadIdx.x * SM120_STREAMK_ACCUMULATORS);
#pragma unroll
    for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 4; ++n_atom) {
            float4 value;
            asm volatile("ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];\n"
                : "=f"(value.x), "=f"(value.y), "=f"(value.z), "=f"(value.w)
                : "l"(source + m_atom * 4 + n_atom) : "memory");
            if (first) {
                accumulator[m_atom][n_atom][0] = value.x;
                accumulator[m_atom][n_atom][1] = value.y;
                accumulator[m_atom][n_atom][2] = value.z;
                accumulator[m_atom][n_atom][3] = value.w;
            } else {
                accumulator[m_atom][n_atom][0] =
                    __fadd_rn(accumulator[m_atom][n_atom][0], value.x);
                accumulator[m_atom][n_atom][1] =
                    __fadd_rn(accumulator[m_atom][n_atom][1], value.y);
                accumulator[m_atom][n_atom][2] =
                    __fadd_rn(accumulator[m_atom][n_atom][2], value.z);
                accumulator[m_atom][n_atom][3] =
                    __fadd_rn(accumulator[m_atom][n_atom][3], value.w);
            }
        }
    }
}

static __device__ __forceinline__ void sm120_streamk_raise(unsigned* flag) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;\n" :: "l"(flag), "r"(1U) : "memory");
}

static __device__ __forceinline__ void sm120_streamk_await(const unsigned* flag) {
    unsigned value;
    do {
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(value) : "l"(flag) : "memory");
    } while (value == 0U);
}

static __device__ __forceinline__ void sm120_streamk_clear(unsigned* flag) {
    asm volatile("st.relaxed.gpu.global.u32 [%0], %1;\n" :: "l"(flag), "r"(0U) : "memory");
}

template <int M, int N, int Stages>
static __device__ __forceinline__ void sm120_tf32_tn_streamk_produce_stage(
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

// The production mainloop over one contiguous K-tile range of one tile,
// starting from zeroed accumulators. Barriers are re-armed per segment.
template <int M, int N, int Stages>
static __device__ __forceinline__ void sm120_tf32_tn_streamk_segment(
    unsigned char* storage, unsigned payload, unsigned full_base,
    unsigned empty_base, const Sm120Tf32StageContext& stage_context,
    int tile_begin, int tile_count, int warp_m, int warp_n,
    float (&accumulator)[2][4][4]) {
    constexpr int warps = Sm120Tf32Storage<M, N, Stages>::threads / 32;
    int warp = (int)threadIdx.x >> 5;
    int lane = (int)threadIdx.x & 31;
    (void)payload;
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
                sm120_tf32_tn_streamk_produce_stage<M, N, Stages>(
                    stage_context, tile, tile_begin + tile);
            }
        }
    }
    if constexpr (Stages == 2) sm120_sync_warp();
    for (int tile = 0; tile < tile_count; ++tile) {
        int stage = tile % Stages;
        unsigned generation = (unsigned)(tile / Stages);
        sm120_wait_barrier(full_base + stage * 8, generation & 1U);
        sm120_tf32_issue_stage<Sm120Tn, M, N, Stages>(
            storage, stage, warp_m, warp_n, accumulator);
        if (lane == 0) {
            sm120_arrive_empty(empty_base + stage * 8);
        }
        if (warp == 0 && lane == 0) {
            int refill = tile + Stages;
            if (refill < tile_count) {
                sm120_wait_barrier(empty_base + stage * 8, generation & 1U);
                sm120_tf32_tn_streamk_produce_stage<M, N, Stages>(
                    stage_context, refill, tile_begin + refill);
            }
        }
        if constexpr (Stages == 2) sm120_sync_warp();
    }
    // Every warp has consumed its last stage before the storage is reused by
    // the next segment or the CTA leaves.
    __syncthreads();
}

template <int M, int N>
static __device__ __forceinline__ void sm120_tf32_tn_streamk_epilogue(
    void* output, int rows, int columns, int output_row, int output_column,
    int warp_m, int warp_n, const float* bias, const Sm120KernelParams& params,
    const float (&accumulator)[2][4][4]) {
    int lane = (int)threadIdx.x & 31;
    int group = lane >> 2;
    int thread = lane & 3;
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
                int column = output_column + warp_n + n_atom * 8 + 2 * thread;
                Sm120Tf32PairValue pair = {
                    accumulator[m_atom][n_atom][element],
                    accumulator[m_atom][n_atom][element + 1]};
                sm120_tf32_store_pair<Sm120Tn>(
                    output, row, column, pair, bias, params, full_tile);
            }
        }
    }
}

template <int M, int N, int Stages>
static __device__ __forceinline__ void sm120_tf32_tn_streamk_pair_kernel(
    void* output, float* partial, unsigned* flags,
    const CUtensorMap& a_map, const CUtensorMap& b_map,
    const float* bias, const Sm120KernelParams& params) {
    constexpr int Op = Sm120Tn;
    constexpr int threads = Sm120Tf32Storage<M, N, Stages>::threads;
    constexpr long long slab_floats = (long long)threads * SM120_STREAMK_ACCUMULATORS;
    assert(params.alpha == 1.0f || bias == nullptr);
    extern __shared__ __align__(1024) unsigned char storage[];
    constexpr int stage_bytes = Sm120Tf32Storage<M, N, Stages>::stage_bytes;
    unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(storage));
    unsigned payload = shared;
    unsigned full_base = shared + Stages * stage_bytes;
    unsigned empty_base = full_base + 64;
    int rows = sm120_tf32_rows<Op>(params);
    int columns = sm120_tf32_columns<Op>(params);
    int column_tiles = 1 + (columns - 1) / N;
    int row_tiles = 1 + (rows - 1) / M;
    int reduction = sm120_tf32_reduction<Op>(params);
    long long k_tiles = 1 + (reduction - 1) / 32;
    long long tiles = (long long)row_tiles * column_tiles;
    long long units = tiles * k_tiles;
    int grid = (int)gridDim.x;
    int cta = (int)blockIdx.x;
    Sm120StreamKRange mine = sm120_streamk_range(units, grid, cta);
    int warp = (int)threadIdx.x >> 5;
    constexpr int warp_columns = N / 32;
    int warp_m = (warp / warp_columns) * 32;
    int warp_n = (warp % warp_columns) * 32;
    float accumulator[2][4][4];
    long long first_tile = mine.first / k_tiles;

    for (long long unit = mine.last; unit > mine.first;) {
        long long tile = (unit - 1) / k_tiles;
        int k_end = (int)(unit - tile * k_tiles);
        int k_begin = (int)max(0LL, (long long)k_end - (unit - mine.first));
        int output_row = (int)(tile / column_tiles) * M;
        int output_column = (int)(tile % column_tiles) * N;
        const Sm120Tf32StageContext stage_context = {
            &a_map, &b_map, &params, payload, full_base, output_row, output_column};
        sm120_tf32_tn_streamk_segment<M, N, Stages>(
            storage, payload, full_base, empty_base, stage_context,
            k_begin, k_end - k_begin, warp_m, warp_n, accumulator);
        bool covers_start = k_begin == 0;
        bool covers_end = k_end == (int)k_tiles;
        if (covers_start && covers_end) {
            sm120_tf32_tn_streamk_epilogue<M, N>(
                output, rows, columns, output_row, output_column,
                warp_m, warp_n, bias, params, accumulator);
        } else if (!covers_end) {
            // A lower contributor: slot 0 when this is my first tile, slot 1
            // when it is my last tile after an earlier one.
            int slot = tile == first_tile ? 0 : 1;
            float* slab = partial + ((long long)cta * 2 + slot) * slab_floats;
            sm120_streamk_store_slab(slab, accumulator);
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0) {
                sm120_streamk_raise(flags + (long long)cta * 2 + slot);
            }
        } else {
            // The owner: combine every lower contributor in ascending CTA
            // order, then my own accumulators, then the production epilogue.
            float own[2][4][4];
#pragma unroll
            for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
                for (int n_atom = 0; n_atom < 4; ++n_atom) {
#pragma unroll
                    for (int element = 0; element < 4; ++element) {
                        own[m_atom][n_atom][element] = accumulator[m_atom][n_atom][element];
                    }
                }
            }
            int first_cta = sm120_streamk_cta_of(units, grid, tile * k_tiles);
            bool first = true;
            for (int source = first_cta; source < cta; ++source) {
                Sm120StreamKRange theirs = sm120_streamk_range(units, grid, source);
                int slot = tile == theirs.first / k_tiles ? 0 : 1;
                unsigned* flag = flags + (long long)source * 2 + slot;
                if (threadIdx.x == 0) {
                    sm120_streamk_await(flag);
                }
                __syncthreads();
                sm120_streamk_add_slab(
                    partial + ((long long)source * 2 + slot) * slab_floats,
                    accumulator, first);
                first = false;
                __syncthreads();
                if (threadIdx.x == 0) {
                    sm120_streamk_clear(flag);
                }
            }
#pragma unroll
            for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
                for (int n_atom = 0; n_atom < 4; ++n_atom) {
#pragma unroll
                    for (int element = 0; element < 4; ++element) {
                        accumulator[m_atom][n_atom][element] = __fadd_rn(
                            accumulator[m_atom][n_atom][element],
                            own[m_atom][n_atom][element]);
                    }
                }
            }
            sm120_tf32_tn_streamk_epilogue<M, N>(
                output, rows, columns, output_row, output_column,
                warp_m, warp_n, bias, params, accumulator);
        }
        unit -= k_end - k_begin;
    }
}

template <int M, int N, int Stages>
static __device__ __forceinline__ void sm120_tf32_tn_streamk_pair_entry(
    void* output, float* partial, unsigned* flags,
    const CUtensorMap& a_map, const CUtensorMap& b_map,
    const float* bias, const Sm120KernelParams& params) {
    if (sm120_tf32_reduction<Sm120Tn>(params) == 0) {
        int columns = sm120_tf32_columns<Sm120Tn>(params);
        int rows = sm120_tf32_rows<Sm120Tn>(params);
        int column_tiles = 1 + (columns - 1) / N;
        int row_tiles = 1 + (rows - 1) / M;
        for (int tile = (int)blockIdx.x; tile < row_tiles * column_tiles;
             tile += (int)gridDim.x) {
            int output_row = (tile / column_tiles) * M;
            int output_column = (tile % column_tiles) * N;
            for (int linear = (int)threadIdx.x; linear < M * N; linear += (int)blockDim.x) {
                int row = output_row + linear / N;
                int column = output_column + linear % N;
                if (row < rows && column < columns) {
                    sm120_tf32_store<Sm120Tn>(output, row, column, 0.0f, bias, params);
                }
            }
        }
        return;
    }
    sm120_tf32_tn_streamk_pair_kernel<M, N, Stages>(
        output, partial, flags, a_map, b_map, bias, params);
}

#define SM120_DEFINE_TF32_TN_STREAMK_KERNEL(NAME, M, N, STAGES)              \
extern "C" __global__ __launch_bounds__((M * N) / 32) void NAME(            \
    void* output, float* partial, unsigned* flags,                             \
    const __grid_constant__ CUtensorMap a_map,                                 \
    const __grid_constant__ CUtensorMap b_map, const float* bias,              \
    const __grid_constant__ Sm120KernelParams params) {                        \
    sm120_tf32_tn_streamk_pair_entry<M, N, STAGES>(                            \
        output, partial, flags, a_map, b_map, bias, params);                   \
}

SM120_DEFINE_TF32_TN_STREAMK_KERNEL(
    gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s4_pair_exp_streamk_v1, 64, 128, 4)
SM120_DEFINE_TF32_TN_STREAMK_KERNEL(
    gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s3_pair_exp_streamk_v1, 64, 128, 3)
SM120_DEFINE_TF32_TN_STREAMK_KERNEL(
    gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s2_pair_exp_streamk_v1, 64, 128, 2)

#endif
