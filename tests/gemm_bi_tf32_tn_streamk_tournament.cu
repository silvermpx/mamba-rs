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
//
// Every CTA leaves a timing trace behind the last slab of the partial
// buffer: entry, the mainloop bounds of its first and last segment, the end
// of its fixup, exit, and the segment count with the unit count of both
// segments. The harness reads it back to split a launch into prologue,
// mainloop rate, fixup and epilogue; the stamps cost a few instructions.
#if defined(__CUDA_ARCH__) && \
    (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)

static constexpr int SM120_EXP_STREAMK_ACCUMULATORS = 32;

struct Sm120ExpStreamKRange {
    long long first;
    long long last;
};

// Units [first, last) of CTA `cta`: the first `remainder` CTAs take one unit
// more, so every CTA differs from any other by at most one unit.
static __device__ __forceinline__ Sm120ExpStreamKRange sm120_exp_streamk_range(
    long long units, int grid, int cta) {
    long long base = units / grid;
    long long remainder = units % grid;
    long long first = (long long)cta * base + min((long long)cta, remainder);
    long long last = first + base + (cta < remainder ? 1 : 0);
    return {first, last};
}

// The CTA whose range contains `unit`: the inverse of the dealing formula.
static __device__ __forceinline__ int sm120_exp_streamk_cta_of(
    long long units, int grid, long long unit) {
    long long base = units / grid;
    long long remainder = units % grid;
    long long wide = remainder * (base + 1);
    if (unit < wide) return (int)(unit / (base + 1));
    return (int)(remainder + (unit - wide) / base);
}

// The timing trace is compiled in only when the harness asks for it: a
// %globaltimer read costs well over half a microsecond, so a traced kernel
// is not the kernel being timed.
#ifndef SM120_EXP_STREAMK_TRACE
#define SM120_EXP_STREAMK_TRACE 0
#endif

static __device__ __forceinline__ unsigned long long sm120_exp_streamk_now() {
#if SM120_EXP_STREAMK_TRACE
    unsigned long long now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    return now;
#else
    return 0ULL;
#endif
}

static __device__ __forceinline__ void sm120_exp_streamk_store_slab(
    float* slab, const float (&accumulator)[2][4][4]) {
    float4* destination = reinterpret_cast<float4*>(
        slab + (long long)threadIdx.x * SM120_EXP_STREAMK_ACCUMULATORS);
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

static __device__ __forceinline__ void sm120_exp_streamk_add_slab(
    const float* slab, float (&accumulator)[2][4][4], bool first) {
    const float4* source = reinterpret_cast<const float4*>(
        slab + (long long)threadIdx.x * SM120_EXP_STREAMK_ACCUMULATORS);
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

static __device__ __forceinline__ void sm120_exp_streamk_raise(unsigned* flag) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;\n" :: "l"(flag), "r"(1U) : "memory");
}

static __device__ __forceinline__ void sm120_exp_streamk_await(const unsigned* flag) {
    unsigned value;
    do {
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(value) : "l"(flag) : "memory");
    } while (value == 0U);
}

static __device__ __forceinline__ void sm120_exp_streamk_clear(unsigned* flag) {
    asm volatile("st.relaxed.gpu.global.u32 [%0], %1;\n" :: "l"(flag), "r"(0U) : "memory");
}

template <int M, int N, int Stages>
static __device__ __forceinline__ void sm120_tf32_tn_exp_streamk_produce_stage(
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
static __device__ __forceinline__ void sm120_tf32_tn_exp_streamk_segment(
    unsigned char* storage, unsigned payload, unsigned full_base,
    unsigned empty_base, const Sm120Tf32StageContext& stage_context,
    int tile_begin, int tile_count, int warp_m, int warp_n,
    float (&accumulator)[2][4][4], unsigned long long& mainloop_begin,
    unsigned long long& mainloop_end) {
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
                sm120_tf32_tn_exp_streamk_produce_stage<M, N, Stages>(
                    stage_context, tile, tile_begin + tile);
            }
        }
    }
    if constexpr (Stages == 2) sm120_sync_warp();
    mainloop_begin = sm120_exp_streamk_now();
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
                sm120_tf32_tn_exp_streamk_produce_stage<M, N, Stages>(
                    stage_context, refill, tile_begin + refill);
            }
        }
        if constexpr (Stages == 2) sm120_sync_warp();
    }
    // Every warp has consumed its last stage before the storage is reused by
    // the next segment or the CTA leaves.
    __syncthreads();
    mainloop_end = sm120_exp_streamk_now();
}

template <int M, int N>
static __device__ __forceinline__ void sm120_tf32_tn_exp_streamk_epilogue(
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
static __device__ __forceinline__ void sm120_tf32_tn_exp_streamk_drained_kernel(
    void* output, float* partial, unsigned* flags,
    const CUtensorMap& a_map, const CUtensorMap& b_map,
    const float* bias, const Sm120KernelParams& params) {
    constexpr int Op = Sm120Tn;
    constexpr int threads = Sm120Tf32Storage<M, N, Stages>::threads;
    constexpr long long slab_floats = (long long)threads * SM120_EXP_STREAMK_ACCUMULATORS;
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
    Sm120ExpStreamKRange mine = sm120_exp_streamk_range(units, grid, cta);
    int warp = (int)threadIdx.x >> 5;
    constexpr int warp_columns = N / 32;
    int warp_m = (warp / warp_columns) * 32;
    int warp_n = (warp % warp_columns) * 32;
    float accumulator[2][4][4];
    long long first_tile = mine.first / k_tiles;
    unsigned long long t_entry = sm120_exp_streamk_now();
    unsigned long long t_first_begin = 0, t_first_end = 0;
    unsigned long long t_last_begin = 0, t_last_end = 0, t_fixup_end = 0;
    int segments = 0, units_first = 0, units_last = 0;

    for (long long unit = mine.last; unit > mine.first;) {
        long long tile = (unit - 1) / k_tiles;
        int k_end = (int)(unit - tile * k_tiles);
        int k_begin = (int)max(0LL, (long long)k_end - (unit - mine.first));
        int output_row = (int)(tile / column_tiles) * M;
        int output_column = (int)(tile % column_tiles) * N;
        const Sm120Tf32StageContext stage_context = {
            &a_map, &b_map, &params, payload, full_base, output_row, output_column};
        sm120_tf32_tn_exp_streamk_segment<M, N, Stages>(
            storage, payload, full_base, empty_base, stage_context,
            k_begin, k_end - k_begin, warp_m, warp_n, accumulator,
            t_last_begin, t_last_end);
        if (segments == 0) {
            t_first_begin = t_last_begin;
            t_first_end = t_last_end;
            units_first = k_end - k_begin;
        }
        units_last = k_end - k_begin;
        ++segments;
        bool covers_start = k_begin == 0;
        bool covers_end = k_end == (int)k_tiles;
        if (covers_start && covers_end) {
            sm120_tf32_tn_exp_streamk_epilogue<M, N>(
                output, rows, columns, output_row, output_column,
                warp_m, warp_n, bias, params, accumulator);
        } else if (!covers_end) {
            // A lower contributor: slot 0 when this is my first tile, slot 1
            // when it is my last tile after an earlier one.
            int slot = tile == first_tile ? 0 : 1;
            float* slab = partial + ((long long)cta * 2 + slot) * slab_floats;
            sm120_exp_streamk_store_slab(slab, accumulator);
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0) {
                sm120_exp_streamk_raise(flags + (long long)cta * 2 + slot);
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
            int first_cta = sm120_exp_streamk_cta_of(units, grid, tile * k_tiles);
            bool first = true;
            for (int source = first_cta; source < cta; ++source) {
                Sm120ExpStreamKRange theirs = sm120_exp_streamk_range(units, grid, source);
                int slot = tile == theirs.first / k_tiles ? 0 : 1;
                unsigned* flag = flags + (long long)source * 2 + slot;
                if (threadIdx.x == 0) {
                    sm120_exp_streamk_await(flag);
                }
                __syncthreads();
                sm120_exp_streamk_add_slab(
                    partial + ((long long)source * 2 + slot) * slab_floats,
                    accumulator, first);
                first = false;
                __syncthreads();
                if (threadIdx.x == 0) {
                    sm120_exp_streamk_clear(flag);
                }
            }
            t_fixup_end = sm120_exp_streamk_now();
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
            sm120_tf32_tn_exp_streamk_epilogue<M, N>(
                output, rows, columns, output_row, output_column,
                warp_m, warp_n, bias, params, accumulator);
        }
        unit -= k_end - k_begin;
    }
    if (SM120_EXP_STREAMK_TRACE && threadIdx.x == 0) {
        unsigned long long* trace = reinterpret_cast<unsigned long long*>(
            partial + (long long)grid * 2 * slab_floats) + (long long)cta * 8;
        trace[0] = t_entry;
        trace[1] = t_first_begin;
        trace[2] = t_first_end;
        trace[3] = t_last_begin;
        trace[4] = t_last_end;
        trace[5] = t_fixup_end;
        trace[6] = sm120_exp_streamk_now();
        trace[7] = ((unsigned long long)segments << 48)
            | ((unsigned long long)units_first << 24)
            | (unsigned long long)units_last;
    }
}

// The persistent grid walks every tile of an empty reduction.
template <int M, int N>
static __device__ __forceinline__ void sm120_tf32_tn_exp_streamk_zero_reduction(
    void* output, const float* bias, const Sm120KernelParams& params) {
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
}

template <int M, int N, int Stages>
static __device__ __forceinline__ void sm120_tf32_tn_exp_streamk_drained_entry(
    void* output, float* partial, unsigned* flags,
    const CUtensorMap& a_map, const CUtensorMap& b_map,
    const float* bias, const Sm120KernelParams& params) {
    if (sm120_tf32_reduction<Sm120Tn>(params) == 0) {
        sm120_tf32_tn_exp_streamk_zero_reduction<M, N>(output, bias, params);
        return;
    }
    sm120_tf32_tn_exp_streamk_drained_kernel<M, N, Stages>(
        output, partial, flags, a_map, b_map, bias, params);
}

// A cursor over one CTA's stage sequence: the segments of its range from the
// end, each ascending in k. `unit` is the absolute unit of the current stage.
struct Sm120ExpStreamKCursor {
    int unit;
    int segment_begin;
    int segment_end;
    int tile;
};

static __device__ __forceinline__ void sm120_exp_streamk_cursor_open(
    Sm120ExpStreamKCursor& cursor, int range_first, int range_end, int k_tiles) {
    cursor.tile = (range_end - 1) / k_tiles;
    cursor.segment_begin = max(cursor.tile * k_tiles, range_first);
    cursor.segment_end = range_end;
    cursor.unit = cursor.segment_begin;
}

// Steps to the next stage of the sequence; false once the range is spent.
static __device__ __forceinline__ bool sm120_exp_streamk_cursor_advance(
    Sm120ExpStreamKCursor& cursor, int range_first, int k_tiles) {
    if (cursor.unit + 1 < cursor.segment_end) {
        ++cursor.unit;
        return true;
    }
    if (cursor.segment_begin == range_first) {
        return false;
    }
    sm120_exp_streamk_cursor_open(cursor, range_first, cursor.segment_begin, k_tiles);
    return true;
}

static __device__ __forceinline__ void sm120_exp_streamk_zero(
    float (&accumulator)[2][4][4]) {
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
}

// Folds the lower contributors' slabs into the accumulators: the slabs are
// summed in ascending CTA order, then the CTA's own value is added last, the
// same operation order as the first design. Every slab is read through
// ordinary cache-global loads so the compiler issues the eight loads of one
// slab together instead of one behind the other.
static __device__ __forceinline__ void sm120_exp_streamk_fold_slabs(
    const float* partial, long long slab_floats, long long units, int grid,
    int first_cta, int cta, int tile, int k_tiles,
    float (&accumulator)[2][4][4]) {
    const long long lane_offset = (long long)threadIdx.x * SM120_EXP_STREAMK_ACCUMULATORS;
    float sum[2][4][4];
    bool first = true;
    for (int source = first_cta; source < cta; ++source) {
        Sm120ExpStreamKRange theirs = sm120_exp_streamk_range(units, grid, source);
        int slot = tile == (int)(theirs.first / k_tiles) ? 0 : 1;
        const float4* slab = reinterpret_cast<const float4*>(
            partial + ((long long)source * 2 + slot) * slab_floats + lane_offset);
        float4 value[2][4];
#pragma unroll
        for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
            for (int n_atom = 0; n_atom < 4; ++n_atom) {
                value[m_atom][n_atom] = __ldcg(slab + m_atom * 4 + n_atom);
            }
        }
#pragma unroll
        for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
            for (int n_atom = 0; n_atom < 4; ++n_atom) {
                float4 v = value[m_atom][n_atom];
                if (first) {
                    sum[m_atom][n_atom][0] = v.x;
                    sum[m_atom][n_atom][1] = v.y;
                    sum[m_atom][n_atom][2] = v.z;
                    sum[m_atom][n_atom][3] = v.w;
                } else {
                    sum[m_atom][n_atom][0] = __fadd_rn(sum[m_atom][n_atom][0], v.x);
                    sum[m_atom][n_atom][1] = __fadd_rn(sum[m_atom][n_atom][1], v.y);
                    sum[m_atom][n_atom][2] = __fadd_rn(sum[m_atom][n_atom][2], v.z);
                    sum[m_atom][n_atom][3] = __fadd_rn(sum[m_atom][n_atom][3], v.w);
                }
            }
        }
        first = false;
    }
#pragma unroll
    for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 4; ++n_atom) {
#pragma unroll
            for (int element = 0; element < 4; ++element) {
                accumulator[m_atom][n_atom][element] = __fadd_rn(
                    sum[m_atom][n_atom][element], accumulator[m_atom][n_atom][element]);
            }
        }
    }
}

// The register-lean fold for a kernel that keeps two CTAs resident: one
// fragment atom at a time, the source loop inside, four temporaries.
static __device__ __forceinline__ void sm120_exp_streamk_fold_slabs_lean(
    const float* partial, long long slab_floats, long long units, int grid,
    int first_cta, int cta, int tile, int k_tiles,
    float (&accumulator)[2][4][4]) {
    const long long lane_offset = (long long)threadIdx.x * SM120_EXP_STREAMK_ACCUMULATORS;
#pragma unroll
    for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 4; ++n_atom) {
            float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            bool first = true;
            for (int source = first_cta; source < cta; ++source) {
                Sm120ExpStreamKRange theirs = sm120_exp_streamk_range(units, grid, source);
                int slot = tile == (int)(theirs.first / k_tiles) ? 0 : 1;
                const float4* slab = reinterpret_cast<const float4*>(
                    partial + ((long long)source * 2 + slot) * slab_floats + lane_offset);
                float4 value = __ldcg(slab + m_atom * 4 + n_atom);
                if (first) {
                    sum = value;
                } else {
                    sum.x = __fadd_rn(sum.x, value.x);
                    sum.y = __fadd_rn(sum.y, value.y);
                    sum.z = __fadd_rn(sum.z, value.z);
                    sum.w = __fadd_rn(sum.w, value.w);
                }
                first = false;
            }
            accumulator[m_atom][n_atom][0] = __fadd_rn(sum.x, accumulator[m_atom][n_atom][0]);
            accumulator[m_atom][n_atom][1] = __fadd_rn(sum.y, accumulator[m_atom][n_atom][1]);
            accumulator[m_atom][n_atom][2] = __fadd_rn(sum.z, accumulator[m_atom][n_atom][2]);
            accumulator[m_atom][n_atom][3] = __fadd_rn(sum.w, accumulator[m_atom][n_atom][3]);
        }
    }
}

// The pair epilogue with every read of the old tile issued before the first
// write: a full tile reads its sixteen pairs in one burst rather than one
// read-modify-write at a time.
template <int M, int N>
static __device__ __forceinline__ void sm120_tf32_tn_exp_streamk_epilogue_burst(
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
    if (!full_tile) {
        sm120_tf32_tn_exp_streamk_epilogue<M, N>(
            output, rows, columns, output_row, output_column,
            warp_m, warp_n, bias, params, accumulator);
        return;
    }
    float* base = static_cast<float*>(output);
    float2 old[2][4][2];
#pragma unroll
    for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 4; ++n_atom) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                int row = output_row + warp_m + m_atom * 16 + group + half * 8;
                int column = output_column + warp_n + n_atom * 8 + 2 * thread;
                old[m_atom][n_atom][half] = *reinterpret_cast<const float2*>(
                    base + static_cast<long long>(row) * params.ldc + column);
            }
        }
    }
#pragma unroll
    for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 4; ++n_atom) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                int row = output_row + warp_m + m_atom * 16 + group + half * 8;
                int column = output_column + warp_n + n_atom * 8 + 2 * thread;
                float2 pair = old[m_atom][n_atom][half];
                float first = sm120_tf32_epilogue<Sm120Tn>(
                    accumulator[m_atom][n_atom][half * 2], pair.x, bias, column, params);
                float second = sm120_tf32_epilogue<Sm120Tn>(
                    accumulator[m_atom][n_atom][half * 2 + 1], pair.y, bias, column + 1, params);
                *reinterpret_cast<float2*>(
                    base + static_cast<long long>(row) * params.ldc + column) =
                    make_float2(first, second);
            }
        }
    }
}

// Launch-constant state of the continuous pipeline.
struct Sm120ExpStreamKFlow {
    const CUtensorMap* a_map;
    const CUtensorMap* b_map;
    const Sm120KernelParams* params;
    unsigned payload;
    unsigned full_base;
    unsigned empty_base;
    int column_tiles;
    int k_tiles;
    int range_first;
    int total;
};

template <int M, int N, int Stages>
static __device__ __forceinline__ void sm120_exp_streamk_flow_produce(
    const Sm120ExpStreamKFlow& flow, const Sm120ExpStreamKCursor& cursor, int step) {
    const Sm120Tf32StageContext context = {
        flow.a_map, flow.b_map, flow.params, flow.payload, flow.full_base,
        (cursor.tile / flow.column_tiles) * M, (cursor.tile % flow.column_tiles) * N};
    sm120_tf32_tn_exp_streamk_produce_stage<M, N, Stages>(
        context, step, cursor.unit - cursor.tile * flow.k_tiles);
}

// The production mainloop over one segment, on a pipeline that keeps
// running: steps are numbered across the whole range, so the stage and the
// barrier phase continue from the previous segment, and the producer lane
// refills with whatever the cursor holds next, which may already belong to
// the segment after this one.
template <int M, int N, int Stages>
static __device__ __forceinline__ void sm120_tf32_tn_exp_streamk_flow_segment(
    unsigned char* storage, const Sm120ExpStreamKFlow& flow,
    Sm120ExpStreamKCursor& producer, int step_base, int tile_count,
    int warp_m, int warp_n, float (&accumulator)[2][4][4],
    unsigned long long* trace, bool first_segment) {
    int warp = (int)threadIdx.x >> 5;
    int lane = (int)threadIdx.x & 31;
    sm120_exp_streamk_zero(accumulator);
    if (SM120_EXP_STREAMK_TRACE && threadIdx.x == 0) {
        unsigned long long now = sm120_exp_streamk_now();
        if (first_segment) trace[1] = now;
        trace[3] = now;
    }
    for (int tile = 0; tile < tile_count; ++tile) {
        int step = step_base + tile;
        int stage = step % Stages;
        unsigned generation = (unsigned)(step / Stages);
        sm120_wait_barrier(flow.full_base + stage * 8, generation & 1U);
        sm120_tf32_issue_stage<Sm120Tn, M, N, Stages>(
            storage, stage, warp_m, warp_n, accumulator);
        if (lane == 0) {
            sm120_arrive_empty(flow.empty_base + stage * 8);
        }
        if (warp == 0 && lane == 0) {
            int refill = step + Stages;
            if (refill < flow.total) {
                sm120_wait_barrier(flow.empty_base + stage * 8, generation & 1U);
                sm120_exp_streamk_flow_produce<M, N, Stages>(flow, producer, refill);
                sm120_exp_streamk_cursor_advance(producer, flow.range_first, flow.k_tiles);
            }
        }
        if constexpr (Stages == 2) sm120_sync_warp();
    }
    if (SM120_EXP_STREAMK_TRACE && threadIdx.x == 0) {
        unsigned long long now = sm120_exp_streamk_now();
        if (first_segment) trace[2] = now;
        trace[4] = now;
    }
}

// The second stream-K design: the same segment walk as the first, on one
// pipeline for the whole range. The barriers are armed once per launch and
// the producer lane runs `Stages` steps ahead across segment boundaries, so
// the loads of the next segment are in flight while the CTA publishes a slab
// or runs an epilogue. The owner polls every contributor flag in parallel
// and releases them together, and the publisher fences once from one thread
// after the block barrier instead of once per thread.
// `Wide` selects the register-rich boundary code (batched slab fold and the
// burst epilogue); a kernel that must keep two CTAs resident cannot afford
// those registers and takes the lean fold and the pair-by-pair store.
template <int M, int N, int Stages, bool Wide>
static __device__ __forceinline__ void sm120_tf32_tn_exp_streamk_flowing_kernel(
    void* output, float* partial, unsigned* flags,
    const CUtensorMap& a_map, const CUtensorMap& b_map,
    const float* bias, const Sm120KernelParams& params) {
    constexpr int Op = Sm120Tn;
    constexpr int threads = Sm120Tf32Storage<M, N, Stages>::threads;
    constexpr int warps = threads / 32;
    constexpr long long slab_floats = (long long)threads * SM120_EXP_STREAMK_ACCUMULATORS;
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
    int k_tiles = 1 + (reduction - 1) / 32;
    long long units = (long long)row_tiles * column_tiles * k_tiles;
    int grid = (int)gridDim.x;
    int cta = (int)blockIdx.x;
    Sm120ExpStreamKRange mine = sm120_exp_streamk_range(units, grid, cta);
    int range_first = (int)mine.first;
    int range_end = (int)mine.last;
    int first_tile = range_first / k_tiles;
    const Sm120ExpStreamKFlow flow = {
        &a_map, &b_map, &params, payload, full_base, empty_base,
        column_tiles, k_tiles, range_first, range_end - range_first};
    int warp = (int)threadIdx.x >> 5;
    int lane = (int)threadIdx.x & 31;
    constexpr int warp_columns = N / 32;
    int warp_m = (warp / warp_columns) * 32;
    int warp_n = (warp % warp_columns) * 32;
    float accumulator[2][4][4];
    unsigned long long* trace = reinterpret_cast<unsigned long long*>(
        partial + (long long)grid * 2 * slab_floats) + (long long)cta * 8;
    int segments = 0, units_first = 0, units_last = 0;

    if (SM120_EXP_STREAMK_TRACE && threadIdx.x == 0) {
        trace[0] = sm120_exp_streamk_now();
        trace[5] = 0;
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

    Sm120ExpStreamKCursor producer;
    sm120_exp_streamk_cursor_open(producer, range_first, range_end, k_tiles);
    if (warp == 0 && lane == 0) {
#pragma unroll
        for (int step = 0; step < Stages; ++step) {
            if (step < flow.total) {
                sm120_exp_streamk_flow_produce<M, N, Stages>(flow, producer, step);
                sm120_exp_streamk_cursor_advance(producer, range_first, k_tiles);
            }
        }
    }
    if constexpr (Stages == 2) sm120_sync_warp();

    int step_base = 0;
    for (int unit = range_end; unit > range_first;) {
        int tile = (unit - 1) / k_tiles;
        int k_end = unit - tile * k_tiles;
        int k_begin = max(0, k_end - (unit - range_first));
        int output_row = (tile / column_tiles) * M;
        int output_column = (tile % column_tiles) * N;
        sm120_tf32_tn_exp_streamk_flow_segment<M, N, Stages>(
            storage, flow, producer, step_base, k_end - k_begin,
            warp_m, warp_n, accumulator, trace, segments == 0);
        step_base += k_end - k_begin;
        if (segments == 0) {
            units_first = k_end - k_begin;
        }
        units_last = k_end - k_begin;
        ++segments;
        bool covers_start = k_begin == 0;
        bool covers_end = k_end == k_tiles;
        if (covers_start && covers_end) {
            if constexpr (Wide) {
                sm120_tf32_tn_exp_streamk_epilogue_burst<M, N>(
                    output, rows, columns, output_row, output_column,
                    warp_m, warp_n, bias, params, accumulator);
            } else {
                sm120_tf32_tn_exp_streamk_epilogue<M, N>(
                    output, rows, columns, output_row, output_column,
                    warp_m, warp_n, bias, params, accumulator);
            }
        } else if (!covers_end) {
            int slot = tile == first_tile ? 0 : 1;
            sm120_exp_streamk_store_slab(
                partial + ((long long)cta * 2 + slot) * slab_floats, accumulator);
            __syncthreads();
            if (threadIdx.x == 0) {
                __threadfence();
                sm120_exp_streamk_raise(flags + (long long)cta * 2 + slot);
            }
        } else {
            int first_cta = sm120_exp_streamk_cta_of(units, grid, (long long)tile * k_tiles);
            int sources = cta - first_cta;
            for (int index = (int)threadIdx.x; index < sources; index += threads) {
                int source = first_cta + index;
                Sm120ExpStreamKRange theirs = sm120_exp_streamk_range(units, grid, source);
                int slot = tile == (int)(theirs.first / k_tiles) ? 0 : 1;
                sm120_exp_streamk_await(flags + (long long)source * 2 + slot);
            }
            __syncthreads();
            if constexpr (Wide) {
                sm120_exp_streamk_fold_slabs(
                    partial, slab_floats, units, grid, first_cta, cta, tile, k_tiles,
                    accumulator);
            } else {
                sm120_exp_streamk_fold_slabs_lean(
                    partial, slab_floats, units, grid, first_cta, cta, tile, k_tiles,
                    accumulator);
            }
            __syncthreads();
            for (int index = (int)threadIdx.x; index < sources; index += threads) {
                int source = first_cta + index;
                Sm120ExpStreamKRange theirs = sm120_exp_streamk_range(units, grid, source);
                int slot = tile == (int)(theirs.first / k_tiles) ? 0 : 1;
                sm120_exp_streamk_clear(flags + (long long)source * 2 + slot);
            }
            if (SM120_EXP_STREAMK_TRACE && threadIdx.x == 0) trace[5] = sm120_exp_streamk_now();
            if constexpr (Wide) {
                sm120_tf32_tn_exp_streamk_epilogue_burst<M, N>(
                    output, rows, columns, output_row, output_column,
                    warp_m, warp_n, bias, params, accumulator);
            } else {
                sm120_tf32_tn_exp_streamk_epilogue<M, N>(
                    output, rows, columns, output_row, output_column,
                    warp_m, warp_n, bias, params, accumulator);
            }
        }
        unit -= k_end - k_begin;
    }
    if (SM120_EXP_STREAMK_TRACE && threadIdx.x == 0) {
        trace[6] = sm120_exp_streamk_now();
        trace[7] = ((unsigned long long)segments << 48)
            | ((unsigned long long)units_first << 24)
            | (unsigned long long)units_last;
    }
}

template <int M, int N, int Stages, bool Wide>
static __device__ __forceinline__ void sm120_tf32_tn_exp_streamk_flowing_entry(
    void* output, float* partial, unsigned* flags,
    const CUtensorMap& a_map, const CUtensorMap& b_map,
    const float* bias, const Sm120KernelParams& params) {
    if (sm120_tf32_reduction<Sm120Tn>(params) == 0) {
        sm120_tf32_tn_exp_streamk_zero_reduction<M, N>(output, bias, params);
        return;
    }
    sm120_tf32_tn_exp_streamk_flowing_kernel<M, N, Stages, Wide>(
        output, partial, flags, a_map, b_map, bias, params);
}

#define SM120_DEFINE_TF32_TN_EXP_STREAMK_DRAINED_KERNEL(NAME, M, N, STAGES)              \
extern "C" __global__ __launch_bounds__((M * N) / 32) void NAME(            \
    void* output, float* partial, unsigned* flags,                             \
    const __grid_constant__ CUtensorMap a_map,                                 \
    const __grid_constant__ CUtensorMap b_map, const float* bias,              \
    const __grid_constant__ Sm120KernelParams params) {                        \
    sm120_tf32_tn_exp_streamk_drained_entry<M, N, STAGES>(                            \
        output, partial, flags, a_map, b_map, bias, params);                   \
}

SM120_DEFINE_TF32_TN_EXP_STREAMK_DRAINED_KERNEL(
    gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s4_pair_exp_streamk_drained, 64, 128, 4)
SM120_DEFINE_TF32_TN_EXP_STREAMK_DRAINED_KERNEL(
    gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s3_pair_exp_streamk_drained, 64, 128, 3)
SM120_DEFINE_TF32_TN_EXP_STREAMK_DRAINED_KERNEL(
    gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s2_pair_exp_streamk_drained, 64, 128, 2)


// A 128x128 tile on eight warps, each owning 32x64 of it: two m-atoms and
// eight n-atoms per warp. The A fragments, whose TN pattern walks down the
// swizzled planes, are loaded half as often per mma as in the 64x128 tile,
// and every k-tile moves a third less operand data per flop through L2.
static constexpr int SM120_EXP_WIDE_ACCUMULATORS = 64;
static constexpr int SM120_EXP_WIDE_THREADS = 256;

template <int Op, int M, int N, int Stages>
static __device__ __forceinline__ void sm120_exp_wide_load_issue(
    unsigned char* storage, int stage, int warp_m, int warp_n, int k8,
    unsigned (&a_fragments)[2][4], unsigned (&b_fragments)[8][2]) {
    int lane = (int)threadIdx.x & 31;
    int group = lane >> 2;
    int thread = lane & 3;
#pragma unroll
    for (int m_atom = 0; m_atom < 2; ++m_atom) {
        int row = warp_m + m_atom * 16 + group;
        a_fragments[m_atom][0] = sm120_tf32_rna(
            sm120_tf32_load_a<Op, M, N, Stages>(storage, stage, row, k8 + thread));
        a_fragments[m_atom][1] = sm120_tf32_rna(
            sm120_tf32_load_a<Op, M, N, Stages>(storage, stage, row + 8, k8 + thread));
        a_fragments[m_atom][2] = sm120_tf32_rna(
            sm120_tf32_load_a<Op, M, N, Stages>(storage, stage, row, k8 + thread + 4));
        a_fragments[m_atom][3] = sm120_tf32_rna(
            sm120_tf32_load_a<Op, M, N, Stages>(storage, stage, row + 8, k8 + thread + 4));
    }
#pragma unroll
    for (int n_atom = 0; n_atom < 8; ++n_atom) {
        int column = warp_n + n_atom * 8 + group;
        b_fragments[n_atom][0] = sm120_tf32_rna(
            sm120_tf32_load_b<Op, M, N, Stages>(storage, stage, k8 + thread, column));
        b_fragments[n_atom][1] = sm120_tf32_rna(
            sm120_tf32_load_b<Op, M, N, Stages>(storage, stage, k8 + thread + 4, column));
    }
}

template <int Op, int M, int N, int Stages>
static __device__ __forceinline__ void sm120_exp_wide_issue_stage(
    unsigned char* storage, int stage, int warp_m, int warp_n,
    float (&accumulator)[2][8][4]) {
    unsigned a_fragments[2][2][4];
    unsigned b_fragments[2][8][2];
    sm120_exp_wide_load_issue<Op, M, N, Stages>(
        storage, stage, warp_m, warp_n, 0, a_fragments[0], b_fragments[0]);
#pragma unroll
    for (int issue = 0; issue < 4; ++issue) {
        int k8 = issue * 8;
        int current = issue & 1;
        int next = current ^ 1;
        if (issue + 1 < 4) {
            sm120_exp_wide_load_issue<Op, M, N, Stages>(storage, stage,
                warp_m, warp_n, k8 + 8, a_fragments[next], b_fragments[next]);
        }
#pragma unroll
        for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
            for (int n_atom = 0; n_atom < 8; ++n_atom) {
                sm120_tf32_mma_m16n8k8(
                    accumulator[m_atom][n_atom],
                    a_fragments[current][m_atom], b_fragments[current][n_atom]);
            }
        }
    }
}

static __device__ __forceinline__ void sm120_exp_wide_zero(float (&accumulator)[2][8][4]) {
#pragma unroll
    for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 8; ++n_atom) {
#pragma unroll
            for (int element = 0; element < 4; ++element) {
                accumulator[m_atom][n_atom][element] = 0.0f;
            }
        }
    }
}

template <int M, int N, int Stages>
static __device__ __forceinline__ void sm120_exp_wide_segment(
    unsigned char* storage, const Sm120ExpStreamKFlow& flow,
    Sm120ExpStreamKCursor& producer, int step_base, int tile_count,
    int warp_m, int warp_n, float (&accumulator)[2][8][4],
    unsigned long long* trace, bool first_segment) {
    int warp = (int)threadIdx.x >> 5;
    int lane = (int)threadIdx.x & 31;
    sm120_exp_wide_zero(accumulator);
    if (SM120_EXP_STREAMK_TRACE && threadIdx.x == 0) {
        unsigned long long now = sm120_exp_streamk_now();
        if (first_segment) trace[1] = now;
        trace[3] = now;
    }
    for (int tile = 0; tile < tile_count; ++tile) {
        int step = step_base + tile;
        int stage = step % Stages;
        unsigned generation = (unsigned)(step / Stages);
        sm120_wait_barrier(flow.full_base + stage * 8, generation & 1U);
        sm120_exp_wide_issue_stage<Sm120Tn, M, N, Stages>(
            storage, stage, warp_m, warp_n, accumulator);
        if (lane == 0) {
            sm120_arrive_empty(flow.empty_base + stage * 8);
        }
        if (warp == 0 && lane == 0) {
            int refill = step + Stages;
            if (refill < flow.total) {
                sm120_wait_barrier(flow.empty_base + stage * 8, generation & 1U);
                sm120_exp_streamk_flow_produce<M, N, Stages>(flow, producer, refill);
                sm120_exp_streamk_cursor_advance(producer, flow.range_first, flow.k_tiles);
            }
        }
        if constexpr (Stages == 2) sm120_sync_warp();
    }
    if (SM120_EXP_STREAMK_TRACE && threadIdx.x == 0) {
        unsigned long long now = sm120_exp_streamk_now();
        if (first_segment) trace[2] = now;
        trace[4] = now;
    }
}

static __device__ __forceinline__ void sm120_exp_wide_store_slab(
    float* slab, const float (&accumulator)[2][8][4]) {
    float4* destination = reinterpret_cast<float4*>(
        slab + (long long)threadIdx.x * SM120_EXP_WIDE_ACCUMULATORS);
#pragma unroll
    for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 8; ++n_atom) {
            float4 value = make_float4(
                accumulator[m_atom][n_atom][0], accumulator[m_atom][n_atom][1],
                accumulator[m_atom][n_atom][2], accumulator[m_atom][n_atom][3]);
            asm volatile("st.global.cg.v4.f32 [%0], {%1, %2, %3, %4};\n"
                :: "l"(destination + m_atom * 8 + n_atom),
                   "f"(value.x), "f"(value.y), "f"(value.z), "f"(value.w) : "memory");
        }
    }
}

static __device__ __forceinline__ void sm120_exp_wide_fold_slabs(
    const float* partial, long long slab_floats, long long units, int grid,
    int first_cta, int cta, int tile, int k_tiles,
    float (&accumulator)[2][8][4]) {
    const long long lane_offset = (long long)threadIdx.x * SM120_EXP_WIDE_ACCUMULATORS;
    float sum[2][8][4];
    bool first = true;
    for (int source = first_cta; source < cta; ++source) {
        Sm120ExpStreamKRange theirs = sm120_exp_streamk_range(units, grid, source);
        int slot = tile == (int)(theirs.first / k_tiles) ? 0 : 1;
        const float4* slab = reinterpret_cast<const float4*>(
            partial + ((long long)source * 2 + slot) * slab_floats + lane_offset);
        float4 value[2][8];
#pragma unroll
        for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
            for (int n_atom = 0; n_atom < 8; ++n_atom) {
                value[m_atom][n_atom] = __ldcg(slab + m_atom * 8 + n_atom);
            }
        }
#pragma unroll
        for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
            for (int n_atom = 0; n_atom < 8; ++n_atom) {
                float4 v = value[m_atom][n_atom];
                if (first) {
                    sum[m_atom][n_atom][0] = v.x;
                    sum[m_atom][n_atom][1] = v.y;
                    sum[m_atom][n_atom][2] = v.z;
                    sum[m_atom][n_atom][3] = v.w;
                } else {
                    sum[m_atom][n_atom][0] = __fadd_rn(sum[m_atom][n_atom][0], v.x);
                    sum[m_atom][n_atom][1] = __fadd_rn(sum[m_atom][n_atom][1], v.y);
                    sum[m_atom][n_atom][2] = __fadd_rn(sum[m_atom][n_atom][2], v.z);
                    sum[m_atom][n_atom][3] = __fadd_rn(sum[m_atom][n_atom][3], v.w);
                }
            }
        }
        first = false;
    }
#pragma unroll
    for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 8; ++n_atom) {
#pragma unroll
            for (int element = 0; element < 4; ++element) {
                accumulator[m_atom][n_atom][element] = __fadd_rn(
                    sum[m_atom][n_atom][element], accumulator[m_atom][n_atom][element]);
            }
        }
    }
}

template <int M, int N>
static __device__ __forceinline__ void sm120_exp_wide_epilogue(
    void* output, int rows, int columns, int output_row, int output_column,
    int warp_m, int warp_n, const float* bias, const Sm120KernelParams& params,
    const float (&accumulator)[2][8][4]) {
    int lane = (int)threadIdx.x & 31;
    int group = lane >> 2;
    int thread = lane & 3;
    bool full_tile = output_row + M <= rows
        && output_column + N <= columns
        && (reinterpret_cast<unsigned long long>(output) & 7ULL) == 0ULL
        && (params.ldc & 1) == 0;
    if (!full_tile) {
#pragma unroll
        for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
            for (int n_atom = 0; n_atom < 8; ++n_atom) {
#pragma unroll
                for (int element = 0; element < 4; element += 2) {
                    int row = output_row + warp_m + m_atom * 16
                        + group + (element >= 2 ? 8 : 0);
                    int column = output_column + warp_n + n_atom * 8 + 2 * thread;
                    Sm120Tf32PairValue pair = {
                        accumulator[m_atom][n_atom][element],
                        accumulator[m_atom][n_atom][element + 1]};
                    sm120_tf32_store_pair<Sm120Tn>(
                        output, row, column, pair, bias, params, false);
                }
            }
        }
        return;
    }
    float* base = static_cast<float*>(output);
    float2 old[2][8][2];
#pragma unroll
    for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 8; ++n_atom) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                int row = output_row + warp_m + m_atom * 16 + group + half * 8;
                int column = output_column + warp_n + n_atom * 8 + 2 * thread;
                old[m_atom][n_atom][half] = *reinterpret_cast<const float2*>(
                    base + static_cast<long long>(row) * params.ldc + column);
            }
        }
    }
#pragma unroll
    for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 8; ++n_atom) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                int row = output_row + warp_m + m_atom * 16 + group + half * 8;
                int column = output_column + warp_n + n_atom * 8 + 2 * thread;
                float2 pair = old[m_atom][n_atom][half];
                float first = sm120_tf32_epilogue<Sm120Tn>(
                    accumulator[m_atom][n_atom][half * 2], pair.x, bias, column, params);
                float second = sm120_tf32_epilogue<Sm120Tn>(
                    accumulator[m_atom][n_atom][half * 2 + 1], pair.y, bias, column + 1, params);
                *reinterpret_cast<float2*>(
                    base + static_cast<long long>(row) * params.ldc + column) =
                    make_float2(first, second);
            }
        }
    }
}

template <int M, int N, int Stages>
static __device__ __forceinline__ void sm120_tf32_tn_exp_streamk_wide_kernel(
    void* output, float* partial, unsigned* flags,
    const CUtensorMap& a_map, const CUtensorMap& b_map,
    const float* bias, const Sm120KernelParams& params) {
    constexpr int Op = Sm120Tn;
    constexpr int threads = SM120_EXP_WIDE_THREADS;
    constexpr int warps = threads / 32;
    constexpr long long slab_floats = (long long)threads * SM120_EXP_WIDE_ACCUMULATORS;
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
    int k_tiles = 1 + (reduction - 1) / 32;
    long long units = (long long)row_tiles * column_tiles * k_tiles;
    int grid = (int)gridDim.x;
    int cta = (int)blockIdx.x;
    Sm120ExpStreamKRange mine = sm120_exp_streamk_range(units, grid, cta);
    int range_first = (int)mine.first;
    int range_end = (int)mine.last;
    int first_tile = range_first / k_tiles;
    const Sm120ExpStreamKFlow flow = {
        &a_map, &b_map, &params, payload, full_base, empty_base,
        column_tiles, k_tiles, range_first, range_end - range_first};
    int warp = (int)threadIdx.x >> 5;
    int lane = (int)threadIdx.x & 31;
    int warp_m = (warp / 2) * 32;
    int warp_n = (warp % 2) * 64;
    float accumulator[2][8][4];
    unsigned long long* trace = reinterpret_cast<unsigned long long*>(
        partial + (long long)grid * 2 * slab_floats) + (long long)cta * 8;
    int segments = 0, units_first = 0, units_last = 0;

    if (SM120_EXP_STREAMK_TRACE && threadIdx.x == 0) {
        trace[0] = sm120_exp_streamk_now();
        trace[5] = 0;
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

    Sm120ExpStreamKCursor producer;
    sm120_exp_streamk_cursor_open(producer, range_first, range_end, k_tiles);
    if (warp == 0 && lane == 0) {
#pragma unroll
        for (int step = 0; step < Stages; ++step) {
            if (step < flow.total) {
                sm120_exp_streamk_flow_produce<M, N, Stages>(flow, producer, step);
                sm120_exp_streamk_cursor_advance(producer, range_first, k_tiles);
            }
        }
    }
    if constexpr (Stages == 2) sm120_sync_warp();

    int step_base = 0;
    for (int unit = range_end; unit > range_first;) {
        int tile = (unit - 1) / k_tiles;
        int k_end = unit - tile * k_tiles;
        int k_begin = max(0, k_end - (unit - range_first));
        int output_row = (tile / column_tiles) * M;
        int output_column = (tile % column_tiles) * N;
        sm120_exp_wide_segment<M, N, Stages>(
            storage, flow, producer, step_base, k_end - k_begin,
            warp_m, warp_n, accumulator, trace, segments == 0);
        step_base += k_end - k_begin;
        if (segments == 0) {
            units_first = k_end - k_begin;
        }
        units_last = k_end - k_begin;
        ++segments;
        bool covers_start = k_begin == 0;
        bool covers_end = k_end == k_tiles;
        if (covers_start && covers_end) {
            sm120_exp_wide_epilogue<M, N>(
                output, rows, columns, output_row, output_column,
                warp_m, warp_n, bias, params, accumulator);
        } else if (!covers_end) {
            int slot = tile == first_tile ? 0 : 1;
            sm120_exp_wide_store_slab(
                partial + ((long long)cta * 2 + slot) * slab_floats, accumulator);
            __syncthreads();
            if (threadIdx.x == 0) {
                __threadfence();
                sm120_exp_streamk_raise(flags + (long long)cta * 2 + slot);
            }
        } else {
            int first_cta = sm120_exp_streamk_cta_of(units, grid, (long long)tile * k_tiles);
            int sources = cta - first_cta;
            for (int index = (int)threadIdx.x; index < sources; index += threads) {
                int source = first_cta + index;
                Sm120ExpStreamKRange theirs = sm120_exp_streamk_range(units, grid, source);
                int slot = tile == (int)(theirs.first / k_tiles) ? 0 : 1;
                sm120_exp_streamk_await(flags + (long long)source * 2 + slot);
            }
            __syncthreads();
            sm120_exp_wide_fold_slabs(
                partial, slab_floats, units, grid, first_cta, cta, tile, k_tiles,
                accumulator);
            __syncthreads();
            for (int index = (int)threadIdx.x; index < sources; index += threads) {
                int source = first_cta + index;
                Sm120ExpStreamKRange theirs = sm120_exp_streamk_range(units, grid, source);
                int slot = tile == (int)(theirs.first / k_tiles) ? 0 : 1;
                sm120_exp_streamk_clear(flags + (long long)source * 2 + slot);
            }
            if (SM120_EXP_STREAMK_TRACE && threadIdx.x == 0) trace[5] = sm120_exp_streamk_now();
            sm120_exp_wide_epilogue<M, N>(
                output, rows, columns, output_row, output_column,
                warp_m, warp_n, bias, params, accumulator);
        }
        unit -= k_end - k_begin;
    }
    if (SM120_EXP_STREAMK_TRACE && threadIdx.x == 0) {
        trace[6] = sm120_exp_streamk_now();
        trace[7] = ((unsigned long long)segments << 48)
            | ((unsigned long long)units_first << 24)
            | (unsigned long long)units_last;
    }
}

template <int M, int N, int Stages>
static __device__ __forceinline__ void sm120_tf32_tn_exp_streamk_wide_entry(
    void* output, float* partial, unsigned* flags,
    const CUtensorMap& a_map, const CUtensorMap& b_map,
    const float* bias, const Sm120KernelParams& params) {
    if (sm120_tf32_reduction<Sm120Tn>(params) == 0) {
        sm120_tf32_tn_exp_streamk_zero_reduction<M, N>(output, bias, params);
        return;
    }
    sm120_tf32_tn_exp_streamk_wide_kernel<M, N, Stages>(
        output, partial, flags, a_map, b_map, bias, params);
}

#define SM120_DEFINE_TF32_TN_EXP_STREAMK_WIDE_KERNEL(NAME, M, N, STAGES)           \
extern "C" __global__ __launch_bounds__(SM120_EXP_WIDE_THREADS, 1) void NAME(   \
    void* output, float* partial, unsigned* flags,                             \
    const __grid_constant__ CUtensorMap a_map,                                 \
    const __grid_constant__ CUtensorMap b_map, const float* bias,              \
    const __grid_constant__ Sm120KernelParams params) {                        \
    sm120_tf32_tn_exp_streamk_wide_entry<M, N, STAGES>(                          \
        output, partial, flags, a_map, b_map, bias, params);                   \
}

SM120_DEFINE_TF32_TN_EXP_STREAMK_WIDE_KERNEL(
    gemm_bi_tn_sm120_tma_mma_tf32_v1_m128n128_bk32_s3_pair_exp_streamk_wide, 128, 128, 3)
SM120_DEFINE_TF32_TN_EXP_STREAMK_WIDE_KERNEL(
    gemm_bi_tn_sm120_tma_mma_tf32_v1_m128n128_bk32_s2_pair_exp_streamk_wide, 128, 128, 2)

#define SM120_DEFINE_TF32_TN_EXP_STREAMK_FLOWING_KERNEL(NAME, M, N, STAGES, BLOCKS)   \
extern "C" __global__ __launch_bounds__((M * N) / 32, BLOCKS) void NAME(    \
    void* output, float* partial, unsigned* flags,                             \
    const __grid_constant__ CUtensorMap a_map,                                 \
    const __grid_constant__ CUtensorMap b_map, const float* bias,              \
    const __grid_constant__ Sm120KernelParams params) {                        \
    sm120_tf32_tn_exp_streamk_flowing_entry<M, N, STAGES, (BLOCKS) == 1>(               \
        output, partial, flags, a_map, b_map, bias, params);                   \
}

SM120_DEFINE_TF32_TN_EXP_STREAMK_FLOWING_KERNEL(
    gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s4_pair_exp_streamk_flowing, 64, 128, 4, 1)
SM120_DEFINE_TF32_TN_EXP_STREAMK_FLOWING_KERNEL(
    gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s3_pair_exp_streamk_flowing, 64, 128, 3, 1)
SM120_DEFINE_TF32_TN_EXP_STREAMK_FLOWING_KERNEL(
    gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s2_pair_exp_streamk_flowing, 64, 128, 2, 2)

#endif
