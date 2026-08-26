#if defined(__CUDA_ARCH__) && \
    (__CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030)

#if __CUDACC_VER_MAJOR__ >= 13
struct alignas(128) CUtensorMap {
#else
struct alignas(64) CUtensorMap {
#endif
    unsigned long long opaque[16];
};
static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap size changed");
#if __CUDACC_VER_MAJOR__ >= 13
static_assert(alignof(CUtensorMap) == 128, "CUtensorMap alignment changed");
#else
static_assert(alignof(CUtensorMap) == 64, "CUtensorMap alignment changed");
#endif

struct Sm100KernelParams {
    int a_x;
    int a_y;
    int b_x;
    int b_y;
    float alpha;
    float beta;
    int m;
    int k;
    int n;
    int ldc;
};

static_assert(sizeof(Sm100KernelParams) == 40,
              "SM100 kernel parameter size changed");
static_assert(alignof(Sm100KernelParams) == 4,
              "SM100 kernel parameter alignment changed");
// Ten ordered 4-byte fields in 40 bytes leave no internal or tail padding.
static_assert(sizeof(((Sm100KernelParams*)0)->a_x) == 4,
              "SM100 A x origin size changed");
static_assert(sizeof(((Sm100KernelParams*)0)->a_y) == 4,
              "SM100 A y origin size changed");
static_assert(sizeof(((Sm100KernelParams*)0)->b_x) == 4,
              "SM100 B x origin size changed");
static_assert(sizeof(((Sm100KernelParams*)0)->b_y) == 4,
              "SM100 B y origin size changed");
static_assert(sizeof(((Sm100KernelParams*)0)->alpha) == 4,
              "SM100 alpha size changed");
static_assert(sizeof(((Sm100KernelParams*)0)->beta) == 4,
              "SM100 beta size changed");
static_assert(sizeof(((Sm100KernelParams*)0)->m) == 4,
              "SM100 M size changed");
static_assert(sizeof(((Sm100KernelParams*)0)->k) == 4,
              "SM100 K size changed");
static_assert(sizeof(((Sm100KernelParams*)0)->n) == 4,
              "SM100 N size changed");
static_assert(sizeof(((Sm100KernelParams*)0)->ldc) == 4,
              "SM100 output stride size changed");

enum Sm100Op {
    Sm100Nn = 0,
    Sm100Tn = 1,
    Sm100Nt = 2,
};

template <int Columns, int Stages>
struct Sm100Storage {
    // A stage starts with 16 KiB for A, followed by 8 or 16 KiB for B.
    // Full and empty barriers follow the payload; the TMEM base is last.
    static constexpr int stage_bytes = Columns == 64 ? 24576 : 32768;
    static constexpr int payload_bytes = stage_bytes * Stages;
    static constexpr int barrier_bytes = 16 * Stages;
    static constexpr int dynamic_bytes =
        (payload_bytes + barrier_bytes + 4 + 255) & ~255;
};

static_assert(Sm100Storage<64, 2>::dynamic_bytes == 49408,
              "N64/S2 shared size changed");
static_assert(Sm100Storage<64, 3>::dynamic_bytes == 73984,
              "N64/S3 shared size changed");
static_assert(Sm100Storage<64, 4>::dynamic_bytes == 98560,
              "N64/S4 shared size changed");
static_assert(Sm100Storage<128, 2>::dynamic_bytes == 65792,
              "N128/S2 shared size changed");
static_assert(Sm100Storage<128, 3>::dynamic_bytes == 98560,
              "N128/S3 shared size changed");
static_assert(Sm100Storage<128, 4>::dynamic_bytes == 131328,
              "N128/S4 shared size changed");

static __host__ __device__ constexpr unsigned long long sm100_desc_fields(
    unsigned start, unsigned leading, unsigned stride) {
    return static_cast<unsigned long long>(start & 0x3fffU) |
           (static_cast<unsigned long long>(leading & 0x3fffU) << 16) |
           (static_cast<unsigned long long>(stride & 0x3fffU) << 32) |
           (1ULL << 46) | (2ULL << 61);
}

static_assert(((sm100_desc_fields(0, 1, 64) >> 46) & 3ULL) == 1ULL,
              "descriptor version changed");
static_assert(((sm100_desc_fields(0, 1, 64) >> 52) & 1ULL) == 0ULL,
              "descriptor leading mode changed");
static_assert(((sm100_desc_fields(0, 1, 64) >> 49) & 7ULL) == 0ULL,
              "descriptor base offset changed");
static_assert(((sm100_desc_fields(0, 1, 64) >> 61) & 7ULL) == 2ULL,
              "descriptor swizzle changed");

static __device__ __forceinline__ unsigned long long sm100_desc(
    const void* pointer, unsigned leading, unsigned stride) {
    // This is legacy relative-leading-offset mode with base_offset 0 and SW128.
    unsigned address =
        static_cast<unsigned>(__cvta_generic_to_shared(pointer));
    return sm100_desc_fields((address >> 4) & 0x3fffU, leading, stride);
}

static __device__ __forceinline__ void sm100_init_barrier(
    unsigned barrier) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                 :: "r"(barrier) : "memory");
}

static __device__ __forceinline__ void sm100_wait_barrier(
    unsigned barrier, unsigned phase) {
    unsigned ready;
    do {
        asm volatile(
            "{ .reg .pred p; "
            "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 "
            "p, [%1], %2; "
            "selp.b32 %0, 1, 0, p; }"
            : "=r"(ready) : "r"(barrier), "r"(phase) : "memory");
    } while (!ready);
}

static __device__ __forceinline__ void sm100_expect_transaction(
    unsigned barrier, unsigned bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 "
        "_, [%0], %1;"
        :: "r"(barrier), "r"(bytes) : "memory");
}

static __device__ __forceinline__ void sm100_tma_copy(
    unsigned destination, unsigned long long map, int x, int y,
    int origin_x, int origin_y, unsigned barrier) {
    int map_x = x + origin_x;
    int map_y = y + origin_y;
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile."
        "mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];"
        :: "r"(destination), "l"(map), "r"(map_x), "r"(map_y),
           "r"(barrier)
        : "memory");
}

static __device__ __forceinline__ void sm100_tcgen_before() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

static __device__ __forceinline__ void sm100_tcgen_after() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

static __device__ __forceinline__ void sm100_tmem_alloc(
    unsigned pointer, int columns) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 "
        "[%0], %1;"
        :: "r"(pointer), "r"(columns) : "memory");
}

static __device__ __forceinline__ void sm100_tmem_relinquish() {
    asm volatile(
        "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"
        ::: "memory");
}

static __device__ __forceinline__ void sm100_tmem_dealloc(
    unsigned pointer, int columns) {
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
        :: "r"(pointer), "r"(columns) : "memory");
}

static __device__ __forceinline__ void sm100_tmem_store8(
    unsigned destination, const unsigned (&values)[8]) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x8.b32 "
        "[%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
        :: "r"(destination), "r"(values[0]), "r"(values[1]),
           "r"(values[2]), "r"(values[3]), "r"(values[4]),
           "r"(values[5]), "r"(values[6]), "r"(values[7])
        : "memory");
}

static __device__ __forceinline__ void sm100_tmem_load8(
    unsigned source, unsigned (&values)[8]) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=r"(values[0]), "=r"(values[1]), "=r"(values[2]),
          "=r"(values[3]), "=r"(values[4]), "=r"(values[5]),
          "=r"(values[6]), "=r"(values[7])
        : "r"(source) : "memory");
}

static __device__ __forceinline__ void sm100_tmem_wait_store() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

static __device__ __forceinline__ void sm100_tmem_wait_load() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

static __device__ __forceinline__ void sm100_tcgen_commit(
    unsigned barrier) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one."
        "shared::cluster.b64 [%0];"
        :: "r"(barrier) : "memory");
}

template <typename T, int Op, int Columns>
struct Sm100Instruction;

template <> struct Sm100Instruction<__half, Sm100Nt, 64> {
    static constexpr unsigned value = 0x08100010;
};
template <> struct Sm100Instruction<__nv_bfloat16, Sm100Nt, 64> {
    static constexpr unsigned value = 0x08100490;
};
template <> struct Sm100Instruction<__half, Sm100Nn, 64> {
    static constexpr unsigned value = 0x08110010;
};
template <> struct Sm100Instruction<__nv_bfloat16, Sm100Nn, 64> {
    static constexpr unsigned value = 0x08110490;
};
template <> struct Sm100Instruction<__half, Sm100Tn, 64> {
    static constexpr unsigned value = 0x08118010;
};
template <> struct Sm100Instruction<__nv_bfloat16, Sm100Tn, 64> {
    static constexpr unsigned value = 0x08118490;
};
template <> struct Sm100Instruction<__half, Sm100Nt, 128> {
    static constexpr unsigned value = 0x08200010;
};
template <> struct Sm100Instruction<__nv_bfloat16, Sm100Nt, 128> {
    static constexpr unsigned value = 0x08200490;
};
template <> struct Sm100Instruction<__half, Sm100Nn, 128> {
    static constexpr unsigned value = 0x08210010;
};
template <> struct Sm100Instruction<__nv_bfloat16, Sm100Nn, 128> {
    static constexpr unsigned value = 0x08210490;
};
template <> struct Sm100Instruction<__half, Sm100Tn, 128> {
    static constexpr unsigned value = 0x08218010;
};
template <> struct Sm100Instruction<__nv_bfloat16, Sm100Tn, 128> {
    static constexpr unsigned value = 0x08218490;
};

template <typename T, int Op, int Columns>
static __device__ __forceinline__ void sm100_tcgen_mma(
    unsigned accumulator, unsigned long long a, unsigned long long b,
    int enable_input_d) {
    unsigned zero = 0;
    unsigned instruction = Sm100Instruction<T, Op, Columns>::value;
    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p, %4, 0; "
        "tcgen05.mma.cta_group::1.kind::f16 "
        "[%0], %1, %2, %3, {%5, %6, %7, %8}, p; }"
        :: "r"(accumulator), "l"(a), "l"(b), "r"(instruction),
           "r"(enable_input_d), "r"(zero), "r"(zero), "r"(zero),
           "r"(zero)
        : "memory");
}

template <int Op, int Columns, int Stages>
static __device__ __forceinline__ void sm100_produce_stage(
    const CUtensorMap& a_map, const CUtensorMap& b_map,
    const Sm100KernelParams& params, unsigned shared,
    unsigned full_barrier, int tile, int output_row, int output_col) {
    constexpr int stage_bytes = Sm100Storage<Columns, Stages>::stage_bytes;
    unsigned stage = shared + (tile % Stages) * stage_bytes;
    unsigned a_destination = stage;
    unsigned b_destination = stage + 16384;
    unsigned long long a_descriptor =
        reinterpret_cast<unsigned long long>(&a_map);
    unsigned long long b_descriptor =
        reinterpret_cast<unsigned long long>(&b_map);
    int reduction = tile * 64;

    sm100_expect_transaction(full_barrier, stage_bytes);
    if (Op == Sm100Nn) {
        sm100_tma_copy(a_destination, a_descriptor, reduction, output_row,
                       params.a_x, params.a_y, full_barrier);
        sm100_tma_copy(b_destination, b_descriptor, output_col, reduction,
                       params.b_x, params.b_y, full_barrier);
        if (Columns == 128) {
            sm100_tma_copy(b_destination + 8192, b_descriptor,
                           output_col + 64, reduction,
                           params.b_x, params.b_y, full_barrier);
        }
    } else if (Op == Sm100Tn) {
        sm100_tma_copy(a_destination, a_descriptor, output_row, reduction,
                       params.a_x, params.a_y, full_barrier);
        sm100_tma_copy(a_destination + 8192, a_descriptor, output_row + 64,
                       reduction, params.a_x, params.a_y, full_barrier);
        sm100_tma_copy(b_destination, b_descriptor, output_col, reduction,
                       params.b_x, params.b_y, full_barrier);
        if (Columns == 128) {
            sm100_tma_copy(b_destination + 8192, b_descriptor,
                           output_col + 64, reduction,
                           params.b_x, params.b_y, full_barrier);
        }
    } else {
        sm100_tma_copy(a_destination, a_descriptor, reduction, output_row,
                       params.a_x, params.a_y, full_barrier);
        sm100_tma_copy(b_destination, b_descriptor, reduction, output_col,
                       params.b_x, params.b_y, full_barrier);
    }
}

template <int Op, int Columns, int Stages>
static __device__ __forceinline__ void sm100_stage_descriptors(
    unsigned char* storage, int tile, unsigned long long& a,
    unsigned long long& b, unsigned& a_step, unsigned& b_step) {
    unsigned char* stage = storage +
        (tile % Stages) * Sm100Storage<Columns, Stages>::stage_bytes;
    if (Op == Sm100Tn) {
        // TN reads the two stacked A halves through the transpose mode.
        a = sm100_desc(stage, 512, 64);
        a_step = 128;
    } else {
        a = sm100_desc(stage, 1, 64);
        a_step = 2;
    }
    if (Op == Sm100Nt) {
        b = sm100_desc(stage + 16384, 1, 64);
        b_step = 2;
    } else {
        b = sm100_desc(stage + 16384, Columns == 64 ? 0 : 512, 64);
        b_step = 128;
    }
}

template <typename T, int Op, int Columns, int Stages>
static __device__ __forceinline__ void sm100_issue_tile(
    unsigned accumulator, unsigned char* storage, int tile,
    int& enable_input_d) {
    unsigned long long a;
    unsigned long long b;
    unsigned a_step;
    unsigned b_step;
    sm100_stage_descriptors<Op, Columns, Stages>(storage, tile, a, b,
                                                 a_step, b_step);
#pragma unroll
    for (int slab = 0; slab < 4; ++slab) {
        sm100_tcgen_mma<T, Op, Columns>(
            accumulator, a + slab * a_step, b + slab * b_step,
            enable_input_d);
        enable_input_d = 1;
    }
}

template <int Columns, bool ProducerSchedule>
static __device__ __forceinline__ int sm100_chunk_begin() {
    return ProducerSchedule ? static_cast<int>(threadIdx.x >> 7) : 0;
}

template <bool ProducerSchedule>
static __device__ __forceinline__ int sm100_chunk_stride() {
    return ProducerSchedule ? 2 : 1;
}

static __device__ __forceinline__ unsigned sm100_warp_accumulator(
    unsigned accumulator) {
    // TMEM lane is taddr[31:16]; each warp owns its fixed 32-lane quarter.
    unsigned warp_in_group = static_cast<unsigned>((threadIdx.x >> 5) & 3);
    return accumulator + ((warp_in_group * 32U) << 16);
}

template <int Columns, bool ProducerSchedule>
static __device__ __forceinline__ void sm100_seed_bias(
    unsigned accumulator, const float* bias, int output_col,
    int output_columns) {
    // The two-warpgroup schedule gives each group disjoint TMEM columns.
    int begin = sm100_chunk_begin<Columns, ProducerSchedule>();
    int stride = sm100_chunk_stride<ProducerSchedule>();
    unsigned warp_accumulator = sm100_warp_accumulator(accumulator);
    for (int chunk = begin; chunk < Columns / 8; chunk += stride) {
        unsigned values[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            int column = output_col + chunk * 8 + j;
            float value = column < output_columns ? bias[column] : 0.0f;
            values[j] = __float_as_uint(value);
        }
        sm100_tmem_store8(warp_accumulator + chunk * 8, values);
        sm100_tmem_wait_store();
    }
    sm100_tcgen_before();
}

template <typename T>
static __device__ __forceinline__ T sm100_from_float(float value);

template <>
__device__ __forceinline__ __half sm100_from_float(float value) {
    return from_f_f16(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 sm100_from_float(float value) {
    return from_f_bf16(value);
}

template <typename T, int Op>
static __device__ __forceinline__ void sm100_store_pair(
    void* output, int row, int column, int stride, float alpha, float beta,
    float first, float second) {
    long long offset = static_cast<long long>(row) * stride + column;
    if (Op == Sm100Tn) {
        float* destination = static_cast<float*>(output) + offset;
        float v0 = __fmaf_rn(alpha, first, destination[0]);
        float v1 = __fmaf_rn(alpha, second, destination[1]);
        if ((reinterpret_cast<unsigned long long>(destination) & 7ULL) == 0) {
            float2 pair = {v0, v1};
            *reinterpret_cast<float2*>(destination) = pair;
        } else {
            destination[0] = v0;
            destination[1] = v1;
        }
    } else {
        T* destination = static_cast<T*>(output) + offset;
        float v0 = alpha == 1.0f ? first : __fmul_rn(alpha, first);
        float v1 = alpha == 1.0f ? second : __fmul_rn(alpha, second);
        if (Op == Sm100Nn && beta != 0.0f) {
            v0 = __fmaf_rn(beta, to_f(destination[0]), v0);
            v1 = __fmaf_rn(beta, to_f(destination[1]), v1);
        }
        if ((reinterpret_cast<unsigned long long>(destination) & 3ULL) == 0) {
            sgb_store_pair_rne(destination, v0, v1);
        } else {
            destination[0] = sm100_from_float<T>(v0);
            destination[1] = sm100_from_float<T>(v1);
        }
    }
}

template <typename T, int Op>
static __device__ __forceinline__ void sm100_store_scalar(
    void* output, int row, int column, int stride, float alpha, float beta,
    float accumulator) {
    long long offset = static_cast<long long>(row) * stride + column;
    if (Op == Sm100Tn) {
        float* destination = static_cast<float*>(output) + offset;
        destination[0] = __fmaf_rn(alpha, accumulator, destination[0]);
    } else {
        T* destination = static_cast<T*>(output) + offset;
        float value = alpha == 1.0f
            ? accumulator
            : __fmul_rn(alpha, accumulator);
        if (Op == Sm100Nn && beta != 0.0f) {
            value = __fmaf_rn(beta, to_f(destination[0]), value);
        }
        destination[0] = sm100_from_float<T>(value);
    }
}

template <typename T, int Op, int Columns, bool ProducerSchedule>
static __device__ __forceinline__ void sm100_epilogue(
    void* output, unsigned accumulator, float alpha, float beta,
    int output_rows, int output_columns, int output_stride,
    int output_row, int output_col) {
    int warp_in_group = (threadIdx.x >> 5) & 3;
    int lane = threadIdx.x & 31;
    int row = output_row + warp_in_group * 32 + lane;
    int begin = sm100_chunk_begin<Columns, ProducerSchedule>();
    int stride = sm100_chunk_stride<ProducerSchedule>();
    unsigned warp_accumulator = sm100_warp_accumulator(accumulator);

    sm100_tcgen_after();
    for (int chunk = begin; chunk < Columns / 8; chunk += stride) {
        unsigned words[8];
        sm100_tmem_load8(warp_accumulator + chunk * 8, words);
        sm100_tmem_wait_load();
        if (row >= output_rows) continue;
#pragma unroll
        for (int j = 0; j < 8; j += 2) {
            int column = output_col + chunk * 8 + j;
            if (column >= output_columns) continue;
            float first = __uint_as_float(words[j]);
            if (column + 1 < output_columns) {
                float second = __uint_as_float(words[j + 1]);
                sm100_store_pair<T, Op>(output, row, column, output_stride,
                                        alpha, beta, first, second);
            } else {
                sm100_store_scalar<T, Op>(
                    output, row, column, output_stride, alpha, beta, first);
            }
        }
    }
    sm100_tcgen_before();
}

template <typename T, int Op, int Columns, int Stages>
static __device__ __forceinline__ void sm100_compact_mainloop(
    const CUtensorMap& a_map, const CUtensorMap& b_map,
    const Sm100KernelParams& params, unsigned char* storage,
    unsigned shared, unsigned full_base, unsigned empty_base,
    unsigned accumulator, int tile_count, int output_row, int output_col,
    int seed) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (warp != 0) return;

    int prologue = tile_count < Stages ? tile_count : Stages;
    for (int tile = 0; tile < prologue; ++tile) {
        if (lane == 0) {
            sm100_produce_stage<Op, Columns, Stages>(
                a_map, b_map, params, shared, full_base + tile * 8, tile,
                output_row, output_col);
        }
    }

    int enable_input_d = seed;
    for (int tile = 0; tile < tile_count; ++tile) {
        int stage = tile % Stages;
        unsigned generation = static_cast<unsigned>(tile / Stages);
        // Full and empty barriers alternate parity whenever a stage wraps.
        sm100_wait_barrier(full_base + stage * 8, generation & 1U);
        if (lane == 0) {
            sm100_tcgen_after();
            sm100_issue_tile<T, Op, Columns, Stages>(
                accumulator, storage, tile, enable_input_d);
            sm100_tcgen_commit(empty_base + stage * 8);
        }
        int refill = tile + Stages;
        if (refill < tile_count) {
            sm100_wait_barrier(empty_base + stage * 8, generation & 1U);
            if (lane == 0) {
                sm100_produce_stage<Op, Columns, Stages>(
                    a_map, b_map, params, shared, full_base + stage * 8,
                    refill, output_row, output_col);
            }
        }
    }
}

template <int Op, int Columns, int Stages>
static __device__ __forceinline__ void sm100_producer_mainloop(
    const CUtensorMap& a_map, const CUtensorMap& b_map,
    const Sm100KernelParams& params, unsigned shared,
    unsigned full_base, unsigned empty_base, int tile_count,
    int output_row, int output_col) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (warp != 4) return;
    for (int tile = 0; tile < tile_count; ++tile) {
        int stage = tile % Stages;
        unsigned generation = static_cast<unsigned>(tile / Stages);
        if (generation != 0) {
            sm100_wait_barrier(empty_base + stage * 8,
                               (generation - 1U) & 1U);
        }
        if (lane == 0) {
            sm100_produce_stage<Op, Columns, Stages>(
                a_map, b_map, params, shared, full_base + stage * 8,
                tile, output_row, output_col);
        }
    }
}

template <typename T, int Op, int Columns, int Stages>
static __device__ __forceinline__ void sm100_consumer_mainloop(
    unsigned char* storage, unsigned full_base, unsigned empty_base,
    unsigned accumulator, int tile_count, int seed) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (warp != 5) return;
    int enable_input_d = seed;
    for (int tile = 0; tile < tile_count; ++tile) {
        int stage = tile % Stages;
        unsigned generation = static_cast<unsigned>(tile / Stages);
        sm100_wait_barrier(full_base + stage * 8, generation & 1U);
        if (lane == 0) {
            sm100_tcgen_after();
            sm100_issue_tile<T, Op, Columns, Stages>(
                accumulator, storage, tile, enable_input_d);
            sm100_tcgen_commit(empty_base + stage * 8);
        }
    }
}

template <int Stages>
static __device__ __forceinline__ void sm100_wait_final_commits(
    unsigned empty_base, int tile_count) {
    int used = tile_count < Stages ? tile_count : Stages;
    for (int stage = 0; stage < used; ++stage) {
        int last = stage + ((tile_count - 1 - stage) / Stages) * Stages;
        unsigned phase = static_cast<unsigned>(last / Stages) & 1U;
        sm100_wait_barrier(empty_base + stage * 8, phase);
    }
}

template <typename T, int Op, int Columns, int Stages,
          bool ProducerSchedule>
static __device__ __forceinline__ void sm100_kernel(
    void* output, const CUtensorMap& a_map, const CUtensorMap& b_map,
    const float* bias, const Sm100KernelParams& params) {
    assert(params.alpha == 1.0f || bias == nullptr);
    extern __shared__ __align__(1024) unsigned char storage[];
    constexpr int payload_bytes = Sm100Storage<Columns, Stages>::payload_bytes;
    constexpr int management_warp = ProducerSchedule ? 4 : 0;

    unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(storage));
    unsigned full_base = shared + payload_bytes;
    unsigned empty_base = full_base + Stages * 8;
    unsigned pointer_address = empty_base + Stages * 8;
    int warp = threadIdx.x >> 5;

    int output_rows = Op == Sm100Tn ? params.k : params.m;
    int output_columns = Op == Sm100Nt ? params.k : params.n;
    int reduction = Op == Sm100Nn
        ? params.k
        : (Op == Sm100Tn ? params.m : params.n);
    unsigned column_tiles = 1U +
        (static_cast<unsigned>(output_columns) - 1U) /
            static_cast<unsigned>(Columns);
    unsigned output_row_value = (blockIdx.x / column_tiles) * 128U;
    unsigned output_col_value =
        (blockIdx.x % column_tiles) * static_cast<unsigned>(Columns);
    int output_row = static_cast<int>(output_row_value);
    int output_col = static_cast<int>(output_col_value);
    int tile_count = static_cast<int>(
        (static_cast<unsigned long long>(reduction) + 63ULL) / 64ULL);

    if (warp == management_warp) {
        sm100_tmem_alloc(pointer_address, Columns);
    }
    if (threadIdx.x == management_warp * 32) {
        for (int stage = 0; stage < Stages; ++stage) {
            sm100_init_barrier(full_base + stage * 8);
            sm100_init_barrier(empty_base + stage * 8);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    unsigned accumulator =
        *reinterpret_cast<unsigned*>(storage + payload_bytes + 16 * Stages);
    int seed = Op == Sm100Nn && bias != nullptr;
    if (seed) {
        sm100_seed_bias<Columns, ProducerSchedule>(
            accumulator, bias, output_col, output_columns);
    }
    __syncthreads();

    if (ProducerSchedule) {
        // p8 assigns TMA to warp 4 and MMA issue to warp 5.
        sm100_producer_mainloop<Op, Columns, Stages>(
            a_map, b_map, params, shared, full_base, empty_base,
            tile_count, output_row, output_col);
        sm100_consumer_mainloop<T, Op, Columns, Stages>(
            storage, full_base, empty_base, accumulator, tile_count, seed);
    } else {
        // c4 keeps TMA and MMA issue in warp 0 while the CTA handles output.
        sm100_compact_mainloop<T, Op, Columns, Stages>(
            a_map, b_map, params, storage, shared, full_base, empty_base,
            accumulator, tile_count, output_row, output_col, seed);
    }

    sm100_wait_final_commits<Stages>(empty_base, tile_count);
    sm100_epilogue<T, Op, Columns, ProducerSchedule>(
        output, accumulator, params.alpha, params.beta, output_rows,
        output_columns, params.ldc, output_row, output_col);
    __syncthreads();

    // Every reader fences and leaves TMEM before its management warp frees it.
    if (warp == management_warp) {
        sm100_tcgen_after();
        sm100_tmem_relinquish();
        sm100_tmem_dealloc(accumulator, Columns);
    }
}

#define SM100_DEFINE_KERNEL(NAME, TYPE, OP, COLUMNS, STAGES, PRODUCER, THREADS) \
extern "C" __global__ __launch_bounds__(THREADS)                              \
void NAME(void* output, const __grid_constant__ CUtensorMap a_map,             \
          const __grid_constant__ CUtensorMap b_map, const float* bias,        \
          const __grid_constant__ Sm100KernelParams params) {                  \
    sm100_kernel<TYPE, OP, COLUMNS, STAGES, PRODUCER>(                         \
        output, a_map, b_map, bias, params);                                   \
}

SM100_DEFINE_KERNEL(sgemm_bi_nn_sm100_tcgen_m128n64_bk64_s2_c4_bf16, __nv_bfloat16, Sm100Nn, 64, 2, false, 128)
SM100_DEFINE_KERNEL(sgemm_bi_nn_sm100_tcgen_m128n64_bk64_s2_c4_f16, __half, Sm100Nn, 64, 2, false, 128)
SM100_DEFINE_KERNEL(sgemm_bi_nn_sm100_tcgen_m128n64_bk64_s2_p8_bf16, __nv_bfloat16, Sm100Nn, 64, 2, true, 256)
SM100_DEFINE_KERNEL(sgemm_bi_nn_sm100_tcgen_m128n64_bk64_s2_p8_f16, __half, Sm100Nn, 64, 2, true, 256)
SM100_DEFINE_KERNEL(sgemm_bi_nn_sm100_tcgen_m128n64_bk64_s3_c4_bf16, __nv_bfloat16, Sm100Nn, 64, 3, false, 128)
SM100_DEFINE_KERNEL(sgemm_bi_nn_sm100_tcgen_m128n64_bk64_s3_c4_f16, __half, Sm100Nn, 64, 3, false, 128)
SM100_DEFINE_KERNEL(sgemm_bi_nn_sm100_tcgen_m128n64_bk64_s3_p8_bf16, __nv_bfloat16, Sm100Nn, 64, 3, true, 256)
SM100_DEFINE_KERNEL(sgemm_bi_nn_sm100_tcgen_m128n64_bk64_s3_p8_f16, __half, Sm100Nn, 64, 3, true, 256)
SM100_DEFINE_KERNEL(sgemm_bi_nn_sm100_tcgen_m128n64_bk64_s4_c4_bf16, __nv_bfloat16, Sm100Nn, 64, 4, false, 128)
SM100_DEFINE_KERNEL(sgemm_bi_nn_sm100_tcgen_m128n64_bk64_s4_c4_f16, __half, Sm100Nn, 64, 4, false, 128)
SM100_DEFINE_KERNEL(sgemm_bi_nn_sm100_tcgen_m128n64_bk64_s4_p8_bf16, __nv_bfloat16, Sm100Nn, 64, 4, true, 256)
SM100_DEFINE_KERNEL(sgemm_bi_nn_sm100_tcgen_m128n64_bk64_s4_p8_f16, __half, Sm100Nn, 64, 4, true, 256)
SM100_DEFINE_KERNEL(sgemm_bi_nn_sm100_tcgen_m128n128_bk64_s2_c4_bf16, __nv_bfloat16, Sm100Nn, 128, 2, false, 128)
SM100_DEFINE_KERNEL(sgemm_bi_nn_sm100_tcgen_m128n128_bk64_s2_c4_f16, __half, Sm100Nn, 128, 2, false, 128)
SM100_DEFINE_KERNEL(sgemm_bi_nn_sm100_tcgen_m128n128_bk64_s2_p8_bf16, __nv_bfloat16, Sm100Nn, 128, 2, true, 256)
SM100_DEFINE_KERNEL(sgemm_bi_nn_sm100_tcgen_m128n128_bk64_s2_p8_f16, __half, Sm100Nn, 128, 2, true, 256)
SM100_DEFINE_KERNEL(sgemm_bi_nn_sm100_tcgen_m128n128_bk64_s3_c4_bf16, __nv_bfloat16, Sm100Nn, 128, 3, false, 128)
SM100_DEFINE_KERNEL(sgemm_bi_nn_sm100_tcgen_m128n128_bk64_s3_c4_f16, __half, Sm100Nn, 128, 3, false, 128)
SM100_DEFINE_KERNEL(sgemm_bi_nn_sm100_tcgen_m128n128_bk64_s3_p8_bf16, __nv_bfloat16, Sm100Nn, 128, 3, true, 256)
SM100_DEFINE_KERNEL(sgemm_bi_nn_sm100_tcgen_m128n128_bk64_s3_p8_f16, __half, Sm100Nn, 128, 3, true, 256)
SM100_DEFINE_KERNEL(sgemm_bi_nn_sm100_tcgen_m128n128_bk64_s4_c4_bf16, __nv_bfloat16, Sm100Nn, 128, 4, false, 128)
SM100_DEFINE_KERNEL(sgemm_bi_nn_sm100_tcgen_m128n128_bk64_s4_c4_f16, __half, Sm100Nn, 128, 4, false, 128)
SM100_DEFINE_KERNEL(sgemm_bi_nn_sm100_tcgen_m128n128_bk64_s4_p8_bf16, __nv_bfloat16, Sm100Nn, 128, 4, true, 256)
SM100_DEFINE_KERNEL(sgemm_bi_nn_sm100_tcgen_m128n128_bk64_s4_p8_f16, __half, Sm100Nn, 128, 4, true, 256)

SM100_DEFINE_KERNEL(sgemm_bi_tn_sm100_tcgen_m128n64_bk64_s2_c4_bf16, __nv_bfloat16, Sm100Tn, 64, 2, false, 128)
SM100_DEFINE_KERNEL(sgemm_bi_tn_sm100_tcgen_m128n64_bk64_s2_c4_f16, __half, Sm100Tn, 64, 2, false, 128)
SM100_DEFINE_KERNEL(sgemm_bi_tn_sm100_tcgen_m128n64_bk64_s2_p8_bf16, __nv_bfloat16, Sm100Tn, 64, 2, true, 256)
SM100_DEFINE_KERNEL(sgemm_bi_tn_sm100_tcgen_m128n64_bk64_s2_p8_f16, __half, Sm100Tn, 64, 2, true, 256)
SM100_DEFINE_KERNEL(sgemm_bi_tn_sm100_tcgen_m128n64_bk64_s3_c4_bf16, __nv_bfloat16, Sm100Tn, 64, 3, false, 128)
SM100_DEFINE_KERNEL(sgemm_bi_tn_sm100_tcgen_m128n64_bk64_s3_c4_f16, __half, Sm100Tn, 64, 3, false, 128)
SM100_DEFINE_KERNEL(sgemm_bi_tn_sm100_tcgen_m128n64_bk64_s3_p8_bf16, __nv_bfloat16, Sm100Tn, 64, 3, true, 256)
SM100_DEFINE_KERNEL(sgemm_bi_tn_sm100_tcgen_m128n64_bk64_s3_p8_f16, __half, Sm100Tn, 64, 3, true, 256)
SM100_DEFINE_KERNEL(sgemm_bi_tn_sm100_tcgen_m128n64_bk64_s4_c4_bf16, __nv_bfloat16, Sm100Tn, 64, 4, false, 128)
SM100_DEFINE_KERNEL(sgemm_bi_tn_sm100_tcgen_m128n64_bk64_s4_c4_f16, __half, Sm100Tn, 64, 4, false, 128)
SM100_DEFINE_KERNEL(sgemm_bi_tn_sm100_tcgen_m128n64_bk64_s4_p8_bf16, __nv_bfloat16, Sm100Tn, 64, 4, true, 256)
SM100_DEFINE_KERNEL(sgemm_bi_tn_sm100_tcgen_m128n64_bk64_s4_p8_f16, __half, Sm100Tn, 64, 4, true, 256)
SM100_DEFINE_KERNEL(sgemm_bi_tn_sm100_tcgen_m128n128_bk64_s2_c4_bf16, __nv_bfloat16, Sm100Tn, 128, 2, false, 128)
SM100_DEFINE_KERNEL(sgemm_bi_tn_sm100_tcgen_m128n128_bk64_s2_c4_f16, __half, Sm100Tn, 128, 2, false, 128)
SM100_DEFINE_KERNEL(sgemm_bi_tn_sm100_tcgen_m128n128_bk64_s2_p8_bf16, __nv_bfloat16, Sm100Tn, 128, 2, true, 256)
SM100_DEFINE_KERNEL(sgemm_bi_tn_sm100_tcgen_m128n128_bk64_s2_p8_f16, __half, Sm100Tn, 128, 2, true, 256)
SM100_DEFINE_KERNEL(sgemm_bi_tn_sm100_tcgen_m128n128_bk64_s3_c4_bf16, __nv_bfloat16, Sm100Tn, 128, 3, false, 128)
SM100_DEFINE_KERNEL(sgemm_bi_tn_sm100_tcgen_m128n128_bk64_s3_c4_f16, __half, Sm100Tn, 128, 3, false, 128)
SM100_DEFINE_KERNEL(sgemm_bi_tn_sm100_tcgen_m128n128_bk64_s3_p8_bf16, __nv_bfloat16, Sm100Tn, 128, 3, true, 256)
SM100_DEFINE_KERNEL(sgemm_bi_tn_sm100_tcgen_m128n128_bk64_s3_p8_f16, __half, Sm100Tn, 128, 3, true, 256)
SM100_DEFINE_KERNEL(sgemm_bi_tn_sm100_tcgen_m128n128_bk64_s4_c4_bf16, __nv_bfloat16, Sm100Tn, 128, 4, false, 128)
SM100_DEFINE_KERNEL(sgemm_bi_tn_sm100_tcgen_m128n128_bk64_s4_c4_f16, __half, Sm100Tn, 128, 4, false, 128)
SM100_DEFINE_KERNEL(sgemm_bi_tn_sm100_tcgen_m128n128_bk64_s4_p8_bf16, __nv_bfloat16, Sm100Tn, 128, 4, true, 256)
SM100_DEFINE_KERNEL(sgemm_bi_tn_sm100_tcgen_m128n128_bk64_s4_p8_f16, __half, Sm100Tn, 128, 4, true, 256)

SM100_DEFINE_KERNEL(sgemm_bi_nt_sm100_tcgen_m128n64_bk64_s2_c4_bf16, __nv_bfloat16, Sm100Nt, 64, 2, false, 128)
SM100_DEFINE_KERNEL(sgemm_bi_nt_sm100_tcgen_m128n64_bk64_s2_c4_f16, __half, Sm100Nt, 64, 2, false, 128)
SM100_DEFINE_KERNEL(sgemm_bi_nt_sm100_tcgen_m128n64_bk64_s2_p8_bf16, __nv_bfloat16, Sm100Nt, 64, 2, true, 256)
SM100_DEFINE_KERNEL(sgemm_bi_nt_sm100_tcgen_m128n64_bk64_s2_p8_f16, __half, Sm100Nt, 64, 2, true, 256)
SM100_DEFINE_KERNEL(sgemm_bi_nt_sm100_tcgen_m128n64_bk64_s3_c4_bf16, __nv_bfloat16, Sm100Nt, 64, 3, false, 128)
SM100_DEFINE_KERNEL(sgemm_bi_nt_sm100_tcgen_m128n64_bk64_s3_c4_f16, __half, Sm100Nt, 64, 3, false, 128)
SM100_DEFINE_KERNEL(sgemm_bi_nt_sm100_tcgen_m128n64_bk64_s3_p8_bf16, __nv_bfloat16, Sm100Nt, 64, 3, true, 256)
SM100_DEFINE_KERNEL(sgemm_bi_nt_sm100_tcgen_m128n64_bk64_s3_p8_f16, __half, Sm100Nt, 64, 3, true, 256)
SM100_DEFINE_KERNEL(sgemm_bi_nt_sm100_tcgen_m128n64_bk64_s4_c4_bf16, __nv_bfloat16, Sm100Nt, 64, 4, false, 128)
SM100_DEFINE_KERNEL(sgemm_bi_nt_sm100_tcgen_m128n64_bk64_s4_c4_f16, __half, Sm100Nt, 64, 4, false, 128)
SM100_DEFINE_KERNEL(sgemm_bi_nt_sm100_tcgen_m128n64_bk64_s4_p8_bf16, __nv_bfloat16, Sm100Nt, 64, 4, true, 256)
SM100_DEFINE_KERNEL(sgemm_bi_nt_sm100_tcgen_m128n64_bk64_s4_p8_f16, __half, Sm100Nt, 64, 4, true, 256)
SM100_DEFINE_KERNEL(sgemm_bi_nt_sm100_tcgen_m128n128_bk64_s2_c4_bf16, __nv_bfloat16, Sm100Nt, 128, 2, false, 128)
SM100_DEFINE_KERNEL(sgemm_bi_nt_sm100_tcgen_m128n128_bk64_s2_c4_f16, __half, Sm100Nt, 128, 2, false, 128)
SM100_DEFINE_KERNEL(sgemm_bi_nt_sm100_tcgen_m128n128_bk64_s2_p8_bf16, __nv_bfloat16, Sm100Nt, 128, 2, true, 256)
SM100_DEFINE_KERNEL(sgemm_bi_nt_sm100_tcgen_m128n128_bk64_s2_p8_f16, __half, Sm100Nt, 128, 2, true, 256)
SM100_DEFINE_KERNEL(sgemm_bi_nt_sm100_tcgen_m128n128_bk64_s3_c4_bf16, __nv_bfloat16, Sm100Nt, 128, 3, false, 128)
SM100_DEFINE_KERNEL(sgemm_bi_nt_sm100_tcgen_m128n128_bk64_s3_c4_f16, __half, Sm100Nt, 128, 3, false, 128)
SM100_DEFINE_KERNEL(sgemm_bi_nt_sm100_tcgen_m128n128_bk64_s3_p8_bf16, __nv_bfloat16, Sm100Nt, 128, 3, true, 256)
SM100_DEFINE_KERNEL(sgemm_bi_nt_sm100_tcgen_m128n128_bk64_s3_p8_f16, __half, Sm100Nt, 128, 3, true, 256)
SM100_DEFINE_KERNEL(sgemm_bi_nt_sm100_tcgen_m128n128_bk64_s4_c4_bf16, __nv_bfloat16, Sm100Nt, 128, 4, false, 128)
SM100_DEFINE_KERNEL(sgemm_bi_nt_sm100_tcgen_m128n128_bk64_s4_c4_f16, __half, Sm100Nt, 128, 4, false, 128)
SM100_DEFINE_KERNEL(sgemm_bi_nt_sm100_tcgen_m128n128_bk64_s4_p8_bf16, __nv_bfloat16, Sm100Nt, 128, 4, true, 256)
SM100_DEFINE_KERNEL(sgemm_bi_nt_sm100_tcgen_m128n128_bk64_s4_p8_f16, __half, Sm100Nt, 128, 4, true, 256)

#undef SM100_DEFINE_KERNEL

#endif
