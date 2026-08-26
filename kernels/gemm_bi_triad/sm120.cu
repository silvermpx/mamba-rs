#if defined(__CUDA_ARCH__) && \
    (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)

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

struct Sm120KernelParams {
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

static_assert(sizeof(Sm120KernelParams) == 40,
              "SM120 kernel parameter size changed");
static_assert(alignof(Sm120KernelParams) == 4,
              "SM120 kernel parameter alignment changed");
static_assert(sizeof(((Sm120KernelParams*)0)->a_x) == 4,
              "SM120 A x origin size changed");
static_assert(sizeof(((Sm120KernelParams*)0)->a_y) == 4,
              "SM120 A y origin size changed");
static_assert(sizeof(((Sm120KernelParams*)0)->b_x) == 4,
              "SM120 B x origin size changed");
static_assert(sizeof(((Sm120KernelParams*)0)->b_y) == 4,
              "SM120 B y origin size changed");
static_assert(sizeof(((Sm120KernelParams*)0)->alpha) == 4,
              "SM120 alpha size changed");
static_assert(sizeof(((Sm120KernelParams*)0)->beta) == 4,
              "SM120 beta size changed");
static_assert(sizeof(((Sm120KernelParams*)0)->m) == 4,
              "SM120 M size changed");
static_assert(sizeof(((Sm120KernelParams*)0)->k) == 4,
              "SM120 K size changed");
static_assert(sizeof(((Sm120KernelParams*)0)->n) == 4,
              "SM120 N size changed");
static_assert(sizeof(((Sm120KernelParams*)0)->ldc) == 4,
              "SM120 output stride size changed");

enum Sm120Op {
    Sm120Nn = 0,
    Sm120Tn = 1,
    Sm120Nt = 2,
};

// Tensor maps use CU_TENSOR_MAP_DATA_TYPE_UINT16 for both activation types.
// BK32 consumes CU_TENSOR_MAP_SWIZZLE_64B in 32-element planes; BK64 consumes
// CU_TENSOR_MAP_SWIZZLE_128B in 64-element planes.
#define SM120_SWIZZLE_64B 4
#define SM120_SWIZZLE_128B 8

template <int M, int N, int BK, int Stages>
struct Sm120Storage {
    static constexpr int threads = (M / 32) * (N / 32) * 32;
    static constexpr int stage_bytes = (M + N) * BK * 2;
    static constexpr int dynamic_bytes = 128 + Stages * stage_bytes;
};

#define SM120_CHECK_STORAGE(M, N, BK, STAGES, BYTES)                         \
    static_assert(Sm120Storage<M, N, BK, STAGES>::dynamic_bytes == BYTES,    \
                  "SM120 dynamic shared size changed")

SM120_CHECK_STORAGE(64, 64, 32, 2, 16512);
SM120_CHECK_STORAGE(64, 64, 32, 3, 24704);
SM120_CHECK_STORAGE(64, 64, 64, 2, 32896);
SM120_CHECK_STORAGE(64, 64, 64, 3, 49280);
SM120_CHECK_STORAGE(128, 64, 32, 2, 24704);
SM120_CHECK_STORAGE(128, 64, 32, 3, 36992);
SM120_CHECK_STORAGE(128, 64, 64, 2, 49280);
SM120_CHECK_STORAGE(128, 64, 64, 3, 73856);
SM120_CHECK_STORAGE(64, 128, 32, 2, 24704);
SM120_CHECK_STORAGE(64, 128, 32, 3, 36992);
SM120_CHECK_STORAGE(64, 128, 64, 2, 49280);
SM120_CHECK_STORAGE(64, 128, 64, 3, 73856);
SM120_CHECK_STORAGE(128, 128, 32, 2, 32896);
SM120_CHECK_STORAGE(128, 128, 32, 3, 49280);
SM120_CHECK_STORAGE(128, 128, 64, 2, 65664);
SM120_CHECK_STORAGE(128, 128, 64, 3, 98432);

template <int Arrivals>
static __device__ __forceinline__ void sm120_init_barrier(unsigned barrier) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                 :: "r"(barrier), "n"(Arrivals) : "memory");
}

static __device__ __forceinline__ void sm120_wait_barrier(
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

template <int Bytes>
static __device__ __forceinline__ void sm120_expect_transaction(
    unsigned barrier) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 "
        "_, [%0], %1;"
        :: "r"(barrier), "n"(Bytes) : "memory");
}

static __device__ __forceinline__ void sm120_arrive_empty(unsigned barrier) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
                 :: "r"(barrier) : "memory");
}

static __device__ __forceinline__ void sm120_sync_warp() {
    __syncwarp();
}

static __device__ __forceinline__ void sm120_tma_copy(
    unsigned destination, unsigned long long map, int x, int y,
    int origin_x, int origin_y, unsigned barrier) {
    int map_x = x + origin_x;
    int map_y = y + origin_y;
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];"
        :: "r"(destination), "l"(map), "r"(map_x), "r"(map_y),
           "r"(barrier)
        : "memory");
}

template <int Groups>
static __device__ __forceinline__ unsigned sm120_swizzled_address_impl(
    unsigned shared_address, unsigned logical_row, unsigned element) {
    constexpr unsigned groups = Groups;
    unsigned offset = (shared_address / 128U) % groups;
    unsigned logical_chunk = element / 8U;
    unsigned physical_chunk =
        logical_chunk ^ ((logical_row + offset) % groups);
    return shared_address + logical_row * groups * 16U +
           physical_chunk * 16U;
}

template <int BK>
static __device__ __forceinline__ unsigned sm120_swizzled_address(
    unsigned shared_address, unsigned logical_row, unsigned element) {
    if constexpr (BK == 32) {
        constexpr unsigned groups = 4;
        static_assert(groups == SM120_SWIZZLE_64B,
                      "SM120 SW64 group count changed");
        return sm120_swizzled_address_impl<SM120_SWIZZLE_64B>(
            shared_address, logical_row, element);
    } else {
        constexpr unsigned groups = 8;
        static_assert(groups == SM120_SWIZZLE_128B,
                      "SM120 SW128 group count changed");
        return sm120_swizzled_address_impl<SM120_SWIZZLE_128B>(
            shared_address, logical_row, element);
    }
}

static __device__ __forceinline__ void sm120_load_x4(
    unsigned address, unsigned (&fragment)[4]) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(fragment[0]), "=r"(fragment[1]), "=r"(fragment[2]),
          "=r"(fragment[3])
        : "r"(address) : "memory");
}

static __device__ __forceinline__ void sm120_load_x4_transpose(
    unsigned address, unsigned (&fragment)[4]) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(fragment[0]), "=r"(fragment[1]), "=r"(fragment[2]),
          "=r"(fragment[3])
        : "r"(address) : "memory");
}

static __device__ __forceinline__ void sm120_load_x2(
    unsigned address, unsigned (&fragment)[2]) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
        : "=r"(fragment[0]), "=r"(fragment[1])
        : "r"(address) : "memory");
}

static __device__ __forceinline__ void sm120_load_x2_transpose(
    unsigned address, unsigned (&fragment)[2]) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
        : "=r"(fragment[0]), "=r"(fragment[1])
        : "r"(address) : "memory");
}

template <typename T>
struct Sm120Mma;

template <>
struct Sm120Mma<__half> {
    static __device__ __forceinline__ void issue(
        float (&accumulator)[4], const unsigned (&a)[4],
        const unsigned (&b)[2]) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
            "{%0, %1, %2, %3};"
            : "+f"(accumulator[0]), "+f"(accumulator[1]),
              "+f"(accumulator[2]), "+f"(accumulator[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "r"(b[0]), "r"(b[1]) : "memory");
    }
};

template <>
struct Sm120Mma<__nv_bfloat16> {
    static __device__ __forceinline__ void issue(
        float (&accumulator)[4], const unsigned (&a)[4],
        const unsigned (&b)[2]) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
            "{%0, %1, %2, %3};"
            : "+f"(accumulator[0]), "+f"(accumulator[1]),
              "+f"(accumulator[2]), "+f"(accumulator[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "r"(b[0]), "r"(b[1]) : "memory");
    }
};

struct Sm120Pipeline {
    unsigned payload;
    unsigned full;
    unsigned empty;
    int output_row;
    int output_col;
};

template <int Op, int M, int N, int BK, int Stages>
static __device__ __forceinline__ void sm120_produce_stage(
    const CUtensorMap& a_map, const CUtensorMap& b_map,
    const Sm120KernelParams& params, const Sm120Pipeline& pipeline,
    int tile) {
    constexpr int plane_bytes = BK * BK * 2;
    constexpr int a_bytes = M * BK * 2;
    constexpr int stage_bytes = Sm120Storage<M, N, BK, Stages>::stage_bytes;
    unsigned stage = pipeline.payload + (tile % Stages) * stage_bytes;
    unsigned a_destination = stage;
    unsigned b_destination = stage + a_bytes;
    unsigned barrier = pipeline.full + (tile % Stages) * 8;
    unsigned long long a_descriptor =
        reinterpret_cast<unsigned long long>(&a_map);
    unsigned long long b_descriptor =
        reinterpret_cast<unsigned long long>(&b_map);
    int reduction = tile * BK;

    sm120_expect_transaction<stage_bytes>(barrier);
    if constexpr (Op == Sm120Nn) {
        sm120_tma_copy(a_destination, a_descriptor, reduction,
                       pipeline.output_row, params.a_x, params.a_y, barrier);
#pragma unroll
        for (int plane = 0; plane < N / BK; ++plane) {
            int column = pipeline.output_col + plane * BK;
            sm120_tma_copy(b_destination + plane * plane_bytes, b_descriptor,
                           column, reduction, params.b_x, params.b_y, barrier);
        }
    } else if constexpr (Op == Sm120Tn) {
#pragma unroll
        for (int plane = 0; plane < M / BK; ++plane) {
            int row = pipeline.output_row + plane * BK;
            sm120_tma_copy(a_destination + plane * plane_bytes, a_descriptor,
                           row, reduction, params.a_x, params.a_y, barrier);
        }
#pragma unroll
        for (int plane = 0; plane < N / BK; ++plane) {
            int column = pipeline.output_col + plane * BK;
            sm120_tma_copy(b_destination + plane * plane_bytes, b_descriptor,
                           column, reduction, params.b_x, params.b_y, barrier);
        }
    } else {
        sm120_tma_copy(a_destination, a_descriptor, reduction,
                       pipeline.output_row, params.a_x, params.a_y, barrier);
        sm120_tma_copy(b_destination, b_descriptor, reduction,
                       pipeline.output_col, params.b_x, params.b_y, barrier);
    }
}

template <int Op, int M, int BK>
static __device__ __forceinline__ void sm120_load_a_fragments(
    unsigned stage, int warp_m, int slab,
    unsigned (&a_fragment)[2][4]) {
    constexpr int plane_bytes = BK * BK * 2;
    unsigned a_base = stage;
    int lane = threadIdx.x & 31;
    int row8 = lane & 7;
    int quadrant = lane >> 3;
    int k0 = slab * 16;

#pragma unroll
    for (int fm = 0; fm < 2; ++fm) {
        if constexpr (Op == Sm120Tn) {
            int logical_row = k0 + ((quadrant & 2) ? 8 : 0) + row8;
            int output_element =
                warp_m + fm * 16 + ((quadrant & 1) ? 8 : 0);
            unsigned plane = a_base +
                static_cast<unsigned>((output_element / BK) * plane_bytes);
            unsigned address = sm120_swizzled_address<BK>(
                plane, static_cast<unsigned>(logical_row),
                static_cast<unsigned>(output_element % BK));
            sm120_load_x4_transpose(address, a_fragment[fm]);
        } else {
            int logical_row =
                warp_m + fm * 16 + ((quadrant & 1) ? 8 : 0) + row8;
            int element = k0 + ((quadrant & 2) ? 8 : 0);
            unsigned address = sm120_swizzled_address<BK>(
                a_base, static_cast<unsigned>(logical_row),
                static_cast<unsigned>(element));
            sm120_load_x4(address, a_fragment[fm]);
        }
    }
}

template <int Op, int M, int BK>
static __device__ __forceinline__ void sm120_load_b_fragment(
    unsigned stage, int warp_n, int slab, int fn,
    unsigned (&b_fragment)[2]) {
    constexpr int plane_bytes = BK * BK * 2;
    constexpr int a_bytes = M * BK * 2;
    unsigned b_base = stage + a_bytes;
    int lane = threadIdx.x & 31;
    int row8 = lane & 7;
    int quadrant = lane >> 3;
    int k0 = slab * 16;

    if constexpr (Op == Sm120Nt) {
        int logical_row = warp_n + fn * 8 + row8;
        int element = k0 + ((quadrant & 1) ? 8 : 0);
        unsigned address = sm120_swizzled_address<BK>(
            b_base, static_cast<unsigned>(logical_row),
            static_cast<unsigned>(element));
        sm120_load_x2(address, b_fragment);
    } else {
        int logical_row = k0 + ((quadrant & 1) ? 8 : 0) + row8;
        int output_element = warp_n + fn * 8;
        unsigned plane = b_base +
            static_cast<unsigned>((output_element / BK) * plane_bytes);
        unsigned address = sm120_swizzled_address<BK>(
            plane, static_cast<unsigned>(logical_row),
            static_cast<unsigned>(output_element % BK));
        sm120_load_x2_transpose(address, b_fragment);
    }
}

template <typename T, int Op, int M, int N, int BK, int Stages>
static __device__ __forceinline__ void sm120_issue_stage(
    unsigned payload, int tile, int warp_m, int warp_n,
    float (&accumulator)[2][4][4]) {
    constexpr int stage_bytes = Sm120Storage<M, N, BK, Stages>::stage_bytes;
    unsigned stage = payload + (tile % Stages) * stage_bytes;
#pragma unroll
    for (int slab = 0; slab < BK / 16; ++slab) {
        unsigned a_fragment[2][4];
        sm120_load_a_fragments<Op, M, BK>(
            stage, warp_m, slab, a_fragment);
#pragma unroll
        for (int fn = 0; fn < 4; ++fn) {
            unsigned b_fragment[2];
            sm120_load_b_fragment<Op, M, BK>(
                stage, warp_n, slab, fn, b_fragment);
#pragma unroll
            for (int fm = 0; fm < 2; ++fm) {
                Sm120Mma<T>::issue(
                    accumulator[fm][fn], a_fragment[fm], b_fragment);
            }
        }
    }
}

template <typename T>
static __device__ __forceinline__ T sm120_from_float(float value);

template <>
__device__ __forceinline__ __half sm120_from_float(float value) {
    return from_f_f16(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 sm120_from_float(float value) {
    return from_f_bf16(value);
}

struct Sm120Output {
    void* pointer;
    float alpha;
    float beta;
    int rows;
    int columns;
    int stride;
    int row_tile;
    int column_tile;
    int warp_columns;
};

template <typename T, int Op>
static __device__ __forceinline__ void sm120_store_pair(
    const Sm120Output& output, int row, int column,
    float first, float second) {
    long long offset = static_cast<long long>(row) * output.stride + column;
    if constexpr (Op == Sm120Tn) {
        float* destination = static_cast<float*>(output.pointer) + offset;
        float v0 = __fmaf_rn(output.alpha, first, destination[0]);
        float v1 = __fmaf_rn(output.alpha, second, destination[1]);
        if ((reinterpret_cast<unsigned long long>(destination) & 7ULL) == 0) {
            float2 pair = {v0, v1};
            *reinterpret_cast<float2*>(destination) = pair;
        } else {
            destination[0] = v0;
            destination[1] = v1;
        }
    } else {
        T* destination = static_cast<T*>(output.pointer) + offset;
        float v0 = __fmul_rn(output.alpha, first);
        float v1 = __fmul_rn(output.alpha, second);
        if constexpr (Op == Sm120Nn) {
            if (output.beta != 0.0f) {
                v0 = __fmaf_rn(output.beta, to_f(destination[0]), v0);
                v1 = __fmaf_rn(output.beta, to_f(destination[1]), v1);
            }
        }
        if ((reinterpret_cast<unsigned long long>(destination) & 3ULL) == 0) {
            sgb_store_pair_rne(destination, v0, v1);
        } else {
            destination[0] = sm120_from_float<T>(v0);
            destination[1] = sm120_from_float<T>(v1);
        }
    }
}

template <typename T, int Op>
static __device__ __forceinline__ void sm120_store_scalar(
    const Sm120Output& output, int row, int column, float accumulator) {
    long long offset = static_cast<long long>(row) * output.stride + column;
    if constexpr (Op == Sm120Tn) {
        float* destination = static_cast<float*>(output.pointer) + offset;
        destination[0] =
            __fmaf_rn(output.alpha, accumulator, destination[0]);
    } else {
        T* destination = static_cast<T*>(output.pointer) + offset;
        float value = __fmul_rn(output.alpha, accumulator);
        if constexpr (Op == Sm120Nn) {
            if (output.beta != 0.0f) {
                value = __fmaf_rn(output.beta, to_f(destination[0]), value);
            }
        }
        destination[0] = sm120_from_float<T>(value);
    }
}

template <typename T, int Op>
static __device__ __forceinline__ void sm120_epilogue(
    const Sm120Output& output, const float (&accumulator)[2][4][4]) {
    int lane = threadIdx.x & 31;
    int group = lane >> 2;
    int pair = lane & 3;
    int warp = threadIdx.x >> 5;

#pragma unroll
    for (int fm = 0; fm < 2; ++fm) {
#pragma unroll
        for (int fn = 0; fn < 4; ++fn) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                int row = output.row_tile + (warp / output.warp_columns) * 32 +
                          fm * 16 + group + half * 8;
                int column = output.column_tile +
                             (warp % output.warp_columns) * 32 +
                             fn * 8 + pair * 2;
                if (row >= output.rows || column >= output.columns) continue;
                int element = half * 2;
                if (column + 1 < output.columns) {
                    sm120_store_pair<T, Op>(output, row, column,
                        accumulator[fm][fn][element],
                        accumulator[fm][fn][element + 1]);
                } else {
                    sm120_store_scalar<T, Op>(
                        output, row, column, accumulator[fm][fn][element]);
                }
            }
        }
    }
}

template <int Op>
static __device__ __forceinline__ void sm120_initialize_accumulator(
    float (&accumulator)[2][4][4], const float* bias,
    int output_columns, int output_col, int warp_n) {
    int pair = threadIdx.x & 3;
#pragma unroll
    for (int fm = 0; fm < 2; ++fm) {
#pragma unroll
        for (int fn = 0; fn < 4; ++fn) {
            float first = 0.0f;
            float second = 0.0f;
            if constexpr (Op == Sm120Nn) {
                if (bias != nullptr) {
                    int column = output_col + warp_n + fn * 8 + pair * 2;
                    if (column < output_columns) first = bias[column];
                    if (column + 1 < output_columns) second = bias[column + 1];
                }
            }
            accumulator[fm][fn][0] = first;
            accumulator[fm][fn][1] = second;
            accumulator[fm][fn][2] = first;
            accumulator[fm][fn][3] = second;
        }
    }
}

template <typename T, int Op, int M, int N, int BK, int Stages>
static __device__ __forceinline__ void sm120_kernel(
    void* output, const CUtensorMap& a_map, const CUtensorMap& b_map,
    const float* bias, const Sm120KernelParams& params) {
    constexpr int threads = Sm120Storage<M, N, BK, Stages>::threads;
    extern __shared__ __align__(128) unsigned char storage[];
    unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(storage));
    int output_rows = Op == Sm120Tn ? params.k : params.m;
    int output_columns = Op == Sm120Nt ? params.k : params.n;
    int reduction =
        Op == Sm120Nn ? params.k : (Op == Sm120Tn ? params.m : params.n);
    int column_tiles = 1 + (output_columns - 1) / N;
    int output_row = (blockIdx.x / column_tiles) * M;
    int output_col = (blockIdx.x % column_tiles) * N;
    int tile_count = 1 + (reduction - 1) / BK;
    Sm120Pipeline pipeline = {
        shared + 128,
        shared,
        shared + 64,
        output_row,
        output_col,
    };
    int warp = threadIdx.x >> 5;

    if (threadIdx.x == 0) {
#pragma unroll
        for (int stage = 0; stage < Stages; ++stage) {
            sm120_init_barrier<1>(pipeline.full + stage * 8);
            sm120_init_barrier<threads>(pipeline.empty + stage * 8);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    if (warp == 0 && (threadIdx.x & 31) == 0) {
#pragma unroll
        for (int tile = 0; tile < Stages; ++tile) {
            if (tile < tile_count) {
                sm120_produce_stage<Op, M, N, BK, Stages>(
                    a_map, b_map, params, pipeline, tile);
            }
        }
    }
    sm120_sync_warp();

    int warp_columns = N / 32;
    int warp_m = (warp / warp_columns) * 32;
    int warp_n = (warp % warp_columns) * 32;
    float accumulator[2][4][4];
    sm120_initialize_accumulator<Op>(
        accumulator, bias, output_columns, output_col, warp_n);
    for (int tile = 0; tile < tile_count; ++tile) {
        int stage = tile % Stages;
        unsigned generation = static_cast<unsigned>(tile / Stages);
        sm120_wait_barrier(pipeline.full + stage * 8, generation & 1U);
        sm120_issue_stage<T, Op, M, N, BK, Stages>(
            pipeline.payload, tile, warp_m, warp_n, accumulator);
        sm120_arrive_empty(pipeline.empty + stage * 8);
        if (warp == 0 && (threadIdx.x & 31) == 0) {
            int refill = tile + Stages;
            if (refill < tile_count) {
                sm120_wait_barrier(
                    pipeline.empty + stage * 8, generation & 1U);
                sm120_produce_stage<Op, M, N, BK, Stages>(
                    a_map, b_map, params, pipeline, refill);
            }
        }
        sm120_sync_warp();
    }

    Sm120Output destination = {
        output,
        params.alpha,
        params.beta,
        output_rows,
        output_columns,
        params.ldc,
        output_row,
        output_col,
        warp_columns,
    };
    sm120_epilogue<T, Op>(destination, accumulator);
}

#define SM120_DEFINE_KERNEL(NAME, TYPE, OP, M, N, BK, STAGES)                \
    extern "C" __global__ __launch_bounds__((M * N) / 32)                  \
    void NAME(void* output,                                                   \
              const __grid_constant__ CUtensorMap a_map,                     \
              const __grid_constant__ CUtensorMap b_map,                     \
              const float* bias,                                              \
              const __grid_constant__ Sm120KernelParams params) {            \
        sm120_kernel<TYPE, OP, M, N, BK, STAGES>(                             \
            output, a_map, b_map, bias, params);                              \
    }

SM120_DEFINE_KERNEL(sgemm_bi_nn_sm120_tma_64x64_bk32_s2_bf16, __nv_bfloat16, Sm120Nn, 64, 64, 32, 2)
SM120_DEFINE_KERNEL(sgemm_bi_nn_sm120_tma_64x64_bk32_s2_f16, __half, Sm120Nn, 64, 64, 32, 2)
SM120_DEFINE_KERNEL(sgemm_bi_nn_sm120_tma_64x64_bk32_s3_bf16, __nv_bfloat16, Sm120Nn, 64, 64, 32, 3)
SM120_DEFINE_KERNEL(sgemm_bi_nn_sm120_tma_64x64_bk32_s3_f16, __half, Sm120Nn, 64, 64, 32, 3)
SM120_DEFINE_KERNEL(sgemm_bi_nn_sm120_tma_64x64_bk64_s2_bf16, __nv_bfloat16, Sm120Nn, 64, 64, 64, 2)
SM120_DEFINE_KERNEL(sgemm_bi_nn_sm120_tma_64x64_bk64_s2_f16, __half, Sm120Nn, 64, 64, 64, 2)
SM120_DEFINE_KERNEL(sgemm_bi_nn_sm120_tma_64x64_bk64_s3_bf16, __nv_bfloat16, Sm120Nn, 64, 64, 64, 3)
SM120_DEFINE_KERNEL(sgemm_bi_nn_sm120_tma_64x64_bk64_s3_f16, __half, Sm120Nn, 64, 64, 64, 3)
SM120_DEFINE_KERNEL(sgemm_bi_nn_sm120_tma_128x64_bk32_s2_bf16, __nv_bfloat16, Sm120Nn, 128, 64, 32, 2)
SM120_DEFINE_KERNEL(sgemm_bi_nn_sm120_tma_128x64_bk32_s2_f16, __half, Sm120Nn, 128, 64, 32, 2)
SM120_DEFINE_KERNEL(sgemm_bi_nn_sm120_tma_128x64_bk32_s3_bf16, __nv_bfloat16, Sm120Nn, 128, 64, 32, 3)
SM120_DEFINE_KERNEL(sgemm_bi_nn_sm120_tma_128x64_bk32_s3_f16, __half, Sm120Nn, 128, 64, 32, 3)
SM120_DEFINE_KERNEL(sgemm_bi_nn_sm120_tma_128x64_bk64_s2_bf16, __nv_bfloat16, Sm120Nn, 128, 64, 64, 2)
SM120_DEFINE_KERNEL(sgemm_bi_nn_sm120_tma_128x64_bk64_s2_f16, __half, Sm120Nn, 128, 64, 64, 2)
SM120_DEFINE_KERNEL(sgemm_bi_nn_sm120_tma_128x64_bk64_s3_bf16, __nv_bfloat16, Sm120Nn, 128, 64, 64, 3)
SM120_DEFINE_KERNEL(sgemm_bi_nn_sm120_tma_128x64_bk64_s3_f16, __half, Sm120Nn, 128, 64, 64, 3)
SM120_DEFINE_KERNEL(sgemm_bi_nn_sm120_tma_64x128_bk32_s2_bf16, __nv_bfloat16, Sm120Nn, 64, 128, 32, 2)
SM120_DEFINE_KERNEL(sgemm_bi_nn_sm120_tma_64x128_bk32_s2_f16, __half, Sm120Nn, 64, 128, 32, 2)
SM120_DEFINE_KERNEL(sgemm_bi_nn_sm120_tma_64x128_bk32_s3_bf16, __nv_bfloat16, Sm120Nn, 64, 128, 32, 3)
SM120_DEFINE_KERNEL(sgemm_bi_nn_sm120_tma_64x128_bk32_s3_f16, __half, Sm120Nn, 64, 128, 32, 3)
SM120_DEFINE_KERNEL(sgemm_bi_nn_sm120_tma_64x128_bk64_s2_bf16, __nv_bfloat16, Sm120Nn, 64, 128, 64, 2)
SM120_DEFINE_KERNEL(sgemm_bi_nn_sm120_tma_64x128_bk64_s2_f16, __half, Sm120Nn, 64, 128, 64, 2)
SM120_DEFINE_KERNEL(sgemm_bi_nn_sm120_tma_64x128_bk64_s3_bf16, __nv_bfloat16, Sm120Nn, 64, 128, 64, 3)
SM120_DEFINE_KERNEL(sgemm_bi_nn_sm120_tma_64x128_bk64_s3_f16, __half, Sm120Nn, 64, 128, 64, 3)
SM120_DEFINE_KERNEL(sgemm_bi_nn_sm120_tma_128x128_bk32_s2_bf16, __nv_bfloat16, Sm120Nn, 128, 128, 32, 2)
SM120_DEFINE_KERNEL(sgemm_bi_nn_sm120_tma_128x128_bk32_s2_f16, __half, Sm120Nn, 128, 128, 32, 2)
SM120_DEFINE_KERNEL(sgemm_bi_nn_sm120_tma_128x128_bk32_s3_bf16, __nv_bfloat16, Sm120Nn, 128, 128, 32, 3)
SM120_DEFINE_KERNEL(sgemm_bi_nn_sm120_tma_128x128_bk32_s3_f16, __half, Sm120Nn, 128, 128, 32, 3)
SM120_DEFINE_KERNEL(sgemm_bi_nn_sm120_tma_128x128_bk64_s2_bf16, __nv_bfloat16, Sm120Nn, 128, 128, 64, 2)
SM120_DEFINE_KERNEL(sgemm_bi_nn_sm120_tma_128x128_bk64_s2_f16, __half, Sm120Nn, 128, 128, 64, 2)
SM120_DEFINE_KERNEL(sgemm_bi_nn_sm120_tma_128x128_bk64_s3_bf16, __nv_bfloat16, Sm120Nn, 128, 128, 64, 3)
SM120_DEFINE_KERNEL(sgemm_bi_nn_sm120_tma_128x128_bk64_s3_f16, __half, Sm120Nn, 128, 128, 64, 3)

SM120_DEFINE_KERNEL(sgemm_bi_tn_sm120_tma_64x64_bk32_s2_bf16, __nv_bfloat16, Sm120Tn, 64, 64, 32, 2)
SM120_DEFINE_KERNEL(sgemm_bi_tn_sm120_tma_64x64_bk32_s2_f16, __half, Sm120Tn, 64, 64, 32, 2)
SM120_DEFINE_KERNEL(sgemm_bi_tn_sm120_tma_64x64_bk32_s3_bf16, __nv_bfloat16, Sm120Tn, 64, 64, 32, 3)
SM120_DEFINE_KERNEL(sgemm_bi_tn_sm120_tma_64x64_bk32_s3_f16, __half, Sm120Tn, 64, 64, 32, 3)
SM120_DEFINE_KERNEL(sgemm_bi_tn_sm120_tma_64x64_bk64_s2_bf16, __nv_bfloat16, Sm120Tn, 64, 64, 64, 2)
SM120_DEFINE_KERNEL(sgemm_bi_tn_sm120_tma_64x64_bk64_s2_f16, __half, Sm120Tn, 64, 64, 64, 2)
SM120_DEFINE_KERNEL(sgemm_bi_tn_sm120_tma_64x64_bk64_s3_bf16, __nv_bfloat16, Sm120Tn, 64, 64, 64, 3)
SM120_DEFINE_KERNEL(sgemm_bi_tn_sm120_tma_64x64_bk64_s3_f16, __half, Sm120Tn, 64, 64, 64, 3)
SM120_DEFINE_KERNEL(sgemm_bi_tn_sm120_tma_128x64_bk32_s2_bf16, __nv_bfloat16, Sm120Tn, 128, 64, 32, 2)
SM120_DEFINE_KERNEL(sgemm_bi_tn_sm120_tma_128x64_bk32_s2_f16, __half, Sm120Tn, 128, 64, 32, 2)
SM120_DEFINE_KERNEL(sgemm_bi_tn_sm120_tma_128x64_bk32_s3_bf16, __nv_bfloat16, Sm120Tn, 128, 64, 32, 3)
SM120_DEFINE_KERNEL(sgemm_bi_tn_sm120_tma_128x64_bk32_s3_f16, __half, Sm120Tn, 128, 64, 32, 3)
SM120_DEFINE_KERNEL(sgemm_bi_tn_sm120_tma_128x64_bk64_s2_bf16, __nv_bfloat16, Sm120Tn, 128, 64, 64, 2)
SM120_DEFINE_KERNEL(sgemm_bi_tn_sm120_tma_128x64_bk64_s2_f16, __half, Sm120Tn, 128, 64, 64, 2)
SM120_DEFINE_KERNEL(sgemm_bi_tn_sm120_tma_128x64_bk64_s3_bf16, __nv_bfloat16, Sm120Tn, 128, 64, 64, 3)
SM120_DEFINE_KERNEL(sgemm_bi_tn_sm120_tma_128x64_bk64_s3_f16, __half, Sm120Tn, 128, 64, 64, 3)
SM120_DEFINE_KERNEL(sgemm_bi_tn_sm120_tma_64x128_bk32_s2_bf16, __nv_bfloat16, Sm120Tn, 64, 128, 32, 2)
SM120_DEFINE_KERNEL(sgemm_bi_tn_sm120_tma_64x128_bk32_s2_f16, __half, Sm120Tn, 64, 128, 32, 2)
SM120_DEFINE_KERNEL(sgemm_bi_tn_sm120_tma_64x128_bk32_s3_bf16, __nv_bfloat16, Sm120Tn, 64, 128, 32, 3)
SM120_DEFINE_KERNEL(sgemm_bi_tn_sm120_tma_64x128_bk32_s3_f16, __half, Sm120Tn, 64, 128, 32, 3)
SM120_DEFINE_KERNEL(sgemm_bi_tn_sm120_tma_64x128_bk64_s2_bf16, __nv_bfloat16, Sm120Tn, 64, 128, 64, 2)
SM120_DEFINE_KERNEL(sgemm_bi_tn_sm120_tma_64x128_bk64_s2_f16, __half, Sm120Tn, 64, 128, 64, 2)
SM120_DEFINE_KERNEL(sgemm_bi_tn_sm120_tma_64x128_bk64_s3_bf16, __nv_bfloat16, Sm120Tn, 64, 128, 64, 3)
SM120_DEFINE_KERNEL(sgemm_bi_tn_sm120_tma_64x128_bk64_s3_f16, __half, Sm120Tn, 64, 128, 64, 3)
SM120_DEFINE_KERNEL(sgemm_bi_tn_sm120_tma_128x128_bk32_s2_bf16, __nv_bfloat16, Sm120Tn, 128, 128, 32, 2)
SM120_DEFINE_KERNEL(sgemm_bi_tn_sm120_tma_128x128_bk32_s2_f16, __half, Sm120Tn, 128, 128, 32, 2)
SM120_DEFINE_KERNEL(sgemm_bi_tn_sm120_tma_128x128_bk32_s3_bf16, __nv_bfloat16, Sm120Tn, 128, 128, 32, 3)
SM120_DEFINE_KERNEL(sgemm_bi_tn_sm120_tma_128x128_bk32_s3_f16, __half, Sm120Tn, 128, 128, 32, 3)
SM120_DEFINE_KERNEL(sgemm_bi_tn_sm120_tma_128x128_bk64_s2_bf16, __nv_bfloat16, Sm120Tn, 128, 128, 64, 2)
SM120_DEFINE_KERNEL(sgemm_bi_tn_sm120_tma_128x128_bk64_s2_f16, __half, Sm120Tn, 128, 128, 64, 2)
SM120_DEFINE_KERNEL(sgemm_bi_tn_sm120_tma_128x128_bk64_s3_bf16, __nv_bfloat16, Sm120Tn, 128, 128, 64, 3)
SM120_DEFINE_KERNEL(sgemm_bi_tn_sm120_tma_128x128_bk64_s3_f16, __half, Sm120Tn, 128, 128, 64, 3)

SM120_DEFINE_KERNEL(sgemm_bi_nt_sm120_tma_64x64_bk32_s2_bf16, __nv_bfloat16, Sm120Nt, 64, 64, 32, 2)
SM120_DEFINE_KERNEL(sgemm_bi_nt_sm120_tma_64x64_bk32_s2_f16, __half, Sm120Nt, 64, 64, 32, 2)
SM120_DEFINE_KERNEL(sgemm_bi_nt_sm120_tma_64x64_bk32_s3_bf16, __nv_bfloat16, Sm120Nt, 64, 64, 32, 3)
SM120_DEFINE_KERNEL(sgemm_bi_nt_sm120_tma_64x64_bk32_s3_f16, __half, Sm120Nt, 64, 64, 32, 3)
SM120_DEFINE_KERNEL(sgemm_bi_nt_sm120_tma_64x64_bk64_s2_bf16, __nv_bfloat16, Sm120Nt, 64, 64, 64, 2)
SM120_DEFINE_KERNEL(sgemm_bi_nt_sm120_tma_64x64_bk64_s2_f16, __half, Sm120Nt, 64, 64, 64, 2)
SM120_DEFINE_KERNEL(sgemm_bi_nt_sm120_tma_64x64_bk64_s3_bf16, __nv_bfloat16, Sm120Nt, 64, 64, 64, 3)
SM120_DEFINE_KERNEL(sgemm_bi_nt_sm120_tma_64x64_bk64_s3_f16, __half, Sm120Nt, 64, 64, 64, 3)
SM120_DEFINE_KERNEL(sgemm_bi_nt_sm120_tma_128x64_bk32_s2_bf16, __nv_bfloat16, Sm120Nt, 128, 64, 32, 2)
SM120_DEFINE_KERNEL(sgemm_bi_nt_sm120_tma_128x64_bk32_s2_f16, __half, Sm120Nt, 128, 64, 32, 2)
SM120_DEFINE_KERNEL(sgemm_bi_nt_sm120_tma_128x64_bk32_s3_bf16, __nv_bfloat16, Sm120Nt, 128, 64, 32, 3)
SM120_DEFINE_KERNEL(sgemm_bi_nt_sm120_tma_128x64_bk32_s3_f16, __half, Sm120Nt, 128, 64, 32, 3)
SM120_DEFINE_KERNEL(sgemm_bi_nt_sm120_tma_128x64_bk64_s2_bf16, __nv_bfloat16, Sm120Nt, 128, 64, 64, 2)
SM120_DEFINE_KERNEL(sgemm_bi_nt_sm120_tma_128x64_bk64_s2_f16, __half, Sm120Nt, 128, 64, 64, 2)
SM120_DEFINE_KERNEL(sgemm_bi_nt_sm120_tma_128x64_bk64_s3_bf16, __nv_bfloat16, Sm120Nt, 128, 64, 64, 3)
SM120_DEFINE_KERNEL(sgemm_bi_nt_sm120_tma_128x64_bk64_s3_f16, __half, Sm120Nt, 128, 64, 64, 3)
SM120_DEFINE_KERNEL(sgemm_bi_nt_sm120_tma_64x128_bk32_s2_bf16, __nv_bfloat16, Sm120Nt, 64, 128, 32, 2)
SM120_DEFINE_KERNEL(sgemm_bi_nt_sm120_tma_64x128_bk32_s2_f16, __half, Sm120Nt, 64, 128, 32, 2)
SM120_DEFINE_KERNEL(sgemm_bi_nt_sm120_tma_64x128_bk32_s3_bf16, __nv_bfloat16, Sm120Nt, 64, 128, 32, 3)
SM120_DEFINE_KERNEL(sgemm_bi_nt_sm120_tma_64x128_bk32_s3_f16, __half, Sm120Nt, 64, 128, 32, 3)
SM120_DEFINE_KERNEL(sgemm_bi_nt_sm120_tma_64x128_bk64_s2_bf16, __nv_bfloat16, Sm120Nt, 64, 128, 64, 2)
SM120_DEFINE_KERNEL(sgemm_bi_nt_sm120_tma_64x128_bk64_s2_f16, __half, Sm120Nt, 64, 128, 64, 2)
SM120_DEFINE_KERNEL(sgemm_bi_nt_sm120_tma_64x128_bk64_s3_bf16, __nv_bfloat16, Sm120Nt, 64, 128, 64, 3)
SM120_DEFINE_KERNEL(sgemm_bi_nt_sm120_tma_64x128_bk64_s3_f16, __half, Sm120Nt, 64, 128, 64, 3)
SM120_DEFINE_KERNEL(sgemm_bi_nt_sm120_tma_128x128_bk32_s2_bf16, __nv_bfloat16, Sm120Nt, 128, 128, 32, 2)
SM120_DEFINE_KERNEL(sgemm_bi_nt_sm120_tma_128x128_bk32_s2_f16, __half, Sm120Nt, 128, 128, 32, 2)
SM120_DEFINE_KERNEL(sgemm_bi_nt_sm120_tma_128x128_bk32_s3_bf16, __nv_bfloat16, Sm120Nt, 128, 128, 32, 3)
SM120_DEFINE_KERNEL(sgemm_bi_nt_sm120_tma_128x128_bk32_s3_f16, __half, Sm120Nt, 128, 128, 32, 3)
SM120_DEFINE_KERNEL(sgemm_bi_nt_sm120_tma_128x128_bk64_s2_bf16, __nv_bfloat16, Sm120Nt, 128, 128, 64, 2)
SM120_DEFINE_KERNEL(sgemm_bi_nt_sm120_tma_128x128_bk64_s2_f16, __half, Sm120Nt, 128, 128, 64, 2)
SM120_DEFINE_KERNEL(sgemm_bi_nt_sm120_tma_128x128_bk64_s3_bf16, __nv_bfloat16, Sm120Nt, 128, 128, 64, 3)
SM120_DEFINE_KERNEL(sgemm_bi_nt_sm120_tma_128x128_bk64_s3_f16, __half, Sm120Nt, 128, 128, 64, 3)

#undef SM120_DEFINE_KERNEL
#undef SM120_CHECK_STORAGE
#undef SM120_SWIZZLE_128B
#undef SM120_SWIZZLE_64B

#endif
