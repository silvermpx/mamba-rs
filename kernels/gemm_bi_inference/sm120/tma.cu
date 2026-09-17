// Deterministic SM120 inference GEMM for homogeneous bf16/f16 operands.
// One CTA owns each output tile and walks K in ascending order. There is no
// split-K, inter-CTA reduction, or atomic update anywhere in this file.

#if defined(__CUDA_ARCH__) && \
    (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)

#if __CUDACC_VER_MAJOR__ >= 13
struct alignas(128) GbfSm120HalfTensorMap {
#else
struct alignas(64) GbfSm120HalfTensorMap {
#endif
    unsigned long long opaque[16];
};

static_assert(sizeof(GbfSm120HalfTensorMap) == 128,
              "Fixed SM120 half tensor-map size changed");
#if __CUDACC_VER_MAJOR__ >= 13
static_assert(alignof(GbfSm120HalfTensorMap) == 128,
              "Fixed SM120 half tensor-map alignment changed");
#else
static_assert(alignof(GbfSm120HalfTensorMap) == 64,
              "Fixed SM120 half tensor-map alignment changed");
#endif

struct GbfSm120HalfParams {
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

static_assert(sizeof(GbfSm120HalfParams) == 40,
              "Fixed SM120 half parameter size changed");
static_assert(alignof(GbfSm120HalfParams) == 4,
              "Fixed SM120 half parameter alignment changed");
static_assert(__is_standard_layout(GbfSm120HalfParams),
              "Fixed SM120 half parameters must remain standard layout");

#define GBF_SM120_HALF_SWIZZLE_64B 4
#define GBF_SM120_HALF_SWIZZLE_128B 8

template <int M, int N, int BK, int Stages>
struct GbfSm120HalfStorage {
    static constexpr bool wide_m_warp =
        M == 128 && N == 128 && BK == 32;
    static constexpr int compute_warps =
        wide_m_warp ? 8 : (M / 32) * (N / 32);
    static constexpr int threads = compute_warps * 32;
    static constexpr int stage_bytes = (M + N) * BK * 2;
    static constexpr int dynamic_bytes = 128 + Stages * stage_bytes;
};

#define GBF_SM120_HALF_CHECK_STORAGE(M, N, BK, STAGES, BYTES)                \
    static_assert(                                                           \
        GbfSm120HalfStorage<M, N, BK, STAGES>::dynamic_bytes == BYTES,       \
        "Fixed SM120 half dynamic shared size changed")

GBF_SM120_HALF_CHECK_STORAGE(64, 64, 32, 2, 16512);
GBF_SM120_HALF_CHECK_STORAGE(64, 64, 32, 3, 24704);
GBF_SM120_HALF_CHECK_STORAGE(64, 64, 64, 2, 32896);
GBF_SM120_HALF_CHECK_STORAGE(64, 64, 64, 3, 49280);
GBF_SM120_HALF_CHECK_STORAGE(128, 64, 32, 2, 24704);
GBF_SM120_HALF_CHECK_STORAGE(128, 64, 32, 3, 36992);
GBF_SM120_HALF_CHECK_STORAGE(128, 64, 64, 2, 49280);
GBF_SM120_HALF_CHECK_STORAGE(128, 64, 64, 3, 73856);
GBF_SM120_HALF_CHECK_STORAGE(64, 128, 32, 2, 24704);
GBF_SM120_HALF_CHECK_STORAGE(64, 128, 32, 3, 36992);
GBF_SM120_HALF_CHECK_STORAGE(64, 128, 64, 2, 49280);
GBF_SM120_HALF_CHECK_STORAGE(64, 128, 64, 3, 73856);
GBF_SM120_HALF_CHECK_STORAGE(128, 128, 32, 2, 32896);
GBF_SM120_HALF_CHECK_STORAGE(128, 128, 32, 3, 49280);
GBF_SM120_HALF_CHECK_STORAGE(128, 128, 64, 2, 65664);
GBF_SM120_HALF_CHECK_STORAGE(128, 128, 64, 3, 98432);

template <int Arrivals>
static __device__ __forceinline__ void gbf_sm120_half_init_barrier(
    unsigned barrier) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                 :: "r"(barrier), "n"(Arrivals) : "memory");
}

static __device__ __forceinline__ void gbf_sm120_half_wait_barrier(
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
static __device__ __forceinline__ void gbf_sm120_half_expect_transaction(
    unsigned barrier) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 "
        "_, [%0], %1;"
        :: "r"(barrier), "n"(Bytes) : "memory");
}

static __device__ __forceinline__ void gbf_sm120_half_arrive_empty(
    unsigned barrier) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
                 :: "r"(barrier) : "memory");
}

static __device__ __forceinline__ void gbf_sm120_half_tma_copy(
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

template <int Groups>
static __device__ __forceinline__ unsigned
gbf_sm120_half_swizzled_address_impl(
    unsigned shared_address, unsigned logical_row, unsigned element) {
    unsigned logical_chunk = element / 8U;
    unsigned row_start = shared_address + logical_row * Groups * 16U;
    unsigned phase = (row_start / 128U) % Groups;
    unsigned physical_chunk = logical_chunk ^ phase;
    return row_start + physical_chunk * 16U;
}

template <int BK>
static __device__ __forceinline__ unsigned gbf_sm120_half_swizzled_address(
    unsigned shared_address, unsigned logical_row, unsigned element) {
    if constexpr (BK == 32) {
        return gbf_sm120_half_swizzled_address_impl<
            GBF_SM120_HALF_SWIZZLE_64B>(
                shared_address, logical_row, element);
    } else {
        static_assert(BK == 64, "Fixed SM120 half BK must be 32 or 64");
        return gbf_sm120_half_swizzled_address_impl<
            GBF_SM120_HALF_SWIZZLE_128B>(
                shared_address, logical_row, element);
    }
}

static __device__ __forceinline__ void gbf_sm120_half_load_x4(
    unsigned address, unsigned (&fragment)[4]) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(fragment[0]), "=r"(fragment[1]), "=r"(fragment[2]),
          "=r"(fragment[3])
        : "r"(address) : "memory");
}

static __device__ __forceinline__ void gbf_sm120_half_load_x2_transpose(
    unsigned address, unsigned (&fragment)[2]) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
        : "=r"(fragment[0]), "=r"(fragment[1])
        : "r"(address) : "memory");
}

template <typename T>
struct GbfSm120HalfMma;

template <>
struct GbfSm120HalfMma<__half> {
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
struct GbfSm120HalfMma<__nv_bfloat16> {
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

struct GbfSm120HalfPipeline {
    unsigned payload;
    unsigned full;
    unsigned empty;
    int output_row;
    int output_column;
};

template <int M, int N, int BK, int Stages>
static __device__ __forceinline__ void gbf_sm120_half_produce_stage(
    const GbfSm120HalfTensorMap& a_map,
    const GbfSm120HalfTensorMap& b_map,
    const GbfSm120HalfParams& params,
    const GbfSm120HalfPipeline& pipeline,
    int tile) {
    constexpr int plane_bytes = BK * BK * 2;
    constexpr int a_bytes = M * BK * 2;
    constexpr int stage_bytes =
        GbfSm120HalfStorage<M, N, BK, Stages>::stage_bytes;
    unsigned stage = pipeline.payload + (tile % Stages) * stage_bytes;
    unsigned a_destination = stage;
    unsigned b_destination = stage + a_bytes;
    unsigned barrier = pipeline.full + (tile % Stages) * 8;
    unsigned long long a_descriptor =
        reinterpret_cast<unsigned long long>(&a_map);
    unsigned long long b_descriptor =
        reinterpret_cast<unsigned long long>(&b_map);
    int reduction = tile * BK;

    gbf_sm120_half_expect_transaction<stage_bytes>(barrier);
    gbf_sm120_half_tma_copy(
        a_destination, a_descriptor, reduction, pipeline.output_row,
        params.a_x, params.a_y, barrier);
    if constexpr (M == 128 && N == 64 && BK == 32) {
        gbf_sm120_half_tma_copy(
            b_destination, b_descriptor, pipeline.output_column, reduction,
            params.b_x, params.b_y, barrier);
    } else {
#pragma unroll
        for (int plane = 0; plane < N / BK; ++plane) {
            int column = pipeline.output_column + plane * BK;
            gbf_sm120_half_tma_copy(
                b_destination + plane * plane_bytes, b_descriptor,
                column, reduction, params.b_x, params.b_y, barrier);
        }
    }
}

template <int M, int N, int BK, int Stages, int MAtoms>
static __device__ __forceinline__ void gbf_sm120_half_load_a_fragments(
    unsigned stage, int warp_m, int slab,
    unsigned (&a_fragment)[MAtoms][4]) {
    constexpr bool compact_addresses =
        M == 128 && N == 128 && BK == 32 && Stages == 2;
    int lane = threadIdx.x & 31;
    int row8 = lane & 7;
    int quadrant = lane >> 3;
    int k0 = slab * 16;

#pragma unroll
    for (int fragment = 0; fragment < MAtoms; ++fragment) {
        int logical_row;
        int element;
        if constexpr (compact_addresses) {
            logical_row = warp_m + fragment * 16 + (lane & 15);
            element = k0 + ((lane & 16) >> 1);
        } else {
            logical_row = warp_m + fragment * 16
                + ((quadrant & 1) ? 8 : 0) + row8;
            element = k0 + ((quadrant & 2) ? 8 : 0);
        }
        unsigned address = gbf_sm120_half_swizzled_address<BK>(
            stage, static_cast<unsigned>(logical_row),
            static_cast<unsigned>(element));
        gbf_sm120_half_load_x4(address, a_fragment[fragment]);
    }
}

template <int M, int N, int BK, int Stages>
static __device__ __forceinline__ void gbf_sm120_half_load_b_fragment(
    unsigned stage, int warp_n, int slab, int fragment,
    unsigned (&b_fragment)[2]) {
    constexpr int plane_bytes = BK * BK * 2;
    constexpr int a_bytes = M * BK * 2;
    constexpr bool compact_addresses =
        (Stages == 2 &&
         ((M == 128 && N == 64) ||
          (M == 128 && N == 128 && BK == 64))) ||
        (Stages == 3 &&
         ((M == 64 && N == 64 && BK == 32) ||
          (M == 64 && N == 128) ||
          (M == 128 && N == 128)));
    unsigned b_base = stage + a_bytes;
    int lane = threadIdx.x & 31;
    int row8 = lane & 7;
    int k0 = slab * 16;
    int logical_row;
    if constexpr (compact_addresses) {
        logical_row = k0 + (lane & 15);
    } else {
        int quadrant = lane >> 3;
        logical_row = k0 + ((quadrant & 1) ? 8 : 0) + row8;
    }
    int output_element = warp_n + fragment * 8;
    unsigned address;
    if constexpr (M == 128 && N == 64 && BK == 32) {
        constexpr int wide_plane_bytes = 64 * BK * 2;
        unsigned plane = b_base + static_cast<unsigned>(
            (output_element / 64) * wide_plane_bytes);
        address = gbf_sm120_half_swizzled_address_impl<
            GBF_SM120_HALF_SWIZZLE_128B>(
                plane, static_cast<unsigned>(logical_row),
                static_cast<unsigned>(output_element % 64));
    } else {
        unsigned plane = b_base + static_cast<unsigned>(
            (output_element / BK) * plane_bytes);
        address = gbf_sm120_half_swizzled_address<BK>(
            plane, static_cast<unsigned>(logical_row),
            static_cast<unsigned>(output_element % BK));
    }
    gbf_sm120_half_load_x2_transpose(address, b_fragment);
}

template <typename T, int M, int N, int BK, int Stages,
          int MAtoms, bool RotatingStage>
static __device__ __forceinline__ void gbf_sm120_half_issue_stage(
    unsigned payload, int stage_or_tile, int warp_m, int warp_n,
    float (&accumulator)[MAtoms][4][4]) {
    constexpr int stage_bytes =
        GbfSm120HalfStorage<M, N, BK, Stages>::stage_bytes;
    constexpr bool lookahead_a = BK == 64 && Stages == 2 && M == 64;
    constexpr bool b_before_next_a = M == 64 && N == 64;
    int stage_index;
    if constexpr (RotatingStage) {
        stage_index = stage_or_tile;
    } else {
        stage_index = stage_or_tile % Stages;
    }
    unsigned stage = payload + stage_index * stage_bytes;
    unsigned a_fragment[lookahead_a ? 2 : 1][MAtoms][4];
    if constexpr (lookahead_a) {
        gbf_sm120_half_load_a_fragments<M, N, BK, Stages, MAtoms>(
            stage, warp_m, 0, a_fragment[0]);
    }

#pragma unroll
    for (int slab = 0; slab < BK / 16; ++slab) {
        int current_a = lookahead_a ? slab & 1 : 0;
        if constexpr (!lookahead_a) {
            gbf_sm120_half_load_a_fragments<M, N, BK, Stages, MAtoms>(
                stage, warp_m, slab, a_fragment[0]);
        }
        if constexpr (lookahead_a && !b_before_next_a) {
            if (slab + 1 < BK / 16) {
                gbf_sm120_half_load_a_fragments<M, N, BK, Stages, MAtoms>(
                    stage, warp_m, slab + 1, a_fragment[current_a ^ 1]);
            }
        }
        if constexpr (N == 64 && BK == 64 && Stages == 2) {
            unsigned b_fragment[2][2];
            gbf_sm120_half_load_b_fragment<M, N, BK, Stages>(
                stage, warp_n, slab, 0, b_fragment[0]);
            if constexpr (lookahead_a && b_before_next_a) {
                if (slab + 1 < BK / 16) {
                    gbf_sm120_half_load_a_fragments<
                        M, N, BK, Stages, MAtoms>(
                            stage, warp_m, slab + 1,
                            a_fragment[current_a ^ 1]);
                }
            }
#pragma unroll
            for (int fragment = 0; fragment < 4; ++fragment) {
                int current = fragment & 1;
                int next = current ^ 1;
                if (fragment + 1 < 4) {
                    gbf_sm120_half_load_b_fragment<M, N, BK, Stages>(
                        stage, warp_n, slab, fragment + 1,
                        b_fragment[next]);
                }
#pragma unroll
                for (int row_fragment = 0;
                     row_fragment < MAtoms;
                     ++row_fragment) {
                    GbfSm120HalfMma<T>::issue(
                        accumulator[row_fragment][fragment],
                        a_fragment[current_a][row_fragment],
                        b_fragment[current]);
                }
            }
        } else {
#pragma unroll
            for (int fragment = 0; fragment < 4; ++fragment) {
                unsigned b_fragment[2];
                gbf_sm120_half_load_b_fragment<M, N, BK, Stages>(
                    stage, warp_n, slab, fragment, b_fragment);
                if constexpr (lookahead_a && b_before_next_a) {
                    if (fragment == 0 && slab + 1 < BK / 16) {
                        gbf_sm120_half_load_a_fragments<
                            M, N, BK, Stages, MAtoms>(
                                stage, warp_m, slab + 1,
                                a_fragment[current_a ^ 1]);
                    }
                }
#pragma unroll
                for (int row_fragment = 0;
                     row_fragment < MAtoms;
                     ++row_fragment) {
                    GbfSm120HalfMma<T>::issue(
                        accumulator[row_fragment][fragment],
                        a_fragment[current_a][row_fragment], b_fragment);
                }
            }
        }
    }
}

template <typename T>
static __device__ __forceinline__ T gbf_sm120_half_from_float(float value);

template <>
__device__ __forceinline__ __half
gbf_sm120_half_from_float<__half>(float value) {
    return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16
gbf_sm120_half_from_float<__nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}

template <>
__device__ __forceinline__ float
gbf_sm120_half_from_float<float>(float value) {
    return value;
}

static __device__ __forceinline__ float gbf_sm120_half_to_float(__half value) {
    return __half2float(value);
}

static __device__ __forceinline__ float gbf_sm120_half_to_float(
    __nv_bfloat16 value) {
    return __bfloat162float(value);
}

static __device__ __forceinline__ float gbf_sm120_half_to_float(float value) {
    return value;
}

static __device__ __forceinline__ void gbf_sm120_half_store_pair_rne(
    __half* destination, float first, float second) {
    *reinterpret_cast<__half2*>(destination) =
        __floats2half2_rn(first, second);
}

static __device__ __forceinline__ void gbf_sm120_half_store_pair_rne(
    __nv_bfloat16* destination, float first, float second) {
    *reinterpret_cast<__nv_bfloat162*>(destination) =
        __floats2bfloat162_rn(first, second);
}

static __device__ __forceinline__ void gbf_sm120_half_store_pair_rne(
    float* destination, float first, float second) {
    *reinterpret_cast<float2*>(destination) = make_float2(first, second);
}

template <typename T>
struct GbfSm120HalfDirectOutput {
    static constexpr bool value = true;
};

struct GbfSm120HalfOutput {
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

template <typename T>
static __device__ __forceinline__ void gbf_sm120_half_store_pair(
    const GbfSm120HalfOutput& output, int row, int column,
    float first, float second) {
    long long offset = static_cast<long long>(row) * output.stride + column;
    T* destination = static_cast<T*>(output.pointer) + offset;
    float first_value = first;
    float second_value = second;
    if constexpr (!GbfSm120HalfDirectOutput<T>::value) {
        first_value = __fmul_rn(output.alpha, first);
        second_value = __fmul_rn(output.alpha, second);
        if (output.beta != 0.0f) {
            first_value = __fmaf_rn(
                output.beta, gbf_sm120_half_to_float(destination[0]),
                first_value);
            second_value = __fmaf_rn(
                output.beta, gbf_sm120_half_to_float(destination[1]),
                second_value);
        }
    }
    constexpr unsigned long long pair_alignment = sizeof(T) * 2ULL;
    if ((reinterpret_cast<unsigned long long>(destination)
         & (pair_alignment - 1ULL)) == 0) {
        gbf_sm120_half_store_pair_rne(
            destination, first_value, second_value);
    } else {
        destination[0] = gbf_sm120_half_from_float<T>(first_value);
        destination[1] = gbf_sm120_half_from_float<T>(second_value);
    }
}

template <typename T>
static __device__ __forceinline__ void gbf_sm120_half_store_scalar(
    const GbfSm120HalfOutput& output, int row, int column,
    float accumulator) {
    long long offset = static_cast<long long>(row) * output.stride + column;
    T* destination = static_cast<T*>(output.pointer) + offset;
    float value = accumulator;
    if constexpr (!GbfSm120HalfDirectOutput<T>::value) {
        value = __fmul_rn(output.alpha, accumulator);
        if (output.beta != 0.0f) {
            value = __fmaf_rn(
                output.beta, gbf_sm120_half_to_float(destination[0]), value);
        }
    }
    destination[0] = gbf_sm120_half_from_float<T>(value);
}

template <typename T, int M, int N, int MAtoms>
static __device__ __forceinline__ void gbf_sm120_half_epilogue_full(
    const GbfSm120HalfOutput& output,
    const float (&accumulator)[MAtoms][4][4]) {
    int lane = threadIdx.x & 31;
    int group = lane >> 2;
    int pair = lane & 3;
    int warp = threadIdx.x >> 5;
    int warp_row = output.row_tile
        + (warp / (N / 32)) * (MAtoms * 16);
    int warp_column = output.column_tile
        + (warp % (N / 32)) * 32 + pair * 2;

#pragma unroll
    for (int row_fragment = 0;
         row_fragment < MAtoms;
         ++row_fragment) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            int row = warp_row + row_fragment * 16 + group + half * 8;
            T* row_destination = static_cast<T*>(output.pointer)
                + static_cast<long long>(row) * output.stride
                + warp_column;
            int element = half * 2;
#pragma unroll
            for (int column_fragment = 0;
                 column_fragment < 4;
                 ++column_fragment) {
                T* destination = row_destination + column_fragment * 8;
                float first =
                    accumulator[row_fragment][column_fragment][element];
                float second =
                    accumulator[row_fragment][column_fragment][element + 1];
                if constexpr (!GbfSm120HalfDirectOutput<T>::value) {
                    first = __fmul_rn(output.alpha, first);
                    second = __fmul_rn(output.alpha, second);
                    if (output.beta != 0.0f) {
                        first = __fmaf_rn(
                            output.beta,
                            gbf_sm120_half_to_float(destination[0]), first);
                        second = __fmaf_rn(
                            output.beta,
                            gbf_sm120_half_to_float(destination[1]), second);
                    }
                }
                gbf_sm120_half_store_pair_rne(destination, first, second);
            }
        }
    }
}

template <typename T, int M, int N, int BK, int Stages, int MAtoms>
static __device__ __forceinline__ void gbf_sm120_half_epilogue(
    const GbfSm120HalfOutput& output,
    const float (&accumulator)[MAtoms][4][4]) {
    constexpr bool row_major =
        !(M == 64 && N == 64 && BK == 32 && Stages == 3);
    if constexpr (row_major) {
        constexpr unsigned long long pair_alignment = sizeof(T) * 2ULL;
        bool full = output.row_tile <= output.rows - M
            && output.column_tile <= output.columns - N
            && (reinterpret_cast<unsigned long long>(output.pointer)
                & (pair_alignment - 1ULL)) == 0
            && (output.stride & 1) == 0;
        if (full) {
            gbf_sm120_half_epilogue_full<T, M, N, MAtoms>(
                output, accumulator);
            return;
        }
    }

    int lane = threadIdx.x & 31;
    int group = lane >> 2;
    int pair = lane & 3;
    int warp = threadIdx.x >> 5;
#pragma unroll
    for (int row_fragment = 0;
         row_fragment < MAtoms;
         ++row_fragment) {
#pragma unroll
        for (int column_fragment = 0;
             column_fragment < 4;
             ++column_fragment) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                int row = output.row_tile
                    + (warp / output.warp_columns) * (MAtoms * 16)
                    + row_fragment * 16 + group + half * 8;
                int column = output.column_tile
                    + (warp % output.warp_columns) * 32
                    + column_fragment * 8 + pair * 2;
                if (row >= output.rows || column >= output.columns) continue;
                int element = half * 2;
                if (column + 1 < output.columns) {
                    gbf_sm120_half_store_pair<T>(
                        output, row, column,
                        accumulator[row_fragment][column_fragment][element],
                        accumulator[row_fragment][column_fragment][element + 1]);
                } else {
                    gbf_sm120_half_store_scalar<T>(
                        output, row, column,
                        accumulator[row_fragment][column_fragment][element]);
                }
            }
        }
    }
}

template <int MAtoms>
static __device__ __forceinline__ void gbf_sm120_half_initialize_accumulator(
    float (&accumulator)[MAtoms][4][4], const float* bias,
    int columns, int output_column, int warp_n) {
    int pair = threadIdx.x & 3;
#pragma unroll
    for (int row_fragment = 0;
         row_fragment < MAtoms;
         ++row_fragment) {
#pragma unroll
        for (int column_fragment = 0;
             column_fragment < 4;
             ++column_fragment) {
            float first = 0.0f;
            float second = 0.0f;
            if (bias != nullptr) {
                int column = output_column + warp_n
                    + column_fragment * 8 + pair * 2;
                if (column < columns) first = bias[column];
                if (column + 1 < columns) second = bias[column + 1];
            }
            accumulator[row_fragment][column_fragment][0] = first;
            accumulator[row_fragment][column_fragment][1] = second;
            accumulator[row_fragment][column_fragment][2] = first;
            accumulator[row_fragment][column_fragment][3] = second;
        }
    }
}

template <typename TInput, typename TOutput, int M, int N, int BK, int Stages>
static __device__ __forceinline__ void gbf_sm120_half_kernel(
    void* output,
    const GbfSm120HalfTensorMap& a_map,
    const GbfSm120HalfTensorMap& b_map,
    const float* bias,
    const GbfSm120HalfParams& params) {
    constexpr int threads =
        GbfSm120HalfStorage<M, N, BK, Stages>::threads;
    constexpr int warps = threads / 32;
    constexpr int row_fragments =
        GbfSm120HalfStorage<M, N, BK, Stages>::wide_m_warp ? 4 : 2;
    constexpr bool rotating_stage =
        (M == 64 && N == 128) ||
        (M == 128 && N == 64 && Stages == 2) ||
        (M == 128 && N == 128 && BK == 64 && Stages == 3);
    extern __shared__ __align__(128) unsigned char storage[];
    unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(storage));
    int column_tiles = 1 + (params.n - 1) / N;
    int output_row = (blockIdx.x / column_tiles) * M;
    int output_column = (blockIdx.x % column_tiles) * N;
    int tile_count = 1 + (params.k - 1) / BK;
    GbfSm120HalfPipeline pipeline = {
        shared + 128,
        shared,
        shared + 64,
        output_row,
        output_column,
    };
    int warp = threadIdx.x >> 5;

    if (threadIdx.x == 0) {
#pragma unroll
        for (int stage = 0; stage < Stages; ++stage) {
            gbf_sm120_half_init_barrier<1>(
                pipeline.full + stage * 8);
            gbf_sm120_half_init_barrier<warps>(
                pipeline.empty + stage * 8);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    if (warp == 0 && (threadIdx.x & 31) == 0) {
#pragma unroll
        for (int tile = 0; tile < Stages; ++tile) {
            if (tile < tile_count) {
                gbf_sm120_half_produce_stage<M, N, BK, Stages>(
                    a_map, b_map, params, pipeline, tile);
            }
        }
    }
    __syncwarp();

    int warp_columns = N / 32;
    int warp_m = (warp / warp_columns) * (row_fragments * 16);
    int warp_n = (warp % warp_columns) * 32;
    float accumulator[row_fragments][4][4];
    gbf_sm120_half_initialize_accumulator<row_fragments>(
        accumulator, bias, params.n, output_column, warp_n);
    int rotating_stage_index = 0;
    unsigned rotating_phase = 0;
    for (int tile = 0; tile < tile_count; ++tile) {
        int stage;
        unsigned phase;
        if constexpr (rotating_stage) {
            stage = rotating_stage_index;
            phase = rotating_phase;
        } else {
            stage = tile % Stages;
            phase = static_cast<unsigned>(tile / Stages) & 1U;
        }
        gbf_sm120_half_wait_barrier(
            pipeline.full + stage * 8, phase);
        gbf_sm120_half_issue_stage<
            TInput, M, N, BK, Stages, row_fragments, rotating_stage>(
                pipeline.payload, rotating_stage ? stage : tile,
                warp_m, warp_n, accumulator);
        __syncwarp();
        if ((threadIdx.x & 31) == 0) {
            gbf_sm120_half_arrive_empty(
                pipeline.empty + stage * 8);
        }
        if (warp == 0 && (threadIdx.x & 31) == 0) {
            int refill = tile + Stages;
            if (refill < tile_count) {
                gbf_sm120_half_wait_barrier(
                    pipeline.empty + stage * 8, phase);
                gbf_sm120_half_produce_stage<M, N, BK, Stages>(
                    a_map, b_map, params, pipeline, refill);
            }
        }
        __syncwarp();
        if constexpr (rotating_stage) {
            if (++rotating_stage_index == Stages) {
                rotating_stage_index = 0;
                rotating_phase ^= 1U;
            }
        }
    }

    GbfSm120HalfOutput destination = {
        output,
        params.alpha,
        params.beta,
        params.m,
        params.n,
        params.ldc,
        output_row,
        output_column,
        warp_columns,
    };
    gbf_sm120_half_epilogue<
        TOutput, M, N, BK, Stages, row_fragments>(
            destination, accumulator);
}

#define GBF_SM120_HALF_DEFINE_KERNEL(                                        \
    NAME, INPUT_TYPE, OUTPUT_TYPE, M, N, BK, STAGES)                         \
    extern "C" __global__                                                    \
    __launch_bounds__(GbfSm120HalfStorage<M, N, BK, STAGES>::threads)        \
    void NAME(                                                               \
        void* output,                                                        \
        const __grid_constant__ GbfSm120HalfTensorMap a_map,                 \
        const __grid_constant__ GbfSm120HalfTensorMap b_map,                 \
        const float* bias,                                                   \
        const __grid_constant__ GbfSm120HalfParams params) {                 \
        gbf_sm120_half_kernel<INPUT_TYPE, OUTPUT_TYPE, M, N, BK, STAGES>(    \
            output, a_map, b_map, bias, params);                             \
    }

#define GBF_SM120_HALF_DEFINE_PAIR(M, N, BK, STAGES)                         \
    GBF_SM120_HALF_DEFINE_KERNEL(                                            \
        nn_sm120_tma_##M##x##N##_bk##BK##_s##STAGES##_bf16,          \
        __nv_bfloat16, __nv_bfloat16, M, N, BK, STAGES)                      \
    GBF_SM120_HALF_DEFINE_KERNEL(                                            \
        nn_sm120_tma_##M##x##N##_bk##BK##_s##STAGES##_f16,           \
        __half, __half, M, N, BK, STAGES)                                    \
    GBF_SM120_HALF_DEFINE_KERNEL(                                            \
        nn_sm120_tma_##M##x##N##_bk##BK##_s##STAGES##_f32out_bf16,   \
        __nv_bfloat16, float, M, N, BK, STAGES)                              \
    GBF_SM120_HALF_DEFINE_KERNEL(                                            \
        nn_sm120_tma_##M##x##N##_bk##BK##_s##STAGES##_f32out_f16,    \
        __half, float, M, N, BK, STAGES)

GBF_SM120_HALF_DEFINE_PAIR(64, 64, 64, 2)
GBF_SM120_HALF_DEFINE_PAIR(64, 128, 64, 2)
GBF_SM120_HALF_DEFINE_PAIR(128, 64, 32, 3)
GBF_SM120_HALF_DEFINE_PAIR(128, 128, 32, 2)
GBF_SM120_HALF_DEFINE_PAIR(128, 128, 32, 3)

#undef GBF_SM120_HALF_DEFINE_PAIR
#undef GBF_SM120_HALF_DEFINE_KERNEL
#undef GBF_SM120_HALF_CHECK_STORAGE
#undef GBF_SM120_HALF_SWIZZLE_128B
#undef GBF_SM120_HALF_SWIZZLE_64B

#endif
