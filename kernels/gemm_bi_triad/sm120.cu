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
static_assert(__is_standard_layout(Sm120KernelParams),
              "SM120 kernel parameters must remain standard layout");
// Ten ordered 4-byte fields in 40 bytes leave no internal or tail padding.
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
    static constexpr bool wide_m_warp =
        M == 128 && N == 128 && BK == 32;
    static constexpr int compute_warps =
        wide_m_warp ? 8 : (M / 32) * (N / 32);
    static constexpr int threads = compute_warps * 32;
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
    unsigned logical_chunk = element / 8U;
    unsigned row_start = shared_address + logical_row * groups * 16U;
    unsigned phase = (row_start / 128U) % groups;
    unsigned physical_chunk = logical_chunk ^ phase;
    return row_start + physical_chunk * 16U;
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

// Loads k-tile `k_tile` of the tile at `pipeline.output_row/col` into the
// stage of `step`. The tiled kernels step through one tile in order, so
// step and k-tile coincide; the stream-K schedule numbers steps across a
// range of tiles and hands over the k-tile of the unit the step carries.
template <int Op, int M, int N, int BK, int Stages>
static __device__ __forceinline__ void sm120_produce_stage_at(
    const CUtensorMap& a_map, const CUtensorMap& b_map,
    const Sm120KernelParams& params, const Sm120Pipeline& pipeline,
    int step, int k_tile) {
    constexpr int plane_bytes = BK * BK * 2;
    constexpr int a_bytes = M * BK * 2;
    constexpr int stage_bytes = Sm120Storage<M, N, BK, Stages>::stage_bytes;
    unsigned stage = pipeline.payload + (step % Stages) * stage_bytes;
    unsigned a_destination = stage;
    unsigned b_destination = stage + a_bytes;
    unsigned barrier = pipeline.full + (step % Stages) * 8;
    unsigned long long a_descriptor =
        reinterpret_cast<unsigned long long>(&a_map);
    unsigned long long b_descriptor =
        reinterpret_cast<unsigned long long>(&b_map);
    int reduction = k_tile * BK;

    sm120_expect_transaction<stage_bytes>(barrier);
    if constexpr (Op == Sm120Nn) {
        sm120_tma_copy(a_destination, a_descriptor, reduction,
                       pipeline.output_row, params.a_x, params.a_y, barrier);
        if constexpr (M == 128 && N == 64 && BK == 32) {
            sm120_tma_copy(b_destination, b_descriptor,
                           pipeline.output_col, reduction,
                           params.b_x, params.b_y, barrier);
        } else {
#pragma unroll
            for (int plane = 0; plane < N / BK; ++plane) {
                int column = pipeline.output_col + plane * BK;
                sm120_tma_copy(
                    b_destination + plane * plane_bytes, b_descriptor,
                    column, reduction, params.b_x, params.b_y, barrier);
            }
        }
    } else if constexpr (Op == Sm120Tn) {
        if constexpr (BK == 32 && (M == 64 || N == 64)) {
            constexpr int wide_plane_bytes = 64 * BK * 2;
#pragma unroll
            for (int plane = 0; plane < M / 64; ++plane) {
                int row = pipeline.output_row + plane * 64;
                sm120_tma_copy(
                    a_destination + plane * wide_plane_bytes, a_descriptor,
                    row, reduction, params.a_x, params.a_y, barrier);
            }
#pragma unroll
            for (int plane = 0; plane < N / 64; ++plane) {
                int column = pipeline.output_col + plane * 64;
                sm120_tma_copy(
                    b_destination + plane * wide_plane_bytes, b_descriptor,
                    column, reduction, params.b_x, params.b_y, barrier);
            }
        } else {
#pragma unroll
            for (int plane = 0; plane < M / BK; ++plane) {
                int row = pipeline.output_row + plane * BK;
                sm120_tma_copy(
                    a_destination + plane * plane_bytes, a_descriptor,
                    row, reduction, params.a_x, params.a_y, barrier);
            }
#pragma unroll
            for (int plane = 0; plane < N / BK; ++plane) {
                int column = pipeline.output_col + plane * BK;
                sm120_tma_copy(
                    b_destination + plane * plane_bytes, b_descriptor,
                    column, reduction, params.b_x, params.b_y, barrier);
            }
        }
    } else {
        sm120_tma_copy(a_destination, a_descriptor, reduction,
                       pipeline.output_row, params.a_x, params.a_y, barrier);
        sm120_tma_copy(b_destination, b_descriptor, reduction,
                       pipeline.output_col, params.b_x, params.b_y, barrier);
    }
}

template <int Op, int M, int N, int BK, int Stages>
static __device__ __forceinline__ void sm120_produce_stage(
    const CUtensorMap& a_map, const CUtensorMap& b_map,
    const Sm120KernelParams& params, const Sm120Pipeline& pipeline,
    int tile) {
    sm120_produce_stage_at<Op, M, N, BK, Stages>(
        a_map, b_map, params, pipeline, tile, tile);
}

template <int Op, int M, int N, int BK, int Stages, int MAtoms>
static __device__ __forceinline__ void sm120_load_a_fragments(
    unsigned stage, int warp_m, int slab,
    unsigned (&a_fragment)[MAtoms][4]) {
    constexpr int plane_bytes = BK * BK * 2;
    unsigned a_base = stage;
    constexpr bool compact_a_addresses =
        (Op == Sm120Nt &&
         ((M == 64 && N == 128 && BK == 64 && Stages == 2) ||
          (M == 128 && N == 128 && BK == 32 && Stages == 3))) ||
        (Op == Sm120Nn && M == 128 && N == 128 && BK == 32 &&
         Stages == 2);
    int lane = threadIdx.x & 31;
    int row8 = lane & 7;
    int quadrant = lane >> 3;
    int k0 = slab * 16;

#pragma unroll
    for (int fm = 0; fm < MAtoms; ++fm) {
        if constexpr (Op == Sm120Tn) {
            int logical_row = k0 + ((quadrant & 2) ? 8 : 0) + row8;
            int output_element =
                warp_m + fm * 16 + ((quadrant & 1) ? 8 : 0);
            unsigned address;
            if constexpr (BK == 32 && (M == 64 || N == 64)) {
                constexpr int wide_plane_bytes = 64 * BK * 2;
                unsigned plane = a_base + static_cast<unsigned>(
                    (output_element / 64) * wide_plane_bytes);
                address = sm120_swizzled_address_impl<SM120_SWIZZLE_128B>(
                    plane, static_cast<unsigned>(logical_row),
                    static_cast<unsigned>(output_element % 64));
            } else {
                unsigned plane = a_base +
                    static_cast<unsigned>((output_element / BK) * plane_bytes);
                address = sm120_swizzled_address<BK>(
                    plane, static_cast<unsigned>(logical_row),
                    static_cast<unsigned>(output_element % BK));
            }
            sm120_load_x4_transpose(address, a_fragment[fm]);
        } else {
            int logical_row;
            int element;
            if constexpr (compact_a_addresses) {
                logical_row = warp_m + fm * 16 + (lane & 15);
                element = k0 + ((lane & 16) >> 1);
            } else {
                logical_row =
                    warp_m + fm * 16 +
                    ((quadrant & 1) ? 8 : 0) + row8;
                element = k0 + ((quadrant & 2) ? 8 : 0);
            }
            unsigned address = sm120_swizzled_address<BK>(
                a_base, static_cast<unsigned>(logical_row),
                static_cast<unsigned>(element));
            sm120_load_x4(address, a_fragment[fm]);
        }
    }
}

template <int Op, int M, int N, int BK, int Stages>
static __device__ __forceinline__ void sm120_load_b_fragment(
    unsigned stage, int warp_n, int slab, int fn,
    unsigned (&b_fragment)[2]) {
    constexpr int plane_bytes = BK * BK * 2;
    constexpr int a_bytes = M * BK * 2;
    constexpr bool compact_tn_x2_addresses = Op == Sm120Tn &&
        (Stages == 3 ||
         (BK == 32 && ((M == 64 && N == 128) ||
                       (M == 128 && N == 64))));
    constexpr bool compact_nn_x2_addresses = Op == Sm120Nn &&
        ((Stages == 2 &&
          ((M == 128 && N == 64) ||
           (M == 128 && N == 128 && BK == 64))) ||
         (Stages == 3 &&
          ((M == 64 && N == 64 && BK == 32) ||
           (M == 64 && N == 128) ||
           (M == 128 && N == 128))));
    constexpr bool compact_x2_addresses =
        compact_tn_x2_addresses || compact_nn_x2_addresses;
    unsigned b_base = stage + a_bytes;
    int lane = threadIdx.x & 31;
    int row8 = lane & 7;
    int k0 = slab * 16;

    if constexpr (Op == Sm120Nt) {
        int logical_row = warp_n + fn * 8 + row8;
        int quadrant = lane >> 3;
        int element = k0 + ((quadrant & 1) ? 8 : 0);
        unsigned address = sm120_swizzled_address<BK>(
            b_base, static_cast<unsigned>(logical_row),
            static_cast<unsigned>(element));
        sm120_load_x2(address, b_fragment);
    } else {
        int logical_row;
        if constexpr (compact_x2_addresses) {
            logical_row = k0 + (lane & 15);
        } else {
            int quadrant = lane >> 3;
            logical_row = k0 + ((quadrant & 1) ? 8 : 0) + row8;
        }
        int output_element = warp_n + fn * 8;
        unsigned address;
        if constexpr (((Op == Sm120Nn && M == 128 && N == 64) ||
                       (Op == Sm120Tn && (M == 64 || N == 64))) &&
                      BK == 32) {
            constexpr int wide_plane_bytes = 64 * BK * 2;
            unsigned plane = b_base +
                static_cast<unsigned>((output_element / 64) * wide_plane_bytes);
            address = sm120_swizzled_address_impl<SM120_SWIZZLE_128B>(
                plane, static_cast<unsigned>(logical_row),
                static_cast<unsigned>(output_element % 64));
        } else {
            unsigned plane = b_base +
                static_cast<unsigned>((output_element / BK) * plane_bytes);
            address = sm120_swizzled_address<BK>(
                plane, static_cast<unsigned>(logical_row),
                static_cast<unsigned>(output_element % BK));
        }
        sm120_load_x2_transpose(address, b_fragment);
    }
}

template <typename T, int Op, int M, int N, int BK, int Stages,
          int MAtoms, bool RotatingStage>
static __device__ __forceinline__ void sm120_issue_stage(
    unsigned payload, int stage_or_tile, int warp_m, int warp_n,
    float (&accumulator)[MAtoms][4][4]) {
    constexpr int stage_bytes = Sm120Storage<M, N, BK, Stages>::stage_bytes;
    constexpr bool lookahead_a = BK == 64 && Stages == 2 &&
        ((Op == Sm120Nn && M == 64) ||
         (Op == Sm120Tn && N == 128) ||
         (Op == Sm120Nt && M == 128 && N == 128));
    constexpr bool b_before_next_a =
        (Op == Sm120Nn && M == 64 && N == 64) ||
        (Op == Sm120Tn && M == 64 && N == 128);
    int stage_index;
    if constexpr (RotatingStage) {
        stage_index = stage_or_tile;
    } else {
        stage_index = stage_or_tile % Stages;
    }
    unsigned stage = payload + stage_index * stage_bytes;
    unsigned a_fragment[lookahead_a ? 2 : 1][MAtoms][4];
    if constexpr (lookahead_a) {
        sm120_load_a_fragments<Op, M, N, BK, Stages, MAtoms>(
            stage, warp_m, 0, a_fragment[0]);
    }
#pragma unroll
    for (int slab = 0; slab < BK / 16; ++slab) {
        int current_a = lookahead_a ? slab & 1 : 0;
        if constexpr (!lookahead_a) {
            sm120_load_a_fragments<Op, M, N, BK, Stages, MAtoms>(
                stage, warp_m, slab, a_fragment[0]);
        }
        if constexpr (lookahead_a && !b_before_next_a) {
            if (slab + 1 < BK / 16) {
                sm120_load_a_fragments<Op, M, N, BK, Stages, MAtoms>(
                    stage, warp_m, slab + 1, a_fragment[current_a ^ 1]);
            }
        }
        if constexpr (N == 64 && BK == 64 && Stages == 2) {
            unsigned b_fragment[2][2];
            sm120_load_b_fragment<Op, M, N, BK, Stages>(
                stage, warp_n, slab, 0, b_fragment[0]);
            if constexpr (lookahead_a && b_before_next_a) {
                if (slab + 1 < BK / 16) {
                    sm120_load_a_fragments<Op, M, N, BK, Stages, MAtoms>(
                        stage, warp_m, slab + 1,
                        a_fragment[current_a ^ 1]);
                }
            }
#pragma unroll
            for (int fn = 0; fn < 4; ++fn) {
                int current = fn & 1;
                int next = current ^ 1;
                if (fn + 1 < 4) {
                    sm120_load_b_fragment<Op, M, N, BK, Stages>(
                        stage, warp_n, slab, fn + 1, b_fragment[next]);
                }
#pragma unroll
                for (int fm = 0; fm < MAtoms; ++fm) {
                    Sm120Mma<T>::issue(
                        accumulator[fm][fn], a_fragment[current_a][fm],
                        b_fragment[current]);
                }
            }
        } else {
#pragma unroll
            for (int fn = 0; fn < 4; ++fn) {
                unsigned b_fragment[2];
                sm120_load_b_fragment<Op, M, N, BK, Stages>(
                    stage, warp_n, slab, fn, b_fragment);
                if constexpr (lookahead_a && b_before_next_a) {
                    if (fn == 0 && slab + 1 < BK / 16) {
                        sm120_load_a_fragments<Op, M, N, BK, Stages, MAtoms>(
                            stage, warp_m, slab + 1,
                            a_fragment[current_a ^ 1]);
                    }
                }
#pragma unroll
                for (int fm = 0; fm < MAtoms; ++fm) {
                    Sm120Mma<T>::issue(
                        accumulator[fm][fn], a_fragment[current_a][fm],
                        b_fragment);
                }
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
            gemm_bi_store_pair_rne(destination, v0, v1);
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

template <int M, int N, int MAtoms>
static __device__ __forceinline__ void sm120_epilogue_tn_full(
    const Sm120Output& output,
    const float (&accumulator)[MAtoms][4][4]) {
    int lane = threadIdx.x & 31;
    int group = lane >> 2;
    int pair = lane & 3;
    int warp = threadIdx.x >> 5;
    int warp_row = output.row_tile +
        (warp / (N / 32)) * (MAtoms * 16);
    int warp_column = output.column_tile +
        (warp % (N / 32)) * 32 + pair * 2;

#pragma unroll
    for (int fm = 0; fm < MAtoms; ++fm) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            int row = warp_row + fm * 16 + group + half * 8;
            float* row_destination = static_cast<float*>(output.pointer) +
                static_cast<long long>(row) * output.stride + warp_column;
            int element = half * 2;
#pragma unroll
            for (int fn = 0; fn < 4; ++fn) {
                float* destination = row_destination + fn * 8;
                float2 value = {
                    __fmaf_rn(output.alpha,
                        accumulator[fm][fn][element], destination[0]),
                    __fmaf_rn(output.alpha,
                        accumulator[fm][fn][element + 1], destination[1]),
                };
                *reinterpret_cast<float2*>(destination) = value;
            }
        }
    }
}

template <typename T, int Op, int M, int N, int MAtoms>
static __device__ __forceinline__ void sm120_epilogue_half_full(
    const Sm120Output& output,
    const float (&accumulator)[MAtoms][4][4]) {
    int lane = threadIdx.x & 31;
    int group = lane >> 2;
    int pair = lane & 3;
    int warp = threadIdx.x >> 5;
    int warp_row = output.row_tile +
        (warp / (N / 32)) * (MAtoms * 16);
    int warp_column = output.column_tile +
        (warp % (N / 32)) * 32 + pair * 2;

#pragma unroll
    for (int fm = 0; fm < MAtoms; ++fm) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            int row = warp_row + fm * 16 + group + half * 8;
            T* row_destination = static_cast<T*>(output.pointer) +
                static_cast<long long>(row) * output.stride + warp_column;
            int element = half * 2;
#pragma unroll
            for (int fn = 0; fn < 4; ++fn) {
                T* destination = row_destination + fn * 8;
                float v0 = __fmul_rn(
                    output.alpha, accumulator[fm][fn][element]);
                float v1 = __fmul_rn(
                    output.alpha, accumulator[fm][fn][element + 1]);
                if constexpr (Op == Sm120Nn) {
                    if (output.beta != 0.0f) {
                        v0 = __fmaf_rn(
                            output.beta, to_f(destination[0]), v0);
                        v1 = __fmaf_rn(
                            output.beta, to_f(destination[1]), v1);
                    }
                }
                gemm_bi_store_pair_rne(destination, v0, v1);
            }
        }
    }
}

template <typename T, int Op, int M, int N, int BK, int Stages,
          int MAtoms>
static __device__ __forceinline__ void sm120_epilogue(
    const Sm120Output& output,
    const float (&accumulator)[MAtoms][4][4]) {
    constexpr bool row_major_half =
        (Op == Sm120Nn &&
         !(M == 64 && N == 64 && BK == 32 && Stages == 3)) ||
        (Op == Sm120Nt &&
         ((M == 64 && N == 64) ||
          (M == 128 && N == 128) ||
          (M == 128 && N == 64 && BK == 32 && Stages == 2) ||
          (BK == 64 && Stages == 3)));
    if constexpr (Op == Sm120Tn) {
        bool full = output.row_tile <= output.rows - M &&
                    output.column_tile <= output.columns - N &&
                    (reinterpret_cast<unsigned long long>(output.pointer) & 7ULL) == 0 &&
                    (output.stride & 1) == 0;
        if (full) {
            sm120_epilogue_tn_full<M, N, MAtoms>(output, accumulator);
            return;
        }
    } else if constexpr (row_major_half) {
        bool full = output.row_tile <= output.rows - M &&
                    output.column_tile <= output.columns - N &&
                    (reinterpret_cast<unsigned long long>(output.pointer) & 3ULL) == 0 &&
                    (output.stride & 1) == 0;
        if (full) {
            sm120_epilogue_half_full<T, Op, M, N, MAtoms>(
                output, accumulator);
            return;
        }
    }
    int lane = threadIdx.x & 31;
    int group = lane >> 2;
    int pair = lane & 3;
    int warp = threadIdx.x >> 5;

#pragma unroll
    for (int fm = 0; fm < MAtoms; ++fm) {
#pragma unroll
        for (int fn = 0; fn < 4; ++fn) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                int row = output.row_tile +
                          (warp / output.warp_columns) * (MAtoms * 16) +
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

template <int Op, int MAtoms>
static __device__ __forceinline__ void sm120_initialize_accumulator(
    float (&accumulator)[MAtoms][4][4], const float* bias,
    int output_columns, int output_col, int warp_n) {
    int pair = threadIdx.x & 3;
#pragma unroll
    for (int fm = 0; fm < MAtoms; ++fm) {
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
    constexpr int warps = threads / 32;
    constexpr int m_atoms =
        Sm120Storage<M, N, BK, Stages>::wide_m_warp ? 4 : 2;
    constexpr bool rotating_stage =
        (Op == Sm120Nn &&
         ((M == 64 && N == 128) ||
          (M == 128 && N == 64 && Stages == 2) ||
          (M == 128 && N == 128 && BK == 64 && Stages == 3))) ||
        (Op == Sm120Tn &&
         ((M == 64 && N == 128 && Stages == 3) ||
          (M == 128 && N == 128 && BK == 64 && Stages == 3))) ||
        (Op == Sm120Nt && BK == 64 &&
         (Stages == 3 || (M == 64 && N == 64 && Stages == 2)));
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
            sm120_init_barrier<warps>(pipeline.empty + stage * 8);
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
    int warp_m = (warp / warp_columns) * (m_atoms * 16);
    int warp_n = (warp % warp_columns) * 32;
    float accumulator[m_atoms][4][4];
    sm120_initialize_accumulator<Op, m_atoms>(
        accumulator, bias, output_columns, output_col, warp_n);
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
        sm120_wait_barrier(pipeline.full + stage * 8, phase);
        sm120_issue_stage<T, Op, M, N, BK, Stages, m_atoms, rotating_stage>(
            pipeline.payload, rotating_stage ? stage : tile,
            warp_m, warp_n, accumulator);
        sm120_sync_warp();
        if ((threadIdx.x & 31) == 0) {
            sm120_arrive_empty(pipeline.empty + stage * 8);
        }
        if (warp == 0 && (threadIdx.x & 31) == 0) {
            int refill = tile + Stages;
            if (refill < tile_count) {
                sm120_wait_barrier(
                    pipeline.empty + stage * 8, phase);
                sm120_produce_stage<Op, M, N, BK, Stages>(
                    a_map, b_map, params, pipeline, refill);
            }
        }
        sm120_sync_warp();
        if constexpr (rotating_stage) {
            if (++rotating_stage_index == Stages) {
                rotating_stage_index = 0;
                rotating_phase ^= 1U;
            }
        }
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
    sm120_epilogue<T, Op, M, N, BK, Stages, m_atoms>(
        destination, accumulator);
}

#define SM120_DEFINE_KERNEL(NAME, TYPE, OP, M, N, BK, STAGES)                \
    extern "C" __global__                                                   \
    __launch_bounds__(Sm120Storage<M, N, BK, STAGES>::threads)               \
    void NAME(void* output,                                                   \
              const __grid_constant__ CUtensorMap a_map,                     \
              const __grid_constant__ CUtensorMap b_map,                     \
              const float* bias,                                              \
              const __grid_constant__ Sm120KernelParams params) {            \
        sm120_kernel<TYPE, OP, M, N, BK, STAGES>(                             \
            output, a_map, b_map, bias, params);                              \
    }

SM120_DEFINE_KERNEL(gemm_bi_nn_sm120_tma_64x64_bk32_s2_bf16, __nv_bfloat16, Sm120Nn, 64, 64, 32, 2)
SM120_DEFINE_KERNEL(gemm_bi_nn_sm120_tma_64x64_bk32_s2_f16, __half, Sm120Nn, 64, 64, 32, 2)
SM120_DEFINE_KERNEL(gemm_bi_nn_sm120_tma_64x64_bk32_s3_bf16, __nv_bfloat16, Sm120Nn, 64, 64, 32, 3)
SM120_DEFINE_KERNEL(gemm_bi_nn_sm120_tma_64x64_bk32_s3_f16, __half, Sm120Nn, 64, 64, 32, 3)
SM120_DEFINE_KERNEL(gemm_bi_nn_sm120_tma_64x64_bk64_s2_bf16, __nv_bfloat16, Sm120Nn, 64, 64, 64, 2)
SM120_DEFINE_KERNEL(gemm_bi_nn_sm120_tma_64x64_bk64_s2_f16, __half, Sm120Nn, 64, 64, 64, 2)
SM120_DEFINE_KERNEL(gemm_bi_nn_sm120_tma_64x64_bk64_s3_bf16, __nv_bfloat16, Sm120Nn, 64, 64, 64, 3)
SM120_DEFINE_KERNEL(gemm_bi_nn_sm120_tma_64x64_bk64_s3_f16, __half, Sm120Nn, 64, 64, 64, 3)
SM120_DEFINE_KERNEL(gemm_bi_nn_sm120_tma_128x64_bk32_s2_bf16, __nv_bfloat16, Sm120Nn, 128, 64, 32, 2)
SM120_DEFINE_KERNEL(gemm_bi_nn_sm120_tma_128x64_bk32_s2_f16, __half, Sm120Nn, 128, 64, 32, 2)
SM120_DEFINE_KERNEL(gemm_bi_nn_sm120_tma_128x64_bk32_s3_bf16, __nv_bfloat16, Sm120Nn, 128, 64, 32, 3)
SM120_DEFINE_KERNEL(gemm_bi_nn_sm120_tma_128x64_bk32_s3_f16, __half, Sm120Nn, 128, 64, 32, 3)
SM120_DEFINE_KERNEL(gemm_bi_nn_sm120_tma_128x64_bk64_s2_bf16, __nv_bfloat16, Sm120Nn, 128, 64, 64, 2)
SM120_DEFINE_KERNEL(gemm_bi_nn_sm120_tma_128x64_bk64_s2_f16, __half, Sm120Nn, 128, 64, 64, 2)
SM120_DEFINE_KERNEL(gemm_bi_nn_sm120_tma_128x64_bk64_s3_bf16, __nv_bfloat16, Sm120Nn, 128, 64, 64, 3)
SM120_DEFINE_KERNEL(gemm_bi_nn_sm120_tma_128x64_bk64_s3_f16, __half, Sm120Nn, 128, 64, 64, 3)
SM120_DEFINE_KERNEL(gemm_bi_nn_sm120_tma_64x128_bk32_s2_bf16, __nv_bfloat16, Sm120Nn, 64, 128, 32, 2)
SM120_DEFINE_KERNEL(gemm_bi_nn_sm120_tma_64x128_bk32_s2_f16, __half, Sm120Nn, 64, 128, 32, 2)
SM120_DEFINE_KERNEL(gemm_bi_nn_sm120_tma_64x128_bk32_s3_bf16, __nv_bfloat16, Sm120Nn, 64, 128, 32, 3)
SM120_DEFINE_KERNEL(gemm_bi_nn_sm120_tma_64x128_bk32_s3_f16, __half, Sm120Nn, 64, 128, 32, 3)
SM120_DEFINE_KERNEL(gemm_bi_nn_sm120_tma_64x128_bk64_s2_bf16, __nv_bfloat16, Sm120Nn, 64, 128, 64, 2)
SM120_DEFINE_KERNEL(gemm_bi_nn_sm120_tma_64x128_bk64_s2_f16, __half, Sm120Nn, 64, 128, 64, 2)
SM120_DEFINE_KERNEL(gemm_bi_nn_sm120_tma_64x128_bk64_s3_bf16, __nv_bfloat16, Sm120Nn, 64, 128, 64, 3)
SM120_DEFINE_KERNEL(gemm_bi_nn_sm120_tma_64x128_bk64_s3_f16, __half, Sm120Nn, 64, 128, 64, 3)
SM120_DEFINE_KERNEL(gemm_bi_nn_sm120_tma_128x128_bk32_s2_bf16, __nv_bfloat16, Sm120Nn, 128, 128, 32, 2)
SM120_DEFINE_KERNEL(gemm_bi_nn_sm120_tma_128x128_bk32_s2_f16, __half, Sm120Nn, 128, 128, 32, 2)
SM120_DEFINE_KERNEL(gemm_bi_nn_sm120_tma_128x128_bk32_s3_bf16, __nv_bfloat16, Sm120Nn, 128, 128, 32, 3)
SM120_DEFINE_KERNEL(gemm_bi_nn_sm120_tma_128x128_bk32_s3_f16, __half, Sm120Nn, 128, 128, 32, 3)
SM120_DEFINE_KERNEL(gemm_bi_nn_sm120_tma_128x128_bk64_s2_bf16, __nv_bfloat16, Sm120Nn, 128, 128, 64, 2)
SM120_DEFINE_KERNEL(gemm_bi_nn_sm120_tma_128x128_bk64_s2_f16, __half, Sm120Nn, 128, 128, 64, 2)
SM120_DEFINE_KERNEL(gemm_bi_nn_sm120_tma_128x128_bk64_s3_bf16, __nv_bfloat16, Sm120Nn, 128, 128, 64, 3)
SM120_DEFINE_KERNEL(gemm_bi_nn_sm120_tma_128x128_bk64_s3_f16, __half, Sm120Nn, 128, 128, 64, 3)

SM120_DEFINE_KERNEL(gemm_bi_tn_sm120_tma_64x64_bk32_s2_bf16, __nv_bfloat16, Sm120Tn, 64, 64, 32, 2)
SM120_DEFINE_KERNEL(gemm_bi_tn_sm120_tma_64x64_bk32_s2_f16, __half, Sm120Tn, 64, 64, 32, 2)
SM120_DEFINE_KERNEL(gemm_bi_tn_sm120_tma_64x64_bk32_s3_bf16, __nv_bfloat16, Sm120Tn, 64, 64, 32, 3)
SM120_DEFINE_KERNEL(gemm_bi_tn_sm120_tma_64x64_bk32_s3_f16, __half, Sm120Tn, 64, 64, 32, 3)
SM120_DEFINE_KERNEL(gemm_bi_tn_sm120_tma_64x64_bk64_s2_bf16, __nv_bfloat16, Sm120Tn, 64, 64, 64, 2)
SM120_DEFINE_KERNEL(gemm_bi_tn_sm120_tma_64x64_bk64_s2_f16, __half, Sm120Tn, 64, 64, 64, 2)
SM120_DEFINE_KERNEL(gemm_bi_tn_sm120_tma_64x64_bk64_s3_bf16, __nv_bfloat16, Sm120Tn, 64, 64, 64, 3)
SM120_DEFINE_KERNEL(gemm_bi_tn_sm120_tma_64x64_bk64_s3_f16, __half, Sm120Tn, 64, 64, 64, 3)
SM120_DEFINE_KERNEL(gemm_bi_tn_sm120_tma_128x64_bk32_s2_bf16, __nv_bfloat16, Sm120Tn, 128, 64, 32, 2)
SM120_DEFINE_KERNEL(gemm_bi_tn_sm120_tma_128x64_bk32_s2_f16, __half, Sm120Tn, 128, 64, 32, 2)
SM120_DEFINE_KERNEL(gemm_bi_tn_sm120_tma_128x64_bk32_s3_bf16, __nv_bfloat16, Sm120Tn, 128, 64, 32, 3)
SM120_DEFINE_KERNEL(gemm_bi_tn_sm120_tma_128x64_bk32_s3_f16, __half, Sm120Tn, 128, 64, 32, 3)
SM120_DEFINE_KERNEL(gemm_bi_tn_sm120_tma_128x64_bk64_s2_bf16, __nv_bfloat16, Sm120Tn, 128, 64, 64, 2)
SM120_DEFINE_KERNEL(gemm_bi_tn_sm120_tma_128x64_bk64_s2_f16, __half, Sm120Tn, 128, 64, 64, 2)
SM120_DEFINE_KERNEL(gemm_bi_tn_sm120_tma_128x64_bk64_s3_bf16, __nv_bfloat16, Sm120Tn, 128, 64, 64, 3)
SM120_DEFINE_KERNEL(gemm_bi_tn_sm120_tma_128x64_bk64_s3_f16, __half, Sm120Tn, 128, 64, 64, 3)
SM120_DEFINE_KERNEL(gemm_bi_tn_sm120_tma_64x128_bk32_s2_bf16, __nv_bfloat16, Sm120Tn, 64, 128, 32, 2)
SM120_DEFINE_KERNEL(gemm_bi_tn_sm120_tma_64x128_bk32_s2_f16, __half, Sm120Tn, 64, 128, 32, 2)
SM120_DEFINE_KERNEL(gemm_bi_tn_sm120_tma_64x128_bk32_s3_bf16, __nv_bfloat16, Sm120Tn, 64, 128, 32, 3)
SM120_DEFINE_KERNEL(gemm_bi_tn_sm120_tma_64x128_bk32_s3_f16, __half, Sm120Tn, 64, 128, 32, 3)
SM120_DEFINE_KERNEL(gemm_bi_tn_sm120_tma_64x128_bk64_s2_bf16, __nv_bfloat16, Sm120Tn, 64, 128, 64, 2)
SM120_DEFINE_KERNEL(gemm_bi_tn_sm120_tma_64x128_bk64_s2_f16, __half, Sm120Tn, 64, 128, 64, 2)
SM120_DEFINE_KERNEL(gemm_bi_tn_sm120_tma_64x128_bk64_s3_bf16, __nv_bfloat16, Sm120Tn, 64, 128, 64, 3)
SM120_DEFINE_KERNEL(gemm_bi_tn_sm120_tma_64x128_bk64_s3_f16, __half, Sm120Tn, 64, 128, 64, 3)
SM120_DEFINE_KERNEL(gemm_bi_tn_sm120_tma_128x128_bk32_s2_bf16, __nv_bfloat16, Sm120Tn, 128, 128, 32, 2)
SM120_DEFINE_KERNEL(gemm_bi_tn_sm120_tma_128x128_bk32_s2_f16, __half, Sm120Tn, 128, 128, 32, 2)
SM120_DEFINE_KERNEL(gemm_bi_tn_sm120_tma_128x128_bk32_s3_bf16, __nv_bfloat16, Sm120Tn, 128, 128, 32, 3)
SM120_DEFINE_KERNEL(gemm_bi_tn_sm120_tma_128x128_bk32_s3_f16, __half, Sm120Tn, 128, 128, 32, 3)
SM120_DEFINE_KERNEL(gemm_bi_tn_sm120_tma_128x128_bk64_s2_bf16, __nv_bfloat16, Sm120Tn, 128, 128, 64, 2)
SM120_DEFINE_KERNEL(gemm_bi_tn_sm120_tma_128x128_bk64_s2_f16, __half, Sm120Tn, 128, 128, 64, 2)
SM120_DEFINE_KERNEL(gemm_bi_tn_sm120_tma_128x128_bk64_s3_bf16, __nv_bfloat16, Sm120Tn, 128, 128, 64, 3)
SM120_DEFINE_KERNEL(gemm_bi_tn_sm120_tma_128x128_bk64_s3_f16, __half, Sm120Tn, 128, 128, 64, 3)

SM120_DEFINE_KERNEL(gemm_bi_nt_sm120_tma_64x64_bk32_s2_bf16, __nv_bfloat16, Sm120Nt, 64, 64, 32, 2)
SM120_DEFINE_KERNEL(gemm_bi_nt_sm120_tma_64x64_bk32_s2_f16, __half, Sm120Nt, 64, 64, 32, 2)
SM120_DEFINE_KERNEL(gemm_bi_nt_sm120_tma_64x64_bk32_s3_bf16, __nv_bfloat16, Sm120Nt, 64, 64, 32, 3)
SM120_DEFINE_KERNEL(gemm_bi_nt_sm120_tma_64x64_bk32_s3_f16, __half, Sm120Nt, 64, 64, 32, 3)
SM120_DEFINE_KERNEL(gemm_bi_nt_sm120_tma_64x64_bk64_s2_bf16, __nv_bfloat16, Sm120Nt, 64, 64, 64, 2)
SM120_DEFINE_KERNEL(gemm_bi_nt_sm120_tma_64x64_bk64_s2_f16, __half, Sm120Nt, 64, 64, 64, 2)
SM120_DEFINE_KERNEL(gemm_bi_nt_sm120_tma_64x64_bk64_s3_bf16, __nv_bfloat16, Sm120Nt, 64, 64, 64, 3)
SM120_DEFINE_KERNEL(gemm_bi_nt_sm120_tma_64x64_bk64_s3_f16, __half, Sm120Nt, 64, 64, 64, 3)
SM120_DEFINE_KERNEL(gemm_bi_nt_sm120_tma_128x64_bk32_s2_bf16, __nv_bfloat16, Sm120Nt, 128, 64, 32, 2)
SM120_DEFINE_KERNEL(gemm_bi_nt_sm120_tma_128x64_bk32_s2_f16, __half, Sm120Nt, 128, 64, 32, 2)
SM120_DEFINE_KERNEL(gemm_bi_nt_sm120_tma_128x64_bk32_s3_bf16, __nv_bfloat16, Sm120Nt, 128, 64, 32, 3)
SM120_DEFINE_KERNEL(gemm_bi_nt_sm120_tma_128x64_bk32_s3_f16, __half, Sm120Nt, 128, 64, 32, 3)
SM120_DEFINE_KERNEL(gemm_bi_nt_sm120_tma_128x64_bk64_s2_bf16, __nv_bfloat16, Sm120Nt, 128, 64, 64, 2)
SM120_DEFINE_KERNEL(gemm_bi_nt_sm120_tma_128x64_bk64_s2_f16, __half, Sm120Nt, 128, 64, 64, 2)
SM120_DEFINE_KERNEL(gemm_bi_nt_sm120_tma_128x64_bk64_s3_bf16, __nv_bfloat16, Sm120Nt, 128, 64, 64, 3)
SM120_DEFINE_KERNEL(gemm_bi_nt_sm120_tma_128x64_bk64_s3_f16, __half, Sm120Nt, 128, 64, 64, 3)
SM120_DEFINE_KERNEL(gemm_bi_nt_sm120_tma_64x128_bk32_s2_bf16, __nv_bfloat16, Sm120Nt, 64, 128, 32, 2)
SM120_DEFINE_KERNEL(gemm_bi_nt_sm120_tma_64x128_bk32_s2_f16, __half, Sm120Nt, 64, 128, 32, 2)
SM120_DEFINE_KERNEL(gemm_bi_nt_sm120_tma_64x128_bk32_s3_bf16, __nv_bfloat16, Sm120Nt, 64, 128, 32, 3)
SM120_DEFINE_KERNEL(gemm_bi_nt_sm120_tma_64x128_bk32_s3_f16, __half, Sm120Nt, 64, 128, 32, 3)
SM120_DEFINE_KERNEL(gemm_bi_nt_sm120_tma_64x128_bk64_s2_bf16, __nv_bfloat16, Sm120Nt, 64, 128, 64, 2)
SM120_DEFINE_KERNEL(gemm_bi_nt_sm120_tma_64x128_bk64_s2_f16, __half, Sm120Nt, 64, 128, 64, 2)
SM120_DEFINE_KERNEL(gemm_bi_nt_sm120_tma_64x128_bk64_s3_bf16, __nv_bfloat16, Sm120Nt, 64, 128, 64, 3)
SM120_DEFINE_KERNEL(gemm_bi_nt_sm120_tma_64x128_bk64_s3_f16, __half, Sm120Nt, 64, 128, 64, 3)
SM120_DEFINE_KERNEL(gemm_bi_nt_sm120_tma_128x128_bk32_s2_bf16, __nv_bfloat16, Sm120Nt, 128, 128, 32, 2)
SM120_DEFINE_KERNEL(gemm_bi_nt_sm120_tma_128x128_bk32_s2_f16, __half, Sm120Nt, 128, 128, 32, 2)
SM120_DEFINE_KERNEL(gemm_bi_nt_sm120_tma_128x128_bk32_s3_bf16, __nv_bfloat16, Sm120Nt, 128, 128, 32, 3)
SM120_DEFINE_KERNEL(gemm_bi_nt_sm120_tma_128x128_bk32_s3_f16, __half, Sm120Nt, 128, 128, 32, 3)
SM120_DEFINE_KERNEL(gemm_bi_nt_sm120_tma_128x128_bk64_s2_bf16, __nv_bfloat16, Sm120Nt, 128, 128, 64, 2)
SM120_DEFINE_KERNEL(gemm_bi_nt_sm120_tma_128x128_bk64_s2_f16, __half, Sm120Nt, 128, 128, 64, 2)
SM120_DEFINE_KERNEL(gemm_bi_nt_sm120_tma_128x128_bk64_s3_bf16, __nv_bfloat16, Sm120Nt, 128, 128, 64, 3)
SM120_DEFINE_KERNEL(gemm_bi_nt_sm120_tma_128x128_bk64_s3_f16, __half, Sm120Nt, 128, 128, 64, 3)

template <int M, int N, int Stages>
struct Sm120Tf32Storage {
    static constexpr int threads = (M / 32) * (N / 32) * 32;
    static constexpr int stage_bytes = (M + N) * 32 * 4;
    static constexpr int dynamic_bytes = 128 + Stages * stage_bytes;
};

static_assert(Sm120Tf32Storage<128, 64, 2>::dynamic_bytes == 49280);
static_assert(Sm120Tf32Storage<128, 64, 3>::dynamic_bytes == 73856);
static_assert(Sm120Tf32Storage<64, 128, 2>::dynamic_bytes == 49280);
static_assert(Sm120Tf32Storage<64, 128, 3>::dynamic_bytes == 73856);
static_assert(Sm120Tf32Storage<64, 128, 4>::dynamic_bytes == 98432);
static_assert(Sm120Tf32Storage<64, 64, 2>::dynamic_bytes == 32896);

template <int Op>
static __device__ __forceinline__ int sm120_tf32_rows(
    const Sm120KernelParams& params) {
    return Op == Sm120Tn ? params.k : params.m;
}

template <int Op>
static __device__ __forceinline__ int sm120_tf32_columns(
    const Sm120KernelParams& params) {
    return Op == Sm120Nt ? params.k : params.n;
}

template <int Op>
static __device__ __forceinline__ int sm120_tf32_reduction(
    const Sm120KernelParams& params) {
    return Op == Sm120Nn ? params.k : (Op == Sm120Tn ? params.m : params.n);
}

static __device__ __forceinline__ unsigned sm120_tf32_sw128_offset(
    unsigned plane_base, unsigned logical_row, unsigned element) {
    unsigned chunk = element / 4;
    unsigned element_in_vector = element & 3;
    unsigned offset = (plane_base / 128) % 8;
    unsigned physical_chunk = chunk ^ ((logical_row + offset) % 8);
    return plane_base + logical_row * 128 + physical_chunk * 16
        + element_in_vector * 4;
}

static __device__ __forceinline__ unsigned sm120_tf32_rna(float value) {
    unsigned result;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(result) : "f"(value));
    return result;
}

static __device__ __forceinline__ void sm120_tf32_mma_m16n8k8(
    float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
}

template <int Op>
static __device__ __forceinline__ float sm120_tf32_epilogue(
    float accumulator, float old_output, const float* bias, int column,
    const Sm120KernelParams& params) {
    if constexpr (Op == Sm120Nn) {
        (void)bias;
        (void)column;
        float value = params.alpha == 1.0f
            ? accumulator
            : __fmul_rn(params.alpha, accumulator);
        if (params.beta == 0.0f) return value;
        return __fmaf_rn(params.beta, old_output, value);
    } else if constexpr (Op == Sm120Tn) {
        (void)bias;
        (void)column;
        return __fmaf_rn(params.alpha, accumulator, old_output);
    } else {
        (void)old_output;
        (void)bias;
        (void)column;
        return params.alpha == 1.0f
            ? accumulator
            : __fmul_rn(params.alpha, accumulator);
    }
}

template <int Op>
static __device__ __forceinline__ void sm120_tf32_store(
    void* output, int row, int column, float accumulator, const float* bias,
    const Sm120KernelParams& params) {
    if (row >= sm120_tf32_rows<Op>(params) ||
        column >= sm120_tf32_columns<Op>(params)) return;
    float* destination = static_cast<float*>(output) +
        static_cast<long long>(row) * params.ldc + column;
    float old_output = 0.0f;
    if constexpr (Op == Sm120Tn) {
        old_output = *destination;
    } else if constexpr (Op == Sm120Nn) {
        if (params.beta != 0.0f) old_output = *destination;
    }
    float value = sm120_tf32_epilogue<Op>(
        accumulator, old_output, bias, column, params);
#line 2001 "mamba_tf32_k0_zero_store"
    *destination = value;
#line 760 "sm120.cu"
}

struct Sm120Tf32PairValue {
    float first;
    float second;
};

template <int Op>
static __device__ __forceinline__ void sm120_tf32_store_pair(
    void* output, int row, int column, Sm120Tf32PairValue accumulator,
    const float* bias, const Sm120KernelParams& params, bool full_tile) {
    if (full_tile) {
        float* destination = static_cast<float*>(output)
            + static_cast<long long>(row) * params.ldc + column;
        float2 old_pair = *reinterpret_cast<const float2*>(destination);
        float first = sm120_tf32_epilogue<Op>(
            accumulator.first, old_pair.x, bias, column, params);
        float second = sm120_tf32_epilogue<Op>(
            accumulator.second, old_pair.y, bias, column + 1, params);
        *reinterpret_cast<float2*>(destination) = make_float2(first, second);
        return;
    }
    sm120_tf32_store<Op>(
        output, row, column, accumulator.first, bias, params);
    sm120_tf32_store<Op>(
        output, row, column + 1, accumulator.second, bias, params);
}

template <int Op, int M, int N>
static __device__ __forceinline__ void sm120_tf32_zero_reduction_epilogue(
    void* output, const float* bias, const Sm120KernelParams& params) {
    (void)&sm120_tf32_epilogue<Op>;
    int columns = sm120_tf32_columns<Op>(params);
    int column_tiles = 1 + (columns - 1) / N;
    int output_row = (int)blockIdx.x / column_tiles * M;
    int output_column = (int)blockIdx.x % column_tiles * N;
    for (int linear = (int)threadIdx.x;
         linear < M * N;
         linear += (int)blockDim.x) {
        int row = output_row + linear / N;
        int column = output_column + linear % N;
        if (row < sm120_tf32_rows<Op>(params) && column < columns) {
            float accumulator =
                Op == Sm120Nn && bias != nullptr ? bias[column] : 0.0f;
            sm120_tf32_store<Op>(output, row, column, accumulator, bias, params);
        }
    }
}

struct Sm120Tf32StageContext {
    const CUtensorMap* a_map;
    const CUtensorMap* b_map;
    const Sm120KernelParams* params;
    unsigned payload;
    unsigned full_base;
    int output_row;
    int output_column;
};

// Loads k-tile `k_tile` of the tile named by `context` into stage slot
// `slot % Stages`. The tiled kernels pass the same index for both; a
// persistent kernel that walks several tiles on one pipeline does not.
template <int Op, int M, int N, int Stages>
static __device__ __forceinline__ void sm120_tf32_produce_stage_at(
    const Sm120Tf32StageContext& context, int slot, int k_tile) {
    constexpr int plane_bytes = 4096;
    constexpr int stage_bytes = Sm120Tf32Storage<M, N, Stages>::stage_bytes;
    unsigned stage_index = (unsigned)(slot % Stages);
    unsigned stage = context.payload + stage_index * stage_bytes;
    unsigned barrier = context.full_base + stage_index * 8;
    unsigned a_destination = stage;
    unsigned b_destination = stage + M * 32 * 4;
    unsigned long long a_descriptor =
        reinterpret_cast<unsigned long long>(context.a_map);
    unsigned long long b_descriptor =
        reinterpret_cast<unsigned long long>(context.b_map);
    const Sm120KernelParams& params = *context.params;
    int reduction = k_tile * 32;
    sm120_expect_transaction<Sm120Tf32Storage<M, N, Stages>::stage_bytes>(barrier);
    if constexpr (Op == Sm120Nn) {
        sm120_tma_copy(a_destination, a_descriptor, reduction, context.output_row,
            params.a_x, params.a_y, barrier);
#pragma unroll
        for (int plane = 0; plane < N / 32; ++plane) {
            sm120_tma_copy(b_destination + plane * plane_bytes, b_descriptor,
                context.output_column + plane * 32, reduction,
                params.b_x, params.b_y, barrier);
        }
    } else if constexpr (Op == Sm120Tn) {
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
    } else {
        sm120_tma_copy(a_destination, a_descriptor, reduction, context.output_row,
            params.a_x, params.a_y, barrier);
        sm120_tma_copy(b_destination, b_descriptor, reduction, context.output_column,
            params.b_x, params.b_y, barrier);
    }
}

template <int Op, int M, int N, int Stages>
static __device__ __forceinline__ void sm120_tf32_produce_stage(
    const Sm120Tf32StageContext& context, int tile) {
    sm120_tf32_produce_stage_at<Op, M, N, Stages>(context, tile, tile);
}

template <int Op, int M, int N, int Stages>
static __device__ __forceinline__ float sm120_tf32_load_a(
    unsigned char* storage, int stage, int row, int reduction) {
    constexpr int stage_bytes = Sm120Tf32Storage<M, N, Stages>::stage_bytes;
    unsigned plane = (unsigned)(row / 32) * 4096U;
    unsigned logical_row = (unsigned)(row & 31);
    unsigned element = (unsigned)reduction;
    if constexpr (Op == Sm120Tn) {
        logical_row = (unsigned)reduction;
        element = (unsigned)(row & 31);
    }
    unsigned offset = sm120_tf32_sw128_offset(plane, logical_row, element);
    return *reinterpret_cast<float*>(storage + stage * stage_bytes + offset);
}

template <int Op, int M, int N, int Stages>
static __device__ __forceinline__ float sm120_tf32_load_b(
    unsigned char* storage, int stage, int reduction, int column) {
    constexpr int stage_bytes = Sm120Tf32Storage<M, N, Stages>::stage_bytes;
    unsigned base = M * 32 * 4;
    unsigned plane = base + (unsigned)(column / 32) * 4096U;
    unsigned logical_row = (unsigned)reduction;
    unsigned element = (unsigned)(column & 31);
    if constexpr (Op == Sm120Nt) {
        logical_row = (unsigned)(column & 31);
        element = (unsigned)reduction;
    }
    unsigned offset = sm120_tf32_sw128_offset(plane, logical_row, element);
    return *reinterpret_cast<float*>(storage + stage * stage_bytes + offset);
}

template <int Op, int M, int N, int Stages>
static __device__ __forceinline__ void sm120_tf32_load_issue(
    unsigned char* storage, int stage, int warp_m, int warp_n, int k8,
    unsigned (&a_fragments)[2][4], unsigned (&b_fragments)[4][2]) {
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
    for (int n_atom = 0; n_atom < 4; ++n_atom) {
        int column = warp_n + n_atom * 8 + group;
        b_fragments[n_atom][0] = sm120_tf32_rna(
            sm120_tf32_load_b<Op, M, N, Stages>(storage, stage, k8 + thread, column));
        b_fragments[n_atom][1] = sm120_tf32_rna(
            sm120_tf32_load_b<Op, M, N, Stages>(storage, stage, k8 + thread + 4, column));
    }
}

template <int Op, int M, int N, int Stages>
static __device__ __forceinline__ void sm120_tf32_issue_stage(
    unsigned char* storage, int stage, int warp_m, int warp_n,
    float (&accumulator)[2][4][4]) {
    unsigned a_fragments[2][2][4];
    unsigned b_fragments[2][4][2];
    sm120_tf32_load_issue<Op, M, N, Stages>(
        storage, stage, warp_m, warp_n, 0, a_fragments[0], b_fragments[0]);
#pragma unroll
    for (int issue = 0; issue < 4; ++issue) {
        int k8 = issue * 8;
        int current = issue & 1;
        int next = current ^ 1;
        if (issue + 1 < 4) {
            sm120_tf32_load_issue<Op, M, N, Stages>(storage, stage,
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

template <int Op, int M, int N, int Stages>
static __device__ __forceinline__ void sm120_tf32_kernel(
    void* output, const CUtensorMap& a_map, const CUtensorMap& b_map,
    const float* bias, const Sm120KernelParams& params) {
    assert(params.alpha == 1.0f || bias == nullptr);
    extern __shared__ __align__(1024) unsigned char storage[];
    constexpr int stage_bytes = Sm120Tf32Storage<M, N, Stages>::stage_bytes;
    constexpr int warps = Sm120Tf32Storage<M, N, Stages>::threads / 32;
    unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(storage));
    unsigned payload = shared;
    unsigned full_base = shared + Stages * stage_bytes;
    unsigned empty_base = full_base + 64;
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
                int column = output_column + warp_n + n_atom * 8
                    + 2 * thread + (element & 1);
                float seed = 0.0f;
                if constexpr (Op == Sm120Nn) {
                    if (column < params.n && bias != nullptr) {
                        seed = bias[column];
                    }
                }
                accumulator[m_atom][n_atom][element] = seed;
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
                sm120_tf32_produce_stage<Op, M, N, Stages>(stage_context, tile);
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
                sm120_tf32_produce_stage<Op, M, N, Stages>(stage_context, refill);
            }
        }
        if constexpr (Stages == 2) sm120_sync_warp();
    }

#pragma unroll
    for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 4; ++n_atom) {
#pragma unroll
            for (int element = 0; element < 4; ++element) {
                int row = output_row + warp_m + m_atom * 16
                    + group + (element >= 2 ? 8 : 0);
                int column = output_column + warp_n + n_atom * 8
                    + 2 * thread + (element & 1);
                sm120_tf32_store<Op>(output, row, column,
                    accumulator[m_atom][n_atom][element], bias, params);
            }
        }
    }
}

template <int Op, int M, int N, int Stages>
static __device__ __forceinline__ void sm120_tf32_pair_kernel(
    void* output, const CUtensorMap& a_map, const CUtensorMap& b_map,
    const float* bias, const Sm120KernelParams& params) {
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
                sm120_tf32_produce_stage<Op, M, N, Stages>(stage_context, tile);
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
                sm120_tf32_produce_stage<Op, M, N, Stages>(
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

template <int M, int StorageN, int LogicalBK, int Stages>
struct Sm120Tf32RectWideStorage {
    static constexpr int slabs = LogicalBK / 32;
    static constexpr int slab_bytes = (M + StorageN) * 32 * 4;
    static constexpr int stage_bytes = slabs * slab_bytes;
    static constexpr int dynamic_bytes = 128 + Stages * stage_bytes;
};

static_assert(Sm120Tf32RectWideStorage<80, 32, 64, 2>::dynamic_bytes == 57472);

template <int M, int StorageN, int LogicalBK, int Stages>
static __device__ __forceinline__ void sm120_tf32_rect_wide_produce_stage(
    const Sm120Tf32StageContext& context, int tile) {
    constexpr int slabs = Sm120Tf32RectWideStorage<
        M, StorageN, LogicalBK, Stages>::slabs;
    constexpr int slab_bytes = Sm120Tf32RectWideStorage<
        M, StorageN, LogicalBK, Stages>::slab_bytes;
    constexpr int stage_bytes = Sm120Tf32RectWideStorage<
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
static __device__ __forceinline__ float sm120_tf32_rect_wide_load_a(
    unsigned char* storage, int stage, int row, int reduction) {
    constexpr int slab_bytes = Sm120Tf32RectWideStorage<
        M, StorageN, LogicalBK, Stages>::slab_bytes;
    constexpr int stage_bytes = Sm120Tf32RectWideStorage<
        M, StorageN, LogicalBK, Stages>::stage_bytes;
    int slab = reduction / 32;
    unsigned slab_base = (unsigned)(slab * slab_bytes);
    unsigned offset = sm120_tf32_sw128_offset(
        slab_base, (unsigned)row, (unsigned)(reduction & 31));
    return *reinterpret_cast<float*>(
        storage + stage * stage_bytes + offset);
}

template <int M, int StorageN, int LogicalBK, int Stages>
static __device__ __forceinline__ float sm120_tf32_rect_wide_load_b(
    unsigned char* storage, int stage, int reduction, int column) {
    constexpr int slab_bytes = Sm120Tf32RectWideStorage<
        M, StorageN, LogicalBK, Stages>::slab_bytes;
    constexpr int stage_bytes = Sm120Tf32RectWideStorage<
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
static __device__ __forceinline__ void sm120_tf32_rect_wide_load_issue(
    unsigned char* storage, int stage, int warp_m, int k8,
    unsigned (&a_fragment)[1][4], unsigned (&b_fragment)[NAtoms][2]) {
    int lane = (int)threadIdx.x & 31;
    int group = lane >> 2;
    int thread = lane & 3;
    int row = warp_m + group;
    a_fragment[0][0] = sm120_tf32_rna(
        sm120_tf32_rect_wide_load_a<M, StorageN, LogicalBK, Stages>(
            storage, stage, row, k8 + thread));
    a_fragment[0][1] = sm120_tf32_rna(
        sm120_tf32_rect_wide_load_a<M, StorageN, LogicalBK, Stages>(
            storage, stage, row + 8, k8 + thread));
    a_fragment[0][2] = sm120_tf32_rna(
        sm120_tf32_rect_wide_load_a<M, StorageN, LogicalBK, Stages>(
            storage, stage, row, k8 + thread + 4));
    a_fragment[0][3] = sm120_tf32_rna(
        sm120_tf32_rect_wide_load_a<M, StorageN, LogicalBK, Stages>(
            storage, stage, row + 8, k8 + thread + 4));
#pragma unroll
    for (int n_atom = 0; n_atom < NAtoms; ++n_atom) {
        int column = n_atom * 8 + group;
        b_fragment[n_atom][0] = sm120_tf32_rna(
            sm120_tf32_rect_wide_load_b<M, StorageN, LogicalBK, Stages>(
                storage, stage, k8 + thread, column));
        b_fragment[n_atom][1] = sm120_tf32_rna(
            sm120_tf32_rect_wide_load_b<M, StorageN, LogicalBK, Stages>(
                storage, stage, k8 + thread + 4, column));
    }
}

template <int NAtoms, int M, int StorageN, int LogicalBK, int Stages>
static __device__ __forceinline__ void sm120_tf32_rect_wide_issue_stage(
    unsigned char* storage, int stage, int warp_m,
    float (&accumulator)[1][NAtoms][4]) {
    unsigned a_fragments[2][1][4];
    unsigned b_fragments[2][NAtoms][2];
    sm120_tf32_rect_wide_load_issue<NAtoms, M, StorageN, LogicalBK, Stages>(
        storage, stage, warp_m, 0, a_fragments[0], b_fragments[0]);
#pragma unroll
    for (int issue = 0; issue < LogicalBK / 8; ++issue) {
        int current = issue & 1;
        int next = current ^ 1;
        if (issue + 1 < LogicalBK / 8) {
            sm120_tf32_rect_wide_load_issue<
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

template <int M, int N, int StorageN, int LogicalBK, int Stages>
static __device__ __forceinline__ void sm120_tf32_rect_wide_kernel(
    void* output, const CUtensorMap& a_map, const CUtensorMap& b_map,
    const float* bias, const Sm120KernelParams& params) {
    constexpr int NAtoms = (N + 7) / 8;
    constexpr int stage_bytes = Sm120Tf32RectWideStorage<
        M, StorageN, LogicalBK, Stages>::stage_bytes;
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
            sm120_init_barrier<5>(empty_base + stage * 8);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    if (warp == 0 && lane == 0) {
#pragma unroll
        for (int tile = 0; tile < Stages; ++tile) {
            if (tile < tile_count) {
                sm120_tf32_rect_wide_produce_stage<
                    M, StorageN, LogicalBK, Stages>(stage_context, tile);
            }
        }
    }
    if constexpr (Stages == 2) sm120_sync_warp();

    int warp_m = warp * 16;
    int group = lane >> 2;
    int thread = lane & 3;
    // The bias rides in the accumulator seed, the way the other NN routes
    // carry it, so the epilogue stays a plain scale.
    float accumulator[1][NAtoms][4];
#pragma unroll
    for (int n_atom = 0; n_atom < NAtoms; ++n_atom) {
#pragma unroll
        for (int element = 0; element < 4; ++element) {
            int column = output_column + n_atom * 8 + 2 * thread + (element & 1);
            float seed = 0.0f;
            if (column < params.n && bias != nullptr) seed = bias[column];
            accumulator[0][n_atom][element] = seed;
        }
    }

    for (int tile = 0; tile < tile_count; ++tile) {
        int stage = tile % Stages;
        unsigned generation = (unsigned)(tile / Stages);
        sm120_wait_barrier(full_base + stage * 8, generation & 1U);
        sm120_tf32_rect_wide_issue_stage<
            NAtoms, M, StorageN, LogicalBK, Stages>(
                storage, stage, warp_m, accumulator);
        if (lane == 0) sm120_arrive_empty(empty_base + stage * 8);
        if (warp == 0 && lane == 0) {
            int refill = tile + Stages;
            if (refill < tile_count) {
                sm120_wait_barrier(
                    empty_base + stage * 8, generation & 1U);
                sm120_tf32_rect_wide_produce_stage<
                    M, StorageN, LogicalBK, Stages>(stage_context, refill);
            }
        }
        if constexpr (Stages == 2) sm120_sync_warp();
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


template <int Op, int M, int N, int Stages>
static __device__ __forceinline__ void sm120_tf32_entry(
    void* output, const CUtensorMap& a_map, const CUtensorMap& b_map,
    const float* bias, const Sm120KernelParams& params) {
    int reduction = sm120_tf32_reduction<Op>(params);
#line 1001 "mamba_tf32_k0_guard"
    bool zero_reduction = reduction == 0;
#line 1002 "mamba_tf32_k0_branch"
    if (zero_reduction) {
        sm120_tf32_zero_reduction_epilogue<Op, M, N>(output, bias, params);
        return;
    }
#line 1120 "sm120.cu"
    sm120_tf32_kernel<Op, M, N, Stages>(output, a_map, b_map, bias, params);
}

static __device__ __forceinline__ void sm120_tf32_rect_wide_entry(
    void* output, const CUtensorMap& a_map, const CUtensorMap& b_map,
    const float* bias, const Sm120KernelParams& params) {
    assert(params.alpha == 1.0f || bias == nullptr);
    if (params.k == 0) {
        sm120_tf32_zero_reduction_epilogue<Sm120Nn, 80, 32>(
            output, bias, params);
        return;
    }
    sm120_tf32_rect_wide_kernel<80, 32, 32, 64, 2>(
        output, a_map, b_map, bias, params);
}

template <int Op, int M, int N, int Stages>
static __device__ __forceinline__ void sm120_tf32_pair_entry(
    void* output, const CUtensorMap& a_map, const CUtensorMap& b_map,
    const float* bias, const Sm120KernelParams& params) {
    if (sm120_tf32_reduction<Op>(params) == 0) {
        sm120_tf32_zero_reduction_epilogue<Op, M, N>(output, bias, params);
        return;
    }
    sm120_tf32_pair_kernel<Op, M, N, Stages>(
        output, a_map, b_map, bias, params);
}


// TN stream-K. The reduction of every output tile is cut into K-tile units
// and the whole (tile, k) space is dealt to a persistent grid in equal
// contiguous ranges, so the machine stays full whatever the tile count. A
// CTA whose range covers a whole tile stores it directly. A tile that
// straddles CTAs is combined by the CTA holding its last unit, which adds
// the lower contributors' slabs in ascending CTA order and its own value
// last, all with __fadd_rn: the partition points are a pure function of the
// shape and the grid, so the bits are deterministic and replay-stable.
//
// Each CTA walks its range from the end, so the segment it publishes comes
// first and the one it owns last: a publisher never waits, and an owner
// waits only on slabs that are already being produced. The pipeline is one
// for the whole range: the producer lane runs `Stages` steps ahead across
// segment boundaries, and the barriers are armed once per launch.
//
// Contributors publish through fragment-order slabs and per-slot flags with
// release/acquire at GPU scope; owners clear the flags they consumed, so the
// workspace is reusable across launches without a reset. The grid must be
// no larger than the device keeps resident at once, so that every wait is
// on a CTA that is running; the host derives the grid from that.

static constexpr int SM120_STREAMK_ACCUMULATORS = 32;
static constexpr int SM120_STREAMK_SLOTS = 2;

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

// A cursor over one CTA's stage sequence: the segments of its range from the
// end, each ascending in k. `unit` is the absolute unit of the current stage.
struct Sm120StreamKCursor {
    int unit;
    int segment_begin;
    int segment_end;
    int tile;
};

static __device__ __forceinline__ void sm120_streamk_cursor_open(
    Sm120StreamKCursor& cursor, int range_first, int range_end, int k_tiles) {
    cursor.tile = (range_end - 1) / k_tiles;
    cursor.segment_begin = max(cursor.tile * k_tiles, range_first);
    cursor.segment_end = range_end;
    cursor.unit = cursor.segment_begin;
}

// Steps to the next stage of the sequence; false once the range is spent.
static __device__ __forceinline__ bool sm120_streamk_cursor_advance(
    Sm120StreamKCursor& cursor, int range_first, int k_tiles) {
    if (cursor.unit + 1 < cursor.segment_end) {
        ++cursor.unit;
        return true;
    }
    if (cursor.segment_begin == range_first) {
        return false;
    }
    sm120_streamk_cursor_open(cursor, range_first, cursor.segment_begin, k_tiles);
    return true;
}

static __device__ __forceinline__ void sm120_streamk_zero(
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

// Launch-constant state of the pipeline.
struct Sm120StreamKFlow {
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
static __device__ __forceinline__ void sm120_streamk_produce(
    const Sm120StreamKFlow& flow, const Sm120StreamKCursor& cursor, int step) {
    const Sm120Tf32StageContext context = {
        flow.a_map, flow.b_map, flow.params, flow.payload, flow.full_base,
        (cursor.tile / flow.column_tiles) * M, (cursor.tile % flow.column_tiles) * N};
    sm120_tf32_produce_stage_at<Sm120Tn, M, N, Stages>(
        context, step, cursor.unit - cursor.tile * flow.k_tiles);
}

// The mainloop over one segment. Steps are numbered across the whole range,
// so the stage and the barrier phase continue from the previous segment,
// and the producer lane refills with whatever the cursor holds next, which
// may already belong to the segment after this one.
template <int M, int N, int Stages>
static __device__ __forceinline__ void sm120_streamk_segment(
    unsigned char* storage, const Sm120StreamKFlow& flow,
    Sm120StreamKCursor& producer, int step_base, int tile_count,
    int warp_m, int warp_n, float (&accumulator)[2][4][4]) {
    int warp = (int)threadIdx.x >> 5;
    int lane = (int)threadIdx.x & 31;
    sm120_streamk_zero(accumulator);
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
                sm120_streamk_produce<M, N, Stages>(flow, producer, refill);
                sm120_streamk_cursor_advance(producer, flow.range_first, flow.k_tiles);
            }
        }
        if constexpr (Stages == 2) sm120_sync_warp();
    }
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
            asm volatile("st.global.cg.v4.f32 [%0], {%1, %2, %3, %4};\n"
                :: "l"(destination + m_atom * 4 + n_atom),
                   "f"(value.x), "f"(value.y), "f"(value.z), "f"(value.w) : "memory");
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

// Folds the lower contributors' slabs into the accumulators: the slabs are
// summed in ascending CTA order, then the CTA's own value is added last.
// Every slab is read through ordinary cache-global loads so the eight loads
// of one slab issue together.
static __device__ __forceinline__ void sm120_streamk_fold_slabs(
    const float* partial, long long slab_floats, long long units, int grid,
    int first_cta, int cta, int tile, int k_tiles,
    float (&accumulator)[2][4][4]) {
    const long long lane_offset = (long long)threadIdx.x * SM120_STREAMK_ACCUMULATORS;
    float sum[2][4][4];
    bool first = true;
    for (int source = first_cta; source < cta; ++source) {
        Sm120StreamKRange theirs = sm120_streamk_range(units, grid, source);
        int slot = tile == (int)(theirs.first / k_tiles) ? 0 : 1;
        const float4* slab = reinterpret_cast<const float4*>(
            partial + ((long long)source * SM120_STREAMK_SLOTS + slot) * slab_floats
            + lane_offset);
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

// The pair epilogue with every read of the old tile issued before the first
// write; a partial tile at the edge stores element by element.
template <int M, int N>
static __device__ __forceinline__ void sm120_streamk_epilogue(
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
                        output, row, column, pair, bias, params, false);
                }
            }
        }
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
                    accumulator[m_atom][n_atom][half * 2 + 1], pair.y, bias, column + 1,
                    params);
                *reinterpret_cast<float2*>(
                    base + static_cast<long long>(row) * params.ldc + column) =
                    make_float2(first, second);
            }
        }
    }
}

// The persistent grid walks every tile of an empty reduction.
template <int M, int N>
static __device__ __forceinline__ void sm120_streamk_zero_reduction(
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
static __device__ __forceinline__ void sm120_tf32_tn_streamk_kernel(
    void* output, float* partial, unsigned* flags,
    const CUtensorMap& a_map, const CUtensorMap& b_map,
    const float* bias, const Sm120KernelParams& params) {
    constexpr int Op = Sm120Tn;
    constexpr int threads = Sm120Tf32Storage<M, N, Stages>::threads;
    constexpr int warps = threads / 32;
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
    int k_tiles = 1 + (reduction - 1) / 32;
    long long units = (long long)row_tiles * column_tiles * k_tiles;
    int grid = (int)gridDim.x;
    int cta = (int)blockIdx.x;
    Sm120StreamKRange mine = sm120_streamk_range(units, grid, cta);
    int range_first = (int)mine.first;
    int range_end = (int)mine.last;
    int first_tile = range_first / k_tiles;
    const Sm120StreamKFlow flow = {
        &a_map, &b_map, &params, payload, full_base, empty_base,
        column_tiles, k_tiles, range_first, range_end - range_first};
    int warp = (int)threadIdx.x >> 5;
    int lane = (int)threadIdx.x & 31;
    constexpr int warp_columns = N / 32;
    int warp_m = (warp / warp_columns) * 32;
    int warp_n = (warp % warp_columns) * 32;
    float accumulator[2][4][4];

    if (threadIdx.x == 0) {
#pragma unroll
        for (int stage = 0; stage < Stages; ++stage) {
            sm120_init_barrier<1>(full_base + stage * 8);
            sm120_init_barrier<warps>(empty_base + stage * 8);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    Sm120StreamKCursor producer;
    sm120_streamk_cursor_open(producer, range_first, range_end, k_tiles);
    if (warp == 0 && lane == 0) {
#pragma unroll
        for (int step = 0; step < Stages; ++step) {
            if (step < flow.total) {
                sm120_streamk_produce<M, N, Stages>(flow, producer, step);
                sm120_streamk_cursor_advance(producer, range_first, k_tiles);
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
        sm120_streamk_segment<M, N, Stages>(
            storage, flow, producer, step_base, k_end - k_begin,
            warp_m, warp_n, accumulator);
        step_base += k_end - k_begin;
        bool covers_start = k_begin == 0;
        bool covers_end = k_end == k_tiles;
        if (covers_start && covers_end) {
            sm120_streamk_epilogue<M, N>(
                output, rows, columns, output_row, output_column,
                warp_m, warp_n, bias, params, accumulator);
        } else if (!covers_end) {
            int slot = tile == first_tile ? 0 : 1;
            sm120_streamk_store_slab(
                partial + ((long long)cta * SM120_STREAMK_SLOTS + slot) * slab_floats,
                accumulator);
            __syncthreads();
            if (threadIdx.x == 0) {
                __threadfence();
                sm120_streamk_raise(flags + (long long)cta * SM120_STREAMK_SLOTS + slot);
            }
        } else {
            int first_cta = sm120_streamk_cta_of(units, grid, (long long)tile * k_tiles);
            int sources = cta - first_cta;
            for (int index = (int)threadIdx.x; index < sources; index += threads) {
                int source = first_cta + index;
                Sm120StreamKRange theirs = sm120_streamk_range(units, grid, source);
                int slot = tile == (int)(theirs.first / k_tiles) ? 0 : 1;
                sm120_streamk_await(flags + (long long)source * SM120_STREAMK_SLOTS + slot);
            }
            __syncthreads();
            sm120_streamk_fold_slabs(
                partial, slab_floats, units, grid, first_cta, cta, tile, k_tiles,
                accumulator);
            __syncthreads();
            for (int index = (int)threadIdx.x; index < sources; index += threads) {
                int source = first_cta + index;
                Sm120StreamKRange theirs = sm120_streamk_range(units, grid, source);
                int slot = tile == (int)(theirs.first / k_tiles) ? 0 : 1;
                sm120_streamk_clear(flags + (long long)source * SM120_STREAMK_SLOTS + slot);
            }
            sm120_streamk_epilogue<M, N>(
                output, rows, columns, output_row, output_column,
                warp_m, warp_n, bias, params, accumulator);
        }
        unit -= k_end - k_begin;
    }
}

template <int M, int N, int Stages>
static __device__ __forceinline__ void sm120_tf32_tn_streamk_entry(
    void* output, float* partial, unsigned* flags,
    const CUtensorMap& a_map, const CUtensorMap& b_map,
    const float* bias, const Sm120KernelParams& params) {
    if (sm120_tf32_reduction<Sm120Tn>(params) == 0) {
        sm120_streamk_zero_reduction<M, N>(output, bias, params);
        return;
    }
    sm120_tf32_tn_streamk_kernel<M, N, Stages>(
        output, partial, flags, a_map, b_map, bias, params);
}

// ---------------------------------------------------------------------------
// Half TN over the stream-K schedule. The training batch runs its reduction
// ten thousand rows deep over a few dozen output tiles, so the tiled kernel
// leaves most of the device idle; here every CTA of a persistent grid walks
// a contiguous range of (tile, k-tile) units dealt by the same formula the
// TF32 kernel uses, stores a partial slab where its range ends inside a
// tile, and the CTA that finishes a tile folds the lower contributors' slabs
// in ascending CTA order before the ordinary epilogue. The fold order is a
// function of the shape and the grid alone, so the output is bit-stable
// run to run and independent of scheduling.
// ---------------------------------------------------------------------------

template <int MAtoms>
static __device__ __forceinline__ void sm120_half_streamk_zero(
    float (&accumulator)[MAtoms][4][4]) {
#pragma unroll
    for (int fm = 0; fm < MAtoms; ++fm) {
#pragma unroll
        for (int fn = 0; fn < 4; ++fn) {
#pragma unroll
            for (int element = 0; element < 4; ++element) {
                accumulator[fm][fn][element] = 0.0f;
            }
        }
    }
}

template <int MAtoms>
static __device__ __forceinline__ void sm120_half_streamk_store_slab(
    float* slab, const float (&accumulator)[MAtoms][4][4]) {
    float4* destination = reinterpret_cast<float4*>(
        slab + (long long)threadIdx.x * (MAtoms * 16));
#pragma unroll
    for (int fm = 0; fm < MAtoms; ++fm) {
#pragma unroll
        for (int fn = 0; fn < 4; ++fn) {
            float4 value = make_float4(
                accumulator[fm][fn][0], accumulator[fm][fn][1],
                accumulator[fm][fn][2], accumulator[fm][fn][3]);
            asm volatile("st.global.cg.v4.f32 [%0], {%1, %2, %3, %4};\n"
                :: "l"(destination + fm * 4 + fn),
                   "f"(value.x), "f"(value.y), "f"(value.z), "f"(value.w) : "memory");
        }
    }
}

// Sums the lower contributors' slabs in ascending CTA order and adds the
// CTA's own value last, exactly as the TF32 fold does.
template <int MAtoms>
static __device__ __forceinline__ void sm120_half_streamk_fold_slabs(
    const float* partial, long long slab_floats, long long units, int grid,
    int first_cta, int cta, int tile, int k_tiles,
    float (&accumulator)[MAtoms][4][4]) {
    const long long lane_offset = (long long)threadIdx.x * (MAtoms * 16);
    float sum[MAtoms][4][4];
    bool first = true;
    for (int source = first_cta; source < cta; ++source) {
        Sm120StreamKRange theirs = sm120_streamk_range(units, grid, source);
        int slot = tile == (int)(theirs.first / k_tiles) ? 0 : 1;
        const float4* slab = reinterpret_cast<const float4*>(
            partial + ((long long)source * SM120_STREAMK_SLOTS + slot) * slab_floats
            + lane_offset);
        float4 value[MAtoms][4];
#pragma unroll
        for (int fm = 0; fm < MAtoms; ++fm) {
#pragma unroll
            for (int fn = 0; fn < 4; ++fn) {
                value[fm][fn] = __ldcg(slab + fm * 4 + fn);
            }
        }
#pragma unroll
        for (int fm = 0; fm < MAtoms; ++fm) {
#pragma unroll
            for (int fn = 0; fn < 4; ++fn) {
                float4 v = value[fm][fn];
                if (first) {
                    sum[fm][fn][0] = v.x;
                    sum[fm][fn][1] = v.y;
                    sum[fm][fn][2] = v.z;
                    sum[fm][fn][3] = v.w;
                } else {
                    sum[fm][fn][0] = __fadd_rn(sum[fm][fn][0], v.x);
                    sum[fm][fn][1] = __fadd_rn(sum[fm][fn][1], v.y);
                    sum[fm][fn][2] = __fadd_rn(sum[fm][fn][2], v.z);
                    sum[fm][fn][3] = __fadd_rn(sum[fm][fn][3], v.w);
                }
            }
        }
        first = false;
    }
#pragma unroll
    for (int fm = 0; fm < MAtoms; ++fm) {
#pragma unroll
        for (int fn = 0; fn < 4; ++fn) {
#pragma unroll
            for (int element = 0; element < 4; ++element) {
                accumulator[fm][fn][element] = __fadd_rn(
                    sum[fm][fn][element], accumulator[fm][fn][element]);
            }
        }
    }
}

struct Sm120HalfStreamKFlow {
    const CUtensorMap* a_map;
    const CUtensorMap* b_map;
    const Sm120KernelParams* params;
    Sm120Pipeline pipeline;
    int column_tiles;
    int k_tiles;
    int range_first;
    int total;
};

// Refills the stage of `step` with the unit the cursor holds, which may
// already belong to the next tile of the range.
template <int M, int N, int BK, int Stages>
static __device__ __forceinline__ void sm120_half_streamk_produce(
    const Sm120HalfStreamKFlow& flow, const Sm120StreamKCursor& cursor, int step) {
    Sm120Pipeline at = flow.pipeline;
    at.output_row = (cursor.tile / flow.column_tiles) * M;
    at.output_col = (cursor.tile % flow.column_tiles) * N;
    sm120_produce_stage_at<Sm120Tn, M, N, BK, Stages>(
        *flow.a_map, *flow.b_map, *flow.params, at, step,
        cursor.unit - cursor.tile * flow.k_tiles);
}

// The mainloop over one segment; steps are numbered across the whole range
// so the stage and the barrier phase carry over from the previous segment.
template <typename T, int M, int N, int BK, int Stages, int MAtoms>
static __device__ __forceinline__ void sm120_half_streamk_segment(
    const Sm120HalfStreamKFlow& flow, Sm120StreamKCursor& producer,
    int step_base, int tile_count, int warp_m, int warp_n,
    float (&accumulator)[MAtoms][4][4]) {
    int warp = (int)threadIdx.x >> 5;
    int lane = (int)threadIdx.x & 31;
    sm120_half_streamk_zero<MAtoms>(accumulator);
    for (int tile = 0; tile < tile_count; ++tile) {
        int step = step_base + tile;
        int stage = step % Stages;
        unsigned phase = (unsigned)(step / Stages) & 1U;
        sm120_wait_barrier(flow.pipeline.full + stage * 8, phase);
        sm120_issue_stage<T, Sm120Tn, M, N, BK, Stages, MAtoms, false>(
            flow.pipeline.payload, step, warp_m, warp_n, accumulator);
        sm120_sync_warp();
        if (lane == 0) {
            sm120_arrive_empty(flow.pipeline.empty + stage * 8);
        }
        if (warp == 0 && lane == 0) {
            int refill = step + Stages;
            if (refill < flow.total) {
                sm120_wait_barrier(flow.pipeline.empty + stage * 8, phase);
                sm120_half_streamk_produce<M, N, BK, Stages>(flow, producer, refill);
                sm120_streamk_cursor_advance(producer, flow.range_first, flow.k_tiles);
            }
        }
        sm120_sync_warp();
    }
}

template <typename T, int M, int N, int BK, int Stages>
static __device__ __forceinline__ void sm120_half_tn_streamk_kernel(
    void* output, float* partial, unsigned* flags,
    const CUtensorMap& a_map, const CUtensorMap& b_map,
    const Sm120KernelParams& params) {
    constexpr int Op = Sm120Tn;
    constexpr int threads = Sm120Storage<M, N, BK, Stages>::threads;
    constexpr int warps = threads / 32;
    constexpr int m_atoms =
        Sm120Storage<M, N, BK, Stages>::wide_m_warp ? 4 : 2;
    constexpr long long slab_floats = (long long)threads * (m_atoms * 16);
    extern __shared__ __align__(128) unsigned char storage[];
    unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(storage));
    int rows = params.k;
    int columns = params.n;
    int reduction = params.m;
    int column_tiles = 1 + (columns - 1) / N;
    int row_tiles = 1 + (rows - 1) / M;
    int k_tiles = 1 + (reduction - 1) / BK;
    long long units = (long long)row_tiles * column_tiles * k_tiles;
    int grid = (int)gridDim.x;
    int cta = (int)blockIdx.x;
    Sm120StreamKRange mine = sm120_streamk_range(units, grid, cta);
    int range_first = (int)mine.first;
    int range_end = (int)mine.last;
    int first_tile = range_first / k_tiles;
    const Sm120HalfStreamKFlow flow = {
        &a_map, &b_map, &params,
        {shared + 128, shared, shared + 64, 0, 0},
        column_tiles, k_tiles, range_first, range_end - range_first};
    int warp = (int)threadIdx.x >> 5;
    int lane = (int)threadIdx.x & 31;
    int warp_columns = N / 32;
    int warp_m = (warp / warp_columns) * (m_atoms * 16);
    int warp_n = (warp % warp_columns) * 32;
    float accumulator[m_atoms][4][4];

    if (threadIdx.x == 0) {
#pragma unroll
        for (int stage = 0; stage < Stages; ++stage) {
            sm120_init_barrier<1>(flow.pipeline.full + stage * 8);
            sm120_init_barrier<warps>(flow.pipeline.empty + stage * 8);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    Sm120StreamKCursor producer;
    sm120_streamk_cursor_open(producer, range_first, range_end, k_tiles);
    if (warp == 0 && lane == 0) {
#pragma unroll
        for (int step = 0; step < Stages; ++step) {
            if (step < flow.total) {
                sm120_half_streamk_produce<M, N, BK, Stages>(flow, producer, step);
                sm120_streamk_cursor_advance(producer, range_first, k_tiles);
            }
        }
    }
    sm120_sync_warp();

    int step_base = 0;
    for (int unit = range_end; unit > range_first;) {
        int tile = (unit - 1) / k_tiles;
        int k_end = unit - tile * k_tiles;
        int k_begin = max(0, k_end - (unit - range_first));
        int output_row = (tile / column_tiles) * M;
        int output_column = (tile % column_tiles) * N;
        sm120_half_streamk_segment<T, M, N, BK, Stages, m_atoms>(
            flow, producer, step_base, k_end - k_begin, warp_m, warp_n,
            accumulator);
        step_base += k_end - k_begin;
        bool covers_start = k_begin == 0;
        bool covers_end = k_end == k_tiles;
        Sm120Output destination = {
            output, params.alpha, params.beta, rows, columns, params.ldc,
            output_row, output_column, warp_columns};
        if (covers_start && covers_end) {
            sm120_epilogue<T, Op, M, N, BK, Stages, m_atoms>(
                destination, accumulator);
        } else if (!covers_end) {
            int slot = tile == first_tile ? 0 : 1;
            sm120_half_streamk_store_slab<m_atoms>(
                partial + ((long long)cta * SM120_STREAMK_SLOTS + slot) * slab_floats,
                accumulator);
            __syncthreads();
            if (threadIdx.x == 0) {
                __threadfence();
                sm120_streamk_raise(flags + (long long)cta * SM120_STREAMK_SLOTS + slot);
            }
        } else {
            int first_cta = sm120_streamk_cta_of(units, grid, (long long)tile * k_tiles);
            int sources = cta - first_cta;
            for (int index = (int)threadIdx.x; index < sources; index += threads) {
                int source = first_cta + index;
                Sm120StreamKRange theirs = sm120_streamk_range(units, grid, source);
                int slot = tile == (int)(theirs.first / k_tiles) ? 0 : 1;
                sm120_streamk_await(flags + (long long)source * SM120_STREAMK_SLOTS + slot);
            }
            __syncthreads();
            sm120_half_streamk_fold_slabs<m_atoms>(
                partial, slab_floats, units, grid, first_cta, cta, tile, k_tiles,
                accumulator);
            __syncthreads();
            for (int index = (int)threadIdx.x; index < sources; index += threads) {
                int source = first_cta + index;
                Sm120StreamKRange theirs = sm120_streamk_range(units, grid, source);
                int slot = tile == (int)(theirs.first / k_tiles) ? 0 : 1;
                sm120_streamk_clear(flags + (long long)source * SM120_STREAMK_SLOTS + slot);
            }
            sm120_epilogue<T, Op, M, N, BK, Stages, m_atoms>(
                destination, accumulator);
        }
        unit -= k_end - k_begin;
    }
}

// One resident CTA per multiprocessor, as the TF32 stream-K kernel: the
// grid is the multiprocessor count and every wait is on a running CTA.
#define SM120_DEFINE_HALF_TN_STREAMK_KERNEL(NAME, TYPE, M, N, BK, STAGES)   \
extern "C" __global__                                                       \
__launch_bounds__(Sm120Storage<M, N, BK, STAGES>::threads, 1) void NAME(     \
    void* output, float* partial, unsigned* flags,                             \
    const __grid_constant__ CUtensorMap a_map,                                 \
    const __grid_constant__ CUtensorMap b_map, const float* bias,              \
    const __grid_constant__ Sm120KernelParams params) {                        \
    (void)bias;                                                                \
    sm120_half_tn_streamk_kernel<TYPE, M, N, BK, STAGES>(                      \
        output, partial, flags, a_map, b_map, params);                         \
}
SM120_DEFINE_HALF_TN_STREAMK_KERNEL(gemm_bi_tn_sm120_tma_64x64_bk64_s3_streamk_bf16, __nv_bfloat16, 64, 64, 64, 3)
SM120_DEFINE_HALF_TN_STREAMK_KERNEL(gemm_bi_tn_sm120_tma_64x64_bk64_s3_streamk_f16, __half, 64, 64, 64, 3)

#define SM120_DEFINE_TF32_KERNEL(NAME, OP, M, N, STAGES)                     \
extern "C" __global__ __launch_bounds__((M * N) / 32) void NAME(            \
    void* output, const __grid_constant__ CUtensorMap a_map,                   \
    const __grid_constant__ CUtensorMap b_map, const float* bias,              \
    const __grid_constant__ Sm120KernelParams params) {                        \
    sm120_tf32_entry<OP, M, N, STAGES>(output, a_map, b_map, bias, params);    \
}

#define SM120_DEFINE_TF32_PAIR_KERNEL(NAME, OP, M, N, STAGES)                \
extern "C" __global__ __launch_bounds__((M * N) / 32) void NAME(            \
    void* output, const __grid_constant__ CUtensorMap a_map,                  \
    const __grid_constant__ CUtensorMap b_map, const float* bias,             \
    const __grid_constant__ Sm120KernelParams params) {                       \
    sm120_tf32_pair_entry<OP, M, N, STAGES>(                                 \
        output, a_map, b_map, bias, params);                                  \
}

SM120_DEFINE_TF32_KERNEL(gemm_bi_nn_sm120_tma_mma_tf32_v1_m128n64_bk32_s2, Sm120Nn, 128, 64, 2)
SM120_DEFINE_TF32_KERNEL(gemm_bi_nn_sm120_tma_mma_tf32_v1_m128n64_bk32_s3, Sm120Nn, 128, 64, 3)
SM120_DEFINE_TF32_KERNEL(gemm_bi_nn_sm120_tma_mma_tf32_v1_m64n128_bk32_s2, Sm120Nn, 64, 128, 2)
SM120_DEFINE_TF32_KERNEL(gemm_bi_nn_sm120_tma_mma_tf32_v1_m64n128_bk32_s3, Sm120Nn, 64, 128, 3)
SM120_DEFINE_TF32_KERNEL(gemm_bi_nn_sm120_tma_mma_tf32_v1_m64n64_bk32_s2, Sm120Nn, 64, 64, 2)
extern "C" __global__ __launch_bounds__(160)
void gemm_bi_nn_sm120_tma_mma_tf32_v1_m80n32_bk64_s2(
    void* output, const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap b_map, const float* bias,
    const __grid_constant__ Sm120KernelParams params) {
    sm120_tf32_rect_wide_entry(output, a_map, b_map, bias, params);
}
SM120_DEFINE_TF32_KERNEL(gemm_bi_tn_sm120_tma_mma_tf32_v1_m128n64_bk32_s2, Sm120Tn, 128, 64, 2)
SM120_DEFINE_TF32_KERNEL(gemm_bi_tn_sm120_tma_mma_tf32_v1_m128n64_bk32_s3, Sm120Tn, 128, 64, 3)
SM120_DEFINE_TF32_KERNEL(gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s2, Sm120Tn, 64, 128, 2)
SM120_DEFINE_TF32_KERNEL(gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s3, Sm120Tn, 64, 128, 3)
SM120_DEFINE_TF32_PAIR_KERNEL(gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s4_pair, Sm120Tn, 64, 128, 4)

// One resident CTA per multiprocessor: the register file is the kernel's
// alone, and the compiler schedules the mainloop for that.
#define SM120_DEFINE_TF32_TN_STREAMK_KERNEL(NAME, M, N, STAGES)              \
extern "C" __global__ __launch_bounds__((M * N) / 32, 1) void NAME(         \
    void* output, float* partial, unsigned* flags,                             \
    const __grid_constant__ CUtensorMap a_map,                                 \
    const __grid_constant__ CUtensorMap b_map, const float* bias,              \
    const __grid_constant__ Sm120KernelParams params) {                        \
    sm120_tf32_tn_streamk_entry<M, N, STAGES>(                                 \
        output, partial, flags, a_map, b_map, bias, params);                   \
}

SM120_DEFINE_TF32_TN_STREAMK_KERNEL(gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s3_pair_streamk, 64, 128, 3)
SM120_DEFINE_TF32_KERNEL(gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n64_bk32_s2, Sm120Tn, 64, 64, 2)
SM120_DEFINE_TF32_KERNEL(gemm_bi_nt_sm120_tma_mma_tf32_v1_m128n64_bk32_s2, Sm120Nt, 128, 64, 2)
SM120_DEFINE_TF32_KERNEL(gemm_bi_nt_sm120_tma_mma_tf32_v1_m128n64_bk32_s3, Sm120Nt, 128, 64, 3)
SM120_DEFINE_TF32_KERNEL(gemm_bi_nt_sm120_tma_mma_tf32_v1_m64n128_bk32_s2, Sm120Nt, 64, 128, 2)
SM120_DEFINE_TF32_KERNEL(gemm_bi_nt_sm120_tma_mma_tf32_v1_m64n128_bk32_s3, Sm120Nt, 64, 128, 3)
SM120_DEFINE_TF32_KERNEL(gemm_bi_nt_sm120_tma_mma_tf32_v1_m64n64_bk32_s2, Sm120Nt, 64, 64, 2)

template <typename A, typename B> struct Sm120Tf32SameType { static constexpr bool value = false; };
template <typename A> struct Sm120Tf32SameType<A, A> { static constexpr bool value = true; };
using Sm120Tf32KernelSignature = void (*)(
    void*, CUtensorMap, CUtensorMap, const float*, Sm120KernelParams);
using Sm120Tf32KernelSignatureStreamK = void (*)(
    void*, float*, unsigned*, CUtensorMap, CUtensorMap, const float*, Sm120KernelParams);
#define TF32_ASSERT_KERNEL_SIGNATURE(NAME, ...) \
    static_assert(Sm120Tf32SameType<decltype(&NAME), Sm120Tf32KernelSignature##__VA_ARGS__>::value, "TF32 kernel signature")

TF32_ASSERT_KERNEL_SIGNATURE(gemm_bi_nn_sm120_tma_mma_tf32_v1_m128n64_bk32_s2);
TF32_ASSERT_KERNEL_SIGNATURE(gemm_bi_nn_sm120_tma_mma_tf32_v1_m128n64_bk32_s3);
TF32_ASSERT_KERNEL_SIGNATURE(gemm_bi_nn_sm120_tma_mma_tf32_v1_m64n128_bk32_s2);
TF32_ASSERT_KERNEL_SIGNATURE(gemm_bi_nn_sm120_tma_mma_tf32_v1_m64n128_bk32_s3);
TF32_ASSERT_KERNEL_SIGNATURE(gemm_bi_nn_sm120_tma_mma_tf32_v1_m64n64_bk32_s2);
TF32_ASSERT_KERNEL_SIGNATURE(gemm_bi_nn_sm120_tma_mma_tf32_v1_m80n32_bk64_s2);
TF32_ASSERT_KERNEL_SIGNATURE(gemm_bi_tn_sm120_tma_mma_tf32_v1_m128n64_bk32_s2);
TF32_ASSERT_KERNEL_SIGNATURE(gemm_bi_tn_sm120_tma_mma_tf32_v1_m128n64_bk32_s3);
TF32_ASSERT_KERNEL_SIGNATURE(gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s2);
TF32_ASSERT_KERNEL_SIGNATURE(gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s3);
TF32_ASSERT_KERNEL_SIGNATURE(gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s4_pair);
TF32_ASSERT_KERNEL_SIGNATURE(gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s3_pair_streamk, StreamK);
TF32_ASSERT_KERNEL_SIGNATURE(gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n64_bk32_s2);
TF32_ASSERT_KERNEL_SIGNATURE(gemm_bi_nt_sm120_tma_mma_tf32_v1_m128n64_bk32_s2);
TF32_ASSERT_KERNEL_SIGNATURE(gemm_bi_nt_sm120_tma_mma_tf32_v1_m128n64_bk32_s3);
TF32_ASSERT_KERNEL_SIGNATURE(gemm_bi_nt_sm120_tma_mma_tf32_v1_m64n128_bk32_s2);
TF32_ASSERT_KERNEL_SIGNATURE(gemm_bi_nt_sm120_tma_mma_tf32_v1_m64n128_bk32_s3);
TF32_ASSERT_KERNEL_SIGNATURE(gemm_bi_nt_sm120_tma_mma_tf32_v1_m64n64_bk32_s2);

#undef TF32_ASSERT_KERNEL_SIGNATURE
#undef SM120_DEFINE_TF32_TN_STREAMK_KERNEL
#undef SM120_DEFINE_HALF_TN_STREAMK_KERNEL
#undef SM120_DEFINE_TF32_PAIR_KERNEL
#undef SM120_DEFINE_TF32_KERNEL

#undef SM120_DEFINE_KERNEL
#undef SM120_CHECK_STORAGE
#undef SM120_SWIZZLE_128B
#undef SM120_SWIZZLE_64B

#endif
