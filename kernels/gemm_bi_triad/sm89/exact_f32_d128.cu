/*
 * Exact-F32 TN weight gradient for the d_model-128 classifier shapes:
 * output[KOut x N] += alpha * A^T[KOut x 1024] . B[1024 x N], reduced over
 * the 1024 rows of both operands. The Rust owner expands this body once per
 * route; only the namespace, the shape, the shared-memory and grid pins and
 * the export symbol are route parameters.
 *
 * Bit contract, the one the split-M fold of the scalar family defines: for
 * every output element the 1024-row reduction is cut into 64 chunks of 16
 * rows; each chunk is an f32 fma chain over its 16 rows in ascending order
 * seeded from +0; chunk 0 seeds an f64 accumulator, chunks 1..63 are added
 * to it in ascending order with round-to-nearest; the epilogue multiplies by
 * alpha in f64, rounds once to f32 and adds that to the existing output
 * value in f32. Seeding is expressed here as adding chunk 0 to negative
 * zero, which is the same value for every input.
 *
 * What differs from a serial walk over the chunks is only where the work
 * happens. A CTA holds Groups warp groups; in round r group g computes the
 * f32 partial of chunk r*Groups+g for the whole tile, the groups exchange
 * partials through shared memory, and every thread then folds one output
 * over the round's Groups chunks in ascending chunk order. The fold order is
 * therefore unchanged, the chunk partials come from the same fma chain, and
 * the CTA keeps Groups chunks of loads in flight per stage.
 *
 * The float-to-double widening runs as integer bit manipulation instead of
 * the hardware conversion: conversions to 64-bit types issue at the same
 * rate as the f64 add itself, so the hardware cvt would double the time on
 * the f64 pipe, which is the floor of this kernel. The widening is exact for
 * every input, NaN payloads included.
 */

// SM89_EXACT_F32_D128_ROUTE_BEGIN
namespace __SM89_EXACT_F32_D128_NAMESPACE__ {

constexpr int MRed = 1024;
constexpr int KOut = __SM89_EXACT_F32_D128_K_OUT__;
constexpr int N = __SM89_EXACT_F32_D128_N__;
constexpr int BK = 16;
constexpr int Chunks = MRed / BK;

static_assert(MRed % BK == 0, "the exact reduction must contain whole chunks");
static_assert(Chunks == 64, "the Split-M reduction tree changed");

__device__ __forceinline__ bool aligned_16(const void* pointer) {
    return ((unsigned long long)pointer & 15ULL) == 0ULL;
}

template <bool CacheGlobal>
__device__ __forceinline__ void copy_16(unsigned destination, const float* source) {
    if (CacheGlobal) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :: "r"(destination), "l"(source));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                     :: "r"(destination), "l"(source));
    }
}

__device__ __forceinline__ void commit_group() {
    asm volatile("cp.async.commit_group;\n");
}

template <int Pending>
__device__ __forceinline__ void wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(Pending));
}

// Widening of the unusual values: zeros, subnormals, infinities and NaNs. A
// NaN widens the way the hardware widens it: the quiet bit is set, then the
// payload is shifted into the double's fraction.
__device__ __forceinline__ double widen_full(unsigned u) {
    unsigned sign = u & 0x80000000u;
    unsigned mag = u & 0x7fffffffu;
    unsigned hi = sign | ((mag >> 3) + 0x38000000u);
    unsigned lo = u << 29;
    if (mag == 0u) {
        hi = sign;
        lo = 0u;
    } else if (mag < 0x00800000u) {
        // Subnormal: value = mag * 2^-149. Shift the leading one to bit 23,
        // then it is a normal double with exponent 897 - shift.
        unsigned shift = (unsigned)__clz((int)mag) - 8u;
        unsigned normalized = (mag << shift) & 0x007fffffu;
        hi = sign | ((897u - shift) << 20) | (normalized >> 3);
        lo = normalized << 29;
    } else if (mag >= 0x7f800000u) {
        if (mag == 0x7f800000u) {
            hi = sign | 0x7ff00000u;
            lo = 0u;
        } else {
            unsigned quiet = mag | 0x00400000u;
            hi = sign | 0x7ff00000u | ((quiet & 0x007fffffu) >> 3);
            lo = quiet << 29;
        }
    }
    return __hiloint2double((int)hi, (int)lo);
}

// Fast path for normal values and zeros; the warp takes the full path only
// when some lane holds a subnormal, an infinity or a NaN. One unsigned
// compare separates the normal range from everything else.
__device__ __forceinline__ double widen(float value) {
    unsigned u = __float_as_uint(value);
    unsigned mag = u & 0x7fffffffu;
    bool unusual = (mag - 0x00800000u) >= 0x7f000000u;
    unsigned sign = u & 0x80000000u;
    unsigned hi = (mag == 0u) ? sign : (sign | ((mag >> 3) + 0x38000000u));
    unsigned lo = u << 29;
    if (__any_sync(0xffffffffu, unusual && mag != 0u)) {
        return widen_full(u);
    }
    return __hiloint2double((int)hi, (int)lo);
}

template <int TileM, int TileN, int Groups, int RowsPerThread, int ColsPerThread,
          int Stages, bool CacheGlobal>
struct Kernel {
    static constexpr int GroupThreads = TileM * TileN / (RowsPerThread * ColsPerThread);
    static constexpr int Threads = Groups * GroupThreads;
    static constexpr int Rounds = Chunks / Groups;
    static constexpr int OutputsPerThread = RowsPerThread * ColsPerThread;
    static constexpr int FoldPerThread = OutputsPerThread / Groups;
    static constexpr int ColumnGroups = TileN / ColsPerThread;
    static constexpr int AChunk = BK * TileM;
    static constexpr int BChunk = BK * TileN;
    static constexpr int ChunkFloats = AChunk + BChunk;
    static constexpr int StageFloats = Groups * ChunkFloats;
    static constexpr int AVectors = AChunk / 4;
    static constexpr int BVectors = BChunk / 4;
    static constexpr int ChunkVectors = AVectors + BVectors;
    static constexpr int StageVectors = Groups * ChunkVectors;
    static constexpr int XchgFloats = (Groups > 1) ? Groups * OutputsPerThread * GroupThreads : 0;
    static constexpr int SharedFloats = Stages * StageFloats + 2 * XchgFloats;
    static constexpr int SharedBytes = SharedFloats * (int)sizeof(float);
    static constexpr int RowTiles = KOut / TileM;
    static constexpr int ColumnTiles = N / TileN;
    static constexpr int Grid = RowTiles * ColumnTiles;

    static_assert(Chunks % Groups == 0, "groups must divide the chunk count");
    static_assert(OutputsPerThread % Groups == 0, "every thread folds whole outputs");
    static_assert(GroupThreads % 32 == 0, "a group is whole warps");
    static_assert(TileM % RowsPerThread == 0 && TileN % ColsPerThread == 0,
                  "thread tiles cover the CTA tile exactly");
    static_assert(KOut % TileM == 0 && N % TileN == 0, "tiles cover the output exactly");
    static_assert(TileM % 4 == 0 && TileN % 4 == 0, "16-byte copies need whole vectors");
    static_assert(ColsPerThread == 4 || ColsPerThread == 2 || ColsPerThread == 1,
                  "column tile is a vector width");
    static_assert(RowsPerThread == 1 || RowsPerThread == 2 || RowsPerThread == 4,
                  "row tile is a vector width");
    static_assert(SharedBytes <= 49152,
                  "a launch without the raised dynamic limit fits in 48 KB");
    static_assert(Stages >= 2, "the pipeline needs a landed stage and one in flight");

    static constexpr int VectorsPerThread = (StageVectors + Threads - 1) / Threads;

    // Per-thread copy plan: the shared-memory byte offset inside a stage and
    // the global source of round 0, fixed for the whole kernel. A round only
    // advances the source by its row stride.
    struct CopyPlan {
        const float* source[VectorsPerThread];
        unsigned destination[VectorsPerThread];
        int advance[VectorsPerThread];
        bool valid[VectorsPerThread];
    };

    static __device__ __forceinline__ CopyPlan plan_copies(
        const float* __restrict__ a,
        const float* __restrict__ b,
        int output_row_base,
        int output_column_base
    ) {
        CopyPlan plan;
        #pragma unroll
        for (int i = 0; i < VectorsPerThread; ++i) {
            int vector = (int)threadIdx.x + i * Threads;
            plan.valid[i] = vector < StageVectors;
            int slot = vector / ChunkVectors;
            int within = vector - slot * ChunkVectors;
            if (within < AVectors) {
                int reduction = within / (TileM / 4);
                int local = (within - reduction * (TileM / 4)) << 2;
                int row = slot * BK + reduction;
                plan.source[i] = a + (long long)row * KOut + output_row_base + local;
                plan.destination[i] =
                    (unsigned)(slot * ChunkFloats + reduction * TileM + local) * sizeof(float);
                plan.advance[i] = Groups * BK * KOut;
            } else {
                int w = within - AVectors;
                int reduction = w / (TileN / 4);
                int local = (w - reduction * (TileN / 4)) << 2;
                int row = slot * BK + reduction;
                plan.source[i] = b + (long long)row * N + output_column_base + local;
                plan.destination[i] =
                    (unsigned)(slot * ChunkFloats + AChunk + reduction * TileN + local)
                        * sizeof(float);
                plan.advance[i] = Groups * BK * N;
            }
        }
        return plan;
    }

    static __device__ __forceinline__ void issue_round(
        unsigned stage_shared_base, CopyPlan& plan) {
        #pragma unroll
        for (int i = 0; i < VectorsPerThread; ++i) {
            if (plan.valid[i]) {
                copy_16<CacheGlobal>(stage_shared_base + plan.destination[i], plan.source[i]);
            }
            plan.source[i] += plan.advance[i];
        }
        commit_group();
    }

    static __device__ __forceinline__ void load_rows(
        const float* a_tile, int reduction, int row_base, float (&values)[RowsPerThread]) {
        const float* p = a_tile + reduction * TileM + row_base;
        if (RowsPerThread == 4) {
            float4 v = *reinterpret_cast<const float4*>(p);
            values[0] = v.x; values[1] = v.y; values[2] = v.z; values[3] = v.w;
        } else if (RowsPerThread == 2) {
            float2 v = *reinterpret_cast<const float2*>(p);
            values[0] = v.x; values[1] = v.y;
        } else {
            values[0] = p[0];
        }
    }

    static __device__ __forceinline__ void load_columns(
        const float* b_tile, int reduction, int column_base, float (&values)[ColsPerThread]) {
        const float* p = b_tile + reduction * TileN + column_base;
        if (ColsPerThread == 4) {
            float4 v = *reinterpret_cast<const float4*>(p);
            values[0] = v.x; values[1] = v.y; values[2] = v.z; values[3] = v.w;
        } else if (ColsPerThread == 2) {
            float2 v = *reinterpret_cast<const float2*>(p);
            values[0] = v.x; values[1] = v.y;
        } else {
            values[0] = p[0];
        }
    }

    static __device__ __forceinline__ void compute_round(
        const float* stage_base, int group, int row_base, int column_base,
        float (&partial)[OutputsPerThread]) {
        const float* a_tile = stage_base + group * ChunkFloats;
        const float* b_tile = a_tile + AChunk;
        #pragma unroll
        for (int o = 0; o < OutputsPerThread; ++o) {
            partial[o] = 0.0f;
        }
        #pragma unroll
        for (int reduction = 0; reduction < BK; ++reduction) {
            float rows[RowsPerThread];
            float columns[ColsPerThread];
            load_rows(a_tile, reduction, row_base, rows);
            load_columns(b_tile, reduction, column_base, columns);
            #pragma unroll
            for (int i = 0; i < RowsPerThread; ++i) {
                #pragma unroll
                for (int j = 0; j < ColsPerThread; ++j) {
                    partial[i * ColsPerThread + j] =
                        __fmaf_rn(rows[i], columns[j], partial[i * ColsPerThread + j]);
                }
            }
        }
    }

    static __device__ __forceinline__ void publish(
        float* xchg_round, int group, int lane_in_group,
        const float (&partial)[OutputsPerThread]) {
        if (Groups > 1) {
            #pragma unroll
            for (int o = 0; o < OutputsPerThread; ++o) {
                xchg_round[(group * OutputsPerThread + o) * GroupThreads + lane_in_group] =
                    partial[o];
            }
        }
    }

    // Folds one round's chunks, ascending, into the f64 sums. With several
    // groups every partial comes from the exchange (the own group's too: one
    // shared load is cheaper than a select chain); a single group folds its
    // own registers.
    static __device__ __forceinline__ void fold_round(
        const float* xchg_round, int group, int lane_in_group,
        const float (&partial)[OutputsPerThread], double (&sums)[FoldPerThread]) {
        #pragma unroll
        for (int j = 0; j < Groups; ++j) {
            #pragma unroll
            for (int f = 0; f < FoldPerThread; ++f) {
                float value;
                if (Groups == 1) {
                    value = partial[f];
                } else {
                    int owned = group + Groups * f;
                    value = xchg_round[(j * OutputsPerThread + owned) * GroupThreads + lane_in_group];
                }
                sums[f] = __dadd_rn(sums[f], widen(value));
            }
        }
    }

    static __device__ __forceinline__ void run(
        float* __restrict__ output,
        const float* __restrict__ a,
        const float* __restrict__ b,
        float alpha,
        int m,
        int k,
        int n
    ) {
        if (!output || !a || !b || m != MRed || k != KOut || n != N
            || blockDim.x != Threads || blockDim.y != 1 || blockDim.z != 1
            || gridDim.x != Grid || gridDim.y != 1 || gridDim.z != 1
            || !aligned_16(a) || !aligned_16(b)) {
            return;
        }

        int tile_m = (int)blockIdx.x % RowTiles;
        int tile_n = (int)blockIdx.x / RowTiles;
        int output_row_base = tile_m * TileM;
        int output_column_base = tile_n * TileN;

        int group = (int)threadIdx.x / GroupThreads;
        int lane_in_group = (int)threadIdx.x - group * GroupThreads;
        int row_group = lane_in_group / ColumnGroups;
        int column_group = lane_in_group - row_group * ColumnGroups;
        int row_base = row_group * RowsPerThread;
        int column_base = column_group * ColsPerThread;

        extern __shared__ __align__(16) float shared[];
        float* stages = shared;
        float* xchg = shared + Stages * StageFloats;
        unsigned stages_shared = __cvta_generic_to_shared(stages);
        constexpr unsigned StageBytes = (unsigned)StageFloats * sizeof(float);
        CopyPlan plan = plan_copies(a, b, output_row_base, output_column_base);

        #pragma unroll
        for (int s = 0; s < Stages - 1; ++s) {
            if (s < Rounds) {
                issue_round(stages_shared + s * StageBytes, plan);
            } else {
                commit_group();
            }
        }

        float partial[OutputsPerThread];
        // Negative zero is the identity of round-to-nearest addition for
        // every double, including both zeros, so adding chunk 0 to it yields
        // exactly the value a fold seeded from chunk 0 holds, and the fold
        // needs no first-chunk special case.
        double sums[FoldPerThread];
        #pragma unroll
        for (int f = 0; f < FoldPerThread; ++f) {
            sums[f] = -0.0;
        }

        // Two exchange buffers, used in alternate rounds, so one barrier per
        // round suffices: a round publishes into its own buffer and folds the
        // previous round's, which the barrier at the top already made
        // visible.
        wait_group<Stages - 2>();
        __syncthreads();
        if (Stages - 1 < Rounds) {
            issue_round(stages_shared + ((Stages - 1) % Stages) * StageBytes, plan);
        } else {
            commit_group();
        }
        compute_round(stages, group, row_base, column_base, partial);
        publish(xchg, group, lane_in_group, partial);

        #pragma unroll 1
        for (int round = 1; round < Rounds; ++round) {
            wait_group<Stages - 2>();
            __syncthreads();

            int prefetch = round + Stages - 1;
            if (prefetch < Rounds) {
                issue_round(stages_shared + (prefetch % Stages) * StageBytes, plan);
            } else {
                commit_group();
            }

            fold_round(xchg + ((round - 1) & 1) * XchgFloats, group, lane_in_group,
                       partial, sums);
            compute_round(stages + (round % Stages) * StageFloats, group, row_base,
                          column_base, partial);
            publish(xchg + (round & 1) * XchgFloats, group, lane_in_group, partial);
        }

        __syncthreads();
        fold_round(xchg + ((Rounds - 1) & 1) * XchgFloats, group, lane_in_group,
                   partial, sums);

        #pragma unroll
        for (int f = 0; f < FoldPerThread; ++f) {
            int owned = group + Groups * f;
            int i = owned / ColsPerThread;
            int j = owned - i * ColsPerThread;
            int row = output_row_base + row_base + i;
            int column = output_column_base + column_base + j;
            long long index = (long long)row * N + column;
            double scaled = __dmul_rn((double)alpha, sums[f]);
            output[index] = __fadd_rn(output[index], __double2float_rn(scaled));
        }
    }
};

using Route = Kernel<16, 16, 8, 2, 4, 2, true>;

static_assert(Route::Threads == 256, "the CTA is eight warps");
static_assert(Route::SharedBytes == __SM89_EXACT_F32_D128_SHARED_BYTES__,
              "shared-memory contract changed");
static_assert(Route::Grid == __SM89_EXACT_F32_D128_GRID__, "flat-grid contract changed");

} // namespace __SM89_EXACT_F32_D128_NAMESPACE__

extern "C" __global__ __launch_bounds__(256, 2)
void __SM89_EXACT_F32_D128_SYMBOL__(
    float* output,
    const float* a,
    const float* b,
    float alpha,
    int m,
    int k,
    int n
) {
    __SM89_EXACT_F32_D128_NAMESPACE__::Route::run(output, a, b, alpha, m, k, n);
}
// SM89_EXACT_F32_D128_ROUTE_END
