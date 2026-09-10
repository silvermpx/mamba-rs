// Fixed-owned NN wide TF32 candidate. This is the production adaptation of
// internal/experiments/sm89-nn-wide-rna-compatible.cu at SHA256
// c0c9eb735374620eaf8a023345ee86359af9f748df53fc5cf7d07ee7051f65c4.
// The 128x128/BK32/S3 pipeline, ascending k8 MMA sequence, explicit
// cvt.rna.tf32.f32 conversion, bias initialization, tail handling, and
// scalar/vector epilogues are unchanged. Only Fixed-private dependency names,
// the production export, and the self-contained parameter/store helpers differ.
// Compose this fragment after Fixed common.cuh and tf32.cu, on sm_89 only.
//
// Export: gemm_bi_nn_fixed_rna_wide_tf32_v1_m128n128_bk32_s3
// ABI: (float* output, const float* a, const float* b, const float* bias,
//       GbfTf32WideParams params).
// Params: {float alpha,beta; int m,k,n,lda,ldb,ldc;} (32 bytes, alignment 4).
// Grid=(ceil(m/128)*ceil(n/128),1,1); block=(256,1,1).
// Dynamic shared=98,304 bytes, opt in the function. No scratch/counters.
// Same vector-input admission as the existing wide body: A/B base alignment
// 16 bytes and lda/ldb divisible by four, valid row-major extents/strides,
// dimensions/tile counts within device and signed-int limits. Host skips
// empty output. K=0 uses the unchanged zero-reduction epilogue. Fixed callers
// must pass this 32-byte struct with alpha=1,beta=0, not GbfTf32Params.
//
// Byte-level conversion rationale:
// Fixed's gbf_tf32_rna(float) applies exactly cvt.rna.tf32.f32 to the input
// register and returns its .b32 result. The new helper bit-reinterprets the
// incoming unsigned fragment as float, then uses the same PTX and constraints.
// This preserves all input bits until that instruction, including signed zero,
// subnormals, infinities, signaling/quiet NaNs and every NaN payload. It also
// preserves whatever output bits the conversion instruction produces, without
// an integer mask or a hand-written special-value transformation.
//
// For a hypothetical upper19-bit TF32 layout, finite nearest-away rounding
// equals ((bits+0x1000) & 0xffffe000). That algebra does not prove the PTX
// instruction's complete NaN payload behavior. In the unguarded old add,
// 0x7f801000 -> 0x7f802000; 0x7fffffff -> 0x80000fff; and
// 0xffffffff -> 0x00000fff. A finite-only add would preserve the special input
// bits, but the available evidence does not certify those bits as cvt.rna's
// result for every NaN. Therefore this twin deliberately uses explicit cvt.
//
// NVIDIA PTX ISA 9.3 defines RNA as nearest with ties away from zero and
// describes TF32's internal layout as implementation-defined:
// https://docs.nvidia.com/cuda/parallel-thread-execution/
// This source-level match is not GPU proof of a cross-route batch ladder.
// Main must still compare raw conversion results, exceptional-value GEMMs,
// repeated/batch-prefix bits and performance before any admission.

struct GbfTf32WideParams {
    float alpha;
    float beta;
    int m;
    int k;
    int n;
    int lda;
    int ldb;
    int ldc;
};

static_assert(sizeof(GbfTf32WideParams) == 32, "Fixed RNA-wide parameter ABI drift");
static_assert(alignof(GbfTf32WideParams) == 4, "Fixed RNA-wide parameter alignment drift");
static_assert(__is_standard_layout(GbfTf32WideParams),
              "Fixed RNA-wide parameters must remain standard layout");

__device__ __forceinline__ int tf32wrc_a_index(int row, int k) {
    int chunk = (k >> 2) ^ (row & 7);
    return row * 32 + chunk * 4 + (k & 3);
}

__device__ __forceinline__ int tf32wrc_b_index(int k, int column) {
    int chunk = (column >> 2) ^ ((k & 3) << 1);
    return k * 128 + chunk * 4 + (column & 3);
}

// One quarter of a stage's copies: the A and B rows this thread owns in
// slice `slice` (two 16-byte cp.async per thread), so the eight copies of a
// stage interleave with the four k8 steps of the stage being computed.
// Each thread owns eight 16-byte copies per stage (four of A, four of B):
// their global pointers, shared destinations and the row / column bounds
// depend on the tile and the thread only, so they are computed once; a
// stage advances the pointers by its k slab and clamps the copy length by
// the remaining reduction (zero bytes zero-fill the slot; a zero-length
// copy keeps a valid address).
struct Tf32wrcCopyPlan {
    const float* a_source[4];
    const float* b_source[4];
    unsigned a_destination[4];
    unsigned b_destination[4];
    int a_k_offset;
    int b_k_row;
    bool a_row_valid[4];
    int b_column_bytes[4];
};

__device__ __forceinline__ void tf32wrc_copy_plan(
    float* a_stage0, float* b_stage0, const float* a, const float* b,
    const GbfTf32WideParams& params, int tile_row, int tile_column, Tf32wrcCopyPlan& plan) {
    plan.a_k_offset = ((int)threadIdx.x & 7) * 4;
    plan.b_k_row = (int)threadIdx.x >> 5;
#pragma unroll
    for (int slice = 0; slice < 4; ++slice) {
        int linear = (int)threadIdx.x + slice * 256;
        int row = linear >> 3;
        int global_row = tile_row + row;
        plan.a_row_valid[slice] = global_row < params.m;
        plan.a_source[slice] = a + (long long)(plan.a_row_valid[slice] ? global_row : 0) * params.lda + plan.a_k_offset;
        plan.a_destination[slice] = (unsigned)__cvta_generic_to_shared(a_stage0 + tf32wrc_a_index(row, plan.a_k_offset));
        int k_row = linear >> 5;
        int column = (linear & 31) * 4;
        int global_column = tile_column + column;
        int columns = params.n - global_column;
        columns = columns < 0 ? 0 : (columns > 4 ? 4 : columns);
        plan.b_column_bytes[slice] = columns * 4;
        plan.b_source[slice] = b + (long long)k_row * params.ldb + (columns > 0 ? global_column : 0);
        plan.b_destination[slice] = (unsigned)__cvta_generic_to_shared(b_stage0 + tf32wrc_b_index(k_row, column));
    }
}

// One quarter of a stage's copies. `stage_bytes` is the stage's byte offset
// in the ring, `k_base` the stage's first k; the pointers already point at
// the stage (the caller advances them).
__device__ __forceinline__ void tf32wrc_stage_slice(
    const Tf32wrcCopyPlan& plan, unsigned stage_bytes, int k_base, int reduction, int slice) {
    {
        int remaining = reduction - k_base - plan.a_k_offset;
        remaining = remaining < 0 ? 0 : (remaining > 4 ? 4 : remaining);
        int bytes = plan.a_row_valid[slice] ? remaining * 4 : 0;
        gbf_tf32_copy_cg(plan.a_destination[slice] + stage_bytes, plan.a_source[slice], bytes);
    }
    {
        int bytes = k_base + plan.b_k_row + slice * 8 < reduction ? plan.b_column_bytes[slice] : 0;
        gbf_tf32_copy_cg(plan.b_destination[slice] + stage_bytes, plan.b_source[slice], bytes);
    }
}

__device__ __forceinline__ void tf32wrc_advance_plan(Tf32wrcCopyPlan& plan, long long b_rows) {
#pragma unroll
    for (int slice = 0; slice < 4; ++slice) {
        plan.a_source[slice] += 32;
        plan.b_source[slice] += b_rows;
    }
}

__device__ __forceinline__ void tf32wrc_stage_async(
    const Tf32wrcCopyPlan& plan, unsigned stage_bytes, int k_base, int reduction) {
#pragma unroll
    for (int slice = 0; slice < 4; ++slice) {
        tf32wrc_stage_slice(plan, stage_bytes, k_base, reduction, slice);
    }
    asm volatile("cp.async.commit_group;\n" ::);
}

struct Tf32wrcFragments {
    unsigned a[4][4];
    unsigned b[4][2];
};

// The shared-memory element offsets of this lane's fragments, computed once
// per thread: the swizzle of A depends on the k8 step through its chunk
// index, so the ldmatrix offsets are kept per (atom, step); the swizzle of B
// depends only on the lane's k row within the step, so one offset per
// (atom, half) serves every step with the step's row term added.
struct Tf32wrcFragmentOffsets {
    int a[4][4];
    int b[4][2];
};

__device__ __forceinline__ void tf32wrc_fragment_offsets(
    int warp_m, int warp_n, int group, int thread, int lane, Tf32wrcFragmentOffsets& offsets) {
    int a_row = warp_m + (lane & 15);
    int a_k = (lane >> 4) << 2;
#pragma unroll
    for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
        for (int step = 0; step < 4; ++step) {
            offsets.a[m_atom][step] = tf32wrc_a_index(a_row + m_atom * 16, step * 8 + a_k);
        }
    }
#pragma unroll
    for (int n_atom = 0; n_atom < 4; ++n_atom) {
        int column = warp_n + n_atom * 8 + group;
        offsets.b[n_atom][0] = tf32wrc_b_index(thread, column);
        offsets.b[n_atom][1] = tf32wrc_b_index(thread + 4, column);
    }
}

// Keep the operand conversion instruction identical to gbf_tf32_rna in
// kernels/gemm_bi_fixed/tf32.cu. __uint_as_float is a bit reinterpretation;
// it performs no FP arithmetic that could quiet or canonicalize an sNaN
// before the PTX instruction sees the original 32-bit payload.
__device__ __forceinline__ unsigned tf32wrc_round(unsigned bits) {
    float value = __uint_as_float(bits);
    unsigned result;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(result) : "f"(value));
    return result;
}

// The A fragment of one 16 x 8 atom through ldmatrix.x4: lanes 0-7 address
// rows 0-7 at k8, 8-15 rows 8-15 at k8, 16-23 rows 0-7 at k8 + 4 and 24-31
// rows 8-15 at k8 + 4, so the four registers land as a0..a3 of the mma
// (row g / row g + 8 at k t, then the same at k t + 4). Every row address is
// a 16-byte chunk of the swizzled stage.
__device__ __forceinline__ void tf32wrc_load_fragments(
    const float* a_stage, const float* b_stage, int step,
    const Tf32wrcFragmentOffsets& offsets, Tf32wrcFragments& fragments) {
    unsigned a_base = (unsigned)__cvta_generic_to_shared(a_stage);
#pragma unroll
    for (int m_atom = 0; m_atom < 4; ++m_atom) {
        unsigned address = a_base + (unsigned)offsets.a[m_atom][step] * 4U;
        unsigned raw0, raw1, raw2, raw3;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(raw0), "=r"(raw1), "=r"(raw2), "=r"(raw3)
                     : "r"(address));
        fragments.a[m_atom][0] = tf32wrc_round(raw0);
        fragments.a[m_atom][1] = tf32wrc_round(raw1);
        fragments.a[m_atom][2] = tf32wrc_round(raw2);
        fragments.a[m_atom][3] = tf32wrc_round(raw3);
    }
    const float* b_step = b_stage + step * 8 * 128;
#pragma unroll
    for (int n_atom = 0; n_atom < 4; ++n_atom) {
        fragments.b[n_atom][0] = tf32wrc_round(__float_as_uint(b_step[offsets.b[n_atom][0]]));
        fragments.b[n_atom][1] = tf32wrc_round(__float_as_uint(b_step[offsets.b[n_atom][1]]));
    }
}

__device__ __forceinline__ void tf32wrc_mma(const Tf32wrcFragments& fragments, float (&acc)[4][4][4]) {
#pragma unroll
    for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 4; ++n_atom) {
            gbf_tf32_mma_m16n8k8(acc[m_atom][n_atom], fragments.a[m_atom], fragments.b[n_atom]);
        }
    }
}

__device__ __forceinline__ void tf32wrc_store(
    float* output, int row, int column, float accumulator,
    const GbfTf32WideParams& params) {
    if (row >= params.m || column >= params.n) return;
    float value = params.alpha == 1.0f
        ? accumulator
        : __fmul_rn(params.alpha, accumulator);
    float* destination = output + (long long)row * params.ldc + column;
    if (params.beta != 0.0f) {
        value = __fmaf_rn(params.beta, *destination, value);
    }
    *destination = value;
}

template <int BM, int BN>
__device__ __forceinline__ void tf32wrc_zero_reduction(
    float* output, const float* bias, const GbfTf32WideParams& params) {
    int column_tiles = (params.n + BN - 1) / BN;
    int tile_row = (int)blockIdx.x / column_tiles * BM;
    int tile_column = (int)blockIdx.x % column_tiles * BN;
    for (int linear = (int)threadIdx.x; linear < BM * BN; linear += (int)blockDim.x) {
        int row = tile_row + linear / BN;
        int column = tile_column + linear % BN;
        if (row < params.m && column < params.n) {
            float accumulator = bias == nullptr ? 0.0f : bias[column];
            tf32wrc_store(output, row, column, accumulator, params);
        }
    }
}

template <int Stages>
__device__ __forceinline__ void tf32wrc_nn_kernel(
    float* output, const float* a, const float* b, const float* bias,
    GbfTf32WideParams params) {
    extern __shared__ __align__(16) unsigned char tf32wrc_shared[];
    float* a_stages = reinterpret_cast<float*>(tf32wrc_shared);
    float* b_stages = a_stages + Stages * 128 * 32;
    int column_tiles = (params.n + 127) / 128;
    int tile_row = (int)blockIdx.x / column_tiles * 128;
    int tile_column = (int)blockIdx.x % column_tiles * 128;
    int warp = (int)threadIdx.x >> 5;
    int lane = (int)threadIdx.x & 31;
    int warp_m = (warp >> 2) * 64;
    int warp_n = (warp & 3) * 32;
    int group = lane >> 2;
    int thread = lane & 3;
    float acc[4][4][4];
#pragma unroll
    for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 4; ++n_atom) {
#pragma unroll
            for (int element = 0; element < 4; ++element) {
                int row = tile_row + warp_m + m_atom * 16 + group + (element >= 2 ? 8 : 0);
                int column = tile_column + warp_n + n_atom * 8 + 2 * thread + (element & 1);
                acc[m_atom][n_atom][element] =
                    row < params.m && column < params.n && bias != nullptr ? bias[column] : 0.0f;
            }
        }
    }
    Tf32wrcFragmentOffsets offsets;
    tf32wrc_fragment_offsets(warp_m, warp_n, group, thread, lane, offsets);
    Tf32wrcCopyPlan plan;
    tf32wrc_copy_plan(a_stages, b_stages, a, b, params, tile_row, tile_column, plan);
    long long b_slab_rows = 32LL * params.ldb;
    unsigned tile_count = (static_cast<unsigned>(params.k) + 31U) / 32U;
#pragma unroll
    for (unsigned tile = 0; tile < Stages - 1; ++tile) {
        if (tile < tile_count) {
            tf32wrc_stage_async(plan, tile * 128 * 32 * 4U, (int)(tile * 32U), params.k);
            tf32wrc_advance_plan(plan, b_slab_rows);
        } else {
            asm volatile("cp.async.commit_group;\n" ::);
        }
    }
    int read_stage = 0;
    for (unsigned tile = 0; tile < tile_count; ++tile) {
        asm volatile("cp.async.wait_group %0;\n" :: "n"(Stages - 2));
        // The barrier also retires every warp's reads of the stage the copies
        // below overwrite (the one computed in the previous iteration).
        __syncthreads();
        unsigned next = tile + Stages - 1;
        bool has_next = next < tile_count;
        int write_stage = read_stage == 0 ? Stages - 1 : read_stage - 1;
        unsigned write_bytes = (unsigned)write_stage * 128U * 32U * 4U;
        const float* a_read = a_stages + read_stage * 128 * 32;
        const float* b_read = b_stages + read_stage * 32 * 128;
        Tf32wrcFragments fragments[2];
        tf32wrc_load_fragments(a_read, b_read, 0, offsets, fragments[0]);
#pragma unroll
        for (int issue = 0; issue < 4; ++issue) {
            if (has_next) {
                tf32wrc_stage_slice(plan, write_bytes, (int)(next * 32U), params.k, issue);
            }
            if (issue < 3) {
                tf32wrc_load_fragments(a_read, b_read, issue + 1, offsets, fragments[(issue + 1) & 1]);
            }
            tf32wrc_mma(fragments[issue & 1], acc);
        }
        asm volatile("cp.async.commit_group;\n" ::);
        if (has_next) tf32wrc_advance_plan(plan, b_slab_rows);
        if (++read_stage == Stages) read_stage = 0;
    }
    // Epilogue: the tile goes through shared memory (row stride 136 floats:
    // the fragment-shaped float2 writes land in two wavefronts) and out as
    // 16-byte row segments, one warp per 512-byte row, when the output is
    // 16-byte addressable and the tile's columns lie inside n; each element
    // sees the same alpha and beta arithmetic as the scalar store.
    __syncthreads();
    float* tile = reinterpret_cast<float*>(tf32wrc_shared);
    bool vector_rows = tile_column + 128 <= params.n && (params.ldc & 3) == 0
        && gbf_aligned16(output);
    if (vector_rows) {
#pragma unroll
        for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
            for (int n_atom = 0; n_atom < 4; ++n_atom) {
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    int row = warp_m + m_atom * 16 + group + half * 8;
                    int column = warp_n + n_atom * 8 + 2 * thread;
                    *reinterpret_cast<float2*>(tile + row * 136 + column) = make_float2(
                        acc[m_atom][n_atom][2 * half], acc[m_atom][n_atom][2 * half + 1]);
                }
            }
        }
        __syncthreads();
        bool scale = params.alpha != 1.0f;
        bool blend = params.beta != 0.0f;
#pragma unroll 4
        for (int linear = (int)threadIdx.x; linear < 128 * 32; linear += 256) {
            int row = linear >> 5;
            int chunk = (linear & 31) * 4;
            int global_row = tile_row + row;
            if (global_row >= params.m) continue;
            float4 value = *reinterpret_cast<const float4*>(tile + row * 136 + chunk);
            float* destination = output + (long long)global_row * params.ldc + tile_column + chunk;
            if (scale) {
                value.x = __fmul_rn(params.alpha, value.x);
                value.y = __fmul_rn(params.alpha, value.y);
                value.z = __fmul_rn(params.alpha, value.z);
                value.w = __fmul_rn(params.alpha, value.w);
            }
            if (blend) {
                float4 old = *reinterpret_cast<const float4*>(destination);
                value.x = __fmaf_rn(params.beta, old.x, value.x);
                value.y = __fmaf_rn(params.beta, old.y, value.y);
                value.z = __fmaf_rn(params.beta, old.z, value.z);
                value.w = __fmaf_rn(params.beta, old.w, value.w);
            }
            *reinterpret_cast<float4*>(destination) = value;
        }
        return;
    }
#pragma unroll
    for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 4; ++n_atom) {
#pragma unroll
            for (int element = 0; element < 4; ++element) {
                int row = tile_row + warp_m + m_atom * 16 + group + (element >= 2 ? 8 : 0);
                int column = tile_column + warp_n + n_atom * 8 + 2 * thread + (element & 1);
                tf32wrc_store(output, row, column, acc[m_atom][n_atom][element], params);
            }
        }
    }
}


extern "C" __global__ __launch_bounds__(256, 1)
void gemm_bi_nn_fixed_rna_wide_tf32_v1_m128n128_bk32_s3(
    float* output, const float* a, const float* b, const float* bias,
    GbfTf32WideParams params) {
    if (params.k == 0) {
        tf32wrc_zero_reduction<128, 128>(output, bias, params);
        return;
    }
    tf32wrc_nn_kernel<3>(output, a, b, bias, params);
}

template <typename A, typename B> struct GbfRnaSameType { static constexpr bool value = false; };
template <typename A> struct GbfRnaSameType<A, A> { static constexpr bool value = true; };
using GbfRnaWideSignature = void (*)(
    float*, const float*, const float*, const float*, GbfTf32WideParams);
static_assert(GbfRnaSameType<
    decltype(&gemm_bi_nn_fixed_rna_wide_tf32_v1_m128n128_bk32_s3),
    GbfRnaWideSignature>::value, "Fixed NN RNA-wide ABI");
