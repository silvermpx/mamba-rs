// The wide deterministic TF32 tile: NN on 128 x 128 with eight computing
// warps (two rows by four columns of 64 x 32 warp tiles), BK 32, three
// cp.async stages in XOR-swizzled unpadded shared memory (32 KB per stage,
// 96 KB in all). The 128 x 64 tile above keeps four of its eight warps out of
// the mma loop; here every warp computes. One CTA fits a multiprocessor, so
// the two warps a scheduler holds cannot hide latency for each other: the
// main loop hides it inside the warp instead. The A fragments arrive by
// ldmatrix (one instruction per 16 x 8 atom instead of four loads), the
// fragments of the next k8 step are loaded while the current step's mma
// chain runs, and the next stage's cp.async are spread over the four k8
// steps rather than issued as one burst that fills the memory queue. Per
// output element the mma chain is the same ascending k8 sequence as every
// other portable TF32 tile, so the output is bit-identical to them.
// Composed into the portable module for every sm80-family target except
// CC 12.x, which keeps the compute_120 module byte-identical to the one its
// TF32 cohort's portable twin was frozen against.

__device__ __forceinline__ int tf32w_a_index(int row, int k) {
    int chunk = (k >> 2) ^ (row & 7);
    return row * 32 + chunk * 4 + (k & 3);
}

__device__ __forceinline__ int tf32w_b_index(int k, int column) {
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
struct Tf32wCopyPlan {
    const float* a_source[4];
    const float* b_source[4];
    unsigned a_destination[4];
    unsigned b_destination[4];
    int a_k_offset;
    int b_k_row;
    bool a_row_valid[4];
    int b_column_bytes[4];
};

__device__ __forceinline__ void tf32w_copy_plan(
    float* a_stage0, float* b_stage0, const float* a, const float* b,
    const Sm80Tf32KernelParams& params, int tile_row, int tile_column, Tf32wCopyPlan& plan) {
    plan.a_k_offset = ((int)threadIdx.x & 7) * 4;
    plan.b_k_row = (int)threadIdx.x >> 5;
#pragma unroll
    for (int slice = 0; slice < 4; ++slice) {
        int linear = (int)threadIdx.x + slice * 256;
        int row = linear >> 3;
        int global_row = tile_row + row;
        plan.a_row_valid[slice] = global_row < params.m;
        plan.a_source[slice] = a + (long long)(plan.a_row_valid[slice] ? global_row : 0) * params.lda + plan.a_k_offset;
        plan.a_destination[slice] = (unsigned)__cvta_generic_to_shared(a_stage0 + tf32w_a_index(row, plan.a_k_offset));
        int k_row = linear >> 5;
        int column = (linear & 31) * 4;
        int global_column = tile_column + column;
        int columns = params.n - global_column;
        columns = columns < 0 ? 0 : (columns > 4 ? 4 : columns);
        plan.b_column_bytes[slice] = columns * 4;
        plan.b_source[slice] = b + (long long)k_row * params.ldb + (columns > 0 ? global_column : 0);
        plan.b_destination[slice] = (unsigned)__cvta_generic_to_shared(b_stage0 + tf32w_b_index(k_row, column));
    }
}

// One quarter of a stage's copies. `stage_bytes` is the stage's byte offset
// in the ring, `k_base` the stage's first k; the pointers already point at
// the stage (the caller advances them).
__device__ __forceinline__ void tf32w_stage_slice(
    const Tf32wCopyPlan& plan, unsigned stage_bytes, int k_base, int reduction, int slice) {
    {
        int remaining = reduction - k_base - plan.a_k_offset;
        remaining = remaining < 0 ? 0 : (remaining > 4 ? 4 : remaining);
        int bytes = plan.a_row_valid[slice] ? remaining * 4 : 0;
        cp_async_16_zfill_l2(plan.a_destination[slice] + stage_bytes, plan.a_source[slice], bytes);
    }
    {
        int bytes = k_base + plan.b_k_row + slice * 8 < reduction ? plan.b_column_bytes[slice] : 0;
        cp_async_16_zfill_l2(plan.b_destination[slice] + stage_bytes, plan.b_source[slice], bytes);
    }
}

__device__ __forceinline__ void tf32w_advance_plan(Tf32wCopyPlan& plan, long long b_rows) {
#pragma unroll
    for (int slice = 0; slice < 4; ++slice) {
        plan.a_source[slice] += 32;
        plan.b_source[slice] += b_rows;
    }
}

__device__ __forceinline__ void tf32w_stage_async(
    const Tf32wCopyPlan& plan, unsigned stage_bytes, int k_base, int reduction) {
#pragma unroll
    for (int slice = 0; slice < 4; ++slice) {
        tf32w_stage_slice(plan, stage_bytes, k_base, reduction, slice);
    }
    asm volatile("cp.async.commit_group;\n" ::);
}

struct Tf32wFragments {
    unsigned a[4][4];
    unsigned b[4][2];
};

// The shared-memory element offsets of this lane's fragments, computed once
// per thread: the swizzle of A depends on the k8 step through its chunk
// index, so the ldmatrix offsets are kept per (atom, step); the swizzle of B
// depends only on the lane's k row within the step, so one offset per
// (atom, half) serves every step with the step's row term added.
struct Tf32wFragmentOffsets {
    int a[4][4];
    int b[4][2];
};

__device__ __forceinline__ void tf32w_fragment_offsets(
    int warp_m, int warp_n, int group, int thread, int lane, Tf32wFragmentOffsets& offsets) {
    int a_row = warp_m + (lane & 15);
    int a_k = (lane >> 4) << 2;
#pragma unroll
    for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
        for (int step = 0; step < 4; ++step) {
            offsets.a[m_atom][step] = tf32w_a_index(a_row + m_atom * 16, step * 8 + a_k);
        }
    }
#pragma unroll
    for (int n_atom = 0; n_atom < 4; ++n_atom) {
        int column = warp_n + n_atom * 8 + group;
        offsets.b[n_atom][0] = tf32w_b_index(thread, column);
        offsets.b[n_atom][1] = tf32w_b_index(thread + 4, column);
    }
}

// TF32 rounding: half an ulp of the ten-bit mantissa, the power of two of the
// operand's own exponent scaled by 2^-11, added in floating point before the
// tensor core drops the low 13 bits (measured on sm_89: they never reach the
// product). For every normal value the upper 19 bits of the sum are those of
// the integer add of 0x1000, the cvt.rna.tf32.f32 result, the carry into the
// exponent and the overflow to infinity included. A subnormal gets no half
// ulp and rounds toward zero, as the tensor core truncates it. A NaN stays a
// NaN, which the integer add does not do: 0x7fffffff, the NaN GPU arithmetic
// produces, carries into the sign bit and reads as a signed zero. There is no
// predicate, so the conversions of a step do not compete for the few
// predicate registers.
__device__ __forceinline__ unsigned tf32w_round(unsigned bits) {
    return __float_as_uint(fmaf(__uint_as_float(bits & 0xff800000U), 1.0f / 2048.0f, __uint_as_float(bits)));
}

// The A fragment of one 16 x 8 atom through ldmatrix.x4: lanes 0-7 address
// rows 0-7 at k8, 8-15 rows 8-15 at k8, 16-23 rows 0-7 at k8 + 4 and 24-31
// rows 8-15 at k8 + 4, so the four registers land as a0..a3 of the mma
// (row g / row g + 8 at k t, then the same at k t + 4). Every row address is
// a 16-byte chunk of the swizzled stage.
__device__ __forceinline__ void tf32w_load_fragments(
    const float* a_stage, const float* b_stage, int step,
    const Tf32wFragmentOffsets& offsets, Tf32wFragments& fragments) {
    unsigned a_base = (unsigned)__cvta_generic_to_shared(a_stage);
#pragma unroll
    for (int m_atom = 0; m_atom < 4; ++m_atom) {
        unsigned address = a_base + (unsigned)offsets.a[m_atom][step] * 4U;
        unsigned raw0, raw1, raw2, raw3;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(raw0), "=r"(raw1), "=r"(raw2), "=r"(raw3)
                     : "r"(address));
        fragments.a[m_atom][0] = tf32w_round(raw0);
        fragments.a[m_atom][1] = tf32w_round(raw1);
        fragments.a[m_atom][2] = tf32w_round(raw2);
        fragments.a[m_atom][3] = tf32w_round(raw3);
    }
    const float* b_step = b_stage + step * 8 * 128;
#pragma unroll
    for (int n_atom = 0; n_atom < 4; ++n_atom) {
        fragments.b[n_atom][0] = tf32w_round(__float_as_uint(b_step[offsets.b[n_atom][0]]));
        fragments.b[n_atom][1] = tf32w_round(__float_as_uint(b_step[offsets.b[n_atom][1]]));
    }
}

__device__ __forceinline__ void tf32w_mma(const Tf32wFragments& fragments, float (&acc)[4][4][4]) {
#pragma unroll
    for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 4; ++n_atom) {
            tf32_mma_m16n8k8(acc[m_atom][n_atom], fragments.a[m_atom], fragments.b[n_atom]);
        }
    }
}

template <int Stages>
__device__ __forceinline__ void tf32w_nn_kernel(
    float* output, const float* a, const float* b, const float* bias,
    Sm80Tf32KernelParams params) {
    extern __shared__ __align__(16) unsigned char tf32w_shared[];
    float* a_stages = reinterpret_cast<float*>(tf32w_shared);
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
    Tf32wFragmentOffsets offsets;
    tf32w_fragment_offsets(warp_m, warp_n, group, thread, lane, offsets);
    Tf32wCopyPlan plan;
    tf32w_copy_plan(a_stages, b_stages, a, b, params, tile_row, tile_column, plan);
    long long b_slab_rows = 32LL * params.ldb;
    unsigned tile_count = (static_cast<unsigned>(params.k) + 31U) / 32U;
#pragma unroll
    for (unsigned tile = 0; tile < Stages - 1; ++tile) {
        if (tile < tile_count) {
            tf32w_stage_async(plan, tile * 128 * 32 * 4U, (int)(tile * 32U), params.k);
            tf32w_advance_plan(plan, b_slab_rows);
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
        Tf32wFragments fragments[2];
        tf32w_load_fragments(a_read, b_read, 0, offsets, fragments[0]);
#pragma unroll
        for (int issue = 0; issue < 4; ++issue) {
            if (has_next) {
                tf32w_stage_slice(plan, write_bytes, (int)(next * 32U), params.k, issue);
            }
            if (issue < 3) {
                tf32w_load_fragments(a_read, b_read, issue + 1, offsets, fragments[(issue + 1) & 1]);
            }
            tf32w_mma(fragments[issue & 1], acc);
        }
        asm volatile("cp.async.commit_group;\n" ::);
        if (has_next) tf32w_advance_plan(plan, b_slab_rows);
        if (++read_stage == Stages) read_stage = 0;
    }
    // Epilogue: the tile goes through shared memory (row stride 136 floats:
    // the fragment-shaped float2 writes land in two wavefronts) and out as
    // 16-byte row segments, one warp per 512-byte row, when the output is
    // 16-byte addressable and the tile's columns lie inside n; each element
    // sees the same alpha and beta arithmetic as the scalar store.
    __syncthreads();
    float* tile = reinterpret_cast<float*>(tf32w_shared);
    bool vector_rows = tile_column + 128 <= params.n && (params.ldc & 3) == 0
        && is_aligned_16(output);
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
                tf32_store<SgbTf32Nn>(output, row, column, acc[m_atom][n_atom][element], bias, params);
            }
        }
    }
}


extern "C" __global__ __launch_bounds__(256, 1)
void nn_sm80_mma_tf32_m128n128_bk32_s3(
    float* output, const float* a, const float* b, const float* bias,
    Sm80Tf32KernelParams params) {
    if (params.k == 0) {
        tf32_zero_reduction_epilogue<SgbTf32Nn, 128, 128>(output, bias, params);
        return;
    }
    tf32w_nn_kernel<3>(output, a, b, bias, params);
}
