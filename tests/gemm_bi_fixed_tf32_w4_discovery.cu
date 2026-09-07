// Test-only four-warp deterministic-TF32 discovery candidates.

struct GbfTf32W4Params {
    float alpha;
    float beta;
    int m;
    int k;
    int n;
    int lda;
    int ldb;
    int ldc;
};

static_assert(sizeof(GbfTf32W4Params) == 32, "W4 parameter ABI");
static_assert(alignof(GbfTf32W4Params) == 4, "W4 parameter alignment");

__device__ __forceinline__ int tf32w4_a_index(int row, int k) {
    int chunk = (k >> 2) ^ (row & 3);
    return row * 16 + chunk * 4 + (k & 3);
}

template <int BN>
__device__ __forceinline__ int tf32w4_b_index(int k, int column) {
    int chunk = (column >> 2) ^ ((k & 3) << 1);
    return k * BN + chunk * 4 + (column & 3);
}

template <int BN>
struct Tf32w4CopyPlan {
    const float* a_source[4];
    const float* b_source[BN / 32];
    unsigned a_destination[4];
    unsigned b_destination[BN / 32];
    int a_k_offset;
    int b_k_row[BN / 32];
    bool a_row_valid[4];
    int b_column_bytes[BN / 32];
};

template <int BN>
__device__ __forceinline__ void tf32w4_copy_plan(
    float* a_stage0, float* b_stage0, const float* a, const float* b,
    const GbfTf32W4Params& params, int tile_row, int tile_column,
    Tf32w4CopyPlan<BN>& plan) {
    plan.a_k_offset = ((int)threadIdx.x & 3) * 4;
#pragma unroll
    for (int slice = 0; slice < 4; ++slice) {
        int linear = (int)threadIdx.x + slice * 128;
        int row = linear >> 2;
        int global_row = tile_row + row;
        plan.a_row_valid[slice] = global_row < params.m;
        plan.a_source[slice] =
            a + (long long)(plan.a_row_valid[slice] ? global_row : 0) * params.lda
            + plan.a_k_offset;
        plan.a_destination[slice] = (unsigned)__cvta_generic_to_shared(
            a_stage0 + tf32w4_a_index(row, plan.a_k_offset));
    }
#pragma unroll
    for (int slice = 0; slice < BN / 32; ++slice) {
        int linear = (int)threadIdx.x + slice * 128;
        int chunks = BN / 4;
        int k_row = linear / chunks;
        int column = (linear % chunks) * 4;
        int global_column = tile_column + column;
        int columns = params.n - global_column;
        columns = columns < 0 ? 0 : (columns > 4 ? 4 : columns);
        plan.b_k_row[slice] = k_row;
        plan.b_column_bytes[slice] = columns * 4;
        plan.b_source[slice] =
            b + (long long)k_row * params.ldb + (columns > 0 ? global_column : 0);
        plan.b_destination[slice] = (unsigned)__cvta_generic_to_shared(
            b_stage0 + tf32w4_b_index<BN>(k_row, column));
    }
}

template <int BN>
__device__ __forceinline__ void tf32w4_stage_slice(
    const Tf32w4CopyPlan<BN>& plan, unsigned a_stage_bytes,
    unsigned b_stage_bytes, int k_base, int reduction, int issue) {
    int remaining = reduction - k_base - plan.a_k_offset;
    remaining = remaining < 0 ? 0 : (remaining > 4 ? 4 : remaining);
    int a_bytes = plan.a_row_valid[issue] ? remaining * 4 : 0;
    gbf_tf32_copy_cg(
        plan.a_destination[issue] + a_stage_bytes,
        plan.a_source[issue], a_bytes);
    if (issue < BN / 32) {
        int b_bytes = k_base + plan.b_k_row[issue] < reduction
            ? plan.b_column_bytes[issue]
            : 0;
        gbf_tf32_copy_cg(
            plan.b_destination[issue] + b_stage_bytes,
            plan.b_source[issue], b_bytes);
    }
}

template <int BN>
__device__ __forceinline__ void tf32w4_stage_async(
    const Tf32w4CopyPlan<BN>& plan, unsigned a_stage_bytes,
    unsigned b_stage_bytes, int k_base, int reduction) {
#pragma unroll
    for (int issue = 0; issue < 4; ++issue) {
        tf32w4_stage_slice(
            plan, a_stage_bytes, b_stage_bytes, k_base, reduction, issue);
    }
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int BN>
__device__ __forceinline__ void tf32w4_advance_plan(
    Tf32w4CopyPlan<BN>& plan, long long b_rows) {
#pragma unroll
    for (int slice = 0; slice < 4; ++slice) plan.a_source[slice] += 16;
#pragma unroll
    for (int slice = 0; slice < BN / 32; ++slice) plan.b_source[slice] += b_rows;
}

template <int BN>
struct Tf32w4Fragments {
    unsigned a[4][4];
    unsigned b[BN / 16][2];
};

template <int BN>
struct Tf32w4FragmentOffsets {
    int a[4][2];
    int b[BN / 16][2];
};

template <int BN>
__device__ __forceinline__ void tf32w4_fragment_offsets(
    int warp_m, int warp_n, int group, int thread, int lane,
    Tf32w4FragmentOffsets<BN>& offsets) {
    int a_row = warp_m + (lane & 15);
    int a_k = (lane >> 4) << 2;
#pragma unroll
    for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
        for (int step = 0; step < 2; ++step) {
            offsets.a[m_atom][step] =
                tf32w4_a_index(a_row + m_atom * 16, step * 8 + a_k);
        }
    }
#pragma unroll
    for (int n_atom = 0; n_atom < BN / 16; ++n_atom) {
        int column = warp_n + n_atom * 8 + group;
        offsets.b[n_atom][0] = tf32w4_b_index<BN>(thread, column);
        offsets.b[n_atom][1] = tf32w4_b_index<BN>(thread + 4, column);
    }
}

__device__ __forceinline__ unsigned tf32w4_round(unsigned bits) {
    float value = __uint_as_float(bits);
    unsigned result;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(result) : "f"(value));
    return result;
}

template <int BN>
__device__ __forceinline__ void tf32w4_load_fragments(
    const float* a_stage, const float* b_stage, int step,
    const Tf32w4FragmentOffsets<BN>& offsets, Tf32w4Fragments<BN>& fragments) {
    unsigned a_base = (unsigned)__cvta_generic_to_shared(a_stage);
#pragma unroll
    for (int m_atom = 0; m_atom < 4; ++m_atom) {
        unsigned address = a_base + (unsigned)offsets.a[m_atom][step] * 4U;
        unsigned raw0, raw1, raw2, raw3;
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(raw0), "=r"(raw1), "=r"(raw2), "=r"(raw3)
            : "r"(address));
        fragments.a[m_atom][0] = tf32w4_round(raw0);
        fragments.a[m_atom][1] = tf32w4_round(raw1);
        fragments.a[m_atom][2] = tf32w4_round(raw2);
        fragments.a[m_atom][3] = tf32w4_round(raw3);
    }
    const float* b_step = b_stage + step * 8 * BN;
#pragma unroll
    for (int n_atom = 0; n_atom < BN / 16; ++n_atom) {
        fragments.b[n_atom][0] =
            tf32w4_round(__float_as_uint(b_step[offsets.b[n_atom][0]]));
        fragments.b[n_atom][1] =
            tf32w4_round(__float_as_uint(b_step[offsets.b[n_atom][1]]));
    }
}

template <int BN>
__device__ __forceinline__ void tf32w4_mma(
    const Tf32w4Fragments<BN>& fragments, float (&acc)[4][BN / 16][4]) {
#pragma unroll
    for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < BN / 16; ++n_atom) {
            gbf_tf32_mma_m16n8k8(
                acc[m_atom][n_atom], fragments.a[m_atom], fragments.b[n_atom]);
        }
    }
}

__device__ __forceinline__ void tf32w4_store(
    float* output, int row, int column, float accumulator,
    const GbfTf32W4Params& params) {
    if (row >= params.m || column >= params.n) return;
    float value = params.alpha == 1.0f
        ? accumulator
        : __fmul_rn(params.alpha, accumulator);
    float* destination = output + (long long)row * params.ldc + column;
    if (params.beta != 0.0f) value = __fmaf_rn(params.beta, *destination, value);
    *destination = value;
}

template <int BN>
__device__ __forceinline__ void tf32w4_zero_reduction(
    float* output, const float* bias, const GbfTf32W4Params& params) {
    int column_tiles = (params.n + BN - 1) / BN;
    int tile_row = (int)blockIdx.x / column_tiles * 128;
    int tile_column = (int)blockIdx.x % column_tiles * BN;
    for (int linear = (int)threadIdx.x; linear < 128 * BN;
         linear += (int)blockDim.x) {
        int row = tile_row + linear / BN;
        int column = tile_column + linear % BN;
        if (row < params.m && column < params.n) {
            float accumulator = bias == nullptr ? 0.0f : bias[column];
            tf32w4_store(output, row, column, accumulator, params);
        }
    }
}

template <int BN>
__device__ __forceinline__ void tf32w4_kernel(
    float* output, const float* a, const float* b, const float* bias,
    GbfTf32W4Params params) {
    extern __shared__ __align__(16) unsigned char shared_bytes[];
    float* a_stages = reinterpret_cast<float*>(shared_bytes);
    float* b_stages = a_stages + 5 * 128 * 16;
    int column_tiles = (params.n + BN - 1) / BN;
    int tile_row = (int)blockIdx.x / column_tiles * 128;
    int tile_column = (int)blockIdx.x % column_tiles * BN;
    int warp = (int)threadIdx.x >> 5;
    int lane = (int)threadIdx.x & 31;
    int warp_m = (warp >> 1) * 64;
    int warp_n = (warp & 1) * (BN / 2);
    int group = lane >> 2;
    int thread = lane & 3;
    float acc[4][BN / 16][4];
#pragma unroll
    for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < BN / 16; ++n_atom) {
#pragma unroll
            for (int element = 0; element < 4; ++element) {
                int row = tile_row + warp_m + m_atom * 16 + group
                    + (element >= 2 ? 8 : 0);
                int column = tile_column + warp_n + n_atom * 8
                    + 2 * thread + (element & 1);
                acc[m_atom][n_atom][element] =
                    row < params.m && column < params.n && bias != nullptr
                    ? bias[column]
                    : 0.0f;
            }
        }
    }
    Tf32w4FragmentOffsets<BN> offsets;
    tf32w4_fragment_offsets<BN>(warp_m, warp_n, group, thread, lane, offsets);
    Tf32w4CopyPlan<BN> plan;
    tf32w4_copy_plan<BN>(
        a_stages, b_stages, a, b, params, tile_row, tile_column, plan);
    long long b_slab_rows = 16LL * params.ldb;
    unsigned tile_count =
        2U * ((static_cast<unsigned>(params.k) + 31U) / 32U);
#pragma unroll
    for (unsigned tile = 0; tile < 4; ++tile) {
        if (tile < tile_count) {
            tf32w4_stage_async<BN>(
                plan, tile * 128U * 16U * 4U, tile * 16U * BN * 4U,
                (int)(tile * 16U), params.k);
            tf32w4_advance_plan<BN>(plan, b_slab_rows);
        } else {
            asm volatile("cp.async.commit_group;\n" ::);
        }
    }
    for (unsigned tile = 0; tile < tile_count; ++tile) {
        asm volatile("cp.async.wait_group 3;\n" ::);
        __syncthreads();
        unsigned next = tile + 4;
        bool has_next = next < tile_count;
        int read_stage = (int)(tile % 5U);
        int write_stage = (int)(next % 5U);
        unsigned write_a_bytes = (unsigned)write_stage * 128U * 16U * 4U;
        unsigned write_b_bytes = (unsigned)write_stage * 16U * BN * 4U;
        const float* a_read = a_stages + read_stage * 128 * 16;
        const float* b_read = b_stages + read_stage * 16 * BN;
        Tf32w4Fragments<BN> fragments[2];
        tf32w4_load_fragments<BN>(a_read, b_read, 0, offsets, fragments[0]);
#pragma unroll
        for (int step = 0; step < 2; ++step) {
            if (has_next) {
                tf32w4_stage_slice<BN>(
                    plan, write_a_bytes, write_b_bytes,
                    (int)(next * 16U), params.k, step * 2);
                tf32w4_stage_slice<BN>(
                    plan, write_a_bytes, write_b_bytes,
                    (int)(next * 16U), params.k, step * 2 + 1);
            }
            if (step == 0) {
                tf32w4_load_fragments<BN>(
                    a_read, b_read, 1, offsets, fragments[1]);
            }
            tf32w4_mma<BN>(fragments[step], acc);
        }
        asm volatile("cp.async.commit_group;\n" ::);
        if (has_next) tf32w4_advance_plan<BN>(plan, b_slab_rows);
    }
    __syncthreads();
    float* tile_output = reinterpret_cast<float*>(shared_bytes);
    constexpr int OutputStride = BN + 8;
    bool vector_rows = tile_column + BN <= params.n && (params.ldc & 3) == 0
        && gbf_aligned16(output);
    if (vector_rows) {
#pragma unroll
        for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
            for (int n_atom = 0; n_atom < BN / 16; ++n_atom) {
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    int row = warp_m + m_atom * 16 + group + half * 8;
                    int column = warp_n + n_atom * 8 + 2 * thread;
                    *reinterpret_cast<float2*>(tile_output + row * OutputStride + column) =
                        make_float2(
                            acc[m_atom][n_atom][2 * half],
                            acc[m_atom][n_atom][2 * half + 1]);
                }
            }
        }
        __syncthreads();
        bool scale = params.alpha != 1.0f;
        bool blend = params.beta != 0.0f;
        for (int linear = (int)threadIdx.x; linear < 128 * (BN / 4);
             linear += 128) {
            int row = linear / (BN / 4);
            int chunk = (linear % (BN / 4)) * 4;
            int global_row = tile_row + row;
            if (global_row >= params.m) continue;
            float4 value = *reinterpret_cast<const float4*>(
                tile_output + row * OutputStride + chunk);
            float* destination =
                output + (long long)global_row * params.ldc + tile_column + chunk;
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
        for (int n_atom = 0; n_atom < BN / 16; ++n_atom) {
#pragma unroll
            for (int element = 0; element < 4; ++element) {
                int row = tile_row + warp_m + m_atom * 16 + group
                    + (element >= 2 ? 8 : 0);
                int column = tile_column + warp_n + n_atom * 8
                    + 2 * thread + (element & 1);
                tf32w4_store(
                    output, row, column, acc[m_atom][n_atom][element], params);
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(128, 1)
void gemm_bi_nn_fixed_rna_tf32_exp_m128n96_bk16_s5_w4(
    float* output, const float* a, const float* b, const float* bias,
    GbfTf32W4Params params) {
    if (params.k == 0) {
        tf32w4_zero_reduction<96>(output, bias, params);
        return;
    }
    tf32w4_kernel<96>(output, a, b, bias, params);
}

extern "C" __global__ __launch_bounds__(128, 1)
void gemm_bi_nn_fixed_rna_tf32_exp_m128n128_bk16_s5_w4(
    float* output, const float* a, const float* b, const float* bias,
    GbfTf32W4Params params) {
    if (params.k == 0) {
        tf32w4_zero_reduction<128>(output, bias, params);
        return;
    }
    tf32w4_kernel<128>(output, a, b, bias, params);
}
