// BEGIN RETAINED TN_N96
namespace sm89_tf32_joint_tn_n96 {
// Ada Fixed TF32 M128xN96/BK32/S3 finalist. Optional forced route until qualified.

struct GbfTf32N96Params {
    float alpha;
    float beta;
    int m;
    int k;
    int n;
    int lda;
    int ldb;
    int ldc;
};

static_assert(sizeof(GbfTf32N96Params) == 32, "N96 parameter ABI");
static_assert(alignof(GbfTf32N96Params) == 4, "N96 parameter alignment");

__device__ __forceinline__ int tf32n96_a_index(int row, int k) {
    int chunk = (k >> 2) ^ (row & 7);
    return row * 32 + chunk * 4 + (k & 3);
}

__device__ __forceinline__ int tf32n96_b_index(int k, int column) {
    int chunk = (column >> 2) ^ ((k & 3) << 1);
    return k * 96 + chunk * 4 + (column & 3);
}

struct Tf32n96CopyPlan {
    const float* a_source[4];
    const float* b_source[3];
    unsigned a_destination[4];
    unsigned b_destination[3];
    int a_k_offset;
    int b_k_row[3];
    bool a_row_valid[4];
    int b_column_bytes[3];
};

__device__ __forceinline__ void tf32n96_copy_plan(
    float* a_stage0, float* b_stage0, const float* a, const float* b,
    const GbfTf32N96Params& params, int tile_row, int tile_column,
    Tf32n96CopyPlan& plan) {
    plan.a_k_offset = ((int)threadIdx.x & 7) * 4;
#pragma unroll
    for (int slice = 0; slice < 4; ++slice) {
        int linear = (int)threadIdx.x + slice * 256;
        int row = linear >> 3;
        int global_row = tile_row + row;
        plan.a_row_valid[slice] = global_row < params.m;
        plan.a_source[slice] =
            a + (long long)(plan.a_row_valid[slice] ? global_row : 0) * params.lda
            + plan.a_k_offset;
        plan.a_destination[slice] = (unsigned)__cvta_generic_to_shared(
            a_stage0 + tf32n96_a_index(row, plan.a_k_offset));
    }
#pragma unroll
    for (int slice = 0; slice < 3; ++slice) {
        int linear = (int)threadIdx.x + slice * 256;
        int k_row = linear / 24;
        int column = (linear % 24) * 4;
        int global_column = tile_column + column;
        int columns = params.n - global_column;
        columns = columns < 0 ? 0 : (columns > 4 ? 4 : columns);
        plan.b_k_row[slice] = k_row;
        plan.b_column_bytes[slice] = columns * 4;
        plan.b_source[slice] =
            b + (long long)k_row * params.ldb + (columns > 0 ? global_column : 0);
        plan.b_destination[slice] = (unsigned)__cvta_generic_to_shared(
            b_stage0 + tf32n96_b_index(k_row, column));
    }
}

__device__ __forceinline__ void tf32n96_stage_slice(
    const Tf32n96CopyPlan& plan, unsigned a_stage_bytes,
    unsigned b_stage_bytes, int k_base, int reduction, int issue) {
    {
        int remaining = reduction - k_base - plan.a_k_offset;
        remaining = remaining < 0 ? 0 : (remaining > 4 ? 4 : remaining);
        int bytes = plan.a_row_valid[issue] ? remaining * 4 : 0;
        gbf_tf32_copy_cg(
            plan.a_destination[issue] + a_stage_bytes,
            plan.a_source[issue], bytes);
    }
    if (issue < 3) {
        int bytes = k_base + plan.b_k_row[issue] < reduction
            ? plan.b_column_bytes[issue]
            : 0;
        gbf_tf32_copy_cg(
            plan.b_destination[issue] + b_stage_bytes,
            plan.b_source[issue], bytes);
    }
}

__device__ __forceinline__ void tf32n96_advance_plan(
    Tf32n96CopyPlan& plan, long long b_rows) {
#pragma unroll
    for (int slice = 0; slice < 4; ++slice) plan.a_source[slice] += 32;
#pragma unroll
    for (int slice = 0; slice < 3; ++slice) plan.b_source[slice] += b_rows;
}

__device__ __forceinline__ void tf32n96_stage_async(
    const Tf32n96CopyPlan& plan, unsigned a_stage_bytes,
    unsigned b_stage_bytes, int k_base, int reduction) {
#pragma unroll
    for (int issue = 0; issue < 4; ++issue) {
        tf32n96_stage_slice(
            plan, a_stage_bytes, b_stage_bytes, k_base, reduction, issue);
    }
    asm volatile("cp.async.commit_group;\n" ::);
}

struct Tf32n96Fragments {
    unsigned a[4][4];
    unsigned b[3][2];
};

struct Tf32n96FragmentOffsets {
    int a[4][4];
    int b[3][2];
};

__device__ __forceinline__ void tf32n96_fragment_offsets(
    int warp_m, int warp_n, int group, int thread, int lane,
    Tf32n96FragmentOffsets& offsets) {
    int a_row = warp_m + (lane & 15);
    int a_k = (lane >> 4) << 2;
#pragma unroll
    for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
        for (int step = 0; step < 4; ++step) {
            offsets.a[m_atom][step] =
                tf32n96_a_index(a_row + m_atom * 16, step * 8 + a_k);
        }
    }
#pragma unroll
    for (int n_atom = 0; n_atom < 3; ++n_atom) {
        int column = warp_n + n_atom * 8 + group;
        offsets.b[n_atom][0] = tf32n96_b_index(thread, column);
        offsets.b[n_atom][1] = tf32n96_b_index(thread + 4, column);
    }
}

__device__ __forceinline__ unsigned tf32n96_round(unsigned bits) {
    float value = __uint_as_float(bits);
    unsigned result;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(result) : "f"(value));
    return result;
}

__device__ __forceinline__ void tf32n96_load_fragments(
    const float* a_stage, const float* b_stage, int step,
    const Tf32n96FragmentOffsets& offsets, Tf32n96Fragments& fragments) {
    unsigned a_base = (unsigned)__cvta_generic_to_shared(a_stage);
#pragma unroll
    for (int m_atom = 0; m_atom < 4; ++m_atom) {
        unsigned address = a_base + (unsigned)offsets.a[m_atom][step] * 4U;
        unsigned raw0, raw1, raw2, raw3;
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(raw0), "=r"(raw1), "=r"(raw2), "=r"(raw3)
            : "r"(address));
        fragments.a[m_atom][0] = raw0;
        fragments.a[m_atom][1] = raw1;
        fragments.a[m_atom][2] = raw2;
        fragments.a[m_atom][3] = raw3;
    }
    const float* b_step = b_stage + step * 8 * 96;
#pragma unroll
    for (int n_atom = 0; n_atom < 3; ++n_atom) {
        fragments.b[n_atom][0] =
            tf32n96_round(__float_as_uint(b_step[offsets.b[n_atom][0]]));
        fragments.b[n_atom][1] =
            tf32n96_round(__float_as_uint(b_step[offsets.b[n_atom][1]]));
    }
}

__device__ __forceinline__ void tf32n96_mma(
    const Tf32n96Fragments& fragments, float (&acc)[4][3][4]) {
#pragma unroll
    for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 3; ++n_atom) {
            gbf_tf32_mma_m16n8k8(
                acc[m_atom][n_atom], fragments.a[m_atom], fragments.b[n_atom]);
        }
    }
}

__device__ __forceinline__ void tf32n96_store(
    float* output, int row, int column, float accumulator,
    const GbfTf32N96Params& params) {
    if (row >= params.m || column >= params.n) return;
    float* destination = output + (long long)row * params.ldc + column;
    *destination = __fmaf_rn(params.alpha, accumulator, *destination);
}

__device__ __forceinline__ void tf32n96_zero_reduction(
    float* output, const float* bias, const GbfTf32N96Params& params) {
    int column_tiles = (params.n + 95) / 96;
    int tile_row = (int)blockIdx.x / column_tiles * 128;
    int tile_column = (int)blockIdx.x % column_tiles * 96;
    for (int linear = (int)threadIdx.x; linear < 128 * 96;
         linear += (int)blockDim.x) {
        int row = tile_row + linear / 96;
        int column = tile_column + linear % 96;
        if (row < params.m && column < params.n) {
            float accumulator = bias == nullptr ? 0.0f : bias[column];
            tf32n96_store(output, row, column, accumulator, params);
        }
    }
}

__device__ __forceinline__ void tf32n96_kernel(
    float* output, const float* a, const float* b, const float* bias,
    GbfTf32N96Params params) {
    extern __shared__ __align__(16) unsigned char shared_bytes[];
    float* a_stages = reinterpret_cast<float*>(shared_bytes);
    float* b_stages = a_stages + 3 * 128 * 32;
    int column_tiles = (params.n + 95) / 96;
    int tile_row = (int)blockIdx.x / column_tiles * 128;
    int tile_column = (int)blockIdx.x % column_tiles * 96;
    int warp = (int)threadIdx.x >> 5;
    int lane = (int)threadIdx.x & 31;
    int warp_m = (warp >> 2) * 64;
    int warp_n = (warp & 3) * 24;
    int group = lane >> 2;
    int thread = lane & 3;
    float acc[4][3][4];
#pragma unroll
    for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 3; ++n_atom) {
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
    Tf32n96FragmentOffsets offsets;
    tf32n96_fragment_offsets(warp_m, warp_n, group, thread, lane, offsets);
    Tf32n96CopyPlan plan;
    tf32n96_copy_plan(
        a_stages, b_stages, a, b, params, tile_row, tile_column, plan);
    long long b_slab_rows = 32LL * params.ldb;
    unsigned tile_count = (static_cast<unsigned>(params.k) + 31U) / 32U;
#pragma unroll
    for (unsigned tile = 0; tile < 2; ++tile) {
        if (tile < tile_count) {
            tf32n96_stage_async(
                plan, tile * 128U * 32U * 4U, tile * 32U * 96U * 4U,
                (int)(tile * 32U), params.k);
            tf32n96_advance_plan(plan, b_slab_rows);
        } else {
            asm volatile("cp.async.commit_group;\n" ::);
        }
    }
    int read_stage = 0;
    for (unsigned tile = 0; tile < tile_count; ++tile) {
        asm volatile("cp.async.wait_group 1;\n" ::);
        __syncthreads();
        unsigned next = tile + 2;
        bool has_next = next < tile_count;
        int write_stage = read_stage == 0 ? 2 : read_stage - 1;
        unsigned write_a_bytes = (unsigned)write_stage * 128U * 32U * 4U;
        unsigned write_b_bytes = (unsigned)write_stage * 32U * 96U * 4U;
        const float* a_read = a_stages + read_stage * 128 * 32;
        const float* b_read = b_stages + read_stage * 32 * 96;
        Tf32n96Fragments fragments[2];
        tf32n96_load_fragments(a_read, b_read, 0, offsets, fragments[0]);
#pragma unroll
        for (int issue = 0; issue < 4; ++issue) {
            if (has_next) {
                tf32n96_stage_slice(
                    plan, write_a_bytes, write_b_bytes,
                    (int)(next * 32U), params.k, issue);
            }
            if (issue < 3) {
                tf32n96_load_fragments(
                    a_read, b_read, issue + 1, offsets, fragments[(issue + 1) & 1]);
            }
            tf32n96_mma(fragments[issue & 1], acc);
        }
        asm volatile("cp.async.commit_group;\n" ::);
        if (has_next) tf32n96_advance_plan(plan, b_slab_rows);
        if (++read_stage == 3) read_stage = 0;
    }
    __syncthreads();
    float* tile_output = reinterpret_cast<float*>(shared_bytes);
    bool vector_rows = tile_column + 96 <= params.n && (params.ldc & 3) == 0
        && gbf_aligned16(output);
    if (vector_rows) {
#pragma unroll
        for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
            for (int n_atom = 0; n_atom < 3; ++n_atom) {
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    int row = warp_m + m_atom * 16 + group + half * 8;
                    int column = warp_n + n_atom * 8 + 2 * thread;
                    *reinterpret_cast<float2*>(tile_output + row * 104 + column) =
                        make_float2(
                            acc[m_atom][n_atom][2 * half],
                            acc[m_atom][n_atom][2 * half + 1]);
                }
            }
        }
        __syncthreads();
#pragma unroll 4
        for (int linear = (int)threadIdx.x; linear < 128 * 24; linear += 256) {
            int row = linear / 24;
            int chunk = (linear % 24) * 4;
            int global_row = tile_row + row;
            if (global_row >= params.m) continue;
            float4 value = *reinterpret_cast<const float4*>(
                tile_output + row * 104 + chunk);
            float* destination =
                output + (long long)global_row * params.ldc + tile_column + chunk;
            float4 old = *reinterpret_cast<const float4*>(destination);
            value.x = __fmaf_rn(params.alpha, value.x, old.x);
            value.y = __fmaf_rn(params.alpha, value.y, old.y);
            value.z = __fmaf_rn(params.alpha, value.z, old.z);
            value.w = __fmaf_rn(params.alpha, value.w, old.w);
            *reinterpret_cast<float4*>(destination) = value;
        }
        return;
    }
#pragma unroll
    for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 3; ++n_atom) {
#pragma unroll
            for (int element = 0; element < 4; ++element) {
                int row = tile_row + warp_m + m_atom * 16 + group
                    + (element >= 2 ? 8 : 0);
                int column = tile_column + warp_n + n_atom * 8
                    + 2 * thread + (element & 1);
                tf32n96_store(
                    output, row, column, acc[m_atom][n_atom][element], params);
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(256, 1)
void gemm_bi_tn_sm89_tf32_pre_rna_m128n96_bk32_s3_v1(
    float* output, const float* a, const float* b, const float* bias,
    GbfTf32N96Params params) {
    if (params.k == 0) {
        tf32n96_zero_reduction(output, bias, params);
        return;
    }
    tf32n96_kernel(output, a, b, bias, params);
}



struct GbfTf32TnTransposeParams {
    int rows;
    int columns;
    int output_stride;
};

static_assert(sizeof(GbfTf32TnTransposeParams) == 12, "TN transpose parameter ABI");
static_assert(alignof(GbfTf32TnTransposeParams) == 4, "TN transpose parameter alignment");

extern "C" __global__ __launch_bounds__(256, 1)
void gemm_bi_tn_sm89_tf32_pre_rna_transpose_32x32_v1(
    const unsigned* input, unsigned* output, GbfTf32TnTransposeParams params) {
    if (params.rows < 0 || params.columns < 0 || params.output_stride < params.rows
        || (params.output_stride & 3) != 0) return;
    __shared__ unsigned tile[32][33];
    int input_column = (int)blockIdx.x * 32 + (int)threadIdx.x;
    int input_row_base = (int)blockIdx.y * 32 + (int)threadIdx.y;
#pragma unroll
    for (int offset = 0; offset < 32; offset += 8) {
        int input_row = input_row_base + offset;
        tile[(int)threadIdx.y + offset][(int)threadIdx.x] =
            input_row < params.rows && input_column < params.columns
            ? input[(long long)input_row * params.columns + input_column]
            : 0U;
    }
    __syncthreads();
    int output_column = (int)blockIdx.y * 32 + (int)threadIdx.x;
    int output_row_base = (int)blockIdx.x * 32 + (int)threadIdx.y;
#pragma unroll
    for (int offset = 0; offset < 32; offset += 8) {
        int output_row = output_row_base + offset;
        if (output_row < params.columns && output_column < params.output_stride) {
            output[(long long)output_row * params.output_stride + output_column] =
                output_column < params.rows
                ? tf32n96_round(tile[(int)threadIdx.x][(int)threadIdx.y + offset])
                : 0U;
        }
    }
}
}
// END RETAINED TN_N96

// BEGIN RETAINED TN_M64N64
namespace sm89_tf32_joint_tn_m64n64 {
// Ada Fixed TF32 M128xN96/BK32/S3 finalist. Optional forced route until qualified.

struct GbfTf32N96Params {
    float alpha;
    float beta;
    int m;
    int k;
    int n;
    int lda;
    int ldb;
    int ldc;
};

static_assert(sizeof(GbfTf32N96Params) == 32, "N96 parameter ABI");
static_assert(alignof(GbfTf32N96Params) == 4, "N96 parameter alignment");

__device__ __forceinline__ int tf32m64n64_a_index(int row, int k) {
    int chunk = (k >> 2) ^ (row & 7);
    return row * 32 + chunk * 4 + (k & 3);
}

__device__ __forceinline__ int tf32m64n64_b_index(int k, int column) {
    int chunk = (column >> 2) ^ ((k & 3) << 1);
    return k * 64 + chunk * 4 + (column & 3);
}

struct Tf32m64n64CopyPlan {
    const float* a_source[2];
    const float* b_source[2];
    unsigned a_destination[2];
    unsigned b_destination[2];
    int a_k_offset;
    int b_k_row[2];
    bool a_row_valid[2];
    int b_column_bytes[2];
};

__device__ __forceinline__ void tf32m64n64_copy_plan(
    float* a_stage0, float* b_stage0, const float* a, const float* b,
    const GbfTf32N96Params& params, int tile_row, int tile_column,
    Tf32m64n64CopyPlan& plan) {
    plan.a_k_offset = ((int)threadIdx.x & 7) * 4;
#pragma unroll
    for (int slice = 0; slice < 2; ++slice) {
        int linear = (int)threadIdx.x + slice * 256;
        int row = linear >> 3;
        int global_row = tile_row + row;
        plan.a_row_valid[slice] = global_row < params.m;
        plan.a_source[slice] =
            a + (long long)(plan.a_row_valid[slice] ? global_row : 0) * params.lda
            + plan.a_k_offset;
        plan.a_destination[slice] = (unsigned)__cvta_generic_to_shared(
            a_stage0 + tf32m64n64_a_index(row, plan.a_k_offset));
    }
#pragma unroll
    for (int slice = 0; slice < 2; ++slice) {
        int linear = (int)threadIdx.x + slice * 256;
        int k_row = linear / 16;
        int column = (linear % 16) * 4;
        int global_column = tile_column + column;
        int columns = params.n - global_column;
        columns = columns < 0 ? 0 : (columns > 4 ? 4 : columns);
        plan.b_k_row[slice] = k_row;
        plan.b_column_bytes[slice] = columns * 4;
        plan.b_source[slice] =
            b + (long long)k_row * params.ldb + (columns > 0 ? global_column : 0);
        plan.b_destination[slice] = (unsigned)__cvta_generic_to_shared(
            b_stage0 + tf32m64n64_b_index(k_row, column));
    }
}

__device__ __forceinline__ void tf32m64n64_stage_slice(
    const Tf32m64n64CopyPlan& plan, unsigned a_stage_bytes,
    unsigned b_stage_bytes, int k_base, int reduction, int issue) {
    int remaining = reduction - k_base - plan.a_k_offset;
    remaining = remaining < 0 ? 0 : (remaining > 4 ? 4 : remaining);
    int a_bytes = plan.a_row_valid[issue] ? remaining * 4 : 0;
    gbf_tf32_copy_cg(
        plan.a_destination[issue] + a_stage_bytes,
        plan.a_source[issue], a_bytes);
    int b_bytes = k_base + plan.b_k_row[issue] < reduction
        ? plan.b_column_bytes[issue]
        : 0;
    gbf_tf32_copy_cg(
        plan.b_destination[issue] + b_stage_bytes,
        plan.b_source[issue], b_bytes);
}

__device__ __forceinline__ void tf32m64n64_advance_plan(
    Tf32m64n64CopyPlan& plan, long long b_rows) {
#pragma unroll
    for (int slice = 0; slice < 2; ++slice) plan.a_source[slice] += 32;
#pragma unroll
    for (int slice = 0; slice < 2; ++slice) plan.b_source[slice] += b_rows;
}

__device__ __forceinline__ void tf32m64n64_stage_async(
    const Tf32m64n64CopyPlan& plan, unsigned a_stage_bytes,
    unsigned b_stage_bytes, int k_base, int reduction) {
#pragma unroll
    for (int issue = 0; issue < 2; ++issue) {
        tf32m64n64_stage_slice(
            plan, a_stage_bytes, b_stage_bytes, k_base, reduction, issue);
    }
    asm volatile("cp.async.commit_group;\n" ::);
}

struct Tf32m64n64Fragments {
    unsigned a[2][4];
    unsigned b[2][2];
};

struct Tf32m64n64FragmentOffsets {
    int a[2][4];
    int b[2][2];
};

__device__ __forceinline__ void tf32m64n64_fragment_offsets(
    int warp_m, int warp_n, int group, int thread, int lane,
    Tf32m64n64FragmentOffsets& offsets) {
    int a_row = warp_m + (lane & 15);
    int a_k = (lane >> 4) << 2;
#pragma unroll
    for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
        for (int step = 0; step < 4; ++step) {
            offsets.a[m_atom][step] =
                tf32m64n64_a_index(a_row + m_atom * 16, step * 8 + a_k);
        }
    }
#pragma unroll
    for (int n_atom = 0; n_atom < 2; ++n_atom) {
        int column = warp_n + n_atom * 8 + group;
        offsets.b[n_atom][0] = tf32m64n64_b_index(thread, column);
        offsets.b[n_atom][1] = tf32m64n64_b_index(thread + 4, column);
    }
}

__device__ __forceinline__ unsigned tf32m64n64_round(unsigned bits) {
    float value = __uint_as_float(bits);
    unsigned result;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(result) : "f"(value));
    return result;
}

__device__ __forceinline__ void tf32m64n64_load_fragments(
    const float* a_stage, const float* b_stage, int step,
    const Tf32m64n64FragmentOffsets& offsets, Tf32m64n64Fragments& fragments) {
    unsigned a_base = (unsigned)__cvta_generic_to_shared(a_stage);
#pragma unroll
    for (int m_atom = 0; m_atom < 2; ++m_atom) {
        unsigned address = a_base + (unsigned)offsets.a[m_atom][step] * 4U;
        unsigned raw0, raw1, raw2, raw3;
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(raw0), "=r"(raw1), "=r"(raw2), "=r"(raw3)
            : "r"(address));
        fragments.a[m_atom][0] = raw0;
        fragments.a[m_atom][1] = raw1;
        fragments.a[m_atom][2] = raw2;
        fragments.a[m_atom][3] = raw3;
    }
    const float* b_step = b_stage + step * 8 * 64;
#pragma unroll
    for (int n_atom = 0; n_atom < 2; ++n_atom) {
        fragments.b[n_atom][0] =
            tf32m64n64_round(__float_as_uint(b_step[offsets.b[n_atom][0]]));
        fragments.b[n_atom][1] =
            tf32m64n64_round(__float_as_uint(b_step[offsets.b[n_atom][1]]));
    }
}

__device__ __forceinline__ void tf32m64n64_mma(
    const Tf32m64n64Fragments& fragments, float (&acc)[2][2][4]) {
#pragma unroll
    for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 2; ++n_atom) {
            gbf_tf32_mma_m16n8k8(
                acc[m_atom][n_atom], fragments.a[m_atom], fragments.b[n_atom]);
        }
    }
}

__device__ __forceinline__ void tf32m64n64_store(
    float* output, int row, int column, float accumulator,
    const GbfTf32N96Params& params) {
    if (row >= params.m || column >= params.n) return;
    float* destination = output + (long long)row * params.ldc + column;
    *destination = __fmaf_rn(params.alpha, accumulator, *destination);
}

__device__ __forceinline__ void tf32m64n64_zero_reduction(
    float* output, const float* bias, const GbfTf32N96Params& params) {
    int column_tiles = (params.n + 63) / 64;
    int tile_row = (int)blockIdx.x / column_tiles * 64;
    int tile_column = (int)blockIdx.x % column_tiles * 64;
    for (int linear = (int)threadIdx.x; linear < 64 * 64;
         linear += (int)blockDim.x) {
        int row = tile_row + linear / 64;
        int column = tile_column + linear % 64;
        if (row < params.m && column < params.n) {
            float accumulator = bias == nullptr ? 0.0f : bias[column];
            tf32m64n64_store(output, row, column, accumulator, params);
        }
    }
}

__device__ __forceinline__ void tf32m64n64_kernel(
    float* output, const float* a, const float* b, const float* bias,
    GbfTf32N96Params params) {
    extern __shared__ __align__(16) unsigned char shared_bytes[];
    float* a_stages = reinterpret_cast<float*>(shared_bytes);
    float* b_stages = a_stages + 3 * 64 * 32;
    int column_tiles = (params.n + 63) / 64;
    int tile_row = (int)blockIdx.x / column_tiles * 64;
    int tile_column = (int)blockIdx.x % column_tiles * 64;
    int warp = (int)threadIdx.x >> 5;
    int lane = (int)threadIdx.x & 31;
    int warp_m = (warp >> 2) * 32;
    int warp_n = (warp & 3) * 16;
    int group = lane >> 2;
    int thread = lane & 3;
    float acc[2][2][4];
#pragma unroll
    for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 2; ++n_atom) {
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
    Tf32m64n64FragmentOffsets offsets;
    tf32m64n64_fragment_offsets(warp_m, warp_n, group, thread, lane, offsets);
    Tf32m64n64CopyPlan plan;
    tf32m64n64_copy_plan(
        a_stages, b_stages, a, b, params, tile_row, tile_column, plan);
    long long b_slab_rows = 32LL * params.ldb;
    unsigned tile_count = (static_cast<unsigned>(params.k) + 31U) / 32U;
#pragma unroll
    for (unsigned tile = 0; tile < 2; ++tile) {
        if (tile < tile_count) {
            tf32m64n64_stage_async(
                plan, tile * 64U * 32U * 4U, tile * 32U * 64U * 4U,
                (int)(tile * 32U), params.k);
            tf32m64n64_advance_plan(plan, b_slab_rows);
        } else {
            asm volatile("cp.async.commit_group;\n" ::);
        }
    }
    int read_stage = 0;
    for (unsigned tile = 0; tile < tile_count; ++tile) {
        asm volatile("cp.async.wait_group 1;\n" ::);
        __syncthreads();
        unsigned next = tile + 2;
        bool has_next = next < tile_count;
        int write_stage = read_stage == 0 ? 2 : read_stage - 1;
        unsigned write_a_bytes = (unsigned)write_stage * 64U * 32U * 4U;
        unsigned write_b_bytes = (unsigned)write_stage * 32U * 64U * 4U;
        const float* a_read = a_stages + read_stage * 64 * 32;
        const float* b_read = b_stages + read_stage * 32 * 64;
        Tf32m64n64Fragments fragments[2];
        tf32m64n64_load_fragments(a_read, b_read, 0, offsets, fragments[0]);
#pragma unroll
        for (int issue = 0; issue < 4; ++issue) {
            if (has_next && issue < 2) {
                tf32m64n64_stage_slice(
                    plan, write_a_bytes, write_b_bytes,
                    (int)(next * 32U), params.k, issue);
            }
            if (issue < 3) {
                tf32m64n64_load_fragments(
                    a_read, b_read, issue + 1, offsets, fragments[(issue + 1) & 1]);
            }
            tf32m64n64_mma(fragments[issue & 1], acc);
        }
        asm volatile("cp.async.commit_group;\n" ::);
        if (has_next) tf32m64n64_advance_plan(plan, b_slab_rows);
        if (++read_stage == 3) read_stage = 0;
    }
    __syncthreads();
    float* tile_output = reinterpret_cast<float*>(shared_bytes);
    bool vector_rows = tile_column + 64 <= params.n && (params.ldc & 3) == 0
        && gbf_aligned16(output);
    if (vector_rows) {
#pragma unroll
        for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
            for (int n_atom = 0; n_atom < 2; ++n_atom) {
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    int row = warp_m + m_atom * 16 + group + half * 8;
                    int column = warp_n + n_atom * 8 + 2 * thread;
                    *reinterpret_cast<float2*>(tile_output + row * 72 + column) =
                        make_float2(
                            acc[m_atom][n_atom][2 * half],
                            acc[m_atom][n_atom][2 * half + 1]);
                }
            }
        }
        __syncthreads();
#pragma unroll 4
        for (int linear = (int)threadIdx.x; linear < 64 * 16; linear += 256) {
            int row = linear / 16;
            int chunk = (linear % 16) * 4;
            int global_row = tile_row + row;
            if (global_row >= params.m) continue;
            float4 value = *reinterpret_cast<const float4*>(
                tile_output + row * 72 + chunk);
            float* destination =
                output + (long long)global_row * params.ldc + tile_column + chunk;
            float4 old = *reinterpret_cast<const float4*>(destination);
            value.x = __fmaf_rn(params.alpha, value.x, old.x);
            value.y = __fmaf_rn(params.alpha, value.y, old.y);
            value.z = __fmaf_rn(params.alpha, value.z, old.z);
            value.w = __fmaf_rn(params.alpha, value.w, old.w);
            *reinterpret_cast<float4*>(destination) = value;
        }
        return;
    }
#pragma unroll
    for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 2; ++n_atom) {
#pragma unroll
            for (int element = 0; element < 4; ++element) {
                int row = tile_row + warp_m + m_atom * 16 + group
                    + (element >= 2 ? 8 : 0);
                int column = tile_column + warp_n + n_atom * 8
                    + 2 * thread + (element & 1);
                tf32m64n64_store(
                    output, row, column, acc[m_atom][n_atom][element], params);
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(256, 1)
void gemm_bi_tn_sm89_tf32_pre_rna_m64n64_bk32_s3_v1(
    float* output, const float* a, const float* b, const float* bias,
    GbfTf32N96Params params) {
    if (params.k == 0) {
        tf32m64n64_zero_reduction(output, bias, params);
        return;
    }
    tf32m64n64_kernel(output, a, b, bias, params);
}



struct GbfTf32TnTransposeParams {
    int rows;
    int columns;
    int output_stride;
};

static_assert(sizeof(GbfTf32TnTransposeParams) == 12, "TN transpose parameter ABI");
static_assert(alignof(GbfTf32TnTransposeParams) == 4, "TN transpose parameter alignment");

}
// END RETAINED TN_M64N64

// BEGIN RETAINED NN_DIRECT_N96
namespace sm89_tf32_joint_nn_direct_n96 {
// Ada Fixed TF32 M128xN96/BK32/S3 finalist. Optional forced route until qualified.

struct GbfTf32N96Params {
    float alpha;
    float beta;
    int m;
    int k;
    int n;
    int lda;
    int ldb;
    int ldc;
};

static_assert(sizeof(GbfTf32N96Params) == 32, "N96 parameter ABI");
static_assert(alignof(GbfTf32N96Params) == 4, "N96 parameter alignment");

__device__ __forceinline__ int tf32n96_a_index(int row, int k) {
    int chunk = (k >> 2) ^ (row & 7);
    return row * 32 + chunk * 4 + (k & 3);
}

__device__ __forceinline__ int tf32n96_b_index(int k, int column) {
    int chunk = (column >> 2) ^ ((k & 3) << 1);
    return k * 96 + chunk * 4 + (column & 3);
}

struct Tf32n96CopyPlan {
    const float* a_source[4];
    const float* b_source[3];
    unsigned a_destination[4];
    unsigned b_destination[3];
    int a_k_offset;
    int b_k_row[3];
    bool a_row_valid[4];
    int b_column_bytes[3];
};

__device__ __forceinline__ void tf32n96_copy_plan(
    float* a_stage0, float* b_stage0, const float* a, const float* b,
    const GbfTf32N96Params& params, int tile_row, int tile_column,
    Tf32n96CopyPlan& plan) {
    plan.a_k_offset = ((int)threadIdx.x & 7) * 4;
#pragma unroll
    for (int slice = 0; slice < 4; ++slice) {
        int linear = (int)threadIdx.x + slice * 256;
        int row = linear >> 3;
        int global_row = tile_row + row;
        plan.a_row_valid[slice] = global_row < params.m;
        plan.a_source[slice] =
            a + (long long)(plan.a_row_valid[slice] ? global_row : 0) * params.lda
            + plan.a_k_offset;
        plan.a_destination[slice] = (unsigned)__cvta_generic_to_shared(
            a_stage0 + tf32n96_a_index(row, plan.a_k_offset));
    }
#pragma unroll
    for (int slice = 0; slice < 3; ++slice) {
        int linear = (int)threadIdx.x + slice * 256;
        int k_row = linear / 24;
        int column = (linear % 24) * 4;
        int global_column = tile_column + column;
        int columns = params.n - global_column;
        columns = columns < 0 ? 0 : (columns > 4 ? 4 : columns);
        plan.b_k_row[slice] = k_row;
        plan.b_column_bytes[slice] = columns * 4;
        plan.b_source[slice] =
            b + (long long)k_row * params.ldb + (columns > 0 ? global_column : 0);
        plan.b_destination[slice] = (unsigned)__cvta_generic_to_shared(
            b_stage0 + tf32n96_b_index(k_row, column));
    }
}

__device__ __forceinline__ void tf32n96_stage_slice(
    const Tf32n96CopyPlan& plan, unsigned a_stage_bytes,
    unsigned b_stage_bytes, int k_base, int reduction, int issue) {
    {
        int remaining = reduction - k_base - plan.a_k_offset;
        remaining = remaining < 0 ? 0 : (remaining > 4 ? 4 : remaining);
        int bytes = plan.a_row_valid[issue] ? remaining * 4 : 0;
        gbf_tf32_copy_cg(
            plan.a_destination[issue] + a_stage_bytes,
            plan.a_source[issue], bytes);
    }
    if (issue < 3) {
        int bytes = k_base + plan.b_k_row[issue] < reduction
            ? plan.b_column_bytes[issue]
            : 0;
        gbf_tf32_copy_cg(
            plan.b_destination[issue] + b_stage_bytes,
            plan.b_source[issue], bytes);
    }
}

__device__ __forceinline__ void tf32n96_advance_plan(
    Tf32n96CopyPlan& plan, long long b_rows) {
#pragma unroll
    for (int slice = 0; slice < 4; ++slice) plan.a_source[slice] += 32;
#pragma unroll
    for (int slice = 0; slice < 3; ++slice) plan.b_source[slice] += b_rows;
}

__device__ __forceinline__ void tf32n96_stage_async(
    const Tf32n96CopyPlan& plan, unsigned a_stage_bytes,
    unsigned b_stage_bytes, int k_base, int reduction) {
#pragma unroll
    for (int issue = 0; issue < 4; ++issue) {
        tf32n96_stage_slice(
            plan, a_stage_bytes, b_stage_bytes, k_base, reduction, issue);
    }
    asm volatile("cp.async.commit_group;\n" ::);
}

struct Tf32n96Fragments {
    unsigned a[4][4];
    unsigned b[3][2];
};

struct Tf32n96FragmentOffsets {
    int a[4][4];
    int b[3][2];
};

__device__ __forceinline__ void tf32n96_fragment_offsets(
    int warp_m, int warp_n, int group, int thread, int lane,
    Tf32n96FragmentOffsets& offsets) {
    int a_row = warp_m + (lane & 15);
    int a_k = (lane >> 4) << 2;
#pragma unroll
    for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
        for (int step = 0; step < 4; ++step) {
            offsets.a[m_atom][step] =
                tf32n96_a_index(a_row + m_atom * 16, step * 8 + a_k);
        }
    }
#pragma unroll
    for (int n_atom = 0; n_atom < 3; ++n_atom) {
        int column = warp_n + n_atom * 8 + group;
        offsets.b[n_atom][0] = tf32n96_b_index(thread, column);
        offsets.b[n_atom][1] = tf32n96_b_index(thread + 4, column);
    }
}

__device__ __forceinline__ unsigned tf32n96_round(unsigned bits) {
    return bits + 0x1000U;
}

__device__ __forceinline__ void tf32n96_load_fragments(
    const float* a_stage, const float* b_stage, int step,
    const Tf32n96FragmentOffsets& offsets, Tf32n96Fragments& fragments) {
    unsigned a_base = (unsigned)__cvta_generic_to_shared(a_stage);
#pragma unroll
    for (int m_atom = 0; m_atom < 4; ++m_atom) {
        unsigned address = a_base + (unsigned)offsets.a[m_atom][step] * 4U;
        unsigned raw0, raw1, raw2, raw3;
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(raw0), "=r"(raw1), "=r"(raw2), "=r"(raw3)
            : "r"(address));
        fragments.a[m_atom][0] = tf32n96_round(raw0);
        fragments.a[m_atom][1] = tf32n96_round(raw1);
        fragments.a[m_atom][2] = tf32n96_round(raw2);
        fragments.a[m_atom][3] = tf32n96_round(raw3);
    }
    const float* b_step = b_stage + step * 8 * 96;
#pragma unroll
    for (int n_atom = 0; n_atom < 3; ++n_atom) {
        fragments.b[n_atom][0] =
            tf32n96_round(__float_as_uint(b_step[offsets.b[n_atom][0]]));
        fragments.b[n_atom][1] =
            tf32n96_round(__float_as_uint(b_step[offsets.b[n_atom][1]]));
    }
}

__device__ __forceinline__ void tf32n96_mma(
    const Tf32n96Fragments& fragments, float (&acc)[4][3][4]) {
#pragma unroll
    for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 3; ++n_atom) {
            gbf_tf32_mma_m16n8k8(
                acc[m_atom][n_atom], fragments.a[m_atom], fragments.b[n_atom]);
        }
    }
}

__device__ __forceinline__ void tf32n96_store(
    float* output, int row, int column, float accumulator,
    const GbfTf32N96Params& params) {
    if (row >= params.m || column >= params.n) return;
    float value = params.alpha == 1.0f
        ? accumulator
        : __fmul_rn(params.alpha, accumulator);
    float* destination = output + (long long)row * params.ldc + column;
    if (params.beta != 0.0f) value = __fmaf_rn(params.beta, *destination, value);
    *destination = value;
}

__device__ __forceinline__ void tf32n96_zero_reduction(
    float* output, const float* bias, const GbfTf32N96Params& params) {
    int column_tiles = (params.n + 95) / 96;
    int tile_row = (int)blockIdx.x / column_tiles * 128;
    int tile_column = (int)blockIdx.x % column_tiles * 96;
    for (int linear = (int)threadIdx.x; linear < 128 * 96;
         linear += (int)blockDim.x) {
        int row = tile_row + linear / 96;
        int column = tile_column + linear % 96;
        if (row < params.m && column < params.n) {
            float accumulator = bias == nullptr ? 0.0f : bias[column];
            tf32n96_store(output, row, column, accumulator, params);
        }
    }
}

__device__ __forceinline__ void tf32n96_kernel(
    float* output, const float* a, const float* b, const float* bias,
    GbfTf32N96Params params) {
    extern __shared__ __align__(16) unsigned char shared_bytes[];
    float* a_stages = reinterpret_cast<float*>(shared_bytes);
    float* b_stages = a_stages + 3 * 128 * 32;
    int column_tiles = (params.n + 95) / 96;
    int tile_row = (int)blockIdx.x / column_tiles * 128;
    int tile_column = (int)blockIdx.x % column_tiles * 96;
    int warp = (int)threadIdx.x >> 5;
    int lane = (int)threadIdx.x & 31;
    int warp_m = (warp >> 2) * 64;
    int warp_n = (warp & 3) * 24;
    int group = lane >> 2;
    int thread = lane & 3;
    float acc[4][3][4];
#pragma unroll
    for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 3; ++n_atom) {
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
    Tf32n96FragmentOffsets offsets;
    tf32n96_fragment_offsets(warp_m, warp_n, group, thread, lane, offsets);
    Tf32n96CopyPlan plan;
    tf32n96_copy_plan(
        a_stages, b_stages, a, b, params, tile_row, tile_column, plan);
    long long b_slab_rows = 32LL * params.ldb;
    unsigned tile_count = (static_cast<unsigned>(params.k) + 31U) / 32U;
#pragma unroll
    for (unsigned tile = 0; tile < 2; ++tile) {
        if (tile < tile_count) {
            tf32n96_stage_async(
                plan, tile * 128U * 32U * 4U, tile * 32U * 96U * 4U,
                (int)(tile * 32U), params.k);
            tf32n96_advance_plan(plan, b_slab_rows);
        } else {
            asm volatile("cp.async.commit_group;\n" ::);
        }
    }
    int read_stage = 0;
    for (unsigned tile = 0; tile < tile_count; ++tile) {
        asm volatile("cp.async.wait_group 1;\n" ::);
        __syncthreads();
        unsigned next = tile + 2;
        bool has_next = next < tile_count;
        int write_stage = read_stage == 0 ? 2 : read_stage - 1;
        unsigned write_a_bytes = (unsigned)write_stage * 128U * 32U * 4U;
        unsigned write_b_bytes = (unsigned)write_stage * 32U * 96U * 4U;
        const float* a_read = a_stages + read_stage * 128 * 32;
        const float* b_read = b_stages + read_stage * 32 * 96;
        Tf32n96Fragments fragments[2];
        tf32n96_load_fragments(a_read, b_read, 0, offsets, fragments[0]);
#pragma unroll
        for (int issue = 0; issue < 4; ++issue) {
            if (has_next) {
                tf32n96_stage_slice(
                    plan, write_a_bytes, write_b_bytes,
                    (int)(next * 32U), params.k, issue);
            }
            if (issue < 3) {
                tf32n96_load_fragments(
                    a_read, b_read, issue + 1, offsets, fragments[(issue + 1) & 1]);
            }
            tf32n96_mma(fragments[issue & 1], acc);
        }
        asm volatile("cp.async.commit_group;\n" ::);
        if (has_next) tf32n96_advance_plan(plan, b_slab_rows);
        if (++read_stage == 3) read_stage = 0;
    }
    __syncthreads();
    bool direct_pair_epilogue = params.alpha == 1.0f && params.beta == 0.0f
        && tile_row + 128 <= params.m && tile_column + 96 <= params.n
        && (params.ldc & 1) == 0
        && (reinterpret_cast<unsigned long long>(output) & 7ULL) == 0ULL;
    if (direct_pair_epilogue) {
#pragma unroll
        for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
            for (int n_atom = 0; n_atom < 3; ++n_atom) {
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    int row = tile_row + warp_m + m_atom * 16 + group + half * 8;
                    int column = tile_column + warp_n + n_atom * 8 + 2 * thread;
                    float* destination = output + (long long)row * params.ldc + column;
                    *reinterpret_cast<float2*>(destination) = make_float2(
                        acc[m_atom][n_atom][2 * half],
                        acc[m_atom][n_atom][2 * half + 1]);
                }
            }
        }
        return;
    }
    float* tile_output = reinterpret_cast<float*>(shared_bytes);
    bool vector_rows = tile_column + 96 <= params.n && (params.ldc & 3) == 0
        && gbf_aligned16(output);
    if (vector_rows) {
#pragma unroll
        for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
            for (int n_atom = 0; n_atom < 3; ++n_atom) {
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    int row = warp_m + m_atom * 16 + group + half * 8;
                    int column = warp_n + n_atom * 8 + 2 * thread;
                    *reinterpret_cast<float2*>(tile_output + row * 104 + column) =
                        make_float2(
                            acc[m_atom][n_atom][2 * half],
                            acc[m_atom][n_atom][2 * half + 1]);
                }
            }
        }
        __syncthreads();
        bool scale = params.alpha != 1.0f;
        bool blend = params.beta != 0.0f;
#pragma unroll 4
        for (int linear = (int)threadIdx.x; linear < 128 * 24; linear += 256) {
            int row = linear / 24;
            int chunk = (linear % 24) * 4;
            int global_row = tile_row + row;
            if (global_row >= params.m) continue;
            float4 value = *reinterpret_cast<const float4*>(
                tile_output + row * 104 + chunk);
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
        for (int n_atom = 0; n_atom < 3; ++n_atom) {
#pragma unroll
            for (int element = 0; element < 4; ++element) {
                int row = tile_row + warp_m + m_atom * 16 + group
                    + (element >= 2 ? 8 : 0);
                int column = tile_column + warp_n + n_atom * 8
                    + 2 * thread + (element & 1);
                tf32n96_store(
                    output, row, column, acc[m_atom][n_atom][element], params);
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(256, 1)
void gemm_bi_nn_sm89_tf32_addhalf_m128n96_bk32_s3_direct_v1(
    float* output, const float* a, const float* b, const float* bias,
    GbfTf32N96Params params) {
    if (params.k == 0) {
        tf32n96_zero_reduction(output, bias, params);
        return;
    }
    tf32n96_kernel(output, a, b, bias, params);
}

}
// END RETAINED NN_DIRECT_N96

// BEGIN RETAINED NN_N96
namespace sm89_tf32_joint_nn_n96 {
// Ada Fixed TF32 M128xN96/BK32/S3 finalist. Optional forced route until qualified.

struct GbfTf32N96Params {
    float alpha;
    float beta;
    int m;
    int k;
    int n;
    int lda;
    int ldb;
    int ldc;
};

static_assert(sizeof(GbfTf32N96Params) == 32, "N96 parameter ABI");
static_assert(alignof(GbfTf32N96Params) == 4, "N96 parameter alignment");

__device__ __forceinline__ int tf32n96_a_index(int row, int k) {
    int chunk = (k >> 2) ^ (row & 7);
    return row * 32 + chunk * 4 + (k & 3);
}

__device__ __forceinline__ int tf32n96_b_index(int k, int column) {
    int chunk = (column >> 2) ^ ((k & 3) << 1);
    return k * 96 + chunk * 4 + (column & 3);
}

struct Tf32n96CopyPlan {
    const float* a_source[4];
    const float* b_source[3];
    unsigned a_destination[4];
    unsigned b_destination[3];
    int a_k_offset;
    int b_k_row[3];
    bool a_row_valid[4];
    int b_column_bytes[3];
};

__device__ __forceinline__ void tf32n96_copy_plan(
    float* a_stage0, float* b_stage0, const float* a, const float* b,
    const GbfTf32N96Params& params, int tile_row, int tile_column,
    Tf32n96CopyPlan& plan) {
    plan.a_k_offset = ((int)threadIdx.x & 7) * 4;
#pragma unroll
    for (int slice = 0; slice < 4; ++slice) {
        int linear = (int)threadIdx.x + slice * 256;
        int row = linear >> 3;
        int global_row = tile_row + row;
        plan.a_row_valid[slice] = global_row < params.m;
        plan.a_source[slice] =
            a + (long long)(plan.a_row_valid[slice] ? global_row : 0) * params.lda
            + plan.a_k_offset;
        plan.a_destination[slice] = (unsigned)__cvta_generic_to_shared(
            a_stage0 + tf32n96_a_index(row, plan.a_k_offset));
    }
#pragma unroll
    for (int slice = 0; slice < 3; ++slice) {
        int linear = (int)threadIdx.x + slice * 256;
        int k_row = linear / 24;
        int column = (linear % 24) * 4;
        int global_column = tile_column + column;
        int columns = params.n - global_column;
        columns = columns < 0 ? 0 : (columns > 4 ? 4 : columns);
        plan.b_k_row[slice] = k_row;
        plan.b_column_bytes[slice] = columns * 4;
        plan.b_source[slice] =
            b + (long long)k_row * params.ldb + (columns > 0 ? global_column : 0);
        plan.b_destination[slice] = (unsigned)__cvta_generic_to_shared(
            b_stage0 + tf32n96_b_index(k_row, column));
    }
}

__device__ __forceinline__ void tf32n96_stage_slice(
    const Tf32n96CopyPlan& plan, unsigned a_stage_bytes,
    unsigned b_stage_bytes, int k_base, int reduction, int issue) {
    {
        int remaining = reduction - k_base - plan.a_k_offset;
        remaining = remaining < 0 ? 0 : (remaining > 4 ? 4 : remaining);
        int bytes = plan.a_row_valid[issue] ? remaining * 4 : 0;
        gbf_tf32_copy_cg(
            plan.a_destination[issue] + a_stage_bytes,
            plan.a_source[issue], bytes);
    }
    if (issue < 3) {
        int bytes = k_base + plan.b_k_row[issue] < reduction
            ? plan.b_column_bytes[issue]
            : 0;
        gbf_tf32_copy_cg(
            plan.b_destination[issue] + b_stage_bytes,
            plan.b_source[issue], bytes);
    }
}

__device__ __forceinline__ void tf32n96_advance_plan(
    Tf32n96CopyPlan& plan, long long b_rows) {
#pragma unroll
    for (int slice = 0; slice < 4; ++slice) plan.a_source[slice] += 32;
#pragma unroll
    for (int slice = 0; slice < 3; ++slice) plan.b_source[slice] += b_rows;
}

__device__ __forceinline__ void tf32n96_stage_async(
    const Tf32n96CopyPlan& plan, unsigned a_stage_bytes,
    unsigned b_stage_bytes, int k_base, int reduction) {
#pragma unroll
    for (int issue = 0; issue < 4; ++issue) {
        tf32n96_stage_slice(
            plan, a_stage_bytes, b_stage_bytes, k_base, reduction, issue);
    }
    asm volatile("cp.async.commit_group;\n" ::);
}

struct Tf32n96Fragments {
    unsigned a[4][4];
    unsigned b[3][2];
};

struct Tf32n96FragmentOffsets {
    int a[4][4];
    int b[3][2];
};

__device__ __forceinline__ void tf32n96_fragment_offsets(
    int warp_m, int warp_n, int group, int thread, int lane,
    Tf32n96FragmentOffsets& offsets) {
    int a_row = warp_m + (lane & 15);
    int a_k = (lane >> 4) << 2;
#pragma unroll
    for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
        for (int step = 0; step < 4; ++step) {
            offsets.a[m_atom][step] =
                tf32n96_a_index(a_row + m_atom * 16, step * 8 + a_k);
        }
    }
#pragma unroll
    for (int n_atom = 0; n_atom < 3; ++n_atom) {
        int column = warp_n + n_atom * 8 + group;
        offsets.b[n_atom][0] = tf32n96_b_index(thread, column);
        offsets.b[n_atom][1] = tf32n96_b_index(thread + 4, column);
    }
}

__device__ __forceinline__ unsigned tf32n96_round(unsigned bits) {
    return bits + 0x1000U;
}

__device__ __forceinline__ void tf32n96_load_fragments(
    const float* a_stage, const float* b_stage, int step,
    const Tf32n96FragmentOffsets& offsets, Tf32n96Fragments& fragments) {
    unsigned a_base = (unsigned)__cvta_generic_to_shared(a_stage);
#pragma unroll
    for (int m_atom = 0; m_atom < 4; ++m_atom) {
        unsigned address = a_base + (unsigned)offsets.a[m_atom][step] * 4U;
        unsigned raw0, raw1, raw2, raw3;
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(raw0), "=r"(raw1), "=r"(raw2), "=r"(raw3)
            : "r"(address));
        fragments.a[m_atom][0] = tf32n96_round(raw0);
        fragments.a[m_atom][1] = tf32n96_round(raw1);
        fragments.a[m_atom][2] = tf32n96_round(raw2);
        fragments.a[m_atom][3] = tf32n96_round(raw3);
    }
    const float* b_step = b_stage + step * 8 * 96;
#pragma unroll
    for (int n_atom = 0; n_atom < 3; ++n_atom) {
        fragments.b[n_atom][0] =
            tf32n96_round(__float_as_uint(b_step[offsets.b[n_atom][0]]));
        fragments.b[n_atom][1] =
            tf32n96_round(__float_as_uint(b_step[offsets.b[n_atom][1]]));
    }
}

__device__ __forceinline__ void tf32n96_mma(
    const Tf32n96Fragments& fragments, float (&acc)[4][3][4]) {
#pragma unroll
    for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 3; ++n_atom) {
            gbf_tf32_mma_m16n8k8(
                acc[m_atom][n_atom], fragments.a[m_atom], fragments.b[n_atom]);
        }
    }
}

__device__ __forceinline__ void tf32n96_store(
    float* output, int row, int column, float accumulator,
    const GbfTf32N96Params& params) {
    if (row >= params.m || column >= params.n) return;
    float value = params.alpha == 1.0f
        ? accumulator
        : __fmul_rn(params.alpha, accumulator);
    float* destination = output + (long long)row * params.ldc + column;
    if (params.beta != 0.0f) value = __fmaf_rn(params.beta, *destination, value);
    *destination = value;
}

__device__ __forceinline__ void tf32n96_zero_reduction(
    float* output, const float* bias, const GbfTf32N96Params& params) {
    int column_tiles = (params.n + 95) / 96;
    int tile_row = (int)blockIdx.x / column_tiles * 128;
    int tile_column = (int)blockIdx.x % column_tiles * 96;
    for (int linear = (int)threadIdx.x; linear < 128 * 96;
         linear += (int)blockDim.x) {
        int row = tile_row + linear / 96;
        int column = tile_column + linear % 96;
        if (row < params.m && column < params.n) {
            float accumulator = bias == nullptr ? 0.0f : bias[column];
            tf32n96_store(output, row, column, accumulator, params);
        }
    }
}

__device__ __forceinline__ void tf32n96_kernel(
    float* output, const float* a, const float* b, const float* bias,
    GbfTf32N96Params params) {
    extern __shared__ __align__(16) unsigned char shared_bytes[];
    float* a_stages = reinterpret_cast<float*>(shared_bytes);
    float* b_stages = a_stages + 3 * 128 * 32;
    int column_tiles = (params.n + 95) / 96;
    int tile_row = (int)blockIdx.x / column_tiles * 128;
    int tile_column = (int)blockIdx.x % column_tiles * 96;
    int warp = (int)threadIdx.x >> 5;
    int lane = (int)threadIdx.x & 31;
    int warp_m = (warp >> 2) * 64;
    int warp_n = (warp & 3) * 24;
    int group = lane >> 2;
    int thread = lane & 3;
    float acc[4][3][4];
#pragma unroll
    for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 3; ++n_atom) {
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
    Tf32n96FragmentOffsets offsets;
    tf32n96_fragment_offsets(warp_m, warp_n, group, thread, lane, offsets);
    Tf32n96CopyPlan plan;
    tf32n96_copy_plan(
        a_stages, b_stages, a, b, params, tile_row, tile_column, plan);
    long long b_slab_rows = 32LL * params.ldb;
    unsigned tile_count = (static_cast<unsigned>(params.k) + 31U) / 32U;
#pragma unroll
    for (unsigned tile = 0; tile < 2; ++tile) {
        if (tile < tile_count) {
            tf32n96_stage_async(
                plan, tile * 128U * 32U * 4U, tile * 32U * 96U * 4U,
                (int)(tile * 32U), params.k);
            tf32n96_advance_plan(plan, b_slab_rows);
        } else {
            asm volatile("cp.async.commit_group;\n" ::);
        }
    }
    int read_stage = 0;
    for (unsigned tile = 0; tile < tile_count; ++tile) {
        asm volatile("cp.async.wait_group 1;\n" ::);
        __syncthreads();
        unsigned next = tile + 2;
        bool has_next = next < tile_count;
        int write_stage = read_stage == 0 ? 2 : read_stage - 1;
        unsigned write_a_bytes = (unsigned)write_stage * 128U * 32U * 4U;
        unsigned write_b_bytes = (unsigned)write_stage * 32U * 96U * 4U;
        const float* a_read = a_stages + read_stage * 128 * 32;
        const float* b_read = b_stages + read_stage * 32 * 96;
        Tf32n96Fragments fragments[2];
        tf32n96_load_fragments(a_read, b_read, 0, offsets, fragments[0]);
#pragma unroll
        for (int issue = 0; issue < 4; ++issue) {
            if (has_next) {
                tf32n96_stage_slice(
                    plan, write_a_bytes, write_b_bytes,
                    (int)(next * 32U), params.k, issue);
            }
            if (issue < 3) {
                tf32n96_load_fragments(
                    a_read, b_read, issue + 1, offsets, fragments[(issue + 1) & 1]);
            }
            tf32n96_mma(fragments[issue & 1], acc);
        }
        asm volatile("cp.async.commit_group;\n" ::);
        if (has_next) tf32n96_advance_plan(plan, b_slab_rows);
        if (++read_stage == 3) read_stage = 0;
    }
    __syncthreads();
    float* tile_output = reinterpret_cast<float*>(shared_bytes);
    bool vector_rows = tile_column + 96 <= params.n && (params.ldc & 3) == 0
        && gbf_aligned16(output);
    if (vector_rows) {
#pragma unroll
        for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
            for (int n_atom = 0; n_atom < 3; ++n_atom) {
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    int row = warp_m + m_atom * 16 + group + half * 8;
                    int column = warp_n + n_atom * 8 + 2 * thread;
                    *reinterpret_cast<float2*>(tile_output + row * 104 + column) =
                        make_float2(
                            acc[m_atom][n_atom][2 * half],
                            acc[m_atom][n_atom][2 * half + 1]);
                }
            }
        }
        __syncthreads();
        bool scale = params.alpha != 1.0f;
        bool blend = params.beta != 0.0f;
#pragma unroll 4
        for (int linear = (int)threadIdx.x; linear < 128 * 24; linear += 256) {
            int row = linear / 24;
            int chunk = (linear % 24) * 4;
            int global_row = tile_row + row;
            if (global_row >= params.m) continue;
            float4 value = *reinterpret_cast<const float4*>(
                tile_output + row * 104 + chunk);
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
        for (int n_atom = 0; n_atom < 3; ++n_atom) {
#pragma unroll
            for (int element = 0; element < 4; ++element) {
                int row = tile_row + warp_m + m_atom * 16 + group
                    + (element >= 2 ? 8 : 0);
                int column = tile_column + warp_n + n_atom * 8
                    + 2 * thread + (element & 1);
                tf32n96_store(
                    output, row, column, acc[m_atom][n_atom][element], params);
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(256, 1)
void gemm_bi_nn_sm89_tf32_addhalf_m128n96_bk32_s3_v1(
    float* output, const float* a, const float* b, const float* bias,
    GbfTf32N96Params params) {
    if (params.k == 0) {
        tf32n96_zero_reduction(output, bias, params);
        return;
    }
    tf32n96_kernel(output, a, b, bias, params);
}

}
// END RETAINED NN_N96

// BEGIN RETAINED TN_M64N96_S2
namespace sm89_tf32_joint_tn_m64n96_s2 {
// Ada TF32 TN M64xN96/BK32/S2 Prism winner for CUDA 13.2.

struct GbfTf32N96Params {
    float alpha;
    float beta;
    int m;
    int k;
    int n;
    int lda;
    int ldb;
    int ldc;
};

static_assert(sizeof(GbfTf32N96Params) == 32, "N96 parameter ABI");
static_assert(alignof(GbfTf32N96Params) == 4, "N96 parameter alignment");

__device__ __forceinline__ int tf32m64n96_a_index(int row, int k) {
    int chunk = (k >> 2) ^ (row & 7);
    return row * 32 + chunk * 4 + (k & 3);
}

__device__ __forceinline__ int tf32m64n96_b_index(int k, int column) {
    int chunk = (column >> 2) ^ ((k & 3) << 1);
    return k * 96 + chunk * 4 + (column & 3);
}

struct Tf32m64n96CopyPlan {
    const float* a_source[2];
    const float* b_source[3];
    unsigned a_destination[2];
    unsigned b_destination[3];
    int a_k_offset;
    int b_k_row[3];
    bool a_row_valid[2];
    int b_column_bytes[3];
};

__device__ __forceinline__ void tf32m64n96_copy_plan(
    float* a_stage0, float* b_stage0, const float* a, const float* b,
    const GbfTf32N96Params& params, int tile_row, int tile_column,
    Tf32m64n96CopyPlan& plan) {
    plan.a_k_offset = ((int)threadIdx.x & 7) * 4;
#pragma unroll
    for (int slice = 0; slice < 2; ++slice) {
        int linear = (int)threadIdx.x + slice * 256;
        int row = linear >> 3;
        int global_row = tile_row + row;
        plan.a_row_valid[slice] = global_row < params.m;
        plan.a_source[slice] =
            a + (long long)(plan.a_row_valid[slice] ? global_row : 0) * params.lda
            + plan.a_k_offset;
        plan.a_destination[slice] = (unsigned)__cvta_generic_to_shared(
            a_stage0 + tf32m64n96_a_index(row, plan.a_k_offset));
    }
#pragma unroll
    for (int slice = 0; slice < 3; ++slice) {
        int linear = (int)threadIdx.x + slice * 256;
        int k_row = linear / 24;
        int column = (linear % 24) * 4;
        int global_column = tile_column + column;
        int columns = params.n - global_column;
        columns = columns < 0 ? 0 : (columns > 4 ? 4 : columns);
        plan.b_k_row[slice] = k_row;
        plan.b_column_bytes[slice] = columns * 4;
        plan.b_source[slice] =
            b + (long long)k_row * params.ldb + (columns > 0 ? global_column : 0);
        plan.b_destination[slice] = (unsigned)__cvta_generic_to_shared(
            b_stage0 + tf32m64n96_b_index(k_row, column));
    }
}

__device__ __forceinline__ void tf32m64n96_stage_slice(
    const Tf32m64n96CopyPlan& plan, unsigned a_stage_bytes,
    unsigned b_stage_bytes, int k_base, int reduction, int issue) {
    if (issue < 2) {
        int remaining = reduction - k_base - plan.a_k_offset;
        remaining = remaining < 0 ? 0 : (remaining > 4 ? 4 : remaining);
        int a_bytes = plan.a_row_valid[issue] ? remaining * 4 : 0;
        gbf_tf32_copy_cg(
            plan.a_destination[issue] + a_stage_bytes,
            plan.a_source[issue], a_bytes);
    }
    if (issue < 3) {
        int b_bytes = k_base + plan.b_k_row[issue] < reduction
            ? plan.b_column_bytes[issue]
            : 0;
        gbf_tf32_copy_cg(
            plan.b_destination[issue] + b_stage_bytes,
            plan.b_source[issue], b_bytes);
    }
}

__device__ __forceinline__ void tf32m64n96_advance_plan(
    Tf32m64n96CopyPlan& plan, long long b_rows) {
#pragma unroll
    for (int slice = 0; slice < 2; ++slice) plan.a_source[slice] += 32;
#pragma unroll
    for (int slice = 0; slice < 3; ++slice) plan.b_source[slice] += b_rows;
}

__device__ __forceinline__ void tf32m64n96_stage_async(
    const Tf32m64n96CopyPlan& plan, unsigned a_stage_bytes,
    unsigned b_stage_bytes, int k_base, int reduction) {
#pragma unroll
    for (int issue = 0; issue < 4; ++issue) {
        tf32m64n96_stage_slice(
            plan, a_stage_bytes, b_stage_bytes, k_base, reduction, issue);
    }
    asm volatile("cp.async.commit_group;\n" ::);
}

struct Tf32m64n96Fragments {
    unsigned a[2][4];
    unsigned b[3][2];
};

struct Tf32m64n96FragmentOffsets {
    int a[2][4];
    int b[3][2];
};

__device__ __forceinline__ void tf32m64n96_fragment_offsets(
    int warp_m, int warp_n, int group, int thread, int lane,
    Tf32m64n96FragmentOffsets& offsets) {
    int a_row = warp_m + (lane & 15);
    int a_k = (lane >> 4) << 2;
#pragma unroll
    for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
        for (int step = 0; step < 4; ++step) {
            offsets.a[m_atom][step] =
                tf32m64n96_a_index(a_row + m_atom * 16, step * 8 + a_k);
        }
    }
#pragma unroll
    for (int n_atom = 0; n_atom < 3; ++n_atom) {
        int column = warp_n + n_atom * 8 + group;
        offsets.b[n_atom][0] = tf32m64n96_b_index(thread, column);
        offsets.b[n_atom][1] = tf32m64n96_b_index(thread + 4, column);
    }
}

__device__ __forceinline__ unsigned tf32m64n96_round(unsigned bits) {
    float value = __uint_as_float(bits);
    unsigned result;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(result) : "f"(value));
    return result;
}

__device__ __forceinline__ void tf32m64n96_load_fragments(
    const float* a_stage, const float* b_stage, int k8,
    const Tf32m64n96FragmentOffsets& offsets, Tf32m64n96Fragments& fragments) {
    int step = k8 >> 3;
    unsigned a_base = (unsigned)__cvta_generic_to_shared(a_stage);
#pragma unroll
    for (int m_atom = 0; m_atom < 2; ++m_atom) {
        unsigned address = a_base + (unsigned)offsets.a[m_atom][step] * 4U;
        unsigned raw0, raw1, raw2, raw3;
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(raw0), "=r"(raw1), "=r"(raw2), "=r"(raw3)
            : "r"(address));
        fragments.a[m_atom][0] = raw0;
        fragments.a[m_atom][1] = raw1;
        fragments.a[m_atom][2] = raw2;
        fragments.a[m_atom][3] = raw3;
    }
    const float* b_step = b_stage + k8 * 96;
#pragma unroll
    for (int n_atom = 0; n_atom < 3; ++n_atom) {
        fragments.b[n_atom][0] =
            tf32m64n96_round(__float_as_uint(b_step[offsets.b[n_atom][0]]));
        fragments.b[n_atom][1] =
            tf32m64n96_round(__float_as_uint(b_step[offsets.b[n_atom][1]]));
    }
}

__device__ __forceinline__ void tf32m64n96_mma(
    const Tf32m64n96Fragments& fragments, float (&acc)[2][3][4]) {
#pragma unroll
    for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 3; ++n_atom) {
            gbf_tf32_mma_m16n8k8(
                acc[m_atom][n_atom], fragments.a[m_atom], fragments.b[n_atom]);
        }
    }
}

__device__ __forceinline__ void tf32m64n96_store(
    float* output, int row, int column, float accumulator,
    const GbfTf32N96Params& params) {
    if (row >= params.m || column >= params.n) return;
    float* destination = output + (long long)row * params.ldc + column;
    *destination = __fmaf_rn(params.alpha, accumulator, *destination);
}

__device__ __forceinline__ void tf32m64n96_zero_reduction(
    float* output, const float* bias, const GbfTf32N96Params& params) {
    int column_tiles = (params.n + 95) / 96;
    int tile_row = (int)blockIdx.x / column_tiles * 64;
    int tile_column = (int)blockIdx.x % column_tiles * 96;
    for (int linear = (int)threadIdx.x; linear < 64 * 96;
         linear += (int)blockDim.x) {
        int row = tile_row + linear / 96;
        int column = tile_column + linear % 96;
        if (row < params.m && column < params.n) {
            float accumulator = bias == nullptr ? 0.0f : bias[column];
            tf32m64n96_store(output, row, column, accumulator, params);
        }
    }
}

__device__ __forceinline__ void tf32m64n96_kernel(
    float* output, const float* a, const float* b, const float* bias,
    GbfTf32N96Params params) {
    extern __shared__ __align__(16) unsigned char shared_bytes[];
    float* a_stages = reinterpret_cast<float*>(shared_bytes);
    float* b_stages = a_stages + 2 * 64 * 32;
    int column_tiles = (params.n + 95) / 96;
    int tile_row = (int)blockIdx.x / column_tiles * 64;
    int tile_column = (int)blockIdx.x % column_tiles * 96;
    int warp = (int)threadIdx.x >> 5;
    int lane = (int)threadIdx.x & 31;
    int warp_m = (warp >> 2) * 32;
    int warp_n = (warp & 3) * 24;
    int group = lane >> 2;
    int thread = lane & 3;
    float acc[2][3][4];
#pragma unroll
    for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 3; ++n_atom) {
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
    Tf32m64n96FragmentOffsets offsets;
    tf32m64n96_fragment_offsets(warp_m, warp_n, group, thread, lane, offsets);
    Tf32m64n96CopyPlan plan;
    tf32m64n96_copy_plan(
        a_stages, b_stages, a, b, params, tile_row, tile_column, plan);
    long long b_slab_rows = 32LL * params.ldb;
    unsigned tile_count = (static_cast<unsigned>(params.k) + 31U) / 32U;
    if (tile_count != 0) {
        tf32m64n96_stage_async(plan, 0, 0, 0, params.k);
        tf32m64n96_advance_plan(plan, b_slab_rows);
    } else {
        asm volatile("cp.async.commit_group;\n" ::);
    }
    int read_stage = 0;
    const int k8_offsets[4] = {0, 8, 16, 24};
    for (unsigned tile = 0; tile < tile_count; ++tile) {
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        unsigned next = tile + 1;
        bool has_next = next < tile_count;
        int write_stage = read_stage ^ 1;
        unsigned write_a_bytes = (unsigned)write_stage * 64U * 32U * 4U;
        unsigned write_b_bytes = (unsigned)write_stage * 32U * 96U * 4U;
        const float* a_read = a_stages + read_stage * 64 * 32;
        const float* b_read = b_stages + read_stage * 32 * 96;
        Tf32m64n96Fragments fragments[2];
        tf32m64n96_load_fragments(
            a_read, b_read, k8_offsets[0], offsets, fragments[0]);
#pragma unroll
        for (int issue = 0; issue < 4; ++issue) {
            if (has_next) {
                tf32m64n96_stage_slice(
                    plan, write_a_bytes, write_b_bytes,
                    (int)(next * 32U), params.k, issue);
            }
            if (issue < 3) {
                tf32m64n96_load_fragments(
                    a_read, b_read, k8_offsets[issue + 1], offsets,
                    fragments[(issue + 1) & 1]);
            }
            tf32m64n96_mma(fragments[issue & 1], acc);
        }
        asm volatile("cp.async.commit_group;\n" ::);
        if (has_next) tf32m64n96_advance_plan(plan, b_slab_rows);
        read_stage ^= 1;
    }
    __syncthreads();
    float* tile_output = reinterpret_cast<float*>(shared_bytes);
    bool vector_rows = tile_column + 96 <= params.n && (params.ldc & 3) == 0
        && gbf_aligned16(output);
    if (vector_rows) {
#pragma unroll
        for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
            for (int n_atom = 0; n_atom < 3; ++n_atom) {
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    int row = warp_m + m_atom * 16 + group + half * 8;
                    int column = warp_n + n_atom * 8 + 2 * thread;
                    *reinterpret_cast<float2*>(tile_output + row * 104 + column) =
                        make_float2(
                            acc[m_atom][n_atom][2 * half],
                            acc[m_atom][n_atom][2 * half + 1]);
                }
            }
        }
        __syncthreads();
#pragma unroll 4
        for (int linear = (int)threadIdx.x; linear < 64 * 24; linear += 256) {
            int row = linear / 24;
            int chunk = (linear % 24) * 4;
            int global_row = tile_row + row;
            if (global_row >= params.m) continue;
            float4 value = *reinterpret_cast<const float4*>(
                tile_output + row * 104 + chunk);
            float* destination =
                output + (long long)global_row * params.ldc + tile_column + chunk;
            float4 old = *reinterpret_cast<const float4*>(destination);
            value.x = __fmaf_rn(params.alpha, value.x, old.x);
            value.y = __fmaf_rn(params.alpha, value.y, old.y);
            value.z = __fmaf_rn(params.alpha, value.z, old.z);
            value.w = __fmaf_rn(params.alpha, value.w, old.w);
            *reinterpret_cast<float4*>(destination) = value;
        }
        return;
    }
#pragma unroll
    for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 3; ++n_atom) {
#pragma unroll
            for (int element = 0; element < 4; ++element) {
                int row = tile_row + warp_m + m_atom * 16 + group
                    + (element >= 2 ? 8 : 0);
                int column = tile_column + warp_n + n_atom * 8
                    + 2 * thread + (element & 1);
                tf32m64n96_store(
                    output, row, column, acc[m_atom][n_atom][element], params);
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(256, 2)
void gemm_bi_tn_sm89_tf32_pre_rna_m64n96_bk32_s2_v1(
    float* output, const float* a, const float* b, const float* bias,
    GbfTf32N96Params params) {
    if (params.k == 0) {
        tf32m64n96_zero_reduction(output, bias, params);
        return;
    }
    tf32m64n96_kernel(output, a, b, bias, params);
}



struct GbfTf32TnTransposeParams {
    int rows;
    int columns;
    int output_stride;
};

static_assert(sizeof(GbfTf32TnTransposeParams) == 12, "TN transpose parameter ABI");
static_assert(alignof(GbfTf32TnTransposeParams) == 4, "TN transpose parameter alignment");

}
// END RETAINED TN_M64N96_S2

// BEGIN RETAINED NT_A_LDMATRIX_N96
namespace sm89_tf32_joint_nt_a_ldmatrix_n96 {
// Ada NT d768-in M128xN96/BK32/S3 A-ldmatrix winner for CUDA 13.2.
struct GbfTf32NtN96Params { float alpha; float beta; int m; int k; int n; int lda; int ldb; int ldc; };
static_assert(sizeof(GbfTf32NtN96Params) == 32, "NT N96 parameter ABI");

template <typename T>
__device__ __forceinline__ const T* nt_n96_cp_async_source(
    const T* base, long long valid_offset, int valid_bytes) {
    return valid_bytes == 0 ? base : base + valid_offset;
}

__device__ __forceinline__ unsigned nt_n96_rna(float value) {
    unsigned result;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(result) : "f"(value));
    return result;
}

template <int BM>
__device__ __forceinline__ void nt_n96_cp_async_16_zfill(
    unsigned shared_dst, const void* global_src, int valid_bytes) {
    if constexpr (BM == 16) {
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                     :: "r"(shared_dst), "l"(global_src), "r"(valid_bytes));
    } else {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                     :: "r"(shared_dst), "l"(global_src), "r"(valid_bytes));
    }
}

__device__ __forceinline__ void nt_n96_cp_async_4x4_zfill(
    unsigned shared_dst, const float* global_src, int valid_bytes) {
#pragma unroll
    for (int element = 0; element < 4; ++element) {
        int bytes = valid_bytes >= (element + 1) * 4 ? 4 : 0;
        const float* source = bytes == 0 ? global_src : global_src + element;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
                     :: "r"(shared_dst + element * 4), "l"(source), "r"(bytes));
    }
}

template <bool Narrow, int BM>
__device__ __forceinline__ void nt_n96_cp_async_zfill(
    unsigned shared_dst, const float* global_src, int valid_bytes) {
    if constexpr (!Narrow) {
        if (valid_bytes == 16) {
            nt_n96_cp_async_16_zfill<BM>(shared_dst, global_src, valid_bytes);
        } else {
            nt_n96_cp_async_4x4_zfill(shared_dst, global_src, valid_bytes);
        }
    } else if (valid_bytes == 16 && gbf_aligned16(global_src)) {
        nt_n96_cp_async_16_zfill<BM>(shared_dst, global_src, valid_bytes);
    } else {
        nt_n96_cp_async_4x4_zfill(shared_dst, global_src, valid_bytes);
    }
}

// BEGIN MEASURED NT_A_LDMATRIX_N96 BODY
struct __align__(16) NtN96S3Storage {
    float a[3][128][32];
    float b[3][96][32];
};

static_assert(sizeof(NtN96S3Storage) == 86016, "NT N96 S3 storage");

// A stage plane keeps 32 reduction floats per row; the 16-byte chunk index
// is folded with the row so that eight consecutive rows never share a bank
// group, for the copies and for ldmatrix alike.
__device__ __forceinline__ int nt_n96_slot(int row, int reduction) {
    return row * 32 + (reduction ^ ((row & 7) << 2));
}

// The tensor core reads the upper 19 bits of a tf32 operand. Adding half an
// ulp of the kept mantissa before that truncation rounds every finite value
// to the nearest, the result cvt.rna gives, in one instruction.
__device__ __forceinline__ unsigned nt_n96_add_half(unsigned bits) {
    return bits + 0x1000U;
}

__device__ __forceinline__ void nt_n96_copy_cg(
    unsigned shared_dst, const void* global_src, int valid_bytes) {
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;\n"
                 :: "r"(shared_dst), "l"(global_src), "r"(valid_bytes));
}

// Both operands are reduction-contiguous, so every thread owns one 16-byte
// chunk of four A rows and three B rows for the whole kernel; the sources
// advance by one slab per K-tile and the destinations by one stage plane.
struct NtN96CopyPlan {
    const float* a_source[4];
    const float* b_source[3];
    unsigned a_destination[4];
    unsigned b_destination[3];
    bool a_row_valid[4];
    bool b_row_valid[3];
    int reduction_offset;
};

__device__ __forceinline__ void nt_n96_copy_plan(
    NtN96S3Storage* storage, const float* a, const float* b,
    const GbfTf32NtN96Params& params, int tile_row, int tile_column,
    NtN96CopyPlan& plan) {
    plan.reduction_offset = ((int)threadIdx.x & 7) * 4;
#pragma unroll
    for (int slice = 0; slice < 4; ++slice) {
        int row = ((int)threadIdx.x + slice * 256) >> 3;
        int global_row = tile_row + row;
        plan.a_row_valid[slice] = global_row < params.m;
        plan.a_source[slice] =
            a + (long long)(plan.a_row_valid[slice] ? global_row : 0) * params.lda
            + plan.reduction_offset;
        plan.a_destination[slice] = (unsigned)__cvta_generic_to_shared(
            &storage->a[0][0][0] + nt_n96_slot(row, plan.reduction_offset));
    }
#pragma unroll
    for (int slice = 0; slice < 3; ++slice) {
        int column = ((int)threadIdx.x + slice * 256) >> 3;
        int global_column = tile_column + column;
        plan.b_row_valid[slice] = global_column < params.k;
        plan.b_source[slice] =
            b + (long long)(plan.b_row_valid[slice] ? global_column : 0) * params.ldb
            + plan.reduction_offset;
        plan.b_destination[slice] = (unsigned)__cvta_generic_to_shared(
            &storage->b[0][0][0] + nt_n96_slot(column, plan.reduction_offset));
    }
}

__device__ __forceinline__ void nt_n96_stage_slice(
    const NtN96CopyPlan& plan, unsigned a_stage_bytes, unsigned b_stage_bytes,
    int reduction_base, int reduction, int issue) {
    int remaining = reduction - reduction_base - plan.reduction_offset;
    remaining = remaining < 0 ? 0 : (remaining > 4 ? 4 : remaining);
    int bytes = remaining * 4;
    nt_n96_copy_cg(
        plan.a_destination[issue] + a_stage_bytes, plan.a_source[issue],
        plan.a_row_valid[issue] ? bytes : 0);
    if (issue < 3) {
        nt_n96_copy_cg(
            plan.b_destination[issue] + b_stage_bytes, plan.b_source[issue],
            plan.b_row_valid[issue] ? bytes : 0);
    }
}

__device__ __forceinline__ void nt_n96_advance_plan(NtN96CopyPlan& plan) {
#pragma unroll
    for (int slice = 0; slice < 4; ++slice) plan.a_source[slice] += 32;
#pragma unroll
    for (int slice = 0; slice < 3; ++slice) plan.b_source[slice] += 32;
}

__device__ __forceinline__ void nt_n96_stage_async(
    const NtN96CopyPlan& plan, unsigned a_stage_bytes, unsigned b_stage_bytes,
    int reduction_base, int reduction) {
#pragma unroll
    for (int issue = 0; issue < 4; ++issue) {
        nt_n96_stage_slice(
            plan, a_stage_bytes, b_stage_bytes, reduction_base, reduction, issue);
    }
    asm volatile("cp.async.commit_group;\n" ::);
}

struct NtN96Fragments {
    unsigned a[4][4];
    unsigned b[3][2];
};

// This lane's ldmatrix row addresses in stage 0, per k8 step: the four A
// atoms of sixteen rows, the first two B atoms as one x4 and the third as
// an x2. The stage plane offset is added at load time.
struct NtN96FragmentAddresses {
    unsigned a[4][4];
    unsigned b01[4];
    unsigned b2[4];
};

__device__ __forceinline__ void nt_n96_fragment_addresses(
    NtN96S3Storage* storage, int warp_m, int warp_n, int lane,
    NtN96FragmentAddresses& addresses) {
    unsigned a_base = (unsigned)__cvta_generic_to_shared(&storage->a[0][0][0]);
    unsigned b_base = (unsigned)__cvta_generic_to_shared(&storage->b[0][0][0]);
    int a_row = warp_m + (lane & 15);
    int a_reduction = (lane >> 4) << 2;
    int b_reduction = ((lane >> 3) & 1) << 2;
    int b01_column = warp_n + (((lane >> 4) & 1) << 3) + (lane & 7);
    int b2_column = warp_n + 16 + (lane & 7);
#pragma unroll
    for (int step = 0; step < 4; ++step) {
#pragma unroll
        for (int m_atom = 0; m_atom < 4; ++m_atom) {
            addresses.a[m_atom][step] = a_base
                + (unsigned)nt_n96_slot(a_row + m_atom * 16, step * 8 + a_reduction) * 4U;
        }
        addresses.b01[step] =
            b_base + (unsigned)nt_n96_slot(b01_column, step * 8 + b_reduction) * 4U;
        addresses.b2[step] =
            b_base + (unsigned)nt_n96_slot(b2_column, step * 8 + b_reduction) * 4U;
    }
}

__device__ __forceinline__ void nt_n96_load_fragments(
    const NtN96FragmentAddresses& addresses, unsigned a_stage_bytes,
    unsigned b_stage_bytes, int step, NtN96Fragments& fragments) {
#pragma unroll
    for (int m_atom = 0; m_atom < 4; ++m_atom) {
        unsigned raw0, raw1, raw2, raw3;
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(raw0), "=r"(raw1), "=r"(raw2), "=r"(raw3)
            : "r"(addresses.a[m_atom][step] + a_stage_bytes));
        fragments.a[m_atom][0] = nt_n96_add_half(raw0);
        fragments.a[m_atom][1] = nt_n96_add_half(raw1);
        fragments.a[m_atom][2] = nt_n96_add_half(raw2);
        fragments.a[m_atom][3] = nt_n96_add_half(raw3);
    }
    {
        unsigned raw0, raw1, raw2, raw3;
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(raw0), "=r"(raw1), "=r"(raw2), "=r"(raw3)
            : "r"(addresses.b01[step] + b_stage_bytes));
        fragments.b[0][0] = nt_n96_add_half(raw0);
        fragments.b[0][1] = nt_n96_add_half(raw1);
        fragments.b[1][0] = nt_n96_add_half(raw2);
        fragments.b[1][1] = nt_n96_add_half(raw3);
    }
    {
        unsigned raw0, raw1;
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
            : "=r"(raw0), "=r"(raw1)
            : "r"(addresses.b2[step] + b_stage_bytes));
        fragments.b[2][0] = nt_n96_add_half(raw0);
        fragments.b[2][1] = nt_n96_add_half(raw1);
    }
}

__device__ __forceinline__ void nt_n96_mma(
    const NtN96Fragments& fragments, float (&acc)[4][3][4]) {
#pragma unroll
    for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 3; ++n_atom) {
            gbf_tf32_mma_m16n8k8(
                acc[m_atom][n_atom], fragments.a[m_atom], fragments.b[n_atom]);
        }
    }
}

__device__ __forceinline__ void nt_n96_zero(
    float* output, GbfTf32NtN96Params params, int tile_row, int tile_column) {
    for (int linear = (int)threadIdx.x; linear < 128 * 96; linear += 256) {
        int row = tile_row + linear / 96;
        int column = tile_column + linear % 96;
        if (row < params.m && column < params.k) {
            output[(long long)row * params.ldc + column] =
                params.alpha == 1.0f ? 0.0f : __fmul_rn(params.alpha, 0.0f);
        }
    }
}

__device__ __forceinline__ void nt_n96_s3_kernel(
    float* output, const float* a, const float* b, GbfTf32NtN96Params params) {
    int column_tiles = (params.k + 95) / 96;
    int tile_row = (int)blockIdx.x / column_tiles * 128;
    int tile_column = (int)blockIdx.x % column_tiles * 96;
    if (params.n == 0) {
        nt_n96_zero(output, params, tile_row, tile_column);
        return;
    }
    extern __shared__ __align__(16) unsigned char shared_bytes[];
    auto* storage = reinterpret_cast<NtN96S3Storage*>(shared_bytes);
    int warp = (int)threadIdx.x >> 5;
    int lane = (int)threadIdx.x & 31;
    int warp_m = (warp >> 2) * 64;
    int warp_n = (warp & 3) * 24;
    int group = lane >> 2;
    int thread = lane & 3;
    float acc[4][3][4] = {};
    NtN96FragmentAddresses addresses;
    nt_n96_fragment_addresses(storage, warp_m, warp_n, lane, addresses);
    NtN96CopyPlan plan;
    nt_n96_copy_plan(storage, a, b, params, tile_row, tile_column, plan);
    unsigned tile_count = (static_cast<unsigned>(params.n) + 31U) / 32U;
#pragma unroll
    for (unsigned tile = 0; tile < 2; ++tile) {
        if (tile < tile_count) {
            nt_n96_stage_async(
                plan, tile * 128U * 32U * 4U, tile * 96U * 32U * 4U,
                (int)(tile * 32U), params.n);
            nt_n96_advance_plan(plan);
        } else {
            asm volatile("cp.async.commit_group;\n" ::);
        }
    }
    int read_stage = 0;
    for (unsigned tile = 0; tile < tile_count; ++tile) {
        asm volatile("cp.async.wait_group 1;\n" ::);
        __syncthreads();
        unsigned next = tile + 2;
        bool has_next = next < tile_count;
        int write_stage = read_stage == 0 ? 2 : read_stage - 1;
        unsigned write_a_bytes = (unsigned)write_stage * 128U * 32U * 4U;
        unsigned write_b_bytes = (unsigned)write_stage * 96U * 32U * 4U;
        unsigned read_a_bytes = (unsigned)read_stage * 128U * 32U * 4U;
        unsigned read_b_bytes = (unsigned)read_stage * 96U * 32U * 4U;
        NtN96Fragments fragments[2];
        nt_n96_load_fragments(addresses, read_a_bytes, read_b_bytes, 0, fragments[0]);
#pragma unroll
        for (int issue = 0; issue < 4; ++issue) {
            if (has_next) {
                nt_n96_stage_slice(
                    plan, write_a_bytes, write_b_bytes, (int)(next * 32U), params.n, issue);
            }
            if (issue < 3) {
                nt_n96_load_fragments(
                    addresses, read_a_bytes, read_b_bytes, issue + 1,
                    fragments[(issue + 1) & 1]);
            }
            nt_n96_mma(fragments[issue & 1], acc);
        }
        asm volatile("cp.async.commit_group;\n" ::);
        if (has_next) nt_n96_advance_plan(plan);
        if (++read_stage == 3) read_stage = 0;
    }
    __syncthreads();
    float* tile_output = reinterpret_cast<float*>(shared_bytes);
    bool vector_rows = tile_column + 96 <= params.k && (params.ldc & 3) == 0
        && (reinterpret_cast<unsigned long long>(output) & 15ull) == 0ull;
    if (vector_rows) {
#pragma unroll
        for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
            for (int n_atom = 0; n_atom < 3; ++n_atom) {
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    int row = warp_m + m_atom * 16 + group + half * 8;
                    int column = warp_n + n_atom * 8 + 2 * thread;
                    *reinterpret_cast<float2*>(tile_output + row * 104 + column) =
                        make_float2(
                            acc[m_atom][n_atom][2 * half],
                            acc[m_atom][n_atom][2 * half + 1]);
                }
            }
        }
        __syncthreads();
#pragma unroll 4
        for (int linear = (int)threadIdx.x; linear < 128 * 24; linear += 256) {
            int row = linear / 24;
            int chunk = (linear % 24) * 4;
            int global_row = tile_row + row;
            if (global_row >= params.m) continue;
            float4 value = *reinterpret_cast<const float4*>(
                tile_output + row * 104 + chunk);
            if (params.alpha != 1.0f) {
                value.x = __fmul_rn(params.alpha, value.x);
                value.y = __fmul_rn(params.alpha, value.y);
                value.z = __fmul_rn(params.alpha, value.z);
                value.w = __fmul_rn(params.alpha, value.w);
            }
            *reinterpret_cast<float4*>(
                output + (long long)global_row * params.ldc + tile_column + chunk) = value;
        }
        return;
    }
#pragma unroll
    for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 3; ++n_atom) {
#pragma unroll
            for (int element = 0; element < 4; ++element) {
                int row = tile_row + warp_m + m_atom * 16 + group + (element >= 2 ? 8 : 0);
                int column = tile_column + warp_n + n_atom * 8 + 2 * thread + (element & 1);
                if (row < params.m && column < params.k) {
                    float value = params.alpha == 1.0f
                        ? acc[m_atom][n_atom][element]
                        : __fmul_rn(params.alpha, acc[m_atom][n_atom][element]);
                    output[(long long)row * params.ldc + column] = value;
                }
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(256, 1)
void gemm_bi_nt_sm89_tf32_a_ldmatrix_m128n96_bk32_s3_v1(
    float* output, const float* a, const float* b, const float* bias,
    GbfTf32NtN96Params params) {
    (void)bias;
    nt_n96_s3_kernel(output, a, b, params);
}
// END MEASURED NT_A_LDMATRIX_N96 BODY
}
// END RETAINED NT_A_LDMATRIX_N96
