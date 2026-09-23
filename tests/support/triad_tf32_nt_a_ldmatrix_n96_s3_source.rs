pub const SYMBOL: &str = "nt_test_a_ldmatrix_n96_s3_sm89";

const SM80_SOURCE: &str = include_str!("../../kernels/gemm_bi_triad/sm80/mma.cu");

pub fn candidate_source() -> String {
    compose(SYMBOL, A_LDMATRIX_LOAD)
}

fn compose(symbol: &str, a_load: &str) -> String {
    format!("{SM80_SOURCE}\n{N96_BODY}")
        .replace("NT_N96_SYMBOL", symbol)
        .replace("NT_N96_A_LOAD", a_load)
}

const A_LDMATRIX_LOAD: &str = r#"#pragma unroll
        for (int m_atom = 0; m_atom < 4; ++m_atom) {
            int row = warp_m + m_atom * 16 + (lane & 15);
            int reduction = k8 + ((lane >> 4) << 2);
            unsigned address = (unsigned)__cvta_generic_to_shared(
                &nt_n96_a_slot(storage, stage, row, reduction));
            unsigned raw0, raw1, raw2, raw3;
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(raw0), "=r"(raw1), "=r"(raw2), "=r"(raw3)
                : "r"(address));
            a_fragments[m_atom][0] = tf32_rna(__uint_as_float(raw0));
            a_fragments[m_atom][1] = tf32_rna(__uint_as_float(raw1));
            a_fragments[m_atom][2] = tf32_rna(__uint_as_float(raw2));
            a_fragments[m_atom][3] = tf32_rna(__uint_as_float(raw3));
        }"#;

const N96_BODY: &str = r#"
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

// The tensor core reads the upper 19 bits of a tf32 operand. Half an ulp of
// the kept mantissa, added in floating point from the operand's own exponent
// before that truncation, gives every normal value the cvt.rna result, keeps
// a NaN a NaN and needs no predicate.
__device__ __forceinline__ unsigned nt_n96_add_half(unsigned bits) {
    return __float_as_uint(fmaf(__uint_as_float(bits & 0xff800000U), 1.0f / 2048.0f, __uint_as_float(bits)));
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
    const Sm80Tf32KernelParams& params, int tile_row, int tile_column,
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

// A whole 16-byte chunk, for a stage whose K slab lies inside the
// reduction: no length to clamp, so the copy is the address alone. A row
// past the matrix reads the clamped row the plan keeps, and the outputs it
// feeds are never stored.
__device__ __forceinline__ void nt_n96_copy_whole(unsigned shared_dst, const void* global_src) {
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                 :: "r"(shared_dst), "l"(global_src));
}

__device__ __forceinline__ void nt_n96_stage_slice_whole(
    const NtN96CopyPlan& plan, unsigned a_stage_bytes, unsigned b_stage_bytes, int issue) {
    nt_n96_copy_whole(plan.a_destination[issue] + a_stage_bytes, plan.a_source[issue]);
    if (issue < 3) {
        nt_n96_copy_whole(plan.b_destination[issue] + b_stage_bytes, plan.b_source[issue]);
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
            tf32_mma_m16n8k8(
                acc[m_atom][n_atom], fragments.a[m_atom], fragments.b[n_atom]);
        }
    }
}

// The ring stages one tile of the main loop reads, fills and reads next,
// and the K slab of the stage it fills.
struct NtN96TileStages {
    unsigned read_a;
    unsigned read_b;
    unsigned write_a;
    unsigned write_b;
    unsigned next_a;
    unsigned next_b;
    int fill_k;
    bool fills;
    bool has_following;
};

// One quarter of the next stage's copies; `Whole` is the copy form of a
// stage whose slab lies inside the reduction.
template <bool Whole>
__device__ __forceinline__ void nt_n96_fill_slice(
    const NtN96CopyPlan& plan, const NtN96TileStages& stages, int reduction, int issue) {
    if (Whole) {
        nt_n96_stage_slice_whole(plan, stages.write_a, stages.write_b, issue);
    } else if (stages.fills) {
        nt_n96_stage_slice(
            plan, stages.write_a, stages.write_b, stages.fill_k, reduction, issue);
    }
}

// One tile of the main loop. The stage wait and barrier sit in step 2,
// after its mma, and step 3 loads the next tile's step-0 fragments, so a
// tile starts on its mma rather than on the barrier. Every read of the
// tile's stage is issued before that barrier, and the copies into the slot
// read two tiles back start only after the previous tile's barrier.
template <bool Whole>
__device__ __forceinline__ void nt_n96_tile(
    NtN96CopyPlan& plan, const NtN96FragmentAddresses& addresses,
    const NtN96TileStages& stages, int reduction, NtN96Fragments (&fragments)[2],
    float (&acc)[4][3][4]) {
#pragma unroll
    for (int step = 0; step < 3; ++step) {
        nt_n96_fill_slice<Whole>(plan, stages, reduction, step);
        if (step == 2) {
            nt_n96_fill_slice<Whole>(plan, stages, reduction, 3);
            asm volatile("cp.async.commit_group;\n" ::);
        }
        nt_n96_load_fragments(
            addresses, stages.read_a, stages.read_b, step + 1, fragments[(step + 1) & 1]);
        nt_n96_mma(fragments[step & 1], acc);
    }
    if (stages.fills) nt_n96_advance_plan(plan);
    asm volatile("cp.async.wait_group 1;\n" ::);
    __syncthreads();
    if (stages.has_following) {
        nt_n96_load_fragments(addresses, stages.next_a, stages.next_b, 0, fragments[0]);
    }
    nt_n96_mma(fragments[1], acc);
}

__device__ __forceinline__ void nt_n96_zero(
    float* output, Sm80Tf32KernelParams params, int tile_row, int tile_column) {
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
    float* output, const float* a, const float* b, Sm80Tf32KernelParams params) {
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
    asm volatile("cp.async.wait_group 1;\n" ::);
    __syncthreads();
    NtN96Fragments fragments[2];
    nt_n96_load_fragments(addresses, 0U, 0U, 0, fragments[0]);
    int read_stage = 0;
    for (unsigned tile = 0; tile < tile_count; ++tile) {
        unsigned fill = tile + 2;
        int write_stage = read_stage == 0 ? 2 : read_stage - 1;
        int next_stage = read_stage == 2 ? 0 : read_stage + 1;
        NtN96TileStages stages;
        stages.read_a = (unsigned)read_stage * 128U * 32U * 4U;
        stages.read_b = (unsigned)read_stage * 96U * 32U * 4U;
        stages.write_a = (unsigned)write_stage * 128U * 32U * 4U;
        stages.write_b = (unsigned)write_stage * 96U * 32U * 4U;
        stages.next_a = (unsigned)next_stage * 128U * 32U * 4U;
        stages.next_b = (unsigned)next_stage * 96U * 32U * 4U;
        stages.fill_k = (int)(fill * 32U);
        stages.fills = fill < tile_count;
        stages.has_following = tile + 1 < tile_count;
        if (stages.fills && stages.fill_k + 32 <= params.n) {
            nt_n96_tile<true>(plan, addresses, stages, params.n, fragments, acc);
        } else {
            nt_n96_tile<false>(plan, addresses, stages, params.n, fragments, acc);
        }
        read_stage = next_stage;
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
void NT_N96_SYMBOL(
    float* output, const float* a, const float* b, const float* bias,
    Sm80Tf32KernelParams params) {
    (void)bias;
    nt_n96_s3_kernel(output, a, b, params);
}
"#;
