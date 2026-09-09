#[path = "triad_tf32_tn_transpose_rna_m64n64_source.rs"]
mod parent;
#[path = "triad_tn_transpose_n96_source.rs"]
mod raw_parent;

pub const GEMM_SYMBOL: &str = "gemm_bi_tn_test_transpose_pre_rna_m64n128_sm89_bk32_s2_reuse";
pub const RETAINED_GEMM_SYMBOL: &str = parent::GEMM_SYMBOL;
pub const TRANSPOSE_SYMBOL: &str = parent::TRANSPOSE_SYMBOL;
pub const BLOCK_THREADS: u32 = 256;
pub const DYNAMIC_SHARED_BYTES: usize = 49_152;
pub const MAX_REGISTERS: i32 = 128;
pub const REQUIRED_OCCUPANCY: u32 = 2;
pub const K8_ISSUE_OFFSETS: [u32; 4] = [0, 8, 16, 24];
pub const TILE: (u32, u32, u32) = (64, 128, 32);
pub const STAGES: u32 = 2;
pub const ACCUMULATORS_PER_THREAD: u32 = 32;

const FIXED_N96_SOURCE: &str = include_str!("../../kernels/gemm_bi_fixed/tf32_rna_n96.cu");

pub fn retained_source(fixed_n96: &str) -> Result<String, String> {
    let raw = raw_parent::candidate_source(fixed_n96)?;
    parent::candidate_source(&raw)
}

pub fn candidate_source(fixed_n96: &str) -> Result<String, String> {
    let mut source = retained_source(fixed_n96)?;
    replace_exact(
        &mut source,
        &format!(
            "extern \"C\" __global__ __launch_bounds__(256, 1)\nvoid {}(",
            parent::GEMM_SYMBOL
        ),
        &format!("extern \"C\" __global__ __launch_bounds__(256, 2)\nvoid {GEMM_SYMBOL}(",),
        "GEMM export and occupancy launch bound",
    )?;
    replace_exact(
        &mut source,
        "    return k * 64 + chunk * 4 + (column & 3);",
        "    return k * 128 + chunk * 4 + (column & 3);",
        "B shared stride",
    )?;
    replace_range(
        &mut source,
        "struct Tf32m64n64CopyPlan {",
        "struct Tf32m64n64Fragments {",
        COPY_PIPELINE,
        "copy pipeline",
    )?;
    replace_range(
        &mut source,
        "struct Tf32m64n64Fragments {",
        "__device__ __forceinline__ void tf32m64n64_store(",
        FRAGMENT_PIPELINE,
        "fragment pipeline",
    )?;
    replace_range(
        &mut source,
        "__device__ __forceinline__ void tf32m64n64_zero_reduction(",
        "__device__ __forceinline__ void tf32m64n64_kernel(",
        ZERO_REDUCTION,
        "zero reduction",
    )?;
    let export_anchor =
        format!("extern \"C\" __global__ __launch_bounds__(256, 2)\nvoid {GEMM_SYMBOL}(");
    replace_range(
        &mut source,
        "__device__ __forceinline__ void tf32m64n64_kernel(",
        &export_anchor,
        M64N128_KERNEL,
        "M64N128 kernel",
    )?;
    replace_all_exact(&mut source, "Tf32m64n64", "Tf32m64n128", 14)?;
    replace_all_exact(&mut source, "tf32m64n64", "tf32m64n128", 35)?;
    for (needle, expected) in [
        ("float acc[2][4][4];", 1),
        ("int warp_m = (warp >> 2) * 32;", 1),
        ("int warp_n = (warp & 3) * 32;", 1),
        ("for (int issue = 0; issue < 4; ++issue)", 2),
        ("if (has_next)", 2),
        ("tf32m64n128_round(__float_as_uint(b_step", 2),
        ("fragments.a[m_atom][0] = raw0;", 1),
        (GEMM_SYMBOL, 1),
        (TRANSPOSE_SYMBOL, 1),
    ] {
        require_count(&source, needle, expected)?;
    }
    if source.contains(parent::GEMM_SYMBOL) {
        return Err("M64N128 candidate retained the M64N64 GEMM export".into());
    }
    Ok(source)
}

pub fn restore_retained_source(candidate: &str) -> Result<String, String> {
    let expected = candidate_source(FIXED_N96_SOURCE)?;
    if candidate != expected {
        return Err("M64N128 candidate does not match the frozen reversible transform".into());
    }
    retained_source(FIXED_N96_SOURCE)
}

const COPY_PIPELINE: &str = r#"static_assert(
    2 * (64 * 32 + 32 * 128) * 4 == 49152,
    "M64N128 BK32 S2 shared ABI");

struct Tf32m64n64CopyPlan {
    const float* a_source[2];
    const float* b_source[4];
    unsigned a_destination[2];
    unsigned b_destination[4];
    int a_k_offset;
    int b_k_row[4];
    bool a_row_valid[2];
    int b_column_bytes[4];
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
    for (int slice = 0; slice < 4; ++slice) {
        int linear = (int)threadIdx.x + slice * 256;
        int k_row = linear / 32;
        int column = (linear % 32) * 4;
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
    if (issue < 2) {
        int remaining = reduction - k_base - plan.a_k_offset;
        remaining = remaining < 0 ? 0 : (remaining > 4 ? 4 : remaining);
        int a_bytes = plan.a_row_valid[issue] ? remaining * 4 : 0;
        gbf_tf32_copy_cg(
            plan.a_destination[issue] + a_stage_bytes,
            plan.a_source[issue], a_bytes);
    }
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
    for (int slice = 0; slice < 4; ++slice) plan.b_source[slice] += b_rows;
}

__device__ __forceinline__ void tf32m64n64_stage_async(
    const Tf32m64n64CopyPlan& plan, unsigned a_stage_bytes,
    unsigned b_stage_bytes, int k_base, int reduction) {
#pragma unroll
    for (int issue = 0; issue < 4; ++issue) {
        tf32m64n64_stage_slice(
            plan, a_stage_bytes, b_stage_bytes, k_base, reduction, issue);
    }
    asm volatile("cp.async.commit_group;\n" ::);
}

"#;

const FRAGMENT_PIPELINE: &str = r#"struct Tf32m64n64Fragments {
    unsigned a[2][4];
    unsigned b[4][2];
};

struct Tf32m64n64FragmentOffsets {
    int a[2][4];
    int b[4][2];
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
    for (int n_atom = 0; n_atom < 4; ++n_atom) {
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
    const float* b_step = b_stage + step * 8 * 128;
#pragma unroll
    for (int n_atom = 0; n_atom < 4; ++n_atom) {
        fragments.b[n_atom][0] =
            tf32m64n64_round(__float_as_uint(b_step[offsets.b[n_atom][0]]));
        fragments.b[n_atom][1] =
            tf32m64n64_round(__float_as_uint(b_step[offsets.b[n_atom][1]]));
    }
}

__device__ __forceinline__ void tf32m64n64_mma(
    const Tf32m64n64Fragments& fragments, float (&acc)[2][4][4]) {
#pragma unroll
    for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 4; ++n_atom) {
            gbf_tf32_mma_m16n8k8(
                acc[m_atom][n_atom], fragments.a[m_atom], fragments.b[n_atom]);
        }
    }
}

"#;

const ZERO_REDUCTION: &str = r#"__device__ __forceinline__ void tf32m64n64_zero_reduction(
    float* output, const float* bias, const GbfTf32N96Params& params) {
    int column_tiles = (params.n + 127) / 128;
    int tile_row = (int)blockIdx.x / column_tiles * 64;
    int tile_column = (int)blockIdx.x % column_tiles * 128;
    for (int linear = (int)threadIdx.x; linear < 64 * 128;
         linear += (int)blockDim.x) {
        int row = tile_row + linear / 128;
        int column = tile_column + linear % 128;
        if (row < params.m && column < params.n) {
            float accumulator = bias == nullptr ? 0.0f : bias[column];
            tf32m64n64_store(output, row, column, accumulator, params);
        }
    }
}

"#;

const M64N128_KERNEL: &str = r#"__device__ __forceinline__ void tf32m64n64_kernel(
    float* output, const float* a, const float* b, const float* bias,
    GbfTf32N96Params params) {
    extern __shared__ __align__(16) unsigned char shared_bytes[];
    float* a_stages = reinterpret_cast<float*>(shared_bytes);
    float* b_stages = a_stages + 2 * 64 * 32;
    int column_tiles = (params.n + 127) / 128;
    int tile_row = (int)blockIdx.x / column_tiles * 64;
    int tile_column = (int)blockIdx.x % column_tiles * 128;
    int warp = (int)threadIdx.x >> 5;
    int lane = (int)threadIdx.x & 31;
    int warp_m = (warp >> 2) * 32;
    int warp_n = (warp & 3) * 32;
    int group = lane >> 2;
    int thread = lane & 3;
    float acc[2][4][4];
#pragma unroll
    for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 4; ++n_atom) {
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
    if (tile_count != 0) {
        tf32m64n64_stage_async(plan, 0, 0, 0, params.k);
        tf32m64n64_advance_plan(plan, b_slab_rows);
    }
    int read_stage = 0;
    for (unsigned tile = 0; tile < tile_count; ++tile) {
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        unsigned next = tile + 1;
        bool has_next = next < tile_count;
        int write_stage = read_stage ^ 1;
        unsigned write_a_bytes = (unsigned)write_stage * 64U * 32U * 4U;
        unsigned write_b_bytes = (unsigned)write_stage * 32U * 128U * 4U;
        const float* a_read = a_stages + read_stage * 64 * 32;
        const float* b_read = b_stages + read_stage * 32 * 128;
        Tf32m64n64Fragments fragments[2];
        tf32m64n64_load_fragments(a_read, b_read, 0, offsets, fragments[0]);
#pragma unroll
        for (int issue = 0; issue < 4; ++issue) {
            if (has_next) {
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
        read_stage ^= 1;
    }
    __syncthreads();
    float* tile_output = reinterpret_cast<float*>(shared_bytes);
    bool vector_rows = tile_column + 128 <= params.n && (params.ldc & 3) == 0
        && gbf_aligned16(output);
    if (vector_rows) {
#pragma unroll
        for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
            for (int n_atom = 0; n_atom < 4; ++n_atom) {
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    int row = warp_m + m_atom * 16 + group + half * 8;
                    int column = warp_n + n_atom * 8 + 2 * thread;
                    *reinterpret_cast<float2*>(tile_output + row * 136 + column) =
                        make_float2(
                            acc[m_atom][n_atom][2 * half],
                            acc[m_atom][n_atom][2 * half + 1]);
                }
            }
        }
        __syncthreads();
#pragma unroll 4
        for (int linear = (int)threadIdx.x; linear < 64 * 32; linear += 256) {
            int row = linear / 32;
            int chunk = (linear % 32) * 4;
            int global_row = tile_row + row;
            if (global_row >= params.m) continue;
            float4 value = *reinterpret_cast<const float4*>(
                tile_output + row * 136 + chunk);
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
        for (int n_atom = 0; n_atom < 4; ++n_atom) {
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

"#;

fn replace_exact(
    source: &mut String,
    before: &str,
    after: &str,
    label: &str,
) -> Result<(), String> {
    require_count(source, before, 1).map_err(|error| format!("{label}: {error}"))?;
    *source = source.replacen(before, after, 1);
    Ok(())
}

fn replace_all_exact(
    source: &mut String,
    before: &str,
    after: &str,
    expected: usize,
) -> Result<(), String> {
    require_count(source, before, expected)?;
    *source = source.replace(before, after);
    Ok(())
}

fn replace_range(
    source: &mut String,
    start: &str,
    end: &str,
    replacement: &str,
    label: &str,
) -> Result<(), String> {
    require_count(source, start, 1).map_err(|error| format!("{label}: {error}"))?;
    require_count(source, end, 1).map_err(|error| format!("{label}: {error}"))?;
    let begin = source.find(start).unwrap();
    let finish = source.find(end).unwrap();
    if begin >= finish {
        return Err(format!("M64N128 {label} boundaries are reversed"));
    }
    source.replace_range(begin..finish, replacement);
    Ok(())
}

fn require_count(source: &str, needle: &str, expected: usize) -> Result<(), String> {
    let actual = source.matches(needle).count();
    if actual == expected {
        Ok(())
    } else {
        Err(format!(
            "M64N128 source anchor {needle:?}: expected {expected}, observed {actual}"
        ))
    }
}
