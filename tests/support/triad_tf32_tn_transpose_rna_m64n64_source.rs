// The n96 transpose adapter this candidate derives from; the joint source
// contract reads it through here so the module has one instance.
#[path = "triad_tf32_tn_transpose_rna_n96_source.rs"]
pub mod parent;

pub const GEMM_SYMBOL: &str = "tn_test_transpose_pre_rna_m64n64_sm89_bk32_s3_wave";
pub const TRANSPOSE_SYMBOL: &str = parent::CANDIDATE_TRANSPOSE_SYMBOL;

pub fn retained_parent_source(raw_tn_n96: &str) -> Result<String, String> {
    parent::compose_candidate_source(raw_tn_n96)
}

pub fn candidate_source(raw_tn_n96: &str) -> Result<String, String> {
    let mut source = retained_parent_source(raw_tn_n96)?;
    replace_exact(
        &mut source,
        parent::CANDIDATE_GEMM_SYMBOL,
        GEMM_SYMBOL,
        "GEMM export",
    )?;
    replace_exact(
        &mut source,
        "    return k * 96 + chunk * 4 + (column & 3);",
        "    return k * 64 + chunk * 4 + (column & 3);",
        "B shared stride",
    )?;
    replace_range(
        &mut source,
        "struct Tf32n96CopyPlan {",
        "struct Tf32n96Fragments {",
        COPY_PIPELINE,
        "copy pipeline",
    )?;
    replace_range(
        &mut source,
        "struct Tf32n96Fragments {",
        "__device__ __forceinline__ void tf32n96_store(",
        FRAGMENT_PIPELINE,
        "fragment pipeline",
    )?;
    replace_range(
        &mut source,
        "__device__ __forceinline__ void tf32n96_zero_reduction(",
        "__device__ __forceinline__ void tf32n96_kernel(",
        ZERO_REDUCTION,
        "zero-reduction geometry",
    )?;
    let export_anchor =
        format!("extern \"C\" __global__ __launch_bounds__(256, 1)\nvoid {GEMM_SYMBOL}(");
    replace_range(
        &mut source,
        "__device__ __forceinline__ void tf32n96_kernel(",
        &export_anchor,
        M64N64_KERNEL,
        "M64N64 kernel",
    )?;
    replace_all_exact(&mut source, "Tf32n96", "Tf32m64n64", 14)?;
    replace_all_exact(&mut source, "tf32n96", "tf32m64n64", 35)?;
    for (needle, expected) in [
        ("float acc[2][2][4];", 1),
        ("int warp_m = (warp >> 2) * 32;", 1),
        ("int warp_n = (warp & 3) * 16;", 1),
        ("for (int issue = 0; issue < 4; ++issue)", 1),
        ("if (has_next && issue < 2)", 1),
        ("tf32m64n64_round(__float_as_uint(b_step", 2),
        ("fragments.a[m_atom][0] = raw0;", 1),
        (GEMM_SYMBOL, 1),
        (TRANSPOSE_SYMBOL, 1),
    ] {
        require_count(&source, needle, expected)?;
    }
    if source.contains(parent::CANDIDATE_GEMM_SYMBOL) {
        return Err("M64N64 candidate retained the N96 GEMM export".into());
    }
    Ok(source)
}

const COPY_PIPELINE: &str = r#"struct Tf32n96CopyPlan {
    const float* a_source[2];
    const float* b_source[2];
    unsigned a_destination[2];
    unsigned b_destination[2];
    int a_k_offset;
    int b_k_row[2];
    bool a_row_valid[2];
    int b_column_bytes[2];
};

__device__ __forceinline__ void tf32n96_copy_plan(
    float* a_stage0, float* b_stage0, const float* a, const float* b,
    const GbfTf32N96Params& params, int tile_row, int tile_column,
    Tf32n96CopyPlan& plan) {
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
            a_stage0 + tf32n96_a_index(row, plan.a_k_offset));
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
            b_stage0 + tf32n96_b_index(k_row, column));
    }
}

__device__ __forceinline__ void tf32n96_stage_slice(
    const Tf32n96CopyPlan& plan, unsigned a_stage_bytes,
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

__device__ __forceinline__ void tf32n96_advance_plan(
    Tf32n96CopyPlan& plan, long long b_rows) {
#pragma unroll
    for (int slice = 0; slice < 2; ++slice) plan.a_source[slice] += 32;
#pragma unroll
    for (int slice = 0; slice < 2; ++slice) plan.b_source[slice] += b_rows;
}

__device__ __forceinline__ void tf32n96_stage_async(
    const Tf32n96CopyPlan& plan, unsigned a_stage_bytes,
    unsigned b_stage_bytes, int k_base, int reduction) {
#pragma unroll
    for (int issue = 0; issue < 2; ++issue) {
        tf32n96_stage_slice(
            plan, a_stage_bytes, b_stage_bytes, k_base, reduction, issue);
    }
    asm volatile("cp.async.commit_group;\n" ::);
}

"#;

const FRAGMENT_PIPELINE: &str = r#"struct Tf32n96Fragments {
    unsigned a[2][4];
    unsigned b[2][2];
};

struct Tf32n96FragmentOffsets {
    int a[2][4];
    int b[2][2];
};

__device__ __forceinline__ void tf32n96_fragment_offsets(
    int warp_m, int warp_n, int group, int thread, int lane,
    Tf32n96FragmentOffsets& offsets) {
    int a_row = warp_m + (lane & 15);
    int a_k = (lane >> 4) << 2;
#pragma unroll
    for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
        for (int step = 0; step < 4; ++step) {
            offsets.a[m_atom][step] =
                tf32n96_a_index(a_row + m_atom * 16, step * 8 + a_k);
        }
    }
#pragma unroll
    for (int n_atom = 0; n_atom < 2; ++n_atom) {
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
            tf32n96_round(__float_as_uint(b_step[offsets.b[n_atom][0]]));
        fragments.b[n_atom][1] =
            tf32n96_round(__float_as_uint(b_step[offsets.b[n_atom][1]]));
    }
}

__device__ __forceinline__ void tf32n96_mma(
    const Tf32n96Fragments& fragments, float (&acc)[2][2][4]) {
#pragma unroll
    for (int m_atom = 0; m_atom < 2; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 2; ++n_atom) {
            gbf_tf32_mma_m16n8k8(
                acc[m_atom][n_atom], fragments.a[m_atom], fragments.b[n_atom]);
        }
    }
}

"#;

const ZERO_REDUCTION: &str = r#"__device__ __forceinline__ void tf32n96_zero_reduction(
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
            tf32n96_store(output, row, column, accumulator, params);
        }
    }
}

"#;

const M64N64_KERNEL: &str = r#"__device__ __forceinline__ void tf32n96_kernel(
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
                plan, tile * 64U * 32U * 4U, tile * 32U * 64U * 4U,
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
        unsigned write_a_bytes = (unsigned)write_stage * 64U * 32U * 4U;
        unsigned write_b_bytes = (unsigned)write_stage * 32U * 64U * 4U;
        const float* a_read = a_stages + read_stage * 64 * 32;
        const float* b_read = b_stages + read_stage * 32 * 64;
        Tf32n96Fragments fragments[2];
        tf32n96_load_fragments(a_read, b_read, 0, offsets, fragments[0]);
#pragma unroll
        for (int issue = 0; issue < 4; ++issue) {
            if (has_next && issue < 2) {
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
                tf32n96_store(
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
        return Err(format!("M64N64 {label} boundaries are reversed"));
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
            "M64N64 source anchor {needle:?}: expected {expected}, observed {actual}"
        ))
    }
}
