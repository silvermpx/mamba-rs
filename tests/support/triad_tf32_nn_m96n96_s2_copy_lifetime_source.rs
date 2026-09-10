#[path = "triad_tf32_nn_m96n96_s2_direct_source.rs"]
mod parent;

pub const SYMBOL: &str =
    "gemm_bi_nn_triad_sm89_add_half_tf32_direct_epilogue_exp_m96n96_bk32_s2_copy_lifetime";
pub const RETAINED_M96_SYMBOL: &str = parent::SYMBOL;
pub const RETAINED_DIRECT_SYMBOL: &str = parent::RETAINED_SYMBOL;
pub const TILE: (u32, u32, u32) = parent::TILE;
pub const BLOCK_THREADS: u32 = parent::BLOCK_THREADS;
pub const STAGES: u32 = parent::STAGES;
pub const DYNAMIC_SHARED_BYTES: usize = parent::DYNAMIC_SHARED_BYTES;
pub const MAX_REGISTERS: i32 = parent::MAX_REGISTERS;
pub const REQUIRED_OCCUPANCY: u32 = parent::REQUIRED_OCCUPANCY;
pub const ACCUMULATORS_PER_THREAD: u32 = parent::ACCUMULATORS_PER_THREAD;
pub const K8_ISSUE_OFFSETS: [u32; 4] = parent::K8_ISSUE_OFFSETS;
pub const REMOVED_PERSISTENT_PLAN_SLOTS: u32 = 37;

const FIXED_N96_SOURCE: &str = include_str!("../../kernels/gemm_bi_inference/tf32_rna_n96.cu");

pub const fn grid(m: usize, n: usize) -> usize {
    parent::grid(m, n)
}

pub const fn direct_pair_coordinates(
    warp: usize,
    m_atom: usize,
    n_atom: usize,
    half: usize,
    lane: usize,
) -> ((usize, usize), (usize, usize)) {
    parent::direct_pair_coordinates(warp, m_atom, n_atom, half, lane)
}

pub fn retained_m96_source(fixed_n96: &str) -> Result<String, String> {
    parent::candidate_source(fixed_n96)
}

pub fn retained_direct_source(fixed_n96: &str) -> Result<String, String> {
    parent::retained_source(fixed_n96)
}

pub fn candidate_source(fixed_n96: &str) -> Result<String, String> {
    let mut source = retained_m96_source(fixed_n96)?;
    replace_exact(
        &mut source,
        &format!(
            "extern \"C\" __global__ __launch_bounds__(192, 2)\nvoid {}(",
            parent::SYMBOL
        ),
        &format!("extern \"C\" __global__ __launch_bounds__(192, 2)\nvoid {SYMBOL}(",),
        "export",
    )?;
    replace_range(
        &mut source,
        "static_assert(\n    2 * (96 * 32 + 32 * 96) * 4 == 49152,",
        "struct Tf32n96Fragments {",
        COPY_LIFETIME_PIPELINE,
        "copy lifetime pipeline",
    )?;
    replace_range(
        &mut source,
        "    Tf32n96CopyPlan plan;",
        "    bool direct_pair_epilogue =",
        ON_DEMAND_MAINLOOP,
        "on-demand mainloop",
    )?;

    for (needle, expected) in [
        ("return bits + 0x1000U;", 1),
        ("float acc[2][6][4];", 1),
        ("for (int issue = 0; issue < 4; ++issue)", 2),
        ("gbf_tf32_mma_m16n8k8(", 1),
        ("*reinterpret_cast<float2*>(destination) = make_float2(", 1),
        ("tf32n96_stage_slice_ondemand(", 3),
        (SYMBOL, 1),
    ] {
        require_count(&source, needle, expected)?;
    }
    for stopped in [
        "Tf32n96CopyPlan",
        "tf32n96_copy_plan(",
        "tf32n96_advance_plan(",
    ] {
        if source.contains(stopped) {
            return Err(format!(
                "copy-lifetime candidate retained stopped state {stopped:?}"
            ));
        }
    }
    if source.contains(&format!("void {}(", RETAINED_M96_SYMBOL))
        || source.contains(&format!("void {}(", RETAINED_DIRECT_SYMBOL))
    {
        return Err("copy-lifetime candidate retained an ancestor export".into());
    }
    Ok(source)
}

pub fn restore_retained_m96_source(candidate: &str) -> Result<String, String> {
    let expected = candidate_source(FIXED_N96_SOURCE)?;
    if candidate != expected {
        return Err(
            "copy-lifetime candidate does not match the frozen reversible transform".into(),
        );
    }
    retained_m96_source(FIXED_N96_SOURCE)
}

const COPY_LIFETIME_PIPELINE: &str = r#"static_assert(
    2 * (96 * 32 + 32 * 96) * 4 == 49152,
    "M96N96 BK32 S2 shared ABI");

__device__ __forceinline__ void tf32n96_stage_slice_ondemand(
    float* a_stage, float* b_stage, const float* a, const float* b,
    const GbfTf32N96Params& params, int tile_row, int tile_column,
    int k_base, int issue) {
    int linear = (int)threadIdx.x + issue * 192;
    {
        int a_k = ((int)threadIdx.x & 7) * 4;
        int a_row = linear >> 3;
        int global_row = tile_row + a_row;
        int remaining = params.k - k_base - a_k;
        remaining = remaining < 0 ? 0 : (remaining > 4 ? 4 : remaining);
        int bytes = global_row < params.m ? remaining * 4 : 0;
        const float* source =
            a + (long long)(global_row < params.m ? global_row : 0) * params.lda
            + k_base + a_k;
        unsigned destination = (unsigned)__cvta_generic_to_shared(
            a_stage + tf32n96_a_index(a_row, a_k));
        gbf_tf32_copy_cg(destination, source, bytes);
    }
    {
        int b_k = linear / 24;
        int column = (linear % 24) * 4;
        int global_column = tile_column + column;
        int columns = params.n - global_column;
        columns = columns < 0 ? 0 : (columns > 4 ? 4 : columns);
        bool row_valid = k_base + b_k < params.k;
        const float* source =
            b + (long long)(row_valid ? k_base + b_k : 0) * params.ldb
            + (columns > 0 ? global_column : 0);
        unsigned destination = (unsigned)__cvta_generic_to_shared(
            b_stage + tf32n96_b_index(b_k, column));
        gbf_tf32_copy_cg(destination, source, row_valid ? columns * 4 : 0);
    }
}

__device__ __forceinline__ void tf32n96_stage_async_ondemand(
    float* a_stage, float* b_stage, const float* a, const float* b,
    const GbfTf32N96Params& params, int tile_row, int tile_column, int k_base) {
#pragma unroll
    for (int issue = 0; issue < 4; ++issue) {
        tf32n96_stage_slice_ondemand(
            a_stage, b_stage, a, b, params,
            tile_row, tile_column, k_base, issue);
    }
    asm volatile("cp.async.commit_group;\n" ::);
}

"#;

const ON_DEMAND_MAINLOOP: &str = r#"    unsigned tile_count = (static_cast<unsigned>(params.k) + 31U) / 32U;
    if (tile_count != 0) {
        tf32n96_stage_async_ondemand(
            a_stages, b_stages, a, b, params, tile_row, tile_column, 0);
    }
    int read_stage = 0;
    for (unsigned tile = 0; tile < tile_count; ++tile) {
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        unsigned next = tile + 1;
        bool has_next = next < tile_count;
        int write_stage = read_stage ^ 1;
        float* a_write = a_stages + write_stage * 96 * 32;
        float* b_write = b_stages + write_stage * 32 * 96;
        const float* a_read = a_stages + read_stage * 96 * 32;
        const float* b_read = b_stages + read_stage * 32 * 96;
        Tf32n96Fragments fragments[2];
        tf32n96_load_fragments(a_read, b_read, 0, offsets, fragments[0]);
#pragma unroll
        for (int issue = 0; issue < 4; ++issue) {
            if (has_next) {
                tf32n96_stage_slice_ondemand(
                    a_write, b_write, a, b, params,
                    tile_row, tile_column, (int)(next * 32U), issue);
            }
            if (issue < 3) {
                tf32n96_load_fragments(
                    a_read, b_read, issue + 1, offsets, fragments[(issue + 1) & 1]);
            }
            tf32n96_mma(fragments[issue & 1], acc);
        }
        asm volatile("cp.async.commit_group;\n" ::);
        read_stage ^= 1;
    }
    __syncthreads();
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
        return Err(format!("copy-lifetime {label} boundaries are reversed"));
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
            "copy-lifetime source anchor {needle:?}: expected {expected}, observed {actual}"
        ))
    }
}
