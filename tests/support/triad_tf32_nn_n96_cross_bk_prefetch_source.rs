#[path = "triad_tf32_nn_n96_direct_epilogue_source.rs"]
mod direct_parent;
#[path = "triad_nn_n96_source.rs"]
mod triad_parent;

pub const SYMBOL: &str =
    "gemm_bi_nn_triad_sm89_add_half_tf32_direct_epilogue_exp_m128n96_bk32_s3_cross_bk_prefetch";
pub const RETAINED_SYMBOL: &str = direct_parent::SYMBOL;
pub const TILE: (u32, u32, u32) = (128, 96, 32);
pub const BLOCK_THREADS: u32 = 256;
pub const STAGES: u32 = 3;
pub const DYNAMIC_SHARED_BYTES: usize = 86_016;
pub const MAX_REGISTERS: i32 = 124;
pub const REQUIRED_OCCUPANCY: u32 = 1;
pub const PARAM_BYTES: usize = 32;
pub const PARAM_ALIGNMENT: usize = 4;
pub const K8_ISSUE_OFFSETS: [u32; 4] = [0, 8, 16, 24];

pub const fn cross_bk_prefetch_boundaries(tile_count: usize) -> usize {
    tile_count.saturating_sub(2)
}

const FIXED_N96_SOURCE: &str = include_str!("../../kernels/gemm_bi_fixed/tf32_rna_n96.cu");

pub const fn grid(m: usize, n: usize) -> usize {
    direct_parent::n96_grid(m, n)
}

pub const fn direct_pair_coordinates(
    warp: usize,
    m_atom: usize,
    n_atom: usize,
    half: usize,
    lane: usize,
) -> ((usize, usize), (usize, usize)) {
    direct_parent::direct_pair_coordinates(warp, m_atom, n_atom, half, lane)
}

pub fn retained_source(fixed_n96: &str) -> Result<String, String> {
    let triad = triad_parent::compose_triad_nn_n96_source(fixed_n96)?;
    direct_parent::compose_candidate_source(&triad)
}

pub fn candidate_source(fixed_n96: &str) -> Result<String, String> {
    let mut source = retained_source(fixed_n96)?;
    replace_exact(
        &mut source,
        RETAINED_MAINLOOP,
        CROSS_BK_PREFETCH_MAINLOOP,
        "cross-BK fragment0 prefetch schedule",
    )?;
    replace_exact(&mut source, RETAINED_SYMBOL, SYMBOL, "export symbol")?;

    for (needle, expected) in [
        ("return bits + 0x1000U;", 1),
        ("float acc[4][3][4];", 1),
        ("Tf32n96Fragments fragments[2];", 1),
        ("for (int issue = 0; issue < 3; ++issue)", 1),
        ("(int)(next * 32U), params.k, 3);", 1),
        ("bool cross_bk_prefetch = has_next;", 1),
        ("tf32n96_mma(fragments[1], acc);", 1),
        ("cp.async.wait_group 1;", 3),
        ("gbf_tf32_mma_m16n8k8(", 1),
        ("*reinterpret_cast<float2*>(destination) = make_float2(", 1),
        (SYMBOL, 1),
    ] {
        require_count(&source, needle, expected)?;
    }
    if source.contains(&format!("void {}(", RETAINED_SYMBOL)) {
        return Err("cross-BK candidate retained the ancestor export".into());
    }
    Ok(source)
}

pub fn restore_retained_source(candidate: &str) -> Result<String, String> {
    let expected = candidate_source(FIXED_N96_SOURCE)?;
    if candidate != expected {
        return Err("cross-BK candidate does not match the frozen reversible transform".into());
    }
    retained_source(FIXED_N96_SOURCE)
}

const RETAINED_MAINLOOP: &str = r#"    Tf32n96CopyPlan plan;
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
    }"#;

const CROSS_BK_PREFETCH_MAINLOOP: &str = r#"    Tf32n96CopyPlan plan;
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
    Tf32n96Fragments fragments[2];
    asm volatile("cp.async.wait_group 1;\n" ::);
    __syncthreads();
    tf32n96_load_fragments(
        a_stages, b_stages, 0, offsets, fragments[0]);
    for (unsigned tile = 0; tile < tile_count; ++tile) {
        unsigned next = tile + 2;
        bool has_next = next < tile_count;
        int write_stage = read_stage == 0 ? 2 : read_stage - 1;
        unsigned write_a_bytes = (unsigned)write_stage * 128U * 32U * 4U;
        unsigned write_b_bytes = (unsigned)write_stage * 32U * 96U * 4U;
        const float* a_read = a_stages + read_stage * 128 * 32;
        const float* b_read = b_stages + read_stage * 32 * 96;
#pragma unroll
        for (int issue = 0; issue < 3; ++issue) {
            if (has_next) {
                tf32n96_stage_slice(
                    plan, write_a_bytes, write_b_bytes,
                    (int)(next * 32U), params.k, issue);
            }
            tf32n96_load_fragments(
                a_read, b_read, issue + 1, offsets, fragments[(issue + 1) & 1]);
            tf32n96_mma(fragments[issue & 1], acc);
        }
        if (has_next) {
            tf32n96_stage_slice(
                plan, write_a_bytes, write_b_bytes,
                (int)(next * 32U), params.k, 3);
        }
        bool cross_bk_prefetch = has_next;
        if (cross_bk_prefetch) {
            asm volatile("cp.async.commit_group;\n" ::);
            asm volatile("cp.async.wait_group 1;\n" ::);
            __syncthreads();
            int following_stage = read_stage + 1;
            if (following_stage == 3) following_stage = 0;
            tf32n96_load_fragments(
                a_stages + following_stage * 128 * 32,
                b_stages + following_stage * 32 * 96,
                0, offsets, fragments[0]);
        }
        tf32n96_mma(fragments[1], acc);
        if (!cross_bk_prefetch) {
            asm volatile("cp.async.commit_group;\n" ::);
        }
        if (has_next) tf32n96_advance_plan(plan, b_slab_rows);
        if (++read_stage == 3) read_stage = 0;
        bool final_drain_boundary = !has_next && tile + 1 < tile_count;
        if (final_drain_boundary) {
            asm volatile("cp.async.wait_group 1;\n" ::);
            __syncthreads();
            tf32n96_load_fragments(
                a_stages + read_stage * 128 * 32,
                b_stages + read_stage * 32 * 96,
                0, offsets, fragments[0]);
        }
    }"#;

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

fn require_count(source: &str, needle: &str, expected: usize) -> Result<(), String> {
    let actual = source.matches(needle).count();
    if actual == expected {
        Ok(())
    } else {
        Err(format!(
            "cross-BK source anchor {needle:?}: expected {expected}, observed {actual}"
        ))
    }
}
