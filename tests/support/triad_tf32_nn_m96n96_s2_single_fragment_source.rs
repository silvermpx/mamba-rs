#[path = "triad_tf32_nn_m96n96_s2_direct_source.rs"]
mod parent;

pub const SYMBOL: &str =
    "gemm_bi_nn_triad_sm89_add_half_tf32_direct_epilogue_exp_m96n96_bk32_s2_single_fragment";
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
pub const FRAGMENT_WORDS_PER_BANK: u32 = 20;
pub const REMOVED_OVERLAPPING_FRAGMENT_WORDS: u32 = 20;

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
    replace_exact(
        &mut source,
        DOUBLE_BUFFERED_FRAGMENT_ISSUE,
        SINGLE_BUFFERED_FRAGMENT_ISSUE,
        "fragment issue lifetime",
    )?;

    for (needle, expected) in [
        ("return bits + 0x1000U;", 1),
        ("float acc[2][6][4];", 1),
        ("struct Tf32n96CopyPlan {", 1),
        ("for (int issue = 0; issue < 4; ++issue)", 2),
        ("gbf_tf32_mma_m16n8k8(", 1),
        ("Tf32n96Fragments fragments;", 1),
        ("*reinterpret_cast<float2*>(destination) = make_float2(", 1),
        (SYMBOL, 1),
    ] {
        require_count(&source, needle, expected)?;
    }
    for stopped in [
        "Tf32n96Fragments fragments[2];",
        "fragments[(issue + 1) & 1]",
        "fragments[issue & 1]",
    ] {
        if source.contains(stopped) {
            return Err(format!(
                "single-fragment candidate retained stopped state {stopped:?}"
            ));
        }
    }
    if source.contains(&format!("void {}(", RETAINED_M96_SYMBOL))
        || source.contains(&format!("void {}(", RETAINED_DIRECT_SYMBOL))
    {
        return Err("single-fragment candidate retained an ancestor export".into());
    }
    Ok(source)
}

pub fn restore_retained_m96_source(candidate: &str) -> Result<String, String> {
    let expected = candidate_source(FIXED_N96_SOURCE)?;
    if candidate != expected {
        return Err(
            "single-fragment candidate does not match the frozen reversible transform".into(),
        );
    }
    retained_m96_source(FIXED_N96_SOURCE)
}

const DOUBLE_BUFFERED_FRAGMENT_ISSUE: &str = r#"        Tf32n96Fragments fragments[2];
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
"#;

const SINGLE_BUFFERED_FRAGMENT_ISSUE: &str = r#"#pragma unroll
        for (int issue = 0; issue < 4; ++issue) {
            if (has_next) {
                tf32n96_stage_slice(
                    plan, write_a_bytes, write_b_bytes,
                    (int)(next * 32U), params.k, issue);
            }
            Tf32n96Fragments fragments;
            tf32n96_load_fragments(a_read, b_read, issue, offsets, fragments);
            tf32n96_mma(fragments, acc);
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

fn require_count(source: &str, needle: &str, expected: usize) -> Result<(), String> {
    let actual = source.matches(needle).count();
    if actual == expected {
        Ok(())
    } else {
        Err(format!(
            "single-fragment source anchor {needle:?}: expected {expected}, observed {actual}"
        ))
    }
}
