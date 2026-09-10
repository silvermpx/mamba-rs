#[path = "triad_tf32_nn_n96_direct_epilogue_source.rs"]
mod direct_parent;
#[path = "triad_nn_n96_source.rs"]
mod triad_parent;

pub const SYMBOL: &str =
    "gemm_bi_nn_triad_sm89_add_half_tf32_direct_epilogue_exp_m128n96_bk32_s3_grid_constant";
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

const FIXED_N96_SOURCE: &str = include_str!("../../kernels/gemm_bi_inference/tf32_rna_n96.cu");

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
        INTERNAL_BY_VALUE_SIGNATURE,
        INTERNAL_CONST_REF_SIGNATURE,
        "internal Params threading",
    )?;
    replace_exact(
        &mut source,
        &format!(
            "extern \"C\" __global__ __launch_bounds__(256, 1)\nvoid {}(\n    float* output, const float* a, const float* b, const float* bias,\n    GbfTf32N96Params params) {{",
            direct_parent::SYMBOL
        ),
        &format!(
            "extern \"C\" __global__ __launch_bounds__(256, 1)\nvoid {SYMBOL}(\n    float* output, const float* a, const float* b, const float* bias,\n    const __grid_constant__ GbfTf32N96Params params) {{"
        ),
        "export and grid-constant Params",
    )?;

    for (needle, expected) in [
        ("static_assert(sizeof(GbfTf32N96Params) == 32", 1),
        ("static_assert(alignof(GbfTf32N96Params) == 4", 1),
        ("const __grid_constant__ GbfTf32N96Params params", 1),
        ("const GbfTf32N96Params& params", 4),
        ("return bits + 0x1000U;", 1),
        ("for (int issue = 0; issue < 4; ++issue)", 2),
        ("gbf_tf32_mma_m16n8k8(", 1),
        ("*reinterpret_cast<float2*>(destination) = make_float2(", 1),
        (SYMBOL, 1),
    ] {
        require_count(&source, needle, expected)?;
    }
    if source.contains(&format!("void {}(", RETAINED_SYMBOL)) {
        return Err("grid-constant candidate retained the ancestor export".into());
    }
    Ok(source)
}

pub fn restore_retained_source(candidate: &str) -> Result<String, String> {
    let expected = candidate_source(FIXED_N96_SOURCE)?;
    if candidate != expected {
        return Err(
            "grid-constant candidate does not match the frozen reversible transform".into(),
        );
    }
    retained_source(FIXED_N96_SOURCE)
}

const INTERNAL_BY_VALUE_SIGNATURE: &str = r#"__device__ __forceinline__ void tf32n96_kernel(
    float* output, const float* a, const float* b, const float* bias,
    GbfTf32N96Params params) {"#;

const INTERNAL_CONST_REF_SIGNATURE: &str = r#"__device__ __forceinline__ void tf32n96_kernel(
    float* output, const float* a, const float* b, const float* bias,
    const GbfTf32N96Params& params) {"#;

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
            "grid-constant source anchor {needle:?}: expected {expected}, observed {actual}"
        ))
    }
}
