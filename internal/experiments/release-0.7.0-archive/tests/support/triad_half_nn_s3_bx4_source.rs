pub const SYMBOL: &str = "gemm_bi_nn_test_fixed_sm89_tc128_s3_bx4_f16";
pub const RETAINED_SYMBOL: &str = "gemm_bi_nn_fixed_sm89_tc128_s3_v1_f16";
pub const TARGET: (usize, usize, usize) = (2_048, 768, 3_072);
pub const TARGET_GRID: u32 = 96;
pub const BLOCK_THREADS: u32 = 256;
pub const DYNAMIC_SHARED_BYTES: usize = 98_304;
pub const REQUIRED_OCCUPANCY: u32 = 1;
pub const MAX_REGISTERS: i32 = 192;
const EXPECTED_RETAINED_FNV64: u64 = 0x723b_b1f9_e4ef_26e7;

const CONSUMER_ANCHOR: &str = r#"template <typename T>
static __device__ __forceinline__ void consume_fragments("#;

const BX4_LOADER: &str = r#"static __device__ __forceinline__ void load_fragments_bx4(
    unsigned a_stage, unsigned b_stage, int issue,
    const FragmentOffsets& offsets, Fragments& fragments) {
#pragma unroll
    for (int atom = 0; atom < 4; ++atom) {
        unsigned address = a_stage + layout::a_fragment_issue(offsets.a[atom], issue);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(fragments.a[atom][0]), "=r"(fragments.a[atom][1]),
              "=r"(fragments.a[atom][2]), "=r"(fragments.a[atom][3]) : "r"(address));
    }
    int lane = (int)threadIdx.x & 31;
    int lm_q = lane >> 3;
    int lm_r = lane & 7;
    int warp_n = (((int)threadIdx.x >> 5) & 3) * 32;
    {
        unsigned address = b_stage + 2U * (unsigned)layout::b_index(
            issue * 16 + ((lm_q & 1) ? 8 : 0) + lm_r,
            warp_n + ((lm_q & 2) ? 8 : 0));
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(fragments.b[0][0]), "=r"(fragments.b[0][1]),
              "=r"(fragments.b[1][0]), "=r"(fragments.b[1][1]) : "r"(address));
    }
    {
        unsigned address = b_stage + 2U * (unsigned)layout::b_index(
            issue * 16 + ((lm_q & 1) ? 8 : 0) + lm_r,
            warp_n + 16 + ((lm_q & 2) ? 8 : 0));
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(fragments.b[2][0]), "=r"(fragments.b[2][1]),
              "=r"(fragments.b[3][0]), "=r"(fragments.b[3][1]) : "r"(address));
    }
}

"#;

pub const fn bx4_coordinate(pair: usize, lane: usize) -> (usize, usize, usize) {
    let q = lane >> 3;
    (
        (lane & 7) + if q & 1 != 0 { 8 } else { 0 },
        pair * 16 + if q & 2 != 0 { 8 } else { 0 },
        q,
    )
}

pub fn retained_source(swizzle: &str, s3: &str, layout: &str) -> Result<String, String> {
    let include = "#include \"sm89_half_swizzle_layout.cuh\"";
    if swizzle.matches(include).count() != 1 {
        return Err("Fixed S3 swizzle layout include seam changed".into());
    }
    let source = format!("{}\n{s3}", swizzle.replacen(include, layout, 1));
    require_retained_hash(&source)?;
    Ok(source)
}

pub fn candidate_source(swizzle: &str, s3: &str, layout: &str) -> Result<String, String> {
    let mut source = retained_source(swizzle, s3, layout)?;
    replace_exact(
        &mut source,
        CONSUMER_ANCHOR,
        &format!("{BX4_LOADER}{CONSUMER_ANCHOR}"),
        1,
        "loader insertion",
    )?;
    replace_exact(
        &mut source,
        "sm89_fixed_half_swizzle::load_fragments(",
        "sm89_fixed_half_swizzle::load_fragments_bx4(",
        7,
        "S3 callsites",
    )?;
    replace_exact(
        &mut source,
        &format!("void {RETAINED_SYMBOL}("),
        &format!("void {SYMBOL}("),
        1,
        "F16 export",
    )?;
    Ok(source)
}

pub fn restore_retained_source(candidate: &str) -> Result<String, String> {
    let mut source = candidate.to_owned();
    replace_exact(
        &mut source,
        "sm89_fixed_half_swizzle::load_fragments_bx4(",
        "sm89_fixed_half_swizzle::load_fragments(",
        7,
        "restored S3 callsites",
    )?;
    replace_exact(
        &mut source,
        &format!("void {SYMBOL}("),
        &format!("void {RETAINED_SYMBOL}("),
        1,
        "restored F16 export",
    )?;
    replace_exact(
        &mut source,
        &format!("{BX4_LOADER}{CONSUMER_ANCHOR}"),
        CONSUMER_ANCHOR,
        1,
        "restored loader insertion",
    )?;
    require_retained_hash(&source)?;
    Ok(source)
}

pub fn all_strata_below(strata: &[[f64; 2]], threshold: f64) -> bool {
    strata.len() == 4
        && threshold.is_finite()
        && threshold > 0.0
        && strata.iter().all(|[p50, p95]| {
            p50.is_finite()
                && p95.is_finite()
                && *p50 > 0.0
                && *p95 > 0.0
                && *p50 < threshold
                && *p95 < threshold
        })
}

fn replace_exact(
    source: &mut String,
    before: &str,
    after: &str,
    expected: usize,
    label: &str,
) -> Result<(), String> {
    let actual = source.matches(before).count();
    if actual != expected {
        return Err(format!(
            "Fixed S3 B-x4 {label} seam changed: expected {expected}, observed {actual}"
        ));
    }
    *source = source.replace(before, after);
    Ok(())
}

fn require_retained_hash(source: &str) -> Result<(), String> {
    let observed = fnv64(source.as_bytes());
    if observed != EXPECTED_RETAINED_FNV64 {
        return Err(format!(
            "Fixed S3 retained source changed: expected {EXPECTED_RETAINED_FNV64:#018x}, observed {observed:#018x}"
        ));
    }
    Ok(())
}

fn fnv64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}
