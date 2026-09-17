use std::collections::BTreeSet;

const PRELUDE: &str = include_str!("../../../../kernels/_typed_prelude.cuh");
const CONTRACT: &str = include_str!("../../../../kernels/gemm_bi_triad/contract.cuh");
const COMMON: &str = include_str!("../../../../kernels/gemm_bi_triad/common.cuh");
const EPILOGUE: &str = include_str!("../../../../kernels/gemm_bi_triad/epilogue.cuh");
const MMA16: &str = include_str!("../../../../kernels/gemm_bi_triad/mma16.cuh");
pub const OWNER: &str = include_str!("../../../../kernels/gemm_bi_triad/sm89/half_tn.cu");

pub const OWNER_SHA256: &str = "e01bc5d622c9fc3a71841217a69031580b5f4d94d590d2d8c90931aec2741dcb";
pub const PRELUDE_SHA256: &str = "0c9b2345c643417406d75403df11f6fb96af7ce82f198ee551086f7c19020948";
pub const CONTRACT_SHA256: &str =
    "a8df19198d57a15d1fb84ea32f085c35ddd56eb407cb1bd172f1810eba12df6f";
pub const COMMON_SHA256: &str = "ae91ca724cb6ae8a14753b3c7260e18e64ea4027cd36b104322d00d34d287a88";
pub const EPILOGUE_SHA256: &str =
    "88f198be891f1316ee540f210e014b55d61e14263d71cd710c17762fb9d1674a";
pub const MMA16_SHA256: &str = "8906c2da4db43c1b51a1ebed5d3ab8c9da29c4c4f8a248131ead8df7149d54f1";

const OWNER_FNV64: u64 = 0xb4d9_9185_f5af_9250;
const PRELUDE_FNV64: u64 = 0x672e_6c67_8676_9881;
const CONTRACT_FNV64: u64 = 0xaab2_41d0_80b4_217d;
const COMMON_FNV64: u64 = 0xd069_42d6_c5b4_4868;
const EPILOGUE_FNV64: u64 = 0xa9bc_47a2_e161_5149;
const MMA16_FNV64: u64 = 0x5407_785d_d217_d044;

pub const COMPACT_SYMBOL_PREFIX: &str = "tn_sm89_m64n64_bk64_s2_compact_bxor_";
pub const REGPIPE_VEC2_SYMBOL_PREFIX: &str = "tn_sm89_m64n64_bk64_s2_regpipe_vec2_";
pub const SMALL16_SYMBOL_PREFIX: &str = "tn_sm89_m16n16_bk64_s2_ldb72_";
pub const COMPACT_BF16_SYMBOL: &str = "tn_sm89_m64n64_bk64_s2_compact_bxor_bf16";
pub const COMPACT_F16_SYMBOL: &str = "tn_sm89_m64n64_bk64_s2_compact_bxor_f16";
pub const REGPIPE_VEC2_BF16_SYMBOL: &str = "tn_sm89_m64n64_bk64_s2_regpipe_vec2_bf16";
pub const REGPIPE_VEC2_F16_SYMBOL: &str = "tn_sm89_m64n64_bk64_s2_regpipe_vec2_f16";
pub const SMALL16_BF16_SYMBOL: &str = "tn_sm89_m16n16_bk64_s2_ldb72_bf16";
pub const SMALL16_F16_SYMBOL: &str = "tn_sm89_m16n16_bk64_s2_ldb72_f16";

pub const HALF_TN_DRIVER_ABI: [(u32, u32); 7] =
    [(0, 8), (8, 8), (16, 8), (24, 4), (28, 4), (32, 4), (36, 4)];
pub const HALF_TN_TERMINAL_ARGUMENT: u32 = 7;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sm89HalfTnKernelKind {
    CompactBxor,
    RegpipeVec2,
    Small16Bk64S2Ldb72,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sm89HalfTnDtype {
    F16,
    Bf16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sm89HalfTnKernelSpec {
    pub symbol: &'static str,
    pub kind: Sm89HalfTnKernelKind,
    pub dtype: Sm89HalfTnDtype,
    pub tile: (u32, u32),
    pub bk: u32,
    pub stages: u8,
    pub block: (u32, u32, u32),
    pub dynamic_shared_bytes: u32,
    pub static_shared_bytes: u32,
    pub local_bytes: u32,
    pub register_cap: u32,
    pub occupancy_gate: u32,
    pub abi_parameter_count: u32,
    pub abi_parameter_bytes: u32,
}

const fn spec(
    symbol: &'static str,
    kind: Sm89HalfTnKernelKind,
    dtype: Sm89HalfTnDtype,
) -> Sm89HalfTnKernelSpec {
    Sm89HalfTnKernelSpec {
        symbol,
        kind,
        dtype,
        tile: (64, 64),
        bk: 64,
        stages: 2,
        block: (128, 1, 1),
        dynamic_shared_bytes: 0,
        static_shared_bytes: 32_768,
        local_bytes: 0,
        register_cap: 128,
        occupancy_gate: 3,
        abi_parameter_count: 7,
        abi_parameter_bytes: 40,
    }
}

const fn small16_spec(symbol: &'static str, dtype: Sm89HalfTnDtype) -> Sm89HalfTnKernelSpec {
    Sm89HalfTnKernelSpec {
        symbol,
        kind: Sm89HalfTnKernelKind::Small16Bk64S2Ldb72,
        dtype,
        tile: (16, 16),
        bk: 64,
        stages: 2,
        block: (32, 1, 1),
        dynamic_shared_bytes: 0,
        static_shared_bytes: 36_864,
        local_bytes: 0,
        register_cap: 128,
        occupancy_gate: 2,
        abi_parameter_count: 7,
        abi_parameter_bytes: 40,
    }
}

pub const SM89_HALF_TN_KERNEL_SPECS: [Sm89HalfTnKernelSpec; 6] = [
    spec(
        COMPACT_F16_SYMBOL,
        Sm89HalfTnKernelKind::CompactBxor,
        Sm89HalfTnDtype::F16,
    ),
    spec(
        COMPACT_BF16_SYMBOL,
        Sm89HalfTnKernelKind::CompactBxor,
        Sm89HalfTnDtype::Bf16,
    ),
    spec(
        REGPIPE_VEC2_F16_SYMBOL,
        Sm89HalfTnKernelKind::RegpipeVec2,
        Sm89HalfTnDtype::F16,
    ),
    spec(
        REGPIPE_VEC2_BF16_SYMBOL,
        Sm89HalfTnKernelKind::RegpipeVec2,
        Sm89HalfTnDtype::Bf16,
    ),
    small16_spec(SMALL16_F16_SYMBOL, Sm89HalfTnDtype::F16),
    small16_spec(SMALL16_BF16_SYMBOL, Sm89HalfTnDtype::Bf16),
];

pub fn kernel_spec(symbol: &str) -> Option<&'static Sm89HalfTnKernelSpec> {
    SM89_HALF_TN_KERNEL_SPECS
        .iter()
        .find(|spec| spec.symbol == symbol)
}

const COMPACT_BEGIN: &str = "// SM89_HALF_TN_COMPACT_BEGIN";
const COMPACT_END: &str = "// SM89_HALF_TN_COMPACT_END";
const REGPIPE_VEC2_BEGIN: &str = "// SM89_HALF_TN_REGPIPE_VEC2_BEGIN";
const REGPIPE_VEC2_END: &str = "// SM89_HALF_TN_REGPIPE_VEC2_END";
const SMALL16_BEGIN: &str = "// SM89_HALF_TN_SMALL16_BEGIN";
const SMALL16_END: &str = "// SM89_HALF_TN_SMALL16_END";

fn fnv1a64(source: &str) -> u64 {
    source
        .as_bytes()
        .iter()
        .fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
        })
}

fn require_frozen(label: &str, source: &str, expected: u64) -> Result<(), String> {
    let observed = fnv1a64(source);
    if observed == expected {
        Ok(())
    } else {
        Err(format!(
            "SM89 half-TN {label} source drift: expected {expected:#018x}, observed {observed:#018x}"
        ))
    }
}

fn section<'a>(source: &'a str, begin: &str, end: &str) -> Result<&'a str, String> {
    let begin_at = source
        .find(begin)
        .ok_or_else(|| format!("SM89 half-TN owner is missing {begin}"))?;
    let body_at = begin_at + begin.len();
    let end_at = source[body_at..]
        .find(end)
        .map(|offset| body_at + offset)
        .ok_or_else(|| format!("SM89 half-TN owner is missing {end}"))?;
    if source[body_at..end_at].contains(begin) || source[end_at + end.len()..].contains(end) {
        return Err("SM89 half-TN owner duplicated a family marker".into());
    }
    Ok(source[body_at..end_at].trim())
}

pub fn family_source<'a>(source: &'a str, symbol: &str) -> Option<&'a str> {
    let (begin, end) = match symbol {
        COMPACT_F16_SYMBOL | COMPACT_BF16_SYMBOL => (COMPACT_BEGIN, COMPACT_END),
        REGPIPE_VEC2_F16_SYMBOL | REGPIPE_VEC2_BF16_SYMBOL => {
            (REGPIPE_VEC2_BEGIN, REGPIPE_VEC2_END)
        }
        SMALL16_F16_SYMBOL | SMALL16_BF16_SYMBOL => (SMALL16_BEGIN, SMALL16_END),
        _ => return None,
    };
    section(source, begin, end).ok()
}

fn validate_dependency_sources() -> Result<(), String> {
    for (label, source, expected) in [
        ("typed prelude", PRELUDE, PRELUDE_FNV64),
        ("contract", CONTRACT, CONTRACT_FNV64),
        ("common", COMMON, COMMON_FNV64),
        ("epilogue", EPILOGUE, EPILOGUE_FNV64),
        ("mma16", MMA16, MMA16_FNV64),
        ("owner", OWNER, OWNER_FNV64),
    ] {
        require_frozen(label, source, expected)?;
    }
    for (label, source) in [
        ("typed prelude", PRELUDE),
        ("contract", CONTRACT),
        ("common", COMMON),
        ("epilogue", EPILOGUE),
        ("mma16", MMA16),
    ] {
        if source.contains("extern \"C\" __global__") {
            return Err(format!(
                "SM89 half-TN dependency {label} unexpectedly exports a kernel"
            ));
        }
    }
    Ok(())
}

fn join(parts: &[&str]) -> String {
    parts
        .iter()
        .map(|part| part.trim_end())
        .collect::<Vec<_>>()
        .join("\n")
        + "\n"
}

pub fn compose_source() -> Result<String, String> {
    validate_dependency_sources()?;
    let source = join(&[PRELUDE, CONTRACT, COMMON, EPILOGUE, MMA16, OWNER]);
    validate_source_text(&source)?;
    Ok(source)
}

pub fn compose_fragment_for_sm89_half() -> Result<String, String> {
    validate_dependency_sources()?;
    let source = join(&[CONTRACT, COMMON, EPILOGUE, MMA16, OWNER]);
    validate_source_text(&source)?;
    Ok(source)
}

pub fn export_inventory(source: &str) -> Result<Vec<&'static str>, String> {
    let compact = section(source, COMPACT_BEGIN, COMPACT_END)?;
    let vec2 = section(source, REGPIPE_VEC2_BEGIN, REGPIPE_VEC2_END)?;
    for (family, prefix) in [
        (compact, COMPACT_SYMBOL_PREFIX),
        (vec2, REGPIPE_VEC2_SYMBOL_PREFIX),
    ] {
        if family.matches(&format!("void {prefix}##SUFFIX(")).count() != 1
            || family
                .matches("GEMM_BI_DEFINE_GEMM_BI_TN_TC64(bf16,")
                .count()
                != 1
            || family
                .matches("GEMM_BI_DEFINE_GEMM_BI_TN_TC64(f16,")
                .count()
                != 1
        {
            return Err(format!(
                "SM89 half-TN family does not represent exactly two dtype exports: {prefix}"
            ));
        }
    }
    let small16 = section(source, SMALL16_BEGIN, SMALL16_END)?;
    for invocation in ["bf16", "f16"]
        .map(|suffix| format!("DEFINE_GEMM_BI_SMALL16_TN({SMALL16_SYMBOL_PREFIX}{suffix},"))
    {
        if small16.matches(invocation.as_str()).count() != 1 {
            return Err(format!(
                "SM89 half-TN small16 family does not represent exact export {invocation}"
            ));
        }
    }
    if small16.matches("DEFINE_GEMM_BI_SMALL16_TN(").count() != 3
        || small16.contains("small32")
        || small16.contains("probe_")
        || small16.contains("PROBE_")
    {
        return Err("SM89 half-TN small16 family retained a discovery namespace or export".into());
    }
    if source.matches("extern \"C\" __global__").count() != 3 {
        return Err("SM89 half-TN source contains a foreign or missing export declaration".into());
    }
    Ok(vec![
        COMPACT_BF16_SYMBOL,
        COMPACT_F16_SYMBOL,
        REGPIPE_VEC2_BF16_SYMBOL,
        REGPIPE_VEC2_F16_SYMBOL,
        SMALL16_BF16_SYMBOL,
        SMALL16_F16_SYMBOL,
    ])
}

pub fn validate_source_text(source: &str) -> Result<(), String> {
    for marker in ["_test_", "_exp_", "tn_tc64_", "__SM89_HALF_TN_", "small32"] {
        if source.contains(marker) {
            return Err(format!(
                "SM89 half-TN source retained forbidden marker {marker}"
            ));
        }
    }
    let expected = SM89_HALF_TN_KERNEL_SPECS
        .iter()
        .map(|spec| spec.symbol)
        .collect::<BTreeSet<_>>();
    let exports = export_inventory(source)?;
    let observed = exports.iter().copied().collect::<BTreeSet<_>>();
    if exports.len() != expected.len() || observed != expected {
        return Err(format!(
            "SM89 half-TN export inventory changed: expected {expected:?}, observed {exports:?}"
        ));
    }
    Ok(())
}

pub fn validate_source() -> Result<(), String> {
    for (label, digest) in [
        ("owner", OWNER_SHA256),
        ("typed prelude", PRELUDE_SHA256),
        ("contract", CONTRACT_SHA256),
        ("common", COMMON_SHA256),
        ("epilogue", EPILOGUE_SHA256),
        ("mma16", MMA16_SHA256),
    ] {
        if digest.len() != 64
            || !digest
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            return Err(format!(
                "SM89 half-TN {label} SHA-256 freeze is not lowercase hex"
            ));
        }
    }
    validate_dependency_sources()?;
    validate_source_text(&compose_source()?)
}
