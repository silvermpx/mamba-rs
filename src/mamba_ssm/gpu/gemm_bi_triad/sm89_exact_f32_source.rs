use std::collections::BTreeSet;

const PRELUDE: &str = include_str!("../../../../kernels/_typed_prelude.cuh");
pub const OWNER_TEMPLATE: &str =
    include_str!("../../../../kernels/gemm_bi_triad/sm80/exact_f32.cu");

pub const OWNER_SHA256: &str = "cdcb768216699f41553e73492a32d92717c62889a4a329ca1990360b361541c7";
pub const OWNER_SHA256_BYTES: [u8; 32] = [
    0xcd, 0xcb, 0x76, 0x82, 0x16, 0x69, 0x9f, 0x41, 0x55, 0x3e, 0x73, 0x49, 0x2a, 0x32, 0xd9, 0x27,
    0x17, 0xc6, 0x28, 0x89, 0xa4, 0xa3, 0x29, 0xca, 0x19, 0x90, 0x36, 0x0b, 0x36, 0x15, 0x41, 0xc7,
];

pub const D768_IN_FUSED_SYMBOL: &str = "tn_sm89_f32_n64_dual_chunk_fused_finalize";
pub const D768_OUT_RAW_SYMBOL: &str = "tn_sm89_f32_m64n64_bk16_s2_d768_out_raw";
pub const PRISM_RAW_SYMBOL: &str = "tn_sm89_f32_m64n64_bk16_s2_prism_raw";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sm89ExactF32TnRoute {
    D768InDualChunkFused,
    D768OutDirectBk16,
    PrismDirectBk16,
}

impl Sm89ExactF32TnRoute {
    pub const fn symbol(self) -> &'static str {
        match self {
            Self::D768InDualChunkFused => D768_IN_FUSED_SYMBOL,
            Self::D768OutDirectBk16 => D768_OUT_RAW_SYMBOL,
            Self::PrismDirectBk16 => PRISM_RAW_SYMBOL,
        }
    }
}

const DUAL_BEGIN: &str = "// SM89_EXACT_F32_DUAL_BEGIN";
const DUAL_END: &str = "// SM89_EXACT_F32_DUAL_END";
const EPILOGUE_BEGIN: &str = "// SM89_EXACT_F32_FUSED_EPILOGUE_BEGIN";
const EPILOGUE_END: &str = "// SM89_EXACT_F32_FUSED_EPILOGUE_END";
const DIRECT_BEGIN: &str = "// SM89_EXACT_F32_DIRECT_BEGIN";
const DIRECT_END: &str = "// SM89_EXACT_F32_DIRECT_END";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sm89ExactF32KernelKind {
    DirectSplitMRaw,
    DualChunkFusedFinalize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct Sm89ExactF32DualChunkParams {
    pub alpha: f32,
    pub m: i32,
    pub n: i32,
    pub k0: i32,
    pub k1: i32,
    pub lda: i32,
    pub ldb: i32,
    pub ldc: i32,
}

unsafe impl cudarc::driver::DeviceRepr for Sm89ExactF32DualChunkParams {}

const _: () = assert!(std::mem::size_of::<Sm89ExactF32DualChunkParams>() == 32);
const _: () = assert!(std::mem::align_of::<Sm89ExactF32DualChunkParams>() == 4);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sm89ExactF32KernelSpec {
    pub symbol: &'static str,
    pub kind: Sm89ExactF32KernelKind,
    pub shape: (usize, usize, usize),
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub dynamic_shared_bytes: u32,
    pub static_shared_bytes: u32,
    pub register_cap: u32,
    pub occupancy_gate: u32,
    pub chunks: u32,
    pub m_chunk: u32,
    pub abi_parameter_count: u32,
    pub abi_parameter_bytes: u32,
}

pub const SM89_EXACT_F32_KERNEL_SPECS: [Sm89ExactF32KernelSpec; 3] = [
    Sm89ExactF32KernelSpec {
        symbol: D768_IN_FUSED_SYMBOL,
        kind: Sm89ExactF32KernelKind::DualChunkFusedFinalize,
        shape: (2_048, 768, 3_072),
        grid: (576, 1, 1),
        block: (128, 1, 1),
        dynamic_shared_bytes: 0,
        static_shared_bytes: 32_768,
        register_cap: 168,
        occupancy_gate: 3,
        chunks: 2,
        m_chunk: 1_024,
        abi_parameter_count: 4,
        abi_parameter_bytes: 56,
    },
    Sm89ExactF32KernelSpec {
        symbol: D768_OUT_RAW_SYMBOL,
        kind: Sm89ExactF32KernelKind::DirectSplitMRaw,
        shape: (2_048, 1_536, 768),
        grid: (288, 1, 4),
        block: (128, 1, 1),
        dynamic_shared_bytes: 0,
        static_shared_bytes: 16_384,
        register_cap: 128,
        occupancy_gate: 4,
        chunks: 4,
        m_chunk: 512,
        abi_parameter_count: 7,
        abi_parameter_bytes: 40,
    },
    Sm89ExactF32KernelSpec {
        symbol: PRISM_RAW_SYMBOL,
        kind: Sm89ExactF32KernelKind::DirectSplitMRaw,
        shape: (4_621, 384, 1_928),
        grid: (186, 1, 6),
        block: (128, 1, 1),
        dynamic_shared_bytes: 0,
        static_shared_bytes: 16_384,
        register_cap: 128,
        occupancy_gate: 4,
        chunks: 6,
        m_chunk: 784,
        abi_parameter_count: 7,
        abi_parameter_bytes: 40,
    },
];

pub fn kernel_spec(symbol: &str) -> Option<&'static Sm89ExactF32KernelSpec> {
    SM89_EXACT_F32_KERNEL_SPECS
        .iter()
        .find(|spec| spec.symbol == symbol)
}

fn section<'a>(source: &'a str, begin: &str, end: &str) -> Result<&'a str, String> {
    let begin_at = source
        .find(begin)
        .ok_or_else(|| format!("SM89 exact-F32 owner is missing {begin}"))?;
    let body_at = begin_at + begin.len();
    let end_at = source[body_at..]
        .find(end)
        .map(|offset| body_at + offset)
        .ok_or_else(|| format!("SM89 exact-F32 owner is missing {end}"))?;
    Ok(source[body_at..end_at].trim())
}

pub fn compose_source() -> Result<String, String> {
    let dual = section(OWNER_TEMPLATE, DUAL_BEGIN, DUAL_END)?;
    let epilogue = section(OWNER_TEMPLATE, EPILOGUE_BEGIN, EPILOGUE_END)?;
    let direct = section(OWNER_TEMPLATE, DIRECT_BEGIN, DIRECT_END)?;
    let fused = dual
        .replace("__DUAL_SYMBOL__", D768_IN_FUSED_SYMBOL)
        .replace("__DUAL_EPILOGUE__", epilogue);
    let d768_out = direct.replace("__SM89_EXACT_F32_DIRECT_SYMBOL__", D768_OUT_RAW_SYMBOL);
    let prism = direct.replace("__SM89_EXACT_F32_DIRECT_SYMBOL__", PRISM_RAW_SYMBOL);
    Ok(format!("{PRELUDE}\n{fused}\n{d768_out}\n{prism}\n"))
}

pub fn export_inventory(source: &str) -> Result<Vec<&str>, String> {
    let mut exports = Vec::new();
    let mut remaining = source;
    while let Some(extern_at) = remaining.find("extern \"C\"") {
        let after_extern = &remaining[extern_at + "extern \"C\"".len()..];
        let void_at = after_extern
            .find("void ")
            .ok_or_else(|| "SM89 exact-F32 extern-C declaration is not a void export".to_owned())?;
        let declaration = &after_extern[void_at + "void ".len()..];
        let name_end = declaration
            .find('(')
            .ok_or_else(|| "SM89 exact-F32 export has no parameter list".to_owned())?;
        let name = declaration[..name_end].trim();
        if name.is_empty()
            || !name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        {
            return Err(format!("invalid SM89 exact-F32 export name {name:?}"));
        }
        exports.push(name);
        remaining = declaration;
    }
    Ok(exports)
}

pub fn source_unit<'a>(source: &'a str, symbol: &str) -> Option<&'a str> {
    let symbol_at = source.find(symbol)?;
    if symbol == D768_IN_FUSED_SYMBOL {
        let start = source[..symbol_at].rfind("struct DualChunkParams")?;
        let end_marker = "#undef DUAL_BM";
        let marker_at = source[symbol_at..].find(end_marker)? + symbol_at;
        let end = source[marker_at..]
            .find('\n')
            .map_or(source.len(), |offset| marker_at + offset);
        return Some(&source[start..end]);
    }
    if symbol == D768_OUT_RAW_SYMBOL || symbol == PRISM_RAW_SYMBOL {
        let start = source[..symbol_at].rfind("#define PRISM_DIRECT_TN_BM")?;
        let end_marker = "#undef PRISM_DIRECT_TN_SMEM_BYTES";
        let marker_at = source[symbol_at..].find(end_marker)? + symbol_at;
        let end = source[marker_at..]
            .find('\n')
            .map_or(source.len(), |offset| marker_at + offset);
        return Some(&source[start..end]);
    }
    None
}

pub fn validate_source_text(source: &str) -> Result<(), String> {
    for marker in [
        "_test_",
        "_exp_",
        "__DUAL_SYMBOL__",
        "__DUAL_EPILOGUE__",
        "__SM89_EXACT_F32_DIRECT_SYMBOL__",
    ] {
        if source.contains(marker) {
            return Err(format!(
                "SM89 exact-F32 source retained discovery marker {marker}"
            ));
        }
    }
    let exports = export_inventory(source)?;
    let expected = [D768_IN_FUSED_SYMBOL, D768_OUT_RAW_SYMBOL, PRISM_RAW_SYMBOL]
        .into_iter()
        .collect::<BTreeSet<_>>();
    let observed = exports.iter().copied().collect::<BTreeSet<_>>();
    if exports.len() != expected.len() || observed != expected {
        return Err(format!(
            "SM89 exact-F32 export inventory changed: expected {expected:?}, observed {exports:?}"
        ));
    }
    for symbol in expected {
        if source_unit(source, symbol).is_none() {
            return Err(format!(
                "SM89 exact-F32 export has no sealed source unit: {symbol}"
            ));
        }
    }
    Ok(())
}

pub fn validate_source() -> Result<(), String> {
    if OWNER_SHA256.len() != 64
        || !OWNER_SHA256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err("SM89 exact-F32 owner SHA-256 freeze is not lowercase hex".into());
    }
    let encoded_bytes = OWNER_SHA256_BYTES
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    if encoded_bytes != OWNER_SHA256 {
        return Err("SM89 exact-F32 owner SHA-256 text/byte freezes disagree".into());
    }
    validate_source_text(&compose_source()?)
}
