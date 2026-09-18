use std::collections::BTreeSet;

pub const OWNER_TEMPLATE: &str =
    include_str!("../../../../kernels/gemm_bi_triad/sm80/exact_f32_d128.cu");

pub const OWNER_SHA256: &str = "3d5226abe9d28b17fd0c6d02718da57a2bd2fa5906fbe1305cac9f60d019e70f";
pub const OWNER_SHA256_BYTES: [u8; 32] = [
    0x3d, 0x52, 0x26, 0xab, 0xe9, 0xd2, 0x8b, 0x17, 0xfd, 0x0c, 0x6d, 0x02, 0x71, 0x8d, 0xa5, 0x7a,
    0x2b, 0xd2, 0xfa, 0x59, 0x06, 0xfb, 0xe1, 0x30, 0x5c, 0xac, 0x9f, 0x60, 0xd0, 0x19, 0xe7, 0x0f,
];

pub const D128_IN_SYMBOL: &str = "tn_sm89_f32_d128_in_m16n16_g8_s2_cg";
pub const D128_OUT_SYMBOL: &str = "tn_sm89_f32_d128_out_m16n16_g8_s2_cg";

/// Separate CUDA Driver arguments: output, A, B, alpha, M, K, N.
/// Each pair is the byte offset and size in the kernel parameter buffer.
pub const DIRECT_FOLD_DRIVER_ABI: [(u32, u32); 7] =
    [(0, 8), (8, 8), (16, 8), (24, 4), (28, 4), (32, 4), (36, 4)];
pub const DIRECT_FOLD_TERMINAL_ARGUMENT: u32 = 7;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sm89ExactF32D128KernelKind {
    DirectF64FoldFinal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sm89ExactF32D128Route {
    D128InDirectFold,
    D128OutDirectFold,
}

impl Sm89ExactF32D128Route {
    pub const fn symbol(self) -> &'static str {
        match self {
            Self::D128InDirectFold => D128_IN_SYMBOL,
            Self::D128OutDirectFold => D128_OUT_SYMBOL,
        }
    }
}

/// Frozen launch and resource limits for the two direct folds that stand in
/// for the 64-chunk split-M reduction. These bounds are qualification
/// requirements, not live receipts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sm89ExactF32D128KernelSpec {
    pub symbol: &'static str,
    pub kind: Sm89ExactF32D128KernelKind,
    pub shape: (usize, usize, usize),
    pub tile: (u32, u32),
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub dynamic_shared_bytes: u32,
    pub static_shared_bytes: u32,
    pub local_bytes: u32,
    pub register_cap: u32,
    pub occupancy_gate: u32,
    pub chunks: u32,
    pub m_chunk: u32,
    pub abi_parameter_count: u32,
    pub abi_parameter_bytes: u32,
}

pub const SM89_EXACT_F32_D128_KERNEL_SPECS: [Sm89ExactF32D128KernelSpec; 2] = [
    Sm89ExactF32D128KernelSpec {
        symbol: D128_IN_SYMBOL,
        kind: Sm89ExactF32D128KernelKind::DirectF64FoldFinal,
        shape: (1_024, 128, 512),
        tile: (16, 16),
        grid: (256, 1, 1),
        block: (256, 1, 1),
        dynamic_shared_bytes: 49_152,
        static_shared_bytes: 0,
        local_bytes: 0,
        register_cap: 128,
        occupancy_gate: 2,
        chunks: 64,
        m_chunk: 16,
        abi_parameter_count: 7,
        abi_parameter_bytes: 40,
    },
    Sm89ExactF32D128KernelSpec {
        symbol: D128_OUT_SYMBOL,
        kind: Sm89ExactF32D128KernelKind::DirectF64FoldFinal,
        shape: (1_024, 256, 128),
        tile: (16, 16),
        grid: (128, 1, 1),
        block: (256, 1, 1),
        dynamic_shared_bytes: 49_152,
        static_shared_bytes: 0,
        local_bytes: 0,
        register_cap: 128,
        occupancy_gate: 2,
        chunks: 64,
        m_chunk: 16,
        abi_parameter_count: 7,
        abi_parameter_bytes: 40,
    },
];

pub fn kernel_spec(symbol: &str) -> Option<&'static Sm89ExactF32D128KernelSpec> {
    SM89_EXACT_F32_D128_KERNEL_SPECS
        .iter()
        .find(|spec| spec.symbol == symbol)
}

const ROUTE_BEGIN: &str = "// SM89_EXACT_F32_D128_ROUTE_BEGIN";
const ROUTE_END: &str = "// SM89_EXACT_F32_D128_ROUTE_END";
const NAMESPACE_PLACEHOLDER: &str = "__SM89_EXACT_F32_D128_NAMESPACE__";
const K_OUT_PLACEHOLDER: &str = "__SM89_EXACT_F32_D128_K_OUT__";
const N_PLACEHOLDER: &str = "__SM89_EXACT_F32_D128_N__";
const SHARED_PLACEHOLDER: &str = "__SM89_EXACT_F32_D128_SHARED_BYTES__";
const GRID_PLACEHOLDER: &str = "__SM89_EXACT_F32_D128_GRID__";
const SYMBOL_PLACEHOLDER: &str = "__SM89_EXACT_F32_D128_SYMBOL__";

const D128_IN_NAMESPACE: &str = "GemmBiTnSm89ExactF32D128In";
const D128_OUT_NAMESPACE: &str = "GemmBiTnSm89ExactF32D128Out";

#[derive(Clone, Copy)]
struct RouteConfig {
    namespace: &'static str,
    symbol: &'static str,
    k_out: &'static str,
    n: &'static str,
    shared_bytes: &'static str,
    grid: &'static str,
}

const ROUTES: [RouteConfig; 2] = [
    RouteConfig {
        namespace: D128_IN_NAMESPACE,
        symbol: D128_IN_SYMBOL,
        k_out: "128",
        n: "512",
        shared_bytes: "49152",
        grid: "256",
    },
    RouteConfig {
        namespace: D128_OUT_NAMESPACE,
        symbol: D128_OUT_SYMBOL,
        k_out: "256",
        n: "128",
        shared_bytes: "49152",
        grid: "128",
    },
];

fn section<'a>(source: &'a str, begin: &str, end: &str) -> Result<&'a str, String> {
    let begin_at = source
        .find(begin)
        .ok_or_else(|| format!("SM89 exact-F32 d128 owner is missing {begin}"))?;
    let body_at = begin_at + begin.len();
    let end_at = source[body_at..]
        .find(end)
        .map(|offset| body_at + offset)
        .ok_or_else(|| format!("SM89 exact-F32 d128 owner is missing {end}"))?;
    if source[body_at..end_at].contains(begin) || source[end_at + end.len()..].contains(end) {
        return Err("SM89 exact-F32 d128 owner duplicated a route marker".into());
    }
    Ok(source[body_at..end_at].trim())
}

fn instantiate(template: &str, route: RouteConfig) -> String {
    template
        .replace(NAMESPACE_PLACEHOLDER, route.namespace)
        .replace(K_OUT_PLACEHOLDER, route.k_out)
        .replace(N_PLACEHOLDER, route.n)
        .replace(SHARED_PLACEHOLDER, route.shared_bytes)
        .replace(GRID_PLACEHOLDER, route.grid)
        .replace(SYMBOL_PLACEHOLDER, route.symbol)
}

pub fn compose_source() -> Result<String, String> {
    let template = section(OWNER_TEMPLATE, ROUTE_BEGIN, ROUTE_END)?;
    Ok(ROUTES
        .into_iter()
        .map(|route| instantiate(template, route))
        .collect::<Vec<_>>()
        .join("\n\n")
        + "\n")
}

pub fn export_inventory(source: &str) -> Result<Vec<&str>, String> {
    let mut exports = Vec::new();
    let mut remaining = source;
    while let Some(extern_at) = remaining.find("extern \"C\"") {
        let after_extern = &remaining[extern_at + "extern \"C\"".len()..];
        let void_at = after_extern.find("void ").ok_or_else(|| {
            "SM89 exact-F32 d128 extern-C declaration is not a void export".to_owned()
        })?;
        let declaration = &after_extern[void_at + "void ".len()..];
        let name_end = declaration
            .find('(')
            .ok_or_else(|| "SM89 exact-F32 d128 export has no parameter list".to_owned())?;
        let name = declaration[..name_end].trim();
        if name.is_empty()
            || !name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        {
            return Err(format!("invalid SM89 exact-F32 d128 export name {name:?}"));
        }
        exports.push(name);
        remaining = declaration;
    }
    Ok(exports)
}

pub fn source_unit<'a>(source: &'a str, symbol: &str) -> Option<&'a str> {
    let namespace = match symbol {
        D128_IN_SYMBOL => D128_IN_NAMESPACE,
        D128_OUT_SYMBOL => D128_OUT_NAMESPACE,
        _ => return None,
    };
    let start = source.find(&format!("namespace {namespace} {{"))?;
    let end = if symbol == D128_IN_SYMBOL {
        source[start..]
            .find(&format!("namespace {D128_OUT_NAMESPACE} {{"))
            .map(|offset| start + offset)
            .unwrap_or(source.len())
    } else {
        source.len()
    };
    let unit = source[start..end].trim();
    unit.contains(symbol).then_some(unit)
}

pub fn validate_source_text(source: &str) -> Result<(), String> {
    for marker in [
        "_test_",
        "_exp_",
        "foldpipe",
        "M8N32",
        "M16N16T128",
        NAMESPACE_PLACEHOLDER,
        K_OUT_PLACEHOLDER,
        N_PLACEHOLDER,
        SHARED_PLACEHOLDER,
        GRID_PLACEHOLDER,
        SYMBOL_PLACEHOLDER,
    ] {
        if source.contains(marker) {
            return Err(format!(
                "SM89 exact-F32 d128 source retained forbidden marker {marker}"
            ));
        }
    }
    let exports = export_inventory(source)?;
    let expected = [D128_IN_SYMBOL, D128_OUT_SYMBOL]
        .into_iter()
        .collect::<BTreeSet<_>>();
    let observed = exports.iter().copied().collect::<BTreeSet<_>>();
    if exports.len() != expected.len() || observed != expected {
        return Err(format!(
            "SM89 exact-F32 d128 export inventory changed: expected {expected:?}, observed {exports:?}"
        ));
    }
    for symbol in expected {
        if source_unit(source, symbol).is_none() {
            return Err(format!(
                "SM89 exact-F32 d128 export has no sealed source unit: {symbol}"
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
        return Err("SM89 exact-F32 d128 owner SHA-256 freeze is not lowercase hex".into());
    }
    let encoded_bytes = OWNER_SHA256_BYTES
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    if encoded_bytes != OWNER_SHA256 {
        return Err("SM89 exact-F32 d128 owner SHA-256 text/byte freezes disagree".into());
    }
    validate_source_text(&compose_source()?)
}
