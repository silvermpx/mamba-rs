//! The source owner of the batch-invariant TF32 forward's wide and
//! multistage NN tiles and of the multistage input-gradient tiles
//! (`kernels/gemm_bi_triad/sm80/tf32_rna.cu`). The module is its own so that
//! no other module's identity moves with it; it composes on every SM80-tier
//! target. Every tile rounds both operands with cvt.rna and walks the
//! reduction in the portable tiles' k8 order; the forward's tiles seed an
//! output with its bias, so any of them gives an output the portable tiles'
//! bits, and the weight gradient's split members fold the portable split
//! members' partitions in their order.

use std::collections::BTreeSet;

pub const SOURCE: &str = include_str!("../../../kernels/gemm_bi_triad/sm80/tf32_rna.cu");

pub const SOURCE_SHA256: &str = "511c4997835f32b470e893b6573f6d843427dcadc17323f25d9060f8a91e8dd1";

pub const NN_M128N96_S3_SYMBOL: &str = "nn_sm80_tf32_rna_m128n96_w2x4_bk32_s3";
pub const NN_M96N96_S3_SYMBOL: &str = "nn_sm80_tf32_rna_m96n96_w2x4_bk32_s3";
pub const NN_M128N128_S3_SYMBOL: &str = "nn_sm80_tf32_rna_m128n128_bk32_s3";
pub const NN_MS_M64N128_BK16_S6_SYMBOL: &str = "nn_sm80_tf32_rna_ms_m64n128_bk16_s6";
pub const NN_MS_M128N64_S3_SYMBOL: &str = "nn_sm80_tf32_rna_ms_m128n64_bk32_s3";
pub const NN_MS_M64N64_S3_SYMBOL: &str = "nn_sm80_tf32_rna_ms_m64n64_bk32_s3";
pub const NN_MS_PERSIST_M128N64_BK16_S3_SYMBOL: &str =
    "nn_sm80_tf32_rna_ms_persist_m128n64_bk16_s3";
pub const NT_MS_M64N128_BK16_S6_SYMBOL: &str = "nt_sm80_tf32_rna_ms_m64n128_bk16_s6";
pub const NT_MS_M128N64_S3_SYMBOL: &str = "nt_sm80_tf32_rna_ms_m128n64_bk32_s3";
pub const NT_MS_M128N128_BK16_S5_SYMBOL: &str = "nt_sm80_tf32_rna_ms_m128n128_bk16_s5";
pub const TN_MS_SPLITK8_M128N64_S3_SYMBOL: &str = "tn_sm80_tf32_rna_ms_splitk8_m128n64_bk32_s3";
pub const TN_MS_SPLITK8_M64N64_S4_SYMBOL: &str = "tn_sm80_tf32_rna_ms_splitk8_m64n64_bk32_s4";
pub const TN_MS_SPLITK8_M64N64_S4_ANY_SYMBOL: &str =
    "tn_sm80_tf32_rna_ms_splitk8_m64n64_bk32_s4_any";

/// Every export of the module, in lexicographic order.
pub const TF32_RNA_SYMBOLS: [&str; 13] = [
    NN_M128N128_S3_SYMBOL,
    NN_M128N96_S3_SYMBOL,
    NN_M96N96_S3_SYMBOL,
    NN_MS_M128N64_S3_SYMBOL,
    NN_MS_M64N128_BK16_S6_SYMBOL,
    NN_MS_M64N64_S3_SYMBOL,
    NN_MS_PERSIST_M128N64_BK16_S3_SYMBOL,
    NT_MS_M128N128_BK16_S5_SYMBOL,
    NT_MS_M128N64_S3_SYMBOL,
    NT_MS_M64N128_BK16_S6_SYMBOL,
    TN_MS_SPLITK8_M128N64_S3_SYMBOL,
    TN_MS_SPLITK8_M64N64_S4_SYMBOL,
    TN_MS_SPLITK8_M64N64_S4_ANY_SYMBOL,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tf32RnaAbiParameter {
    pub offset: u32,
    pub size: u32,
}

/// Output, A, B and bias pointers, then the 32-byte parameter block.
pub const GEMM_DRIVER_ABI: [Tf32RnaAbiParameter; 5] = [
    Tf32RnaAbiParameter { offset: 0, size: 8 },
    Tf32RnaAbiParameter { offset: 8, size: 8 },
    Tf32RnaAbiParameter {
        offset: 16,
        size: 8,
    },
    Tf32RnaAbiParameter {
        offset: 24,
        size: 8,
    },
    Tf32RnaAbiParameter {
        offset: 32,
        size: 32,
    },
];

/// Output, partial planes, arrival counters, A, B and bias pointers, then
/// the 32-byte parameter block: the portable split-K members' ABI.
pub const SPLITK_DRIVER_ABI: [Tf32RnaAbiParameter; 7] = [
    Tf32RnaAbiParameter { offset: 0, size: 8 },
    Tf32RnaAbiParameter { offset: 8, size: 8 },
    Tf32RnaAbiParameter {
        offset: 16,
        size: 8,
    },
    Tf32RnaAbiParameter {
        offset: 24,
        size: 8,
    },
    Tf32RnaAbiParameter {
        offset: 32,
        size: 8,
    },
    Tf32RnaAbiParameter {
        offset: 40,
        size: 8,
    },
    Tf32RnaAbiParameter {
        offset: 48,
        size: 32,
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tf32RnaKernelKind {
    /// A tile of the wide NN template: ldmatrix A, scalar B.
    Wide,
    /// The 128 x 128 tile on eight computing warps.
    Wide128,
    /// A multistage tile.
    Multistage,
    /// The multistage 128 x 64 forward on a resident grid of two CTAs a
    /// multiprocessor, which walks the output tiles.
    Persistent,
    /// A multistage member of the weight gradient's eight-way split.
    SplitK8,
}

impl Tf32RnaKernelKind {
    /// The kernel's Driver parameter ABI.
    pub const fn driver_abi(self) -> &'static [Tf32RnaAbiParameter] {
        match self {
            Self::SplitK8 => &SPLITK_DRIVER_ABI,
            Self::Wide | Self::Wide128 | Self::Multistage | Self::Persistent => &GEMM_DRIVER_ABI,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tf32RnaKernelSpec {
    pub symbol: &'static str,
    pub kind: Tf32RnaKernelKind,
    pub block: (u32, u32, u32),
    pub dynamic_shared_bytes: u32,
    pub register_cap: u32,
    pub local_bytes: u32,
    pub minimum_active_blocks: u32,
}

const fn spec(
    symbol: &'static str,
    kind: Tf32RnaKernelKind,
    threads: u32,
    dynamic_shared_bytes: u32,
    minimum_active_blocks: u32,
) -> Tf32RnaKernelSpec {
    Tf32RnaKernelSpec {
        symbol,
        kind,
        block: (threads, 1, 1),
        dynamic_shared_bytes,
        // The launch bounds set the compiler's register budget; the gate
        // keeps the hardware's ceiling, and a tile that spills to local
        // memory is excluded rather than served slowly.
        register_cap: 255,
        local_bytes: 0,
        minimum_active_blocks,
    }
}

/// The launch geometry of every tile: threads, the dynamic shared memory of
/// its stages, and the CTAs a multiprocessor must hold at once.
pub const TF32_RNA_KERNEL_SPECS: [Tf32RnaKernelSpec; 13] = [
    spec(
        NN_M128N96_S3_SYMBOL,
        Tf32RnaKernelKind::Wide,
        256,
        86_016,
        1,
    ),
    spec(NN_M96N96_S3_SYMBOL, Tf32RnaKernelKind::Wide, 256, 73_728, 1),
    spec(
        NN_M128N128_S3_SYMBOL,
        Tf32RnaKernelKind::Wide128,
        256,
        98_304,
        1,
    ),
    spec(
        NN_MS_M64N128_BK16_S6_SYMBOL,
        Tf32RnaKernelKind::Multistage,
        128,
        76_800,
        1,
    ),
    spec(
        NN_MS_M128N64_S3_SYMBOL,
        Tf32RnaKernelKind::Multistage,
        128,
        76_800,
        1,
    ),
    // Its launch bounds ask the compiler for two CTAs' registers, but two
    // of its 52 KB rings exceed the 100 KB a CC 8.6, 8.9 or 12.x
    // multiprocessor holds; one CTA is the gate everywhere.
    spec(
        NN_MS_M64N64_S3_SYMBOL,
        Tf32RnaKernelKind::Multistage,
        128,
        52_224,
        1,
    ),
    // Two of its 38 KB rings fit every SM80-tier multiprocessor, and the
    // resident grid it launches counts on both.
    spec(
        NN_MS_PERSIST_M128N64_BK16_S3_SYMBOL,
        Tf32RnaKernelKind::Persistent,
        128,
        38_400,
        2,
    ),
    spec(
        NT_MS_M64N128_BK16_S6_SYMBOL,
        Tf32RnaKernelKind::Multistage,
        128,
        73_728,
        1,
    ),
    spec(
        NT_MS_M128N64_S3_SYMBOL,
        Tf32RnaKernelKind::Multistage,
        128,
        73_728,
        1,
    ),
    spec(
        NT_MS_M128N128_BK16_S5_SYMBOL,
        Tf32RnaKernelKind::Multistage,
        128,
        81_920,
        1,
    ),
    spec(
        TN_MS_SPLITK8_M128N64_S3_SYMBOL,
        Tf32RnaKernelKind::SplitK8,
        128,
        79_872,
        1,
    ),
    spec(
        TN_MS_SPLITK8_M64N64_S4_SYMBOL,
        Tf32RnaKernelKind::SplitK8,
        128,
        73_728,
        1,
    ),
    spec(
        TN_MS_SPLITK8_M64N64_S4_ANY_SYMBOL,
        Tf32RnaKernelKind::SplitK8,
        128,
        73_728,
        1,
    ),
];

pub fn kernel_spec(symbol: &str) -> Option<&'static Tf32RnaKernelSpec> {
    TF32_RNA_KERNEL_SPECS
        .iter()
        .find(|spec| spec.symbol == symbol)
}

/// The exports of a source, sorted.
pub fn export_inventory(source: &str) -> Result<Vec<&str>, String> {
    let mut exports = Vec::new();
    let mut remaining = source;
    while let Some(extern_at) = remaining.find("extern \"C\"") {
        let after_extern = &remaining[extern_at + "extern \"C\"".len()..];
        let void_at = after_extern
            .find("void ")
            .ok_or_else(|| "TF32 RNA extern-C declaration is not a void export".to_owned())?;
        let declaration = &after_extern[void_at + "void ".len()..];
        let name_end = declaration
            .find('(')
            .ok_or_else(|| "TF32 RNA export has no parameter list".to_owned())?;
        let name = declaration[..name_end].trim();
        if name.is_empty()
            || !name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        {
            return Err(format!("invalid TF32 RNA export name {name:?}"));
        }
        exports.push(name);
        remaining = declaration;
    }
    exports.sort_unstable();
    Ok(exports)
}

/// The exports a source carries, expanded from the multistage instance
/// macro, against the module's inventory.
pub fn validate_source_text(source: &str) -> Result<(), String> {
    let mut observed = export_inventory(source)?
        .into_iter()
        .filter(|name| !name.contains("NAME"))
        .collect::<Vec<_>>();
    for macro_name in [
        "TF32_RNA_MS_TILE(",
        "TF32_RNA_MS_TN_SPLIT8(",
        "TF32_RNA_MS_TN_SPLIT8_ANY(",
        "TF32_RNA_MS_NN_PERSISTENT(",
    ] {
        let mut remaining = source;
        while let Some(at) = remaining.find(macro_name) {
            let rest = &remaining[at + macro_name.len()..];
            let name = rest
                .split(',')
                .next()
                .map(str::trim)
                .ok_or_else(|| "TF32 RNA multistage instance has no name".to_owned())?;
            if name != "NAME" {
                observed.push(name);
            }
            remaining = rest;
        }
    }
    observed.sort_unstable();
    let expected = TF32_RNA_SYMBOLS.iter().copied().collect::<BTreeSet<_>>();
    let unique = observed.iter().copied().collect::<BTreeSet<_>>();
    if observed.len() != unique.len() || unique != expected {
        return Err(format!(
            "TF32 RNA export inventory changed: expected {expected:?}, observed {observed:?}"
        ));
    }
    // Every operand rounds with cvt.rna: neither the multistage tiles'
    // integer add nor the 128 x 128 tile's half-ulp add may come back.
    if source.contains("bits += 0x1000U") {
        return Err("TF32 RNA source carries the integer operand rounding".into());
    }
    let wide_round =
        "asm(\"cvt.rna.tf32.f32 %0, %1;\" : \"=r\"(result) : \"f\"(__uint_as_float(bits)));";
    if source.matches(wide_round).count() != 1 {
        return Err("the TF32 RNA 128 x 128 tile does not round with cvt.rna".into());
    }
    Ok(())
}

pub fn compose_source() -> Result<String, String> {
    validate_source_text(SOURCE)?;
    Ok(SOURCE.to_owned())
}

pub fn validate_source() -> Result<(), String> {
    if SOURCE_SHA256.len() != 64
        || !SOURCE_SHA256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err("TF32 RNA source SHA-256 freeze is not lowercase hex".into());
    }
    validate_source_text(SOURCE)
}
