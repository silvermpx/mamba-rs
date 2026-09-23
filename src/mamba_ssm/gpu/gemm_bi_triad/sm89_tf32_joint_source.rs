use std::collections::BTreeSet;

pub const SOURCE: &str = include_str!("../../../../kernels/gemm_bi_triad/sm80/tf32_joint.cu");
pub const PRIMITIVES: &str =
    include_str!("../../../../kernels/gemm_bi_triad/sm80/tf32_joint_primitives.cuh");
/// The wide-tile fragments: one file per operand layout, each exporting
/// the tiles measured on the Ada board after the joint module was sealed.
pub const WIDE_NT: &str = include_str!("../../../../kernels/gemm_bi_triad/sm80/tf32_wide_nt.cu");
pub const WIDE_NT_EPI: &str =
    include_str!("../../../../kernels/gemm_bi_triad/sm80/tf32_wide_nt_epi.cu");
pub const WIDE_TN: &str = include_str!("../../../../kernels/gemm_bi_triad/sm80/tf32_wide_tn.cu");
pub const WIDE_NN: &str = include_str!("../../../../kernels/gemm_bi_triad/sm80/tf32_wide_nn.cu");

pub const SOURCE_SHA256: &str = "49db8e286cf2d335314fa6b8a92d1288dcdaa2f8a0efcac5458f556c7c1702a6";
pub const PRIMITIVES_SHA256: &str =
    "c16e81fdcc4745352c97ee7daa39f2629716d7ebe38b6eea0a91393268303b0e";
pub const WIDE_NT_SHA256: &str = "4e74410d3045e8fc21583c64c409777b29593c81a1884974e1c6a701d3fc5458";
pub const WIDE_NT_EPI_SHA256: &str =
    "823820682537ac24c07ad8535fdddf7f94fefb25a838a5def3fa70c8b80a79d2";
pub const WIDE_TN_SHA256: &str = "06d1f793a3cc7390d8dab3e7f69cea1c80b992be13e0d03f415e578d6e105ffe";
pub const WIDE_NN_SHA256: &str = "ce4cc1f1267e8788ee1c50ea17bb9e4bb8d56a9cd525fc074c6fcf1286762885";

pub const COPY_CG_PRIMITIVE: &str = "gbf_tf32_copy_cg";
pub const MMA_M16N8K8_PRIMITIVE: &str = "gbf_tf32_mma_m16n8k8";
pub const ALIGNMENT_PRIMITIVE: &str = "gbf_aligned16";

pub const TN_PRE_RNA_TRANSPOSE_SYMBOL: &str = "tn_sm89_tf32_pre_rna_transpose_32x32";
pub const TN_PRE_RNA_N96_SYMBOL: &str = "tn_sm89_tf32_pre_rna_m128n96_bk32_s3";
pub const TN_PRE_RNA_M64N64_SYMBOL: &str = "tn_sm89_tf32_pre_rna_m64n64_bk32_s3";
pub const TN_PRE_RNA_M64N96_S2_SYMBOL: &str = "tn_sm89_tf32_pre_rna_m64n96_bk32_s2";
pub const NN_ADD_HALF_DIRECT_N96_SYMBOL: &str = "nn_sm89_tf32_addhalf_m128n96_bk32_s3_direct";
pub const NN_ADD_HALF_N96_SYMBOL: &str = "nn_sm89_tf32_addhalf_m128n96_bk32_s3";
pub const NT_A_LDMATRIX_N96_SYMBOL: &str = "nt_sm89_tf32_a_ldmatrix_m128n96_bk32_s3";
pub const NT_RNA_M144N96_S2_SYMBOL: &str = "nt_sm89_tf32_rna_m144n96_w3x4_bk32_s2";
pub const NT_ROWSTAGE_M128N192_S2_SYMBOL: &str = "nt_sm89_tf32_rowstage_m128n192_w2x4_bk32_s2";
pub const TN_DIRECT_M192N192_S2_SYMBOL: &str = "tn_sm89_tf32_m192n192_w3x4_bk32_s2";
pub const TN_PRE_RNA_M96N192_S2_SYMBOL: &str = "tn_sm89_tf32_pre_rna_m96n192_w3x4_bk32_s2";
pub const TN_PRE_RNA_M96N96_S3_SYMBOL: &str = "tn_sm89_tf32_pre_rna_m96n96_bk32_s3";

/// The seven exports of the sealed owner file in lexicographic order, the
/// order the inventory scan reports them in.
pub const SM89_TF32_JOINT_SEALED_SYMBOLS: [&str; 7] = [
    NN_ADD_HALF_N96_SYMBOL,
    NN_ADD_HALF_DIRECT_N96_SYMBOL,
    NT_A_LDMATRIX_N96_SYMBOL,
    TN_PRE_RNA_N96_SYMBOL,
    TN_PRE_RNA_M64N64_SYMBOL,
    TN_PRE_RNA_M64N96_S2_SYMBOL,
    TN_PRE_RNA_TRANSPOSE_SYMBOL,
];

/// Every export of the composed module, the sealed seven and the five wide
/// tiles, in lexicographic order.
pub const SM89_TF32_JOINT_SYMBOLS: [&str; 12] = [
    NN_ADD_HALF_N96_SYMBOL,
    NN_ADD_HALF_DIRECT_N96_SYMBOL,
    NT_A_LDMATRIX_N96_SYMBOL,
    NT_RNA_M144N96_S2_SYMBOL,
    NT_ROWSTAGE_M128N192_S2_SYMBOL,
    TN_DIRECT_M192N192_S2_SYMBOL,
    TN_PRE_RNA_N96_SYMBOL,
    TN_PRE_RNA_M64N64_SYMBOL,
    TN_PRE_RNA_M64N96_S2_SYMBOL,
    TN_PRE_RNA_M96N192_S2_SYMBOL,
    TN_PRE_RNA_M96N96_S3_SYMBOL,
    TN_PRE_RNA_TRANSPOSE_SYMBOL,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct Sm89Tf32JointTransposeParams {
    pub rows: i32,
    pub columns: i32,
    pub output_stride: i32,
}

const _: () = assert!(std::mem::size_of::<Sm89Tf32JointTransposeParams>() == 12);
const _: () = assert!(std::mem::align_of::<Sm89Tf32JointTransposeParams>() == 4);

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct Sm89Tf32JointGemmParams {
    pub alpha: f32,
    pub beta: f32,
    pub m: i32,
    pub k: i32,
    pub n: i32,
    pub lda: i32,
    pub ldb: i32,
    pub ldc: i32,
}

const _: () = assert!(std::mem::size_of::<Sm89Tf32JointGemmParams>() == 32);
const _: () = assert!(std::mem::align_of::<Sm89Tf32JointGemmParams>() == 4);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sm89Tf32JointAbiParameter {
    pub offset: u32,
    pub size: u32,
}

pub const TRANSPOSE_DRIVER_ABI: [Sm89Tf32JointAbiParameter; 3] = [
    Sm89Tf32JointAbiParameter { offset: 0, size: 8 },
    Sm89Tf32JointAbiParameter { offset: 8, size: 8 },
    Sm89Tf32JointAbiParameter {
        offset: 16,
        size: 12,
    },
];
pub const TRANSPOSE_TERMINAL_ARGUMENT: u32 = 3;

pub const GEMM_DRIVER_ABI: [Sm89Tf32JointAbiParameter; 5] = [
    Sm89Tf32JointAbiParameter { offset: 0, size: 8 },
    Sm89Tf32JointAbiParameter { offset: 8, size: 8 },
    Sm89Tf32JointAbiParameter {
        offset: 16,
        size: 8,
    },
    Sm89Tf32JointAbiParameter {
        offset: 24,
        size: 8,
    },
    Sm89Tf32JointAbiParameter {
        offset: 32,
        size: 32,
    },
];
pub const GEMM_TERMINAL_ARGUMENT: u32 = 5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sm89Tf32JointKernelKind {
    NnAddHalfDirectM128N96Bk32S3,
    NnAddHalfM128N96Bk32S3,
    NtALdmatrixM128N96Bk32S3,
    TnPreRnaM128N96Bk32S3,
    TnPreRnaM64N64Bk32S3,
    TnPreRnaM64N96Bk32S2,
    TnPreRnaTranspose32x32,
    NtRnaM144N96Bk32S2,
    NtRowstageM128N192Bk32S2,
    TnDirectM192N192Bk32S2,
    TnPreRnaM96N192Bk32S2,
    TnPreRnaM96N96Bk32S3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sm89Tf32JointKernelSpec {
    pub symbol: &'static str,
    pub kind: Sm89Tf32JointKernelKind,
    pub block: (u32, u32, u32),
    pub dynamic_shared_bytes: u32,
    pub static_shared_bytes: u32,
    pub register_cap: u32,
    pub local_bytes: u32,
    pub minimum_max_threads: u32,
    pub minimum_active_blocks: Option<u32>,
    pub abi_parameters: &'static [Sm89Tf32JointAbiParameter],
    pub terminal_argument: u32,
    pub abi_parameter_bytes: u32,
    pub params_bytes: u32,
    pub params_alignment: u32,
}

const fn gemm_spec(
    symbol: &'static str,
    kind: Sm89Tf32JointKernelKind,
    dynamic_shared_bytes: u32,
    register_cap: u32,
    minimum_active_blocks: u32,
) -> Sm89Tf32JointKernelSpec {
    wide_spec(
        symbol,
        kind,
        256,
        dynamic_shared_bytes,
        register_cap,
        0,
        minimum_active_blocks,
    )
}

/// A GEMM export at its own CTA width: the wide tiles run eight or twelve
/// warps.
const fn wide_spec(
    symbol: &'static str,
    kind: Sm89Tf32JointKernelKind,
    threads: u32,
    dynamic_shared_bytes: u32,
    register_cap: u32,
    local_bytes: u32,
    minimum_active_blocks: u32,
) -> Sm89Tf32JointKernelSpec {
    Sm89Tf32JointKernelSpec {
        symbol,
        kind,
        block: (threads, 1, 1),
        dynamic_shared_bytes,
        static_shared_bytes: 0,
        register_cap,
        local_bytes,
        minimum_max_threads: 256,
        minimum_active_blocks: Some(minimum_active_blocks),
        abi_parameters: &GEMM_DRIVER_ABI,
        terminal_argument: GEMM_TERMINAL_ARGUMENT,
        abi_parameter_bytes: 64,
        params_bytes: 32,
        params_alignment: 4,
    }
}

// Register caps cover the largest spill-free allocation observed across the
// supported CUDA 12.8, 13.0, and 13.2 JITs. CUDA 13.2 allocates 126/126/127
// registers for the original N96 kernels; CUDA 12.8 and 13.0 allocate
// 135/135/135. The NT route with both operands through ldmatrix and the
// half-ulp add allocates 116 on all three.
// Occupancy remains sealed independently below.
pub const SM89_TF32_JOINT_KERNEL_SPECS: [Sm89Tf32JointKernelSpec; 12] = [
    gemm_spec(
        NN_ADD_HALF_N96_SYMBOL,
        Sm89Tf32JointKernelKind::NnAddHalfM128N96Bk32S3,
        86_016,
        135,
        1,
    ),
    gemm_spec(
        NN_ADD_HALF_DIRECT_N96_SYMBOL,
        Sm89Tf32JointKernelKind::NnAddHalfDirectM128N96Bk32S3,
        86_016,
        135,
        1,
    ),
    gemm_spec(
        NT_A_LDMATRIX_N96_SYMBOL,
        Sm89Tf32JointKernelKind::NtALdmatrixM128N96Bk32S3,
        86_016,
        128,
        1,
    ),
    wide_spec(
        NT_RNA_M144N96_S2_SYMBOL,
        Sm89Tf32JointKernelKind::NtRnaM144N96Bk32S2,
        384,
        61_440,
        168,
        0,
        1,
    ),
    wide_spec(
        NT_ROWSTAGE_M128N192_S2_SYMBOL,
        Sm89Tf32JointKernelKind::NtRowstageM128N192Bk32S2,
        256,
        81_920,
        255,
        0,
        1,
    ),
    wide_spec(
        TN_DIRECT_M192N192_S2_SYMBOL,
        Sm89Tf32JointKernelKind::TnDirectM192N192Bk32S2,
        384,
        98_304,
        168,
        88,
        1,
    ),
    gemm_spec(
        TN_PRE_RNA_N96_SYMBOL,
        Sm89Tf32JointKernelKind::TnPreRnaM128N96Bk32S3,
        86_016,
        135,
        1,
    ),
    gemm_spec(
        TN_PRE_RNA_M64N64_SYMBOL,
        Sm89Tf32JointKernelKind::TnPreRnaM64N64Bk32S3,
        49_152,
        83,
        2,
    ),
    gemm_spec(
        TN_PRE_RNA_M64N96_S2_SYMBOL,
        Sm89Tf32JointKernelKind::TnPreRnaM64N96Bk32S2,
        40_960,
        128,
        2,
    ),
    wide_spec(
        TN_PRE_RNA_M96N192_S2_SYMBOL,
        Sm89Tf32JointKernelKind::TnPreRnaM96N192Bk32S2,
        384,
        73_728,
        168,
        0,
        1,
    ),
    wide_spec(
        TN_PRE_RNA_M96N96_S3_SYMBOL,
        Sm89Tf32JointKernelKind::TnPreRnaM96N96Bk32S3,
        256,
        73_728,
        255,
        0,
        1,
    ),
    Sm89Tf32JointKernelSpec {
        symbol: TN_PRE_RNA_TRANSPOSE_SYMBOL,
        kind: Sm89Tf32JointKernelKind::TnPreRnaTranspose32x32,
        block: (32, 8, 1),
        dynamic_shared_bytes: 0,
        static_shared_bytes: 4_224,
        register_cap: 26,
        local_bytes: 0,
        minimum_max_threads: 256,
        minimum_active_blocks: None,
        abi_parameters: &TRANSPOSE_DRIVER_ABI,
        terminal_argument: TRANSPOSE_TERMINAL_ARGUMENT,
        abi_parameter_bytes: 28,
        params_bytes: 12,
        params_alignment: 4,
    },
];

pub fn kernel_spec(symbol: &str) -> Option<&'static Sm89Tf32JointKernelSpec> {
    SM89_TF32_JOINT_KERNEL_SPECS
        .iter()
        .find(|spec| spec.symbol == symbol)
}

pub fn compose_source() -> Result<String, String> {
    validate_primitives_text(PRIMITIVES)?;
    validate_sealed_source_text(SOURCE)?;
    let composed = format!(
        "{PRIMITIVES}\n{SOURCE}\n{}\n{}\n{}\n{}",
        WIDE_NT.trim_end(),
        WIDE_NT_EPI.trim_end(),
        WIDE_TN.trim_end(),
        WIDE_NN.trim_end()
    );
    validate_source_text(&composed)?;
    Ok(composed)
}

pub fn validate_primitives_text(source: &str) -> Result<(), String> {
    if source.contains("extern \"C\"") {
        return Err("SM89 TF32 joint primitives must not export kernels".into());
    }
    let mut observed = BTreeSet::new();
    let mut remaining = source;
    const PREFIX: &str = "__device__ __forceinline__ void ";
    while let Some(function_at) = remaining.find(PREFIX) {
        let declaration = &remaining[function_at + PREFIX.len()..];
        let name_end = declaration
            .find('(')
            .ok_or_else(|| "SM89 TF32 joint primitive has no parameter list".to_owned())?;
        let name = declaration[..name_end].trim();
        if !name.starts_with("gbf_tf32_") {
            return Err(format!("foreign SM89 TF32 joint primitive {name}"));
        }
        observed.insert(name);
        remaining = declaration;
    }
    let expected = [COPY_CG_PRIMITIVE, MMA_M16N8K8_PRIMITIVE]
        .into_iter()
        .collect::<BTreeSet<_>>();
    if observed != expected {
        return Err(format!(
            "SM89 TF32 joint primitive inventory changed: expected {expected:?}, observed {observed:?}"
        ));
    }
    for primitive in expected {
        let signature = format!("{PREFIX}{primitive}(");
        if source.matches(&signature).count() != 1 {
            return Err(format!(
                "SM89 TF32 joint primitive must have one definition: {primitive}"
            ));
        }
    }
    let alignment_signature = "static __device__ __forceinline__ bool gbf_aligned16(const void* p)";
    if source.matches(alignment_signature).count() != 1 {
        return Err("SM89 TF32 joint alignment primitive must have one definition".into());
    }
    let mut names = BTreeSet::new();
    let mut tokens = source;
    while let Some(name_at) = tokens.find("gbf_") {
        let name = &tokens[name_at..];
        let name_end = name
            .bytes()
            .position(|byte| !(byte.is_ascii_alphanumeric() || byte == b'_'))
            .unwrap_or(name.len());
        names.insert(&name[..name_end]);
        tokens = &name[name_end..];
    }
    let expected = [
        COPY_CG_PRIMITIVE,
        MMA_M16N8K8_PRIMITIVE,
        ALIGNMENT_PRIMITIVE,
    ]
    .into_iter()
    .collect::<BTreeSet<_>>();
    if names != expected {
        return Err(format!(
            "SM89 TF32 joint primitive references changed: expected {expected:?}, observed {names:?}"
        ));
    }
    Ok(())
}

pub fn export_inventory(source: &str) -> Result<Vec<&str>, String> {
    let mut exports = Vec::new();
    let mut remaining = source;
    while let Some(extern_at) = remaining.find("extern \"C\"") {
        let after_extern = &remaining[extern_at + "extern \"C\"".len()..];
        let void_at = after_extern.find("void ").ok_or_else(|| {
            "SM89 TF32 joint extern-C declaration is not a void export".to_owned()
        })?;
        let declaration = &after_extern[void_at + "void ".len()..];
        let name_end = declaration
            .find('(')
            .ok_or_else(|| "SM89 TF32 joint export has no parameter list".to_owned())?;
        let name = declaration[..name_end].trim();
        if name.is_empty()
            || !name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        {
            return Err(format!("invalid SM89 TF32 joint export name {name:?}"));
        }
        exports.push(name);
        remaining = declaration;
    }
    exports.sort_unstable();
    Ok(exports)
}

fn validate_inventory(source: &str, symbols: &[&str]) -> Result<(), String> {
    for marker in ["_test_", "_exp_"] {
        if source.contains(marker) {
            return Err(format!(
                "SM89 TF32 joint source retained discovery marker {marker}"
            ));
        }
    }
    let exports = export_inventory(source)?;
    let expected = symbols.iter().copied().collect::<BTreeSet<_>>();
    let observed = exports.iter().copied().collect::<BTreeSet<_>>();
    if exports.len() != expected.len() || observed != expected {
        return Err(format!(
            "SM89 TF32 joint export inventory changed: expected {expected:?}, observed {exports:?}"
        ));
    }
    Ok(())
}

/// The sealed owner file alone: its seven exports.
pub fn validate_sealed_source_text(source: &str) -> Result<(), String> {
    validate_inventory(source, &SM89_TF32_JOINT_SEALED_SYMBOLS)
}

/// The composed module: the sealed seven and the five wide tiles.
pub fn validate_source_text(source: &str) -> Result<(), String> {
    validate_inventory(source, &SM89_TF32_JOINT_SYMBOLS)
}

/// The four wide-tile fragments with their frozen digests: name, text,
/// expected SHA-256. The digest comparison lives with the callers because
/// this file is also compiled standalone by the source contract test.
pub const WIDE_FRAGMENTS: [(&str, &str, &str); 4] = [
    ("wide NT", WIDE_NT, WIDE_NT_SHA256),
    ("wide NT epilogue", WIDE_NT_EPI, WIDE_NT_EPI_SHA256),
    ("wide TN", WIDE_TN, WIDE_TN_SHA256),
    ("wide NN", WIDE_NN, WIDE_NN_SHA256),
];

pub fn validate_source() -> Result<(), String> {
    for (name, digest) in [
        ("source", SOURCE_SHA256),
        ("primitives", PRIMITIVES_SHA256),
        ("wide NT", WIDE_NT_SHA256),
        ("wide NT epilogue", WIDE_NT_EPI_SHA256),
        ("wide TN", WIDE_TN_SHA256),
        ("wide NN", WIDE_NN_SHA256),
    ] {
        if digest.len() != 64
            || !digest
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            return Err(format!(
                "SM89 TF32 joint {name} SHA-256 freeze is not lowercase hex"
            ));
        }
    }
    validate_source_text(&compose_source()?)
}
