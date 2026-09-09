use std::collections::BTreeSet;

pub const SOURCE: &str = include_str!("../../../../kernels/gemm_bi_triad/sm89_tf32_joint.cu");
pub const PRIMITIVES: &str =
    include_str!("../../../../kernels/gemm_bi_triad/sm89_tf32_joint_primitives.cuh");

pub const SOURCE_SHA256: &str = "e1e8a2ad1d2d03b4d0e02730f087eab1c26cfc7712f867fbbead13b032e3624c";
pub const PRIMITIVES_SHA256: &str =
    "c16e81fdcc4745352c97ee7daa39f2629716d7ebe38b6eea0a91393268303b0e";

pub const COPY_CG_PRIMITIVE: &str = "gbf_tf32_copy_cg";
pub const MMA_M16N8K8_PRIMITIVE: &str = "gbf_tf32_mma_m16n8k8";
pub const ALIGNMENT_PRIMITIVE: &str = "gbf_aligned16";

pub const TN_PRE_RNA_TRANSPOSE_SYMBOL: &str = "gemm_bi_tn_sm89_tf32_pre_rna_transpose_32x32_v1";
pub const TN_PRE_RNA_N96_SYMBOL: &str = "gemm_bi_tn_sm89_tf32_pre_rna_m128n96_bk32_s3_v1";
pub const TN_PRE_RNA_M64N64_SYMBOL: &str = "gemm_bi_tn_sm89_tf32_pre_rna_m64n64_bk32_s3_v1";
pub const NN_ADD_HALF_DIRECT_N96_SYMBOL: &str =
    "gemm_bi_nn_sm89_tf32_addhalf_m128n96_bk32_s3_direct_v1";
pub const NN_ADD_HALF_N96_SYMBOL: &str = "gemm_bi_nn_sm89_tf32_addhalf_m128n96_bk32_s3_v1";

pub const SM89_TF32_JOINT_SYMBOLS: [&str; 5] = [
    NN_ADD_HALF_DIRECT_N96_SYMBOL,
    NN_ADD_HALF_N96_SYMBOL,
    TN_PRE_RNA_N96_SYMBOL,
    TN_PRE_RNA_M64N64_SYMBOL,
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
    TnPreRnaM128N96Bk32S3,
    TnPreRnaM64N64Bk32S3,
    TnPreRnaTranspose32x32,
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
    Sm89Tf32JointKernelSpec {
        symbol,
        kind,
        block: (256, 1, 1),
        dynamic_shared_bytes,
        static_shared_bytes: 0,
        register_cap,
        local_bytes: 0,
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
// supported CUDA 12.8, 13.0, and 13.2 JITs. CUDA 13.2 allocates 124/124/127
// registers for the N96 kernels; CUDA 12.8 and 13.0 allocate 131/131/135.
// Occupancy remains sealed independently below.
pub const SM89_TF32_JOINT_KERNEL_SPECS: [Sm89Tf32JointKernelSpec; 5] = [
    gemm_spec(
        NN_ADD_HALF_DIRECT_N96_SYMBOL,
        Sm89Tf32JointKernelKind::NnAddHalfDirectM128N96Bk32S3,
        86_016,
        131,
        1,
    ),
    gemm_spec(
        NN_ADD_HALF_N96_SYMBOL,
        Sm89Tf32JointKernelKind::NnAddHalfM128N96Bk32S3,
        86_016,
        131,
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
    validate_source_text(SOURCE)?;
    Ok(format!("{PRIMITIVES}\n{SOURCE}"))
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

pub fn validate_source_text(source: &str) -> Result<(), String> {
    for marker in ["_test_", "_exp_"] {
        if source.contains(marker) {
            return Err(format!(
                "SM89 TF32 joint source retained discovery marker {marker}"
            ));
        }
    }
    let exports = export_inventory(source)?;
    let expected = SM89_TF32_JOINT_SYMBOLS.into_iter().collect::<BTreeSet<_>>();
    let observed = exports.iter().copied().collect::<BTreeSet<_>>();
    if exports.len() != expected.len() || observed != expected {
        return Err(format!(
            "SM89 TF32 joint export inventory changed: expected {expected:?}, observed {exports:?}"
        ));
    }
    Ok(())
}

pub fn validate_source() -> Result<(), String> {
    for (name, digest) in [("source", SOURCE_SHA256), ("primitives", PRIMITIVES_SHA256)] {
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
