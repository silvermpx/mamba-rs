use crate::mamba_ssm::gpu::{
    dtype::WeightDtype,
    kernel_identity::{
        ArtifactIdentity, ArtifactKind, COMPILER_REVISION, COMPOSER_REVISION, CompilerIdentity,
        ModuleKind, NUMERIC_ABI_REVISION, ResolvedGemmOp, SCHEDULE_REVISION,
    },
};

use super::contract::{F32TriadOperands, F32TriadRequest, F32TriadShape};

const BASE_SOURCE: &str = include_str!("../../../../kernels/gemm_bi_triad/sm89/half.cu");

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Sm89HalfRoute {
    NnM128N128Bk64S3,
    TnM64N64Bk64S2CompactBxor,
    TnM64N64Bk64S2RegpipeVec2,
    NtM128N128Bk64S3Bxor,
    NtM96N128Bk64S3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sm89HalfKernelSpec {
    pub route: Sm89HalfRoute,
    pub op: ResolvedGemmOp,
    pub dtype: WeightDtype,
    pub symbol: &'static str,
    pub tile: (u32, u32),
    pub bk: u32,
    pub stages: u8,
    pub threads: u32,
    pub dynamic_shared_bytes: u32,
    pub static_shared_bytes: u32,
    pub register_cap: u32,
    pub occupancy_gate: u32,
}

pub const SM89_HALF_KERNEL_SPECS: [Sm89HalfKernelSpec; 10] = [
    Sm89HalfKernelSpec {
        route: Sm89HalfRoute::NnM128N128Bk64S3,
        op: ResolvedGemmOp::Nn,
        dtype: WeightDtype::F16,
        symbol: "nn_sm89_m128n128_bk64_s3_f16",
        tile: (128, 128),
        bk: 64,
        stages: 3,
        threads: 256,
        dynamic_shared_bytes: 98_304,
        static_shared_bytes: 0,
        register_cap: 188,
        occupancy_gate: 1,
    },
    Sm89HalfKernelSpec {
        route: Sm89HalfRoute::NnM128N128Bk64S3,
        op: ResolvedGemmOp::Nn,
        dtype: WeightDtype::Bf16,
        symbol: "nn_sm89_m128n128_bk64_s3_bf16",
        tile: (128, 128),
        bk: 64,
        stages: 3,
        threads: 256,
        dynamic_shared_bytes: 98_304,
        static_shared_bytes: 0,
        register_cap: 188,
        occupancy_gate: 1,
    },
    Sm89HalfKernelSpec {
        route: Sm89HalfRoute::TnM64N64Bk64S2CompactBxor,
        op: ResolvedGemmOp::Tn,
        dtype: WeightDtype::F16,
        symbol: super::sm89_half_tn_source::COMPACT_F16_SYMBOL,
        tile: (64, 64),
        bk: 64,
        stages: 2,
        threads: 128,
        dynamic_shared_bytes: 0,
        static_shared_bytes: 32_768,
        register_cap: 128,
        occupancy_gate: 3,
    },
    Sm89HalfKernelSpec {
        route: Sm89HalfRoute::TnM64N64Bk64S2CompactBxor,
        op: ResolvedGemmOp::Tn,
        dtype: WeightDtype::Bf16,
        symbol: super::sm89_half_tn_source::COMPACT_BF16_SYMBOL,
        tile: (64, 64),
        bk: 64,
        stages: 2,
        threads: 128,
        dynamic_shared_bytes: 0,
        static_shared_bytes: 32_768,
        register_cap: 128,
        occupancy_gate: 3,
    },
    Sm89HalfKernelSpec {
        route: Sm89HalfRoute::TnM64N64Bk64S2RegpipeVec2,
        op: ResolvedGemmOp::Tn,
        dtype: WeightDtype::F16,
        symbol: super::sm89_half_tn_source::REGPIPE_VEC2_F16_SYMBOL,
        tile: (64, 64),
        bk: 64,
        stages: 2,
        threads: 128,
        dynamic_shared_bytes: 0,
        static_shared_bytes: 32_768,
        register_cap: 128,
        occupancy_gate: 3,
    },
    Sm89HalfKernelSpec {
        route: Sm89HalfRoute::TnM64N64Bk64S2RegpipeVec2,
        op: ResolvedGemmOp::Tn,
        dtype: WeightDtype::Bf16,
        symbol: super::sm89_half_tn_source::REGPIPE_VEC2_BF16_SYMBOL,
        tile: (64, 64),
        bk: 64,
        stages: 2,
        threads: 128,
        dynamic_shared_bytes: 0,
        static_shared_bytes: 32_768,
        register_cap: 128,
        occupancy_gate: 3,
    },
    Sm89HalfKernelSpec {
        route: Sm89HalfRoute::NtM128N128Bk64S3Bxor,
        op: ResolvedGemmOp::Nt,
        dtype: WeightDtype::F16,
        symbol: "nt_sm89_m128n128_bk64_s3_bxor_f16",
        tile: (128, 128),
        bk: 64,
        stages: 3,
        threads: 256,
        dynamic_shared_bytes: 98_304,
        static_shared_bytes: 0,
        register_cap: 168,
        occupancy_gate: 1,
    },
    Sm89HalfKernelSpec {
        route: Sm89HalfRoute::NtM128N128Bk64S3Bxor,
        op: ResolvedGemmOp::Nt,
        dtype: WeightDtype::Bf16,
        symbol: "nt_sm89_m128n128_bk64_s3_bxor_bf16",
        tile: (128, 128),
        bk: 64,
        stages: 3,
        threads: 256,
        dynamic_shared_bytes: 98_304,
        static_shared_bytes: 0,
        register_cap: 168,
        occupancy_gate: 1,
    },
    Sm89HalfKernelSpec {
        route: Sm89HalfRoute::NtM96N128Bk64S3,
        op: ResolvedGemmOp::Nt,
        dtype: WeightDtype::F16,
        symbol: "nt_sm89_m96n128_bk64_s3_f16",
        tile: (96, 128),
        bk: 64,
        stages: 3,
        threads: 384,
        dynamic_shared_bytes: 86_016,
        static_shared_bytes: 0,
        register_cap: 120,
        occupancy_gate: 1,
    },
    Sm89HalfKernelSpec {
        route: Sm89HalfRoute::NtM96N128Bk64S3,
        op: ResolvedGemmOp::Nt,
        dtype: WeightDtype::Bf16,
        symbol: "nt_sm89_m96n128_bk64_s3_bf16",
        tile: (96, 128),
        bk: 64,
        stages: 3,
        threads: 384,
        dynamic_shared_bytes: 86_016,
        static_shared_bytes: 0,
        register_cap: 120,
        occupancy_gate: 1,
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum Sm89HalfRuntimeRoute {
    Legacy(Sm89HalfRoute),
    TnSmall16Bk64S2Ldb72,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Sm89HalfRuntimeSpec {
    pub(super) route: Sm89HalfRuntimeRoute,
    pub(super) op: ResolvedGemmOp,
    pub(super) dtype: WeightDtype,
    pub(super) symbol: &'static str,
    pub(super) tile: (u32, u32),
    pub(super) bk: u32,
    pub(super) stages: u8,
    pub(super) threads: u32,
    pub(super) dynamic_shared_bytes: u32,
    pub(super) static_shared_bytes: u32,
    pub(super) register_cap: u32,
    pub(super) occupancy_gate: u32,
}

const SM89_HALF_SMALL16_RUNTIME_SPECS: [Sm89HalfRuntimeSpec; 2] = [
    Sm89HalfRuntimeSpec {
        route: Sm89HalfRuntimeRoute::TnSmall16Bk64S2Ldb72,
        op: ResolvedGemmOp::Tn,
        dtype: WeightDtype::F16,
        symbol: super::sm89_half_tn_source::SMALL16_F16_SYMBOL,
        tile: (16, 16),
        bk: 64,
        stages: 2,
        threads: 32,
        dynamic_shared_bytes: 0,
        static_shared_bytes: 36_864,
        register_cap: 78,
        occupancy_gate: 2,
    },
    Sm89HalfRuntimeSpec {
        route: Sm89HalfRuntimeRoute::TnSmall16Bk64S2Ldb72,
        op: ResolvedGemmOp::Tn,
        dtype: WeightDtype::Bf16,
        symbol: super::sm89_half_tn_source::SMALL16_BF16_SYMBOL,
        tile: (16, 16),
        bk: 64,
        stages: 2,
        threads: 32,
        dynamic_shared_bytes: 0,
        static_shared_bytes: 36_864,
        register_cap: 78,
        occupancy_gate: 2,
    },
];

impl From<Sm89HalfKernelSpec> for Sm89HalfRuntimeSpec {
    fn from(spec: Sm89HalfKernelSpec) -> Self {
        Self {
            route: Sm89HalfRuntimeRoute::Legacy(spec.route),
            op: spec.op,
            dtype: spec.dtype,
            symbol: spec.symbol,
            tile: spec.tile,
            bk: spec.bk,
            stages: spec.stages,
            threads: spec.threads,
            dynamic_shared_bytes: spec.dynamic_shared_bytes,
            static_shared_bytes: spec.static_shared_bytes,
            register_cap: spec.register_cap,
            occupancy_gate: spec.occupancy_gate,
        }
    }
}

pub(super) fn runtime_kernel_specs() -> impl Iterator<Item = Sm89HalfRuntimeSpec> {
    SM89_HALF_KERNEL_SPECS
        .into_iter()
        .map(Into::into)
        .chain(SM89_HALF_SMALL16_RUNTIME_SPECS)
}

pub(super) fn runtime_kernel_spec(symbol: &str) -> Option<Sm89HalfRuntimeSpec> {
    runtime_kernel_specs().find(|spec| spec.symbol == symbol)
}

type Sm89HalfRuntimeAutoCell = (
    ResolvedGemmOp,
    WeightDtype,
    (usize, usize, usize),
    Sm89HalfRuntimeRoute,
);

const SM89_HALF_RUNTIME_AUTO_CELLS: &[Sm89HalfRuntimeAutoCell] = &[
    (
        ResolvedGemmOp::Tn,
        WeightDtype::F16,
        (1024, 256, 128),
        Sm89HalfRuntimeRoute::TnSmall16Bk64S2Ldb72,
    ),
    (
        ResolvedGemmOp::Tn,
        WeightDtype::Bf16,
        (1024, 256, 128),
        Sm89HalfRuntimeRoute::TnSmall16Bk64S2Ldb72,
    ),
];

/// One automatically served Ada half cell: op, dtype, (m, k, n) and the
/// route the frozen discovery ledger proved for it.
pub type Sm89HalfAutoCell = (
    ResolvedGemmOp,
    WeightDtype,
    (usize, usize, usize),
    Sm89HalfRoute,
);

/// The Ada half cells the automatic selector serves, proven by the
/// frozen discovery ledger.
pub const SM89_HALF_AUTO_CELLS: &[Sm89HalfAutoCell] = &[
    (
        ResolvedGemmOp::Nn,
        WeightDtype::F16,
        (2048, 768, 3072),
        Sm89HalfRoute::NnM128N128Bk64S3,
    ),
    (
        ResolvedGemmOp::Nn,
        WeightDtype::Bf16,
        (2048, 768, 3072),
        Sm89HalfRoute::NnM128N128Bk64S3,
    ),
    (
        ResolvedGemmOp::Nn,
        WeightDtype::F16,
        (2048, 1536, 768),
        Sm89HalfRoute::NnM128N128Bk64S3,
    ),
    (
        ResolvedGemmOp::Nn,
        WeightDtype::Bf16,
        (2048, 1536, 768),
        Sm89HalfRoute::NnM128N128Bk64S3,
    ),
    (
        ResolvedGemmOp::Nn,
        WeightDtype::F16,
        (4621, 384, 1928),
        Sm89HalfRoute::NnM128N128Bk64S3,
    ),
    (
        ResolvedGemmOp::Nn,
        WeightDtype::Bf16,
        (4621, 384, 1928),
        Sm89HalfRoute::NnM128N128Bk64S3,
    ),
    (
        ResolvedGemmOp::Tn,
        WeightDtype::F16,
        (2048, 768, 3072),
        Sm89HalfRoute::TnM64N64Bk64S2RegpipeVec2,
    ),
    (
        ResolvedGemmOp::Tn,
        WeightDtype::Bf16,
        (2048, 768, 3072),
        Sm89HalfRoute::TnM64N64Bk64S2RegpipeVec2,
    ),
    (
        ResolvedGemmOp::Tn,
        WeightDtype::F16,
        (2048, 1536, 768),
        Sm89HalfRoute::TnM64N64Bk64S2CompactBxor,
    ),
    (
        ResolvedGemmOp::Tn,
        WeightDtype::Bf16,
        (2048, 1536, 768),
        Sm89HalfRoute::TnM64N64Bk64S2RegpipeVec2,
    ),
    (
        ResolvedGemmOp::Tn,
        WeightDtype::F16,
        (4621, 384, 1928),
        Sm89HalfRoute::TnM64N64Bk64S2CompactBxor,
    ),
    (
        ResolvedGemmOp::Tn,
        WeightDtype::Bf16,
        (4621, 384, 1928),
        Sm89HalfRoute::TnM64N64Bk64S2CompactBxor,
    ),
    (
        ResolvedGemmOp::Nt,
        WeightDtype::F16,
        (2048, 768, 3072),
        Sm89HalfRoute::NtM128N128Bk64S3Bxor,
    ),
    (
        ResolvedGemmOp::Nt,
        WeightDtype::Bf16,
        (2048, 768, 3072),
        Sm89HalfRoute::NtM128N128Bk64S3Bxor,
    ),
    (
        ResolvedGemmOp::Nt,
        WeightDtype::F16,
        (4621, 384, 1928),
        Sm89HalfRoute::NtM128N128Bk64S3Bxor,
    ),
    (
        ResolvedGemmOp::Nt,
        WeightDtype::Bf16,
        (4621, 384, 1928),
        Sm89HalfRoute::NtM128N128Bk64S3Bxor,
    ),
    (
        ResolvedGemmOp::Nt,
        WeightDtype::F16,
        (2048, 1536, 768),
        Sm89HalfRoute::NtM96N128Bk64S3,
    ),
    (
        ResolvedGemmOp::Nt,
        WeightDtype::Bf16,
        (2048, 1536, 768),
        Sm89HalfRoute::NtM96N128Bk64S3,
    ),
];

#[derive(Clone, Copy)]
pub(crate) struct Sm89HalfAutoContext {
    pub(crate) compiler: Option<CompilerIdentity>,
    pub(crate) artifact: Option<ArtifactIdentity>,
    pub(crate) compute_capability: (u32, u32),
    pub(crate) multiprocessor_count: u32,
}

#[derive(Clone, Copy)]
pub(crate) struct Sm89HalfAutoRequest {
    pub(crate) request: F32TriadRequest,
    pub(crate) operands: F32TriadOperands,
    pub(crate) dtype: WeightDtype,
}

#[derive(Clone, Copy)]
struct Sm89HalfAutoIdentity {
    nvrtc_version: (i32, i32),
    compile_key: [u8; 32],
    artifact_digest: [u8; 32],
    source_digest: [u8; 32],
    header_manifest_digest: [u8; 32],
    nvrtc_library_domain: [u8; 32],
}

impl Sm89HalfAutoIdentity {
    fn matches(self, compiler: CompilerIdentity, artifact: ArtifactIdentity) -> bool {
        compiler.nvrtc_version == self.nvrtc_version
            && compiler.source_digest == self.source_digest
            && compiler.invocation_digest == self.compile_key
            && compiler.header_manifest_digest == self.header_manifest_digest
            && compiler.nvrtc_library_domain == self.nvrtc_library_domain
            && artifact.compile_key == self.compile_key
            && artifact.artifact_digest == self.artifact_digest
    }
}

/// Complete-module identities from the initial six-cohort census. Each
/// toolkit produced the same identity at cap16 and cap64; raw-word admission
/// remains a separate release gate.
const SM89_HALF_AUTO_IDENTITIES: [Sm89HalfAutoIdentity; 3] = [
    Sm89HalfAutoIdentity {
        nvrtc_version: (12, 8),
        compile_key: [
            53, 88, 65, 230, 45, 14, 201, 183, 247, 226, 214, 102, 144, 121, 155, 158, 171, 74,
            215, 231, 20, 233, 129, 192, 56, 63, 212, 253, 130, 161, 215, 223,
        ],
        artifact_digest: [
            105, 101, 89, 217, 145, 231, 253, 49, 68, 241, 63, 238, 53, 183, 251, 140, 203, 212,
            135, 137, 147, 113, 57, 50, 48, 6, 116, 137, 82, 190, 211, 123,
        ],
        source_digest: [
            67, 118, 205, 134, 184, 14, 106, 71, 66, 38, 133, 164, 196, 63, 122, 71, 94, 46, 103,
            161, 25, 15, 186, 238, 157, 109, 231, 110, 68, 194, 89, 107,
        ],
        header_manifest_digest: [
            22, 87, 107, 243, 151, 28, 253, 49, 242, 12, 194, 42, 91, 0, 77, 36, 127, 52, 18, 137,
            4, 139, 202, 102, 4, 163, 21, 132, 25, 25, 209, 30,
        ],
        nvrtc_library_domain: [
            38, 176, 163, 160, 32, 68, 255, 203, 193, 105, 63, 216, 62, 146, 97, 190, 255, 166,
            146, 164, 251, 207, 227, 172, 94, 157, 140, 135, 152, 11, 177, 85,
        ],
    },
    Sm89HalfAutoIdentity {
        nvrtc_version: (13, 0),
        compile_key: [
            65, 65, 203, 117, 136, 145, 250, 24, 36, 71, 155, 185, 173, 182, 193, 13, 226, 32, 12,
            56, 26, 29, 37, 231, 223, 124, 129, 142, 182, 183, 227, 116,
        ],
        artifact_digest: [
            103, 139, 45, 1, 147, 23, 36, 71, 36, 139, 182, 27, 193, 230, 132, 93, 75, 43, 85, 5,
            173, 112, 30, 188, 224, 87, 251, 56, 133, 10, 247, 127,
        ],
        source_digest: [
            67, 118, 205, 134, 184, 14, 106, 71, 66, 38, 133, 164, 196, 63, 122, 71, 94, 46, 103,
            161, 25, 15, 186, 238, 157, 109, 231, 110, 68, 194, 89, 107,
        ],
        header_manifest_digest: [
            242, 235, 68, 21, 160, 216, 233, 162, 18, 142, 9, 52, 72, 241, 130, 135, 159, 123, 44,
            26, 14, 135, 237, 109, 93, 33, 180, 4, 11, 182, 177, 226,
        ],
        nvrtc_library_domain: [
            112, 155, 145, 195, 107, 251, 14, 217, 102, 238, 105, 173, 200, 214, 248, 127, 241, 16,
            238, 207, 61, 251, 80, 96, 54, 127, 24, 60, 230, 20, 235, 13,
        ],
    },
    Sm89HalfAutoIdentity {
        nvrtc_version: (13, 2),
        compile_key: [
            206, 165, 241, 150, 102, 4, 108, 38, 147, 144, 42, 227, 187, 156, 10, 246, 252, 102,
            71, 126, 117, 23, 202, 241, 78, 142, 200, 27, 199, 21, 109, 173,
        ],
        artifact_digest: [
            46, 14, 19, 187, 72, 120, 118, 238, 197, 130, 145, 21, 142, 84, 215, 9, 116, 224, 25,
            29, 152, 19, 97, 51, 65, 21, 9, 204, 206, 152, 222, 82,
        ],
        source_digest: [
            67, 118, 205, 134, 184, 14, 106, 71, 66, 38, 133, 164, 196, 63, 122, 71, 94, 46, 103,
            161, 25, 15, 186, 238, 157, 109, 231, 110, 68, 194, 89, 107,
        ],
        header_manifest_digest: [
            102, 251, 184, 239, 126, 11, 177, 44, 8, 222, 75, 16, 227, 229, 86, 208, 35, 131, 183,
            95, 193, 160, 100, 5, 18, 117, 67, 208, 14, 27, 201, 237,
        ],
        nvrtc_library_domain: [
            208, 49, 165, 62, 185, 114, 53, 183, 15, 98, 246, 82, 147, 45, 177, 189, 247, 40, 234,
            34, 156, 140, 168, 9, 213, 60, 95, 253, 145, 100, 38, 135,
        ],
    },
];

pub(crate) fn select_sm89_half_auto_cell(
    context: Sm89HalfAutoContext,
    request: Sm89HalfAutoRequest,
) -> Option<Sm89HalfRuntimeSpec> {
    let compiler = context.compiler?;
    let artifact = context.artifact?;
    if context.compute_capability != (8, 9)
        || context.multiprocessor_count != 142
        || artifact.module_kind != ModuleKind::TriadSm89Half
        || artifact.artifact_kind != ArtifactKind::Ptx
        || compiler.target.as_str() != "sm_89"
        || !compiler.nvrtc_library_known
        || compiler.output_kind != ArtifactKind::Ptx
        || compiler.composer_revision != COMPOSER_REVISION
        || compiler.compiler_revision != COMPILER_REVISION
        || compiler.numeric_abi_revision != NUMERIC_ABI_REVISION
        || compiler.schedule_revision != SCHEDULE_REVISION
    {
        return None;
    }
    SM89_HALF_AUTO_IDENTITIES
        .iter()
        .find(|identity| identity.matches(compiler, artifact))?;
    let dims = (
        request.request.shape.m,
        request.request.shape.k,
        request.request.shape.n,
    );
    if request.request.shape != F32TriadShape::contiguous(request.request.op, dims)
        || request.operands.alpha.to_bits() != 1.0_f32.to_bits()
        || request.operands.beta.to_bits()
            != if request.request.op == ResolvedGemmOp::Tn {
                1.0_f32.to_bits()
            } else {
                0.0_f32.to_bits()
            }
        || [
            request.operands.output,
            request.operands.a,
            request.operands.b,
        ]
        .into_iter()
        .any(|pointer| pointer == 0 || !pointer.is_multiple_of(16))
        || match request.request.op {
            ResolvedGemmOp::Nn => request
                .operands
                .bias
                .is_some_and(|pointer| pointer == 0 || !pointer.is_multiple_of(4)),
            ResolvedGemmOp::Tn | ResolvedGemmOp::Nt => request.operands.bias.is_some(),
        }
    {
        return None;
    }
    let legacy_route = SM89_HALF_AUTO_CELLS
        .iter()
        .find(|&&(op, dtype, shape, _)| {
            op == request.request.op && dtype == request.dtype && shape == dims
        })
        .map(|&(_, _, _, route)| Sm89HalfRuntimeRoute::Legacy(route));
    let runtime_route = SM89_HALF_RUNTIME_AUTO_CELLS
        .iter()
        .find(|&&(op, dtype, shape, _)| {
            op == request.request.op && dtype == request.dtype && shape == dims
        })
        .map(|&(_, _, _, route)| route);
    let route = legacy_route.or(runtime_route)?;
    runtime_kernel_specs().find(|spec| spec.route == route && spec.dtype == request.dtype)
}

pub(super) fn kernel_spec(
    route: Sm89HalfRoute,
    dtype: WeightDtype,
) -> Option<&'static Sm89HalfKernelSpec> {
    SM89_HALF_KERNEL_SPECS
        .iter()
        .find(|spec| spec.route == route && spec.dtype == dtype)
}

pub(super) fn compose_sm89_half_source() -> Result<String, String> {
    super::sm89_half_tn_source::validate_source()?;
    let tn_fragment = super::sm89_half_tn_source::compose_fragment_for_sm89_half()?;
    let source = format!("{}\n{}", BASE_SOURCE.trim_end(), tn_fragment);
    for spec in runtime_kernel_specs() {
        let represented_once = match spec.route {
            Sm89HalfRuntimeRoute::Legacy(Sm89HalfRoute::NtM128N128Bk64S3Bxor) => {
                source
                    .matches("nt_sm89_m128n128_bk64_s3_bxor_##SUFFIX")
                    .count()
                    == 1
                    && source
                        .matches(match spec.dtype {
                            WeightDtype::F16 => "SM89_HALF_NT_S3_EXPORT(__half, f16)",
                            WeightDtype::Bf16 => "SM89_HALF_NT_S3_EXPORT(__nv_bfloat16, bf16)",
                            WeightDtype::F32 | WeightDtype::Tf32 => {
                                return Err("TriadSm89Half owns no f32 export".into());
                            }
                        })
                        .count()
                        == 1
            }
            Sm89HalfRuntimeRoute::Legacy(
                Sm89HalfRoute::TnM64N64Bk64S2CompactBxor | Sm89HalfRoute::TnM64N64Bk64S2RegpipeVec2,
            )
            | Sm89HalfRuntimeRoute::TnSmall16Bk64S2Ldb72 => {
                super::sm89_half_tn_source::kernel_spec(spec.symbol)
                    .and_then(|_| super::sm89_half_tn_source::family_source(&source, spec.symbol))
                    .is_some()
            }
            _ => source.matches(spec.symbol).count() == 1,
        };
        if !represented_once {
            return Err(format!(
                "TriadSm89Half source must export {} exactly once",
                spec.symbol
            ));
        }
    }
    for forbidden in ["_test_", "nn_sm89_tc128_s3_", "nn_sm89_tc128_swizzle_"] {
        if source.contains(forbidden) {
            return Err(format!(
                "TriadSm89Half source retained non-production marker {forbidden}"
            ));
        }
    }
    Ok(source)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mamba_ssm::gpu::kernel_identity::{
        ArtifactKind, COMPILER_REVISION, COMPOSER_REVISION, CudaTarget, ModuleKind,
        NUMERIC_ABI_REVISION, SCHEDULE_REVISION,
    };

    fn digest(hex: &str) -> [u8; 32] {
        assert_eq!(hex.len(), 64);
        std::array::from_fn(|index| u8::from_str_radix(&hex[index * 2..index * 2 + 2], 16).unwrap())
    }

    fn measured_auto_identities() -> [Sm89HalfAutoIdentity; 3] {
        [
            Sm89HalfAutoIdentity {
                nvrtc_version: (12, 8),
                compile_key: digest(
                    "355841e62d0ec9b7f7e2d66690799b9eab4ad7e714e981c0383fd4fd82a1d7df",
                ),
                artifact_digest: digest(
                    "696559d991e7fd3144f13fee35b7fb8ccbd48789937139323006748952bed37b",
                ),
                source_digest: digest(
                    "4376cd86b80e6a47422685a4c43f7a475e2e67a1190fbaee9d6de76e44c2596b",
                ),
                header_manifest_digest: digest(
                    "16576bf3971cfd31f20cc22a5b004d247f341289048bca6604a315841919d11e",
                ),
                nvrtc_library_domain: digest(
                    "26b0a3a02044ffcbc1693fd83e9261beffa692a4fbcfe3ac5e9d8c87980bb155",
                ),
            },
            Sm89HalfAutoIdentity {
                nvrtc_version: (13, 0),
                compile_key: digest(
                    "4141cb758891fa1824479bb9adb6c10de2200c381a1d25e7df7c818eb6b7e374",
                ),
                artifact_digest: digest(
                    "678b2d0193172447248bb61bc1e6845d4b2b5505ad701ebce057fb38850af77f",
                ),
                source_digest: digest(
                    "4376cd86b80e6a47422685a4c43f7a475e2e67a1190fbaee9d6de76e44c2596b",
                ),
                header_manifest_digest: digest(
                    "f2eb4415a0d8e9a2128e093448f182879f7b2c1a0e87ed6d5d21b4040bb6b1e2",
                ),
                nvrtc_library_domain: digest(
                    "709b91c36bfb0ed966ee69adc8d6f87ff110eecf3dfb5060367f183ce614eb0d",
                ),
            },
            Sm89HalfAutoIdentity {
                nvrtc_version: (13, 2),
                compile_key: digest(
                    "cea5f19666046c2693902ae3bb9c0af6fc66477e7517caf14e8ec81bc7156dad",
                ),
                artifact_digest: digest(
                    "2e0e13bb487876eec58291158e54d70974e0191d98136133411509ccce98de52",
                ),
                source_digest: digest(
                    "4376cd86b80e6a47422685a4c43f7a475e2e67a1190fbaee9d6de76e44c2596b",
                ),
                header_manifest_digest: digest(
                    "66fbb8ef7e0bb12c08de4b10e3e556d02383b75fc1a06405127543d00e1bc9ed",
                ),
                nvrtc_library_domain: digest(
                    "d031a53eb97235b70f62f652932db1bdf728ea229c8ca809d53c5ffd91642687",
                ),
            },
        ]
    }

    fn auto_context_for(identity: Sm89HalfAutoIdentity) -> Sm89HalfAutoContext {
        Sm89HalfAutoContext {
            compiler: Some(CompilerIdentity {
                source_digest: identity.source_digest,
                invocation_digest: identity.compile_key,
                header_manifest_digest: identity.header_manifest_digest,
                target: CudaTarget::new("sm_89").unwrap(),
                nvrtc_version: identity.nvrtc_version,
                nvrtc_library_domain: identity.nvrtc_library_domain,
                nvrtc_library_known: true,
                output_kind: ArtifactKind::Ptx,
                composer_revision: COMPOSER_REVISION,
                compiler_revision: COMPILER_REVISION,
                numeric_abi_revision: NUMERIC_ABI_REVISION,
                schedule_revision: SCHEDULE_REVISION,
            }),
            artifact: Some(ArtifactIdentity {
                module_kind: ModuleKind::TriadSm89Half,
                artifact_kind: ArtifactKind::Ptx,
                compile_key: identity.compile_key,
                artifact_digest: identity.artifact_digest,
            }),
            compute_capability: (8, 9),
            multiprocessor_count: 142,
        }
    }

    fn auto_context(nvrtc_version: (i32, i32)) -> Sm89HalfAutoContext {
        let identity = SM89_HALF_AUTO_IDENTITIES
            .iter()
            .find(|identity| identity.nvrtc_version == nvrtc_version)
            .copied()
            .unwrap_or(Sm89HalfAutoIdentity {
                nvrtc_version,
                compile_key: [5; 32],
                artifact_digest: [6; 32],
                source_digest: [1; 32],
                header_manifest_digest: [3; 32],
                nvrtc_library_domain: [4; 32],
            });
        Sm89HalfAutoContext {
            compiler: Some(CompilerIdentity {
                source_digest: identity.source_digest,
                invocation_digest: identity.compile_key,
                header_manifest_digest: identity.header_manifest_digest,
                target: CudaTarget::new("sm_89").unwrap(),
                nvrtc_version,
                nvrtc_library_domain: identity.nvrtc_library_domain,
                nvrtc_library_known: true,
                output_kind: ArtifactKind::Ptx,
                composer_revision: COMPOSER_REVISION,
                compiler_revision: COMPILER_REVISION,
                numeric_abi_revision: NUMERIC_ABI_REVISION,
                schedule_revision: SCHEDULE_REVISION,
            }),
            artifact: Some(ArtifactIdentity {
                module_kind: ModuleKind::TriadSm89Half,
                artifact_kind: ArtifactKind::Ptx,
                compile_key: identity.compile_key,
                artifact_digest: identity.artifact_digest,
            }),
            compute_capability: (8, 9),
            multiprocessor_count: 142,
        }
    }

    #[test]
    fn triad_retained_identity_half_bindings_match_all_three_measured_modules() {
        let expected = measured_auto_identities();
        let request = auto_request(ResolvedGemmOp::Tn, WeightDtype::F16, (1024, 256, 128));
        for identity in expected {
            let context = auto_context_for(identity);
            assert!(
                select_sm89_half_auto_cell(context, request).is_some(),
                "measured CUDA {:?} half module must enter AUTO",
                identity.nvrtc_version
            );

            let mut changed = context;
            changed.compiler.as_mut().unwrap().nvrtc_version.1 += 1;
            assert_declined(changed, request, "measured toolkit");
            let mut changed = context;
            changed.compiler.as_mut().unwrap().source_digest[0] ^= 1;
            assert_declined(changed, request, "measured source");
            let mut changed = context;
            changed.artifact.as_mut().unwrap().compile_key[0] ^= 1;
            assert_declined(changed, request, "measured compile key");
            let mut changed = context;
            changed.artifact.as_mut().unwrap().artifact_digest[0] ^= 1;
            assert_declined(changed, request, "measured artifact");
        }
        for (actual, expected) in SM89_HALF_AUTO_IDENTITIES.iter().zip(expected) {
            assert_eq!(actual.nvrtc_version, expected.nvrtc_version);
            assert_eq!(actual.compile_key, expected.compile_key);
            assert_eq!(actual.artifact_digest, expected.artifact_digest);
            assert_eq!(actual.source_digest, expected.source_digest);
            assert_eq!(
                actual.header_manifest_digest,
                expected.header_manifest_digest
            );
            assert_eq!(actual.nvrtc_library_domain, expected.nvrtc_library_domain);
        }
        assert_eq!(
            SM89_HALF_AUTO_IDENTITIES
                .iter()
                .map(|identity| identity.compile_key)
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            3
        );
    }

    fn auto_request(
        op: ResolvedGemmOp,
        dtype: WeightDtype,
        dims: (usize, usize, usize),
    ) -> Sm89HalfAutoRequest {
        Sm89HalfAutoRequest {
            request: F32TriadRequest {
                op,
                shape: super::super::contract::F32TriadShape::contiguous(op, dims),
            },
            operands: F32TriadOperands {
                output: 0x1000,
                a: 0x2000,
                b: 0x3000,
                bias: (op == ResolvedGemmOp::Nn).then_some(0x4004),
                alpha: 1.0,
                beta: if op == ResolvedGemmOp::Tn { 1.0 } else { 0.0 },
            },
            dtype,
        }
    }

    #[test]
    fn production_source_has_ten_owned_exports_and_five_physical_families() {
        let source = compose_sm89_half_source().unwrap();
        assert_eq!(SM89_HALF_KERNEL_SPECS.len(), 10);
        assert_eq!(
            SM89_HALF_KERNEL_SPECS
                .iter()
                .map(|spec| spec.route)
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            5
        );
        assert!(!source.contains("nn_sm89_tc128_swizzle_"));
    }

    #[test]
    fn triad_retained_half_private_registry_has_twelve_specs_and_twenty_auto_cells() {
        let specs = runtime_kernel_specs().collect::<Vec<_>>();
        assert_eq!(specs.len(), 12);
        assert_eq!(
            specs
                .iter()
                .map(|spec| spec.route)
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            6
        );
        assert_eq!(
            SM89_HALF_AUTO_CELLS.len() + SM89_HALF_RUNTIME_AUTO_CELLS.len(),
            20
        );
        for (symbol, dtype) in [
            (
                super::super::sm89_half_tn_source::SMALL16_BF16_SYMBOL,
                WeightDtype::Bf16,
            ),
            (
                super::super::sm89_half_tn_source::SMALL16_F16_SYMBOL,
                WeightDtype::F16,
            ),
        ] {
            let spec = runtime_kernel_spec(symbol).expect("private small16 runtime spec");
            assert_eq!(spec.route, Sm89HalfRuntimeRoute::TnSmall16Bk64S2Ldb72);
            assert_eq!(spec.op, ResolvedGemmOp::Tn);
            assert_eq!(spec.dtype, dtype);
            assert_eq!(spec.tile, (16, 16));
            assert_eq!(spec.bk, 64);
            assert_eq!(spec.stages, 2);
            assert_eq!(spec.threads, 32);
            assert_eq!(spec.dynamic_shared_bytes, 0);
            assert_eq!(spec.static_shared_bytes, 36_864);
            assert_eq!(spec.register_cap, 78);
            assert_eq!(spec.occupancy_gate, 2);
        }
    }

    #[test]
    fn triad_retained_half_selector_admits_only_the_exact_small16_cells() {
        for (dtype, symbol) in [
            (
                WeightDtype::Bf16,
                super::super::sm89_half_tn_source::SMALL16_BF16_SYMBOL,
            ),
            (
                WeightDtype::F16,
                super::super::sm89_half_tn_source::SMALL16_F16_SYMBOL,
            ),
        ] {
            let selected = select_sm89_half_auto_cell(
                auto_context((13, 2)),
                auto_request(ResolvedGemmOp::Tn, dtype, (1024, 256, 128)),
            )
            .expect("exact retained small16 AUTO cell");
            assert_eq!(selected.symbol, symbol);
        }
    }

    #[test]
    fn triad_retained_half_small16_selector_rejects_contract_drift() {
        let context = auto_context((13, 2));
        let valid = auto_request(ResolvedGemmOp::Tn, WeightDtype::Bf16, (1024, 256, 128));
        assert!(select_sm89_half_auto_cell(context, valid).is_some());

        for field in ["m", "k", "n", "lda", "ldb", "ldc"] {
            let mut changed = valid;
            match field {
                "m" => changed.request.shape.m += 1,
                "k" => changed.request.shape.k += 1,
                "n" => changed.request.shape.n += 1,
                "lda" => changed.request.shape.lda += 1,
                "ldb" => changed.request.shape.ldb += 1,
                "ldc" => changed.request.shape.ldc += 1,
                _ => unreachable!(),
            }
            assert_declined(context, changed, field);
        }
        for field in ["output", "a", "b"] {
            let mut null = valid;
            let mut misaligned = valid;
            match field {
                "output" => {
                    null.operands.output = 0;
                    misaligned.operands.output += 4;
                }
                "a" => {
                    null.operands.a = 0;
                    misaligned.operands.a += 2;
                }
                "b" => {
                    null.operands.b = 0;
                    misaligned.operands.b += 2;
                }
                _ => unreachable!(),
            }
            assert_declined(context, null, field);
            assert_declined(context, misaligned, field);
        }
        let mut changed = valid;
        changed.dtype = WeightDtype::F32;
        assert_declined(context, changed, "dtype");
        let mut changed = valid;
        changed.operands.alpha = 0.5;
        assert_declined(context, changed, "alpha");
        let mut changed = valid;
        changed.operands.beta = 0.0;
        assert_declined(context, changed, "beta");
        let mut changed = valid;
        changed.operands.bias = Some(0x4000);
        assert_declined(context, changed, "bias");

        let mut changed = context;
        changed.compute_capability = (9, 0);
        assert_declined(changed, valid, "compute capability");
        let mut changed = context;
        changed.multiprocessor_count = 141;
        assert_declined(changed, valid, "SM count");
        let mut changed = context;
        changed.compiler = None;
        assert_declined(changed, valid, "compiler holder");
        let mut changed = context;
        changed.artifact = None;
        assert_declined(changed, valid, "artifact holder");
    }

    #[test]
    fn auto_registry_is_exactly_the_eighteen_proven_cells() {
        use ResolvedGemmOp::{Nn, Nt, Tn};
        use Sm89HalfRoute::{
            NnM128N128Bk64S3, NtM96N128Bk64S3, NtM128N128Bk64S3Bxor, TnM64N64Bk64S2CompactBxor,
            TnM64N64Bk64S2RegpipeVec2,
        };
        use WeightDtype::{Bf16, F16};

        assert_eq!(
            SM89_HALF_AUTO_CELLS,
            &[
                (Nn, F16, (2048, 768, 3072), NnM128N128Bk64S3),
                (Nn, Bf16, (2048, 768, 3072), NnM128N128Bk64S3),
                (Nn, F16, (2048, 1536, 768), NnM128N128Bk64S3),
                (Nn, Bf16, (2048, 1536, 768), NnM128N128Bk64S3),
                (Nn, F16, (4621, 384, 1928), NnM128N128Bk64S3),
                (Nn, Bf16, (4621, 384, 1928), NnM128N128Bk64S3),
                (Tn, F16, (2048, 768, 3072), TnM64N64Bk64S2RegpipeVec2),
                (Tn, Bf16, (2048, 768, 3072), TnM64N64Bk64S2RegpipeVec2),
                (Tn, F16, (2048, 1536, 768), TnM64N64Bk64S2CompactBxor),
                (Tn, Bf16, (2048, 1536, 768), TnM64N64Bk64S2RegpipeVec2),
                (Tn, F16, (4621, 384, 1928), TnM64N64Bk64S2CompactBxor),
                (Tn, Bf16, (4621, 384, 1928), TnM64N64Bk64S2CompactBxor),
                (Nt, F16, (2048, 768, 3072), NtM128N128Bk64S3Bxor),
                (Nt, Bf16, (2048, 768, 3072), NtM128N128Bk64S3Bxor),
                (Nt, F16, (4621, 384, 1928), NtM128N128Bk64S3Bxor),
                (Nt, Bf16, (4621, 384, 1928), NtM128N128Bk64S3Bxor),
                (Nt, F16, (2048, 1536, 768), NtM96N128Bk64S3),
                (Nt, Bf16, (2048, 1536, 768), NtM96N128Bk64S3),
            ]
        );
    }

    #[test]
    fn bxor_resource_ceiling_accepts_the_measured_cuda128_register_count() {
        for spec in SM89_HALF_KERNEL_SPECS
            .iter()
            .filter(|spec| spec.route == Sm89HalfRoute::NtM128N128Bk64S3Bxor)
        {
            assert_eq!(spec.register_cap, 168);
        }
    }

    #[test]
    fn selector_admits_all_eighteen_cells_on_each_exact_toolkit_cohort() {
        for nvrtc in [(12, 8), (13, 0), (13, 2)] {
            for &(op, dtype, dims, route) in SM89_HALF_AUTO_CELLS {
                let selected =
                    select_sm89_half_auto_cell(auto_context(nvrtc), auto_request(op, dtype, dims))
                        .unwrap_or_else(|| panic!("{nvrtc:?}/{op:?}/{dtype:?}/{dims:?} declined"));
                assert_eq!(selected.route, Sm89HalfRuntimeRoute::Legacy(route));
                assert_eq!(selected.op, op);
                assert_eq!(selected.dtype, dtype);
            }
        }
    }

    fn assert_declined(context: Sm89HalfAutoContext, request: Sm89HalfAutoRequest, label: &str) {
        assert!(
            select_sm89_half_auto_cell(context, request).is_none(),
            "unqualified {label} entered the SM89 half AUTO cohort"
        );
    }

    #[test]
    fn selector_fails_closed_on_device_toolkit_compiler_and_artifact_drift() {
        let valid_request = auto_request(ResolvedGemmOp::Nn, WeightDtype::Bf16, (2048, 768, 3072));
        let valid = auto_context((13, 2));

        let mut changed = valid;
        changed.compute_capability = (8, 6);
        assert_declined(changed, valid_request, "compute capability");
        let mut changed = valid;
        changed.multiprocessor_count = 141;
        assert_declined(changed, valid_request, "SM count");
        for nvrtc in [(12, 7), (12, 9), (13, 1), (13, 3)] {
            assert_declined(auto_context(nvrtc), valid_request, "NVRTC version");
        }
        let mut changed = valid;
        changed.compiler.as_mut().unwrap().nvrtc_library_known = false;
        assert_declined(changed, valid_request, "unknown NVRTC library");
        let mut changed = valid;
        changed.compiler.as_mut().unwrap().target = CudaTarget::new("compute_89").unwrap();
        assert_declined(changed, valid_request, "compiler target");
        let mut changed = valid;
        changed.compiler.as_mut().unwrap().output_kind = ArtifactKind::Cubin;
        assert_declined(changed, valid_request, "compiler output kind");
        let mut changed = valid;
        changed.compiler.as_mut().unwrap().schedule_revision += 1;
        assert_declined(changed, valid_request, "schedule revision");
        type CompilerMutation = (&'static str, fn(&mut CompilerIdentity));
        let revision_mutations: [CompilerMutation; 3] = [
            ("composer revision", |compiler: &mut CompilerIdentity| {
                compiler.composer_revision += 1
            }),
            ("compiler revision", |compiler: &mut CompilerIdentity| {
                compiler.compiler_revision += 1
            }),
            ("numeric ABI revision", |compiler: &mut CompilerIdentity| {
                compiler.numeric_abi_revision += 1
            }),
        ];
        for (label, mutate) in revision_mutations {
            let mut changed = valid;
            mutate(changed.compiler.as_mut().unwrap());
            assert_declined(changed, valid_request, label);
        }
        let compiler_digest_mutations: [CompilerMutation; 4] = [
            ("source digest", |compiler: &mut CompilerIdentity| {
                compiler.source_digest[0] ^= 1
            }),
            ("invocation digest", |compiler: &mut CompilerIdentity| {
                compiler.invocation_digest[0] ^= 1
            }),
            (
                "header manifest digest",
                |compiler: &mut CompilerIdentity| compiler.header_manifest_digest[0] ^= 1,
            ),
            ("NVRTC library domain", |compiler: &mut CompilerIdentity| {
                compiler.nvrtc_library_domain[0] ^= 1
            }),
        ];
        for (label, mutate) in compiler_digest_mutations {
            let mut changed = valid;
            mutate(changed.compiler.as_mut().unwrap());
            assert_declined(changed, valid_request, label);
        }
        let mut changed = valid;
        changed.artifact.as_mut().unwrap().module_kind = ModuleKind::TriadSm89Finalist;
        assert_declined(changed, valid_request, "artifact module owner");
        let mut changed = valid;
        changed.artifact.as_mut().unwrap().artifact_kind = ArtifactKind::Cubin;
        assert_declined(changed, valid_request, "artifact kind");
        let mut changed = valid;
        changed.artifact.as_mut().unwrap().compile_key[0] ^= 1;
        assert_declined(changed, valid_request, "compile key");
        let mut changed = valid;
        changed.artifact.as_mut().unwrap().artifact_digest[0] ^= 1;
        assert_declined(changed, valid_request, "artifact digest");
        let mut changed = valid;
        changed.compiler = None;
        assert_declined(changed, valid_request, "missing compiler identity");
        let mut changed = valid;
        changed.artifact = None;
        assert_declined(changed, valid_request, "missing artifact identity");
    }

    #[test]
    fn selector_fails_closed_on_request_contract_drift() {
        let context = auto_context((13, 2));
        let valid = auto_request(ResolvedGemmOp::Nt, WeightDtype::F16, (2048, 1536, 768));

        let mut changed = valid;
        changed.request.shape.lda += 1;
        assert_declined(context, changed, "noncanonical lda");
        let mut changed = valid;
        changed.request.shape.ldb += 1;
        assert_declined(context, changed, "noncanonical ldb");
        let mut changed = valid;
        changed.request.shape.ldc += 1;
        assert_declined(context, changed, "noncanonical ldc");
        let mut changed = valid;
        changed.request.shape.m += 1;
        assert_declined(context, changed, "neighbor shape");
        let mut changed = valid;
        changed.request.op = ResolvedGemmOp::Tn;
        assert_declined(context, changed, "TN operation");
        let mut changed = valid;
        changed.dtype = WeightDtype::F32;
        assert_declined(context, changed, "f32 dtype");
        for field in ["output", "a", "b"] {
            let mut changed = valid;
            match field {
                "output" => changed.operands.output = 0,
                "a" => changed.operands.a += 2,
                "b" => changed.operands.b += 4,
                _ => unreachable!(),
            }
            assert_declined(context, changed, field);
        }
        let mut changed = valid;
        changed.operands.alpha = 2.0;
        assert_declined(context, changed, "alpha");
        let mut changed = valid;
        changed.operands.beta = 1.0;
        assert_declined(context, changed, "beta");
        let mut changed = valid;
        changed.operands.bias = Some(0x4000);
        assert_declined(context, changed, "NT bias");

        let mut nn = auto_request(ResolvedGemmOp::Nn, WeightDtype::Bf16, (2048, 768, 3072));
        nn.operands.bias = Some(0);
        assert_declined(context, nn, "null Some bias");
        nn.operands.bias = Some(0x4002);
        assert_declined(context, nn, "misaligned NN bias");

        let mut tn = auto_request(ResolvedGemmOp::Tn, WeightDtype::F16, (2048, 768, 3072));
        tn.operands.beta = 0.0;
        assert_declined(context, tn, "TN beta");
        tn.operands.beta = 1.0;
        tn.operands.bias = Some(0x4000);
        assert_declined(context, tn, "TN bias");
    }
}
