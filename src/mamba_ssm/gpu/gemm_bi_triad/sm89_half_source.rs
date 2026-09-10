use crate::mamba_ssm::gpu::{
    dtype::WeightDtype,
    kernel_identity::{
        ArtifactIdentity, ArtifactKind, COMPILER_REVISION, COMPOSER_REVISION, CompilerIdentity,
        ModuleKind, NUMERIC_ABI_REVISION, ResolvedGemmOp, SCHEDULE_REVISION,
    },
};

use super::contract::{F32TriadOperands, F32TriadRequest, F32TriadShape};

const BASE_SOURCE: &str = include_str!("../../../../kernels/gemm_bi_triad/sm89_half.cu");

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

const fn spec(
    route: Sm89HalfRoute,
    op: ResolvedGemmOp,
    dtype: WeightDtype,
    symbol: &'static str,
    tile: (u32, u32),
    bk: u32,
    stages: u8,
    threads: u32,
    dynamic_shared_bytes: u32,
    static_shared_bytes: u32,
    register_cap: u32,
    occupancy_gate: u32,
) -> Sm89HalfKernelSpec {
    Sm89HalfKernelSpec {
        route,
        op,
        dtype,
        symbol,
        tile,
        bk,
        stages,
        threads,
        dynamic_shared_bytes,
        static_shared_bytes,
        register_cap,
        occupancy_gate,
    }
}

pub const SM89_HALF_KERNEL_SPECS: [Sm89HalfKernelSpec; 10] = [
    spec(
        Sm89HalfRoute::NnM128N128Bk64S3,
        ResolvedGemmOp::Nn,
        WeightDtype::F16,
        "gemm_bi_nn_sm89_m128n128_bk64_s3_v1_f16",
        (128, 128),
        64,
        3,
        256,
        98_304,
        0,
        188,
        1,
    ),
    spec(
        Sm89HalfRoute::NnM128N128Bk64S3,
        ResolvedGemmOp::Nn,
        WeightDtype::Bf16,
        "gemm_bi_nn_sm89_m128n128_bk64_s3_v1_bf16",
        (128, 128),
        64,
        3,
        256,
        98_304,
        0,
        188,
        1,
    ),
    spec(
        Sm89HalfRoute::TnM64N64Bk64S2CompactBxor,
        ResolvedGemmOp::Tn,
        WeightDtype::F16,
        super::sm89_half_tn_source::COMPACT_F16_SYMBOL,
        (64, 64),
        64,
        2,
        128,
        0,
        32_768,
        128,
        3,
    ),
    spec(
        Sm89HalfRoute::TnM64N64Bk64S2CompactBxor,
        ResolvedGemmOp::Tn,
        WeightDtype::Bf16,
        super::sm89_half_tn_source::COMPACT_BF16_SYMBOL,
        (64, 64),
        64,
        2,
        128,
        0,
        32_768,
        128,
        3,
    ),
    spec(
        Sm89HalfRoute::TnM64N64Bk64S2RegpipeVec2,
        ResolvedGemmOp::Tn,
        WeightDtype::F16,
        super::sm89_half_tn_source::REGPIPE_VEC2_F16_SYMBOL,
        (64, 64),
        64,
        2,
        128,
        0,
        32_768,
        128,
        3,
    ),
    spec(
        Sm89HalfRoute::TnM64N64Bk64S2RegpipeVec2,
        ResolvedGemmOp::Tn,
        WeightDtype::Bf16,
        super::sm89_half_tn_source::REGPIPE_VEC2_BF16_SYMBOL,
        (64, 64),
        64,
        2,
        128,
        0,
        32_768,
        128,
        3,
    ),
    spec(
        Sm89HalfRoute::NtM128N128Bk64S3Bxor,
        ResolvedGemmOp::Nt,
        WeightDtype::F16,
        "gemm_bi_nt_sm89_m128n128_bk64_s3_bxor_v1_f16",
        (128, 128),
        64,
        3,
        256,
        98_304,
        0,
        168,
        1,
    ),
    spec(
        Sm89HalfRoute::NtM128N128Bk64S3Bxor,
        ResolvedGemmOp::Nt,
        WeightDtype::Bf16,
        "gemm_bi_nt_sm89_m128n128_bk64_s3_bxor_v1_bf16",
        (128, 128),
        64,
        3,
        256,
        98_304,
        0,
        168,
        1,
    ),
    spec(
        Sm89HalfRoute::NtM96N128Bk64S3,
        ResolvedGemmOp::Nt,
        WeightDtype::F16,
        "gemm_bi_nt_sm89_m96n128_bk64_s3_v1_f16",
        (96, 128),
        64,
        3,
        384,
        86_016,
        0,
        120,
        1,
    ),
    spec(
        Sm89HalfRoute::NtM96N128Bk64S3,
        ResolvedGemmOp::Nt,
        WeightDtype::Bf16,
        "gemm_bi_nt_sm89_m96n128_bk64_s3_v1_bf16",
        (96, 128),
        64,
        3,
        384,
        86_016,
        0,
        120,
        1,
    ),
];

/// Exact Ada half-Triad AUTO cohort proven by the frozen discovery ledger.
pub const SM89_HALF_AUTO_CELLS: &[(
    ResolvedGemmOp,
    WeightDtype,
    (usize, usize, usize),
    Sm89HalfRoute,
)] = &[
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

const SM89_HALF_AUTO_IDENTITIES: [Sm89HalfAutoIdentity; 3] = [
    Sm89HalfAutoIdentity {
        nvrtc_version: (12, 8),
        compile_key: [
            159, 255, 251, 92, 130, 88, 114, 78, 220, 210, 91, 61, 113, 102, 42, 10, 188, 205, 238,
            13, 69, 18, 148, 230, 248, 5, 37, 107, 17, 18, 151, 22,
        ],
        artifact_digest: [
            113, 201, 94, 27, 28, 71, 177, 134, 197, 168, 255, 61, 203, 33, 165, 153, 186, 250,
            103, 165, 215, 117, 157, 82, 9, 167, 7, 213, 235, 17, 68, 240,
        ],
        source_digest: [
            41, 196, 241, 83, 119, 233, 164, 215, 130, 54, 236, 198, 179, 140, 228, 80, 184, 224,
            190, 150, 106, 167, 223, 244, 200, 255, 223, 123, 187, 39, 114, 149,
        ],
        header_manifest_digest: [
            45, 244, 28, 98, 175, 253, 239, 32, 178, 62, 159, 216, 214, 147, 98, 160, 95, 88, 75,
            135, 154, 95, 166, 45, 97, 34, 10, 25, 9, 42, 85, 50,
        ],
        nvrtc_library_domain: [
            38, 176, 163, 160, 32, 68, 255, 203, 193, 105, 63, 216, 62, 146, 97, 190, 255, 166,
            146, 164, 251, 207, 227, 172, 94, 157, 140, 135, 152, 11, 177, 85,
        ],
    },
    Sm89HalfAutoIdentity {
        nvrtc_version: (13, 0),
        compile_key: [
            151, 76, 104, 44, 242, 189, 45, 141, 90, 32, 111, 104, 254, 45, 52, 248, 77, 195, 235,
            16, 122, 175, 206, 26, 58, 246, 136, 50, 255, 105, 12, 145,
        ],
        artifact_digest: [
            250, 208, 145, 82, 4, 21, 203, 3, 103, 180, 215, 157, 79, 66, 169, 249, 60, 10, 197,
            87, 115, 162, 236, 26, 139, 92, 121, 190, 51, 165, 102, 14,
        ],
        source_digest: [
            41, 196, 241, 83, 119, 233, 164, 215, 130, 54, 236, 198, 179, 140, 228, 80, 184, 224,
            190, 150, 106, 167, 223, 244, 200, 255, 223, 123, 187, 39, 114, 149,
        ],
        header_manifest_digest: [
            107, 136, 12, 146, 208, 46, 51, 146, 159, 167, 177, 121, 143, 26, 35, 125, 83, 188,
            103, 81, 155, 94, 51, 218, 179, 57, 9, 122, 32, 88, 64, 4,
        ],
        nvrtc_library_domain: [
            112, 155, 145, 195, 107, 251, 14, 217, 102, 238, 105, 173, 200, 214, 248, 127, 241, 16,
            238, 207, 61, 251, 80, 96, 54, 127, 24, 60, 230, 20, 235, 13,
        ],
    },
    Sm89HalfAutoIdentity {
        nvrtc_version: (13, 2),
        compile_key: [
            108, 96, 214, 146, 176, 119, 90, 234, 54, 91, 150, 127, 12, 181, 26, 112, 33, 35, 130,
            83, 176, 151, 88, 99, 252, 88, 122, 97, 117, 114, 93, 115,
        ],
        artifact_digest: [
            19, 140, 250, 30, 92, 224, 155, 156, 238, 224, 118, 21, 140, 181, 135, 202, 163, 112,
            148, 12, 21, 227, 12, 114, 2, 67, 243, 20, 112, 131, 238, 164,
        ],
        source_digest: [
            41, 196, 241, 83, 119, 233, 164, 215, 130, 54, 236, 198, 179, 140, 228, 80, 184, 224,
            190, 150, 106, 167, 223, 244, 200, 255, 223, 123, 187, 39, 114, 149,
        ],
        header_manifest_digest: [
            251, 115, 54, 248, 169, 136, 243, 129, 157, 45, 255, 254, 87, 253, 18, 66, 240, 69, 80,
            91, 231, 46, 0, 137, 136, 125, 44, 96, 180, 119, 77, 2,
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
) -> Option<&'static Sm89HalfKernelSpec> {
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
    let &(_, _, _, route) = SM89_HALF_AUTO_CELLS
        .iter()
        .find(|&&(op, dtype, shape, _)| {
            op == request.request.op && dtype == request.dtype && shape == dims
        })?;
    kernel_spec(route, request.dtype)
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
    for spec in SM89_HALF_KERNEL_SPECS {
        let represented_once = match spec.route {
            Sm89HalfRoute::NtM128N128Bk64S3Bxor => {
                source
                    .matches("gemm_bi_nt_sm89_m128n128_bk64_s3_bxor_v1_##SUFFIX")
                    .count()
                    == 1
                    && source
                        .matches(match spec.dtype {
                            WeightDtype::F16 => "SM89_HALF_NT_S3_EXPORT(__half, f16)",
                            WeightDtype::Bf16 => "SM89_HALF_NT_S3_EXPORT(__nv_bfloat16, bf16)",
                            WeightDtype::F32 => {
                                return Err("TriadSm89Half owns no f32 export".into());
                            }
                        })
                        .count()
                        == 1
            }
            Sm89HalfRoute::TnM64N64Bk64S2CompactBxor | Sm89HalfRoute::TnM64N64Bk64S2RegpipeVec2 => {
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
    for forbidden in [
        "_test_",
        "gemm_bi_nn_fixed_sm89_tc128_s3_v1_",
        "gemm_bi_nn_fixed_sm89_tc128_swizzle_v1_",
    ] {
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
        assert!(!source.contains("gemm_bi_nn_fixed_sm89_tc128_swizzle_v1_"));
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
                assert_eq!(selected.route, route);
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
        let revision_mutations: [(&str, fn(&mut CompilerIdentity)); 3] = [
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
        let compiler_digest_mutations: [(&str, fn(&mut CompilerIdentity)); 4] = [
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
