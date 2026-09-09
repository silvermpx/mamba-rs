use crate::mamba_ssm::gpu::{
    dtype::WeightDtype,
    kernel_identity::{
        ArtifactIdentity, ArtifactKind, COMPILER_REVISION, COMPOSER_REVISION, CompilerIdentity,
        ModuleKind, NUMERIC_ABI_REVISION, ResolvedGemmOp, SCHEDULE_REVISION,
    },
};

use super::contract::{F32TriadOperands, F32TriadRequest, F32TriadShape};

const SOURCE: &str = include_str!("../../../../kernels/gemm_bi_triad/sm89_half.cu");

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Sm89HalfRoute {
    NnM128N128Bk64S3,
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
    pub register_cap: u32,
    pub occupancy_gate: u32,
}

const fn spec(
    route: Sm89HalfRoute,
    op: ResolvedGemmOp,
    dtype: WeightDtype,
    symbol: &'static str,
    tile: (u32, u32),
    threads: u32,
    dynamic_shared_bytes: u32,
    register_cap: u32,
) -> Sm89HalfKernelSpec {
    Sm89HalfKernelSpec {
        route,
        op,
        dtype,
        symbol,
        tile,
        bk: 64,
        stages: 3,
        threads,
        dynamic_shared_bytes,
        register_cap,
        occupancy_gate: 1,
    }
}

pub const SM89_HALF_KERNEL_SPECS: [Sm89HalfKernelSpec; 6] = [
    spec(
        Sm89HalfRoute::NnM128N128Bk64S3,
        ResolvedGemmOp::Nn,
        WeightDtype::F16,
        "gemm_bi_nn_sm89_m128n128_bk64_s3_v1_f16",
        (128, 128),
        256,
        98_304,
        188,
    ),
    spec(
        Sm89HalfRoute::NnM128N128Bk64S3,
        ResolvedGemmOp::Nn,
        WeightDtype::Bf16,
        "gemm_bi_nn_sm89_m128n128_bk64_s3_v1_bf16",
        (128, 128),
        256,
        98_304,
        188,
    ),
    spec(
        Sm89HalfRoute::NtM128N128Bk64S3Bxor,
        ResolvedGemmOp::Nt,
        WeightDtype::F16,
        "gemm_bi_nt_sm89_m128n128_bk64_s3_bxor_v1_f16",
        (128, 128),
        256,
        98_304,
        168,
    ),
    spec(
        Sm89HalfRoute::NtM128N128Bk64S3Bxor,
        ResolvedGemmOp::Nt,
        WeightDtype::Bf16,
        "gemm_bi_nt_sm89_m128n128_bk64_s3_bxor_v1_bf16",
        (128, 128),
        256,
        98_304,
        168,
    ),
    spec(
        Sm89HalfRoute::NtM96N128Bk64S3,
        ResolvedGemmOp::Nt,
        WeightDtype::F16,
        "gemm_bi_nt_sm89_m96n128_bk64_s3_v1_f16",
        (96, 128),
        384,
        86_016,
        120,
    ),
    spec(
        Sm89HalfRoute::NtM96N128Bk64S3,
        ResolvedGemmOp::Nt,
        WeightDtype::Bf16,
        "gemm_bi_nt_sm89_m96n128_bk64_s3_v1_bf16",
        (96, 128),
        384,
        86_016,
        120,
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
        || !matches!(compiler.nvrtc_version, (12, 8) | (13, 0) | (13, 2))
        || !compiler.nvrtc_library_known
        || compiler.output_kind != ArtifactKind::Ptx
        || compiler.composer_revision != COMPOSER_REVISION
        || compiler.compiler_revision != COMPILER_REVISION
        || compiler.numeric_abi_revision != NUMERIC_ABI_REVISION
        || compiler.schedule_revision != SCHEDULE_REVISION
    {
        return None;
    }
    let dims = (
        request.request.shape.m,
        request.request.shape.k,
        request.request.shape.n,
    );
    if request.request.shape != F32TriadShape::contiguous(request.request.op, dims)
        || request.operands.alpha.to_bits() != 1.0_f32.to_bits()
        || request.operands.beta.to_bits() != 0.0_f32.to_bits()
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
    for spec in SM89_HALF_KERNEL_SPECS {
        let represented_once = match spec.route {
            Sm89HalfRoute::NtM128N128Bk64S3Bxor => {
                SOURCE
                    .matches("gemm_bi_nt_sm89_m128n128_bk64_s3_bxor_v1_##SUFFIX")
                    .count()
                    == 1
                    && SOURCE
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
            _ => SOURCE.matches(spec.symbol).count() == 1,
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
        if SOURCE.contains(forbidden) {
            return Err(format!(
                "TriadSm89Half source retained non-production marker {forbidden}"
            ));
        }
    }
    Ok(SOURCE.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mamba_ssm::gpu::kernel_identity::{
        ArtifactKind, COMPILER_REVISION, COMPOSER_REVISION, CudaTarget, ModuleKind,
        NUMERIC_ABI_REVISION, SCHEDULE_REVISION,
    };

    fn auto_context(nvrtc_version: (i32, i32)) -> Sm89HalfAutoContext {
        Sm89HalfAutoContext {
            compiler: Some(CompilerIdentity {
                source_digest: [1; 32],
                invocation_digest: [2; 32],
                header_manifest_digest: [3; 32],
                target: CudaTarget::new("sm_89").unwrap(),
                nvrtc_version,
                nvrtc_library_domain: [4; 32],
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
                compile_key: [5; 32],
                artifact_digest: [6; 32],
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
                beta: 0.0,
            },
            dtype,
        }
    }

    #[test]
    fn production_source_has_six_owned_exports_and_three_physical_families() {
        let source = compose_sm89_half_source().unwrap();
        assert_eq!(SM89_HALF_KERNEL_SPECS.len(), 6);
        assert_eq!(
            SM89_HALF_KERNEL_SPECS
                .iter()
                .map(|spec| spec.route)
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            3
        );
        assert!(!source.contains("gemm_bi_nn_fixed_sm89_tc128_swizzle_v1_"));
    }

    #[test]
    fn auto_registry_is_exactly_the_eleven_proven_cells() {
        use ResolvedGemmOp::{Nn, Nt};
        use Sm89HalfRoute::{NnM128N128Bk64S3, NtM96N128Bk64S3, NtM128N128Bk64S3Bxor};
        use WeightDtype::{Bf16, F16};

        assert_eq!(
            SM89_HALF_AUTO_CELLS,
            &[
                (Nn, Bf16, (2048, 768, 3072), NnM128N128Bk64S3),
                (Nn, F16, (2048, 1536, 768), NnM128N128Bk64S3),
                (Nn, Bf16, (2048, 1536, 768), NnM128N128Bk64S3),
                (Nn, F16, (4621, 384, 1928), NnM128N128Bk64S3),
                (Nn, Bf16, (4621, 384, 1928), NnM128N128Bk64S3),
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
    fn selector_admits_all_eleven_cells_on_each_exact_toolkit_cohort() {
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
        assert_declined(changed, valid_request, "compiler revision tuple");
        let mut changed = valid;
        changed.artifact.as_mut().unwrap().module_kind = ModuleKind::TriadSm89Finalist;
        assert_declined(changed, valid_request, "artifact module owner");
        let mut changed = valid;
        changed.artifact.as_mut().unwrap().artifact_kind = ArtifactKind::Cubin;
        assert_declined(changed, valid_request, "artifact kind");
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
    }
}
