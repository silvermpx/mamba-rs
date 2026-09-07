#![cfg(feature = "cuda")]

use mamba_rs::gpu::context::F32TriadPolicy;
use mamba_rs::gpu::gemm_bi_triad::{
    F32TriadAvailability, F32TriadRequest, F32TriadSelection, F32TriadShape, Tf32QualifiedModule,
    resolve_f32_triad_auto,
};
use mamba_rs::gpu::kernel_identity::{
    ArtifactIdentity, ArtifactKind, COMPILER_REVISION, COMPOSER_REVISION, CompilerIdentity,
    CudaTarget, DeviceCaps, DeviceIdentity, DriverIdentity, ModuleKind, NUMERIC_ABI_REVISION,
    ResolvedGemmOp, SCHEDULE_REVISION,
};

fn sm89_availability() -> F32TriadAvailability {
    let target = CudaTarget::new("sm_89").unwrap();
    let module_kind = ModuleKind::TriadSm80;
    let nvrtc_version = (13, 2);
    F32TriadAvailability {
        portable: Some(Tf32QualifiedModule {
            module_kind,
            target,
            artifact: ArtifactIdentity {
                module_kind,
                artifact_kind: ArtifactKind::Ptx,
                compile_key: [4; 32],
                artifact_digest: [2; 32],
            },
            compiler: CompilerIdentity {
                source_digest: [3; 32],
                invocation_digest: [4; 32],
                header_manifest_digest: [5; 32],
                target,
                nvrtc_version,
                nvrtc_library_domain: [6; 32],
                nvrtc_library_known: true,
                output_kind: ArtifactKind::Ptx,
                composer_revision: COMPOSER_REVISION,
                compiler_revision: COMPILER_REVISION,
                numeric_abi_revision: NUMERIC_ABI_REVISION,
                schedule_revision: SCHEDULE_REVISION,
            },
            device: DeviceIdentity {
                compute_capability: (8, 9),
                multiprocessor_count: 142,
                target,
                driver: DriverIdentity {
                    api_version: 13_020,
                    build_sources: 1,
                    build_digest: [7; 32],
                },
            },
            device_caps: DeviceCaps {
                compute_capability: (8, 9),
                nvrtc_version,
                accepted_target: Some(target),
                optin_shared_bytes: 99_000,
                tensor_map_access: false,
            },
        }),
        specialized: None,
        finalist: None,
    }
}

#[test]
fn request_only_selector_fails_closed_without_operand_evidence() {
    let cases = [
        (ResolvedGemmOp::Tn, (65, 49, 129)),
        (ResolvedGemmOp::Tn, (49, 65, 129)),
        (ResolvedGemmOp::Tn, (129, 131, 100)),
        (ResolvedGemmOp::Nt, (49, 65, 129)),
        (ResolvedGemmOp::Nt, (65, 49, 129)),
    ];
    for (op, dims) in cases {
        let request = F32TriadRequest {
            op,
            shape: F32TriadShape::contiguous(op, dims),
        };
        assert_eq!(
            resolve_f32_triad_auto(
                F32TriadPolicy::AllowDeterministicTf32V1,
                request,
                sm89_availability(),
            )
            .unwrap(),
            F32TriadSelection::ScalarFmaV1,
            "request-only selector promoted {op:?} {dims:?} without operand evidence",
        );
    }
}
