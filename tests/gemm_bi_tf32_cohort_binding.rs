//! Does the frozen TF32 cohort of this board bind on this stack? The host
//! tests prove the cohorts describe the tree; only a live context proves the
//! board's compiled module matches a frozen identity. This smoke qualifies
//! the automatic TF32 policy on the projection shapes and reports which
//! module served each one; a stack the cohorts do not describe serves them
//! all from the exact families and fails here.

use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, F32TriadPolicy, GpuCtx};
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
    PhysicalQualificationF32Epilogue, PhysicalQualificationRequest, PhysicalQualificationRoute,
    qualify_physical_launch,
};
use mamba_rs::mamba_ssm::gpu::kernel_identity::{ModuleKind, ResolvedGemmOp};

#[test]
#[ignore = "requires a CUDA device whose TF32 cohort is frozen in the tree"]
fn tf32_cohort_binds_on_this_board() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Triad);
    ctx.set_f32_triad_policy(F32TriadPolicy::AllowDeterministicTf32V1);
    let shapes = [
        (2048, 768, 3072),
        (2048, 1536, 768),
        (4621, 384, 1928),
        (1024, 256, 128),
        (1024, 128, 512),
    ];
    let mut tf32_served = 0;
    for dims in shapes {
        let request = PhysicalQualificationRequest::contiguous_f32(
            ResolvedGemmOp::Nn,
            dims,
            PhysicalQualificationRoute::F32Policy(F32TriadPolicy::AllowDeterministicTf32V1),
            PhysicalQualificationF32Epilogue::new(1.0, 0.0, false),
        );
        let qualified = qualify_physical_launch(&ctx, request)
            .unwrap_or_else(|error| panic!("qualify {dims:?}: {error}"));
        let nodes = qualified.evidence().nodes();
        let served = nodes
            .iter()
            .map(|node| format!("{:?}:{}", node.module_kind, node.symbol))
            .collect::<Vec<_>>()
            .join(" ");
        let tf32 = nodes.iter().any(|node| {
            matches!(
                node.module_kind,
                ModuleKind::TriadSm80
                    | ModuleKind::TriadSm90a
                    | ModuleKind::TriadSm100
                    | ModuleKind::TriadSm120
            ) && node.symbol.contains("tf32")
        });
        tf32_served += usize::from(tf32);
        println!(
            "cc={:?} dims={dims:?} tf32={tf32} served={served}",
            device.compute_capability
        );
        if device.compute_capability == (8, 9)
            && device.multiprocessor_count() == 142
            && matches!(
                dims,
                (2048, 768, 3072) | (2048, 1536, 768) | (4621, 384, 1928)
            )
        {
            assert!(
                nodes.iter().any(|node| {
                    node.module_kind == ModuleKind::TriadSm80
                        && node.symbol == "gemm_bi_nn_sm80_mma_tf32_v1_m128n128_bk32_s3"
                }),
                "SM89 requalified wide winner did not serve {dims:?}: {served}",
            );
        }
    }
    assert!(
        tf32_served > 0,
        "no projection shape reached a TF32 route: the board's cohort does not bind on this stack"
    );
}

#[test]
#[ignore = "requires the frozen RTX 6000 Ada TF32 bias cohort"]
fn sm89_tf32_bias_cohort_serves_the_qualified_wide_epilogues() {
    let device = GpuDevice::new(0).expect("CUDA device");
    assert_eq!(device.compute_capability, (8, 9));
    assert_eq!(device.multiprocessor_count(), 142);
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Triad);
    ctx.set_f32_triad_policy(F32TriadPolicy::AllowDeterministicTf32V1);
    for dims in [
        (2048, 768, 3072),
        (2048, 1536, 768),
        (4621, 384, 1928),
        (4096, 3072, 1536),
        (2048, 3072, 768),
    ] {
        let request = PhysicalQualificationRequest::contiguous_f32(
            ResolvedGemmOp::Nn,
            dims,
            PhysicalQualificationRoute::F32Policy(F32TriadPolicy::AllowDeterministicTf32V1),
            PhysicalQualificationF32Epilogue::new(1.0, 0.0, true),
        );
        let launch = qualify_physical_launch(&ctx, request).expect("qualify Ada bias AUTO");
        assert!(
            launch.evidence().nodes().iter().any(|node| {
                node.module_kind == ModuleKind::TriadSm80
                    && node.symbol == "gemm_bi_nn_sm80_mma_tf32_v1_m128n128_bk32_s3"
            }),
            "qualified bias AUTO served the wrong body for {dims:?}"
        );
        println!("Ada bias AUTO {dims:?}: wide M128N128/S3 served");
    }
}
