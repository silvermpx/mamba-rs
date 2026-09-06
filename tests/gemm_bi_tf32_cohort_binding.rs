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
    Tf32PhysicalRoute, Tf32Sm120Route, Tf32Sm120Stages, Tf32Sm120Tile, qualify_physical_launch,
};
use mamba_rs::mamba_ssm::gpu::kernel_identity::{
    ModuleKind, ResolvedGemmOp, ResolvedNumericContract, digest_hex,
};

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

#[test]
#[ignore = "requires the fresh CC12.0/170SM CUDA13.2 driver595.58.03 cohort; run after frozen comparator queue"]
fn sm120_tf32_fresh_595_58_03_auto_symbols_graphs_and_bits() {
    let device = GpuDevice::new(0).expect("CUDA device");
    assert_eq!(device.compute_capability, (12, 0));
    assert_eq!(device.multiprocessor_count(), 170);
    let ctx = GpuCtx::new(&device).expect("fresh cohort context");
    let binding = ctx
        .kernels
        .f32_triad_availability()
        .specialized
        .expect("bound SM120 module");
    assert_eq!(binding.module_kind, ModuleKind::TriadSm120);
    assert_eq!(binding.compiler.nvrtc_version, (13, 2));
    assert!(binding.compiler.nvrtc_library_known);
    assert_eq!(
        digest_hex(&binding.device.driver.build_digest),
        "bbe8397f6ef11a506a502d127d5eb82745a515ab64f9b2901dab4e834bf190d8"
    );
    assert_eq!(
        digest_hex(&binding.artifact.artifact_digest),
        "1cbfd2318610ff5e105eed18ec269453246fef30ca23ac94ac4d3378dfe89246"
    );

    // Independent literal physical expectations, in public M/K/N order.
    let cells = [
        (
            (2048, 768, 3072),
            Tf32Sm120Tile::M64N128,
            (64, 128),
            768,
            256,
            49_280,
            "gemm_bi_nn_sm120_tma_mma_tf32_v1_m64n128_bk32_s2",
        ),
        (
            (2048, 1536, 768),
            Tf32Sm120Tile::M64N64,
            (64, 64),
            384,
            128,
            32_896,
            "gemm_bi_nn_sm120_tma_mma_tf32_v1_m64n64_bk32_s2",
        ),
        (
            (4621, 384, 1928),
            Tf32Sm120Tile::M64N128,
            (64, 128),
            1168,
            256,
            49_280,
            "gemm_bi_nn_sm120_tma_mma_tf32_v1_m64n128_bk32_s2",
        ),
    ];
    for (dims, tile, tile_shape, grid, threads, shared, symbol) in cells {
        let original = ctx.gemm_route();
        let mut auto_words = None;
        for (mode, route) in [
            (
                "auto",
                PhysicalQualificationRoute::F32Policy(F32TriadPolicy::AllowDeterministicTf32V1),
            ),
            (
                "forced",
                PhysicalQualificationRoute::Tf32Forced(Tf32PhysicalRoute::Sm120TmaMmaTf32RnaV1(
                    Tf32Sm120Route {
                        tile,
                        stages: Tf32Sm120Stages::S2,
                    },
                )),
            ),
        ] {
            let request = PhysicalQualificationRequest::contiguous_f32(
                ResolvedGemmOp::Nn,
                dims,
                route,
                PhysicalQualificationF32Epilogue::new(1.0, 0.0, false),
            );
            let mut qualified = qualify_physical_launch(&ctx, request)
                .unwrap_or_else(|error| panic!("{dims:?}/{mode}: {error}"));
            qualified.validate_timed_request(&ctx, request).unwrap();
            let evidence = qualified.evidence();
            assert_eq!(evidence.launch_count(), 1);
            assert!(
                evidence.eager_graph_equal(),
                "actual captured/eager nodes differ"
            );
            assert_eq!(evidence.route_identity().tuning_table_revision, 40);
            let [node] = evidence.nodes() else {
                panic!("one actual GEMM expected");
            };
            assert_eq!(node.symbol, symbol, "{dims:?}/{mode}: wrong actual symbol");
            assert_eq!(node.module_kind, ModuleKind::TriadSm120);
            assert_eq!(node.logical_op, ResolvedGemmOp::Nn);
            assert_eq!(node.shape, dims);
            assert_eq!(node.strides, (dims.1, dims.2, dims.2));
            assert_eq!(node.tile, Some(tile_shape));
            assert_eq!(
                node.numeric_contract,
                Some(ResolvedNumericContract::Sm120TmaMmaTf32RnaV1)
            );
            assert_eq!(node.launch.grid_dim, (grid, 1, 1));
            assert_eq!(node.launch.block_dim, (threads, 1, 1));
            assert_eq!(node.launch.shared_mem_bytes, shared);
            assert_ne!(node.launch.arguments_digest, [0; 32]);

            // Independent full-output dyadic probe detects no-op/operand-role
            // errors. Zero products normalize to +0 after the +0 MMA seed.
            // This is a bounded numeric smoke, not general TF32 accuracy proof.
            for graph in [false, true] {
                qualified.seed_f32_nn_single_term_probe(&ctx).unwrap();
                if graph {
                    qualified.measure_graph_window_ms(&ctx, 1).unwrap();
                } else {
                    qualified.measure_eager_window_ms(&ctx, 1).unwrap();
                }
                let actual = qualified.f32_output_bits(&ctx).unwrap();
                assert_eq!(actual.len(), dims.0 * dims.2);
                for (index, word) in actual.into_iter().enumerate() {
                    let row = index / dims.2;
                    let column = index % dims.2;
                    let a = ((row % 7) as i32 - 3) as f32 * 0.125;
                    let b = ((column % 11) as i32 - 5) as f32 * 0.125;
                    assert_eq!(
                        word,
                        (a * b + 0.0).to_bits(),
                        "{dims:?}/{mode}/graph={graph}/probe[{index}]"
                    );
                }
            }
            // Same full finite corpus and reset before each actual execution:
            // eager/graph/graph/eager bits, then AUTO versus its forced twin.
            let mut repeated = None;
            for graph in [false, true, true, false] {
                qualified.seed_f32_operands(&ctx, 0x1205_32a1).unwrap();
                if graph {
                    qualified.measure_graph_window_ms(&ctx, 1).unwrap();
                } else {
                    qualified.measure_eager_window_ms(&ctx, 1).unwrap();
                }
                let words = qualified.f32_output_bits(&ctx).unwrap();
                assert!(words.iter().all(|word| f32::from_bits(*word).is_finite()));
                if let Some(expected) = &repeated {
                    assert!(
                        expected == &words,
                        "{dims:?}/{mode}: eager/graph/repeat bits"
                    );
                } else {
                    repeated = Some(words);
                }
            }
            if mode == "auto" {
                auto_words = repeated;
            } else {
                assert!(
                    auto_words == repeated,
                    "{dims:?}: AUTO/forced same-symbol bits"
                );
            }
            drop(qualified);
            assert_eq!(ctx.gemm_route(), original, "holder leaked policy");
        }
        // F32 facade buffers are contiguous, not red-zoned. The separate
        // retained full qualification/sanitizers own guard coverage, not this smoke.
        println!(
            "fresh595.58.03 {dims:?}: actual AUTO/forced {symbol}, dyadic numeric and eager/graph/repeat bits passed"
        );
    }
}
