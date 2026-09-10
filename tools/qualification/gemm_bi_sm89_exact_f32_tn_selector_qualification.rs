use mamba_rs::mamba_ssm::gpu::GemmMode;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum QualificationPhase {
    ModuleAbiResources,
    RawScratch,
    TargetEager,
    TargetGraph,
    Tail,
    Exceptional,
    NonUnitAlpha,
    K0,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CaseReceipt {
    phases: Vec<QualificationPhase>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScreenOutcome {
    Once3Stop,
    Once7Stop,
    Retained,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct BatchReceipt {
    cases: Vec<CaseReceipt>,
    screen: ScreenOutcome,
    fast_labelled: bool,
}

const REQUIRED_PHASES: [QualificationPhase; 8] = [
    QualificationPhase::ModuleAbiResources,
    QualificationPhase::RawScratch,
    QualificationPhase::TargetEager,
    QualificationPhase::TargetGraph,
    QualificationPhase::Tail,
    QualificationPhase::Exceptional,
    QualificationPhase::NonUnitAlpha,
    QualificationPhase::K0,
];

fn tn_policy_facade_supports_epilogue(alpha: f32, beta: f32) -> bool {
    alpha.to_bits() == 1.0_f32.to_bits() && beta.to_bits() == 1.0_f32.to_bits()
}

fn tn_k0_expected_output_bits(alpha: f32, output: &[f32]) -> Vec<u32> {
    output
        .iter()
        .map(|value| alpha.mul_add(0.0, *value).to_bits())
        .collect()
}

fn validate_batch_receipt(receipt: &BatchReceipt) -> Result<(), String> {
    if receipt.cases.len() != 3 {
        return Err(format!(
            "exact-F32 TN admission requires three cases, got {}",
            receipt.cases.len()
        ));
    }
    for (index, case) in receipt.cases.iter().enumerate() {
        if case.phases != REQUIRED_PHASES {
            return Err(format!(
                "exact-F32 TN case {index} phase order changed: {:?}",
                case.phases
            ));
        }
    }
    if receipt.fast_labelled != matches!(receipt.screen, ScreenOutcome::Retained) {
        return Err("Fast may be labelled only after the once3 and once7 retained gates".into());
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[path = "../../tests/support/fixed_full_mantissa.rs"]
mod full_mantissa;
#[cfg(feature = "cuda")]
#[path = "support/sm89_exact_f32_tn_admission.rs"]
mod raw_probe;

#[cfg(feature = "cuda")]
mod live {
    use super::*;
    use crate::raw_probe::{
        RawCell, raw_seed_words, run_k0_auto_probe, run_raw_probe, run_raw_probe_repeated,
    };
    use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, F32TriadPolicy, GpuCtx};
    use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
    use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
        D768_IN_FUSED_SYMBOL, D768_OUT_RAW_SYMBOL, PRISM_RAW_SYMBOL,
        PhysicalQualificationF32Epilogue, PhysicalQualificationRequest, PhysicalQualificationRoute,
        QualifiedPhysicalLaunch, SM89_EXACT_F32_KERNEL_SPECS, Sm89ExactF32TnRoute,
        qualify_physical_launch,
    };
    use mamba_rs::mamba_ssm::gpu::kernel_identity::{
        ModuleKind, PhysicalLaunchKind, PolicyDtype, ResolvedGemmOp, ResolvedNumericContract,
        ResolvedOutputOwnership,
    };
    use serde_json::json;

    const OPS_PER_WINDOW: usize = 20;
    const PRIOR_SYMBOLS: [&str; 2] = ["gemm_bi_tn_splitm_partial_aligned", "gemm_bi_splitm_reduce"];

    #[derive(Clone, Copy)]
    struct Case {
        route: Sm89ExactF32TnRoute,
        target: RawCell,
        tail: RawCell,
        symbols: [&'static str; 2],
        grids: [(u32, u32, u32); 2],
        blocks: [(u32, u32, u32); 2],
        modules: [ModuleKind; 2],
        tiles: [(u32, u32); 2],
        numeric: [ResolvedNumericContract; 2],
        ownership: [ResolvedOutputOwnership; 2],
    }

    const CASES: [Case; 3] = [
        Case {
            route: Sm89ExactF32TnRoute::D768InDualChunkFused,
            target: RawCell {
                dims: (2_048, 768, 3_072),
                chunks: 2,
                m_chunk: 1_024,
            },
            tail: RawCell {
                dims: (2_047, 68, 132),
                chunks: 2,
                m_chunk: 1_024,
            },
            symbols: ["gemm_bi_transpose_f32_32x16_d768_v1", D768_IN_FUSED_SYMBOL],
            grids: [(24, 64, 1), (576, 1, 1)],
            blocks: [(32, 16, 1), (128, 1, 1)],
            modules: [ModuleKind::TriadScalar, ModuleKind::TriadSm89ExactF32],
            tiles: [(32, 32), (64, 64)],
            numeric: [
                ResolvedNumericContract::ScalarFmaV1,
                ResolvedNumericContract::ScalarFmaTnSplitMF64ReduceV1,
            ],
            ownership: [
                ResolvedOutputOwnership::OneCtaPerOutputTileV1,
                ResolvedOutputOwnership::OneCtaPerOutputTileV1,
            ],
        },
        Case {
            route: Sm89ExactF32TnRoute::D768OutDirectBk16,
            target: RawCell {
                dims: (2_048, 1_536, 768),
                chunks: 4,
                m_chunk: 512,
            },
            tail: RawCell {
                dims: (2_047, 68, 132),
                chunks: 4,
                m_chunk: 512,
            },
            symbols: [D768_OUT_RAW_SYMBOL, "gemm_bi_splitm_reduce"],
            grids: [(288, 1, 4), (4_608, 1, 1)],
            blocks: [(128, 1, 1), (256, 1, 1)],
            modules: [ModuleKind::TriadSm89ExactF32, ModuleKind::TriadScalar],
            tiles: [(64, 64), (1, 1)],
            numeric: [
                ResolvedNumericContract::ScalarFmaTnSplitMPartialV1,
                ResolvedNumericContract::ScalarFmaTnSplitMF64ReduceV1,
            ],
            ownership: [
                ResolvedOutputOwnership::OneCtaPerOutputTilePerSplitMPartitionV1,
                ResolvedOutputOwnership::OneThreadPerOutputElementFixedSplitMReduceV1,
            ],
        },
        Case {
            route: Sm89ExactF32TnRoute::PrismDirectBk16,
            target: RawCell {
                dims: (4_621, 384, 1_928),
                chunks: 6,
                m_chunk: 784,
            },
            tail: RawCell {
                dims: (4_620, 68, 132),
                chunks: 6,
                m_chunk: 784,
            },
            symbols: [PRISM_RAW_SYMBOL, "gemm_bi_splitm_reduce"],
            grids: [(186, 1, 6), (2_892, 1, 1)],
            blocks: [(128, 1, 1), (256, 1, 1)],
            modules: [ModuleKind::TriadSm89ExactF32, ModuleKind::TriadScalar],
            tiles: [(64, 64), (1, 1)],
            numeric: [
                ResolvedNumericContract::ScalarFmaTnSplitMPartialV1,
                ResolvedNumericContract::ScalarFmaTnSplitMF64ReduceV1,
            ],
            ownership: [
                ResolvedOutputOwnership::OneCtaPerOutputTilePerSplitMPartitionV1,
                ResolvedOutputOwnership::OneThreadPerOutputElementFixedSplitMReduceV1,
            ],
        },
    ];

    #[derive(Clone, Copy)]
    enum TimedPath {
        Eager,
        Graph,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum AdmissionMode {
        PreAdmission,
        PostAdmission,
    }

    impl AdmissionMode {
        fn require(self) -> Result<(), String> {
            let expected = match self {
                Self::PreAdmission => "0",
                Self::PostAdmission => "1",
            };
            if std::env::var("MAMBA_SM89_EXACT_F32_EXPECT_AUTO")
                .ok()
                .as_deref()
                != Some(expected)
            {
                return Err(format!(
                    "{self:?} requires MAMBA_SM89_EXACT_F32_EXPECT_AUTO={expected}"
                ));
            }
            Ok(())
        }

        fn label(self) -> &'static str {
            match self {
                Self::PreAdmission => "pre_admission",
                Self::PostAdmission => "post_admission",
            }
        }
    }

    fn runtime() -> Result<GpuCtx, String> {
        let device = GpuDevice::new(0)?;
        let ctx = GpuCtx::new(&device)?;
        let identity = ctx.gemm_route();
        if identity.device.compute_capability != (8, 9)
            || identity.device.multiprocessor_count != 142
        {
            return Err(format!(
                "SM89 exact-F32 TN qualification requires CC8.9/142SM, got {:?}/{}SM",
                identity.device.compute_capability, identity.device.multiprocessor_count
            ));
        }
        ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
        ctx.set_bi_gemm_family(BiGemmFamily::Triad);
        ctx.set_bi_tensor_cores(false);
        ctx.set_f32_triad_policy(F32TriadPolicy::ExactScalarFmaV1);
        Ok(ctx)
    }

    fn request(
        route: PhysicalQualificationRoute,
        dims: (usize, usize, usize),
        alpha: f32,
    ) -> PhysicalQualificationRequest {
        PhysicalQualificationRequest::contiguous_f32(
            ResolvedGemmOp::Tn,
            dims,
            route,
            PhysicalQualificationF32Epilogue::new(alpha, 1.0, false),
        )
    }

    fn node_symbols(launch: &QualifiedPhysicalLaunch<'_>) -> Vec<&'static str> {
        launch
            .evidence()
            .nodes()
            .iter()
            .map(|node| node.symbol)
            .collect()
    }

    fn require_module_abi_resources(ctx: &GpuCtx) -> Result<(), String> {
        let artifact = ctx
            .kernels
            .triad_sm89_exact_f32_artifact_identity()
            .ok_or_else(|| "exact-F32 module has no artifact identity".to_string())?;
        let compiler = ctx
            .kernels
            .triad_sm89_exact_f32_compiler_identity()
            .ok_or_else(|| "exact-F32 module has no compiler identity".to_string())?;
        if artifact.module_kind != ModuleKind::TriadSm89ExactF32
            || compiler.target.as_str() != "sm_89"
            || ctx.kernels.artifact_set_identity().sm89_exact_f32 != Some(artifact)
        {
            return Err("exact-F32 module/compiler/artifact identity changed".into());
        }
        let exclusions = ctx.kernels.triad_sm89_exact_f32_exclusions();
        if !exclusions.is_empty() {
            return Err(format!(
                "exact-F32 ABI/SASS/resource inventory excluded symbols: {exclusions:?}"
            ));
        }
        for spec in SM89_EXACT_F32_KERNEL_SPECS {
            let function = ctx
                .kernels
                .triad_sm89_exact_f32_function(spec.symbol)
                .ok_or_else(|| format!("missing qualified function {}", spec.symbol))?;
            let registers = function
                .num_regs()
                .map_err(|error| format!("{} registers: {error:?}", spec.symbol))?;
            let local = function
                .local_size_bytes()
                .map_err(|error| format!("{} local bytes: {error:?}", spec.symbol))?;
            let shared = function
                .shared_size_bytes()
                .map_err(|error| format!("{} shared bytes: {error:?}", spec.symbol))?;
            let occupancy = function
                .occupancy_max_active_blocks_per_multiprocessor(
                    spec.block.0,
                    spec.dynamic_shared_bytes as usize,
                    None,
                )
                .map_err(|error| format!("{} occupancy: {error:?}", spec.symbol))?;
            if registers < 0
                || registers as u32 > spec.register_cap
                || local != 0
                || shared < 0
                || shared as u32 != spec.static_shared_bytes
                || occupancy < spec.occupancy_gate
            {
                return Err(format!(
                    "{} resource stop: regs={registers}/{} local={local} shared={shared}/{} occupancy={occupancy}/{}",
                    spec.symbol, spec.register_cap, spec.static_shared_bytes, spec.occupancy_gate
                ));
            }
            println!(
                "{}",
                json!({
                    "schema":"MambaTriadSm89ExactF32B2ResourceV1",
                    "symbol":spec.symbol,
                    "abi_parameters":spec.abi_parameter_count,
                    "abi_bytes":spec.abi_parameter_bytes,
                    "registers":registers,
                    "local_bytes":local,
                    "static_shared_bytes":shared,
                    "occupancy":occupancy
                })
            );
        }
        Ok(())
    }

    fn require_large_tn_identity(
        ctx: &GpuCtx,
        launch: &QualifiedPhysicalLaunch<'_>,
        case: Case,
    ) -> Result<(), String> {
        let evidence = launch.evidence();
        if evidence.route_identity() != &ctx.gemm_route()
            || evidence.evidence_scope() != "eager_preflight_same_launcher"
            || !evidence.eager_graph_equal()
            || evidence.launch_count() != 2
            || evidence.single_launch_symbol().is_some()
            || evidence.single_launch_tile().is_some()
        {
            return Err(format!("{:?} route-level identity changed", case.route));
        }
        let (m, k, n) = case.target.dims;
        for (index, node) in evidence.nodes().iter().enumerate() {
            if node.kind != PhysicalLaunchKind::Gemm
                || node.symbol != case.symbols[index]
                || node.module_kind != case.modules[index]
                || node.logical_op != ResolvedGemmOp::Tn
                || node.logical_dtype != PolicyDtype::F32
                || node.execution_dtype != PolicyDtype::F32
                || node.shape != case.target.dims
                || node.strides != (k, n, n)
                || node.tile != Some(case.tiles[index])
                || node.numeric_contract != Some(case.numeric[index])
                || node.ownership != Some(case.ownership[index])
                || node.launch.grid_dim != case.grids[index]
                || node.launch.block_dim != case.blocks[index]
                || node.launch.shared_mem_bytes != 0
            {
                return Err(format!(
                    "{:?} physical node {index} changed: {node:?}",
                    case.route
                ));
            }
        }
        if m == 0 {
            return Err("target case unexpectedly has zero reduction".into());
        }
        Ok(())
    }

    fn require_prior_auto_identity(
        ctx: &GpuCtx,
        launch: &QualifiedPhysicalLaunch<'_>,
        case: Case,
    ) -> Result<(), String> {
        let evidence = launch.evidence();
        let (_, k, n) = case.target.dims;
        let expected_grids = [
            (
                (k.div_ceil(128) * n.div_ceil(128)) as u32,
                1,
                case.target.chunks as u32,
            ),
            ((k * n).div_ceil(256) as u32, 1, 1),
        ];
        let expected_tiles = [(128, 128), (1, 1)];
        let expected_numeric = [
            ResolvedNumericContract::ScalarFmaV1,
            ResolvedNumericContract::ScalarFmaTnSplitMF64ReduceV1,
        ];
        let expected_ownership = [
            ResolvedOutputOwnership::OneCtaPerOutputTileV1,
            ResolvedOutputOwnership::OneThreadPerOutputElementFixedSplitMReduceV1,
        ];
        if node_symbols(launch) != PRIOR_SYMBOLS
            || evidence.route_identity() != &ctx.gemm_route()
            || evidence.evidence_scope() != "eager_preflight_same_launcher"
            || !evidence.eager_graph_equal()
            || evidence.launch_count() != 2
            || evidence.single_launch_symbol().is_some()
            || evidence.single_launch_tile().is_some()
            || evidence.nodes().iter().enumerate().any(|(index, node)| {
                node.module_kind != ModuleKind::TriadScalar
                    || node.kind != PhysicalLaunchKind::Gemm
                    || node.logical_op != ResolvedGemmOp::Tn
                    || node.logical_dtype != PolicyDtype::F32
                    || node.execution_dtype != PolicyDtype::F32
                    || node.shape != case.target.dims
                    || node.strides != (k, n, n)
                    || node.tile != Some(expected_tiles[index])
                    || node.numeric_contract != Some(expected_numeric[index])
                    || node.ownership != Some(expected_ownership[index])
                    || node.launch.grid_dim != expected_grids[index]
                    || node.launch.block_dim != (256, 1, 1)
                    || node.launch.shared_mem_bytes != 0
            })
        {
            return Err(format!(
                "prior actual AUTO physical identity changed: {:?}",
                evidence.nodes()
            ));
        }
        Ok(())
    }

    fn exercise_path(
        ctx: &GpuCtx,
        launch: &mut QualifiedPhysicalLaunch<'_>,
        words: &(Vec<u32>, Vec<u32>, Vec<u32>),
        path: TimedPath,
        expected: &[u32],
        forced: bool,
    ) -> Result<(), String> {
        let original_inputs = (words.1.clone(), words.2.clone());
        for repeat in 0..2 {
            launch.upload_exact_unbiased_f32_words(ctx, &words.0, &words.1, &words.2)?;
            match (path, forced) {
                (TimedPath::Eager, true) => {
                    launch.measure_prevalidated_forced_eager_window_ms(ctx, 1)?;
                }
                (TimedPath::Eager, false) => {
                    launch.measure_eager_window_ms(ctx, 1)?;
                }
                (TimedPath::Graph, _) => {
                    launch.measure_graph_window_ms(ctx, 1)?;
                }
            }
            let actual = launch.f32_output_bits(ctx)?;
            if actual != expected {
                return Err(format!("qualified output differs at repeat {repeat}"));
            }
            if launch.f32_operand_bits(ctx)? != original_inputs {
                return Err(format!("qualified launch modified A/B at repeat {repeat}"));
            }
            let guards = launch.validate_red_zones(ctx)?;
            if guards.allocation_count() != 3 || guards.element_count() == 0 {
                return Err("qualified output/A/B guard inventory changed".into());
            }
        }
        Ok(())
    }

    fn forced_must_reject(
        ctx: &GpuCtx,
        route: Sm89ExactF32TnRoute,
        dims: (usize, usize, usize),
        alpha: f32,
    ) -> Result<(), String> {
        match qualify_physical_launch(
            ctx,
            request(
                PhysicalQualificationRoute::Sm89ExactF32TnForced(route),
                dims,
                alpha,
            ),
        ) {
            Ok(_) => Err(format!(
                "forced {:?} incorrectly admitted dims={dims:?} alpha={alpha}",
                route
            )),
            Err(_) => Ok(()),
        }
    }

    fn exercise_auto_raw_case(
        ctx: &GpuCtx,
        case: Case,
        cell: RawCell,
        alpha: f32,
        exceptional: bool,
    ) -> Result<(), String> {
        let raw = run_raw_probe(ctx, case.route, cell, alpha, exceptional)?;
        if raw.compared_output_words != cell.dims.1 * cell.dims.2
            || raw.guarded_allocations != 7
            || raw.guarded_elements == 0
            || (case.route == Sm89ExactF32TnRoute::D768InDualChunkFused
                && raw.compared_transpose_words != cell.dims.0 * cell.dims.1)
            || (case.route != Sm89ExactF32TnRoute::D768InDualChunkFused
                && raw.compared_partial_words != cell.chunks * cell.dims.1 * cell.dims.2)
        {
            return Err(format!("{:?} raw/scratch inventory changed", case.route));
        }
        println!(
            "{}",
            json!({
                "schema":"MambaTriadSm89ExactF32B2RawProbeV1",
                "route":format!("{:?}",case.route),
                "dims":cell.dims,
                "alpha_bits":alpha.to_bits(),
                "exceptional":exceptional,
                "partial_words":raw.compared_partial_words,
                "transpose_words":raw.compared_transpose_words,
                "output_words":raw.compared_output_words,
                "guarded_allocations":raw.guarded_allocations,
                "guarded_elements":raw.guarded_elements,
                "exact":true
            })
        );
        let words = raw_seed_words(cell, exceptional)?;
        let auto_request = request(
            PhysicalQualificationRoute::F32Policy(F32TriadPolicy::ExactScalarFmaV1),
            cell.dims,
            alpha,
        );
        if !tn_policy_facade_supports_epilogue(alpha, 1.0) {
            return match qualify_physical_launch(ctx, auto_request) {
                Ok(_) => Err("public AUTO admitted an epilogue the production wrapper cannot express"
                    .into()),
                Err(error)
                    if error.contains(
                        "F32 policy qualification epilogue must match the production wrapper alpha and beta",
                    ) =>
                {
                    Ok(())
                }
                Err(error) => Err(format!(
                    "public AUTO rejected the unsupported epilogue for an unexpected reason: {error}"
                )),
            };
        }
        let mut auto = qualify_physical_launch(ctx, auto_request)?;
        if node_symbols(&auto).iter().any(|symbol| {
            [D768_IN_FUSED_SYMBOL, D768_OUT_RAW_SYMBOL, PRISM_RAW_SYMBOL].contains(symbol)
        }) {
            return Err(
                "public AUTO admitted an exact-F32 B2 route before the route-set evidence".into(),
            );
        }
        auto.upload_exact_unbiased_f32_words(ctx, &words.0, &words.1, &words.2)?;
        let original_inputs = auto.f32_operand_bits(ctx)?;
        auto.measure_eager_window_ms(ctx, 1)?;
        let eager = auto.f32_output_bits(ctx)?;
        if auto.f32_operand_bits(ctx)? != original_inputs {
            return Err("prior AUTO eager launch modified A/B".into());
        }
        auto.validate_red_zones(ctx)?;
        auto.upload_exact_unbiased_f32_words(ctx, &words.0, &words.1, &words.2)?;
        auto.measure_graph_window_ms(ctx, 1)?;
        if auto.f32_output_bits(ctx)? != eager || auto.f32_operand_bits(ctx)? != original_inputs {
            return Err("prior AUTO eager/graph bits or A/B immutability changed".into());
        }
        auto.validate_red_zones(ctx)?;
        Ok(())
    }

    fn qualify_case(
        forced_ctx: &GpuCtx,
        auto_ctx: &GpuCtx,
        case: Case,
    ) -> Result<CaseReceipt, String> {
        let target_raw = run_raw_probe(auto_ctx, case.route, case.target, 1.0, false)?;
        println!(
            "{}",
            json!({
                "schema":"MambaTriadSm89ExactF32B2RawProbeV1",
                "route":format!("{:?}",case.route),
                "dims":case.target.dims,
                "alpha_bits":1.0_f32.to_bits(),
                "exceptional":false,
                "partial_words":target_raw.compared_partial_words,
                "transpose_words":target_raw.compared_transpose_words,
                "output_words":target_raw.compared_output_words,
                "guarded_allocations":target_raw.guarded_allocations,
                "guarded_elements":target_raw.guarded_elements,
                "exact":true
            })
        );
        let target_words = raw_seed_words(case.target, false)?;
        let forced_request = request(
            PhysicalQualificationRoute::Sm89ExactF32TnForced(case.route),
            case.target.dims,
            1.0,
        );
        let mut forced = qualify_physical_launch(forced_ctx, forced_request)?;
        require_large_tn_identity(forced_ctx, &forced, case)?;
        exercise_path(
            forced_ctx,
            &mut forced,
            &target_words,
            TimedPath::Eager,
            &target_raw.output_bits,
            true,
        )?;
        exercise_path(
            forced_ctx,
            &mut forced,
            &target_words,
            TimedPath::Graph,
            &target_raw.output_bits,
            true,
        )?;
        drop(forced);

        let mut auto = qualify_physical_launch(
            auto_ctx,
            request(
                PhysicalQualificationRoute::F32Policy(F32TriadPolicy::ExactScalarFmaV1),
                case.target.dims,
                1.0,
            ),
        )?;
        require_prior_auto_identity(auto_ctx, &auto, case)?;
        exercise_path(
            auto_ctx,
            &mut auto,
            &target_words,
            TimedPath::Eager,
            &target_raw.output_bits,
            false,
        )?;
        exercise_path(
            auto_ctx,
            &mut auto,
            &target_words,
            TimedPath::Graph,
            &target_raw.output_bits,
            false,
        )?;
        drop(auto);

        forced_must_reject(forced_ctx, case.route, case.tail.dims, 1.0)?;
        exercise_auto_raw_case(auto_ctx, case, case.tail, 1.0, false)?;
        exercise_auto_raw_case(auto_ctx, case, case.tail, 1.0, true)?;
        forced_must_reject(forced_ctx, case.route, case.target.dims, -0.75)?;
        exercise_auto_raw_case(auto_ctx, case, case.tail, -0.75, false)?;
        forced_must_reject(
            forced_ctx,
            case.route,
            (0, case.tail.dims.1, case.tail.dims.2),
            1.0,
        )?;
        run_k0_auto_probe(auto_ctx, case.tail.dims.1, case.tail.dims.2)?;

        Ok(CaseReceipt {
            phases: REQUIRED_PHASES.to_vec(),
        })
    }

    fn quantile(values: &[f64], q: f64) -> f64 {
        let mut values = values.to_vec();
        values.sort_by(f64::total_cmp);
        values[((values.len() - 1) as f64 * q).ceil() as usize]
    }

    fn timed_window(
        ctx: &GpuCtx,
        launch: &mut QualifiedPhysicalLaunch<'_>,
        words: &(Vec<u32>, Vec<u32>, Vec<u32>),
        path: TimedPath,
        forced: bool,
        expected: &[u32],
    ) -> Result<f64, String> {
        launch.upload_exact_unbiased_f32_words(ctx, &words.0, &words.1, &words.2)?;
        let total = match (path, forced) {
            (TimedPath::Eager, true) => {
                launch.measure_prevalidated_forced_eager_window_ms(ctx, OPS_PER_WINDOW)?
            }
            (TimedPath::Eager, false) => launch.measure_eager_window_ms(ctx, OPS_PER_WINDOW)?,
            (TimedPath::Graph, _) => launch.measure_graph_window_ms(ctx, OPS_PER_WINDOW)?,
        };
        if !total.is_finite() || total <= 0.0 {
            return Err("invalid exact-F32 timing window".into());
        }
        if launch.f32_output_bits(ctx)? != expected {
            return Err("timed exact-F32 output differs from the untimed 20-chain oracle".into());
        }
        if launch.f32_operand_bits(ctx)? != (words.1.clone(), words.2.clone()) {
            return Err("timed exact-F32 launch modified A/B".into());
        }
        let guards = launch.validate_red_zones(ctx)?;
        if guards.allocation_count() != 3 || guards.element_count() == 0 {
            return Err("timed exact-F32 guard inventory changed".into());
        }
        Ok(total / OPS_PER_WINDOW as f64)
    }

    fn screen_batch(
        forced_ctx: &GpuCtx,
        auto_ctx: &GpuCtx,
        windows: usize,
    ) -> Result<bool, String> {
        let mut all_strict = true;
        for case in CASES {
            let words = raw_seed_words(case.target, false)?;
            let repeated_oracle = run_raw_probe_repeated(
                auto_ctx,
                case.route,
                case.target,
                1.0,
                false,
                OPS_PER_WINDOW,
            )?
            .output_bits;
            let mut forced = qualify_physical_launch(
                forced_ctx,
                request(
                    PhysicalQualificationRoute::Sm89ExactF32TnForced(case.route),
                    case.target.dims,
                    1.0,
                ),
            )?;
            let mut auto = qualify_physical_launch(
                auto_ctx,
                request(
                    PhysicalQualificationRoute::F32Policy(F32TriadPolicy::ExactScalarFmaV1),
                    case.target.dims,
                    1.0,
                ),
            )?;
            require_large_tn_identity(forced_ctx, &forced, case)?;
            require_prior_auto_identity(auto_ctx, &auto, case)?;
            for (path_name, path) in [("eager", TimedPath::Eager), ("graph", TimedPath::Graph)] {
                for (order_name, candidate_first) in [("ABBA", true), ("BAAB", false)] {
                    let mut ratios = Vec::with_capacity(windows);
                    let mut observations = Vec::with_capacity(windows);
                    for _ in 0..windows {
                        let (candidate_a, reference_a, reference_b, candidate_b, raw_bracket) =
                            if candidate_first {
                                let candidate_a = timed_window(
                                    forced_ctx,
                                    &mut forced,
                                    &words,
                                    path,
                                    true,
                                    &repeated_oracle,
                                )?;
                                let reference_a = timed_window(
                                    auto_ctx,
                                    &mut auto,
                                    &words,
                                    path,
                                    false,
                                    &repeated_oracle,
                                )?;
                                let reference_b = timed_window(
                                    auto_ctx,
                                    &mut auto,
                                    &words,
                                    path,
                                    false,
                                    &repeated_oracle,
                                )?;
                                let candidate_b = timed_window(
                                    forced_ctx,
                                    &mut forced,
                                    &words,
                                    path,
                                    true,
                                    &repeated_oracle,
                                )?;
                                (
                                    candidate_a,
                                    reference_a,
                                    reference_b,
                                    candidate_b,
                                    [candidate_a, reference_a, reference_b, candidate_b],
                                )
                            } else {
                                let reference_a = timed_window(
                                    auto_ctx,
                                    &mut auto,
                                    &words,
                                    path,
                                    false,
                                    &repeated_oracle,
                                )?;
                                let candidate_a = timed_window(
                                    forced_ctx,
                                    &mut forced,
                                    &words,
                                    path,
                                    true,
                                    &repeated_oracle,
                                )?;
                                let candidate_b = timed_window(
                                    forced_ctx,
                                    &mut forced,
                                    &words,
                                    path,
                                    true,
                                    &repeated_oracle,
                                )?;
                                let reference_b = timed_window(
                                    auto_ctx,
                                    &mut auto,
                                    &words,
                                    path,
                                    false,
                                    &repeated_oracle,
                                )?;
                                (
                                    candidate_a,
                                    reference_a,
                                    reference_b,
                                    candidate_b,
                                    [reference_a, candidate_a, candidate_b, reference_b],
                                )
                            };
                        observations.push(raw_bracket);
                        ratios.push(
                            ((candidate_a + candidate_b) * 0.5)
                                / ((reference_a + reference_b) * 0.5),
                        );
                    }
                    let p50 = quantile(&ratios, 0.50);
                    let p95 = quantile(&ratios, 0.95);
                    let strict = p50 < 0.99 && p95 < 0.99;
                    all_strict &= strict;
                    println!(
                        "{}",
                        json!({
                            "schema":"MambaTriadSm89ExactF32B2PairedScreenV1",
                            "route":format!("{:?}",case.route),
                            "dims":case.target.dims,
                            "path":path_name,
                            "order":order_name,
                            "windows":windows,
                            "ops_per_window":OPS_PER_WINDOW,
                            "observations_ms_per_op":observations,
                            "ratios":ratios,
                            "ratio_p50":p50,
                            "ratio_p95":p95,
                            "strict_099":strict,
                            "comparator":"current_actual_AUTO_prior_fallback"
                        })
                    );
                }
            }
        }
        Ok(all_strict)
    }

    #[test]
    #[ignore = "requires an exclusive SM89 GPU; root is the sole GPU owner"]
    fn sm89_exact_f32_large_tn_pre_admission_forced_vs_prior_actual_auto() -> Result<(), String> {
        let mode = AdmissionMode::PreAdmission;
        mode.require()?;
        let forced_ctx = runtime()?;
        let auto_ctx = runtime()?;
        require_module_abi_resources(&forced_ctx)?;

        let mut cases = Vec::with_capacity(CASES.len());
        for case in CASES {
            cases.push(qualify_case(&forced_ctx, &auto_ctx, case)?);
        }
        let once3 = screen_batch(&forced_ctx, &auto_ctx, 3)?;
        let (screen, fast_labelled) = if once3 {
            if screen_batch(&forced_ctx, &auto_ctx, 7)? {
                println!(
                    "{}",
                    json!({
                        "schema":"MambaTriadSm89ExactF32B2FastComparatorV1",
                        "status":"separately_labelled_deferred_to_frozen_discovery_receipts",
                        "admission_comparator":"current_actual_AUTO_prior_fallback"
                    })
                );
                (ScreenOutcome::Retained, true)
            } else {
                (ScreenOutcome::Once7Stop, false)
            }
        } else {
            (ScreenOutcome::Once3Stop, false)
        };
        let receipt = BatchReceipt {
            cases,
            screen,
            fast_labelled,
        };
        validate_batch_receipt(&receipt)?;
        println!(
            "{}",
            json!({
                "schema":"MambaTriadSm89ExactF32B2AdmissionReceiptV1",
                "mode":mode.label(),
                "screen":format!("{:?}",receipt.screen),
                "fast_labelled":receipt.fast_labelled,
                "auto_admitted":false
            })
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires an exclusive SM89 GPU; root is the sole GPU owner"]
    fn sm89_exact_f32_large_tn_post_admission_actual_auto_exactness() -> Result<(), String> {
        let mode = AdmissionMode::PostAdmission;
        mode.require()?;
        let ctx = runtime()?;
        require_module_abi_resources(&ctx)?;

        for case in CASES {
            let oracle = run_raw_probe(&ctx, case.route, case.target, 1.0, false)?;
            if oracle.compared_output_words != case.target.dims.1 * case.target.dims.2
                || oracle.guarded_allocations != 7
                || oracle.guarded_elements == 0
            {
                return Err(format!(
                    "{:?} post-admission raw oracle inventory changed",
                    case.route
                ));
            }
            let words = raw_seed_words(case.target, false)?;
            let mut auto = qualify_physical_launch(
                &ctx,
                request(
                    PhysicalQualificationRoute::F32Policy(F32TriadPolicy::ExactScalarFmaV1),
                    case.target.dims,
                    1.0,
                ),
            )?;
            require_large_tn_identity(&ctx, &auto, case)?;
            exercise_path(
                &ctx,
                &mut auto,
                &words,
                TimedPath::Eager,
                &oracle.output_bits,
                false,
            )?;
            exercise_path(
                &ctx,
                &mut auto,
                &words,
                TimedPath::Graph,
                &oracle.output_bits,
                false,
            )?;
            println!(
                "{}",
                json!({
                    "schema":"MambaTriadSm89ExactF32PostAdmissionAutoV1",
                    "mode":mode.label(),
                    "route":format!("{:?}",case.route),
                    "dims":case.target.dims,
                    "eager_graph_identity_exact":true,
                    "repeated_bits_exact":true,
                    "guards_exact":true,
                    "inputs_immutable":true,
                    "auto_admitted":true
                })
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod native_contract_tests {
    use super::*;

    fn complete_case() -> CaseReceipt {
        CaseReceipt {
            phases: vec![
                QualificationPhase::ModuleAbiResources,
                QualificationPhase::RawScratch,
                QualificationPhase::TargetEager,
                QualificationPhase::TargetGraph,
                QualificationPhase::Tail,
                QualificationPhase::Exceptional,
                QualificationPhase::NonUnitAlpha,
                QualificationPhase::K0,
            ],
        }
    }

    #[test]
    fn qualification_receipt_accepts_only_three_complete_cases_and_ordered_screen() {
        let receipt = BatchReceipt {
            cases: vec![complete_case(), complete_case(), complete_case()],
            screen: ScreenOutcome::Retained,
            fast_labelled: true,
        };
        validate_batch_receipt(&receipt).unwrap();

        for case_index in 0..3 {
            for phase_index in 0..8 {
                let mut mutated = receipt.clone();
                mutated.cases[case_index].phases.remove(phase_index);
                assert!(validate_batch_receipt(&mutated).is_err());
            }
        }
        let mut swapped = receipt.clone();
        swapped.cases[1].phases.swap(1, 2);
        assert!(validate_batch_receipt(&swapped).is_err());
    }

    #[test]
    fn qualification_receipt_fail_closes_performance_ladder() {
        for (screen, fast_labelled, accepted) in [
            (ScreenOutcome::Once3Stop, false, true),
            (ScreenOutcome::Once3Stop, true, false),
            (ScreenOutcome::Once7Stop, false, true),
            (ScreenOutcome::Once7Stop, true, false),
            (ScreenOutcome::Retained, false, false),
            (ScreenOutcome::Retained, true, true),
        ] {
            let receipt = BatchReceipt {
                cases: vec![complete_case(), complete_case(), complete_case()],
                screen,
                fast_labelled,
            };
            assert_eq!(validate_batch_receipt(&receipt).is_ok(), accepted);
        }
    }

    #[test]
    fn tn_policy_facade_is_exercised_only_for_the_production_epilogue() {
        assert!(tn_policy_facade_supports_epilogue(1.0, 1.0));
        assert!(!tn_policy_facade_supports_epilogue(-0.75, 1.0));
        assert!(!tn_policy_facade_supports_epilogue(1.0, 0.0));
        assert!(!tn_policy_facade_supports_epilogue(1.0, -0.0));
    }

    #[test]
    fn tn_k0_oracle_matches_fma_not_bitwise_identity_for_signed_zero() {
        let output = [
            f32::from_bits(0x0000_0000),
            f32::from_bits(0x8000_0000),
            f32::from_bits(0x3f80_1000),
            f32::from_bits(0xbf80_1000),
            f32::from_bits(0x0000_0001),
            f32::from_bits(0x8000_0001),
        ];
        assert_eq!(
            tn_k0_expected_output_bits(1.0, &output),
            [
                0x0000_0000,
                0x0000_0000,
                0x3f80_1000,
                0xbf80_1000,
                0x0000_0001,
                0x8000_0001,
            ]
        );
    }
}
