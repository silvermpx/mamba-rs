//! First live wiring smoke for the six retained Ada TF32 joint cells.
//!
//! This test intentionally stops before timing and cohort admission. It proves
//! the forced production launch manifest and compares its exact bits with the
//! portable route that served the cell before the joint module. The prior
//! route is forced so the comparator is identical across toolkit-specific
//! AUTO cohort identities.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LiteralOp {
    Nn,
    Tn,
    Nt,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LiteralRoute {
    TnN96,
    TnM64N64,
    TnM64N96S2,
    NnDirectN96,
    NnN96,
    NtALdmatrixN96,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LiteralModule {
    Sm80,
    Sm89Finalist,
    Sm89Joint,
}

impl LiteralModule {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Sm80 => "TriadSm80",
            Self::Sm89Finalist => "TriadSm89Finalist",
            Self::Sm89Joint => "TriadSm89Tf32Joint",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LiteralCase {
    name: &'static str,
    op: LiteralOp,
    dims: (usize, usize, usize),
    route: LiteralRoute,
    gemm_symbol: &'static str,
    old_auto_symbol: &'static str,
    transform_symbol: Option<&'static str>,
    transpose_grid: Option<(u32, u32, u32)>,
    gemm_grid: (u32, u32, u32),
    gemm_tile: (u32, u32),
    gemm_shared_bytes: u32,
}

const TRANSPOSE_SYMBOL: &str = "gemm_bi_tn_sm89_tf32_pre_rna_transpose_32x32_v1";
const TN_N96_SYMBOL: &str = "gemm_bi_tn_sm89_tf32_pre_rna_m128n96_bk32_s3_v1";
const TN_M64N96_S2_SYMBOL: &str = "gemm_bi_tn_sm89_tf32_pre_rna_m64n96_bk32_s2_v1";
const LOWER_TN_M64N64_SYMBOL: &str = "gemm_bi_tn_sm89_tf32_pre_rna_m64n64_bk32_s3_v1";
const NN_DIRECT_N96_SYMBOL: &str = "gemm_bi_nn_sm89_tf32_addhalf_m128n96_bk32_s3_direct_v1";
const NN_N96_SYMBOL: &str = "gemm_bi_nn_sm89_tf32_addhalf_m128n96_bk32_s3_v1";
const NT_A_LDMATRIX_N96_SYMBOL: &str = "gemm_bi_nt_sm89_tf32_a_ldmatrix_m128n96_bk32_s3_v1";
const OLD_TN_M64N64_SYMBOL: &str = "gemm_bi_tn_sm80_mma_tf32_v1_m64n64_bk32_s2";
const OLD_TN_M128N64_SYMBOL: &str = "gemm_bi_tn_sm80_mma_tf32_v1_m128n64_bk32_s3";
const OLD_NN_M128N128_SYMBOL: &str = "gemm_bi_nn_sm80_mma_tf32_v1_m128n128_bk32_s3";

const CASES: [LiteralCase; 6] = [
    LiteralCase {
        name: "tn_d768_in",
        op: LiteralOp::Tn,
        dims: (2_048, 768, 3_072),
        route: LiteralRoute::TnN96,
        gemm_symbol: TN_N96_SYMBOL,
        old_auto_symbol: OLD_TN_M64N64_SYMBOL,
        transform_symbol: Some(TRANSPOSE_SYMBOL),
        transpose_grid: Some((24, 64, 1)),
        gemm_grid: (192, 1, 1),
        gemm_tile: (128, 96),
        gemm_shared_bytes: 86_016,
    },
    LiteralCase {
        name: "tn_d768_out",
        op: LiteralOp::Tn,
        dims: (2_048, 1_536, 768),
        route: LiteralRoute::TnN96,
        gemm_symbol: TN_N96_SYMBOL,
        old_auto_symbol: OLD_TN_M64N64_SYMBOL,
        transform_symbol: Some(TRANSPOSE_SYMBOL),
        transpose_grid: Some((48, 64, 1)),
        gemm_grid: (96, 1, 1),
        gemm_tile: (128, 96),
        gemm_shared_bytes: 86_016,
    },
    LiteralCase {
        name: "tn_prism",
        op: LiteralOp::Tn,
        dims: (4_621, 384, 1_928),
        route: LiteralRoute::TnM64N96S2,
        gemm_symbol: TN_M64N96_S2_SYMBOL,
        old_auto_symbol: OLD_TN_M128N64_SYMBOL,
        transform_symbol: Some(TRANSPOSE_SYMBOL),
        transpose_grid: Some((12, 145, 1)),
        gemm_grid: (126, 1, 1),
        gemm_tile: (64, 96),
        gemm_shared_bytes: 40_960,
    },
    LiteralCase {
        name: "nn_prism",
        op: LiteralOp::Nn,
        dims: (4_621, 384, 1_928),
        route: LiteralRoute::NnDirectN96,
        gemm_symbol: NN_DIRECT_N96_SYMBOL,
        old_auto_symbol: OLD_NN_M128N128_SYMBOL,
        transform_symbol: None,
        transpose_grid: None,
        gemm_grid: (777, 1, 1),
        gemm_tile: (128, 96),
        gemm_shared_bytes: 86_016,
    },
    LiteralCase {
        name: "nn_d768_out",
        op: LiteralOp::Nn,
        dims: (2_048, 1_536, 768),
        route: LiteralRoute::NnN96,
        gemm_symbol: NN_N96_SYMBOL,
        old_auto_symbol: OLD_NN_M128N128_SYMBOL,
        transform_symbol: None,
        transpose_grid: None,
        gemm_grid: (128, 1, 1),
        gemm_tile: (128, 96),
        gemm_shared_bytes: 86_016,
    },
    LiteralCase {
        name: "nt_d768_in",
        op: LiteralOp::Nt,
        dims: (2_048, 768, 3_072),
        route: LiteralRoute::NtALdmatrixN96,
        gemm_symbol: NT_A_LDMATRIX_N96_SYMBOL,
        old_auto_symbol: "gemm_bi_nt_sm89_mma_tf32_compact8_v1_m128n64_bk32_s2",
        transform_symbol: None,
        transpose_grid: None,
        gemm_grid: (128, 1, 1),
        gemm_tile: (128, 96),
        gemm_shared_bytes: 86_016,
    },
];

fn post_admission_symbol_and_module(
    case: LiteralCase,
    nvrtc: (i32, i32),
) -> (&'static str, LiteralModule) {
    if matches!(nvrtc, (12, 8) | (13, 0)) {
        return match case.name {
            "tn_prism" => (LOWER_TN_M64N64_SYMBOL, LiteralModule::Sm89Joint),
            "nn_prism" => (case.old_auto_symbol, LiteralModule::Sm80),
            "nt_d768_in" => (case.old_auto_symbol, LiteralModule::Sm89Finalist),
            _ => (case.gemm_symbol, LiteralModule::Sm89Joint),
        };
    }
    (case.gemm_symbol, LiteralModule::Sm89Joint)
}

#[test]
fn six_cell_literal_map_is_complete_and_has_independent_launch_geometry() {
    let expected_keys = [
        ("tn_d768_in", LiteralOp::Tn, (2_048, 768, 3_072)),
        ("tn_d768_out", LiteralOp::Tn, (2_048, 1_536, 768)),
        ("tn_prism", LiteralOp::Tn, (4_621, 384, 1_928)),
        ("nn_prism", LiteralOp::Nn, (4_621, 384, 1_928)),
        ("nn_d768_out", LiteralOp::Nn, (2_048, 1_536, 768)),
        ("nt_d768_in", LiteralOp::Nt, (2_048, 768, 3_072)),
    ];
    assert_eq!(
        CASES.map(|case| (case.name, case.op, case.dims)),
        expected_keys,
        "the wiring smoke must cover each retained logical cell exactly once",
    );

    for case in CASES {
        let (output_rows, output_columns) = match case.op {
            LiteralOp::Nn => (case.dims.0, case.dims.2),
            LiteralOp::Tn => (case.dims.1, case.dims.2),
            LiteralOp::Nt => (case.dims.0, case.dims.1),
        };
        let expected_grid = (
            u32::try_from(output_rows.div_ceil(case.gemm_tile.0 as usize)).unwrap()
                * u32::try_from(output_columns.div_ceil(case.gemm_tile.1 as usize)).unwrap(),
            1,
            1,
        );
        assert_eq!(case.gemm_grid, expected_grid, "{} GEMM grid", case.name);
        assert_eq!(
            case.transpose_grid.is_some(),
            case.op == LiteralOp::Tn,
            "{} transform cardinality",
            case.name,
        );
        assert_eq!(
            case.transform_symbol.is_some(),
            case.op == LiteralOp::Tn,
            "{} transform symbol cardinality",
            case.name,
        );
    }
}

#[test]
fn toolkit_literal_map_keeps_lower_winners_and_adds_nt_only_on_cuda_13_2() {
    for (nvrtc, expected) in [
        (
            (12, 8),
            [
                (TN_N96_SYMBOL, LiteralModule::Sm89Joint),
                (TN_N96_SYMBOL, LiteralModule::Sm89Joint),
                (LOWER_TN_M64N64_SYMBOL, LiteralModule::Sm89Joint),
                (OLD_NN_M128N128_SYMBOL, LiteralModule::Sm80),
                (NN_N96_SYMBOL, LiteralModule::Sm89Joint),
                (
                    "gemm_bi_nt_sm89_mma_tf32_compact8_v1_m128n64_bk32_s2",
                    LiteralModule::Sm89Finalist,
                ),
            ],
        ),
        (
            (13, 0),
            [
                (TN_N96_SYMBOL, LiteralModule::Sm89Joint),
                (TN_N96_SYMBOL, LiteralModule::Sm89Joint),
                (LOWER_TN_M64N64_SYMBOL, LiteralModule::Sm89Joint),
                (OLD_NN_M128N128_SYMBOL, LiteralModule::Sm80),
                (NN_N96_SYMBOL, LiteralModule::Sm89Joint),
                (
                    "gemm_bi_nt_sm89_mma_tf32_compact8_v1_m128n64_bk32_s2",
                    LiteralModule::Sm89Finalist,
                ),
            ],
        ),
        (
            (13, 2),
            [
                (TN_N96_SYMBOL, LiteralModule::Sm89Joint),
                (TN_N96_SYMBOL, LiteralModule::Sm89Joint),
                (TN_M64N96_S2_SYMBOL, LiteralModule::Sm89Joint),
                (NN_DIRECT_N96_SYMBOL, LiteralModule::Sm89Joint),
                (NN_N96_SYMBOL, LiteralModule::Sm89Joint),
                (NT_A_LDMATRIX_N96_SYMBOL, LiteralModule::Sm89Joint),
            ],
        ),
    ] {
        assert_eq!(
            CASES.map(|case| post_admission_symbol_and_module(case, nvrtc)),
            expected,
            "CUDA {nvrtc:?} AUTO winner map",
        );
    }
}

#[cfg(feature = "cuda")]
mod live {
    use super::*;
    use mamba_rs::mamba_ssm::gpu::context::{F32TriadPolicy, GpuCtx};
    use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
    use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
        PhysicalQualificationF32Epilogue, PhysicalQualificationRequest, PhysicalQualificationRoute,
        QualifiedPhysicalLaunch, Tf32PhysicalRoute, Tf32PortableRoute, Tf32PortableStages,
        Tf32PortableTile, presize_physical_qualification_suite, qualify_physical_launch,
    };
    use mamba_rs::mamba_ssm::gpu::kernel_identity::{
        ModuleKind, PhysicalLaunchKind, PolicyDtype, ResolvedGemmOp, ResolvedNumericContract,
        ResolvedOutputOwnership,
    };

    fn op(case: LiteralCase) -> ResolvedGemmOp {
        match case.op {
            LiteralOp::Nn => ResolvedGemmOp::Nn,
            LiteralOp::Tn => ResolvedGemmOp::Tn,
            LiteralOp::Nt => ResolvedGemmOp::Nt,
        }
    }

    fn route(case: LiteralCase) -> Tf32PhysicalRoute {
        match case.route {
            LiteralRoute::TnN96 => Tf32PhysicalRoute::Sm89TnPreRnaN96V1,
            LiteralRoute::TnM64N64 => Tf32PhysicalRoute::Sm89TnPreRnaM64N64V1,
            LiteralRoute::TnM64N96S2 => Tf32PhysicalRoute::Sm89TnPreRnaM64N96S2V1,
            LiteralRoute::NnDirectN96 => Tf32PhysicalRoute::Sm89NnDirectN96V1,
            LiteralRoute::NnN96 => Tf32PhysicalRoute::Sm89NnN96V1,
            LiteralRoute::NtALdmatrixN96 => Tf32PhysicalRoute::Sm89NtALdmatrixN96V1,
        }
    }

    fn auto_expected_case(mut case: LiteralCase, nvrtc: (i32, i32)) -> LiteralCase {
        if case.name == "tn_prism" && matches!(nvrtc, (12, 8) | (13, 0)) {
            case.route = LiteralRoute::TnM64N64;
            case.gemm_symbol = LOWER_TN_M64N64_SYMBOL;
            case.gemm_grid = (186, 1, 1);
            case.gemm_tile = (64, 64);
            case.gemm_shared_bytes = 49_152;
        }
        case
    }

    fn prior_route(case: LiteralCase) -> Tf32PhysicalRoute {
        let (tile, stages) = match case.old_auto_symbol {
            OLD_TN_M64N64_SYMBOL => (Tf32PortableTile::M64N64, Tf32PortableStages::S2),
            OLD_TN_M128N64_SYMBOL => (Tf32PortableTile::M128N64, Tf32PortableStages::S3),
            OLD_NN_M128N128_SYMBOL => (Tf32PortableTile::M128N128, Tf32PortableStages::S3),
            "gemm_bi_nt_sm89_mma_tf32_compact8_v1_m128n64_bk32_s2" => {
                return Tf32PhysicalRoute::Sm89MmaTf32Compact8V1;
            }
            symbol => panic!("unknown prior TF32 route symbol {symbol}"),
        };
        Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute { tile, stages })
    }

    fn literal_module(module: ModuleKind) -> Option<LiteralModule> {
        match module {
            ModuleKind::TriadSm80 => Some(LiteralModule::Sm80),
            ModuleKind::TriadSm89Finalist => Some(LiteralModule::Sm89Finalist),
            ModuleKind::TriadSm89Tf32Joint => Some(LiteralModule::Sm89Joint),
            _ => None,
        }
    }

    fn request(case: LiteralCase, forced: bool) -> PhysicalQualificationRequest {
        let op = op(case);
        PhysicalQualificationRequest::contiguous_f32(
            op,
            case.dims,
            if forced {
                PhysicalQualificationRoute::Tf32Forced(route(case))
            } else {
                PhysicalQualificationRoute::Tf32Forced(prior_route(case))
            },
            PhysicalQualificationF32Epilogue::new(
                1.0,
                if op == ResolvedGemmOp::Tn { 1.0 } else { 0.0 },
                false,
            ),
        )
    }

    fn auto_request(case: LiteralCase) -> PhysicalQualificationRequest {
        let op = op(case);
        PhysicalQualificationRequest::contiguous_f32(
            op,
            case.dims,
            PhysicalQualificationRoute::F32Policy(F32TriadPolicy::AllowDeterministicTf32V1),
            PhysicalQualificationF32Epilogue::new(
                1.0,
                if op == ResolvedGemmOp::Tn { 1.0 } else { 0.0 },
                false,
            ),
        )
    }

    fn assert_forced_manifest(
        case: LiteralCase,
        qualified: &QualifiedPhysicalLaunch<'_>,
    ) -> Result<(), String> {
        let evidence = qualified.evidence();
        let expected_count = if case.op == LiteralOp::Tn { 2 } else { 1 };
        if evidence.nodes().len() != expected_count
            || evidence.launch_count() != expected_count as u32
            || !evidence.eager_graph_equal()
            || evidence.uniform_module_kind() != Some(ModuleKind::TriadSm89Tf32Joint)
        {
            return Err(format!(
                "{} forced eager/prepared manifest mismatch: {:?}",
                case.name,
                evidence.nodes(),
            ));
        }

        let logical_strides = match case.op {
            LiteralOp::Nn | LiteralOp::Tn => (case.dims.1, case.dims.2, case.dims.2),
            LiteralOp::Nt => (case.dims.2, case.dims.2, case.dims.1),
        };
        let gemm = evidence
            .nodes()
            .last()
            .ok_or_else(|| format!("{} has no GEMM node", case.name))?;
        let expected_numeric = match case.op {
            LiteralOp::Tn => ResolvedNumericContract::MmaTf32PreRnaAV1,
            LiteralOp::Nn => ResolvedNumericContract::MmaTf32AddHalfUlpV1,
            LiteralOp::Nt => ResolvedNumericContract::MmaTf32RnaV1,
        };
        if gemm.kind != PhysicalLaunchKind::Gemm
            || gemm.symbol != case.gemm_symbol
            || gemm.module_kind != ModuleKind::TriadSm89Tf32Joint
            || gemm.logical_op != op(case)
            || gemm.logical_dtype != PolicyDtype::F32
            || gemm.execution_dtype != PolicyDtype::F32
            || gemm.shape != case.dims
            || gemm.strides != logical_strides
            || gemm.tile != Some(case.gemm_tile)
            || gemm.numeric_contract != Some(expected_numeric)
            || gemm.ownership != Some(ResolvedOutputOwnership::OneCtaPerOutputTileV1)
            || gemm.launch.grid_dim != case.gemm_grid
            || gemm.launch.block_dim != (256, 1, 1)
            || gemm.launch.shared_mem_bytes != case.gemm_shared_bytes
        {
            return Err(format!("{} forced GEMM node drifted: {gemm:?}", case.name));
        }

        if let Some(grid) = case.transpose_grid {
            let transform = &evidence.nodes()[0];
            if transform.kind != PhysicalLaunchKind::InputTransform
                || Some(transform.symbol) != case.transform_symbol
                || transform.module_kind != ModuleKind::TriadSm89Tf32Joint
                || transform.logical_op != ResolvedGemmOp::Tn
                || transform.logical_dtype != PolicyDtype::F32
                || transform.execution_dtype != PolicyDtype::F32
                || transform.shape != case.dims
                || transform.strides != logical_strides
                || transform.tile.is_some()
                || transform.numeric_contract.is_some()
                || transform.ownership.is_some()
                || transform.launch.grid_dim != grid
                || transform.launch.block_dim != (32, 8, 1)
                || transform.launch.shared_mem_bytes != 0
                || transform.launch.arguments_digest == gemm.launch.arguments_digest
            {
                return Err(format!(
                    "{} forced transform node drifted: {transform:?}",
                    case.name,
                ));
            }
        }
        Ok(())
    }

    fn assert_prior_auto_manifest(
        case: LiteralCase,
        qualified: &QualifiedPhysicalLaunch<'_>,
    ) -> Result<(), String> {
        let evidence = qualified.evidence();
        let [node] = evidence.nodes() else {
            return Err(format!(
                "{} prior public AUTO must have one node: {:?}",
                case.name,
                evidence.nodes(),
            ));
        };
        if !evidence.eager_graph_equal()
            || node.kind != PhysicalLaunchKind::Gemm
            || node.symbol != case.old_auto_symbol
            || node.module_kind != prior_route(case).module_kind()
            || node.logical_op != op(case)
            || node.logical_dtype != PolicyDtype::F32
            || node.execution_dtype != PolicyDtype::F32
            || node.shape != case.dims
        {
            return Err(format!(
                "{} public AUTO no longer resolves to the pre-admission route: {node:?}",
                case.name,
            ));
        }
        Ok(())
    }

    fn compare_one_path(
        case: LiteralCase,
        forced_ctx: &GpuCtx,
        forced: &mut QualifiedPhysicalLaunch<'_>,
        auto_ctx: &GpuCtx,
        auto: &mut QualifiedPhysicalLaunch<'_>,
        graph: bool,
        salt: u64,
    ) -> Result<Vec<u32>, String> {
        forced.seed_f32_operands(forced_ctx, salt)?;
        auto.seed_f32_operands(auto_ctx, salt)?;
        let forced_inputs = forced.f32_operand_bits(forced_ctx)?;
        if auto.f32_operand_bits(auto_ctx)? != forced_inputs {
            return Err(format!("{} seeded A/B differ between contexts", case.name));
        }

        if graph {
            forced.measure_graph_window_ms(forced_ctx, 1)?;
            auto.measure_graph_window_ms(auto_ctx, 1)?;
        } else {
            forced.measure_eager_window_ms(forced_ctx, 1)?;
            auto.measure_eager_window_ms(auto_ctx, 1)?;
        }
        let forced_bits = forced.f32_output_bits(forced_ctx)?;
        let auto_bits = auto.f32_output_bits(auto_ctx)?;
        if forced_bits != auto_bits {
            return Err(format!(
                "{} {} bits differ from prior public AUTO",
                case.name,
                if graph { "graph" } else { "eager" },
            ));
        }
        if forced.f32_operand_bits(forced_ctx)? != forced_inputs
            || auto.f32_operand_bits(auto_ctx)? != forced_inputs
        {
            return Err(format!(
                "{} {} launch modified A or B",
                case.name,
                if graph { "graph" } else { "eager" },
            ));
        }
        Ok(forced_bits)
    }

    #[test]
    #[ignore = "requires an idle RTX 6000 Ada and explicit CUDA toolkit selection"]
    fn sm89_tf32_joint_six_cell_forced_wiring_matches_prior_auto() -> Result<(), String> {
        let forced_device = GpuDevice::new(0)?;
        let forced_ctx = GpuCtx::new(&forced_device)?;
        let auto_device = GpuDevice::new(0)?;
        let auto_ctx = GpuCtx::new(&auto_device)?;

        let identity = forced_ctx.gemm_route();
        if identity.device.compute_capability != (8, 9)
            || identity.device.multiprocessor_count != 142
        {
            return Err(format!(
                "TF32 joint wiring smoke requires CC8.9/142SM, got {:?}/{}SM",
                identity.device.compute_capability, identity.device.multiprocessor_count,
            ));
        }
        let compiler = forced_ctx
            .kernels
            .triad_sm89_tf32_joint_compiler_identity()
            .ok_or_else(|| {
                format!(
                    "TriadSm89Tf32Joint did not bind: {:?}",
                    forced_ctx.kernels.triad_sm89_tf32_joint_rejection(),
                )
            })?;
        if compiler.nvrtc_version != (13, 2) {
            return Err(format!(
                "M64N96/S2 wiring smoke requires CUDA 13.2, got {:?}",
                compiler.nvrtc_version,
            ));
        }

        let forced_requests = CASES.map(|case| request(case, true));
        let auto_requests = CASES.map(|case| request(case, false));
        presize_physical_qualification_suite(&forced_ctx, &forced_requests)?;
        presize_physical_qualification_suite(&auto_ctx, &auto_requests)?;

        for (index, case) in CASES.into_iter().enumerate() {
            let forced_request = forced_requests[index];
            let auto_request = auto_requests[index];
            let mut forced = qualify_physical_launch(&forced_ctx, forced_request)?;
            let mut auto = qualify_physical_launch(&auto_ctx, auto_request)?;
            assert_forced_manifest(case, &forced)?;
            assert_prior_auto_manifest(case, &auto)?;

            let mut repeated = None;
            for (path_index, graph) in [false, false, true, true].into_iter().enumerate() {
                let bits = compare_one_path(
                    case,
                    &forced_ctx,
                    &mut forced,
                    &auto_ctx,
                    &mut auto,
                    graph,
                    0x89_7f_3200_u64 ^ index as u64,
                )?;
                if let Some(expected) = &repeated {
                    if &bits != expected {
                        return Err(format!(
                            "{} path {path_index} is not bit-repeatable",
                            case.name,
                        ));
                    }
                } else {
                    repeated = Some(bits);
                }
            }

            let forced_guards = forced.validate_red_zones(&forced_ctx)?;
            let auto_guards = auto.validate_red_zones(&auto_ctx)?;
            if forced_guards.allocation_count() < 3
                || forced_guards.element_count() == 0
                || auto_guards.allocation_count() < 3
                || auto_guards.element_count() == 0
            {
                return Err(format!("{} did not validate all facade guards", case.name));
            }

            println!(
                "{{\"schema\":\"MambaTriadSm89Tf32JointWiringSmokeV1\",\"case\":\"{}\",\"dims\":[{},{},{}],\"forced_symbol\":\"{}\",\"prior_auto_symbol\":\"{}\",\"forced_nodes\":{},\"eager_graph_manifest_equal\":true,\"prior_auto_bits_equal\":true,\"repeatable\":true,\"inputs_unchanged\":true,\"guards_checked\":true}}",
                case.name,
                case.dims.0,
                case.dims.1,
                case.dims.2,
                case.gemm_symbol,
                case.old_auto_symbol,
                if case.op == LiteralOp::Tn { 2 } else { 1 },
            );

            drop(auto);
            drop(forced);
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires an idle RTX 6000 Ada and explicit CUDA toolkit selection"]
    fn sm89_tf32_joint_post_admission_auto_uses_the_toolkit_winner_map() -> Result<(), String> {
        let auto_device = GpuDevice::new(0)?;
        let auto_ctx = GpuCtx::new(&auto_device)?;
        let prior_device = GpuDevice::new(0)?;
        let prior_ctx = GpuCtx::new(&prior_device)?;
        let identity = auto_ctx.gemm_route();
        if identity.device.compute_capability != (8, 9)
            || identity.device.multiprocessor_count != 142
        {
            return Err(format!(
                "TF32 joint post-admission smoke requires CC8.9/142SM, got {:?}/{}SM",
                identity.device.compute_capability, identity.device.multiprocessor_count,
            ));
        }
        let nvrtc = auto_ctx
            .kernels
            .triad_sm89_tf32_joint_compiler_identity()
            .ok_or_else(|| "TriadSm89Tf32Joint compiler identity is absent".to_string())?
            .nvrtc_version;
        if !matches!(nvrtc, (12, 8) | (13, 0) | (13, 2)) {
            return Err(format!("no frozen post-admission map for CUDA {nvrtc:?}"));
        }

        let auto_requests = CASES.map(auto_request);
        let prior_requests = CASES.map(|case| request(case, false));
        presize_physical_qualification_suite(&auto_ctx, &auto_requests)?;
        presize_physical_qualification_suite(&prior_ctx, &prior_requests)?;

        for (index, case) in CASES.into_iter().enumerate() {
            let mut auto = qualify_physical_launch(&auto_ctx, auto_requests[index])?;
            let mut prior = qualify_physical_launch(&prior_ctx, prior_requests[index])?;
            let expected_case = auto_expected_case(case, nvrtc);
            let (expected_symbol, expected_module) = post_admission_symbol_and_module(case, nvrtc);
            let keeps_prior = expected_symbol == case.old_auto_symbol;
            if keeps_prior {
                assert_prior_auto_manifest(case, &auto)?;
            } else {
                assert_forced_manifest(expected_case, &auto)?;
            }
            if auto
                .evidence()
                .uniform_module_kind()
                .and_then(literal_module)
                != Some(expected_module)
            {
                return Err(format!("{} AUTO module drifted", case.name));
            }
            assert_prior_auto_manifest(case, &prior)?;

            let mut repeated = None;
            for (path_index, graph) in [false, false, true, true].into_iter().enumerate() {
                let bits = compare_one_path(
                    case,
                    &auto_ctx,
                    &mut auto,
                    &prior_ctx,
                    &mut prior,
                    graph,
                    0x89_7f_32a0_u64 ^ index as u64,
                )?;
                if let Some(expected) = &repeated {
                    if &bits != expected {
                        return Err(format!(
                            "{} admitted AUTO path {path_index} is not bit-repeatable",
                            case.name,
                        ));
                    }
                } else {
                    repeated = Some(bits);
                }
            }

            let auto_guards = auto.validate_red_zones(&auto_ctx)?;
            let prior_guards = prior.validate_red_zones(&prior_ctx)?;
            if auto_guards.allocation_count() < 3
                || auto_guards.element_count() == 0
                || prior_guards.allocation_count() < 3
                || prior_guards.element_count() == 0
            {
                return Err(format!("{} did not validate all facade guards", case.name));
            }
            println!(
                "{{\"schema\":\"MambaTriadSm89Tf32JointPostAdmissionV1\",\"cuda\":[{},{}],\"case\":\"{}\",\"auto_symbol\":\"{}\",\"module\":\"{}\",\"eager_graph_manifest_equal\":true,\"prior_bits_equal\":true,\"repeatable\":true,\"inputs_unchanged\":true,\"guards_checked\":true}}",
                nvrtc.0,
                nvrtc.1,
                case.name,
                expected_symbol,
                expected_module.as_str(),
            );
        }
        Ok(())
    }
}
