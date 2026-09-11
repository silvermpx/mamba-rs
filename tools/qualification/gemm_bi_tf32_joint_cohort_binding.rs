//! Performance qualification, before admission, for the retained Ada TF32 joint cells.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PairOrder {
    Abba,
    Baab,
}

impl PairOrder {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Abba => "abba",
            Self::Baab => "baab",
        }
    }
}

fn paired_ratio(order: PairOrder, raw_us: [f64; 4]) -> Result<f64, String> {
    if raw_us
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(format!("invalid paired timing sample {raw_us:?}"));
    }
    let (candidate_us, comparator_us) = match order {
        PairOrder::Abba => ((raw_us[0] + raw_us[3]) * 0.5, (raw_us[1] + raw_us[2]) * 0.5),
        PairOrder::Baab => ((raw_us[1] + raw_us[2]) * 0.5, (raw_us[0] + raw_us[3]) * 0.5),
    };
    let ratio = candidate_us / comparator_us;
    if !ratio.is_finite() || ratio <= 0.0 {
        return Err(format!("invalid candidate/comparator ratio {ratio}"));
    }
    Ok(ratio)
}

fn nearest_rank(values: &[f64], quantile: f64) -> Result<f64, String> {
    if values.is_empty() || !(0.0..=1.0).contains(&quantile) {
        return Err("nearest-rank requires samples and a quantile in [0,1]".into());
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err("nearest-rank rejects nonfinite samples".into());
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let rank = (quantile * sorted.len() as f64).ceil() as usize;
    Ok(sorted[rank.clamp(1, sorted.len()) - 1])
}

#[test]
fn paired_ratio_and_nearest_rank_are_order_unambiguous() {
    assert_eq!(
        paired_ratio(PairOrder::Abba, [8.0, 10.0, 10.0, 8.0]).unwrap(),
        0.8
    );
    assert_eq!(
        paired_ratio(PairOrder::Baab, [10.0, 8.0, 8.0, 10.0]).unwrap(),
        0.8
    );
    assert_eq!(nearest_rank(&[0.8, 0.9, 0.7], 0.50).unwrap(), 0.8);
    assert_eq!(nearest_rank(&[0.8, 0.9, 0.7], 0.95).unwrap(), 0.9);
    assert!(paired_ratio(PairOrder::Abba, [0.0, 1.0, 1.0, 1.0]).is_err());
    assert!(nearest_rank(&[], 0.50).is_err());
}

#[cfg(feature = "cuda")]
mod live {
    use super::*;
    use mamba_rs::mamba_ssm::gpu::GemmMode;
    use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, F32TriadPolicy, GpuCtx};
    use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
    use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
        PhysicalQualificationF32Epilogue, PhysicalQualificationRequest, PhysicalQualificationRoute,
        QualifiedPhysicalLaunch, Tf32PhysicalRoute, Tf32PortableRoute, Tf32PortableStages,
        Tf32PortableTile, presize_physical_qualification_suite, qualify_physical_launch,
    };
    use mamba_rs::mamba_ssm::gpu::kernel_identity::{ModuleKind, ResolvedGemmOp, digest_hex};

    const SCREEN_WINDOWS: usize = 3;
    const OFFICIAL_WINDOWS: usize = 7;
    const TARGET_WINDOW_US: f64 = 40_000.0;
    const MAX_ITERATIONS: usize = 4_096;
    const ADMISSION_RATIO: f64 = 0.99;

    #[derive(Clone, Copy, Debug)]
    enum Path {
        Eager,
        Graph,
    }

    impl Path {
        const fn as_str(self) -> &'static str {
            match self {
                Self::Eager => "eager",
                Self::Graph => "graph",
            }
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct Case {
        name: &'static str,
        op: ResolvedGemmOp,
        dims: (usize, usize, usize),
        route: Tf32PhysicalRoute,
        candidate_symbol: &'static str,
        prior_route: Tf32PhysicalRoute,
        prior_symbol: &'static str,
    }

    const CASES: [Case; 5] = [
        Case {
            name: "tn_d768_in",
            op: ResolvedGemmOp::Tn,
            dims: (2_048, 768, 3_072),
            route: Tf32PhysicalRoute::Sm89TnPreRnaN96V1,
            candidate_symbol: "gemm_bi_tn_sm89_tf32_pre_rna_m128n96_bk32_s3_v1",
            prior_route: Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                tile: Tf32PortableTile::M64N64,
                stages: Tf32PortableStages::S2,
            }),
            prior_symbol: "gemm_bi_tn_sm80_mma_tf32_v1_m64n64_bk32_s2",
        },
        Case {
            name: "tn_d768_out",
            op: ResolvedGemmOp::Tn,
            dims: (2_048, 1_536, 768),
            route: Tf32PhysicalRoute::Sm89TnPreRnaN96V1,
            candidate_symbol: "gemm_bi_tn_sm89_tf32_pre_rna_m128n96_bk32_s3_v1",
            prior_route: Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                tile: Tf32PortableTile::M64N64,
                stages: Tf32PortableStages::S2,
            }),
            prior_symbol: "gemm_bi_tn_sm80_mma_tf32_v1_m64n64_bk32_s2",
        },
        Case {
            name: "tn_prism",
            op: ResolvedGemmOp::Tn,
            dims: (4_621, 384, 1_928),
            route: Tf32PhysicalRoute::Sm89TnPreRnaM64N64V1,
            candidate_symbol: "gemm_bi_tn_sm89_tf32_pre_rna_m64n64_bk32_s3_v1",
            prior_route: Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                tile: Tf32PortableTile::M128N64,
                stages: Tf32PortableStages::S3,
            }),
            prior_symbol: "gemm_bi_tn_sm80_mma_tf32_v1_m128n64_bk32_s3",
        },
        Case {
            name: "nn_prism",
            op: ResolvedGemmOp::Nn,
            dims: (4_621, 384, 1_928),
            route: Tf32PhysicalRoute::Sm89NnDirectN96V1,
            candidate_symbol: "gemm_bi_nn_sm89_tf32_addhalf_m128n96_bk32_s3_direct_v1",
            prior_route: Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                tile: Tf32PortableTile::M128N128,
                stages: Tf32PortableStages::S3,
            }),
            prior_symbol: "gemm_bi_nn_sm80_mma_tf32_v1_m128n128_bk32_s3",
        },
        Case {
            name: "nn_d768_out",
            op: ResolvedGemmOp::Nn,
            dims: (2_048, 1_536, 768),
            route: Tf32PhysicalRoute::Sm89NnN96V1,
            candidate_symbol: "gemm_bi_nn_sm89_tf32_addhalf_m128n96_bk32_s3_v1",
            prior_route: Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                tile: Tf32PortableTile::M128N128,
                stages: Tf32PortableStages::S3,
            }),
            prior_symbol: "gemm_bi_nn_sm80_mma_tf32_v1_m128n128_bk32_s3",
        },
    ];

    const NEW_PRODUCTION_BODY_CASES: [Case; 2] = [
        Case {
            name: "nt_d768_in_new_body",
            op: ResolvedGemmOp::Nt,
            dims: (2_048, 768, 3_072),
            route: Tf32PhysicalRoute::Sm89NtALdmatrixN96V1,
            candidate_symbol: "gemm_bi_nt_sm89_tf32_a_ldmatrix_m128n96_bk32_s3_v1",
            prior_route: Tf32PhysicalRoute::Sm89MmaTf32Compact8V1,
            prior_symbol: "gemm_bi_nt_sm89_mma_tf32_compact8_v1_m128n64_bk32_s2",
        },
        Case {
            name: "tn_prism_new_body",
            op: ResolvedGemmOp::Tn,
            dims: (4_621, 384, 1_928),
            route: Tf32PhysicalRoute::Sm89TnPreRnaM64N96S2V1,
            candidate_symbol: "gemm_bi_tn_sm89_tf32_pre_rna_m64n96_bk32_s2_v1",
            prior_route: Tf32PhysicalRoute::Sm89TnPreRnaM64N64V1,
            prior_symbol: "gemm_bi_tn_sm89_tf32_pre_rna_m64n64_bk32_s3_v1",
        },
    ];

    /// Screens, not admissions: routes that already exist, tried on cells
    /// the tables never measured them on (the deep 4096-row products, the
    /// d128 weight gradients) and one retile of an admitted cell. Every
    /// ratio is recorded whether or not the candidate wins.
    const RETILE_AND_DEEP_CASES: [Case; 11] = [
        Case {
            name: "tn_d768_in_m64n96",
            op: ResolvedGemmOp::Tn,
            dims: (2_048, 768, 3_072),
            route: Tf32PhysicalRoute::Sm89TnPreRnaM64N96S2V1,
            candidate_symbol: "gemm_bi_tn_sm89_tf32_pre_rna_m64n96_bk32_s2_v1",
            prior_route: Tf32PhysicalRoute::Sm89TnPreRnaN96V1,
            prior_symbol: "gemm_bi_tn_sm89_tf32_pre_rna_m128n96_bk32_s3_v1",
        },
        Case {
            name: "tn_large_deep_n96",
            op: ResolvedGemmOp::Tn,
            dims: (4_096, 3_072, 1_536),
            route: Tf32PhysicalRoute::Sm89TnPreRnaN96V1,
            candidate_symbol: "gemm_bi_tn_sm89_tf32_pre_rna_m128n96_bk32_s3_v1",
            prior_route: Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                tile: Tf32PortableTile::M64N64,
                stages: Tf32PortableStages::S2,
            }),
            prior_symbol: "gemm_bi_tn_sm80_mma_tf32_v1_m64n64_bk32_s2",
        },
        Case {
            name: "tn_large_deep_m64n96",
            op: ResolvedGemmOp::Tn,
            dims: (4_096, 3_072, 1_536),
            route: Tf32PhysicalRoute::Sm89TnPreRnaM64N96S2V1,
            candidate_symbol: "gemm_bi_tn_sm89_tf32_pre_rna_m64n96_bk32_s2_v1",
            prior_route: Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                tile: Tf32PortableTile::M64N64,
                stages: Tf32PortableStages::S2,
            }),
            prior_symbol: "gemm_bi_tn_sm80_mma_tf32_v1_m64n64_bk32_s2",
        },
        Case {
            name: "tn_large_deep_m64n64",
            op: ResolvedGemmOp::Tn,
            dims: (4_096, 3_072, 1_536),
            route: Tf32PhysicalRoute::Sm89TnPreRnaM64N64V1,
            candidate_symbol: "gemm_bi_tn_sm89_tf32_pre_rna_m64n64_bk32_s3_v1",
            prior_route: Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                tile: Tf32PortableTile::M64N64,
                stages: Tf32PortableStages::S2,
            }),
            prior_symbol: "gemm_bi_tn_sm80_mma_tf32_v1_m64n64_bk32_s2",
        },
        Case {
            name: "nn_large_deep_n96",
            op: ResolvedGemmOp::Nn,
            dims: (4_096, 3_072, 1_536),
            route: Tf32PhysicalRoute::Sm89NnN96V1,
            candidate_symbol: "gemm_bi_nn_sm89_tf32_addhalf_m128n96_bk32_s3_v1",
            prior_route: Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                tile: Tf32PortableTile::M128N128,
                stages: Tf32PortableStages::S3,
            }),
            prior_symbol: "gemm_bi_nn_sm80_mma_tf32_v1_m128n128_bk32_s3",
        },
        Case {
            name: "nn_large_deep_direct_n96",
            op: ResolvedGemmOp::Nn,
            dims: (4_096, 3_072, 1_536),
            route: Tf32PhysicalRoute::Sm89NnDirectN96V1,
            candidate_symbol: "gemm_bi_nn_sm89_tf32_addhalf_m128n96_bk32_s3_direct_v1",
            prior_route: Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                tile: Tf32PortableTile::M128N128,
                stages: Tf32PortableStages::S3,
            }),
            prior_symbol: "gemm_bi_nn_sm80_mma_tf32_v1_m128n128_bk32_s3",
        },
        Case {
            name: "nt_large_deep_a_ldmatrix",
            op: ResolvedGemmOp::Nt,
            dims: (4_096, 3_072, 1_536),
            route: Tf32PhysicalRoute::Sm89NtALdmatrixN96V1,
            candidate_symbol: "gemm_bi_nt_sm89_tf32_a_ldmatrix_m128n96_bk32_s3_v1",
            prior_route: Tf32PhysicalRoute::Sm89MmaTf32Compact8V1,
            prior_symbol: "gemm_bi_nt_sm89_mma_tf32_compact8_v1_m128n64_bk32_s2",
        },
        Case {
            name: "tn_d128_in_splitk8_s3",
            op: ResolvedGemmOp::Tn,
            dims: (1_024, 128, 512),
            route: Tf32PhysicalRoute::MmaTf32RnaSplitK8V1(Tf32PortableRoute {
                tile: Tf32PortableTile::M32N32,
                stages: Tf32PortableStages::S3,
            }),
            candidate_symbol: "gemm_bi_tn_sm80_mma_tf32_splitk8_v1_m32n32_bk32_s3",
            prior_route: Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                tile: Tf32PortableTile::M16N32,
                stages: Tf32PortableStages::S4,
            }),
            prior_symbol: "gemm_bi_tn_sm80_mma_tf32_v1_m16n32_bk32_s4",
        },
        Case {
            name: "tn_d128_in_splitk8_s4",
            op: ResolvedGemmOp::Tn,
            dims: (1_024, 128, 512),
            route: Tf32PhysicalRoute::MmaTf32RnaSplitK8V1(Tf32PortableRoute {
                tile: Tf32PortableTile::M32N32,
                stages: Tf32PortableStages::S4,
            }),
            candidate_symbol: "gemm_bi_tn_sm80_mma_tf32_splitk8_v1_m32n32_bk32_s4",
            prior_route: Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                tile: Tf32PortableTile::M16N32,
                stages: Tf32PortableStages::S4,
            }),
            prior_symbol: "gemm_bi_tn_sm80_mma_tf32_v1_m16n32_bk32_s4",
        },
        Case {
            name: "tn_d128_out_splitk8_s3",
            op: ResolvedGemmOp::Tn,
            dims: (1_024, 256, 128),
            route: Tf32PhysicalRoute::MmaTf32RnaSplitK8V1(Tf32PortableRoute {
                tile: Tf32PortableTile::M32N32,
                stages: Tf32PortableStages::S3,
            }),
            candidate_symbol: "gemm_bi_tn_sm80_mma_tf32_splitk8_v1_m32n32_bk32_s3",
            prior_route: Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                tile: Tf32PortableTile::M16N32,
                stages: Tf32PortableStages::S4,
            }),
            prior_symbol: "gemm_bi_tn_sm80_mma_tf32_v1_m16n32_bk32_s4",
        },
        Case {
            name: "tn_d128_out_splitk8_s4",
            op: ResolvedGemmOp::Tn,
            dims: (1_024, 256, 128),
            route: Tf32PhysicalRoute::MmaTf32RnaSplitK8V1(Tf32PortableRoute {
                tile: Tf32PortableTile::M32N32,
                stages: Tf32PortableStages::S4,
            }),
            candidate_symbol: "gemm_bi_tn_sm80_mma_tf32_splitk8_v1_m32n32_bk32_s4",
            prior_route: Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                tile: Tf32PortableTile::M16N32,
                stages: Tf32PortableStages::S4,
            }),
            prior_symbol: "gemm_bi_tn_sm80_mma_tf32_v1_m16n32_bk32_s4",
        },
    ];

    /// The rebuilt NT body against the compact8 finalist on the cells the
    /// finalist serves on the lower toolkits (on CUDA 13.2 the d768 in_proj
    /// row is the production-body comparison above).
    const NT_BODY_CASES: [Case; 3] = [
        Case {
            name: "nt_d768_in_ldmatrix",
            op: ResolvedGemmOp::Nt,
            dims: (2_048, 768, 3_072),
            route: Tf32PhysicalRoute::Sm89NtALdmatrixN96V1,
            candidate_symbol: "gemm_bi_nt_sm89_tf32_a_ldmatrix_m128n96_bk32_s3_v1",
            prior_route: Tf32PhysicalRoute::Sm89MmaTf32Compact8V1,
            prior_symbol: "gemm_bi_nt_sm89_mma_tf32_compact8_v1_m128n64_bk32_s2",
        },
        Case {
            name: "nt_d768_out_ldmatrix",
            op: ResolvedGemmOp::Nt,
            dims: (2_048, 1_536, 768),
            route: Tf32PhysicalRoute::Sm89NtALdmatrixN96V1,
            candidate_symbol: "gemm_bi_nt_sm89_tf32_a_ldmatrix_m128n96_bk32_s3_v1",
            prior_route: Tf32PhysicalRoute::Sm89MmaTf32Compact8V1,
            prior_symbol: "gemm_bi_nt_sm89_mma_tf32_compact8_v1_m128n64_bk32_s2",
        },
        Case {
            name: "nt_prism_ldmatrix",
            op: ResolvedGemmOp::Nt,
            dims: (4_621, 384, 1_928),
            route: Tf32PhysicalRoute::Sm89NtALdmatrixN96V1,
            candidate_symbol: "gemm_bi_nt_sm89_tf32_a_ldmatrix_m128n96_bk32_s3_v1",
            prior_route: Tf32PhysicalRoute::Sm89MmaTf32Compact8V1,
            prior_symbol: "gemm_bi_nt_sm89_mma_tf32_compact8_v1_m128n64_bk32_s2",
        },
    ];

    struct PhaseResult {
        ratios: Vec<f64>,
        candidate_p50_us: f64,
        comparator_p50_us: f64,
        ratio_p50: f64,
        ratio_p95: f64,
        candidate_iterations: usize,
        comparator_iterations: usize,
    }

    fn configure(ctx: &GpuCtx) {
        ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
        ctx.set_bi_gemm_family(BiGemmFamily::Triad);
        ctx.set_f32_triad_policy(F32TriadPolicy::AllowDeterministicTf32V1);
    }

    fn request(case: Case, candidate: bool) -> PhysicalQualificationRequest {
        PhysicalQualificationRequest::contiguous_f32(
            case.op,
            case.dims,
            if candidate {
                PhysicalQualificationRoute::Tf32Forced(case.route)
            } else {
                PhysicalQualificationRoute::Tf32Forced(case.prior_route)
            },
            PhysicalQualificationF32Epilogue::new(
                1.0,
                if case.op == ResolvedGemmOp::Tn {
                    1.0
                } else {
                    0.0
                },
                false,
            ),
        )
    }

    fn measure_us(
        ctx: &GpuCtx,
        launch: &mut QualifiedPhysicalLaunch<'_>,
        path: Path,
        iterations: usize,
    ) -> Result<f64, String> {
        let milliseconds = match path {
            Path::Eager => launch.measure_prevalidated_forced_eager_window_ms(ctx, iterations)?,
            Path::Graph => launch.measure_graph_window_ms(ctx, iterations)?,
        };
        let microseconds = milliseconds * 1_000.0 / iterations as f64;
        if !microseconds.is_finite() || microseconds <= 0.0 {
            return Err(format!("invalid {} sample {microseconds}", path.as_str()));
        }
        Ok(microseconds)
    }

    fn calibrate(
        ctx: &GpuCtx,
        launch: &mut QualifiedPhysicalLaunch<'_>,
        path: Path,
    ) -> Result<usize, String> {
        measure_us(ctx, launch, path, 2)?;
        let pilot_us = measure_us(ctx, launch, path, 4)?;
        let iterations = (TARGET_WINDOW_US / pilot_us).ceil() as usize;
        Ok(iterations.clamp(1, MAX_ITERATIONS))
    }

    /// The two timed arms of one pre-admission pair: the candidate route
    /// and its prior comparator, each on its own context.
    struct PairArms<'a, 'b> {
        candidate_ctx: &'a GpuCtx,
        candidate: &'a mut QualifiedPhysicalLaunch<'b>,
        comparator_ctx: &'a GpuCtx,
        comparator: &'a mut QualifiedPhysicalLaunch<'b>,
    }

    fn measure_raw_pair(
        arms: &mut PairArms<'_, '_>,
        path: Path,
        order: PairOrder,
        iterations: (usize, usize),
    ) -> Result<[f64; 4], String> {
        let (candidate_iterations, comparator_iterations) = iterations;
        match order {
            PairOrder::Abba => Ok([
                measure_us(
                    arms.candidate_ctx,
                    arms.candidate,
                    path,
                    candidate_iterations,
                )?,
                measure_us(
                    arms.comparator_ctx,
                    arms.comparator,
                    path,
                    comparator_iterations,
                )?,
                measure_us(
                    arms.comparator_ctx,
                    arms.comparator,
                    path,
                    comparator_iterations,
                )?,
                measure_us(
                    arms.candidate_ctx,
                    arms.candidate,
                    path,
                    candidate_iterations,
                )?,
            ]),
            PairOrder::Baab => Ok([
                measure_us(
                    arms.comparator_ctx,
                    arms.comparator,
                    path,
                    comparator_iterations,
                )?,
                measure_us(
                    arms.candidate_ctx,
                    arms.candidate,
                    path,
                    candidate_iterations,
                )?,
                measure_us(
                    arms.candidate_ctx,
                    arms.candidate,
                    path,
                    candidate_iterations,
                )?,
                measure_us(
                    arms.comparator_ctx,
                    arms.comparator,
                    path,
                    comparator_iterations,
                )?,
            ]),
        }
    }

    fn run_phase(
        case: Case,
        arms: &mut PairArms<'_, '_>,
        path: Path,
        order: PairOrder,
        phase: &str,
        windows: usize,
        salt: u64,
    ) -> Result<PhaseResult, String> {
        arms.candidate.seed_f32_operands(arms.candidate_ctx, salt)?;
        arms.comparator
            .seed_f32_operands(arms.comparator_ctx, salt)?;
        let expected_inputs = arms.candidate.f32_operand_bits(arms.candidate_ctx)?;
        if arms.comparator.f32_operand_bits(arms.comparator_ctx)? != expected_inputs {
            return Err(format!("{} timing inputs differ", case.name));
        }

        let candidate_iterations = calibrate(arms.candidate_ctx, arms.candidate, path)?;
        let comparator_iterations = calibrate(arms.comparator_ctx, arms.comparator, path)?;
        arms.candidate.seed_f32_operands(arms.candidate_ctx, salt)?;
        arms.comparator
            .seed_f32_operands(arms.comparator_ctx, salt)?;

        let mut ratios = Vec::with_capacity(windows);
        let mut candidate_us = Vec::with_capacity(windows);
        let mut comparator_us = Vec::with_capacity(windows);
        for _ in 0..windows {
            let raw = measure_raw_pair(
                arms,
                path,
                order,
                (candidate_iterations, comparator_iterations),
            )?;
            let ratio = paired_ratio(order, raw)?;
            let (candidate_sample, comparator_sample) = match order {
                PairOrder::Abba => ((raw[0] + raw[3]) * 0.5, (raw[1] + raw[2]) * 0.5),
                PairOrder::Baab => ((raw[1] + raw[2]) * 0.5, (raw[0] + raw[3]) * 0.5),
            };
            candidate_us.push(candidate_sample);
            comparator_us.push(comparator_sample);
            ratios.push(ratio);
        }

        if arms.candidate.f32_operand_bits(arms.candidate_ctx)? != expected_inputs
            || arms.comparator.f32_operand_bits(arms.comparator_ctx)? != expected_inputs
        {
            return Err(format!("{} timed launch modified A or B", case.name));
        }
        let candidate_guards = arms.candidate.validate_red_zones(arms.candidate_ctx)?;
        let comparator_guards = arms.comparator.validate_red_zones(arms.comparator_ctx)?;
        if candidate_guards.allocation_count() < 3
            || comparator_guards.allocation_count() < 3
            || candidate_guards.element_count() == 0
            || comparator_guards.element_count() == 0
        {
            return Err(format!("{} timing did not validate all guards", case.name));
        }

        let result = PhaseResult {
            ratio_p50: nearest_rank(&ratios, 0.50)?,
            ratio_p95: nearest_rank(&ratios, 0.95)?,
            candidate_p50_us: nearest_rank(&candidate_us, 0.50)?,
            comparator_p50_us: nearest_rank(&comparator_us, 0.50)?,
            ratios,
            candidate_iterations,
            comparator_iterations,
        };
        println!(
            "{{\"schema\":\"MambaTriadSm89Tf32JointPreAdmissionV1\",\"case\":\"{}\",\"dims\":[{},{},{}],\"path\":\"{}\",\"order\":\"{}\",\"phase\":\"{}\",\"windows\":{},\"candidate_iterations\":{},\"comparator_iterations\":{},\"candidate_p50_us\":{:.9},\"comparator_p50_us\":{:.9},\"ratio_direction\":\"candidate_over_prior_portable\",\"ratios\":{:?},\"ratio_p50\":{:.9},\"ratio_p95\":{:.9},\"pass\":{}}}",
            case.name,
            case.dims.0,
            case.dims.1,
            case.dims.2,
            path.as_str(),
            order.as_str(),
            phase,
            windows,
            result.candidate_iterations,
            result.comparator_iterations,
            result.candidate_p50_us,
            result.comparator_p50_us,
            result.ratios,
            result.ratio_p50,
            result.ratio_p95,
            result.ratio_p50 < ADMISSION_RATIO && result.ratio_p95 < ADMISSION_RATIO,
        );
        Ok(result)
    }

    fn phase_passed(result: &PhaseResult) -> bool {
        result.ratio_p50 < ADMISSION_RATIO && result.ratio_p95 < ADMISSION_RATIO
    }

    fn expected_rejections(nvrtc: (i32, i32)) -> Result<&'static [&'static str], String> {
        match nvrtc {
            (12, 8) | (13, 0) => Ok(&["nn_prism/eager screen", "nn_prism/graph screen"]),
            (13, 2) => Ok(&[]),
            _ => Err(format!(
                "no frozen joint pre-admission expectation for NVRTC {}.{}",
                nvrtc.0, nvrtc.1,
            )),
        }
    }

    #[test]
    #[ignore = "requires an Ada GPU and explicit CUDA toolkit selection"]
    fn sm89_portable_identity_inventory() -> Result<(), String> {
        let device = GpuDevice::new(0)?;
        if device.compute_capability != (8, 9) || device.multiprocessor_count() != 142 {
            return Err(format!(
                "portable identity inventory requires exact CC8.9/142SM, got {:?}/{}SM",
                device.compute_capability,
                device.multiprocessor_count(),
            ));
        }
        let ctx = GpuCtx::new(&device)?;
        let portable = ctx
            .kernels
            .f32_triad_availability()
            .portable
            .ok_or_else(|| "portable TF32 module identity is absent".to_string())?;
        println!(
            "{{\"schema\":\"MambaTriadPortableIdentityV1\",\"target\":\"{}\",\"nvrtc\":[{},{}],\"compile_key\":\"{}\",\"artifact_digest\":\"{}\",\"source_digest\":\"{}\",\"invocation_digest\":\"{}\",\"header_manifest_digest\":\"{}\",\"nvrtc_library_domain\":\"{}\",\"driver_build_sources\":{},\"driver_build_digest\":\"{}\"}}",
            portable.target.as_str(),
            portable.compiler.nvrtc_version.0,
            portable.compiler.nvrtc_version.1,
            digest_hex(&portable.artifact.compile_key),
            digest_hex(&portable.artifact.artifact_digest),
            digest_hex(&portable.compiler.source_digest),
            digest_hex(&portable.compiler.invocation_digest),
            digest_hex(&portable.compiler.header_manifest_digest),
            digest_hex(&portable.compiler.nvrtc_library_domain),
            portable.device.driver.build_sources,
            digest_hex(&portable.device.driver.build_digest),
        );
        Ok(())
    }

    /// The production-body protocol over `cases`, recording every ratio: a
    /// candidate that loses its once3 screen is reported and skipped, one
    /// that passes is confirmed with once7, and only a drifted binding fails
    /// the run. Any of the three Ada toolkits may run it.
    fn screen_cases_recording_every_ratio(cases: &[Case]) -> Result<(), String> {
        let candidate_device = GpuDevice::new(0)?;
        let comparator_device = GpuDevice::new(0)?;
        for device in [&candidate_device, &comparator_device] {
            if device.compute_capability != (8, 9) || device.multiprocessor_count() != 142 {
                return Err(format!(
                    "the screen requires exact CC8.9/142SM, got {:?}/{}SM",
                    device.compute_capability,
                    device.multiprocessor_count(),
                ));
            }
        }
        let candidate_ctx = GpuCtx::new(&candidate_device)?;
        let comparator_ctx = GpuCtx::new(&comparator_device)?;
        configure(&candidate_ctx);
        configure(&comparator_ctx);
        let nvrtc = candidate_ctx
            .kernels
            .triad_sm89_tf32_joint_compiler_identity()
            .ok_or_else(|| "TriadSm89Tf32Joint compiler identity is absent".to_string())?
            .nvrtc_version;
        let candidate_requests: Vec<_> = cases.iter().map(|case| request(*case, true)).collect();
        let comparator_requests: Vec<_> = cases.iter().map(|case| request(*case, false)).collect();
        presize_physical_qualification_suite(&candidate_ctx, &candidate_requests)?;
        presize_physical_qualification_suite(&comparator_ctx, &comparator_requests)?;
        for (index, case) in cases.iter().copied().enumerate() {
            let mut candidate = qualify_physical_launch(&candidate_ctx, candidate_requests[index])?;
            let mut comparator =
                qualify_physical_launch(&comparator_ctx, comparator_requests[index])?;
            let candidate_evidence = candidate.evidence();
            let comparator_evidence = comparator.evidence();
            if !candidate_evidence.eager_graph_equal()
                || candidate_evidence.uniform_module_kind() != Some(case.route.module_kind())
                || candidate_evidence.nodes().last().map(|node| node.symbol)
                    != Some(case.candidate_symbol)
                || !comparator_evidence.eager_graph_equal()
                || comparator_evidence.uniform_module_kind() != Some(case.prior_route.module_kind())
                || comparator_evidence.nodes().last().map(|node| node.symbol)
                    != Some(case.prior_symbol)
            {
                return Err(format!(
                    "{} screen binding drifted: candidate={:?} comparator={:?}",
                    case.name,
                    candidate_evidence.nodes(),
                    comparator_evidence.nodes(),
                ));
            }
            for path in [Path::Eager, Path::Graph] {
                let mut screen_ratios = Vec::new();
                let mut screens_pass = true;
                for order in [PairOrder::Abba, PairOrder::Baab] {
                    let screen = run_phase(
                        case,
                        &mut PairArms {
                            candidate_ctx: &candidate_ctx,
                            candidate: &mut candidate,
                            comparator_ctx: &comparator_ctx,
                            comparator: &mut comparator,
                        },
                        path,
                        order,
                        "retile_deep_screen_once3",
                        SCREEN_WINDOWS,
                        0x897f_3330_u64 ^ index as u64,
                    )?;
                    screens_pass &= phase_passed(&screen);
                    screen_ratios.push((
                        screen.ratio_p50,
                        screen.ratio_p95,
                        screen.candidate_p50_us,
                        screen.comparator_p50_us,
                    ));
                }
                let mut official_ratios = Vec::new();
                let mut officials_pass = screens_pass;
                if screens_pass {
                    for order in [PairOrder::Abba, PairOrder::Baab] {
                        let official = run_phase(
                            case,
                            &mut PairArms {
                                candidate_ctx: &candidate_ctx,
                                candidate: &mut candidate,
                                comparator_ctx: &comparator_ctx,
                                comparator: &mut comparator,
                            },
                            path,
                            order,
                            "retile_deep_official_once7",
                            OFFICIAL_WINDOWS,
                            0x897f_3370_u64 ^ index as u64,
                        )?;
                        officials_pass &= phase_passed(&official);
                        official_ratios.push((
                            official.ratio_p50,
                            official.ratio_p95,
                            official.candidate_p50_us,
                            official.comparator_p50_us,
                        ));
                    }
                }
                println!(
                    "{{\"schema\":\"MambaTriadSm89Tf32RetileDeepScreenV1\",\"case\":\"{}\",\"op\":\"{:?}\",\"dims\":{:?},\"nvrtc\":[{},{}],\"candidate\":\"{}\",\"prior\":\"{}\",\"path\":\"{}\",\"screen_once3\":{:?},\"official_once7\":{:?},\"retain\":{}}}",
                    case.name,
                    case.op,
                    case.dims,
                    nvrtc.0,
                    nvrtc.1,
                    case.candidate_symbol,
                    case.prior_symbol,
                    path.as_str(),
                    screen_ratios,
                    official_ratios,
                    officials_pass,
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires an idle RTX 6000 Ada on CUDA 12.8/13.0/13.2; records every ratio of the retiled and deep cells"]
    fn sm89_tf32_screens_retiled_and_deep_cells_once3_then_once7() -> Result<(), String> {
        screen_cases_recording_every_ratio(&RETILE_AND_DEEP_CASES)
    }

    #[test]
    #[ignore = "requires an idle RTX 6000 Ada on CUDA 12.8/13.0/13.2; the NT body on the compact8 cells"]
    fn sm89_tf32_screens_nt_body_on_the_compact8_cells_once3_then_once7() -> Result<(), String> {
        screen_cases_recording_every_ratio(&NT_BODY_CASES)
    }

    #[test]
    #[ignore = "requires an idle RTX 6000 Ada with CUDA 13.2"]
    fn sm89_tf32_joint_two_new_production_bodies_once3_then_once7() -> Result<(), String> {
        let candidate_device = GpuDevice::new(0)?;
        let comparator_device = GpuDevice::new(0)?;
        for device in [&candidate_device, &comparator_device] {
            if device.compute_capability != (8, 9) || device.multiprocessor_count() != 142 {
                return Err(format!(
                    "new-body timing requires exact CC8.9/142SM, got {:?}/{}SM",
                    device.compute_capability,
                    device.multiprocessor_count(),
                ));
            }
        }
        let candidate_ctx = GpuCtx::new(&candidate_device)?;
        let comparator_ctx = GpuCtx::new(&comparator_device)?;
        configure(&candidate_ctx);
        configure(&comparator_ctx);
        let nvrtc = candidate_ctx
            .kernels
            .triad_sm89_tf32_joint_compiler_identity()
            .ok_or_else(|| "TriadSm89Tf32Joint compiler identity is absent".to_string())?
            .nvrtc_version;
        if nvrtc != (13, 2) {
            return Err(format!("new-body timing requires CUDA 13.2, got {nvrtc:?}"));
        }

        let candidate_requests = NEW_PRODUCTION_BODY_CASES.map(|case| request(case, true));
        let comparator_requests = NEW_PRODUCTION_BODY_CASES.map(|case| request(case, false));
        presize_physical_qualification_suite(&candidate_ctx, &candidate_requests)?;
        presize_physical_qualification_suite(&comparator_ctx, &comparator_requests)?;

        for (index, case) in NEW_PRODUCTION_BODY_CASES.into_iter().enumerate() {
            let mut candidate = qualify_physical_launch(&candidate_ctx, candidate_requests[index])?;
            let mut comparator =
                qualify_physical_launch(&comparator_ctx, comparator_requests[index])?;
            let candidate_evidence = candidate.evidence();
            let comparator_evidence = comparator.evidence();
            if !candidate_evidence.eager_graph_equal()
                || candidate_evidence.uniform_module_kind() != Some(case.route.module_kind())
                || candidate_evidence.nodes().last().map(|node| node.symbol)
                    != Some(case.candidate_symbol)
                || !comparator_evidence.eager_graph_equal()
                || comparator_evidence.uniform_module_kind() != Some(case.prior_route.module_kind())
                || comparator_evidence.nodes().last().map(|node| node.symbol)
                    != Some(case.prior_symbol)
            {
                return Err(format!(
                    "{} production binding drifted: candidate={:?} comparator={:?}",
                    case.name,
                    candidate_evidence.nodes(),
                    comparator_evidence.nodes(),
                ));
            }

            for path in [Path::Eager, Path::Graph] {
                let mut screens_pass = true;
                for order in [PairOrder::Abba, PairOrder::Baab] {
                    let screen = run_phase(
                        case,
                        &mut PairArms {
                            candidate_ctx: &candidate_ctx,
                            candidate: &mut candidate,
                            comparator_ctx: &comparator_ctx,
                            comparator: &mut comparator,
                        },
                        path,
                        order,
                        "production_screen_once3",
                        SCREEN_WINDOWS,
                        0x897f_3230_u64 ^ index as u64,
                    )?;
                    screens_pass &= phase_passed(&screen);
                }
                if !screens_pass {
                    return Err(format!(
                        "{} {} production screen once3 failed",
                        case.name,
                        path.as_str(),
                    ));
                }
                for order in [PairOrder::Abba, PairOrder::Baab] {
                    let official = run_phase(
                        case,
                        &mut PairArms {
                            candidate_ctx: &candidate_ctx,
                            candidate: &mut candidate,
                            comparator_ctx: &comparator_ctx,
                            comparator: &mut comparator,
                        },
                        path,
                        order,
                        "production_official_once7",
                        OFFICIAL_WINDOWS,
                        0x897f_3270_u64 ^ index as u64,
                    )?;
                    if !phase_passed(&official) {
                        return Err(format!(
                            "{} {}/{} production official once7 failed",
                            case.name,
                            path.as_str(),
                            order.as_str(),
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires an idle RTX 6000 Ada and explicit CUDA 12.8/13.0/13.2 selection"]
    fn sm89_tf32_joint_qualifies_winning_cells_once3_then_once7() -> Result<(), String> {
        let candidate_device = GpuDevice::new(0)?;
        let comparator_device = GpuDevice::new(0)?;
        for device in [&candidate_device, &comparator_device] {
            if device.compute_capability != (8, 9) || device.multiprocessor_count() != 142 {
                return Err(format!(
                    "joint timing requires exact CC8.9/142SM, got {:?}/{}SM",
                    device.compute_capability,
                    device.multiprocessor_count(),
                ));
            }
        }
        let candidate_ctx = GpuCtx::new(&candidate_device)?;
        let comparator_ctx = GpuCtx::new(&comparator_device)?;
        configure(&candidate_ctx);
        configure(&comparator_ctx);

        let compiler = candidate_ctx
            .kernels
            .triad_sm89_tf32_joint_compiler_identity()
            .ok_or_else(|| "TriadSm89Tf32Joint compiler identity is absent".to_string())?;
        let artifact = candidate_ctx
            .kernels
            .triad_sm89_tf32_joint_artifact_identity()
            .ok_or_else(|| "TriadSm89Tf32Joint artifact identity is absent".to_string())?;
        let portable = candidate_ctx
            .kernels
            .f32_triad_availability()
            .portable
            .ok_or_else(|| "portable TF32 module identity is absent".to_string())?;
        println!(
            "{{\"schema\":\"MambaTriadPortableIdentityV1\",\"target\":\"{}\",\"nvrtc\":[{},{}],\"compile_key\":\"{}\",\"artifact_digest\":\"{}\",\"source_digest\":\"{}\",\"invocation_digest\":\"{}\",\"header_manifest_digest\":\"{}\",\"nvrtc_library_domain\":\"{}\",\"driver_build_sources\":{},\"driver_build_digest\":\"{}\"}}",
            portable.target.as_str(),
            portable.compiler.nvrtc_version.0,
            portable.compiler.nvrtc_version.1,
            digest_hex(&portable.artifact.compile_key),
            digest_hex(&portable.artifact.artifact_digest),
            digest_hex(&portable.compiler.source_digest),
            digest_hex(&portable.compiler.invocation_digest),
            digest_hex(&portable.compiler.header_manifest_digest),
            digest_hex(&portable.compiler.nvrtc_library_domain),
            portable.device.driver.build_sources,
            digest_hex(&portable.device.driver.build_digest),
        );
        let candidate_requests = CASES.map(|case| request(case, true));
        let comparator_requests = CASES.map(|case| request(case, false));
        presize_physical_qualification_suite(&candidate_ctx, &candidate_requests)?;
        presize_physical_qualification_suite(&comparator_ctx, &comparator_requests)?;

        let mut official_rows = 0usize;
        let mut failures = Vec::new();
        for (index, case) in CASES.into_iter().enumerate() {
            let mut candidate = qualify_physical_launch(&candidate_ctx, candidate_requests[index])?;
            let mut comparator =
                qualify_physical_launch(&comparator_ctx, comparator_requests[index])?;
            let candidate_evidence = candidate.evidence();
            if !candidate_evidence.eager_graph_equal()
                || candidate_evidence.uniform_module_kind() != Some(ModuleKind::TriadSm89Tf32Joint)
                || candidate_evidence.nodes().last().map(|node| node.symbol)
                    != Some(case.candidate_symbol)
                || candidate_evidence.launch_count()
                    != if case.op == ResolvedGemmOp::Tn { 2 } else { 1 }
            {
                return Err(format!(
                    "{} candidate physical manifest drifted: {:?}",
                    case.name,
                    candidate_evidence.nodes(),
                ));
            }
            let comparator_evidence = comparator.evidence();
            if !comparator_evidence.eager_graph_equal()
                || comparator_evidence.uniform_module_kind() != Some(ModuleKind::TriadSm80)
                || comparator_evidence.nodes().last().map(|node| node.symbol)
                    != Some(case.prior_symbol)
            {
                return Err(format!(
                    "{} prior portable comparator drifted: {:?}",
                    case.name,
                    comparator_evidence.nodes(),
                ));
            }
            let comparator_symbols = comparator_evidence
                .nodes()
                .iter()
                .map(|node| node.symbol)
                .collect::<Vec<_>>();
            println!(
                "{{\"schema\":\"MambaTriadSm89Tf32JointPreAdmissionBindingV1\",\"case\":\"{}\",\"candidate_symbol\":\"{}\",\"candidate_nodes\":{},\"prior_symbols\":{:?},\"prior_nodes\":{}}}",
                case.name,
                case.candidate_symbol,
                candidate_evidence.launch_count(),
                comparator_symbols,
                comparator_evidence.launch_count(),
            );

            for path in [Path::Eager, Path::Graph] {
                let mut screens_pass = true;
                for order in [PairOrder::Abba, PairOrder::Baab] {
                    let screen = run_phase(
                        case,
                        &mut PairArms {
                            candidate_ctx: &candidate_ctx,
                            candidate: &mut candidate,
                            comparator_ctx: &comparator_ctx,
                            comparator: &mut comparator,
                        },
                        path,
                        order,
                        "screen3",
                        SCREEN_WINDOWS,
                        0x897f_3200_u64 ^ index as u64,
                    )?;
                    screens_pass &= phase_passed(&screen);
                }
                if !screens_pass {
                    failures.push(format!("{}/{} screen", case.name, path.as_str()));
                    continue;
                }
                for order in [PairOrder::Abba, PairOrder::Baab] {
                    let official = run_phase(
                        case,
                        &mut PairArms {
                            candidate_ctx: &candidate_ctx,
                            candidate: &mut candidate,
                            comparator_ctx: &comparator_ctx,
                            comparator: &mut comparator,
                        },
                        path,
                        order,
                        "official7",
                        OFFICIAL_WINDOWS,
                        0x897f_3270_u64 ^ index as u64,
                    )?;
                    if phase_passed(&official) {
                        official_rows += 1;
                    } else {
                        failures.push(format!(
                            "{}/{}/{} official",
                            case.name,
                            path.as_str(),
                            order.as_str(),
                        ));
                    }
                }
            }
        }

        let expected_rejections = expected_rejections(compiler.nvrtc_version)?;
        let expected_official_rows = 20usize
            .checked_sub(expected_rejections.len() * 2)
            .ok_or_else(|| "invalid expected joint rejection count".to_string())?;
        println!(
            "{{\"schema\":\"MambaTriadSm89Tf32JointPreAdmissionCompletionV1\",\"target\":\"{}\",\"nvrtc\":[{},{}],\"compile_key\":\"{}\",\"artifact_digest\":\"{}\",\"source_digest\":\"{}\",\"invocation_digest\":\"{}\",\"header_manifest_digest\":\"{}\",\"nvrtc_library_domain\":\"{}\",\"cells\":{},\"official_rows\":{},\"expected_official_rows\":{},\"rejections\":{:?}}}",
            compiler.target.as_str(),
            compiler.nvrtc_version.0,
            compiler.nvrtc_version.1,
            digest_hex(&artifact.compile_key),
            digest_hex(&artifact.artifact_digest),
            digest_hex(&compiler.source_digest),
            digest_hex(&compiler.invocation_digest),
            digest_hex(&compiler.header_manifest_digest),
            digest_hex(&compiler.nvrtc_library_domain),
            CASES.len(),
            official_rows,
            expected_official_rows,
            failures,
        );
        let actual_rejections = failures.iter().map(String::as_str).collect::<Vec<_>>();
        if official_rows != expected_official_rows
            || actual_rejections.as_slice() != expected_rejections
        {
            return Err(format!(
                "joint pre-admission outcome drifted: rows={official_rows}/{expected_official_rows}, rejections={failures:?}/{expected_rejections:?}",
            ));
        }
        Ok(())
    }
}
