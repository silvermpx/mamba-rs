use super::super::dtype::WeightDtype;
use super::super::{
    context::F32TriadPolicy,
    kernel_identity::{
        ArtifactIdentity, ArtifactKind, COMPILER_REVISION, COMPOSER_REVISION, CompilerIdentity,
        ModuleKind, NUMERIC_ABI_REVISION, SCHEDULE_REVISION,
    },
};
use super::contract::{
    F32_TF32_TUNING_REVISION, F32TriadAvailability, F32TriadOperands, F32TriadRequest,
    F32TriadSelection, F32TriadShape, Sm90aForcedRoute, Sm90aOp, Sm90aShape,
    Sm90aWarpgroupSchedule, Sm100ForcedRoute, Sm100Op, Sm100TargetCandidate, Sm100TargetKind,
    Sm120Bk, Sm120FmaRoute, Sm120FmaTile, Sm120ForcedRoute, Sm120LaunchOperands, Sm120MapRequest,
    Sm120Op, Sm120PhysicalRoute, Sm120Schedule, Sm120Shape, Sm120Stages, Sm120TargetCandidate,
    Sm120Tile, Tf32PhysicalRoute, Tf32QualifiedModule, tf32_kernel_spec,
    validate_sm120_map_request,
};
use super::contract::{GemmDims, checked_mul3, checked_tile_grid, checked_usize};
use crate::mamba_ssm::gpu::context::HalfTriadPolicy;
use crate::mamba_ssm::gpu::kernel_identity::DeviceCaps;
use crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Tf32ExactShape {
    output_rows: usize,
    output_columns: usize,
    reduction: usize,
}

#[cfg(test)]
mod sm89_exact_f32_tn_admission_tests {
    use super::*;
    use crate::mamba_ssm::gpu::gemm_bi_triad::Sm89ExactF32TnRoute;
    use crate::mamba_ssm::gpu::kernel_identity::CudaTarget;

    const KNOWN_IDENTITIES: [Sm89ExactF32QualificationIdentity; 3] = [
        Sm89ExactF32QualificationIdentity {
            nvrtc_version: (12, 8),
            compile_key: [
                66, 121, 129, 237, 122, 5, 88, 82, 236, 7, 126, 122, 204, 5, 28, 245, 51, 163, 1,
                244, 197, 59, 224, 245, 31, 83, 101, 169, 218, 21, 222, 22,
            ],
            artifact_digest: [
                192, 15, 106, 13, 105, 184, 164, 11, 31, 141, 174, 79, 21, 42, 80, 52, 86, 155,
                178, 162, 8, 182, 10, 93, 0, 196, 222, 120, 250, 4, 71, 67,
            ],
            source_digest: [
                167, 87, 116, 48, 219, 158, 201, 49, 230, 222, 239, 38, 231, 204, 212, 118, 36, 74,
                143, 52, 227, 224, 0, 233, 195, 114, 128, 121, 124, 104, 219, 160,
            ],
            header_manifest_digest: [
                17, 224, 104, 25, 33, 251, 173, 232, 90, 254, 188, 30, 128, 5, 220, 249, 183, 198,
                166, 77, 231, 111, 1, 159, 159, 185, 247, 48, 224, 199, 213, 168,
            ],
            nvrtc_library_domain: [
                38, 176, 163, 160, 32, 68, 255, 203, 193, 105, 63, 216, 62, 146, 97, 190, 255, 166,
                146, 164, 251, 207, 227, 172, 94, 157, 140, 135, 152, 11, 177, 85,
            ],
        },
        Sm89ExactF32QualificationIdentity {
            nvrtc_version: (13, 0),
            compile_key: [
                17, 240, 71, 134, 204, 61, 203, 250, 181, 106, 102, 167, 33, 191, 189, 121, 115,
                54, 140, 172, 44, 112, 143, 78, 93, 63, 120, 198, 158, 93, 217, 55,
            ],
            artifact_digest: [
                21, 143, 23, 229, 192, 9, 54, 115, 56, 50, 91, 4, 68, 248, 40, 151, 5, 148, 165,
                142, 49, 59, 139, 250, 89, 134, 115, 75, 9, 179, 160, 147,
            ],
            source_digest: [
                167, 87, 116, 48, 219, 158, 201, 49, 230, 222, 239, 38, 231, 204, 212, 118, 36, 74,
                143, 52, 227, 224, 0, 233, 195, 114, 128, 121, 124, 104, 219, 160,
            ],
            header_manifest_digest: [
                255, 141, 156, 152, 51, 210, 171, 101, 199, 86, 7, 9, 70, 23, 25, 158, 210, 13, 70,
                188, 225, 14, 188, 9, 199, 160, 251, 252, 222, 202, 162, 109,
            ],
            nvrtc_library_domain: [
                112, 155, 145, 195, 107, 251, 14, 217, 102, 238, 105, 173, 200, 214, 248, 127, 241,
                16, 238, 207, 61, 251, 80, 96, 54, 127, 24, 60, 230, 20, 235, 13,
            ],
        },
        Sm89ExactF32QualificationIdentity {
            nvrtc_version: (13, 2),
            compile_key: [
                175, 136, 89, 91, 161, 213, 134, 177, 20, 218, 221, 181, 219, 213, 169, 101, 215,
                74, 33, 159, 206, 226, 69, 196, 173, 106, 89, 24, 184, 227, 21, 173,
            ],
            artifact_digest: [
                169, 42, 21, 147, 251, 133, 206, 249, 195, 108, 82, 6, 48, 205, 200, 98, 80, 76,
                64, 189, 98, 37, 69, 98, 89, 24, 12, 129, 180, 177, 111, 25,
            ],
            source_digest: [
                167, 87, 116, 48, 219, 158, 201, 49, 230, 222, 239, 38, 231, 204, 212, 118, 36, 74,
                143, 52, 227, 224, 0, 233, 195, 114, 128, 121, 124, 104, 219, 160,
            ],
            header_manifest_digest: [
                150, 71, 34, 137, 0, 196, 78, 181, 39, 250, 42, 207, 200, 39, 49, 125, 79, 43, 199,
                7, 93, 216, 18, 120, 177, 226, 113, 202, 103, 60, 101, 55,
            ],
            nvrtc_library_domain: [
                208, 49, 165, 62, 185, 114, 53, 183, 15, 98, 246, 82, 147, 45, 177, 189, 247, 40,
                234, 34, 156, 140, 168, 9, 213, 60, 95, 253, 145, 100, 38, 135,
            ],
        },
    ];

    type AdmissionCase = (
        Sm89ExactF32TnRoute,
        (usize, usize, usize),
        ScalarDispatchPlan,
        ScalarDispatchPlan,
    );
    const CASES: [AdmissionCase; 3] = [
        (
            Sm89ExactF32TnRoute::D768InDualChunkFused,
            (2_048, 768, 3_072),
            ScalarDispatchPlan::TnD768InSm89DualChunkQualified,
            ScalarDispatchPlan::TnSplitM {
                m_chunk: 1_024,
                chunks: 2,
            },
        ),
        (
            Sm89ExactF32TnRoute::D768OutDirectBk16,
            (2_048, 1_536, 768),
            ScalarDispatchPlan::TnD768OutSm89DirectBk16Qualified,
            ScalarDispatchPlan::TnSplitM {
                m_chunk: 512,
                chunks: 4,
            },
        ),
        (
            Sm89ExactF32TnRoute::PrismDirectBk16,
            (4_621, 384, 1_928),
            ScalarDispatchPlan::TnPrismSm89DirectBk16Qualified,
            ScalarDispatchPlan::TnSplitM {
                m_chunk: 784,
                chunks: 6,
            },
        ),
    ];

    fn facts(identity: Sm89ExactF32QualificationIdentity) -> ScalarLaunchFacts {
        let compiler = CompilerIdentity {
            source_digest: identity.source_digest,
            invocation_digest: identity.compile_key,
            header_manifest_digest: identity.header_manifest_digest,
            target: crate::mamba_ssm::gpu::kernel_identity::CudaTarget::new("sm_89").unwrap(),
            nvrtc_version: identity.nvrtc_version,
            nvrtc_library_domain: identity.nvrtc_library_domain,
            nvrtc_library_known: true,
            output_kind: ArtifactKind::Ptx,
            composer_revision: COMPOSER_REVISION,
            compiler_revision: COMPILER_REVISION,
            numeric_abi_revision: NUMERIC_ABI_REVISION,
            schedule_revision: SCHEDULE_REVISION,
        };
        ScalarLaunchFacts {
            scalar_artifact: ArtifactIdentity {
                module_kind: ModuleKind::TriadScalar,
                artifact_kind: ArtifactKind::Ptx,
                compile_key: [1; 32],
                artifact_digest: [2; 32],
            },
            scalar_compiler: compiler,
            fixed_artifact: ArtifactIdentity {
                module_kind: ModuleKind::Fixed,
                artifact_kind: ArtifactKind::Ptx,
                compile_key: [3; 32],
                artifact_digest: [4; 32],
            },
            fixed_compiler: compiler,
            fixed_copyplan_loaded: false,
            sm89_exact_f32_artifact: Some(ArtifactIdentity {
                module_kind: ModuleKind::TriadSm89ExactF32,
                artifact_kind: ArtifactKind::Ptx,
                compile_key: identity.compile_key,
                artifact_digest: identity.artifact_digest,
            }),
            sm89_exact_f32_compiler: Some(compiler),
            sm89_exact_f32_symbols_loaded: [true; 3],
            sm89_exact_f32_d128_artifact: None,
            sm89_exact_f32_d128_compiler: None,
            sm89_exact_f32_d128_symbols_loaded: [false; 2],
            compute_capability: (8, 9),
            multiprocessor_count: 142,
        }
    }

    fn operands() -> F32TriadOperands {
        F32TriadOperands {
            output: 0x3000,
            a: 0x1000,
            b: 0x2000,
            bias: None,
            alpha: 1.0,
            beta: 1.0,
        }
    }

    #[test]
    fn actual_auto_admits_literal_large_tn_geometry_for_each_known_toolkit() {
        for identity in KNOWN_IDENTITIES {
            for (route, dims, expected, _) in CASES {
                let request = F32TriadRequest {
                    op: ResolvedGemmOp::Tn,
                    shape: F32TriadShape::contiguous(ResolvedGemmOp::Tn, dims),
                };
                assert_eq!(
                    scalar_launch_plan(facts(identity), request, operands()).unwrap(),
                    expected,
                    "AUTO did not admit {route:?} for NVRTC {:?}",
                    identity.nvrtc_version,
                );
            }
        }
    }

    #[test]
    fn actual_auto_large_tn_admission_fails_closed() {
        for (symbol_index, (route, dims, expected, fallback)) in CASES.into_iter().enumerate() {
            let request = F32TriadRequest {
                op: ResolvedGemmOp::Tn,
                shape: F32TriadShape::contiguous(ResolvedGemmOp::Tn, dims),
            };
            let base = facts(KNOWN_IDENTITIES[2]);
            assert_eq!(
                scalar_launch_plan(base, request, operands()).unwrap(),
                expected
            );

            for (field, delta) in [("m", 1), ("k", 1), ("n", 1)] {
                let neighboring_dims = match field {
                    "m" => (dims.0 + delta, dims.1, dims.2),
                    "k" => (dims.0, dims.1 + delta, dims.2),
                    "n" => (dims.0, dims.1, dims.2 + delta),
                    _ => unreachable!(),
                };
                let wrong = F32TriadRequest {
                    op: ResolvedGemmOp::Tn,
                    shape: F32TriadShape::contiguous(ResolvedGemmOp::Tn, neighboring_dims),
                };
                assert_ne!(
                    scalar_launch_plan(base, wrong, operands()).unwrap(),
                    expected,
                    "neighboring {field} shape was admitted",
                );
            }
            let wrong_op = F32TriadRequest {
                op: ResolvedGemmOp::Nt,
                shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, dims),
            };
            assert_ne!(
                scalar_launch_plan(base, wrong_op, operands()).unwrap(),
                expected
            );
            for field in ["lda", "ldb", "ldc"] {
                let mut wrong = request;
                match field {
                    "lda" => wrong.shape.lda += 1,
                    "ldb" => wrong.shape.ldb += 1,
                    "ldc" => wrong.shape.ldc += 1,
                    _ => unreachable!(),
                }
                assert_eq!(
                    scalar_launch_plan(base, wrong, operands()).unwrap(),
                    fallback,
                    "bad {field} did not retain the literal prior route",
                );
            }

            for mutation in [
                F32TriadOperands {
                    output: 0,
                    ..operands()
                },
                F32TriadOperands { a: 0, ..operands() },
                F32TriadOperands { b: 0, ..operands() },
                F32TriadOperands {
                    output: 0x3004,
                    ..operands()
                },
                F32TriadOperands {
                    a: 0x1004,
                    ..operands()
                },
                F32TriadOperands {
                    b: 0x2004,
                    ..operands()
                },
                F32TriadOperands {
                    bias: Some(0x4000),
                    ..operands()
                },
                F32TriadOperands {
                    alpha: 0.5,
                    ..operands()
                },
                F32TriadOperands {
                    beta: 0.0,
                    ..operands()
                },
            ] {
                assert_eq!(
                    scalar_launch_plan(base, request, mutation).unwrap(),
                    fallback,
                    "invalid operand contract admitted {route:?}: {mutation:?}",
                );
            }

            let mut missing = base;
            missing.sm89_exact_f32_symbols_loaded[symbol_index] = false;
            let mut mutations = vec![missing];
            let mut mutation = base;
            mutation.compute_capability = (9, 0);
            mutations.push(mutation);
            mutation = base;
            mutation.multiprocessor_count = 141;
            mutations.push(mutation);
            mutation = base;
            mutation.sm89_exact_f32_artifact = None;
            mutations.push(mutation);
            mutation = base;
            mutation.sm89_exact_f32_compiler = None;
            mutations.push(mutation);
            mutation = base;
            mutation
                .sm89_exact_f32_artifact
                .as_mut()
                .unwrap()
                .module_kind = ModuleKind::TriadScalar;
            mutations.push(mutation);
            mutation = base;
            mutation
                .sm89_exact_f32_artifact
                .as_mut()
                .unwrap()
                .artifact_kind = ArtifactKind::Cubin;
            mutations.push(mutation);
            mutation = base;
            mutation
                .sm89_exact_f32_artifact
                .as_mut()
                .unwrap()
                .compile_key[0] ^= 1;
            mutations.push(mutation);
            mutation = base;
            mutation
                .sm89_exact_f32_artifact
                .as_mut()
                .unwrap()
                .artifact_digest[0] ^= 1;
            mutations.push(mutation);
            mutation = base;
            mutation
                .sm89_exact_f32_compiler
                .as_mut()
                .unwrap()
                .source_digest[0] ^= 1;
            mutations.push(mutation);
            mutation = base;
            mutation
                .sm89_exact_f32_compiler
                .as_mut()
                .unwrap()
                .invocation_digest[0] ^= 1;
            mutations.push(mutation);
            mutation = base;
            mutation
                .sm89_exact_f32_compiler
                .as_mut()
                .unwrap()
                .header_manifest_digest[0] ^= 1;
            mutations.push(mutation);
            mutation = base;
            mutation.sm89_exact_f32_compiler.as_mut().unwrap().target =
                CudaTarget::new("sm_90").unwrap();
            mutations.push(mutation);
            mutation = base;
            mutation
                .sm89_exact_f32_compiler
                .as_mut()
                .unwrap()
                .nvrtc_version = (13, 1);
            mutations.push(mutation);
            mutation = base;
            mutation
                .sm89_exact_f32_compiler
                .as_mut()
                .unwrap()
                .nvrtc_library_domain[0] ^= 1;
            mutations.push(mutation);
            mutation = base;
            mutation
                .sm89_exact_f32_compiler
                .as_mut()
                .unwrap()
                .nvrtc_library_known = false;
            mutations.push(mutation);
            mutation = base;
            mutation
                .sm89_exact_f32_compiler
                .as_mut()
                .unwrap()
                .output_kind = ArtifactKind::Cubin;
            mutations.push(mutation);
            mutation = base;
            mutation
                .sm89_exact_f32_compiler
                .as_mut()
                .unwrap()
                .composer_revision += 1;
            mutations.push(mutation);
            mutation = base;
            mutation
                .sm89_exact_f32_compiler
                .as_mut()
                .unwrap()
                .compiler_revision += 1;
            mutations.push(mutation);
            mutation = base;
            mutation
                .sm89_exact_f32_compiler
                .as_mut()
                .unwrap()
                .numeric_abi_revision += 1;
            mutations.push(mutation);
            mutation = base;
            mutation
                .sm89_exact_f32_compiler
                .as_mut()
                .unwrap()
                .schedule_revision += 1;
            mutations.push(mutation);
            for mutation in mutations {
                assert_eq!(
                    scalar_launch_plan(mutation, request, operands()).unwrap(),
                    fallback,
                    "identity mutation admitted {route:?}: {mutation:?}",
                );
            }

            assert_eq!(
                forced_sm89_exact_f32_plan(base, request, operands(), route).unwrap(),
                expected
            );
        }
    }
}

#[cfg(test)]
mod sm89_exact_f32_d128_admission_tests {
    use super::*;
    use crate::mamba_ssm::gpu::gemm_bi_triad::Sm89ExactF32D128Route;
    use crate::mamba_ssm::gpu::kernel_identity::CudaTarget;

    const CASES: [(
        Sm89ExactF32D128Route,
        (usize, usize, usize),
        ScalarDispatchPlan,
    ); 2] = [
        (
            Sm89ExactF32D128Route::D128InDirectFold,
            (1_024, 128, 512),
            ScalarDispatchPlan::TnD128InSm89DirectFoldQualified,
        ),
        (
            Sm89ExactF32D128Route::D128OutDirectFold,
            (1_024, 256, 128),
            ScalarDispatchPlan::TnD128OutSm89DirectFoldQualified,
        ),
    ];

    const FALLBACK: ScalarDispatchPlan = ScalarDispatchPlan::TnSplitM {
        m_chunk: 16,
        chunks: 64,
    };

    fn facts(identity: Sm89ExactF32D128QualificationIdentity) -> ScalarLaunchFacts {
        let compiler = CompilerIdentity {
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
        };
        ScalarLaunchFacts {
            scalar_artifact: ArtifactIdentity {
                module_kind: ModuleKind::TriadScalar,
                artifact_kind: ArtifactKind::Ptx,
                compile_key: [1; 32],
                artifact_digest: [2; 32],
            },
            scalar_compiler: compiler,
            fixed_artifact: ArtifactIdentity {
                module_kind: ModuleKind::Fixed,
                artifact_kind: ArtifactKind::Ptx,
                compile_key: [3; 32],
                artifact_digest: [4; 32],
            },
            fixed_compiler: compiler,
            fixed_copyplan_loaded: false,
            sm89_exact_f32_artifact: None,
            sm89_exact_f32_compiler: None,
            sm89_exact_f32_symbols_loaded: [false; 3],
            sm89_exact_f32_d128_artifact: Some(ArtifactIdentity {
                module_kind: ModuleKind::TriadSm89ExactF32D128,
                artifact_kind: ArtifactKind::Ptx,
                compile_key: identity.compile_key,
                artifact_digest: identity.artifact_digest,
            }),
            sm89_exact_f32_d128_compiler: Some(compiler),
            sm89_exact_f32_d128_symbols_loaded: [true; 2],
            compute_capability: (8, 9),
            multiprocessor_count: 142,
        }
    }

    fn request(dims: (usize, usize, usize)) -> F32TriadRequest {
        F32TriadRequest {
            op: ResolvedGemmOp::Tn,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Tn, dims),
        }
    }

    fn operands() -> F32TriadOperands {
        F32TriadOperands {
            output: 0x3000,
            a: 0x1000,
            b: 0x2000,
            bias: None,
            alpha: 1.0,
            beta: 1.0,
        }
    }

    #[test]
    fn all_three_candidates_bind_both_forced_symbols() {
        assert_eq!(SM89_EXACT_F32_D128_QUALIFICATION_CANDIDATES.len(), 3);
        for identity in SM89_EXACT_F32_D128_QUALIFICATION_CANDIDATES {
            let facts = facts(*identity);
            assert!(identity.matches(facts));
            for (route, dims, selected) in CASES {
                assert_eq!(scalar_dispatch_plan(request(dims), 142).unwrap(), FALLBACK);
                assert_eq!(
                    forced_sm89_exact_f32_d128_plan(facts, request(dims), operands(), route)
                        .unwrap(),
                    selected,
                );
            }
        }
    }

    #[test]
    fn actual_auto_admits_exactly_the_three_live_passed_d128_cohorts() {
        assert_eq!(
            SM89_EXACT_F32_D128_EVIDENCE_COHORTS,
            SM89_EXACT_F32_D128_QUALIFICATION_CANDIDATES
        );
        for identity in SM89_EXACT_F32_D128_EVIDENCE_COHORTS {
            let facts = facts(*identity);
            for (route, dims, selected) in CASES {
                assert_eq!(
                    scalar_launch_plan(facts, request(dims), operands()).unwrap(),
                    selected,
                    "AUTO did not admit {route:?} on NVRTC {:?}",
                    identity.nvrtc_version
                );
            }
        }
    }

    #[test]
    fn other_sm80_boards_reach_the_d128_routes_through_the_proof_tier() {
        let ada = facts(SM89_EXACT_F32_D128_QUALIFICATION_CANDIDATES[2]);
        for (route, dims, selected) in CASES {
            assert_eq!(
                scalar_proof_plan(ada, request(dims), operands()).unwrap(),
                Some(selected),
                "{route:?} is the proof candidate on the evidence board too"
            );
            for cc in [(8, 0), (8, 6), (8, 7), (9, 0), (10, 0), (12, 0)] {
                let board = ScalarLaunchFacts {
                    compute_capability: cc,
                    ..ada
                };
                assert_eq!(
                    scalar_launch_plan(board, request(dims), operands()).unwrap(),
                    FALLBACK,
                    "{route:?} holds no frozen evidence on {cc:?}"
                );
                assert_eq!(
                    scalar_proof_plan(board, request(dims), operands()).unwrap(),
                    Some(selected),
                    "{route:?} is the proof candidate on {cc:?}"
                );
            }
            for multiprocessor_count in [84, 108, 132, 148, 170] {
                let board = ScalarLaunchFacts {
                    compute_capability: (9, 0),
                    multiprocessor_count,
                    ..ada
                };
                let expected = (scalar_dispatch_plan(request(dims), multiprocessor_count).unwrap()
                    == FALLBACK)
                    .then_some(selected);
                assert_eq!(
                    scalar_proof_plan(board, request(dims), operands()).unwrap(),
                    expected,
                    "{route:?} stands in for exactly the split it reproduces at {multiprocessor_count} SMs"
                );
            }
            let pre_ampere = ScalarLaunchFacts {
                compute_capability: (7, 5),
                ..ada
            };
            assert_eq!(
                scalar_proof_plan(pre_ampere, request(dims), operands()).unwrap(),
                None
            );
            let unloaded = ScalarLaunchFacts {
                compute_capability: (9, 0),
                sm89_exact_f32_d128_symbols_loaded: [false; 2],
                ..ada
            };
            assert_eq!(
                scalar_proof_plan(unloaded, request(dims), operands()).unwrap(),
                None
            );
            let nonunit = F32TriadOperands {
                alpha: -0.75,
                ..operands()
            };
            let hopper = ScalarLaunchFacts {
                compute_capability: (9, 0),
                ..ada
            };
            assert_eq!(
                scalar_proof_plan(hopper, request(dims), nonunit).unwrap(),
                None
            );
        }
    }

    #[test]
    fn forced_route_accepts_nonunit_alpha_while_auto_epilogue_stays_exact() {
        let facts = facts(SM89_EXACT_F32_D128_QUALIFICATION_CANDIDATES[2]);
        for (route, dims, selected) in CASES {
            let nonunit = F32TriadOperands {
                alpha: -0.75,
                ..operands()
            };
            assert_eq!(
                forced_sm89_exact_f32_d128_plan(facts, request(dims), nonunit, route).unwrap(),
                selected,
            );
            assert_eq!(
                scalar_launch_plan(facts, request(dims), nonunit).unwrap(),
                FALLBACK
            );
        }
    }

    #[test]
    fn d128_routes_fail_closed_on_every_shape_stride_and_operand_neighbor() {
        let facts = facts(SM89_EXACT_F32_D128_QUALIFICATION_CANDIDATES[2]);
        for (route, dims, _) in CASES {
            for (axis, delta) in [(0, -1_isize), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1)] {
                let mut neighbor = [dims.0, dims.1, dims.2];
                neighbor[axis] = neighbor[axis].checked_add_signed(delta).unwrap();
                assert!(
                    forced_sm89_exact_f32_d128_plan(
                        facts,
                        request((neighbor[0], neighbor[1], neighbor[2])),
                        operands(),
                        route,
                    )
                    .is_err(),
                    "shape axis {axis} delta {delta}"
                );
            }
            for field in ["lda", "ldb", "ldc"] {
                let mut neighbor = request(dims);
                match field {
                    "lda" => neighbor.shape.lda += 1,
                    "ldb" => neighbor.shape.ldb += 1,
                    "ldc" => neighbor.shape.ldc += 1,
                    _ => unreachable!(),
                }
                assert!(
                    forced_sm89_exact_f32_d128_plan(facts, neighbor, operands(), route).is_err(),
                    "stride {field}"
                );
            }
            let wrong_op = F32TriadRequest {
                op: ResolvedGemmOp::Nt,
                shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, dims),
            };
            assert!(forced_sm89_exact_f32_d128_plan(facts, wrong_op, operands(), route).is_err());
            for rejected in [
                F32TriadOperands {
                    output: 0,
                    ..operands()
                },
                F32TriadOperands { a: 0, ..operands() },
                F32TriadOperands { b: 0, ..operands() },
                F32TriadOperands {
                    output: 0x3004,
                    ..operands()
                },
                F32TriadOperands {
                    a: 0x1004,
                    ..operands()
                },
                F32TriadOperands {
                    b: 0x2004,
                    ..operands()
                },
                F32TriadOperands {
                    bias: Some(0x4000),
                    ..operands()
                },
                F32TriadOperands {
                    beta: 0.0,
                    ..operands()
                },
                F32TriadOperands {
                    beta: -0.0,
                    ..operands()
                },
            ] {
                assert!(
                    forced_sm89_exact_f32_d128_plan(facts, request(dims), rejected, route).is_err(),
                    "operand mutation {rejected:?}"
                );
            }
        }
    }

    #[test]
    fn each_symbol_and_identity_field_fails_closed_independently() {
        for (symbol_index, (route, dims, _)) in CASES.into_iter().enumerate() {
            let base = facts(SM89_EXACT_F32_D128_QUALIFICATION_CANDIDATES[2]);
            let mut mutations = Vec::new();
            let mut mutation = base;
            mutation.sm89_exact_f32_d128_symbols_loaded[symbol_index] = false;
            mutations.push(mutation);
            mutation = base;
            mutation.compute_capability = (9, 0);
            mutations.push(mutation);
            mutation = base;
            mutation.multiprocessor_count = 141;
            mutations.push(mutation);
            mutation = base;
            mutation.sm89_exact_f32_d128_artifact = None;
            mutations.push(mutation);
            mutation = base;
            mutation.sm89_exact_f32_d128_compiler = None;
            mutations.push(mutation);
            mutation = base;
            mutation
                .sm89_exact_f32_d128_artifact
                .as_mut()
                .unwrap()
                .module_kind = ModuleKind::TriadScalar;
            mutations.push(mutation);
            mutation = base;
            mutation
                .sm89_exact_f32_d128_artifact
                .as_mut()
                .unwrap()
                .artifact_kind = ArtifactKind::Cubin;
            mutations.push(mutation);
            mutation = base;
            mutation
                .sm89_exact_f32_d128_artifact
                .as_mut()
                .unwrap()
                .compile_key[0] ^= 1;
            mutations.push(mutation);
            mutation = base;
            mutation
                .sm89_exact_f32_d128_artifact
                .as_mut()
                .unwrap()
                .artifact_digest[0] ^= 1;
            mutations.push(mutation);
            mutation = base;
            mutation
                .sm89_exact_f32_d128_compiler
                .as_mut()
                .unwrap()
                .source_digest[0] ^= 1;
            mutations.push(mutation);
            mutation = base;
            mutation
                .sm89_exact_f32_d128_compiler
                .as_mut()
                .unwrap()
                .invocation_digest[0] ^= 1;
            mutations.push(mutation);
            mutation = base;
            mutation
                .sm89_exact_f32_d128_compiler
                .as_mut()
                .unwrap()
                .header_manifest_digest[0] ^= 1;
            mutations.push(mutation);
            mutation = base;
            mutation
                .sm89_exact_f32_d128_compiler
                .as_mut()
                .unwrap()
                .target = CudaTarget::new("sm_90").unwrap();
            mutations.push(mutation);
            mutation = base;
            mutation
                .sm89_exact_f32_d128_compiler
                .as_mut()
                .unwrap()
                .nvrtc_version = (13, 1);
            mutations.push(mutation);
            mutation = base;
            mutation
                .sm89_exact_f32_d128_compiler
                .as_mut()
                .unwrap()
                .nvrtc_library_domain[0] ^= 1;
            mutations.push(mutation);
            mutation = base;
            mutation
                .sm89_exact_f32_d128_compiler
                .as_mut()
                .unwrap()
                .nvrtc_library_known = false;
            mutations.push(mutation);
            mutation = base;
            mutation
                .sm89_exact_f32_d128_compiler
                .as_mut()
                .unwrap()
                .output_kind = ArtifactKind::Cubin;
            mutations.push(mutation);
            mutation = base;
            mutation
                .sm89_exact_f32_d128_compiler
                .as_mut()
                .unwrap()
                .composer_revision += 1;
            mutations.push(mutation);
            mutation = base;
            mutation
                .sm89_exact_f32_d128_compiler
                .as_mut()
                .unwrap()
                .compiler_revision += 1;
            mutations.push(mutation);
            mutation = base;
            mutation
                .sm89_exact_f32_d128_compiler
                .as_mut()
                .unwrap()
                .numeric_abi_revision += 1;
            mutations.push(mutation);
            mutation = base;
            mutation
                .sm89_exact_f32_d128_compiler
                .as_mut()
                .unwrap()
                .schedule_revision += 1;
            mutations.push(mutation);
            for rejected in mutations {
                assert!(
                    forced_sm89_exact_f32_d128_plan(rejected, request(dims), operands(), route)
                        .is_err(),
                    "identity mutation admitted {route:?}: {rejected:?}"
                );
                assert_eq!(
                    scalar_launch_plan(rejected, request(dims), operands()).unwrap(),
                    FALLBACK,
                    "AUTO admitted identity mutation for {route:?}: {rejected:?}"
                );
            }
            let sibling = 1 - symbol_index;
            let mut only_sibling = base;
            only_sibling.sm89_exact_f32_d128_symbols_loaded[sibling] = false;
            assert!(
                forced_sm89_exact_f32_d128_plan(only_sibling, request(dims), operands(), route)
                    .is_ok(),
                "excluding sibling disabled {route:?}"
            );
            assert_eq!(
                scalar_launch_plan(only_sibling, request(dims), operands()).unwrap(),
                CASES[symbol_index].2,
                "AUTO sibling exclusion disabled {route:?}"
            );
        }
    }
}

impl Tf32ExactShape {
    /// The contiguous layout the cell was measured under.
    fn contiguous(self, op: ResolvedGemmOp) -> super::contract::F32TriadShape {
        let dims = match op {
            ResolvedGemmOp::Nn => (self.output_rows, self.reduction, self.output_columns),
            ResolvedGemmOp::Tn => (self.reduction, self.output_rows, self.output_columns),
            ResolvedGemmOp::Nt => (self.output_rows, self.output_columns, self.reduction),
        };
        super::contract::F32TriadShape::contiguous(op, dims)
    }

    fn matches_contiguous(self, request: F32TriadRequest) -> bool {
        request.shape == self.contiguous(request.op)
    }
}

/// Which operands the portable kernels stage with vector loads: an operand
/// whose leading dimension is a multiple of four floats takes the 16-byte
/// path, any other takes scalar loads. The two paths run different code, so
/// a measured cell speaks only for shapes staged the same way.
fn tf32_staging_class(shape: super::contract::F32TriadShape) -> (bool, bool) {
    (shape.lda.is_multiple_of(4), shape.ldb.is_multiple_of(4))
}

/// The archived cells record whether their exact strides use scalar or vector
/// staging. Every class still requires the concrete epilogue and aligned base
/// pointers that were present during qualification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tf32AutoOperandGate {
    /// TN/NT forbids bias and this exact stride tuple forces scalar staging,
    /// so four-byte-aligned production pointers stay in the measured path.
    RequestContractSafe,
    /// NN permits bias, but the archived point measured `bias=None` only.
    RequiresNoBiasEvidence,
    /// The route can stage asynchronously, but pointer-offset evidence is absent.
    RequiresVectorAlignmentEvidence,
    /// Both bias and pointer-alignment evidence are absent.
    RequiresNoBiasAndVectorAlignmentEvidence,
}

fn tf32_auto_operands_match(op: ResolvedGemmOp, operands: F32TriadOperands) -> bool {
    let epilogue_matches = match op {
        ResolvedGemmOp::Nn | ResolvedGemmOp::Nt => {
            operands.alpha.to_bits() == 1.0_f32.to_bits()
                && operands.beta.to_bits() == 0.0_f32.to_bits()
                && operands.bias.is_none()
        }
        ResolvedGemmOp::Tn => {
            operands.alpha.to_bits() == 1.0_f32.to_bits()
                && operands.beta.to_bits() == 1.0_f32.to_bits()
                && operands.bias.is_none()
        }
    };
    epilogue_matches
        && [operands.output, operands.a, operands.b]
            .into_iter()
            .all(|pointer| pointer != 0 && pointer.is_multiple_of(16))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Tf32AutoQualificationIdentity {
    module_kind: ModuleKind,
    module_target: &'static str,
    device_target: &'static str,
    compute_capability: (u32, u32),
    multiprocessor_count: u32,
    nvrtc_version: (i32, i32),
    optin_shared_bytes: u32,
    tensor_map_access: bool,
    compile_key: [u8; 32],
    artifact_digest: [u8; 32],
    source_digest: [u8; 32],
    invocation_digest: [u8; 32],
    header_manifest_digest: [u8; 32],
    nvrtc_library_domain: [u8; 32],
}

impl Tf32AutoQualificationIdentity {
    fn matches(self, module: Tf32QualifiedModule) -> bool {
        self.mismatch(module).is_none()
    }

    /// The first identity field the bound module does not satisfy, or `None`
    /// when the cohort applies to it. A decline on this path used to be a bare
    /// `None`; the field name is what tells a wrong board, a moved toolkit and
    /// an edited kernel apart.
    fn mismatch(self, module: Tf32QualifiedModule) -> Option<&'static str> {
        let checks: [(&'static str, bool); 28] = [
            ("module kind", module.module_kind == self.module_kind),
            (
                "module target",
                module.target.as_str() == self.module_target,
            ),
            (
                "artifact module kind",
                module.artifact.module_kind == self.module_kind,
            ),
            (
                "artifact kind",
                module.artifact.artifact_kind == ArtifactKind::Ptx,
            ),
            (
                "compile key",
                module.artifact.compile_key == self.compile_key,
            ),
            (
                "artifact digest",
                module.artifact.artifact_digest == self.artifact_digest,
            ),
            (
                "source digest",
                module.compiler.source_digest == self.source_digest,
            ),
            (
                "invocation digest",
                module.compiler.invocation_digest == self.invocation_digest,
            ),
            (
                "header manifest digest",
                module.compiler.header_manifest_digest == self.header_manifest_digest,
            ),
            (
                "compiler target",
                module.compiler.target.as_str() == self.module_target,
            ),
            (
                "nvrtc version",
                module.compiler.nvrtc_version == self.nvrtc_version,
            ),
            (
                "nvrtc library domain",
                module.compiler.nvrtc_library_domain == self.nvrtc_library_domain,
            ),
            ("nvrtc library known", module.compiler.nvrtc_library_known),
            (
                "compiler output kind",
                module.compiler.output_kind == ArtifactKind::Ptx,
            ),
            (
                "composer revision",
                module.compiler.composer_revision == COMPOSER_REVISION,
            ),
            (
                "compiler revision",
                module.compiler.compiler_revision == COMPILER_REVISION,
            ),
            (
                "numeric abi revision",
                module.compiler.numeric_abi_revision == NUMERIC_ABI_REVISION,
            ),
            (
                "schedule revision",
                module.compiler.schedule_revision == SCHEDULE_REVISION,
            ),
            (
                "device compute capability",
                module.device.compute_capability == self.compute_capability,
            ),
            (
                "multiprocessor count",
                module.device.multiprocessor_count == self.multiprocessor_count,
            ),
            (
                "device target",
                module.device.target.as_str() == self.device_target,
            ),
            (
                "caps compute capability",
                module.device_caps.compute_capability == self.compute_capability,
            ),
            (
                "caps nvrtc version",
                module.device_caps.nvrtc_version == self.nvrtc_version,
            ),
            (
                "accepted target",
                module
                    .device_caps
                    .accepted_target
                    .is_some_and(|target| target.as_str() == self.module_target),
            ),
            (
                "opt-in shared bytes",
                module.device_caps.optin_shared_bytes == self.optin_shared_bytes,
            ),
            (
                "tensor map access",
                module.device_caps.tensor_map_access == self.tensor_map_access,
            ),
            ("cohort compute capability", true),
            ("cohort multiprocessor count", true),
        ];
        checks
            .into_iter()
            .find(|(_, holds)| !holds)
            .map(|(field, _)| field)
    }
}

const SM89_TF32_QUALIFIED_TUNING_REVISION: u16 = F32_TF32_TUNING_REVISION;
const SM89_TF32_QUALIFICATION_IDENTITY: Tf32AutoQualificationIdentity =
    Tf32AutoQualificationIdentity {
        module_kind: ModuleKind::TriadSm80,
        module_target: "sm_89",
        device_target: "sm_89",
        compute_capability: (8, 9),
        multiprocessor_count: 142,
        nvrtc_version: (13, 2),
        optin_shared_bytes: 101376,
        tensor_map_access: false,
        compile_key: [
            224, 96, 64, 128, 40, 115, 80, 43, 118, 227, 103, 206, 217, 19, 27, 119, 0, 99, 142,
            50, 25, 234, 49, 114, 58, 215, 35, 193, 199, 81, 119, 36,
        ],
        artifact_digest: [
            208, 216, 246, 246, 192, 161, 136, 78, 52, 90, 107, 113, 124, 0, 140, 159, 94, 9, 168,
            63, 12, 160, 172, 152, 165, 247, 65, 253, 70, 13, 126, 68,
        ],
        source_digest: [
            19, 130, 62, 215, 174, 153, 255, 129, 82, 208, 196, 106, 32, 82, 148, 17, 100, 189,
            112, 107, 52, 217, 149, 177, 205, 208, 23, 199, 115, 114, 0, 36,
        ],
        invocation_digest: [
            224, 96, 64, 128, 40, 115, 80, 43, 118, 227, 103, 206, 217, 19, 27, 119, 0, 99, 142,
            50, 25, 234, 49, 114, 58, 215, 35, 193, 199, 81, 119, 36,
        ],
        header_manifest_digest: [
            75, 22, 2, 166, 11, 33, 251, 177, 146, 26, 10, 166, 112, 157, 154, 10, 194, 82, 149,
            70, 133, 147, 43, 13, 203, 214, 7, 99, 136, 158, 174, 27,
        ],
        nvrtc_library_domain: [
            208, 49, 165, 62, 185, 114, 53, 183, 15, 98, 246, 82, 147, 45, 177, 189, 247, 40, 234,
            34, 156, 140, 168, 9, 213, 60, 95, 253, 145, 100, 38, 135,
        ],
    };

const fn sm89_observed_tf32_identity(
    module_kind: ModuleKind,
    nvrtc_version: (i32, i32),
    compile_key: [u8; 32],
    artifact_digest: [u8; 32],
    source_digest: [u8; 32],
    header_manifest_digest: [u8; 32],
    nvrtc_library_domain: [u8; 32],
) -> Tf32AutoQualificationIdentity {
    Tf32AutoQualificationIdentity {
        module_kind,
        module_target: "sm_89",
        device_target: "sm_89",
        compute_capability: (8, 9),
        multiprocessor_count: 142,
        nvrtc_version,
        optin_shared_bytes: 101_376,
        tensor_map_access: false,
        compile_key,
        artifact_digest,
        source_digest,
        invocation_digest: compile_key,
        header_manifest_digest,
        nvrtc_library_domain,
    }
}

const SM89_PORTABLE_TF32_IDENTITY_CUDA_12_8: Tf32AutoQualificationIdentity =
    sm89_observed_tf32_identity(
        ModuleKind::TriadSm80,
        (12, 8),
        [
            108, 118, 78, 194, 63, 27, 160, 175, 171, 37, 236, 252, 23, 156, 221, 17, 121, 76, 231,
            252, 70, 188, 244, 48, 207, 22, 157, 49, 238, 159, 79, 121,
        ],
        [
            250, 130, 75, 246, 179, 227, 73, 41, 194, 133, 91, 73, 156, 206, 71, 246, 105, 196, 82,
            0, 60, 88, 223, 241, 90, 114, 9, 64, 243, 222, 30, 205,
        ],
        [
            19, 130, 62, 215, 174, 153, 255, 129, 82, 208, 196, 106, 32, 82, 148, 17, 100, 189,
            112, 107, 52, 217, 149, 177, 205, 208, 23, 199, 115, 114, 0, 36,
        ],
        [
            20, 247, 46, 114, 95, 159, 85, 254, 5, 255, 78, 190, 206, 242, 34, 139, 19, 82, 158,
            141, 114, 155, 128, 63, 5, 31, 18, 144, 163, 95, 235, 67,
        ],
        [
            38, 176, 163, 160, 32, 68, 255, 203, 193, 105, 63, 216, 62, 146, 97, 190, 255, 166,
            146, 164, 251, 207, 227, 172, 94, 157, 140, 135, 152, 11, 177, 85,
        ],
    );

const SM89_PORTABLE_TF32_IDENTITY_CUDA_13_0: Tf32AutoQualificationIdentity =
    sm89_observed_tf32_identity(
        ModuleKind::TriadSm80,
        (13, 0),
        [
            194, 20, 83, 245, 169, 127, 142, 243, 87, 112, 226, 119, 37, 84, 149, 255, 183, 161,
            196, 73, 77, 21, 39, 220, 206, 135, 71, 236, 255, 116, 158, 119,
        ],
        [
            182, 46, 9, 207, 7, 10, 165, 236, 4, 123, 239, 182, 74, 141, 160, 219, 162, 58, 115,
            59, 50, 216, 197, 243, 16, 181, 218, 95, 214, 179, 172, 212,
        ],
        [
            19, 130, 62, 215, 174, 153, 255, 129, 82, 208, 196, 106, 32, 82, 148, 17, 100, 189,
            112, 107, 52, 217, 149, 177, 205, 208, 23, 199, 115, 114, 0, 36,
        ],
        [
            235, 40, 251, 186, 8, 248, 156, 8, 61, 33, 13, 202, 71, 10, 65, 80, 116, 84, 9, 52,
            233, 247, 187, 213, 154, 15, 167, 22, 71, 195, 180, 136,
        ],
        [
            112, 155, 145, 195, 107, 251, 14, 217, 102, 238, 105, 173, 200, 214, 248, 127, 241, 16,
            238, 207, 61, 251, 80, 96, 54, 127, 24, 60, 230, 20, 235, 13,
        ],
    );

const SM89_JOINT_TF32_IDENTITY_CUDA_12_8: Tf32AutoQualificationIdentity =
    sm89_observed_tf32_identity(
        ModuleKind::TriadSm89Tf32Joint,
        (12, 8),
        [
            151, 182, 254, 89, 160, 216, 168, 5, 227, 232, 58, 84, 158, 215, 224, 164, 223, 51,
            245, 200, 152, 122, 6, 5, 248, 185, 67, 249, 40, 247, 6, 28,
        ],
        [
            218, 169, 91, 231, 57, 231, 34, 75, 236, 237, 80, 104, 65, 214, 25, 48, 9, 153, 149,
            254, 156, 225, 241, 127, 162, 114, 140, 23, 149, 251, 112, 169,
        ],
        [
            89, 203, 88, 237, 40, 157, 244, 6, 14, 69, 23, 71, 190, 238, 134, 195, 6, 188, 110,
            235, 25, 76, 134, 116, 246, 27, 117, 195, 25, 92, 15, 81,
        ],
        [
            134, 3, 170, 185, 85, 163, 208, 3, 81, 104, 139, 243, 30, 23, 171, 198, 233, 142, 149,
            185, 97, 94, 174, 114, 45, 160, 244, 226, 196, 163, 10, 103,
        ],
        [
            38, 176, 163, 160, 32, 68, 255, 203, 193, 105, 63, 216, 62, 146, 97, 190, 255, 166,
            146, 164, 251, 207, 227, 172, 94, 157, 140, 135, 152, 11, 177, 85,
        ],
    );

const SM89_JOINT_TF32_IDENTITY_CUDA_13_0: Tf32AutoQualificationIdentity =
    sm89_observed_tf32_identity(
        ModuleKind::TriadSm89Tf32Joint,
        (13, 0),
        [
            163, 70, 57, 177, 73, 45, 135, 163, 114, 112, 104, 53, 92, 44, 234, 11, 212, 93, 61,
            89, 21, 87, 169, 50, 94, 235, 83, 98, 182, 223, 145, 77,
        ],
        [
            140, 64, 160, 57, 197, 189, 104, 189, 108, 105, 108, 37, 107, 218, 245, 56, 239, 139,
            128, 240, 119, 32, 80, 89, 67, 101, 129, 18, 61, 34, 193, 191,
        ],
        [
            89, 203, 88, 237, 40, 157, 244, 6, 14, 69, 23, 71, 190, 238, 134, 195, 6, 188, 110,
            235, 25, 76, 134, 116, 246, 27, 117, 195, 25, 92, 15, 81,
        ],
        [
            134, 3, 170, 185, 85, 163, 208, 3, 81, 104, 139, 243, 30, 23, 171, 198, 233, 142, 149,
            185, 97, 94, 174, 114, 45, 160, 244, 226, 196, 163, 10, 103,
        ],
        [
            112, 155, 145, 195, 107, 251, 14, 217, 102, 238, 105, 173, 200, 214, 248, 127, 241, 16,
            238, 207, 61, 251, 80, 96, 54, 127, 24, 60, 230, 20, 235, 13,
        ],
    );

const SM89_JOINT_TF32_IDENTITY_CUDA_13_2: Tf32AutoQualificationIdentity =
    sm89_observed_tf32_identity(
        ModuleKind::TriadSm89Tf32Joint,
        (13, 2),
        [
            100, 93, 80, 23, 87, 210, 154, 55, 174, 7, 206, 91, 173, 172, 86, 202, 205, 224, 219,
            18, 78, 238, 82, 27, 188, 57, 53, 121, 6, 205, 131, 118,
        ],
        [
            210, 102, 28, 66, 200, 167, 175, 105, 68, 134, 10, 188, 116, 141, 72, 93, 141, 194,
            133, 106, 204, 148, 243, 135, 227, 17, 251, 103, 234, 83, 116, 164,
        ],
        [
            89, 203, 88, 237, 40, 157, 244, 6, 14, 69, 23, 71, 190, 238, 134, 195, 6, 188, 110,
            235, 25, 76, 134, 116, 246, 27, 117, 195, 25, 92, 15, 81,
        ],
        [
            134, 3, 170, 185, 85, 163, 208, 3, 81, 104, 139, 243, 30, 23, 171, 198, 233, 142, 149,
            185, 97, 94, 174, 114, 45, 160, 244, 226, 196, 163, 10, 103,
        ],
        [
            208, 49, 165, 62, 185, 114, 53, 183, 15, 98, 246, 82, 147, 45, 177, 189, 247, 40, 234,
            34, 156, 140, 168, 9, 213, 60, 95, 253, 145, 100, 38, 135,
        ],
    );

const SM120_TF32_QUALIFIED_TUNING_REVISION: u16 = F32_TF32_TUNING_REVISION;
#[cfg(test)]
const SM120_TF32_PORTABLE_QUALIFICATION_IDENTITY_CUDA_13_2: Tf32AutoQualificationIdentity =
    Tf32AutoQualificationIdentity {
        module_kind: ModuleKind::TriadSm80,
        module_target: "compute_120",
        device_target: "sm_120",
        compute_capability: (12, 0),
        multiprocessor_count: 170,
        nvrtc_version: (13, 2),
        optin_shared_bytes: 101_376,
        tensor_map_access: true,
        compile_key: [
            152, 216, 153, 132, 10, 117, 104, 1, 208, 62, 15, 125, 19, 15, 218, 83, 180, 144, 4,
            144, 211, 131, 159, 120, 140, 116, 131, 206, 41, 62, 77, 183,
        ],
        artifact_digest: [
            240, 103, 177, 251, 174, 24, 81, 83, 102, 109, 200, 106, 209, 179, 67, 191, 217, 255,
            129, 130, 70, 139, 203, 226, 105, 153, 35, 119, 142, 224, 129, 125,
        ],
        source_digest: [
            248, 83, 171, 12, 79, 34, 196, 226, 18, 202, 31, 229, 38, 226, 2, 51, 123, 119, 253,
            51, 62, 115, 206, 189, 212, 224, 105, 0, 75, 218, 119, 29,
        ],
        invocation_digest: [
            152, 216, 153, 132, 10, 117, 104, 1, 208, 62, 15, 125, 19, 15, 218, 83, 180, 144, 4,
            144, 211, 131, 159, 120, 140, 116, 131, 206, 41, 62, 77, 183,
        ],
        header_manifest_digest: [
            165, 215, 84, 8, 228, 132, 48, 141, 150, 196, 78, 149, 7, 182, 197, 11, 248, 193, 166,
            225, 180, 28, 187, 88, 5, 215, 224, 21, 237, 254, 181, 167,
        ],
        nvrtc_library_domain: [
            14, 13, 195, 250, 169, 151, 174, 150, 68, 46, 246, 47, 252, 2, 100, 13, 54, 26, 92, 80,
            224, 180, 229, 223, 42, 5, 188, 9, 134, 254, 98, 65,
        ],
    };
#[cfg(test)]
const SM120_TF32_QUALIFICATION_IDENTITY_CUDA_12_8: Tf32AutoQualificationIdentity =
    Tf32AutoQualificationIdentity {
        module_kind: ModuleKind::TriadSm120,
        module_target: "compute_120",
        device_target: "sm_120",
        compute_capability: (12, 0),
        multiprocessor_count: 170,
        nvrtc_version: (12, 8),
        optin_shared_bytes: 101_376,
        tensor_map_access: true,
        compile_key: [
            166, 206, 169, 76, 240, 169, 84, 100, 217, 7, 12, 212, 25, 4, 118, 118, 179, 222, 227,
            156, 49, 139, 185, 110, 235, 81, 26, 54, 11, 54, 128, 33,
        ],
        artifact_digest: [
            91, 106, 28, 188, 79, 11, 154, 143, 66, 25, 178, 253, 11, 152, 42, 219, 121, 139, 185,
            24, 232, 18, 21, 222, 72, 132, 54, 144, 222, 61, 247, 171,
        ],
        source_digest: [
            36, 109, 46, 53, 235, 5, 156, 214, 150, 236, 116, 50, 93, 74, 34, 51, 18, 250, 96, 95,
            232, 79, 245, 143, 23, 246, 99, 141, 153, 184, 24, 44,
        ],
        invocation_digest: [
            166, 206, 169, 76, 240, 169, 84, 100, 217, 7, 12, 212, 25, 4, 118, 118, 179, 222, 227,
            156, 49, 139, 185, 110, 235, 81, 26, 54, 11, 54, 128, 33,
        ],
        header_manifest_digest: [
            254, 128, 56, 210, 41, 106, 84, 102, 144, 148, 104, 242, 181, 2, 247, 203, 59, 245, 94,
            236, 49, 74, 64, 150, 164, 59, 2, 159, 109, 139, 29, 48,
        ],
        nvrtc_library_domain: [
            38, 176, 163, 160, 32, 68, 255, 203, 193, 105, 63, 216, 62, 146, 97, 190, 255, 166,
            146, 164, 251, 207, 227, 172, 94, 157, 140, 135, 152, 11, 177, 85,
        ],
    };

#[cfg(test)]
const SM120_TF32_QUALIFICATION_IDENTITY_CUDA_13_0: Tf32AutoQualificationIdentity =
    Tf32AutoQualificationIdentity {
        module_kind: ModuleKind::TriadSm120,
        module_target: "compute_120",
        device_target: "sm_120",
        compute_capability: (12, 0),
        multiprocessor_count: 170,
        nvrtc_version: (13, 0),
        optin_shared_bytes: 101_376,
        tensor_map_access: true,
        compile_key: [
            218, 36, 138, 227, 73, 184, 235, 221, 246, 237, 231, 11, 39, 108, 209, 191, 166, 32,
            167, 221, 251, 169, 63, 201, 34, 81, 127, 134, 136, 166, 71, 79,
        ],
        artifact_digest: [
            81, 129, 108, 141, 121, 6, 25, 109, 73, 170, 104, 7, 116, 44, 14, 147, 172, 23, 50,
            242, 58, 150, 71, 46, 170, 185, 125, 196, 70, 239, 181, 184,
        ],
        source_digest: [
            36, 109, 46, 53, 235, 5, 156, 214, 150, 236, 116, 50, 93, 74, 34, 51, 18, 250, 96, 95,
            232, 79, 245, 143, 23, 246, 99, 141, 153, 184, 24, 44,
        ],
        invocation_digest: [
            218, 36, 138, 227, 73, 184, 235, 221, 246, 237, 231, 11, 39, 108, 209, 191, 166, 32,
            167, 221, 251, 169, 63, 201, 34, 81, 127, 134, 136, 166, 71, 79,
        ],
        header_manifest_digest: [
            244, 255, 248, 65, 139, 210, 195, 70, 200, 110, 198, 240, 127, 94, 116, 23, 20, 124, 1,
            59, 240, 56, 205, 193, 109, 243, 231, 108, 243, 248, 10, 217,
        ],
        nvrtc_library_domain: [
            112, 155, 145, 195, 107, 251, 14, 217, 102, 238, 105, 173, 200, 214, 248, 127, 241, 16,
            238, 207, 61, 251, 80, 96, 54, 127, 24, 60, 230, 20, 235, 13,
        ],
    };

/// Frozen CUDA 13.2 qualification identity.
#[cfg(test)]
const SM120_TF32_QUALIFICATION_IDENTITY: Tf32AutoQualificationIdentity =
    Tf32AutoQualificationIdentity {
        module_kind: ModuleKind::TriadSm120,
        module_target: "compute_120",
        device_target: "sm_120",
        compute_capability: (12, 0),
        multiprocessor_count: 170,
        nvrtc_version: (13, 2),
        optin_shared_bytes: 101_376,
        tensor_map_access: true,
        compile_key: [
            112, 238, 69, 136, 32, 242, 124, 85, 78, 231, 125, 63, 137, 201, 144, 207, 0, 93, 126,
            62, 86, 109, 196, 135, 234, 127, 136, 146, 46, 195, 54, 69,
        ],
        artifact_digest: [
            80, 206, 171, 198, 77, 105, 232, 87, 90, 121, 116, 211, 68, 150, 5, 152, 12, 177, 180,
            45, 222, 248, 149, 42, 206, 55, 37, 7, 60, 217, 4, 65,
        ],
        source_digest: [
            101, 223, 188, 193, 100, 34, 155, 115, 243, 247, 111, 161, 80, 22, 226, 138, 58, 223,
            154, 210, 50, 128, 48, 235, 31, 230, 217, 126, 227, 208, 26, 175,
        ],
        invocation_digest: [
            112, 238, 69, 136, 32, 242, 124, 85, 78, 231, 125, 63, 137, 201, 144, 207, 0, 93, 126,
            62, 86, 109, 196, 135, 234, 127, 136, 146, 46, 195, 54, 69,
        ],
        header_manifest_digest: [
            144, 90, 202, 198, 154, 11, 239, 32, 177, 45, 241, 189, 47, 183, 11, 111, 139, 211,
            143, 197, 48, 236, 2, 191, 245, 242, 177, 36, 191, 141, 145, 87,
        ],
        nvrtc_library_domain: [
            14, 13, 195, 250, 169, 151, 174, 150, 68, 46, 246, 47, 252, 2, 100, 13, 54, 26, 92, 80,
            224, 180, 229, 223, 42, 5, 188, 9, 134, 254, 98, 65,
        ],
    };

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Tf32AutoCell {
    op: crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp,
    shape: Tf32ExactShape,
    route: Tf32PhysicalRoute,
    operand_gate: Tf32AutoOperandGate,
}

/// One frozen qualification: the module identity it was measured on, the
/// portable module that served its portable routes (a specialized cohort
/// only; a cell measured into the portable module holds only while that
/// module is the one measured), the current dispatch epoch authorized to use
/// those unchanged measurements, and its cells. An epoch-only host change
/// carries retained identity/cell literals forward; it does not claim fresh
/// GPU qualification or rewrite the acquisition epoch in archived records.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Tf32AutoEvidenceCohort {
    identity: Tf32AutoQualificationIdentity,
    portable: Option<Tf32AutoQualificationIdentity>,
    tuning_revision: u16,
    cells: &'static [Tf32AutoCell],
}

const fn sm89_tf32_cell(
    op: crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp,
    output_rows: usize,
    output_columns: usize,
    reduction: usize,
    tile: super::contract::Tf32PortableTile,
    stages: super::contract::Tf32PortableStages,
    operand_gate: Tf32AutoOperandGate,
) -> Tf32AutoCell {
    Tf32AutoCell {
        op,
        shape: Tf32ExactShape {
            output_rows,
            output_columns,
            reduction,
        },
        route: Tf32PhysicalRoute::MmaTf32Rna(super::contract::Tf32PortableRoute { tile, stages }),
        operand_gate,
    }
}

/// A measured SM89 cell whose winner is not the plain portable route: the
/// split-K arms are physical routes of the same portable family.
const fn sm89_tf32_route_cell(
    op: crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp,
    dims: (usize, usize, usize),
    route: Tf32PhysicalRoute,
    operand_gate: Tf32AutoOperandGate,
) -> Tf32AutoCell {
    let (output_rows, output_columns, reduction) = dims;
    Tf32AutoCell {
        op,
        shape: Tf32ExactShape {
            output_rows,
            output_columns,
            reduction,
        },
        route,
        operand_gate,
    }
}

use super::contract::Tf32PortableStages::{S2, S3, S4};
use super::contract::Tf32PortableTile::{M16N32, M32N32, M64N64, M128N64, M128N128};
use crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp::{Nn, Nt, Tn};
use Tf32AutoOperandGate::{
    RequestContractSafe, RequiresNoBiasAndVectorAlignmentEvidence, RequiresNoBiasEvidence,
    RequiresVectorAlignmentEvidence,
};

/// Exact SM89 TF32 evidence inventory. Production preparation supplies the
/// concrete epilogue and pointers required to match an archived cell. The
/// request-only API therefore stays on the scalar route. Requalification
/// records follow the archived rows; the last matching record takes precedence.
const SM89_TF32_EVIDENCE_CELLS: &[Tf32AutoCell] = &[
    sm89_tf32_cell(
        Nn,
        16,
        2048,
        512,
        M16N32,
        S4,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(Nn, 49, 129, 65, M16N32, S4, RequiresNoBiasEvidence),
    sm89_tf32_cell(Nn, 65, 129, 49, M16N32, S4, RequiresNoBiasEvidence),
    sm89_tf32_cell(Nn, 129, 100, 131, M16N32, S4, RequiresNoBiasEvidence),
    sm89_tf32_cell(
        Nn,
        512,
        768,
        3072,
        M64N64,
        S3,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Nn,
        1024,
        128,
        256,
        M16N32,
        S4,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Nn,
        1024,
        512,
        128,
        M64N64,
        S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Nn,
        2048,
        768,
        1536,
        M128N64,
        S3,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Nn,
        2048,
        768,
        3072,
        M128N64,
        S3,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Nn,
        2048,
        3072,
        768,
        M64N64,
        S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Nn,
        4096,
        768,
        512,
        M128N64,
        S3,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Nn,
        4096,
        1536,
        3072,
        M64N64,
        S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Nn,
        4621,
        1928,
        384,
        M64N64,
        S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Tn,
        16,
        2048,
        512,
        M16N32,
        S4,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(Tn, 49, 129, 65, M16N32, S4, RequestContractSafe),
    sm89_tf32_cell(Tn, 65, 129, 49, M16N32, S4, RequestContractSafe),
    sm89_tf32_route_cell(
        Tn,
        (128, 512, 1024),
        Tf32PhysicalRoute::MmaTf32RnaSplitK8(super::contract::Tf32PortableRoute {
            tile: M32N32,
            stages: S4,
        }),
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(Tn, 131, 100, 129, M16N32, S4, RequestContractSafe),
    sm89_tf32_route_cell(
        Tn,
        (256, 128, 1024),
        Tf32PhysicalRoute::MmaTf32RnaSplitK8(super::contract::Tf32PortableRoute {
            tile: M32N32,
            stages: S3,
        }),
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Tn,
        512,
        384,
        256,
        M64N64,
        S3,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Tn,
        3072,
        768,
        512,
        M64N64,
        S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Tn,
        8192,
        128,
        128,
        M64N64,
        S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Nt,
        16,
        512,
        2048,
        M16N32,
        S4,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(Nt, 49, 65, 129, M16N32, S4, RequestContractSafe),
    sm89_tf32_cell(Nt, 65, 49, 129, M16N32, S4, RequestContractSafe),
    sm89_tf32_cell(
        Nt,
        128,
        8192,
        128,
        M64N64,
        S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Nt,
        512,
        16,
        2048,
        M16N32,
        S4,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Nt,
        512,
        3072,
        768,
        M64N64,
        S3,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Nt,
        1024,
        256,
        128,
        M16N32,
        S4,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Nt,
        2048,
        768,
        3072,
        M128N64,
        S3,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Nt,
        2048,
        1536,
        768,
        M128N64,
        S3,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Nt,
        2048,
        3072,
        768,
        M128N64,
        S3,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Nt,
        4096,
        512,
        768,
        M128N64,
        S3,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Nt,
        4621,
        384,
        1928,
        M128N64,
        S3,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Tn,
        768,
        3072,
        2048,
        M64N64,
        S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Tn,
        1536,
        768,
        2048,
        M64N64,
        S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Tn,
        384,
        1928,
        4621,
        M128N64,
        S3,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Tn,
        3072,
        1536,
        4096,
        M64N64,
        S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Nt,
        4096,
        3072,
        1536,
        M128N64,
        S3,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Tn,
        3072,
        768,
        2048,
        M64N64,
        S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_route_cell(
        Nt,
        (1024, 128, 512),
        Tf32PhysicalRoute::MmaTf32RnaSplitK4(super::contract::Tf32PortableRoute {
            tile: M16N32,
            stages: S3,
        }),
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Nn,
        10400,
        1536,
        384,
        M64N64,
        S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Nt,
        10400,
        384,
        1536,
        M128N64,
        S3,
        RequiresVectorAlignmentEvidence,
    ),
    // The hot training projections of the SM89 census, qualified on the
    // RTX 6000 Ada (internal/perf/sm89-requal-hot-20260904): prism_out_proj,
    // prism_input_proj, batch_input_proj, batch_out_proj, rect_tall and
    // underfill. Three TN cells (prism_out_proj, batch_input_proj,
    // batch_out_proj) kept the scalar winner: the TN TF32 family has no split
    // schedule for a 72-tile grid over a ten-thousand-row reduction.
    // nn_prism_out_proj: gemm_bi_nn_sm80_mma_tf32_m128n128_bk32_s3
    sm89_tf32_cell(
        Nn,
        4621,
        384,
        768,
        M128N128,
        S3,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    // nt_prism_out_proj: gemm_bi_nt_sm80_mma_tf32_m64n64_bk32_s2
    sm89_tf32_cell(
        Nt,
        4621,
        768,
        384,
        M64N64,
        S2,
        RequiresVectorAlignmentEvidence,
    ),
    // nn_prism_input_proj: gemm_bi_nn_sm80_mma_tf32_m128n128_bk32_s3
    sm89_tf32_cell(
        Nn,
        4621,
        384,
        1024,
        M128N128,
        S3,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    // tn_prism_input_proj: gemm_bi_tn_sm80_mma_tf32_splitk8_m64n64_bk32_s3
    sm89_tf32_route_cell(
        Tn,
        (1024, 384, 4621),
        Tf32PhysicalRoute::MmaTf32RnaSplitK8(super::contract::Tf32PortableRoute {
            tile: M64N64,
            stages: S3,
        }),
        RequiresVectorAlignmentEvidence,
    ),
    // nt_prism_input_proj: gemm_bi_nt_sm80_mma_tf32_m64n64_bk32_s2
    sm89_tf32_cell(
        Nt,
        4621,
        1024,
        384,
        M64N64,
        S2,
        RequiresVectorAlignmentEvidence,
    ),
    // nn_batch_input_proj: gemm_bi_nn_sm80_mma_tf32_m128n128_bk32_s3
    sm89_tf32_cell(
        Nn,
        10400,
        384,
        384,
        M128N128,
        S3,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    // nt_batch_input_proj: gemm_bi_nt_sm80_mma_tf32_m64n64_bk32_s2
    sm89_tf32_cell(
        Nt,
        10400,
        384,
        384,
        M64N64,
        S2,
        RequiresVectorAlignmentEvidence,
    ),
    // nn_batch_out_proj: gemm_bi_nn_sm80_mma_tf32_m128n128_bk32_s3
    sm89_tf32_cell(
        Nn,
        10400,
        384,
        768,
        M128N128,
        S3,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    // nt_batch_out_proj: gemm_bi_nt_sm80_mma_tf32_m128n64_bk32_s3
    sm89_tf32_cell(
        Nt,
        10400,
        768,
        384,
        M128N64,
        S3,
        RequiresVectorAlignmentEvidence,
    ),
    // nn_rect_tall: gemm_bi_nn_sm80_mma_tf32_m128n128_bk32_s3
    sm89_tf32_cell(
        Nn,
        4096,
        768,
        512,
        M128N128,
        S3,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    // tn_rect_tall: gemm_bi_tn_sm80_mma_tf32_splitk8_m64n64_bk32_s3
    sm89_tf32_route_cell(
        Tn,
        (512, 768, 4096),
        Tf32PhysicalRoute::MmaTf32RnaSplitK8(super::contract::Tf32PortableRoute {
            tile: M64N64,
            stages: S3,
        }),
        RequiresVectorAlignmentEvidence,
    ),
    // nt_rect_tall: gemm_bi_nt_sm80_mma_tf32_m128n64_bk32_s3
    sm89_tf32_cell(
        Nt,
        4096,
        512,
        768,
        M128N64,
        S3,
        RequiresVectorAlignmentEvidence,
    ),
    // nn_underfill: gemm_bi_nn_sm80_mma_tf32_splitk2_m16n32_bk32_s4
    sm89_tf32_route_cell(
        Nn,
        (256, 384, 512),
        Tf32PhysicalRoute::MmaTf32RnaSplitK2(super::contract::Tf32PortableRoute {
            tile: M16N32,
            stages: S4,
        }),
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    // tn_underfill: gemm_bi_tn_sm80_mma_tf32_m16n32_bk32_s4
    sm89_tf32_cell(
        Tn,
        512,
        384,
        256,
        M16N32,
        S4,
        RequiresVectorAlignmentEvidence,
    ),
    // nt_underfill: gemm_bi_nt_sm80_mma_tf32_m16n32_bk32_s4
    sm89_tf32_cell(
        Nt,
        256,
        512,
        384,
        M16N32,
        S4,
        RequiresVectorAlignmentEvidence,
    ),
    // tn_prism_out_proj: gemm_bi_tn_sm80_mma_tf32_splitk8_m64n64_bk32_s3
    sm89_tf32_route_cell(
        Tn,
        (768, 384, 4621),
        Tf32PhysicalRoute::MmaTf32RnaSplitK8(super::contract::Tf32PortableRoute {
            tile: M64N64,
            stages: S3,
        }),
        RequiresVectorAlignmentEvidence,
    ),
    // nn_d768_in_proj: gemm_bi_nn_sm80_mma_tf32_m128n128_bk32_s3
    sm89_tf32_cell(
        Nn,
        2048,
        3072,
        768,
        M128N128,
        S3,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    // nn_d768_out_proj: gemm_bi_nn_sm80_mma_tf32_m128n128_bk32_s3
    sm89_tf32_cell(
        Nn,
        2048,
        768,
        1536,
        M128N128,
        S3,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    // nn_prism_in_proj: gemm_bi_nn_sm80_mma_tf32_m128n128_bk32_s3
    sm89_tf32_cell(
        Nn,
        4621,
        1928,
        384,
        M128N128,
        S3,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    // nn_large_deep: gemm_bi_nn_sm80_mma_tf32_m128n128_bk32_s3
    sm89_tf32_cell(
        Nn,
        4096,
        1536,
        3072,
        M128N128,
        S3,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    // nn_large: gemm_bi_nn_sm80_mma_tf32_m128n128_bk32_s3
    sm89_tf32_cell(
        Nn,
        2048,
        768,
        3072,
        M128N128,
        S3,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    // nn_d128_in_proj: gemm_bi_nn_sm80_mma_tf32_m64n64_bk32_s2
    sm89_tf32_cell(
        Nn,
        1024,
        512,
        128,
        M64N64,
        S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    // nn_d128_out_proj: gemm_bi_nn_sm80_mma_tf32_m16n32_bk32_s4
    sm89_tf32_cell(
        Nn,
        1024,
        128,
        256,
        M16N32,
        S4,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    // nn_batch_in_proj: gemm_bi_nn_sm80_mma_tf32_m128n128_bk32_s3
    sm89_tf32_cell(
        Nn,
        10400,
        1536,
        384,
        M128N128,
        S3,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    // tn_batch_input_proj: gemm_bi_tn_sm80_mma_tf32_splitk8_m64n64_bk32_s3
    sm89_tf32_route_cell(
        Tn,
        (384, 384, 10400),
        Tf32PhysicalRoute::MmaTf32RnaSplitK8(super::contract::Tf32PortableRoute {
            tile: M64N64,
            stages: S3,
        }),
        RequiresVectorAlignmentEvidence,
    ),
    // tn_batch_out_proj: gemm_bi_tn_sm80_mma_tf32_splitk8_m64n64_bk32_s3
    sm89_tf32_route_cell(
        Tn,
        (768, 384, 10400),
        Tf32PhysicalRoute::MmaTf32RnaSplitK8(super::contract::Tf32PortableRoute {
            tile: M64N64,
            stages: S3,
        }),
        RequiresVectorAlignmentEvidence,
    ),
];

const fn sm120_tf32_streamk_cell(
    op: crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp,
    output_rows: usize,
    output_columns: usize,
    reduction: usize,
    stages: super::contract::Tf32Sm120Stages,
    operand_gate: Tf32AutoOperandGate,
) -> Tf32AutoCell {
    Tf32AutoCell {
        op,
        shape: Tf32ExactShape {
            output_rows,
            output_columns,
            reduction,
        },
        route: Tf32PhysicalRoute::Sm120TmaMmaTf32RnaStreamKV1(super::contract::Tf32Sm120Route {
            tile: super::contract::Tf32Sm120Tile::M64N128,
            stages,
        }),
        operand_gate,
    }
}

const fn sm120_tf32_cell(
    op: crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp,
    output_rows: usize,
    output_columns: usize,
    reduction: usize,
    tile: super::contract::Tf32Sm120Tile,
    stages: super::contract::Tf32Sm120Stages,
    operand_gate: Tf32AutoOperandGate,
) -> Tf32AutoCell {
    Tf32AutoCell {
        op,
        shape: Tf32ExactShape {
            output_rows,
            output_columns,
            reduction,
        },
        route: Tf32PhysicalRoute::Sm120TmaMmaTf32Rna(super::contract::Tf32Sm120Route {
            tile,
            stages,
        }),
        operand_gate,
    }
}

/// Exact CUDA 12.8 SM120 TF32 evidence inventory from the production
/// projection suite. This is intentionally literal: CUDA 12.8 and 13.2 chose
/// different routes for TN d768 input projection.
#[cfg(test)]
const SM120_TF32_EVIDENCE_CELLS_CUDA_12_8: &[Tf32AutoCell] = &[
    sm120_tf32_cell(
        Nn,
        2048,
        3072,
        768,
        super::contract::Tf32Sm120Tile::M64N128,
        super::contract::Tf32Sm120Stages::S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Tn,
        768,
        3072,
        2048,
        super::contract::Tf32Sm120Tile::M64N128,
        super::contract::Tf32Sm120Stages::S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nt,
        2048,
        768,
        3072,
        super::contract::Tf32Sm120Tile::M64N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nn,
        2048,
        768,
        1536,
        super::contract::Tf32Sm120Tile::M64N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Tn,
        1536,
        768,
        2048,
        super::contract::Tf32Sm120Tile::M64N128,
        super::contract::Tf32Sm120Stages::S4,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nt,
        2048,
        1536,
        768,
        super::contract::Tf32Sm120Tile::M64N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nn,
        4621,
        1928,
        384,
        super::contract::Tf32Sm120Tile::M128N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Tn,
        384,
        1928,
        4621,
        super::contract::Tf32Sm120Tile::M64N128,
        super::contract::Tf32Sm120Stages::S4,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nt,
        4621,
        384,
        1928,
        super::contract::Tf32Sm120Tile::M64N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresVectorAlignmentEvidence,
    ),
];

/// Exact CUDA 13.0 SM120 TF32 evidence inventory from the production
/// projection suite. Routes are bound to the exact 13.0 artifact and device
/// identity rather than inferred from either neighboring toolchain cohort.
#[cfg(test)]
const SM120_TF32_EVIDENCE_CELLS_CUDA_13_0: &[Tf32AutoCell] = &[
    sm120_tf32_cell(
        Nn,
        2048,
        3072,
        768,
        super::contract::Tf32Sm120Tile::M128N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Tn,
        768,
        3072,
        2048,
        super::contract::Tf32Sm120Tile::M128N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nt,
        2048,
        768,
        3072,
        super::contract::Tf32Sm120Tile::M64N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nn,
        2048,
        768,
        1536,
        super::contract::Tf32Sm120Tile::M64N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Tn,
        1536,
        768,
        2048,
        super::contract::Tf32Sm120Tile::M128N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nt,
        2048,
        1536,
        768,
        super::contract::Tf32Sm120Tile::M64N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nn,
        4621,
        1928,
        384,
        super::contract::Tf32Sm120Tile::M64N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Tn,
        384,
        1928,
        4621,
        super::contract::Tf32Sm120Tile::M128N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nt,
        4621,
        384,
        1928,
        super::contract::Tf32Sm120Tile::M64N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresVectorAlignmentEvidence,
    ),
];

/// Exact CUDA 13.2 SM120 TF32 evidence inventory from the production
/// projection suite. Operand-aware dispatch remains scalar unless the full
/// artifact and device identity matches this cohort.
#[cfg(test)]
const SM120_TF32_EVIDENCE_CELLS: &[Tf32AutoCell] = &[
    Tf32AutoCell {
        op: Tn,
        shape: Tf32ExactShape {
            output_rows: 512,
            output_columns: 384,
            reduction: 256,
        },
        route: Tf32PhysicalRoute::MmaTf32Rna(super::contract::Tf32PortableRoute {
            tile: M16N32,
            stages: S4,
        }),
        operand_gate: RequiresVectorAlignmentEvidence,
    },
    sm120_tf32_cell(
        Nn,
        512,
        768,
        3072,
        super::contract::Tf32Sm120Tile::M80N32Bk64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nn,
        2048,
        3072,
        768,
        super::contract::Tf32Sm120Tile::M64N128,
        super::contract::Tf32Sm120Stages::S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm120_tf32_streamk_cell(
        Tn,
        768,
        3072,
        2048,
        super::contract::Tf32Sm120Stages::S3,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nt,
        2048,
        768,
        3072,
        super::contract::Tf32Sm120Tile::M64N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nn,
        2048,
        768,
        1536,
        super::contract::Tf32Sm120Tile::M64N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm120_tf32_streamk_cell(
        Tn,
        1536,
        768,
        2048,
        super::contract::Tf32Sm120Stages::S3,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nt,
        2048,
        1536,
        768,
        super::contract::Tf32Sm120Tile::M64N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nn,
        4621,
        1928,
        384,
        super::contract::Tf32Sm120Tile::M128N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm120_tf32_streamk_cell(
        Tn,
        384,
        1928,
        4621,
        super::contract::Tf32Sm120Stages::S3,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nt,
        4621,
        384,
        1928,
        super::contract::Tf32Sm120Tile::M64N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresVectorAlignmentEvidence,
    ),
];

/// Fresh current-source CUDA 12.8 identities from the complete 24-key V2
/// packet under driver 595.84. Twenty-three keys admitted; true G10 stayed exact.
const SM120_TF32_QUALIFICATION_IDENTITY_CUDA_12_8_DRIVER_595_84: Tf32AutoQualificationIdentity =
    Tf32AutoQualificationIdentity {
        module_kind: ModuleKind::TriadSm120,
        module_target: "compute_120",
        device_target: "sm_120",
        compute_capability: (12, 0),
        multiprocessor_count: 170,
        nvrtc_version: (12, 8),
        optin_shared_bytes: 101376,
        tensor_map_access: true,
        compile_key: [
            132, 192, 75, 102, 174, 40, 187, 247, 149, 218, 223, 15, 141, 136, 49, 6, 3, 224, 158,
            95, 254, 234, 211, 120, 131, 253, 125, 21, 244, 106, 76, 83,
        ],
        artifact_digest: [
            171, 133, 41, 159, 53, 47, 98, 207, 49, 83, 34, 1, 58, 251, 253, 237, 183, 189, 3, 61,
            14, 189, 155, 112, 246, 10, 144, 138, 44, 44, 244, 233,
        ],
        source_digest: [
            36, 139, 183, 205, 251, 199, 80, 186, 15, 165, 225, 86, 237, 248, 80, 16, 137, 15, 170,
            125, 244, 171, 145, 106, 235, 151, 216, 194, 43, 181, 208, 53,
        ],
        invocation_digest: [
            132, 192, 75, 102, 174, 40, 187, 247, 149, 218, 223, 15, 141, 136, 49, 6, 3, 224, 158,
            95, 254, 234, 211, 120, 131, 253, 125, 21, 244, 106, 76, 83,
        ],
        header_manifest_digest: [
            107, 105, 213, 249, 88, 161, 184, 94, 143, 181, 202, 198, 90, 59, 174, 52, 166, 245,
            68, 74, 117, 121, 37, 229, 242, 249, 64, 158, 46, 3, 146, 200,
        ],
        nvrtc_library_domain: [
            38, 176, 163, 160, 32, 68, 255, 203, 193, 105, 63, 216, 62, 146, 97, 190, 255, 166,
            146, 164, 251, 207, 227, 172, 94, 157, 140, 135, 152, 11, 177, 85,
        ],
    };

const SM120_TF32_PORTABLE_QUALIFICATION_IDENTITY_CUDA_12_8_DRIVER_595_84:
    Tf32AutoQualificationIdentity = Tf32AutoQualificationIdentity {
    module_kind: ModuleKind::TriadSm80,
    module_target: "compute_120",
    device_target: "sm_120",
    compute_capability: (12, 0),
    multiprocessor_count: 170,
    nvrtc_version: (12, 8),
    optin_shared_bytes: 101376,
    tensor_map_access: true,
    compile_key: [
        199, 145, 160, 37, 64, 39, 127, 235, 182, 239, 88, 221, 245, 21, 50, 79, 129, 99, 193, 9,
        186, 113, 81, 164, 62, 131, 163, 4, 204, 153, 238, 180,
    ],
    artifact_digest: [
        62, 17, 231, 140, 31, 152, 101, 132, 235, 3, 175, 174, 192, 10, 33, 0, 5, 230, 112, 151,
        162, 245, 19, 231, 169, 100, 8, 184, 12, 149, 37, 63,
    ],
    source_digest: [
        248, 83, 171, 12, 79, 34, 196, 226, 18, 202, 31, 229, 38, 226, 2, 51, 123, 119, 253, 51,
        62, 115, 206, 189, 212, 224, 105, 0, 75, 218, 119, 29,
    ],
    invocation_digest: [
        199, 145, 160, 37, 64, 39, 127, 235, 182, 239, 88, 221, 245, 21, 50, 79, 129, 99, 193, 9,
        186, 113, 81, 164, 62, 131, 163, 4, 204, 153, 238, 180,
    ],
    header_manifest_digest: [
        7, 5, 151, 177, 61, 28, 130, 64, 239, 221, 141, 50, 38, 174, 147, 219, 73, 30, 21, 213, 37,
        14, 242, 205, 239, 142, 174, 43, 43, 210, 194, 118,
    ],
    nvrtc_library_domain: [
        38, 176, 163, 160, 32, 68, 255, 203, 193, 105, 63, 216, 62, 146, 97, 190, 255, 166, 146,
        164, 251, 207, 227, 172, 94, 157, 140, 135, 152, 11, 177, 85,
    ],
};

/// Fresh current-source CUDA 13.0 identities from the retained 24-key packet
/// under driver 595.84. Twenty-three keys admitted; true G10 stayed exact.
const SM120_TF32_QUALIFICATION_IDENTITY_CUDA_13_0_DRIVER_595_84: Tf32AutoQualificationIdentity =
    Tf32AutoQualificationIdentity {
        module_kind: ModuleKind::TriadSm120,
        module_target: "compute_120",
        device_target: "sm_120",
        compute_capability: (12, 0),
        multiprocessor_count: 170,
        nvrtc_version: (13, 0),
        optin_shared_bytes: 101376,
        tensor_map_access: true,
        compile_key: [
            177, 13, 162, 229, 113, 18, 50, 12, 177, 76, 146, 99, 33, 103, 124, 138, 174, 128, 13,
            210, 183, 78, 203, 127, 79, 216, 243, 238, 214, 147, 255, 40,
        ],
        artifact_digest: [
            11, 17, 185, 170, 132, 100, 215, 204, 34, 200, 23, 181, 117, 237, 214, 12, 148, 68, 99,
            61, 45, 96, 146, 160, 246, 196, 185, 204, 231, 110, 83, 55,
        ],
        source_digest: [
            36, 139, 183, 205, 251, 199, 80, 186, 15, 165, 225, 86, 237, 248, 80, 16, 137, 15, 170,
            125, 244, 171, 145, 106, 235, 151, 216, 194, 43, 181, 208, 53,
        ],
        invocation_digest: [
            177, 13, 162, 229, 113, 18, 50, 12, 177, 76, 146, 99, 33, 103, 124, 138, 174, 128, 13,
            210, 183, 78, 203, 127, 79, 216, 243, 238, 214, 147, 255, 40,
        ],
        header_manifest_digest: [
            185, 56, 225, 115, 89, 191, 148, 212, 209, 134, 65, 155, 74, 209, 60, 31, 158, 99, 216,
            194, 67, 97, 175, 76, 252, 59, 169, 30, 55, 83, 206, 218,
        ],
        nvrtc_library_domain: [
            112, 155, 145, 195, 107, 251, 14, 217, 102, 238, 105, 173, 200, 214, 248, 127, 241, 16,
            238, 207, 61, 251, 80, 96, 54, 127, 24, 60, 230, 20, 235, 13,
        ],
    };

const SM120_TF32_PORTABLE_QUALIFICATION_IDENTITY_CUDA_13_0_DRIVER_595_84:
    Tf32AutoQualificationIdentity = Tf32AutoQualificationIdentity {
    module_kind: ModuleKind::TriadSm80,
    module_target: "compute_120",
    device_target: "sm_120",
    compute_capability: (12, 0),
    multiprocessor_count: 170,
    nvrtc_version: (13, 0),
    optin_shared_bytes: 101376,
    tensor_map_access: true,
    compile_key: [
        78, 225, 87, 120, 190, 216, 217, 82, 241, 56, 244, 41, 8, 141, 70, 35, 45, 138, 153, 180,
        159, 61, 207, 76, 21, 19, 223, 164, 75, 39, 178, 48,
    ],
    artifact_digest: [
        194, 226, 42, 242, 147, 213, 225, 72, 96, 127, 121, 60, 232, 188, 47, 147, 175, 206, 58,
        255, 202, 228, 161, 212, 88, 129, 172, 135, 1, 159, 21, 229,
    ],
    source_digest: [
        248, 83, 171, 12, 79, 34, 196, 226, 18, 202, 31, 229, 38, 226, 2, 51, 123, 119, 253, 51,
        62, 115, 206, 189, 212, 224, 105, 0, 75, 218, 119, 29,
    ],
    invocation_digest: [
        78, 225, 87, 120, 190, 216, 217, 82, 241, 56, 244, 41, 8, 141, 70, 35, 45, 138, 153, 180,
        159, 61, 207, 76, 21, 19, 223, 164, 75, 39, 178, 48,
    ],
    header_manifest_digest: [
        215, 86, 53, 120, 9, 5, 218, 148, 34, 165, 196, 176, 165, 177, 204, 100, 53, 5, 161, 140,
        142, 24, 140, 31, 125, 214, 241, 126, 144, 170, 26, 128,
    ],
    nvrtc_library_domain: [
        112, 155, 145, 195, 107, 251, 14, 217, 102, 238, 105, 173, 200, 214, 248, 127, 241, 16,
        238, 207, 61, 251, 80, 96, 54, 127, 24, 60, 230, 20, 235, 13,
    ],
};

/// Frozen CUDA 13.2 qualification identity, measured on the 595.84 driver
/// build and applied on every driver that loads the same artifact.
const SM120_TF32_QUALIFICATION_IDENTITY_CUDA_13_2_DRIVER_595_84: Tf32AutoQualificationIdentity =
    Tf32AutoQualificationIdentity {
        module_kind: ModuleKind::TriadSm120,
        module_target: "compute_120",
        device_target: "sm_120",
        compute_capability: (12, 0),
        multiprocessor_count: 170,
        nvrtc_version: (13, 2),
        optin_shared_bytes: 101376,
        tensor_map_access: true,
        compile_key: [
            161, 21, 120, 21, 231, 39, 162, 110, 82, 73, 43, 75, 145, 145, 102, 200, 102, 47, 196,
            51, 28, 170, 184, 46, 43, 66, 191, 134, 182, 223, 50, 241,
        ],
        artifact_digest: [
            106, 95, 235, 158, 70, 210, 91, 99, 94, 200, 78, 83, 179, 154, 173, 100, 79, 151, 216,
            63, 160, 108, 203, 200, 217, 76, 149, 250, 50, 173, 50, 53,
        ],
        source_digest: [
            36, 139, 183, 205, 251, 199, 80, 186, 15, 165, 225, 86, 237, 248, 80, 16, 137, 15, 170,
            125, 244, 171, 145, 106, 235, 151, 216, 194, 43, 181, 208, 53,
        ],
        invocation_digest: [
            161, 21, 120, 21, 231, 39, 162, 110, 82, 73, 43, 75, 145, 145, 102, 200, 102, 47, 196,
            51, 28, 170, 184, 46, 43, 66, 191, 134, 182, 223, 50, 241,
        ],
        header_manifest_digest: [
            83, 5, 9, 183, 38, 144, 254, 0, 97, 45, 96, 198, 191, 90, 169, 121, 167, 151, 73, 229,
            199, 42, 148, 27, 144, 107, 218, 103, 117, 26, 249, 129,
        ],
        nvrtc_library_domain: [
            208, 49, 165, 62, 185, 114, 53, 183, 15, 98, 246, 82, 147, 45, 177, 189, 247, 40, 234,
            34, 156, 140, 168, 9, 213, 60, 95, 253, 145, 100, 38, 135,
        ],
    };

/// Portable identity paired with the CUDA 13.2 specialized module on 595.84.
/// The portable module the 595.84 cohort measured its portable routes on:
/// a portable cell of that cohort is admitted only while the board binds
/// this exact module beside the SM120 one.
const SM120_TF32_PORTABLE_QUALIFICATION_IDENTITY_CUDA_13_2_DRIVER_595_84:
    Tf32AutoQualificationIdentity = Tf32AutoQualificationIdentity {
    module_kind: ModuleKind::TriadSm80,
    module_target: "compute_120",
    device_target: "sm_120",
    compute_capability: (12, 0),
    multiprocessor_count: 170,
    nvrtc_version: (13, 2),
    optin_shared_bytes: 101376,
    tensor_map_access: true,
    compile_key: [
        188, 71, 140, 5, 199, 170, 187, 192, 19, 111, 242, 3, 52, 145, 171, 220, 158, 182, 94, 75,
        247, 127, 202, 22, 170, 178, 25, 204, 228, 5, 59, 122,
    ],
    artifact_digest: [
        235, 72, 246, 129, 24, 17, 240, 95, 126, 21, 163, 223, 218, 64, 171, 114, 6, 81, 121, 87,
        253, 45, 16, 108, 156, 252, 196, 37, 169, 112, 190, 50,
    ],
    source_digest: [
        248, 83, 171, 12, 79, 34, 196, 226, 18, 202, 31, 229, 38, 226, 2, 51, 123, 119, 253, 51,
        62, 115, 206, 189, 212, 224, 105, 0, 75, 218, 119, 29,
    ],
    invocation_digest: [
        188, 71, 140, 5, 199, 170, 187, 192, 19, 111, 242, 3, 52, 145, 171, 220, 158, 182, 94, 75,
        247, 127, 202, 22, 170, 178, 25, 204, 228, 5, 59, 122,
    ],
    header_manifest_digest: [
        165, 215, 84, 8, 228, 132, 48, 141, 150, 196, 78, 149, 7, 182, 197, 11, 248, 193, 166, 225,
        180, 28, 187, 88, 5, 215, 224, 21, 237, 254, 181, 167,
    ],
    nvrtc_library_domain: [
        208, 49, 165, 62, 185, 114, 53, 183, 15, 98, 246, 82, 147, 45, 177, 189, 247, 40, 234, 34,
        156, 140, 168, 9, 213, 60, 95, 253, 145, 100, 38, 135,
    ],
};

/// The CUDA 13.2 identity of the same module on the NVRTC 13.2.78 build, the
/// build the rented RTX 5090 boxes carry; the 595.84 constant above was
/// minted on the 13.2.51 build. Same source and headers, another compiler.
const SM120_TF32_QUALIFICATION_IDENTITY_CUDA_13_2_NVRTC_13_2_78: Tf32AutoQualificationIdentity =
    Tf32AutoQualificationIdentity {
        module_kind: ModuleKind::TriadSm120,
        module_target: "compute_120",
        device_target: "sm_120",
        compute_capability: (12, 0),
        multiprocessor_count: 170,
        nvrtc_version: (13, 2),
        optin_shared_bytes: 101376,
        tensor_map_access: true,
        compile_key: [
            83, 14, 55, 193, 108, 148, 105, 217, 72, 113, 63, 255, 96, 137, 255, 91, 224, 104, 191,
            43, 26, 29, 117, 150, 99, 196, 14, 84, 146, 153, 181, 3,
        ],
        artifact_digest: [
            7, 141, 64, 95, 152, 80, 167, 54, 4, 197, 229, 150, 80, 189, 106, 29, 34, 107, 147, 46,
            178, 113, 1, 240, 37, 8, 81, 124, 178, 255, 60, 161,
        ],
        source_digest: [
            36, 139, 183, 205, 251, 199, 80, 186, 15, 165, 225, 86, 237, 248, 80, 16, 137, 15, 170,
            125, 244, 171, 145, 106, 235, 151, 216, 194, 43, 181, 208, 53,
        ],
        invocation_digest: [
            83, 14, 55, 193, 108, 148, 105, 217, 72, 113, 63, 255, 96, 137, 255, 91, 224, 104, 191,
            43, 26, 29, 117, 150, 99, 196, 14, 84, 146, 153, 181, 3,
        ],
        header_manifest_digest: [
            83, 5, 9, 183, 38, 144, 254, 0, 97, 45, 96, 198, 191, 90, 169, 121, 167, 151, 73, 229,
            199, 42, 148, 27, 144, 107, 218, 103, 117, 26, 249, 129,
        ],
        nvrtc_library_domain: [
            220, 223, 96, 48, 189, 148, 19, 101, 183, 103, 158, 213, 226, 50, 226, 19, 235, 24,
            147, 98, 12, 178, 250, 224, 154, 245, 36, 3, 79, 180, 23, 212,
        ],
    };

/// The CUDA 13.2 identity of the same module on the NVRTC 13.2.78 build, the
/// build the rented RTX 5090 boxes carry; the 595.84 constant above was
/// minted on the 13.2.51 build. Same source and headers, another compiler.
const SM120_TF32_PORTABLE_QUALIFICATION_IDENTITY_CUDA_13_2_NVRTC_13_2_78:
    Tf32AutoQualificationIdentity = Tf32AutoQualificationIdentity {
    module_kind: ModuleKind::TriadSm80,
    module_target: "compute_120",
    device_target: "sm_120",
    compute_capability: (12, 0),
    multiprocessor_count: 170,
    nvrtc_version: (13, 2),
    optin_shared_bytes: 101376,
    tensor_map_access: true,
    compile_key: [
        181, 176, 245, 62, 216, 202, 217, 20, 121, 165, 210, 34, 228, 163, 54, 0, 54, 223, 46, 243,
        176, 29, 207, 45, 170, 255, 148, 250, 200, 31, 45, 93,
    ],
    artifact_digest: [
        52, 15, 227, 135, 175, 180, 249, 167, 251, 98, 32, 132, 30, 113, 28, 255, 51, 228, 67, 213,
        78, 99, 135, 191, 195, 61, 46, 119, 107, 72, 89, 151,
    ],
    source_digest: [
        248, 83, 171, 12, 79, 34, 196, 226, 18, 202, 31, 229, 38, 226, 2, 51, 123, 119, 253, 51,
        62, 115, 206, 189, 212, 224, 105, 0, 75, 218, 119, 29,
    ],
    invocation_digest: [
        181, 176, 245, 62, 216, 202, 217, 20, 121, 165, 210, 34, 228, 163, 54, 0, 54, 223, 46, 243,
        176, 29, 207, 45, 170, 255, 148, 250, 200, 31, 45, 93,
    ],
    header_manifest_digest: [
        165, 215, 84, 8, 228, 132, 48, 141, 150, 196, 78, 149, 7, 182, 197, 11, 248, 193, 166, 225,
        180, 28, 187, 88, 5, 215, 224, 21, 237, 254, 181, 167,
    ],
    nvrtc_library_domain: [
        220, 223, 96, 48, 189, 148, 19, 101, 183, 103, 158, 213, 226, 50, 226, 19, 235, 24, 147,
        98, 12, 178, 250, 224, 154, 245, 36, 3, 79, 180, 23, 212,
    ],
};

/// Stable current-source route/key manifest independently requalified on CUDA
/// 12.8, 13.0 and 13.2 under driver 595.84. All three packets admitted these
/// same 23 keys; true G10 `Tn(128,128,8192)` lost and is intentionally absent.
const SM120_TF32_EVIDENCE_CELLS_DRIVER_595_84_RETAINED: &[Tf32AutoCell] = &[
    sm120_tf32_cell(
        Nn,
        2048,
        3072,
        768,
        super::contract::Tf32Sm120Tile::M64N128,
        super::contract::Tf32Sm120Stages::S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm120_tf32_streamk_cell(
        Tn,
        768,
        3072,
        2048,
        super::contract::Tf32Sm120Stages::S3,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nt,
        2048,
        768,
        3072,
        super::contract::Tf32Sm120Tile::M64N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nn,
        2048,
        768,
        1536,
        super::contract::Tf32Sm120Tile::M64N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm120_tf32_streamk_cell(
        Tn,
        1536,
        768,
        2048,
        super::contract::Tf32Sm120Stages::S3,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nt,
        2048,
        1536,
        768,
        super::contract::Tf32Sm120Tile::M64N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nn,
        4621,
        1928,
        384,
        super::contract::Tf32Sm120Tile::M64N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm120_tf32_streamk_cell(
        Tn,
        384,
        1928,
        4621,
        super::contract::Tf32Sm120Stages::S3,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nt,
        4621,
        384,
        1928,
        super::contract::Tf32Sm120Tile::M64N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nn,
        4096,
        1536,
        3072,
        super::contract::Tf32Sm120Tile::M64N128,
        super::contract::Tf32Sm120Stages::S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm120_tf32_streamk_cell(
        Tn,
        3072,
        1536,
        4096,
        super::contract::Tf32Sm120Stages::S3,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nt,
        4096,
        3072,
        1536,
        super::contract::Tf32Sm120Tile::M64N128,
        super::contract::Tf32Sm120Stages::S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nn,
        2048,
        768,
        3072,
        super::contract::Tf32Sm120Tile::M64N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm120_tf32_streamk_cell(
        Tn,
        3072,
        768,
        2048,
        super::contract::Tf32Sm120Stages::S3,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nt,
        2048,
        3072,
        768,
        super::contract::Tf32Sm120Tile::M64N128,
        super::contract::Tf32Sm120Stages::S2,
        RequiresVectorAlignmentEvidence,
    ),
    Tf32AutoCell {
        op: Nn,
        shape: Tf32ExactShape {
            output_rows: 1024,
            output_columns: 512,
            reduction: 128,
        },
        route: Tf32PhysicalRoute::MmaTf32Rna(super::contract::Tf32PortableRoute {
            tile: super::contract::Tf32PortableTile::M64N64,
            stages: super::contract::Tf32PortableStages::S3,
        }),
        operand_gate: RequiresNoBiasAndVectorAlignmentEvidence,
    },
    Tf32AutoCell {
        op: Tn,
        shape: Tf32ExactShape {
            output_rows: 128,
            output_columns: 512,
            reduction: 1024,
        },
        route: Tf32PhysicalRoute::MmaTf32Rna(super::contract::Tf32PortableRoute {
            tile: super::contract::Tf32PortableTile::M16N32,
            stages: super::contract::Tf32PortableStages::S4,
        }),
        operand_gate: RequiresVectorAlignmentEvidence,
    },
    Tf32AutoCell {
        op: Nt,
        shape: Tf32ExactShape {
            output_rows: 1024,
            output_columns: 128,
            reduction: 512,
        },
        route: Tf32PhysicalRoute::MmaTf32Rna(super::contract::Tf32PortableRoute {
            tile: super::contract::Tf32PortableTile::M16N16,
            stages: super::contract::Tf32PortableStages::S4,
        }),
        operand_gate: RequiresVectorAlignmentEvidence,
    },
    Tf32AutoCell {
        op: Tn,
        shape: Tf32ExactShape {
            output_rows: 256,
            output_columns: 128,
            reduction: 1024,
        },
        route: Tf32PhysicalRoute::MmaTf32Rna(super::contract::Tf32PortableRoute {
            tile: super::contract::Tf32PortableTile::M16N16,
            stages: super::contract::Tf32PortableStages::S4,
        }),
        operand_gate: RequiresVectorAlignmentEvidence,
    },
    // The deep reductions of the training batch and the split candidate
    // (internal/perf/sm120-requal-deepk-20260903): no split-K arm won, and the
    // TN batch projection admitted no TF32 route at all.
    sm120_tf32_cell(
        Nn,
        10400,
        1536,
        384,
        super::contract::Tf32Sm120Tile::M64N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    Tf32AutoCell {
        op: Nt,
        shape: Tf32ExactShape {
            output_rows: 128,
            output_columns: 8192,
            reduction: 128,
        },
        route: Tf32PhysicalRoute::MmaTf32Rna(super::contract::Tf32PortableRoute {
            tile: super::contract::Tf32PortableTile::M64N64,
            stages: super::contract::Tf32PortableStages::S2,
        }),
        operand_gate: RequiresVectorAlignmentEvidence,
    },
    sm120_tf32_cell(
        Nt,
        10400,
        384,
        1536,
        super::contract::Tf32Sm120Tile::M64N128,
        super::contract::Tf32Sm120Stages::S2,
        RequiresVectorAlignmentEvidence,
    ),
    // Requalified on the current 595.84 / CUDA 13.2 cohort at tuning
    // revision 45. This portable route beat the exact production fallback in
    // both eager and graph paths over the independent 21+101-window protocol.
    Tf32AutoCell {
        op: Tn,
        shape: Tf32ExactShape {
            output_rows: 512,
            output_columns: 384,
            reduction: 256,
        },
        route: Tf32PhysicalRoute::MmaTf32Rna(super::contract::Tf32PortableRoute {
            tile: super::contract::Tf32PortableTile::M16N32,
            stages: super::contract::Tf32PortableStages::S4,
        }),
        operand_gate: RequiresVectorAlignmentEvidence,
    },
];

/// The SM120 cohorts that describe this tree: every entry is frozen against
/// the module source the tree contains, so each one can match a board. A
/// cohort is keyed by the toolkit, the composed source and the board, never
/// by the driver build: the kernels' bits come from the compiler and the
/// source, and a box may run any driver that loads the artifact.
const SM120_TF32_EVIDENCE_COHORTS: &[Tf32AutoEvidenceCohort] = &[
    Tf32AutoEvidenceCohort {
        identity: SM120_TF32_QUALIFICATION_IDENTITY_CUDA_12_8_DRIVER_595_84,
        portable: Some(SM120_TF32_PORTABLE_QUALIFICATION_IDENTITY_CUDA_12_8_DRIVER_595_84),
        tuning_revision: SM120_TF32_QUALIFIED_TUNING_REVISION,
        cells: SM120_TF32_EVIDENCE_CELLS_DRIVER_595_84_RETAINED,
    },
    Tf32AutoEvidenceCohort {
        identity: SM120_TF32_QUALIFICATION_IDENTITY_CUDA_13_0_DRIVER_595_84,
        portable: Some(SM120_TF32_PORTABLE_QUALIFICATION_IDENTITY_CUDA_13_0_DRIVER_595_84),
        tuning_revision: SM120_TF32_QUALIFIED_TUNING_REVISION,
        cells: SM120_TF32_EVIDENCE_CELLS_DRIVER_595_84_RETAINED,
    },
    Tf32AutoEvidenceCohort {
        identity: SM120_TF32_QUALIFICATION_IDENTITY_CUDA_13_2_DRIVER_595_84,
        portable: Some(SM120_TF32_PORTABLE_QUALIFICATION_IDENTITY_CUDA_13_2_DRIVER_595_84),
        tuning_revision: SM120_TF32_QUALIFIED_TUNING_REVISION,
        cells: SM120_TF32_EVIDENCE_CELLS_DRIVER_595_84_RETAINED,
    },
    // The same cells, compiled by the NVRTC 13.2.78 build: a cohort binds
    // one compiler build, and the rented boards carry this one.
    Tf32AutoEvidenceCohort {
        identity: SM120_TF32_QUALIFICATION_IDENTITY_CUDA_13_2_NVRTC_13_2_78,
        portable: Some(SM120_TF32_PORTABLE_QUALIFICATION_IDENTITY_CUDA_13_2_NVRTC_13_2_78),
        tuning_revision: SM120_TF32_QUALIFIED_TUNING_REVISION,
        cells: SM120_TF32_EVIDENCE_CELLS_DRIVER_595_84_RETAINED,
    },
];

/// The SM120 cohorts frozen against a module source this tree no longer
/// contains. None of them can match a board, so they leave the runtime table;
/// their literal manifests stay as the record of what those stacks chose,
/// pinned by the tests, until a board of that stack requalifies them.
#[cfg(test)]
const SM120_TF32_RETIRED_COHORTS: &[Tf32AutoEvidenceCohort] = &[
    Tf32AutoEvidenceCohort {
        identity: SM120_TF32_QUALIFICATION_IDENTITY_CUDA_12_8,
        portable: None,
        tuning_revision: SM120_TF32_QUALIFIED_TUNING_REVISION,
        cells: SM120_TF32_EVIDENCE_CELLS_CUDA_12_8,
    },
    Tf32AutoEvidenceCohort {
        identity: SM120_TF32_QUALIFICATION_IDENTITY_CUDA_13_0,
        portable: None,
        tuning_revision: SM120_TF32_QUALIFIED_TUNING_REVISION,
        cells: SM120_TF32_EVIDENCE_CELLS_CUDA_13_0,
    },
    Tf32AutoEvidenceCohort {
        identity: SM120_TF32_QUALIFICATION_IDENTITY,
        portable: Some(SM120_TF32_PORTABLE_QUALIFICATION_IDENTITY_CUDA_13_2),
        tuning_revision: SM120_TF32_QUALIFIED_TUNING_REVISION,
        cells: SM120_TF32_EVIDENCE_CELLS,
    },
];

/// The portable cohorts: boards without a specialized module read these.
const SM89_TF32_EVIDENCE_COHORTS: &[Tf32AutoEvidenceCohort] = &[Tf32AutoEvidenceCohort {
    identity: SM89_TF32_QUALIFICATION_IDENTITY,
    portable: None,
    tuning_revision: SM89_TF32_QUALIFIED_TUNING_REVISION,
    cells: SM89_TF32_EVIDENCE_CELLS,
}];

/// CUDA 12.8 and 13.0 share the same exact-cell winner map. Eight rows move
/// to the joint Ada module; NN Prism retains the measured portable winner.
const SM89_JOINT_TF32_EVIDENCE_CELLS_LOWER: &[Tf32AutoCell] = &[
    sm89_tf32_route_cell(
        Tn,
        (768, 3072, 2048),
        Tf32PhysicalRoute::Sm89TnPreRnaM96N192S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_route_cell(
        Tn,
        (1536, 768, 2048),
        Tf32PhysicalRoute::Sm89TnPreRnaM96N96S3,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_route_cell(
        Tn,
        (384, 1928, 4621),
        Tf32PhysicalRoute::Sm89TnPreRnaM64N64,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Nn,
        4621,
        1928,
        384,
        M128N128,
        S3,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm89_tf32_route_cell(
        Nn,
        (2048, 768, 1536),
        Tf32PhysicalRoute::Sm89NnN96,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm89_tf32_route_cell(
        Tn,
        (3072, 1536, 4096),
        Tf32PhysicalRoute::Sm89TnDirectM192N192S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_route_cell(
        Nt,
        (2048, 768, 3072),
        Tf32PhysicalRoute::Sm89NtALdmatrixN96,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm89_tf32_route_cell(
        Nt,
        (2048, 1536, 768),
        Tf32PhysicalRoute::Sm89NtALdmatrixN96,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm89_tf32_route_cell(
        Nt,
        (4096, 3072, 1536),
        Tf32PhysicalRoute::Sm89NtRowstageM128N192S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm89_tf32_route_cell(
        Nt,
        (4621, 384, 1928),
        Tf32PhysicalRoute::Sm89NtRnaM144N96S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
];

/// CUDA 13.2 independently qualified the prior five rows and the NT d768-in
/// A-ldmatrix winner into the joint Ada module.
const SM89_JOINT_TF32_EVIDENCE_CELLS_CUDA_13_2: &[Tf32AutoCell] = &[
    sm89_tf32_route_cell(
        Tn,
        (768, 3072, 2048),
        Tf32PhysicalRoute::Sm89TnPreRnaM96N192S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_route_cell(
        Tn,
        (1536, 768, 2048),
        Tf32PhysicalRoute::Sm89TnPreRnaM96N96S3,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_route_cell(
        Tn,
        (384, 1928, 4621),
        Tf32PhysicalRoute::Sm89TnPreRnaM64N96S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_route_cell(
        Nn,
        (4621, 1928, 384),
        Tf32PhysicalRoute::Sm89NnDirectN96,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm89_tf32_route_cell(
        Nn,
        (2048, 768, 1536),
        Tf32PhysicalRoute::Sm89NnN96,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm89_tf32_route_cell(
        Nt,
        (2048, 768, 3072),
        Tf32PhysicalRoute::Sm89NtALdmatrixN96,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm89_tf32_route_cell(
        Tn,
        (3072, 1536, 4096),
        Tf32PhysicalRoute::Sm89TnDirectM192N192S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_route_cell(
        Nt,
        (4096, 3072, 1536),
        Tf32PhysicalRoute::Sm89NtRowstageM128N192S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm89_tf32_route_cell(
        Nt,
        (2048, 1536, 768),
        Tf32PhysicalRoute::Sm89NtALdmatrixN96,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm89_tf32_route_cell(
        Nt,
        (4621, 384, 1928),
        Tf32PhysicalRoute::Sm89NtRnaM144N96S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
];

const SM89_JOINT_TF32_EVIDENCE_COHORTS: &[Tf32AutoEvidenceCohort] = &[
    Tf32AutoEvidenceCohort {
        identity: SM89_JOINT_TF32_IDENTITY_CUDA_12_8,
        portable: Some(SM89_PORTABLE_TF32_IDENTITY_CUDA_12_8),
        tuning_revision: super::contract::SM89_TF32_JOINT_TUNING_REVISION,
        cells: SM89_JOINT_TF32_EVIDENCE_CELLS_LOWER,
    },
    Tf32AutoEvidenceCohort {
        identity: SM89_JOINT_TF32_IDENTITY_CUDA_13_0,
        portable: Some(SM89_PORTABLE_TF32_IDENTITY_CUDA_13_0),
        tuning_revision: super::contract::SM89_TF32_JOINT_TUNING_REVISION,
        cells: SM89_JOINT_TF32_EVIDENCE_CELLS_LOWER,
    },
    Tf32AutoEvidenceCohort {
        identity: SM89_JOINT_TF32_IDENTITY_CUDA_13_2,
        portable: None,
        tuning_revision: super::contract::SM89_TF32_JOINT_TUNING_REVISION,
        cells: SM89_JOINT_TF32_EVIDENCE_CELLS_CUDA_13_2,
    },
];

/// CUDA 13.2 complete-module identity from the initial six-cohort census.
/// This unreleased binding does not claim the pending raw-word admission.
const SM89_FINALIST_TF32_IDENTITY_CUDA_13_2: Tf32AutoQualificationIdentity =
    Tf32AutoQualificationIdentity {
        module_kind: ModuleKind::TriadSm89Finalist,
        module_target: "sm_89",
        device_target: "sm_89",
        compute_capability: (8, 9),
        multiprocessor_count: 142,
        nvrtc_version: (13, 2),
        optin_shared_bytes: 101376,
        tensor_map_access: false,
        compile_key: [
            171, 72, 1, 138, 112, 54, 19, 38, 208, 129, 191, 41, 41, 158, 244, 127, 79, 40, 99, 49,
            73, 206, 169, 103, 201, 172, 35, 60, 12, 95, 195, 131,
        ],
        artifact_digest: [
            190, 17, 135, 37, 87, 248, 254, 189, 253, 86, 216, 34, 88, 221, 126, 171, 119, 72, 132,
            112, 219, 177, 41, 138, 20, 181, 202, 179, 77, 120, 172, 182,
        ],
        source_digest: [
            82, 166, 175, 139, 13, 222, 124, 184, 54, 18, 147, 104, 190, 115, 200, 50, 199, 252,
            198, 102, 115, 223, 190, 95, 64, 108, 122, 6, 113, 37, 216, 186,
        ],
        invocation_digest: [
            171, 72, 1, 138, 112, 54, 19, 38, 208, 129, 191, 41, 41, 158, 244, 127, 79, 40, 99, 49,
            73, 206, 169, 103, 201, 172, 35, 60, 12, 95, 195, 131,
        ],
        header_manifest_digest: [
            184, 72, 182, 75, 105, 29, 110, 26, 225, 224, 22, 241, 217, 227, 94, 196, 6, 169, 65,
            189, 215, 201, 109, 53, 157, 12, 178, 193, 25, 238, 247, 108,
        ],
        nvrtc_library_domain: [
            208, 49, 165, 62, 185, 114, 53, 183, 15, 98, 246, 82, 147, 45, 177, 189, 247, 40, 234,
            34, 156, 140, 168, 9, 213, 60, 95, 253, 145, 100, 38, 135,
        ],
    };

/// CUDA 12.8 complete-module identity from the initial six-cohort census.
const SM89_FINALIST_TF32_IDENTITY_CUDA_12_8: Tf32AutoQualificationIdentity =
    Tf32AutoQualificationIdentity {
        module_kind: ModuleKind::TriadSm89Finalist,
        module_target: "sm_89",
        device_target: "sm_89",
        compute_capability: (8, 9),
        multiprocessor_count: 142,
        nvrtc_version: (12, 8),
        optin_shared_bytes: 101376,
        tensor_map_access: false,
        compile_key: [
            235, 193, 182, 130, 116, 103, 174, 159, 63, 206, 131, 175, 205, 64, 119, 151, 34, 203,
            179, 89, 18, 217, 30, 232, 30, 140, 241, 92, 201, 156, 78, 7,
        ],
        artifact_digest: [
            87, 216, 213, 81, 91, 104, 49, 85, 154, 249, 63, 47, 147, 126, 34, 203, 252, 254, 162,
            89, 29, 79, 133, 116, 48, 163, 77, 144, 64, 15, 203, 51,
        ],
        source_digest: [
            82, 166, 175, 139, 13, 222, 124, 184, 54, 18, 147, 104, 190, 115, 200, 50, 199, 252,
            198, 102, 115, 223, 190, 95, 64, 108, 122, 6, 113, 37, 216, 186,
        ],
        invocation_digest: [
            235, 193, 182, 130, 116, 103, 174, 159, 63, 206, 131, 175, 205, 64, 119, 151, 34, 203,
            179, 89, 18, 217, 30, 232, 30, 140, 241, 92, 201, 156, 78, 7,
        ],
        header_manifest_digest: [
            139, 228, 11, 246, 18, 217, 251, 94, 204, 32, 205, 80, 22, 252, 163, 148, 110, 85, 76,
            215, 186, 230, 211, 154, 40, 128, 98, 235, 215, 179, 135, 235,
        ],
        nvrtc_library_domain: [
            38, 176, 163, 160, 32, 68, 255, 203, 193, 105, 63, 216, 62, 146, 97, 190, 255, 166,
            146, 164, 251, 207, 227, 172, 94, 157, 140, 135, 152, 11, 177, 85,
        ],
    };

/// CUDA 13.0 complete-module identity from the initial six-cohort census.
const SM89_FINALIST_TF32_IDENTITY_CUDA_13_0: Tf32AutoQualificationIdentity =
    Tf32AutoQualificationIdentity {
        module_kind: ModuleKind::TriadSm89Finalist,
        module_target: "sm_89",
        device_target: "sm_89",
        compute_capability: (8, 9),
        multiprocessor_count: 142,
        nvrtc_version: (13, 0),
        optin_shared_bytes: 101376,
        tensor_map_access: false,
        compile_key: [
            252, 44, 190, 165, 133, 77, 131, 111, 149, 245, 3, 142, 141, 94, 215, 174, 96, 5, 162,
            16, 19, 14, 248, 24, 17, 242, 63, 169, 130, 116, 115, 62,
        ],
        artifact_digest: [
            181, 62, 15, 149, 230, 51, 192, 86, 101, 145, 185, 134, 13, 101, 238, 11, 176, 248, 56,
            87, 22, 170, 82, 116, 201, 115, 211, 5, 250, 41, 3, 163,
        ],
        source_digest: [
            82, 166, 175, 139, 13, 222, 124, 184, 54, 18, 147, 104, 190, 115, 200, 50, 199, 252,
            198, 102, 115, 223, 190, 95, 64, 108, 122, 6, 113, 37, 216, 186,
        ],
        invocation_digest: [
            252, 44, 190, 165, 133, 77, 131, 111, 149, 245, 3, 142, 141, 94, 215, 174, 96, 5, 162,
            16, 19, 14, 248, 24, 17, 242, 63, 169, 130, 116, 115, 62,
        ],
        header_manifest_digest: [
            45, 18, 8, 151, 34, 97, 78, 200, 98, 21, 60, 224, 91, 37, 226, 19, 106, 79, 195, 76,
            18, 221, 54, 189, 152, 128, 253, 23, 75, 182, 167, 207,
        ],
        nvrtc_library_domain: [
            112, 155, 145, 195, 107, 251, 14, 217, 102, 238, 105, 173, 200, 214, 248, 127, 241, 16,
            238, 207, 61, 251, 80, 96, 54, 127, 24, 60, 230, 20, 235, 13,
        ],
    };

const SM89_FINALIST_TF32_NT_CELLS: &[Tf32AutoCell] = &[
    sm89_tf32_route_cell(
        Nt,
        (2048, 768, 3072),
        Tf32PhysicalRoute::Sm89MmaTf32Compact8,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_route_cell(
        Nt,
        (2048, 1536, 768),
        Tf32PhysicalRoute::Sm89MmaTf32Compact8,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_route_cell(
        Nt,
        (4621, 384, 1928),
        Tf32PhysicalRoute::Sm89MmaTf32Compact8,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_route_cell(
        Nt,
        (4096, 3072, 1536),
        Tf32PhysicalRoute::Sm89MmaTf32Compact8,
        RequiresVectorAlignmentEvidence,
    ),
];

/// Separate finalist evidence never rewrites the incumbent portable cohort.
const SM89_FINALIST_TF32_EVIDENCE_COHORTS: &[Tf32AutoEvidenceCohort] = &[
    Tf32AutoEvidenceCohort {
        identity: SM89_FINALIST_TF32_IDENTITY_CUDA_12_8,
        portable: None,
        tuning_revision: super::contract::SM89_FINALIST_TUNING_REVISION,
        cells: SM89_FINALIST_TF32_NT_CELLS,
    },
    Tf32AutoEvidenceCohort {
        identity: SM89_FINALIST_TF32_IDENTITY_CUDA_13_0,
        portable: None,
        tuning_revision: super::contract::SM89_FINALIST_TUNING_REVISION,
        cells: SM89_FINALIST_TF32_NT_CELLS,
    },
    Tf32AutoEvidenceCohort {
        identity: SM89_FINALIST_TF32_IDENTITY_CUDA_13_2,
        portable: None,
        tuning_revision: super::contract::SM89_FINALIST_TUNING_REVISION,
        cells: SM89_FINALIST_TF32_NT_CELLS,
    },
];

/// No SM90a board has frozen a TF32 cohort yet; the family declines to the
/// portable ladder until one does.
const SM90A_TF32_EVIDENCE_COHORTS: &[Tf32AutoEvidenceCohort] = &[];

/// No SM100-family board has frozen a TF32 cohort yet; the family declines
/// to the portable ladder until one does.
const SM100_TF32_EVIDENCE_COHORTS: &[Tf32AutoEvidenceCohort] = &[];

/// The frozen TF32 evidence cohorts of the specialized module family a
/// board binds, with the family's name for the decline report.
fn tf32_evidence_cohorts(
    module: Tf32QualifiedModule,
) -> (&'static str, &'static [Tf32AutoEvidenceCohort]) {
    match module.module_kind {
        ModuleKind::TriadSm90a => ("SM90a", SM90A_TF32_EVIDENCE_COHORTS),
        ModuleKind::TriadSm100 => ("SM100", SM100_TF32_EVIDENCE_COHORTS),
        ModuleKind::TriadSm120 => ("SM120", SM120_TF32_EVIDENCE_COHORTS),
        // The specialized slot holds an architecture module alone; any
        // other kind here is a wiring error, and no cohort admits it.
        _ => ("unknown", &[]),
    }
}

fn matching_tf32_cohort(
    module: Tf32QualifiedModule,
    cohorts: &[Tf32AutoEvidenceCohort],
) -> Option<&Tf32AutoEvidenceCohort> {
    cohorts
        .iter()
        .find(|cohort| cohort.identity.matches(module))
}

fn measured_tf32_cell(
    request: F32TriadRequest,
    operands: F32TriadOperands,
    cells: &[Tf32AutoCell],
) -> Option<Tf32PhysicalRoute> {
    // Later qualification records supersede earlier results for the same
    // exact shape. Searching from the front can silently serve an obsolete
    // tile even when the cohort contains a measured replacement.
    cells
        .iter()
        .rfind(|cell| {
            cell.op == request.op
                && cell.shape.matches_contiguous(request)
                && tf32_auto_operands_match(request.op, operands)
        })
        .map(|cell| cell.route)
}

/// How far, per axis and in natural-log units, a shape may sit from the
/// nearest measured portable cell and still take that cell's tile: a
/// factor of four in output rows, output columns and reduction. The
/// leave-one-out check over the measured portable cells settled the width:
/// at a factor of four the band reaches 37 of the 58 cells and names the
/// measured tile for 31 of them, at a factor of eight it reaches 54 and
/// names it for 37, so the wider band buys its reach with a coin-flip on
/// the extra cells. Outside the band the exact f32 family serves, as it
/// did before.
const TF32_PORTABLE_NEIGHBOUR_LOG_BOUND: f64 = 1.386_294_361_119_890_6;

/// The portable tile of the nearest measured cell of the same operation,
/// for a shape no cell names exactly. Only the portable tier is widened
/// this way: its tiles keep one owner CTA per output tile and one k order,
/// so any tile of the tier gives the same bits, and what the neighbour
/// contributes is a speed choice. The specialized tiles stay on their
/// exact cells. A neighbour measured on a larger output may carry a wide
/// tile that leaves this board idle, so when the shape fills less than one
/// wave with it and less than half of what the neighbour filled, the 64x64
/// tile serves at the neighbour's pipeline depth, as the SM120 rule does.
fn nearest_tf32_portable_cell(
    request: F32TriadRequest,
    operands: F32TriadOperands,
    cells: &[Tf32AutoCell],
    multiprocessors: u32,
) -> Option<Tf32PhysicalRoute> {
    nearest_tf32_portable_cell_within(
        request,
        operands,
        cells,
        multiprocessors,
        TF32_PORTABLE_NEIGHBOUR_LOG_BOUND,
    )
}

fn nearest_tf32_portable_cell_within(
    request: F32TriadRequest,
    operands: F32TriadOperands,
    cells: &[Tf32AutoCell],
    multiprocessors: u32,
    log_bound: f64,
) -> Option<Tf32PhysicalRoute> {
    if multiprocessors == 0 || !tf32_auto_operands_match(request.op, operands) {
        return None;
    }
    let shape = request.shape;
    if shape != F32TriadShape::contiguous(request.op, (shape.m, shape.k, shape.n)) {
        return None;
    }
    let geometry = |rows: usize, columns: usize, reduction: usize| {
        [rows as f64, columns as f64, reduction as f64].map(f64::ln)
    };
    let target = geometry(
        shape.output_rows(request.op),
        shape.output_columns(request.op),
        shape.reduction(request.op),
    );
    let staging = tf32_staging_class(shape);
    let mut best: Option<(f64, &Tf32AutoCell)> = None;
    for cell in cells {
        if cell.op != request.op
            || cell.route.module_kind() != ModuleKind::TriadSm80
            || tf32_staging_class(cell.shape.contiguous(cell.op)) != staging
        {
            continue;
        }
        let axes = geometry(
            cell.shape.output_rows,
            cell.shape.output_columns,
            cell.shape.reduction,
        );
        let deltas = axes
            .iter()
            .zip(target.iter())
            .map(|(cell_axis, target_axis)| (cell_axis - target_axis).abs());
        if deltas.clone().any(|delta| delta > log_bound) {
            continue;
        }
        let distance = deltas.map(|delta| delta * delta).sum::<f64>().sqrt();
        // A later record supersedes an earlier one at the same distance,
        // as it does for an exact match.
        if best.is_none_or(|(best_distance, _)| distance <= best_distance) {
            best = Some((distance, cell));
        }
    }
    let (_, neighbour) = best?;
    let Tf32PhysicalRoute::MmaTf32Rna(portable) = neighbour.route else {
        return Some(neighbour.route);
    };
    let tile = tf32_kernel_spec(request.op, neighbour.route).ok()?.tile;
    let grid = |rows: usize, columns: usize| {
        (rows as f64 / f64::from(tile.0)).ceil() * (columns as f64 / f64::from(tile.1)).ceil()
    };
    let target_grid = grid(
        shape.output_rows(request.op),
        shape.output_columns(request.op),
    );
    let neighbour_grid = grid(neighbour.shape.output_rows, neighbour.shape.output_columns);
    let wide = matches!(
        portable.tile,
        super::contract::Tf32PortableTile::M128N64 | super::contract::Tf32PortableTile::M128N128
    );
    if wide && target_grid < f64::from(multiprocessors) && target_grid * 2.0 < neighbour_grid {
        return Some(Tf32PhysicalRoute::MmaTf32Rna(
            super::contract::Tf32PortableRoute {
                tile: super::contract::Tf32PortableTile::M64N64,
                stages: portable.stages,
            },
        ));
    }
    Some(neighbour.route)
}

/// Separately qualified NN bias epilogues on the frozen Ada module. Their
/// qualification measured 21 discovery and 101 final
/// windows per order/path, with repeat, graph and red-zone gates. No-bias
/// evidence elsewhere does not authorize this epilogue or adjacent shapes.
fn sm89_measured_tf32_bias_route(
    request: F32TriadRequest,
    operands: F32TriadOperands,
    cohort: &Tf32AutoEvidenceCohort,
) -> Option<Tf32PhysicalRoute> {
    if cohort.identity != SM89_TF32_QUALIFICATION_IDENTITY
        || request.op != ResolvedGemmOp::Nn
        || !operands
            .bias
            .is_some_and(|pointer| pointer != 0 && pointer.is_multiple_of(16))
        || !tf32_auto_operands_match(
            request.op,
            F32TriadOperands {
                bias: None,
                ..operands
            },
        )
    {
        return None;
    }
    let qualified = [
        (2048, 3072, 768),
        (2048, 768, 1536),
        (4621, 1928, 384),
        (4096, 1536, 3072),
        (2048, 768, 3072),
    ];
    qualified
        .into_iter()
        .any(|(output_rows, output_columns, reduction)| {
            Tf32ExactShape {
                output_rows,
                output_columns,
                reduction,
            }
            .matches_contiguous(request)
        })
        .then_some(Tf32PhysicalRoute::MmaTf32Rna(
            super::contract::Tf32PortableRoute {
                tile: M128N128,
                stages: S3,
            },
        ))
}

fn measured_tf32_route_with_operands(
    request: F32TriadRequest,
    operands: F32TriadOperands,
    availability: F32TriadAvailability,
    tuning_revision: u16,
) -> Option<Tf32PhysicalRoute> {
    if let Some(joint) = availability.joint
        && let Some(cohort) = matching_tf32_cohort(joint, SM89_JOINT_TF32_EVIDENCE_COHORTS)
        && cohort.tuning_revision == super::contract::SM89_TF32_JOINT_TUNING_REVISION
        && let Some(route) = measured_tf32_cell(request, operands, cohort.cells)
    {
        let portable_twin_matches = route.module_kind() != ModuleKind::TriadSm80
            || cohort.portable.is_some_and(|twin| {
                availability
                    .portable
                    .is_some_and(|portable| twin.matches(portable))
            });
        if portable_twin_matches && resolve_tf32_forced(request, availability, route).is_ok() {
            return Some(route);
        }
    }
    if let Some(finalist) = availability.finalist
        && let Some(cohort) = matching_tf32_cohort(finalist, SM89_FINALIST_TF32_EVIDENCE_COHORTS)
        && cohort.tuning_revision == super::contract::SM89_FINALIST_TUNING_REVISION
        && let Some(route) = measured_tf32_cell(request, operands, cohort.cells)
        && resolve_tf32_forced(request, availability, route).is_ok()
    {
        return Some(route);
    }
    // A board with a specialized module reads that family's cohorts; a
    // board without one reads the portable cohorts. Either way the cohort
    // must describe this stack whole: its identity, its tuning revision
    // and, for a portable route of a specialized cohort, the portable
    // module it measured that route on.
    let (family, module, cohorts) = match availability.specialized {
        Some(module) => {
            let (family, cohorts) = tf32_evidence_cohorts(module);
            (family, module, cohorts)
        }
        None => (
            "portable",
            availability.portable?,
            SM89_TF32_EVIDENCE_COHORTS,
        ),
    };
    let Some(cohort) = matching_tf32_cohort(module, cohorts) else {
        static NO_COHORT: std::sync::Once = std::sync::Once::new();
        crate::mamba_ssm::gpu::diagnostics::warn_once(&NO_COHORT, || {
            let newest = cohorts
                .last()
                .and_then(|cohort| cohort.identity.mismatch(module))
                .unwrap_or("no cohort is frozen for this family");
            format!(
                "no {family} TF32 evidence cohort matches this stack (newest cohort differs at: \
                 {newest}); the common TF32 routes serve through the first-use proof and the \
                 exact f32 family serves the rest until a requalification is frozen"
            )
        });
        return None;
    };
    if cohort.tuning_revision != tuning_revision {
        static STALE_TUNING: std::sync::Once = std::sync::Once::new();
        crate::mamba_ssm::gpu::diagnostics::warn_once(&STALE_TUNING, || {
            format!(
                "the {family} TF32 evidence cohort is authorized for dispatch epoch {} and this \
                 build runs epoch {tuning_revision}; the common TF32 routes serve through the \
                 first-use proof and the exact f32 family serves the rest until the dispatch \
                 epoch is reviewed",
                cohort.tuning_revision
            )
        });
        return None;
    }
    let route = measured_tf32_cell(request, operands, cohort.cells)
        .or_else(|| sm89_measured_tf32_bias_route(request, operands, cohort))
        .or_else(|| {
            availability.specialized.is_none().then(|| {
                nearest_tf32_portable_cell(
                    request,
                    operands,
                    cohort.cells,
                    availability.multiprocessors,
                )
            })?
        })?;
    if availability.specialized.is_some() && route.module_kind() == ModuleKind::TriadSm80 {
        let twin = cohort.portable?;
        if !availability
            .portable
            .is_some_and(|portable| twin.matches(portable))
        {
            return None;
        }
    }
    Some(route)
}

pub fn resolve_f32_triad_auto(
    policy: F32TriadPolicy,
    request: F32TriadRequest,
    availability: F32TriadAvailability,
) -> Result<F32TriadSelection, String> {
    resolve_f32_triad_auto_impl(policy, request, None, availability)
}

pub(super) fn resolve_f32_triad_auto_with_operands(
    policy: F32TriadPolicy,
    request: F32TriadRequest,
    operands: F32TriadOperands,
    availability: F32TriadAvailability,
) -> Result<F32TriadSelection, String> {
    resolve_f32_triad_auto_impl(policy, request, Some(operands), availability)
}

fn resolve_f32_triad_auto_impl(
    policy: F32TriadPolicy,
    request: F32TriadRequest,
    operands: Option<F32TriadOperands>,
    availability: F32TriadAvailability,
) -> Result<F32TriadSelection, String> {
    request.shape.validate(request.op)?;
    match policy {
        F32TriadPolicy::ExactScalarFma => {
            Ok(exact_or_scalar_selection(request, operands, availability))
        }
        F32TriadPolicy::AllowDeterministicTf32 => {
            if availability.portable.is_none()
                && availability.specialized.is_none()
                && availability.finalist.is_none()
                && availability.joint.is_none()
            {
                return Ok(F32TriadSelection::ScalarFma);
            }
            let route = match operands {
                Some(operands) => measured_tf32_route_with_operands(
                    request,
                    operands,
                    availability,
                    F32_TF32_TUNING_REVISION,
                ),
                None => None,
            };
            let Some(route) = route else {
                if let Some(operands) = operands
                    && let Some(selection) = proof_tf32_selection(request, operands, availability)
                {
                    return Ok(selection);
                }
                return Ok(exact_or_scalar_selection(request, operands, availability));
            };
            match resolve_tf32_forced(request, availability, route) {
                Ok(route) => Ok(F32TriadSelection::Tf32(route)),
                Err(reason) => {
                    static DECLINED: std::sync::Once = std::sync::Once::new();
                    crate::mamba_ssm::gpu::diagnostics::warn_once(&DECLINED, || {
                        format!(
                            "deterministic TF32 route {route:?} is measured for this shape but \
                             does not bind on this stack ({reason}); the exact family serves \
                             instead"
                        )
                    });
                    Ok(exact_or_scalar_selection(request, operands, availability))
                }
            }
        }
    }
}

/// The Ada evidence offered to a board that holds no cohort of its own.
/// The portable route the Ada portable cohort names for this cell serves by
/// design under the TF32 policy on every board that binds the portable
/// module: single-owner tiles and a fixed k-order make it deterministic
/// wherever it compiles. The specialized route for the same cell, when its
/// module is bound here, is offered as a candidate that must first
/// reproduce that portable route bit for bit on this board.
/// A module the proof tier may trust: its structure is what this build
/// composes and compiles, whatever board or toolkit produced its digests.
/// The frozen cohorts pin the provenance as well; this tier pins only what
/// a drifted provenance cannot change.
fn portable_module_well_formed(module: Tf32QualifiedModule) -> bool {
    module.module_kind == ModuleKind::TriadSm80
        && module.artifact.module_kind == ModuleKind::TriadSm80
        && module.artifact.artifact_kind == ArtifactKind::Ptx
        && module.compiler.output_kind == ArtifactKind::Ptx
        && module.compiler.nvrtc_library_known
        && module.compiler.composer_revision == COMPOSER_REVISION
        && module.compiler.compiler_revision == COMPILER_REVISION
        && module.compiler.numeric_abi_revision == NUMERIC_ABI_REVISION
        && module.compiler.schedule_revision == SCHEDULE_REVISION
        && module.compiler.target == module.target
        && module.device_caps.accepted_target == Some(module.target)
}

/// Whether the portable module this board bound composed the kernel a
/// common route names. The portable module of a CC 12.x board leaves the
/// extension kernels out (the wide TF32 tile and the split-K families),
/// and a common cell measured on the Ada may still name one; such a route
/// is declined here rather than failing at launch.
fn portable_route_composed(
    portable: Tf32QualifiedModule,
    op: ResolvedGemmOp,
    route: Tf32PhysicalRoute,
) -> bool {
    if route.module_kind() != ModuleKind::TriadSm80
        || super::contract::sm80_target_composes_extensions(portable.target.as_str())
    {
        return true;
    }
    match super::contract::tf32_kernel_spec(op, route) {
        Ok(spec) => super::contract::tf32_route_specs_for(ModuleKind::TriadSm80, false)
            .any(|base| base.symbol == spec.symbol),
        Err(_) => false,
    }
}

fn proof_tf32_selection(
    request: F32TriadRequest,
    operands: F32TriadOperands,
    availability: F32TriadAvailability,
) -> Option<F32TriadSelection> {
    let portable = availability
        .portable
        .filter(|module| portable_module_well_formed(*module))?;
    // A board's own measured cells were consulted before this point; a
    // shape they do not name takes the common tier here, on every board:
    // the measured common cell when one names the shape exactly, else the
    // nearest cell's portable tile within the band, each proven at first
    // use against the reference of the same numeric contract, and only
    // when this board's portable module composed the kernel it names.
    let composed =
        |route: &Tf32PhysicalRoute| portable_route_composed(portable, request.op, *route);
    let reference = measured_tf32_cell(request, operands, SM89_TF32_EVIDENCE_CELLS)
        .filter(|route| route.module_kind() == ModuleKind::TriadSm80)
        .filter(composed)
        .or_else(|| {
            nearest_tf32_portable_cell(
                request,
                operands,
                SM89_TF32_EVIDENCE_CELLS,
                availability.multiprocessors,
            )
            .filter(composed)
        })?;
    resolve_tf32_forced(request, availability, reference).ok()?;
    let specialized = [
        (
            availability.joint,
            SM89_JOINT_TF32_EVIDENCE_COHORTS
                .last()
                .map(|cohort| cohort.cells),
        ),
        (
            availability.finalist,
            SM89_FINALIST_TF32_EVIDENCE_COHORTS
                .last()
                .map(|cohort| cohort.cells),
        ),
    ];
    let candidate = specialized
        .into_iter()
        .filter_map(|(module, cells)| module.and(cells))
        .find_map(|cells| measured_tf32_cell(request, operands, cells))
        .filter(|route| {
            route.module_kind() != ModuleKind::TriadSm80
                && resolve_tf32_forced(request, availability, *route).is_ok()
        });
    Some(match candidate {
        Some(candidate) => F32TriadSelection::Tf32Proof {
            candidate,
            reference,
        },
        None => F32TriadSelection::Tf32(reference),
    })
}

/// The exact-F32 family is the floor under both policies. A shape with no
/// measured TF32 route still runs on the SM120 exact routes when the operands
/// admit them, and only falls through to the plain scalar chain when they do
/// not: allowing TF32 must never select something slower than forbidding it.
pub(super) fn exact_or_scalar_selection(
    request: F32TriadRequest,
    operands: Option<F32TriadOperands>,
    availability: F32TriadAvailability,
) -> F32TriadSelection {
    operands
        .and_then(|operands| sm120_fma_exact_route(request, operands, availability))
        .map_or(
            F32TriadSelection::ScalarFma,
            F32TriadSelection::ExactSm120Fma,
        )
}

pub fn resolve_tf32_forced(
    request: F32TriadRequest,
    availability: F32TriadAvailability,
    route: Tf32PhysicalRoute,
) -> Result<Tf32PhysicalRoute, String> {
    request.shape.validate(request.op)?;
    let (module_kind, dynamic_shared_bytes) = match route {
        Tf32PhysicalRoute::MmaTf32RnaSplitK2(_)
        | Tf32PhysicalRoute::MmaTf32RnaSplitK4(_)
        | Tf32PhysicalRoute::MmaTf32RnaSplitK8(_) => {
            let spec = super::contract::tf32_splitk_spec(request.op, route)?;
            (ModuleKind::TriadSm80, spec.dynamic_shared_bytes)
        }
        _ => {
            let spec = tf32_kernel_spec(request.op, route)?;
            (spec.module_kind, spec.dynamic_shared_bytes)
        }
    };
    let binding = match route {
        Tf32PhysicalRoute::MmaTf32Rna(_)
        | Tf32PhysicalRoute::MmaTf32RnaSplitK2(_)
        | Tf32PhysicalRoute::MmaTf32RnaSplitK4(_)
        | Tf32PhysicalRoute::MmaTf32RnaSplitK8(_) => availability.portable,
        Tf32PhysicalRoute::Sm89MmaTf32Compact8 => availability.finalist,
        Tf32PhysicalRoute::Sm89TnPreRnaN96
        | Tf32PhysicalRoute::Sm89TnPreRnaM64N64
        | Tf32PhysicalRoute::Sm89TnPreRnaM64N96S2
        | Tf32PhysicalRoute::Sm89NnDirectN96
        | Tf32PhysicalRoute::Sm89NnN96
        | Tf32PhysicalRoute::Sm89NtALdmatrixN96
        | Tf32PhysicalRoute::Sm89NtRnaM144N96S2
        | Tf32PhysicalRoute::Sm89NtRowstageM128N192S2
        | Tf32PhysicalRoute::Sm89TnDirectM192N192S2
        | Tf32PhysicalRoute::Sm89TnPreRnaM96N192S2
        | Tf32PhysicalRoute::Sm89TnPreRnaM96N96S3 => availability.joint,
        Tf32PhysicalRoute::Sm90aWgmmaTf32Tma(_)
        | Tf32PhysicalRoute::Sm100Tcgen05Tf32Tma(_)
        | Tf32PhysicalRoute::Sm120TmaMmaTf32Rna(_)
        | Tf32PhysicalRoute::Sm120TmaMmaTf32RnaStreamKV1(_)
        | Tf32PhysicalRoute::Sm120TmaFmaExact(_) => availability.specialized,
    }
    .ok_or_else(|| format!("forced TF32 route {route:?} has no qualified module"))?;
    ensure_tf32_binding_contract(binding, module_kind, dynamic_shared_bytes, route)?;
    if let Tf32PhysicalRoute::Sm120TmaFmaExact(exact) = route
        && binding.sm120_fma_exclusions.is_excluded(request.op, exact)
    {
        let symbol = tf32_kernel_spec(request.op, route)?.symbol;
        return Err(format!(
            "forced exact-F32 route {symbol} is excluded on this toolkit"
        ));
    }
    if !sm89_joint_route_matches_request(route, request) {
        return Err(format!(
            "forced TF32 route {route:?} is not qualified for {request:?}"
        ));
    }
    Ok(route)
}

/// The cells a joint route may be forced on: the ones its cohorts admit
/// and the ones the retile and deep-cell screens measure it on (the deep
/// 4096-row product and the d768 in_proj retile). Production selection
/// still comes only from the cohort tables.
fn sm89_joint_route_matches_request(route: Tf32PhysicalRoute, request: F32TriadRequest) -> bool {
    let shape = request.shape;
    let dims = (shape.m, shape.k, shape.n);
    let contiguous = shape == F32TriadShape::contiguous(request.op, dims);
    const DEEP: (usize, usize, usize) = (4_096, 3_072, 1_536);
    match route {
        Tf32PhysicalRoute::Sm89TnPreRnaN96 => {
            request.op == ResolvedGemmOp::Tn
                && matches!(dims, (2_048, 768, 3_072) | (2_048, 1_536, 768) | DEEP)
                && contiguous
        }
        Tf32PhysicalRoute::Sm89TnPreRnaM64N64 => {
            request.op == ResolvedGemmOp::Tn
                && matches!(dims, (4_621, 384, 1_928) | DEEP)
                && contiguous
        }
        Tf32PhysicalRoute::Sm89TnPreRnaM64N96S2 => {
            request.op == ResolvedGemmOp::Tn
                && matches!(dims, (4_621, 384, 1_928) | (2_048, 768, 3_072) | DEEP)
                && contiguous
        }
        Tf32PhysicalRoute::Sm89NnDirectN96 => {
            request.op == ResolvedGemmOp::Nn
                && matches!(dims, (4_621, 384, 1_928) | DEEP)
                && contiguous
        }
        Tf32PhysicalRoute::Sm89NnN96 => {
            request.op == ResolvedGemmOp::Nn
                && matches!(dims, (2_048, 1_536, 768) | DEEP)
                && contiguous
        }
        Tf32PhysicalRoute::Sm89NtALdmatrixN96 => {
            request.op == ResolvedGemmOp::Nt
                && matches!(
                    dims,
                    (2_048, 768, 3_072) | (2_048, 1_536, 768) | (4_621, 384, 1_928) | DEEP
                )
                && contiguous
        }
        Tf32PhysicalRoute::Sm89NtRnaM144N96S2 => {
            request.op == ResolvedGemmOp::Nt && dims == (4_621, 384, 1_928) && contiguous
        }
        Tf32PhysicalRoute::Sm89NtRowstageM128N192S2 => {
            request.op == ResolvedGemmOp::Nt && dims == DEEP && contiguous
        }
        Tf32PhysicalRoute::Sm89TnDirectM192N192S2 => {
            request.op == ResolvedGemmOp::Tn && dims == DEEP && contiguous
        }
        Tf32PhysicalRoute::Sm89TnPreRnaM96N192S2 => {
            request.op == ResolvedGemmOp::Tn && dims == (2_048, 768, 3_072) && contiguous
        }
        Tf32PhysicalRoute::Sm89TnPreRnaM96N96S3 => {
            request.op == ResolvedGemmOp::Tn && dims == (2_048, 1_536, 768) && contiguous
        }
        _ => true,
    }
}

fn ensure_tf32_binding_contract(
    binding: Tf32QualifiedModule,
    module_kind: ModuleKind,
    dynamic_shared_bytes: u32,
    route: Tf32PhysicalRoute,
) -> Result<(), String> {
    if binding.module_kind != module_kind
        || binding.module_kind != route.module_kind()
        || binding.artifact.module_kind != binding.module_kind
    {
        return Err(format!(
            "forced TF32 route {route:?} has the wrong module identity"
        ));
    }
    if binding.target != binding.compiler.target
        || binding.device_caps.accepted_target != Some(binding.target)
        || binding.compiler.output_kind != binding.artifact.artifact_kind
        || binding.artifact.compile_key != binding.compiler.invocation_digest
        || binding.compiler.composer_revision != COMPOSER_REVISION
        || binding.compiler.compiler_revision != COMPILER_REVISION
        || binding.compiler.numeric_abi_revision != NUMERIC_ABI_REVISION
        || binding.compiler.schedule_revision != SCHEDULE_REVISION
    {
        return Err(format!(
            "forced TF32 route {route:?} has an inconsistent compiler or artifact identity"
        ));
    }
    if binding.device.compute_capability != binding.device_caps.compute_capability
        || binding.compiler.nvrtc_version != binding.device_caps.nvrtc_version
    {
        return Err(format!(
            "forced TF32 route {route:?} has an inconsistent device identity"
        ));
    }
    if !target_admits_route(binding, route) {
        return Err(format!(
            "forced TF32 route {route:?} is not admitted by target {}",
            binding.target.as_str()
        ));
    }
    if !matches!(
        route,
        Tf32PhysicalRoute::MmaTf32Rna(_)
            | Tf32PhysicalRoute::MmaTf32RnaSplitK2(_)
            | Tf32PhysicalRoute::MmaTf32RnaSplitK4(_)
            | Tf32PhysicalRoute::MmaTf32RnaSplitK8(_)
            | Tf32PhysicalRoute::Sm89MmaTf32Compact8
            | Tf32PhysicalRoute::Sm89TnPreRnaN96
            | Tf32PhysicalRoute::Sm89TnPreRnaM64N64
            | Tf32PhysicalRoute::Sm89TnPreRnaM64N96S2
            | Tf32PhysicalRoute::Sm89NnDirectN96
            | Tf32PhysicalRoute::Sm89NnN96
            | Tf32PhysicalRoute::Sm89NtALdmatrixN96
            | Tf32PhysicalRoute::Sm89NtRnaM144N96S2
            | Tf32PhysicalRoute::Sm89NtRowstageM128N192S2
            | Tf32PhysicalRoute::Sm89TnDirectM192N192S2
            | Tf32PhysicalRoute::Sm89TnPreRnaM96N192S2
            | Tf32PhysicalRoute::Sm89TnPreRnaM96N96S3
    ) && !binding.device_caps.tensor_map_access
    {
        return Err(format!(
            "forced TF32 route {route:?} requires tensor-map access"
        ));
    }
    if binding.device_caps.optin_shared_bytes < dynamic_shared_bytes {
        return Err(format!(
            "forced TF32 route {route:?} needs {} shared bytes, device admits {}",
            dynamic_shared_bytes, binding.device_caps.optin_shared_bytes
        ));
    }
    Ok(())
}

fn target_admits_route(binding: Tf32QualifiedModule, route: Tf32PhysicalRoute) -> bool {
    let cc = binding.device.compute_capability;
    let target = binding.target.as_str();
    let device_target = binding.device.target.as_str();
    match route {
        Tf32PhysicalRoute::MmaTf32Rna(_)
        | Tf32PhysicalRoute::MmaTf32RnaSplitK2(_)
        | Tf32PhysicalRoute::MmaTf32RnaSplitK4(_)
        | Tf32PhysicalRoute::MmaTf32RnaSplitK8(_) => matches!(
            (cc, target, device_target),
            ((8, 0), "sm_80", "sm_80")
                | ((8, 6), "sm_86", "sm_86")
                | ((8, 7), "sm_87", "sm_87")
                | ((8, 9), "sm_89", "sm_89")
                | ((9, 0), "sm_90a", "sm_90a")
                | ((10, 0), "sm_100a", "sm_100a")
                | ((10, 1), "sm_101a", "sm_101a")
                | ((10, 3), "sm_103a", "sm_103a")
                | ((10, 7), "sm_107a", "sm_107a")
                | ((11, 0), "sm_110a", "sm_110a")
                | ((12, 0), "compute_120", "sm_120")
                | ((12, 1), "compute_121", "sm_121")
                | ((12, 1), "compute_120", "sm_120")
        ),
        // The Ada-found routes are sm_80-tier PTX: any board that compiles
        // their module for its own target runs them, and the first-use
        // proof decides their admission where no cohort does.
        Tf32PhysicalRoute::Sm89MmaTf32Compact8
        | Tf32PhysicalRoute::Sm89TnPreRnaN96
        | Tf32PhysicalRoute::Sm89TnPreRnaM64N64
        | Tf32PhysicalRoute::Sm89TnPreRnaM64N96S2
        | Tf32PhysicalRoute::Sm89NnDirectN96
        | Tf32PhysicalRoute::Sm89NnN96
        | Tf32PhysicalRoute::Sm89NtALdmatrixN96
        | Tf32PhysicalRoute::Sm89NtRnaM144N96S2
        | Tf32PhysicalRoute::Sm89NtRowstageM128N192S2
        | Tf32PhysicalRoute::Sm89TnDirectM192N192S2
        | Tf32PhysicalRoute::Sm89TnPreRnaM96N192S2
        | Tf32PhysicalRoute::Sm89TnPreRnaM96N96S3 => {
            cc.0 >= 8 && super::modules::sm80_ptx_target(target) == Some(device_target)
        }
        Tf32PhysicalRoute::Sm90aWgmmaTf32Tma(_) => {
            (cc, target, device_target) == ((9, 0), "sm_90a", "sm_90a")
        }
        Tf32PhysicalRoute::Sm100Tcgen05Tf32Tma(_) => matches!(
            (cc, target, device_target),
            ((10, 0), "compute_100f", "sm_100f")
                | ((10, 0), "compute_100a", "sm_100a")
                | ((10, 3), "compute_103f", "sm_103f")
                | ((10, 3), "compute_103a", "sm_103a")
                | ((10, 7), "compute_107f", "sm_107f")
                | ((10, 7), "compute_107a", "sm_107a")
                | ((11, 0), "compute_110f", "sm_110f")
                | ((11, 0), "compute_110a", "sm_110a")
        ),
        Tf32PhysicalRoute::Sm120TmaMmaTf32Rna(_)
        | Tf32PhysicalRoute::Sm120TmaMmaTf32RnaStreamKV1(_)
        | Tf32PhysicalRoute::Sm120TmaFmaExact(_) => matches!(
            (cc, target, device_target),
            ((12, 0), "compute_120", "sm_120")
                | ((12, 1), "compute_121", "sm_121")
                | ((12, 1), "compute_120", "sm_120")
        ),
    }
}

/// One exact-F32 SM120 cell measured under the exact policy: the shape in
/// the performance-matrix convention with contiguous strides, and the arm
/// that won its official 101-window qualification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Sm120FmaMeasuredCell {
    op: ResolvedGemmOp,
    shape: F32TriadShape,
    route: Sm120FmaRoute,
}

const fn sm120_fma_cell(
    op: ResolvedGemmOp,
    dims: (usize, usize, usize),
    tile: Sm120FmaTile,
    kvec: bool,
    splits: u8,
) -> Sm120FmaMeasuredCell {
    let (m, k, n) = dims;
    let (lda, ldb, ldc) = match op {
        ResolvedGemmOp::Nn => (k, n, n),
        ResolvedGemmOp::Tn => (k, n, n),
        ResolvedGemmOp::Nt => (n, n, k),
    };
    Sm120FmaMeasuredCell {
        op,
        shape: F32TriadShape {
            m,
            k,
            n,
            lda,
            ldb,
            ldc,
        },
        route: Sm120FmaRoute { tile, kvec, splits },
    }
}

/// The measured cells, each carrying the arm that won its qualification:
/// the nine research projections against the scalar production route
/// (1.22x to 1.60x, internal/perf/sm120-exact-nn-wide-20260903), and the
/// classifier serve and training-batch projections against what the exact
/// policy resolved before them (1.21x to 2.07x,
/// internal/perf/sm120-exact-wide-product-20260903).
const SM120_FMA_MEASURED_CELLS_CC120_170: [Sm120FmaMeasuredCell; 18] = [
    sm120_fma_cell(
        ResolvedGemmOp::Nn,
        (2_048, 768, 3_072),
        Sm120FmaTile::M128N64,
        false,
        1,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Nn,
        (2_048, 1_536, 768),
        Sm120FmaTile::M128N64,
        false,
        4,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Nn,
        (4_621, 384, 1_928),
        Sm120FmaTile::M64N128,
        false,
        1,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Tn,
        (2_048, 768, 3_072),
        Sm120FmaTile::M64N128,
        false,
        1,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Tn,
        (2_048, 1_536, 768),
        Sm120FmaTile::M128N64,
        false,
        2,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Tn,
        (4_621, 384, 1_928),
        Sm120FmaTile::M128N64,
        false,
        5,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Nt,
        (2_048, 768, 3_072),
        Sm120FmaTile::M128N64,
        true,
        5,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Nt,
        (2_048, 1_536, 768),
        Sm120FmaTile::M64N128,
        false,
        2,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Nt,
        (4_621, 384, 1_928),
        Sm120FmaTile::M128N64,
        true,
        3,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Nn,
        (4_621, 768, 384),
        Sm120FmaTile::M64N64,
        false,
        2,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Nn,
        (4_621, 1_024, 384),
        Sm120FmaTile::M64N64,
        false,
        2,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Nn,
        (10_400, 384, 384),
        Sm120FmaTile::M128N64,
        false,
        1,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Nn,
        (10_400, 768, 384),
        Sm120FmaTile::M128N64,
        false,
        1,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Tn,
        (10_400, 384, 1_536),
        Sm120FmaTile::M64N128,
        false,
        5,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Tn,
        (4_096, 3_072, 1_536),
        Sm120FmaTile::M64N128,
        false,
        2,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Nt,
        (10_400, 384, 1_536),
        Sm120FmaTile::M128N64,
        false,
        1,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Nt,
        (10_400, 384, 384),
        Sm120FmaTile::M64N64,
        false,
        1,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Nt,
        (4_621, 768, 384),
        Sm120FmaTile::M64N64,
        false,
        1,
    ),
];

/// Shapes past the measured cells take the exact family when they fill the
/// device with whole tiles on their own: at least one tile per
/// multiprocessor, no split, and a reduction long enough for the pipeline
/// to matter. The floor was three; the product-shape screen showed the
/// family beating the scalar route at 1.3 tiles per multiprocessor (1.68x
/// on the 4621x384 output) and at 2.9 (1.81x on the 10400x384 output).
const SM120_FMA_GENERIC_TILES_PER_MULTIPROCESSOR: usize = 1;
const SM120_FMA_GENERIC_MIN_EDGE: usize = 128;
const SM120_FMA_GENERIC_MIN_REDUCTION: usize = 256;

fn sm120_fma_generic_route(
    request: F32TriadRequest,
    multiprocessor_count: u32,
) -> Option<Sm120FmaRoute> {
    let (tile, kvec) = match request.op {
        ResolvedGemmOp::Nn => (Sm120FmaTile::M128N64, false),
        ResolvedGemmOp::Tn => (Sm120FmaTile::M64N128, false),
        ResolvedGemmOp::Nt => (Sm120FmaTile::M128N64, true),
    };
    let rows = request.shape.output_rows(request.op);
    let columns = request.shape.output_columns(request.op);
    let reduction = request.shape.reduction(request.op);
    if rows < SM120_FMA_GENERIC_MIN_EDGE
        || columns < SM120_FMA_GENERIC_MIN_EDGE
        || reduction < SM120_FMA_GENERIC_MIN_REDUCTION
    {
        return None;
    }
    let (bm, bn) = tile.dims();
    let tiles = rows
        .div_ceil(bm as usize)
        .checked_mul(columns.div_ceil(bn as usize))?;
    let required =
        (multiprocessor_count as usize).checked_mul(SM120_FMA_GENERIC_TILES_PER_MULTIPROCESSOR)?;
    (tiles >= required).then_some(Sm120FmaRoute {
        tile,
        kvec,
        splits: 1,
    })
}

fn f32_pointer_is_tma_aligned(pointer: super::contract::CUptr) -> bool {
    pointer != 0 && pointer.is_multiple_of(16)
}

/// The exact-F32 SM120 route for a request under the exact policy: the
/// specialized module must be bound on a CC 12.0 device, every operand must
/// sit on a 16-byte boundary with float4 leading dimensions (the tensor
/// maps demand it), and the shape must be a measured cell or fill the
/// device on its own. Anything else keeps the scalar routes.
pub(super) fn sm120_fma_exact_route(
    request: F32TriadRequest,
    operands: F32TriadOperands,
    availability: F32TriadAvailability,
) -> Option<Sm120FmaRoute> {
    let qualified = availability.specialized?;
    // The kernels are built for both minors of the family and the route
    // admission already names sm_121, so the gate follows the kernel rather
    // than the board the cells happen to be measured on.
    if qualified.module_kind != ModuleKind::TriadSm120
        || !matches!(qualified.device.compute_capability, (12, 0) | (12, 1))
        || !qualified.device_caps.tensor_map_access
    {
        return None;
    }
    let shape = request.shape;
    if shape.reduction(request.op) == 0
        || !f32_pointer_is_tma_aligned(operands.a)
        || !f32_pointer_is_tma_aligned(operands.b)
        || !f32_pointer_is_tma_aligned(operands.output)
        || !shape.lda.is_multiple_of(4)
        || !shape.ldb.is_multiple_of(4)
    {
        return None;
    }
    let measured = SM120_FMA_MEASURED_CELLS_CC120_170
        .iter()
        .find(|cell| cell.op == request.op && cell.shape == shape)
        .map(|cell| cell.route)
        .filter(|_| qualified.device.multiprocessor_count == 170);
    let route = measured
        .filter(|&route| {
            !qualified
                .sm120_fma_exclusions
                .is_excluded(request.op, route)
        })
        .or_else(|| {
            let mut route =
                sm120_fma_generic_route(request, qualified.device.multiprocessor_count)?;
            if let Some(measured) = measured {
                route.splits = measured.splits;
            }
            (!qualified
                .sm120_fma_exclusions
                .is_excluded(request.op, route))
            .then_some(route)
        })?;
    super::launch::sm120_fma_launch_plan(request, route).ok()?;
    Some(route)
}

pub const SM90A_AUTO_CELLS: &[Sm90aForcedRoute] = &[];
pub const SM100_AUTO_CELLS_CC100: &[Sm100ForcedRoute] = &[];
pub const SM100_AUTO_CELLS_CC103: &[Sm100ForcedRoute] = &[];
/// Automatic CC 10.7 routes. Empty until a board measures them.
pub const SM100_AUTO_CELLS_CC107: &[Sm100ForcedRoute] = &[];
/// Automatic CC 11.0 routes. Empty until a board measures them.
pub const SM100_AUTO_CELLS_CC110: &[Sm100ForcedRoute] = &[];

/// An automatic SM100 request: the measured table names the tile, the
/// caller only the operation, operands and shape.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sm100AutoRequest {
    pub op: Sm100Op,
    pub dtype: WeightDtype,
    pub shape: super::contract::Sm100Shape,
    pub a_ptr: super::contract::CUptr,
    pub b_ptr: super::contract::CUptr,
    pub operands: super::contract::Sm100LaunchOperands,
}

/// The measured SM100 cells of a board's compute capability. A minor with
/// no table has no cell to find and declines like an uncovered shape.
pub fn sm100_auto_cells(device_cc: (i32, i32)) -> &'static [Sm100ForcedRoute] {
    match device_cc {
        (10, 0) => SM100_AUTO_CELLS_CC100,
        (10, 3) => SM100_AUTO_CELLS_CC103,
        (10, 7) => SM100_AUTO_CELLS_CC107,
        (11, 0) => SM100_AUTO_CELLS_CC110,
        _ => &[],
    }
}

/// Resolves a measured SM100 cell for the request, or declines to the
/// portable caller. The table is per compute capability and stays empty
/// until a board of that capability qualifies its cells.
pub fn resolve_sm100_auto(
    device_cc: (i32, i32),
    module_target: Option<Sm100TargetCandidate>,
    request: Sm100AutoRequest,
) -> Option<Sm100ForcedRoute> {
    resolve_sm100_auto_from_cells(
        sm100_auto_cells(device_cc),
        device_cc,
        module_target,
        request,
    )
}

fn resolve_sm100_auto_from_cells(
    cells: &[Sm100ForcedRoute],
    device_cc: (i32, i32),
    module_target: Option<Sm100TargetCandidate>,
    request: Sm100AutoRequest,
) -> Option<Sm100ForcedRoute> {
    // A measured cell first; anything the table does not cover takes the wave
    // rule: the wide tile only when the output still fills a device with it,
    // deeper stages and the larger schedule only when the reduction is deep
    // enough to feed them. The map, operand and admission checks below apply
    // to both sources equally.
    let route = cells
        .iter()
        .copied()
        .find(|cell| {
            cell.op == request.op && cell.dtype == request.dtype && cell.shape == request.shape
        })
        .unwrap_or_else(|| {
            let (rows, columns, reduction) = match request.op {
                Sm100Op::Nn => (request.shape.m, request.shape.n, request.shape.k),
                Sm100Op::Tn => (request.shape.k, request.shape.n, request.shape.m),
                Sm100Op::Nt => (request.shape.m, request.shape.k, request.shape.n),
            };
            // No board of this family is qualified yet, so the tile floor is
            // the family's own SM counts: the smallest announced part carries
            // well over a hundred multiprocessors, and 128 wide tiles cover it.
            let wide_tiles = rows.div_ceil(128).saturating_mul(columns.div_ceil(128));
            let tile = if columns >= 128 && wide_tiles >= 128 {
                super::contract::Sm100Tile::M128N128
            } else {
                super::contract::Sm100Tile::M128N64
            };
            let stages = if reduction >= 2048 {
                super::contract::Sm100Stages::S4
            } else if reduction >= 512 {
                super::contract::Sm100Stages::S3
            } else {
                super::contract::Sm100Stages::S2
            };
            let schedule = if reduction >= 1024 {
                super::contract::Sm100Schedule::P8
            } else {
                super::contract::Sm100Schedule::C4
            };
            Sm100ForcedRoute {
                op: request.op,
                dtype: request.dtype,
                physical: super::contract::Sm100PhysicalRoute {
                    tile,
                    stages,
                    schedule,
                },
                shape: request.shape,
            }
        });
    super::contract::validate_sm100_map_request(super::contract::Sm100MapRequest {
        op: request.op,
        dtype: request.dtype,
        tile: route.physical.tile,
        a_ptr: request.a_ptr,
        b_ptr: request.b_ptr,
        shape: request.shape,
    })
    .ok()?;
    if !sm100_auto_operands_supported(request.op, request.operands) {
        return None;
    }
    (resolve_sm100_forced(device_cc, module_target, route).ok()? == Some(route)).then_some(route)
}

/// The epilogue contexts the measured SM100 cells were qualified under:
/// the same law as the SM120 tiles, until a board measures otherwise.
fn sm100_auto_operands_supported(
    op: Sm100Op,
    operands: super::contract::Sm100LaunchOperands,
) -> bool {
    let output_alignment = if op == Sm100Op::Tn { 4 } else { 2 };
    operands.output_ptr != 0
        && operands.output_ptr.is_multiple_of(output_alignment)
        && (operands.bias_ptr == 0 || operands.bias_ptr.is_multiple_of(4))
        && match op {
            Sm100Op::Nn => operands.bias_ptr == 0 || operands.alpha == 1.0,
            Sm100Op::Tn => operands.bias_ptr == 0 && operands.beta == 1.0,
            Sm100Op::Nt => operands.bias_ptr == 0 && operands.beta == 0.0,
        }
}

/// An automatic SM90a request: the measured table names the warpgroup
/// schedule, the caller only the operation, operands and shape.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sm90aAutoRequest {
    pub op: Sm90aOp,
    pub dtype: WeightDtype,
    pub shape: Sm90aShape,
    pub a_ptr: super::contract::CUptr,
    pub b_ptr: super::contract::CUptr,
    pub operands: super::contract::Sm90aLaunchOperands,
}

/// Resolves a measured SM90a cell for the request, or declines to the
/// portable caller. The table stays empty until a CC 9.0 board qualifies
/// its cells.
pub fn resolve_sm90a_auto(
    device_cc: (i32, i32),
    module_available: bool,
    request: Sm90aAutoRequest,
) -> Option<Sm90aForcedRoute> {
    resolve_sm90a_auto_from_cells(SM90A_AUTO_CELLS, device_cc, module_available, request)
}

fn resolve_sm90a_auto_from_cells(
    cells: &[Sm90aForcedRoute],
    device_cc: (i32, i32),
    module_available: bool,
    request: Sm90aAutoRequest,
) -> Option<Sm90aForcedRoute> {
    // A measured cell first; anything the table does not cover takes the wave
    // rule, so a Hopper board runs its own wgmma kernels rather than the
    // portable tiles. The second warpgroup of Wg2 is a dedicated TMA producer:
    // it earns its threads on a deep reduction and costs occupancy on a
    // shallow one, so the reduction depth is the axis the rule reads. The map,
    // operand and admission checks below apply to both sources equally.
    let route = cells
        .iter()
        .copied()
        .find(|cell| {
            cell.op == request.op && cell.dtype == request.dtype && cell.shape == request.shape
        })
        .unwrap_or_else(|| {
            let reduction = match request.op {
                Sm90aOp::Nn => request.shape.k,
                Sm90aOp::Tn => request.shape.m,
                Sm90aOp::Nt => request.shape.n,
            };
            let schedule = if reduction >= 1024 {
                Sm90aWarpgroupSchedule::Wg2
            } else {
                Sm90aWarpgroupSchedule::Wg1
            };
            Sm90aForcedRoute {
                op: request.op,
                dtype: request.dtype,
                schedule,
                shape: request.shape,
            }
        });
    super::contract::validate_sm90a_map_request(super::contract::Sm90aMapRequest {
        op: request.op,
        dtype: request.dtype,
        a_ptr: request.a_ptr,
        b_ptr: request.b_ptr,
        shape: request.shape,
    })
    .ok()?;
    if !sm90a_auto_operands_supported(request.op, request.operands) {
        return None;
    }
    let resolved = resolve_sm90a_forced(
        device_cc,
        module_available,
        route.op,
        route.dtype,
        route.schedule,
        route.shape,
    )
    .ok()?;
    (resolved == Some(route)).then_some(route)
}

/// The epilogue contexts the measured SM90a cells were qualified under:
/// the same law as the SM120 tiles, until a board measures otherwise.
fn sm90a_auto_operands_supported(
    op: Sm90aOp,
    operands: super::contract::Sm90aLaunchOperands,
) -> bool {
    let output_alignment = if op == Sm90aOp::Tn { 4 } else { 2 };
    operands.output_ptr != 0
        && operands.output_ptr.is_multiple_of(output_alignment)
        && (operands.bias_ptr == 0 || operands.bias_ptr.is_multiple_of(4))
        && match op {
            Sm90aOp::Nn => operands.bias_ptr == 0 || operands.alpha == 1.0,
            Sm90aOp::Tn => operands.bias_ptr == 0 && operands.beta == 1.0,
            Sm90aOp::Nt => operands.bias_ptr == 0 && operands.beta == 0.0,
        }
}
/// Measured BF16/F16 NN/TN/NT cells eligible for automatic CC 12.0 dispatch.
///
/// These routes are exact shape-and-stride matches; this is not a heuristic
/// table and it must not be reused for another compute-capability minor.
pub const SM120_AUTO_CELLS_CC120: &[Sm120ForcedRoute] = &[
    Sm120ForcedRoute {
        op: Sm120Op::Nn,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S2,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 2048,
            k: 1536,
            n: 768,
            lda: 1536,
            ldb: 768,
            ldc: 768,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nn,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S2,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 2048,
            k: 3072,
            n: 768,
            lda: 3072,
            ldb: 768,
            ldc: 768,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nn,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N64,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S2,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 4096,
            k: 3072,
            n: 1536,
            lda: 3072,
            ldb: 1536,
            ldc: 1536,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nn,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S2,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 2048,
            k: 1536,
            n: 768,
            lda: 1536,
            ldb: 768,
            ldc: 768,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nn,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S2,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 2048,
            k: 3072,
            n: 768,
            lda: 3072,
            ldb: 768,
            ldc: 768,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nn,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N64,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S2,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 4096,
            k: 3072,
            n: 1536,
            lda: 3072,
            ldb: 1536,
            ldc: 1536,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N128,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S3,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 2048,
            k: 1536,
            n: 768,
            lda: 1536,
            ldb: 768,
            ldc: 768,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N128,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S3,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 2048,
            k: 3072,
            n: 768,
            lda: 3072,
            ldb: 768,
            ldc: 768,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N128,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S3,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 4096,
            k: 3072,
            n: 1536,
            lda: 3072,
            ldb: 1536,
            ldc: 1536,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N128,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S3,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 2048,
            k: 1536,
            n: 768,
            lda: 1536,
            ldb: 768,
            ldc: 768,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N128,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S3,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 2048,
            k: 3072,
            n: 768,
            lda: 3072,
            ldb: 768,
            ldc: 768,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N128,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S3,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 4096,
            k: 3072,
            n: 1536,
            lda: 3072,
            ldb: 1536,
            ldc: 1536,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nt,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S2,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 2048,
            k: 1536,
            n: 768,
            lda: 768,
            ldb: 768,
            ldc: 1536,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nt,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N64,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S2,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 2048,
            k: 3072,
            n: 768,
            lda: 768,
            ldb: 768,
            ldc: 3072,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nt,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N128,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S3,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 4096,
            k: 3072,
            n: 1536,
            lda: 1536,
            ldb: 1536,
            ldc: 3072,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nt,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S2,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 2048,
            k: 1536,
            n: 768,
            lda: 768,
            ldb: 768,
            ldc: 1536,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nt,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N64,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S2,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 2048,
            k: 3072,
            n: 768,
            lda: 768,
            ldb: 768,
            ldc: 3072,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nt,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N128,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S3,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 4096,
            k: 3072,
            n: 1536,
            lda: 1536,
            ldb: 1536,
            ldc: 3072,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nn,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N64,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S2,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 2048,
            k: 768,
            n: 3072,
            lda: 768,
            ldb: 3072,
            ldc: 3072,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nn,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S2,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 4621,
            k: 384,
            n: 1928,
            lda: 384,
            ldb: 1928,
            ldc: 1928,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nn,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N64,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S2,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 2048,
            k: 768,
            n: 3072,
            lda: 768,
            ldb: 3072,
            ldc: 3072,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nn,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S2,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 4621,
            k: 384,
            n: 1928,
            lda: 384,
            ldb: 1928,
            ldc: 1928,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N128,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S3,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 2048,
            k: 768,
            n: 3072,
            lda: 768,
            ldb: 3072,
            ldc: 3072,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N128,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S3,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 4621,
            k: 384,
            n: 1928,
            lda: 384,
            ldb: 1928,
            ldc: 1928,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N128,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S3,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 2048,
            k: 768,
            n: 3072,
            lda: 768,
            ldb: 3072,
            ldc: 3072,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N128,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S3,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 4621,
            k: 384,
            n: 1928,
            lda: 384,
            ldb: 1928,
            ldc: 1928,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nt,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S2,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 2048,
            k: 768,
            n: 3072,
            lda: 3072,
            ldb: 3072,
            ldc: 768,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nt,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N128,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S3,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 4621,
            k: 384,
            n: 1928,
            lda: 1928,
            ldb: 1928,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nt,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S2,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 2048,
            k: 768,
            n: 3072,
            lda: 3072,
            ldb: 3072,
            ldc: 768,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nt,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N128,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S3,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 4621,
            k: 384,
            n: 1928,
            lda: 1928,
            ldb: 1928,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nn,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S2,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 4621,
            k: 768,
            n: 384,
            lda: 768,
            ldb: 384,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nn,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S2,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 4621,
            k: 1024,
            n: 384,
            lda: 1024,
            ldb: 384,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nn,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N64,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S2,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 10400,
            k: 384,
            n: 384,
            lda: 384,
            ldb: 384,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nn,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N128,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S3,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 10400,
            k: 384,
            n: 1536,
            lda: 384,
            ldb: 1536,
            ldc: 1536,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nn,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N64,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S2,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 10400,
            k: 768,
            n: 384,
            lda: 768,
            ldb: 384,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nn,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S2,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 4621,
            k: 768,
            n: 384,
            lda: 768,
            ldb: 384,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nn,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S2,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 4621,
            k: 1024,
            n: 384,
            lda: 1024,
            ldb: 384,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nn,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N64,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S2,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 10400,
            k: 384,
            n: 384,
            lda: 384,
            ldb: 384,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nn,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N128,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S3,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 10400,
            k: 384,
            n: 1536,
            lda: 384,
            ldb: 1536,
            ldc: 1536,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nn,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N64,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S2,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 10400,
            k: 768,
            n: 384,
            lda: 768,
            ldb: 384,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S3,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 4621,
            k: 768,
            n: 384,
            lda: 768,
            ldb: 384,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S3,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 4621,
            k: 1024,
            n: 384,
            lda: 1024,
            ldb: 384,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S3,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 10400,
            k: 384,
            n: 384,
            lda: 384,
            ldb: 384,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S3,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 10400,
            k: 384,
            n: 1536,
            lda: 384,
            ldb: 1536,
            ldc: 1536,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S3,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 10400,
            k: 768,
            n: 384,
            lda: 768,
            ldb: 384,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S3,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 4621,
            k: 768,
            n: 384,
            lda: 768,
            ldb: 384,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S3,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 4621,
            k: 1024,
            n: 384,
            lda: 1024,
            ldb: 384,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S3,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 10400,
            k: 384,
            n: 384,
            lda: 384,
            ldb: 384,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S3,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 10400,
            k: 384,
            n: 1536,
            lda: 384,
            ldb: 1536,
            ldc: 1536,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S3,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 10400,
            k: 768,
            n: 384,
            lda: 768,
            ldb: 384,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nt,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N128,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S2,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 4621,
            k: 768,
            n: 384,
            lda: 384,
            ldb: 384,
            ldc: 768,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nt,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N128,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S3,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 4621,
            k: 1024,
            n: 384,
            lda: 384,
            ldb: 384,
            ldc: 1024,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nt,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N64,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S2,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 10400,
            k: 384,
            n: 384,
            lda: 384,
            ldb: 384,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nt,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N64,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S2,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 10400,
            k: 384,
            n: 1536,
            lda: 1536,
            ldb: 1536,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nt,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N128,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S3,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 10400,
            k: 768,
            n: 384,
            lda: 384,
            ldb: 384,
            ldc: 768,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nt,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N128,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S2,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 4621,
            k: 768,
            n: 384,
            lda: 384,
            ldb: 384,
            ldc: 768,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nt,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N128,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S3,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 4621,
            k: 1024,
            n: 384,
            lda: 384,
            ldb: 384,
            ldc: 1024,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nt,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N64,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S2,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 10400,
            k: 384,
            n: 384,
            lda: 384,
            ldb: 384,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nt,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N64,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S2,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 10400,
            k: 384,
            n: 1536,
            lda: 1536,
            ldb: 1536,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nt,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N128,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S3,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 10400,
            k: 768,
            n: 384,
            lda: 384,
            ldb: 384,
            ldc: 768,
        },
    },
];
/// Automatic CC 12.1 routes. Empty until separate physical evidence exists.
pub const SM120_AUTO_CELLS_CC121: &[Sm120ForcedRoute] = &[];

/// Stream-K half cells measured on the 5090 (CC 12.0, driver 595.84;
/// internal/perf/sm120-streamk-tn-20260904): the TN shapes whose tile grid
/// underfills the 170 multiprocessors. Each names the persistent-grid twin of
/// the M64N64/BK64/S3 body and is served only under
/// `HalfTriadPolicy::AllowStreamKFixedOrder`; the tiled table keeps its own
/// cell for every one of these shapes, so the default policy loses nothing.
/// Against the best tiled route of the same shape (p50, both dtypes within
/// 0.01): 10400x384x384 0.32, 10400x768x384 0.53, 4621x768x384 0.60,
/// 4621x1024x384 0.74, 4621x384x1928 0.93, 10400x384x1536 0.96. Where the
/// tile grid already covers the multiprocessors the same body loses 1.22-1.77x,
/// which is why no such shape is listed and why the neighbour rule declines
/// a stream-K neighbour for a filled grid.
pub const SM120_STREAMK_CELLS_CC120: &[Sm120ForcedRoute] = &[
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S3,
            schedule: Sm120Schedule::StreamK,
        },
        shape: Sm120Shape {
            m: 4621,
            k: 384,
            n: 1928,
            lda: 384,
            ldb: 1928,
            ldc: 1928,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S3,
            schedule: Sm120Schedule::StreamK,
        },
        shape: Sm120Shape {
            m: 4621,
            k: 768,
            n: 384,
            lda: 768,
            ldb: 384,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S3,
            schedule: Sm120Schedule::StreamK,
        },
        shape: Sm120Shape {
            m: 4621,
            k: 1024,
            n: 384,
            lda: 1024,
            ldb: 384,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S3,
            schedule: Sm120Schedule::StreamK,
        },
        shape: Sm120Shape {
            m: 10400,
            k: 384,
            n: 384,
            lda: 384,
            ldb: 384,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S3,
            schedule: Sm120Schedule::StreamK,
        },
        shape: Sm120Shape {
            m: 10400,
            k: 384,
            n: 1536,
            lda: 384,
            ldb: 1536,
            ldc: 1536,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S3,
            schedule: Sm120Schedule::StreamK,
        },
        shape: Sm120Shape {
            m: 10400,
            k: 768,
            n: 384,
            lda: 768,
            ldb: 384,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S3,
            schedule: Sm120Schedule::StreamK,
        },
        shape: Sm120Shape {
            m: 4621,
            k: 384,
            n: 1928,
            lda: 384,
            ldb: 1928,
            ldc: 1928,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S3,
            schedule: Sm120Schedule::StreamK,
        },
        shape: Sm120Shape {
            m: 4621,
            k: 768,
            n: 384,
            lda: 768,
            ldb: 384,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S3,
            schedule: Sm120Schedule::StreamK,
        },
        shape: Sm120Shape {
            m: 4621,
            k: 1024,
            n: 384,
            lda: 1024,
            ldb: 384,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S3,
            schedule: Sm120Schedule::StreamK,
        },
        shape: Sm120Shape {
            m: 10400,
            k: 384,
            n: 384,
            lda: 384,
            ldb: 384,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S3,
            schedule: Sm120Schedule::StreamK,
        },
        shape: Sm120Shape {
            m: 10400,
            k: 384,
            n: 1536,
            lda: 384,
            ldb: 1536,
            ldc: 1536,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S3,
            schedule: Sm120Schedule::StreamK,
        },
        shape: Sm120Shape {
            m: 10400,
            k: 768,
            n: 384,
            lda: 768,
            ldb: 384,
            ldc: 384,
        },
    },
];

/// CC 12.1 has no measured stream-K cell yet.
pub const SM120_STREAMK_CELLS_CC121: &[Sm120ForcedRoute] = &[];

/// Logical operands for an SM120 automatic route lookup.
///
/// The selected physical route comes solely from the per-minor qualified
/// table; callers never nominate a tile, BK, or pipeline stage.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::mamba_ssm::gpu) struct Sm120AutoRequest {
    pub op: Sm120Op,
    pub dtype: WeightDtype,
    pub shape: Sm120Shape,
    pub a_ptr: super::contract::CUptr,
    pub b_ptr: super::contract::CUptr,
    pub operands: Sm120LaunchOperands,
    /// Live multiprocessor count of the board: the neighbour rule refuses a
    /// tile whose grid cannot fill one wave of it.
    pub multiprocessors: u32,
    /// The live half-precision policy. Only its stream-K permission opens
    /// the measured stream-K cells; every other request sees the tiled table.
    pub half_policy: HalfTriadPolicy,
}

/// Resolves only a measured, minor-specific SM120 cell. Every unsupported
/// capability or request detail declines to the portable caller with `None`.
pub(super) fn resolve_sm120_auto(
    caps: DeviceCaps,
    module_target: Option<Sm120TargetCandidate>,
    request: Sm120AutoRequest,
) -> Option<Sm120ForcedRoute> {
    if !crate::mamba_ssm::gpu::device::is_sm120_family(caps.compute_capability) {
        return None;
    }
    // A minor without a measured table has no cell to find and declines the
    // way an uncovered shape does.
    let (tiled, stream_k) = match caps.compute_capability {
        (12, 0) => (SM120_AUTO_CELLS_CC120, SM120_STREAMK_CELLS_CC120),
        (12, 1) => (SM120_AUTO_CELLS_CC121, SM120_STREAMK_CELLS_CC121),
        _ => (&[][..], &[][..]),
    };
    // The stream-K cells open only under the half policy that permits their
    // fixed-order fold. A permitted request that no stream-K cell serves, or
    // whose own grid already fills the device, falls through to the tiled
    // table exactly as an unpermitted one does.
    if request.half_policy == HalfTriadPolicy::AllowStreamKFixedOrder
        && let Some(route) = resolve_sm120_auto_from_cells(stream_k, caps, module_target, request)
    {
        return Some(route);
    }
    resolve_sm120_auto_from_cells(tiled, caps, module_target, request)
}

/// How far, per axis and in natural-log units, a shape may sit from the
/// nearest measured cell and still take that cell's tile: a factor of eight
/// in output rows, output columns and reduction. Inside that band the
/// leave-one-out check over the sixty measured cells loses 8 percent on
/// average to the best tile (internal/perf/sm120-half-census-20260903b);
/// outside it the portable tensor-core tiles serve, as they did before.
const SM120_AUTO_NEIGHBOUR_LOG_BOUND: f64 = 2.079_441_541_679_836;

/// Output rows, output columns and reduction length of an SM120 request,
/// the three axes a tile choice depends on.
fn sm120_auto_geometry(op: Sm120Op, shape: Sm120Shape) -> [f64; 3] {
    let (rows, columns, reduction) = match op {
        Sm120Op::Nn => (shape.m, shape.n, shape.k),
        Sm120Op::Tn => (shape.k, shape.n, shape.m),
        Sm120Op::Nt => (shape.m, shape.k, shape.n),
    };
    [rows as f64, columns as f64, reduction as f64]
}

/// The tile of the nearest measured cell of the same operation and dtype,
/// when one lies within the neighbour band. Ties keep table order, so the
/// choice is deterministic for a given table.
fn nearest_sm120_cell(
    cells: &[Sm120ForcedRoute],
    op: Sm120Op,
    dtype: WeightDtype,
    shape: Sm120Shape,
    multiprocessors: u32,
) -> Option<Sm120PhysicalRoute> {
    let target = sm120_auto_geometry(op, shape).map(f64::ln);
    let mut best: Option<(f64, &Sm120ForcedRoute)> = None;
    for cell in cells {
        if cell.op != op || cell.dtype != dtype {
            continue;
        }
        let axes = sm120_auto_geometry(op, cell.shape)
            .map(f64::ln)
            .iter()
            .zip(target.iter())
            .map(|(cell_axis, target_axis)| (cell_axis - target_axis).abs())
            .collect::<Vec<_>>();
        if axes
            .iter()
            .any(|axis| *axis > SM120_AUTO_NEIGHBOUR_LOG_BOUND)
        {
            continue;
        }
        let distance = axes.iter().map(|axis| axis * axis).sum::<f64>().sqrt();
        if best.is_none_or(|(best_distance, _)| distance < best_distance) {
            best = Some((distance, cell));
        }
    }
    let (_, neighbour) = best?;
    let mut physical = neighbour.physical;
    // A neighbour measured on a larger output may carry a tile whose grid
    // leaves most of this board idle. When the shape fills less than one
    // wave with that tile and less than half of what the neighbour filled,
    // the smallest tile serves instead: the census shows it winning every
    // cell that far into underfill. The pipeline depth and reduction step
    // stay the neighbour's. A neighbour that won underfilled itself keeps
    // its tile for shapes that are underfilled the same way.
    let grid = |cell_shape: Sm120Shape| {
        let [rows, columns, _] = sm120_auto_geometry(op, cell_shape);
        (rows / f64::from(physical.tile.output_rows())).ceil()
            * (columns / f64::from(physical.tile.output_columns())).ceil()
    };
    let target_grid = grid(shape);
    if target_grid < f64::from(multiprocessors) && target_grid * 2.0 < grid(neighbour.shape) {
        physical.tile = Sm120Tile::M64N64;
    }
    // A stream-K neighbour earns its persistent grid only while the target's
    // own grid underfills the device; a shape whose tiles already cover the
    // multiprocessors declines here so the tiled table serves it (measured on
    // the 5090: stream-K loses 22-77% wherever the tiles reach the SM count).
    if physical.schedule == Sm120Schedule::StreamK && target_grid >= f64::from(multiprocessors) {
        return None;
    }
    Some(physical)
}

fn resolve_sm120_auto_from_cells(
    cells: &[Sm120ForcedRoute],
    caps: DeviceCaps,
    module_target: Option<Sm120TargetCandidate>,
    request: Sm120AutoRequest,
) -> Option<Sm120ForcedRoute> {
    let measured = cells.iter().copied().find(|cell| {
        cell.op == request.op && cell.dtype == request.dtype && cell.shape == request.shape
    });
    // A shape no cell names takes the tile of the nearest measured cell, so
    // every shape in the measured band has a path; the exact cell still wins
    // where one exists.
    let route = match measured {
        Some(cell) => cell,
        None => Sm120ForcedRoute {
            op: request.op,
            dtype: request.dtype,
            physical: nearest_sm120_cell(
                cells,
                request.op,
                request.dtype,
                request.shape,
                request.multiprocessors,
            )?,
            shape: request.shape,
        },
    };
    let maps = Sm120MapRequest {
        op: request.op,
        dtype: request.dtype,
        tile: route.physical.tile,
        bk: route.physical.bk,
        a_ptr: request.a_ptr,
        b_ptr: request.b_ptr,
        shape: request.shape,
    };
    validate_sm120_map_request(maps).ok()?;
    if !sm120_auto_operands_supported(request.op, request.operands) {
        return None;
    }
    match sm120_forced_decline(caps, module_target, route) {
        Ok(None) => Some(route),
        Ok(Some(reason)) => {
            static DECLINED: std::sync::Once = std::sync::Once::new();
            crate::mamba_ssm::gpu::diagnostics::warn_once(&DECLINED, || {
                format!(
                    "SM120 half cell {route:?} is measured for this shape but this board \
                     declines it ({reason}); the portable tensor-core tiles serve instead"
                )
            });
            None
        }
        Err(error) => {
            static INVALID: std::sync::Once = std::sync::Once::new();
            crate::mamba_ssm::gpu::diagnostics::warn_once(&INVALID, || {
                format!("SM120 half cell {route:?} is not a launchable route: {error}")
            });
            None
        }
    }
}

fn sm120_auto_operands_supported(op: Sm120Op, operands: Sm120LaunchOperands) -> bool {
    let output_alignment = if op == Sm120Op::Tn { 4 } else { 2 };
    operands.output_ptr != 0
        && operands.output_ptr.is_multiple_of(output_alignment)
        && (operands.bias_ptr == 0 || operands.bias_ptr.is_multiple_of(4))
        && match op {
            Sm120Op::Nn => operands.bias_ptr == 0 || operands.alpha == 1.0,
            Sm120Op::Tn => operands.bias_ptr == 0 && operands.beta == 1.0,
            Sm120Op::Nt => operands.bias_ptr == 0 && operands.beta == 0.0,
        }
}

const SM100_CC100_TARGETS: [Sm100TargetCandidate; 2] = [
    Sm100TargetCandidate {
        device_cc: (10, 0),
        nvrtc_arch: "compute_100f",
        ptx_target: "sm_100f",
        kind: Sm100TargetKind::Family,
    },
    Sm100TargetCandidate {
        device_cc: (10, 0),
        nvrtc_arch: "compute_100a",
        ptx_target: "sm_100a",
        kind: Sm100TargetKind::Exact,
    },
];

const SM100_CC103_TARGETS: [Sm100TargetCandidate; 2] = [
    Sm100TargetCandidate {
        device_cc: (10, 3),
        nvrtc_arch: "compute_103f",
        ptx_target: "sm_103f",
        kind: Sm100TargetKind::Family,
    },
    Sm100TargetCandidate {
        device_cc: (10, 3),
        nvrtc_arch: "compute_103a",
        ptx_target: "sm_103a",
        kind: Sm100TargetKind::Exact,
    },
];

const SM100_CC107_TARGETS: [Sm100TargetCandidate; 2] = [
    Sm100TargetCandidate {
        device_cc: (10, 7),
        nvrtc_arch: "compute_107f",
        ptx_target: "sm_107f",
        kind: Sm100TargetKind::Family,
    },
    Sm100TargetCandidate {
        device_cc: (10, 7),
        nvrtc_arch: "compute_107a",
        ptx_target: "sm_107a",
        kind: Sm100TargetKind::Exact,
    },
];

const SM100_CC110_TARGETS: [Sm100TargetCandidate; 2] = [
    Sm100TargetCandidate {
        device_cc: (11, 0),
        nvrtc_arch: "compute_110f",
        ptx_target: "sm_110f",
        kind: Sm100TargetKind::Family,
    },
    Sm100TargetCandidate {
        device_cc: (11, 0),
        nvrtc_arch: "compute_110a",
        ptx_target: "sm_110a",
        kind: Sm100TargetKind::Exact,
    },
];

pub fn sm100_target_candidates(cc: (i32, i32)) -> &'static [Sm100TargetCandidate] {
    match cc {
        (10, 0) => &SM100_CC100_TARGETS,
        (10, 3) => &SM100_CC103_TARGETS,
        (10, 7) => &SM100_CC107_TARGETS,
        (11, 0) => &SM100_CC110_TARGETS,
        _ => &[],
    }
}

/// The SM100 candidates a toolkit can compile and that the contract has
/// verified. Family-specific targets (`compute_100f` and siblings) and the
/// CC 10.3 targets arrived with CUDA 12.9, the CC 11.0 targets with CUDA
/// 13.2, the CC 10.7 targets with CUDA 13.4 (PTX ISA 9.4), and CUDA 12.8
/// assembles the tcgen allocation with a different
/// instruction pairing than the one the contract freezes; below 12.9 the
/// family is not offered at all rather than run unverified.
pub fn sm100_target_candidates_for_nvrtc(
    cc: (i32, i32),
    nvrtc_version: (i32, i32),
) -> Vec<Sm100TargetCandidate> {
    sm100_target_candidates(cc)
        .iter()
        .copied()
        .filter(|candidate| {
            let floor = match candidate.device_cc {
                (10, 7) => (13, 4),
                (11, 0) => (13, 2),
                _ => (12, 9),
            };
            nvrtc_version >= floor
        })
        .collect()
}

const SM120_CC120_TARGETS: [Sm120TargetCandidate; 1] = [Sm120TargetCandidate {
    device_cc: (12, 0),
    nvrtc_arch: "compute_120",
    ptx_target: "sm_120",
}];

const SM120_CC121_FALLBACK_TARGETS: [Sm120TargetCandidate; 1] = [Sm120TargetCandidate {
    device_cc: (12, 1),
    nvrtc_arch: "compute_120",
    ptx_target: "sm_120",
}];

const SM120_CC121_TARGETS: [Sm120TargetCandidate; 2] = [
    Sm120TargetCandidate {
        device_cc: (12, 1),
        nvrtc_arch: "compute_121",
        ptx_target: "sm_121",
    },
    Sm120TargetCandidate {
        device_cc: (12, 1),
        nvrtc_arch: "compute_120",
        ptx_target: "sm_120",
    },
];

/// Returns compiler targets valid for the exact device and NVRTC version.
///
/// An empty slice means the specialized module must decline to the portable
/// Triad path. Target support does not by itself qualify an automatic cell.
pub fn sm120_target_candidates(
    cc: (i32, i32),
    nvrtc: (i32, i32),
) -> &'static [Sm120TargetCandidate] {
    match (cc, nvrtc) {
        ((12, 0), version) if version >= (12, 8) => &SM120_CC120_TARGETS,
        ((12, 1), version) if version >= (12, 9) => &SM120_CC121_TARGETS,
        ((12, 1), version) if version >= (12, 8) => &SM120_CC121_FALLBACK_TARGETS,
        _ => &[],
    }
}

/// Validates a caller-selected SM120 route against device, target, and resources.
///
/// This forced resolver exists for qualification and census tooling. Production
/// typed dispatch uses the private minor-specific automatic resolver instead.
pub fn resolve_sm120_forced(
    caps: DeviceCaps,
    module_target: Option<Sm120TargetCandidate>,
    route: Sm120ForcedRoute,
) -> Result<Option<Sm120ForcedRoute>, String> {
    Ok(sm120_forced_decline(caps, module_target, route)?
        .is_none()
        .then_some(route))
}

/// The first board-level reason `route` cannot launch here, or `None` when
/// it can. [`resolve_sm120_forced`] collapses this into its option; the
/// automatic path reports it, so a wrong board is told apart from a missing
/// cell.
pub(super) fn sm120_forced_decline(
    caps: DeviceCaps,
    module_target: Option<Sm120TargetCandidate>,
    route: Sm120ForcedRoute,
) -> Result<Option<&'static str>, String> {
    route.shape.validate(route.op)?;
    if !matches!(route.dtype, WeightDtype::Bf16 | WeightDtype::F16) {
        return Err("SM120 TMA supports bf16 and f16 operands only".into());
    }
    let spec = route.kernel_spec()?;
    let Some(module_target) = module_target else {
        return Ok(Some("no SM120 module is bound"));
    };
    let device_cc = (
        i32::try_from(caps.compute_capability.0)
            .map_err(|_| "SM120 device CC major exceeds i32::MAX".to_string())?,
        i32::try_from(caps.compute_capability.1)
            .map_err(|_| "SM120 device CC minor exceeds i32::MAX".to_string())?,
    );
    let accepted = caps
        .accepted_target
        .map(|target| target.as_str().to_owned());
    Ok(if module_target.device_cc != device_cc {
        Some("the bound SM120 module was compiled for another compute capability")
    } else if !sm120_target_candidates(device_cc, caps.nvrtc_version).contains(&module_target) {
        Some("this toolkit has no SM120 target for the device")
    } else if accepted.as_deref() != Some(module_target.nvrtc_arch) {
        Some("the driver accepted a different target than the bound module")
    } else if !caps.tensor_map_access {
        Some("the driver exposes no tensor-map access")
    } else if caps.optin_shared_bytes < spec.dynamic_shared_bytes {
        Some("the device opt-in shared memory is below the kernel's staging")
    } else {
        None
    })
}

pub fn resolve_sm100_forced(
    device_cc: (i32, i32),
    module_target: Option<Sm100TargetCandidate>,
    route: Sm100ForcedRoute,
) -> Result<Option<Sm100ForcedRoute>, String> {
    route.shape.validate(route.op)?;
    if !matches!(route.dtype, WeightDtype::Bf16 | WeightDtype::F16) {
        return Err("SM100 TCGEN supports bf16 and f16 operands only".into());
    }
    route.kernel_spec()?;
    let Some(module_target) = module_target else {
        return Ok(None);
    };
    if module_target.device_cc != device_cc
        || !sm100_target_candidates(device_cc).contains(&module_target)
    {
        return Ok(None);
    }
    Ok(Some(route))
}

pub fn resolve_sm90a_forced(
    device_cc: (i32, i32),
    module_available: bool,
    op: Sm90aOp,
    dtype: WeightDtype,
    schedule: Sm90aWarpgroupSchedule,
    shape: Sm90aShape,
) -> Result<Option<Sm90aForcedRoute>, String> {
    shape.validate(op)?;
    if !matches!(dtype, WeightDtype::Bf16 | WeightDtype::F16) {
        return Err("SM90a WGMMA supports bf16 and f16 operands only".into());
    }
    if device_cc != (9, 0) || !module_available {
        return Ok(None);
    }
    Ok(Some(Sm90aForcedRoute {
        op,
        dtype,
        schedule,
        shape,
    }))
}

// ── Split-M TN partition heuristic ──

/// Scratch cap for split-M partials, in f32 elements. Must not exceed the
/// `splitk_scratch` allocation in kernels.rs.
pub(super) const SPLITM_TN_SCRATCH_CAP: usize = 1 << 23;
/// m_chunk alignment (BK of the TN tile).
pub(super) const SPLITM_TN_BK_ALIGN: u32 = 16;

/// Decide the split-M factor for the TN (dW) kernel on underfilled grids.
/// Returns `(m_chunk, f_final)` or `None` when the plain kernel is fine.
#[inline]
pub(super) fn splitm_tn_partition(
    batch: usize,
    n_in: usize,
    n_out: usize,
    multiprocessor_count: u32,
) -> Result<Option<(usize, usize)>, String> {
    // No n_in floor: the partial kernel predicates K_out < 128 exactly
    // like the plain kernel, and a small-K dW against a large batch
    // reduction underfills the grid without the split (K_out=24, N=768
    // ran six CTAs). The split changes the dW summation order versus
    // the plain kernel; run-to-run and per-shape determinism hold — the
    // partition is a pure function of the shape and immutable device size.
    if !(n_out >= 128 && batch >= 256) {
        return Ok(None);
    }
    let policy = scalar_wave_policy(multiprocessor_count)?;
    let target_blocks = wave_target_blocks(
        multiprocessor_count,
        policy.tn_split_m_wave_numerator,
        policy.tn_split_m_wave_denominator,
    )?;
    let batch_u32 = u32::try_from(batch).map_err(|_| "TN split-M batch exceeds u32::MAX")?;
    let k_tiles = u32::try_from(n_in)
        .map_err(|_| "TN split-M input width exceeds u32::MAX")?
        .div_ceil(128);
    let n_tiles = u32::try_from(n_out)
        .map_err(|_| "TN split-M output width exceeds u32::MAX")?
        .div_ceil(128);
    let base_blocks = k_tiles
        .checked_mul(n_tiles)
        .ok_or_else(|| "TN split-M base grid overflows u32".to_string())?;
    if base_blocks == 0 || base_blocks >= target_blocks {
        return Ok(None);
    }
    let f_grid = target_blocks.div_ceil(base_blocks);
    let output_elements = n_in
        .checked_mul(n_out)
        .ok_or_else(|| "TN split-M output size overflows usize".to_string())?;
    let f_scratch_cap = u32::try_from(SPLITM_TN_SCRATCH_CAP / output_elements)
        .map_err(|_| "TN split-M scratch factor exceeds u32::MAX")?;
    let f = f_grid.min(f_scratch_cap).max(1);
    let m_chunk_raw = batch_u32.div_ceil(f);
    let m_chunk = m_chunk_raw
        .checked_add(SPLITM_TN_BK_ALIGN - 1)
        .ok_or_else(|| "TN split-M aligned chunk overflows u32".to_string())?
        & !(SPLITM_TN_BK_ALIGN - 1);
    let f_final = batch_u32.div_ceil(m_chunk);
    let scratch_elements = usize::try_from(f_final)
        .map_err(|_| "TN split-M partition count exceeds usize")?
        .checked_mul(output_elements)
        .ok_or_else(|| "TN split-M scratch size overflows usize".to_string())?;
    if f_final < 2 || scratch_elements > SPLITM_TN_SCRATCH_CAP {
        return Ok(None);
    }
    Ok(Some((
        usize::try_from(m_chunk).map_err(|_| "TN split-M chunk exceeds usize")?,
        usize::try_from(f_final).map_err(|_| "TN split-M partition count exceeds usize")?,
    )))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TnNarrowSplitMCell {
    shape: F32TriadShape,
    m_chunk: usize,
    chunks: usize,
}

const TN_NARROW_SPLITM_SM120_CC120_170_CELLS: [TnNarrowSplitMCell; 3] = [
    TnNarrowSplitMCell {
        shape: F32TriadShape {
            m: 1024,
            k: 47,
            n: 17,
            lda: 47,
            ldb: 17,
            ldc: 17,
        },
        m_chunk: 32,
        chunks: 32,
    },
    TnNarrowSplitMCell {
        shape: F32TriadShape {
            m: 1024,
            k: 128,
            n: 25,
            lda: 128,
            ldb: 25,
            ldc: 25,
        },
        m_chunk: 32,
        chunks: 32,
    },
    TnNarrowSplitMCell {
        shape: F32TriadShape {
            m: 4096,
            k: 64,
            n: 64,
            lda: 64,
            ldb: 64,
            ldc: 64,
        },
        m_chunk: 48,
        chunks: 86,
    },
];

fn qualified_tn_narrow_splitm_cell(request: F32TriadRequest) -> Option<TnNarrowSplitMCell> {
    (request.op == crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp::Tn).then_some(())?;
    TN_NARROW_SPLITM_SM120_CC120_170_CELLS
        .iter()
        .copied()
        .find(|cell| cell.shape == request.shape)
}

/// The facts a tuned scalar cell is admitted on. The f32 policy is not one
/// of them: a tuned exact-FMA route is the floor the TF32 policy falls back
/// to, so it serves under either policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ScalarLaunchFacts {
    pub scalar_artifact: ArtifactIdentity,
    pub scalar_compiler: CompilerIdentity,
    pub fixed_artifact: ArtifactIdentity,
    pub fixed_compiler: CompilerIdentity,
    pub fixed_copyplan_loaded: bool,
    pub sm89_exact_f32_artifact: Option<ArtifactIdentity>,
    pub sm89_exact_f32_compiler: Option<CompilerIdentity>,
    pub sm89_exact_f32_symbols_loaded: [bool; 3],
    pub sm89_exact_f32_d128_artifact: Option<ArtifactIdentity>,
    pub sm89_exact_f32_d128_compiler: Option<CompilerIdentity>,
    pub sm89_exact_f32_d128_symbols_loaded: [bool; 2],
    pub compute_capability: (u32, u32),
    pub multiprocessor_count: u32,
}

/// How a scalar route measured on the Ada board is admitted on this one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ScalarAdmission {
    /// A frozen evidence cohort matches the board and the compiled module.
    Evidence,
    /// The qualification tooling's own candidate cohorts, for forced routes.
    Candidate,
    /// The board runs the instruction tier the module was compiled for and
    /// proves the route's output words against the reference at first use.
    Proof,
}

/// The boards whose portable-tier compile of an Ada Triad module admits its
/// routes to the first-use proof: the whole sm_80 tier.
fn scalar_proof_board(facts: ScalarLaunchFacts) -> bool {
    facts.compute_capability.0 >= 8
}

/// The boards whose Fixed module composes the Ada overlay and so may prove
/// its copy-plan routes: the sm_80 tier without the CC 12 family, which
/// keeps its Fixed module byte-identical to the one its own cohorts pin.
fn fixed_overlay_proof_board(facts: ScalarLaunchFacts) -> bool {
    facts.compute_capability.0 >= 8
        && !crate::mamba_ssm::gpu::device::is_sm120_family(facts.compute_capability)
        && facts.fixed_copyplan_loaded
}

fn scalar_admitted(
    admission: ScalarAdmission,
    facts: ScalarLaunchFacts,
    evidence: fn(ScalarLaunchFacts) -> bool,
) -> bool {
    match admission {
        ScalarAdmission::Evidence | ScalarAdmission::Candidate => evidence(facts),
        ScalarAdmission::Proof => scalar_proof_board(facts),
    }
}

fn fixed_overlay_admitted(
    admission: ScalarAdmission,
    facts: ScalarLaunchFacts,
    evidence: fn(ScalarLaunchFacts) -> bool,
) -> bool {
    match admission {
        ScalarAdmission::Evidence | ScalarAdmission::Candidate => evidence(facts),
        ScalarAdmission::Proof => fixed_overlay_proof_board(facts),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Sm89ExactF32D128QualificationIdentity {
    nvrtc_version: (i32, i32),
    compile_key: [u8; 32],
    artifact_digest: [u8; 32],
    source_digest: [u8; 32],
    header_manifest_digest: [u8; 32],
    nvrtc_library_domain: [u8; 32],
}

impl Sm89ExactF32D128QualificationIdentity {
    fn matches(self, facts: ScalarLaunchFacts) -> bool {
        let Some(artifact) = facts.sm89_exact_f32_d128_artifact else {
            return false;
        };
        let Some(compiler) = facts.sm89_exact_f32_d128_compiler else {
            return false;
        };
        artifact.module_kind == ModuleKind::TriadSm89ExactF32D128
            && artifact.artifact_kind == ArtifactKind::Ptx
            && artifact.compile_key == self.compile_key
            && artifact.artifact_digest == self.artifact_digest
            && compiler.source_digest == self.source_digest
            && compiler.invocation_digest == self.compile_key
            && compiler.header_manifest_digest == self.header_manifest_digest
            && compiler.target.as_str() == "sm_89"
            && compiler.nvrtc_version == self.nvrtc_version
            && compiler.nvrtc_library_domain == self.nvrtc_library_domain
            && compiler.nvrtc_library_known
            && compiler.output_kind == ArtifactKind::Ptx
            && compiler.composer_revision == COMPOSER_REVISION
            && compiler.compiler_revision == COMPILER_REVISION
            && compiler.numeric_abi_revision == NUMERIC_ABI_REVISION
            && compiler.schedule_revision == SCHEDULE_REVISION
    }
}

const SM89_EXACT_F32_D128_SOURCE_DIGEST: [u8; 32] = [
    0x61, 0x7f, 0x74, 0x63, 0x10, 0xee, 0x68, 0xee, 0xba, 0xe1, 0xfa, 0x52, 0x6f, 0x6d, 0x97, 0x44,
    0x69, 0x26, 0xe8, 0xb9, 0x0a, 0xdf, 0x49, 0x10, 0x4b, 0x5a, 0x33, 0x9c, 0x3c, 0xc1, 0x32, 0x6f,
];

const SM89_EXACT_F32_D128_HEADER_MANIFEST_DIGEST: [u8; 32] = [
    0x01, 0x89, 0x50, 0x95, 0x30, 0x55, 0x5f, 0x96, 0x15, 0x20, 0x71, 0x2f, 0x39, 0xe1, 0x02, 0x57,
    0x76, 0xf4, 0x30, 0x19, 0x3b, 0x33, 0x53, 0x3d, 0x8b, 0xee, 0xbf, 0xcc, 0x38, 0xab, 0x8c, 0xfe,
];

// Exact live module/ABI/resource identities from CUDA 12.8/13.0/13.2.
// AUTO evidence remains empty until exact-bit and performance qualification passes.
const SM89_EXACT_F32_D128_QUALIFICATION_CANDIDATES: &[Sm89ExactF32D128QualificationIdentity] = &[
    Sm89ExactF32D128QualificationIdentity {
        nvrtc_version: (12, 8),
        compile_key: [
            0xce, 0xcd, 0xd0, 0x14, 0x4b, 0x62, 0xfe, 0x37, 0xb4, 0x5b, 0x24, 0xa9, 0x60, 0x88,
            0xc0, 0x4c, 0x33, 0x70, 0x9b, 0x23, 0x5f, 0xce, 0x9f, 0xb1, 0x30, 0x4e, 0xc3, 0xd5,
            0x45, 0x3c, 0x0c, 0xeb,
        ],
        artifact_digest: [
            0xd5, 0xf9, 0x6d, 0x75, 0x3f, 0x54, 0x28, 0x51, 0x15, 0x19, 0x86, 0x0e, 0x44, 0xc1,
            0xce, 0x58, 0x85, 0xed, 0xe1, 0x7d, 0x8a, 0x1a, 0x27, 0xbd, 0xe9, 0xaf, 0x3b, 0x99,
            0xa9, 0xe9, 0xa8, 0x93,
        ],
        source_digest: SM89_EXACT_F32_D128_SOURCE_DIGEST,
        header_manifest_digest: SM89_EXACT_F32_D128_HEADER_MANIFEST_DIGEST,
        nvrtc_library_domain: [
            0x26, 0xb0, 0xa3, 0xa0, 0x20, 0x44, 0xff, 0xcb, 0xc1, 0x69, 0x3f, 0xd8, 0x3e, 0x92,
            0x61, 0xbe, 0xff, 0xa6, 0x92, 0xa4, 0xfb, 0xcf, 0xe3, 0xac, 0x5e, 0x9d, 0x8c, 0x87,
            0x98, 0x0b, 0xb1, 0x55,
        ],
    },
    Sm89ExactF32D128QualificationIdentity {
        nvrtc_version: (13, 0),
        compile_key: [
            0xe1, 0x30, 0xdd, 0xd9, 0x03, 0x6b, 0xba, 0x9f, 0xf2, 0xd0, 0xbb, 0xcb, 0x83, 0x57,
            0xab, 0x1f, 0x6b, 0x7e, 0x60, 0x4c, 0x12, 0x80, 0x80, 0xe8, 0x2d, 0xb4, 0x40, 0xf7,
            0xba, 0xd9, 0x3a, 0x0a,
        ],
        artifact_digest: [
            0x43, 0x4b, 0xa1, 0xad, 0x3b, 0x2a, 0xef, 0xaa, 0xc6, 0x25, 0x56, 0x4d, 0xda, 0x57,
            0x19, 0x27, 0xb1, 0x13, 0x8b, 0x4c, 0xa7, 0x64, 0x0a, 0x88, 0xb7, 0x9d, 0xbb, 0x16,
            0x49, 0x80, 0x76, 0x4b,
        ],
        source_digest: SM89_EXACT_F32_D128_SOURCE_DIGEST,
        header_manifest_digest: SM89_EXACT_F32_D128_HEADER_MANIFEST_DIGEST,
        nvrtc_library_domain: [
            0x70, 0x9b, 0x91, 0xc3, 0x6b, 0xfb, 0x0e, 0xd9, 0x66, 0xee, 0x69, 0xad, 0xc8, 0xd6,
            0xf8, 0x7f, 0xf1, 0x10, 0xee, 0xcf, 0x3d, 0xfb, 0x50, 0x60, 0x36, 0x7f, 0x18, 0x3c,
            0xe6, 0x14, 0xeb, 0x0d,
        ],
    },
    Sm89ExactF32D128QualificationIdentity {
        nvrtc_version: (13, 2),
        compile_key: [
            0x64, 0x58, 0x91, 0xaf, 0xed, 0x71, 0x2b, 0x73, 0x7a, 0x87, 0xcb, 0xd9, 0x76, 0x2f,
            0xad, 0x59, 0x83, 0xb0, 0xa6, 0x7e, 0x9b, 0xfc, 0x66, 0x2c, 0xc4, 0x0d, 0x81, 0x59,
            0xad, 0x35, 0x9a, 0x3f,
        ],
        artifact_digest: [
            0xd7, 0xf2, 0xce, 0x5b, 0x16, 0xc9, 0x6e, 0x5a, 0x60, 0xa6, 0x44, 0x42, 0x4e, 0xba,
            0xe6, 0x2a, 0x52, 0x66, 0xc9, 0x8d, 0x6d, 0xd8, 0xed, 0xde, 0x0c, 0xf7, 0x6b, 0xf3,
            0x8a, 0x57, 0x20, 0x5e,
        ],
        source_digest: SM89_EXACT_F32_D128_SOURCE_DIGEST,
        header_manifest_digest: SM89_EXACT_F32_D128_HEADER_MANIFEST_DIGEST,
        nvrtc_library_domain: [
            0xd0, 0x31, 0xa5, 0x3e, 0xb9, 0x72, 0x35, 0xb7, 0x0f, 0x62, 0xf6, 0x52, 0x93, 0x2d,
            0xb1, 0xbd, 0xf7, 0x28, 0xea, 0x22, 0x9c, 0x8c, 0xa8, 0x09, 0xd5, 0x3c, 0x5f, 0xfd,
            0x91, 0x64, 0x26, 0x87,
        ],
    },
];
// Admission provenance: all three candidate identities passed the complete
// module, exact-bit, guarded eager/graph, and once3/once7 actual-AUTO screen
// recorded under `internal/perf/ada-d128-assembly-20260910/`.
const SM89_EXACT_F32_D128_EVIDENCE_COHORTS: &[Sm89ExactF32D128QualificationIdentity] = &[
    SM89_EXACT_F32_D128_QUALIFICATION_CANDIDATES[0],
    SM89_EXACT_F32_D128_QUALIFICATION_CANDIDATES[1],
    SM89_EXACT_F32_D128_QUALIFICATION_CANDIDATES[2],
];

fn qualified_sm89_exact_f32_d128_environment(facts: ScalarLaunchFacts) -> bool {
    facts.compute_capability == (8, 9)
        && facts.multiprocessor_count == 142
        && SM89_EXACT_F32_D128_EVIDENCE_COHORTS
            .iter()
            .copied()
            .any(|identity| identity.matches(facts))
}

fn qualified_sm89_exact_f32_d128_candidate_environment(facts: ScalarLaunchFacts) -> bool {
    facts.compute_capability == (8, 9)
        && facts.multiprocessor_count == 142
        && SM89_EXACT_F32_D128_QUALIFICATION_CANDIDATES
            .iter()
            .copied()
            .any(|identity| identity.matches(facts))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Sm89ExactF32QualificationIdentity {
    nvrtc_version: (i32, i32),
    compile_key: [u8; 32],
    artifact_digest: [u8; 32],
    source_digest: [u8; 32],
    header_manifest_digest: [u8; 32],
    nvrtc_library_domain: [u8; 32],
}

impl Sm89ExactF32QualificationIdentity {
    fn matches(self, facts: ScalarLaunchFacts) -> bool {
        let Some(artifact) = facts.sm89_exact_f32_artifact else {
            return false;
        };
        let Some(compiler) = facts.sm89_exact_f32_compiler else {
            return false;
        };
        artifact.module_kind == ModuleKind::TriadSm89ExactF32
            && artifact.artifact_kind == ArtifactKind::Ptx
            && artifact.compile_key == self.compile_key
            && artifact.artifact_digest == self.artifact_digest
            && compiler.source_digest == self.source_digest
            && compiler.invocation_digest == self.compile_key
            && compiler.header_manifest_digest == self.header_manifest_digest
            && compiler.target.as_str() == "sm_89"
            && compiler.nvrtc_version == self.nvrtc_version
            && compiler.nvrtc_library_domain == self.nvrtc_library_domain
            && compiler.nvrtc_library_known
            && compiler.output_kind == ArtifactKind::Ptx
            && compiler.composer_revision == COMPOSER_REVISION
            && compiler.compiler_revision == COMPILER_REVISION
            && compiler.numeric_abi_revision == NUMERIC_ABI_REVISION
            && compiler.schedule_revision == SCHEDULE_REVISION
    }
}

// Populated only after the three CUDA 12.8/13.0/13.2 live qualification
// records are frozen. An empty cohort deliberately keeps public AUTO on its
// retained portable Split-M routes while forced qualification remains usable.
const SM89_EXACT_F32_SOURCE_DIGEST: [u8; 32] = [
    167, 87, 116, 48, 219, 158, 201, 49, 230, 222, 239, 38, 231, 204, 212, 118, 36, 74, 143, 52,
    227, 224, 0, 233, 195, 114, 128, 121, 124, 104, 219, 160,
];

const SM89_EXACT_F32_QUALIFICATION_CANDIDATES: &[Sm89ExactF32QualificationIdentity] = &[
    Sm89ExactF32QualificationIdentity {
        nvrtc_version: (12, 8),
        compile_key: [
            66, 121, 129, 237, 122, 5, 88, 82, 236, 7, 126, 122, 204, 5, 28, 245, 51, 163, 1, 244,
            197, 59, 224, 245, 31, 83, 101, 169, 218, 21, 222, 22,
        ],
        artifact_digest: [
            192, 15, 106, 13, 105, 184, 164, 11, 31, 141, 174, 79, 21, 42, 80, 52, 86, 155, 178,
            162, 8, 182, 10, 93, 0, 196, 222, 120, 250, 4, 71, 67,
        ],
        source_digest: SM89_EXACT_F32_SOURCE_DIGEST,
        header_manifest_digest: [
            17, 224, 104, 25, 33, 251, 173, 232, 90, 254, 188, 30, 128, 5, 220, 249, 183, 198, 166,
            77, 231, 111, 1, 159, 159, 185, 247, 48, 224, 199, 213, 168,
        ],
        nvrtc_library_domain: [
            38, 176, 163, 160, 32, 68, 255, 203, 193, 105, 63, 216, 62, 146, 97, 190, 255, 166,
            146, 164, 251, 207, 227, 172, 94, 157, 140, 135, 152, 11, 177, 85,
        ],
    },
    Sm89ExactF32QualificationIdentity {
        nvrtc_version: (13, 0),
        compile_key: [
            17, 240, 71, 134, 204, 61, 203, 250, 181, 106, 102, 167, 33, 191, 189, 121, 115, 54,
            140, 172, 44, 112, 143, 78, 93, 63, 120, 198, 158, 93, 217, 55,
        ],
        artifact_digest: [
            21, 143, 23, 229, 192, 9, 54, 115, 56, 50, 91, 4, 68, 248, 40, 151, 5, 148, 165, 142,
            49, 59, 139, 250, 89, 134, 115, 75, 9, 179, 160, 147,
        ],
        source_digest: SM89_EXACT_F32_SOURCE_DIGEST,
        header_manifest_digest: [
            255, 141, 156, 152, 51, 210, 171, 101, 199, 86, 7, 9, 70, 23, 25, 158, 210, 13, 70,
            188, 225, 14, 188, 9, 199, 160, 251, 252, 222, 202, 162, 109,
        ],
        nvrtc_library_domain: [
            112, 155, 145, 195, 107, 251, 14, 217, 102, 238, 105, 173, 200, 214, 248, 127, 241, 16,
            238, 207, 61, 251, 80, 96, 54, 127, 24, 60, 230, 20, 235, 13,
        ],
    },
    Sm89ExactF32QualificationIdentity {
        nvrtc_version: (13, 2),
        compile_key: [
            175, 136, 89, 91, 161, 213, 134, 177, 20, 218, 221, 181, 219, 213, 169, 101, 215, 74,
            33, 159, 206, 226, 69, 196, 173, 106, 89, 24, 184, 227, 21, 173,
        ],
        artifact_digest: [
            169, 42, 21, 147, 251, 133, 206, 249, 195, 108, 82, 6, 48, 205, 200, 98, 80, 76, 64,
            189, 98, 37, 69, 98, 89, 24, 12, 129, 180, 177, 111, 25,
        ],
        source_digest: SM89_EXACT_F32_SOURCE_DIGEST,
        header_manifest_digest: [
            150, 71, 34, 137, 0, 196, 78, 181, 39, 250, 42, 207, 200, 39, 49, 125, 79, 43, 199, 7,
            93, 216, 18, 120, 177, 226, 113, 202, 103, 60, 101, 55,
        ],
        nvrtc_library_domain: [
            208, 49, 165, 62, 185, 114, 53, 183, 15, 98, 246, 82, 147, 45, 177, 189, 247, 40, 234,
            34, 156, 140, 168, 9, 213, 60, 95, 253, 145, 100, 38, 135,
        ],
    },
];

// Admission provenance: the supported-toolkit evidence is frozen in
// `internal/perf/ada-large-tn-admission-20260910/report.md`; its reused prior
// CUDA 13.2 qualification is detailed in
// `internal/perf/ada-sm89-exact-f32-b2-20260909/report.md`.
const SM89_EXACT_F32_EVIDENCE_COHORTS: &[Sm89ExactF32QualificationIdentity] = &[
    SM89_EXACT_F32_QUALIFICATION_CANDIDATES[0],
    SM89_EXACT_F32_QUALIFICATION_CANDIDATES[1],
    SM89_EXACT_F32_QUALIFICATION_CANDIDATES[2],
];

fn qualified_sm89_exact_f32_environment(facts: ScalarLaunchFacts) -> bool {
    facts.compute_capability == (8, 9)
        && facts.multiprocessor_count == 142
        && SM89_EXACT_F32_EVIDENCE_COHORTS
            .iter()
            .copied()
            .any(|identity| identity.matches(facts))
}

fn qualified_sm89_exact_f32_candidate_environment(facts: ScalarLaunchFacts) -> bool {
    facts.compute_capability == (8, 9)
        && facts.multiprocessor_count == 142
        && SM89_EXACT_F32_QUALIFICATION_CANDIDATES
            .iter()
            .copied()
            .any(|identity| identity.matches(facts))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FixedCopyPlanQualificationIdentity {
    nvrtc_version: (i32, i32),
    compile_key: [u8; 32],
    artifact_digest: [u8; 32],
    source_digest: [u8; 32],
    header_manifest_digest: [u8; 32],
    nvrtc_library_domain: [u8; 32],
}

impl FixedCopyPlanQualificationIdentity {
    fn matches(self, facts: ScalarLaunchFacts) -> bool {
        facts.fixed_copyplan_loaded
            && facts.fixed_artifact.module_kind == ModuleKind::Fixed
            && facts.fixed_artifact.artifact_kind == ArtifactKind::Ptx
            && facts.fixed_artifact.compile_key == self.compile_key
            && facts.fixed_artifact.artifact_digest == self.artifact_digest
            && facts.fixed_compiler.source_digest == self.source_digest
            && facts.fixed_compiler.invocation_digest == self.compile_key
            && facts.fixed_compiler.header_manifest_digest == self.header_manifest_digest
            && facts.fixed_compiler.target.as_str() == "sm_89"
            && facts.fixed_compiler.nvrtc_version == self.nvrtc_version
            && facts.fixed_compiler.nvrtc_library_domain == self.nvrtc_library_domain
            && facts.fixed_compiler.nvrtc_library_known
            && facts.fixed_compiler.output_kind == ArtifactKind::Ptx
            && facts.fixed_compiler.composer_revision == COMPOSER_REVISION
            && facts.fixed_compiler.compiler_revision == COMPILER_REVISION
            && facts.fixed_compiler.numeric_abi_revision == NUMERIC_ABI_REVISION
            && facts.fixed_compiler.schedule_revision == SCHEDULE_REVISION
    }
}

const FIXED_COPYPLAN_SOURCE_DIGEST_CAP16: [u8; 32] = [
    228, 106, 241, 227, 21, 211, 35, 219, 106, 191, 90, 209, 117, 105, 54, 113, 80, 182, 50, 5,
    169, 247, 81, 115, 170, 248, 8, 29, 180, 94, 14, 144,
];

const FIXED_COPYPLAN_SOURCE_DIGEST_CAP64: [u8; 32] = [
    115, 17, 29, 85, 12, 211, 55, 120, 203, 161, 47, 39, 235, 56, 204, 139, 48, 228, 23, 180, 11,
    100, 105, 243, 243, 109, 220, 56, 21, 224, 220, 171,
];

/// Exact Fixed identities from the six initial complete-module compilations,
/// ordered by CUDA toolkit and then state capacity (16, 64). Fixed carries the
/// fold and Inference overlays, so its source, key and artifact remain
/// capacity-specific even when the toolkit is unchanged.
const FIXED_COPYPLAN_EVIDENCE_COHORTS: &[FixedCopyPlanQualificationIdentity] = &[
    FixedCopyPlanQualificationIdentity {
        nvrtc_version: (12, 8),
        compile_key: [
            40, 38, 84, 166, 63, 118, 16, 227, 231, 122, 248, 220, 26, 188, 177, 61, 237, 52, 57,
            90, 237, 253, 178, 38, 114, 107, 155, 157, 217, 66, 243, 246,
        ],
        artifact_digest: [
            48, 83, 107, 208, 145, 221, 237, 54, 223, 79, 190, 228, 100, 240, 151, 149, 253, 214,
            106, 122, 60, 135, 46, 108, 179, 206, 192, 180, 64, 5, 242, 211,
        ],
        source_digest: FIXED_COPYPLAN_SOURCE_DIGEST_CAP16,
        header_manifest_digest: [
            41, 90, 28, 248, 109, 177, 114, 166, 101, 88, 33, 169, 246, 40, 246, 75, 177, 57, 3,
            231, 218, 127, 19, 175, 21, 243, 133, 66, 41, 157, 25, 172,
        ],
        nvrtc_library_domain: [
            38, 176, 163, 160, 32, 68, 255, 203, 193, 105, 63, 216, 62, 146, 97, 190, 255, 166,
            146, 164, 251, 207, 227, 172, 94, 157, 140, 135, 152, 11, 177, 85,
        ],
    },
    FixedCopyPlanQualificationIdentity {
        nvrtc_version: (12, 8),
        compile_key: [
            232, 157, 8, 230, 245, 198, 87, 201, 15, 189, 133, 47, 195, 181, 46, 144, 211, 254,
            183, 9, 91, 230, 9, 58, 41, 152, 145, 93, 63, 57, 196, 219,
        ],
        artifact_digest: [
            199, 51, 148, 167, 38, 145, 169, 213, 65, 27, 87, 115, 57, 69, 247, 254, 13, 109, 196,
            198, 236, 53, 133, 214, 87, 161, 11, 100, 241, 142, 119, 132,
        ],
        source_digest: FIXED_COPYPLAN_SOURCE_DIGEST_CAP64,
        header_manifest_digest: [
            95, 72, 218, 83, 232, 0, 212, 34, 241, 28, 99, 193, 37, 110, 226, 125, 210, 6, 31, 25,
            126, 59, 205, 91, 114, 143, 133, 18, 228, 74, 197, 153,
        ],
        nvrtc_library_domain: [
            38, 176, 163, 160, 32, 68, 255, 203, 193, 105, 63, 216, 62, 146, 97, 190, 255, 166,
            146, 164, 251, 207, 227, 172, 94, 157, 140, 135, 152, 11, 177, 85,
        ],
    },
    FixedCopyPlanQualificationIdentity {
        nvrtc_version: (13, 0),
        compile_key: [
            150, 113, 145, 232, 99, 212, 108, 81, 23, 216, 108, 167, 29, 147, 58, 37, 156, 181, 43,
            102, 253, 226, 168, 137, 169, 39, 136, 158, 13, 2, 87, 153,
        ],
        artifact_digest: [
            29, 145, 211, 101, 64, 243, 47, 108, 137, 22, 198, 148, 235, 117, 224, 190, 140, 112,
            250, 179, 89, 41, 112, 176, 171, 69, 120, 30, 171, 37, 19, 160,
        ],
        source_digest: FIXED_COPYPLAN_SOURCE_DIGEST_CAP16,
        header_manifest_digest: [
            65, 180, 50, 8, 184, 220, 49, 112, 66, 46, 53, 126, 16, 203, 46, 224, 202, 8, 75, 10,
            147, 239, 224, 31, 241, 212, 227, 195, 159, 211, 60, 194,
        ],
        nvrtc_library_domain: [
            112, 155, 145, 195, 107, 251, 14, 217, 102, 238, 105, 173, 200, 214, 248, 127, 241, 16,
            238, 207, 61, 251, 80, 96, 54, 127, 24, 60, 230, 20, 235, 13,
        ],
    },
    FixedCopyPlanQualificationIdentity {
        nvrtc_version: (13, 0),
        compile_key: [
            62, 106, 196, 94, 150, 126, 236, 129, 124, 167, 163, 234, 112, 121, 152, 206, 148, 171,
            60, 231, 206, 79, 211, 92, 158, 153, 103, 10, 209, 12, 50, 217,
        ],
        artifact_digest: [
            158, 250, 100, 169, 63, 150, 216, 122, 162, 206, 190, 241, 203, 158, 124, 66, 11, 185,
            233, 12, 5, 198, 216, 146, 192, 166, 218, 120, 200, 171, 152, 196,
        ],
        source_digest: FIXED_COPYPLAN_SOURCE_DIGEST_CAP64,
        header_manifest_digest: [
            183, 131, 32, 166, 164, 22, 77, 115, 215, 224, 240, 101, 40, 121, 115, 97, 216, 132,
            213, 100, 168, 50, 100, 145, 188, 188, 120, 143, 170, 210, 18, 158,
        ],
        nvrtc_library_domain: [
            112, 155, 145, 195, 107, 251, 14, 217, 102, 238, 105, 173, 200, 214, 248, 127, 241, 16,
            238, 207, 61, 251, 80, 96, 54, 127, 24, 60, 230, 20, 235, 13,
        ],
    },
    FixedCopyPlanQualificationIdentity {
        nvrtc_version: (13, 2),
        compile_key: [
            64, 234, 181, 34, 254, 181, 41, 162, 25, 207, 72, 187, 38, 9, 49, 174, 162, 202, 121,
            227, 239, 11, 140, 167, 222, 173, 137, 58, 225, 223, 63, 24,
        ],
        artifact_digest: [
            131, 208, 156, 113, 96, 92, 98, 221, 166, 89, 8, 187, 182, 255, 28, 237, 233, 149, 91,
            19, 107, 164, 240, 185, 24, 217, 4, 77, 77, 30, 69, 228,
        ],
        source_digest: FIXED_COPYPLAN_SOURCE_DIGEST_CAP16,
        header_manifest_digest: [
            156, 37, 38, 105, 194, 169, 155, 210, 128, 109, 239, 74, 65, 38, 170, 164, 62, 150, 7,
            243, 164, 210, 107, 112, 158, 60, 159, 9, 61, 122, 244, 54,
        ],
        nvrtc_library_domain: [
            208, 49, 165, 62, 185, 114, 53, 183, 15, 98, 246, 82, 147, 45, 177, 189, 247, 40, 234,
            34, 156, 140, 168, 9, 213, 60, 95, 253, 145, 100, 38, 135,
        ],
    },
    FixedCopyPlanQualificationIdentity {
        nvrtc_version: (13, 2),
        compile_key: [
            244, 82, 102, 107, 95, 169, 215, 120, 179, 137, 166, 13, 212, 20, 14, 130, 251, 162,
            183, 173, 155, 17, 83, 117, 236, 143, 120, 213, 30, 86, 201, 218,
        ],
        artifact_digest: [
            57, 155, 175, 93, 19, 222, 42, 191, 85, 198, 183, 206, 85, 176, 97, 9, 104, 146, 163,
            110, 155, 78, 19, 186, 32, 82, 251, 4, 105, 154, 91, 115,
        ],
        source_digest: FIXED_COPYPLAN_SOURCE_DIGEST_CAP64,
        header_manifest_digest: [
            245, 204, 65, 250, 5, 45, 192, 124, 0, 78, 252, 37, 44, 87, 101, 38, 109, 151, 163, 65,
            14, 251, 78, 249, 166, 229, 159, 54, 190, 91, 84, 79,
        ],
        nvrtc_library_domain: [
            208, 49, 165, 62, 185, 114, 53, 183, 15, 98, 246, 82, 147, 45, 177, 189, 247, 40, 234,
            34, 156, 140, 168, 9, 213, 60, 95, 253, 145, 100, 38, 135,
        ],
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ScalarTransposeQualificationIdentity {
    nvrtc_version: (i32, i32),
    compile_key: [u8; 32],
    artifact_digest: [u8; 32],
    source_digest: [u8; 32],
    header_manifest_digest: [u8; 32],
    nvrtc_library_domain: [u8; 32],
}

impl ScalarTransposeQualificationIdentity {
    fn matches(self, facts: ScalarLaunchFacts) -> bool {
        facts.scalar_artifact.module_kind == ModuleKind::TriadScalar
            && facts.scalar_artifact.artifact_kind == ArtifactKind::Ptx
            && facts.scalar_artifact.compile_key == self.compile_key
            && facts.scalar_artifact.artifact_digest == self.artifact_digest
            && facts.scalar_compiler.source_digest == self.source_digest
            && facts.scalar_compiler.invocation_digest == self.compile_key
            && facts.scalar_compiler.header_manifest_digest == self.header_manifest_digest
            && facts.scalar_compiler.target.as_str() == "sm_89"
            && facts.scalar_compiler.nvrtc_version == self.nvrtc_version
            && facts.scalar_compiler.nvrtc_library_domain == self.nvrtc_library_domain
            && facts.scalar_compiler.nvrtc_library_known
            && facts.scalar_compiler.output_kind == ArtifactKind::Ptx
            && facts.scalar_compiler.composer_revision == COMPOSER_REVISION
            && facts.scalar_compiler.compiler_revision == COMPILER_REVISION
            && facts.scalar_compiler.numeric_abi_revision == NUMERIC_ABI_REVISION
            && facts.scalar_compiler.schedule_revision == SCHEDULE_REVISION
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct NtFixedCopyPlanComposedQualificationIdentity {
    scalar: ScalarTransposeQualificationIdentity,
    fixed: FixedCopyPlanQualificationIdentity,
}

impl NtFixedCopyPlanComposedQualificationIdentity {
    fn matches(self, facts: ScalarLaunchFacts) -> bool {
        self.scalar.nvrtc_version == self.fixed.nvrtc_version
            && self.scalar.matches(facts)
            && self.fixed.matches(facts)
    }
}

const TRIAD_SCALAR_TRANSPOSE_SOURCE_DIGEST: [u8; 32] = [
    75, 17, 4, 143, 156, 191, 255, 113, 44, 252, 78, 209, 236, 56, 35, 216, 122, 163, 181, 77, 204,
    60, 111, 7, 11, 96, 190, 49, 92, 41, 236, 17,
];

const TRIAD_SCALAR_TRANSPOSE_IDENTITY_CUDA_12_8: ScalarTransposeQualificationIdentity =
    ScalarTransposeQualificationIdentity {
        nvrtc_version: (12, 8),
        compile_key: [
            91, 63, 101, 184, 44, 105, 94, 155, 113, 77, 199, 170, 210, 99, 141, 192, 19, 236, 146,
            100, 117, 103, 20, 134, 197, 221, 33, 10, 51, 255, 202, 152,
        ],
        artifact_digest: [
            241, 105, 79, 17, 181, 142, 8, 14, 202, 196, 149, 185, 176, 59, 22, 3, 214, 221, 64,
            101, 148, 199, 204, 67, 23, 179, 139, 141, 223, 235, 80, 125,
        ],
        source_digest: TRIAD_SCALAR_TRANSPOSE_SOURCE_DIGEST,
        header_manifest_digest: [
            145, 108, 170, 21, 194, 14, 167, 107, 142, 20, 233, 83, 15, 25, 174, 3, 22, 206, 207,
            245, 49, 163, 167, 147, 43, 37, 223, 96, 214, 43, 231, 142,
        ],
        nvrtc_library_domain: [
            38, 176, 163, 160, 32, 68, 255, 203, 193, 105, 63, 216, 62, 146, 97, 190, 255, 166,
            146, 164, 251, 207, 227, 172, 94, 157, 140, 135, 152, 11, 177, 85,
        ],
    };

const TRIAD_SCALAR_TRANSPOSE_IDENTITY_CUDA_13_0: ScalarTransposeQualificationIdentity =
    ScalarTransposeQualificationIdentity {
        nvrtc_version: (13, 0),
        compile_key: [
            28, 99, 11, 117, 75, 17, 22, 215, 10, 187, 226, 150, 182, 175, 30, 23, 175, 130, 7,
            170, 12, 98, 34, 239, 241, 36, 56, 176, 82, 119, 136, 180,
        ],
        artifact_digest: [
            37, 99, 151, 71, 78, 223, 36, 115, 82, 228, 173, 73, 69, 185, 38, 192, 193, 20, 19,
            189, 201, 211, 107, 72, 240, 166, 12, 47, 95, 9, 159, 246,
        ],
        source_digest: TRIAD_SCALAR_TRANSPOSE_SOURCE_DIGEST,
        header_manifest_digest: [
            124, 181, 100, 127, 48, 32, 30, 118, 196, 50, 152, 12, 206, 241, 71, 106, 249, 30, 220,
            156, 219, 194, 121, 84, 191, 233, 139, 210, 32, 22, 239, 31,
        ],
        nvrtc_library_domain: [
            112, 155, 145, 195, 107, 251, 14, 217, 102, 238, 105, 173, 200, 214, 248, 127, 241, 16,
            238, 207, 61, 251, 80, 96, 54, 127, 24, 60, 230, 20, 235, 13,
        ],
    };

const TRIAD_SCALAR_TRANSPOSE_IDENTITY_CUDA_13_2: ScalarTransposeQualificationIdentity =
    ScalarTransposeQualificationIdentity {
        nvrtc_version: (13, 2),
        compile_key: [
            71, 197, 215, 158, 137, 217, 179, 49, 83, 212, 39, 230, 102, 240, 129, 237, 229, 94,
            174, 126, 141, 21, 13, 54, 16, 161, 33, 23, 219, 167, 125, 189,
        ],
        artifact_digest: [
            107, 167, 24, 70, 126, 212, 160, 36, 204, 143, 121, 2, 164, 55, 116, 135, 140, 222, 51,
            93, 33, 52, 19, 100, 151, 120, 86, 161, 232, 163, 116, 82,
        ],
        source_digest: TRIAD_SCALAR_TRANSPOSE_SOURCE_DIGEST,
        header_manifest_digest: [
            250, 112, 31, 143, 180, 144, 31, 11, 170, 60, 180, 213, 4, 172, 199, 141, 236, 254, 41,
            36, 143, 155, 246, 99, 73, 211, 207, 70, 3, 90, 56, 44,
        ],
        nvrtc_library_domain: [
            208, 49, 165, 62, 185, 114, 53, 183, 15, 98, 246, 82, 147, 45, 177, 189, 247, 40, 234,
            34, 156, 140, 168, 9, 213, 60, 95, 253, 145, 100, 38, 135,
        ],
    };

/// Frozen whole-pipeline identities for the portable transpose followed by
/// the Fixed CopyPlan kernel. Each entry binds both modules to the same live
/// toolkit domain; a Fixed-only match is intentionally insufficient.
const NT_FIXED_COPYPLAN_COMPOSED_QUALIFICATION_CANDIDATES:
    &[NtFixedCopyPlanComposedQualificationIdentity] = &[
    NtFixedCopyPlanComposedQualificationIdentity {
        scalar: TRIAD_SCALAR_TRANSPOSE_IDENTITY_CUDA_12_8,
        fixed: FIXED_COPYPLAN_EVIDENCE_COHORTS[0],
    },
    NtFixedCopyPlanComposedQualificationIdentity {
        scalar: TRIAD_SCALAR_TRANSPOSE_IDENTITY_CUDA_12_8,
        fixed: FIXED_COPYPLAN_EVIDENCE_COHORTS[1],
    },
    NtFixedCopyPlanComposedQualificationIdentity {
        scalar: TRIAD_SCALAR_TRANSPOSE_IDENTITY_CUDA_13_0,
        fixed: FIXED_COPYPLAN_EVIDENCE_COHORTS[2],
    },
    NtFixedCopyPlanComposedQualificationIdentity {
        scalar: TRIAD_SCALAR_TRANSPOSE_IDENTITY_CUDA_13_0,
        fixed: FIXED_COPYPLAN_EVIDENCE_COHORTS[3],
    },
    NtFixedCopyPlanComposedQualificationIdentity {
        scalar: TRIAD_SCALAR_TRANSPOSE_IDENTITY_CUDA_13_2,
        fixed: FIXED_COPYPLAN_EVIDENCE_COHORTS[4],
    },
    NtFixedCopyPlanComposedQualificationIdentity {
        scalar: TRIAD_SCALAR_TRANSPOSE_IDENTITY_CUDA_13_2,
        fixed: FIXED_COPYPLAN_EVIDENCE_COHORTS[5],
    },
];

// The six measured complete-module pairs cover cap16/cap64 on each retained
// toolkit. Admission is an alias of that sealed set so no unmeasured pair can
// be appended here.
const NT_FIXED_COPYPLAN_COMPOSED_EVIDENCE_COHORTS:
    &[NtFixedCopyPlanComposedQualificationIdentity] =
    NT_FIXED_COPYPLAN_COMPOSED_QUALIFICATION_CANDIDATES;

// These three toolkit projections keep TN independent of the Fixed holder and
// of the paired cap-specific Fixed identity.
const TN_M16N16_SPLITM16_SM89_BINDINGS: [ScalarTransposeQualificationIdentity; 3] = [
    NT_FIXED_COPYPLAN_COMPOSED_QUALIFICATION_CANDIDATES[0].scalar,
    NT_FIXED_COPYPLAN_COMPOSED_QUALIFICATION_CANDIDATES[2].scalar,
    NT_FIXED_COPYPLAN_COMPOSED_QUALIFICATION_CANDIDATES[4].scalar,
];

fn qualified_nt_fixed_copyplan_sibling_environment(facts: ScalarLaunchFacts) -> bool {
    facts.compute_capability == (8, 9)
        && facts.multiprocessor_count == 142
        && NT_FIXED_COPYPLAN_COMPOSED_EVIDENCE_COHORTS
            .iter()
            .copied()
            .any(|identity| identity.matches(facts))
}

fn qualified_scalar_sm89_ada142_tn_environment(facts: ScalarLaunchFacts) -> bool {
    facts.compute_capability == (8, 9)
        && facts.multiprocessor_count == 142
        && TN_M16N16_SPLITM16_SM89_BINDINGS
            .iter()
            .copied()
            .any(|identity| identity.matches(facts))
}

#[cfg(test)]
pub(super) fn scalar_sm89_composed_test_facts(index: usize) -> ScalarLaunchFacts {
    let identity = NT_FIXED_COPYPLAN_COMPOSED_QUALIFICATION_CANDIDATES[index];
    let compiler = |source_digest,
                    invocation_digest,
                    header_manifest_digest,
                    nvrtc_version,
                    nvrtc_library_domain| CompilerIdentity {
        source_digest,
        invocation_digest,
        header_manifest_digest,
        target: crate::mamba_ssm::gpu::kernel_identity::CudaTarget::new("sm_89").unwrap(),
        nvrtc_version,
        nvrtc_library_domain,
        nvrtc_library_known: true,
        output_kind: ArtifactKind::Ptx,
        composer_revision: COMPOSER_REVISION,
        compiler_revision: COMPILER_REVISION,
        numeric_abi_revision: NUMERIC_ABI_REVISION,
        schedule_revision: SCHEDULE_REVISION,
    };
    ScalarLaunchFacts {
        scalar_artifact: ArtifactIdentity {
            module_kind: ModuleKind::TriadScalar,
            artifact_kind: ArtifactKind::Ptx,
            compile_key: identity.scalar.compile_key,
            artifact_digest: identity.scalar.artifact_digest,
        },
        scalar_compiler: compiler(
            identity.scalar.source_digest,
            identity.scalar.compile_key,
            identity.scalar.header_manifest_digest,
            identity.scalar.nvrtc_version,
            identity.scalar.nvrtc_library_domain,
        ),
        fixed_artifact: ArtifactIdentity {
            module_kind: ModuleKind::Fixed,
            artifact_kind: ArtifactKind::Ptx,
            compile_key: identity.fixed.compile_key,
            artifact_digest: identity.fixed.artifact_digest,
        },
        fixed_compiler: compiler(
            identity.fixed.source_digest,
            identity.fixed.compile_key,
            identity.fixed.header_manifest_digest,
            identity.fixed.nvrtc_version,
            identity.fixed.nvrtc_library_domain,
        ),
        fixed_copyplan_loaded: true,
        sm89_exact_f32_artifact: None,
        sm89_exact_f32_compiler: None,
        sm89_exact_f32_symbols_loaded: [false; 3],
        sm89_exact_f32_d128_artifact: None,
        sm89_exact_f32_d128_compiler: None,
        sm89_exact_f32_d128_symbols_loaded: [false; 2],
        compute_capability: (8, 9),
        multiprocessor_count: 142,
    }
}

fn qualified_fixed_copyplan_environment(facts: ScalarLaunchFacts) -> bool {
    facts.compute_capability == (8, 9)
        && facts.multiprocessor_count == 142
        && FIXED_COPYPLAN_EVIDENCE_COHORTS
            .iter()
            .copied()
            .any(|identity| identity.matches(facts))
}

const NN_FIXED_COPYPLAN_SM89_CELLS: [F32TriadShape; 4] = [
    F32TriadShape {
        m: 2048,
        k: 768,
        n: 3072,
        lda: 768,
        ldb: 3072,
        ldc: 3072,
    },
    F32TriadShape {
        m: 2048,
        k: 1536,
        n: 768,
        lda: 1536,
        ldb: 768,
        ldc: 768,
    },
    F32TriadShape {
        m: 4621,
        k: 384,
        n: 1928,
        lda: 384,
        ldb: 1928,
        ldc: 1928,
    },
    // The deep 4096-row forward: the plain kernel held it at 23.5 TFLOP/s
    // where the copy plan holds its other cells at 33 to 35, the one cell
    // of the release table that ran slower than 0.6.9.
    F32TriadShape {
        m: 4096,
        k: 3072,
        n: 1536,
        lda: 3072,
        ldb: 1536,
        ldc: 1536,
    },
];

const NT_D768_OUT_FIXED_COPYPLAN_SM89_CELL: F32TriadShape = F32TriadShape {
    m: 2048,
    k: 1536,
    n: 768,
    lda: 768,
    ldb: 768,
    ldc: 1536,
};

const NT_D768_IN_FIXED_COPYPLAN_SM89_CELL: F32TriadShape = F32TriadShape {
    m: 2048,
    k: 768,
    n: 3072,
    lda: 3072,
    ldb: 3072,
    ldc: 768,
};

const NT_PRISM_FIXED_COPYPLAN_SM89_CELL: F32TriadShape = F32TriadShape {
    m: 4621,
    k: 384,
    n: 1928,
    lda: 1928,
    ldb: 1928,
    ldc: 384,
};

fn qualified_scalar_sm120_cc120_170_nvrtc132_environment(facts: ScalarLaunchFacts) -> bool {
    facts.compute_capability == (12, 0)
        && facts.multiprocessor_count == 170
        && facts.scalar_artifact.module_kind == ModuleKind::TriadScalar
        && facts.scalar_artifact.artifact_kind == facts.scalar_compiler.output_kind
        && facts.scalar_artifact.compile_key == facts.scalar_compiler.invocation_digest
        && facts.scalar_artifact.artifact_digest != [0; 32]
        && facts.scalar_compiler.invocation_digest != [0; 32]
        && facts.scalar_compiler.target.as_str() == "compute_120"
        && facts.scalar_compiler.nvrtc_version == (13, 2)
        && facts.scalar_compiler.nvrtc_library_known
        && facts.scalar_compiler.nvrtc_library_domain != [0; 32]
}

const NN_M64N64_SM120_CC120_170_NVRTC132_CELLS: [F32TriadShape; 7] = [
    F32TriadShape {
        m: 2048,
        k: 3072,
        n: 768,
        lda: 3072,
        ldb: 768,
        ldc: 768,
    },
    F32TriadShape {
        m: 4096,
        k: 3072,
        n: 1536,
        lda: 3072,
        ldb: 1536,
        ldc: 1536,
    },
    F32TriadShape {
        m: 512,
        k: 3072,
        n: 768,
        lda: 3072,
        ldb: 768,
        ldc: 768,
    },
    F32TriadShape {
        m: 4096,
        k: 512,
        n: 768,
        lda: 512,
        ldb: 768,
        ldc: 768,
    },
    F32TriadShape {
        m: 2048,
        k: 768,
        n: 3072,
        lda: 768,
        ldb: 3072,
        ldc: 3072,
    },
    F32TriadShape {
        m: 2048,
        k: 1536,
        n: 768,
        lda: 1536,
        ldb: 768,
        ldc: 768,
    },
    F32TriadShape {
        m: 4621,
        k: 384,
        n: 1928,
        lda: 384,
        ldb: 1928,
        ldc: 1928,
    },
];

fn qualified_nn_m64n64_request(request: F32TriadRequest) -> bool {
    request.op == crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp::Nn
        && NN_M64N64_SM120_CC120_170_NVRTC132_CELLS.contains(&request.shape)
}

fn qualified_nn_m64n64_operands(operands: F32TriadOperands) -> bool {
    const ALIGNMENT: u64 = 16;
    operands.bias.is_none()
        && operands.alpha.to_bits() == 1.0_f32.to_bits()
        && operands.beta.to_bits() == 0.0_f32.to_bits()
        && [operands.output, operands.a, operands.b]
            .into_iter()
            .all(|pointer| pointer != 0 && pointer.is_multiple_of(ALIGNMENT))
}

const NN_M32N64_SPLITK32_SM120_CC120_170_NVRTC132_CELL: F32TriadShape = F32TriadShape {
    m: 128,
    k: 8_192,
    n: 128,
    lda: 8_192,
    ldb: 128,
    ldc: 128,
};

fn qualified_nn_m32n64_splitk32_request(request: F32TriadRequest) -> bool {
    request.op == crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp::Nn
        && request.shape == NN_M32N64_SPLITK32_SM120_CC120_170_NVRTC132_CELL
}

fn qualified_nn_m32n64_splitk32_operands(operands: F32TriadOperands) -> bool {
    qualified_nn_m64n64_operands(operands)
}

const NT_D768_TRANSPOSE_M64N64_SM120_CC120_170_NVRTC132_CELL: F32TriadShape = F32TriadShape {
    m: 2048,
    k: 768,
    n: 3072,
    lda: 3072,
    ldb: 3072,
    ldc: 768,
};

const NT_D768_OUT_TRANSPOSE_M64N64_SM120_CC120_170_NVRTC132_CELL: F32TriadShape = F32TriadShape {
    m: 2048,
    k: 1536,
    n: 768,
    lda: 768,
    ldb: 768,
    ldc: 1536,
};

const NT_D768_OUT_TRANSPOSE_ELEMENTS: usize = 1_179_648;

const NT_LARGE_DEEP_TRANSPOSE_M64N64_SM120_CC120_170_NVRTC132_CELL: F32TriadShape = F32TriadShape {
    m: 4096,
    k: 3072,
    n: 1536,
    lda: 1536,
    ldb: 1536,
    ldc: 3072,
};

const NT_PRISM_VECTOR_SM120_CC120_170_NVRTC132_CELL: F32TriadShape = F32TriadShape {
    m: 4621,
    k: 384,
    n: 1928,
    lda: 1928,
    ldb: 1928,
    ldc: 384,
};

const NT_PRISM_TRANSPOSE_ELEMENTS: usize = 740_352;

// The deep 4096-row product's transposed input: the largest exact-route
// transpose, which sized the scratch before the TF32 weight gradient of the
// same shape needed more.
const NT_LARGE_DEEP_TRANSPOSE_ELEMENTS: usize = 4_718_592;

const NT_D128_OUT_TRANSPOSE_M64N64_SM120_CC120_170_NVRTC132_CELL: F32TriadShape = F32TriadShape {
    m: 1024,
    k: 256,
    n: 128,
    lda: 128,
    ldb: 128,
    ldc: 256,
};

const NT_D128_OUT_TRANSPOSE_ELEMENTS: usize = 32_768;

const TN_M16N16_SPLITM16_SM120_CC120_170_NVRTC132_CELL: F32TriadShape = F32TriadShape {
    m: 256,
    k: 512,
    n: 384,
    lda: 512,
    ldb: 384,
    ldc: 384,
};

fn qualified_nt_m2n16_splitk32_request(request: F32TriadRequest) -> bool {
    request.op == crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp::Nt
        && request.shape
            == F32TriadShape::contiguous(
                crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp::Nt,
                (512, 16, 2_048),
            )
}

fn qualified_tn_m16n16_splitm16_request(request: F32TriadRequest) -> bool {
    request.op == crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp::Tn
        && request.shape == TN_M16N16_SPLITM16_SM120_CC120_170_NVRTC132_CELL
}

fn qualified_tn_m16n16_splitm16_operands(operands: F32TriadOperands) -> bool {
    const ALIGNMENT: u64 = 16;
    operands.bias.is_none()
        && operands.alpha.to_bits() == 1.0_f32.to_bits()
        && operands.beta.to_bits() == 1.0_f32.to_bits()
        && [operands.output, operands.a, operands.b]
            .into_iter()
            .all(|pointer| pointer != 0 && pointer.is_multiple_of(ALIGNMENT))
}

fn qualified_nt_d768_transpose_m64n64_request(request: F32TriadRequest) -> bool {
    request.op == crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp::Nt
        && request.shape == NT_D768_TRANSPOSE_M64N64_SM120_CC120_170_NVRTC132_CELL
}

fn qualified_nt_d768_out_transpose_m64n64_request(request: F32TriadRequest) -> bool {
    request.op == crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp::Nt
        && request.shape == NT_D768_OUT_TRANSPOSE_M64N64_SM120_CC120_170_NVRTC132_CELL
        && request.shape.k.checked_mul(request.shape.n) == Some(NT_D768_OUT_TRANSPOSE_ELEMENTS)
}

fn qualified_nt_large_deep_transpose_m64n64_request(request: F32TriadRequest) -> bool {
    request.op == crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp::Nt
        && request.shape == NT_LARGE_DEEP_TRANSPOSE_M64N64_SM120_CC120_170_NVRTC132_CELL
        && request.shape.k.checked_mul(request.shape.n) == Some(NT_LARGE_DEEP_TRANSPOSE_ELEMENTS)
}

fn qualified_nt_prism_vector_request(request: F32TriadRequest) -> bool {
    request.op == crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp::Nt
        && request.shape == NT_PRISM_VECTOR_SM120_CC120_170_NVRTC132_CELL
        && request.shape.k.checked_mul(request.shape.n) == Some(NT_PRISM_TRANSPOSE_ELEMENTS)
}

fn qualified_nt_d128_out_transpose_m64n64_request(request: F32TriadRequest) -> bool {
    request.op == crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp::Nt
        && request.shape == NT_D128_OUT_TRANSPOSE_M64N64_SM120_CC120_170_NVRTC132_CELL
        && request.shape.k.checked_mul(request.shape.n) == Some(NT_D128_OUT_TRANSPOSE_ELEMENTS)
}

fn qualified_nt_d768_transpose_m64n64_operands(operands: F32TriadOperands) -> bool {
    const ALIGNMENT: u64 = 16;
    operands.bias.is_none()
        && operands.alpha.to_bits() == 1.0_f32.to_bits()
        && operands.beta.to_bits() == 0.0_f32.to_bits()
        && [operands.output, operands.a, operands.b]
            .into_iter()
            .all(|pointer| pointer != 0 && pointer.is_multiple_of(ALIGNMENT))
}

fn qualified_tn_narrow_splitm_operands(operands: F32TriadOperands) -> bool {
    let alignment = std::mem::align_of::<f32>() as u64;
    operands.bias.is_none()
        && operands.alpha.to_bits() == 1.0_f32.to_bits()
        && operands.beta.to_bits() == 1.0_f32.to_bits()
        && [operands.output, operands.a, operands.b]
            .into_iter()
            .all(|pointer| pointer != 0 && pointer.is_multiple_of(alignment))
}

fn qualified_sm89_exact_f32_tn_operands(operands: F32TriadOperands) -> bool {
    const ALIGNMENT: u64 = 16;
    operands.bias.is_none()
        && operands.alpha.to_bits() == 1.0_f32.to_bits()
        && operands.beta.to_bits() == 1.0_f32.to_bits()
        && [operands.output, operands.a, operands.b]
            .into_iter()
            .all(|pointer| pointer != 0 && pointer.is_multiple_of(ALIGNMENT))
}

fn qualified_sm89_exact_f32_d128_forced_operands(operands: F32TriadOperands) -> bool {
    const ALIGNMENT: u64 = 16;
    operands.bias.is_none()
        && operands.beta.to_bits() == 1.0_f32.to_bits()
        && [operands.output, operands.a, operands.b]
            .into_iter()
            .all(|pointer| pointer != 0 && pointer.is_multiple_of(ALIGNMENT))
}

const TN_D768_IN_SM89_EXACT_F32_CELL: F32TriadShape = F32TriadShape {
    m: 2_048,
    k: 768,
    n: 3_072,
    lda: 768,
    ldb: 3_072,
    ldc: 3_072,
};
const TN_D768_OUT_SM89_EXACT_F32_CELL: F32TriadShape = F32TriadShape {
    m: 2_048,
    k: 1_536,
    n: 768,
    lda: 1_536,
    ldb: 768,
    ldc: 768,
};
const TN_PRISM_SM89_EXACT_F32_CELL: F32TriadShape = F32TriadShape {
    m: 4_621,
    k: 384,
    n: 1_928,
    lda: 384,
    ldb: 1_928,
    ldc: 1_928,
};

const TN_D128_IN_SM89_EXACT_F32_CELL: F32TriadShape = F32TriadShape {
    m: 1_024,
    k: 128,
    n: 512,
    lda: 128,
    ldb: 512,
    ldc: 512,
};

const TN_D128_OUT_SM89_EXACT_F32_CELL: F32TriadShape = F32TriadShape {
    m: 1_024,
    k: 256,
    n: 128,
    lda: 256,
    ldb: 128,
    ldc: 128,
};

fn sm89_exact_f32_d128_plan_for_route(
    facts: ScalarLaunchFacts,
    request: F32TriadRequest,
    operands: F32TriadOperands,
    route: super::sm89_exact_f32_d128_source::Sm89ExactF32D128Route,
    admission: ScalarAdmission,
) -> Result<ScalarDispatchPlan, String> {
    use super::sm89_exact_f32_d128_source::Sm89ExactF32D128Route;
    let (shape, plan, symbol_index) = match route {
        Sm89ExactF32D128Route::D128InDirectFold => (
            TN_D128_IN_SM89_EXACT_F32_CELL,
            ScalarDispatchPlan::TnD128InSm89DirectFoldQualified,
            0,
        ),
        Sm89ExactF32D128Route::D128OutDirectFold => (
            TN_D128_OUT_SM89_EXACT_F32_CELL,
            ScalarDispatchPlan::TnD128OutSm89DirectFoldQualified,
            1,
        ),
    };
    let actual_fallback = scalar_dispatch_plan(request, facts.multiprocessor_count)?;
    let required_fallback = ScalarDispatchPlan::TnSplitM {
        m_chunk: 16,
        chunks: 64,
    };
    let environment_ok = match admission {
        ScalarAdmission::Evidence => qualified_sm89_exact_f32_d128_environment(facts),
        ScalarAdmission::Candidate => qualified_sm89_exact_f32_d128_candidate_environment(facts),
        ScalarAdmission::Proof => scalar_proof_board(facts),
    };
    let operands_ok = if admission == ScalarAdmission::Candidate {
        qualified_sm89_exact_f32_d128_forced_operands(operands)
    } else {
        qualified_sm89_exact_f32_tn_operands(operands)
    };
    if actual_fallback != required_fallback
        || request.op != crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp::Tn
        || request.shape != shape
        || !operands_ok
        || !environment_ok
        || !facts.sm89_exact_f32_d128_symbols_loaded[symbol_index]
    {
        return Ok(actual_fallback);
    }
    Ok(plan)
}

pub(super) fn forced_sm89_exact_f32_d128_plan(
    facts: ScalarLaunchFacts,
    request: F32TriadRequest,
    operands: F32TriadOperands,
    route: super::sm89_exact_f32_d128_source::Sm89ExactF32D128Route,
) -> Result<ScalarDispatchPlan, String> {
    let plan = sm89_exact_f32_d128_plan_for_route(
        facts,
        request,
        operands,
        route,
        ScalarAdmission::Candidate,
    )?;
    if matches!(
        plan,
        ScalarDispatchPlan::TnD128InSm89DirectFoldQualified
            | ScalarDispatchPlan::TnD128OutSm89DirectFoldQualified
    ) {
        Ok(plan)
    } else {
        Err(format!(
            "SM89 exact-F32 d128 forced route {:?} failed shape, operand, symbol, or candidate-identity qualification",
            route
        ))
    }
}

fn sm89_exact_f32_plan_for_route(
    facts: ScalarLaunchFacts,
    request: F32TriadRequest,
    operands: F32TriadOperands,
    route: super::sm89_exact_f32_source::Sm89ExactF32TnRoute,
    admission: ScalarAdmission,
) -> Result<ScalarDispatchPlan, String> {
    use super::sm89_exact_f32_source::Sm89ExactF32TnRoute;
    let (shape, fallback, plan, symbol_index) = match route {
        Sm89ExactF32TnRoute::D768InDualChunkFused => (
            TN_D768_IN_SM89_EXACT_F32_CELL,
            ScalarDispatchPlan::TnSplitM {
                m_chunk: 1_024,
                chunks: 2,
            },
            ScalarDispatchPlan::TnD768InSm89DualChunkQualified,
            0,
        ),
        Sm89ExactF32TnRoute::D768OutDirectBk16 => (
            TN_D768_OUT_SM89_EXACT_F32_CELL,
            ScalarDispatchPlan::TnSplitM {
                m_chunk: 512,
                chunks: 4,
            },
            ScalarDispatchPlan::TnD768OutSm89DirectBk16Qualified,
            1,
        ),
        Sm89ExactF32TnRoute::PrismDirectBk16 => (
            TN_PRISM_SM89_EXACT_F32_CELL,
            ScalarDispatchPlan::TnSplitM {
                m_chunk: 784,
                chunks: 6,
            },
            ScalarDispatchPlan::TnPrismSm89DirectBk16Qualified,
            2,
        ),
    };
    let actual_fallback = scalar_dispatch_plan(request, facts.multiprocessor_count)?;
    let environment_ok = match admission {
        ScalarAdmission::Evidence => qualified_sm89_exact_f32_environment(facts),
        ScalarAdmission::Candidate => qualified_sm89_exact_f32_candidate_environment(facts),
        ScalarAdmission::Proof => scalar_proof_board(facts),
    };
    if actual_fallback != fallback
        || request.op != crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp::Tn
        || request.shape != shape
        || !qualified_sm89_exact_f32_tn_operands(operands)
        || !environment_ok
        || !facts.sm89_exact_f32_symbols_loaded[symbol_index]
    {
        return Ok(actual_fallback);
    }
    Ok(plan)
}

pub(super) fn forced_sm89_exact_f32_plan(
    facts: ScalarLaunchFacts,
    request: F32TriadRequest,
    operands: F32TriadOperands,
    route: super::sm89_exact_f32_source::Sm89ExactF32TnRoute,
) -> Result<ScalarDispatchPlan, String> {
    let plan =
        sm89_exact_f32_plan_for_route(facts, request, operands, route, ScalarAdmission::Candidate)?;
    if matches!(
        plan,
        ScalarDispatchPlan::TnD768InSm89DualChunkQualified
            | ScalarDispatchPlan::TnD768OutSm89DirectBk16Qualified
            | ScalarDispatchPlan::TnPrismSm89DirectBk16Qualified
    ) {
        Ok(plan)
    } else {
        Err(format!(
            "SM89 exact-F32 forced route {:?} failed shape, operand, symbol, or candidate-identity qualification",
            route
        ))
    }
}

/// Resolves the scalar physical launch plan from request, operands, and the
/// complete device/compiler policy facts available to both prepared and raw
/// launch paths, by the frozen evidence alone. Cells without evidence retain
/// the ordinary scalar plan.
pub(super) fn scalar_launch_plan(
    facts: ScalarLaunchFacts,
    request: F32TriadRequest,
    operands: F32TriadOperands,
) -> Result<ScalarDispatchPlan, String> {
    scalar_plan_with_admission(facts, request, operands, ScalarAdmission::Evidence)
}

/// The plan a board without frozen evidence for the cell would serve once
/// the first-use proof admits it; `None` when the cell has no candidate.
pub(super) fn scalar_proof_plan(
    facts: ScalarLaunchFacts,
    request: F32TriadRequest,
    operands: F32TriadOperands,
) -> Result<Option<ScalarDispatchPlan>, String> {
    let plan = scalar_plan_with_admission(facts, request, operands, ScalarAdmission::Proof)?;
    Ok((plan != scalar_dispatch_plan(request, facts.multiprocessor_count)?).then_some(plan))
}

fn scalar_plan_with_admission(
    facts: ScalarLaunchFacts,
    request: F32TriadRequest,
    operands: F32TriadOperands,
    admission: ScalarAdmission,
) -> Result<ScalarDispatchPlan, String> {
    let fallback = scalar_dispatch_plan(request, facts.multiprocessor_count)?;
    for route in [
        super::sm89_exact_f32_d128_source::Sm89ExactF32D128Route::D128InDirectFold,
        super::sm89_exact_f32_d128_source::Sm89ExactF32D128Route::D128OutDirectFold,
    ] {
        let plan = sm89_exact_f32_d128_plan_for_route(facts, request, operands, route, admission)?;
        if plan != fallback {
            return Ok(plan);
        }
    }
    for route in [
        super::sm89_exact_f32_source::Sm89ExactF32TnRoute::D768InDualChunkFused,
        super::sm89_exact_f32_source::Sm89ExactF32TnRoute::D768OutDirectBk16,
        super::sm89_exact_f32_source::Sm89ExactF32TnRoute::PrismDirectBk16,
    ] {
        let plan = sm89_exact_f32_plan_for_route(facts, request, operands, route, admission)?;
        if plan != fallback {
            return Ok(plan);
        }
    }
    if fallback == (ScalarDispatchPlan::NnFinal { slim: false })
        && fixed_overlay_admitted(admission, facts, qualified_fixed_copyplan_environment)
        && request.op == crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp::Nn
        && NN_FIXED_COPYPLAN_SM89_CELLS.contains(&request.shape)
        && qualified_nn_m64n64_operands(operands)
    {
        return Ok(ScalarDispatchPlan::NnSm89FixedCopyPlanQualified);
    }
    if fallback == (ScalarDispatchPlan::NtFinal { slim: false })
        && fixed_overlay_admitted(admission, facts, qualified_fixed_copyplan_environment)
        && request.op == crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp::Nt
        && request.shape == NT_D768_OUT_FIXED_COPYPLAN_SM89_CELL
        && qualified_nt_d768_transpose_m64n64_operands(operands)
    {
        return Ok(ScalarDispatchPlan::NtD768OutSm89FixedCopyPlanQualified);
    }
    if fallback == (ScalarDispatchPlan::NtFinal { slim: false })
        && fixed_overlay_admitted(
            admission,
            facts,
            qualified_nt_fixed_copyplan_sibling_environment,
        )
        && request.op == crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp::Nt
        && request.shape == NT_D768_IN_FIXED_COPYPLAN_SM89_CELL
        && qualified_nt_d768_transpose_m64n64_operands(operands)
    {
        return Ok(ScalarDispatchPlan::NtD768InSm89FixedCopyPlanQualified);
    }
    if fallback == (ScalarDispatchPlan::NtFinal { slim: true })
        && fixed_overlay_admitted(
            admission,
            facts,
            qualified_nt_fixed_copyplan_sibling_environment,
        )
        && request.op == crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp::Nt
        && request.shape == NT_PRISM_FIXED_COPYPLAN_SM89_CELL
        && qualified_nt_d768_transpose_m64n64_operands(operands)
    {
        return Ok(ScalarDispatchPlan::NtPrismSm89FixedCopyPlanQualified);
    }
    if fallback == (ScalarDispatchPlan::NtFinal { slim: false })
        && fixed_overlay_admitted(
            admission,
            facts,
            qualified_nt_fixed_copyplan_sibling_environment,
        )
        && request.op == crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp::Nt
        && qualified_nt_large_deep_transpose_m64n64_request(request)
        && qualified_nt_d768_transpose_m64n64_operands(operands)
    {
        return Ok(ScalarDispatchPlan::NtLargeDeepSm89FixedCopyPlanQualified);
    }
    if fallback
        == (ScalarDispatchPlan::NtSplitKMain {
            n_main: 2_048,
            n_tail: 0,
        })
        && qualified_scalar_sm120_cc120_170_nvrtc132_environment(facts)
        && qualified_nt_m2n16_splitk32_request(request)
        && qualified_nt_d768_transpose_m64n64_operands(operands)
    {
        return Ok(ScalarDispatchPlan::NtM2N16SplitK32Qualified);
    }
    if fallback == (ScalarDispatchPlan::NnFinal { slim: false })
        && qualified_scalar_sm120_cc120_170_nvrtc132_environment(facts)
        && qualified_nn_m64n64_request(request)
        && qualified_nn_m64n64_operands(operands)
    {
        return Ok(ScalarDispatchPlan::NnM64N64Qualified);
    }
    if fallback == ScalarDispatchPlan::NnSplitKThin
        && qualified_scalar_sm120_cc120_170_nvrtc132_environment(facts)
        && qualified_nn_m32n64_splitk32_request(request)
        && qualified_nn_m32n64_splitk32_operands(operands)
    {
        return Ok(ScalarDispatchPlan::NnM32N64SplitK32Qualified);
    }
    if fallback == (ScalarDispatchPlan::NtFinal { slim: false })
        && qualified_scalar_sm120_cc120_170_nvrtc132_environment(facts)
        && qualified_nt_d768_transpose_m64n64_request(request)
        && qualified_nt_d768_transpose_m64n64_operands(operands)
    {
        return Ok(ScalarDispatchPlan::NtD768TransposeM64N64Qualified);
    }
    if fallback == (ScalarDispatchPlan::NtFinal { slim: false })
        && qualified_scalar_sm120_cc120_170_nvrtc132_environment(facts)
        && qualified_nt_d768_out_transpose_m64n64_request(request)
        && qualified_nt_d768_transpose_m64n64_operands(operands)
    {
        return Ok(ScalarDispatchPlan::NtD768OutTransposeM64N64Qualified);
    }
    if fallback == (ScalarDispatchPlan::NtFinal { slim: false })
        && qualified_scalar_sm120_cc120_170_nvrtc132_environment(facts)
        && qualified_nt_large_deep_transpose_m64n64_request(request)
        && qualified_nt_d768_transpose_m64n64_operands(operands)
    {
        return Ok(ScalarDispatchPlan::NtLargeDeepTransposeM64N64Qualified);
    }
    if fallback == (ScalarDispatchPlan::NtFinal { slim: true })
        && qualified_scalar_sm120_cc120_170_nvrtc132_environment(facts)
        && qualified_nt_prism_vector_request(request)
        && qualified_nt_d768_transpose_m64n64_operands(operands)
    {
        return Ok(ScalarDispatchPlan::NtPrismVectorQualified);
    }
    if fallback
        == (ScalarDispatchPlan::NtSplitKMain {
            n_main: 128,
            n_tail: 0,
        })
        && qualified_scalar_sm120_cc120_170_nvrtc132_environment(facts)
        && qualified_nt_d128_out_transpose_m64n64_request(request)
        && qualified_nt_d768_transpose_m64n64_operands(operands)
    {
        return Ok(ScalarDispatchPlan::NtD128OutTransposeM64N64Qualified);
    }
    if fallback
        == (ScalarDispatchPlan::TnSplitM {
            m_chunk: 16,
            chunks: 16,
        })
        && (qualified_scalar_sm120_cc120_170_nvrtc132_environment(facts)
            || scalar_admitted(
                admission,
                facts,
                qualified_scalar_sm89_ada142_tn_environment,
            ))
        && qualified_tn_m16n16_splitm16_request(request)
        && qualified_tn_m16n16_splitm16_operands(operands)
    {
        return Ok(ScalarDispatchPlan::TnM16N16SplitM16Qualified);
    }
    if fallback != ScalarDispatchPlan::TnNarrow
        || !qualified_scalar_sm120_cc120_170_nvrtc132_environment(facts)
        || !qualified_tn_narrow_splitm_operands(operands)
    {
        return Ok(fallback);
    }
    let Some(cell) = qualified_tn_narrow_splitm_cell(request) else {
        return Ok(fallback);
    };
    Ok(ScalarDispatchPlan::TnNarrowSplitM {
        m_chunk: cell.m_chunk,
        chunks: cell.chunks,
    })
}

/// Minimum N (output cols) before the dispatcher switches from Slim-N tiles
/// to Big-N tiles. Below this, Slim-N (BN=64) packs better; above it Big-N
/// (BN=128) wins on wave occupancy. The threshold once gated a cuBLAS
/// fallback; that fallback is gone, so this is only a tile-pick boundary.
pub(super) const GEMM_CUSTOM_MIN: usize = 128;

/// Boundary between Slim-N and Big tile variants (by output N dimension).
pub(super) const GEMM_SLIM_MAX: usize = 512;

/// v6.5 Phase C-1.5av: separate Slim Split-K NT-via-T n_in cap for backward dx.
/// The forward Slim NN path uses N as output dim → GEMM_SLIM_MAX=512 bounds
/// wave-fill correctness there. But NT-via-T backward dx reads n_in (input dim
/// of original forward), and the kernel itself tiles arbitrary n_in via N-axis
/// tiling — the 512 cap is conservative, not load-bearing. v6.5 multi-step
/// Production input widths can exceed 512; 768 keeps those shapes on Slim
/// Split-K NT-via-T
/// with F=4 K-tile partials (576 blocks vs plain Big NT 144 blocks).
/// Determinism preserved: F is shape-keyed (function of n_out, not batch).
pub(super) const GEMM_SLIM_NT_NIN_MAX: usize = 768;

/// M threshold below which we force Slim-N even for N ≥ 129 (wave underfill protection).
/// At M < 512, Big tile BM=128 gives ≤4 M-blocks; adding N-blocks via Slim's BN=64 (vs Big's BN=128)
/// doubles grid to reduce wave underfill on Ada's 142 SMs. Only matters when N ≥ 129 (otherwise slim already chosen).
pub(super) const GEMM_M_SLIM_FORCE: usize = 512;

/// Single source of truth for Split-K/M scratch buffer cap, in f32 elements.
/// Must match `splitk_scratch` allocation in `kernels.rs` (1 << 23 = 8M f32 = 32 MB).
/// All Split-K dispatch gates (NN fwd, NT bwd_dx, Split-M TN bwd_dw) read this.
pub(super) const SPLITK_SCRATCH_CAP: usize = 1 << 23;

/// One device-scope completion counter per fused TF32 split-K output tile.
pub(super) const TF32_SPLITK_COUNTER_CAP: usize = 1 << 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ScalarDispatchPlan {
    NnUltraThin,
    NnNarrowSmall,
    NnNarrow,
    NnGemv,
    NnSplitKThinTail { k_main: usize, k_tail: usize },
    NnSplitKThin,
    NnSplitKSlim { chunks: u32 },
    NnM32N64SplitK32Qualified,
    NnM64N64Qualified,
    NnSm89FixedCopyPlanQualified,
    NnFinal { slim: bool },
    TnGemv,
    TnNarrow,
    TnNarrowSplitM { m_chunk: usize, chunks: usize },
    TnSplitM { m_chunk: usize, chunks: usize },
    TnD768InSm89DualChunkQualified,
    TnD768OutSm89DirectBk16Qualified,
    TnPrismSm89DirectBk16Qualified,
    TnD128InSm89DirectFoldQualified,
    TnD128OutSm89DirectFoldQualified,
    TnM16N16SplitM16Qualified,
    TnFinal { slim: bool },
    NtNarrow,
    NtSmallBatchWide,
    NtGemv,
    NtSplitKTail { k_main: usize, k_tail: usize },
    NtSplitKMain { n_main: usize, n_tail: usize },
    NtSplitKSlim { chunks: u32 },
    NtMidBatchWide,
    NtM2N16SplitK32Qualified,
    NtD768TransposeM64N64Qualified,
    NtD768OutTransposeM64N64Qualified,
    NtD768OutSm89FixedCopyPlanQualified,
    NtD768InSm89FixedCopyPlanQualified,
    NtPrismSm89FixedCopyPlanQualified,
    NtLargeDeepSm89FixedCopyPlanQualified,
    NtLargeDeepTransposeM64N64Qualified,
    NtPrismVectorQualified,
    NtD128OutTransposeM64N64Qualified,
    NtFinal { slim: bool },
}

impl ScalarDispatchPlan {
    pub(super) const fn needs_split_scratch(self) -> bool {
        matches!(
            self,
            Self::NnSplitKThinTail { .. }
                | Self::NnSplitKThin
                | Self::NnSplitKSlim { .. }
                | Self::NnM32N64SplitK32Qualified
                | Self::TnNarrowSplitM { .. }
                | Self::TnSplitM { .. }
                | Self::TnD768OutSm89DirectBk16Qualified
                | Self::TnPrismSm89DirectBk16Qualified
                | Self::NtSplitKTail { .. }
                | Self::NtSplitKMain { .. }
                | Self::NtSplitKSlim { .. }
        )
    }

    pub(super) const fn needs_transpose_scratch(self) -> bool {
        matches!(
            self,
            Self::NtSplitKTail { .. }
                | Self::NtSplitKMain { .. }
                | Self::NtSplitKSlim { .. }
                | Self::NtD768TransposeM64N64Qualified
                | Self::NtD768OutTransposeM64N64Qualified
                | Self::NtD768OutSm89FixedCopyPlanQualified
                | Self::NtD768InSm89FixedCopyPlanQualified
                | Self::NtPrismSm89FixedCopyPlanQualified
                | Self::NtLargeDeepSm89FixedCopyPlanQualified
                | Self::NtLargeDeepTransposeM64N64Qualified
                | Self::NtPrismVectorQualified
                | Self::NtD128OutTransposeM64N64Qualified
                | Self::TnD768InSm89DualChunkQualified
        )
    }
}

fn slim_final(output_rows: usize, output_columns: usize) -> bool {
    output_columns <= GEMM_SLIM_MAX
        || (output_rows < GEMM_M_SLIM_FORCE && output_columns >= GEMM_CUSTOM_MIN)
}

fn scalar_wave_policy(
    multiprocessor_count: u32,
) -> Result<crate::mamba_ssm::gpu::kernel_identity::ScalarWavePolicy, String> {
    if multiprocessor_count == 0 {
        return Err("scalar dispatch requires a nonzero multiprocessor count".into());
    }
    let policy = crate::mamba_ssm::gpu::kernel_identity::ScalarWavePolicy::current();
    if policy.thin_split_wave_denominator == 0
        || policy.slim_split_wave_denominator == 0
        || policy.tn_split_m_wave_denominator == 0
    {
        return Err("scalar dispatch wave policy has a zero denominator".into());
    }
    Ok(policy)
}

fn grid_below_waves(
    blocks: u32,
    multiprocessor_count: u32,
    numerator: u64,
    denominator: u64,
) -> Result<bool, String> {
    let blocks = u64::from(blocks)
        .checked_mul(denominator)
        .ok_or_else(|| "scalar dispatch block-wave comparison overflows u64".to_string())?;
    let target = u64::from(multiprocessor_count)
        .checked_mul(numerator)
        .ok_or_else(|| "scalar dispatch SM-wave comparison overflows u64".to_string())?;
    Ok(blocks < target)
}

fn wave_target_blocks(
    multiprocessor_count: u32,
    numerator: u64,
    denominator: u64,
) -> Result<u32, String> {
    let scaled = u64::from(multiprocessor_count)
        .checked_mul(numerator)
        .ok_or_else(|| "scalar dispatch target wave count overflows u64".to_string())?;
    u32::try_from(scaled.div_ceil(denominator))
        .map_err(|_| "scalar dispatch target block count exceeds u32::MAX".into())
}

fn scalar_nn_plan(dims: GemmDims, multiprocessor_count: u32) -> Result<ScalarDispatchPlan, String> {
    let policy = scalar_wave_policy(multiprocessor_count)?;
    let (batch, n_in, n_out) = dims.tuple();
    if (1..32).contains(&batch) && (32..=2048).contains(&n_in) && n_out >= 32 {
        return Ok(ScalarDispatchPlan::NnUltraThin);
    }
    if (2..=127).contains(&n_out) && (1..=64).contains(&batch) {
        return Ok(ScalarDispatchPlan::NnNarrowSmall);
    }
    if (2..=127).contains(&n_out) {
        return Ok(ScalarDispatchPlan::NnNarrow);
    }
    if n_out == 1 && n_in >= 32 {
        return Ok(ScalarDispatchPlan::NnGemv);
    }
    let plain_slim_blocks = checked_tile_grid(dims.m_u32, 128, dims.n_u32, 64)?;
    let underfill = grid_below_waves(
        plain_slim_blocks,
        multiprocessor_count,
        policy.thin_split_wave_numerator,
        policy.thin_split_wave_denominator,
    )?;
    if (32..=1024).contains(&batch)
        && (64..=2048).contains(&n_out)
        && n_out.is_multiple_of(4)
        && n_in >= 33
        && !n_in.is_multiple_of(32)
        && underfill
    {
        let k_tail = n_in % 32;
        let k_main = n_in - k_tail;
        if k_main >= 32
            && checked_mul3(k_main / 32, batch, n_out, "NN K-tail scratch")? <= SPLITK_SCRATCH_CAP
        {
            return Ok(ScalarDispatchPlan::NnSplitKThinTail { k_main, k_tail });
        }
    }
    if (32..=1024).contains(&batch)
        && (64..=2048).contains(&n_out)
        && n_out.is_multiple_of(4)
        && n_in >= 32
        && n_in.is_multiple_of(32)
        && checked_mul3(n_in / 32, batch, n_out, "NN split-K scratch")? <= SPLITK_SCRATCH_CAP
        && underfill
    {
        return Ok(ScalarDispatchPlan::NnSplitKThin);
    }
    if batch > 1024
        && (128..=GEMM_SLIM_MAX).contains(&n_out)
        && n_in >= 64
        && n_in.is_multiple_of(32)
    {
        let chunks = dims.k_u32.div_ceil(64);
        if chunks >= 6
            && checked_mul3(
                checked_usize(chunks, "NN slim split-K chunks")?,
                batch,
                n_out,
                "NN slim split-K scratch",
            )? <= SPLITK_SCRATCH_CAP
            && plain_slim_blocks > 0
            && grid_below_waves(
                plain_slim_blocks,
                multiprocessor_count,
                policy.slim_split_wave_numerator,
                policy.slim_split_wave_denominator,
            )?
        {
            return Ok(ScalarDispatchPlan::NnSplitKSlim { chunks });
        }
    }
    if batch < 128 && n_out >= 128 {
        return Ok(ScalarDispatchPlan::NnNarrow);
    }
    if batch >= GEMM_CUSTOM_MIN && n_out >= GEMM_CUSTOM_MIN {
        return Ok(ScalarDispatchPlan::NnFinal {
            slim: slim_final(batch, n_out),
        });
    }
    Err(format!(
        "UNCOVERED scalar NN route M={batch} K={n_in} N={n_out}"
    ))
}

pub(in crate::mamba_ssm::gpu) fn tc_half_policy_prefers_scalar_forward(
    compute_capability: (u32, u32),
    dims: (usize, usize, usize),
    multiprocessor_count: u32,
) -> Result<bool, String> {
    let policy = super::super::kernel_identity::Sm80TcPolicy::current();
    if compute_capability != policy.deep_split_k_compute_capability
        || dims.2 != policy.deep_split_k_output_columns
    {
        return Ok(false);
    }
    let plan = scalar_nn_plan(GemmDims::nn(dims, dims.1)?, multiprocessor_count)?;
    Ok(match plan {
        ScalarDispatchPlan::NnSplitKThinTail { .. } => {
            dims.1 >= policy.deep_split_k_tail_min_reduction
        }
        ScalarDispatchPlan::NnSplitKThin => dims.1 >= policy.deep_split_k_aligned_min_reduction,
        _ => false,
    })
}

fn scalar_tn_plan(dims: GemmDims, multiprocessor_count: u32) -> Result<ScalarDispatchPlan, String> {
    scalar_wave_policy(multiprocessor_count)?;
    let (batch, n_in, n_out) = dims.tuple();
    if n_out == 1 && n_in >= 4 && batch >= 32 {
        return Ok(ScalarDispatchPlan::TnGemv);
    }
    if (2..=127).contains(&n_out) {
        return Ok(ScalarDispatchPlan::TnNarrow);
    }
    if let Some((m_chunk, chunks)) = splitm_tn_partition(batch, n_in, n_out, multiprocessor_count)?
    {
        return Ok(ScalarDispatchPlan::TnSplitM { m_chunk, chunks });
    }
    if n_in >= 1 && n_out >= GEMM_CUSTOM_MIN {
        return Ok(ScalarDispatchPlan::TnFinal {
            slim: slim_final(n_in, n_out),
        });
    }
    Err(format!(
        "UNCOVERED scalar TN route M={batch} K={n_in} N={n_out}"
    ))
}

fn scalar_nt_plan(dims: GemmDims, multiprocessor_count: u32) -> Result<ScalarDispatchPlan, String> {
    let policy = scalar_wave_policy(multiprocessor_count)?;
    let (batch, n_in, n_out) = dims.tuple();
    if (2..=127).contains(&n_out) {
        return Ok(ScalarDispatchPlan::NtNarrow);
    }
    if n_out == 1 {
        return Ok(ScalarDispatchPlan::NtGemv);
    }
    if matches!((batch, n_in, n_out), (512, 16, 2048) | (16, 512, 2048)) {
        return Ok(ScalarDispatchPlan::NtSplitKMain {
            n_main: 2048,
            n_tail: 0,
        });
    }
    let plain_slim_blocks = checked_tile_grid(dims.m_u32, 128, dims.k_u32, 64)?;
    let underfill = grid_below_waves(
        plain_slim_blocks,
        multiprocessor_count,
        policy.thin_split_wave_numerator,
        policy.thin_split_wave_denominator,
    )?;
    if batch < 32 && n_out >= 128 {
        return Ok(ScalarDispatchPlan::NtSmallBatchWide);
    }
    if (32..=1024).contains(&batch)
        && (64..=4096).contains(&n_in)
        && n_in >= 33
        && !n_in.is_multiple_of(32)
        && (32..=2048).contains(&n_out)
        && n_out.is_multiple_of(32)
        && underfill
    {
        let k_tail = n_in % 32;
        let k_main = n_in - k_tail;
        if k_main >= 32
            && k_main.checked_mul(n_out).is_some_and(|elements| {
                elements <= super::contract::SCALAR_GENERIC_TRANSPOSE_ROUTE_CAP_ELEMENTS
            })
            && checked_mul3(n_out / 32, batch, k_main, "NT K-tail scratch")? <= SPLITK_SCRATCH_CAP
        {
            return Ok(ScalarDispatchPlan::NtSplitKTail { k_main, k_tail });
        }
    }
    let n_tail = n_out % 32;
    let n_main = n_out - n_tail;
    if (32..=1024).contains(&batch)
        && (64..=4096).contains(&n_in)
        && n_in.is_multiple_of(32)
        && (32..=2048).contains(&n_out)
        && n_main >= 32
        && dims.kn <= super::contract::SCALAR_GENERIC_TRANSPOSE_ROUTE_CAP_ELEMENTS
        && checked_mul3(n_main / 32, batch, n_in, "NT split-K scratch")? <= SPLITK_SCRATCH_CAP
        && underfill
    {
        return Ok(ScalarDispatchPlan::NtSplitKMain { n_main, n_tail });
    }
    if batch > 1024
        && (128..=GEMM_SLIM_NT_NIN_MAX).contains(&n_in)
        && n_out >= 64
        && n_out.is_multiple_of(32)
        && dims.kn <= super::contract::SCALAR_GENERIC_TRANSPOSE_ROUTE_CAP_ELEMENTS
    {
        let chunks = dims.n_u32.div_ceil(64);
        if chunks >= 2
            && checked_mul3(
                checked_usize(chunks, "NT slim split-K chunks")?,
                batch,
                n_in,
                "NT slim split-K scratch",
            )? <= SPLITK_SCRATCH_CAP
            && plain_slim_blocks > 0
            && grid_below_waves(
                plain_slim_blocks,
                multiprocessor_count,
                policy.slim_split_wave_numerator,
                policy.slim_split_wave_denominator,
            )?
        {
            return Ok(ScalarDispatchPlan::NtSplitKSlim { chunks });
        }
    }
    if (32..128).contains(&batch) && n_out >= 128 {
        return Ok(ScalarDispatchPlan::NtMidBatchWide);
    }
    if batch >= GEMM_CUSTOM_MIN {
        return Ok(ScalarDispatchPlan::NtFinal {
            slim: slim_final(batch, n_in),
        });
    }
    Err(format!(
        "UNCOVERED scalar NT route M={batch} K={n_in} N={n_out}"
    ))
}

pub(super) fn scalar_dispatch_plan(
    request: F32TriadRequest,
    multiprocessor_count: u32,
) -> Result<ScalarDispatchPlan, String> {
    request.shape.validate(request.op)?;
    scalar_wave_policy(multiprocessor_count)?;
    let dims = (request.shape.m, request.shape.k, request.shape.n);
    match request.op {
        crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp::Nn => {
            scalar_nn_plan(GemmDims::nn(dims, request.shape.lda)?, multiprocessor_count)
        }
        crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp::Tn => {
            scalar_tn_plan(GemmDims::tn(dims)?, multiprocessor_count)
        }
        crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp::Nt => {
            scalar_nt_plan(GemmDims::nt(dims)?, multiprocessor_count)
        }
    }
}

pub(super) fn nn_routes_to_big(
    batch: usize,
    n_in: usize,
    n_out: usize,
    multiprocessor_count: u32,
) -> bool {
    GemmDims::nn((batch, n_in, n_out), n_in)
        .and_then(|dims| scalar_nn_plan(dims, multiprocessor_count))
        .is_ok_and(|plan| matches!(plan, ScalarDispatchPlan::NnFinal { slim: false }))
}

pub(super) fn tn_routes_to_big(
    batch: usize,
    n_in: usize,
    n_out: usize,
    multiprocessor_count: u32,
) -> bool {
    GemmDims::tn((batch, n_in, n_out))
        .and_then(|dims| scalar_tn_plan(dims, multiprocessor_count))
        .is_ok_and(|plan| matches!(plan, ScalarDispatchPlan::TnFinal { slim: false }))
}

pub(super) fn nt_routes_to_big(
    batch: usize,
    n_in: usize,
    n_out: usize,
    multiprocessor_count: u32,
) -> bool {
    GemmDims::nt((batch, n_in, n_out))
        .and_then(|dims| scalar_nt_plan(dims, multiprocessor_count))
        .is_ok_and(|plan| matches!(plan, ScalarDispatchPlan::NtFinal { slim: false }))
}

#[cfg(test)]
mod scalar_wave_policy_tests {
    use super::{
        FIXED_COPYPLAN_EVIDENCE_COHORTS, FixedCopyPlanQualificationIdentity,
        NT_FIXED_COPYPLAN_COMPOSED_EVIDENCE_COHORTS,
        NT_FIXED_COPYPLAN_COMPOSED_QUALIFICATION_CANDIDATES,
        NtFixedCopyPlanComposedQualificationIdentity, ScalarDispatchPlan, ScalarLaunchFacts,
        ScalarTransposeQualificationIdentity, TN_M16N16_SPLITM16_SM89_BINDINGS,
        qualified_fixed_copyplan_environment, qualified_nt_fixed_copyplan_sibling_environment,
        qualified_scalar_sm89_ada142_tn_environment, scalar_dispatch_plan, scalar_launch_plan,
        scalar_proof_plan,
    };
    use crate::mamba_ssm::gpu::gemm_bi_triad::{F32TriadOperands, F32TriadRequest, F32TriadShape};
    use crate::mamba_ssm::gpu::kernel_identity::{
        ArtifactIdentity, ArtifactKind, COMPILER_REVISION, COMPOSER_REVISION, CompilerIdentity,
        CudaTarget, ModuleKind, NUMERIC_ABI_REVISION, ResolvedGemmOp, SCHEDULE_REVISION,
    };

    const RTX_6000_ADA_SMS: u32 = 142;

    #[derive(Clone, Copy)]
    struct MeasuredComposedRow {
        nvrtc: (i32, i32),
        fixed_source: &'static str,
        fixed_key: &'static str,
        fixed_artifact: &'static str,
        fixed_header: &'static str,
        scalar_key: &'static str,
        scalar_artifact: &'static str,
        scalar_header: &'static str,
        library: &'static str,
    }

    fn digest(hex: &str) -> [u8; 32] {
        assert_eq!(hex.len(), 64);
        std::array::from_fn(|index| u8::from_str_radix(&hex[index * 2..index * 2 + 2], 16).unwrap())
    }

    fn measured_composed_identities() -> [NtFixedCopyPlanComposedQualificationIdentity; 6] {
        const SCALAR_SOURCE: &str =
            "4b11048f9cbfff712cfc4ed1ec3823d87aa3b54dcc3c6f070b60be315c29ec11";
        let rows = [
            MeasuredComposedRow {
                nvrtc: (12, 8),
                fixed_source: "e46af1e315d323db6abf5ad17569367150b63205a9f75173aaf8081db45e0e90",
                fixed_key: "282654a63f7610e3e77af8dc1abcb13ded34395aedfdb226726b9b9dd942f3f6",
                fixed_artifact: "30536bd091dded36df4fbee464f09795fdd66a7a3c872e6cb3cec0b44005f2d3",
                fixed_header: "295a1cf86db172a6655821a9f628f64bb13903e7da7f13af15f38542299d19ac",
                scalar_key: "5b3f65b82c695e9b714dc7aad2638dc013ec926475671486c5dd210a33ffca98",
                scalar_artifact: "f1694f11b58e080ecac495b9b03b1603d6dd406594c7cc4317b38b8ddfeb507d",
                scalar_header: "916caa15c20ea76b8e14e9530f19ae0316cecff531a3a7932b25df60d62be78e",
                library: "26b0a3a02044ffcbc1693fd83e9261beffa692a4fbcfe3ac5e9d8c87980bb155",
            },
            MeasuredComposedRow {
                nvrtc: (12, 8),
                fixed_source: "73111d550cd33778cba12f27eb38cc8b30e417b40b6469f3f36ddc3815e0dcab",
                fixed_key: "e89d08e6f5c657c90fbd852fc3b52e90d3feb7095be6093a2998915d3f39c4db",
                fixed_artifact: "c73394a72691a9d5411b57733945f7fe0d6dc4c6ec3585d657a10b64f18e7784",
                fixed_header: "5f48da53e800d422f11c63c1256ee27dd2061f197e3bcd5b728f8512e44ac599",
                scalar_key: "5b3f65b82c695e9b714dc7aad2638dc013ec926475671486c5dd210a33ffca98",
                scalar_artifact: "f1694f11b58e080ecac495b9b03b1603d6dd406594c7cc4317b38b8ddfeb507d",
                scalar_header: "916caa15c20ea76b8e14e9530f19ae0316cecff531a3a7932b25df60d62be78e",
                library: "26b0a3a02044ffcbc1693fd83e9261beffa692a4fbcfe3ac5e9d8c87980bb155",
            },
            MeasuredComposedRow {
                nvrtc: (13, 0),
                fixed_source: "e46af1e315d323db6abf5ad17569367150b63205a9f75173aaf8081db45e0e90",
                fixed_key: "967191e863d46c5117d86ca71d933a259cb52b66fde2a889a927889e0d025799",
                fixed_artifact: "1d91d36540f32f6c8916c694eb75e0be8c70fab3592970b0ab45781eab2513a0",
                fixed_header: "41b43208b8dc3170422e357e10cb2ee0ca084b0a93efe01ff1d4e3c39fd33cc2",
                scalar_key: "1c630b754b1116d70abbe296b6af1e17af8207aa0c6222eff12438b0527788b4",
                scalar_artifact: "256397474edf247352e4ad4945b926c0c11413bdc9d36b48f0a60c2f5f099ff6",
                scalar_header: "7cb5647f30201e76c432980ccef1476af91edc9cdbc27954bfe98bd22016ef1f",
                library: "709b91c36bfb0ed966ee69adc8d6f87ff110eecf3dfb5060367f183ce614eb0d",
            },
            MeasuredComposedRow {
                nvrtc: (13, 0),
                fixed_source: "73111d550cd33778cba12f27eb38cc8b30e417b40b6469f3f36ddc3815e0dcab",
                fixed_key: "3e6ac45e967eec817ca7a3ea707998ce94ab3ce7ce4fd35c9e99670ad10c32d9",
                fixed_artifact: "9efa64a93f96d87aa2cebef1cb9e7c420bb9e90c05c6d892c0a6da78c8ab98c4",
                fixed_header: "b78320a6a4164d73d7e0f06528797361d884d564a8326491bcbc788faad2129e",
                scalar_key: "1c630b754b1116d70abbe296b6af1e17af8207aa0c6222eff12438b0527788b4",
                scalar_artifact: "256397474edf247352e4ad4945b926c0c11413bdc9d36b48f0a60c2f5f099ff6",
                scalar_header: "7cb5647f30201e76c432980ccef1476af91edc9cdbc27954bfe98bd22016ef1f",
                library: "709b91c36bfb0ed966ee69adc8d6f87ff110eecf3dfb5060367f183ce614eb0d",
            },
            MeasuredComposedRow {
                nvrtc: (13, 2),
                fixed_source: "e46af1e315d323db6abf5ad17569367150b63205a9f75173aaf8081db45e0e90",
                fixed_key: "40eab522feb529a219cf48bb260931aea2ca79e3ef0b8ca7dead893ae1df3f18",
                fixed_artifact: "83d09c71605c62dda65908bbb6ff1cede9955b136ba4f0b918d9044d4d1e45e4",
                fixed_header: "9c252669c2a99bd2806def4a4126aaa43e9607f3a4d26b709e3c9f093d7af436",
                scalar_key: "47c5d79e89d9b33153d427e666f081ede55eae7e8d150d3610a12117dba77dbd",
                scalar_artifact: "6ba718467ed4a024cc8f7902a43774878cde335d21341364977856a1e8a37452",
                scalar_header: "fa701f8fb4901f0baa3cb4d504acc78decfe29248f9bf66349d3cf46035a382c",
                library: "d031a53eb97235b70f62f652932db1bdf728ea229c8ca809d53c5ffd91642687",
            },
            MeasuredComposedRow {
                nvrtc: (13, 2),
                fixed_source: "73111d550cd33778cba12f27eb38cc8b30e417b40b6469f3f36ddc3815e0dcab",
                fixed_key: "f452666b5fa9d778b389a60dd4140e82fba2b7ad9b115375ec8f78d51e56c9da",
                fixed_artifact: "399baf5d13de2abf55c6b7ce55b061096892a36e9b4e13ba2052fb04699a5b73",
                fixed_header: "f5cc41fa052dc07c004efc252c5765266d97a3410efb4ef9a6e59f36be5b544f",
                scalar_key: "47c5d79e89d9b33153d427e666f081ede55eae7e8d150d3610a12117dba77dbd",
                scalar_artifact: "6ba718467ed4a024cc8f7902a43774878cde335d21341364977856a1e8a37452",
                scalar_header: "fa701f8fb4901f0baa3cb4d504acc78decfe29248f9bf66349d3cf46035a382c",
                library: "d031a53eb97235b70f62f652932db1bdf728ea229c8ca809d53c5ffd91642687",
            },
        ];
        rows.map(|row| NtFixedCopyPlanComposedQualificationIdentity {
            scalar: ScalarTransposeQualificationIdentity {
                nvrtc_version: row.nvrtc,
                compile_key: digest(row.scalar_key),
                artifact_digest: digest(row.scalar_artifact),
                source_digest: digest(SCALAR_SOURCE),
                header_manifest_digest: digest(row.scalar_header),
                nvrtc_library_domain: digest(row.library),
            },
            fixed: FixedCopyPlanQualificationIdentity {
                nvrtc_version: row.nvrtc,
                compile_key: digest(row.fixed_key),
                artifact_digest: digest(row.fixed_artifact),
                source_digest: digest(row.fixed_source),
                header_manifest_digest: digest(row.fixed_header),
                nvrtc_library_domain: digest(row.library),
            },
        })
    }

    fn facts_for_measured_pair(
        identity: NtFixedCopyPlanComposedQualificationIdentity,
    ) -> ScalarLaunchFacts {
        let compiler = |source_digest,
                        invocation_digest,
                        header_manifest_digest,
                        nvrtc_version,
                        nvrtc_library_domain| CompilerIdentity {
            source_digest,
            invocation_digest,
            header_manifest_digest,
            target: CudaTarget::new("sm_89").unwrap(),
            nvrtc_version,
            nvrtc_library_domain,
            nvrtc_library_known: true,
            output_kind: ArtifactKind::Ptx,
            composer_revision: COMPOSER_REVISION,
            compiler_revision: COMPILER_REVISION,
            numeric_abi_revision: NUMERIC_ABI_REVISION,
            schedule_revision: SCHEDULE_REVISION,
        };
        ScalarLaunchFacts {
            scalar_artifact: ArtifactIdentity {
                module_kind: ModuleKind::TriadScalar,
                artifact_kind: ArtifactKind::Ptx,
                compile_key: identity.scalar.compile_key,
                artifact_digest: identity.scalar.artifact_digest,
            },
            scalar_compiler: compiler(
                identity.scalar.source_digest,
                identity.scalar.compile_key,
                identity.scalar.header_manifest_digest,
                identity.scalar.nvrtc_version,
                identity.scalar.nvrtc_library_domain,
            ),
            fixed_artifact: ArtifactIdentity {
                module_kind: ModuleKind::Fixed,
                artifact_kind: ArtifactKind::Ptx,
                compile_key: identity.fixed.compile_key,
                artifact_digest: identity.fixed.artifact_digest,
            },
            fixed_compiler: compiler(
                identity.fixed.source_digest,
                identity.fixed.compile_key,
                identity.fixed.header_manifest_digest,
                identity.fixed.nvrtc_version,
                identity.fixed.nvrtc_library_domain,
            ),
            fixed_copyplan_loaded: true,
            sm89_exact_f32_artifact: None,
            sm89_exact_f32_compiler: None,
            sm89_exact_f32_symbols_loaded: [false; 3],
            sm89_exact_f32_d128_artifact: None,
            sm89_exact_f32_d128_compiler: None,
            sm89_exact_f32_d128_symbols_loaded: [false; 2],
            compute_capability: (8, 9),
            multiprocessor_count: RTX_6000_ADA_SMS,
        }
    }

    fn plan(
        op: ResolvedGemmOp,
        dims: (usize, usize, usize),
        multiprocessor_count: u32,
    ) -> Result<ScalarDispatchPlan, String> {
        scalar_dispatch_plan(
            F32TriadRequest {
                op,
                shape: F32TriadShape::contiguous(op, dims),
            },
            multiprocessor_count,
        )
    }

    fn tn_admission_compiler(
        target: &str,
        nvrtc_version: (i32, i32),
        library_domain: [u8; 32],
        library_known: bool,
    ) -> CompilerIdentity {
        CompilerIdentity {
            source_digest: [1; 32],
            invocation_digest: [2; 32],
            header_manifest_digest: [3; 32],
            target: CudaTarget::new(target).unwrap(),
            nvrtc_version,
            nvrtc_library_domain: library_domain,
            nvrtc_library_known: library_known,
            output_kind: ArtifactKind::Ptx,
            composer_revision: COMPOSER_REVISION,
            compiler_revision: COMPILER_REVISION,
            numeric_abi_revision: NUMERIC_ABI_REVISION,
            schedule_revision: SCHEDULE_REVISION,
        }
    }

    fn tn_admission_facts() -> ScalarLaunchFacts {
        let compiler = tn_admission_compiler("compute_120", (13, 2), [4; 32], true);
        ScalarLaunchFacts {
            scalar_artifact: ArtifactIdentity {
                module_kind: ModuleKind::TriadScalar,
                artifact_kind: ArtifactKind::Ptx,
                compile_key: compiler.invocation_digest,
                artifact_digest: [6; 32],
            },
            scalar_compiler: compiler,
            fixed_artifact: ArtifactIdentity {
                module_kind: ModuleKind::Fixed,
                artifact_kind: ArtifactKind::Ptx,
                compile_key: [0; 32],
                artifact_digest: [0; 32],
            },
            fixed_compiler: compiler,
            fixed_copyplan_loaded: false,
            sm89_exact_f32_artifact: None,
            sm89_exact_f32_compiler: None,
            sm89_exact_f32_symbols_loaded: [false; 3],
            sm89_exact_f32_d128_artifact: None,
            sm89_exact_f32_d128_compiler: None,
            sm89_exact_f32_d128_symbols_loaded: [false; 2],
            compute_capability: (12, 0),
            multiprocessor_count: 170,
        }
    }

    fn tn_admission_request(dims: (usize, usize, usize)) -> F32TriadRequest {
        F32TriadRequest {
            op: ResolvedGemmOp::Tn,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Tn, dims),
        }
    }

    fn tn_admission_operands() -> F32TriadOperands {
        F32TriadOperands {
            output: 0x3000,
            a: 0x1000,
            b: 0x2000,
            bias: None,
            alpha: 1.0,
            beta: 1.0,
        }
    }

    fn admitted_tn_plan(
        facts: ScalarLaunchFacts,
        request: F32TriadRequest,
        operands: F32TriadOperands,
    ) -> ScalarDispatchPlan {
        scalar_launch_plan(facts, request, operands).unwrap()
    }

    fn nn_qualified_request(dims: (usize, usize, usize)) -> F32TriadRequest {
        F32TriadRequest {
            op: ResolvedGemmOp::Nn,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nn, dims),
        }
    }

    fn nn_qualified_operands() -> F32TriadOperands {
        F32TriadOperands {
            output: 0x3000,
            a: 0x1000,
            b: 0x2000,
            bias: None,
            alpha: 1.0,
            beta: 0.0,
        }
    }

    fn fixed_copyplan_facts(index: usize) -> ScalarLaunchFacts {
        let identity = FIXED_COPYPLAN_EVIDENCE_COHORTS[index];
        let compiler = CompilerIdentity {
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
        };
        ScalarLaunchFacts {
            scalar_artifact: ArtifactIdentity {
                module_kind: ModuleKind::TriadScalar,
                artifact_kind: ArtifactKind::Ptx,
                compile_key: [8; 32],
                artifact_digest: [9; 32],
            },
            scalar_compiler: CompilerIdentity {
                invocation_digest: [8; 32],
                ..compiler
            },
            fixed_artifact: ArtifactIdentity {
                module_kind: ModuleKind::Fixed,
                artifact_kind: ArtifactKind::Ptx,
                compile_key: identity.compile_key,
                artifact_digest: identity.artifact_digest,
            },
            fixed_compiler: compiler,
            fixed_copyplan_loaded: true,
            sm89_exact_f32_artifact: None,
            sm89_exact_f32_compiler: None,
            sm89_exact_f32_symbols_loaded: [false; 3],
            sm89_exact_f32_d128_artifact: None,
            sm89_exact_f32_d128_compiler: None,
            sm89_exact_f32_d128_symbols_loaded: [false; 2],
            compute_capability: (8, 9),
            multiprocessor_count: RTX_6000_ADA_SMS,
        }
    }

    fn nt_fixed_copyplan_sibling_facts(index: usize) -> ScalarLaunchFacts {
        let identity = NT_FIXED_COPYPLAN_COMPOSED_QUALIFICATION_CANDIDATES[index];
        let mut facts = fixed_copyplan_facts(index);
        facts.scalar_artifact = ArtifactIdentity {
            module_kind: ModuleKind::TriadScalar,
            artifact_kind: ArtifactKind::Ptx,
            compile_key: identity.scalar.compile_key,
            artifact_digest: identity.scalar.artifact_digest,
        };
        facts.scalar_compiler = CompilerIdentity {
            source_digest: identity.scalar.source_digest,
            invocation_digest: identity.scalar.compile_key,
            header_manifest_digest: identity.scalar.header_manifest_digest,
            target: CudaTarget::new("sm_89").unwrap(),
            nvrtc_version: identity.scalar.nvrtc_version,
            nvrtc_library_domain: identity.scalar.nvrtc_library_domain,
            nvrtc_library_known: true,
            output_kind: ArtifactKind::Ptx,
            composer_revision: COMPOSER_REVISION,
            compiler_revision: COMPILER_REVISION,
            numeric_abi_revision: NUMERIC_ABI_REVISION,
            schedule_revision: SCHEDULE_REVISION,
        };
        facts
    }

    #[test]
    fn scalar_dispatch_rejects_zero_multiprocessors() {
        for op in [ResolvedGemmOp::Nn, ResolvedGemmOp::Tn, ResolvedGemmOp::Nt] {
            let error = plan(op, (128, 128, 128), 0)
                .expect_err("scalar dispatch requires a nonzero multiprocessor count");
            assert!(error.contains("multiprocessor count"), "{error}");
        }
    }

    #[test]
    fn nn_thin_split_admission_uses_the_live_multiprocessor_count() {
        let dims = (512, 64, 2048);
        assert_eq!(
            plan(ResolvedGemmOp::Nn, dims, 108).unwrap(),
            ScalarDispatchPlan::NnFinal { slim: false }
        );
        assert_eq!(
            plan(ResolvedGemmOp::Nn, dims, RTX_6000_ADA_SMS).unwrap(),
            ScalarDispatchPlan::NnSplitKThin
        );
    }

    #[test]
    fn nt_split_admission_uses_the_live_multiprocessor_count() {
        let dims = (512, 2048, 128);
        assert_eq!(
            plan(ResolvedGemmOp::Nt, dims, 108).unwrap(),
            ScalarDispatchPlan::NtFinal { slim: false }
        );
        assert_eq!(
            plan(ResolvedGemmOp::Nt, dims, RTX_6000_ADA_SMS).unwrap(),
            ScalarDispatchPlan::NtSplitKMain {
                n_main: 128,
                n_tail: 0,
            }
        );
    }

    #[test]
    fn nt_generic_split_admission_keeps_the_pre_large_deep_scratch_boundary() {
        assert_eq!(
            plan(ResolvedGemmOp::Nt, (32, 2_304, 2_048), 170).unwrap(),
            ScalarDispatchPlan::NtMidBatchWide
        );
    }

    #[test]
    fn nt_split_admission_promotes_the_two_qualified_thin_output_cells() {
        for multiprocessor_count in [20, 56, 82, 108, 120, 128, 132, 142, 148, 170] {
            for dims in [(512, 16, 2048), (16, 512, 2048)] {
                assert_eq!(
                    plan(ResolvedGemmOp::Nt, dims, multiprocessor_count).unwrap(),
                    ScalarDispatchPlan::NtSplitKMain {
                        n_main: 2048,
                        n_tail: 0,
                    },
                    "{dims:?} {multiprocessor_count} SMs"
                );
            }
        }

        for (dims, expected) in [
            ((512, 17, 2048), ScalarDispatchPlan::NtFinal { slim: true }),
            ((17, 512, 2048), ScalarDispatchPlan::NtSmallBatchWide),
            ((512, 16, 2016), ScalarDispatchPlan::NtFinal { slim: true }),
            ((16, 512, 2016), ScalarDispatchPlan::NtSmallBatchWide),
        ] {
            assert_eq!(
                plan(ResolvedGemmOp::Nt, dims, 170).unwrap(),
                expected,
                "neighbor {dims:?}"
            );
        }
    }

    #[test]
    fn nt_m2n16_exact_cell_requires_shape_operands_and_qualified_environment() {
        let facts = tn_admission_facts();
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nt,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, (512, 16, 2_048)),
        };
        let operands = nn_qualified_operands();
        assert_eq!(
            scalar_launch_plan(facts, request, operands).unwrap(),
            ScalarDispatchPlan::NtM2N16SplitK32Qualified
        );
        for dims in [
            (511, 16, 2_048),
            (513, 16, 2_048),
            (512, 15, 2_048),
            (512, 17, 2_048),
            (512, 16, 2_047),
            (512, 16, 2_049),
        ] {
            let neighbor = F32TriadRequest {
                op: ResolvedGemmOp::Nt,
                shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, dims),
            };
            assert_ne!(
                scalar_launch_plan(facts, neighbor, operands).unwrap(),
                ScalarDispatchPlan::NtM2N16SplitK32Qualified,
                "neighbor {dims:?}"
            );
        }
        for mutation in [
            F32TriadOperands {
                alpha: -1.0,
                ..operands
            },
            F32TriadOperands {
                beta: 1.0,
                ..operands
            },
            F32TriadOperands {
                bias: Some(0x4000),
                ..operands
            },
            F32TriadOperands {
                output: 0,
                ..operands
            },
            F32TriadOperands {
                a: operands.a + 4,
                ..operands
            },
            F32TriadOperands {
                b: operands.b + 4,
                ..operands
            },
        ] {
            assert_ne!(
                scalar_launch_plan(facts, request, mutation).unwrap(),
                ScalarDispatchPlan::NtM2N16SplitK32Qualified
            );
        }
        let mut mutations = [facts; 3];
        mutations[0].compute_capability = (8, 9);
        mutations[1].multiprocessor_count = 169;
        mutations[2].scalar_compiler.nvrtc_version = (13, 0);
        for mutation in mutations {
            assert_ne!(
                scalar_launch_plan(mutation, request, operands).unwrap(),
                ScalarDispatchPlan::NtM2N16SplitK32Qualified
            );
        }
    }

    #[test]
    fn nn_m32n64_splitk32_exact_cell_is_fail_closed() {
        let facts = tn_admission_facts();
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nn,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nn, (128, 8_192, 128)),
        };
        let operands = nn_qualified_operands();
        let fallback = scalar_dispatch_plan(request, facts.multiprocessor_count).unwrap();
        assert_eq!(fallback, ScalarDispatchPlan::NnSplitKThin);
        assert_eq!(
            scalar_launch_plan(facts, request, operands).unwrap(),
            ScalarDispatchPlan::NnM32N64SplitK32Qualified
        );

        for dims in [
            (127, 8_192, 128),
            (129, 8_192, 128),
            (128, 8_191, 128),
            (128, 8_193, 128),
            (128, 8_192, 127),
            (128, 8_192, 129),
        ] {
            let neighbor = F32TriadRequest {
                op: ResolvedGemmOp::Nn,
                shape: F32TriadShape::contiguous(ResolvedGemmOp::Nn, dims),
            };
            assert_ne!(
                scalar_launch_plan(facts, neighbor, operands).unwrap(),
                ScalarDispatchPlan::NnM32N64SplitK32Qualified,
                "neighbor {dims:?}"
            );
        }
        for shape in [
            F32TriadShape {
                lda: request.shape.lda + 1,
                ..request.shape
            },
            F32TriadShape {
                ldb: request.shape.ldb + 1,
                ..request.shape
            },
            F32TriadShape {
                ldc: request.shape.ldc + 1,
                ..request.shape
            },
        ] {
            let strided = F32TriadRequest {
                op: ResolvedGemmOp::Nn,
                shape,
            };
            assert_ne!(
                scalar_launch_plan(facts, strided, operands).unwrap(),
                ScalarDispatchPlan::NnM32N64SplitK32Qualified
            );
        }
        for op in [ResolvedGemmOp::Tn, ResolvedGemmOp::Nt] {
            let wrong_op = F32TriadRequest {
                op,
                shape: F32TriadShape::contiguous(op, (128, 8_192, 128)),
            };
            assert_ne!(
                scalar_launch_plan(facts, wrong_op, operands).unwrap(),
                ScalarDispatchPlan::NnM32N64SplitK32Qualified
            );
        }
        for mutation in [
            F32TriadOperands {
                alpha: -1.0,
                ..operands
            },
            F32TriadOperands {
                beta: -0.0,
                ..operands
            },
            F32TriadOperands {
                bias: Some(0x4000),
                ..operands
            },
            F32TriadOperands {
                output: 0,
                ..operands
            },
            F32TriadOperands { a: 0, ..operands },
            F32TriadOperands { b: 0, ..operands },
            F32TriadOperands {
                output: operands.output + 4,
                ..operands
            },
            F32TriadOperands {
                a: operands.a + 4,
                ..operands
            },
            F32TriadOperands {
                b: operands.b + 4,
                ..operands
            },
        ] {
            assert_eq!(
                scalar_launch_plan(facts, request, mutation).unwrap(),
                fallback
            );
        }

        let mut mutations = [facts; 13];
        mutations[0].compute_capability = (8, 9);
        mutations[1].compute_capability = (12, 1);
        mutations[2].multiprocessor_count = 169;
        mutations[3].scalar_compiler.nvrtc_version = (13, 0);
        mutations[4].scalar_compiler.target = CudaTarget::new("compute_121").unwrap();
        mutations[5].scalar_compiler.nvrtc_library_known = false;
        mutations[6].scalar_compiler.nvrtc_library_domain = [0; 32];
        mutations[7].scalar_artifact.compile_key[0] ^= 1;
        mutations[8].scalar_artifact.artifact_digest = [0; 32];
        mutations[9].scalar_artifact.module_kind = ModuleKind::TriadSm80;
        mutations[10].scalar_compiler.invocation_digest = [0; 32];
        mutations[11].scalar_artifact.artifact_kind = ArtifactKind::Cubin;
        mutations[12].scalar_compiler.output_kind = ArtifactKind::Cubin;
        for mutation in mutations {
            assert_eq!(
                scalar_launch_plan(mutation, request, operands).unwrap(),
                fallback
            );
        }
    }

    #[test]
    fn tn_split_m_partition_targets_two_live_waves() {
        let dims = (4096, 128, 128);
        assert_eq!(
            plan(ResolvedGemmOp::Tn, dims, 108).unwrap(),
            ScalarDispatchPlan::TnSplitM {
                m_chunk: 32,
                chunks: 128,
            }
        );
        assert_eq!(
            plan(ResolvedGemmOp::Tn, dims, RTX_6000_ADA_SMS).unwrap(),
            ScalarDispatchPlan::TnSplitM {
                m_chunk: 16,
                chunks: 256,
            }
        );
    }

    #[test]
    fn tn_small_input_splitm_gap_is_covered_by_the_generic_member() {
        let plan = plan(ResolvedGemmOp::Tn, (4096, 24, 768), RTX_6000_ADA_SMS).unwrap();
        assert!(matches!(
            plan,
            ScalarDispatchPlan::TnSplitM {
                m_chunk: 96,
                chunks: 43,
            }
        ));
    }

    #[test]
    fn tn_narrow_split_m_selector_admits_only_the_three_exact_cells() {
        for (dims, expected) in [
            ((1024, 47, 17), (32, 32)),
            ((1024, 128, 25), (32, 32)),
            ((4096, 64, 64), (48, 86)),
        ] {
            assert_eq!(
                admitted_tn_plan(
                    tn_admission_facts(),
                    tn_admission_request(dims),
                    tn_admission_operands(),
                ),
                ScalarDispatchPlan::TnNarrowSplitM {
                    m_chunk: expected.0,
                    chunks: expected.1,
                },
                "{dims:?}"
            );
        }

        let request = tn_admission_request((1024, 47, 17));
        let operands = tn_admission_operands();
        for admitted in [
            F32TriadOperands {
                output: operands.output + 4,
                ..operands
            },
            F32TriadOperands {
                a: operands.a + 4,
                ..operands
            },
            F32TriadOperands {
                b: operands.b + 4,
                ..operands
            },
            F32TriadOperands {
                a: operands.a + 4,
                b: operands.b + 4,
                ..operands
            },
        ] {
            assert_eq!(
                admitted_tn_plan(tn_admission_facts(), request, admitted),
                ScalarDispatchPlan::TnNarrowSplitM {
                    m_chunk: 32,
                    chunks: 32,
                }
            );
        }

        for dims in [
            (256, 32, 2),
            (4096, 128, 96),
            (4096, 256, 101),
            (4111, 257, 127),
        ] {
            assert_eq!(
                admitted_tn_plan(
                    tn_admission_facts(),
                    tn_admission_request(dims),
                    tn_admission_operands(),
                ),
                ScalarDispatchPlan::TnNarrow,
                "rejected tournament cell {dims:?}"
            );
        }
    }

    #[test]
    fn tn_narrow_split_m_selector_admits_from_independent_scalar_facts() {
        assert_eq!(
            admitted_tn_plan(
                tn_admission_facts(),
                tn_admission_request((1024, 47, 17)),
                tn_admission_operands(),
            ),
            ScalarDispatchPlan::TnNarrowSplitM {
                m_chunk: 32,
                chunks: 32,
            }
        );
    }

    #[test]
    fn tn_narrow_split_m_selector_requires_the_exact_device_and_compiler_domain() {
        let request = tn_admission_request((1024, 47, 17));
        let operands = tn_admission_operands();
        let fallback = ScalarDispatchPlan::TnNarrow;

        let mut wrong_arch = tn_admission_facts();
        wrong_arch.compute_capability = (12, 1);
        assert_eq!(admitted_tn_plan(wrong_arch, request, operands), fallback);

        let mut wrong_sm_count = tn_admission_facts();
        wrong_sm_count.multiprocessor_count = 169;
        assert_eq!(
            admitted_tn_plan(wrong_sm_count, request, operands),
            fallback
        );

        let mut wrong_nvrtc = tn_admission_facts();
        wrong_nvrtc.scalar_compiler.nvrtc_version = (13, 1);
        assert_eq!(admitted_tn_plan(wrong_nvrtc, request, operands), fallback);

        let mut unknown_library = tn_admission_facts();
        unknown_library.scalar_compiler.nvrtc_library_known = false;
        assert_eq!(
            admitted_tn_plan(unknown_library, request, operands),
            fallback
        );

        let mut wrong_compile_key = tn_admission_facts();
        wrong_compile_key.scalar_artifact.compile_key = [8; 32];
        assert_eq!(
            admitted_tn_plan(wrong_compile_key, request, operands),
            fallback
        );

        let mut wrong_target = tn_admission_facts();
        wrong_target.scalar_compiler.target = CudaTarget::new("compute_121").unwrap();
        assert_eq!(admitted_tn_plan(wrong_target, request, operands), fallback);
    }

    #[test]
    fn tn_narrow_split_m_selector_rejects_neighboring_requests_and_epilogues() {
        let facts = tn_admission_facts();
        let request = tn_admission_request((1024, 47, 17));
        let operands = tn_admission_operands();
        let fallback = ScalarDispatchPlan::TnNarrow;

        let nn_request = F32TriadRequest {
            op: ResolvedGemmOp::Nn,
            shape: request.shape,
        };
        assert_eq!(
            admitted_tn_plan(facts, nn_request, operands),
            ScalarDispatchPlan::NnNarrow
        );

        for dims in [(1023, 47, 17), (1024, 48, 17), (1024, 47, 18)] {
            assert_eq!(
                admitted_tn_plan(facts, tn_admission_request(dims), operands),
                fallback,
                "shape neighbor {dims:?}"
            );
        }
        for shape in [
            F32TriadShape {
                lda: request.shape.lda + 1,
                ..request.shape
            },
            F32TriadShape {
                ldb: request.shape.ldb + 1,
                ..request.shape
            },
            F32TriadShape {
                ldc: request.shape.ldc + 1,
                ..request.shape
            },
        ] {
            assert_eq!(
                admitted_tn_plan(
                    facts,
                    F32TriadRequest {
                        op: ResolvedGemmOp::Tn,
                        shape,
                    },
                    operands,
                ),
                fallback,
                "stride neighbor {shape:?}"
            );
        }
        for rejected in [
            F32TriadOperands {
                bias: Some(0x4000),
                ..operands
            },
            F32TriadOperands {
                alpha: f32::from_bits(1.0_f32.to_bits() + 1),
                ..operands
            },
            F32TriadOperands {
                beta: 0.0,
                ..operands
            },
            F32TriadOperands {
                output: 0,
                ..operands
            },
            F32TriadOperands {
                output: operands.output + 2,
                ..operands
            },
            F32TriadOperands { a: 0, ..operands },
            F32TriadOperands {
                a: operands.a + 2,
                ..operands
            },
            F32TriadOperands { b: 0, ..operands },
            F32TriadOperands {
                b: operands.b + 2,
                ..operands
            },
        ] {
            assert_eq!(admitted_tn_plan(facts, request, rejected), fallback);
        }
    }

    #[test]
    fn nn_m64n64_selector_admits_only_the_seven_measured_cells() {
        let facts = tn_admission_facts();
        let operands = nn_qualified_operands();
        let cells = [
            (2_048, 3_072, 768),
            (4_096, 3_072, 1_536),
            (512, 3_072, 768),
            (4_096, 512, 768),
            (2_048, 768, 3_072),
            (2_048, 1_536, 768),
            (4_621, 384, 1_928),
        ];

        for dims in cells {
            let request = nn_qualified_request(dims);
            let plan = scalar_launch_plan(facts, request, operands).unwrap();
            assert_eq!(format!("{plan:?}"), "NnM64N64Qualified", "cell {dims:?}");

            let dimensions = [dims.0, dims.1, dims.2];
            for axis in 0..3 {
                for delta in [-1_isize, 1] {
                    let mut neighbor = dimensions;
                    neighbor[axis] = neighbor[axis].checked_add_signed(delta).unwrap();
                    let neighbor = (neighbor[0], neighbor[1], neighbor[2]);
                    let plan = scalar_launch_plan(facts, nn_qualified_request(neighbor), operands)
                        .unwrap();
                    assert_ne!(
                        format!("{plan:?}"),
                        "NnM64N64Qualified",
                        "shape neighbor {neighbor:?}"
                    );
                }
            }

            for shape in [
                F32TriadShape {
                    lda: request.shape.lda + 1,
                    ..request.shape
                },
                F32TriadShape {
                    ldb: request.shape.ldb + 1,
                    ..request.shape
                },
                F32TriadShape {
                    ldc: request.shape.ldc + 1,
                    ..request.shape
                },
            ] {
                let plan = scalar_launch_plan(
                    facts,
                    F32TriadRequest {
                        op: ResolvedGemmOp::Nn,
                        shape,
                    },
                    operands,
                )
                .unwrap();
                assert_ne!(
                    format!("{plan:?}"),
                    "NnM64N64Qualified",
                    "stride mutation {shape:?}"
                );
            }
        }
    }

    #[test]
    fn nn_fixed_copyplan_selector_admits_only_the_exact_measured_cells() {
        let operands = nn_qualified_operands();
        let cells = [
            (2_048, 768, 3_072),
            (2_048, 1_536, 768),
            (4_621, 384, 1_928),
            (4_096, 3_072, 1_536),
        ];
        for cohort in 0..FIXED_COPYPLAN_EVIDENCE_COHORTS.len() {
            let facts = fixed_copyplan_facts(cohort);
            for dims in cells {
                let plan = scalar_launch_plan(facts, nn_qualified_request(dims), operands).unwrap();
                assert_eq!(
                    format!("{plan:?}"),
                    "NnSm89FixedCopyPlanQualified",
                    "cohort {cohort} cell {dims:?}"
                );
            }
        }
    }

    #[test]
    fn nn_fixed_copyplan_selector_fails_closed_to_the_prior_plan() {
        const SELECTED: &str = "NnSm89FixedCopyPlanQualified";
        let facts = fixed_copyplan_facts(2);
        let operands = nn_qualified_operands();
        let request = nn_qualified_request((2_048, 1_536, 768));
        let fallback = scalar_dispatch_plan(request, facts.multiprocessor_count).unwrap();
        assert_eq!(fallback, ScalarDispatchPlan::NnFinal { slim: false });
        let transposed_unmeasured = nn_qualified_request((2_048, 3_072, 768));
        assert_ne!(
            format!(
                "{:?}",
                scalar_launch_plan(facts, transposed_unmeasured, operands).unwrap()
            ),
            SELECTED,
            "the dimension-swapped shape was not measured"
        );

        let dimensions = [request.shape.m, request.shape.k, request.shape.n];
        for axis in 0..3 {
            for delta in [-1_isize, 1] {
                let mut neighbor = dimensions;
                neighbor[axis] = neighbor[axis].checked_add_signed(delta).unwrap();
                let neighbor = (neighbor[0], neighbor[1], neighbor[2]);
                assert_ne!(
                    format!(
                        "{:?}",
                        scalar_launch_plan(facts, nn_qualified_request(neighbor), operands)
                            .unwrap()
                    ),
                    SELECTED,
                    "shape neighbor {neighbor:?}"
                );
            }
        }
        for shape in [
            F32TriadShape {
                lda: request.shape.lda + 1,
                ..request.shape
            },
            F32TriadShape {
                ldb: request.shape.ldb + 1,
                ..request.shape
            },
            F32TriadShape {
                ldc: request.shape.ldc + 1,
                ..request.shape
            },
        ] {
            assert_ne!(
                format!(
                    "{:?}",
                    scalar_launch_plan(
                        facts,
                        F32TriadRequest {
                            op: ResolvedGemmOp::Nn,
                            shape,
                        },
                        operands,
                    )
                    .unwrap()
                ),
                SELECTED,
                "stride mutation {shape:?}"
            );
        }
        for op in [ResolvedGemmOp::Tn, ResolvedGemmOp::Nt] {
            let request = F32TriadRequest {
                op,
                shape: F32TriadShape::contiguous(op, (2_048, 1_536, 768)),
            };
            assert_ne!(
                format!(
                    "{:?}",
                    scalar_launch_plan(facts, request, operands).unwrap()
                ),
                SELECTED,
                "op {op:?}"
            );
        }
        for mutation in [
            F32TriadOperands {
                alpha: -1.0,
                ..operands
            },
            F32TriadOperands {
                beta: -0.0,
                ..operands
            },
            F32TriadOperands {
                beta: 1.0,
                ..operands
            },
            F32TriadOperands {
                bias: Some(0x4000),
                ..operands
            },
            F32TriadOperands {
                output: 0,
                ..operands
            },
            F32TriadOperands {
                output: operands.output + 4,
                ..operands
            },
            F32TriadOperands { a: 0, ..operands },
            F32TriadOperands {
                a: operands.a + 4,
                ..operands
            },
            F32TriadOperands { b: 0, ..operands },
            F32TriadOperands {
                b: operands.b + 4,
                ..operands
            },
        ] {
            assert_eq!(
                scalar_launch_plan(facts, request, mutation).unwrap(),
                fallback,
                "operand mutation {mutation:?}"
            );
        }

        let mut mutations = Vec::new();
        let mut mutation = facts;
        mutation.compute_capability = (9, 0);
        mutations.push(mutation);
        mutation = facts;
        mutation.multiprocessor_count = 141;
        mutations.push(mutation);
        mutation = facts;
        mutation.fixed_copyplan_loaded = false;
        mutations.push(mutation);
        mutation = facts;
        mutation.fixed_compiler.nvrtc_version = (13, 1);
        mutations.push(mutation);
        mutation = facts;
        mutation.fixed_compiler.source_digest[0] ^= 1;
        mutations.push(mutation);
        mutation = facts;
        mutation.fixed_compiler.invocation_digest[0] ^= 1;
        mutations.push(mutation);
        mutation = facts;
        mutation.fixed_compiler.header_manifest_digest[0] ^= 1;
        mutations.push(mutation);
        mutation = facts;
        mutation.fixed_compiler.nvrtc_library_domain[0] ^= 1;
        mutations.push(mutation);
        mutation = facts;
        mutation.fixed_artifact.compile_key[0] ^= 1;
        mutations.push(mutation);
        mutation = facts;
        mutation.fixed_artifact.artifact_digest[0] ^= 1;
        mutations.push(mutation);
        mutation = facts;
        mutation.fixed_artifact.module_kind = ModuleKind::TriadScalar;
        mutations.push(mutation);
        for mutation in mutations {
            assert_eq!(
                scalar_launch_plan(mutation, request, operands).unwrap(),
                fallback,
                "identity mutation {mutation:?}"
            );
        }
    }

    #[test]
    fn nt_d768_transpose_m64n64_selector_is_an_exact_cell() {
        let facts = tn_admission_facts();
        let operands = nn_qualified_operands();
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nt,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, (2_048, 768, 3_072)),
        };
        assert_eq!(
            scalar_launch_plan(facts, request, operands).unwrap(),
            ScalarDispatchPlan::NtD768TransposeM64N64Qualified
        );
        assert!(ScalarDispatchPlan::NtD768TransposeM64N64Qualified.needs_transpose_scratch());
        assert!(!ScalarDispatchPlan::NtD768TransposeM64N64Qualified.needs_split_scratch());

        let dimensions = [request.shape.m, request.shape.k, request.shape.n];
        for axis in 0..3 {
            for delta in [-1_isize, 1] {
                let mut neighbor = dimensions;
                neighbor[axis] = neighbor[axis].checked_add_signed(delta).unwrap();
                let neighbor = F32TriadRequest {
                    op: ResolvedGemmOp::Nt,
                    shape: F32TriadShape::contiguous(
                        ResolvedGemmOp::Nt,
                        (neighbor[0], neighbor[1], neighbor[2]),
                    ),
                };
                assert_ne!(
                    scalar_launch_plan(facts, neighbor, operands).unwrap(),
                    ScalarDispatchPlan::NtD768TransposeM64N64Qualified
                );
            }
        }
        for shape in [
            F32TriadShape {
                lda: request.shape.lda + 1,
                ..request.shape
            },
            F32TriadShape {
                ldb: request.shape.ldb + 1,
                ..request.shape
            },
            F32TriadShape {
                ldc: request.shape.ldc + 1,
                ..request.shape
            },
        ] {
            let mutated = F32TriadRequest {
                op: ResolvedGemmOp::Nt,
                shape,
            };
            assert_ne!(
                scalar_launch_plan(facts, mutated, operands).unwrap(),
                ScalarDispatchPlan::NtD768TransposeM64N64Qualified
            );
        }
        for op in [ResolvedGemmOp::Nn, ResolvedGemmOp::Tn] {
            let mutated = F32TriadRequest {
                op,
                shape: F32TriadShape::contiguous(op, (2_048, 768, 3_072)),
            };
            assert_ne!(
                scalar_launch_plan(facts, mutated, operands).unwrap(),
                ScalarDispatchPlan::NtD768TransposeM64N64Qualified
            );
        }
    }

    #[test]
    fn nt_d768_out_transpose_m64n64_selector_is_a_distinct_exact_cell() {
        let facts = tn_admission_facts();
        let operands = nn_qualified_operands();
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nt,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, (2_048, 1_536, 768)),
        };
        assert_eq!(
            scalar_launch_plan(facts, request, operands).unwrap(),
            ScalarDispatchPlan::NtD768OutTransposeM64N64Qualified
        );
        assert_ne!(
            ScalarDispatchPlan::NtD768OutTransposeM64N64Qualified,
            ScalarDispatchPlan::NtD768TransposeM64N64Qualified
        );
        assert!(ScalarDispatchPlan::NtD768OutTransposeM64N64Qualified.needs_transpose_scratch());
        assert!(!ScalarDispatchPlan::NtD768OutTransposeM64N64Qualified.needs_split_scratch());

        let dimensions = [request.shape.m, request.shape.k, request.shape.n];
        for axis in 0..3 {
            for delta in [-1_isize, 1] {
                let mut neighbor = dimensions;
                neighbor[axis] = neighbor[axis].checked_add_signed(delta).unwrap();
                let neighbor = F32TriadRequest {
                    op: ResolvedGemmOp::Nt,
                    shape: F32TriadShape::contiguous(
                        ResolvedGemmOp::Nt,
                        (neighbor[0], neighbor[1], neighbor[2]),
                    ),
                };
                assert_ne!(
                    scalar_launch_plan(facts, neighbor, operands).unwrap(),
                    ScalarDispatchPlan::NtD768OutTransposeM64N64Qualified
                );
            }
        }
        for shape in [
            F32TriadShape {
                lda: request.shape.lda + 1,
                ..request.shape
            },
            F32TriadShape {
                ldb: request.shape.ldb + 1,
                ..request.shape
            },
            F32TriadShape {
                ldc: request.shape.ldc + 1,
                ..request.shape
            },
        ] {
            assert_ne!(
                scalar_launch_plan(
                    facts,
                    F32TriadRequest {
                        op: ResolvedGemmOp::Nt,
                        shape,
                    },
                    operands,
                )
                .unwrap(),
                ScalarDispatchPlan::NtD768OutTransposeM64N64Qualified
            );
        }
        for op in [ResolvedGemmOp::Nn, ResolvedGemmOp::Tn] {
            let mutated = F32TriadRequest {
                op,
                shape: F32TriadShape::contiguous(op, (2_048, 1_536, 768)),
            };
            assert_ne!(
                scalar_launch_plan(facts, mutated, operands).unwrap(),
                ScalarDispatchPlan::NtD768OutTransposeM64N64Qualified
            );
        }
    }

    #[test]
    fn other_sm80_boards_reach_the_fixed_copyplan_routes_through_the_proof_tier() {
        let ada = fixed_copyplan_facts(2);
        let operands = nn_qualified_operands();
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nt,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, (2_048, 1_536, 768)),
        };
        let selected = ScalarDispatchPlan::NtD768OutSm89FixedCopyPlanQualified;
        assert_eq!(
            scalar_proof_plan(ada, request, operands).unwrap(),
            Some(selected)
        );
        for cc in [(8, 0), (8, 6), (8, 7), (9, 0), (10, 0), (11, 0)] {
            let board = ScalarLaunchFacts {
                compute_capability: cc,
                ..ada
            };
            assert_eq!(
                scalar_launch_plan(board, request, operands).unwrap(),
                ScalarDispatchPlan::NtFinal { slim: false },
                "no frozen evidence on {cc:?}"
            );
            assert_eq!(
                scalar_proof_plan(board, request, operands).unwrap(),
                Some(selected),
                "the overlay route is the proof candidate on {cc:?}"
            );
        }
        for cc in [(7, 5), (12, 0), (12, 1)] {
            let board = ScalarLaunchFacts {
                compute_capability: cc,
                ..ada
            };
            assert_eq!(
                scalar_proof_plan(board, request, operands).unwrap(),
                None,
                "the overlay is not composed on {cc:?}"
            );
        }
        let unloaded = ScalarLaunchFacts {
            compute_capability: (9, 0),
            fixed_copyplan_loaded: false,
            ..ada
        };
        assert_eq!(
            scalar_proof_plan(unloaded, request, operands).unwrap(),
            None
        );
    }

    #[test]
    fn nt_d768_out_fixed_copyplan_selector_is_one_exact_measured_cell() {
        let facts = fixed_copyplan_facts(2);
        let operands = nn_qualified_operands();
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nt,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, (2_048, 1_536, 768)),
        };
        let selected = ScalarDispatchPlan::NtD768OutSm89FixedCopyPlanQualified;
        assert_eq!(
            scalar_dispatch_plan(request, facts.multiprocessor_count).unwrap(),
            ScalarDispatchPlan::NtFinal { slim: false }
        );
        assert_eq!(
            scalar_launch_plan(facts, request, operands).unwrap(),
            selected
        );
        assert!(selected.needs_transpose_scratch());
        assert!(!selected.needs_split_scratch());
        assert_eq!(request.shape.k * request.shape.n, 1_179_648);

        for dims in [
            (2_047, 1_536, 768),
            (2_049, 1_536, 768),
            (2_048, 1_535, 768),
            (2_048, 1_537, 768),
            (2_048, 1_536, 767),
            (2_048, 1_536, 769),
        ] {
            assert_ne!(
                scalar_launch_plan(
                    facts,
                    F32TriadRequest {
                        op: ResolvedGemmOp::Nt,
                        shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, dims),
                    },
                    operands,
                )
                .unwrap(),
                selected,
                "neighbor {dims:?}"
            );
        }
        for cohort in 0..FIXED_COPYPLAN_EVIDENCE_COHORTS.len() {
            assert_eq!(
                scalar_launch_plan(fixed_copyplan_facts(cohort), request, operands).unwrap(),
                selected,
                "cohort {cohort}"
            );
        }
        let mut absent = facts;
        absent.fixed_copyplan_loaded = false;
        assert_eq!(
            scalar_launch_plan(absent, request, operands).unwrap(),
            ScalarDispatchPlan::NtFinal { slim: false }
        );
    }

    #[test]
    fn nt_fixed_copyplan_sibling_selector_admits_only_all_six_measured_cohorts() {
        assert_eq!(
            NT_FIXED_COPYPLAN_COMPOSED_EVIDENCE_COHORTS,
            NT_FIXED_COPYPLAN_COMPOSED_QUALIFICATION_CANDIDATES,
            "admission must contain exactly the six measured composed candidates"
        );
    }

    #[test]
    fn triad_retained_identity_fixed_scalar_pairs_match_the_six_measured_cohorts() {
        let expected = measured_composed_identities();
        assert_eq!(
            NT_FIXED_COPYPLAN_COMPOSED_QUALIFICATION_CANDIDATES.len(),
            6,
            "the six capacity-specific measured pairs must all be staged"
        );
        assert_eq!(FIXED_COPYPLAN_EVIDENCE_COHORTS.len(), 6);

        for (index, identity) in expected.iter().copied().enumerate() {
            let facts = facts_for_measured_pair(identity);
            assert!(
                qualified_fixed_copyplan_environment(facts),
                "Fixed cohort {index}"
            );
            assert!(
                qualified_nt_fixed_copyplan_sibling_environment(facts),
                "scalar+Fixed cohort {index}"
            );
            assert!(
                qualified_scalar_sm89_ada142_tn_environment(facts),
                "scalar projection for cohort {index}"
            );

            let mut changed = facts;
            changed.scalar_compiler.nvrtc_version.1 += 1;
            changed.fixed_compiler.nvrtc_version.1 += 1;
            assert!(!qualified_nt_fixed_copyplan_sibling_environment(changed));
            let mut changed = facts;
            changed.fixed_compiler.source_digest[0] ^= 1;
            assert!(!qualified_fixed_copyplan_environment(changed));
            assert!(!qualified_nt_fixed_copyplan_sibling_environment(changed));
            let mut changed = facts;
            changed.scalar_artifact.compile_key[0] ^= 1;
            assert!(!qualified_nt_fixed_copyplan_sibling_environment(changed));
            let mut changed = facts;
            changed.fixed_artifact.artifact_digest[0] ^= 1;
            assert!(!qualified_fixed_copyplan_environment(changed));
            assert!(!qualified_nt_fixed_copyplan_sibling_environment(changed));

            let other_toolkit = expected[(index + 2) % expected.len()].scalar;
            let mut crossed = facts;
            crossed.scalar_artifact.compile_key = other_toolkit.compile_key;
            crossed.scalar_artifact.artifact_digest = other_toolkit.artifact_digest;
            crossed.scalar_compiler.source_digest = other_toolkit.source_digest;
            crossed.scalar_compiler.invocation_digest = other_toolkit.compile_key;
            crossed.scalar_compiler.header_manifest_digest = other_toolkit.header_manifest_digest;
            crossed.scalar_compiler.nvrtc_version = other_toolkit.nvrtc_version;
            crossed.scalar_compiler.nvrtc_library_domain = other_toolkit.nvrtc_library_domain;
            assert!(qualified_fixed_copyplan_environment(crossed));
            assert!(!qualified_nt_fixed_copyplan_sibling_environment(crossed));
        }

        assert_eq!(
            NT_FIXED_COPYPLAN_COMPOSED_QUALIFICATION_CANDIDATES,
            expected.as_slice()
        );
        assert_eq!(
            NT_FIXED_COPYPLAN_COMPOSED_QUALIFICATION_CANDIDATES
                .iter()
                .map(|identity| identity.scalar.compile_key)
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            3
        );
        assert_eq!(
            TN_M16N16_SPLITM16_SM89_BINDINGS,
            [expected[0].scalar, expected[2].scalar, expected[4].scalar]
        );
    }

    #[test]
    fn nt_fixed_copyplan_sibling_selector_admits_both_retained_cells() {
        let operands = nn_qualified_operands();
        let cells = [
            (
                (2_048, 768, 3_072),
                ScalarDispatchPlan::NtD768InSm89FixedCopyPlanQualified,
                ScalarDispatchPlan::NtFinal { slim: false },
            ),
            (
                (4_621, 384, 1_928),
                ScalarDispatchPlan::NtPrismSm89FixedCopyPlanQualified,
                ScalarDispatchPlan::NtFinal { slim: true },
            ),
        ];
        for (cohort, candidate) in NT_FIXED_COPYPLAN_COMPOSED_QUALIFICATION_CANDIDATES
            .iter()
            .enumerate()
        {
            let facts = nt_fixed_copyplan_sibling_facts(cohort);
            assert!(
                candidate.matches(facts),
                "cohort {cohort} candidate identity"
            );
            for (dims, selected, fallback) in cells {
                let request = F32TriadRequest {
                    op: ResolvedGemmOp::Nt,
                    shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, dims),
                };
                assert_eq!(
                    scalar_dispatch_plan(request, facts.multiprocessor_count).unwrap(),
                    fallback,
                    "cohort {cohort} cell {dims:?} prior plan"
                );
                assert_eq!(
                    scalar_launch_plan(facts, request, operands).unwrap(),
                    selected,
                    "cohort {cohort} cell {dims:?} retained route"
                );
            }
        }
    }

    #[test]
    fn nt_fixed_copyplan_sibling_admission_does_not_change_existing_nn_or_d768_out_routes() {
        let operands = nn_qualified_operands();
        for cohort in 0..NT_FIXED_COPYPLAN_COMPOSED_QUALIFICATION_CANDIDATES.len() {
            let facts = nt_fixed_copyplan_sibling_facts(cohort);
            let nn = F32TriadRequest {
                op: ResolvedGemmOp::Nn,
                shape: F32TriadShape::contiguous(ResolvedGemmOp::Nn, (2_048, 768, 3_072)),
            };
            assert_eq!(
                scalar_launch_plan(facts, nn, operands).unwrap(),
                ScalarDispatchPlan::NnSm89FixedCopyPlanQualified,
                "cohort {cohort} NN"
            );
            let d768_out = F32TriadRequest {
                op: ResolvedGemmOp::Nt,
                shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, (2_048, 1_536, 768)),
            };
            assert_eq!(
                scalar_launch_plan(facts, d768_out, operands).unwrap(),
                ScalarDispatchPlan::NtD768OutSm89FixedCopyPlanQualified,
                "cohort {cohort} d768-out"
            );
        }
    }

    #[test]
    fn triad_retained_scalar_ada_tn_m16n16_is_exact_and_fixed_holder_independent() {
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Tn,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Tn, (256, 512, 384)),
        };
        let operands = F32TriadOperands {
            beta: 1.0,
            ..nn_qualified_operands()
        };
        for cohort in [0, 2, 4] {
            let mut facts = nt_fixed_copyplan_sibling_facts(cohort);
            facts.fixed_copyplan_loaded = false;
            facts.fixed_artifact.compile_key = [0; 32];
            facts.fixed_artifact.artifact_digest = [0; 32];
            assert_eq!(
                scalar_launch_plan(facts, request, operands).unwrap(),
                ScalarDispatchPlan::TnM16N16SplitM16Qualified,
                "Ada scalar cohort {cohort}"
            );
        }

        let fallback = scalar_dispatch_plan(request, RTX_6000_ADA_SMS).unwrap();
        assert_eq!(
            fallback,
            ScalarDispatchPlan::TnSplitM {
                m_chunk: 16,
                chunks: 16,
            }
        );
        let facts = nt_fixed_copyplan_sibling_facts(2);
        for dims in [(257, 512, 384), (256, 513, 384), (256, 512, 385)] {
            assert_ne!(
                scalar_launch_plan(
                    facts,
                    F32TriadRequest {
                        op: ResolvedGemmOp::Tn,
                        shape: F32TriadShape::contiguous(ResolvedGemmOp::Tn, dims),
                    },
                    operands,
                )
                .unwrap(),
                ScalarDispatchPlan::TnM16N16SplitM16Qualified
            );
        }
        for shape in [
            F32TriadShape {
                lda: request.shape.lda + 1,
                ..request.shape
            },
            F32TriadShape {
                ldb: request.shape.ldb + 1,
                ..request.shape
            },
            F32TriadShape {
                ldc: request.shape.ldc + 1,
                ..request.shape
            },
        ] {
            assert_ne!(
                scalar_launch_plan(
                    facts,
                    F32TriadRequest {
                        op: ResolvedGemmOp::Tn,
                        shape,
                    },
                    operands,
                )
                .unwrap(),
                ScalarDispatchPlan::TnM16N16SplitM16Qualified
            );
        }
        for shape in [
            F32TriadShape {
                lda: request.shape.lda - 1,
                ..request.shape
            },
            F32TriadShape {
                ldb: request.shape.ldb - 1,
                ..request.shape
            },
            F32TriadShape {
                ldc: request.shape.ldc - 1,
                ..request.shape
            },
        ] {
            assert!(
                scalar_launch_plan(
                    facts,
                    F32TriadRequest {
                        op: ResolvedGemmOp::Tn,
                        shape,
                    },
                    operands,
                )
                .is_err()
            );
        }
        for op in [ResolvedGemmOp::Nn, ResolvedGemmOp::Nt] {
            let wrong_op = F32TriadRequest {
                op,
                shape: F32TriadShape::contiguous(op, (256, 512, 384)),
            };
            assert_ne!(
                scalar_launch_plan(facts, wrong_op, operands).unwrap(),
                ScalarDispatchPlan::TnM16N16SplitM16Qualified
            );
        }
        for mutation in [
            F32TriadOperands {
                alpha: f32::from_bits(1.0_f32.to_bits() + 1),
                ..operands
            },
            F32TriadOperands {
                beta: -0.0,
                ..operands
            },
            F32TriadOperands {
                bias: Some(0x4000),
                ..operands
            },
            F32TriadOperands {
                output: 0,
                ..operands
            },
            F32TriadOperands { a: 0, ..operands },
            F32TriadOperands { b: 0, ..operands },
            F32TriadOperands {
                output: operands.output + 4,
                ..operands
            },
            F32TriadOperands {
                a: operands.a + 4,
                ..operands
            },
            F32TriadOperands {
                b: operands.b + 4,
                ..operands
            },
        ] {
            assert_ne!(
                scalar_launch_plan(facts, request, mutation).unwrap(),
                ScalarDispatchPlan::TnM16N16SplitM16Qualified
            );
        }
        let mut identity_mutations = Vec::new();
        let mut mutation = facts;
        mutation.compute_capability = (8, 8);
        identity_mutations.push(mutation);
        mutation = facts;
        mutation.multiprocessor_count = 141;
        identity_mutations.push(mutation);
        mutation = facts;
        mutation.scalar_artifact.artifact_digest[0] ^= 1;
        identity_mutations.push(mutation);
        mutation = facts;
        mutation.scalar_compiler.header_manifest_digest[0] ^= 1;
        identity_mutations.push(mutation);
        for mutation in identity_mutations {
            assert_ne!(
                scalar_launch_plan(mutation, request, operands).unwrap(),
                ScalarDispatchPlan::TnM16N16SplitM16Qualified
            );
        }
    }

    #[test]
    fn triad_retained_scalar_ada_large_deep_requires_the_exact_composed_pair() {
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nt,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, (4_096, 3_072, 1_536)),
        };
        let operands = nn_qualified_operands();
        let fallback = scalar_dispatch_plan(request, RTX_6000_ADA_SMS).unwrap();
        assert_eq!(fallback, ScalarDispatchPlan::NtFinal { slim: false });
        for cohort in 0..NT_FIXED_COPYPLAN_COMPOSED_QUALIFICATION_CANDIDATES.len() {
            assert_eq!(
                scalar_launch_plan(nt_fixed_copyplan_sibling_facts(cohort), request, operands,)
                    .unwrap(),
                ScalarDispatchPlan::NtLargeDeepSm89FixedCopyPlanQualified,
                "composed cohort {cohort}"
            );
        }

        let facts = nt_fixed_copyplan_sibling_facts(2);
        let other = nt_fixed_copyplan_sibling_facts(1);
        let mut pair_mismatch = facts;
        pair_mismatch.fixed_artifact = other.fixed_artifact;
        pair_mismatch.fixed_compiler = other.fixed_compiler;
        let mut holder_missing = facts;
        holder_missing.fixed_copyplan_loaded = false;
        for mutation in [pair_mismatch, holder_missing] {
            assert_eq!(
                scalar_launch_plan(mutation, request, operands).unwrap(),
                fallback,
                "composed identity or holder mutation {mutation:?}"
            );
        }

        for dims in [
            (4_097, 3_072, 1_536),
            (4_096, 3_073, 1_536),
            (4_096, 3_072, 1_537),
        ] {
            assert_ne!(
                scalar_launch_plan(
                    facts,
                    F32TriadRequest {
                        op: ResolvedGemmOp::Nt,
                        shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, dims),
                    },
                    operands,
                )
                .unwrap(),
                ScalarDispatchPlan::NtLargeDeepSm89FixedCopyPlanQualified
            );
        }
        for shape in [
            F32TriadShape {
                lda: request.shape.lda + 1,
                ..request.shape
            },
            F32TriadShape {
                ldb: request.shape.ldb + 1,
                ..request.shape
            },
            F32TriadShape {
                ldc: request.shape.ldc + 1,
                ..request.shape
            },
        ] {
            assert_ne!(
                scalar_launch_plan(
                    facts,
                    F32TriadRequest {
                        op: ResolvedGemmOp::Nt,
                        shape,
                    },
                    operands,
                )
                .unwrap(),
                ScalarDispatchPlan::NtLargeDeepSm89FixedCopyPlanQualified
            );
        }
        for shape in [
            F32TriadShape {
                lda: request.shape.lda - 1,
                ..request.shape
            },
            F32TriadShape {
                ldb: request.shape.ldb - 1,
                ..request.shape
            },
            F32TriadShape {
                ldc: request.shape.ldc - 1,
                ..request.shape
            },
        ] {
            assert!(
                scalar_launch_plan(
                    facts,
                    F32TriadRequest {
                        op: ResolvedGemmOp::Nt,
                        shape,
                    },
                    operands,
                )
                .is_err()
            );
        }
        for op in [ResolvedGemmOp::Nn, ResolvedGemmOp::Tn] {
            let wrong_op = F32TriadRequest {
                op,
                shape: F32TriadShape::contiguous(op, (4_096, 3_072, 1_536)),
            };
            assert_ne!(
                scalar_launch_plan(facts, wrong_op, operands).unwrap(),
                ScalarDispatchPlan::NtLargeDeepSm89FixedCopyPlanQualified
            );
        }
        for mutation in [
            F32TriadOperands {
                alpha: f32::from_bits(1.0_f32.to_bits() + 1),
                ..operands
            },
            F32TriadOperands {
                beta: -0.0,
                ..operands
            },
            F32TriadOperands {
                bias: Some(0x4000),
                ..operands
            },
            F32TriadOperands {
                output: 0,
                ..operands
            },
            F32TriadOperands { a: 0, ..operands },
            F32TriadOperands { b: 0, ..operands },
            F32TriadOperands {
                output: operands.output + 4,
                ..operands
            },
            F32TriadOperands {
                a: operands.a + 4,
                ..operands
            },
            F32TriadOperands {
                b: operands.b + 4,
                ..operands
            },
        ] {
            assert_ne!(
                scalar_launch_plan(facts, request, mutation).unwrap(),
                ScalarDispatchPlan::NtLargeDeepSm89FixedCopyPlanQualified
            );
        }
    }

    #[test]
    fn nt_fixed_copyplan_sibling_selector_fails_closed_on_contract_mutations() {
        let facts = nt_fixed_copyplan_sibling_facts(2);
        let operands = nn_qualified_operands();
        for (dims, fallback) in [
            (
                (2_048, 768, 3_072),
                ScalarDispatchPlan::NtFinal { slim: false },
            ),
            (
                (4_621, 384, 1_928),
                ScalarDispatchPlan::NtFinal { slim: true },
            ),
        ] {
            let request = F32TriadRequest {
                op: ResolvedGemmOp::Nt,
                shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, dims),
            };
            let dimensions = [request.shape.m, request.shape.k, request.shape.n];
            for axis in 0..3 {
                for delta in [-1_isize, 1] {
                    let mut neighbor = dimensions;
                    neighbor[axis] = neighbor[axis].checked_add_signed(delta).unwrap();
                    let neighbor = F32TriadRequest {
                        op: ResolvedGemmOp::Nt,
                        shape: F32TriadShape::contiguous(
                            ResolvedGemmOp::Nt,
                            (neighbor[0], neighbor[1], neighbor[2]),
                        ),
                    };
                    assert_eq!(
                        scalar_launch_plan(facts, neighbor, operands).unwrap(),
                        scalar_dispatch_plan(neighbor, facts.multiprocessor_count).unwrap(),
                        "shape neighbor {neighbor:?} must retain prior AUTO"
                    );
                }
            }
            for shape in [
                F32TriadShape {
                    lda: request.shape.lda + 1,
                    ..request.shape
                },
                F32TriadShape {
                    ldb: request.shape.ldb + 1,
                    ..request.shape
                },
                F32TriadShape {
                    ldc: request.shape.ldc + 1,
                    ..request.shape
                },
            ] {
                let mutated = F32TriadRequest {
                    op: ResolvedGemmOp::Nt,
                    shape,
                };
                assert_eq!(
                    scalar_launch_plan(facts, mutated, operands).unwrap(),
                    scalar_dispatch_plan(mutated, facts.multiprocessor_count).unwrap(),
                    "stride mutation {shape:?} must retain prior AUTO"
                );
            }
            for mutation in [
                F32TriadOperands {
                    alpha: -1.0,
                    ..operands
                },
                F32TriadOperands {
                    beta: -0.0,
                    ..operands
                },
                F32TriadOperands {
                    beta: 1.0,
                    ..operands
                },
                F32TriadOperands {
                    bias: Some(0x4000),
                    ..operands
                },
                F32TriadOperands {
                    output: 0,
                    ..operands
                },
                F32TriadOperands {
                    output: operands.output + 4,
                    ..operands
                },
                F32TriadOperands { a: 0, ..operands },
                F32TriadOperands {
                    a: operands.a + 4,
                    ..operands
                },
                F32TriadOperands { b: 0, ..operands },
                F32TriadOperands {
                    b: operands.b + 4,
                    ..operands
                },
            ] {
                assert_eq!(
                    scalar_launch_plan(facts, request, mutation).unwrap(),
                    fallback,
                    "operand mutation {mutation:?}"
                );
            }

            let mut environment_mutations = Vec::new();
            let mut mutation = facts;
            mutation.compute_capability = (9, 0);
            environment_mutations.push(mutation);
            mutation = facts;
            mutation.multiprocessor_count = 141;
            environment_mutations.push(mutation);
            mutation = facts;
            mutation.fixed_copyplan_loaded = false;
            environment_mutations.push(mutation);
            mutation = facts;
            mutation.fixed_compiler.nvrtc_version = (13, 1);
            environment_mutations.push(mutation);
            mutation = facts;
            mutation.fixed_compiler.source_digest[0] ^= 1;
            environment_mutations.push(mutation);
            mutation = facts;
            mutation.fixed_compiler.invocation_digest[0] ^= 1;
            environment_mutations.push(mutation);
            mutation = facts;
            mutation.fixed_compiler.header_manifest_digest[0] ^= 1;
            environment_mutations.push(mutation);
            mutation = facts;
            mutation.fixed_compiler.nvrtc_library_domain[0] ^= 1;
            environment_mutations.push(mutation);
            mutation = facts;
            mutation.fixed_artifact.compile_key[0] ^= 1;
            environment_mutations.push(mutation);
            mutation = facts;
            mutation.fixed_artifact.artifact_digest[0] ^= 1;
            environment_mutations.push(mutation);
            mutation = facts;
            mutation.fixed_artifact.module_kind = ModuleKind::TriadScalar;
            environment_mutations.push(mutation);
            mutation = facts;
            mutation.fixed_artifact.artifact_kind = ArtifactKind::Cubin;
            environment_mutations.push(mutation);
            mutation = facts;
            mutation.fixed_compiler.target = CudaTarget::new("compute_89").unwrap();
            environment_mutations.push(mutation);
            mutation = facts;
            mutation.fixed_compiler.nvrtc_library_known = false;
            environment_mutations.push(mutation);
            mutation = facts;
            mutation.fixed_compiler.output_kind = ArtifactKind::Cubin;
            environment_mutations.push(mutation);
            mutation = facts;
            mutation.scalar_artifact.module_kind = ModuleKind::TriadSm80;
            environment_mutations.push(mutation);
            mutation = facts;
            mutation.scalar_artifact.artifact_kind = ArtifactKind::Cubin;
            environment_mutations.push(mutation);
            mutation = facts;
            mutation.scalar_artifact.compile_key[0] ^= 1;
            environment_mutations.push(mutation);
            mutation = facts;
            mutation.scalar_artifact.artifact_digest[0] ^= 1;
            environment_mutations.push(mutation);
            mutation = facts;
            mutation.scalar_compiler.source_digest[0] ^= 1;
            environment_mutations.push(mutation);
            mutation = facts;
            mutation.scalar_compiler.invocation_digest[0] ^= 1;
            environment_mutations.push(mutation);
            mutation = facts;
            mutation.scalar_compiler.header_manifest_digest[0] ^= 1;
            environment_mutations.push(mutation);
            mutation = facts;
            mutation.scalar_compiler.nvrtc_library_domain[0] ^= 1;
            environment_mutations.push(mutation);
            mutation = facts;
            mutation.scalar_compiler.nvrtc_version = (13, 1);
            environment_mutations.push(mutation);
            mutation = facts;
            mutation.scalar_compiler.target = CudaTarget::new("compute_89").unwrap();
            environment_mutations.push(mutation);
            mutation = facts;
            mutation.scalar_compiler.nvrtc_library_known = false;
            environment_mutations.push(mutation);
            mutation = facts;
            mutation.scalar_compiler.output_kind = ArtifactKind::Cubin;
            environment_mutations.push(mutation);
            mutation = facts;
            mutation.scalar_compiler.composer_revision = COMPOSER_REVISION + 1;
            environment_mutations.push(mutation);
            mutation = facts;
            mutation.scalar_compiler.compiler_revision = COMPILER_REVISION + 1;
            environment_mutations.push(mutation);
            mutation = facts;
            mutation.scalar_compiler.numeric_abi_revision = NUMERIC_ABI_REVISION + 1;
            environment_mutations.push(mutation);
            mutation = facts;
            mutation.scalar_compiler.schedule_revision = SCHEDULE_REVISION + 1;
            environment_mutations.push(mutation);
            for mutation in environment_mutations {
                assert_eq!(
                    scalar_launch_plan(mutation, request, operands).unwrap(),
                    fallback,
                    "identity mutation {mutation:?}"
                );
            }
        }
    }

    #[test]
    fn nt_large_deep_transpose_m64n64_selector_is_an_exact_extent_cell() {
        let facts = tn_admission_facts();
        let operands = nn_qualified_operands();
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nt,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, (4_096, 3_072, 1_536)),
        };
        assert_eq!(
            request.shape.k.checked_mul(request.shape.n),
            Some(super::NT_LARGE_DEEP_TRANSPOSE_ELEMENTS)
        );
        const {
            assert!(
                super::NT_LARGE_DEEP_TRANSPOSE_ELEMENTS
                    <= super::super::contract::SCALAR_TRANSPOSE_SCRATCH_CAP_ELEMENTS
            )
        };
        assert_eq!(
            scalar_launch_plan(facts, request, operands).unwrap(),
            ScalarDispatchPlan::NtLargeDeepTransposeM64N64Qualified
        );
        assert!(ScalarDispatchPlan::NtLargeDeepTransposeM64N64Qualified.needs_transpose_scratch());
        assert!(!ScalarDispatchPlan::NtLargeDeepTransposeM64N64Qualified.needs_split_scratch());

        let dimensions = [request.shape.m, request.shape.k, request.shape.n];
        for axis in 0..3 {
            for delta in [-1_isize, 1] {
                let mut neighbor = dimensions;
                neighbor[axis] = neighbor[axis].checked_add_signed(delta).unwrap();
                let neighbor = F32TriadRequest {
                    op: ResolvedGemmOp::Nt,
                    shape: F32TriadShape::contiguous(
                        ResolvedGemmOp::Nt,
                        (neighbor[0], neighbor[1], neighbor[2]),
                    ),
                };
                assert_ne!(
                    scalar_launch_plan(facts, neighbor, operands).unwrap(),
                    ScalarDispatchPlan::NtLargeDeepTransposeM64N64Qualified
                );
            }
        }
        for shape in [
            F32TriadShape {
                lda: request.shape.lda + 1,
                ..request.shape
            },
            F32TriadShape {
                ldb: request.shape.ldb + 1,
                ..request.shape
            },
            F32TriadShape {
                ldc: request.shape.ldc + 1,
                ..request.shape
            },
        ] {
            assert_ne!(
                scalar_launch_plan(
                    facts,
                    F32TriadRequest {
                        op: ResolvedGemmOp::Nt,
                        shape,
                    },
                    operands,
                )
                .unwrap(),
                ScalarDispatchPlan::NtLargeDeepTransposeM64N64Qualified
            );
        }
        for op in [ResolvedGemmOp::Nn, ResolvedGemmOp::Tn] {
            let mutated = F32TriadRequest {
                op,
                shape: F32TriadShape::contiguous(op, (4_096, 3_072, 1_536)),
            };
            assert_ne!(
                scalar_launch_plan(facts, mutated, operands).unwrap(),
                ScalarDispatchPlan::NtLargeDeepTransposeM64N64Qualified
            );
        }
    }

    #[test]
    fn nt_large_deep_transpose_m64n64_rejects_environment_and_operand_mutations() {
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nt,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, (4_096, 3_072, 1_536)),
        };
        let operands = nn_qualified_operands();
        let qualified = ScalarDispatchPlan::NtLargeDeepTransposeM64N64Qualified;
        let mut fact_mutations = Vec::new();

        let mut facts = tn_admission_facts();
        facts.compute_capability = (12, 1);
        fact_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.multiprocessor_count = 169;
        fact_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_compiler.target = CudaTarget::new("compute_121").unwrap();
        fact_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_compiler.nvrtc_version = (13, 1);
        fact_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_compiler.nvrtc_library_known = false;
        fact_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_compiler.nvrtc_library_domain = [0; 32];
        fact_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_artifact.compile_key = [9; 32];
        fact_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_compiler.invocation_digest = [0; 32];
        facts.scalar_artifact.compile_key = [0; 32];
        fact_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_artifact.module_kind = ModuleKind::TriadSm80;
        fact_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_artifact.artifact_kind = ArtifactKind::Cubin;
        fact_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_artifact.artifact_digest = [0; 32];
        fact_mutations.push(facts);

        for facts in fact_mutations {
            assert_ne!(
                scalar_launch_plan(facts, request, operands).unwrap(),
                qualified
            );
        }
        for mutation in [
            F32TriadOperands {
                bias: Some(0x4000),
                ..operands
            },
            F32TriadOperands {
                alpha: f32::from_bits(1.0_f32.to_bits() + 1),
                ..operands
            },
            F32TriadOperands {
                beta: -0.0,
                ..operands
            },
            F32TriadOperands {
                output: 0,
                ..operands
            },
            F32TriadOperands { a: 0, ..operands },
            F32TriadOperands { b: 0, ..operands },
            F32TriadOperands {
                output: operands.output + 4,
                ..operands
            },
            F32TriadOperands {
                a: operands.a + 8,
                ..operands
            },
            F32TriadOperands {
                b: operands.b + 12,
                ..operands
            },
        ] {
            assert_ne!(
                scalar_launch_plan(tn_admission_facts(), request, mutation).unwrap(),
                qualified
            );
        }
    }

    #[test]
    fn nt_prism_vector_selector_is_an_exact_slim_cell() {
        let facts = tn_admission_facts();
        let operands = nn_qualified_operands();
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nt,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, (4_621, 384, 1_928)),
        };
        let qualified = ScalarDispatchPlan::NtPrismVectorQualified;
        assert_eq!(
            scalar_dispatch_plan(request, 170).unwrap(),
            ScalarDispatchPlan::NtFinal { slim: true }
        );
        assert_eq!(
            scalar_launch_plan(facts, request, operands).unwrap(),
            qualified
        );
        assert!(qualified.needs_transpose_scratch());
        assert!(!qualified.needs_split_scratch());

        let dimensions = [request.shape.m, request.shape.k, request.shape.n];
        for axis in 0..3 {
            for delta in [-1_isize, 1] {
                let mut neighbor = dimensions;
                neighbor[axis] = neighbor[axis].checked_add_signed(delta).unwrap();
                let neighbor = F32TriadRequest {
                    op: ResolvedGemmOp::Nt,
                    shape: F32TriadShape::contiguous(
                        ResolvedGemmOp::Nt,
                        (neighbor[0], neighbor[1], neighbor[2]),
                    ),
                };
                assert_ne!(
                    scalar_launch_plan(facts, neighbor, operands).unwrap(),
                    qualified,
                    "dimension neighbor {neighbor:?}"
                );
            }
        }
        for shape in [
            F32TriadShape {
                lda: request.shape.lda + 1,
                ..request.shape
            },
            F32TriadShape {
                ldb: request.shape.ldb + 1,
                ..request.shape
            },
            F32TriadShape {
                ldc: request.shape.ldc + 1,
                ..request.shape
            },
        ] {
            assert_ne!(
                scalar_launch_plan(
                    facts,
                    F32TriadRequest {
                        op: ResolvedGemmOp::Nt,
                        shape,
                    },
                    operands,
                )
                .unwrap(),
                qualified,
                "stride neighbor {shape:?}"
            );
        }
        for op in [ResolvedGemmOp::Nn, ResolvedGemmOp::Tn] {
            let mutated = F32TriadRequest {
                op,
                shape: F32TriadShape::contiguous(op, (4_621, 384, 1_928)),
            };
            assert_ne!(
                scalar_launch_plan(facts, mutated, operands).unwrap(),
                qualified
            );
        }
    }

    #[test]
    fn nt_d128_out_transpose_m64n64_selector_is_an_exact_splitk_cell() {
        let facts = tn_admission_facts();
        let operands = nn_qualified_operands();
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nt,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, (1_024, 256, 128)),
        };
        let qualified = ScalarDispatchPlan::NtD128OutTransposeM64N64Qualified;
        assert_eq!(
            scalar_dispatch_plan(request, 170).unwrap(),
            ScalarDispatchPlan::NtSplitKMain {
                n_main: 128,
                n_tail: 0,
            }
        );
        assert_eq!(
            scalar_launch_plan(facts, request, operands).unwrap(),
            qualified
        );
        assert!(qualified.needs_transpose_scratch());
        assert!(!qualified.needs_split_scratch());

        let dimensions = [request.shape.m, request.shape.k, request.shape.n];
        for axis in 0..3 {
            for delta in [-1_isize, 1] {
                let mut neighbor = dimensions;
                neighbor[axis] = neighbor[axis].checked_add_signed(delta).unwrap();
                let neighbor = F32TriadRequest {
                    op: ResolvedGemmOp::Nt,
                    shape: F32TriadShape::contiguous(
                        ResolvedGemmOp::Nt,
                        (neighbor[0], neighbor[1], neighbor[2]),
                    ),
                };
                assert_ne!(
                    scalar_launch_plan(facts, neighbor, operands).unwrap(),
                    qualified
                );
            }
        }
        for shape in [
            F32TriadShape {
                lda: 129,
                ..request.shape
            },
            F32TriadShape {
                ldb: 129,
                ..request.shape
            },
            F32TriadShape {
                ldc: 257,
                ..request.shape
            },
        ] {
            assert_ne!(
                scalar_launch_plan(
                    facts,
                    F32TriadRequest {
                        op: ResolvedGemmOp::Nt,
                        shape
                    },
                    operands,
                )
                .unwrap(),
                qualified
            );
        }
        for op in [ResolvedGemmOp::Nn, ResolvedGemmOp::Tn] {
            let mutated = F32TriadRequest {
                op,
                shape: F32TriadShape::contiguous(op, (1_024, 256, 128)),
            };
            assert_ne!(
                scalar_launch_plan(facts, mutated, operands).unwrap(),
                qualified
            );
        }

        let mut fact_mutations = Vec::new();
        let mut wrong = facts;
        wrong.compute_capability = (12, 1);
        fact_mutations.push(wrong);
        let mut wrong = facts;
        wrong.multiprocessor_count = 169;
        fact_mutations.push(wrong);
        let mut wrong = facts;
        wrong.scalar_compiler.nvrtc_version = (13, 1);
        fact_mutations.push(wrong);
        let mut wrong = facts;
        wrong.scalar_compiler.nvrtc_library_domain = [0; 32];
        fact_mutations.push(wrong);
        let mut wrong = facts;
        wrong.scalar_compiler.invocation_digest = [0; 32];
        wrong.scalar_artifact.compile_key = [0; 32];
        fact_mutations.push(wrong);
        let mut wrong = facts;
        wrong.scalar_artifact.artifact_digest = [0; 32];
        fact_mutations.push(wrong);
        for wrong in fact_mutations {
            assert_ne!(
                scalar_launch_plan(wrong, request, operands).unwrap(),
                qualified
            );
        }
        for mutation in [
            F32TriadOperands {
                alpha: f32::from_bits(1.0_f32.to_bits() + 1),
                ..operands
            },
            F32TriadOperands {
                beta: -0.0,
                ..operands
            },
            F32TriadOperands {
                bias: Some(0x4000),
                ..operands
            },
            F32TriadOperands {
                b: operands.b + 4,
                ..operands
            },
            F32TriadOperands {
                output: 0,
                ..operands
            },
            F32TriadOperands { a: 0, ..operands },
            F32TriadOperands { b: 0, ..operands },
        ] {
            assert_ne!(
                scalar_launch_plan(facts, request, mutation).unwrap(),
                qualified
            );
        }
    }

    #[test]
    fn nt_prism_vector_rejects_environment_and_operand_mutations() {
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nt,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, (4_621, 384, 1_928)),
        };
        let operands = nn_qualified_operands();
        let qualified = ScalarDispatchPlan::NtPrismVectorQualified;
        let mut fact_mutations = Vec::new();

        let mut facts = tn_admission_facts();
        facts.compute_capability = (12, 1);
        fact_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.compute_capability = (8, 9);
        fact_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.multiprocessor_count = 169;
        fact_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_compiler.target = CudaTarget::new("compute_121").unwrap();
        fact_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_compiler.nvrtc_version = (13, 1);
        fact_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_compiler.nvrtc_version = (13, 3);
        fact_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_compiler.nvrtc_library_known = false;
        fact_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_compiler.nvrtc_library_domain = [0; 32];
        fact_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_artifact.compile_key = [9; 32];
        fact_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_compiler.invocation_digest = [0; 32];
        facts.scalar_artifact.compile_key = [0; 32];
        fact_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_artifact.module_kind = ModuleKind::TriadSm80;
        fact_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_artifact.artifact_kind = ArtifactKind::Cubin;
        fact_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_artifact.artifact_digest = [0; 32];
        fact_mutations.push(facts);

        for facts in fact_mutations {
            assert_ne!(
                scalar_launch_plan(facts, request, operands).unwrap(),
                qualified
            );
        }
        for mutation in [
            F32TriadOperands {
                bias: Some(0x4000),
                ..operands
            },
            F32TriadOperands {
                alpha: f32::from_bits(1.0_f32.to_bits() + 1),
                ..operands
            },
            F32TriadOperands {
                beta: -0.0,
                ..operands
            },
            F32TriadOperands {
                output: 0,
                ..operands
            },
            F32TriadOperands { a: 0, ..operands },
            F32TriadOperands { b: 0, ..operands },
            F32TriadOperands {
                output: operands.output + 4,
                ..operands
            },
            F32TriadOperands {
                a: operands.a + 8,
                ..operands
            },
            F32TriadOperands {
                b: operands.b + 12,
                ..operands
            },
        ] {
            assert_ne!(
                scalar_launch_plan(tn_admission_facts(), request, mutation).unwrap(),
                qualified
            );
        }
        for pointer in [operands.output, operands.a, operands.b] {
            let shifted = F32TriadOperands {
                output: if pointer == operands.output {
                    pointer + 16
                } else {
                    operands.output
                },
                a: if pointer == operands.a {
                    pointer + 16
                } else {
                    operands.a
                },
                b: if pointer == operands.b {
                    pointer + 16
                } else {
                    operands.b
                },
                ..operands
            };
            assert_eq!(
                scalar_launch_plan(tn_admission_facts(), request, shifted).unwrap(),
                qualified
            );
        }
    }

    #[test]
    fn nt_d768_out_transpose_m64n64_requires_qualified_environment_and_operands() {
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nt,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, (2_048, 1_536, 768)),
        };
        let operands = nn_qualified_operands();
        let plan = ScalarDispatchPlan::NtD768OutTransposeM64N64Qualified;
        let mut mutations = Vec::new();
        let mut facts = tn_admission_facts();
        facts.compute_capability = (8, 9);
        mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.multiprocessor_count = 169;
        mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_compiler.target = CudaTarget::new("compute_89").unwrap();
        mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_compiler.nvrtc_version = (13, 1);
        mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_compiler.nvrtc_library_known = false;
        mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_compiler.nvrtc_library_domain = [0; 32];
        mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_artifact.compile_key = [9; 32];
        mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_artifact.artifact_digest = [0; 32];
        mutations.push(facts);
        for facts in mutations {
            assert_ne!(scalar_launch_plan(facts, request, operands).unwrap(), plan);
        }

        for mutation in [
            F32TriadOperands {
                bias: Some(0x4000),
                ..operands
            },
            F32TriadOperands {
                alpha: f32::from_bits(1.0_f32.to_bits() + 1),
                ..operands
            },
            F32TriadOperands {
                beta: -0.0,
                ..operands
            },
            F32TriadOperands {
                output: 0,
                ..operands
            },
            F32TriadOperands { a: 0, ..operands },
            F32TriadOperands { b: 0, ..operands },
            F32TriadOperands {
                output: operands.output + 4,
                ..operands
            },
            F32TriadOperands {
                a: operands.a + 8,
                ..operands
            },
            F32TriadOperands {
                b: operands.b + 12,
                ..operands
            },
        ] {
            assert_ne!(
                scalar_launch_plan(tn_admission_facts(), request, mutation).unwrap(),
                plan
            );
        }
        for aligned in [
            F32TriadOperands {
                output: operands.output + 16,
                ..operands
            },
            F32TriadOperands {
                a: operands.a + 16,
                ..operands
            },
            F32TriadOperands {
                b: operands.b + 16,
                ..operands
            },
        ] {
            assert_eq!(
                scalar_launch_plan(tn_admission_facts(), request, aligned).unwrap(),
                plan
            );
        }
    }

    #[test]
    fn nt_d768_transpose_m64n64_requires_qualified_environment_and_operands() {
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nt,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, (2_048, 768, 3_072)),
        };
        let operands = nn_qualified_operands();
        let mut facts_mutations = Vec::new();
        let mut facts = tn_admission_facts();
        facts.compute_capability = (12, 1);
        facts_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.multiprocessor_count = 169;
        facts_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_compiler.target = CudaTarget::new("compute_121").unwrap();
        facts_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_compiler.nvrtc_version = (13, 1);
        facts_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_compiler.nvrtc_library_known = false;
        facts_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_artifact.compile_key = [9; 32];
        facts_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_compiler.nvrtc_library_domain = [0; 32];
        facts_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_compiler.invocation_digest = [0; 32];
        facts.scalar_artifact.compile_key = [0; 32];
        facts_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_artifact.artifact_digest = [0; 32];
        facts_mutations.push(facts);
        for facts in facts_mutations {
            assert_ne!(
                scalar_launch_plan(facts, request, operands).unwrap(),
                ScalarDispatchPlan::NtD768TransposeM64N64Qualified
            );
        }

        for mutation in [
            F32TriadOperands {
                bias: Some(0x4000),
                ..operands
            },
            F32TriadOperands {
                alpha: f32::from_bits(1.0_f32.to_bits() + 1),
                ..operands
            },
            F32TriadOperands {
                beta: -0.0,
                ..operands
            },
            F32TriadOperands {
                output: operands.output + 4,
                ..operands
            },
            F32TriadOperands {
                a: operands.a + 8,
                ..operands
            },
            F32TriadOperands { b: 0, ..operands },
        ] {
            assert_ne!(
                scalar_launch_plan(tn_admission_facts(), request, mutation).unwrap(),
                ScalarDispatchPlan::NtD768TransposeM64N64Qualified
            );
        }
    }

    #[test]
    fn nn_m64n64_selector_requires_the_exact_scalar_environment() {
        let request = nn_qualified_request((2_048, 768, 3_072));
        let operands = nn_qualified_operands();

        let mut mutations = Vec::new();
        let mut facts = tn_admission_facts();
        facts.compute_capability = (12, 1);
        mutations.push(facts);
        for count in [169, 171] {
            let mut facts = tn_admission_facts();
            facts.multiprocessor_count = count;
            mutations.push(facts);
        }
        let mut facts = tn_admission_facts();
        facts.scalar_artifact.module_kind = ModuleKind::TriadSm80;
        mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_artifact.artifact_kind = ArtifactKind::Cubin;
        mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_artifact.compile_key = [9; 32];
        mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_compiler.target = CudaTarget::new("compute_121").unwrap();
        mutations.push(facts);
        for version in [(12, 8), (13, 1), (13, 3)] {
            let mut facts = tn_admission_facts();
            facts.scalar_compiler.nvrtc_version = version;
            mutations.push(facts);
        }
        let mut facts = tn_admission_facts();
        facts.scalar_compiler.nvrtc_library_known = false;
        mutations.push(facts);

        for facts in mutations {
            let plan = scalar_launch_plan(facts, request, operands).unwrap();
            assert_ne!(
                format!("{plan:?}"),
                "NnM64N64Qualified",
                "environment mutation {facts:?}"
            );
        }
    }

    #[test]
    fn nn_m64n64_selector_requires_the_measured_operand_domain() {
        let facts = tn_admission_facts();
        let request = nn_qualified_request((2_048, 768, 3_072));
        let operands = nn_qualified_operands();

        for admitted in [
            F32TriadOperands {
                output: operands.output + 16,
                ..operands
            },
            F32TriadOperands {
                a: operands.a + 16,
                ..operands
            },
            F32TriadOperands {
                b: operands.b + 16,
                ..operands
            },
        ] {
            let plan = scalar_launch_plan(facts, request, admitted).unwrap();
            assert_eq!(format!("{plan:?}"), "NnM64N64Qualified");
        }

        for mutation in [
            F32TriadOperands {
                bias: Some(0x4000),
                ..operands
            },
            F32TriadOperands {
                alpha: f32::from_bits(1.0_f32.to_bits() + 1),
                ..operands
            },
            F32TriadOperands {
                beta: f32::from_bits(1),
                ..operands
            },
            F32TriadOperands {
                beta: -0.0,
                ..operands
            },
            F32TriadOperands {
                output: 0,
                ..operands
            },
            F32TriadOperands { a: 0, ..operands },
            F32TriadOperands { b: 0, ..operands },
            F32TriadOperands {
                output: operands.output + 4,
                ..operands
            },
            F32TriadOperands {
                a: operands.a + 4,
                ..operands
            },
            F32TriadOperands {
                b: operands.b + 4,
                ..operands
            },
            F32TriadOperands {
                output: operands.output + 8,
                ..operands
            },
            F32TriadOperands {
                a: operands.a + 8,
                ..operands
            },
            F32TriadOperands {
                b: operands.b + 8,
                ..operands
            },
            F32TriadOperands {
                output: operands.output + 12,
                ..operands
            },
            F32TriadOperands {
                a: operands.a + 12,
                ..operands
            },
            F32TriadOperands {
                b: operands.b + 12,
                ..operands
            },
        ] {
            let plan = scalar_launch_plan(facts, request, mutation).unwrap();
            assert_ne!(
                format!("{plan:?}"),
                "NnM64N64Qualified",
                "operand mutation {mutation:?}"
            );
        }

        for op in [ResolvedGemmOp::Tn, ResolvedGemmOp::Nt] {
            let request = F32TriadRequest {
                op,
                shape: F32TriadShape::contiguous(op, (2_048, 768, 3_072)),
            };
            let plan = scalar_launch_plan(facts, request, operands).unwrap();
            assert_ne!(format!("{plan:?}"), "NnM64N64Qualified", "op {op:?}");
        }
    }

    #[test]
    fn nn_slim_split_admission_uses_three_live_waves() {
        let dims = (2560, 384, 512);
        assert_eq!(
            plan(ResolvedGemmOp::Nn, dims, 48).unwrap(),
            ScalarDispatchPlan::NnFinal { slim: true }
        );
        assert_eq!(
            plan(ResolvedGemmOp::Nn, dims, RTX_6000_ADA_SMS).unwrap(),
            ScalarDispatchPlan::NnSplitKSlim { chunks: 6 }
        );
    }

    #[test]
    fn live_sm_policy_preserves_the_frozen_142_sm_scalar_table() {
        let cases = [
            (
                ResolvedGemmOp::Nn,
                (1, 32, 32),
                ScalarDispatchPlan::NnUltraThin,
            ),
            (
                ResolvedGemmOp::Nn,
                (32, 64, 64),
                ScalarDispatchPlan::NnNarrowSmall,
            ),
            (
                ResolvedGemmOp::Nn,
                (128, 64, 64),
                ScalarDispatchPlan::NnNarrow,
            ),
            (ResolvedGemmOp::Nn, (128, 32, 1), ScalarDispatchPlan::NnGemv),
            (
                ResolvedGemmOp::Nn,
                (32, 33, 128),
                ScalarDispatchPlan::NnSplitKThinTail {
                    k_main: 32,
                    k_tail: 1,
                },
            ),
            (
                ResolvedGemmOp::Nn,
                (32, 32, 128),
                ScalarDispatchPlan::NnSplitKThin,
            ),
            (
                ResolvedGemmOp::Nn,
                (2560, 384, 512),
                ScalarDispatchPlan::NnSplitKSlim { chunks: 6 },
            ),
            (
                ResolvedGemmOp::Nn,
                (2048, 128, 512),
                ScalarDispatchPlan::NnFinal { slim: true },
            ),
            (
                ResolvedGemmOp::Nn,
                (2048, 128, 1024),
                ScalarDispatchPlan::NnFinal { slim: false },
            ),
            (ResolvedGemmOp::Tn, (32, 4, 1), ScalarDispatchPlan::TnGemv),
            (ResolvedGemmOp::Tn, (1, 1, 2), ScalarDispatchPlan::TnNarrow),
            (
                ResolvedGemmOp::Tn,
                (256, 128, 128),
                ScalarDispatchPlan::TnSplitM {
                    m_chunk: 16,
                    chunks: 16,
                },
            ),
            (
                ResolvedGemmOp::Tn,
                (128, 128, 128),
                ScalarDispatchPlan::TnFinal { slim: true },
            ),
            (
                ResolvedGemmOp::Tn,
                (128, 1024, 1024),
                ScalarDispatchPlan::TnFinal { slim: false },
            ),
            (
                ResolvedGemmOp::Nt,
                (32, 95, 127),
                ScalarDispatchPlan::NtNarrow,
            ),
            (
                ResolvedGemmOp::Nt,
                (31, 95, 128),
                ScalarDispatchPlan::NtSmallBatchWide,
            ),
            (ResolvedGemmOp::Nt, (1, 7, 1), ScalarDispatchPlan::NtGemv),
            (
                ResolvedGemmOp::Nt,
                (32, 95, 128),
                ScalarDispatchPlan::NtSplitKTail {
                    k_main: 64,
                    k_tail: 31,
                },
            ),
            (
                ResolvedGemmOp::Nt,
                (32, 96, 128),
                ScalarDispatchPlan::NtSplitKMain {
                    n_main: 128,
                    n_tail: 0,
                },
            ),
            (
                ResolvedGemmOp::Nt,
                (2048, 512, 256),
                ScalarDispatchPlan::NtSplitKSlim { chunks: 4 },
            ),
            (
                ResolvedGemmOp::Nt,
                (32, 95, 129),
                ScalarDispatchPlan::NtMidBatchWide,
            ),
            (
                ResolvedGemmOp::Nt,
                (2048, 512, 129),
                ScalarDispatchPlan::NtFinal { slim: true },
            ),
            (
                ResolvedGemmOp::Nt,
                (2048, 1024, 129),
                ScalarDispatchPlan::NtFinal { slim: false },
            ),
        ];

        for (op, dims, expected) in cases {
            assert_eq!(
                plan(op, dims, RTX_6000_ADA_SMS).unwrap(),
                expected,
                "{op:?} {dims:?}"
            );
        }
    }
}

/// Which tensor-core tile variant a TC entry point launched. Returned on
/// success so callers and tests can detect a route that silently failed to
/// launch its selected kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TcTile {
    /// 128x128 CTA tile, 256 threads / 8 warps (`gemm_bi_*_tc_*`).
    Tile128,
    /// 64x64 CTA tile, 128 threads / 4 warps (`gemm_bi_*_tc64_*`).
    Tile64,
    /// 16x32 CTA tile, 128 threads / 4 warps, 4-stage cp.async
    /// (`gemm_bi_nn_tc16_*`) - the decode rung of the ladder. NN
    /// forward only; picked by `tc_pick_tile_forward` for the small-M
    /// and narrow-N bands.
    Thin16,
    /// 128x64 CTA tile, 256 threads / 8 warps, BK32 with three stages.
    /// Forced TN dW experiment only; automatic policy never selects it.
    Rect128x64,
    /// The 64x64 TN dW body over a persistent stream-K grid
    /// (`gemm_bi_tn_tc64_streamk_*`): one CTA per multiprocessor walks a
    /// contiguous range of (tile, slab) units and the tile's last CTA folds
    /// the partial slabs in a fixed order. Its own numeric contract; the
    /// automatic policy selects it only on SM89 under
    /// `HalfTriadPolicy::AllowStreamKFixedOrder`.
    Tile64StreamK,
}

#[derive(Clone, Copy)]
struct TcGeometry {
    rows: usize,
    columns: usize,
    reduction: usize,
}

fn tc_grid_ctas(rows: usize, columns: usize, tile_rows: u64, tile_columns: u64) -> Option<u64> {
    let rows = u64::try_from(rows).ok()?.div_ceil(tile_rows);
    let columns = u64::try_from(columns).ok()?.div_ceil(tile_columns);
    rows.checked_mul(columns)
}

fn tc_pick_square_tile(
    op: Option<super::super::kernel_identity::PolicyOp>,
    geometry: TcGeometry,
    multiprocessor_count: u32,
) -> Option<TcTile> {
    let policy = super::super::kernel_identity::Sm80TcPolicy::current();
    if multiprocessor_count == 0
        || geometry.rows < policy.square_tile_min
        || geometry.columns < policy.square_tile_min
    {
        return None;
    }
    if geometry.rows < policy.large_tile_min || geometry.columns < policy.large_tile_min {
        return Some(TcTile::Tile64);
    }

    let rows128 = u64::try_from(geometry.rows).ok()?.div_ceil(128);
    let columns128 = u64::try_from(geometry.columns).ok()?.div_ceil(128);
    let tiles128 = rows128.checked_mul(columns128)?;
    let sms = u64::from(multiprocessor_count);

    if op == Some(super::super::kernel_identity::PolicyOp::Dw) {
        let short = rows128.min(columns128);
        let long = rows128.max(columns128);
        let rectangular = long >= short.checked_mul(policy.tn_rectangular_min_aspect)?;
        let at_most_one_wave = tiles128.checked_mul(policy.tn_rectangular_wave_denominator)?
            <= sms.checked_mul(policy.tn_rectangular_wave_numerator)?;
        if rectangular && at_most_one_wave {
            return Some(TcTile::Tile64);
        }
    }

    let short_reduction_tile128 = op.is_none()
        && geometry.reduction <= policy.forward_short_reduction_max
        && tiles128.checked_mul(policy.forward_short_reduction_wave_denominator)?
            >= sms.checked_mul(policy.forward_short_reduction_wave_numerator)?;
    let base_tile128 = tiles128.checked_mul(policy.tile128_base_wave_denominator)?
        >= sms.checked_mul(policy.tile128_base_wave_numerator)?;
    if short_reduction_tile128 || base_tile128 {
        Some(TcTile::Tile128)
    } else {
        Some(TcTile::Tile64)
    }
}

/// The forward thin tile covers the narrow rows or columns below Tile64.
/// Its per-element MMA order matches the square tiles, so crossing this
/// scheduling boundary preserves the forward numeric contract.
pub(super) fn tc_pick_tile_forward(
    dims: (usize, usize, usize),
    multiprocessor_count: u32,
) -> Option<TcTile> {
    let (rows, reduction, columns) = dims;
    let policy = super::super::kernel_identity::Sm80TcPolicy::current();
    if multiprocessor_count == 0
        || rows == 0
        || reduction == 0
        || columns < policy.forward_min_columns
    {
        return None;
    }
    if rows <= policy.forward_thin_max_rows
        || (policy.forward_thin_below_square_columns && columns < policy.square_tile_min)
    {
        return Some(TcTile::Thin16);
    }

    let grid64 = tc_grid_ctas(rows, columns, 64, 64)?;
    let sms = u64::from(multiprocessor_count);
    let underfilled = grid64.checked_mul(policy.forward_underfill_wave_denominator)?
        < sms.checked_mul(policy.forward_underfill_wave_numerator)?;
    let measured_column_band = columns < policy.large_tile_min
        || (rows >= policy.large_tile_min && columns >= policy.forward_underfill_min_columns);
    if reduction <= policy.forward_underfill_max_reduction && underfilled && measured_column_band {
        return Some(TcTile::Thin16);
    }

    tc_pick_square_tile(
        None,
        TcGeometry {
            rows,
            columns,
            reduction,
        },
        multiprocessor_count,
    )
}

pub(super) fn tc_pick_tile_backward(
    op: super::super::kernel_identity::PolicyOp,
    dims: (usize, usize, usize),
    multiprocessor_count: u32,
) -> Option<TcTile> {
    let (batch, n_in, n_out) = dims;
    let geometry = match op {
        super::super::kernel_identity::PolicyOp::Dw => TcGeometry {
            rows: n_in,
            columns: n_out,
            reduction: batch,
        },
        super::super::kernel_identity::PolicyOp::Dx => TcGeometry {
            rows: batch,
            columns: n_in,
            reduction: n_out,
        },
    };
    let policy = super::super::kernel_identity::Sm80TcPolicy::current();
    if multiprocessor_count == 0
        || (policy.reject_zero_axes
            && (geometry.rows == 0 || geometry.columns == 0 || geometry.reduction == 0))
    {
        return None;
    }
    if geometry.rows >= policy.square_tile_min && geometry.columns >= policy.square_tile_min {
        return tc_pick_square_tile(Some(op), geometry, multiprocessor_count);
    }

    let one_long_axis =
        geometry.rows >= policy.square_tile_min || geometry.columns >= policy.square_tile_min;
    let grid64 = tc_grid_ctas(geometry.rows, geometry.columns, 64, 64)?;
    (one_long_axis
        && geometry.reduction >= policy.backward_tail_min_reduction
        && grid64 >= policy.backward_tail_min_tile64_ctas)
        .then_some(TcTile::Tile64)
}

pub(super) fn tc_pick_tile_backward_for_device(
    op: super::super::kernel_identity::PolicyOp,
    dims: (usize, usize, usize),
    multiprocessor_count: u32,
    compute_capability: (i32, i32),
    half_policy: HalfTriadPolicy,
) -> Option<TcTile> {
    let portable = tc_pick_tile_backward(op, dims, multiprocessor_count);
    let (major, minor) = compute_capability;
    let composed = u32::try_from(major)
        .ok()
        .zip(u32::try_from(minor).ok())
        .is_some_and(super::contract::portable_extensions_composed_for_cc);
    if !composed || multiprocessor_count == 0 {
        return portable;
    }
    // The stream-K dW schedule serves any 64x64 grid whose (tile, slab)
    // units give every multiprocessor a deep enough reduction, and only
    // when the half policy permits its fixed-order fold; a request that
    // stays on the tiled contract never sees it. Its persistent grid is
    // bounded by the resident CTA count, so the tile count itself no longer
    // limits it: the census shows it winning from one wave of tiles to
    // eight wherever the reduction runs 2048 rows or more. It runs on every
    // board whose portable module composes it; the CC 12.x boards keep
    // their own stream-K kernel instead.
    if op == super::super::kernel_identity::PolicyOp::Dw
        && half_policy == HalfTriadPolicy::AllowStreamKFixedOrder
    {
        let (batch, n_in, n_out) = dims;
        let policy = super::super::kernel_identity::Sm80TcPolicy::current();
        let tiles64 = tc_grid_ctas(n_in, n_out, 64, 64)?;
        let slabs = u64::try_from(batch).ok()?.div_ceil(64);
        let units = tiles64.checked_mul(slabs)?;
        let sms = u64::from(multiprocessor_count);
        let deep_enough = units >= sms.checked_mul(policy.stream_k_min_slabs_per_cta)?;
        if tiles64 > 0 && deep_enough {
            return Some(TcTile::Tile64StreamK);
        }
    }

    // The tile preferences below were measured on the Ada board alone.
    if compute_capability != (8, 9) {
        return portable;
    }
    let (batch, n_in, n_out) = dims;
    let geometry = match op {
        super::super::kernel_identity::PolicyOp::Dw => TcGeometry {
            rows: n_in,
            columns: n_out,
            reduction: batch,
        },
        super::super::kernel_identity::PolicyOp::Dx => TcGeometry {
            rows: batch,
            columns: n_in,
            reduction: n_out,
        },
    };
    if geometry.rows < 128 || geometry.columns < 128 || geometry.reduction == 0 {
        return portable;
    }

    let tiles128 = tc_grid_ctas(geometry.rows, geometry.columns, 128, 128)?;
    let sms = u64::from(multiprocessor_count);
    match op {
        super::super::kernel_identity::PolicyOp::Dw => {
            let four_waves = sms.checked_mul(4)?;
            if geometry.reduction <= 128 && tiles128 >= four_waves {
                Some(TcTile::Tile128)
            } else {
                Some(TcTile::Tile64)
            }
        }
        super::super::kernel_identity::PolicyOp::Dx => {
            let enough_tile128_work = tiles128.checked_mul(2)? >= sms;
            let short = geometry.rows.min(geometry.columns);
            let long = geometry.rows.max(geometry.columns);
            let elongated = long.checked_mul(2)? >= short.checked_mul(3)?;
            if enough_tile128_work && (geometry.reduction >= 1024 || elongated) {
                Some(TcTile::Tile128)
            } else {
                Some(TcTile::Tile64)
            }
        }
    }
}

#[cfg(test)]
mod tc_policy_tests {
    use super::{
        TcTile, tc_half_policy_prefers_scalar_forward, tc_pick_tile_backward,
        tc_pick_tile_backward_for_device, tc_pick_tile_forward,
    };
    use crate::mamba_ssm::gpu::context::HalfTriadPolicy;
    use crate::mamba_ssm::gpu::kernel_identity::{
        FramedSha256, PolicyOp, digest_hex, gemm_dispatch_policy_digest,
    };

    const RTX_6000_ADA_SMS: u32 = 142;

    #[test]
    fn sm89_stream_k_dw_needs_the_half_permission_an_underfilled_grid_and_depth() {
        let pick = |dims, policy, cc| {
            tc_pick_tile_backward_for_device(PolicyOp::Dw, dims, RTX_6000_ADA_SMS, cc, policy)
        };
        let tiled = HalfTriadPolicy::TiledParity;
        let stream_k = HalfTriadPolicy::AllowStreamKFixedOrder;
        // The measured winners (internal/perf/sm89-streamk-tn-20260904, ratio
        // to the best tiled route): batch_input_proj 0.34, batch_out_proj
        // 0.56, prism_out_proj 0.64, prism_input_proj 0.79, rect_tall 0.80,
        // and batch_in_proj 0.86 with 144 tiles on 142 multiprocessors.
        for dims in [
            (10400, 384, 384),
            (10400, 768, 384),
            (4621, 768, 384),
            (4621, 1024, 384),
            (4096, 512, 768),
            (10400, 384, 1536),
        ] {
            assert_ne!(
                pick(dims, tiled, (8, 9)),
                Some(TcTile::Tile64StreamK),
                "{dims:?}"
            );
            assert_eq!(
                pick(dims, stream_k, (8, 9)),
                Some(TcTile::Tile64StreamK),
                "{dims:?}"
            );
            // Every board whose portable module composes the schedule runs
            // it; the CC 12.x boards compose their own stream-K instead.
            assert_eq!(
                pick(dims, stream_k, (8, 6)),
                Some(TcTile::Tile64StreamK),
                "{dims:?}"
            );
            assert_ne!(
                pick(dims, stream_k, (12, 0)),
                Some(TcTile::Tile64StreamK),
                "{dims:?}"
            );
            assert_eq!(
                pick(dims, stream_k, (12, 0)),
                pick(dims, tiled, (12, 0)),
                "{dims:?}"
            );
        }
        // Grids of more than a wave of tiles used to stay tiled; with the
        // persistent grid bounded by the resident CTA count the depth rule
        // alone decides, and these three (prism_in_proj 186 tiles x 73
        // slabs, large 576 x 32, a filled 32 x 24 grid) are deep enough.
        for dims in [(4621, 384, 1928), (2048, 3072, 768), (10400, 2048, 1536)] {
            assert_eq!(
                pick(dims, stream_k, (8, 9)),
                Some(TcTile::Tile64StreamK),
                "{dims:?}"
            );
            assert_ne!(
                pick(dims, tiled, (8, 9)),
                Some(TcTile::Tile64StreamK),
                "{dims:?}"
            );
        }
        // Too few (tile, slab) units to give every multiprocessor a deep
        // reduction keep the tiled pick under both policies: d128_out_proj
        // 8 tiles x 16 slabs, d128_in_proj 16 x 16, underfill 48 x 4,
        // split_candidate 256 x 2, a shallow 512-row batch.
        for dims in [
            (1024, 256, 128),
            (1024, 128, 512),
            (256, 512, 384),
            (128, 8192, 128),
            (512, 384, 384),
        ] {
            assert_eq!(
                pick(dims, stream_k, (8, 9)),
                pick(dims, tiled, (8, 9)),
                "{dims:?}"
            );
            assert_ne!(
                pick(dims, stream_k, (8, 9)),
                Some(TcTile::Tile64StreamK),
                "{dims:?}"
            );
        }
        // dX never takes the dW schedule.
        assert_eq!(
            tc_pick_tile_backward_for_device(
                PolicyOp::Dx,
                (10400, 384, 384),
                RTX_6000_ADA_SMS,
                (8, 9),
                stream_k
            ),
            tc_pick_tile_backward_for_device(
                PolicyOp::Dx,
                (10400, 384, 384),
                RTX_6000_ADA_SMS,
                (8, 9),
                tiled
            )
        );
    }

    #[test]
    fn sm89_half_policy_prefers_proven_split_k_scalar_boundaries_only() {
        let prefers_scalar =
            |dims| tc_half_policy_prefers_scalar_forward((8, 9), dims, RTX_6000_ADA_SMS).unwrap();

        for dims in [
            (127, 511, 128),
            (128, 511, 128),
            (129, 511, 128),
            (128, 513, 128),
            (128, 1023, 128),
            (128, 1024, 128),
            (128, 16_384, 128),
            (128, 16_385, 128),
        ] {
            assert!(prefers_scalar(dims), "approved SM89 scalar cell {dims:?}");
            assert_eq!(
                tc_pick_tile_forward(dims, RTX_6000_ADA_SMS),
                Some(TcTile::Tile64),
                "forced TC selection stays available for {dims:?}"
            );
        }

        for dims in [
            (128, 510, 128),
            (128, 512, 128),
            (128, 768, 128),
            (128, 1023, 127),
            (128, 1023, 129),
            (128, 16_416, 128),
        ] {
            assert!(
                !prefers_scalar(dims),
                "unapproved SM89 scalar cell {dims:?}"
            );
        }
    }

    #[test]
    fn deep_split_k_scalar_preference_is_sm89_specific() {
        for compute_capability in [(8, 0), (8, 6), (8, 7), (9, 0), (10, 0), (12, 0)] {
            assert!(
                !tc_half_policy_prefers_scalar_forward(
                    compute_capability,
                    (128, 8192, 128),
                    RTX_6000_ADA_SMS,
                )
                .unwrap(),
                "unmeasured architecture {compute_capability:?} must keep TC"
            );
        }
    }

    fn selector_tile_name(tile: Option<TcTile>) -> &'static str {
        match tile {
            None => "none",
            Some(TcTile::Thin16) => "thin16",
            Some(TcTile::Tile64) => "tile64",
            Some(TcTile::Tile128) => "tile128",
            Some(TcTile::Rect128x64) => "rect128x64",
            Some(TcTile::Tile64StreamK) => "tile64_streamk",
        }
    }

    fn selector_outcome_label(outcome: [Option<TcTile>; 3]) -> String {
        format!(
            "{},{},{}",
            selector_tile_name(outcome[0]),
            selector_tile_name(outcome[1]),
            selector_tile_name(outcome[2]),
        )
    }

    fn selector_outcome_digest(
        multiprocessor_count: u32,
        shapes: &[(usize, usize, usize); 15],
        outcomes: &[[Option<TcTile>; 3]; 15],
    ) -> [u8; 32] {
        let mut digest = FramedSha256::new(b"gemm-bi-edge-selector-outcomes.v1")
            .required(b"multiprocessor-count", &multiprocessor_count.to_le_bytes())
            .required(b"outcome-count", &45_u64.to_le_bytes());
        let mut index = 0_u64;
        for (shape, outcome) in shapes.iter().zip(outcomes) {
            let shape = format!("m{}_k{}_n{}", shape.0, shape.1, shape.2);
            for (op, tile) in ["nn", "tn", "nt"].into_iter().zip(outcome) {
                digest = digest
                    .required(b"outcome-index", &index.to_le_bytes())
                    .required(b"shape", shape.as_bytes())
                    .required(b"op", op.as_bytes())
                    .required(b"tile", selector_tile_name(*tile).as_bytes());
                index += 1;
            }
        }
        digest.finish()
    }

    #[test]
    fn selector_policy_identity_is_pinned_for_sm80_plus_device_sizes() {
        let mut mismatches = Vec::new();
        for (multiprocessor_count, expected) in [
            (
                48,
                "8ffb9f41f6b4df65840cc56b04d9075d5dc90023bf669af69ed36767f8cd78ae",
            ),
            (
                80,
                "716558ebb2cc88e4c6e478b2dfddba5f183fe16ed58bcfd5fa46b4e147424746",
            ),
            (
                108,
                "62fd8115a3fed77d4a5fb1e958cc1c1f8bfeedc41fd9e86d624bf7f1c438a28f",
            ),
            (
                120,
                "27df93074ce978142c01210058242d9a1b715302f722f656d6b0b6fde370e569",
            ),
            (
                142,
                "05849a39ab8eaafd08f55ee27e72c18b3bb9bc4aa009818be5ba6ae872093012",
            ),
        ] {
            let actual = digest_hex(&gemm_dispatch_policy_digest(multiprocessor_count));
            if actual != expected {
                mismatches.push(format!(
                    "{multiprocessor_count} SM: pinned {expected}, live {actual}"
                ));
            }
        }
        assert!(
            mismatches.is_empty(),
            "policy identity moved; repin every size from the live digests:\n{}",
            mismatches.join("\n")
        );
    }

    #[test]
    fn edge_selector_outcomes_are_pinned_for_sm80_plus_device_sizes() {
        let shapes = [
            (31, 32, 32),
            (32, 32, 32),
            (128, 32, 1),
            (128, 32, 2),
            (128, 32, 127),
            (128, 32, 128),
            (128, 33, 128),
            (127, 513, 16_384),
            (128, 513, 16_384),
            (255, 16, 256),
            (256, 16, 256),
            (16, 192, 256),
            (16, 256, 256),
            (128, 512, 832),
            (129, 512, 576),
        ];
        for (multiprocessor_count, expected, expected_digest) in [
            (
                48,
                [
                    "thin16,none,none",
                    "thin16,none,none",
                    "none,none,none",
                    "none,none,none",
                    "thin16,none,none",
                    "tile64,none,none",
                    "tile64,none,none",
                    "tile64,tile128,tile64",
                    "tile128,tile128,tile64",
                    "tile64,none,tile64",
                    "tile64,tile64,tile64",
                    "thin16,tile64,none",
                    "thin16,tile64,tile64",
                    "tile64,tile128,tile64",
                    "tile64,tile64,tile64",
                ],
                "63b5e7e9a2e9077eb0be28dfc9182cceb0346ae5a599e50432b5f73d4d803089",
            ),
            (
                80,
                [
                    "thin16,none,none",
                    "thin16,none,none",
                    "none,none,none",
                    "none,none,none",
                    "thin16,none,none",
                    "tile64,none,none",
                    "tile64,none,none",
                    "tile64,tile128,tile64",
                    "tile128,tile128,tile64",
                    "tile64,none,tile64",
                    "tile64,tile64,tile64",
                    "thin16,tile64,none",
                    "thin16,tile64,tile64",
                    "tile64,tile64,tile64",
                    "tile64,tile64,tile64",
                ],
                "f7171ad4ff1d4aca745d7fa851a906e32e22eba1532c22f28205b9459c16c5ef",
            ),
            (
                108,
                [
                    "thin16,none,none",
                    "thin16,none,none",
                    "none,none,none",
                    "none,none,none",
                    "thin16,none,none",
                    "tile64,none,none",
                    "tile64,none,none",
                    "tile64,tile128,tile64",
                    "tile128,tile128,tile64",
                    "thin16,none,tile64",
                    "thin16,tile64,tile64",
                    "thin16,tile64,none",
                    "thin16,tile64,tile64",
                    "tile64,tile64,tile64",
                    "tile64,tile64,tile64",
                ],
                "ce7836babc059753e82b0a771cd8cd2db2033ae4e4842783f4d990e64ed90dcb",
            ),
            (
                120,
                [
                    "thin16,none,none",
                    "thin16,none,none",
                    "none,none,none",
                    "none,none,none",
                    "thin16,none,none",
                    "tile64,none,none",
                    "tile64,none,none",
                    "tile64,tile128,tile64",
                    "tile128,tile128,tile64",
                    "thin16,none,tile64",
                    "thin16,tile64,tile64",
                    "thin16,tile64,none",
                    "thin16,tile64,tile64",
                    "tile64,tile64,tile64",
                    "tile64,tile64,tile64",
                ],
                "9ffbbbac14c9bbdbfcc10270ac51d054d24402d02d52bfaf39e9fae2ebfa0090",
            ),
            (
                142,
                [
                    "thin16,none,none",
                    "thin16,none,none",
                    "none,none,none",
                    "none,none,none",
                    "thin16,none,none",
                    "tile64,none,none",
                    "tile64,none,none",
                    "tile64,tile128,tile64",
                    "tile128,tile128,tile64",
                    "thin16,none,tile64",
                    "thin16,tile64,tile64",
                    "thin16,tile64,none",
                    "thin16,tile64,tile64",
                    "thin16,tile64,tile64",
                    "tile64,tile64,tile64",
                ],
                "9c5fe15f7ed7bc426d35f2fce3017ed489aef0958ef7b7a049b66419e9e71bc1",
            ),
        ] {
            let outcomes = shapes.map(|dims| {
                [
                    tc_pick_tile_forward(dims, multiprocessor_count),
                    tc_pick_tile_backward(PolicyOp::Dw, dims, multiprocessor_count),
                    tc_pick_tile_backward(PolicyOp::Dx, dims, multiprocessor_count),
                ]
            });
            assert_eq!(outcomes.map(selector_outcome_label), expected);
            assert_eq!(
                digest_hex(&selector_outcome_digest(
                    multiprocessor_count,
                    &shapes,
                    &outcomes,
                )),
                expected_digest,
            );
        }
    }

    #[test]
    fn edge_selector_outcome_digest_rejects_order_and_route_mutations() {
        let mut shapes = [
            (31, 32, 32),
            (32, 32, 32),
            (128, 32, 1),
            (128, 32, 2),
            (128, 32, 127),
            (128, 32, 128),
            (128, 33, 128),
            (127, 513, 16_384),
            (128, 513, 16_384),
            (255, 16, 256),
            (256, 16, 256),
            (16, 192, 256),
            (16, 256, 256),
            (128, 512, 832),
            (129, 512, 576),
        ];
        let mut outcomes = shapes.map(|dims| {
            [
                tc_pick_tile_forward(dims, 142),
                tc_pick_tile_backward(PolicyOp::Dw, dims, 142),
                tc_pick_tile_backward(PolicyOp::Dx, dims, 142),
            ]
        });
        let frozen = selector_outcome_digest(142, &shapes, &outcomes);

        outcomes[0][0] = Some(TcTile::Tile64);
        assert_ne!(selector_outcome_digest(142, &shapes, &outcomes), frozen);
        outcomes[0][0] = Some(TcTile::Thin16);
        shapes.swap(0, 1);
        assert_ne!(selector_outcome_digest(142, &shapes, &outcomes), frozen);
    }

    fn nn(dims: (usize, usize, usize)) -> Option<TcTile> {
        tc_pick_tile_forward(dims, RTX_6000_ADA_SMS)
    }

    fn dw(dims: (usize, usize, usize)) -> Option<TcTile> {
        tc_pick_tile_backward(PolicyOp::Dw, dims, RTX_6000_ADA_SMS)
    }

    fn dx(dims: (usize, usize, usize)) -> Option<TcTile> {
        tc_pick_tile_backward(PolicyOp::Dx, dims, RTX_6000_ADA_SMS)
    }

    #[test]
    fn nn_policy_uses_underfill_and_short_reduction_wave_rules() {
        assert_eq!(nn((129, 131, 100)), Some(TcTile::Thin16));
        assert_eq!(nn((256, 512, 384)), Some(TcTile::Thin16));
        assert_eq!(nn((512, 16, 2048)), Some(TcTile::Tile128));
    }

    #[test]
    fn nn_short_reduction_boundary_is_inclusive_at_k16() {
        assert_eq!(nn((128, 16, 8064)), Some(TcTile::Tile128));
        assert_eq!(nn((128, 17, 8064)), Some(TcTile::Tile64));
    }

    #[test]
    fn nn_underfill_reduction_boundary_is_inclusive_at_k512() {
        assert_eq!(nn((129, 512, 100)), Some(TcTile::Thin16));
        assert_eq!(nn((129, 513, 100)), Some(TcTile::Tile64));
    }

    #[test]
    fn nn_underfill_grid_boundary_is_strict_between_26_and_27_ctas() {
        assert_eq!(nn((128, 512, 832)), Some(TcTile::Thin16));
        assert_eq!(nn((129, 512, 576)), Some(TcTile::Tile64));
    }

    #[test]
    fn square_half_wave_boundary_is_inclusive_between_70_and_71_ctas() {
        assert_eq!(nn((128, 17, 8960)), Some(TcTile::Tile64));
        assert_eq!(nn((128, 17, 9088)), Some(TcTile::Tile128));
    }

    #[test]
    fn tn_policy_uses_tile64_for_one_wave_rectangles() {
        assert_eq!(dw((2048, 768, 3072)), Some(TcTile::Tile64));
        assert_eq!(dw((2048, 3072, 768)), Some(TcTile::Tile64));
        assert_eq!(dw((512, 3072, 768)), Some(TcTile::Tile64));

        assert_eq!(dw((2048, 1536, 768)), Some(TcTile::Tile128));
        assert_eq!(dw((4096, 3072, 1536)), Some(TcTile::Tile128));
    }

    #[test]
    fn sm89_backward_square_policy_matches_forced_matrix_envelopes() {
        let sm89 = (8, 9);
        let sm80 = (8, 0);

        assert_eq!(
            tc_pick_tile_backward_for_device(
                PolicyOp::Dw,
                (2048, 1536, 768),
                RTX_6000_ADA_SMS,
                sm89,
                HalfTriadPolicy::TiledParity,
            ),
            Some(TcTile::Tile64),
        );
        assert_eq!(
            tc_pick_tile_backward_for_device(
                PolicyOp::Dw,
                (4096, 3072, 1536),
                RTX_6000_ADA_SMS,
                sm89,
                HalfTriadPolicy::TiledParity,
            ),
            Some(TcTile::Tile64),
        );
        assert_eq!(
            tc_pick_tile_backward_for_device(
                PolicyOp::Dx,
                (2048, 1536, 768),
                RTX_6000_ADA_SMS,
                sm89,
                HalfTriadPolicy::TiledParity,
            ),
            Some(TcTile::Tile64),
        );
        assert_eq!(
            tc_pick_tile_backward_for_device(
                PolicyOp::Dx,
                (2048, 3072, 768),
                RTX_6000_ADA_SMS,
                sm89,
                HalfTriadPolicy::TiledParity,
            ),
            Some(TcTile::Tile128),
        );
        assert_eq!(
            tc_pick_tile_backward_for_device(
                PolicyOp::Dw,
                (4096, 3072, 1536),
                RTX_6000_ADA_SMS,
                sm80,
                HalfTriadPolicy::TiledParity,
            ),
            tc_pick_tile_backward(PolicyOp::Dw, (4096, 3072, 1536), RTX_6000_ADA_SMS,),
        );
    }

    #[test]
    fn tn_rectangular_aspect_uses_the_ceil_divided_tile_grid() {
        assert_eq!(dw((1024, 513, 2432)), Some(TcTile::Tile128));
        assert_eq!(dw((1024, 640, 2433)), Some(TcTile::Tile64));
    }

    #[test]
    fn backward_tail_envelope_replaces_exact_shape_admission() {
        assert_eq!(dw((512, 16, 2048)), Some(TcTile::Tile64));
        assert_eq!(dx((512, 16, 2048)), Some(TcTile::Tile64));
        assert_eq!(dx((16, 512, 2048)), Some(TcTile::Tile64));

        assert_eq!(dw((255, 16, 2048)), None);
        assert_eq!(dx((16, 192, 256)), None);
        assert_eq!(dx((16, 256, 256)), Some(TcTile::Tile64));
        assert_eq!(dw((256, 63, 63)), None);
        assert_eq!(dw((256, 0, 256)), None);
    }

    #[test]
    fn backward_tail_reduction_boundary_is_inclusive_at_256() {
        assert_eq!(dw((255, 16, 256)), None);
        assert_eq!(dw((256, 16, 256)), Some(TcTile::Tile64));
    }

    #[test]
    fn one_sub_128_axis_never_uses_the_tile128_square_rule() {
        assert_eq!(nn((127, 513, 16384)), Some(TcTile::Tile64));
        assert_eq!(dw((512, 127, 16384)), Some(TcTile::Tile64));
    }

    #[test]
    fn policy_keeps_long_reduction_and_full_tile_controls() {
        assert_eq!(nn((128, 8192, 128)), Some(TcTile::Tile64));
        assert_eq!(nn((1024, 256, 128)), Some(TcTile::Tile64));
        assert_eq!(nn((2048, 768, 3072)), Some(TcTile::Tile128));

        assert_eq!(dx((1024, 128, 512)), Some(TcTile::Tile64));
        assert_eq!(dx((2048, 768, 3072)), Some(TcTile::Tile128));
        assert_eq!(dx((4096, 3072, 1536)), Some(TcTile::Tile128));
    }
}

#[cfg(test)]
mod sm120_tests {
    use super::Sm120Schedule;
    use super::{
        SM120_AUTO_CELLS_CC120, SM120_AUTO_CELLS_CC121, Sm120AutoRequest, resolve_sm120_auto,
        sm120_target_candidates,
    };
    use super::{SM120_STREAMK_CELLS_CC120, SM120_STREAMK_CELLS_CC121};
    use crate::mamba_ssm::gpu::context::HalfTriadPolicy;
    use crate::mamba_ssm::gpu::dtype::WeightDtype;
    use crate::mamba_ssm::gpu::gemm_bi_triad::contract::{
        Sm120Bk, Sm120ForcedRoute, Sm120LaunchOperands, Sm120Op, Sm120PhysicalRoute, Sm120Shape,
        Sm120Stages, Sm120Tile,
    };
    use crate::mamba_ssm::gpu::kernel_identity::{CudaTarget, DeviceCaps};

    fn targets(cc: (i32, i32), nvrtc: (i32, i32)) -> Vec<(&'static str, &'static str)> {
        sm120_target_candidates(cc, nvrtc)
            .iter()
            .map(|candidate| (candidate.nvrtc_arch, candidate.ptx_target))
            .collect()
    }

    fn caps(
        compute_capability: (u32, u32),
        nvrtc_version: (i32, i32),
        accepted_target: Option<&str>,
    ) -> DeviceCaps {
        DeviceCaps {
            compute_capability,
            nvrtc_version,
            accepted_target: accepted_target.map(|target| CudaTarget::new(target).unwrap()),
            optin_shared_bytes: 101_376,
            tensor_map_access: true,
        }
    }

    fn route(
        op: Sm120Op,
        dtype: WeightDtype,
        dims: (usize, usize, usize),
        physical: Sm120PhysicalRoute,
    ) -> Sm120ForcedRoute {
        Sm120ForcedRoute {
            op,
            dtype,
            physical,
            shape: Sm120Shape::contiguous(op, dims),
        }
    }

    fn qualified_cc120_routes() -> [Sm120ForcedRoute; 60] {
        let m64n64_bk64_s2 = Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S2,
            schedule: Sm120Schedule::Tiled,
        };
        let m128n64_bk32_s2 = Sm120PhysicalRoute {
            tile: Sm120Tile::M128N64,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S2,
            schedule: Sm120Schedule::Tiled,
        };
        let m64n128_bk32_s3 = Sm120PhysicalRoute {
            tile: Sm120Tile::M64N128,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S3,
            schedule: Sm120Schedule::Tiled,
        };
        let m128n128_bk32_s3 = Sm120PhysicalRoute {
            tile: Sm120Tile::M128N128,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S3,
            schedule: Sm120Schedule::Tiled,
        };
        let m128n64_bk64_s2 = Sm120PhysicalRoute {
            tile: Sm120Tile::M128N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S2,
            schedule: Sm120Schedule::Tiled,
        };
        let m128n128_bk64_s3 = Sm120PhysicalRoute {
            tile: Sm120Tile::M128N128,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S3,
            schedule: Sm120Schedule::Tiled,
        };
        let m64n64_bk64_s3 = Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S3,
            schedule: Sm120Schedule::Tiled,
        };
        let m64n128_bk32_s2 = Sm120PhysicalRoute {
            tile: Sm120Tile::M64N128,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S2,
            schedule: Sm120Schedule::Tiled,
        };
        let projection = (2048, 1536, 768);
        let large = (2048, 3072, 768);
        let deep = (4096, 3072, 1536);
        let input_projection = (2048, 768, 3072);
        let prism = (4621, 384, 1928);
        let out_projection = (4621, 768, 384);
        let input_projection_wide = (4621, 1024, 384);
        let batch_input_projection = (10400, 384, 384);
        let batch_in_projection = (10400, 384, 1536);
        let batch_out_projection = (10400, 768, 384);
        [
            route(Sm120Op::Nn, WeightDtype::Bf16, projection, m64n64_bk64_s2),
            route(Sm120Op::Nn, WeightDtype::Bf16, large, m64n64_bk64_s2),
            route(Sm120Op::Nn, WeightDtype::Bf16, deep, m128n64_bk32_s2),
            route(Sm120Op::Nn, WeightDtype::F16, projection, m64n64_bk64_s2),
            route(Sm120Op::Nn, WeightDtype::F16, large, m64n64_bk64_s2),
            route(Sm120Op::Nn, WeightDtype::F16, deep, m128n64_bk32_s2),
            route(Sm120Op::Tn, WeightDtype::Bf16, projection, m64n128_bk32_s3),
            route(Sm120Op::Tn, WeightDtype::Bf16, large, m64n128_bk32_s3),
            route(Sm120Op::Tn, WeightDtype::Bf16, deep, m128n128_bk32_s3),
            route(Sm120Op::Tn, WeightDtype::F16, projection, m64n128_bk32_s3),
            route(Sm120Op::Tn, WeightDtype::F16, large, m64n128_bk32_s3),
            route(Sm120Op::Tn, WeightDtype::F16, deep, m128n128_bk32_s3),
            route(Sm120Op::Nt, WeightDtype::Bf16, projection, m64n64_bk64_s2),
            route(Sm120Op::Nt, WeightDtype::Bf16, large, m128n64_bk32_s2),
            route(Sm120Op::Nt, WeightDtype::Bf16, deep, m128n128_bk32_s3),
            route(Sm120Op::Nt, WeightDtype::F16, projection, m64n64_bk64_s2),
            route(Sm120Op::Nt, WeightDtype::F16, large, m128n64_bk32_s2),
            route(Sm120Op::Nt, WeightDtype::F16, deep, m128n128_bk32_s3),
            route(
                Sm120Op::Nn,
                WeightDtype::Bf16,
                input_projection,
                m128n64_bk32_s2,
            ),
            route(Sm120Op::Nn, WeightDtype::Bf16, prism, m128n64_bk64_s2),
            route(
                Sm120Op::Nn,
                WeightDtype::F16,
                input_projection,
                m128n64_bk32_s2,
            ),
            route(Sm120Op::Nn, WeightDtype::F16, prism, m128n64_bk64_s2),
            route(
                Sm120Op::Tn,
                WeightDtype::Bf16,
                input_projection,
                m64n128_bk32_s3,
            ),
            route(Sm120Op::Tn, WeightDtype::Bf16, prism, m64n128_bk32_s3),
            route(
                Sm120Op::Tn,
                WeightDtype::F16,
                input_projection,
                m64n128_bk32_s3,
            ),
            route(Sm120Op::Tn, WeightDtype::F16, prism, m64n128_bk32_s3),
            route(
                Sm120Op::Nt,
                WeightDtype::Bf16,
                input_projection,
                m64n64_bk64_s2,
            ),
            route(Sm120Op::Nt, WeightDtype::Bf16, prism, m128n128_bk64_s3),
            route(
                Sm120Op::Nt,
                WeightDtype::F16,
                input_projection,
                m64n64_bk64_s2,
            ),
            route(Sm120Op::Nt, WeightDtype::F16, prism, m128n128_bk64_s3),
            route(
                Sm120Op::Nn,
                WeightDtype::Bf16,
                out_projection,
                m64n64_bk64_s2,
            ),
            route(
                Sm120Op::Nn,
                WeightDtype::Bf16,
                input_projection_wide,
                m64n64_bk64_s2,
            ),
            route(
                Sm120Op::Nn,
                WeightDtype::Bf16,
                batch_input_projection,
                m128n64_bk32_s2,
            ),
            route(
                Sm120Op::Nn,
                WeightDtype::Bf16,
                batch_in_projection,
                m128n128_bk32_s3,
            ),
            route(
                Sm120Op::Nn,
                WeightDtype::Bf16,
                batch_out_projection,
                m128n64_bk32_s2,
            ),
            route(
                Sm120Op::Nn,
                WeightDtype::F16,
                out_projection,
                m64n64_bk64_s2,
            ),
            route(
                Sm120Op::Nn,
                WeightDtype::F16,
                input_projection_wide,
                m64n64_bk64_s2,
            ),
            route(
                Sm120Op::Nn,
                WeightDtype::F16,
                batch_input_projection,
                m128n64_bk32_s2,
            ),
            route(
                Sm120Op::Nn,
                WeightDtype::F16,
                batch_in_projection,
                m128n128_bk32_s3,
            ),
            route(
                Sm120Op::Nn,
                WeightDtype::F16,
                batch_out_projection,
                m128n64_bk32_s2,
            ),
            route(
                Sm120Op::Tn,
                WeightDtype::Bf16,
                out_projection,
                m64n64_bk64_s3,
            ),
            route(
                Sm120Op::Tn,
                WeightDtype::Bf16,
                input_projection_wide,
                m64n64_bk64_s3,
            ),
            route(
                Sm120Op::Tn,
                WeightDtype::Bf16,
                batch_input_projection,
                m64n64_bk64_s3,
            ),
            route(
                Sm120Op::Tn,
                WeightDtype::Bf16,
                batch_in_projection,
                m64n64_bk64_s3,
            ),
            route(
                Sm120Op::Tn,
                WeightDtype::Bf16,
                batch_out_projection,
                m64n64_bk64_s3,
            ),
            route(
                Sm120Op::Tn,
                WeightDtype::F16,
                out_projection,
                m64n64_bk64_s3,
            ),
            route(
                Sm120Op::Tn,
                WeightDtype::F16,
                input_projection_wide,
                m64n64_bk64_s3,
            ),
            route(
                Sm120Op::Tn,
                WeightDtype::F16,
                batch_input_projection,
                m64n64_bk64_s3,
            ),
            route(
                Sm120Op::Tn,
                WeightDtype::F16,
                batch_in_projection,
                m64n64_bk64_s3,
            ),
            route(
                Sm120Op::Tn,
                WeightDtype::F16,
                batch_out_projection,
                m64n64_bk64_s3,
            ),
            route(
                Sm120Op::Nt,
                WeightDtype::Bf16,
                out_projection,
                m64n128_bk32_s2,
            ),
            route(
                Sm120Op::Nt,
                WeightDtype::Bf16,
                input_projection_wide,
                m128n128_bk32_s3,
            ),
            route(
                Sm120Op::Nt,
                WeightDtype::Bf16,
                batch_input_projection,
                m128n64_bk32_s2,
            ),
            route(
                Sm120Op::Nt,
                WeightDtype::Bf16,
                batch_in_projection,
                m128n64_bk32_s2,
            ),
            route(
                Sm120Op::Nt,
                WeightDtype::Bf16,
                batch_out_projection,
                m128n128_bk32_s3,
            ),
            route(
                Sm120Op::Nt,
                WeightDtype::F16,
                out_projection,
                m64n128_bk32_s2,
            ),
            route(
                Sm120Op::Nt,
                WeightDtype::F16,
                input_projection_wide,
                m128n128_bk32_s3,
            ),
            route(
                Sm120Op::Nt,
                WeightDtype::F16,
                batch_input_projection,
                m128n64_bk32_s2,
            ),
            route(
                Sm120Op::Nt,
                WeightDtype::F16,
                batch_in_projection,
                m128n64_bk32_s2,
            ),
            route(
                Sm120Op::Nt,
                WeightDtype::F16,
                batch_out_projection,
                m128n128_bk32_s3,
            ),
        ]
    }

    #[test]
    fn a_shape_off_the_table_takes_the_nearest_measured_tile_inside_the_band() {
        use super::{SM120_AUTO_CELLS_CC120, nearest_sm120_cell};
        // The tall rectangle sits nearest the 4621x384-output serve
        // projections and takes their tiles, one per operation.
        let tall = Sm120Shape::contiguous(Sm120Op::Nn, (4096, 512, 768));
        assert_eq!(
            nearest_sm120_cell(
                SM120_AUTO_CELLS_CC120,
                Sm120Op::Nn,
                WeightDtype::Bf16,
                tall,
                170
            ),
            Some(Sm120PhysicalRoute {
                tile: Sm120Tile::M64N64,
                bk: Sm120Bk::Bk64,
                stages: Sm120Stages::S2,
                schedule: Sm120Schedule::Tiled,
            })
        );
        let tall_tn = Sm120Shape::contiguous(Sm120Op::Tn, (4096, 512, 768));
        assert_eq!(
            nearest_sm120_cell(
                SM120_AUTO_CELLS_CC120,
                Sm120Op::Tn,
                WeightDtype::F16,
                tall_tn,
                170,
            ),
            Some(Sm120PhysicalRoute {
                tile: Sm120Tile::M64N64,
                bk: Sm120Bk::Bk64,
                stages: Sm120Stages::S3,
                schedule: Sm120Schedule::Tiled,
            })
        );
        // A measured shape is its own nearest cell.
        let measured = Sm120Shape::contiguous(Sm120Op::Nt, (2048, 3072, 768));
        assert_eq!(
            nearest_sm120_cell(
                SM120_AUTO_CELLS_CC120,
                Sm120Op::Nt,
                WeightDtype::Bf16,
                measured,
                170,
            ),
            Some(Sm120PhysicalRoute {
                tile: Sm120Tile::M128N64,
                bk: Sm120Bk::Bk32,
                stages: Sm120Stages::S2,
                schedule: Sm120Schedule::Tiled,
            })
        );
        // A narrow output whose neighbour carries a 128x128 tile cannot fill
        // one wave with it and takes the smallest tile instead.
        let narrow = Sm120Shape::contiguous(Sm120Op::Nt, (1024, 128, 512));
        let picked = nearest_sm120_cell(
            SM120_AUTO_CELLS_CC120,
            Sm120Op::Nt,
            WeightDtype::Bf16,
            narrow,
            170,
        )
        .expect("the narrow projection is inside the band");
        assert_eq!(picked.tile, Sm120Tile::M64N64);
        // A tiny square is more than a factor of eight from every cell on at
        // least one axis and gets no tile: the portable tiles serve it.
        let tiny = Sm120Shape::contiguous(Sm120Op::Nn, (64, 64, 64));
        assert_eq!(
            nearest_sm120_cell(
                SM120_AUTO_CELLS_CC120,
                Sm120Op::Nn,
                WeightDtype::Bf16,
                tiny,
                170
            ),
            None
        );
    }

    fn request_for(route: Sm120ForcedRoute) -> Sm120AutoRequest {
        let beta = if route.op == Sm120Op::Tn { 1.0 } else { 0.0 };
        Sm120AutoRequest {
            op: route.op,
            dtype: route.dtype,
            shape: route.shape,
            a_ptr: 0x1_0000,
            b_ptr: 0x2_0000,
            multiprocessors: 170,
            half_policy: HalfTriadPolicy::TiledParity,
            operands: Sm120LaunchOperands {
                output_ptr: 0x3_0000,
                bias_ptr: 0,
                alpha: 1.0,
                beta,
            },
        }
    }

    #[test]
    fn sm120_auto_table_matches_the_qualified_cc120_inventory() {
        assert_eq!(SM120_AUTO_CELLS_CC120, qualified_cc120_routes());
        assert!(SM120_AUTO_CELLS_CC121.is_empty());
    }

    #[test]
    fn sm120_auto_selects_every_qualified_cc120_cell() {
        let target = sm120_target_candidates((12, 0), (12, 8))[0];
        for expected in qualified_cc120_routes() {
            assert_eq!(
                resolve_sm120_auto(
                    caps((12, 0), (12, 8), Some("compute_120")),
                    Some(target),
                    request_for(expected),
                ),
                Some(expected)
            );
        }
    }

    #[test]
    fn sm120_auto_declines_unsupported_operands_for_each_operation() {
        let target = sm120_target_candidates((12, 0), (12, 8))[0];
        for op in [Sm120Op::Nn, Sm120Op::Tn, Sm120Op::Nt] {
            let route = qualified_cc120_routes()
                .into_iter()
                .find(|route| route.op == op)
                .expect("qualified route for operation");
            let mut request = request_for(route);
            match op {
                Sm120Op::Nn => {
                    request.operands.bias_ptr = 0x4_0000;
                    request.operands.alpha = 0.5;
                }
                Sm120Op::Tn => request.operands.beta = 0.0,
                Sm120Op::Nt => request.operands.beta = 1.0,
            }
            assert_eq!(
                resolve_sm120_auto(
                    caps((12, 0), (12, 8), Some("compute_120")),
                    Some(target),
                    request,
                ),
                None,
                "unsupported {op:?} operands"
            );
        }
    }

    #[test]
    fn a_streamk_neighbour_serves_only_an_underfilled_grid() {
        // One measured stream-K cell: the batch out projection, whose
        // twelve-by-six tile grid leaves most of the device idle.
        let cell = Sm120ForcedRoute {
            op: Sm120Op::Tn,
            dtype: WeightDtype::Bf16,
            physical: Sm120PhysicalRoute {
                tile: Sm120Tile::M64N64,
                bk: Sm120Bk::Bk64,
                stages: Sm120Stages::S3,
                schedule: Sm120Schedule::StreamK,
            },
            shape: Sm120Shape::contiguous(Sm120Op::Tn, (10400, 768, 384)),
        };
        let cells = [cell];
        // A few columns wider: still a small grid, the persistent schedule
        // carries over.
        let near = Sm120Shape::contiguous(Sm120Op::Tn, (10400, 768, 512));
        assert_eq!(
            super::nearest_sm120_cell(&cells, Sm120Op::Tn, WeightDtype::Bf16, near, 170)
                .map(|physical| physical.schedule),
            Some(Sm120Schedule::StreamK)
        );
        // Inside the neighbour band but with a grid that covers the device
        // several times over: the stream-K cell declines, leaving the shape
        // to the tiled table.
        let wide = Sm120Shape::contiguous(Sm120Op::Tn, (10400, 2048, 2048));
        assert_eq!(
            super::nearest_sm120_cell(&cells, Sm120Op::Tn, WeightDtype::Bf16, wide, 170),
            None
        );
    }

    fn streamk_cc120_routes() -> [Sm120ForcedRoute; 12] {
        let streamk = Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S3,
            schedule: Sm120Schedule::StreamK,
        };
        let shapes = [
            (4621, 384, 1928),
            (4621, 768, 384),
            (4621, 1024, 384),
            (10400, 384, 384),
            (10400, 384, 1536),
            (10400, 768, 384),
        ];
        [WeightDtype::Bf16, WeightDtype::F16]
            .into_iter()
            .flat_map(|dtype| {
                shapes.into_iter().map(move |shape| Sm120ForcedRoute {
                    op: Sm120Op::Tn,
                    dtype,
                    physical: streamk,
                    shape: Sm120Shape::contiguous(Sm120Op::Tn, shape),
                })
            })
            .collect::<Vec<_>>()
            .try_into()
            .expect("six shapes by two dtypes")
    }

    #[test]
    fn streamk_table_is_the_measured_twelve_with_a_tiled_cell_each() {
        assert_eq!(SM120_STREAMK_CELLS_CC120, &streamk_cc120_routes()[..]);
        assert!(SM120_STREAMK_CELLS_CC121.is_empty());
        for cell in SM120_STREAMK_CELLS_CC120 {
            assert_eq!(cell.op, Sm120Op::Tn, "{cell:?}");
            assert_eq!(cell.physical.schedule, Sm120Schedule::StreamK, "{cell:?}");
            assert!(
                cell.kernel_spec()
                    .expect("stream-K cell spec")
                    .symbol
                    .contains("_streamk_"),
                "{cell:?}"
            );
            // The default policy keeps a measured tiled route for the shape.
            assert!(
                SM120_AUTO_CELLS_CC120.iter().any(|tiled| {
                    tiled.op == cell.op
                        && tiled.dtype == cell.dtype
                        && tiled.shape == cell.shape
                        && tiled.physical.schedule == Sm120Schedule::Tiled
                }),
                "{cell:?} has no tiled cell"
            );
        }
    }

    #[test]
    fn streamk_cells_open_only_under_the_half_policy() {
        let target = sm120_target_candidates((12, 0), (12, 8))[0];
        let device = || caps((12, 0), (12, 8), Some("compute_120"));
        for cell in SM120_STREAMK_CELLS_CC120.iter().copied() {
            let tiled = resolve_sm120_auto(device(), Some(target), request_for(cell))
                .expect("the tiled table serves every stream-K shape");
            assert_eq!(tiled.physical.schedule, Sm120Schedule::Tiled, "{cell:?}");
            assert_eq!(tiled.shape, cell.shape);

            let mut permitted = request_for(cell);
            permitted.half_policy = HalfTriadPolicy::AllowStreamKFixedOrder;
            assert_eq!(
                resolve_sm120_auto(device(), Some(target), permitted),
                Some(cell),
                "permitted request must take the measured stream-K cell"
            );
        }
    }

    #[test]
    fn a_permitted_request_with_a_filled_grid_takes_the_tiled_table() {
        // Inside the neighbour band of the 10400-deep stream-K cells (the
        // dW output grows from 384 to 2048 rows, a factor inside the band),
        // but its 32 x 24 tile grid already covers the 170 multiprocessors:
        // the stream-K neighbour declines and the tiled table answers, so the
        // permitted route equals the default one.
        let target = sm120_target_candidates((12, 0), (12, 8))[0];
        let device = || caps((12, 0), (12, 8), Some("compute_120"));
        let template = SM120_STREAMK_CELLS_CC120[0];
        let mut request = request_for(template);
        request.shape = Sm120Shape::contiguous(Sm120Op::Tn, (10400, 2048, 1536));
        let default = resolve_sm120_auto(device(), Some(target), request)
            .expect("the tiled table covers the shape");
        assert_eq!(default.physical.schedule, Sm120Schedule::Tiled);
        request.half_policy = HalfTriadPolicy::AllowStreamKFixedOrder;
        assert_eq!(
            resolve_sm120_auto(device(), Some(target), request),
            Some(default)
        );
    }

    #[test]
    fn sm120_auto_declines_a_nonqualified_shape() {
        let route = qualified_cc120_routes()[6];
        let mut request = request_for(route);
        request.shape.m += 1;
        let target = sm120_target_candidates((12, 0), (12, 8))[0];

        // One row off a measured cell is inside the neighbour band: the
        // request takes that cell's tile on its own shape.
        assert_eq!(
            resolve_sm120_auto(
                caps((12, 0), (12, 8), Some("compute_120")),
                Some(target),
                request,
            ),
            Some(Sm120ForcedRoute {
                op: route.op,
                dtype: route.dtype,
                physical: route.physical,
                shape: request.shape,
            })
        );

        // Far outside the band nothing is measured and the request declines.
        let mut far = request_for(route);
        far.shape = Sm120Shape::contiguous(route.op, (64, 64, 64));
        assert_eq!(
            resolve_sm120_auto(
                caps((12, 0), (12, 8), Some("compute_120")),
                Some(target),
                far,
            ),
            None
        );

        let mut unsupported_dtype = request_for(route);
        unsupported_dtype.dtype = WeightDtype::F32;
        assert_eq!(
            resolve_sm120_auto(
                caps((12, 0), (12, 8), Some("compute_120")),
                Some(target),
                unsupported_dtype,
            ),
            None
        );
    }

    #[test]
    fn sm120_auto_declines_unaligned_inputs_and_outputs() {
        let route = qualified_cc120_routes()[6];
        let target = sm120_target_candidates((12, 0), (12, 8))[0];
        for operand in 0..4 {
            let mut request = request_for(route);
            match operand {
                0 => request.a_ptr += 2,
                1 => request.b_ptr += 2,
                2 => request.operands.output_ptr += 2,
                _ => request.operands.output_ptr = 0,
            }
            assert_eq!(
                resolve_sm120_auto(
                    caps((12, 0), (12, 8), Some("compute_120")),
                    Some(target),
                    request,
                ),
                None,
                "invalid operand {operand}"
            );
        }

        let nn_route = qualified_cc120_routes()[0];
        let mut request = request_for(nn_route);
        request.operands.bias_ptr = 0x4_0002;
        assert_eq!(
            resolve_sm120_auto(
                caps((12, 0), (12, 8), Some("compute_120")),
                Some(target),
                request,
            ),
            None
        );
    }

    #[test]
    fn sm120_auto_declines_without_tensor_map_capability() {
        let route = qualified_cc120_routes()[6];
        let request = request_for(route);
        let target = sm120_target_candidates((12, 0), (12, 8))[0];
        let mut unavailable = caps((12, 0), (12, 8), Some("compute_120"));
        unavailable.tensor_map_access = false;

        assert_eq!(resolve_sm120_auto(unavailable, Some(target), request), None);

        let mut insufficient_shared = caps((12, 0), (12, 8), Some("compute_120"));
        insufficient_shared.optin_shared_bytes = route
            .kernel_spec()
            .expect("qualified route specification")
            .dynamic_shared_bytes
            - 1;
        assert_eq!(
            resolve_sm120_auto(insufficient_shared, Some(target), request),
            None
        );
        assert_eq!(
            resolve_sm120_auto(caps((12, 0), (12, 8), Some("compute_120")), None, request),
            None
        );
        assert_eq!(
            resolve_sm120_auto(
                caps((12, 0), (12, 7), Some("compute_120")),
                Some(target),
                request,
            ),
            None
        );
    }

    #[test]
    fn sm120_auto_declines_a_mismatched_accepted_target() {
        let route = qualified_cc120_routes()[6];
        let request = request_for(route);
        let target = sm120_target_candidates((12, 0), (12, 8))[0];

        assert_eq!(
            resolve_sm120_auto(
                caps((12, 0), (12, 8), Some("compute_121")),
                Some(target),
                request,
            ),
            None
        );
    }

    #[test]
    fn sm120_auto_declines_without_qualified_minor_table() {
        let request = request_for(qualified_cc120_routes()[6]);
        let target = sm120_target_candidates((12, 1), (12, 9))[0];

        assert_eq!(
            resolve_sm120_auto(
                caps((12, 1), (12, 9), Some("compute_121")),
                Some(target),
                request
            ),
            None
        );
    }

    #[test]
    fn sm120_candidates_follow_toolkit_support_and_generic_compatibility() {
        assert!(targets((12, 0), (12, 7)).is_empty());
        assert!(targets((12, 1), (12, 7)).is_empty());
        assert!(targets((11, 0), (13, 2)).is_empty());
        assert!(targets((12, 2), (13, 2)).is_empty());

        assert_eq!(targets((12, 0), (12, 8)), [("compute_120", "sm_120")]);
        assert_eq!(targets((12, 1), (12, 8)), [("compute_120", "sm_120")]);
        assert_eq!(
            targets((12, 1), (12, 9)),
            [("compute_121", "sm_121"), ("compute_120", "sm_120")]
        );
        assert_eq!(
            targets((12, 1), (13, 2)),
            [("compute_121", "sm_121"), ("compute_120", "sm_120")]
        );
    }
}

#[cfg(test)]
mod tf32_tests {
    use super::{
        FIXED_COPYPLAN_EVIDENCE_COHORTS, FIXED_COPYPLAN_SOURCE_DIGEST_CAP16,
        FIXED_COPYPLAN_SOURCE_DIGEST_CAP64, SM89_EXACT_F32_D128_EVIDENCE_COHORTS,
        SM89_EXACT_F32_EVIDENCE_COHORTS, SM89_FINALIST_TF32_EVIDENCE_COHORTS,
        SM89_JOINT_TF32_EVIDENCE_COHORTS, SM89_JOINT_TF32_IDENTITY_CUDA_12_8,
        SM89_JOINT_TF32_IDENTITY_CUDA_13_0, SM89_JOINT_TF32_IDENTITY_CUDA_13_2,
        SM89_PORTABLE_TF32_IDENTITY_CUDA_12_8, SM89_PORTABLE_TF32_IDENTITY_CUDA_13_0,
        SM89_TF32_EVIDENCE_CELLS, SM89_TF32_EVIDENCE_COHORTS, SM89_TF32_QUALIFICATION_IDENTITY,
        SM120_TF32_EVIDENCE_CELLS, SM120_TF32_EVIDENCE_CELLS_DRIVER_595_84_RETAINED,
        SM120_TF32_EVIDENCE_COHORTS,
        SM120_TF32_PORTABLE_QUALIFICATION_IDENTITY_CUDA_12_8_DRIVER_595_84,
        SM120_TF32_PORTABLE_QUALIFICATION_IDENTITY_CUDA_13_0_DRIVER_595_84,
        SM120_TF32_PORTABLE_QUALIFICATION_IDENTITY_CUDA_13_2_DRIVER_595_84,
        SM120_TF32_QUALIFICATION_IDENTITY_CUDA_12_8_DRIVER_595_84,
        SM120_TF32_QUALIFICATION_IDENTITY_CUDA_13_0_DRIVER_595_84,
        SM120_TF32_QUALIFICATION_IDENTITY_CUDA_13_2_DRIVER_595_84, SM120_TF32_RETIRED_COHORTS,
        Tf32AutoEvidenceCohort, Tf32AutoOperandGate, Tf32AutoQualificationIdentity,
        matching_tf32_cohort, measured_tf32_cell, measured_tf32_route_with_operands,
        resolve_f32_triad_auto, resolve_f32_triad_auto_with_operands, resolve_tf32_forced,
    };
    use crate::mamba_ssm::gpu::context::F32TriadPolicy;
    use crate::mamba_ssm::gpu::gemm_bi_triad::contract::{
        F32_TF32_TUNING_REVISION, F32TriadAvailability, F32TriadOperands, F32TriadRequest,
        F32TriadSelection, F32TriadShape, Sm120FmaExclusions, Sm120FmaRoute, Sm120FmaTile,
        TF32_NT_SPLITK8_S3_SPEC, Tf32PhysicalRoute, Tf32PortableRoute, Tf32PortableStages,
        Tf32PortableTile, Tf32QualifiedModule, Tf32Sm90aRoute, Tf32Sm100Route, Tf32Sm120Route,
        Tf32Sm120Stages, Tf32Sm120Tile,
    };
    use crate::mamba_ssm::gpu::gemm_bi_triad::{
        Sm90aWarpgroupSchedule, Sm100Schedule, Sm100Stages, Sm100Tile,
    };
    use crate::mamba_ssm::gpu::kernel_identity::{
        ArtifactIdentity, ArtifactKind, COMPILER_REVISION, COMPOSER_REVISION, CompilerIdentity,
        CudaTarget, DeviceCaps, DeviceIdentity, DriverIdentity, ModuleKind, NUMERIC_ABI_REVISION,
        ResolvedGemmOp, SCHEDULE_REVISION, TUNING_TABLE_REVISION, digest_hex,
    };

    fn qualified_module(
        module_kind: ModuleKind,
        target_name: &str,
        device_target_name: &str,
        compute_capability: (u32, u32),
        tensor_map_access: bool,
        optin_shared_bytes: u32,
    ) -> Tf32QualifiedModule {
        let target = CudaTarget::new(target_name).unwrap();
        let device_target = CudaTarget::new(device_target_name).unwrap();
        let nvrtc_version = (13, 2);
        Tf32QualifiedModule {
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
                compute_capability,
                multiprocessor_count: 142,
                target: device_target,
                driver: DriverIdentity {
                    api_version: 13_020,
                    build_sources: 1,
                    build_digest: [7; 32],
                },
            },
            device_caps: DeviceCaps {
                compute_capability,
                nvrtc_version,
                accepted_target: Some(target),
                optin_shared_bytes,
                tensor_map_access,
            },
            sm120_fma_exclusions: Default::default(),
        }
    }

    fn request(op: ResolvedGemmOp) -> F32TriadRequest {
        F32TriadRequest {
            op,
            shape: F32TriadShape::contiguous(op, (128, 256, 128)),
        }
    }

    fn normalized_request(
        op: ResolvedGemmOp,
        output_rows: usize,
        output_columns: usize,
        reduction: usize,
    ) -> F32TriadRequest {
        let dims = match op {
            ResolvedGemmOp::Nn => (output_rows, reduction, output_columns),
            ResolvedGemmOp::Tn => (reduction, output_rows, output_columns),
            ResolvedGemmOp::Nt => (output_rows, output_columns, reduction),
        };
        F32TriadRequest {
            op,
            shape: F32TriadShape::contiguous(op, dims),
        }
    }

    fn sm89_availability() -> F32TriadAvailability {
        let mut portable = qualified_module(
            ModuleKind::TriadSm80,
            "sm_89",
            "sm_89",
            (8, 9),
            false,
            101_376,
        );
        portable.artifact.compile_key = SM89_TF32_QUALIFICATION_IDENTITY.compile_key;
        portable.artifact.artifact_digest = SM89_TF32_QUALIFICATION_IDENTITY.artifact_digest;
        portable.compiler.source_digest = SM89_TF32_QUALIFICATION_IDENTITY.source_digest;
        portable.compiler.invocation_digest = SM89_TF32_QUALIFICATION_IDENTITY.invocation_digest;
        portable.compiler.header_manifest_digest =
            SM89_TF32_QUALIFICATION_IDENTITY.header_manifest_digest;
        portable.compiler.nvrtc_library_domain =
            SM89_TF32_QUALIFICATION_IDENTITY.nvrtc_library_domain;
        F32TriadAvailability {
            portable: Some(portable),
            specialized: None,
            finalist: None,
            joint: None,
            multiprocessors: 142,
        }
    }

    fn sm89_finalist_availability() -> F32TriadAvailability {
        let mut availability = sm89_availability();
        availability.finalist = Some(qualified_module(
            ModuleKind::TriadSm89Finalist,
            "sm_89",
            "sm_89",
            (8, 9),
            false,
            49_152,
        ));
        availability
    }

    #[test]
    fn sm89_joint_cohorts_freeze_the_toolkit_specific_winner_map() {
        let expected = [
            (
                SM89_JOINT_TF32_IDENTITY_CUDA_12_8,
                Some(SM89_PORTABLE_TF32_IDENTITY_CUDA_12_8),
                10,
                Tf32PhysicalRoute::Sm89TnPreRnaM64N64,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: Tf32PortableTile::M128N128,
                    stages: Tf32PortableStages::S3,
                }),
            ),
            (
                SM89_JOINT_TF32_IDENTITY_CUDA_13_0,
                Some(SM89_PORTABLE_TF32_IDENTITY_CUDA_13_0),
                10,
                Tf32PhysicalRoute::Sm89TnPreRnaM64N64,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: Tf32PortableTile::M128N128,
                    stages: Tf32PortableStages::S3,
                }),
            ),
            (
                SM89_JOINT_TF32_IDENTITY_CUDA_13_2,
                None,
                10,
                Tf32PhysicalRoute::Sm89TnPreRnaM64N96S2,
                Tf32PhysicalRoute::Sm89NnDirectN96,
            ),
        ];
        assert_eq!(SM89_JOINT_TF32_EVIDENCE_COHORTS.len(), expected.len());
        for (cohort, (identity, portable, cells, tn_prism, nn_prism)) in
            SM89_JOINT_TF32_EVIDENCE_COHORTS.iter().zip(expected)
        {
            assert_eq!(cohort.identity, identity);
            assert_eq!(cohort.portable, portable);
            assert_eq!(cohort.cells.len(), cells);
            assert_eq!(cohort.cells[2].route, tn_prism);
            assert_eq!(cohort.cells[3].route, nn_prism);
        }
    }

    fn sm89_joint_availability(
        joint: super::Tf32AutoQualificationIdentity,
        portable: Option<super::Tf32AutoQualificationIdentity>,
    ) -> F32TriadAvailability {
        F32TriadAvailability {
            portable: portable.map(qualified_module_for_auto_identity),
            specialized: None,
            finalist: None,
            joint: Some(qualified_module_for_auto_identity(joint)),
            multiprocessors: 142,
        }
    }

    fn sm89_joint_operands(op: ResolvedGemmOp) -> F32TriadOperands {
        F32TriadOperands {
            output: 0x1000,
            a: 0x2000,
            b: 0x3000,
            bias: None,
            alpha: 1.0,
            beta: if op == ResolvedGemmOp::Tn { 1.0 } else { 0.0 },
        }
    }

    #[test]
    fn sm89_joint_auto_selects_each_toolkit_specific_winner() {
        for cohort in SM89_JOINT_TF32_EVIDENCE_COHORTS {
            let availability = sm89_joint_availability(cohort.identity, cohort.portable);
            assert!(cohort.cells.len() >= 6);
            for cell in cohort.cells {
                let (op, rows, columns, reduction) = (
                    cell.op,
                    cell.shape.output_rows,
                    cell.shape.output_columns,
                    cell.shape.reduction,
                );
                let request = normalized_request(op, rows, columns, reduction);
                assert_eq!(
                    resolve_f32_triad_auto_with_operands(
                        F32TriadPolicy::AllowDeterministicTf32,
                        request,
                        sm89_joint_operands(op),
                        availability,
                    )
                    .unwrap(),
                    F32TriadSelection::Tf32(cell.route),
                    "CUDA {:?} failed to select {op:?}/{rows}/{columns}/{reduction}",
                    cohort.identity.nvrtc_version,
                );
                assert_no_tf32_route(
                    resolve_f32_triad_auto_with_operands(
                        F32TriadPolicy::ExactScalarFma,
                        request,
                        sm89_joint_operands(op),
                        availability,
                    )
                    .unwrap(),
                );
            }
        }
    }

    #[test]
    fn sm89_joint_lower_toolkits_require_the_exact_portable_twin_only_for_nn_prism() {
        for cohort in &SM89_JOINT_TF32_EVIDENCE_COHORTS[..2] {
            let without_portable = sm89_joint_availability(cohort.identity, None);
            for cell in cohort
                .cells
                .iter()
                .filter(|cell| cell.route.module_kind() == ModuleKind::TriadSm89Tf32Joint)
            {
                let request = normalized_request(
                    cell.op,
                    cell.shape.output_rows,
                    cell.shape.output_columns,
                    cell.shape.reduction,
                );
                assert_eq!(
                    resolve_f32_triad_auto_with_operands(
                        F32TriadPolicy::AllowDeterministicTf32,
                        request,
                        sm89_joint_operands(cell.op),
                        without_portable,
                    )
                    .unwrap(),
                    F32TriadSelection::Tf32(cell.route),
                );
            }

            let prism = cohort.cells[3];
            let request = normalized_request(
                prism.op,
                prism.shape.output_rows,
                prism.shape.output_columns,
                prism.shape.reduction,
            );
            assert_no_tf32_route(
                resolve_f32_triad_auto_with_operands(
                    F32TriadPolicy::AllowDeterministicTf32,
                    request,
                    sm89_joint_operands(prism.op),
                    without_portable,
                )
                .unwrap(),
            );
            let mut wrong_portable = qualified_module_for_auto_identity(cohort.portable.unwrap());
            wrong_portable.artifact.artifact_digest[0] ^= 1;
            let wrong_twin = F32TriadAvailability {
                portable: Some(wrong_portable),
                ..without_portable
            };
            assert_no_tf32_route(
                resolve_f32_triad_auto_with_operands(
                    F32TriadPolicy::AllowDeterministicTf32,
                    request,
                    sm89_joint_operands(prism.op),
                    wrong_twin,
                )
                .unwrap(),
            );
        }
    }

    #[test]
    fn sm89_joint_identities_reject_every_single_field_mutation() {
        for cohort in SM89_JOINT_TF32_EVIDENCE_COHORTS {
            let exact = qualified_module_for_auto_identity(cohort.identity);
            assert_eq!(
                matching_tf32_cohort(exact, SM89_JOINT_TF32_EVIDENCE_COHORTS),
                Some(cohort),
            );
            for (field, mutate) in sm120_identity_mutations().into_iter().enumerate() {
                let mut drifted = exact;
                mutate(&mut drifted);
                if drifted == exact {
                    drifted.device_caps.tensor_map_access = !exact.device_caps.tensor_map_access;
                }
                assert!(
                    matching_tf32_cohort(drifted, SM89_JOINT_TF32_EVIDENCE_COHORTS).is_none(),
                    "CUDA {:?} identity mutation {field} was admitted",
                    cohort.identity.nvrtc_version,
                );
            }
        }
    }

    #[test]
    fn sm89_joint_auto_rejects_neighboring_shapes_and_operand_drift() {
        for cohort in SM89_JOINT_TF32_EVIDENCE_COHORTS {
            let availability = sm89_joint_availability(cohort.identity, cohort.portable);
            for cell in cohort.cells {
                let request = normalized_request(
                    cell.op,
                    cell.shape.output_rows,
                    cell.shape.output_columns,
                    cell.shape.reduction,
                );
                for field in 0..6 {
                    let mut neighbor = request;
                    match field {
                        0 => neighbor.shape.m += 1,
                        1 => neighbor.shape.k += 1,
                        2 => neighbor.shape.n += 1,
                        3 => neighbor.shape.lda += 1,
                        4 => neighbor.shape.ldb += 1,
                        5 => neighbor.shape.ldc += 1,
                        _ => unreachable!(),
                    }
                    if let Ok(selection) = resolve_f32_triad_auto_with_operands(
                        F32TriadPolicy::AllowDeterministicTf32,
                        neighbor,
                        sm89_joint_operands(cell.op),
                        availability,
                    ) {
                        if field < 3 {
                            // A shape one element off the cell may take the
                            // portable tier's nearest cell, never the joint route.
                            assert_no_tf32_route(selection);
                        } else {
                            assert_no_tf32_route_for(selection, "neighboring joint request");
                        }
                    }
                }
                for bad_operands in [
                    F32TriadOperands {
                        a: 0,
                        ..sm89_joint_operands(cell.op)
                    },
                    F32TriadOperands {
                        b: 0x3004,
                        ..sm89_joint_operands(cell.op)
                    },
                    F32TriadOperands {
                        output: 0x1004,
                        ..sm89_joint_operands(cell.op)
                    },
                    F32TriadOperands {
                        bias: Some(0x4000),
                        ..sm89_joint_operands(cell.op)
                    },
                    F32TriadOperands {
                        alpha: 0.5,
                        ..sm89_joint_operands(cell.op)
                    },
                    F32TriadOperands {
                        beta: if cell.op == ResolvedGemmOp::Tn {
                            0.0
                        } else {
                            1.0
                        },
                        ..sm89_joint_operands(cell.op)
                    },
                ] {
                    assert_no_tf32_route_for(
                        resolve_f32_triad_auto_with_operands(
                            F32TriadPolicy::AllowDeterministicTf32,
                            request,
                            bad_operands,
                            availability,
                        )
                        .unwrap(),
                        "joint operand drift",
                    );
                }
            }
        }
    }

    /// TF32 fail-closed: a request or operand set that drifts off a measured
    /// cell may still run on the exact family, which carries its own
    /// qualification, but it must never reach a TF32 route.
    /// A drifted request may still land on the portable tier by design,
    /// or on a proof candidate whose reference is that tier, but never on
    /// a specialized route without the proof.
    fn portable_cell_request(cell: &super::Tf32AutoCell) -> (F32TriadRequest, F32TriadOperands) {
        let dims = match cell.op {
            ResolvedGemmOp::Nn => (
                cell.shape.output_rows,
                cell.shape.reduction,
                cell.shape.output_columns,
            ),
            ResolvedGemmOp::Tn => (
                cell.shape.reduction,
                cell.shape.output_rows,
                cell.shape.output_columns,
            ),
            ResolvedGemmOp::Nt => (
                cell.shape.output_rows,
                cell.shape.output_columns,
                cell.shape.reduction,
            ),
        };
        let request = F32TriadRequest {
            op: cell.op,
            shape: F32TriadShape::contiguous(cell.op, dims),
        };
        let operands = F32TriadOperands {
            output: 0x1000,
            a: 0x2000,
            b: 0x3000,
            bias: None,
            alpha: 1.0,
            beta: if cell.op == ResolvedGemmOp::Tn {
                1.0
            } else {
                0.0
            },
        };
        (request, operands)
    }

    /// The portable cells this tree serves exactly: the last record of each
    /// (op, shape) wins, as it does for the exact lookup.
    fn portable_cells_last_wins() -> Vec<&'static super::Tf32AutoCell> {
        let mut seen = std::collections::HashSet::new();
        let mut cells = Vec::new();
        for cell in SM89_TF32_EVIDENCE_CELLS.iter().rev() {
            if cell.route.module_kind() == ModuleKind::TriadSm80
                && seen.insert((cell.op, cell.shape))
            {
                cells.push(cell);
            }
        }
        cells
    }

    #[test]
    fn nearest_portable_cell_reproduces_every_exact_cell_and_keeps_a_far_shape_out() {
        for cell in portable_cells_last_wins() {
            let (request, operands) = portable_cell_request(cell);
            assert_eq!(
                super::nearest_tf32_portable_cell(request, operands, SM89_TF32_EVIDENCE_CELLS, 142),
                Some(cell.route),
                "{:?} {:?}",
                cell.op,
                cell.shape
            );
        }
        // Without a board the band stays shut.
        let (request, operands) = portable_cell_request(&SM89_TF32_EVIDENCE_CELLS[0]);
        assert_eq!(
            super::nearest_tf32_portable_cell(request, operands, SM89_TF32_EVIDENCE_CELLS, 0),
            None
        );
        // A shape a factor of sixteen from every measured reduction is out.
        let far = F32TriadRequest {
            op: ResolvedGemmOp::Nn,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nn, (2048, 49_152, 3072)),
        };
        assert_eq!(
            super::nearest_tf32_portable_cell(far, operands, SM89_TF32_EVIDENCE_CELLS, 142),
            None
        );
        // A drifted stride never matches: the band widens shapes, not layouts.
        let mut strided = request;
        strided.shape.lda += 1;
        assert_eq!(
            super::nearest_tf32_portable_cell(strided, operands, SM89_TF32_EVIDENCE_CELLS, 142),
            None
        );
    }

    /// Leave one measured cell out and ask the band for it: how often the
    /// nearest other cell names the tile the census measured. The rate is
    /// pinned so a table change that weakens the band is noticed.
    #[test]
    fn nearest_portable_cell_leave_one_out_agreement_is_pinned() {
        let cells = portable_cells_last_wins();
        let leave_one_out = |log_bound: f64| {
            let mut agreed = 0usize;
            let mut covered = 0usize;
            let mut disagreements = Vec::new();
            for held_out in &cells {
                let remaining = SM89_TF32_EVIDENCE_CELLS
                    .iter()
                    .filter(|cell| !(cell.op == held_out.op && cell.shape == held_out.shape))
                    .copied()
                    .collect::<Vec<_>>();
                let (request, operands) = portable_cell_request(held_out);
                let Some(chosen) = super::nearest_tf32_portable_cell_within(
                    request, operands, &remaining, 142, log_bound,
                ) else {
                    continue;
                };
                covered += 1;
                if chosen == held_out.route {
                    agreed += 1;
                } else {
                    disagreements.push(format!(
                        "{:?} {:?}: measured {:?}, band {:?}",
                        held_out.op, held_out.shape, held_out.route, chosen
                    ));
                }
            }
            (agreed, covered, disagreements)
        };
        for factor in [2.0_f64, 4.0, 8.0] {
            let (agreed, covered, disagreements) = leave_one_out(factor.ln());
            eprintln!(
                "portable neighbour band x{factor}: {agreed} of {covered} covered cells agree ({} cells, {} disagreements)\n{}",
                cells.len(),
                disagreements.len(),
                disagreements.join("\n")
            );
        }
        let (agreed, covered, _) = leave_one_out(super::TF32_PORTABLE_NEIGHBOUR_LOG_BOUND);
        assert!(
            covered * 10 >= cells.len() * 6,
            "the band covers {covered} of {}",
            cells.len()
        );
        assert!(agreed * 10 >= covered * 8, "{agreed} of {covered} agree");
    }

    fn assert_no_tf32_route(selection: F32TriadSelection) {
        let portable_tier = |route: Tf32PhysicalRoute| route.module_kind() == ModuleKind::TriadSm80;
        assert!(
            match selection {
                F32TriadSelection::Tf32(route) => portable_tier(route),
                F32TriadSelection::Tf32Proof { reference, .. } => portable_tier(reference),
                _ => true,
            },
            "drifted request selected {selection:?}"
        );
    }

    fn assert_no_tf32_route_for(selection: F32TriadSelection, label: &str) {
        assert!(
            !matches!(selection, F32TriadSelection::Tf32(_)),
            "{label}: rejected request selected {selection:?}"
        );
    }

    fn sm120_availability_for(
        identity: super::Tf32AutoQualificationIdentity,
    ) -> F32TriadAvailability {
        let specialized = qualified_module_for_auto_identity(identity);
        F32TriadAvailability {
            portable: None,
            specialized: Some(specialized),
            finalist: None,
            joint: None,
            multiprocessors: 170,
        }
    }

    fn qualified_module_for_auto_identity(
        identity: super::Tf32AutoQualificationIdentity,
    ) -> Tf32QualifiedModule {
        let mut module = qualified_module(
            identity.module_kind,
            identity.module_target,
            identity.device_target,
            identity.compute_capability,
            identity.tensor_map_access,
            identity.optin_shared_bytes,
        );
        module.artifact.compile_key = identity.compile_key;
        module.artifact.artifact_digest = identity.artifact_digest;
        module.compiler.source_digest = identity.source_digest;
        module.compiler.invocation_digest = identity.invocation_digest;
        module.compiler.header_manifest_digest = identity.header_manifest_digest;
        module.compiler.nvrtc_version = identity.nvrtc_version;
        module.compiler.nvrtc_library_domain = identity.nvrtc_library_domain;
        module.device.multiprocessor_count = identity.multiprocessor_count;
        module.device_caps.nvrtc_version = identity.nvrtc_version;
        module
    }

    fn measured_finalist_identities() -> [Tf32AutoQualificationIdentity; 3] {
        fn digest(hex: &str) -> [u8; 32] {
            assert_eq!(hex.len(), 64);
            std::array::from_fn(|index| {
                u8::from_str_radix(&hex[index * 2..index * 2 + 2], 16).unwrap()
            })
        }
        let identity =
            |nvrtc_version,
             compile_key,
             artifact_digest,
             header_manifest_digest,
             nvrtc_library_domain| Tf32AutoQualificationIdentity {
                module_kind: ModuleKind::TriadSm89Finalist,
                module_target: "sm_89",
                device_target: "sm_89",
                compute_capability: (8, 9),
                multiprocessor_count: 142,
                nvrtc_version,
                optin_shared_bytes: 101_376,
                tensor_map_access: false,
                compile_key: digest(compile_key),
                artifact_digest: digest(artifact_digest),
                source_digest: digest(
                    "52a6af8b0dde7cb836129368be73c832c7fcc66673dfbe5f406c7a067125d8ba",
                ),
                invocation_digest: digest(compile_key),
                header_manifest_digest: digest(header_manifest_digest),
                nvrtc_library_domain: digest(nvrtc_library_domain),
            };
        [
            identity(
                (12, 8),
                "ebc1b6827467ae9f3fce83afcd40779722cbb35912d91ee81e8cf15cc99c4e07",
                "57d8d5515b6831559af93f2f937e22cbfcfea2591d4f857430a34d90400fcb33",
                "8be40bf612d9fb5ecc20cd5016fca3946e554cd7bae6d39a288062ebd7b387eb",
                "26b0a3a02044ffcbc1693fd83e9261beffa692a4fbcfe3ac5e9d8c87980bb155",
            ),
            identity(
                (13, 0),
                "fc2cbea5854d836f95f5038e8d5ed7ae6005a210130ef81811f23fa98274733e",
                "b53e0f95e633c0566591b9860d65ee0bb0f8385716aa5274c973d305fa2903a3",
                "2d12089722614ec862153ce05b25e2136a4fc34c12dd36bd9880fd174bb6a7cf",
                "709b91c36bfb0ed966ee69adc8d6f87ff110eecf3dfb5060367f183ce614eb0d",
            ),
            identity(
                (13, 2),
                "ab48018a70361326d081bf29299ef47f4f28633149cea967c9ac233c0c5fc383",
                "be11872557f8febdfd56d82258dd7eab77488470dbb1298a14b5cab34d78acb6",
                "b848b64b691d6e1ae1e016f1d9e35ec406a941bdd7c96d359d0cb2c119eef76c",
                "d031a53eb97235b70f62f652932db1bdf728ea229c8ca809d53c5ffd91642687",
            ),
        ]
    }

    /// The live SM120 cohort of a toolkit.
    fn sm120_cohort(nvrtc_version: (i32, i32)) -> Tf32AutoEvidenceCohort {
        SM120_TF32_EVIDENCE_COHORTS
            .iter()
            .copied()
            .find(|cohort| cohort.identity.nvrtc_version == nvrtc_version)
            .unwrap_or_else(|| panic!("missing live SM120 TF32 CUDA {nvrtc_version:?} cohort"))
    }

    /// A retired SM120 cohort of a toolkit: frozen against a source this tree
    /// no longer contains, kept as the record of what that stack chose.
    fn sm120_retired_cohort(nvrtc_version: (i32, i32)) -> Tf32AutoEvidenceCohort {
        SM120_TF32_RETIRED_COHORTS
            .iter()
            .copied()
            .find(|cohort| cohort.identity.nvrtc_version == nvrtc_version)
            .unwrap_or_else(|| panic!("missing retired SM120 TF32 CUDA {nvrtc_version:?} cohort"))
    }

    fn sm120_identity_mutations() -> [fn(&mut Tf32QualifiedModule); 26] {
        [
            |module| module.module_kind = ModuleKind::Fixed,
            |module| module.target = CudaTarget::new("compute_121").unwrap(),
            |module| module.artifact.module_kind = ModuleKind::Fixed,
            |module| module.artifact.artifact_kind = ArtifactKind::Cubin,
            |module| module.artifact.compile_key[0] ^= 1,
            |module| module.artifact.artifact_digest[0] ^= 1,
            |module| module.compiler.source_digest[0] ^= 1,
            |module| module.compiler.invocation_digest[0] ^= 1,
            |module| module.compiler.header_manifest_digest[0] ^= 1,
            |module| module.compiler.target = CudaTarget::new("compute_121").unwrap(),
            |module| module.compiler.nvrtc_version.1 ^= 1,
            |module| module.compiler.nvrtc_library_domain[0] ^= 1,
            |module| module.compiler.nvrtc_library_known = false,
            |module| module.compiler.output_kind = ArtifactKind::Cubin,
            |module| module.compiler.composer_revision ^= 1,
            |module| module.compiler.compiler_revision ^= 1,
            |module| module.compiler.numeric_abi_revision ^= 1,
            |module| module.compiler.schedule_revision ^= 1,
            |module| module.device.compute_capability = (12, 1),
            |module| module.device.multiprocessor_count -= 1,
            |module| module.device.target = CudaTarget::new("sm_121").unwrap(),
            |module| module.device_caps.compute_capability = (12, 1),
            |module| module.device_caps.nvrtc_version.1 ^= 1,
            |module| module.device_caps.accepted_target = None,
            |module| module.device_caps.optin_shared_bytes -= 1,
            |module| module.device_caps.tensor_map_access = false,
        ]
    }

    fn portable_sm120_coupled_mutations() -> [fn(&mut Tf32QualifiedModule); 6] {
        [
            |module| {
                module.artifact.compile_key[0] ^= 1;
                module.compiler.invocation_digest[0] ^= 1;
            },
            |module| {
                let target = CudaTarget::new("compute_121").unwrap();
                module.target = target;
                module.compiler.target = target;
                module.device_caps.accepted_target = Some(target);
            },
            |module| {
                module.artifact.artifact_kind = ArtifactKind::Cubin;
                module.compiler.output_kind = ArtifactKind::Cubin;
            },
            |module| {
                module.device.compute_capability = (12, 1);
                module.device_caps.compute_capability = (12, 1);
            },
            |module| {
                module.compiler.nvrtc_version = (13, 1);
                module.device_caps.nvrtc_version = (13, 1);
            },
            |module| {
                module.module_kind = ModuleKind::Fixed;
                module.artifact.module_kind = ModuleKind::Fixed;
            },
        ]
    }

    #[derive(Clone, Copy, Debug)]
    enum Sm120ManifestRoute {
        Tiled(Tf32Sm120Tile, Tf32Sm120Stages),
        StreamK(Tf32Sm120Tile, Tf32Sm120Stages),
    }

    impl Sm120ManifestRoute {
        fn route(self) -> Tf32PhysicalRoute {
            match self {
                Self::Tiled(tile, stages) => {
                    Tf32PhysicalRoute::Sm120TmaMmaTf32Rna(Tf32Sm120Route { tile, stages })
                }
                Self::StreamK(tile, stages) => {
                    Tf32PhysicalRoute::Sm120TmaMmaTf32RnaStreamKV1(Tf32Sm120Route { tile, stages })
                }
            }
        }
    }

    type Sm120RouteManifestEntry = (ResolvedGemmOp, usize, usize, usize, Sm120ManifestRoute);

    fn assert_sm120_route_manifest<const N: usize>(
        cells: &[super::Tf32AutoCell],
        expected: [Sm120RouteManifestEntry; N],
    ) {
        assert_eq!(cells.len(), expected.len());
        for (cell, (op, rows, columns, reduction, route)) in cells.iter().zip(expected) {
            assert_eq!(cell.op, op);
            assert_eq!(
                cell.shape,
                super::Tf32ExactShape {
                    output_rows: rows,
                    output_columns: columns,
                    reduction,
                }
            );
            assert_eq!(cell.route, route.route());
        }
    }

    #[test]
    fn tf32_cohorts_follow_the_bound_module_family() {
        let sm120 = qualified_module(
            ModuleKind::TriadSm120,
            "compute_120",
            "sm_120",
            (12, 0),
            true,
            101_376,
        );
        let sm90a = qualified_module(
            ModuleKind::TriadSm90a,
            "sm_90a",
            "sm_90a",
            (9, 0),
            true,
            232_448,
        );
        let sm100 = qualified_module(
            ModuleKind::TriadSm100,
            "compute_100a",
            "sm_100a",
            (10, 0),
            true,
            232_448,
        );
        let (family, cohorts) = super::tf32_evidence_cohorts(sm120);
        assert_eq!(family, "SM120");
        assert!(!cohorts.is_empty());
        // The families without a measured board hold no cohort yet, so the
        // search declines them instead of reading the SM120 evidence.
        for (module, expected) in [(sm90a, "SM90a"), (sm100, "SM100")] {
            let (family, cohorts) = super::tf32_evidence_cohorts(module);
            assert_eq!(family, expected);
            assert!(cohorts.is_empty());
            assert!(super::matching_tf32_cohort(module, cohorts).is_none());
        }
    }

    #[test]
    fn sm120_tf32_cuda_12_8_identity_matches_the_literal_qualification_manifest() {
        let identity = sm120_retired_cohort((12, 8)).identity;
        assert_eq!(identity.module_kind, ModuleKind::TriadSm120);
        assert_eq!(identity.module_target, "compute_120");
        assert_eq!(identity.device_target, "sm_120");
        assert_eq!(identity.compute_capability, (12, 0));
        assert_eq!(identity.multiprocessor_count, 170);
        assert_eq!(identity.nvrtc_version, (12, 8));
        assert_eq!(identity.optin_shared_bytes, 101_376);
        assert!(identity.tensor_map_access);
        assert_eq!(
            digest_hex(&identity.compile_key),
            "a6cea94cf0a95464d9070cd419047676b3dee39c318bb96eeb511a360b368021"
        );
        assert_eq!(
            digest_hex(&identity.artifact_digest),
            "5b6a1cbc4f0b9a8f4219b2fd0b982adb798bb918e81215de48843690de3df7ab"
        );
        assert_eq!(
            digest_hex(&identity.source_digest),
            "246d2e35eb059cd696ec74325d4a223312fa605fe84ff58f17f6638d99b8182c"
        );
        assert_eq!(
            digest_hex(&identity.invocation_digest),
            "a6cea94cf0a95464d9070cd419047676b3dee39c318bb96eeb511a360b368021"
        );
        assert_eq!(
            digest_hex(&identity.header_manifest_digest),
            "fe8038d2296a5466909468f2b502f7cb3bf55eec314a4096a43b029f6d8b1d30"
        );
        assert_eq!(
            digest_hex(&identity.nvrtc_library_domain),
            "26b0a3a02044ffcbc1693fd83e9261beffa692a4fbcfe3ac5e9d8c87980bb155"
        );

        let module = qualified_module_for_auto_identity(identity);
        assert_eq!(module.artifact.module_kind, ModuleKind::TriadSm120);
        assert_eq!(module.artifact.artifact_kind, ArtifactKind::Ptx);
        assert_eq!(module.compiler.target.as_str(), "compute_120");
        assert!(module.compiler.nvrtc_library_known);
        assert_eq!(module.compiler.output_kind, ArtifactKind::Ptx);
        assert_eq!(module.compiler.composer_revision, 1);
        assert_eq!(module.compiler.compiler_revision, COMPILER_REVISION);
        assert_eq!(module.compiler.numeric_abi_revision, 5);
        assert_eq!(module.compiler.schedule_revision, 8);
        assert_eq!(module.device_caps.compute_capability, (12, 0));
        assert_eq!(module.device_caps.nvrtc_version, (12, 8));
        assert_eq!(
            module
                .device_caps
                .accepted_target
                .expect("accepted CUDA 12.8 target")
                .as_str(),
            "compute_120"
        );
        assert_eq!(module.device_caps.optin_shared_bytes, 101_376);
        assert!(module.device_caps.tensor_map_access);
    }

    #[test]
    fn sm120_tf32_cuda_12_8_route_manifest_is_literal_and_complete() {
        let cohort = sm120_retired_cohort((12, 8));
        assert_sm120_route_manifest(
            cohort.cells,
            [
                (
                    ResolvedGemmOp::Nn,
                    2048,
                    3072,
                    768,
                    Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M64N128, Tf32Sm120Stages::S2),
                ),
                (
                    ResolvedGemmOp::Tn,
                    768,
                    3072,
                    2048,
                    Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M64N128, Tf32Sm120Stages::S2),
                ),
                (
                    ResolvedGemmOp::Nt,
                    2048,
                    768,
                    3072,
                    Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M64N64, Tf32Sm120Stages::S2),
                ),
                (
                    ResolvedGemmOp::Nn,
                    2048,
                    768,
                    1536,
                    Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M64N64, Tf32Sm120Stages::S2),
                ),
                (
                    ResolvedGemmOp::Tn,
                    1536,
                    768,
                    2048,
                    Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M64N128, Tf32Sm120Stages::S4),
                ),
                (
                    ResolvedGemmOp::Nt,
                    2048,
                    1536,
                    768,
                    Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M64N64, Tf32Sm120Stages::S2),
                ),
                (
                    ResolvedGemmOp::Nn,
                    4621,
                    1928,
                    384,
                    Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M128N64, Tf32Sm120Stages::S2),
                ),
                (
                    ResolvedGemmOp::Tn,
                    384,
                    1928,
                    4621,
                    Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M64N128, Tf32Sm120Stages::S4),
                ),
                (
                    ResolvedGemmOp::Nt,
                    4621,
                    384,
                    1928,
                    Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M64N64, Tf32Sm120Stages::S2),
                ),
            ],
        );
        for cell in cohort.cells {
            let request = normalized_request(
                cell.op,
                cell.shape.output_rows,
                cell.shape.output_columns,
                cell.shape.reduction,
            );
            let operands = F32TriadOperands {
                output: 0x1000,
                a: 0x2000,
                b: 0x3000,
                bias: None,
                alpha: 1.0,
                beta: if cell.op == ResolvedGemmOp::Tn {
                    1.0
                } else {
                    0.0
                },
            };
            // The record still names the cell; the runtime table no longer
            // holds this cohort, so the resolver declines it.
            assert_eq!(
                measured_tf32_cell(request, operands, cohort.cells),
                Some(cell.route)
            );
            assert_no_tf32_route(
                resolve_f32_triad_auto_with_operands(
                    F32TriadPolicy::AllowDeterministicTf32,
                    request,
                    operands,
                    sm120_availability_for(cohort.identity),
                )
                .unwrap(),
            );
            assert_eq!(
                resolve_f32_triad_auto(
                    F32TriadPolicy::AllowDeterministicTf32,
                    request,
                    sm120_availability_for(cohort.identity),
                )
                .unwrap(),
                F32TriadSelection::ScalarFma,
            );
        }
    }

    #[test]
    fn sm120_tf32_cuda_13_0_identity_matches_the_literal_qualification_manifest() {
        let identity = sm120_retired_cohort((13, 0)).identity;
        assert_eq!(identity.module_kind, ModuleKind::TriadSm120);
        assert_eq!(identity.module_target, "compute_120");
        assert_eq!(identity.device_target, "sm_120");
        assert_eq!(identity.compute_capability, (12, 0));
        assert_eq!(identity.multiprocessor_count, 170);
        assert_eq!(identity.nvrtc_version, (13, 0));
        assert_eq!(identity.optin_shared_bytes, 101_376);
        assert!(identity.tensor_map_access);
        assert_eq!(
            digest_hex(&identity.compile_key),
            "da248ae349b8ebddf6ede70b276cd1bfa620a7ddfba93fc922517f8688a6474f"
        );
        assert_eq!(
            digest_hex(&identity.artifact_digest),
            "51816c8d7906196d49aa6807742c0e93ac1732f23a96472eaab97dc446efb5b8"
        );
        assert_eq!(
            digest_hex(&identity.source_digest),
            "246d2e35eb059cd696ec74325d4a223312fa605fe84ff58f17f6638d99b8182c"
        );
        assert_eq!(
            digest_hex(&identity.invocation_digest),
            "da248ae349b8ebddf6ede70b276cd1bfa620a7ddfba93fc922517f8688a6474f"
        );
        assert_eq!(
            digest_hex(&identity.header_manifest_digest),
            "f4fff8418bd2c346c86ec6f07f5e7417147c013bf038cdc16df3e76cf3f80ad9"
        );
        assert_eq!(
            digest_hex(&identity.nvrtc_library_domain),
            "709b91c36bfb0ed966ee69adc8d6f87ff110eecf3dfb5060367f183ce614eb0d"
        );

        let module = qualified_module_for_auto_identity(identity);
        assert_eq!(module.artifact.module_kind, ModuleKind::TriadSm120);
        assert_eq!(module.artifact.artifact_kind, ArtifactKind::Ptx);
        assert_eq!(module.compiler.target.as_str(), "compute_120");
        assert!(module.compiler.nvrtc_library_known);
        assert_eq!(module.compiler.output_kind, ArtifactKind::Ptx);
        assert_eq!(module.compiler.composer_revision, 1);
        assert_eq!(module.compiler.compiler_revision, COMPILER_REVISION);
        assert_eq!(module.compiler.numeric_abi_revision, 5);
        assert_eq!(module.compiler.schedule_revision, 8);
        assert_eq!(module.device_caps.compute_capability, (12, 0));
        assert_eq!(module.device_caps.nvrtc_version, (13, 0));
        assert_eq!(
            module
                .device_caps
                .accepted_target
                .expect("accepted CUDA 13.0 target")
                .as_str(),
            "compute_120"
        );
        assert_eq!(module.device_caps.optin_shared_bytes, 101_376);
        assert!(module.device_caps.tensor_map_access);
    }

    #[test]
    fn sm120_tf32_cuda_13_0_route_manifest_is_literal_and_complete() {
        let cohort = sm120_retired_cohort((13, 0));
        assert_sm120_route_manifest(
            cohort.cells,
            [
                (
                    ResolvedGemmOp::Nn,
                    2048,
                    3072,
                    768,
                    Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M128N64, Tf32Sm120Stages::S2),
                ),
                (
                    ResolvedGemmOp::Tn,
                    768,
                    3072,
                    2048,
                    Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M128N64, Tf32Sm120Stages::S2),
                ),
                (
                    ResolvedGemmOp::Nt,
                    2048,
                    768,
                    3072,
                    Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M64N64, Tf32Sm120Stages::S2),
                ),
                (
                    ResolvedGemmOp::Nn,
                    2048,
                    768,
                    1536,
                    Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M64N64, Tf32Sm120Stages::S2),
                ),
                (
                    ResolvedGemmOp::Tn,
                    1536,
                    768,
                    2048,
                    Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M128N64, Tf32Sm120Stages::S2),
                ),
                (
                    ResolvedGemmOp::Nt,
                    2048,
                    1536,
                    768,
                    Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M64N64, Tf32Sm120Stages::S2),
                ),
                (
                    ResolvedGemmOp::Nn,
                    4621,
                    1928,
                    384,
                    Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M64N64, Tf32Sm120Stages::S2),
                ),
                (
                    ResolvedGemmOp::Tn,
                    384,
                    1928,
                    4621,
                    Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M128N64, Tf32Sm120Stages::S2),
                ),
                (
                    ResolvedGemmOp::Nt,
                    4621,
                    384,
                    1928,
                    Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M64N64, Tf32Sm120Stages::S2),
                ),
            ],
        );
        for cell in cohort.cells {
            let request = normalized_request(
                cell.op,
                cell.shape.output_rows,
                cell.shape.output_columns,
                cell.shape.reduction,
            );
            let operands = F32TriadOperands {
                output: 0x1000,
                a: 0x2000,
                b: 0x3000,
                bias: None,
                alpha: 1.0,
                beta: if cell.op == ResolvedGemmOp::Tn {
                    1.0
                } else {
                    0.0
                },
            };
            // The record still names the cell; the runtime table no longer
            // holds this cohort, so the resolver declines it.
            assert_eq!(
                measured_tf32_cell(request, operands, cohort.cells),
                Some(cell.route)
            );
            assert_no_tf32_route(
                resolve_f32_triad_auto_with_operands(
                    F32TriadPolicy::AllowDeterministicTf32,
                    request,
                    operands,
                    sm120_availability_for(cohort.identity),
                )
                .unwrap(),
            );
            assert_eq!(
                resolve_f32_triad_auto(
                    F32TriadPolicy::AllowDeterministicTf32,
                    request,
                    sm120_availability_for(cohort.identity),
                )
                .unwrap(),
                F32TriadSelection::ScalarFma,
            );
        }
    }

    #[test]
    fn sm120_tf32_cuda_13_0_all_nine_routes_fail_closed_on_request_or_operand_drift() {
        let cohort = sm120_retired_cohort((13, 0));
        assert_eq!(cohort.cells.len(), 9);
        for cell in cohort.cells {
            let request = normalized_request(
                cell.op,
                cell.shape.output_rows,
                cell.shape.output_columns,
                cell.shape.reduction,
            );
            let operands = F32TriadOperands {
                output: 0x1000,
                a: 0x2000,
                b: 0x3000,
                bias: None,
                alpha: 1.0,
                beta: if cell.op == ResolvedGemmOp::Tn {
                    1.0
                } else {
                    0.0
                },
            };

            for (axis, value) in [
                cell.shape.output_rows,
                cell.shape.output_columns,
                cell.shape.reduction,
            ]
            .into_iter()
            .enumerate()
            {
                for changed in [value - 1, value + 1] {
                    let mut shape = [
                        cell.shape.output_rows,
                        cell.shape.output_columns,
                        cell.shape.reduction,
                    ];
                    shape[axis] = changed;
                    assert_no_tf32_route(
                        resolve_f32_triad_auto_with_operands(
                            F32TriadPolicy::AllowDeterministicTf32,
                            normalized_request(cell.op, shape[0], shape[1], shape[2]),
                            operands,
                            sm120_availability_for(cohort.identity),
                        )
                        .unwrap(),
                    );
                }
            }

            for stride in 0..3 {
                let mut noncontiguous = request;
                match stride {
                    0 => noncontiguous.shape.lda += 1,
                    1 => noncontiguous.shape.ldb += 1,
                    _ => noncontiguous.shape.ldc += 1,
                }
                assert_no_tf32_route(
                    resolve_f32_triad_auto_with_operands(
                        F32TriadPolicy::AllowDeterministicTf32,
                        noncontiguous,
                        operands,
                        sm120_availability_for(cohort.identity),
                    )
                    .unwrap(),
                );
            }

            for rejected in [
                F32TriadOperands {
                    output: 0,
                    ..operands
                },
                F32TriadOperands { a: 0, ..operands },
                F32TriadOperands { b: 0, ..operands },
                F32TriadOperands {
                    output: 0x1004,
                    ..operands
                },
                F32TriadOperands {
                    a: 0x2004,
                    ..operands
                },
                F32TriadOperands {
                    b: 0x3004,
                    ..operands
                },
                F32TriadOperands {
                    bias: Some(0x4000),
                    ..operands
                },
                F32TriadOperands {
                    alpha: 0.5,
                    ..operands
                },
                F32TriadOperands {
                    beta: if cell.op == ResolvedGemmOp::Tn {
                        0.0
                    } else {
                        1.0
                    },
                    ..operands
                },
            ] {
                assert_no_tf32_route(
                    resolve_f32_triad_auto_with_operands(
                        F32TriadPolicy::AllowDeterministicTf32,
                        request,
                        rejected,
                        sm120_availability_for(cohort.identity),
                    )
                    .unwrap(),
                );
            }

            // The exact policy owns the exact-F32 SM120 family: a measured
            // cell resolves to its qualified arm whatever the TF32
            // qualification identity says, everything else stays scalar.
            assert_eq!(
                resolve_f32_triad_auto_with_operands(
                    F32TriadPolicy::ExactScalarFma,
                    request,
                    operands,
                    sm120_availability_for(cohort.identity),
                )
                .unwrap(),
                expected_exact_selection(
                    request,
                    operands,
                    sm120_availability_for(cohort.identity)
                ),
            );
        }
    }

    /// The exact-policy selection the measured-cell table implies: the
    /// official arm for a measured shape on a bound CC 12.0 module with
    /// tensor-map-aligned operands, otherwise the scalar routes.
    fn expected_exact_selection(
        request: F32TriadRequest,
        operands: F32TriadOperands,
        availability: F32TriadAvailability,
    ) -> F32TriadSelection {
        super::sm120_fma_exact_route(request, operands, availability).map_or(
            F32TriadSelection::ScalarFma,
            F32TriadSelection::ExactSm120Fma,
        )
    }

    #[test]
    fn exact_policy_measured_cells_resolve_to_their_official_arms() {
        let identity = sm120_cohort((13, 2)).identity;
        let availability = sm120_availability_for(identity);
        for cell in super::SM120_FMA_MEASURED_CELLS_CC120_170 {
            let request = F32TriadRequest {
                op: cell.op,
                shape: cell.shape,
            };
            let operands = F32TriadOperands {
                output: 0x1000,
                a: 0x2000,
                b: 0x3000,
                bias: None,
                alpha: 1.0,
                beta: if cell.op == ResolvedGemmOp::Tn {
                    1.0
                } else {
                    0.0
                },
            };
            assert_eq!(
                resolve_f32_triad_auto_with_operands(
                    F32TriadPolicy::ExactScalarFma,
                    request,
                    operands,
                    availability,
                )
                .unwrap(),
                F32TriadSelection::ExactSm120Fma(cell.route),
                "{:?} {:?}",
                cell.op,
                cell.shape
            );
            // A misaligned operand or leading dimension keeps the scalar
            // routes: the tensor maps cannot describe it.
            let misaligned = F32TriadOperands {
                a: 0x2004,
                ..operands
            };
            assert_eq!(
                resolve_f32_triad_auto_with_operands(
                    F32TriadPolicy::ExactScalarFma,
                    request,
                    misaligned,
                    availability,
                )
                .unwrap(),
                F32TriadSelection::ScalarFma,
            );
            // Under the TF32 policy a measured cell takes its TF32 route when
            // the cohort has one and the exact family otherwise; it never
            // falls through to the bare scalar route.
            assert!(!matches!(
                resolve_f32_triad_auto_with_operands(
                    F32TriadPolicy::AllowDeterministicTf32,
                    request,
                    operands,
                    availability,
                )
                .unwrap(),
                F32TriadSelection::ScalarFma
                    | F32TriadSelection::Tf32(super::Tf32PhysicalRoute::Sm120TmaFmaExact(_))
            ));
        }
    }

    fn exact_sm120_availability_with_exclusions(
        excluded: &[(ResolvedGemmOp, Sm120FmaRoute)],
        matching_tf32_cohort: bool,
    ) -> F32TriadAvailability {
        let identity = sm120_cohort((13, 2)).identity;
        let mut availability = sm120_availability_for(identity);
        let specialized = availability.specialized.as_mut().unwrap();
        specialized.sm120_fma_exclusions =
            Sm120FmaExclusions::from_routes(excluded).expect("literal exact-F32 routes");
        if !matching_tf32_cohort {
            specialized.compiler.source_digest = [0xee; 32];
        }
        availability
    }

    fn nt_d768_out_request_and_operands() -> (F32TriadRequest, F32TriadOperands) {
        (
            F32TriadRequest {
                op: ResolvedGemmOp::Nt,
                shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, (2_048, 1_536, 768)),
            },
            F32TriadOperands {
                output: 0x1000,
                a: 0x2000,
                b: 0x3000,
                bias: None,
                alpha: 1.0,
                beta: 0.0,
            },
        )
    }

    #[test]
    fn sm120_exact_reachability_excluded_measured_route_uses_generic_under_both_policies() {
        let measured = Sm120FmaRoute {
            tile: Sm120FmaTile::M64N128,
            kvec: false,
            splits: 2,
        };
        let generic = Sm120FmaRoute {
            tile: Sm120FmaTile::M128N64,
            kvec: true,
            splits: 2,
        };
        let availability =
            exact_sm120_availability_with_exclusions(&[(ResolvedGemmOp::Nt, measured)], false);
        let (request, operands) = nt_d768_out_request_and_operands();

        for policy in [
            F32TriadPolicy::ExactScalarFma,
            F32TriadPolicy::AllowDeterministicTf32,
        ] {
            assert_eq!(
                resolve_f32_triad_auto_with_operands(policy, request, operands, availability)
                    .unwrap(),
                F32TriadSelection::ExactSm120Fma(generic),
                "{policy:?} must skip the excluded measured arm"
            );
        }
    }

    #[test]
    fn sm120_exact_reachability_excluded_measured_and_generic_routes_use_scalar() {
        let measured = Sm120FmaRoute {
            tile: Sm120FmaTile::M64N128,
            kvec: false,
            splits: 2,
        };
        let generic = Sm120FmaRoute {
            tile: Sm120FmaTile::M128N64,
            kvec: true,
            splits: 2,
        };
        let availability = exact_sm120_availability_with_exclusions(
            &[
                (ResolvedGemmOp::Nt, measured),
                (ResolvedGemmOp::Nt, generic),
            ],
            false,
        );
        let (request, operands) = nt_d768_out_request_and_operands();

        assert_eq!(
            resolve_f32_triad_auto_with_operands(
                F32TriadPolicy::ExactScalarFma,
                request,
                operands,
                availability,
            )
            .unwrap(),
            F32TriadSelection::ScalarFma,
        );
    }

    #[test]
    fn sm120_exact_reachability_one_exclusion_preserves_unrelated_measured_routes() {
        let excluded = Sm120FmaRoute {
            tile: Sm120FmaTile::M64N128,
            kvec: false,
            splits: 2,
        };
        let availability =
            exact_sm120_availability_with_exclusions(&[(ResolvedGemmOp::Nt, excluded)], true);
        for (op, dims, expected) in [
            (
                ResolvedGemmOp::Nn,
                (2_048, 1_536, 768),
                Sm120FmaRoute {
                    tile: Sm120FmaTile::M128N64,
                    kvec: false,
                    splits: 4,
                },
            ),
            (
                ResolvedGemmOp::Tn,
                (2_048, 1_536, 768),
                Sm120FmaRoute {
                    tile: Sm120FmaTile::M128N64,
                    kvec: false,
                    splits: 2,
                },
            ),
        ] {
            let request = F32TriadRequest {
                op,
                shape: F32TriadShape::contiguous(op, dims),
            };
            let operands = F32TriadOperands {
                output: 0x1000,
                a: 0x2000,
                b: 0x3000,
                bias: None,
                alpha: 1.0,
                beta: if op == ResolvedGemmOp::Tn { 1.0 } else { 0.0 },
            };
            assert_eq!(
                resolve_f32_triad_auto_with_operands(
                    F32TriadPolicy::ExactScalarFma,
                    request,
                    operands,
                    availability,
                )
                .unwrap(),
                F32TriadSelection::ExactSm120Fma(expected),
            );
        }
    }

    #[test]
    fn sm120_exact_reachability_forced_excluded_route_is_rejected_early() {
        let exact = Sm120FmaRoute {
            tile: Sm120FmaTile::M64N128,
            kvec: false,
            splits: 2,
        };
        let availability =
            exact_sm120_availability_with_exclusions(&[(ResolvedGemmOp::Nt, exact)], true);
        let (request, _) = nt_d768_out_request_and_operands();
        let error = resolve_tf32_forced(
            request,
            availability,
            Tf32PhysicalRoute::Sm120TmaFmaExact(exact),
        )
        .expect_err("a forced excluded exact-F32 symbol must fail before preparation");
        assert!(error.contains("excluded on this toolkit"), "{error}");
        assert!(
            error.contains("nt_sm120_tma_fma_m64n128_bk16_s2"),
            "{error}"
        );
    }

    #[test]
    fn sm120_tf32_cuda_13_2_route_manifest_is_literal_and_complete() {
        let cells = sm120_retired_cohort((13, 2)).cells;
        assert_eq!(
            cells[0],
            super::Tf32AutoCell {
                op: ResolvedGemmOp::Tn,
                shape: super::Tf32ExactShape {
                    output_rows: 512,
                    output_columns: 384,
                    reduction: 256,
                },
                route: Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: Tf32PortableTile::M16N32,
                    stages: Tf32PortableStages::S4,
                }),
                operand_gate: Tf32AutoOperandGate::RequiresVectorAlignmentEvidence,
            }
        );
        assert_sm120_route_manifest(
            &cells[1..],
            [
                (
                    ResolvedGemmOp::Nn,
                    512,
                    768,
                    3072,
                    Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M80N32Bk64, Tf32Sm120Stages::S2),
                ),
                (
                    ResolvedGemmOp::Nn,
                    2048,
                    3072,
                    768,
                    Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M64N128, Tf32Sm120Stages::S2),
                ),
                (
                    ResolvedGemmOp::Tn,
                    768,
                    3072,
                    2048,
                    Sm120ManifestRoute::StreamK(Tf32Sm120Tile::M64N128, Tf32Sm120Stages::S3),
                ),
                (
                    ResolvedGemmOp::Nt,
                    2048,
                    768,
                    3072,
                    Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M64N64, Tf32Sm120Stages::S2),
                ),
                (
                    ResolvedGemmOp::Nn,
                    2048,
                    768,
                    1536,
                    Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M64N64, Tf32Sm120Stages::S2),
                ),
                (
                    ResolvedGemmOp::Tn,
                    1536,
                    768,
                    2048,
                    Sm120ManifestRoute::StreamK(Tf32Sm120Tile::M64N128, Tf32Sm120Stages::S3),
                ),
                (
                    ResolvedGemmOp::Nt,
                    2048,
                    1536,
                    768,
                    Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M64N64, Tf32Sm120Stages::S2),
                ),
                (
                    ResolvedGemmOp::Nn,
                    4621,
                    1928,
                    384,
                    Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M128N64, Tf32Sm120Stages::S2),
                ),
                (
                    ResolvedGemmOp::Tn,
                    384,
                    1928,
                    4621,
                    Sm120ManifestRoute::StreamK(Tf32Sm120Tile::M64N128, Tf32Sm120Stages::S3),
                ),
                (
                    ResolvedGemmOp::Nt,
                    4621,
                    384,
                    1928,
                    Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M64N64, Tf32Sm120Stages::S2),
                ),
            ],
        );
    }

    #[test]
    fn tf32_tn_underfill_qualification_uses_current_tuning_revision() {
        assert_eq!(TUNING_TABLE_REVISION, 46);
        assert_eq!(F32_TF32_TUNING_REVISION, 46);
    }

    #[test]
    fn sm120_tf32_tn_underfill_uses_the_requalified_portable_winner() {
        let cohort = sm120_cohort((13, 2));
        let mut availability = sm120_availability_for(cohort.identity);
        availability.portable =
            Some(qualified_module_for_auto_identity(cohort.portable.expect(
                "the live SM120 cohort binds its measured portable module",
            )));
        let request = normalized_request(ResolvedGemmOp::Tn, 512, 384, 256);
        let operands = F32TriadOperands {
            output: 0x1000,
            a: 0x2000,
            b: 0x3000,
            bias: None,
            alpha: 1.0,
            beta: 1.0,
        };

        assert_eq!(
            resolve_f32_triad_auto_with_operands(
                F32TriadPolicy::AllowDeterministicTf32,
                request,
                operands,
                availability,
            )
            .unwrap(),
            F32TriadSelection::Tf32(Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                tile: Tf32PortableTile::M16N32,
                stages: Tf32PortableStages::S4,
            },)),
        );
    }

    #[test]
    fn sm120_tf32_live_portable_tn_cell_is_exact_and_fail_closed() {
        let cohort = sm120_cohort((13, 2));
        let identity = cohort.identity;
        let request = normalized_request(ResolvedGemmOp::Tn, 128, 512, 1024);
        let operands = F32TriadOperands {
            output: 0x1000,
            a: 0x2000,
            b: 0x3000,
            bias: None,
            alpha: 1.0,
            beta: 1.0,
        };
        let route = Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
            tile: Tf32PortableTile::M16N32,
            stages: Tf32PortableStages::S4,
        });
        let availability = || {
            let mut value = sm120_availability_for(identity);
            value.portable =
                Some(qualified_module_for_auto_identity(cohort.portable.expect(
                    "the live SM120 cohort names the portable module it measured",
                )));
            value
        };
        let resolve = |policy, request, operands, availability| {
            resolve_f32_triad_auto_with_operands(policy, request, operands, availability).unwrap()
        };

        assert_eq!(
            resolve(
                F32TriadPolicy::AllowDeterministicTf32,
                request,
                operands,
                availability(),
            ),
            F32TriadSelection::Tf32(route),
        );
        assert_eq!(
            resolve(
                F32TriadPolicy::ExactScalarFma,
                request,
                operands,
                availability(),
            ),
            F32TriadSelection::ScalarFma,
        );

        // A shape the SM120 cell does not name exactly never takes the
        // SM120 kernel; it takes the common tier (a portable tile, or a
        // proof against one) or the exact family, as on every board.
        for drifted in [
            normalized_request(ResolvedGemmOp::Tn, 127, 512, 1024),
            normalized_request(ResolvedGemmOp::Tn, 129, 512, 1024),
            normalized_request(ResolvedGemmOp::Tn, 128, 511, 1024),
            normalized_request(ResolvedGemmOp::Tn, 128, 513, 1024),
            normalized_request(ResolvedGemmOp::Tn, 128, 512, 1023),
            normalized_request(ResolvedGemmOp::Tn, 128, 512, 1025),
            normalized_request(ResolvedGemmOp::Nn, 128, 512, 1024),
            normalized_request(ResolvedGemmOp::Nt, 128, 512, 1024),
        ] {
            assert_no_tf32_route(resolve(
                F32TriadPolicy::AllowDeterministicTf32,
                drifted,
                operands,
                availability(),
            ));
        }
        for mutate in [
            |shape: &mut F32TriadShape| shape.lda += 1,
            |shape: &mut F32TriadShape| shape.ldb += 1,
            |shape: &mut F32TriadShape| shape.ldc += 1,
        ] {
            let mut rejected = request;
            mutate(&mut rejected.shape);
            assert_eq!(
                resolve(
                    F32TriadPolicy::AllowDeterministicTf32,
                    rejected,
                    operands,
                    availability(),
                ),
                F32TriadSelection::ScalarFma,
            );
        }
        for rejected in [
            F32TriadOperands {
                output: 0x1004,
                ..operands
            },
            F32TriadOperands {
                a: 0x2004,
                ..operands
            },
            F32TriadOperands {
                b: 0x3004,
                ..operands
            },
            F32TriadOperands {
                bias: Some(0x4000),
                ..operands
            },
            F32TriadOperands {
                alpha: -0.0,
                ..operands
            },
            F32TriadOperands {
                beta: 0.0,
                ..operands
            },
        ] {
            assert_eq!(
                resolve(
                    F32TriadPolicy::AllowDeterministicTf32,
                    request,
                    rejected,
                    availability(),
                ),
                F32TriadSelection::ScalarFma,
            );
        }
        let mut unavailable = availability();
        unavailable.portable = None;
        assert_eq!(
            resolve(
                F32TriadPolicy::AllowDeterministicTf32,
                request,
                operands,
                unavailable,
            ),
            F32TriadSelection::ScalarFma,
        );
        // A drifted specialized identity is no evidence: the portable
        // tier may serve the cell by design, a specialized route never.
        for mutate in sm120_identity_mutations() {
            let mut rejected = availability();
            mutate(rejected.specialized.as_mut().unwrap());
            let selection = resolve(
                F32TriadPolicy::AllowDeterministicTf32,
                request,
                operands,
                rejected,
            );
            assert!(
                matches!(selection, F32TriadSelection::ScalarFma)
                    || matches!(
                        selection,
                        F32TriadSelection::Tf32(route) if route.module_kind() == ModuleKind::TriadSm80
                    ),
                "drifted specialized identity selected {selection:?}"
            );
        }
        for (specialized, portable) in [
            (
                SM120_TF32_QUALIFICATION_IDENTITY_CUDA_12_8_DRIVER_595_84,
                SM120_TF32_PORTABLE_QUALIFICATION_IDENTITY_CUDA_12_8_DRIVER_595_84,
            ),
            (
                SM120_TF32_QUALIFICATION_IDENTITY_CUDA_13_0_DRIVER_595_84,
                SM120_TF32_PORTABLE_QUALIFICATION_IDENTITY_CUDA_13_0_DRIVER_595_84,
            ),
            (
                SM120_TF32_QUALIFICATION_IDENTITY_CUDA_13_2_DRIVER_595_84,
                SM120_TF32_PORTABLE_QUALIFICATION_IDENTITY_CUDA_13_2_DRIVER_595_84,
            ),
        ] {
            let availability = || {
                let mut value = sm120_availability_for(specialized);
                value.portable = Some(qualified_module_for_auto_identity(portable));
                value
            };
            assert_eq!(
                resolve(
                    F32TriadPolicy::AllowDeterministicTf32,
                    request,
                    operands,
                    availability(),
                ),
                F32TriadSelection::Tf32(route),
                "CUDA {:?} exact portable twin",
                specialized.nvrtc_version,
            );
            // A mismatched portable twin is no evidence for the cohort's
            // route; the portable tier may still serve the cell by design
            // where the twin's structure is sound.
            for mutate in sm120_identity_mutations()
                .into_iter()
                .chain(portable_sm120_coupled_mutations())
            {
                let mut rejected = availability();
                mutate(rejected.portable.as_mut().unwrap());
                let selection = resolve(
                    F32TriadPolicy::AllowDeterministicTf32,
                    request,
                    operands,
                    rejected,
                );
                assert!(
                    matches!(selection, F32TriadSelection::ScalarFma)
                        || matches!(
                            selection,
                            F32TriadSelection::Tf32(route)
                                if route.module_kind() == ModuleKind::TriadSm80
                        ),
                    "CUDA {:?} mismatched portable twin selected {selection:?}",
                    specialized.nvrtc_version,
                );
            }
        }
    }

    #[test]
    fn sm120_tf32_live_nn_cell_is_exact_and_fail_closed() {
        let cohort = sm120_cohort((13, 2));
        let request = normalized_request(ResolvedGemmOp::Nn, 2048, 768, 1536);
        let operands = F32TriadOperands {
            output: 0x1000,
            a: 0x2000,
            b: 0x3000,
            bias: None,
            alpha: 1.0,
            beta: 0.0,
        };
        let route = Tf32PhysicalRoute::Sm120TmaMmaTf32Rna(Tf32Sm120Route {
            tile: Tf32Sm120Tile::M64N64,
            stages: Tf32Sm120Stages::S2,
        });
        let resolve = |policy, request, operands, availability| {
            resolve_f32_triad_auto_with_operands(policy, request, operands, availability).unwrap()
        };

        assert_eq!(
            resolve(
                F32TriadPolicy::AllowDeterministicTf32,
                request,
                operands,
                sm120_availability_for(cohort.identity),
            ),
            F32TriadSelection::Tf32(route),
        );
        // The request-only API carries no operand evidence and never takes a
        // measured cell.
        assert_no_tf32_route(
            resolve_f32_triad_auto(
                F32TriadPolicy::AllowDeterministicTf32,
                request,
                sm120_availability_for(cohort.identity),
            )
            .unwrap(),
        );

        for (name, rejected) in [
            (
                "M-1",
                normalized_request(ResolvedGemmOp::Nn, 2047, 768, 1536),
            ),
            (
                "M+1",
                normalized_request(ResolvedGemmOp::Nn, 2049, 768, 1536),
            ),
            (
                "N-1",
                normalized_request(ResolvedGemmOp::Nn, 2048, 767, 1536),
            ),
            (
                "N+1",
                normalized_request(ResolvedGemmOp::Nn, 2048, 769, 1536),
            ),
            (
                "K-1",
                normalized_request(ResolvedGemmOp::Nn, 2048, 768, 1535),
            ),
            (
                "K+1",
                normalized_request(ResolvedGemmOp::Nn, 2048, 768, 1537),
            ),
            (
                "TN",
                normalized_request(ResolvedGemmOp::Tn, 2048, 768, 1536),
            ),
            (
                "NT",
                normalized_request(ResolvedGemmOp::Nt, 2048, 768, 1536),
            ),
        ] {
            assert_no_tf32_route_for(
                resolve(
                    F32TriadPolicy::AllowDeterministicTf32,
                    rejected,
                    operands,
                    sm120_availability_for(cohort.identity),
                ),
                &format!("live NN cell {name} mutation"),
            );
        }

        for (name, rejected) in [
            (
                "null output",
                F32TriadOperands {
                    output: 0,
                    ..operands
                },
            ),
            ("null A", F32TriadOperands { a: 0, ..operands }),
            ("null B", F32TriadOperands { b: 0, ..operands }),
            (
                "output alignment",
                F32TriadOperands {
                    output: 0x1004,
                    ..operands
                },
            ),
            (
                "A alignment",
                F32TriadOperands {
                    a: 0x2004,
                    ..operands
                },
            ),
            (
                "B alignment",
                F32TriadOperands {
                    b: 0x3004,
                    ..operands
                },
            ),
            (
                "alpha",
                F32TriadOperands {
                    alpha: 0.5,
                    ..operands
                },
            ),
            (
                "beta sign",
                F32TriadOperands {
                    beta: -0.0,
                    ..operands
                },
            ),
            (
                "bias",
                F32TriadOperands {
                    bias: Some(0x4000),
                    ..operands
                },
            ),
        ] {
            assert_no_tf32_route_for(
                resolve(
                    F32TriadPolicy::AllowDeterministicTf32,
                    request,
                    rejected,
                    sm120_availability_for(cohort.identity),
                ),
                &format!("live NN cell {name} operand"),
            );
        }
        assert_no_tf32_route(resolve(
            F32TriadPolicy::ExactScalarFma,
            request,
            operands,
            sm120_availability_for(cohort.identity),
        ));

        for (field, mutate) in sm120_identity_mutations().into_iter().enumerate() {
            let mut availability = sm120_availability_for(cohort.identity);
            mutate(availability.specialized.as_mut().unwrap());
            assert_no_tf32_route_for(
                resolve(
                    F32TriadPolicy::AllowDeterministicTf32,
                    request,
                    operands,
                    availability,
                ),
                &format!("identity field {field}"),
            );
        }
        for version in [(12, 8), (13, 0)] {
            assert_no_tf32_route(resolve(
                F32TriadPolicy::AllowDeterministicTf32,
                request,
                operands,
                sm120_availability_for(sm120_retired_cohort(version).identity),
            ));
        }
    }

    #[test]
    fn sm120_tf32_cuda_12_8_identity_rejects_all_29_single_field_mutations() {
        let cohort = sm120_retired_cohort((12, 8));
        let exact = qualified_module_for_auto_identity(cohort.identity);
        assert_eq!(
            matching_tf32_cohort(exact, SM120_TF32_RETIRED_COHORTS),
            Some(&cohort),
        );
        for (field, mutate) in sm120_identity_mutations().into_iter().enumerate() {
            let mut module = exact;
            mutate(&mut module);
            assert!(
                matching_tf32_cohort(module, SM120_TF32_RETIRED_COHORTS).is_none(),
                "12.8 identity mutation {field} was admitted",
            );
        }
    }

    #[test]
    fn sm120_tf32_cuda_13_0_identity_rejects_all_29_single_field_mutations() {
        let cohort = sm120_retired_cohort((13, 0));
        let exact = qualified_module_for_auto_identity(cohort.identity);
        assert_eq!(
            matching_tf32_cohort(exact, SM120_TF32_RETIRED_COHORTS),
            Some(&cohort),
        );
        for (field, mutate) in sm120_identity_mutations().into_iter().enumerate() {
            let mut module = exact;
            mutate(&mut module);
            assert!(
                matching_tf32_cohort(module, SM120_TF32_RETIRED_COHORTS).is_none(),
                "13.0 identity mutation {field} was admitted",
            );
        }
    }

    #[test]
    fn sm120_tf32_real_cohorts_bind_routes_and_reject_cross_version_splices() {
        let cuda_12_8 = sm120_cohort((12, 8));
        let cuda_13_0 = sm120_cohort((13, 0));
        let cuda_13_2 = sm120_cohort((13, 2));
        let request = normalized_request(ResolvedGemmOp::Tn, 768, 3072, 2048);
        let operands = F32TriadOperands {
            output: 0x1000,
            a: 0x2000,
            b: 0x3000,
            bias: None,
            alpha: 1.0,
            beta: 1.0,
        };
        assert_eq!(
            measured_tf32_cell(request, operands, cuda_12_8.cells,),
            Some(Tf32PhysicalRoute::Sm120TmaMmaTf32RnaStreamKV1(
                Tf32Sm120Route {
                    tile: Tf32Sm120Tile::M64N128,
                    stages: Tf32Sm120Stages::S3,
                },
            )),
        );
        assert_eq!(
            measured_tf32_cell(request, operands, cuda_13_0.cells,),
            Some(Tf32PhysicalRoute::Sm120TmaMmaTf32RnaStreamKV1(
                Tf32Sm120Route {
                    tile: Tf32Sm120Tile::M64N128,
                    stages: Tf32Sm120Stages::S3,
                },
            )),
        );
        assert_eq!(
            measured_tf32_cell(request, operands, cuda_13_2.cells,),
            Some(Tf32PhysicalRoute::Sm120TmaMmaTf32RnaStreamKV1(
                Tf32Sm120Route {
                    tile: Tf32Sm120Tile::M64N128,
                    stages: Tf32Sm120Stages::S3,
                },
            )),
        );

        for live in [cuda_12_8, cuda_13_0, cuda_13_2] {
            assert_eq!(
                resolve_f32_triad_auto_with_operands(
                    F32TriadPolicy::AllowDeterministicTf32,
                    request,
                    operands,
                    sm120_availability_for(live.identity),
                )
                .unwrap(),
                F32TriadSelection::Tf32(Tf32PhysicalRoute::Sm120TmaMmaTf32RnaStreamKV1(
                    Tf32Sm120Route {
                        tile: Tf32Sm120Tile::M64N128,
                        stages: Tf32Sm120Stages::S3,
                    },
                )),
            );
        }

        // Historical lower-toolkit records retain their old manifests but
        // remain unreachable because only fresh current-source identities are live.
        for retired in [sm120_retired_cohort((12, 8)), sm120_retired_cohort((13, 0))] {
            assert_no_tf32_route(
                resolve_f32_triad_auto_with_operands(
                    F32TriadPolicy::AllowDeterministicTf32,
                    request,
                    operands,
                    sm120_availability_for(retired.identity),
                )
                .unwrap(),
            );
        }
        let cohorts = SM120_TF32_EVIDENCE_COHORTS
            .iter()
            .chain(SM120_TF32_RETIRED_COHORTS)
            .copied()
            .collect::<Vec<_>>();
        for source in cohorts.iter().copied() {
            for destination in cohorts.iter().copied() {
                if source.identity.nvrtc_version == destination.identity.nvrtc_version {
                    continue;
                }
                let mut spliced = qualified_module_for_auto_identity(source.identity);
                spliced.compiler.nvrtc_version = destination.identity.nvrtc_version;
                spliced.device_caps.nvrtc_version = destination.identity.nvrtc_version;
                assert!(
                    matching_tf32_cohort(spliced, SM120_TF32_EVIDENCE_COHORTS).is_none(),
                    "CUDA {:?} body admitted with CUDA {:?} versions",
                    source.identity.nvrtc_version,
                    destination.identity.nvrtc_version,
                );
            }
        }
    }

    /// A cohort is frozen against the module source it was measured on. Editing
    /// any kernel of that module changes the source digest, and every cohort
    /// whose digest no longer matches is silently unreachable: the TF32 policy
    /// resolves nothing and falls through to the exact floor with no error.
    /// At least one cohort has to still describe this tree, or the whole
    /// deterministic TF32 family is dead code until someone requalifies it.
    #[test]
    fn sm120_tf32_live_cohort_manifest_is_exact_and_unique() {
        let cohorts = SM120_TF32_EVIDENCE_COHORTS.iter().collect::<Vec<_>>();
        // CUDA 13.2 is frozen twice: once per NVRTC library build the
        // boards carry (13.2.51 on the Ada box, 13.2.78 on the rented
        // RTX 5090s). The library domain tells them apart.
        assert_eq!(
            cohorts
                .iter()
                .map(|cohort| cohort.identity.nvrtc_version)
                .collect::<Vec<_>>(),
            [(12, 8), (13, 0), (13, 2), (13, 2)]
        );
        let (first, second) = (cohorts[2].identity, cohorts[3].identity);
        assert_ne!(first.nvrtc_library_domain, second.nvrtc_library_domain);
        assert_eq!(first.source_digest, second.source_digest);
        assert_eq!(first.header_manifest_digest, second.header_manifest_digest);
        for cohort in cohorts {
            assert_eq!(
                cohort.cells,
                SM120_TF32_EVIDENCE_CELLS_DRIVER_595_84_RETAINED
            );
            assert_eq!(cohort.cells.len(), 23);
            let mut keys = cohort
                .cells
                .iter()
                .map(|cell| (cell.op, cell.shape))
                .collect::<Vec<_>>();
            keys.sort_unstable_by_key(|(op, shape)| {
                (
                    *op as u8,
                    shape.output_rows,
                    shape.output_columns,
                    shape.reduction,
                )
            });
            keys.dedup();
            assert_eq!(keys.len(), cohort.cells.len(), "a shape is measured twice");
            assert!(!cohort.cells.iter().any(|cell| {
                cell.op == ResolvedGemmOp::Tn
                    && cell.shape.output_rows == 128
                    && cell.shape.output_columns == 128
                    && cell.shape.reduction == 8192
            }));
            assert_eq!(
                cohort
                    .cells
                    .iter()
                    .filter(|cell| cell.route.module_kind() == ModuleKind::TriadSm80)
                    .count(),
                6
            );
            assert!(cohort.portable.is_some());
        }
    }

    #[test]
    fn sm120_tf32_lower_toolkit_595_84_identities_are_literal_and_live() {
        fn digest(hex: &str) -> [u8; 32] {
            assert_eq!(hex.len(), 64);
            std::array::from_fn(|index| {
                u8::from_str_radix(&hex[index * 2..index * 2 + 2], 16).unwrap()
            })
        }
        type LowerToolkitCase = (
            Tf32AutoQualificationIdentity,
            Tf32AutoQualificationIdentity,
            (i32, i32),
            &'static str,
            &'static str,
            &'static str,
            &'static str,
            &'static str,
            &'static str,
            &'static str,
        );
        let cases: [LowerToolkitCase; 2] = [
            (
                SM120_TF32_QUALIFICATION_IDENTITY_CUDA_12_8_DRIVER_595_84,
                SM120_TF32_PORTABLE_QUALIFICATION_IDENTITY_CUDA_12_8_DRIVER_595_84,
                (12, 8),
                "84c04b66ae28bbf795dadf0f8d88310603e09e5ffeead37883fd7d15f46a4c53",
                "ab85299f352f62cf315322013afbfdedb7bd033d0ebd9b70f60a908a2c2cf4e9",
                "6b69d5f958a1b85e8fb5cac65a3bae34a6f5444a757925e5f2f9409e2e0392c8",
                "c791a02540277febb6ef58ddf515324f8163c109ba7151a43e83a304cc99eeb4",
                "3e11e78c1f986584eb03afaec00a210005e67097a2f513e7a96408b80c95253f",
                "070597b13d1c8240efdd8d3226ae93db491e15d5250ef2cdef8eae2b2bd2c276",
                "26b0a3a02044ffcbc1693fd83e9261beffa692a4fbcfe3ac5e9d8c87980bb155",
            ),
            (
                SM120_TF32_QUALIFICATION_IDENTITY_CUDA_13_0_DRIVER_595_84,
                SM120_TF32_PORTABLE_QUALIFICATION_IDENTITY_CUDA_13_0_DRIVER_595_84,
                (13, 0),
                "b10da2e57112320cb14c926321677c8aae800dd2b74ecb7f4fd8f3eed693ff28",
                "0b11b9aa8464d7cc22c817b575edd60c9444633d2d6092a0f6c4b9cce76e5337",
                "b938e17359bf94d4d186419b4ad13c1f9e63d8c24361af4cfc3ba91e3753ceda",
                "4ee15778bed8d952f138f429088d46232d8a99b49f3dcf4c1513dfa44b27b230",
                "c2e22af293d5e148607f793ce8bc2f93afce3affcae4a1d45881ac87019f15e5",
                "d75635780905da9422a5c4b0a5b1cc643505a18c8e188c1f7dd6f17e90aa1a80",
                "709b91c36bfb0ed966ee69adc8d6f87ff110eecf3dfb5060367f183ce614eb0d",
            ),
        ];
        for (
            specialized,
            portable,
            version,
            specialized_key,
            specialized_artifact,
            specialized_header,
            portable_key,
            portable_artifact,
            portable_header,
            nvrtc_domain,
        ) in cases
        {
            assert_eq!(specialized.module_kind, ModuleKind::TriadSm120);
            assert_eq!(portable.module_kind, ModuleKind::TriadSm80);
            for identity in [specialized, portable] {
                assert_eq!(identity.module_target, "compute_120");
                assert_eq!(identity.device_target, "sm_120");
                assert_eq!(identity.compute_capability, (12, 0));
                assert_eq!(identity.multiprocessor_count, 170);
                assert_eq!(identity.nvrtc_version, version);
                assert_eq!(identity.optin_shared_bytes, 101376);
                assert!(identity.tensor_map_access);
                assert!(identity.matches(qualified_module_for_auto_identity(identity)));
            }
            assert_eq!(specialized.compile_key, digest(specialized_key));
            assert_eq!(specialized.artifact_digest, digest(specialized_artifact));
            assert_eq!(specialized.invocation_digest, digest(specialized_key));
            assert_eq!(
                specialized.header_manifest_digest,
                digest(specialized_header)
            );
            assert_eq!(portable.compile_key, digest(portable_key));
            assert_eq!(portable.artifact_digest, digest(portable_artifact));
            assert_eq!(portable.invocation_digest, digest(portable_key));
            assert_eq!(portable.header_manifest_digest, digest(portable_header));
            assert_eq!(specialized.nvrtc_library_domain, digest(nvrtc_domain));
            assert_eq!(portable.nvrtc_library_domain, digest(nvrtc_domain));
            assert_eq!(
                specialized.source_digest,
                digest("248bb7cdfbc750ba0fa5e156edf85010890faa7df4ab916aeb97d8c22bb5d035")
            );
            assert_eq!(
                portable.source_digest,
                digest("f853ab0c4f22c4e212ca1fe526e202337b77fd333e73cebdd4e069004bda771d")
            );
        }
    }

    #[test]
    fn at_least_one_sm120_tf32_cohort_matches_this_tree() {
        let live = super::super::modules::module_source_digest(ModuleKind::TriadSm120, "sm_120a")
            .expect("compose the SM120 module source");
        let stale: Vec<usize> = SM120_TF32_EVIDENCE_COHORTS
            .iter()
            .enumerate()
            .filter(|(_, cohort)| cohort.identity.source_digest != live)
            .map(|(index, _)| index)
            .collect();
        assert!(
            stale.is_empty(),
            "a live SM120 TF32 cohort is frozen against a source this tree no \
             longer contains and can never match a board: requalify it against \
             the current kernels or retire it. Stale cohort indices: {stale:?}",
        );
        assert!(!SM120_TF32_EVIDENCE_COHORTS.is_empty());
        for cohort in SM120_TF32_EVIDENCE_COHORTS {
            if let Some(twin) = cohort.portable {
                assert_eq!(twin.module_kind, ModuleKind::TriadSm80);
                // The portable module a CC 12.x board compiles leaves the
                // sm80 stream-K fragment out, so the twin's source is the
                // composition for its own target.
                let portable_live = super::super::modules::module_source_digest(
                    ModuleKind::TriadSm80,
                    twin.module_target,
                )
                .expect("compose the portable module source");
                assert_eq!(
                    twin.source_digest, portable_live,
                    "a live cohort's portable twin is frozen against a portable source this \
                     tree no longer contains"
                );
            }
            assert_eq!(cohort.tuning_revision, F32_TF32_TUNING_REVISION);
        }
        // A retired cohort is retired for exactly this reason.
        for cohort in SM120_TF32_RETIRED_COHORTS {
            assert_ne!(
                cohort.identity.source_digest, live,
                "a retired SM120 TF32 cohort matches the live source; it belongs in the live table"
            );
        }
    }

    /// The Fixed module is composed from the Mamba kernels and the portable
    /// module carries the stream-K kernel; an edit to any of those sources
    /// moves the module identity these Ada cohorts pin, and the routes they
    /// admit fall back without any test noticing. Hold every Ada cohort to
    /// the source this tree composes so such an edit fails here first.
    #[test]
    fn every_ada_cohort_matches_the_source_this_tree_composes() {
        let live = |kind| {
            super::super::modules::module_source_digest(kind, "sm_89")
                .unwrap_or_else(|error| panic!("compose {kind:?} for sm_89: {error}"))
        };
        let fixed = [
            ((12, 8), 16, FIXED_COPYPLAN_SOURCE_DIGEST_CAP16),
            ((12, 8), 64, FIXED_COPYPLAN_SOURCE_DIGEST_CAP64),
            ((13, 0), 16, FIXED_COPYPLAN_SOURCE_DIGEST_CAP16),
            ((13, 0), 64, FIXED_COPYPLAN_SOURCE_DIGEST_CAP64),
            ((13, 2), 16, FIXED_COPYPLAN_SOURCE_DIGEST_CAP16),
            ((13, 2), 64, FIXED_COPYPLAN_SOURCE_DIGEST_CAP64),
        ];
        assert_eq!(FIXED_COPYPLAN_EVIDENCE_COHORTS.len(), fixed.len());
        for (cohort, (nvrtc, state_cap, expected_source)) in
            FIXED_COPYPLAN_EVIDENCE_COHORTS.iter().zip(fixed)
        {
            assert_eq!(cohort.nvrtc_version, nvrtc);
            assert_eq!(cohort.source_digest, expected_source);
            let digest = super::super::modules::module_source_digest_for_compile(
                ModuleKind::Fixed,
                Some((8, 9)),
                "sm_89",
                state_cap,
                nvrtc,
            )
            .unwrap_or_else(|error| {
                panic!("compose Fixed for CUDA {nvrtc:?} cap{state_cap}: {error}")
            });
            assert_eq!(
                cohort.source_digest, digest,
                "a Fixed copy-plan cohort pins a pre-overlay or stale source"
            );
        }
        let tf32 = [
            (
                "portable",
                ModuleKind::TriadSm80,
                SM89_TF32_EVIDENCE_COHORTS,
            ),
            (
                "joint",
                ModuleKind::TriadSm89Tf32Joint,
                SM89_JOINT_TF32_EVIDENCE_COHORTS,
            ),
            (
                "finalist",
                ModuleKind::TriadSm89Finalist,
                SM89_FINALIST_TF32_EVIDENCE_COHORTS,
            ),
        ];
        for (name, kind, cohorts) in tf32 {
            let digest = live(kind);
            for cohort in cohorts {
                assert_eq!(cohort.identity.module_kind, kind, "{name} cohort module");
                assert_eq!(
                    cohort.identity.source_digest, digest,
                    "an Ada {name} TF32 cohort pins a source this tree no longer composes"
                );
            }
        }
        let exact = live(ModuleKind::TriadSm89ExactF32);
        for cohort in SM89_EXACT_F32_EVIDENCE_COHORTS {
            assert_eq!(
                cohort.source_digest, exact,
                "an Ada exact-f32 cohort is stale"
            );
        }
        let d128 = live(ModuleKind::TriadSm89ExactF32D128);
        for cohort in SM89_EXACT_F32_D128_EVIDENCE_COHORTS {
            assert_eq!(
                cohort.source_digest, d128,
                "an Ada exact-f32 d128 cohort is stale"
            );
        }
    }

    #[test]
    fn sm120_tf32_evidence_cohorts_are_exact_and_unambiguous() {
        assert!(!SM120_TF32_EVIDENCE_COHORTS.is_empty());
        // The retired record holds the three stacks that measured earlier
        // sources: CUDA 12.8, CUDA 13.0 and CUDA 13.2.
        assert_eq!(
            SM120_TF32_RETIRED_COHORTS
                .iter()
                .map(|cohort| cohort.identity.nvrtc_version)
                .collect::<Vec<_>>(),
            vec![(12, 8), (13, 0), (13, 2)]
        );
        // One cohort per NVRTC library build: the driver build is not part
        // of the identity, the compiler build is. CUDA 13.2 carries two.
        assert_eq!(
            SM120_TF32_EVIDENCE_COHORTS
                .iter()
                .filter(|cohort| cohort.identity.nvrtc_version == (13, 2))
                .count(),
            2,
        );
        assert!(
            SM120_TF32_EVIDENCE_COHORTS
                .iter()
                .all(|cohort| cohort.identity.nvrtc_version != (13, 1))
        );
        for (index, cohort) in SM120_TF32_EVIDENCE_COHORTS.iter().enumerate() {
            for other in SM120_TF32_EVIDENCE_COHORTS.iter().skip(index + 1) {
                assert!(
                    cohort.identity.nvrtc_version != other.identity.nvrtc_version
                        || cohort.identity.nvrtc_library_domain
                            != other.identity.nvrtc_library_domain,
                    "cohort {index} shares a toolkit and a compiler build with a later cohort",
                );
            }
        }
        for (index, cohort) in SM120_TF32_EVIDENCE_COHORTS.iter().enumerate() {
            assert_eq!(cohort.identity.module_kind, ModuleKind::TriadSm120);
            assert!(!cohort.cells.is_empty());
            assert_eq!(
                SM120_TF32_EVIDENCE_COHORTS
                    .iter()
                    .filter(|candidate| {
                        candidate
                            .identity
                            .matches(qualified_module_for_auto_identity(cohort.identity))
                    })
                    .count(),
                1,
                "cohort {index} did not match exactly once",
            );
            assert_eq!(
                matching_tf32_cohort(
                    qualified_module_for_auto_identity(cohort.identity),
                    SM120_TF32_EVIDENCE_COHORTS,
                ),
                Some(cohort),
            );
            for other in &SM120_TF32_EVIDENCE_COHORTS[index + 1..] {
                assert_ne!(cohort.identity, other.identity);
            }
        }
    }

    #[test]
    fn sm120_tf32_operand_aware_resolution_admits_only_archived_cells() {
        assert_eq!(SM120_TF32_EVIDENCE_CELLS.len(), 11);
        assert_eq!(
            SM120_TF32_EVIDENCE_CELLS
                .iter()
                .filter(|cell| matches!(
                    cell.operand_gate,
                    Tf32AutoOperandGate::RequiresNoBiasAndVectorAlignmentEvidence
                ))
                .count(),
            4
        );
        assert_eq!(
            SM120_TF32_EVIDENCE_CELLS
                .iter()
                .filter(|cell| matches!(
                    cell.operand_gate,
                    Tf32AutoOperandGate::RequiresVectorAlignmentEvidence
                ))
                .count(),
            7
        );
        let unique_keys = SM120_TF32_EVIDENCE_CELLS
            .iter()
            .map(|cell| {
                (
                    cell.op as u8,
                    cell.shape.output_rows,
                    cell.shape.output_columns,
                    cell.shape.reduction,
                )
            })
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(unique_keys.len(), SM120_TF32_EVIDENCE_CELLS.len());
        let expected = [
            (
                ResolvedGemmOp::Nn,
                512,
                768,
                3072,
                Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M80N32Bk64, Tf32Sm120Stages::S2),
            ),
            (
                ResolvedGemmOp::Nn,
                2048,
                3072,
                768,
                Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M64N128, Tf32Sm120Stages::S2),
            ),
            (
                ResolvedGemmOp::Tn,
                768,
                3072,
                2048,
                Sm120ManifestRoute::StreamK(Tf32Sm120Tile::M64N128, Tf32Sm120Stages::S3),
            ),
            (
                ResolvedGemmOp::Nt,
                2048,
                768,
                3072,
                Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M64N64, Tf32Sm120Stages::S2),
            ),
            (
                ResolvedGemmOp::Nn,
                2048,
                768,
                1536,
                Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M64N64, Tf32Sm120Stages::S2),
            ),
            (
                ResolvedGemmOp::Tn,
                1536,
                768,
                2048,
                Sm120ManifestRoute::StreamK(Tf32Sm120Tile::M64N128, Tf32Sm120Stages::S3),
            ),
            (
                ResolvedGemmOp::Nt,
                2048,
                1536,
                768,
                Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M64N64, Tf32Sm120Stages::S2),
            ),
            (
                ResolvedGemmOp::Nn,
                4621,
                1928,
                384,
                Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M128N64, Tf32Sm120Stages::S2),
            ),
            (
                ResolvedGemmOp::Tn,
                384,
                1928,
                4621,
                Sm120ManifestRoute::StreamK(Tf32Sm120Tile::M64N128, Tf32Sm120Stages::S3),
            ),
            (
                ResolvedGemmOp::Nt,
                4621,
                384,
                1928,
                Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M64N64, Tf32Sm120Stages::S2),
            ),
        ];
        assert_eq!(
            SM120_TF32_EVIDENCE_CELLS[0].route,
            Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                tile: Tf32PortableTile::M16N32,
                stages: Tf32PortableStages::S4,
            })
        );
        for (cell, (op, rows, columns, reduction, route)) in
            SM120_TF32_EVIDENCE_CELLS[1..].iter().zip(expected)
        {
            assert_eq!(cell.op, op);
            assert_eq!(
                cell.shape,
                super::Tf32ExactShape {
                    output_rows: rows,
                    output_columns: columns,
                    reduction,
                }
            );
            assert_eq!(cell.route, route.route());
        }
        for cohort in SM120_TF32_EVIDENCE_COHORTS {
            let mut exact_availability = sm120_availability_for(cohort.identity);
            exact_availability.portable = cohort.portable.map(qualified_module_for_auto_identity);
            for cell in cohort.cells {
                assert!(
                    matches!(
                        (cell.op, cell.operand_gate),
                        (
                            ResolvedGemmOp::Nn,
                            Tf32AutoOperandGate::RequiresNoBiasAndVectorAlignmentEvidence
                        ) | (
                            ResolvedGemmOp::Tn,
                            Tf32AutoOperandGate::RequiresVectorAlignmentEvidence
                        ) | (
                            ResolvedGemmOp::Nt,
                            Tf32AutoOperandGate::RequiresVectorAlignmentEvidence
                        )
                    ),
                    "SM120 cell gate must agree with the strict shared operand predicate: {cell:?}",
                );
                let request = normalized_request(
                    cell.op,
                    cell.shape.output_rows,
                    cell.shape.output_columns,
                    cell.shape.reduction,
                );
                let operands = F32TriadOperands {
                    output: 0x1000,
                    a: 0x2000,
                    b: 0x3000,
                    bias: None,
                    alpha: 1.0,
                    beta: if cell.op == ResolvedGemmOp::Tn {
                        1.0
                    } else {
                        0.0
                    },
                };
                assert_eq!(
                    resolve_f32_triad_auto_with_operands(
                        F32TriadPolicy::AllowDeterministicTf32,
                        request,
                        operands,
                        exact_availability,
                    )
                    .unwrap(),
                    F32TriadSelection::Tf32(cell.route),
                );
                assert_eq!(
                    resolve_f32_triad_auto(
                        F32TriadPolicy::AllowDeterministicTf32,
                        request,
                        exact_availability,
                    )
                    .unwrap(),
                    F32TriadSelection::ScalarFma,
                );

                for (axis, value) in [
                    cell.shape.output_rows,
                    cell.shape.output_columns,
                    cell.shape.reduction,
                ]
                .into_iter()
                .enumerate()
                {
                    for changed in [value - 1, value + 1] {
                        let mut shape = [
                            cell.shape.output_rows,
                            cell.shape.output_columns,
                            cell.shape.reduction,
                        ];
                        shape[axis] = changed;
                        assert_no_tf32_route(
                            resolve_f32_triad_auto_with_operands(
                                F32TriadPolicy::AllowDeterministicTf32,
                                normalized_request(cell.op, shape[0], shape[1], shape[2]),
                                operands,
                                exact_availability,
                            )
                            .unwrap(),
                        );
                    }
                }

                for stride in 0..3 {
                    let mut noncontiguous = request;
                    match stride {
                        0 => noncontiguous.shape.lda += 1,
                        1 => noncontiguous.shape.ldb += 1,
                        _ => noncontiguous.shape.ldc += 1,
                    }
                    assert_no_tf32_route(
                        resolve_f32_triad_auto_with_operands(
                            F32TriadPolicy::AllowDeterministicTf32,
                            noncontiguous,
                            operands,
                            exact_availability,
                        )
                        .unwrap(),
                    );
                }

                for rejected in [
                    F32TriadOperands {
                        output: 0,
                        ..operands
                    },
                    F32TriadOperands { a: 0, ..operands },
                    F32TriadOperands { b: 0, ..operands },
                    F32TriadOperands {
                        output: 0x1004,
                        ..operands
                    },
                    F32TriadOperands {
                        a: 0x2004,
                        ..operands
                    },
                    F32TriadOperands {
                        b: 0x3004,
                        ..operands
                    },
                    F32TriadOperands {
                        bias: Some(0x4000),
                        ..operands
                    },
                    F32TriadOperands {
                        alpha: 0.5,
                        ..operands
                    },
                    F32TriadOperands {
                        beta: if cell.op == ResolvedGemmOp::Tn {
                            0.0
                        } else {
                            1.0
                        },
                        ..operands
                    },
                ] {
                    assert_no_tf32_route(
                        resolve_f32_triad_auto_with_operands(
                            F32TriadPolicy::AllowDeterministicTf32,
                            request,
                            rejected,
                            exact_availability,
                        )
                        .unwrap(),
                    );
                }
            }

            let unmeasured = normalized_request(ResolvedGemmOp::Tn, 385, 1928, 4621);
            let operands = F32TriadOperands {
                output: 0x1000,
                a: 0x2000,
                b: 0x3000,
                bias: None,
                alpha: 1.0,
                beta: 1.0,
            };
            assert_no_tf32_route(
                resolve_f32_triad_auto_with_operands(
                    F32TriadPolicy::AllowDeterministicTf32,
                    unmeasured,
                    operands,
                    exact_availability,
                )
                .unwrap(),
            );

            let measured = normalized_request(ResolvedGemmOp::Tn, 384, 1928, 4621);
            assert_eq!(
                resolve_f32_triad_auto_with_operands(
                    F32TriadPolicy::ExactScalarFma,
                    measured,
                    operands,
                    exact_availability,
                )
                .unwrap(),
                expected_exact_selection(measured, operands, exact_availability),
            );

            for mutate in sm120_identity_mutations() {
                let mut availability = exact_availability;
                mutate(availability.specialized.as_mut().unwrap());
                assert_no_tf32_route(
                    resolve_f32_triad_auto_with_operands(
                        F32TriadPolicy::AllowDeterministicTf32,
                        measured,
                        operands,
                        availability,
                    )
                    .unwrap(),
                );
            }
            for version in [(12, 8), (13, 0), (13, 1), (13, 2), (13, 3)]
                .into_iter()
                .filter(|version| *version != cohort.identity.nvrtc_version)
            {
                let mut availability = exact_availability;
                let module = availability.specialized.as_mut().unwrap();
                module.compiler.nvrtc_version = version;
                module.device_caps.nvrtc_version = version;
                assert_no_tf32_route(
                    resolve_f32_triad_auto_with_operands(
                        F32TriadPolicy::AllowDeterministicTf32,
                        measured,
                        operands,
                        availability,
                    )
                    .unwrap(),
                );
            }
            for multiprocessors in [169, 171] {
                let mut availability = exact_availability;
                availability
                    .specialized
                    .as_mut()
                    .unwrap()
                    .device
                    .multiprocessor_count = multiprocessors;
                assert_no_tf32_route(
                    resolve_f32_triad_auto_with_operands(
                        F32TriadPolicy::AllowDeterministicTf32,
                        measured,
                        operands,
                        availability,
                    )
                    .unwrap(),
                );
            }
        }
    }

    #[test]
    fn sm120_tf32_cuda_13_2_identity_matches_the_literal_qualification_manifest() {
        let identity = sm120_retired_cohort((13, 2)).identity;
        assert_eq!(identity.module_kind, ModuleKind::TriadSm120);
        assert_eq!(identity.module_target, "compute_120");
        assert_eq!(identity.device_target, "sm_120");
        assert_eq!(identity.compute_capability, (12, 0));
        assert_eq!(identity.multiprocessor_count, 170);
        assert_eq!(identity.nvrtc_version, (13, 2));
        assert_eq!(identity.optin_shared_bytes, 101_376);
        assert!(identity.tensor_map_access);
        assert_eq!(
            digest_hex(&identity.compile_key),
            "70ee458820f27c554ee77d3f89c990cf005d7e3e566dc487ea7f88922ec33645"
        );
        assert_eq!(
            digest_hex(&identity.artifact_digest),
            "50ceabc64d69e8575a7974d3449605980cb1b42ddef8952ace3725073cd90441"
        );
        assert_eq!(
            digest_hex(&identity.source_digest),
            "65dfbcc164229b73f3f76fa15016e28a3adf9ad2328030eb1fe6d97ee3d01aaf"
        );
        assert_eq!(
            digest_hex(&identity.invocation_digest),
            "70ee458820f27c554ee77d3f89c990cf005d7e3e566dc487ea7f88922ec33645"
        );
        assert_eq!(
            digest_hex(&identity.header_manifest_digest),
            "905acac69a0bef20b12df1bd2fb70b6f8bd38fc530ec02bff5f2b124bf8d9157"
        );
        assert_eq!(
            digest_hex(&identity.nvrtc_library_domain),
            "0e0dc3faa997ae96442ef62ffc02640d361a5c50e0b4e5df2a05bc0986fe6241"
        );

        let module = qualified_module_for_auto_identity(identity);
        assert_eq!(module.artifact.module_kind, ModuleKind::TriadSm120);
        assert_eq!(module.artifact.artifact_kind, ArtifactKind::Ptx);
        assert_eq!(module.compiler.target.as_str(), "compute_120");
        assert!(module.compiler.nvrtc_library_known);
        assert_eq!(module.compiler.output_kind, ArtifactKind::Ptx);
        assert_eq!(module.compiler.composer_revision, 1);
        assert_eq!(module.compiler.compiler_revision, COMPILER_REVISION);
        assert_eq!(module.compiler.numeric_abi_revision, 5);
        assert_eq!(module.compiler.schedule_revision, 8);
        assert_eq!(module.device_caps.compute_capability, (12, 0));
        assert_eq!(module.device_caps.nvrtc_version, (13, 2));
        assert_eq!(
            module
                .device_caps
                .accepted_target
                .expect("accepted CUDA 13.2 target")
                .as_str(),
            "compute_120"
        );
        assert_eq!(module.device_caps.optin_shared_bytes, 101_376);
        assert!(module.device_caps.tensor_map_access);
    }

    const SCALAR_STRIDE_CELLS: [(ResolvedGemmOp, usize, usize, usize); 5] = [
        (ResolvedGemmOp::Tn, 49, 129, 65),
        (ResolvedGemmOp::Tn, 65, 129, 49),
        (ResolvedGemmOp::Tn, 131, 100, 129),
        (ResolvedGemmOp::Nt, 49, 65, 129),
        (ResolvedGemmOp::Nt, 65, 49, 129),
    ];

    const OPERAND_GATED_CELLS: [(ResolvedGemmOp, usize, usize, usize); 29] = [
        (ResolvedGemmOp::Nn, 16, 2048, 512),
        (ResolvedGemmOp::Nn, 49, 129, 65),
        (ResolvedGemmOp::Nn, 65, 129, 49),
        (ResolvedGemmOp::Nn, 129, 100, 131),
        (ResolvedGemmOp::Nn, 512, 768, 3072),
        (ResolvedGemmOp::Nn, 1024, 128, 256),
        (ResolvedGemmOp::Nn, 1024, 512, 128),
        (ResolvedGemmOp::Nn, 2048, 768, 1536),
        (ResolvedGemmOp::Nn, 2048, 768, 3072),
        (ResolvedGemmOp::Nn, 2048, 3072, 768),
        (ResolvedGemmOp::Nn, 4096, 768, 512),
        (ResolvedGemmOp::Nn, 4096, 1536, 3072),
        (ResolvedGemmOp::Nn, 4621, 1928, 384),
        (ResolvedGemmOp::Tn, 16, 2048, 512),
        (ResolvedGemmOp::Tn, 128, 512, 1024),
        (ResolvedGemmOp::Tn, 256, 128, 1024),
        (ResolvedGemmOp::Tn, 512, 384, 256),
        (ResolvedGemmOp::Tn, 3072, 768, 512),
        (ResolvedGemmOp::Tn, 8192, 128, 128),
        (ResolvedGemmOp::Nt, 16, 512, 2048),
        (ResolvedGemmOp::Nt, 128, 8192, 128),
        (ResolvedGemmOp::Nt, 512, 16, 2048),
        (ResolvedGemmOp::Nt, 512, 3072, 768),
        (ResolvedGemmOp::Nt, 1024, 256, 128),
        (ResolvedGemmOp::Nt, 2048, 768, 3072),
        (ResolvedGemmOp::Nt, 2048, 1536, 768),
        (ResolvedGemmOp::Nt, 2048, 3072, 768),
        (ResolvedGemmOp::Nt, 4096, 512, 768),
        (ResolvedGemmOp::Nt, 4621, 384, 1928),
    ];

    #[test]
    fn sm89_tf32_request_only_cells_remain_scalar() {
        for (op, rows, columns, reduction) in SCALAR_STRIDE_CELLS {
            assert_eq!(
                resolve_f32_triad_auto(
                    F32TriadPolicy::AllowDeterministicTf32,
                    normalized_request(op, rows, columns, reduction),
                    sm89_availability(),
                )
                .unwrap(),
                F32TriadSelection::ScalarFma,
                "request-only resolution promoted {op:?} {rows}x{columns}x{reduction}",
            );
        }
    }

    #[test]
    fn sm89_tf32_scalar_stride_cells_have_no_bucket_or_layout_bleed() {
        for (op, rows, columns, reduction) in SCALAR_STRIDE_CELLS {
            let request = normalized_request(op, rows, columns, reduction);
            let operands = F32TriadOperands {
                output: 0x1000,
                a: 0x2000,
                b: 0x3000,
                bias: None,
                alpha: 1.0,
                beta: if op == ResolvedGemmOp::Tn { 1.0 } else { 0.0 },
            };
            let expected = SM89_TF32_EVIDENCE_CELLS
                .iter()
                .rfind(|cell| {
                    cell.op == op
                        && cell.shape.output_rows == rows
                        && cell.shape.output_columns == columns
                        && cell.shape.reduction == reduction
                })
                .unwrap()
                .route;
            assert_eq!(
                resolve_f32_triad_auto_with_operands(
                    F32TriadPolicy::AllowDeterministicTf32,
                    request,
                    operands,
                    sm89_availability(),
                )
                .unwrap(),
                F32TriadSelection::Tf32(expected),
            );
            // These cells stage both operands with scalar loads. The band
            // widens them to a contiguous neighbour that stays scalar-staged
            // and to nothing else: a neighbour that gains a vector-staged
            // operand is served by the vector-staged evidence or not at all.
            let scalar_staged =
                |shape: F32TriadShape| super::tf32_staging_class(shape) == (false, false);
            let vector_staged_cells = SM89_TF32_EVIDENCE_CELLS
                .iter()
                .filter(|cell| !scalar_staged(cell.shape.contiguous(cell.op)))
                .copied()
                .collect::<Vec<_>>();
            for (axis, value) in [rows, columns, reduction].into_iter().enumerate() {
                for changed in [value - 1, value + 1] {
                    let mut normalized = [rows, columns, reduction];
                    normalized[axis] = changed;
                    let neighbour =
                        normalized_request(op, normalized[0], normalized[1], normalized[2]);
                    let served = resolve_f32_triad_auto_with_operands(
                        F32TriadPolicy::AllowDeterministicTf32,
                        neighbour,
                        operands,
                        sm89_availability(),
                    )
                    .unwrap();
                    let label =
                        format!("{op:?} {rows}x{columns}x{reduction} axis {axis} -> {changed}");
                    if scalar_staged(neighbour.shape) {
                        assert_eq!(served, F32TriadSelection::Tf32(expected), "{label}");
                    } else {
                        let from_vector_staged = super::nearest_tf32_portable_cell(
                            neighbour,
                            operands,
                            &vector_staged_cells,
                            142,
                        )
                        .map_or(F32TriadSelection::ScalarFma, F32TriadSelection::Tf32);
                        assert_eq!(served, from_vector_staged, "{label}");
                    }
                }
            }

            for stride in 0..3 {
                let mut noncontiguous = normalized_request(op, rows, columns, reduction);
                match stride {
                    0 => noncontiguous.shape.lda += 1,
                    1 => noncontiguous.shape.ldb += 1,
                    _ => noncontiguous.shape.ldc += 1,
                }
                assert_eq!(
                    resolve_f32_triad_auto_with_operands(
                        F32TriadPolicy::AllowDeterministicTf32,
                        noncontiguous,
                        operands,
                        sm89_availability(),
                    )
                    .unwrap(),
                    F32TriadSelection::ScalarFma,
                    "layout bleed for {op:?} {rows}x{columns}x{reduction} stride {stride}",
                );
            }
        }
    }

    #[test]
    fn sm89_tf32_cells_are_topology_target_and_revision_exact() {
        let selected = normalized_request(ResolvedGemmOp::Tn, 49, 129, 65);
        let operands = F32TriadOperands {
            output: 0x1000,
            a: 0x2000,
            b: 0x3000,
            bias: None,
            alpha: 1.0,
            beta: 1.0,
        };
        let mut wrong_sm_count = sm89_availability();
        wrong_sm_count
            .portable
            .as_mut()
            .unwrap()
            .device
            .multiprocessor_count = 141;
        let mut wrong_sm_count_high = sm89_availability();
        wrong_sm_count_high
            .portable
            .as_mut()
            .unwrap()
            .device
            .multiprocessor_count = 143;
        let cases = [
            F32TriadAvailability::default(),
            wrong_sm_count,
            wrong_sm_count_high,
            F32TriadAvailability {
                portable: Some(qualified_module(
                    ModuleKind::TriadSm80,
                    "sm_89",
                    "sm_89",
                    (8, 8),
                    false,
                    99_000,
                )),
                specialized: None,
                finalist: None,
                joint: None,
                multiprocessors: 142,
            },
            F32TriadAvailability {
                portable: Some(qualified_module(
                    ModuleKind::TriadSm80,
                    "sm_90a",
                    "sm_90a",
                    (9, 0),
                    false,
                    99_000,
                )),
                specialized: None,
                finalist: None,
                joint: None,
                multiprocessors: 142,
            },
            F32TriadAvailability {
                portable: Some(qualified_module(
                    ModuleKind::TriadSm80,
                    "sm_80",
                    "sm_80",
                    (8, 9),
                    false,
                    99_000,
                )),
                specialized: None,
                finalist: None,
                joint: None,
                multiprocessors: 142,
            },
            F32TriadAvailability {
                portable: Some(qualified_module(
                    ModuleKind::TriadSm90a,
                    "sm_89",
                    "sm_89",
                    (8, 9),
                    false,
                    99_000,
                )),
                specialized: None,
                finalist: None,
                joint: None,
                multiprocessors: 142,
            },
        ];
        for (index, availability) in cases.into_iter().enumerate() {
            let selection = resolve_f32_triad_auto_with_operands(
                F32TriadPolicy::AllowDeterministicTf32,
                selected,
                operands,
                availability,
            )
            .unwrap();
            // No module, a target the board does not own, or a module of
            // another kind leaves the exact family serving; a different
            // multiprocessor count or another board of the same tier is
            // served by the portable tier by design.
            let portable_tier = matches!(
                selection,
                F32TriadSelection::Tf32(route) if route.module_kind() == ModuleKind::TriadSm80
            );
            match index {
                1 | 2 | 4 => assert!(portable_tier, "case {index} selected {selection:?}"),
                _ => assert_eq!(
                    selection,
                    F32TriadSelection::ScalarFma,
                    "case {index} selected {selection:?}"
                ),
            }
        }
        assert_eq!(
            measured_tf32_route_with_operands(
                selected,
                operands,
                sm89_availability(),
                F32_TF32_TUNING_REVISION - 1,
            ),
            None,
        );
    }

    #[test]
    fn sm89_tf32_cells_are_nvrtc_13_2_exact() {
        let selected = normalized_request(ResolvedGemmOp::Tn, 49, 129, 65);
        let operands = F32TriadOperands {
            output: 0x1000,
            a: 0x2000,
            b: 0x3000,
            bias: None,
            alpha: 1.0,
            beta: 1.0,
        };
        // Another toolkit holds no cohort: the portable tier serves the
        // cell by design, and only the portable tier.
        for version in [(13, 1), (13, 0), (12, 8), (0, 0)] {
            let mut availability = sm89_availability();
            let portable = availability.portable.as_mut().unwrap();
            portable.compiler.nvrtc_version = version;
            portable.device_caps.nvrtc_version = version;
            let selection = resolve_f32_triad_auto_with_operands(
                F32TriadPolicy::AllowDeterministicTf32,
                selected,
                operands,
                availability,
            )
            .unwrap();
            assert!(
                matches!(
                    selection,
                    F32TriadSelection::Tf32(route) if route.module_kind() == ModuleKind::TriadSm80
                ),
                "NVRTC {version:?} selected {selection:?}"
            );
        }

        let mut unknown_library = sm89_availability();
        unknown_library
            .portable
            .as_mut()
            .unwrap()
            .compiler
            .nvrtc_library_known = false;
        assert_eq!(
            resolve_f32_triad_auto_with_operands(
                F32TriadPolicy::AllowDeterministicTf32,
                selected,
                operands,
                unknown_library,
            )
            .unwrap(),
            F32TriadSelection::ScalarFma,
        );

        for (compiler, device_caps) in [((13, 2), (13, 1)), ((13, 1), (13, 2))] {
            let mut availability = sm89_availability();
            let portable = availability.portable.as_mut().unwrap();
            portable.compiler.nvrtc_version = compiler;
            portable.device_caps.nvrtc_version = device_caps;
            assert_eq!(
                resolve_f32_triad_auto_with_operands(
                    F32TriadPolicy::AllowDeterministicTf32,
                    selected,
                    operands,
                    availability,
                )
                .unwrap(),
                F32TriadSelection::ScalarFma,
                "compiler/device NVRTC mismatch {compiler:?}/{device_caps:?} escaped",
            );
        }
    }

    #[test]
    fn sm89_tf32_operand_gated_cells_remain_scalar() {
        for (op, rows, columns, reduction) in OPERAND_GATED_CELLS {
            assert_eq!(
                resolve_f32_triad_auto(
                    F32TriadPolicy::AllowDeterministicTf32,
                    normalized_request(op, rows, columns, reduction),
                    sm89_availability(),
                )
                .unwrap(),
                F32TriadSelection::ScalarFma,
                "operand-gated {op:?} {rows}x{columns}x{reduction} was promoted",
            );
        }
    }

    #[test]
    fn sm89_tf32_auto_fails_closed_without_operands_alignment_or_exact_identity() {
        for cell in SM89_TF32_EVIDENCE_CELLS {
            let request = normalized_request(
                cell.op,
                cell.shape.output_rows,
                cell.shape.output_columns,
                cell.shape.reduction,
            );
            assert_eq!(
                resolve_f32_triad_auto(
                    F32TriadPolicy::AllowDeterministicTf32,
                    request,
                    sm89_availability(),
                )
                .unwrap(),
                F32TriadSelection::ScalarFma,
                "request-only resolution promoted {:?}",
                cell.shape,
            );

            let operands = F32TriadOperands {
                output: 0x1000,
                a: 0x2000,
                b: 0x3000,
                bias: None,
                alpha: 1.0,
                beta: if cell.op == ResolvedGemmOp::Tn {
                    1.0
                } else {
                    0.0
                },
            };
            for rejected in [
                F32TriadOperands {
                    output: 0x1004,
                    ..operands
                },
                F32TriadOperands {
                    a: 0x2004,
                    ..operands
                },
                F32TriadOperands {
                    b: 0x3004,
                    ..operands
                },
            ] {
                assert_eq!(
                    resolve_f32_triad_auto_with_operands(
                        F32TriadPolicy::AllowDeterministicTf32,
                        request,
                        rejected,
                        sm89_availability(),
                    )
                    .unwrap(),
                    F32TriadSelection::ScalarFma,
                    "misaligned operand promoted {:?}",
                    cell.shape,
                );
            }
        }

        let request = normalized_request(ResolvedGemmOp::Nn, 49, 129, 65);
        let operands = F32TriadOperands {
            output: 0x1000,
            a: 0x2000,
            b: 0x3000,
            bias: None,
            alpha: 1.0,
            beta: 0.0,
        };
        let mutations: [fn(&mut Tf32QualifiedModule); 26] = [
            |module| module.module_kind = ModuleKind::Fixed,
            |module| module.target = CudaTarget::new("sm_80").unwrap(),
            |module| module.artifact.module_kind = ModuleKind::Fixed,
            |module| module.artifact.artifact_kind = ArtifactKind::Cubin,
            |module| module.artifact.compile_key[0] ^= 1,
            |module| module.artifact.artifact_digest[0] ^= 1,
            |module| module.compiler.source_digest[0] ^= 1,
            |module| module.compiler.invocation_digest[0] ^= 1,
            |module| module.compiler.header_manifest_digest[0] ^= 1,
            |module| module.compiler.target = CudaTarget::new("sm_80").unwrap(),
            |module| module.compiler.nvrtc_version = (13, 1),
            |module| module.compiler.nvrtc_library_domain[0] ^= 1,
            |module| module.compiler.nvrtc_library_known = false,
            |module| module.compiler.output_kind = ArtifactKind::Cubin,
            |module| module.compiler.composer_revision ^= 1,
            |module| module.compiler.compiler_revision ^= 1,
            |module| module.compiler.numeric_abi_revision ^= 1,
            |module| module.compiler.schedule_revision ^= 1,
            |module| module.device.compute_capability = (8, 8),
            |module| module.device.multiprocessor_count -= 1,
            |module| module.device.target = CudaTarget::new("sm_80").unwrap(),
            |module| module.device_caps.compute_capability = (8, 8),
            |module| module.device_caps.nvrtc_version = (13, 1),
            |module| module.device_caps.accepted_target = None,
            |module| module.device_caps.optin_shared_bytes -= 1,
            |module| module.device_caps.tensor_map_access = true,
        ];
        // A mutated identity is no evidence. Structural drift (the module
        // kind, artifact and output kinds, an unknown NVRTC library, the
        // revisions, a target the board did not accept) leaves the exact
        // family serving; provenance drift (digests, versions, board) is
        // what another board looks like, and the portable tier serves it by
        // design. A specialized route or a proof candidate never appears.
        let structural: [usize; 10] = [0, 2, 3, 12, 13, 14, 15, 16, 17, 23];
        for (index, mutate) in mutations.into_iter().enumerate() {
            let mut availability = sm89_availability();
            mutate(availability.portable.as_mut().unwrap());
            let selection = resolve_f32_triad_auto_with_operands(
                F32TriadPolicy::AllowDeterministicTf32,
                request,
                operands,
                availability,
            )
            .unwrap();
            if structural.contains(&index) {
                assert_eq!(
                    selection,
                    F32TriadSelection::ScalarFma,
                    "structural mutation {index} was admitted"
                );
            } else {
                assert!(
                    matches!(selection, F32TriadSelection::ScalarFma)
                        || matches!(
                            selection,
                            F32TriadSelection::Tf32(route)
                                if route.module_kind() == ModuleKind::TriadSm80
                        ),
                    "mutation {index} selected {selection:?}"
                );
            }
        }
    }

    #[test]
    fn sm89_tf32_operand_aware_resolution_admits_exact_evidence() {
        for (op, rows, columns, reduction) in
            SCALAR_STRIDE_CELLS.into_iter().chain(OPERAND_GATED_CELLS)
        {
            let operands = F32TriadOperands {
                output: 0x1000,
                a: 0x2000,
                b: 0x3000,
                bias: None,
                alpha: 1.0,
                beta: if op == ResolvedGemmOp::Tn { 1.0 } else { 0.0 },
            };
            assert!(matches!(
                resolve_f32_triad_auto_with_operands(
                    F32TriadPolicy::AllowDeterministicTf32,
                    normalized_request(op, rows, columns, reduction),
                    operands,
                    sm89_availability(),
                )
                .unwrap(),
                F32TriadSelection::Tf32(_)
            ));
        }
    }

    #[test]
    fn sm89_tf32_requalification_overrides_archived_routes() {
        // These winners come from the wide5 and hot selector records. The
        // older, unlabelled rows must not shadow their later requalification.
        let cases = [
            (
                ResolvedGemmOp::Nn,
                2048,
                3072,
                768,
                Tf32PortableTile::M128N128,
                Tf32PortableStages::S3,
            ),
            (
                ResolvedGemmOp::Nn,
                2048,
                768,
                1536,
                Tf32PortableTile::M128N128,
                Tf32PortableStages::S3,
            ),
            (
                ResolvedGemmOp::Nn,
                4621,
                1928,
                384,
                Tf32PortableTile::M128N128,
                Tf32PortableStages::S3,
            ),
            (
                ResolvedGemmOp::Nn,
                4096,
                1536,
                3072,
                Tf32PortableTile::M128N128,
                Tf32PortableStages::S3,
            ),
            (
                ResolvedGemmOp::Nn,
                2048,
                768,
                3072,
                Tf32PortableTile::M128N128,
                Tf32PortableStages::S3,
            ),
            (
                ResolvedGemmOp::Tn,
                512,
                384,
                256,
                Tf32PortableTile::M16N32,
                Tf32PortableStages::S4,
            ),
        ];
        for (op, rows, columns, reduction, tile, stages) in cases {
            let operands = F32TriadOperands {
                output: 0x1000,
                a: 0x2000,
                b: 0x3000,
                bias: None,
                alpha: 1.0,
                beta: if op == ResolvedGemmOp::Tn { 1.0 } else { 0.0 },
            };
            assert_eq!(
                resolve_f32_triad_auto_with_operands(
                    F32TriadPolicy::AllowDeterministicTf32,
                    normalized_request(op, rows, columns, reduction),
                    operands,
                    sm89_availability(),
                )
                .unwrap(),
                F32TriadSelection::Tf32(Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile,
                    stages
                },)),
                "requalified winner is shadowed for {op:?} {rows}x{columns} over {reduction}",
            );
        }
    }

    #[test]
    fn sm89_tf32_bias_wide_requires_its_exact_qualified_epilogue() {
        let operands = F32TriadOperands {
            output: 0x1000,
            a: 0x2000,
            b: 0x3000,
            bias: Some(0x4000),
            alpha: 1.0,
            beta: 0.0,
        };
        let resolve = |request, operands| {
            resolve_f32_triad_auto_with_operands(
                F32TriadPolicy::AllowDeterministicTf32,
                request,
                operands,
                sm89_availability(),
            )
            .unwrap()
        };
        for (rows, columns, reduction) in [
            (2048, 3072, 768),
            (2048, 768, 1536),
            (4621, 1928, 384),
            (4096, 1536, 3072),
            (2048, 768, 3072),
        ] {
            let request = normalized_request(ResolvedGemmOp::Nn, rows, columns, reduction);
            assert_eq!(
                resolve(request, operands),
                F32TriadSelection::Tf32(Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: Tf32PortableTile::M128N128,
                    stages: Tf32PortableStages::S3,
                })),
            );
            for drift in [
                F32TriadOperands {
                    bias: Some(0),
                    ..operands
                },
                F32TriadOperands {
                    bias: Some(0x4004),
                    ..operands
                },
                F32TriadOperands {
                    a: 0x2004,
                    ..operands
                },
                F32TriadOperands {
                    b: 0x3004,
                    ..operands
                },
                F32TriadOperands {
                    output: 0x1004,
                    ..operands
                },
                F32TriadOperands {
                    alpha: 0.5,
                    ..operands
                },
                F32TriadOperands {
                    beta: -0.0,
                    ..operands
                },
                F32TriadOperands {
                    beta: 1.0,
                    ..operands
                },
            ] {
                assert_eq!(resolve(request, drift), F32TriadSelection::ScalarFma);
            }
            assert_eq!(
                resolve(
                    normalized_request(ResolvedGemmOp::Nn, rows + 1, columns, reduction),
                    operands
                ),
                F32TriadSelection::ScalarFma,
                "bias evidence must not expand to adjacent shapes",
            );
        }
        assert_eq!(
            resolve(
                normalized_request(ResolvedGemmOp::Nn, 4096, 768, 512),
                operands
            ),
            F32TriadSelection::ScalarFma,
            "an archived no-bias wide winner does not authorize bias",
        );
    }

    #[test]
    fn sm89_tf32_operand_aware_resolution_rejects_semantic_and_alignment_drift() {
        let request = normalized_request(ResolvedGemmOp::Nn, 2048, 768, 1536);
        let operands = F32TriadOperands {
            output: 0x1000,
            a: 0x2000,
            b: 0x3000,
            bias: None,
            alpha: 1.0,
            beta: 0.0,
        };
        for rejected in [
            F32TriadOperands {
                output: 0x1004,
                ..operands
            },
            F32TriadOperands {
                a: 0x2004,
                ..operands
            },
            F32TriadOperands {
                b: 0x3004,
                ..operands
            },
            F32TriadOperands {
                bias: Some(0x4004),
                ..operands
            },
            F32TriadOperands {
                alpha: f32::from_bits((-0.0_f32).to_bits()),
                ..operands
            },
            F32TriadOperands {
                beta: -0.0,
                ..operands
            },
        ] {
            assert_eq!(
                resolve_f32_triad_auto_with_operands(
                    F32TriadPolicy::AllowDeterministicTf32,
                    request,
                    rejected,
                    sm89_availability(),
                )
                .unwrap(),
                F32TriadSelection::ScalarFma,
            );
        }
    }

    #[test]
    fn sm89_tf32_evidence_inventory_encodes_the_missing_operand_gates() {
        assert_eq!(SM89_TF32_EVIDENCE_CELLS.len(), 69);
        let mut gate_counts = [0_usize; 4];
        for cell in SM89_TF32_EVIDENCE_CELLS {
            let request = normalized_request(
                cell.op,
                cell.shape.output_rows,
                cell.shape.output_columns,
                cell.shape.reduction,
            );
            let async_staging_shape =
                request.shape.lda.is_multiple_of(4) && request.shape.ldb.is_multiple_of(4);
            match cell.operand_gate {
                Tf32AutoOperandGate::RequestContractSafe => {
                    gate_counts[0] += 1;
                    assert_ne!(cell.op, ResolvedGemmOp::Nn);
                    assert!(!async_staging_shape);
                }
                Tf32AutoOperandGate::RequiresNoBiasEvidence => {
                    gate_counts[1] += 1;
                    assert_eq!(cell.op, ResolvedGemmOp::Nn);
                    assert!(!async_staging_shape);
                }
                Tf32AutoOperandGate::RequiresVectorAlignmentEvidence => {
                    gate_counts[2] += 1;
                    assert_ne!(cell.op, ResolvedGemmOp::Nn);
                    assert!(async_staging_shape);
                }
                Tf32AutoOperandGate::RequiresNoBiasAndVectorAlignmentEvidence => {
                    gate_counts[3] += 1;
                    assert_eq!(cell.op, ResolvedGemmOp::Nn);
                    assert!(async_staging_shape);
                }
            }
        }
        assert_eq!(gate_counts, [5, 3, 36, 25]);
    }

    #[test]
    fn sm89_tf32_evidence_inventory_matches_all_qualified_routes() {
        use Tf32AutoOperandGate::{
            RequestContractSafe as Safe, RequiresNoBiasAndVectorAlignmentEvidence as BiasAlign,
            RequiresNoBiasEvidence as Bias, RequiresVectorAlignmentEvidence as Align,
        };
        use Tf32PortableStages::{S2, S3, S4};
        use Tf32PortableTile::{M16N32, M32N32, M64N64, M128N64, M128N128};

        let expected = [
            (
                ResolvedGemmOp::Nn,
                16,
                2048,
                512,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M16N32,
                    stages: S4,
                }),
                BiasAlign,
            ),
            (
                ResolvedGemmOp::Nn,
                49,
                129,
                65,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M16N32,
                    stages: S4,
                }),
                Bias,
            ),
            (
                ResolvedGemmOp::Nn,
                65,
                129,
                49,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M16N32,
                    stages: S4,
                }),
                Bias,
            ),
            (
                ResolvedGemmOp::Nn,
                129,
                100,
                131,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M16N32,
                    stages: S4,
                }),
                Bias,
            ),
            (
                ResolvedGemmOp::Nn,
                512,
                768,
                3072,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M64N64,
                    stages: S3,
                }),
                BiasAlign,
            ),
            (
                ResolvedGemmOp::Nn,
                1024,
                128,
                256,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M16N32,
                    stages: S4,
                }),
                BiasAlign,
            ),
            (
                ResolvedGemmOp::Nn,
                1024,
                512,
                128,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M64N64,
                    stages: S2,
                }),
                BiasAlign,
            ),
            (
                ResolvedGemmOp::Nn,
                2048,
                768,
                1536,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M128N64,
                    stages: S3,
                }),
                BiasAlign,
            ),
            (
                ResolvedGemmOp::Nn,
                2048,
                768,
                3072,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M128N64,
                    stages: S3,
                }),
                BiasAlign,
            ),
            (
                ResolvedGemmOp::Nn,
                2048,
                3072,
                768,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M64N64,
                    stages: S2,
                }),
                BiasAlign,
            ),
            (
                ResolvedGemmOp::Nn,
                4096,
                768,
                512,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M128N64,
                    stages: S3,
                }),
                BiasAlign,
            ),
            (
                ResolvedGemmOp::Nn,
                4096,
                1536,
                3072,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M64N64,
                    stages: S2,
                }),
                BiasAlign,
            ),
            (
                ResolvedGemmOp::Nn,
                4621,
                1928,
                384,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M64N64,
                    stages: S2,
                }),
                BiasAlign,
            ),
            (
                ResolvedGemmOp::Tn,
                16,
                2048,
                512,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M16N32,
                    stages: S4,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Tn,
                49,
                129,
                65,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M16N32,
                    stages: S4,
                }),
                Safe,
            ),
            (
                ResolvedGemmOp::Tn,
                65,
                129,
                49,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M16N32,
                    stages: S4,
                }),
                Safe,
            ),
            (
                ResolvedGemmOp::Tn,
                128,
                512,
                1024,
                Tf32PhysicalRoute::MmaTf32RnaSplitK8(Tf32PortableRoute {
                    tile: M32N32,
                    stages: S4,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Tn,
                131,
                100,
                129,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M16N32,
                    stages: S4,
                }),
                Safe,
            ),
            (
                ResolvedGemmOp::Tn,
                256,
                128,
                1024,
                Tf32PhysicalRoute::MmaTf32RnaSplitK8(Tf32PortableRoute {
                    tile: M32N32,
                    stages: S3,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Tn,
                512,
                384,
                256,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M64N64,
                    stages: S3,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Tn,
                3072,
                768,
                512,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M64N64,
                    stages: S2,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Tn,
                8192,
                128,
                128,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M64N64,
                    stages: S2,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Nt,
                16,
                512,
                2048,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M16N32,
                    stages: S4,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Nt,
                49,
                65,
                129,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M16N32,
                    stages: S4,
                }),
                Safe,
            ),
            (
                ResolvedGemmOp::Nt,
                65,
                49,
                129,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M16N32,
                    stages: S4,
                }),
                Safe,
            ),
            (
                ResolvedGemmOp::Nt,
                128,
                8192,
                128,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M64N64,
                    stages: S2,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Nt,
                512,
                16,
                2048,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M16N32,
                    stages: S4,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Nt,
                512,
                3072,
                768,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M64N64,
                    stages: S3,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Nt,
                1024,
                256,
                128,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M16N32,
                    stages: S4,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Nt,
                2048,
                768,
                3072,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M128N64,
                    stages: S3,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Nt,
                2048,
                1536,
                768,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M128N64,
                    stages: S3,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Nt,
                2048,
                3072,
                768,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M128N64,
                    stages: S3,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Nt,
                4096,
                512,
                768,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M128N64,
                    stages: S3,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Nt,
                4621,
                384,
                1928,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M128N64,
                    stages: S3,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Tn,
                768,
                3072,
                2048,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M64N64,
                    stages: S2,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Tn,
                1536,
                768,
                2048,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M64N64,
                    stages: S2,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Tn,
                384,
                1928,
                4621,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M128N64,
                    stages: S3,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Tn,
                3072,
                1536,
                4096,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M64N64,
                    stages: S2,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Nt,
                4096,
                3072,
                1536,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M128N64,
                    stages: S3,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Tn,
                3072,
                768,
                2048,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M64N64,
                    stages: S2,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Nt,
                1024,
                128,
                512,
                Tf32PhysicalRoute::MmaTf32RnaSplitK4(Tf32PortableRoute {
                    tile: M16N32,
                    stages: S3,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Nn,
                10400,
                1536,
                384,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M64N64,
                    stages: S2,
                }),
                BiasAlign,
            ),
            (
                ResolvedGemmOp::Nt,
                10400,
                384,
                1536,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M128N64,
                    stages: S3,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Nn,
                4621,
                384,
                768,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M128N128,
                    stages: S3,
                }),
                BiasAlign,
            ),
            (
                ResolvedGemmOp::Nt,
                4621,
                768,
                384,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M64N64,
                    stages: S2,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Nn,
                4621,
                384,
                1024,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M128N128,
                    stages: S3,
                }),
                BiasAlign,
            ),
            (
                ResolvedGemmOp::Tn,
                1024,
                384,
                4621,
                Tf32PhysicalRoute::MmaTf32RnaSplitK8(Tf32PortableRoute {
                    tile: M64N64,
                    stages: S3,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Nt,
                4621,
                1024,
                384,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M64N64,
                    stages: S2,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Nn,
                10400,
                384,
                384,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M128N128,
                    stages: S3,
                }),
                BiasAlign,
            ),
            (
                ResolvedGemmOp::Nt,
                10400,
                384,
                384,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M64N64,
                    stages: S2,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Nn,
                10400,
                384,
                768,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M128N128,
                    stages: S3,
                }),
                BiasAlign,
            ),
            (
                ResolvedGemmOp::Nt,
                10400,
                768,
                384,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M128N64,
                    stages: S3,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Nn,
                4096,
                768,
                512,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M128N128,
                    stages: S3,
                }),
                BiasAlign,
            ),
            (
                ResolvedGemmOp::Tn,
                512,
                768,
                4096,
                Tf32PhysicalRoute::MmaTf32RnaSplitK8(Tf32PortableRoute {
                    tile: M64N64,
                    stages: S3,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Nt,
                4096,
                512,
                768,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M128N64,
                    stages: S3,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Nn,
                256,
                384,
                512,
                Tf32PhysicalRoute::MmaTf32RnaSplitK2(Tf32PortableRoute {
                    tile: M16N32,
                    stages: S4,
                }),
                BiasAlign,
            ),
            (
                ResolvedGemmOp::Tn,
                512,
                384,
                256,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M16N32,
                    stages: S4,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Nt,
                256,
                512,
                384,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M16N32,
                    stages: S4,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Tn,
                768,
                384,
                4621,
                Tf32PhysicalRoute::MmaTf32RnaSplitK8(Tf32PortableRoute {
                    tile: M64N64,
                    stages: S3,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Nn,
                2048,
                3072,
                768,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M128N128,
                    stages: S3,
                }),
                BiasAlign,
            ),
            (
                ResolvedGemmOp::Nn,
                2048,
                768,
                1536,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M128N128,
                    stages: S3,
                }),
                BiasAlign,
            ),
            (
                ResolvedGemmOp::Nn,
                4621,
                1928,
                384,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M128N128,
                    stages: S3,
                }),
                BiasAlign,
            ),
            (
                ResolvedGemmOp::Nn,
                4096,
                1536,
                3072,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M128N128,
                    stages: S3,
                }),
                BiasAlign,
            ),
            (
                ResolvedGemmOp::Nn,
                2048,
                768,
                3072,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M128N128,
                    stages: S3,
                }),
                BiasAlign,
            ),
            (
                ResolvedGemmOp::Nn,
                1024,
                512,
                128,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M64N64,
                    stages: S2,
                }),
                BiasAlign,
            ),
            (
                ResolvedGemmOp::Nn,
                1024,
                128,
                256,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M16N32,
                    stages: S4,
                }),
                BiasAlign,
            ),
            (
                ResolvedGemmOp::Nn,
                10400,
                1536,
                384,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: M128N128,
                    stages: S3,
                }),
                BiasAlign,
            ),
            (
                ResolvedGemmOp::Tn,
                384,
                384,
                10400,
                Tf32PhysicalRoute::MmaTf32RnaSplitK8(Tf32PortableRoute {
                    tile: M64N64,
                    stages: S3,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Tn,
                768,
                384,
                10400,
                Tf32PhysicalRoute::MmaTf32RnaSplitK8(Tf32PortableRoute {
                    tile: M64N64,
                    stages: S3,
                }),
                Align,
            ),
        ];

        assert_eq!(SM89_TF32_EVIDENCE_CELLS.len(), expected.len());
        for (cell, (op, rows, columns, reduction, route, operand_gate)) in
            SM89_TF32_EVIDENCE_CELLS.iter().zip(expected)
        {
            assert_eq!(cell.op, op);
            assert_eq!(
                cell.shape,
                super::Tf32ExactShape {
                    output_rows: rows,
                    output_columns: columns,
                    reduction,
                }
            );
            assert_eq!(cell.route, route);
            assert_eq!(cell.operand_gate, operand_gate);
        }
    }

    #[test]
    fn exact_scalar_policy_dominates_the_sm89_tf32_table() {
        for (op, rows, columns, reduction) in
            SCALAR_STRIDE_CELLS.into_iter().chain(OPERAND_GATED_CELLS)
        {
            assert_eq!(
                resolve_f32_triad_auto(
                    F32TriadPolicy::ExactScalarFma,
                    normalized_request(op, rows, columns, reduction),
                    sm89_availability(),
                )
                .unwrap(),
                F32TriadSelection::ScalarFma,
            );
        }
    }

    #[test]
    fn exact_and_unmeasured_auto_resolution_stay_scalar() {
        let portable = qualified_module(
            ModuleKind::TriadSm80,
            "sm_89",
            "sm_89",
            (8, 9),
            false,
            99_000,
        );
        for op in [ResolvedGemmOp::Nn, ResolvedGemmOp::Tn, ResolvedGemmOp::Nt] {
            for availability in [
                F32TriadAvailability::default(),
                F32TriadAvailability {
                    portable: Some(portable),
                    specialized: None,
                    finalist: None,
                    joint: None,
                    multiprocessors: 142,
                },
            ] {
                assert_eq!(
                    resolve_f32_triad_auto(
                        F32TriadPolicy::ExactScalarFma,
                        request(op),
                        availability,
                    )
                    .unwrap(),
                    F32TriadSelection::ScalarFma
                );
                assert_eq!(
                    resolve_f32_triad_auto(
                        F32TriadPolicy::AllowDeterministicTf32,
                        request(op),
                        availability,
                    )
                    .unwrap(),
                    F32TriadSelection::ScalarFma
                );
            }
        }
    }

    #[test]
    fn forced_resolution_accepts_each_exact_qualified_family() {
        let cases = [
            (
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: Tf32PortableTile::M16N32,
                    stages: Tf32PortableStages::S4,
                }),
                F32TriadAvailability {
                    portable: Some(qualified_module(
                        ModuleKind::TriadSm80,
                        "sm_89",
                        "sm_89",
                        (8, 9),
                        false,
                        29_696,
                    )),
                    specialized: None,
                    finalist: None,
                    joint: None,
                    multiprocessors: 142,
                },
            ),
            (
                Tf32PhysicalRoute::Sm90aWgmmaTf32Tma(Tf32Sm90aRoute {
                    schedule: Sm90aWarpgroupSchedule::Wg2,
                }),
                F32TriadAvailability {
                    portable: None,
                    specialized: Some(qualified_module(
                        ModuleKind::TriadSm90a,
                        "sm_90a",
                        "sm_90a",
                        (9, 0),
                        true,
                        73_984,
                    )),
                    finalist: None,
                    joint: None,
                    multiprocessors: 142,
                },
            ),
            (
                Tf32PhysicalRoute::Sm100Tcgen05Tf32Tma(Tf32Sm100Route {
                    tile: Sm100Tile::M128N128,
                    stages: Sm100Stages::S4,
                    schedule: Sm100Schedule::P8,
                }),
                F32TriadAvailability {
                    portable: None,
                    specialized: Some(qualified_module(
                        ModuleKind::TriadSm100,
                        "compute_100a",
                        "sm_100a",
                        (10, 0),
                        true,
                        131_328,
                    )),
                    finalist: None,
                    joint: None,
                    multiprocessors: 142,
                },
            ),
            (
                Tf32PhysicalRoute::Sm120TmaMmaTf32Rna(Tf32Sm120Route {
                    tile: Tf32Sm120Tile::M64N128,
                    stages: Tf32Sm120Stages::S3,
                }),
                F32TriadAvailability {
                    portable: None,
                    specialized: Some(qualified_module(
                        ModuleKind::TriadSm120,
                        "compute_120",
                        "sm_120",
                        (12, 0),
                        true,
                        73_856,
                    )),
                    finalist: None,
                    joint: None,
                    multiprocessors: 142,
                },
            ),
        ];
        for (route, availability) in cases {
            assert_eq!(
                resolve_tf32_forced(request(ResolvedGemmOp::Nn), availability, route).unwrap(),
                route
            );
        }
    }

    #[test]
    fn sm89_joint_forced_routes_are_exact_cell_only_and_unqualified_auto_stays_closed() {
        let joint = qualified_module(
            ModuleKind::TriadSm89Tf32Joint,
            "sm_89",
            "sm_89",
            (8, 9),
            false,
            101_376,
        );
        let availability = F32TriadAvailability {
            joint: Some(joint),
            ..F32TriadAvailability::default()
        };
        for (op, dims, route) in [
            (
                ResolvedGemmOp::Tn,
                (2_048, 768, 3_072),
                Tf32PhysicalRoute::Sm89TnPreRnaN96,
            ),
            (
                ResolvedGemmOp::Tn,
                (2_048, 1_536, 768),
                Tf32PhysicalRoute::Sm89TnPreRnaN96,
            ),
            (
                ResolvedGemmOp::Tn,
                (4_621, 384, 1_928),
                Tf32PhysicalRoute::Sm89TnPreRnaM64N96S2,
            ),
            (
                ResolvedGemmOp::Nn,
                (4_621, 384, 1_928),
                Tf32PhysicalRoute::Sm89NnDirectN96,
            ),
            (
                ResolvedGemmOp::Nn,
                (2_048, 1_536, 768),
                Tf32PhysicalRoute::Sm89NnN96,
            ),
            (
                ResolvedGemmOp::Nt,
                (2_048, 768, 3_072),
                Tf32PhysicalRoute::Sm89NtALdmatrixN96,
            ),
            (
                ResolvedGemmOp::Tn,
                (2_048, 768, 3_072),
                Tf32PhysicalRoute::Sm89TnPreRnaM96N192S2,
            ),
            (
                ResolvedGemmOp::Tn,
                (2_048, 1_536, 768),
                Tf32PhysicalRoute::Sm89TnPreRnaM96N96S3,
            ),
            (
                ResolvedGemmOp::Tn,
                (4_096, 3_072, 1_536),
                Tf32PhysicalRoute::Sm89TnDirectM192N192S2,
            ),
            (
                ResolvedGemmOp::Nt,
                (4_096, 3_072, 1_536),
                Tf32PhysicalRoute::Sm89NtRowstageM128N192S2,
            ),
            (
                ResolvedGemmOp::Nt,
                (4_621, 384, 1_928),
                Tf32PhysicalRoute::Sm89NtRnaM144N96S2,
            ),
        ] {
            let request = F32TriadRequest {
                op,
                shape: F32TriadShape::contiguous(op, dims),
            };
            assert_eq!(
                resolve_tf32_forced(request, availability, route).unwrap(),
                route
            );
            assert_eq!(
                resolve_f32_triad_auto(
                    F32TriadPolicy::AllowDeterministicTf32,
                    request,
                    availability,
                )
                .unwrap(),
                F32TriadSelection::ScalarFma,
                "empty joint cohort must not admit {op:?}/{dims:?}"
            );
            let mut neighbors = Vec::new();
            for field in 0..6 {
                let mut neighbor = request;
                match field {
                    0 => neighbor.shape.m += 1,
                    1 => neighbor.shape.k += 1,
                    2 => neighbor.shape.n += 1,
                    3 => neighbor.shape.lda += 1,
                    4 => neighbor.shape.ldb += 1,
                    5 => neighbor.shape.ldc += 1,
                    _ => unreachable!(),
                }
                neighbors.push(neighbor);
            }
            for neighbor in neighbors {
                assert!(
                    resolve_tf32_forced(neighbor, availability, route).is_err(),
                    "joint route {route:?} accepted neighboring request {neighbor:?}"
                );
            }
        }
    }

    #[test]
    fn sm89_finalist_is_forced_only_and_uses_only_its_own_pointer_binding() {
        let route = Tf32PhysicalRoute::Sm89MmaTf32Compact8;
        let request = normalized_request(ResolvedGemmOp::Nt, 2048, 768, 3072);
        let operands = F32TriadOperands {
            output: 0x1000,
            a: 0x2000,
            b: 0x3000,
            bias: None,
            alpha: 1.0,
            beta: 0.0,
        };
        let availability = sm89_finalist_availability();
        assert_eq!(resolve_tf32_forced(request, availability, route), Ok(route));

        let prior = F32TriadSelection::Tf32(Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
            tile: Tf32PortableTile::M128N64,
            stages: Tf32PortableStages::S3,
        }));
        assert_eq!(
            resolve_f32_triad_auto_with_operands(
                F32TriadPolicy::AllowDeterministicTf32,
                request,
                operands,
                sm89_availability(),
            )
            .unwrap(),
            prior,
            "the incumbent portable baseline changed"
        );
        assert_eq!(
            resolve_f32_triad_auto_with_operands(
                F32TriadPolicy::AllowDeterministicTf32,
                request,
                operands,
                availability,
            )
            .unwrap(),
            prior,
            "an unmeasured finalist binding changed AUTO"
        );

        let mut rejected = availability;
        rejected.finalist = rejected.portable;
        assert_eq!(
            resolve_f32_triad_auto_with_operands(
                F32TriadPolicy::AllowDeterministicTf32,
                request,
                operands,
                rejected,
            )
            .unwrap(),
            prior,
            "a rejected finalist holder changed the portable AUTO route"
        );

        assert_eq!(
            resolve_f32_triad_auto(
                F32TriadPolicy::AllowDeterministicTf32,
                request,
                availability,
            )
            .unwrap(),
            F32TriadSelection::ScalarFma,
            "request-only resolution must remain unable to claim TF32 evidence"
        );

        let mut missing = availability;
        missing.finalist = None;
        assert!(resolve_tf32_forced(request, missing, route).is_err());
        assert!(resolve_tf32_forced(request, rejected, route).is_err());
        let mut missing_portable = availability;
        missing_portable.portable = None;
        assert_eq!(
            resolve_tf32_forced(request, missing_portable, route),
            Ok(route),
            "the finalist holder depends on the portable holder"
        );
    }

    #[test]
    fn sm89_finalist_measured_cohorts_select_and_decline_to_prior_routes() {
        assert!(
            !SM89_FINALIST_TF32_EVIDENCE_COHORTS.is_empty(),
            "no measured finalist has been integrated into AUTO"
        );
        let operands = F32TriadOperands {
            output: 0x1000,
            a: 0x2000,
            b: 0x3000,
            bias: None,
            alpha: 1.0,
            beta: 0.0,
        };
        for cohort in SM89_FINALIST_TF32_EVIDENCE_COHORTS {
            assert_eq!(cohort.identity.module_kind, ModuleKind::TriadSm89Finalist);
            assert_eq!(
                cohort.tuning_revision,
                super::super::contract::SM89_FINALIST_TUNING_REVISION
            );
            assert!(cohort.portable.is_none());
            assert_eq!(
                cohort.cells.len(),
                4,
                "the stage-sliced finalist cohort must include large_deep"
            );
            assert_eq!(
                SM89_FINALIST_TF32_EVIDENCE_COHORTS
                    .iter()
                    .filter(|other| other.identity == cohort.identity)
                    .count(),
                1,
                "duplicate finalist identity"
            );
            let availability = F32TriadAvailability {
                finalist: Some(qualified_module_for_auto_identity(cohort.identity)),
                ..sm89_availability()
            };
            let select = |request, operands, availability| {
                resolve_f32_triad_auto_with_operands(
                    F32TriadPolicy::AllowDeterministicTf32,
                    request,
                    operands,
                    availability,
                )
            };
            let assert_prior = |request, operands, actual: F32TriadAvailability| {
                let previous = F32TriadAvailability {
                    finalist: None,
                    joint: None,
                    ..actual
                };
                assert_eq!(
                    select(request, operands, actual),
                    select(request, operands, previous)
                );
            };
            for cell in cohort.cells {
                assert_eq!(cell.op, ResolvedGemmOp::Nt);
                assert_eq!(cell.route, Tf32PhysicalRoute::Sm89MmaTf32Compact8);
                assert_eq!(
                    cohort
                        .cells
                        .iter()
                        .filter(|other| other.op == cell.op && other.shape == cell.shape)
                        .count(),
                    1,
                    "duplicate finalist shape"
                );
                let dims = [
                    cell.shape.output_rows,
                    cell.shape.output_columns,
                    cell.shape.reduction,
                ];
                let request = normalized_request(cell.op, dims[0], dims[1], dims[2]);
                assert_eq!(
                    select(request, operands, availability),
                    Ok(F32TriadSelection::Tf32(cell.route))
                );

                // The existing mutation inventory's first 25 fields are architecture-neutral;
                // its last field sets tensor-map access false, already false on Ada.
                for mutate in
                    sm120_identity_mutations()
                        .into_iter()
                        .take(25)
                        .chain(std::iter::once(
                            (|module: &mut Tf32QualifiedModule| {
                                module.device_caps.tensor_map_access = true;
                            }) as fn(&mut Tf32QualifiedModule),
                        ))
                {
                    let mut changed = availability;
                    mutate(changed.finalist.as_mut().unwrap());
                    assert!(
                        matching_tf32_cohort(
                            changed.finalist.unwrap(),
                            SM89_FINALIST_TF32_EVIDENCE_COHORTS
                        )
                        .is_none()
                    );
                    assert_prior(request, operands, changed);
                }
                for axis in 0..3 {
                    for step in [-1isize, 1] {
                        let mut changed = dims;
                        changed[axis] = (changed[axis] as isize + step) as usize;
                        assert_prior(
                            normalized_request(cell.op, changed[0], changed[1], changed[2]),
                            operands,
                            availability,
                        );
                    }
                    let mut changed = request;
                    match axis {
                        0 => changed.shape.lda += 1,
                        1 => changed.shape.ldb += 1,
                        _ => changed.shape.ldc += 1,
                    }
                    assert_prior(changed, operands, availability);
                }
                let operand_mutations: [fn(&mut F32TriadOperands); 11] = [
                    |value| value.output += 4,
                    |value| value.a += 4,
                    |value| value.b += 4,
                    |value| value.output = 0,
                    |value| value.a = 0,
                    |value| value.b = 0,
                    |value| value.alpha = -0.75,
                    |value| value.beta = 0.5,
                    |value| value.beta = -0.0,
                    |value| value.bias = Some(0x4000),
                    |value| value.bias = Some(0),
                ];
                for mutate in operand_mutations {
                    let mut changed = operands;
                    mutate(&mut changed);
                    assert_prior(request, changed, availability);
                }
                for finalist in [None, availability.portable] {
                    assert_prior(
                        request,
                        operands,
                        F32TriadAvailability {
                            finalist,
                            ..availability
                        },
                    );
                }
            }
        }
    }

    #[test]
    fn triad_retained_identity_finalist_bindings_match_three_measured_modules() {
        let expected = measured_finalist_identities();
        assert_eq!(SM89_FINALIST_TF32_EVIDENCE_COHORTS.len(), 3);
        for identity in expected {
            let module = qualified_module_for_auto_identity(identity);
            let cohort = matching_tf32_cohort(module, SM89_FINALIST_TF32_EVIDENCE_COHORTS)
                .unwrap_or_else(|| {
                    panic!(
                        "measured CUDA {:?} finalist module must enter AUTO",
                        identity.nvrtc_version
                    )
                });
            assert_eq!(cohort.identity, identity);
            assert_eq!(
                cohort.tuning_revision,
                super::super::contract::SM89_FINALIST_TUNING_REVISION
            );

            let mut changed = module;
            changed.compiler.nvrtc_version.1 += 1;
            assert!(matching_tf32_cohort(changed, SM89_FINALIST_TF32_EVIDENCE_COHORTS).is_none());
            let mut changed = module;
            changed.compiler.source_digest[0] ^= 1;
            assert!(matching_tf32_cohort(changed, SM89_FINALIST_TF32_EVIDENCE_COHORTS).is_none());
            let mut changed = module;
            changed.artifact.compile_key[0] ^= 1;
            assert!(matching_tf32_cohort(changed, SM89_FINALIST_TF32_EVIDENCE_COHORTS).is_none());
            let mut changed = module;
            changed.artifact.artifact_digest[0] ^= 1;
            assert!(matching_tf32_cohort(changed, SM89_FINALIST_TF32_EVIDENCE_COHORTS).is_none());
        }
        let actual = SM89_FINALIST_TF32_EVIDENCE_COHORTS
            .iter()
            .map(|cohort| cohort.identity)
            .collect::<Vec<_>>();
        assert_eq!(actual.as_slice(), expected.as_slice());
        assert_eq!(
            SM89_FINALIST_TF32_EVIDENCE_COHORTS
                .iter()
                .map(|cohort| cohort.identity.compile_key)
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            3
        );
    }

    #[test]
    fn portable_sm110_accepts_the_arch_specific_target_transaction() {
        let route = Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
            tile: Tf32PortableTile::M16N32,
            stages: Tf32PortableStages::S4,
        });
        let availability = F32TriadAvailability {
            portable: Some(qualified_module(
                ModuleKind::TriadSm80,
                "sm_110a",
                "sm_110a",
                (11, 0),
                false,
                29_696,
            )),
            specialized: None,
            finalist: None,
            joint: None,
            multiprocessors: 142,
        };

        assert_eq!(
            resolve_tf32_forced(request(ResolvedGemmOp::Nn), availability, route).unwrap(),
            route
        );
    }

    #[test]
    fn portable_sm101a_accepts_the_exact_target_transaction() {
        let route = Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
            tile: Tf32PortableTile::M16N32,
            stages: Tf32PortableStages::S4,
        });
        let availability = F32TriadAvailability {
            portable: Some(qualified_module(
                ModuleKind::TriadSm80,
                "sm_101a",
                "sm_101a",
                (10, 1),
                false,
                29_696,
            )),
            specialized: None,
            finalist: None,
            joint: None,
            multiprocessors: 142,
        };

        assert_eq!(
            resolve_tf32_forced(request(ResolvedGemmOp::Nn), availability, route).unwrap(),
            route
        );
    }

    #[test]
    fn nt_splitk8_is_forced_only_and_does_not_change_auto_selection() {
        let selected = normalized_request(ResolvedGemmOp::Nt, 64, 384, 1_536);
        assert_eq!(
            resolve_tf32_forced(selected, sm89_availability(), TF32_NT_SPLITK8_S3_SPEC.route,)
                .unwrap(),
            TF32_NT_SPLITK8_S3_SPEC.route
        );
        assert_eq!(
            resolve_f32_triad_auto(
                F32TriadPolicy::AllowDeterministicTf32,
                selected,
                sm89_availability(),
            )
            .unwrap(),
            F32TriadSelection::ScalarFma
        );
    }

    #[test]
    fn specialized_sm110_accepts_only_feature_target_transactions() {
        let route = Tf32PhysicalRoute::Sm100Tcgen05Tf32Tma(Tf32Sm100Route {
            tile: Sm100Tile::M128N64,
            stages: Sm100Stages::S2,
            schedule: Sm100Schedule::C4,
        });
        for (compiler_target, device_target) in
            [("compute_110f", "sm_110f"), ("compute_110a", "sm_110a")]
        {
            let availability = F32TriadAvailability {
                portable: None,
                specialized: Some(qualified_module(
                    ModuleKind::TriadSm100,
                    compiler_target,
                    device_target,
                    (11, 0),
                    true,
                    49_408,
                )),
                finalist: None,
                joint: None,
                multiprocessors: 142,
            };
            assert_eq!(
                resolve_tf32_forced(request(ResolvedGemmOp::Nn), availability, route).unwrap(),
                route
            );
        }

        let generic = F32TriadAvailability {
            portable: None,
            specialized: Some(qualified_module(
                ModuleKind::TriadSm100,
                "sm_110",
                "sm_110",
                (11, 0),
                true,
                49_408,
            )),
            finalist: None,
            joint: None,
            multiprocessors: 142,
        };
        assert!(resolve_tf32_forced(request(ResolvedGemmOp::Nn), generic, route).is_err());
    }

    #[test]
    fn forced_resolution_rejects_incoherent_or_unavailable_bindings() {
        let route = Tf32PhysicalRoute::Sm100Tcgen05Tf32Tma(Tf32Sm100Route {
            tile: Sm100Tile::M128N128,
            stages: Sm100Stages::S4,
            schedule: Sm100Schedule::C4,
        });
        let valid = qualified_module(
            ModuleKind::TriadSm100,
            "compute_100a",
            "sm_100a",
            (10, 0),
            true,
            131_328,
        );
        assert!(
            resolve_tf32_forced(
                request(ResolvedGemmOp::Nn),
                F32TriadAvailability::default(),
                route
            )
            .is_err()
        );
        for invalid in [
            Tf32QualifiedModule {
                module_kind: ModuleKind::TriadSm90a,
                ..valid
            },
            Tf32QualifiedModule {
                target: CudaTarget::new("compute_103a").unwrap(),
                ..valid
            },
            Tf32QualifiedModule {
                device: DeviceIdentity {
                    compute_capability: (10, 3),
                    ..valid.device
                },
                ..valid
            },
            Tf32QualifiedModule {
                device_caps: DeviceCaps {
                    tensor_map_access: false,
                    ..valid.device_caps
                },
                ..valid
            },
            Tf32QualifiedModule {
                device_caps: DeviceCaps {
                    optin_shared_bytes: 131_327,
                    ..valid.device_caps
                },
                ..valid
            },
            Tf32QualifiedModule {
                compiler: CompilerIdentity {
                    numeric_abi_revision: 0,
                    ..valid.compiler
                },
                ..valid
            },
        ] {
            assert!(
                resolve_tf32_forced(
                    request(ResolvedGemmOp::Nn),
                    F32TriadAvailability {
                        portable: None,
                        specialized: Some(invalid),
                        finalist: None,
                        joint: None,
                        multiprocessors: 142,
                    },
                    route,
                )
                .is_err(),
                "accepted {invalid:?}"
            );
        }

        let illegal = Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
            tile: Tf32PortableTile::M16N32,
            stages: Tf32PortableStages::S2,
        });
        assert!(
            resolve_tf32_forced(
                request(ResolvedGemmOp::Nn),
                F32TriadAvailability {
                    portable: Some(qualified_module(
                        ModuleKind::TriadSm80,
                        "sm_89",
                        "sm_89",
                        (8, 9),
                        false,
                        99_000,
                    )),
                    specialized: None,
                    finalist: None,
                    joint: None,
                    multiprocessors: 142,
                },
                illegal,
            )
            .is_err()
        );
    }
}

#[cfg(test)]
mod sm100_toolkit_tests {
    use super::super::contract::Sm100TargetKind;
    use super::sm100_target_candidates_for_nvrtc;

    #[test]
    fn older_toolkits_are_offered_only_the_targets_they_can_name() {
        assert!(sm100_target_candidates_for_nvrtc((10, 0), (12, 8)).is_empty());

        let on_12_9 = sm100_target_candidates_for_nvrtc((10, 0), (12, 9));
        assert_eq!(on_12_9.len(), 2);
        assert_eq!(on_12_9[0].nvrtc_arch, "compute_100f");
        assert_eq!(on_12_9[0].kind, Sm100TargetKind::Family);
        assert_eq!(on_12_9[1].kind, Sm100TargetKind::Exact);

        assert!(sm100_target_candidates_for_nvrtc((10, 3), (12, 8)).is_empty());
        assert_eq!(sm100_target_candidates_for_nvrtc((10, 3), (12, 9)).len(), 2);
        assert!(sm100_target_candidates_for_nvrtc((11, 0), (13, 0)).is_empty());
        assert_eq!(sm100_target_candidates_for_nvrtc((11, 0), (13, 2)).len(), 2);
        assert!(sm100_target_candidates_for_nvrtc((10, 7), (13, 3)).is_empty());
        assert_eq!(sm100_target_candidates_for_nvrtc((10, 7), (13, 4)).len(), 2);
        assert!(sm100_target_candidates_for_nvrtc((12, 0), (13, 2)).is_empty());
    }
}

#[cfg(test)]
mod sm100_sm90a_auto_tests {
    use super::super::contract::{
        Sm90aLaunchOperands, Sm90aShape, Sm100LaunchOperands, Sm100PhysicalRoute, Sm100Schedule,
        Sm100Shape, Sm100Stages, Sm100Tile,
    };
    use super::*;

    fn sm100_request(op: Sm100Op, dims: (usize, usize, usize)) -> Sm100AutoRequest {
        let beta = if op == Sm100Op::Tn { 1.0 } else { 0.0 };
        Sm100AutoRequest {
            op,
            dtype: WeightDtype::Bf16,
            shape: Sm100Shape::contiguous(op, dims),
            a_ptr: 0x1_0000,
            b_ptr: 0x2_0000,
            operands: Sm100LaunchOperands {
                output_ptr: 0x3_0000,
                bias_ptr: 0,
                alpha: 1.0,
                beta,
            },
        }
    }

    fn sm100_cell(op: Sm100Op, dims: (usize, usize, usize)) -> Sm100ForcedRoute {
        Sm100ForcedRoute {
            op,
            dtype: WeightDtype::Bf16,
            physical: Sm100PhysicalRoute {
                tile: Sm100Tile::M128N64,
                stages: Sm100Stages::S2,
                schedule: Sm100Schedule::C4,
            },
            shape: Sm100Shape::contiguous(op, dims),
        }
    }

    #[test]
    fn sm100_tables_are_empty_and_the_wave_rule_serves_the_family() {
        let request = sm100_request(Sm100Op::Nn, (2048, 768, 3072));
        // The measured tables stay empty until a board qualifies its cells,
        // and the wave rule serves the family's boards in the meantime: the
        // output fills a device with wide tiles and the reduction feeds three
        // stages but not the larger schedule.
        let expected = Sm100PhysicalRoute {
            tile: Sm100Tile::M128N128,
            stages: Sm100Stages::S3,
            schedule: Sm100Schedule::C4,
        };
        for device_cc in [(10, 0), (10, 3), (10, 7), (11, 0)] {
            assert!(sm100_auto_cells(device_cc).is_empty());
            let target = sm100_target_candidates(device_cc).first().copied();
            let resolved = resolve_sm100_auto(device_cc, target, request)
                .unwrap_or_else(|| panic!("wave rule must serve {device_cc:?}"));
            assert_eq!(resolved.physical, expected);
            assert_eq!(resolved.shape, request.shape);
        }
        // Off the family nothing resolves, wave rule or not.
        for device_cc in [(12, 0), (9, 0)] {
            assert!(sm100_auto_cells(device_cc).is_empty());
            let target = sm100_target_candidates(device_cc).first().copied();
            assert_eq!(resolve_sm100_auto(device_cc, target, request), None);
        }
    }

    #[test]
    fn a_measured_sm100_cell_wins_on_its_board_and_the_rule_takes_the_rest() {
        let dims = (2048, 768, 3072);
        let cells = [sm100_cell(Sm100Op::Nn, dims)];
        let target = sm100_target_candidates((10, 0))[0];
        let request = sm100_request(Sm100Op::Nn, dims);
        assert_eq!(
            resolve_sm100_auto_from_cells(&cells, (10, 0), Some(target), request),
            Some(cells[0])
        );
        // No bound module, another board, a shape one row off, or an
        // epilogue outside the measured law each decline.
        assert_eq!(
            resolve_sm100_auto_from_cells(&cells, (10, 0), None, request),
            None
        );
        assert_eq!(
            resolve_sm100_auto_from_cells(&cells, (10, 3), Some(target), request),
            None
        );
        // A shape one row off the cell is not a decline any more: it takes
        // the wave rule instead of the measured route.
        let mut off = request;
        off.shape.m += 1;
        let drifted = resolve_sm100_auto_from_cells(&cells, (10, 0), Some(target), off)
            .expect("the wave rule serves the drifted shape");
        assert_ne!(drifted, cells[0]);
        assert_eq!(drifted.shape, off.shape);
        let mut biased = request;
        biased.operands.bias_ptr = 0x4_0000;
        biased.operands.alpha = 2.0;
        assert_eq!(
            resolve_sm100_auto_from_cells(&cells, (10, 0), Some(target), biased),
            None
        );
    }

    fn sm90a_request(op: Sm90aOp, dims: (usize, usize, usize)) -> Sm90aAutoRequest {
        let beta = if op == Sm90aOp::Tn { 1.0 } else { 0.0 };
        Sm90aAutoRequest {
            op,
            dtype: WeightDtype::Bf16,
            shape: Sm90aShape::contiguous(op, dims),
            a_ptr: 0x1_0000,
            b_ptr: 0x2_0000,
            operands: Sm90aLaunchOperands {
                output_ptr: 0x3_0000,
                bias_ptr: 0,
                alpha: 1.0,
                beta,
            },
        }
    }

    #[test]
    fn the_sm90a_wave_rule_serves_hopper_and_measured_cells_stay_scoped() {
        let dims = (2048, 768, 3072);
        let request = sm90a_request(Sm90aOp::Nn, dims);
        assert!(SM90A_AUTO_CELLS.is_empty());
        // With the table empty the wave rule serves Hopper: this reduction is
        // too shallow to feed a dedicated producer warpgroup.
        assert_eq!(
            resolve_sm90a_auto((9, 0), true, request).map(|route| route.schedule),
            Some(Sm90aWarpgroupSchedule::Wg1)
        );
        let mut deep = sm90a_request(Sm90aOp::Nn, (2048, 2048, 3072));
        deep.operands.beta = 0.0;
        assert_eq!(
            resolve_sm90a_auto((9, 0), true, deep).map(|route| route.schedule),
            Some(Sm90aWarpgroupSchedule::Wg2)
        );
        assert_eq!(resolve_sm90a_auto((12, 0), true, request), None);
        let cell = Sm90aForcedRoute {
            op: Sm90aOp::Nn,
            dtype: WeightDtype::Bf16,
            schedule: Sm90aWarpgroupSchedule::Wg1,
            shape: Sm90aShape::contiguous(Sm90aOp::Nn, dims),
        };
        assert_eq!(
            resolve_sm90a_auto_from_cells(&[cell], (9, 0), true, request),
            Some(cell)
        );
        assert_eq!(
            resolve_sm90a_auto_from_cells(&[cell], (9, 0), false, request),
            None
        );
        assert_eq!(
            resolve_sm90a_auto_from_cells(&[cell], (12, 0), true, request),
            None
        );
        let mut biased = request;
        biased.operands.bias_ptr = 0x4_0000;
        biased.operands.alpha = 2.0;
        assert_eq!(
            resolve_sm90a_auto_from_cells(&[cell], (9, 0), true, biased),
            None
        );
    }
}
