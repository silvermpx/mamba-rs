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
    F32TriadSelection, F32TriadShape, SM90A_TILE, SM120_FMA_BK, Sm90aForcedRoute, Sm90aOp,
    Sm90aShape, Sm90aWarpgroupSchedule, Sm100ForcedRoute, Sm100Op, Sm100TargetCandidate,
    Sm100TargetKind, Sm120Bk, Sm120FmaRoute, Sm120FmaTile, Sm120ForcedRoute, Sm120LaunchOperands,
    Sm120MapRequest, Sm120Op, Sm120PhysicalRoute, Sm120Schedule, Sm120Shape, Sm120Stages,
    Sm120TargetCandidate, Sm120Tile, Tf32PhysicalRoute, Tf32QualifiedModule, tf32_kernel_spec,
    validate_sm120_map_request,
};
use super::contract::{GemmDims, checked_mul3, checked_tile_grid, checked_usize};
use crate::gpu::context::HalfTriadPolicy;
use crate::gpu::kernel_identity::DeviceCaps;
use crate::gpu::kernel_identity::ResolvedGemmOp;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Tf32ExactShape {
    output_rows: usize,
    output_columns: usize,
    reduction: usize,
}

#[cfg(test)]
mod sm89_exact_f32_tn_admission_tests {
    use super::*;
    use crate::gpu::gemm_bi_triad::Sm89ExactF32TnRoute;
    use crate::gpu::kernel_identity::CudaTarget;

    const KNOWN_IDENTITIES: [Sm89ExactF32QualificationIdentity; 3] = [
        Sm89ExactF32QualificationIdentity {
            nvrtc_version: (12, 8),
            compile_key: [
                22, 133, 2, 113, 79, 140, 174, 92, 64, 55, 72, 26, 191, 32, 144, 16, 129, 71, 18,
                169, 163, 233, 151, 238, 254, 194, 137, 67, 229, 238, 69, 233,
            ],
            artifact_digest: [
                119, 139, 6, 165, 228, 20, 145, 219, 6, 35, 85, 77, 245, 101, 64, 247, 248, 158,
                74, 71, 51, 150, 150, 18, 252, 81, 0, 54, 194, 68, 144, 112,
            ],
            source_digest: [
                42, 227, 116, 0, 192, 38, 217, 202, 183, 32, 74, 114, 1, 61, 242, 232, 153, 15,
                166, 209, 101, 53, 119, 255, 89, 36, 101, 191, 82, 161, 129, 35,
            ],
            header_manifest_digest: [
                58, 2, 32, 37, 216, 76, 62, 36, 155, 189, 92, 49, 247, 246, 159, 151, 126, 27, 28,
                65, 56, 204, 94, 105, 113, 193, 67, 69, 3, 189, 58, 15,
            ],
            nvrtc_library_domain: [
                38, 176, 163, 160, 32, 68, 255, 203, 193, 105, 63, 216, 62, 146, 97, 190, 255, 166,
                146, 164, 251, 207, 227, 172, 94, 157, 140, 135, 152, 11, 177, 85,
            ],
        },
        Sm89ExactF32QualificationIdentity {
            nvrtc_version: (13, 0),
            compile_key: [
                19, 69, 61, 20, 203, 144, 94, 87, 123, 178, 25, 92, 62, 171, 168, 54, 114, 169,
                222, 191, 56, 153, 243, 49, 147, 85, 9, 200, 223, 137, 172, 184,
            ],
            artifact_digest: [
                31, 208, 236, 148, 231, 189, 226, 21, 247, 7, 136, 202, 101, 180, 43, 62, 225, 201,
                46, 5, 21, 235, 33, 219, 8, 8, 252, 210, 60, 150, 180, 61,
            ],
            source_digest: [
                42, 227, 116, 0, 192, 38, 217, 202, 183, 32, 74, 114, 1, 61, 242, 232, 153, 15,
                166, 209, 101, 53, 119, 255, 89, 36, 101, 191, 82, 161, 129, 35,
            ],
            header_manifest_digest: [
                132, 239, 3, 50, 67, 109, 203, 137, 227, 80, 84, 20, 197, 231, 41, 9, 26, 76, 5,
                67, 51, 57, 230, 106, 95, 57, 79, 61, 144, 75, 194, 62,
            ],
            nvrtc_library_domain: [
                112, 155, 145, 195, 107, 251, 14, 217, 102, 238, 105, 173, 200, 214, 248, 127, 241,
                16, 238, 207, 61, 251, 80, 96, 54, 127, 24, 60, 230, 20, 235, 13,
            ],
        },
        Sm89ExactF32QualificationIdentity {
            nvrtc_version: (13, 2),
            compile_key: [
                83, 124, 13, 149, 236, 157, 74, 190, 239, 221, 245, 103, 76, 179, 93, 193, 227,
                154, 223, 88, 202, 153, 140, 158, 163, 158, 181, 28, 112, 82, 234, 187,
            ],
            artifact_digest: [
                104, 186, 214, 68, 145, 122, 106, 172, 70, 82, 22, 20, 112, 248, 126, 174, 157, 4,
                72, 49, 184, 100, 198, 228, 141, 126, 12, 21, 25, 171, 244, 73,
            ],
            source_digest: [
                42, 227, 116, 0, 192, 38, 217, 202, 183, 32, 74, 114, 1, 61, 242, 232, 153, 15,
                166, 209, 101, 53, 119, 255, 89, 36, 101, 191, 82, 161, 129, 35,
            ],
            header_manifest_digest: [
                252, 26, 221, 32, 31, 26, 175, 118, 209, 248, 228, 199, 168, 92, 58, 101, 51, 220,
                250, 147, 173, 57, 210, 137, 130, 77, 100, 73, 16, 27, 233, 223,
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
            target: crate::gpu::kernel_identity::CudaTarget::new("sm_89").unwrap(),
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
            inference_artifact: ArtifactIdentity {
                module_kind: ModuleKind::Inference,
                artifact_kind: ArtifactKind::Ptx,
                compile_key: [3; 32],
                artifact_digest: [4; 32],
            },
            inference_compiler: compiler,
            inference_copyplan_loaded: false,
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
            sm120_scalar_resources_qualified: false,
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
    fn every_aligned_ada_split_m_takes_the_direct_partial_of_its_partition() {
        let base = facts(KNOWN_IDENTITIES[2]);
        let tn = |dims| F32TriadRequest {
            op: ResolvedGemmOp::Tn,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Tn, dims),
        };
        for dims in [(4_096, 256, 128), (1_024, 1_024, 256), (4_096, 128, 796)] {
            let ScalarDispatchPlan::TnSplitM { m_chunk, chunks } =
                scalar_dispatch_plan(tn(dims)).unwrap()
            else {
                panic!("{dims:?} is a split-M shape");
            };
            assert_eq!(
                scalar_launch_plan(base, tn(dims), operands()).unwrap(),
                ScalarDispatchPlan::TnSm89DirectSplitM { m_chunk, chunks },
                "{dims:?}"
            );
        }
        // A width that is not whole 16-byte chunks, an unaligned operand, a
        // board without the module's evidence and a missing kernel keep the
        // portable partial.
        let fallback = scalar_dispatch_plan(tn((4_096, 130, 128))).unwrap();
        assert_eq!(
            scalar_launch_plan(base, tn((4_096, 130, 128)), operands()).unwrap(),
            fallback
        );
        let unaligned = F32TriadOperands {
            a: 0x1004,
            ..operands()
        };
        let plain = scalar_dispatch_plan(tn((4_096, 256, 128))).unwrap();
        assert_eq!(
            scalar_launch_plan(base, tn((4_096, 256, 128)), unaligned).unwrap(),
            plain
        );
        let other_board = ScalarLaunchFacts {
            compute_capability: (8, 6),
            ..base
        };
        assert_eq!(
            scalar_launch_plan(other_board, tn((4_096, 256, 128)), operands()).unwrap(),
            plain
        );
        let without_kernel = ScalarLaunchFacts {
            sm89_exact_f32_symbols_loaded: [true, false, true],
            ..base
        };
        assert_eq!(
            scalar_launch_plan(without_kernel, tn((4_096, 256, 128)), operands()).unwrap(),
            plain
        );
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
            // A mutation that only drops the cell's own kernel leaves the
            // general direct partial of the same partition, which writes the
            // fallback's bits.
            let general = match fallback {
                ScalarDispatchPlan::TnSplitM { m_chunk, chunks } => {
                    ScalarDispatchPlan::TnSm89DirectSplitM { m_chunk, chunks }
                }
                other => other,
            };
            for mutation in mutations {
                let plan = scalar_launch_plan(mutation, request, operands()).unwrap();
                assert!(
                    plan == fallback || plan == general,
                    "identity mutation admitted {route:?}: {plan:?} on {mutation:?}",
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
    use crate::gpu::gemm_bi_triad::Sm89ExactF32D128Route;
    use crate::gpu::kernel_identity::CudaTarget;

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
            inference_artifact: ArtifactIdentity {
                module_kind: ModuleKind::Inference,
                artifact_kind: ArtifactKind::Ptx,
                compile_key: [3; 32],
                artifact_digest: [4; 32],
            },
            inference_compiler: compiler,
            inference_copyplan_loaded: false,
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
            sm120_scalar_resources_qualified: false,
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
                assert_eq!(scalar_dispatch_plan(request(dims)).unwrap(), FALLBACK);
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
            let hopper = ScalarLaunchFacts {
                compute_capability: (9, 0),
                ..ada
            };
            let expected =
                (scalar_dispatch_plan(request(dims)).unwrap() == FALLBACK).then_some(selected);
            assert_eq!(
                scalar_proof_plan(hopper, request(dims), operands()).unwrap(),
                expected,
                "{route:?} stands in for exactly the split it reproduces"
            );
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

/// The architecture a frozen TF32 evidence cohort belongs to. Evidence
/// belongs to an architecture, not to the card it was measured on: it serves
/// any module of the same kind, built by this tree's composer, compiler and
/// numeric ABI, on a device of the same compute-capability family that offers
/// at least the shared memory the measuring card offered. The card's SM count,
/// the toolkit, the driver and the exact bytes the toolkit produced never
/// decide who may use it (docs/architecture-dispatch.md); each cohort's
/// comment records where its tournament ran.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Tf32EvidenceArchitecture {
    module_kind: ModuleKind,
    compute_capability: (u32, u32),
    optin_shared_bytes: u32,
    tensor_map_access: bool,
}

impl Tf32EvidenceArchitecture {
    fn matches(self, module: Tf32QualifiedModule) -> bool {
        self.mismatch(module).is_none()
    }

    /// The first check that keeps this evidence from the bound module, or
    /// `None` when the evidence serves it. The check's name tells a module of
    /// another architecture from one this tree did not build.
    fn mismatch(self, module: Tf32QualifiedModule) -> Option<&'static str> {
        let checks: [(&'static str, bool); 15] = [
            ("module kind", module.module_kind == self.module_kind),
            (
                "artifact module kind",
                module.artifact.module_kind == self.module_kind,
            ),
            (
                "artifact kind",
                module.artifact.artifact_kind == ArtifactKind::Ptx,
            ),
            (
                "compiler output kind",
                module.compiler.output_kind == ArtifactKind::Ptx,
            ),
            ("nvrtc library known", module.compiler.nvrtc_library_known),
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
            ("compiler target", module.compiler.target == module.target),
            (
                "accepted target",
                module.device_caps.accepted_target == Some(module.target),
            ),
            (
                "device architecture",
                same_architecture(module.device.compute_capability, self.compute_capability),
            ),
            (
                "caps architecture",
                same_architecture(
                    module.device_caps.compute_capability,
                    self.compute_capability,
                ),
            ),
            (
                "opt-in shared bytes",
                module.device_caps.optin_shared_bytes >= self.optin_shared_bytes,
            ),
            (
                "tensor map access",
                module.device_caps.tensor_map_access || !self.tensor_map_access,
            ),
        ];
        checks
            .into_iter()
            .find(|(_, holds)| !holds)
            .map(|(field, _)| field)
    }
}

/// Whether two compute capabilities are one architecture for frozen
/// evidence: the same capability, or two of the SM120 family, whose cards
/// share the instruction set the SM120 module compiles for.
fn same_architecture(device: (u32, u32), evidence: (u32, u32)) -> bool {
    device == evidence
        || (crate::gpu::device::is_sm120_family(device)
            && crate::gpu::device::is_sm120_family(evidence))
}

const SM89_TF32_QUALIFIED_TUNING_REVISION: u16 = F32_TF32_TUNING_REVISION;
/// The Ada portable cohort's architecture: the portable module on an SM89
/// card. Its tournament ran on an RTX 6000 Ada (142 SMs) under NVRTC 13.2; it
/// serves every SM89 card, the L40S and the RTX 4090 among them, on any
/// toolkit.
const SM89_TF32_ARCHITECTURE: Tf32EvidenceArchitecture = Tf32EvidenceArchitecture {
    module_kind: ModuleKind::TriadSm80,
    compute_capability: (8, 9),
    optin_shared_bytes: 101_376,
    tensor_map_access: false,
};

/// The Ada joint module's architecture. Its tournaments ran on an RTX 6000
/// Ada under NVRTC 12.8, 13.0 and 13.2; the 13.2 winners serve every SM89
/// card.
const SM89_JOINT_TF32_ARCHITECTURE: Tf32EvidenceArchitecture = Tf32EvidenceArchitecture {
    module_kind: ModuleKind::TriadSm89Tf32Joint,
    compute_capability: (8, 9),
    optin_shared_bytes: 101_376,
    tensor_map_access: false,
};

const SM120_TF32_QUALIFIED_TUNING_REVISION: u16 = F32_TF32_TUNING_REVISION;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Tf32AutoCell {
    op: crate::gpu::kernel_identity::ResolvedGemmOp,
    shape: Tf32ExactShape,
    route: Tf32PhysicalRoute,
    operand_gate: Tf32AutoOperandGate,
}

/// One architecture's frozen tournament: the module architecture it serves,
/// the portable module architecture its portable routes run on (a
/// specialized cohort only), the dispatch epoch authorized to use those
/// measurements, and its cells. Each architecture has one cohort, so every
/// card of it takes the same route for a shape and gives the same bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Tf32AutoEvidenceCohort {
    identity: Tf32EvidenceArchitecture,
    portable: Option<Tf32EvidenceArchitecture>,
    tuning_revision: u16,
    cells: &'static [Tf32AutoCell],
}

const fn sm89_tf32_cell(
    op: crate::gpu::kernel_identity::ResolvedGemmOp,
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
    op: crate::gpu::kernel_identity::ResolvedGemmOp,
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
use crate::gpu::kernel_identity::ResolvedGemmOp::{Nn, Nt, Tn};
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
    op: crate::gpu::kernel_identity::ResolvedGemmOp,
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
    op: crate::gpu::kernel_identity::ResolvedGemmOp,
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

/// The SM120 architecture's winners, from the tournament on an RTX PRO 6000
/// Blackwell; they serve every SM120 card. An NN cell admits no bias. Three
/// winners stay out because the route served without them measured faster:
/// on both large deep projections a proven Ada kernel the tournament does
/// not time, by 7 and 19 percent, and on the NT d128 out-projection the
/// portable kernel, by 14 percent when launched eagerly. The TN d128
/// out-projection has no cell either: there the exact family ran 25 percent
/// ahead of the best TF32 route.
const SM120_TF32_EVIDENCE_CELLS: &[Tf32AutoCell] = &[
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
        super::contract::Tf32Sm120Tile::M64N128,
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
    sm120_tf32_cell(
        Nn,
        4096,
        1536,
        3072,
        super::contract::Tf32Sm120Tile::M64N128,
        super::contract::Tf32Sm120Stages::S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
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
            stages: super::contract::Tf32PortableStages::S2,
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
        route: Tf32PhysicalRoute::MmaTf32RnaSplitK4(super::contract::Tf32PortableRoute {
            tile: super::contract::Tf32PortableTile::M16N32,
            stages: super::contract::Tf32PortableStages::S3,
        }),
        operand_gate: RequiresVectorAlignmentEvidence,
    },
    Tf32AutoCell {
        op: Nn,
        shape: Tf32ExactShape {
            output_rows: 128,
            output_columns: 128,
            reduction: 8192,
        },
        route: Tf32PhysicalRoute::MmaTf32RnaSplitK4(super::contract::Tf32PortableRoute {
            tile: super::contract::Tf32PortableTile::M16N32,
            stages: super::contract::Tf32PortableStages::S4,
        }),
        operand_gate: RequiresNoBiasAndVectorAlignmentEvidence,
    },
    sm120_tf32_cell(
        Nn,
        10400,
        1536,
        384,
        super::contract::Tf32Sm120Tile::M64N128,
        super::contract::Tf32Sm120Stages::S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Tn,
        8192,
        128,
        128,
        super::contract::Tf32Sm120Tile::M64N128,
        super::contract::Tf32Sm120Stages::S4,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_streamk_cell(
        Tn,
        384,
        1536,
        10400,
        super::contract::Tf32Sm120Stages::S3,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nt,
        128,
        8192,
        128,
        super::contract::Tf32Sm120Tile::M64N128,
        super::contract::Tf32Sm120Stages::S3,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nt,
        10400,
        384,
        1536,
        super::contract::Tf32Sm120Tile::M128N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nn,
        4621,
        384,
        768,
        super::contract::Tf32Sm120Tile::M64N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm120_tf32_streamk_cell(
        Tn,
        768,
        384,
        4621,
        super::contract::Tf32Sm120Stages::S3,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nt,
        4621,
        768,
        384,
        super::contract::Tf32Sm120Tile::M128N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nn,
        4621,
        384,
        1024,
        super::contract::Tf32Sm120Tile::M64N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm120_tf32_streamk_cell(
        Tn,
        1024,
        384,
        4621,
        super::contract::Tf32Sm120Stages::S3,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nt,
        4621,
        1024,
        384,
        super::contract::Tf32Sm120Tile::M64N128,
        super::contract::Tf32Sm120Stages::S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nn,
        10400,
        384,
        384,
        super::contract::Tf32Sm120Tile::M128N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm120_tf32_streamk_cell(
        Tn,
        384,
        384,
        10400,
        super::contract::Tf32Sm120Stages::S3,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nt,
        10400,
        384,
        384,
        super::contract::Tf32Sm120Tile::M64N128,
        super::contract::Tf32Sm120Stages::S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nn,
        10400,
        384,
        768,
        super::contract::Tf32Sm120Tile::M64N128,
        super::contract::Tf32Sm120Stages::S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm120_tf32_streamk_cell(
        Tn,
        768,
        384,
        10400,
        super::contract::Tf32Sm120Stages::S3,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nt,
        10400,
        768,
        384,
        super::contract::Tf32Sm120Tile::M64N128,
        super::contract::Tf32Sm120Stages::S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nn,
        4096,
        768,
        512,
        super::contract::Tf32Sm120Tile::M64N128,
        super::contract::Tf32Sm120Stages::S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm120_tf32_streamk_cell(
        Tn,
        512,
        768,
        4096,
        super::contract::Tf32Sm120Stages::S3,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nt,
        4096,
        512,
        768,
        super::contract::Tf32Sm120Tile::M64N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresVectorAlignmentEvidence,
    ),
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
    Tf32AutoCell {
        op: Nt,
        shape: Tf32ExactShape {
            output_rows: 256,
            output_columns: 512,
            reduction: 384,
        },
        route: Tf32PhysicalRoute::MmaTf32Rna(super::contract::Tf32PortableRoute {
            tile: super::contract::Tf32PortableTile::M16N16,
            stages: super::contract::Tf32PortableStages::S4,
        }),
        operand_gate: RequiresVectorAlignmentEvidence,
    },
];

/// The SM120 specialized module's architecture, and the portable module its
/// portable cells run on. The tournament ran on an RTX PRO 6000 Blackwell
/// (188 SMs) under NVRTC 13.2.78; its winners serve every SM120 card.
const SM120_TF32_ARCHITECTURE: Tf32EvidenceArchitecture = Tf32EvidenceArchitecture {
    module_kind: ModuleKind::TriadSm120,
    compute_capability: (12, 0),
    optin_shared_bytes: 101_376,
    tensor_map_access: true,
};

const SM120_TF32_PORTABLE_ARCHITECTURE: Tf32EvidenceArchitecture = Tf32EvidenceArchitecture {
    module_kind: ModuleKind::TriadSm80,
    compute_capability: (12, 0),
    optin_shared_bytes: 101_376,
    tensor_map_access: true,
};

/// The SM120 architecture's one cohort.
const SM120_TF32_EVIDENCE_COHORTS: &[Tf32AutoEvidenceCohort] = &[Tf32AutoEvidenceCohort {
    identity: SM120_TF32_ARCHITECTURE,
    portable: Some(SM120_TF32_PORTABLE_ARCHITECTURE),
    tuning_revision: SM120_TF32_QUALIFIED_TUNING_REVISION,
    cells: SM120_TF32_EVIDENCE_CELLS,
}];

/// The Ada portable cohort: boards without a specialized module read it.
const SM89_TF32_EVIDENCE_COHORTS: &[Tf32AutoEvidenceCohort] = &[Tf32AutoEvidenceCohort {
    identity: SM89_TF32_ARCHITECTURE,
    portable: None,
    tuning_revision: SM89_TF32_QUALIFIED_TUNING_REVISION,
    cells: SM89_TF32_EVIDENCE_CELLS,
}];

/// CUDA 13.2 independently qualified the prior five rows and the NT d768-in
/// A-ldmatrix winner into the joint Ada module.
const SM89_JOINT_TF32_EVIDENCE_CELLS: &[Tf32AutoCell] = &[
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

const SM89_JOINT_TF32_EVIDENCE_COHORTS: &[Tf32AutoEvidenceCohort] = &[Tf32AutoEvidenceCohort {
    identity: SM89_JOINT_TF32_ARCHITECTURE,
    portable: None,
    tuning_revision: super::contract::SM89_TF32_JOINT_TUNING_REVISION,
    cells: SM89_JOINT_TF32_EVIDENCE_CELLS,
}];

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

/// The Ada finalist module's architecture. Its tournaments ran on an RTX 6000
/// Ada under NVRTC 12.8, 13.0 and 13.2 and froze the same winners each time.
const SM89_FINALIST_TF32_ARCHITECTURE: Tf32EvidenceArchitecture = Tf32EvidenceArchitecture {
    module_kind: ModuleKind::TriadSm89Finalist,
    compute_capability: (8, 9),
    optin_shared_bytes: 101_376,
    tensor_map_access: false,
};

/// Separate finalist evidence never rewrites the incumbent portable cohort.
const SM89_FINALIST_TF32_EVIDENCE_COHORTS: &[Tf32AutoEvidenceCohort] = &[Tf32AutoEvidenceCohort {
    identity: SM89_FINALIST_TF32_ARCHITECTURE,
    portable: None,
    tuning_revision: super::contract::SM89_FINALIST_TUNING_REVISION,
    cells: SM89_FINALIST_TF32_NT_CELLS,
}];

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
/// this way: its plain tiles keep one owner CTA per output tile and one k
/// order, so any of them gives the same bits, and what the neighbour
/// contributes is a speed choice; a split-K neighbour brings its own
/// reduction order, and its partials must fit the fixed workspace (see
/// [`resolve_tf32_forced`]). The specialized tiles stay on their exact
/// cells. A neighbour measured on a larger output may carry a wide
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
    if cohort.identity != SM89_TF32_ARCHITECTURE
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
        crate::gpu::diagnostics::warn_once(&NO_COHORT, || {
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
        crate::gpu::diagnostics::warn_once(&STALE_TUNING, || {
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
                    crate::gpu::diagnostics::warn_once(&DECLINED, || {
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
        F32TriadPolicy::RnaTf32BatchInvariant => {
            rna_batch_invariant_selection(request, operands, availability)
        }
        // The forward folds by its layer's shape on the scalar tiles, whose
        // plan the launcher takes from `exact_batch_invariant_nn_plan`; the
        // gradients are the exact policy's.
        F32TriadPolicy::ExactFmaBatchInvariant => Ok(if request.op == ResolvedGemmOp::Nn {
            F32TriadSelection::ScalarFma
        } else {
            exact_or_scalar_selection(request, operands, availability)
        }),
    }
}

/// The strict policy's selection: the architecture's measured route when it
/// keeps the contract, else the first-use proof tier's routes that keep it,
/// else the portable tier's plain tile for the shape. A shape no route can
/// serve is an error, never the exact family.
fn rna_batch_invariant_selection(
    request: F32TriadRequest,
    operands: Option<F32TriadOperands>,
    availability: F32TriadAvailability,
) -> Result<F32TriadSelection, String> {
    if availability.portable.is_none() {
        return Err(
            "the RNA batch-invariant TF32 policy needs the portable TF32 module, and this \
             context bound none"
                .to_string(),
        );
    }
    let keeps = |route: Tf32PhysicalRoute| rna_batch_invariant_route(request.op, route);
    // The architecture's own forward table, where it has one: every tile it
    // names walks the portable tiles' chain, so it picks by speed alone.
    if let Some(route) = rna_forward_table_route(request, operands, availability) {
        let bound = resolve_tf32_forced(request, availability, route).or_else(|_| {
            resolve_tf32_forced(
                request,
                availability,
                rna_wide_portable_twin(request, route),
            )
        })?;
        return Ok(F32TriadSelection::Tf32(bound));
    }
    // A measured cell keeps its route when the cohort crowned a specialized
    // kernel for it. A cell whose cohort winner is a portable tile or a
    // portable split member takes the gradients' rules instead: they pick
    // among the split members, the multistage tiles and the pre-rounded
    // tiles, most of which the cohort's tournaments never ran (TN
    // 1024x256x1024 37.7 -> 19.4 us on the RTX 6000 Ada), and they cut a
    // reduction by its shape where the cohort split it by its own cell. A
    // tile the rules stage from the operands falls back to the portable rule
    // when this toolkit cannot bind it.
    let measured = operands
        .and_then(|operands| {
            measured_tf32_route_with_operands(
                request,
                operands,
                availability,
                F32_TF32_TUNING_REVISION,
            )
        })
        .filter(|route| keeps(*route))
        .and_then(|route| resolve_tf32_forced(request, availability, route).ok());
    if let Some(route) = measured
        && !matches!(
            route,
            Tf32PhysicalRoute::MmaTf32Rna(_)
                | Tf32PhysicalRoute::MmaTf32RnaSplitK2(_)
                | Tf32PhysicalRoute::MmaTf32RnaSplitK4(_)
                | Tf32PhysicalRoute::MmaTf32RnaSplitK8(_)
        )
    {
        return Ok(F32TriadSelection::Tf32(route));
    }
    let ruled = rna_backward_rule_route(request, operands, availability)
        .and_then(|route| resolve_tf32_forced(request, availability, route).ok())
        .or_else(|| {
            rna_backward_rule_route(request, None, availability)
                .and_then(|route| resolve_tf32_forced(request, availability, route).ok())
        });
    if let Some(bound) = ruled.or(measured) {
        return Ok(F32TriadSelection::Tf32(bound));
    }
    let proven = operands
        .and_then(|operands| proof_tf32_selection(request, operands, availability))
        .and_then(|selection| match selection {
            F32TriadSelection::Tf32(route) => keeps(route).then_some(selection),
            F32TriadSelection::Tf32Proof {
                candidate,
                reference,
            } if keeps(reference) => Some(if keeps(candidate) {
                selection
            } else {
                F32TriadSelection::Tf32(reference)
            }),
            _ => None,
        });
    if let Some(selection) = proven {
        return Ok(selection);
    }
    let route = rna_portable_ladder_route(request);
    resolve_tf32_forced(request, availability, route)
        .map(F32TriadSelection::Tf32)
        .map_err(|reason| {
            format!("the RNA batch-invariant TF32 policy has no route for {request:?}: {reason}")
        })
}

/// A weight gradient's reduction from which the strict policy splits it
/// eight ways: the deep batches whose few output tiles leave the card idle
/// over one long chain.
const RNA_TN_SPLIT_MIN_REDUCTION: usize = 2048;

/// The reduction and the 64 x 64 output grid inside which a small layer's
/// weight gradient splits eight ways below the deep-batch split.
const RNA_TN_SMALL_SPLIT_MIN_REDUCTION: usize = 1024;
const RNA_TN_SMALL_SPLIT_MAX_TILES: usize = 16;

/// The batch below which an input gradient's output is a few 16-row slabs,
/// and the reduction and output bounds inside which splitting its reduction
/// four ways fills the card.
const RNA_NT_NARROW_ROWS: usize = 256;
const RNA_NT_SPLIT_MIN_REDUCTION: usize = 256;
const RNA_NT_SPLIT_MAX_COLUMNS: usize = 1024;

/// Whether the 64 x 96 pre-rounded weight-gradient tile serves rather than
/// the 64 x 64 one: the 64 x 64 grid needs more than one wave of the live
/// multiprocessors. Both tiles walk one chain per output, so the pick reads
/// the live count. It reproduces the RTX 6000 Ada's measured picks (the
/// 590k-output gradients take the wider tile, the 262k-320k ones do not);
/// on an RTX 5090 and an RTX PRO 6000 it ran 144 pre-rounded shapes in 0.98
/// and 0.99 of the time of an output-size bound set on the Ada, none slower.
/// Weighing the busiest multiprocessor's work instead gained 0.92 on the
/// RTX 5090 but slowed 20 shapes on the RTX PRO 6000.
fn rna_pre_tn_wide_pays(rows: usize, columns: usize, multiprocessors: usize) -> bool {
    multiprocessors != 0 && rows.div_ceil(64) * columns.div_ceil(64) > multiprocessors
}

/// The outputs, rows by columns, one CTA of a wide or multistage tile covers.
const fn rna_wide_tile_outputs(tile: super::contract::Tf32RnaWideTile) -> (usize, usize) {
    use super::contract::Tf32RnaWideTile as Wide;
    match tile {
        Wide::M128N96S3 => (128, 96),
        Wide::M96N96S3 => (96, 96),
        Wide::M128N128S3 | Wide::NtMsM128N128Bk16S5 => (128, 128),
        Wide::MsM64N128Bk16S6 | Wide::NtMsM64N128Bk16S6 => (64, 128),
        Wide::MsM128N64S3 | Wide::NtMsM128N64S3 | Wide::MsPersistM128N64Bk16S3 => (128, 64),
        Wide::MsM64N64S3 => (64, 64),
    }
}

/// The strict policy's gradient routes. The gradients only keep cvt.rna and
/// a fixed order, so a split by a shape-only partition count serves them; the
/// members of one partition count fold the same partitions, so which member
/// runs is speed alone and may read the card. A weight gradient over a deep
/// batch splits eight ways, an input gradient of a few rows over a long
/// reduction four ways, and a narrow output takes the 16 x 32 tile. On the
/// Ada a weight gradient no split claims runs over a pre-rounded copy of its
/// input when the joint module is bound and the operands stage whole chunks.
fn rna_backward_rule_route(
    request: F32TriadRequest,
    operands: Option<F32TriadOperands>,
    availability: F32TriadAvailability,
) -> Option<Tf32PhysicalRoute> {
    use super::contract::{Tf32PortableRoute, Tf32PortableStages, Tf32PortableTile};
    let portable = |tile, stages| Tf32PortableRoute { tile, stages };
    let rows = request.shape.output_rows(request.op);
    let columns = request.shape.output_columns(request.op);
    let reduction = request.shape.reduction(request.op);
    if request.op != ResolvedGemmOp::Nn && availability.portable?.device.compute_capability.0 == 12
    {
        return Some(rna_blackwell_backward_route(
            request,
            operands,
            availability,
        ));
    }
    match request.op {
        ResolvedGemmOp::Nn => None,
        // The Ada's multistage members of the same eight partitions, where
        // the rows and columns are whole 16-byte chunks and the operands
        // stage, measured against the portable members (TN 4096x256x2112
        // 123.9 -> 58.4 us, 4096x128x796 33.8 -> 19.5, 4096x256x128 20.5 ->
        // 11.3): 128 x 64 from 256 columns, 64 x 64 from 128.
        ResolvedGemmOp::Tn
            if reduction >= RNA_TN_SPLIT_MIN_REDUCTION
                && columns >= 128
                && rows.is_multiple_of(4)
                && columns.is_multiple_of(4)
                && rna_tiles_stage(request, operands, availability) =>
        {
            Some(Tf32PhysicalRoute::MmaTf32RnaWideSplitK8(
                if columns >= 256 {
                    super::contract::Tf32RnaSplitTile::M128N64S3
                } else {
                    super::contract::Tf32RnaSplitTile::M64N64S4
                },
            ))
        }
        ResolvedGemmOp::Tn if reduction >= RNA_TN_SPLIT_MIN_REDUCTION => {
            let ada = availability.portable?.device.compute_capability == (8, 9);
            let member = if ada {
                // Measured on the RTX 6000 Ada: the 64 x 64 member from 512
                // columns, the 32 x 32 one's four times the CTAs below.
                if columns >= 512 {
                    portable(Tf32PortableTile::M64N64, Tf32PortableStages::S2)
                } else {
                    portable(Tf32PortableTile::M32N32, Tf32PortableStages::S3)
                }
            } else {
                // Under half a wave of 64 x 64 CTAs the 32 x 32 member's four
                // times as many win; from four waves the third stage pays.
                let ctas = rows.div_ceil(64) * columns.div_ceil(64) * 8;
                let sms = availability.multiprocessors as usize;
                if 2 * ctas < sms {
                    portable(Tf32PortableTile::M32N32, Tf32PortableStages::S3)
                } else if ctas >= 4 * sms {
                    portable(Tf32PortableTile::M64N64, Tf32PortableStages::S3)
                } else {
                    portable(Tf32PortableTile::M64N64, Tf32PortableStages::S2)
                }
            };
            Some(Tf32PhysicalRoute::MmaTf32RnaSplitK8(member))
        }
        ResolvedGemmOp::Tn if columns < 128 => Some(Tf32PhysicalRoute::MmaTf32Rna(portable(
            Tf32PortableTile::M16N32,
            Tf32PortableStages::S4,
        ))),
        // A small layer's weight gradient at least 128 wide over a reduction
        // of 1024 or more splits eight ways as well: a 64 x 64 grid of 16
        // tiles or fewer leaves most of the card idle, and the 32 x 32
        // member's four times the CTAs fill it (RTX 6000 Ada: TN
        // 1536x128x384 25.8 -> 14.4 us on the pre-rounded tile, 1024x192x256
        // 18.8 -> 9.9).
        ResolvedGemmOp::Tn
            if reduction >= RNA_TN_SMALL_SPLIT_MIN_REDUCTION
                && rows.div_ceil(64) * columns.div_ceil(64) <= RNA_TN_SMALL_SPLIT_MAX_TILES =>
        {
            Some(Tf32PhysicalRoute::MmaTf32RnaSplitK8(portable(
                Tf32PortableTile::M32N32,
                Tf32PortableStages::S3,
            )))
        }
        ResolvedGemmOp::Tn if rna_pre_tn_admits(request, operands, availability) => Some(
            if reduction >= 256
                && rna_pre_tn_wide_pays(rows, columns, availability.multiprocessors as usize)
            {
                Tf32PhysicalRoute::Sm89TnPreRnaM64N96S2
            } else {
                Tf32PhysicalRoute::Sm89TnPreRnaM64N64
            },
        ),
        ResolvedGemmOp::Nt
            if rows < RNA_NT_NARROW_ROWS
                && reduction >= RNA_NT_SPLIT_MIN_REDUCTION
                && columns <= RNA_NT_SPLIT_MAX_COLUMNS =>
        {
            Some(Tf32PhysicalRoute::MmaTf32RnaSplitK4(portable(
                Tf32PortableTile::M16N32,
                Tf32PortableStages::S3,
            )))
        }
        ResolvedGemmOp::Nt if rows < RNA_NT_NARROW_ROWS => Some(Tf32PhysicalRoute::MmaTf32Rna(
            portable(Tf32PortableTile::M16N32, Tf32PortableStages::S4),
        )),
        // The Ada's multistage input-gradient tiles, measured on an RTX 6000
        // Ada against the portable tiles
        // (NT 1024x1536x384 34.9 -> 16.2 us, 4096x256x2112 75.0 -> 49.8): the
        // 128 x 128 tile once its grid fills two thirds of the card, the
        // 64 x 128 tile's twice the CTAs below.
        ResolvedGemmOp::Nt
            if rows >= 512
                && columns >= 128
                && reduction >= 64
                && rna_tiles_stage(request, operands, availability) =>
        {
            let tiles = rows.div_ceil(128) * columns.div_ceil(128);
            let sms = availability.multiprocessors as usize;
            Some(Tf32PhysicalRoute::MmaTf32RnaWide(if 3 * tiles >= 2 * sms {
                super::contract::Tf32RnaWideTile::NtMsM128N128Bk16S5
            } else {
                super::contract::Tf32RnaWideTile::NtMsM64N128Bk16S6
            }))
        }
        ResolvedGemmOp::Tn | ResolvedGemmOp::Nt => Some(Tf32PhysicalRoute::MmaTf32Rna(portable(
            Tf32PortableTile::M64N64,
            Tf32PortableStages::S2,
        ))),
    }
}

/// The strict policy's gradient routes on a CC 12.x board, as measured on
/// an RTX 5090 (interleaved sweeps,
/// device minimum per launch): the multistage 64 x 64 member of the eight
/// partitions for a deep weight gradient, the 16 x 32 tile for a small
/// layer over a few hundred to a thousand rows (TN 896x256x256 18.1 -> 9.9
/// us, NT 896x256x256 16.0 -> 7.8), the pre-rounded tiles for a wider
/// weight gradient, and the multistage input-gradient tiles from 786k
/// outputs over a reduction of 256 or more (NT 4096x256x2112 77.5 -> 54.9,
/// 1024x833x384 24.2 -> 16.0). The partition counts are the shape's; the
/// tiles walk one chain each, so no pick moves a bit. Where the table names
/// the portable 64 x 64 tile for a staged input gradient, the SM120 TMA
/// 64 x 64 tile walks the same chain ([`rna_nt_tma_twin`]).
fn rna_blackwell_backward_route(
    request: F32TriadRequest,
    operands: Option<F32TriadOperands>,
    availability: F32TriadAvailability,
) -> Tf32PhysicalRoute {
    let table = rna_blackwell_backward_table(request, operands, availability);
    if request.op != ResolvedGemmOp::Nt || !rna_tiles_stage(request, operands, availability) {
        return table;
    }
    rna_nt_tma_twin(table, availability)
}

/// The SM120 TMA 64 x 64 tile in place of the portable 64 x 64 one the
/// table names for a strict input gradient, when the SM120 module is bound
/// with tensor-map access: the same tile and chain fed by TMA. On an RTX
/// 5090 and an RTX PRO 6000 the input gradients ran in 0.91 and 0.90 of the
/// table's time, none slower. Weighing the busiest multiprocessor's work
/// over the TMA, 144 x 96 and multistage tiles gained 0.85 on the RTX 5090
/// but slowed 61 shapes on the RTX PRO 6000, up to 1.42 times.
fn rna_nt_tma_twin(
    table: Tf32PhysicalRoute,
    availability: F32TriadAvailability,
) -> Tf32PhysicalRoute {
    use super::contract::{
        Tf32PortableRoute, Tf32PortableTile, Tf32Sm120Route, Tf32Sm120Stages, Tf32Sm120Tile,
    };
    let tma_binds = availability.specialized.is_some_and(|module| {
        module.module_kind == ModuleKind::TriadSm120 && module.device_caps.tensor_map_access
    });
    match table {
        Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
            tile: Tf32PortableTile::M64N64,
            ..
        }) if tma_binds => Tf32PhysicalRoute::Sm120TmaMmaTf32Rna(Tf32Sm120Route {
            tile: Tf32Sm120Tile::M64N64,
            stages: Tf32Sm120Stages::S2,
        }),
        other => other,
    }
}

/// The CC 12.x gradient table [`rna_blackwell_backward_route`] starts from.
fn rna_blackwell_backward_table(
    request: F32TriadRequest,
    operands: Option<F32TriadOperands>,
    availability: F32TriadAvailability,
) -> Tf32PhysicalRoute {
    use super::contract::{
        Tf32PortableRoute, Tf32PortableStages, Tf32PortableTile, Tf32RnaWideTile,
    };
    let portable = |tile, stages| Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute { tile, stages });
    let (m, k, n) = (request.shape.m, request.shape.k, request.shape.n);
    let rows = request.shape.output_rows(request.op);
    let columns = request.shape.output_columns(request.op);
    let reduction = request.shape.reduction(request.op);
    let staged = rna_tiles_stage(request, operands, availability);
    const SMALL_LAYER_WIDTH: usize = 512;
    const SMALL_LAYER_GRID: usize = 64;
    const MS_NT_MIN_OUTPUTS: usize = 3 << 18;
    const MS_NT_NARROW_OUTPUTS: usize = 1 << 20;
    const MS_NT_SQUARE_OUTPUTS: usize = 1 << 22;
    let small_layer = (RNA_NT_NARROW_ROWS..2_048).contains(&m)
        && k <= SMALL_LAYER_WIDTH
        && n <= SMALL_LAYER_WIDTH;
    if request.op == ResolvedGemmOp::Tn {
        if reduction >= RNA_TN_SPLIT_MIN_REDUCTION {
            // Every deep weight gradient cuts eight ways, whatever its
            // operands' alignment: a CC 12.x portable module carries no
            // split members, so operands the 16-byte member cannot stage go
            // to the member that stages them four bytes at a time, which
            // folds the same partitions to the same bits.
            return Tf32PhysicalRoute::MmaTf32RnaWideSplitK8(
                if staged && columns >= 128 && rows.is_multiple_of(4) && columns.is_multiple_of(4) {
                    super::contract::Tf32RnaSplitTile::M64N64S4
                } else {
                    super::contract::Tf32RnaSplitTile::M64N64S4AnyAlignment
                },
            );
        }
        if small_layer || columns < 128 {
            return portable(Tf32PortableTile::M16N32, Tf32PortableStages::S4);
        }
        if rna_pre_tn_admits(request, operands, availability) {
            return if reduction >= 256
                && rna_pre_tn_wide_pays(rows, columns, availability.multiprocessors as usize)
            {
                Tf32PhysicalRoute::Sm89TnPreRnaM64N96S2
            } else {
                Tf32PhysicalRoute::Sm89TnPreRnaM64N64
            };
        }
        return if rows < RNA_NT_NARROW_ROWS {
            portable(Tf32PortableTile::M16N32, Tf32PortableStages::S4)
        } else {
            portable(Tf32PortableTile::M64N64, Tf32PortableStages::S2)
        };
    }
    if small_layer {
        return if rows.div_ceil(64) * columns.div_ceil(64) < SMALL_LAYER_GRID {
            portable(Tf32PortableTile::M16N32, Tf32PortableStages::S4)
        } else {
            portable(Tf32PortableTile::M64N64, Tf32PortableStages::S2)
        };
    }
    if rows < RNA_NT_NARROW_ROWS {
        return if reduction >= RNA_NT_SPLIT_MIN_REDUCTION && columns <= RNA_NT_SPLIT_MAX_COLUMNS {
            Tf32PhysicalRoute::MmaTf32RnaSplitK4(Tf32PortableRoute {
                tile: Tf32PortableTile::M16N32,
                stages: Tf32PortableStages::S3,
            })
        } else {
            portable(Tf32PortableTile::M16N32, Tf32PortableStages::S4)
        };
    }
    let outputs = rows * columns;
    if staged && reduction >= 256 && columns <= 1_024 && outputs >= MS_NT_MIN_OUTPUTS {
        return Tf32PhysicalRoute::MmaTf32RnaWide(
            if reduction < 1_024 || outputs > MS_NT_SQUARE_OUTPUTS {
                Tf32RnaWideTile::NtMsM64N128Bk16S6
            } else if outputs <= MS_NT_NARROW_OUTPUTS {
                Tf32RnaWideTile::NtMsM128N64S3
            } else {
                Tf32RnaWideTile::NtMsM128N128Bk16S5
            },
        );
    }
    // A few thousand rows over an output up to 512 wide keep the 64 x 64
    // tile, three stages over a short reduction.
    if (2_048..8_192).contains(&rows) && columns <= 512 {
        return portable(
            Tf32PortableTile::M64N64,
            if reduction <= 64 {
                Tf32PortableStages::S3
            } else {
                Tf32PortableStages::S2
            },
        );
    }
    // An input gradient whose reduction is no longer than its output is
    // wide fills the twelve-warp 144 x 96 tile (NT 1024x1536x384 30.4 ->
    // 20.1 us on the RTX 5090); a long reduction under a narrow output
    // keeps the 64 x 64 tile.
    if staged
        && reduction <= columns
        && availability
            .joint
            .is_some_and(|joint| joint.device.compute_capability.0 == 12)
    {
        return Tf32PhysicalRoute::Sm89NtRnaM144N96S2;
    }
    portable(Tf32PortableTile::M64N64, Tf32PortableStages::S2)
}

/// Whether the batch-invariant module's tiles can stage a request's operands
/// on this board: the module is bound on an Ada board and both operands are
/// 16-byte aligned with whole-chunk strides.
fn rna_tiles_stage(
    request: F32TriadRequest,
    operands: Option<F32TriadOperands>,
    availability: F32TriadAvailability,
) -> bool {
    operands.is_some_and(|operands| {
        availability.rna.is_some_and(|rna| {
            let cc = rna.device.compute_capability;
            cc == (8, 9) || cc.0 == 12
        }) && request.shape.lda.is_multiple_of(4)
            && request.shape.ldb.is_multiple_of(4)
            && [operands.a, operands.b]
                .into_iter()
                .all(|pointer| pointer != 0 && pointer.is_multiple_of(16))
    })
}

/// Whether a weight gradient may run the pre-rounded tiles: the joint
/// module is bound on an Ada or CC 12.x board, the gradient accumulates
/// with unit scales into 16-byte aligned operands, the saved activations'
/// rows are whole 16-byte chunks, and the transposed copy fits its scratch.
fn rna_pre_tn_admits(
    request: F32TriadRequest,
    operands: Option<F32TriadOperands>,
    availability: F32TriadAvailability,
) -> bool {
    let Some(operands) = operands else {
        return false;
    };
    let shape = request.shape;
    let copy_stride = shape.m.next_multiple_of(4);
    // The joint module composes on every SM80-tier board; its pre-rounding
    // transpose and tiles are gated for the Ada and CC 12.x JITs.
    availability.joint.is_some_and(|joint| {
        let cc = joint.device.compute_capability;
        cc == (8, 9) || cc.0 == 12
    }) && request.op == ResolvedGemmOp::Tn
        && shape == F32TriadShape::contiguous(ResolvedGemmOp::Tn, (shape.m, shape.k, shape.n))
        && shape.m > 0
        && shape.n.is_multiple_of(4)
        && shape
            .k
            .checked_mul(copy_stride)
            .is_some_and(|floats| floats <= super::contract::SCALAR_TRANSPOSE_SCRATCH_CAP_ELEMENTS)
        && operands.alpha.to_bits() == 1.0_f32.to_bits()
        && operands.beta.to_bits() == 1.0_f32.to_bits()
        && operands.bias.is_none()
        && [operands.output, operands.a, operands.b]
            .into_iter()
            .all(|pointer| pointer != 0 && pointer.is_multiple_of(16))
}

/// The batch-invariant TF32 forward's table on an Ada board (CC 8.9), the
/// architecture it was measured on: the narrow tile below 256 rows, the
/// multistage tiles for a few thousand rows of a layer up to 512 wide, the
/// wide tiles as the rows and columns grow, the 64 x 64 tile where the wide
/// tiles cannot stage the operands. `None` on any other architecture, whose
/// forward keeps the portable tiles until its own table is measured.
fn rna_forward_table_route(
    request: F32TriadRequest,
    operands: Option<F32TriadOperands>,
    availability: F32TriadAvailability,
) -> Option<Tf32PhysicalRoute> {
    use super::contract::{
        Tf32PortableRoute, Tf32PortableStages, Tf32PortableTile, Tf32RnaWideTile,
    };
    let compute_capability = availability.portable?.device.compute_capability;
    if request.op != ResolvedGemmOp::Nn
        || (compute_capability != (8, 9) && compute_capability.0 != 12)
    {
        return None;
    }
    let (m, n) = (request.shape.m, request.shape.n);
    let portable = |tile, stages| Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute { tile, stages });
    let wide = |tile| Tf32PhysicalRoute::MmaTf32RnaWide(tile);
    // The wide and multistage tiles copy whole 16-byte chunks.
    let staged = availability.rna.is_some()
        && request.shape.lda.is_multiple_of(4)
        && request.shape.ldb.is_multiple_of(4)
        && operands.is_some_and(|operands| {
            [operands.a, operands.b]
                .into_iter()
                .all(|pointer| pointer != 0 && pointer.is_multiple_of(16))
        });
    const NARROW_ROWS: usize = 256;
    const NARROW_WIDE_ROWS: usize = 2_048;
    const TALL_ROWS: usize = 8_192;
    const TALLER_ROWS: usize = 16_384;
    if compute_capability.0 == 12 {
        return Some(rna_blackwell_forward_route(
            request,
            staged,
            availability.multiprocessors as usize,
        ));
    }
    Some(if m < NARROW_ROWS {
        portable(Tf32PortableTile::M16N32, Tf32PortableStages::S4)
    } else if staged && (512..TALL_ROWS).contains(&m) && (128..=512).contains(&n) {
        if n > 128 && n <= 384 {
            wide(Tf32RnaWideTile::MsM128N64S3)
        } else {
            wide(Tf32RnaWideTile::MsM64N128Bk16S6)
        }
    } else if n < 128 {
        if staged && m >= NARROW_WIDE_ROWS {
            wide(Tf32RnaWideTile::M96N96S3)
        } else {
            portable(Tf32PortableTile::M16N32, Tf32PortableStages::S4)
        }
    } else if staged {
        if n >= 1_536 && m >= NARROW_WIDE_ROWS {
            wide(Tf32RnaWideTile::M128N128S3)
        } else if n >= 1_536 {
            wide(Tf32RnaWideTile::M128N96S3)
        } else if (n >= 512 && m >= 4_096)
            || (n <= 128 && m >= TALLER_ROWS)
            || (n > 128 && m >= TALL_ROWS)
        {
            wide(Tf32RnaWideTile::M128N128S3)
        } else if n <= 128 && m >= TALL_ROWS {
            wide(Tf32RnaWideTile::M128N96S3)
        } else {
            wide(Tf32RnaWideTile::M96N96S3)
        }
    } else if n <= 128 {
        portable(Tf32PortableTile::M64N64, Tf32PortableStages::S3)
    } else {
        portable(Tf32PortableTile::M64N64, Tf32PortableStages::S2)
    })
}

/// The batch-invariant TF32 forward's table on a CC 12.x board, as measured
/// on an RTX 5090 (interleaved sweeps,
/// device minimum per launch): the narrow tiles for a few columns, the
/// 64 x 64 tile while a narrow forward's tiles fill no more than one wave of
/// the card's own SMs, the multistage tiles over a long reduction of 129 to
/// 256 columns, and the wide tiles as the rows and columns grow (NN
/// 4096x128x796 24.2 -> 20.1 us, 4096x128x16 7.8 -> 3.3, 896x256x256 11.9 ->
/// 7.8). Every tile walks an output's reduction in one chain, so the table
/// moves time and never a bit.
///
/// Staged operands then take the multistage tiles where they pay: the
/// multistage 64 x 64 tile in place of the portable one (0.54 to 0.87 of its
/// time on all 76 such shapes of two RTX 5090 sweeps) and in place of a wide
/// one while its grid fits one wave of the live multiprocessors, and else
/// the multistage 128 x 64 tile where its busiest multiprocessor does at
/// most four fifths of the wide tile's work, counted as whole-card rounds of
/// one CTA each times the tile's outputs. That ran 168 forward shapes in
/// 0.95 of the table's time on an RTX 5090 and 0.97 on an RTX PRO 6000, none
/// slower on either. The 64 x 64 tile over several waves, by the same work
/// count, slowed up to 21 shapes on the RTX PRO 6000.
fn rna_blackwell_forward_route(
    request: F32TriadRequest,
    staged: bool,
    multiprocessors: usize,
) -> Tf32PhysicalRoute {
    use super::contract::{Tf32PortableRoute, Tf32PortableTile, Tf32RnaWideTile};
    let table = rna_blackwell_forward_table(request, staged, multiprocessors);
    if !staged || multiprocessors == 0 {
        return table;
    }
    if rna_persistent_forward(request) {
        return Tf32PhysicalRoute::MmaTf32RnaWide(Tf32RnaWideTile::MsPersistM128N64Bk16S3);
    }
    let tabled = match table {
        Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
            tile: Tf32PortableTile::M64N64,
            ..
        }) => return Tf32PhysicalRoute::MmaTf32RnaWide(Tf32RnaWideTile::MsM64N64S3),
        Tf32PhysicalRoute::MmaTf32RnaWide(tile) => tile,
        other => return other,
    };
    let (m, n) = (request.shape.m, request.shape.n);
    let work = |tile| {
        let (bm, bn) = rna_wide_tile_outputs(tile);
        (m.div_ceil(bm) * n.div_ceil(bn)).div_ceil(multiprocessors) * bm * bn
    };
    let one_wave = m.div_ceil(64) * n.div_ceil(64) <= multiprocessors;
    let wide = Tf32RnaWideTile::MsM128N64S3;
    Tf32PhysicalRoute::MmaTf32RnaWide(if one_wave {
        Tf32RnaWideTile::MsM64N64S3
    } else if work(wide) * 5 <= work(tabled) * 4 {
        wide
    } else {
        tabled
    })
}

/// Whether a CC 12.x forward takes the persistent 128 x 64 tile: a wide
/// output of 2304 to 4096 rows and at least 1536 columns over a reduction of
/// 129 to 256, every tile and every 32-deep group of it whole. Its resident
/// grid pays no pipeline fill per tile, which a short reduction feels most.
/// On an RTX 5090 it ran every shape of a sweep over those rows (every 128
/// rows, 1536 to 2112 columns) as fast as the best other tile or up to 13%
/// faster (NN 4096x256x2112 63.6 -> 57.2 us); below 2304 rows another tile
/// wins at some row counts. Each output is one CTA's chain, as on every
/// other tile, so the pick moves no bit.
fn rna_persistent_forward(request: F32TriadRequest) -> bool {
    let (m, k, n) = (request.shape.m, request.shape.k, request.shape.n);
    request.op == ResolvedGemmOp::Nn
        && (2_304..=4_096).contains(&m)
        && n >= 1_536
        && (129..=256).contains(&k)
        && m.is_multiple_of(128)
        && n.is_multiple_of(64)
        && k.is_multiple_of(32)
}

/// The CC 12.x forward table [`rna_blackwell_forward_route`] starts from.
fn rna_blackwell_forward_table(
    request: F32TriadRequest,
    staged: bool,
    multiprocessors: usize,
) -> Tf32PhysicalRoute {
    use super::contract::{
        Tf32PortableRoute, Tf32PortableStages, Tf32PortableTile, Tf32RnaWideTile,
    };
    let (m, k, n) = (request.shape.m, request.shape.k, request.shape.n);
    let portable = |tile, stages| Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute { tile, stages });
    let wide = |tile| Tf32PhysicalRoute::MmaTf32RnaWide(tile);
    const NARROW_ROWS: usize = 256;
    const NARROW_WIDE_ROWS: usize = 2_048;
    const TALL_ROWS: usize = 8_192;
    const TALLER_ROWS: usize = 16_384;
    const SMALL_LAYER_WIDTH: usize = 512;
    const SMALL_LAYER_GRID: usize = 64;
    if m < NARROW_ROWS {
        return portable(Tf32PortableTile::M16N32, Tf32PortableStages::S4);
    }
    if staged && (129..=256).contains(&n) && k >= 512 {
        return wide(if m.div_ceil(64) * n.div_ceil(64) <= multiprocessors {
            Tf32RnaWideTile::MsM64N64S3
        } else {
            Tf32RnaWideTile::MsM128N64S3
        });
    }
    if (TALLER_ROWS..2 * TALLER_ROWS).contains(&m) && (129..512).contains(&n) && staged {
        return wide(Tf32RnaWideTile::M128N128S3);
    }
    if (NARROW_WIDE_ROWS..TALL_ROWS).contains(&m) {
        if n <= 16 {
            return portable(Tf32PortableTile::M16N16, Tf32PortableStages::S4);
        }
        if n <= 48 {
            return portable(Tf32PortableTile::M16N32, Tf32PortableStages::S4);
        }
        if n <= 128 {
            return portable(
                Tf32PortableTile::M64N64,
                if k.is_multiple_of(4) {
                    Tf32PortableStages::S3
                } else {
                    Tf32PortableStages::S2
                },
            );
        }
        if staged && (512..=640).contains(&n) && m >= 4_096 {
            return wide(Tf32RnaWideTile::M128N128S3);
        }
        if staged && n >= 1_536 {
            return wide(Tf32RnaWideTile::M96N96S3);
        }
        if staged && n >= 512 {
            return wide(Tf32RnaWideTile::M128N96S3);
        }
    }
    if (NARROW_ROWS..NARROW_WIDE_ROWS).contains(&m)
        && k <= SMALL_LAYER_WIDTH
        && n <= SMALL_LAYER_WIDTH
    {
        return if m.div_ceil(64) * n.div_ceil(64) < SMALL_LAYER_GRID {
            portable(Tf32PortableTile::M16N32, Tf32PortableStages::S4)
        } else {
            portable(Tf32PortableTile::M64N64, Tf32PortableStages::S2)
        };
    }
    if n < 128 {
        return if staged && m >= NARROW_WIDE_ROWS {
            wide(Tf32RnaWideTile::M96N96S3)
        } else {
            portable(Tf32PortableTile::M16N32, Tf32PortableStages::S4)
        };
    }
    if !staged {
        return portable(
            Tf32PortableTile::M64N64,
            if n <= 128 {
                Tf32PortableStages::S3
            } else {
                Tf32PortableStages::S2
            },
        );
    }
    if n >= 1_536 {
        wide(Tf32RnaWideTile::M128N96S3)
    } else if (n >= 512 && m >= 4_096) || (n <= 128 && m >= TALLER_ROWS) {
        wide(Tf32RnaWideTile::M128N128S3)
    } else if n <= 128 && m >= TALL_ROWS {
        wide(Tf32RnaWideTile::M128N96S3)
    } else {
        wide(Tf32RnaWideTile::M96N96S3)
    }
}

/// The portable tile that serves a shape whose wide tile this board does
/// not bind: the one the table gives where the wide tiles cannot stage.
fn rna_wide_portable_twin(request: F32TriadRequest, route: Tf32PhysicalRoute) -> Tf32PhysicalRoute {
    use super::contract::{Tf32PortableRoute, Tf32PortableStages, Tf32PortableTile};
    match route {
        Tf32PhysicalRoute::MmaTf32RnaWide(_) if request.shape.n < 128 => {
            Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                tile: Tf32PortableTile::M16N32,
                stages: Tf32PortableStages::S4,
            })
        }
        Tf32PhysicalRoute::MmaTf32RnaWide(_) => Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
            tile: Tf32PortableTile::M64N64,
            stages: if request.shape.n <= 128 {
                Tf32PortableStages::S3
            } else {
                Tf32PortableStages::S2
            },
        }),
        other => other,
    }
}

/// Whether a TF32 route keeps the strict policy's contract. Both operands
/// round with cvt.rna: in registers, after a tensor-map load, or in an
/// explicit pre-rounding pass; the half-ulp shortcut and the tensor map's
/// own tfloat32 conversion round a subnormal differently and are refused.
/// A forward (NN) also takes only the portable tier's plain tiles and the
/// strict policy's own wide and multistage tiles: every one of them seeds an
/// output with its bias and walks the reduction in 32-wide steps of eight
/// from zero to the same zero-filled end, so the tile a row count selects
/// never moves an output's bits.
fn rna_batch_invariant_route(op: ResolvedGemmOp, route: Tf32PhysicalRoute) -> bool {
    use crate::gpu::kernel_identity::ResolvedOperandConversion as Conversion;
    let rounds_rna = match route {
        Tf32PhysicalRoute::MmaTf32RnaSplitK2(_)
        | Tf32PhysicalRoute::MmaTf32RnaSplitK4(_)
        | Tf32PhysicalRoute::MmaTf32RnaSplitK8(_)
        | Tf32PhysicalRoute::MmaTf32RnaWideSplitK8(_) => {
            super::contract::tf32_splitk_spec(op, route).is_ok()
        }
        _ => tf32_kernel_spec(op, route).is_ok_and(|spec| {
            matches!(
                spec.operand_conversion,
                Conversion::RegisterCvtRnaTf32F32
                    | Conversion::TensorMapUint32ThenCvtRnaTf32F32
                    | Conversion::PreRnaAThenRegisterCvtRnaBV1
            )
        }),
    };
    rounds_rna
        && (op != ResolvedGemmOp::Nn
            || matches!(
                route,
                Tf32PhysicalRoute::MmaTf32Rna(_) | Tf32PhysicalRoute::MmaTf32RnaWide(_)
            ))
}

/// The portable plain tile of the strict policy for `request`, bound on this
/// stack: the route that serves a shape whose measured route this toolkit
/// cannot run.
pub(super) fn rna_batch_invariant_ladder(
    request: F32TriadRequest,
    availability: F32TriadAvailability,
) -> Result<Tf32PhysicalRoute, String> {
    resolve_tf32_forced(request, availability, rna_portable_ladder_route(request))
}

/// The portable plain tile the strict policy falls to for a shape no table
/// names: the narrow tiles for a few output rows, the wider ones as rows
/// grow. The ladder reads the output the tile covers, not the request's
/// operand axes: a weight gradient's output is its input width by its output
/// width, whatever the batch. Every one of the tiles keeps the contract.
fn rna_portable_ladder_route(request: F32TriadRequest) -> Tf32PhysicalRoute {
    use super::contract::{Tf32PortableRoute, Tf32PortableStages, Tf32PortableTile};
    let rows = request.shape.output_rows(request.op);
    let columns = request.shape.output_columns(request.op);
    let (tile, stages) = match (rows, columns) {
        (0..=16, 0..=16) => (Tf32PortableTile::M16N16, Tf32PortableStages::S4),
        (0..=16, _) => (Tf32PortableTile::M16N32, Tf32PortableStages::S4),
        (17..=64, _) => (Tf32PortableTile::M64N64, Tf32PortableStages::S3),
        _ => (Tf32PortableTile::M128N64, Tf32PortableStages::S3),
    };
    Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute { tile, stages })
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
        | Tf32PhysicalRoute::MmaTf32RnaSplitK8(_)
        | Tf32PhysicalRoute::MmaTf32RnaWideSplitK8(_) => {
            let spec = super::contract::tf32_splitk_spec(request.op, route)?;
            // A neighbour's split-K route can outgrow the fixed partial
            // and counter workspaces on a larger shape: it does not bind,
            // and the exact family serves.
            super::launch::tf32_splitk_launch_plan(request, spec)?;
            // The portable module carries its extension members only where
            // it composes the extension fragment (not on CC 12.x).
            let extension = super::contract::TF32_SPLITK_EXTENSION_SPECS
                .iter()
                .any(|member| member.route == route && member.op == request.op);
            if extension
                && availability.portable.is_none_or(|portable| {
                    !super::contract::portable_extensions_composed_for_cc(
                        portable.device.compute_capability,
                    )
                })
            {
                return Err(format!(
                    "TF32 split-K extension member {route:?} is not composed on this board"
                ));
            }
            (route.module_kind(), spec.dynamic_shared_bytes)
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
        Tf32PhysicalRoute::MmaTf32RnaWide(_) | Tf32PhysicalRoute::MmaTf32RnaWideSplitK8(_) => {
            availability.rna
        }
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
        // The two pre-rounded members mask every edge of the output and of
        // the reduction, so they run any contiguous weight gradient whose
        // saved activations' rows are whole 16-byte chunks.
        Tf32PhysicalRoute::Sm89TnPreRnaM64N64 | Tf32PhysicalRoute::Sm89TnPreRnaM64N96S2 => {
            request.op == ResolvedGemmOp::Tn && contiguous && shape.n.is_multiple_of(4)
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
        // The twelve-warp input-gradient tile masks every edge of its
        // output and reduction, so it runs any contiguous input gradient.
        Tf32PhysicalRoute::Sm89NtRnaM144N96S2 => request.op == ResolvedGemmOp::Nt && contiguous,
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
            | Tf32PhysicalRoute::MmaTf32RnaWide(_)
            | Tf32PhysicalRoute::MmaTf32RnaWideSplitK8(_)
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
        | Tf32PhysicalRoute::Sm89TnPreRnaM96N96S3
        | Tf32PhysicalRoute::MmaTf32RnaWide(_)
        | Tf32PhysicalRoute::MmaTf32RnaWideSplitK8(_) => {
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
/// the performance-matrix convention with contiguous strides, and the tile
/// and B fragment of the arm that won its official 101-window
/// qualification. How the reduction is cut is not the cell's to say: the
/// portable plan of the shape names the pieces on every architecture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Sm120FmaMeasuredCell {
    op: ResolvedGemmOp,
    shape: F32TriadShape,
    tile: Sm120FmaTile,
    kvec: bool,
}

const fn sm120_fma_cell(
    op: ResolvedGemmOp,
    dims: (usize, usize, usize),
    tile: Sm120FmaTile,
    kvec: bool,
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
        tile,
        kvec,
    }
}

/// The measured cells, each carrying the tile that won its qualification:
/// the nine research projections against the scalar production route
/// (1.22x to 1.60x, internal/perf/sm120-exact-nn-wide-20260903), and the
/// classifier serve and training-batch projections against what the exact
/// policy resolved before them (1.21x to 2.07x,
/// internal/perf/sm120-exact-wide-product-20260903). Those arms also cut
/// some reductions into two to five pieces of their own; the cut now follows
/// the portable plan, so an RTX 5090 writes the bits an RTX 6000 Ada writes.
const SM120_FMA_MEASURED_CELLS: [Sm120FmaMeasuredCell; 18] = [
    sm120_fma_cell(
        ResolvedGemmOp::Nn,
        (2_048, 768, 3_072),
        Sm120FmaTile::M128N64,
        false,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Nn,
        (2_048, 1_536, 768),
        Sm120FmaTile::M128N64,
        false,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Nn,
        (4_621, 384, 1_928),
        Sm120FmaTile::M64N128,
        false,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Tn,
        (2_048, 768, 3_072),
        Sm120FmaTile::M64N128,
        false,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Tn,
        (2_048, 1_536, 768),
        Sm120FmaTile::M128N64,
        false,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Tn,
        (4_621, 384, 1_928),
        Sm120FmaTile::M128N64,
        false,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Nt,
        (2_048, 768, 3_072),
        Sm120FmaTile::M128N64,
        true,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Nt,
        (2_048, 1_536, 768),
        Sm120FmaTile::M64N128,
        false,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Nt,
        (4_621, 384, 1_928),
        Sm120FmaTile::M128N64,
        true,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Nn,
        (4_621, 768, 384),
        Sm120FmaTile::M64N64,
        false,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Nn,
        (4_621, 1_024, 384),
        Sm120FmaTile::M64N64,
        false,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Nn,
        (10_400, 384, 384),
        Sm120FmaTile::M128N64,
        false,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Nn,
        (10_400, 768, 384),
        Sm120FmaTile::M128N64,
        false,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Tn,
        (10_400, 384, 1_536),
        Sm120FmaTile::M64N128,
        false,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Tn,
        (4_096, 3_072, 1_536),
        Sm120FmaTile::M64N128,
        false,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Nt,
        (10_400, 384, 1_536),
        Sm120FmaTile::M128N64,
        false,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Nt,
        (10_400, 384, 384),
        Sm120FmaTile::M64N64,
        false,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Nt,
        (4_621, 768, 384),
        Sm120FmaTile::M64N64,
        false,
    ),
];

/// Shapes past the measured cells take the exact family when their output
/// holds whole tiles enough for the card, an edge of at least a tile and a
/// reduction long enough for the pipeline to matter. The route writes the
/// portable plan's bits whichever tile serves, so the live multiprocessor
/// count decides here. One CTA per tile, a chain or a CTA folding its own
/// pieces, needs half a tile per multiprocessor: on an RTX 5090 that moved
/// 26 of 337 chains to the family, every one faster (0.79 to 0.97 of its
/// time), where one tile per multiprocessor had left them on the scalar
/// route. A cut dealt a CTA per piece keeps a whole tile per
/// multiprocessor: counting its pieces as well ran small weight gradients
/// up to nine times slower, since each unit then holds only a couple of k
/// tiles.
const SM120_FMA_GENERIC_MIN_EDGE: usize = 128;
const SM120_FMA_GENERIC_MIN_REDUCTION: usize = 256;

fn sm120_fma_generic_tile(
    request: F32TriadRequest,
    multiprocessors: u32,
    splits: u8,
) -> Option<(Sm120FmaTile, bool)> {
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
    let per_multiprocessor = if splits == 1 { 2 } else { 1 };
    (multiprocessors > 0 && tiles * per_multiprocessor >= multiprocessors as usize)
        .then_some((tile, kvec))
}

/// The longest piece, in k tiles, that one CTA folds with the others of its
/// tile rather than dealing a CTA to each: the thin plans' 32 inputs and the
/// slim plans' 64.
const SM120_FMA_SHORT_PIECE_TILES: usize = 4;

/// How the portable exact plan of a request cuts its reduction, in the
/// SM120 kernels' 16-deep k tiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Sm120FmaCut {
    /// The pieces the reduction is cut into.
    pub splits: u8,
    /// The k tiles of every piece; the last one may hold fewer.
    pub piece_tiles: usize,
    /// The reduction goes on past the pieces, and the portable reducer
    /// continues the folded sum over it: only a CTA that folds its own
    /// pieces writes those bits.
    pub tail: bool,
    /// The first output column of the plan's column tail: from it on the
    /// portable route writes each output as one chain over the whole
    /// reduction in a kernel of its own, so only a CTA that folds its own
    /// pieces, keeping those columns out of the fold, writes those bits.
    pub chain_from: Option<usize>,
}

/// The portable plan's cut of a request when the SM120 kernels can write
/// its bits: one chain for a plain big or slim tile, the batch chunks of a
/// split-M weight gradient, the 64-input chunks of a slim split-K product,
/// and the 32-input chunks of a thin split-K product, with the tail chain
/// its reducer runs after the sum or the column tail whose chains a kernel
/// of their own writes. Every other plan folds in a way the SM120 kernels
/// do not (a warp-parallel row), and the request keeps the portable route.
pub(super) fn sm120_fma_portable_cut(request: F32TriadRequest) -> Option<Sm120FmaCut> {
    let k_tiles = request
        .shape
        .reduction(request.op)
        .div_ceil(SM120_FMA_BK as usize);
    // The thin split-K partials sum 32 inputs a piece.
    let thin = 32 / SM120_FMA_BK as usize;
    let mut chain_from = None;
    let (splits, piece_tiles, tail) = match scalar_dispatch_plan(request).ok()? {
        ScalarDispatchPlan::NnFinal { .. }
        | ScalarDispatchPlan::TnFinal { .. }
        | ScalarDispatchPlan::NtFinal { .. } => (1, k_tiles, false),
        ScalarDispatchPlan::TnSplitM { m_chunk, chunks } => {
            if !m_chunk.is_multiple_of(SM120_FMA_BK as usize) {
                return None;
            }
            (chunks, m_chunk / SM120_FMA_BK as usize, false)
        }
        ScalarDispatchPlan::NnSplitKSlim { chunks }
        | ScalarDispatchPlan::NtSplitKSlim { chunks } => (
            usize::try_from(chunks).ok()?,
            SLIM_SPLIT_K_CHUNK as usize / SM120_FMA_BK as usize,
            false,
        ),
        ScalarDispatchPlan::NnSplitKThin => (k_tiles / thin, thin, false),
        ScalarDispatchPlan::NnSplitKThinTail { k_main, k_tail } => (k_main / 32, thin, k_tail > 0),
        ScalarDispatchPlan::NtSplitKMain { n_main, n_tail } => (n_main / 32, thin, n_tail > 0),
        // The main columns fold 32-input pieces of the whole reduction; the
        // columns past them are the tail kernel's chains.
        ScalarDispatchPlan::NtSplitKTail { k_main, .. } => {
            chain_from = Some(k_main);
            (k_tiles / thin, thin, false)
        }
        _ => return None,
    };
    let covered = piece_tiles.checked_mul(splits)?;
    let whole = if tail {
        // The pieces end on a whole tile and the tail holds the rest.
        covered < k_tiles
    } else {
        // Every piece but the last is whole and the last holds a tile.
        covered >= k_tiles && piece_tiles.checked_mul(splits.checked_sub(1)?)? < k_tiles
    };
    if piece_tiles == 0 || splits < 1 || !whole {
        return None;
    }
    Some(Sm120FmaCut {
        splits: u8::try_from(splits).ok()?,
        piece_tiles,
        tail,
        chain_from,
    })
}

fn f32_pointer_is_tma_aligned(pointer: super::contract::CUptr) -> bool {
    pointer != 0 && pointer.is_multiple_of(16)
}

/// The exact-F32 SM120 route for a request under the exact policy: the
/// specialized module must be bound on an SM120 device, every operand must
/// sit on a 16-byte boundary with float4 leading dimensions (the tensor
/// maps demand it), the portable plan's cut must be one the kernels write,
/// and the shape must be a measured cell or fill the card with whole tiles.
/// The route deals the portable plan's pieces and folds them as the
/// portable reducers do, so it writes the bits of every other architecture.
/// Anything else keeps the scalar routes.
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
    let cut = sm120_fma_portable_cut(request)?;
    // The portable split refuses a nonzero beta on NN, and its slim tile
    // pads a ragged reduction to 32 inputs where these kernels pad to 16:
    // a zero product turns a negative-zero sum seeded by a bias positive.
    if cut.splits > 1 && request.op == ResolvedGemmOp::Nn && operands.beta != 0.0 {
        return None;
    }
    // The portable column tail overwrites the output with the bare chain
    // and its reducer never reads the old output, so only an unscaled,
    // overwriting product has those bits here.
    if cut.chain_from.is_some() && (operands.alpha != 1.0 || operands.beta != 0.0) {
        return None;
    }
    if request.op == ResolvedGemmOp::Nn
        && operands.bias.is_some()
        && matches!(
            scalar_dispatch_plan(request),
            Ok(ScalarDispatchPlan::NnFinal { slim: true })
        )
        && shape.reduction(request.op).div_ceil(32) * 32
            != shape.reduction(request.op).div_ceil(16) * 16
    {
        return None;
    }
    let multiprocessors = qualified.device.multiprocessor_count;
    // Only a CTA that folds its own pieces can continue the sum over a tail
    // or keep a column tail's chains out of the fold, and it serves pieces
    // of a few k tiles best: on an RTX 5090 the thin and slim cuts it took
    // from the portable route ran in 0.46 to 0.89 of their time on 14 of 16
    // shapes (1.03 and 1.11 on two of 96 and 100 tiles), where a CTA per
    // piece managed 0.78 to 0.97 and lost up to 1.24 on short reductions.
    // Long pieces, the split-M weight gradients', run a CTA each: one CTA
    // walking them in a row was 1.27 to 1.57 times slower.
    let in_cta = cut.splits > 1
        && (cut.tail || cut.chain_from.is_some() || cut.piece_tiles <= SM120_FMA_SHORT_PIECE_TILES);
    let with_cut = |(tile, kvec): (Sm120FmaTile, bool)| Sm120FmaRoute {
        tile,
        kvec,
        splits: cut.splits,
        in_cta,
    };
    let route = SM120_FMA_MEASURED_CELLS
        .iter()
        .find(|cell| cell.op == request.op && cell.shape == shape)
        .map(|cell| with_cut((cell.tile, cell.kvec)))
        .filter(|route| {
            !qualified
                .sm120_fma_exclusions
                .is_excluded(request.op, *route)
        })
        .or_else(|| {
            // A CTA that folds its own pieces fills the card like one chain.
            let dealt = if in_cta || cut.splits == 1 {
                1
            } else {
                cut.splits
            };
            sm120_fma_generic_tile(request, multiprocessors, dealt)
                .map(with_cut)
                .filter(|route| {
                    !qualified
                        .sm120_fma_exclusions
                        .is_excluded(request.op, *route)
                })
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
    let measured = cells.iter().copied().find(|cell| {
        cell.op == request.op && cell.dtype == request.dtype && cell.shape == request.shape
    });
    if measured.is_none() && sm90a_tn_too_small_for_the_wave(request) {
        return None;
    }
    let route = measured.unwrap_or_else(|| {
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

/// Fewer output-tile reduction steps than this leave a weight gradient's
/// wgmma launch too little work to fill the board: each 64 by 128 output tile
/// walks its whole reduction in one CTA, so a small gradient runs a handful
/// of CTAs in series where the portable tiles split the work. On an H100 PCIe
/// every TN cell up to 256 such steps was faster on the portable tiles and
/// every cell from about 2300 faster on wgmma.
pub const SM90A_TN_MIN_TILE_STEPS: usize = 1024;

/// A weight gradient (TN) too small for the wave rule. Its bits depend on
/// the token count anyway, so choosing by shape keeps every promise the
/// forward makes; the forward and the input gradient always take the wave.
fn sm90a_tn_too_small_for_the_wave(request: Sm90aAutoRequest) -> bool {
    if request.op != Sm90aOp::Tn {
        return false;
    }
    let shape = request.shape;
    let tiles = shape.k.div_ceil(SM90A_TILE.0 as usize) * shape.n.div_ceil(SM90A_TILE.1 as usize);
    let steps = shape.m.div_ceil(SM90A_TILE.2 as usize);
    tiles.saturating_mul(steps) < SM90A_TN_MIN_TILE_STEPS
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
/// underfilled its multiprocessors. Each names the persistent-grid twin of
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
pub(in crate::gpu) struct Sm120AutoRequest {
    pub op: Sm120Op,
    pub dtype: WeightDtype,
    pub shape: Sm120Shape,
    pub a_ptr: super::contract::CUptr,
    pub b_ptr: super::contract::CUptr,
    pub operands: Sm120LaunchOperands,
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
    if !crate::gpu::device::is_sm120_family(caps.compute_capability) {
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
///
/// Both bounds of the rule are tile counts of the shape, never the card's:
/// the stream-K decline picks a reduction order and the tile swap a tile
/// whose bits no test has pinned against the neighbour's, so every SM120
/// card must take the same branch. The count is the largest grid a measured
/// stream-K cell won on (384 x 1928 in 64 x 64 tiles: 186, at 0.93 of the
/// best tiled route), read from the table, so a census that adds a winner
/// moves it; past it the census found the same body losing 1.22-1.77x.
const SM120_STREAMK_NEIGHBOUR_MAX_TILES: f64 =
    sm120_streamk_largest_winning_grid(SM120_STREAMK_CELLS_CC120) as f64;

const fn sm120_streamk_largest_winning_grid(cells: &[Sm120ForcedRoute]) -> usize {
    let mut largest = 0;
    let mut index = 0;
    while index < cells.len() {
        let cell = cells[index];
        if matches!(cell.physical.schedule, Sm120Schedule::StreamK) {
            let (rows, columns) = match cell.op {
                Sm120Op::Nn => (cell.shape.m, cell.shape.n),
                Sm120Op::Tn => (cell.shape.k, cell.shape.n),
                Sm120Op::Nt => (cell.shape.m, cell.shape.k),
            };
            let grid = rows.div_ceil(cell.physical.tile.output_rows() as usize)
                * columns.div_ceil(cell.physical.tile.output_columns() as usize);
            if grid > largest {
                largest = grid;
            }
        }
        index += 1;
    }
    largest
}

fn nearest_sm120_cell(
    cells: &[Sm120ForcedRoute],
    op: Sm120Op,
    dtype: WeightDtype,
    shape: Sm120Shape,
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
    // leaves most of the card idle. When the shape's grid with that tile is
    // under the stream-K bound and under half of what the neighbour filled,
    // the smallest tile serves instead: the census shows it winning every
    // cell that far into underfill. The pipeline depth and reduction
    // step stay the neighbour's. A neighbour that won underfilled itself
    // keeps its tile for shapes that are underfilled the same way.
    let grid = |cell_shape: Sm120Shape| {
        let [rows, columns, _] = sm120_auto_geometry(op, cell_shape);
        (rows / f64::from(physical.tile.output_rows())).ceil()
            * (columns / f64::from(physical.tile.output_columns())).ceil()
    };
    let target_grid = grid(shape);
    if target_grid <= SM120_STREAMK_NEIGHBOUR_MAX_TILES && target_grid * 2.0 < grid(neighbour.shape)
    {
        physical.tile = Sm120Tile::M64N64;
    }
    // A stream-K neighbour earns its persistent grid only up to the largest
    // grid stream-K won on; a larger shape declines here so the tiled table
    // serves it.
    if physical.schedule == Sm120Schedule::StreamK
        && target_grid > SM120_STREAMK_NEIGHBOUR_MAX_TILES
    {
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
            physical: nearest_sm120_cell(cells, request.op, request.dtype, request.shape)?,
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
            crate::gpu::diagnostics::warn_once(&DECLINED, || {
                format!(
                    "SM120 half cell {route:?} is measured for this shape but this board \
                     declines it ({reason}); the portable tensor-core tiles serve instead"
                )
            });
            None
        }
        Err(error) => {
            static INVALID: std::sync::Once = std::sync::Once::new();
            crate::gpu::diagnostics::warn_once(&INVALID, || {
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
) -> Result<Option<(usize, usize)>, String> {
    // No n_in floor: the partial kernel predicates K_out < 128 exactly
    // like the plain kernel, and a small-K dW against a large batch
    // reduction underfills the grid without the split (K_out=24, N=768
    // ran six CTAs). The split changes the dW summation order versus
    // the plain kernel; the partition is a pure function of the shape, so
    // every card of every architecture folds alike.
    if !(n_out >= 128 && batch >= 256) {
        return Ok(None);
    }
    splitm_tn_partition_over_tiles(batch, (n_in, n_out), (128, 128))
}

/// The split-M partition of a narrow weight gradient (2 to 127 outputs),
/// the same rule over the narrow kernel's 64 x 32 tiles. Without it a
/// narrow gradient walks the whole batch on a CTA per tile: 24 CTAs for
/// 384 x 101 over 1024 rows, three times the split's time.
#[inline]
pub(super) fn splitm_tn_narrow_partition(
    batch: usize,
    n_in: usize,
    n_out: usize,
) -> Result<Option<(usize, usize)>, String> {
    if !((2..=127).contains(&n_out) && batch >= 256) {
        return Ok(None);
    }
    splitm_tn_partition_over_tiles(batch, (n_in, n_out), (64, 32))
}

/// The row chunks a weight gradient's batch is cut into so its `tile`
/// grid reaches the policy's target block count, every chunk a whole
/// number of the partial kernel's 16-row steps; `None` where the grid
/// already reaches the target or the planes would not fit the scratch.
fn splitm_tn_partition_over_tiles(
    batch: usize,
    (n_in, n_out): (usize, usize),
    (tile_rows, tile_columns): (u32, u32),
) -> Result<Option<(usize, usize)>, String> {
    let target_blocks = scalar_split_policy().tn_split_m_target_blocks;
    let batch_u32 = u32::try_from(batch).map_err(|_| "TN split-M batch exceeds u32::MAX")?;
    let k_tiles = u32::try_from(n_in)
        .map_err(|_| "TN split-M input width exceeds u32::MAX")?
        .div_ceil(tile_rows);
    let n_tiles = u32::try_from(n_out)
        .map_err(|_| "TN split-M output width exceeds u32::MAX")?
        .div_ceil(tile_columns);
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

/// The facts a tuned scalar cell is admitted on. The f32 policy is not one
/// of them: a tuned exact-FMA route is the floor the TF32 policy falls back
/// to, so it serves under either policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ScalarLaunchFacts {
    pub scalar_artifact: ArtifactIdentity,
    pub scalar_compiler: CompilerIdentity,
    pub inference_artifact: ArtifactIdentity,
    pub inference_compiler: CompilerIdentity,
    pub inference_copyplan_loaded: bool,
    pub sm89_exact_f32_artifact: Option<ArtifactIdentity>,
    pub sm89_exact_f32_compiler: Option<CompilerIdentity>,
    pub sm89_exact_f32_symbols_loaded: [bool; 3],
    pub sm89_exact_f32_d128_artifact: Option<ArtifactIdentity>,
    pub sm89_exact_f32_d128_compiler: Option<CompilerIdentity>,
    pub sm89_exact_f32_d128_symbols_loaded: [bool; 2],
    pub compute_capability: (u32, u32),
    /// The SM120 scalar kernels met their resource caps on this toolkit.
    pub sm120_scalar_resources_qualified: bool,
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

/// The boards whose Inference module composes the Ada overlay and so may prove
/// its copy-plan routes: the sm_80 tier without the CC 12 family, which
/// keeps its Inference module byte-identical to the one its own cohorts pin.
fn inference_overlay_proof_board(facts: ScalarLaunchFacts) -> bool {
    facts.compute_capability.0 >= 8
        && !crate::gpu::device::is_sm120_family(facts.compute_capability)
        && facts.inference_copyplan_loaded
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

fn inference_overlay_admitted(
    admission: ScalarAdmission,
    facts: ScalarLaunchFacts,
    evidence: fn(ScalarLaunchFacts) -> bool,
) -> bool {
    match admission {
        ScalarAdmission::Evidence | ScalarAdmission::Candidate => evidence(facts),
        ScalarAdmission::Proof => inference_overlay_proof_board(facts),
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
    0x93, 0x3a, 0x09, 0x89, 0x8c, 0x5c, 0x41, 0x04, 0x95, 0xd1, 0xab, 0xb6, 0x83, 0xc9, 0xe0, 0xf6,
    0xcb, 0x8b, 0x0f, 0x94, 0x2c, 0xd6, 0x26, 0x1c, 0xa7, 0xaf, 0x9a, 0xca, 0x03, 0x49, 0x38, 0x8b,
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
        && SM89_EXACT_F32_D128_EVIDENCE_COHORTS
            .iter()
            .copied()
            .any(|identity| identity.matches(facts))
}

fn qualified_sm89_exact_f32_d128_candidate_environment(facts: ScalarLaunchFacts) -> bool {
    facts.compute_capability == (8, 9)
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
    42, 227, 116, 0, 192, 38, 217, 202, 183, 32, 74, 114, 1, 61, 242, 232, 153, 15, 166, 209, 101,
    53, 119, 255, 89, 36, 101, 191, 82, 161, 129, 35,
];

const SM89_EXACT_F32_QUALIFICATION_CANDIDATES: &[Sm89ExactF32QualificationIdentity] = &[
    Sm89ExactF32QualificationIdentity {
        nvrtc_version: (12, 8),
        compile_key: [
            22, 133, 2, 113, 79, 140, 174, 92, 64, 55, 72, 26, 191, 32, 144, 16, 129, 71, 18, 169,
            163, 233, 151, 238, 254, 194, 137, 67, 229, 238, 69, 233,
        ],
        artifact_digest: [
            119, 139, 6, 165, 228, 20, 145, 219, 6, 35, 85, 77, 245, 101, 64, 247, 248, 158, 74,
            71, 51, 150, 150, 18, 252, 81, 0, 54, 194, 68, 144, 112,
        ],
        source_digest: SM89_EXACT_F32_SOURCE_DIGEST,
        header_manifest_digest: [
            58, 2, 32, 37, 216, 76, 62, 36, 155, 189, 92, 49, 247, 246, 159, 151, 126, 27, 28, 65,
            56, 204, 94, 105, 113, 193, 67, 69, 3, 189, 58, 15,
        ],
        nvrtc_library_domain: [
            38, 176, 163, 160, 32, 68, 255, 203, 193, 105, 63, 216, 62, 146, 97, 190, 255, 166,
            146, 164, 251, 207, 227, 172, 94, 157, 140, 135, 152, 11, 177, 85,
        ],
    },
    Sm89ExactF32QualificationIdentity {
        nvrtc_version: (13, 0),
        compile_key: [
            19, 69, 61, 20, 203, 144, 94, 87, 123, 178, 25, 92, 62, 171, 168, 54, 114, 169, 222,
            191, 56, 153, 243, 49, 147, 85, 9, 200, 223, 137, 172, 184,
        ],
        artifact_digest: [
            31, 208, 236, 148, 231, 189, 226, 21, 247, 7, 136, 202, 101, 180, 43, 62, 225, 201, 46,
            5, 21, 235, 33, 219, 8, 8, 252, 210, 60, 150, 180, 61,
        ],
        source_digest: SM89_EXACT_F32_SOURCE_DIGEST,
        header_manifest_digest: [
            132, 239, 3, 50, 67, 109, 203, 137, 227, 80, 84, 20, 197, 231, 41, 9, 26, 76, 5, 67,
            51, 57, 230, 106, 95, 57, 79, 61, 144, 75, 194, 62,
        ],
        nvrtc_library_domain: [
            112, 155, 145, 195, 107, 251, 14, 217, 102, 238, 105, 173, 200, 214, 248, 127, 241, 16,
            238, 207, 61, 251, 80, 96, 54, 127, 24, 60, 230, 20, 235, 13,
        ],
    },
    Sm89ExactF32QualificationIdentity {
        nvrtc_version: (13, 2),
        compile_key: [
            83, 124, 13, 149, 236, 157, 74, 190, 239, 221, 245, 103, 76, 179, 93, 193, 227, 154,
            223, 88, 202, 153, 140, 158, 163, 158, 181, 28, 112, 82, 234, 187,
        ],
        artifact_digest: [
            104, 186, 214, 68, 145, 122, 106, 172, 70, 82, 22, 20, 112, 248, 126, 174, 157, 4, 72,
            49, 184, 100, 198, 228, 141, 126, 12, 21, 25, 171, 244, 73,
        ],
        source_digest: SM89_EXACT_F32_SOURCE_DIGEST,
        header_manifest_digest: [
            252, 26, 221, 32, 31, 26, 175, 118, 209, 248, 228, 199, 168, 92, 58, 101, 51, 220, 250,
            147, 173, 57, 210, 137, 130, 77, 100, 73, 16, 27, 233, 223,
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
        && SM89_EXACT_F32_EVIDENCE_COHORTS
            .iter()
            .copied()
            .any(|identity| identity.matches(facts))
}

fn qualified_sm89_exact_f32_candidate_environment(facts: ScalarLaunchFacts) -> bool {
    facts.compute_capability == (8, 9)
        && SM89_EXACT_F32_QUALIFICATION_CANDIDATES
            .iter()
            .copied()
            .any(|identity| identity.matches(facts))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct InferenceCopyPlanQualificationIdentity {
    nvrtc_version: (i32, i32),
    compile_key: [u8; 32],
    artifact_digest: [u8; 32],
    source_digest: [u8; 32],
    header_manifest_digest: [u8; 32],
    nvrtc_library_domain: [u8; 32],
}

impl InferenceCopyPlanQualificationIdentity {
    fn matches(self, facts: ScalarLaunchFacts) -> bool {
        facts.inference_copyplan_loaded
            && facts.inference_artifact.module_kind == ModuleKind::Inference
            && facts.inference_artifact.artifact_kind == ArtifactKind::Ptx
            && facts.inference_artifact.compile_key == self.compile_key
            && facts.inference_artifact.artifact_digest == self.artifact_digest
            && facts.inference_compiler.source_digest == self.source_digest
            && facts.inference_compiler.invocation_digest == self.compile_key
            && facts.inference_compiler.header_manifest_digest == self.header_manifest_digest
            && facts.inference_compiler.target.as_str() == "sm_89"
            && facts.inference_compiler.nvrtc_version == self.nvrtc_version
            && facts.inference_compiler.nvrtc_library_domain == self.nvrtc_library_domain
            && facts.inference_compiler.nvrtc_library_known
            && facts.inference_compiler.output_kind == ArtifactKind::Ptx
            && facts.inference_compiler.composer_revision == COMPOSER_REVISION
            && facts.inference_compiler.compiler_revision == COMPILER_REVISION
            && facts.inference_compiler.numeric_abi_revision == NUMERIC_ABI_REVISION
            && facts.inference_compiler.schedule_revision == SCHEDULE_REVISION
    }
}

const INFERENCE_COPYPLAN_SOURCE_DIGEST: [u8; 32] = [
    59, 122, 134, 236, 194, 32, 175, 1, 63, 86, 29, 5, 233, 68, 204, 187, 136, 30, 68, 199, 130,
    86, 55, 193, 149, 196, 223, 55, 45, 89, 166, 252,
];

/// Exact Inference identities minted on Ada, one per retained toolkit. The
/// Inference module takes no state capacity, so one identity serves every
/// capacity on a toolkit.
const INFERENCE_COPYPLAN_EVIDENCE_COHORTS: &[InferenceCopyPlanQualificationIdentity] = &[
    InferenceCopyPlanQualificationIdentity {
        nvrtc_version: (12, 8),
        compile_key: [
            12, 17, 144, 213, 3, 52, 126, 193, 128, 214, 181, 11, 112, 91, 103, 1, 27, 188, 129,
            95, 220, 101, 126, 171, 138, 57, 34, 21, 191, 179, 232, 47,
        ],
        artifact_digest: [
            240, 139, 195, 112, 188, 209, 32, 240, 25, 76, 19, 240, 235, 44, 165, 50, 23, 22, 10,
            1, 136, 2, 204, 13, 152, 32, 93, 146, 111, 244, 17, 113,
        ],
        source_digest: INFERENCE_COPYPLAN_SOURCE_DIGEST,
        header_manifest_digest: [
            92, 118, 31, 241, 96, 57, 48, 196, 146, 88, 75, 119, 178, 215, 121, 17, 104, 244, 209,
            45, 221, 166, 151, 163, 203, 143, 60, 202, 192, 90, 62, 190,
        ],
        nvrtc_library_domain: [
            38, 176, 163, 160, 32, 68, 255, 203, 193, 105, 63, 216, 62, 146, 97, 190, 255, 166,
            146, 164, 251, 207, 227, 172, 94, 157, 140, 135, 152, 11, 177, 85,
        ],
    },
    InferenceCopyPlanQualificationIdentity {
        nvrtc_version: (13, 0),
        compile_key: [
            204, 237, 201, 109, 135, 166, 200, 57, 110, 24, 124, 36, 197, 35, 93, 168, 23, 107,
            174, 81, 134, 162, 78, 133, 46, 199, 193, 70, 116, 21, 124, 175,
        ],
        artifact_digest: [
            95, 241, 16, 132, 243, 96, 90, 136, 187, 131, 85, 109, 226, 138, 72, 252, 188, 102, 75,
            39, 35, 3, 79, 229, 242, 105, 3, 127, 63, 181, 53, 11,
        ],
        source_digest: INFERENCE_COPYPLAN_SOURCE_DIGEST,
        header_manifest_digest: [
            242, 91, 36, 121, 218, 36, 90, 206, 50, 163, 82, 83, 174, 137, 239, 84, 109, 231, 245,
            70, 241, 112, 117, 64, 240, 139, 172, 110, 250, 65, 222, 98,
        ],
        nvrtc_library_domain: [
            112, 155, 145, 195, 107, 251, 14, 217, 102, 238, 105, 173, 200, 214, 248, 127, 241, 16,
            238, 207, 61, 251, 80, 96, 54, 127, 24, 60, 230, 20, 235, 13,
        ],
    },
    InferenceCopyPlanQualificationIdentity {
        nvrtc_version: (13, 2),
        compile_key: [
            108, 118, 1, 22, 58, 210, 9, 206, 176, 208, 76, 105, 108, 197, 114, 218, 77, 36, 37,
            38, 246, 155, 201, 25, 87, 110, 88, 63, 46, 220, 83, 100,
        ],
        artifact_digest: [
            245, 103, 64, 67, 67, 66, 27, 138, 15, 114, 129, 83, 123, 68, 91, 220, 133, 69, 22,
            228, 110, 35, 25, 9, 30, 56, 38, 172, 93, 185, 219, 3,
        ],
        source_digest: INFERENCE_COPYPLAN_SOURCE_DIGEST,
        header_manifest_digest: [
            26, 213, 41, 179, 21, 21, 5, 1, 5, 111, 152, 20, 128, 204, 76, 5, 243, 58, 15, 169,
            138, 53, 138, 232, 230, 63, 199, 84, 103, 82, 108, 13,
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
struct NtInferenceCopyPlanComposedQualificationIdentity {
    scalar: ScalarTransposeQualificationIdentity,
    inference: InferenceCopyPlanQualificationIdentity,
}

impl NtInferenceCopyPlanComposedQualificationIdentity {
    fn matches(self, facts: ScalarLaunchFacts) -> bool {
        self.scalar.nvrtc_version == self.inference.nvrtc_version
            && self.scalar.matches(facts)
            && self.inference.matches(facts)
    }
}

const TRIAD_SCALAR_TRANSPOSE_SOURCE_DIGEST: [u8; 32] = [
    220, 214, 147, 7, 243, 176, 228, 109, 197, 70, 21, 187, 16, 124, 105, 48, 96, 204, 83, 20, 190,
    249, 126, 33, 243, 255, 94, 87, 215, 226, 156, 12,
];

const TRIAD_SCALAR_TRANSPOSE_IDENTITY_CUDA_12_8: ScalarTransposeQualificationIdentity =
    ScalarTransposeQualificationIdentity {
        nvrtc_version: (12, 8),
        compile_key: [
            44, 237, 65, 107, 0, 46, 55, 172, 23, 100, 138, 33, 89, 78, 78, 50, 161, 182, 182, 50,
            149, 124, 149, 162, 151, 48, 100, 117, 158, 172, 173, 183,
        ],
        artifact_digest: [
            160, 132, 192, 216, 157, 233, 246, 109, 117, 55, 155, 251, 39, 200, 42, 211, 38, 10,
            78, 12, 231, 220, 33, 88, 214, 224, 40, 72, 221, 35, 112, 65,
        ],
        source_digest: TRIAD_SCALAR_TRANSPOSE_SOURCE_DIGEST,
        header_manifest_digest: [
            8, 239, 223, 101, 22, 31, 40, 190, 147, 217, 38, 225, 51, 92, 231, 221, 209, 164, 77,
            237, 194, 110, 238, 126, 31, 146, 53, 26, 32, 2, 152, 97,
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
            62, 102, 156, 227, 107, 246, 43, 111, 141, 176, 105, 151, 8, 11, 186, 149, 82, 250, 65,
            244, 138, 228, 18, 240, 163, 104, 170, 250, 93, 29, 143, 201,
        ],
        artifact_digest: [
            10, 222, 73, 89, 128, 204, 255, 252, 199, 19, 255, 242, 118, 99, 19, 16, 228, 55, 81,
            245, 191, 37, 26, 203, 85, 141, 229, 176, 221, 17, 99, 85,
        ],
        source_digest: TRIAD_SCALAR_TRANSPOSE_SOURCE_DIGEST,
        header_manifest_digest: [
            173, 38, 178, 78, 12, 136, 157, 161, 174, 225, 55, 46, 19, 93, 82, 221, 239, 127, 132,
            50, 112, 85, 80, 142, 106, 253, 126, 53, 130, 162, 253, 199,
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
            38, 57, 236, 131, 75, 138, 94, 8, 154, 110, 95, 105, 29, 86, 35, 173, 158, 242, 135,
            170, 36, 0, 66, 225, 254, 31, 174, 17, 238, 176, 210, 222,
        ],
        artifact_digest: [
            153, 2, 240, 81, 207, 243, 56, 85, 120, 7, 188, 76, 9, 149, 84, 247, 86, 138, 110, 162,
            94, 137, 129, 44, 225, 142, 231, 121, 124, 138, 24, 79,
        ],
        source_digest: TRIAD_SCALAR_TRANSPOSE_SOURCE_DIGEST,
        header_manifest_digest: [
            159, 19, 90, 253, 198, 94, 69, 180, 185, 126, 37, 173, 49, 75, 228, 237, 192, 183, 172,
            105, 143, 219, 164, 112, 211, 153, 178, 238, 153, 238, 90, 122,
        ],
        nvrtc_library_domain: [
            208, 49, 165, 62, 185, 114, 53, 183, 15, 98, 246, 82, 147, 45, 177, 189, 247, 40, 234,
            34, 156, 140, 168, 9, 213, 60, 95, 253, 145, 100, 38, 135,
        ],
    };

/// Frozen whole-pipeline identities for the portable transpose followed by
/// the Inference CopyPlan kernel. Each entry binds both modules to the same live
/// toolkit domain; an Inference-only match is intentionally insufficient.
const NT_INFERENCE_COPYPLAN_COMPOSED_QUALIFICATION_CANDIDATES:
    &[NtInferenceCopyPlanComposedQualificationIdentity] = &[
    NtInferenceCopyPlanComposedQualificationIdentity {
        scalar: TRIAD_SCALAR_TRANSPOSE_IDENTITY_CUDA_12_8,
        inference: INFERENCE_COPYPLAN_EVIDENCE_COHORTS[0],
    },
    NtInferenceCopyPlanComposedQualificationIdentity {
        scalar: TRIAD_SCALAR_TRANSPOSE_IDENTITY_CUDA_13_0,
        inference: INFERENCE_COPYPLAN_EVIDENCE_COHORTS[1],
    },
    NtInferenceCopyPlanComposedQualificationIdentity {
        scalar: TRIAD_SCALAR_TRANSPOSE_IDENTITY_CUDA_13_2,
        inference: INFERENCE_COPYPLAN_EVIDENCE_COHORTS[2],
    },
];

// The three measured complete-module pairs, one per retained toolkit.
// Admission is an alias of that sealed set so no unmeasured pair can be
// appended here.
const NT_INFERENCE_COPYPLAN_COMPOSED_EVIDENCE_COHORTS:
    &[NtInferenceCopyPlanComposedQualificationIdentity] =
    NT_INFERENCE_COPYPLAN_COMPOSED_QUALIFICATION_CANDIDATES;

// These three toolkit projections keep TN independent of the Inference holder.
const TN_M16N16_SPLITM16_SM89_BINDINGS: [ScalarTransposeQualificationIdentity; 3] = [
    NT_INFERENCE_COPYPLAN_COMPOSED_QUALIFICATION_CANDIDATES[0].scalar,
    NT_INFERENCE_COPYPLAN_COMPOSED_QUALIFICATION_CANDIDATES[1].scalar,
    NT_INFERENCE_COPYPLAN_COMPOSED_QUALIFICATION_CANDIDATES[2].scalar,
];

fn qualified_nt_inference_copyplan_sibling_environment(facts: ScalarLaunchFacts) -> bool {
    facts.compute_capability == (8, 9)
        && NT_INFERENCE_COPYPLAN_COMPOSED_EVIDENCE_COHORTS
            .iter()
            .copied()
            .any(|identity| identity.matches(facts))
}

fn qualified_scalar_sm89_tn_environment(facts: ScalarLaunchFacts) -> bool {
    facts.compute_capability == (8, 9)
        && TN_M16N16_SPLITM16_SM89_BINDINGS
            .iter()
            .copied()
            .any(|identity| identity.matches(facts))
}

#[cfg(test)]
pub(super) fn scalar_sm89_composed_test_facts(index: usize) -> ScalarLaunchFacts {
    let identity = NT_INFERENCE_COPYPLAN_COMPOSED_QUALIFICATION_CANDIDATES[index];
    let compiler = |source_digest,
                    invocation_digest,
                    header_manifest_digest,
                    nvrtc_version,
                    nvrtc_library_domain| CompilerIdentity {
        source_digest,
        invocation_digest,
        header_manifest_digest,
        target: crate::gpu::kernel_identity::CudaTarget::new("sm_89").unwrap(),
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
        inference_artifact: ArtifactIdentity {
            module_kind: ModuleKind::Inference,
            artifact_kind: ArtifactKind::Ptx,
            compile_key: identity.inference.compile_key,
            artifact_digest: identity.inference.artifact_digest,
        },
        inference_compiler: compiler(
            identity.inference.source_digest,
            identity.inference.compile_key,
            identity.inference.header_manifest_digest,
            identity.inference.nvrtc_version,
            identity.inference.nvrtc_library_domain,
        ),
        inference_copyplan_loaded: true,
        sm89_exact_f32_artifact: None,
        sm89_exact_f32_compiler: None,
        sm89_exact_f32_symbols_loaded: [false; 3],
        sm89_exact_f32_d128_artifact: None,
        sm89_exact_f32_d128_compiler: None,
        sm89_exact_f32_d128_symbols_loaded: [false; 2],
        compute_capability: (8, 9),
        sm120_scalar_resources_qualified: false,
    }
}

fn qualified_inference_copyplan_environment(facts: ScalarLaunchFacts) -> bool {
    facts.compute_capability == (8, 9)
        && INFERENCE_COPYPLAN_EVIDENCE_COHORTS
            .iter()
            .copied()
            .any(|identity| identity.matches(facts))
}

const NN_INFERENCE_COPYPLAN_SM89_CELLS: [F32TriadShape; 4] = [
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

const NT_D768_OUT_INFERENCE_COPYPLAN_SM89_CELL: F32TriadShape = F32TriadShape {
    m: 2048,
    k: 1536,
    n: 768,
    lda: 768,
    ldb: 768,
    ldc: 1536,
};

const NT_D768_IN_INFERENCE_COPYPLAN_SM89_CELL: F32TriadShape = F32TriadShape {
    m: 2048,
    k: 768,
    n: 3072,
    lda: 3072,
    ldb: 3072,
    ldc: 768,
};

const NT_PRISM_INFERENCE_COPYPLAN_SM89_CELL: F32TriadShape = F32TriadShape {
    m: 4621,
    k: 384,
    n: 1928,
    lda: 1928,
    ldb: 1928,
    ldc: 384,
};

/// The SM120 scalar routes' environment: the CC 12.0 generic target, a
/// well-formed scalar module whose kernels met their resource caps, and the
/// toolkit the routes were timed on. On any other toolkit the routes are
/// offered to the first-use proof, which admits those that reproduce the
/// ordinary plan's words.
fn scalar_sm120_admitted(admission: ScalarAdmission, facts: ScalarLaunchFacts) -> bool {
    let measured = match admission {
        ScalarAdmission::Evidence | ScalarAdmission::Candidate => {
            facts.scalar_compiler.nvrtc_version == (13, 2)
        }
        ScalarAdmission::Proof => true,
    };
    measured
        && facts.sm120_scalar_resources_qualified
        && facts.compute_capability == (12, 0)
        && facts.scalar_artifact.module_kind == ModuleKind::TriadScalar
        && facts.scalar_artifact.artifact_kind == facts.scalar_compiler.output_kind
        && facts.scalar_artifact.compile_key == facts.scalar_compiler.invocation_digest
        && facts.scalar_artifact.artifact_digest != [0; 32]
        && facts.scalar_compiler.invocation_digest != [0; 32]
        && facts.scalar_compiler.target.as_str() == "compute_120"
        && facts.scalar_compiler.nvrtc_library_known
        && facts.scalar_compiler.nvrtc_library_domain != [0; 32]
}

const NN_M64N64_SM120_CELLS: [F32TriadShape; 7] = [
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
    request.op == crate::gpu::kernel_identity::ResolvedGemmOp::Nn
        && NN_M64N64_SM120_CELLS.contains(&request.shape)
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

const NN_M32N64_SPLITK32_SM120_CELL: F32TriadShape = F32TriadShape {
    m: 128,
    k: 8_192,
    n: 128,
    lda: 8_192,
    ldb: 128,
    ldc: 128,
};

fn qualified_nn_m32n64_splitk32_request(request: F32TriadRequest) -> bool {
    request.op == crate::gpu::kernel_identity::ResolvedGemmOp::Nn
        && request.shape == NN_M32N64_SPLITK32_SM120_CELL
}

fn qualified_nn_m32n64_splitk32_operands(operands: F32TriadOperands) -> bool {
    qualified_nn_m64n64_operands(operands)
}

const NT_D768_TRANSPOSE_M64N64_SM120_CELL: F32TriadShape = F32TriadShape {
    m: 2048,
    k: 768,
    n: 3072,
    lda: 3072,
    ldb: 3072,
    ldc: 768,
};

const NT_D768_OUT_TRANSPOSE_M64N64_SM120_CELL: F32TriadShape = F32TriadShape {
    m: 2048,
    k: 1536,
    n: 768,
    lda: 768,
    ldb: 768,
    ldc: 1536,
};

const NT_D768_OUT_TRANSPOSE_ELEMENTS: usize = 1_179_648;

const NT_LARGE_DEEP_TRANSPOSE_M64N64_SM120_CELL: F32TriadShape = F32TriadShape {
    m: 4096,
    k: 3072,
    n: 1536,
    lda: 1536,
    ldb: 1536,
    ldc: 3072,
};

const NT_PRISM_VECTOR_SM120_CELL: F32TriadShape = F32TriadShape {
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

const NT_D128_OUT_TRANSPOSE_M64N64_SM120_CELL: F32TriadShape = F32TriadShape {
    m: 1024,
    k: 256,
    n: 128,
    lda: 128,
    ldb: 128,
    ldc: 256,
};

const NT_D128_OUT_TRANSPOSE_ELEMENTS: usize = 32_768;

const TN_M16N16_SPLITM16_SM120_CELL: F32TriadShape = F32TriadShape {
    m: 256,
    k: 512,
    n: 384,
    lda: 512,
    ldb: 384,
    ldc: 384,
};

fn qualified_nt_m2n16_splitk32_request(request: F32TriadRequest) -> bool {
    request.op == crate::gpu::kernel_identity::ResolvedGemmOp::Nt
        && request.shape
            == F32TriadShape::contiguous(
                crate::gpu::kernel_identity::ResolvedGemmOp::Nt,
                (512, 16, 2_048),
            )
}

fn qualified_tn_m16n16_splitm16_request(request: F32TriadRequest) -> bool {
    request.op == crate::gpu::kernel_identity::ResolvedGemmOp::Tn
        && request.shape == TN_M16N16_SPLITM16_SM120_CELL
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
    request.op == crate::gpu::kernel_identity::ResolvedGemmOp::Nt
        && request.shape == NT_D768_TRANSPOSE_M64N64_SM120_CELL
}

fn qualified_nt_d768_out_transpose_m64n64_request(request: F32TriadRequest) -> bool {
    request.op == crate::gpu::kernel_identity::ResolvedGemmOp::Nt
        && request.shape == NT_D768_OUT_TRANSPOSE_M64N64_SM120_CELL
        && request.shape.k.checked_mul(request.shape.n) == Some(NT_D768_OUT_TRANSPOSE_ELEMENTS)
}

fn qualified_nt_large_deep_transpose_m64n64_request(request: F32TriadRequest) -> bool {
    request.op == crate::gpu::kernel_identity::ResolvedGemmOp::Nt
        && request.shape == NT_LARGE_DEEP_TRANSPOSE_M64N64_SM120_CELL
        && request.shape.k.checked_mul(request.shape.n) == Some(NT_LARGE_DEEP_TRANSPOSE_ELEMENTS)
}

fn qualified_nt_prism_vector_request(request: F32TriadRequest) -> bool {
    request.op == crate::gpu::kernel_identity::ResolvedGemmOp::Nt
        && request.shape == NT_PRISM_VECTOR_SM120_CELL
        && request.shape.k.checked_mul(request.shape.n) == Some(NT_PRISM_TRANSPOSE_ELEMENTS)
}

fn qualified_nt_d128_out_transpose_m64n64_request(request: F32TriadRequest) -> bool {
    request.op == crate::gpu::kernel_identity::ResolvedGemmOp::Nt
        && request.shape == NT_D128_OUT_TRANSPOSE_M64N64_SM120_CELL
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
    let actual_fallback = scalar_dispatch_plan(request)?;
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
        || request.op != crate::gpu::kernel_identity::ResolvedGemmOp::Tn
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
    let actual_fallback = scalar_dispatch_plan(request)?;
    let environment_ok = match admission {
        ScalarAdmission::Evidence => qualified_sm89_exact_f32_environment(facts),
        ScalarAdmission::Candidate => qualified_sm89_exact_f32_candidate_environment(facts),
        ScalarAdmission::Proof => scalar_proof_board(facts),
    };
    if actual_fallback != fallback
        || request.op != crate::gpu::kernel_identity::ResolvedGemmOp::Tn
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
    Ok((plan != scalar_dispatch_plan(request)?).then_some(plan))
}

fn scalar_plan_with_admission(
    facts: ScalarLaunchFacts,
    request: F32TriadRequest,
    operands: F32TriadOperands,
    admission: ScalarAdmission,
) -> Result<ScalarDispatchPlan, String> {
    let fallback = scalar_dispatch_plan(request)?;
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
    // Every other split-M partition on a board whose exact module carries
    // the direct partial: the kernel masks every edge of its 64 x 64 tiles
    // and of its chunk, writes the planes the portable partial writes, and
    // the same reducer folds them, so the bits stay the plan's. It takes
    // about half the time of the portable partial. An Ada
    // toolkit the module was qualified on takes it outright; any other board
    // and toolkit proves the partial's planes against the portable ones at
    // first use.
    if let ScalarDispatchPlan::TnSplitM { m_chunk, chunks } = fallback
        && match admission {
            ScalarAdmission::Evidence => qualified_sm89_exact_f32_environment(facts),
            ScalarAdmission::Proof => !qualified_sm89_exact_f32_environment(facts),
            ScalarAdmission::Candidate => false,
        }
        && facts.sm89_exact_f32_symbols_loaded[1]
        && request.shape
            == F32TriadShape::contiguous(
                crate::gpu::kernel_identity::ResolvedGemmOp::Tn,
                (request.shape.m, request.shape.k, request.shape.n),
            )
        && request.shape.k.is_multiple_of(4)
        && request.shape.n.is_multiple_of(4)
        && qualified_sm89_exact_f32_tn_operands(operands)
    {
        return Ok(ScalarDispatchPlan::TnSm89DirectSplitM { m_chunk, chunks });
    }
    if fallback == (ScalarDispatchPlan::NnFinal { slim: false })
        && inference_overlay_admitted(admission, facts, qualified_inference_copyplan_environment)
        && request.op == crate::gpu::kernel_identity::ResolvedGemmOp::Nn
        && NN_INFERENCE_COPYPLAN_SM89_CELLS.contains(&request.shape)
        && qualified_nn_m64n64_operands(operands)
    {
        return Ok(ScalarDispatchPlan::NnSm89InferenceCopyPlanQualified);
    }
    if fallback == (ScalarDispatchPlan::NtFinal { slim: false })
        && inference_overlay_admitted(admission, facts, qualified_inference_copyplan_environment)
        && request.op == crate::gpu::kernel_identity::ResolvedGemmOp::Nt
        && request.shape == NT_D768_OUT_INFERENCE_COPYPLAN_SM89_CELL
        && qualified_nt_d768_transpose_m64n64_operands(operands)
    {
        return Ok(ScalarDispatchPlan::NtD768OutSm89InferenceCopyPlanQualified);
    }
    if fallback == (ScalarDispatchPlan::NtFinal { slim: false })
        && inference_overlay_admitted(
            admission,
            facts,
            qualified_nt_inference_copyplan_sibling_environment,
        )
        && request.op == crate::gpu::kernel_identity::ResolvedGemmOp::Nt
        && request.shape == NT_D768_IN_INFERENCE_COPYPLAN_SM89_CELL
        && qualified_nt_d768_transpose_m64n64_operands(operands)
    {
        return Ok(ScalarDispatchPlan::NtD768InSm89InferenceCopyPlanQualified);
    }
    if fallback == (ScalarDispatchPlan::NtFinal { slim: true })
        && inference_overlay_admitted(
            admission,
            facts,
            qualified_nt_inference_copyplan_sibling_environment,
        )
        && request.op == crate::gpu::kernel_identity::ResolvedGemmOp::Nt
        && request.shape == NT_PRISM_INFERENCE_COPYPLAN_SM89_CELL
        && qualified_nt_d768_transpose_m64n64_operands(operands)
    {
        return Ok(ScalarDispatchPlan::NtPrismSm89InferenceCopyPlanQualified);
    }
    if fallback == (ScalarDispatchPlan::NtFinal { slim: false })
        && inference_overlay_admitted(
            admission,
            facts,
            qualified_nt_inference_copyplan_sibling_environment,
        )
        && request.op == crate::gpu::kernel_identity::ResolvedGemmOp::Nt
        && qualified_nt_large_deep_transpose_m64n64_request(request)
        && qualified_nt_d768_transpose_m64n64_operands(operands)
    {
        return Ok(ScalarDispatchPlan::NtLargeDeepSm89InferenceCopyPlanQualified);
    }
    if fallback
        == (ScalarDispatchPlan::NtSplitKMain {
            n_main: 2_048,
            n_tail: 0,
        })
        && scalar_sm120_admitted(admission, facts)
        && qualified_nt_m2n16_splitk32_request(request)
        && qualified_nt_d768_transpose_m64n64_operands(operands)
    {
        return Ok(ScalarDispatchPlan::NtM2N16SplitK32Qualified);
    }
    if fallback == (ScalarDispatchPlan::NnFinal { slim: false })
        && scalar_sm120_admitted(admission, facts)
        && qualified_nn_m64n64_request(request)
        && qualified_nn_m64n64_operands(operands)
    {
        return Ok(ScalarDispatchPlan::NnM64N64Qualified);
    }
    if fallback == ScalarDispatchPlan::NnSplitKThin
        && scalar_sm120_admitted(admission, facts)
        && qualified_nn_m32n64_splitk32_request(request)
        && qualified_nn_m32n64_splitk32_operands(operands)
    {
        return Ok(ScalarDispatchPlan::NnM32N64SplitK32Qualified);
    }
    if fallback == (ScalarDispatchPlan::NtFinal { slim: false })
        && scalar_sm120_admitted(admission, facts)
        && qualified_nt_d768_transpose_m64n64_request(request)
        && qualified_nt_d768_transpose_m64n64_operands(operands)
    {
        return Ok(ScalarDispatchPlan::NtD768TransposeM64N64Qualified);
    }
    if fallback == (ScalarDispatchPlan::NtFinal { slim: false })
        && scalar_sm120_admitted(admission, facts)
        && qualified_nt_d768_out_transpose_m64n64_request(request)
        && qualified_nt_d768_transpose_m64n64_operands(operands)
    {
        return Ok(ScalarDispatchPlan::NtD768OutTransposeM64N64Qualified);
    }
    if fallback == (ScalarDispatchPlan::NtFinal { slim: false })
        && scalar_sm120_admitted(admission, facts)
        && qualified_nt_large_deep_transpose_m64n64_request(request)
        && qualified_nt_d768_transpose_m64n64_operands(operands)
    {
        return Ok(ScalarDispatchPlan::NtLargeDeepTransposeM64N64Qualified);
    }
    if fallback == (ScalarDispatchPlan::NtFinal { slim: true })
        && scalar_sm120_admitted(admission, facts)
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
        && scalar_sm120_admitted(admission, facts)
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
        && (scalar_sm120_admitted(admission, facts)
            || scalar_admitted(admission, facts, qualified_scalar_sm89_tn_environment))
        && qualified_tn_m16n16_splitm16_request(request)
        && qualified_tn_m16n16_splitm16_operands(operands)
    {
        return Ok(ScalarDispatchPlan::TnM16N16SplitM16Qualified);
    }
    // A slim plain input gradient whose weights the transpose scratch holds
    // may run the slim tile's chain as the NN 64 x 64 tile over the
    // transposed weights: the chain per output is the same, so the bits are
    // the plan's and the choice is speed alone. A CC 12.x board takes it on
    // every such gradient (RTX 5090: NT 4096x128x796 53 -> 29 us,
    // 4096x256x2112 138 -> 112, 16384x256x128 29 -> 26); the other
    // architectures only over a reduction that is not whole 32-column steps,
    // where the slim tile's tail is slow (RTX 6000 Ada: 4096x128x796 62 ->
    // 48) and whole steps keep the slim tile (4096x256x2112 183 against 214).
    // Outputs 16 past a multiple of 32 stay on the slim tile everywhere.
    if fallback == (ScalarDispatchPlan::NtFinal { slim: true })
        && admission == ScalarAdmission::Evidence
        && (facts.compute_capability.0 == 12 || !request.shape.n.is_multiple_of(32))
        && nt_slim_via_transpose_request(request)
        && qualified_nt_d768_transpose_m64n64_operands(operands)
    {
        return Ok(ScalarDispatchPlan::NtSlimViaTranspose);
    }
    Ok(fallback)
}

/// Whether an input gradient's weights fit the transpose scratch and its
/// shape is one the transposed chain serves: the outputs' row stride whole
/// 16-byte chunks and the reduction not 16 past a multiple of 32.
fn nt_slim_via_transpose_request(request: F32TriadRequest) -> bool {
    let shape = request.shape;
    request.op == crate::gpu::kernel_identity::ResolvedGemmOp::Nt
        && shape
            == F32TriadShape::contiguous(
                crate::gpu::kernel_identity::ResolvedGemmOp::Nt,
                (shape.m, shape.k, shape.n),
            )
        && shape.k.is_multiple_of(4)
        && shape.n % 32 != 16
        && shape.k.checked_mul(shape.n).is_some_and(|elements| {
            elements <= super::contract::SCALAR_TRANSPOSE_SCRATCH_CAP_ELEMENTS
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
/// The forward Slim NN path uses N as output dim; NT-via-T backward dx reads
/// n_in, and the kernel tiles any n_in. 768 keeps the production input
/// widths on the slim split with F=4 K-tile partials (576 blocks against
/// the plain big NT's 144). Determinism preserved: F is shape-keyed (a
/// function of n_out, not of the batch).
pub(super) const GEMM_SLIM_NT_NIN_MAX: usize = 768;

/// The reduction from which the slim NT split also serves 512 to 1024 rows.
pub(super) const GEMM_SLIM_NT_SHORT_BATCH_MIN_REDUCTION: usize = 1536;

/// The reduction from which the slim NT split takes over from the thin one
/// past 512 rows, and at 512 rows from [`GEMM_SLIM_NT_MID_MIN_WIDTH`]
/// inputs.
pub(super) const GEMM_SLIM_NT_MID_MIN_REDUCTION: usize = 1024;

/// The input width from which 512 rows over a reduction of 1024 or more
/// take the slim NT split.
pub(super) const GEMM_SLIM_NT_MID_MIN_WIDTH: usize = 512;

/// M threshold below which we force Slim-N even for N ≥ 129 (wave underfill protection).
/// At M < 512, Big tile BM=128 gives ≤4 M-blocks; adding N-blocks via Slim's BN=64 (vs Big's BN=128)
/// doubles grid to reduce wave underfill on Ada's 142 SMs. Only matters when N ≥ 129 (otherwise slim already chosen).
pub(super) const GEMM_M_SLIM_FORCE: usize = 512;

/// Single source of truth for Split-K/M scratch buffer cap, in f32 elements:
/// 32M f32 = 128 MB, allocated on the first split. All Split-K dispatch gates
/// (NN fwd, NT bwd_dx, Split-M TN bwd_dw) and the allocation read this. A
/// 32-wide split of an input gradient over 1024 rows of 384 inputs and 1536
/// outputs keeps 48 planes, 18.9M floats; under a quarter of this cap the
/// gradient fell to the plain slim tile at two and a half times the time.
pub(super) const SPLITK_SCRATCH_CAP: usize = 1 << 25;

/// One device-scope completion counter per fused TF32 split-K output tile.
pub(super) const TF32_SPLITK_COUNTER_CAP: usize = 1 << 16;

/// The tile that writes the batch-invariant chunk fold's partial planes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ExactChunkPartial {
    /// The 128 x 64 slim tile, for a large grid.
    Slim,
    /// A 32 x 64 tile, four times the CTAs of the slim one for a thin batch.
    Small,
    /// The pipelined 16 x 16 tile, for a single row.
    Narrow16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ScalarDispatchPlan {
    NnUltraThin,
    NnNarrowSmall,
    NnNarrow,
    NnGemv,
    NnSplitKThinTail {
        k_main: usize,
        k_tail: usize,
    },
    NnSplitKThin,
    NnSplitKSlim {
        chunks: u32,
    },
    NnM32N64SplitK32Qualified,
    NnM64N64Qualified,
    NnSm89InferenceCopyPlanQualified,
    NnFinal {
        slim: bool,
    },
    /// The batch-invariant exact forward's thin tile, one row per thread.
    NnExactThin,
    /// The batch-invariant exact forward's pipelined 16 x 16 tile.
    NnExactNarrow16,
    /// The batch-invariant exact policy's chunk fold: the partial writes one
    /// 64-input chunk per plane and `splitk_reduce` sums the planes in order
    /// and adds the bias, `piece_rows` rows at a time so the planes fit the
    /// split scratch at any batch. Every partial writes the same planes.
    NnExactChunk {
        partial: ExactChunkPartial,
        chunks: u32,
        piece_rows: u32,
        pieces: u32,
    },
    TnGemv,
    TnNarrow,
    TnNarrowSplitM {
        m_chunk: usize,
        chunks: usize,
    },
    TnSplitM {
        m_chunk: usize,
        chunks: usize,
    },
    TnD768InSm89DualChunkQualified,
    TnD768OutSm89DirectBk16Qualified,
    TnPrismSm89DirectBk16Qualified,
    /// The split-M plan's partition written by the Ada's direct 64 x 64
    /// partial: the same planes and the same fold as the portable partial.
    TnSm89DirectSplitM {
        m_chunk: usize,
        chunks: usize,
    },
    TnD128InSm89DirectFoldQualified,
    TnD128OutSm89DirectFoldQualified,
    TnM16N16SplitM16Qualified,
    TnFinal {
        slim: bool,
    },
    NtNarrow,
    NtSmallBatchWide,
    NtGemv,
    NtSplitKTail {
        k_main: usize,
        k_tail: usize,
    },
    NtSplitKMain {
        n_main: usize,
        n_tail: usize,
    },
    NtSplitKSlim {
        chunks: u32,
    },
    NtMidBatchWide,
    NtM2N16SplitK32Qualified,
    NtD768TransposeM64N64Qualified,
    NtD768OutTransposeM64N64Qualified,
    NtD768OutSm89InferenceCopyPlanQualified,
    NtD768InSm89InferenceCopyPlanQualified,
    NtPrismSm89InferenceCopyPlanQualified,
    NtLargeDeepSm89InferenceCopyPlanQualified,
    NtLargeDeepTransposeM64N64Qualified,
    /// The slim plain tile's chain as the NN 64 x 64 tile over the
    /// transposed weights: the same chain per output, so the same bits.
    NtSlimViaTranspose,
    NtPrismVectorQualified,
    NtD128OutTransposeM64N64Qualified,
    NtFinal {
        slim: bool,
    },
}

impl ScalarDispatchPlan {
    pub(super) const fn needs_split_scratch(self) -> bool {
        matches!(
            self,
            Self::NnSplitKThinTail { .. }
                | Self::NnSplitKThin
                | Self::NnSplitKSlim { .. }
                | Self::NnExactChunk { .. }
                | Self::NnM32N64SplitK32Qualified
                | Self::TnNarrowSplitM { .. }
                | Self::TnSplitM { .. }
                | Self::TnD768OutSm89DirectBk16Qualified
                | Self::TnPrismSm89DirectBk16Qualified
                | Self::TnSm89DirectSplitM { .. }
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
                | Self::NtD768OutSm89InferenceCopyPlanQualified
                | Self::NtD768InSm89InferenceCopyPlanQualified
                | Self::NtPrismSm89InferenceCopyPlanQualified
                | Self::NtLargeDeepSm89InferenceCopyPlanQualified
                | Self::NtLargeDeepTransposeM64N64Qualified
                | Self::NtSlimViaTranspose
                | Self::NtPrismVectorQualified
                | Self::NtD128OutTransposeM64N64Qualified
                | Self::TnD768InSm89DualChunkQualified
        )
    }
}

/// The inputs one piece of a slim split-K product sums, NN and NT alike.
pub(super) const SLIM_SPLIT_K_CHUNK: u32 = 64;

fn slim_final(output_rows: usize, output_columns: usize) -> bool {
    output_columns <= GEMM_SLIM_MAX
        || (output_rows < GEMM_M_SLIM_FORCE && output_columns >= GEMM_CUSTOM_MIN)
}

const fn scalar_split_policy() -> crate::gpu::kernel_identity::ScalarSplitPolicy {
    crate::gpu::kernel_identity::ScalarSplitPolicy::current()
}

fn scalar_nn_plan(dims: GemmDims) -> Result<ScalarDispatchPlan, String> {
    let policy = scalar_split_policy();
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
    // The GEMV kernel serves any reduction: a warp per row, lanes past K
    // add nothing.
    if n_out == 1 {
        return Ok(ScalarDispatchPlan::NnGemv);
    }
    let plain_slim_blocks = checked_tile_grid(dims.m_u32, 128, dims.n_u32, 64)?;
    let underfill = plain_slim_blocks < policy.thin_split_max_tiles;
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
        let chunks = dims.k_u32.div_ceil(SLIM_SPLIT_K_CHUNK);
        if chunks >= 6
            && checked_mul3(
                checked_usize(chunks, "NN slim split-K chunks")?,
                batch,
                n_out,
                "NN slim split-K scratch",
            )? <= SPLITK_SCRATCH_CAP
            && plain_slim_blocks > 0
            && plain_slim_blocks < policy.slim_split_max_tiles
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

pub(in crate::gpu) fn tc_half_policy_prefers_scalar_forward(
    compute_capability: (u32, u32),
    dims: (usize, usize, usize),
) -> Result<bool, String> {
    let policy = super::super::kernel_identity::Sm80TcPolicy::current();
    if compute_capability != policy.deep_split_k_compute_capability
        || dims.2 != policy.deep_split_k_output_columns
    {
        return Ok(false);
    }
    // The scalar split was measured against the tensor-core tile only while
    // its partial planes fit 8M floats; past that the tile keeps the cell.
    const MEASURED_PARTIAL_FLOATS: usize = 1 << 23;
    if checked_mul3(
        dims.1 / 32,
        dims.0,
        dims.2,
        "half split-K preference planes",
    )? > MEASURED_PARTIAL_FLOATS
    {
        return Ok(false);
    }
    let plan = scalar_nn_plan(GemmDims::nn(dims, dims.1)?)?;
    Ok(match plan {
        ScalarDispatchPlan::NnSplitKThinTail { .. } => {
            dims.1 >= policy.deep_split_k_tail_min_reduction
        }
        ScalarDispatchPlan::NnSplitKThin => dims.1 >= policy.deep_split_k_aligned_min_reduction,
        _ => false,
    })
}

fn scalar_tn_plan(dims: GemmDims) -> Result<ScalarDispatchPlan, String> {
    let (batch, n_in, n_out) = dims.tuple();
    // The GEMV kernel serves any batch and width: a warp per weight, lanes
    // past the batch add nothing.
    if n_out == 1 {
        return Ok(ScalarDispatchPlan::TnGemv);
    }
    if (2..=127).contains(&n_out) {
        if let Some((m_chunk, chunks)) = splitm_tn_narrow_partition(batch, n_in, n_out)? {
            return Ok(ScalarDispatchPlan::TnNarrowSplitM { m_chunk, chunks });
        }
        return Ok(ScalarDispatchPlan::TnNarrow);
    }
    if let Some((m_chunk, chunks)) = splitm_tn_partition(batch, n_in, n_out)? {
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

fn scalar_nt_plan(dims: GemmDims) -> Result<ScalarDispatchPlan, String> {
    let policy = scalar_split_policy();
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
    let underfill = plain_slim_blocks < policy.thin_split_max_tiles;
    if batch < 32 && n_out >= 128 {
        return Ok(ScalarDispatchPlan::NtSmallBatchWide);
    }
    // The slim split through the transposed weights, 64-wide chunks: past
    // 1024 rows wherever its own grid stays under the slim bound, and from
    // 512 rows over a reduction of 1536 or more, where it runs in about half
    // the time on both the RTX 5090 and the RTX 6000 Ada (NT 1024x384x1536
    // 99.6 -> 40.4 us and 111.8 -> 61.1). Over a reduction of 1024 to 1535
    // it takes over from the thin split wherever that split's planes would
    // fit, past 512 rows or from 512 inputs at 512 rows: with half the planes
    // to write and fold, the 59 shapes this moves of a 72-shape grid ran on
    // an RTX 6000 Ada in 0.53 of the thin split's time (0.41 to 1.00, none
    // slower), while at 512 rows under 512 inputs the thin split's finer
    // pieces stayed up to 1.24 ahead.
    // Over a shorter reduction at 1024 rows the two cards disagree, and the
    // exact plan is one for all of them. Up to 1024 rows its own grid must
    // fill the thin split's bound.
    if batch >= 512
        && (128..=GEMM_SLIM_NT_NIN_MAX).contains(&n_in)
        && n_out >= 64
        && n_out.is_multiple_of(32)
        && dims.kn <= super::contract::SCALAR_GENERIC_TRANSPOSE_ROUTE_CAP_ELEMENTS
    {
        let chunks = dims.n_u32.div_ceil(SLIM_SPLIT_K_CHUNK);
        let partials = checked_mul3(
            checked_usize(chunks, "NT slim split-K chunks")?,
            batch,
            n_in,
            "NT slim split-K scratch",
        )?;
        let fills_thin_bound = checked_usize(plain_slim_blocks, "NT slim split-K grid")?
            .checked_mul(checked_usize(chunks, "NT slim split-K chunks")?)
            .is_some_and(|ctas| ctas >= policy.thin_split_max_tiles as usize);
        if chunks >= 2
            && partials <= SPLITK_SCRATCH_CAP
            && plain_slim_blocks > 0
            && plain_slim_blocks < policy.slim_split_max_tiles
            && (batch > 1024
                || (n_out >= GEMM_SLIM_NT_SHORT_BATCH_MIN_REDUCTION && fills_thin_bound)
                || (n_out >= GEMM_SLIM_NT_MID_MIN_REDUCTION
                    && (batch > 512 || n_in >= GEMM_SLIM_NT_MID_MIN_WIDTH)
                    && partials <= SPLITK_SCRATCH_CAP / 2
                    && fills_thin_bound))
        {
            return Ok(ScalarDispatchPlan::NtSplitKSlim { chunks });
        }
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

/// Reductions at least this deep fold as 64-input chunks under the
/// batch-invariant exact policy, shallower ones as one chain.
pub(crate) const EXACT_CHUNKS_MIN_K: usize = 512;

/// A narrow layer (fewer outputs than this) folds as chunks from
/// [`EXACT_CHUNKS_NARROW_MIN_K`] inputs: a chain tile has too few columns to
/// keep the card busy, and the chunks add the reduction's parallelism.
pub(crate) const EXACT_CHUNKS_NARROW_N: usize = 128;

/// The chunk threshold of a narrow layer.
pub(crate) const EXACT_CHUNKS_NARROW_MIN_K: usize = 384;

/// The inputs one chunk of a deep layer's fold sums.
pub(crate) const EXACT_CHUNK: usize = 64;

/// Whether a layer of `k` inputs and `n` outputs folds as chunks under the
/// batch-invariant exact policy. The rule reads the layer's shape alone, so
/// a layer folds the same way at every row count, and a CPU twin that
/// applies it reproduces the GPU's bits.
pub(crate) const fn exact_chunks(k: usize, n: usize) -> bool {
    k >= EXACT_CHUNKS_MIN_K || (n < EXACT_CHUNKS_NARROW_N && k >= EXACT_CHUNKS_NARROW_MIN_K)
}

/// The batch-invariant exact policy's forward plan. A shallow layer runs
/// one of the chain tiles, each of which seeds an output with its bias and
/// walks the reduction with one `fma` per input in ascending order, padding
/// a ragged end with zero products; which tile serves is scheduling alone.
/// The slim tile pads to 32 inputs, not 16, so it serves only where the two
/// paddings agree. A deep layer runs the chunk fold. Every tile and every
/// partial writes the same bits, so the thresholds read the card's own
/// multiprocessor count and each card runs the tiles that fill it.
pub(super) fn exact_batch_invariant_nn_plan(
    request: F32TriadRequest,
    multiprocessors: u32,
    compute_capability: (u32, u32),
) -> Result<ScalarDispatchPlan, String> {
    if request.op != ResolvedGemmOp::Nn {
        return Err(format!(
            "the batch-invariant exact forward plan takes NN, not {:?}",
            request.op
        ));
    }
    request.shape.validate(request.op)?;
    if multiprocessors == 0 {
        return Err("the batch-invariant exact plan needs the card's multiprocessor count".into());
    }
    let (m, k, n) = (request.shape.m, request.shape.k, request.shape.n);
    if m == 0 || k == 0 || n == 0 {
        return Err(format!(
            "the batch-invariant exact plan takes no empty axis: {m}x{k}x{n}"
        ));
    }
    let waves = multiprocessors as usize;
    if exact_chunks(k, n) {
        let chunks = k.div_ceil(EXACT_CHUNK);
        let piece_rows = (SPLITK_SCRATCH_CAP / chunks.saturating_mul(n).max(1)).clamp(1, m);
        if chunks.saturating_mul(n) > SPLITK_SCRATCH_CAP {
            return Err(format!(
                "a chunk-fold row of {chunks} chunks x {n} outputs passes the split scratch \
                 of {SPLITK_SCRATCH_CAP} floats"
            ));
        }
        // The slim partial once its grid fills three quarters of a wave of
        // the card, the single-row tile for one row, the small tile between.
        let slim_grid = m
            .div_ceil(128)
            .saturating_mul(n.div_ceil(64))
            .saturating_mul(chunks);
        let partial = if m > 64 && slim_grid.saturating_mul(4) >= waves.saturating_mul(3) {
            ExactChunkPartial::Slim
        } else if m == 1 {
            ExactChunkPartial::Narrow16
        } else {
            ExactChunkPartial::Small
        };
        return Ok(ScalarDispatchPlan::NnExactChunk {
            partial,
            chunks: u32::try_from(chunks)
                .map_err(|_| format!("{chunks} chunks pass the grid's third axis"))?,
            piece_rows: u32::try_from(piece_rows)
                .map_err(|_| format!("a piece of {piece_rows} rows passes u32"))?,
            pieces: u32::try_from(m.div_ceil(piece_rows))
                .map_err(|_| format!("{m} rows in pieces of {piece_rows} pass u32"))?,
        });
    }
    // The 128 x 128 chain tile once its own grid covers the card enough
    // times; the two tiles give the same bits, so the bound is speed alone
    // and differs by architecture. Over the same 164 shapes (64 to 384
    // inputs, up to eight waves), three quarters of a wave came within 0.6%
    // of the best tile per shape on an RTX 6000 Ada, where the large tile
    // ran in 0.53 to 0.94 of the small one's time; an RTX 5090 keeps the
    // small tile longer, and four waves came within 1.0% there, where three
    // quarters lost 3%.
    let quarter_waves = if compute_capability.0 == 12 { 16 } else { 3 };
    let big_grid = m.div_ceil(128).saturating_mul(n.div_ceil(128));
    if m >= GEMM_CUSTOM_MIN
        && n >= GEMM_CUSTOM_MIN
        && big_grid.saturating_mul(4) >= waves.saturating_mul(quarter_waves)
    {
        return Ok(ScalarDispatchPlan::NnFinal { slim: false });
    }
    if m.div_ceil(64).saturating_mul(n.div_ceil(64)) >= waves / 2 {
        return Ok(ScalarDispatchPlan::NnM64N64Qualified);
    }
    // A thin batch: the chain's length sets the time. One row takes the thin
    // tile with a row per thread, every other thin batch the pipelined 16 x
    // 16 tile, whose grid spreads furthest with its next slabs in flight.
    // Both put their 16-row tiles on the grid's row axis.
    if m.div_ceil(16) > EXACT_SLAB_GRID_ROWS_MAX {
        return Ok(ScalarDispatchPlan::NnNarrowSmall);
    }
    Ok(if m == 1 {
        ScalarDispatchPlan::NnExactThin
    } else {
        ScalarDispatchPlan::NnExactNarrow16
    })
}

/// The grid's row axis, where the thin and pipelined 16 x 16 tiles put their
/// 16-row tiles.
pub(super) const EXACT_SLAB_GRID_ROWS_MAX: usize = 65_535;

/// The exact scalar plan of a request: a function of the shape alone, so
/// every card of every architecture cuts each reduction at the same places
/// and writes the same bits.
pub(super) fn scalar_dispatch_plan(request: F32TriadRequest) -> Result<ScalarDispatchPlan, String> {
    request.shape.validate(request.op)?;
    let dims = (request.shape.m, request.shape.k, request.shape.n);
    match request.op {
        crate::gpu::kernel_identity::ResolvedGemmOp::Nn => {
            scalar_nn_plan(GemmDims::nn(dims, request.shape.lda)?)
        }
        crate::gpu::kernel_identity::ResolvedGemmOp::Tn => scalar_tn_plan(GemmDims::tn(dims)?),
        crate::gpu::kernel_identity::ResolvedGemmOp::Nt => scalar_nt_plan(GemmDims::nt(dims)?),
    }
}

pub(super) fn nn_routes_to_big(batch: usize, n_in: usize, n_out: usize) -> bool {
    GemmDims::nn((batch, n_in, n_out), n_in)
        .and_then(scalar_nn_plan)
        .is_ok_and(|plan| matches!(plan, ScalarDispatchPlan::NnFinal { slim: false }))
}

/// Whether the f32 TN cascade runs the narrow kernel unsplit: a narrow
/// weight gradient whose batch the split-M rule leaves whole. A split one
/// folds its chunks in the f32 reducer, so a typed launch upcasts into it.
pub(super) fn tn_routes_to_narrow(batch: usize, n_in: usize, n_out: usize) -> bool {
    GemmDims::tn((batch, n_in, n_out))
        .and_then(scalar_tn_plan)
        .is_ok_and(|plan| plan == ScalarDispatchPlan::TnNarrow)
}

pub(super) fn tn_routes_to_big(batch: usize, n_in: usize, n_out: usize) -> bool {
    GemmDims::tn((batch, n_in, n_out))
        .and_then(scalar_tn_plan)
        .is_ok_and(|plan| matches!(plan, ScalarDispatchPlan::TnFinal { slim: false }))
}

pub(super) fn nt_routes_to_big(batch: usize, n_in: usize, n_out: usize) -> bool {
    GemmDims::nt((batch, n_in, n_out))
        .and_then(scalar_nt_plan)
        .is_ok_and(|plan| matches!(plan, ScalarDispatchPlan::NtFinal { slim: false }))
}

/// Whether the f32 NT cascade runs the narrow kernel over a reduction of
/// 128 or more: a batch under 128 rows that no split-K route takes. No
/// qualified cell replaces either plan on any board, so the plain narrow
/// kernel is the exact f32 route of these shapes everywhere.
pub(super) fn nt_routes_to_wide_narrow(batch: usize, n_in: usize, n_out: usize) -> bool {
    GemmDims::nt((batch, n_in, n_out))
        .and_then(scalar_nt_plan)
        .is_ok_and(|plan| {
            matches!(
                plan,
                ScalarDispatchPlan::NtSmallBatchWide | ScalarDispatchPlan::NtMidBatchWide
            )
        })
}

#[cfg(test)]
mod scalar_split_policy_tests {
    use super::{
        INFERENCE_COPYPLAN_EVIDENCE_COHORTS, InferenceCopyPlanQualificationIdentity,
        NT_INFERENCE_COPYPLAN_COMPOSED_EVIDENCE_COHORTS,
        NT_INFERENCE_COPYPLAN_COMPOSED_QUALIFICATION_CANDIDATES,
        NtInferenceCopyPlanComposedQualificationIdentity, ScalarDispatchPlan, ScalarLaunchFacts,
        ScalarTransposeQualificationIdentity, TN_M16N16_SPLITM16_SM89_BINDINGS,
        qualified_inference_copyplan_environment,
        qualified_nt_inference_copyplan_sibling_environment, qualified_scalar_sm89_tn_environment,
        scalar_dispatch_plan, scalar_launch_plan, scalar_proof_plan,
    };
    use crate::gpu::gemm_bi_triad::{F32TriadOperands, F32TriadRequest, F32TriadShape};
    use crate::gpu::kernel_identity::{
        ArtifactIdentity, ArtifactKind, COMPILER_REVISION, COMPOSER_REVISION, CompilerIdentity,
        CudaTarget, ModuleKind, NUMERIC_ABI_REVISION, ResolvedGemmOp, SCHEDULE_REVISION,
    };

    #[derive(Clone, Copy)]
    struct MeasuredComposedRow {
        nvrtc: (i32, i32),
        inference_source: &'static str,
        inference_key: &'static str,
        inference_artifact: &'static str,
        inference_header: &'static str,
        scalar_key: &'static str,
        scalar_artifact: &'static str,
        scalar_header: &'static str,
        library: &'static str,
    }

    fn digest(hex: &str) -> [u8; 32] {
        assert_eq!(hex.len(), 64);
        std::array::from_fn(|index| u8::from_str_radix(&hex[index * 2..index * 2 + 2], 16).unwrap())
    }

    fn measured_composed_identities() -> [NtInferenceCopyPlanComposedQualificationIdentity; 3] {
        const SCALAR_SOURCE: &str =
            "dcd69307f3b0e46dc54615bb107c693060cc5314bef97e21f3ff5e57d7e29c0c";
        let rows = [
            MeasuredComposedRow {
                nvrtc: (12, 8),
                inference_source: "3b7a86ecc220af013f561d05e944ccbb881e44c7825637c195c4df372d59a6fc",
                inference_key: "0c1190d503347ec180d6b50b705b67011bbc815fdc657eab8a392215bfb3e82f",
                inference_artifact: "f08bc370bcd120f0194c13f0eb2ca53217160a018802cc0d98205d926ff41171",
                inference_header: "5c761ff1603930c492584b77b2d7791168f4d12ddda697a3cb8f3ccac05a3ebe",
                scalar_key: "2ced416b002e37ac17648a21594e4e32a1b6b632957c95a2973064759eacadb7",
                scalar_artifact: "a084c0d89de9f66d75379bfb27c82ad3260a4e0ce7dc2158d6e02848dd237041",
                scalar_header: "08efdf65161f28be93d926e1335ce7ddd1a44dedc26eee7e1f92351a20029861",
                library: "26b0a3a02044ffcbc1693fd83e9261beffa692a4fbcfe3ac5e9d8c87980bb155",
            },
            MeasuredComposedRow {
                nvrtc: (13, 0),
                inference_source: "3b7a86ecc220af013f561d05e944ccbb881e44c7825637c195c4df372d59a6fc",
                inference_key: "ccedc96d87a6c8396e187c24c5235da8176bae5186a24e852ec7c14674157caf",
                inference_artifact: "5ff11084f3605a88bb83556de28a48fcbc664b2723034fe5f269037f3fb5350b",
                inference_header: "f25b2479da245ace32a35253ae89ef546de7f546f1707540f08bac6efa41de62",
                scalar_key: "3e669ce36bf62b6f8db06997080bba9552fa41f48ae412f0a368aafa5d1d8fc9",
                scalar_artifact: "0ade495980ccfffcc713fff276631310e43751f5bf251acb558de5b0dd116355",
                scalar_header: "ad26b24e0c889da1aee1372e135d52ddef7f84327055508e6afd7e3582a2fdc7",
                library: "709b91c36bfb0ed966ee69adc8d6f87ff110eecf3dfb5060367f183ce614eb0d",
            },
            MeasuredComposedRow {
                nvrtc: (13, 2),
                inference_source: "3b7a86ecc220af013f561d05e944ccbb881e44c7825637c195c4df372d59a6fc",
                inference_key: "6c7601163ad209ceb0d04c696cc572da4d242526f69bc919576e583f2edc5364",
                inference_artifact: "f567404343421b8a0f7281537b445bdc854516e46e2319091e3826ac5db9db03",
                inference_header: "1ad529b315150501056f981480cc4c05f33a0fa98a358ae8e63fc75467526c0d",
                scalar_key: "2639ec834b8a5e089a6e5f691d5623ad9ef287aa240042e1fe1fae11eeb0d2de",
                scalar_artifact: "9902f051cff338557807bc4c099554f7568a6ea25e89812ce18ee7797c8a184f",
                scalar_header: "9f135afdc65e45b4b97e25ad314be4edc0b7ac698fdba470d399b2ee99ee5a7a",
                library: "d031a53eb97235b70f62f652932db1bdf728ea229c8ca809d53c5ffd91642687",
            },
        ];
        rows.map(|row| NtInferenceCopyPlanComposedQualificationIdentity {
            scalar: ScalarTransposeQualificationIdentity {
                nvrtc_version: row.nvrtc,
                compile_key: digest(row.scalar_key),
                artifact_digest: digest(row.scalar_artifact),
                source_digest: digest(SCALAR_SOURCE),
                header_manifest_digest: digest(row.scalar_header),
                nvrtc_library_domain: digest(row.library),
            },
            inference: InferenceCopyPlanQualificationIdentity {
                nvrtc_version: row.nvrtc,
                compile_key: digest(row.inference_key),
                artifact_digest: digest(row.inference_artifact),
                source_digest: digest(row.inference_source),
                header_manifest_digest: digest(row.inference_header),
                nvrtc_library_domain: digest(row.library),
            },
        })
    }

    fn facts_for_measured_pair(
        identity: NtInferenceCopyPlanComposedQualificationIdentity,
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
            inference_artifact: ArtifactIdentity {
                module_kind: ModuleKind::Inference,
                artifact_kind: ArtifactKind::Ptx,
                compile_key: identity.inference.compile_key,
                artifact_digest: identity.inference.artifact_digest,
            },
            inference_compiler: compiler(
                identity.inference.source_digest,
                identity.inference.compile_key,
                identity.inference.header_manifest_digest,
                identity.inference.nvrtc_version,
                identity.inference.nvrtc_library_domain,
            ),
            inference_copyplan_loaded: true,
            sm89_exact_f32_artifact: None,
            sm89_exact_f32_compiler: None,
            sm89_exact_f32_symbols_loaded: [false; 3],
            sm89_exact_f32_d128_artifact: None,
            sm89_exact_f32_d128_compiler: None,
            sm89_exact_f32_d128_symbols_loaded: [false; 2],
            compute_capability: (8, 9),
            sm120_scalar_resources_qualified: false,
        }
    }

    fn plan(op: ResolvedGemmOp, dims: (usize, usize, usize)) -> Result<ScalarDispatchPlan, String> {
        scalar_dispatch_plan(F32TriadRequest {
            op,
            shape: F32TriadShape::contiguous(op, dims),
        })
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
            inference_artifact: ArtifactIdentity {
                module_kind: ModuleKind::Inference,
                artifact_kind: ArtifactKind::Ptx,
                compile_key: [0; 32],
                artifact_digest: [0; 32],
            },
            inference_compiler: compiler,
            inference_copyplan_loaded: false,
            sm89_exact_f32_artifact: None,
            sm89_exact_f32_compiler: None,
            sm89_exact_f32_symbols_loaded: [false; 3],
            sm89_exact_f32_d128_artifact: None,
            sm89_exact_f32_d128_compiler: None,
            sm89_exact_f32_d128_symbols_loaded: [false; 2],
            compute_capability: (12, 0),
            sm120_scalar_resources_qualified: true,
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

    fn inference_copyplan_facts(index: usize) -> ScalarLaunchFacts {
        let identity = INFERENCE_COPYPLAN_EVIDENCE_COHORTS[index];
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
            inference_artifact: ArtifactIdentity {
                module_kind: ModuleKind::Inference,
                artifact_kind: ArtifactKind::Ptx,
                compile_key: identity.compile_key,
                artifact_digest: identity.artifact_digest,
            },
            inference_compiler: compiler,
            inference_copyplan_loaded: true,
            sm89_exact_f32_artifact: None,
            sm89_exact_f32_compiler: None,
            sm89_exact_f32_symbols_loaded: [false; 3],
            sm89_exact_f32_d128_artifact: None,
            sm89_exact_f32_d128_compiler: None,
            sm89_exact_f32_d128_symbols_loaded: [false; 2],
            compute_capability: (8, 9),
            sm120_scalar_resources_qualified: false,
        }
    }

    fn nt_inference_copyplan_sibling_facts(index: usize) -> ScalarLaunchFacts {
        let identity = NT_INFERENCE_COPYPLAN_COMPOSED_QUALIFICATION_CANDIDATES[index];
        let mut facts = inference_copyplan_facts(index);
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
    fn nn_thin_split_admission_counts_the_shapes_own_tiles() {
        // 128 plain 128 x 64 tiles split; 192 do not, on every card.
        assert_eq!(
            plan(ResolvedGemmOp::Nn, (512, 64, 2048)).unwrap(),
            ScalarDispatchPlan::NnSplitKThin
        );
        assert!(!matches!(
            plan(ResolvedGemmOp::Nn, (1024, 64, 1536)).unwrap(),
            ScalarDispatchPlan::NnSplitKThin | ScalarDispatchPlan::NnSplitKThinTail { .. }
        ));
    }

    #[test]
    fn nt_split_admission_counts_the_shapes_own_tiles() {
        // The 32-wide split takes the underfilled grid; the slim split takes
        // 512 to 1024 rows only over a reduction of 1536 or more.
        for dims in [(256, 2048, 128), (512, 2048, 128)] {
            assert_eq!(
                plan(ResolvedGemmOp::Nt, dims).unwrap(),
                ScalarDispatchPlan::NtSplitKMain {
                    n_main: 128,
                    n_tail: 0,
                },
                "{dims:?}"
            );
        }
        assert_eq!(
            plan(ResolvedGemmOp::Nt, (1024, 384, 1536)).unwrap(),
            ScalarDispatchPlan::NtSplitKSlim { chunks: 24 }
        );
        assert!(!matches!(
            plan(ResolvedGemmOp::Nt, (1024, 1536, 384)).unwrap(),
            ScalarDispatchPlan::NtSplitKSlim { .. }
        ));
        assert!(!matches!(
            plan(ResolvedGemmOp::Nt, (1024, 1536, 128)).unwrap(),
            ScalarDispatchPlan::NtSplitKMain { .. } | ScalarDispatchPlan::NtSplitKTail { .. }
        ));
    }

    #[test]
    fn nt_generic_split_admission_keeps_the_pre_large_deep_scratch_boundary() {
        assert_eq!(
            plan(ResolvedGemmOp::Nt, (32, 2_304, 2_048)).unwrap(),
            ScalarDispatchPlan::NtMidBatchWide
        );
    }

    #[test]
    fn nt_split_admission_promotes_the_two_qualified_thin_output_cells() {
        for dims in [(512, 16, 2048), (16, 512, 2048)] {
            assert_eq!(
                plan(ResolvedGemmOp::Nt, dims).unwrap(),
                ScalarDispatchPlan::NtSplitKMain {
                    n_main: 2048,
                    n_tail: 0,
                },
                "{dims:?}"
            );
        }

        for (dims, expected) in [
            ((512, 17, 2048), ScalarDispatchPlan::NtFinal { slim: true }),
            ((17, 512, 2048), ScalarDispatchPlan::NtSmallBatchWide),
            ((512, 16, 2016), ScalarDispatchPlan::NtFinal { slim: true }),
            ((16, 512, 2016), ScalarDispatchPlan::NtSmallBatchWide),
        ] {
            assert_eq!(
                plan(ResolvedGemmOp::Nt, dims).unwrap(),
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
        mutations[1].compute_capability = (10, 0);
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
        let fallback = scalar_dispatch_plan(request).unwrap();
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
        mutations[2].compute_capability = (9, 0);
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
    fn tn_split_m_partition_is_a_function_of_the_shape() {
        assert_eq!(
            plan(ResolvedGemmOp::Tn, (4096, 128, 128)).unwrap(),
            ScalarDispatchPlan::TnSplitM {
                m_chunk: 16,
                chunks: 256,
            }
        );
    }

    #[test]
    fn tn_small_input_splitm_gap_is_covered_by_the_generic_member() {
        let plan = plan(ResolvedGemmOp::Tn, (4096, 24, 768)).unwrap();
        assert!(matches!(
            plan,
            ScalarDispatchPlan::TnSplitM {
                m_chunk: 96,
                chunks: 43,
            }
        ));
    }

    #[test]
    fn nt_slim_input_gradients_run_through_the_transposed_weights() {
        let nt = |dims| F32TriadRequest {
            op: ResolvedGemmOp::Nt,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, dims),
        };
        let operands = nn_qualified_operands();
        for compute_capability in [(8, 0), (8, 9), (12, 0)] {
            let mut facts = tn_admission_facts();
            facts.compute_capability = compute_capability;
            for dims in [(4096, 128, 796), (16384, 384, 768), (8192, 128, 796)] {
                assert_eq!(
                    plan(ResolvedGemmOp::Nt, dims).unwrap(),
                    ScalarDispatchPlan::NtFinal { slim: true },
                    "{dims:?}"
                );
                // Whole 32-column reduction steps take the transposed chain
                // on a CC 12.x board only.
                let transposed = compute_capability.0 == 12 || dims.2 % 32 != 0;
                assert_eq!(
                    scalar_launch_plan(facts, nt(dims), operands).unwrap(),
                    if transposed {
                        ScalarDispatchPlan::NtSlimViaTranspose
                    } else {
                        ScalarDispatchPlan::NtFinal { slim: true }
                    },
                    "{compute_capability:?} {dims:?}"
                );
                assert_eq!(
                    scalar_proof_plan(facts, nt(dims), operands).unwrap(),
                    None,
                    "{dims:?}"
                );
            }
            // Outputs 16 past a multiple of 32, an accumulating epilogue, a
            // bias, a scale or an unaligned operand keep the slim tile.
            let slim = ScalarDispatchPlan::NtFinal { slim: true };
            assert_eq!(
                scalar_launch_plan(facts, nt((4096, 128, 784)), operands).unwrap(),
                slim
            );
            for rejected in [
                F32TriadOperands {
                    beta: 1.0,
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
                    b: operands.b + 4,
                    ..operands
                },
            ] {
                assert_eq!(
                    scalar_launch_plan(facts, nt((4096, 128, 796)), rejected).unwrap(),
                    slim
                );
            }
        }
    }

    #[test]
    fn tn_narrow_split_m_is_the_shape_rule_on_every_board() {
        // The narrow band cuts its batch like the wide split-M, over the
        // narrow kernel's 64 x 32 tiles, whatever the board or operands.
        for (dims, expected) in [
            ((1024, 47, 17), Some((16, 64))),
            ((1024, 128, 25), Some((16, 64))),
            ((4096, 64, 64), Some((32, 128))),
            ((1024, 384, 101), Some((96, 11))),
            ((4096, 256, 101), Some((240, 18))),
            ((4111, 257, 127), Some((288, 15))),
            ((256, 32, 2), Some((16, 16))),
            ((255, 32, 2), None),
            ((1024, 64 * 72, 127), None),
        ] {
            let want = match expected {
                Some((m_chunk, chunks)) => ScalarDispatchPlan::TnNarrowSplitM { m_chunk, chunks },
                None => ScalarDispatchPlan::TnNarrow,
            };
            assert_eq!(plan(ResolvedGemmOp::Tn, dims).unwrap(), want, "{dims:?}");
            let mut other_board = tn_admission_facts();
            other_board.compute_capability = (8, 9);
            assert_eq!(
                admitted_tn_plan(
                    other_board,
                    tn_admission_request(dims),
                    tn_admission_operands()
                ),
                want,
                "{dims:?}"
            );
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
    fn nn_inference_copyplan_selector_admits_only_the_exact_measured_cells() {
        let operands = nn_qualified_operands();
        let cells = [
            (2_048, 768, 3_072),
            (2_048, 1_536, 768),
            (4_621, 384, 1_928),
            (4_096, 3_072, 1_536),
        ];
        for cohort in 0..INFERENCE_COPYPLAN_EVIDENCE_COHORTS.len() {
            let facts = inference_copyplan_facts(cohort);
            for dims in cells {
                let plan = scalar_launch_plan(facts, nn_qualified_request(dims), operands).unwrap();
                assert_eq!(
                    format!("{plan:?}"),
                    "NnSm89InferenceCopyPlanQualified",
                    "cohort {cohort} cell {dims:?}"
                );
            }
        }
    }

    #[test]
    fn nn_inference_copyplan_selector_fails_closed_to_the_prior_plan() {
        const SELECTED: &str = "NnSm89InferenceCopyPlanQualified";
        let facts = inference_copyplan_facts(2);
        let operands = nn_qualified_operands();
        let request = nn_qualified_request((2_048, 1_536, 768));
        let fallback = scalar_dispatch_plan(request).unwrap();
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
        mutation.inference_copyplan_loaded = false;
        mutations.push(mutation);
        mutation = facts;
        mutation.inference_compiler.nvrtc_version = (13, 1);
        mutations.push(mutation);
        mutation = facts;
        mutation.inference_compiler.source_digest[0] ^= 1;
        mutations.push(mutation);
        mutation = facts;
        mutation.inference_compiler.invocation_digest[0] ^= 1;
        mutations.push(mutation);
        mutation = facts;
        mutation.inference_compiler.header_manifest_digest[0] ^= 1;
        mutations.push(mutation);
        mutation = facts;
        mutation.inference_compiler.nvrtc_library_domain[0] ^= 1;
        mutations.push(mutation);
        mutation = facts;
        mutation.inference_artifact.compile_key[0] ^= 1;
        mutations.push(mutation);
        mutation = facts;
        mutation.inference_artifact.artifact_digest[0] ^= 1;
        mutations.push(mutation);
        mutation = facts;
        mutation.inference_artifact.module_kind = ModuleKind::TriadScalar;
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
    fn other_sm80_boards_reach_the_inference_copyplan_routes_through_the_proof_tier() {
        let ada = inference_copyplan_facts(2);
        let operands = nn_qualified_operands();
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nt,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, (2_048, 1_536, 768)),
        };
        let selected = ScalarDispatchPlan::NtD768OutSm89InferenceCopyPlanQualified;
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
            inference_copyplan_loaded: false,
            ..ada
        };
        assert_eq!(
            scalar_proof_plan(unloaded, request, operands).unwrap(),
            None
        );
    }

    #[test]
    fn nt_d768_out_inference_copyplan_selector_is_one_exact_measured_cell() {
        let facts = inference_copyplan_facts(2);
        let operands = nn_qualified_operands();
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nt,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, (2_048, 1_536, 768)),
        };
        let selected = ScalarDispatchPlan::NtD768OutSm89InferenceCopyPlanQualified;
        assert_eq!(
            scalar_dispatch_plan(request).unwrap(),
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
        for cohort in 0..INFERENCE_COPYPLAN_EVIDENCE_COHORTS.len() {
            assert_eq!(
                scalar_launch_plan(inference_copyplan_facts(cohort), request, operands).unwrap(),
                selected,
                "cohort {cohort}"
            );
        }
        let mut absent = facts;
        absent.inference_copyplan_loaded = false;
        assert_eq!(
            scalar_launch_plan(absent, request, operands).unwrap(),
            ScalarDispatchPlan::NtFinal { slim: false }
        );
    }

    #[test]
    fn nt_inference_copyplan_sibling_selector_admits_only_all_three_measured_cohorts() {
        assert_eq!(
            NT_INFERENCE_COPYPLAN_COMPOSED_EVIDENCE_COHORTS,
            NT_INFERENCE_COPYPLAN_COMPOSED_QUALIFICATION_CANDIDATES,
            "admission must contain exactly the three measured composed candidates"
        );
    }

    #[test]
    fn triad_retained_identity_inference_scalar_pairs_match_the_three_measured_cohorts() {
        let expected = measured_composed_identities();
        assert_eq!(
            NT_INFERENCE_COPYPLAN_COMPOSED_QUALIFICATION_CANDIDATES.len(),
            3,
            "the three measured pairs must all be staged"
        );
        assert_eq!(INFERENCE_COPYPLAN_EVIDENCE_COHORTS.len(), 3);

        for (index, identity) in expected.iter().copied().enumerate() {
            let facts = facts_for_measured_pair(identity);
            assert!(
                qualified_inference_copyplan_environment(facts),
                "Inference cohort {index}"
            );
            assert!(
                qualified_nt_inference_copyplan_sibling_environment(facts),
                "scalar+Inference cohort {index}"
            );
            assert!(
                qualified_scalar_sm89_tn_environment(facts),
                "scalar projection for cohort {index}"
            );

            let mut changed = facts;
            changed.scalar_compiler.nvrtc_version.1 += 1;
            changed.inference_compiler.nvrtc_version.1 += 1;
            assert!(!qualified_nt_inference_copyplan_sibling_environment(
                changed
            ));
            let mut changed = facts;
            changed.inference_compiler.source_digest[0] ^= 1;
            assert!(!qualified_inference_copyplan_environment(changed));
            assert!(!qualified_nt_inference_copyplan_sibling_environment(
                changed
            ));
            let mut changed = facts;
            changed.scalar_artifact.compile_key[0] ^= 1;
            assert!(!qualified_nt_inference_copyplan_sibling_environment(
                changed
            ));
            let mut changed = facts;
            changed.inference_artifact.artifact_digest[0] ^= 1;
            assert!(!qualified_inference_copyplan_environment(changed));
            assert!(!qualified_nt_inference_copyplan_sibling_environment(
                changed
            ));

            let other_toolkit = expected[(index + 1) % expected.len()].scalar;
            let mut crossed = facts;
            crossed.scalar_artifact.compile_key = other_toolkit.compile_key;
            crossed.scalar_artifact.artifact_digest = other_toolkit.artifact_digest;
            crossed.scalar_compiler.source_digest = other_toolkit.source_digest;
            crossed.scalar_compiler.invocation_digest = other_toolkit.compile_key;
            crossed.scalar_compiler.header_manifest_digest = other_toolkit.header_manifest_digest;
            crossed.scalar_compiler.nvrtc_version = other_toolkit.nvrtc_version;
            crossed.scalar_compiler.nvrtc_library_domain = other_toolkit.nvrtc_library_domain;
            assert!(qualified_inference_copyplan_environment(crossed));
            assert!(!qualified_nt_inference_copyplan_sibling_environment(
                crossed
            ));
        }

        assert_eq!(
            NT_INFERENCE_COPYPLAN_COMPOSED_QUALIFICATION_CANDIDATES,
            expected.as_slice()
        );
        assert_eq!(
            NT_INFERENCE_COPYPLAN_COMPOSED_QUALIFICATION_CANDIDATES
                .iter()
                .map(|identity| identity.scalar.compile_key)
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            3
        );
        assert_eq!(
            TN_M16N16_SPLITM16_SM89_BINDINGS,
            [expected[0].scalar, expected[1].scalar, expected[2].scalar]
        );
    }

    #[test]
    fn nt_inference_copyplan_sibling_selector_admits_both_retained_cells() {
        let operands = nn_qualified_operands();
        let cells = [
            (
                (2_048, 768, 3_072),
                ScalarDispatchPlan::NtD768InSm89InferenceCopyPlanQualified,
                ScalarDispatchPlan::NtFinal { slim: false },
            ),
            (
                (4_621, 384, 1_928),
                ScalarDispatchPlan::NtPrismSm89InferenceCopyPlanQualified,
                ScalarDispatchPlan::NtFinal { slim: true },
            ),
        ];
        for (cohort, candidate) in NT_INFERENCE_COPYPLAN_COMPOSED_QUALIFICATION_CANDIDATES
            .iter()
            .enumerate()
        {
            let facts = nt_inference_copyplan_sibling_facts(cohort);
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
                    scalar_dispatch_plan(request).unwrap(),
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
    fn nt_inference_copyplan_sibling_admission_does_not_change_existing_nn_or_d768_out_routes() {
        let operands = nn_qualified_operands();
        for cohort in 0..NT_INFERENCE_COPYPLAN_COMPOSED_QUALIFICATION_CANDIDATES.len() {
            let facts = nt_inference_copyplan_sibling_facts(cohort);
            let nn = F32TriadRequest {
                op: ResolvedGemmOp::Nn,
                shape: F32TriadShape::contiguous(ResolvedGemmOp::Nn, (2_048, 768, 3_072)),
            };
            assert_eq!(
                scalar_launch_plan(facts, nn, operands).unwrap(),
                ScalarDispatchPlan::NnSm89InferenceCopyPlanQualified,
                "cohort {cohort} NN"
            );
            let d768_out = F32TriadRequest {
                op: ResolvedGemmOp::Nt,
                shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, (2_048, 1_536, 768)),
            };
            assert_eq!(
                scalar_launch_plan(facts, d768_out, operands).unwrap(),
                ScalarDispatchPlan::NtD768OutSm89InferenceCopyPlanQualified,
                "cohort {cohort} d768-out"
            );
        }
    }

    #[test]
    fn triad_retained_scalar_ada_tn_m16n16_is_exact_and_inference_holder_independent() {
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Tn,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Tn, (256, 512, 384)),
        };
        let operands = F32TriadOperands {
            beta: 1.0,
            ..nn_qualified_operands()
        };
        for cohort in 0..3 {
            let mut facts = nt_inference_copyplan_sibling_facts(cohort);
            facts.inference_copyplan_loaded = false;
            facts.inference_artifact.compile_key = [0; 32];
            facts.inference_artifact.artifact_digest = [0; 32];
            assert_eq!(
                scalar_launch_plan(facts, request, operands).unwrap(),
                ScalarDispatchPlan::TnM16N16SplitM16Qualified,
                "Ada scalar cohort {cohort}"
            );
        }

        let fallback = scalar_dispatch_plan(request).unwrap();
        assert_eq!(
            fallback,
            ScalarDispatchPlan::TnSplitM {
                m_chunk: 16,
                chunks: 16,
            }
        );
        let facts = nt_inference_copyplan_sibling_facts(2);
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
        let fallback = scalar_dispatch_plan(request).unwrap();
        assert_eq!(fallback, ScalarDispatchPlan::NtFinal { slim: false });
        for cohort in 0..NT_INFERENCE_COPYPLAN_COMPOSED_QUALIFICATION_CANDIDATES.len() {
            assert_eq!(
                scalar_launch_plan(
                    nt_inference_copyplan_sibling_facts(cohort),
                    request,
                    operands,
                )
                .unwrap(),
                ScalarDispatchPlan::NtLargeDeepSm89InferenceCopyPlanQualified,
                "composed cohort {cohort}"
            );
        }

        let facts = nt_inference_copyplan_sibling_facts(2);
        let other = nt_inference_copyplan_sibling_facts(1);
        let mut pair_mismatch = facts;
        pair_mismatch.inference_artifact = other.inference_artifact;
        pair_mismatch.inference_compiler = other.inference_compiler;
        let mut holder_missing = facts;
        holder_missing.inference_copyplan_loaded = false;
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
                ScalarDispatchPlan::NtLargeDeepSm89InferenceCopyPlanQualified
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
                ScalarDispatchPlan::NtLargeDeepSm89InferenceCopyPlanQualified
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
                ScalarDispatchPlan::NtLargeDeepSm89InferenceCopyPlanQualified
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
                ScalarDispatchPlan::NtLargeDeepSm89InferenceCopyPlanQualified
            );
        }
    }

    #[test]
    fn nt_inference_copyplan_sibling_selector_fails_closed_on_contract_mutations() {
        let facts = nt_inference_copyplan_sibling_facts(2);
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
                    let general = scalar_launch_plan(facts, neighbor, operands).unwrap();
                    assert!(
                        general == scalar_dispatch_plan(neighbor).unwrap()
                            || general == ScalarDispatchPlan::NtSlimViaTranspose,
                        "shape neighbor {neighbor:?} must retain prior AUTO, got {general:?}"
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
                    scalar_dispatch_plan(mutated).unwrap(),
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
            mutation.inference_copyplan_loaded = false;
            environment_mutations.push(mutation);
            mutation = facts;
            mutation.inference_compiler.nvrtc_version = (13, 1);
            environment_mutations.push(mutation);
            mutation = facts;
            mutation.inference_compiler.source_digest[0] ^= 1;
            environment_mutations.push(mutation);
            mutation = facts;
            mutation.inference_compiler.invocation_digest[0] ^= 1;
            environment_mutations.push(mutation);
            mutation = facts;
            mutation.inference_compiler.header_manifest_digest[0] ^= 1;
            environment_mutations.push(mutation);
            mutation = facts;
            mutation.inference_compiler.nvrtc_library_domain[0] ^= 1;
            environment_mutations.push(mutation);
            mutation = facts;
            mutation.inference_artifact.compile_key[0] ^= 1;
            environment_mutations.push(mutation);
            mutation = facts;
            mutation.inference_artifact.artifact_digest[0] ^= 1;
            environment_mutations.push(mutation);
            mutation = facts;
            mutation.inference_artifact.module_kind = ModuleKind::TriadScalar;
            environment_mutations.push(mutation);
            mutation = facts;
            mutation.inference_artifact.artifact_kind = ArtifactKind::Cubin;
            environment_mutations.push(mutation);
            mutation = facts;
            mutation.inference_compiler.target = CudaTarget::new("compute_89").unwrap();
            environment_mutations.push(mutation);
            mutation = facts;
            mutation.inference_compiler.nvrtc_library_known = false;
            environment_mutations.push(mutation);
            mutation = facts;
            mutation.inference_compiler.output_kind = ArtifactKind::Cubin;
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
            // Without the cohort the cell keeps the plan every board serves:
            // the slim tile, or its chain over the transposed weights.
            let general = if fallback == (ScalarDispatchPlan::NtFinal { slim: true }) {
                ScalarDispatchPlan::NtSlimViaTranspose
            } else {
                fallback
            };
            for mutation in environment_mutations {
                assert_eq!(
                    scalar_launch_plan(mutation, request, operands).unwrap(),
                    general,
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
            scalar_dispatch_plan(request).unwrap(),
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
            scalar_dispatch_plan(request).unwrap(),
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
    fn nn_slim_split_admission_counts_the_shapes_own_tiles() {
        assert_eq!(
            plan(ResolvedGemmOp::Nn, (2560, 384, 512)).unwrap(),
            ScalarDispatchPlan::NnSplitKSlim { chunks: 6 }
        );
    }

    #[test]
    fn shape_only_split_policy_keeps_the_frozen_scalar_table() {
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
            assert_eq!(plan(op, dims).unwrap(), expected, "{op:?} {dims:?}");
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
    // The stream-K dW schedule serves a deep reduction over few enough
    // 64x64 tiles (the policy's measured bounds), and only when the half
    // policy permits its fixed-order fold; a request that stays on the
    // tiled contract never sees it. Stream-K or the tiled kernel is a
    // choice of reduction order, so the rule reads the shape alone and every
    // card takes the same one. It runs on every board whose portable module
    // composes it; the CC 12.x boards keep their own stream-K kernel.
    if op == super::super::kernel_identity::PolicyOp::Dw
        && half_policy == HalfTriadPolicy::AllowStreamKFixedOrder
    {
        let (batch, n_in, n_out) = dims;
        let policy = super::super::kernel_identity::Sm80TcPolicy::current();
        let tiles64 = tc_grid_ctas(n_in, n_out, 64, 64)?;
        let slabs = u64::try_from(batch).ok()?.div_ceil(64);
        let max_tiles = if slabs >= policy.stream_k_deep_reduction_slabs {
            policy.stream_k_deep_max_tiles
        } else {
            policy.stream_k_shallow_max_tiles
        };
        if tiles64 > 0 && slabs >= policy.stream_k_min_reduction_slabs && tiles64 <= max_tiles {
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
    use crate::gpu::context::HalfTriadPolicy;
    use crate::gpu::kernel_identity::{
        FramedSha256, PolicyOp, digest_hex, gemm_dispatch_policy_digest,
    };

    const RTX_6000_ADA_SMS: u32 = 142;

    #[test]
    fn stream_k_dw_needs_the_half_permission_a_deep_reduction_and_few_tiles() {
        let pick = |dims, policy, cc| {
            let picks = [24, 82, 108, 128, 132, 142, 170, 188].map(|multiprocessors| {
                tc_pick_tile_backward_for_device(PolicyOp::Dw, dims, multiprocessors, cc, policy)
                    == Some(TcTile::Tile64StreamK)
            });
            // Stream-K or the tiled kernel is a reduction order: every card
            // takes the same one.
            assert!(
                picks.iter().all(|&picked| picked == picks[0]),
                "{dims:?} {policy:?} {cc:?}: {picks:?}"
            );
            tc_pick_tile_backward_for_device(PolicyOp::Dw, dims, RTX_6000_ADA_SMS, cc, policy)
        };
        let tiled = HalfTriadPolicy::TiledParity;
        let stream_k = HalfTriadPolicy::AllowStreamKFixedOrder;
        // Stream-K serves a deep reduction over few tiles: every grid of up to
        // 48 tiles from 2048 rows (the thin and few-tile gradients ran it at
        // 0.29-0.88 of the tiled kernel), and up to 192 tiles from 8192 rows.
        for dims in [
            (2048, 16, 512),
            (4096, 64, 512),
            (2048, 384, 384),
            (10400, 384, 384),
            (10400, 768, 384),
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
        // More tiles over a shallower reduction, and any grid past 192
        // tiles, keep the tiled kernel, which the interleaved measurement
        // found as fast or faster there: prism_out_proj 72 tiles x 73 slabs,
        // prism_input_proj 96 x 73, rect_tall 96 x 64, prism_in_proj 186 x
        // 73, large 576 x 32, a filled 768-tile grid.
        for dims in [
            (4621, 768, 384),
            (4621, 1024, 384),
            (4096, 512, 768),
            (4621, 384, 1928),
            (2048, 3072, 768),
            (10400, 2048, 1536),
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
        // A reduction of 1024 rows or fewer keeps the tiled pick under both
        // policies: d128_out_proj 8 tiles x 16 slabs, d128_in_proj 16 x 16,
        // underfill 48 x 4, split_candidate 256 x 2, a shallow 512-row batch.
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
        let prefers_scalar = |dims| tc_half_policy_prefers_scalar_forward((8, 9), dims).unwrap();

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
                !tc_half_policy_prefers_scalar_forward(compute_capability, (128, 8192, 128))
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
    fn selector_policy_identity_is_pinned_and_the_same_on_every_card() {
        let actual = digest_hex(&gemm_dispatch_policy_digest());
        assert_eq!(
            actual, "dc4bdadc015028be186c7b3f35923cd701b174db9082402c9978e8cd38fad0a8",
            "policy identity moved; repin it from the live digest"
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
    use crate::gpu::context::HalfTriadPolicy;
    use crate::gpu::dtype::WeightDtype;
    use crate::gpu::gemm_bi_triad::contract::{
        Sm120Bk, Sm120ForcedRoute, Sm120LaunchOperands, Sm120Op, Sm120PhysicalRoute, Sm120Shape,
        Sm120Stages, Sm120Tile,
    };
    use crate::gpu::kernel_identity::{CudaTarget, DeviceCaps};

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
            nearest_sm120_cell(SM120_AUTO_CELLS_CC120, Sm120Op::Nn, WeightDtype::Bf16, tall),
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
                tall_tn
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
                measured
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
        )
        .expect("the narrow projection is inside the band");
        assert_eq!(picked.tile, Sm120Tile::M64N64);
        // A tiny square is more than a factor of eight from every cell on at
        // least one axis and gets no tile: the portable tiles serve it.
        let tiny = Sm120Shape::contiguous(Sm120Op::Nn, (64, 64, 64));
        assert_eq!(
            nearest_sm120_cell(SM120_AUTO_CELLS_CC120, Sm120Op::Nn, WeightDtype::Bf16, tiny),
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
    fn the_streamk_neighbour_bound_is_the_largest_measured_winner() {
        // 384 x 1928 in 64 x 64 tiles: a shape of the table, not a card's
        // multiprocessor count.
        assert_eq!(super::SM120_STREAMK_NEIGHBOUR_MAX_TILES, 186.0);
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
            super::nearest_sm120_cell(&cells, Sm120Op::Tn, WeightDtype::Bf16, near)
                .map(|physical| physical.schedule),
            Some(Sm120Schedule::StreamK)
        );
        // Inside the neighbour band but with a grid that covers the device
        // several times over: the stream-K cell declines, leaving the shape
        // to the tiled table.
        let wide = Sm120Shape::contiguous(Sm120Op::Tn, (10400, 2048, 2048));
        assert_eq!(
            super::nearest_sm120_cell(&cells, Sm120Op::Tn, WeightDtype::Bf16, wide),
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
        INFERENCE_COPYPLAN_EVIDENCE_COHORTS, INFERENCE_COPYPLAN_SOURCE_DIGEST,
        SM89_EXACT_F32_D128_EVIDENCE_COHORTS, SM89_EXACT_F32_EVIDENCE_COHORTS,
        SM89_FINALIST_TF32_ARCHITECTURE, SM89_FINALIST_TF32_EVIDENCE_COHORTS,
        SM89_JOINT_TF32_ARCHITECTURE, SM89_JOINT_TF32_EVIDENCE_COHORTS, SM89_TF32_ARCHITECTURE,
        SM89_TF32_EVIDENCE_CELLS, SM89_TF32_EVIDENCE_COHORTS, SM120_TF32_ARCHITECTURE,
        SM120_TF32_EVIDENCE_CELLS, SM120_TF32_EVIDENCE_COHORTS, SM120_TF32_PORTABLE_ARCHITECTURE,
        Tf32AutoEvidenceCohort, Tf32AutoOperandGate, matching_tf32_cohort,
        measured_tf32_route_with_operands, resolve_f32_triad_auto,
        resolve_f32_triad_auto_with_operands, resolve_tf32_forced, rna_batch_invariant_route,
        rna_portable_ladder_route,
    };
    use crate::gpu::context::F32TriadPolicy;
    use crate::gpu::gemm_bi_triad::contract::{
        F32_TF32_TUNING_REVISION, F32TriadAvailability, F32TriadOperands, F32TriadRequest,
        F32TriadSelection, F32TriadShape, Sm120FmaExclusions, Sm120FmaRoute, Sm120FmaTile,
        TF32_NT_SPLITK8_S3_SPEC, Tf32PhysicalRoute, Tf32PortableRoute, Tf32PortableStages,
        Tf32PortableTile, Tf32QualifiedModule, Tf32Sm90aRoute, Tf32Sm100Route, Tf32Sm120Route,
        Tf32Sm120Stages, Tf32Sm120Tile,
    };
    use crate::gpu::gemm_bi_triad::{
        Sm90aWarpgroupSchedule, Sm100Schedule, Sm100Stages, Sm100Tile,
    };
    use crate::gpu::kernel_identity::{
        ArtifactIdentity, ArtifactKind, COMPILER_REVISION, COMPOSER_REVISION, CompilerIdentity,
        CudaTarget, DeviceCaps, DeviceIdentity, DriverIdentity, ModuleKind, NUMERIC_ABI_REVISION,
        ResolvedGemmOp, SCHEDULE_REVISION, TUNING_TABLE_REVISION,
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
        let portable = qualified_module_for_auto_identity(SM89_TF32_ARCHITECTURE);
        F32TriadAvailability {
            portable: Some(portable),
            specialized: None,
            finalist: None,
            joint: None,
            rna: None,
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
    fn sm89_joint_cohort_is_one_table_for_the_architecture() {
        let [cohort] = SM89_JOINT_TF32_EVIDENCE_COHORTS else {
            panic!("SM89 joint must have exactly one TF32 cohort");
        };
        assert_eq!(cohort.identity, SM89_JOINT_TF32_ARCHITECTURE);
        assert_eq!(cohort.portable, None);
        assert_eq!(cohort.cells.len(), 10);
        assert_eq!(
            cohort.cells[2].route,
            Tf32PhysicalRoute::Sm89TnPreRnaM64N96S2
        );
        assert_eq!(cohort.cells[3].route, Tf32PhysicalRoute::Sm89NnDirectN96);
    }

    fn sm89_joint_availability(
        joint: super::Tf32EvidenceArchitecture,
        portable: Option<super::Tf32EvidenceArchitecture>,
    ) -> F32TriadAvailability {
        F32TriadAvailability {
            portable: portable.map(qualified_module_for_auto_identity),
            specialized: None,
            finalist: None,
            joint: Some(qualified_module_for_auto_identity(joint)),
            rna: None,
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
                    "failed to select {op:?}/{rows}/{columns}/{reduction}",
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
    fn sm89_joint_evidence_serves_every_sm89_card_and_no_other_module() {
        let [cohort] = SM89_JOINT_TF32_EVIDENCE_COHORTS else {
            panic!("SM89 joint must have exactly one TF32 cohort");
        };
        let exact = qualified_module_for_auto_identity(cohort.identity);
        assert_eq!(
            matching_tf32_cohort(exact, SM89_JOINT_TF32_EVIDENCE_COHORTS),
            Some(cohort),
        );
        for (field, mutate) in sm120_identity_mutations().into_iter().enumerate() {
            let mut drifted = exact;
            mutate(&mut drifted);
            if drifted == exact {
                continue;
            }
            assert!(
                matching_tf32_cohort(drifted, SM89_JOINT_TF32_EVIDENCE_COHORTS).is_none(),
                "architecture mutation {field} was admitted",
            );
        }
        for (field, mutate) in provenance_mutations().into_iter().enumerate() {
            let mut card = exact;
            mutate(&mut card);
            assert_eq!(
                matching_tf32_cohort(card, SM89_JOINT_TF32_EVIDENCE_COHORTS),
                Some(cohort),
                "another SM89 card (provenance mutation {field}) lost the evidence",
            );
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

    fn sm120_availability_for(identity: super::Tf32EvidenceArchitecture) -> F32TriadAvailability {
        let specialized = qualified_module_for_auto_identity(identity);
        F32TriadAvailability {
            portable: None,
            specialized: Some(specialized),
            finalist: None,
            joint: None,
            rna: None,
            multiprocessors: 170,
        }
    }

    /// The targets a module of the evidence's architecture compiles for.
    fn architecture_targets(
        identity: super::Tf32EvidenceArchitecture,
    ) -> (&'static str, &'static str) {
        match identity.compute_capability {
            (12, _) => ("compute_120", "sm_120"),
            (8, 9) => ("sm_89", "sm_89"),
            other => panic!("no test targets for compute capability {other:?}"),
        }
    }

    /// The module a card of the evidence's architecture binds. The test
    /// module's SM count, toolkit and artifact bytes are its own, never the
    /// measuring card's: evidence serves the whole architecture.
    fn qualified_module_for_auto_identity(
        identity: super::Tf32EvidenceArchitecture,
    ) -> Tf32QualifiedModule {
        let (target, device_target) = architecture_targets(identity);
        qualified_module(
            identity.module_kind,
            target,
            device_target,
            identity.compute_capability,
            identity.tensor_map_access,
            identity.optin_shared_bytes,
        )
    }

    /// The SM120 architecture's one cohort.
    fn sm120_cohort() -> Tf32AutoEvidenceCohort {
        let [cohort] = SM120_TF32_EVIDENCE_COHORTS else {
            panic!("SM120 must have exactly one TF32 cohort")
        };
        *cohort
    }

    /// Changes that move a module out of the evidence's architecture or out
    /// of what this tree builds: each must keep the evidence away.
    fn sm120_identity_mutations() -> [fn(&mut Tf32QualifiedModule); 15] {
        [
            |module| module.module_kind = ModuleKind::Inference,
            |module| module.target = CudaTarget::new("compute_121").unwrap(),
            |module| module.artifact.module_kind = ModuleKind::Inference,
            |module| module.artifact.artifact_kind = ArtifactKind::Cubin,
            |module| module.compiler.target = CudaTarget::new("compute_121").unwrap(),
            |module| module.compiler.nvrtc_library_known = false,
            |module| module.compiler.output_kind = ArtifactKind::Cubin,
            |module| module.compiler.composer_revision ^= 1,
            |module| module.compiler.compiler_revision ^= 1,
            |module| module.compiler.numeric_abi_revision ^= 1,
            |module| module.compiler.schedule_revision ^= 1,
            |module| module.device.compute_capability = (8, 6),
            |module| module.device_caps.accepted_target = None,
            |module| module.device_caps.optin_shared_bytes -= 1,
            |module| module.device_caps.tensor_map_access = false,
        ]
    }

    /// Changes that stay inside the evidence's architecture: another card's
    /// SM count, another toolkit, another driver, the bytes another toolkit
    /// produces. None of them may keep the evidence away.
    fn provenance_mutations() -> [fn(&mut Tf32QualifiedModule); 10] {
        [
            |module| {
                module.artifact.compile_key[0] ^= 1;
                module.compiler.invocation_digest[0] ^= 1;
            },
            |module| module.artifact.artifact_digest[0] ^= 1,
            |module| module.compiler.source_digest[0] ^= 1,
            |module| module.compiler.header_manifest_digest[0] ^= 1,
            |module| {
                module.compiler.nvrtc_version.1 ^= 1;
                module.device_caps.nvrtc_version.1 ^= 1;
            },
            |module| module.compiler.nvrtc_library_domain[0] ^= 1,
            |module| module.device.multiprocessor_count = 188,
            |module| module.device.multiprocessor_count = 84,
            |module| module.device.driver.build_digest[0] ^= 1,
            |module| module.device_caps.optin_shared_bytes += 1024,
        ]
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
    fn sm120_exact_splits_only_where_the_portable_plan_splits() {
        // An RTX 5090 writes an exact product's bits only if it cuts the
        // reduction where every other architecture does: the pieces of the
        // portable plan, nowhere else.
        let identity = sm120_cohort().identity;
        let availability = sm120_availability_for(identity);
        let mut served = [0_usize; 2];
        for op in [ResolvedGemmOp::Nn, ResolvedGemmOp::Tn, ResolvedGemmOp::Nt] {
            for m in [256, 1_024, 2_048, 4_621, 10_400] {
                for k in [128, 384, 768, 1_024, 1_536, 1_928, 3_072] {
                    for n in [128, 384, 768, 1_536, 1_928, 3_072] {
                        let request = F32TriadRequest {
                            op,
                            shape: F32TriadShape::contiguous(op, (m, k, n)),
                        };
                        let operands = F32TriadOperands {
                            output: 0x1000,
                            a: 0x2000,
                            b: 0x3000,
                            bias: None,
                            alpha: 1.0,
                            beta: if op == ResolvedGemmOp::Tn { 1.0 } else { 0.0 },
                        };
                        let Some(route) =
                            super::sm120_fma_exact_route(request, operands, availability)
                        else {
                            continue;
                        };
                        let plan = super::scalar_dispatch_plan(request).unwrap();
                        let cut = super::sm120_fma_portable_cut(request)
                            .expect("a served shape has a portable cut");
                        assert_eq!(route.splits, cut.splits, "{op:?} {m}x{k}x{n} {plan:?}");
                        assert_eq!(
                            route.splits > 1,
                            plan.needs_split_scratch(),
                            "{op:?} {m}x{k}x{n} {plan:?}"
                        );
                        let launch = crate::gpu::gemm_bi_triad::launch::sm120_fma_launch_plan(
                            request, route,
                        )
                        .unwrap();
                        assert_eq!(launch.tiles_per_split, cut.piece_tiles);
                        served[usize::from(route.splits > 1)] += 1;
                    }
                }
            }
        }
        assert!(
            served[0] > 0 && served[1] > 0,
            "the sweep reaches both chains and splits: {served:?}"
        );
    }

    #[test]
    fn sm120_exact_folds_short_pieces_and_tails_in_one_cta() {
        // A thin cut's two-tile pieces and a cut with a tail fold inside one
        // CTA per tile, which half the card's tiles admit; the long pieces
        // of a split-M weight gradient get a CTA each.
        let identity = sm120_cohort().identity;
        let availability = sm120_availability_for(identity);
        let route = |op, dims| {
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
            let route = super::sm120_fma_exact_route(request, operands, availability)?;
            let launch =
                crate::gpu::gemm_bi_triad::launch::sm120_fma_launch_plan(request, route).ok()?;
            Some((route, launch))
        };
        let (thin, launch) =
            route(ResolvedGemmOp::Nn, (1_024, 1_024, 1_024)).expect("a thin cut over 128 tiles");
        assert!(thin.splits > 1 && thin.in_cta, "{thin:?}");
        assert_eq!(launch.tiles_per_split, 2, "{launch:?}");
        assert_eq!(launch.units as usize, launch.tiles, "{launch:?}");
        let tail_request = F32TriadRequest {
            op: ResolvedGemmOp::Nt,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, (1_024, 768, 1_000)),
        };
        assert!(
            matches!(
                super::scalar_dispatch_plan(tail_request),
                Ok(super::ScalarDispatchPlan::NtSplitKMain { n_tail: 8, .. })
            ),
            "NT 1024x768x1000 runs its last eight inputs after the fold"
        );
        let (tail, _) = route(ResolvedGemmOp::Nt, (1_024, 768, 1_000)).expect("a cut with a tail");
        assert!(tail.splits > 1 && tail.in_cta, "{tail:?}");
        let (split_m, _) =
            route(ResolvedGemmOp::Tn, (8_192, 1_024, 1_536)).expect("a split-M gradient");
        assert!(split_m.splits > 1 && !split_m.in_cta, "{split_m:?}");
    }

    #[test]
    fn sm120_exact_keeps_a_column_tail_out_of_the_fold() {
        // NT 1024x833x384 folds 32-input pieces over its first 832 output
        // columns, and the portable route writes the last column as one
        // chain in a kernel of its own: the CTA that folds its own pieces
        // keeps that column's chain apart, and only for the unscaled,
        // overwriting product that portable route writes.
        let identity = sm120_cohort().identity;
        let availability = sm120_availability_for(identity);
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nt,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, (1_024, 833, 384)),
        };
        assert_eq!(
            super::scalar_dispatch_plan(request),
            Ok(super::ScalarDispatchPlan::NtSplitKTail {
                k_main: 832,
                k_tail: 1
            })
        );
        let cut = super::sm120_fma_portable_cut(request).expect("a cut with a column tail");
        assert_eq!(
            cut,
            super::Sm120FmaCut {
                splits: 12,
                piece_tiles: 2,
                tail: false,
                chain_from: Some(832),
            }
        );
        let operands = F32TriadOperands {
            output: 0x1000,
            a: 0x2000,
            b: 0x3000,
            bias: None,
            alpha: 1.0,
            beta: 0.0,
        };
        let route = super::sm120_fma_exact_route(request, operands, availability)
            .expect("112 tiles fold their own pieces");
        assert!(
            route.splits == 12 && route.in_cta && route.kvec,
            "{route:?}"
        );
        let launch =
            crate::gpu::gemm_bi_triad::launch::sm120_fma_launch_plan(request, route).unwrap();
        assert_eq!(launch.chain_from, Some(832), "{launch:?}");
        assert_eq!(launch.units as usize, launch.tiles, "{launch:?}");
        for (alpha, beta) in [(2.0, 0.0), (1.0, 1.0)] {
            assert_eq!(
                super::sm120_fma_exact_route(
                    request,
                    F32TriadOperands {
                        alpha,
                        beta,
                        ..operands
                    },
                    availability
                ),
                None,
                "alpha {alpha} beta {beta}"
            );
        }
        let dealt = Sm120FmaRoute {
            in_cta: false,
            ..route
        };
        assert!(
            crate::gpu::gemm_bi_triad::launch::sm120_fma_launch_plan(request, dealt).is_err(),
            "a CTA per piece cannot keep the tail's chains"
        );
        let scalar_b = Sm120FmaRoute {
            kvec: false,
            ..route
        };
        assert!(
            crate::gpu::gemm_bi_triad::launch::sm120_fma_launch_plan(request, scalar_b).is_err(),
            "only the k-vector NT kernels keep a column tail apart"
        );
    }

    #[test]
    fn sm120_exact_launch_refuses_a_cut_of_its_own() {
        // NT 2048x1536x768 is one chain in the portable plan; the two pieces
        // its measured arm once dealt are refused.
        let (request, _) = nt_d768_out_request_and_operands();
        let route = Sm120FmaRoute {
            tile: Sm120FmaTile::M64N128,
            kvec: false,
            splits: 2,
            in_cta: false,
        };
        let error = crate::gpu::gemm_bi_triad::launch::sm120_fma_launch_plan(request, route)
            .expect_err("a split the portable plan does not make");
        assert!(error.contains("portable plan's cut"), "{error}");
    }

    #[test]
    fn exact_policy_measured_cells_resolve_to_their_official_arms() {
        let identity = sm120_cohort().identity;
        let availability = sm120_availability_for(identity);
        for cell in super::SM120_FMA_MEASURED_CELLS {
            let request = F32TriadRequest {
                op: cell.op,
                shape: cell.shape,
            };
            let cut = super::sm120_fma_portable_cut(request)
                .expect("every measured cell has a portable cut the kernels write");
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
                F32TriadSelection::ExactSm120Fma(Sm120FmaRoute {
                    tile: cell.tile,
                    kvec: cell.kvec,
                    splits: cut.splits,
                    in_cta: cut.splits > 1
                        && (cut.tail
                            || cut.chain_from.is_some()
                            || cut.piece_tiles <= super::SM120_FMA_SHORT_PIECE_TILES),
                }),
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
    ) -> F32TriadAvailability {
        let identity = sm120_cohort().identity;
        let mut availability = sm120_availability_for(identity);
        let specialized = availability.specialized.as_mut().unwrap();
        specialized.sm120_fma_exclusions =
            Sm120FmaExclusions::from_routes(excluded).expect("literal exact-F32 routes");
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
            splits: 1,
            in_cta: false,
        };
        let generic = Sm120FmaRoute {
            tile: Sm120FmaTile::M128N64,
            kvec: true,
            splits: 1,
            in_cta: false,
        };
        let availability =
            exact_sm120_availability_with_exclusions(&[(ResolvedGemmOp::Nt, measured)]);
        let (request, operands) = nt_d768_out_request_and_operands();

        assert_eq!(
            resolve_f32_triad_auto_with_operands(
                F32TriadPolicy::ExactScalarFma,
                request,
                operands,
                availability
            )
            .unwrap(),
            F32TriadSelection::ExactSm120Fma(generic),
            "the exact policy must skip the excluded measured arm"
        );
        // Under the TF32 policy the architecture's TF32 cell for the shape
        // serves before the exact family is consulted.
        assert_eq!(
            resolve_f32_triad_auto_with_operands(
                F32TriadPolicy::AllowDeterministicTf32,
                request,
                operands,
                availability
            )
            .unwrap(),
            F32TriadSelection::Tf32(Tf32PhysicalRoute::Sm120TmaMmaTf32Rna(Tf32Sm120Route {
                tile: Tf32Sm120Tile::M64N128,
                stages: Tf32Sm120Stages::S2,
            })),
        );
    }

    #[test]
    fn sm120_exact_reachability_excluded_measured_and_generic_routes_use_scalar() {
        let measured = Sm120FmaRoute {
            tile: Sm120FmaTile::M64N128,
            kvec: false,
            splits: 2,
            in_cta: false,
        };
        let generic = Sm120FmaRoute {
            tile: Sm120FmaTile::M128N64,
            kvec: true,
            splits: 2,
            in_cta: false,
        };
        let availability = exact_sm120_availability_with_exclusions(&[
            (ResolvedGemmOp::Nt, measured),
            (ResolvedGemmOp::Nt, generic),
        ]);
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
            in_cta: false,
        };
        let availability =
            exact_sm120_availability_with_exclusions(&[(ResolvedGemmOp::Nt, excluded)]);
        for (op, dims, expected) in [
            (
                ResolvedGemmOp::Nn,
                (2_048, 1_536, 768),
                Sm120FmaRoute {
                    tile: Sm120FmaTile::M128N64,
                    kvec: false,
                    splits: 1,
                    in_cta: false,
                },
            ),
            (
                ResolvedGemmOp::Tn,
                (2_048, 1_536, 768),
                Sm120FmaRoute {
                    tile: Sm120FmaTile::M128N64,
                    kvec: false,
                    splits: 4,
                    in_cta: false,
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
            in_cta: false,
        };
        let availability = exact_sm120_availability_with_exclusions(&[(ResolvedGemmOp::Nt, exact)]);
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
    fn tf32_tn_underfill_qualification_uses_current_tuning_revision() {
        assert_eq!(TUNING_TABLE_REVISION, 47);
        assert_eq!(F32_TF32_TUNING_REVISION, 47);
    }

    #[test]
    fn sm120_tf32_tn_underfill_uses_the_requalified_portable_winner() {
        let cohort = sm120_cohort();
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
        let cohort = sm120_cohort();
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
        let twinned = || {
            let mut value = sm120_availability_for(SM120_TF32_ARCHITECTURE);
            value.portable = Some(qualified_module_for_auto_identity(
                SM120_TF32_PORTABLE_ARCHITECTURE,
            ));
            value
        };
        assert_eq!(
            resolve(
                F32TriadPolicy::AllowDeterministicTf32,
                request,
                operands,
                twinned(),
            ),
            F32TriadSelection::Tf32(route),
            "the architecture's portable twin",
        );
        // Another SM120 card binds a portable module of its own toolkit, SM
        // count and driver: the cohort's portable route serves it unchanged.
        for (field, mutate) in provenance_mutations().into_iter().enumerate() {
            let mut card = twinned();
            mutate(card.portable.as_mut().unwrap());
            assert_eq!(
                resolve(
                    F32TriadPolicy::AllowDeterministicTf32,
                    request,
                    operands,
                    card,
                ),
                F32TriadSelection::Tf32(route),
                "provenance mutation {field} of the portable twin lost the route",
            );
        }
        // A portable module of another architecture or another build is no
        // evidence for the cohort's route; the portable tier may still serve
        // the cell by design where the module's structure is sound.
        for mutate in sm120_identity_mutations() {
            let mut rejected = twinned();
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
                "mismatched portable twin selected {selection:?}",
            );
        }
    }

    #[test]
    fn sm120_tf32_live_nn_cell_is_exact_and_fail_closed() {
        let cohort = sm120_cohort();
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
        // Every other SM120 card takes the same route: another SM count,
        // another toolkit, another driver, other artifact bytes.
        for (field, mutate) in provenance_mutations().into_iter().enumerate() {
            let mut availability = sm120_availability_for(cohort.identity);
            mutate(availability.specialized.as_mut().unwrap());
            assert_eq!(
                resolve(
                    F32TriadPolicy::AllowDeterministicTf32,
                    request,
                    operands,
                    availability,
                ),
                F32TriadSelection::Tf32(route),
                "provenance mutation {field} lost the live NN route",
            );
        }
    }

    /// The SM120 architecture has one cohort: the tournament on one card,
    /// serving every card of the architecture, each shape measured once.
    #[test]
    fn sm120_has_one_cohort_for_every_card_of_the_architecture() {
        let cohort = sm120_cohort();
        assert_eq!(cohort.identity, SM120_TF32_ARCHITECTURE);
        assert_eq!(cohort.portable, Some(SM120_TF32_PORTABLE_ARCHITECTURE));
        assert_eq!(cohort.cells, SM120_TF32_EVIDENCE_CELLS);
        assert_eq!(cohort.cells.len(), 39);
        let mut keys = cohort
            .cells
            .iter()
            .map(|cell| {
                (
                    cell.op as u8,
                    cell.shape.output_rows,
                    cell.shape.output_columns,
                    cell.shape.reduction,
                )
            })
            .collect::<Vec<_>>();
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(keys.len(), cohort.cells.len(), "a shape is measured twice");
        assert_eq!(
            cohort
                .cells
                .iter()
                .filter(|cell| cell.route.module_kind() == ModuleKind::TriadSm80)
                .count(),
            6
        );
    }

    /// The exact-f32 Ada cohorts and the Inference copy-plan cohorts still pin
    /// the module source they were measured on; an edit to those sources
    /// would let their routes fall back without any test noticing. Hold them
    /// to the source this tree composes so such an edit fails here first.
    #[test]
    fn every_source_pinned_ada_cohort_matches_the_source_this_tree_composes() {
        let live = |kind| {
            super::super::modules::module_source_digest(kind, "sm_89")
                .unwrap_or_else(|error| panic!("compose {kind:?} for sm_89: {error}"))
        };
        let toolkits = [(12, 8), (13, 0), (13, 2)];
        assert_eq!(INFERENCE_COPYPLAN_EVIDENCE_COHORTS.len(), toolkits.len());
        for (cohort, nvrtc) in INFERENCE_COPYPLAN_EVIDENCE_COHORTS.iter().zip(toolkits) {
            assert_eq!(cohort.nvrtc_version, nvrtc);
            assert_eq!(cohort.source_digest, INFERENCE_COPYPLAN_SOURCE_DIGEST);
            // The module takes no state capacity: every capacity composes
            // the one source the cohort pins.
            for state_cap in [16, 32, 64, 256] {
                let digest = super::super::modules::module_source_digest_for_compile(
                    ModuleKind::Inference,
                    Some((8, 9)),
                    "sm_89",
                    state_cap,
                    nvrtc,
                )
                .unwrap_or_else(|error| {
                    panic!("compose Inference for CUDA {nvrtc:?} cap{state_cap}: {error}")
                });
                assert_eq!(
                    cohort.source_digest, digest,
                    "an Inference copy-plan cohort pins a pre-overlay or stale source"
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

    /// Every card of an architecture reaches its cohort, and nothing outside
    /// the architecture does: a 5090, a PRO 6000, a 5080 and a GB10 all take
    /// the SM120 cohort under any toolkit; an Ada card or a module this tree
    /// did not build takes none.
    #[test]
    fn tf32_cohorts_serve_their_whole_architecture_and_nothing_else() {
        for (cohorts, identity) in [
            (SM120_TF32_EVIDENCE_COHORTS, SM120_TF32_ARCHITECTURE),
            (SM89_TF32_EVIDENCE_COHORTS, SM89_TF32_ARCHITECTURE),
            (
                SM89_JOINT_TF32_EVIDENCE_COHORTS,
                SM89_JOINT_TF32_ARCHITECTURE,
            ),
            (
                SM89_FINALIST_TF32_EVIDENCE_COHORTS,
                SM89_FINALIST_TF32_ARCHITECTURE,
            ),
        ] {
            let [cohort] = cohorts else {
                panic!("{identity:?} must have exactly one cohort");
            };
            let exact = qualified_module_for_auto_identity(identity);
            assert_eq!(matching_tf32_cohort(exact, cohorts), Some(cohort));
            for multiprocessors in [24, 48, 84, 128, 142, 170, 188] {
                for nvrtc_version in [(12, 8), (13, 0), (13, 2), (13, 4)] {
                    let mut card = exact;
                    card.device.multiprocessor_count = multiprocessors;
                    card.compiler.nvrtc_version = nvrtc_version;
                    card.device_caps.nvrtc_version = nvrtc_version;
                    assert_eq!(
                        matching_tf32_cohort(card, cohorts),
                        Some(cohort),
                        "{identity:?} on {multiprocessors} SMs under NVRTC {nvrtc_version:?}",
                    );
                }
            }
            for (field, mutate) in sm120_identity_mutations().into_iter().enumerate() {
                let mut drifted = exact;
                mutate(&mut drifted);
                if drifted == exact {
                    continue;
                }
                assert!(
                    matching_tf32_cohort(drifted, cohorts).is_none(),
                    "{identity:?}: architecture mutation {field} was admitted",
                );
            }
        }
        let mut gb10 = qualified_module_for_auto_identity(SM120_TF32_ARCHITECTURE);
        gb10.device.compute_capability = (12, 1);
        gb10.device_caps.compute_capability = (12, 1);
        assert_eq!(
            matching_tf32_cohort(gb10, SM120_TF32_EVIDENCE_COHORTS),
            Some(&sm120_cohort())
        );
    }

    #[test]
    fn sm120_tf32_operand_aware_resolution_admits_only_archived_cells() {
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
            // Every SM120 card under any toolkit takes the measured route.
            let measured_route = resolve_f32_triad_auto_with_operands(
                F32TriadPolicy::AllowDeterministicTf32,
                measured,
                operands,
                exact_availability,
            )
            .unwrap();
            assert!(matches!(measured_route, F32TriadSelection::Tf32(_)));
            for version in [(12, 8), (13, 0), (13, 1), (13, 2), (13, 3)] {
                for multiprocessors in [84, 169, 170, 171, 188] {
                    let mut availability = exact_availability;
                    let module = availability.specialized.as_mut().unwrap();
                    module.compiler.nvrtc_version = version;
                    module.device_caps.nvrtc_version = version;
                    module.device.multiprocessor_count = multiprocessors;
                    availability.multiprocessors = multiprocessors;
                    assert_eq!(
                        resolve_f32_triad_auto_with_operands(
                            F32TriadPolicy::AllowDeterministicTf32,
                            measured,
                            operands,
                            availability,
                        )
                        .unwrap(),
                        measured_route,
                        "NVRTC {version:?} on {multiprocessors} SMs",
                    );
                }
            }
        }
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
                rna: None,
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
                rna: None,
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
                rna: None,
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
                rna: None,
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
            |module| module.module_kind = ModuleKind::Inference,
            |module| module.target = CudaTarget::new("sm_80").unwrap(),
            |module| module.artifact.module_kind = ModuleKind::Inference,
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
                    rna: None,
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
                    rna: None,
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
                    rna: None,
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
                    rna: None,
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
                    rna: None,
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
    fn sm89_joint_forced_routes_keep_their_cells_and_unqualified_auto_stays_closed() {
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
            // The two pre-rounded weight-gradient tiles and the 144 x 96
            // input-gradient tile mask every edge: the first two run any
            // contiguous gradient whose rows are whole 16-byte chunks, the
            // last any contiguous input gradient; every other joint route
            // stays on its cells.
            let pre_rounded = matches!(
                route,
                Tf32PhysicalRoute::Sm89TnPreRnaM64N64 | Tf32PhysicalRoute::Sm89TnPreRnaM64N96S2
            );
            for neighbor in neighbors {
                let shape = neighbor.shape;
                let contiguous =
                    shape == F32TriadShape::contiguous(neighbor.op, (shape.m, shape.k, shape.n));
                let runs_any_shape = contiguous
                    && ((pre_rounded && shape.n.is_multiple_of(4))
                        || route == Tf32PhysicalRoute::Sm89NtRnaM144N96S2);
                assert_eq!(
                    resolve_tf32_forced(neighbor, availability, route).is_ok(),
                    runs_any_shape,
                    "joint route {route:?} on neighboring request {neighbor:?}"
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

                // A module of another architecture or another build keeps
                // the evidence away; the tensor-map mutation leaves an Ada
                // module as it was and is skipped.
                for mutate in sm120_identity_mutations() {
                    let mut changed = availability;
                    mutate(changed.finalist.as_mut().unwrap());
                    if changed == availability {
                        continue;
                    }
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
    fn finalist_cohort_serves_every_sm89_finalist_module() {
        let [cohort] = SM89_FINALIST_TF32_EVIDENCE_COHORTS else {
            panic!("SM89 finalist must have exactly one TF32 cohort");
        };
        assert_eq!(cohort.identity, SM89_FINALIST_TF32_ARCHITECTURE);
        assert_eq!(
            cohort.tuning_revision,
            super::super::contract::SM89_FINALIST_TUNING_REVISION
        );
        let module = qualified_module_for_auto_identity(SM89_FINALIST_TF32_ARCHITECTURE);
        for (field, mutate) in provenance_mutations().into_iter().enumerate() {
            let mut card = module;
            mutate(&mut card);
            assert_eq!(
                matching_tf32_cohort(card, SM89_FINALIST_TF32_EVIDENCE_COHORTS),
                Some(cohort),
                "provenance mutation {field} lost the finalist cohort",
            );
        }
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
            rna: None,
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
            rna: None,
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
                rna: None,
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
            rna: None,
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
                        rna: None,
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
                    rna: None,
                    multiprocessors: 142,
                },
                illegal,
            )
            .is_err()
        );
    }

    const STRICT: F32TriadPolicy = F32TriadPolicy::RnaTf32BatchInvariant;

    /// The boards the strict policy is checked on: the Ada portable module
    /// alone, with the finalist, with each joint cohort, and the SM120 board
    /// with its portable twin.
    fn strict_boards() -> Vec<F32TriadAvailability> {
        let mut boards = vec![sm89_availability(), sm89_finalist_availability()];
        boards.extend(
            SM89_JOINT_TF32_EVIDENCE_COHORTS
                .iter()
                .map(|cohort| sm89_joint_availability(cohort.identity, cohort.portable))
                .filter(|availability| availability.portable.is_some()),
        );
        boards.push(F32TriadAvailability {
            portable: Some(qualified_module_for_auto_identity(
                SM120_TF32_PORTABLE_ARCHITECTURE,
            )),
            ..sm120_availability_for(SM120_TF32_ARCHITECTURE)
        });
        boards
    }

    fn strict_keeps(op: ResolvedGemmOp, selection: F32TriadSelection) -> bool {
        match selection {
            F32TriadSelection::Tf32(route) => rna_batch_invariant_route(op, route),
            F32TriadSelection::Tf32Proof {
                candidate,
                reference,
            } => {
                rna_batch_invariant_route(op, candidate) && rna_batch_invariant_route(op, reference)
            }
            F32TriadSelection::ScalarFma | F32TriadSelection::ExactSm120Fma(_) => false,
        }
    }

    #[test]
    fn rna_strict_admits_only_rna_routes_and_plain_forward_tiles() {
        use crate::gpu::kernel_identity::ResolvedOperandConversion as Conversion;
        for module_kind in [
            ModuleKind::TriadSm80,
            ModuleKind::TriadSm90a,
            ModuleKind::TriadSm100,
            ModuleKind::TriadSm120,
            ModuleKind::TriadSm89Finalist,
            ModuleKind::TriadSm89Tf32Joint,
            ModuleKind::TriadTf32Rna,
        ] {
            for spec in crate::gpu::gemm_bi_triad::contract::tf32_route_specs_all(module_kind) {
                let rna = matches!(
                    spec.operand_conversion,
                    Conversion::RegisterCvtRnaTf32F32
                        | Conversion::TensorMapUint32ThenCvtRnaTf32F32
                        | Conversion::PreRnaAThenRegisterCvtRnaBV1
                );
                let plain = matches!(
                    spec.route,
                    Tf32PhysicalRoute::MmaTf32Rna(_) | Tf32PhysicalRoute::MmaTf32RnaWide(_)
                );
                assert_eq!(
                    rna_batch_invariant_route(spec.op, spec.route),
                    rna && (spec.op != ResolvedGemmOp::Nn || plain),
                    "{}",
                    spec.symbol,
                );
            }
        }
        for spec in crate::gpu::gemm_bi_triad::contract::tf32_splitk_specs_all() {
            assert_eq!(
                rna_batch_invariant_route(spec.op, spec.route),
                spec.op != ResolvedGemmOp::Nn,
                "{}",
                spec.symbol,
            );
        }
        let wide = Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
            tile: Tf32PortableTile::M128N128,
            stages: Tf32PortableStages::S3,
        });
        assert!(!rna_batch_invariant_route(ResolvedGemmOp::Nn, wide));
    }

    #[test]
    fn rna_strict_forward_takes_a_plain_portable_tile_at_every_row_count() {
        for availability in strict_boards() {
            for (reduction, columns) in [
                (64, 96),
                (96, 40),
                (384, 384),
                (512, 128),
                (768, 768),
                (768, 1536),
                (3072, 768),
            ] {
                for rows in [1, 7, 16, 17, 64, 65, 128, 513, 1024, 4096] {
                    let request = normalized_request(ResolvedGemmOp::Nn, rows, columns, reduction);
                    for operands in [
                        sm89_joint_operands(ResolvedGemmOp::Nn),
                        F32TriadOperands {
                            bias: Some(0x4000),
                            ..sm89_joint_operands(ResolvedGemmOp::Nn)
                        },
                    ] {
                        let selection = resolve_f32_triad_auto_with_operands(
                            STRICT,
                            request,
                            operands,
                            availability,
                        )
                        .unwrap();
                        assert!(
                            matches!(
                                selection,
                                F32TriadSelection::Tf32(Tf32PhysicalRoute::MmaTf32Rna(_))
                            ) && strict_keeps(ResolvedGemmOp::Nn, selection),
                            "{availability:?} {request:?}: {selection:?}",
                        );
                    }
                    // Without operands no wide tile can be staged: the Ada and
                    // CC 12.x tables name a portable tile, every other board
                    // the ladder.
                    let selection = resolve_f32_triad_auto(STRICT, request, availability).unwrap();
                    let tabled = availability.portable.is_some_and(|module| {
                        let cc = module.device.compute_capability;
                        cc == (8, 9) || cc.0 == 12
                    });
                    if tabled {
                        assert!(
                            matches!(
                                selection,
                                F32TriadSelection::Tf32(Tf32PhysicalRoute::MmaTf32Rna(_))
                            ),
                            "{request:?}: {selection:?}"
                        );
                    } else {
                        assert_eq!(
                            selection,
                            F32TriadSelection::Tf32(rna_portable_ladder_route(request)),
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn rna_strict_ada_forward_follows_the_ada_table() {
        use crate::gpu::gemm_bi_triad::contract::Tf32RnaWideTile as Wide;
        let mut availability = sm89_availability();
        availability.rna = Some(qualified_module(
            ModuleKind::TriadTf32Rna,
            "sm_89",
            "sm_89",
            (8, 9),
            false,
            101_376,
        ));
        let plain =
            |tile, stages| Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute { tile, stages });
        let wide = Tf32PhysicalRoute::MmaTf32RnaWide;
        let cases = [
            (
                (128, 256, 2112),
                plain(Tf32PortableTile::M16N32, Tf32PortableStages::S4),
            ),
            ((4096, 256, 2112), wide(Wide::M128N128S3)),
            ((1024, 384, 1536), wide(Wide::M128N96S3)),
            ((1024, 256, 256), wide(Wide::MsM128N64S3)),
            ((1024, 256, 128), wide(Wide::MsM64N128Bk16S6)),
            ((1024, 256, 1024), wide(Wide::M96N96S3)),
            ((4096, 128, 64), wide(Wide::M96N96S3)),
            (
                (1024, 128, 64),
                plain(Tf32PortableTile::M16N32, Tf32PortableStages::S4),
            ),
            ((8192, 256, 128), wide(Wide::M128N96S3)),
            ((16384, 256, 128), wide(Wide::M128N128S3)),
            ((8192, 512, 256), wide(Wide::M128N128S3)),
        ];
        for ((m, k, n), expected) in cases {
            let request = normalized_request(ResolvedGemmOp::Nn, m, n, k);
            let aligned = sm89_joint_operands(ResolvedGemmOp::Nn);
            assert_eq!(
                resolve_f32_triad_auto_with_operands(STRICT, request, aligned, availability)
                    .unwrap(),
                F32TriadSelection::Tf32(expected),
                "{m}x{k}x{n}"
            );
            // A misaligned operand stages no wide tile: its portable twin serves.
            let misaligned = F32TriadOperands {
                a: aligned.a + 4,
                ..aligned
            };
            let selection =
                resolve_f32_triad_auto_with_operands(STRICT, request, misaligned, availability)
                    .unwrap();
            assert!(
                matches!(
                    selection,
                    F32TriadSelection::Tf32(Tf32PhysicalRoute::MmaTf32Rna(_))
                ),
                "{m}x{k}x{n}: {selection:?}"
            );
            // Without the module the same holds.
            let without = F32TriadAvailability {
                rna: None,
                ..availability
            };
            let selection =
                resolve_f32_triad_auto_with_operands(STRICT, request, aligned, without).unwrap();
            assert!(
                matches!(
                    selection,
                    F32TriadSelection::Tf32(Tf32PhysicalRoute::MmaTf32Rna(_))
                ),
                "{m}x{k}x{n}: {selection:?}"
            );
        }
        // A CC 12.x board follows its own table, not the Ada's.
        let mut sm120 = sm120_availability_for(SM120_TF32_ARCHITECTURE);
        sm120.portable = Some(qualified_module_for_auto_identity(
            SM120_TF32_PORTABLE_ARCHITECTURE,
        ));
        sm120.rna = Some(qualified_module(
            ModuleKind::TriadTf32Rna,
            "compute_120",
            "sm_120",
            (12, 0),
            true,
            101_376,
        ));
        let request = normalized_request(ResolvedGemmOp::Nn, 4096, 2112, 256);
        let selection = resolve_f32_triad_auto_with_operands(
            STRICT,
            request,
            sm89_joint_operands(ResolvedGemmOp::Nn),
            sm120,
        )
        .unwrap();
        assert_eq!(
            selection,
            F32TriadSelection::Tf32(super::rna_blackwell_forward_route(
                request,
                true,
                sm120.multiprocessors as usize
            )),
        );
    }

    #[test]
    fn rna_strict_gradients_keep_every_specialized_measured_route_and_never_fall_to_exact() {
        let mut cohorts: Vec<(F32TriadAvailability, &[_])> = SM89_JOINT_TF32_EVIDENCE_COHORTS
            .iter()
            .filter(|cohort| cohort.portable.is_some())
            .map(|cohort| {
                (
                    sm89_joint_availability(cohort.identity, cohort.portable),
                    cohort.cells,
                )
            })
            .collect();
        for cohort in SM120_TF32_EVIDENCE_COHORTS {
            let mut availability = sm120_availability_for(cohort.identity);
            availability.portable = cohort.portable.map(qualified_module_for_auto_identity);
            cohorts.push((availability, cohort.cells));
        }
        cohorts.push((sm89_availability(), SM89_TF32_EVIDENCE_CELLS));
        let mut kept = 0;
        for (availability, cells) in cohorts {
            for cell in cells.iter().filter(|cell| cell.op != ResolvedGemmOp::Nn) {
                let request = normalized_request(
                    cell.op,
                    cell.shape.output_rows,
                    cell.shape.output_columns,
                    cell.shape.reduction,
                );
                let selection = resolve_f32_triad_auto_with_operands(
                    STRICT,
                    request,
                    sm89_joint_operands(cell.op),
                    availability,
                )
                .unwrap();
                assert!(strict_keeps(cell.op, selection), "{cell:?}: {selection:?}");
                let allowed = resolve_f32_triad_auto_with_operands(
                    F32TriadPolicy::AllowDeterministicTf32,
                    request,
                    sm89_joint_operands(cell.op),
                    availability,
                )
                .unwrap();
                // A specialized cohort winner keeps its cell; a portable tile
                // or split member yields to the gradients' shape rules.
                if let F32TriadSelection::Tf32(route) = allowed
                    && rna_batch_invariant_route(cell.op, route)
                    && !matches!(
                        route,
                        Tf32PhysicalRoute::MmaTf32Rna(_)
                            | Tf32PhysicalRoute::MmaTf32RnaSplitK2(_)
                            | Tf32PhysicalRoute::MmaTf32RnaSplitK4(_)
                            | Tf32PhysicalRoute::MmaTf32RnaSplitK8(_)
                    )
                {
                    assert_eq!(selection, allowed, "{cell:?}");
                    kept += 1;
                }
            }
        }
        assert!(
            kept > 0,
            "no measured gradient route kept the strict contract"
        );
    }

    #[test]
    fn rna_strict_gradients_split_by_the_shape_and_fill_by_the_card() {
        let rna_backward_rule_route_for =
            |request, availability| super::rna_backward_rule_route(request, None, availability);
        let split8 = |tile, stages| {
            Some(Tf32PhysicalRoute::MmaTf32RnaSplitK8(Tf32PortableRoute {
                tile,
                stages,
            }))
        };
        let tn = |dims| F32TriadRequest {
            op: ResolvedGemmOp::Tn,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Tn, dims),
        };
        let nt = |dims| F32TriadRequest {
            op: ResolvedGemmOp::Nt,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, dims),
        };
        let ada = sm89_availability();
        // A deep weight gradient splits eight ways on its 128 x 796 output,
        // which the 13 CTAs of one plain tile left idle.
        assert_eq!(
            rna_backward_rule_route_for(tn((4096, 128, 796)), ada),
            split8(Tf32PortableTile::M64N64, Tf32PortableStages::S2)
        );
        assert_eq!(
            rna_backward_rule_route_for(tn((4096, 256, 128)), ada),
            split8(Tf32PortableTile::M32N32, Tf32PortableStages::S3)
        );
        // Every card splits the same shapes; only the member moves.
        for multiprocessors in [24, 82, 108, 128, 132, 142, 170, 188] {
            let mut board = ada;
            board.multiprocessors = multiprocessors;
            for dims in [(4096, 128, 796), (2048, 512, 256), (1024, 384, 1536)] {
                assert_eq!(
                    matches!(
                        rna_backward_rule_route_for(tn(dims), board),
                        Some(Tf32PhysicalRoute::MmaTf32RnaSplitK8(_))
                    ),
                    dims.0 >= 2048,
                    "{dims:?} on {multiprocessors} SMs"
                );
            }
        }
        assert_eq!(
            rna_backward_rule_route_for(tn((1024, 384, 101)), ada),
            Some(Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                tile: Tf32PortableTile::M16N32,
                stages: Tf32PortableStages::S4,
            }))
        );
        // A few rows over a long reduction split four ways; a wide output
        // or a short reduction keeps the narrow tile.
        assert_eq!(
            rna_backward_rule_route_for(nt((64, 384, 1536)), ada),
            Some(Tf32PhysicalRoute::MmaTf32RnaSplitK4(Tf32PortableRoute {
                tile: Tf32PortableTile::M16N32,
                stages: Tf32PortableStages::S3,
            }))
        );
        for dims in [(64, 1536, 384), (64, 384, 101)] {
            assert_eq!(
                rna_backward_rule_route_for(nt(dims), ada),
                Some(Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: Tf32PortableTile::M16N32,
                    stages: Tf32PortableStages::S4,
                })),
                "{dims:?}"
            );
        }
        // With the batch-invariant module bound and staged operands the Ada
        // takes its multistage input-gradient tiles.
        let mut staged = ada;
        staged.rna = Some(qualified_module(
            ModuleKind::TriadTf32Rna,
            "sm_89",
            "sm_89",
            (8, 9),
            false,
            101_376,
        ));
        let aligned = F32TriadOperands {
            output: 0x1000,
            a: 0x2000,
            b: 0x3000,
            bias: None,
            alpha: 1.0,
            beta: 0.0,
        };
        for (dims, tile) in [
            (
                (1024, 1536, 384),
                super::super::contract::Tf32RnaWideTile::NtMsM128N128Bk16S5,
            ),
            (
                (4096, 256, 2112),
                super::super::contract::Tf32RnaWideTile::NtMsM64N128Bk16S6,
            ),
        ] {
            assert_eq!(
                super::rna_backward_rule_route(nt(dims), Some(aligned), staged),
                Some(Tf32PhysicalRoute::MmaTf32RnaWide(tile)),
                "{dims:?}"
            );
            assert_eq!(
                super::rna_backward_rule_route(nt(dims), None, staged),
                Some(Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: Tf32PortableTile::M64N64,
                    stages: Tf32PortableStages::S2,
                })),
                "{dims:?} without operands"
            );
        }
        // A staged deep weight gradient takes the multistage members of the
        // same eight partitions.
        let accumulate = F32TriadOperands {
            beta: 1.0,
            ..aligned
        };
        for (dims, tile) in [
            (
                (4096, 128, 796),
                super::super::contract::Tf32RnaSplitTile::M128N64S3,
            ),
            (
                (4096, 256, 128),
                super::super::contract::Tf32RnaSplitTile::M64N64S4,
            ),
        ] {
            assert_eq!(
                super::rna_backward_rule_route(tn(dims), Some(accumulate), staged),
                Some(Tf32PhysicalRoute::MmaTf32RnaWideSplitK8(tile)),
                "{dims:?}"
            );
        }
        // An output that is not whole chunks keeps the portable member.
        assert!(matches!(
            super::rna_backward_rule_route(tn((4096, 130, 798)), Some(accumulate), staged),
            Some(Tf32PhysicalRoute::MmaTf32RnaSplitK8(_))
        ));
        assert_eq!(
            rna_backward_rule_route_for(
                F32TriadRequest {
                    op: ResolvedGemmOp::Nn,
                    shape: F32TriadShape::contiguous(ResolvedGemmOp::Nn, (4096, 128, 796)),
                },
                ada
            ),
            None
        );
    }

    #[test]
    fn rna_strict_deep_weight_gradient_on_sm120_never_binds_an_uncomposed_member() {
        let mut availability = sm120_availability_for(SM120_TF32_ARCHITECTURE);
        availability.portable = Some(qualified_module(
            ModuleKind::TriadSm80,
            "compute_120",
            "sm_120",
            (12, 0),
            true,
            101_376,
        ));
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Tn,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Tn, (4096, 128, 796)),
        };
        let selection = resolve_f32_triad_auto(STRICT, request, availability).unwrap();
        assert!(
            !matches!(
                selection,
                F32TriadSelection::Tf32(Tf32PhysicalRoute::MmaTf32RnaSplitK8(_))
            ),
            "{selection:?}"
        );
    }

    #[test]
    fn rna_strict_deep_weight_gradient_on_sm120_cuts_eight_ways_at_any_alignment() {
        // The cut of a deep strict weight gradient must not follow where its
        // operands sit: rows of 798 floats or a base off a 16-byte boundary
        // take the member that stages four bytes at a time, which folds the
        // same eight partitions as the 16-byte member.
        let mut availability = sm120_availability_for(SM120_TF32_ARCHITECTURE);
        availability.portable = Some(qualified_module(
            ModuleKind::TriadSm80,
            "compute_120",
            "sm_120",
            (12, 0),
            true,
            101_376,
        ));
        availability.rna = Some(qualified_module(
            ModuleKind::TriadTf32Rna,
            "compute_120",
            "sm_120",
            (12, 0),
            true,
            101_376,
        ));
        let tn = |dims| F32TriadRequest {
            op: ResolvedGemmOp::Tn,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Tn, dims),
        };
        let aligned = F32TriadOperands {
            output: 0x1000,
            a: 0x2000,
            b: 0x3000,
            bias: None,
            alpha: 1.0,
            beta: 1.0,
        };
        let member = |tile| Some(Tf32PhysicalRoute::MmaTf32RnaWideSplitK8(tile));
        use super::super::contract::Tf32RnaSplitTile as Split;
        assert_eq!(
            super::rna_backward_rule_route(tn((4096, 256, 128)), Some(aligned), availability),
            member(Split::M64N64S4)
        );
        let shifted = F32TriadOperands {
            a: 0x2004,
            ..aligned
        };
        assert_eq!(
            super::rna_backward_rule_route(tn((4096, 256, 128)), Some(shifted), availability),
            member(Split::M64N64S4AnyAlignment)
        );
        assert_eq!(
            super::rna_backward_rule_route(tn((4096, 130, 798)), Some(aligned), availability),
            member(Split::M64N64S4AnyAlignment)
        );
        let any = super::super::contract::tf32_splitk_spec(
            ResolvedGemmOp::Tn,
            Tf32PhysicalRoute::MmaTf32RnaWideSplitK8(Split::M64N64S4AnyAlignment),
        )
        .unwrap();
        let whole = super::super::contract::tf32_splitk_spec(
            ResolvedGemmOp::Tn,
            Tf32PhysicalRoute::MmaTf32RnaWideSplitK8(Split::M64N64S4),
        )
        .unwrap();
        assert_eq!(any.partitions, whole.partitions);
        assert_eq!((any.tile, any.bk), (whole.tile, whole.bk));
    }

    #[test]
    fn a_split_member_is_held_to_its_own_kernel_symbol() {
        // The launch checks a route's exclusion by this symbol, so a split
        // member must resolve to its own kernel and not miss the check.
        use super::super::contract::{
            Tf32RnaSplitTile as Split, Tf32RnaWideTile, tf32_route_symbol,
        };
        use super::super::tf32_rna_source::{
            NN_MS_PERSIST_M128N64_BK16_S3_SYMBOL, TN_MS_SPLITK8_M64N64_S4_ANY_SYMBOL,
        };
        assert_eq!(
            tf32_route_symbol(
                ResolvedGemmOp::Tn,
                Tf32PhysicalRoute::MmaTf32RnaWideSplitK8(Split::M64N64S4AnyAlignment)
            ),
            Ok(TN_MS_SPLITK8_M64N64_S4_ANY_SYMBOL)
        );
        assert_eq!(
            tf32_route_symbol(
                ResolvedGemmOp::Nn,
                Tf32PhysicalRoute::MmaTf32RnaWide(Tf32RnaWideTile::MsPersistM128N64Bk16S3)
            ),
            Ok(NN_MS_PERSIST_M128N64_BK16_S3_SYMBOL)
        );
    }

    #[test]
    fn rna_strict_policy_refuses_a_context_without_the_portable_module() {
        let availability = sm120_availability_for(SM120_TF32_ARCHITECTURE);
        for op in [ResolvedGemmOp::Nn, ResolvedGemmOp::Tn, ResolvedGemmOp::Nt] {
            assert!(resolve_f32_triad_auto(STRICT, request(op), availability).is_err());
        }
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

    #[test]
    fn a_small_weight_gradient_leaves_the_wave_and_the_forward_never_does() {
        // A d128 weight gradient: four output tiles over sixteen reduction
        // steps. The portable tiles serve it; the forward and the input
        // gradient of the same shape stay on the wave whatever their size.
        let small = (1024, 256, 128);
        assert_eq!(
            resolve_sm90a_auto((9, 0), true, sm90a_request(Sm90aOp::Tn, small)),
            None
        );
        for op in [Sm90aOp::Nn, Sm90aOp::Nt] {
            assert!(resolve_sm90a_auto((9, 0), true, sm90a_request(op, small)).is_some());
        }
        let large = sm90a_request(Sm90aOp::Tn, (2048, 768, 3072));
        assert!(resolve_sm90a_auto((9, 0), true, large).is_some());
        // A measured cell is served whatever its size.
        let cell = Sm90aForcedRoute {
            op: Sm90aOp::Tn,
            dtype: WeightDtype::Bf16,
            schedule: Sm90aWarpgroupSchedule::Wg2,
            shape: Sm90aShape::contiguous(Sm90aOp::Tn, small),
        };
        assert_eq!(
            resolve_sm90a_auto_from_cells(&[cell], (9, 0), true, sm90a_request(Sm90aOp::Tn, small)),
            Some(cell)
        );
    }
}

#[cfg(test)]
mod exact_batch_invariant_tests {
    use super::{
        EXACT_CHUNK, ExactChunkPartial, SPLITK_SCRATCH_CAP, ScalarDispatchPlan,
        exact_batch_invariant_nn_plan, exact_chunks, resolve_f32_triad_auto,
    };
    use crate::gpu::context::F32TriadPolicy;
    use crate::gpu::gemm_bi_triad::contract::{
        F32TriadAvailability, F32TriadRequest, F32TriadSelection, F32TriadShape,
    };
    use crate::gpu::kernel_identity::ResolvedGemmOp;

    fn nn(m: usize, k: usize, n: usize) -> F32TriadRequest {
        F32TriadRequest {
            op: ResolvedGemmOp::Nn,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nn, (m, k, n)),
        }
    }

    #[test]
    fn exact_batch_invariant_plan_folds_by_the_layer_alone() {
        for multiprocessors in [24, 84, 108, 128, 132, 142, 148, 170, 188] {
            for k in [1, 15, 16, 33, 256, 383, 384, 511, 512, 833, 1536, 3000] {
                for n in [1, 17, 40, 101, 127, 128, 384, 1536, 2112] {
                    for m in [1, 7, 16, 17, 64, 65, 128, 513, 1024, 4096, 32_768] {
                        for compute_capability in [(8, 9), (12, 0)] {
                            let plan = exact_batch_invariant_nn_plan(
                                nn(m, k, n),
                                multiprocessors,
                                compute_capability,
                            )
                            .unwrap();
                            let what = format!(
                                "{m}x{k}x{n} on {multiprocessors} SMs of {compute_capability:?}: {plan:?}"
                            );
                            match plan {
                                ScalarDispatchPlan::NnExactChunk {
                                    partial,
                                    chunks,
                                    piece_rows,
                                    pieces,
                                } => {
                                    assert!(exact_chunks(k, n), "{what}");
                                    assert_eq!(
                                        partial == ExactChunkPartial::Narrow16,
                                        m == 1,
                                        "{what}"
                                    );
                                    assert_eq!(chunks as usize, k.div_ceil(EXACT_CHUNK), "{what}");
                                    let (rows, pieces) = (piece_rows as usize, pieces as usize);
                                    assert!(rows >= 1 && rows <= m, "{what}");
                                    assert!(
                                        rows * pieces >= m && rows * (pieces - 1) < m,
                                        "{what}"
                                    );
                                    assert!(
                                        chunks as usize * rows * n <= SPLITK_SCRATCH_CAP,
                                        "{what}"
                                    );
                                }
                                ScalarDispatchPlan::NnFinal { slim: false } => {
                                    assert!(!exact_chunks(k, n) && m >= 128 && n >= 128, "{what}");
                                }
                                ScalarDispatchPlan::NnExactThin => {
                                    assert!(!exact_chunks(k, n) && m == 1, "{what}");
                                }
                                ScalarDispatchPlan::NnExactNarrow16 => {
                                    assert!(!exact_chunks(k, n) && m > 1, "{what}");
                                }
                                ScalarDispatchPlan::NnM64N64Qualified => {
                                    assert!(!exact_chunks(k, n), "{what}");
                                }
                                other => panic!("{what}: {other:?} is not a batch-invariant tile"),
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn exact_batch_invariant_forward_never_leaves_the_scalar_tiles() {
        let availability = F32TriadAvailability {
            portable: None,
            specialized: None,
            finalist: None,
            joint: None,
            rna: None,
            multiprocessors: 170,
        };
        for (m, k, n) in [
            (1, 64, 96),
            (128, 256, 2112),
            (4096, 512, 256),
            (513, 96, 40),
        ] {
            assert_eq!(
                resolve_f32_triad_auto(
                    F32TriadPolicy::ExactFmaBatchInvariant,
                    nn(m, k, n),
                    availability
                )
                .unwrap(),
                F32TriadSelection::ScalarFma,
            );
        }
        assert!(exact_batch_invariant_nn_plan(nn(0, 64, 96), 170, (12, 0)).is_err());
        assert!(exact_batch_invariant_nn_plan(nn(4, 64, 96), 0, (8, 9)).is_err());
    }
}
