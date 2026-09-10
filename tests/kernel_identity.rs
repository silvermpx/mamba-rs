#![cfg(feature = "cuda")]

use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, F32TriadPolicy, HalfTriadPolicy};
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
    SM120_SCHEDULE_REVISION, SM120_TENSOR_MAP_REVISION, SM120_TUNING_REVISION, Sm120Bk,
    Sm120NumericContract, Sm120Op, Sm120PhysicalRoute, Sm120RouteIdentity, Sm120Schedule,
    Sm120Shape, Sm120Stages, Sm120TargetCandidate, Sm120Tile,
};
use mamba_rs::mamba_ssm::gpu::kernel_identity::{
    ArtifactIdentity, ArtifactKind, BackendSet, CacheEnvelope, CompileKeyMaterial,
    CompilerIdentity, CudaTarget, DeviceCaps, DeviceIdentity, DriverIdentity, FramedSha256,
    GemmPolicy, GemmRouteIdentity, ModuleKind, NUMERIC_ABI_REVISION, NumericContractSet,
    POLICY_REVISION, PhysicalGemmBackend, PhysicalLaunchKind, PolicyDtype, ResolvedGemmLaunchSet,
    ResolvedGemmLaunchSetBuilder, ResolvedGemmOp, ResolvedGemmRoute, ResolvedInstructionFamily,
    ResolvedInstructionShape, ResolvedNumericContract, ResolvedOperandConversion,
    ResolvedOutputOwnership, SCHEDULE_REVISION, ScalarWavePolicyV1, Sm80TcPolicyV3,
    TUNING_TABLE_REVISION, build_artifact_set, build_resolved_gemm_launch_set, canonical_ptx_image,
    gemm_dispatch_policy_digest, route_backend_contract_sets,
};

fn digest(seed: u8) -> [u8; 32] {
    [seed; 32]
}

fn compile_material() -> CompileKeyMaterial {
    CompileKeyMaterial {
        module_kind: ModuleKind::Fixed,
        source: b"source".to_vec(),
        target: b"sm_89".to_vec(),
        argv: vec![
            b"--fmad=true".to_vec(),
            b"--gpu-architecture=sm_89".to_vec(),
        ],
        header_manifest: Some(b"headers".to_vec()),
        nvrtc_version: (13, 2),
        nvrtc_library_domain: Some(b"libnvrtc.so.13".to_vec()),
        output_kind: ArtifactKind::Ptx,
        composer_revision: 1,
        compiler_revision: 1,
        numeric_abi_revision: NUMERIC_ABI_REVISION,
        schedule_revision: SCHEDULE_REVISION,
    }
}

#[test]
fn module_kind_discriminants_are_stable() {
    assert_eq!(ModuleKind::Fixed as u8, 1);
    assert_eq!(ModuleKind::TriadScalar as u8, 2);
    assert_eq!(ModuleKind::TriadSm80 as u8, 3);
    assert_eq!(ModuleKind::TriadSm90a as u8, 4);
    assert_eq!(ModuleKind::TriadSm100 as u8, 5);
    assert_eq!(ModuleKind::TriadSm120 as u8, 6);
    assert_eq!(ModuleKind::Mamba3Combined as u8, 7);
    assert_eq!(ModuleKind::TriadSm89Finalist as u8, 8);
    assert_eq!(ModuleKind::TriadSm89Half as u8, 9);
    assert_eq!(ModuleKind::TriadSm89ExactF32 as u8, 10);
    assert_eq!(ModuleKind::TriadSm89Tf32Joint as u8, 11);
}

#[test]
fn f32_triad_policy_parser_accepts_only_the_versioned_public_spellings() {
    assert_eq!(
        F32TriadPolicy::parse_env_value("exact").unwrap(),
        F32TriadPolicy::ExactScalarFmaV1
    );
    assert_eq!(
        F32TriadPolicy::parse_env_value(" \t\ntf32\r ").unwrap(),
        F32TriadPolicy::AllowDeterministicTf32V1
    );
    assert_eq!(F32TriadPolicy::default(), F32TriadPolicy::ExactScalarFmaV1);
    assert_eq!(F32TriadPolicy::ExactScalarFmaV1 as u8, 0);
    assert_eq!(F32TriadPolicy::AllowDeterministicTf32V1 as u8, 1);

    for rejected in [
        "",
        "EXACT",
        "TF32",
        "1",
        "on",
        "true",
        "yes",
        "off",
        "false",
        "other",
        "\u{a0}exact",
    ] {
        let error = F32TriadPolicy::parse_env_value(rejected)
            .expect_err("unsupported policy spelling must fail closed");
        assert!(error.contains("MAMBA_RS_BI_F32_POLICY"), "{error}");
        assert!(error.contains(&format!("{rejected:?}")), "{error}");
        assert!(error.contains("exact") && error.contains("tf32"), "{error}");
    }
}

#[test]
fn framed_sha256_has_unambiguous_boundaries() {
    let first = FramedSha256::new(b"test")
        .required(b"part", b"ab")
        .required(b"part", b"c")
        .finish();
    let second = FramedSha256::new(b"test")
        .required(b"part", b"a")
        .required(b"part", b"bc")
        .finish();
    let embedded = FramedSha256::new(b"test")
        .required(b"part", b"a\x1fb")
        .finish();
    let split = FramedSha256::new(b"test")
        .required(b"part", b"a")
        .required(b"part", b"b")
        .finish();
    let absent = FramedSha256::new(b"test").optional(b"part", None).finish();
    let empty = FramedSha256::new(b"test")
        .optional(b"part", Some(b""))
        .finish();

    assert_ne!(first, second);
    assert_ne!(embedded, split);
    assert_ne!(absent, empty);
}

#[test]
fn compile_key_covers_every_invocation_field() {
    let base = compile_material();
    let expected = base.digest().expect("complete key material");
    let expected_invocation = base.invocation_digest();
    let mut mutations = Vec::new();

    let mut value = base.clone();
    value.module_kind = ModuleKind::TriadScalar;
    mutations.push(value);
    let mut value = base.clone();
    value.source.push(b'!');
    mutations.push(value);
    let mut value = base.clone();
    value.target.push(b'a');
    mutations.push(value);
    let mut value = base.clone();
    value.argv[0].push(b'!');
    mutations.push(value);
    let mut value = base.clone();
    value.argv.reverse();
    mutations.push(value);
    let mut value = base.clone();
    value.header_manifest.as_mut().unwrap().push(b'!');
    mutations.push(value);
    let mut value = base.clone();
    value.nvrtc_version.1 += 1;
    mutations.push(value);
    let mut value = base.clone();
    value.nvrtc_library_domain.as_mut().unwrap().push(b'!');
    mutations.push(value);
    let mut value = base.clone();
    value.output_kind = ArtifactKind::Cubin;
    mutations.push(value);
    let mut value = base.clone();
    value.composer_revision += 1;
    mutations.push(value);
    let mut value = base.clone();
    value.compiler_revision += 1;
    mutations.push(value);
    let mut value = base.clone();
    value.numeric_abi_revision += 1;
    mutations.push(value);
    let mut value = base.clone();
    value.schedule_revision += 1;
    mutations.push(value);

    for mutation in mutations {
        assert_ne!(mutation.digest().unwrap(), expected);
        assert_ne!(mutation.invocation_digest(), expected_invocation);
    }
    let mut incomplete = base;
    incomplete.header_manifest = None;
    assert!(incomplete.digest().is_none());
    incomplete.header_manifest = Some(b"headers".to_vec());
    incomplete.nvrtc_library_domain = None;
    assert!(incomplete.digest().is_none());
}

#[test]
fn canonical_ptx_requires_one_terminal_nul() {
    assert_eq!(canonical_ptx_image(b"ptx\0").unwrap(), "ptx");
    assert!(canonical_ptx_image(b"ptx").is_err());
    assert!(canonical_ptx_image(b"pt\0x\0").is_err());
    assert!(canonical_ptx_image(&[0xff, 0]).is_err());
}

#[test]
fn backend_contract_sets_match_reachable_dispatch_trees() {
    let policy = |batch_invariant, bi_tensor_cores, bi_gemm_family| GemmPolicy {
        batch_invariant,
        bi_tensor_cores,
        fast_gemm: false,
        cublas_tf32: false,
        f32_triad_policy: F32TriadPolicy::ExactScalarFmaV1,
        half_triad_policy: HalfTriadPolicy::TiledParityV1,
        bi_gemm_family,
    };

    let (backends, contracts) =
        route_backend_contract_sets(policy(false, false, BiGemmFamily::Triad));
    assert_eq!(backends, BackendSet::CUBLAS);
    assert_eq!(contracts, NumericContractSet::CUBLAS_POLICY_V1);

    for tc in [false, true] {
        let (backends, contracts) =
            route_backend_contract_sets(policy(true, tc, BiGemmFamily::Triad));
        assert!(backends.contains(BackendSet::TRIAD));
        assert!(backends.contains(BackendSet::FIXED));
        assert!(contracts.contains(NumericContractSet::TRIAD_SCALAR_FMA_V1));
        assert!(contracts.contains(NumericContractSet::FIXED_MATVEC_TREE_V1));
        assert_eq!(
            contracts.contains(NumericContractSet::TRIAD_MMA_SYNC_V1),
            tc
        );
    }

    for tc in [false, true] {
        let (backends, contracts) =
            route_backend_contract_sets(policy(true, tc, BiGemmFamily::Fixed));
        assert!(backends.contains(BackendSet::TRIAD));
        assert!(backends.contains(BackendSet::FIXED));
        assert!(!contracts.contains(NumericContractSet::FIXED_MATVEC_TREE_V1));
        assert!(contracts.contains(NumericContractSet::FIXED_SCALAR_FMA_V1));
        assert!(contracts.contains(NumericContractSet::FIXED_MMA_SYNC_V1));
        assert!(contracts.contains(NumericContractSet::TRIAD_SCALAR_FMA_V1));
        assert_eq!(
            contracts.contains(NumericContractSet::TRIAD_MMA_SYNC_V1),
            tc
        );
    }
}

#[test]
fn gemm_policy_keeps_cublas_and_deterministic_triad_tf32_independent() {
    let policy = |cublas_tf32, f32_triad_policy| GemmPolicy {
        batch_invariant: true,
        bi_tensor_cores: false,
        fast_gemm: false,
        cublas_tf32,
        f32_triad_policy,
        half_triad_policy: HalfTriadPolicy::TiledParityV1,
        bi_gemm_family: BiGemmFamily::Triad,
    };

    let exact = policy(true, F32TriadPolicy::ExactScalarFmaV1);
    let allow = policy(false, F32TriadPolicy::AllowDeterministicTf32V1);
    assert!(exact.cublas_tf32);
    assert_eq!(exact.f32_triad_policy, F32TriadPolicy::ExactScalarFmaV1);
    assert!(!allow.cublas_tf32);
    assert_eq!(
        allow.f32_triad_policy,
        F32TriadPolicy::AllowDeterministicTf32V1
    );

    let (_, exact_contracts) = route_backend_contract_sets(exact);
    let (_, allow_contracts) = route_backend_contract_sets(allow);
    assert!(!exact_contracts.contains(NumericContractSet::TRIAD_DETERMINISTIC_TF32_V1));
    assert!(allow_contracts.contains(NumericContractSet::TRIAD_DETERMINISTIC_TF32_V1));

    let mut fixed_allow = allow;
    fixed_allow.bi_gemm_family = BiGemmFamily::Fixed;
    let (_, fixed_contracts) = route_backend_contract_sets(fixed_allow);
    assert!(!fixed_contracts.contains(NumericContractSet::TRIAD_DETERMINISTIC_TF32_V1));
}

#[test]
fn half_policy_opens_the_stream_k_contract_only_inside_the_tensor_core_tier() {
    let policy = |bi_tensor_cores, half_triad_policy, bi_gemm_family| GemmPolicy {
        batch_invariant: true,
        bi_tensor_cores,
        fast_gemm: false,
        cublas_tf32: false,
        f32_triad_policy: F32TriadPolicy::ExactScalarFmaV1,
        half_triad_policy,
        bi_gemm_family,
    };
    let stream_k = NumericContractSet::TRIAD_MMA_SYNC_STREAM_K_V1;

    // The default policy never carries the fixed-order fold, tensor cores or not.
    for tc in [false, true] {
        for family in [BiGemmFamily::Triad, BiGemmFamily::Fixed] {
            let (_, contracts) =
                route_backend_contract_sets(policy(tc, HalfTriadPolicy::TiledParityV1, family));
            assert!(!contracts.contains(stream_k), "{tc} {family:?}");
        }
    }
    // The permission reaches a kernel only through the tensor-core tier,
    // under either family.
    for family in [BiGemmFamily::Triad, BiGemmFamily::Fixed] {
        let (_, without_tc) = route_backend_contract_sets(policy(
            false,
            HalfTriadPolicy::AllowStreamKFixedOrderV1,
            family,
        ));
        assert!(!without_tc.contains(stream_k), "{family:?}");
        let (_, with_tc) = route_backend_contract_sets(policy(
            true,
            HalfTriadPolicy::AllowStreamKFixedOrderV1,
            family,
        ));
        assert!(with_tc.contains(stream_k), "{family:?}");
        assert!(
            with_tc.contains(NumericContractSet::TRIAD_MMA_SYNC_V1),
            "{family:?}"
        );
    }
    // Outside the batch-invariant dispatch cuBLAS owns the route.
    let mut cublas = policy(
        true,
        HalfTriadPolicy::AllowStreamKFixedOrderV1,
        BiGemmFamily::Triad,
    );
    cublas.batch_invariant = false;
    assert_eq!(
        route_backend_contract_sets(cublas).1,
        NumericContractSet::CUBLAS_POLICY_V1
    );
}

#[test]
fn cache_envelope_rejects_corruption_and_wrong_identity() {
    let key = digest(1);
    let payload = b"ptx payload";
    let encoded = CacheEnvelope::encode(key, ArtifactKind::Ptx, payload);
    let hit = CacheEnvelope::decode(key, ArtifactKind::Ptx, &encoded).unwrap();
    assert_eq!(hit.payload, payload);
    assert_eq!(hit.artifact_digest, FramedSha256::bytes(payload));

    let mut corrupt = encoded.clone();
    *corrupt.last_mut().unwrap() ^= 1;
    assert!(CacheEnvelope::decode(key, ArtifactKind::Ptx, &corrupt).is_err());
    assert!(CacheEnvelope::decode(digest(2), ArtifactKind::Ptx, &encoded).is_err());
    assert!(CacheEnvelope::decode(key, ArtifactKind::Cubin, &encoded).is_err());
}

fn artifact(kind: ModuleKind, seed: u8) -> ArtifactIdentity {
    ArtifactIdentity {
        module_kind: kind,
        artifact_kind: ArtifactKind::Ptx,
        compile_key: digest(seed),
        artifact_digest: digest(seed + 1),
    }
}

#[test]
fn artifact_set_is_ordered_and_rejects_duplicate_module_kinds() {
    let fixed = artifact(ModuleKind::Fixed, 1);
    let scalar = artifact(ModuleKind::TriadScalar, 3);
    let sm80 = artifact(ModuleKind::TriadSm80, 5);
    let first = build_artifact_set(&[fixed, scalar, sm80]).unwrap();
    assert_eq!(first.module_count, 3);
    assert_eq!(first.fixed, fixed);
    assert_eq!(first.triad_scalar, scalar);
    assert_eq!(first.triad_sm80, sm80);
    assert_eq!(first.specialized, None);
    assert!(build_artifact_set(&[scalar, fixed, sm80]).is_err());
    assert!(build_artifact_set(&[fixed, scalar]).is_err());
    assert!(build_artifact_set(&[fixed, fixed]).is_err());
    let sm90a = artifact(ModuleKind::TriadSm90a, 7);
    let specialized = build_artifact_set(&[fixed, scalar, sm80, sm90a]).unwrap();
    assert_eq!(specialized.module_count, 4);
    assert_eq!(specialized.specialized, Some(sm90a));
    assert!(
        build_artifact_set(&[
            fixed,
            scalar,
            sm80,
            sm90a,
            artifact(ModuleKind::TriadSm100, 9),
        ])
        .is_err()
    );
}

#[test]
fn artifact_set_tracks_all_four_sm89_optional_modules_in_canonical_order() {
    let fixed = artifact(ModuleKind::Fixed, 1);
    let scalar = artifact(ModuleKind::TriadScalar, 3);
    let sm80 = artifact(ModuleKind::TriadSm80, 5);
    let finalist = artifact(ModuleKind::TriadSm89Finalist, 7);
    let half = artifact(ModuleKind::TriadSm89Half, 9);
    let exact = artifact(ModuleKind::TriadSm89ExactF32, 11);
    let joint = artifact(ModuleKind::TriadSm89Tf32Joint, 13);

    let all = build_artifact_set(&[fixed, scalar, sm80, finalist, half, exact, joint]).unwrap();
    assert_eq!(all.module_count, 7);
    assert_eq!(all.specialized, Some(finalist));
    assert_eq!(all.sm89_half, Some(half));
    assert_eq!(all.sm89_exact_f32, Some(exact));
    assert_eq!(all.sm89_tf32_joint, Some(joint));

    let half_only = build_artifact_set(&[fixed, scalar, sm80, half]).unwrap();
    assert_eq!(half_only.module_count, 4);
    assert_eq!(half_only.specialized, None);
    assert_eq!(half_only.sm89_half, Some(half));
    assert_eq!(half_only.sm89_exact_f32, None);
    assert_eq!(half_only.sm89_tf32_joint, None);

    for artifacts in [
        vec![fixed, scalar, sm80, joint],
        vec![fixed, scalar, sm80, exact, joint],
        vec![fixed, scalar, sm80, finalist, exact, joint],
        vec![fixed, scalar, sm80, half, exact, joint],
    ] {
        let set = build_artifact_set(&artifacts).unwrap();
        assert_eq!(set.sm89_tf32_joint, Some(joint));
    }

    let changed_exact = artifact(ModuleKind::TriadSm89ExactF32, 13);
    assert_ne!(
        all.ordered_digest,
        build_artifact_set(&[fixed, scalar, sm80, finalist, half, changed_exact, joint])
            .unwrap()
            .ordered_digest
    );

    let changed_joint = artifact(ModuleKind::TriadSm89Tf32Joint, 15);
    assert_ne!(
        all.ordered_digest,
        build_artifact_set(&[fixed, scalar, sm80, finalist, half, exact, changed_joint])
            .unwrap()
            .ordered_digest
    );

    assert!(build_artifact_set(&[fixed, scalar, sm80, half, finalist]).is_err());
    assert!(build_artifact_set(&[fixed, scalar, sm80, exact, half]).is_err());
    assert!(build_artifact_set(&[fixed, scalar, sm80, exact, finalist]).is_err());
    assert!(build_artifact_set(&[fixed, scalar, sm80, exact, exact]).is_err());
    assert!(build_artifact_set(&[fixed, scalar, sm80, joint, exact]).is_err());
    assert!(build_artifact_set(&[fixed, scalar, sm80, joint, joint]).is_err());
    assert!(
        build_artifact_set(&[
            fixed,
            scalar,
            sm80,
            artifact(ModuleKind::TriadSm90a, 13),
            exact,
        ])
        .is_err()
    );
    assert!(build_artifact_set(&[fixed, scalar, sm80, finalist, finalist]).is_err());
}

#[test]
fn sm80_tc_policy_v3_digest_covers_geometry_waves_device_and_deep_split_k() {
    let policy = Sm80TcPolicyV3::current();
    let hash = policy.digest(142);
    assert_ne!(policy.digest(141), hash);

    macro_rules! assert_field_changes_digest {
        ($field:ident, $value:expr) => {{
            let mut changed = policy;
            changed.$field = $value;
            assert_ne!(changed.digest(142), hash, stringify!($field));
        }};
    }

    assert_field_changes_digest!(reject_zero_axes, !policy.reject_zero_axes);
    assert_field_changes_digest!(square_tile_min, policy.square_tile_min + 1);
    assert_field_changes_digest!(large_tile_min, policy.large_tile_min + 1);
    assert_field_changes_digest!(forward_min_columns, policy.forward_min_columns + 1);
    assert_field_changes_digest!(forward_thin_max_rows, policy.forward_thin_max_rows + 1);
    assert_field_changes_digest!(
        forward_thin_below_square_columns,
        !policy.forward_thin_below_square_columns
    );
    assert_field_changes_digest!(
        forward_underfill_max_reduction,
        policy.forward_underfill_max_reduction + 1
    );
    assert_field_changes_digest!(
        forward_underfill_min_columns,
        policy.forward_underfill_min_columns + 1
    );
    assert_field_changes_digest!(
        forward_underfill_wave_numerator,
        policy.forward_underfill_wave_numerator + 1
    );
    assert_field_changes_digest!(
        forward_underfill_wave_denominator,
        policy.forward_underfill_wave_denominator + 1
    );
    assert_field_changes_digest!(
        forward_short_reduction_max,
        policy.forward_short_reduction_max + 1
    );
    assert_field_changes_digest!(
        forward_short_reduction_wave_numerator,
        policy.forward_short_reduction_wave_numerator + 1
    );
    assert_field_changes_digest!(
        forward_short_reduction_wave_denominator,
        policy.forward_short_reduction_wave_denominator + 1
    );
    assert_field_changes_digest!(
        tile128_base_wave_numerator,
        policy.tile128_base_wave_numerator + 1
    );
    assert_field_changes_digest!(
        tile128_base_wave_denominator,
        policy.tile128_base_wave_denominator + 1
    );
    assert_field_changes_digest!(
        tn_rectangular_min_aspect,
        policy.tn_rectangular_min_aspect + 1
    );
    assert_field_changes_digest!(
        tn_rectangular_wave_numerator,
        policy.tn_rectangular_wave_numerator + 1
    );
    assert_field_changes_digest!(
        tn_rectangular_wave_denominator,
        policy.tn_rectangular_wave_denominator + 1
    );
    assert_field_changes_digest!(
        backward_tail_min_reduction,
        policy.backward_tail_min_reduction + 1
    );
    assert_field_changes_digest!(
        backward_tail_min_tile64_ctas,
        policy.backward_tail_min_tile64_ctas + 1
    );
    assert_field_changes_digest!(
        deep_split_k_compute_capability,
        (
            policy.deep_split_k_compute_capability.0 + 1,
            policy.deep_split_k_compute_capability.1
        )
    );
    assert_field_changes_digest!(
        deep_split_k_output_columns,
        policy.deep_split_k_output_columns + 1
    );
    assert_field_changes_digest!(
        deep_split_k_tail_min_reduction,
        policy.deep_split_k_tail_min_reduction + 1
    );
    assert_field_changes_digest!(
        deep_split_k_aligned_min_reduction,
        policy.deep_split_k_aligned_min_reduction + 1
    );
}

#[test]
fn scalar_wave_policy_v1_digest_covers_every_wave_and_device_field() {
    let policy = ScalarWavePolicyV1::current();
    let hash = policy.digest(142);
    assert_ne!(policy.digest(141), hash);

    macro_rules! assert_field_changes_digest {
        ($field:ident) => {{
            let mut changed = policy;
            changed.$field += 1;
            assert_ne!(changed.digest(142), hash, stringify!($field));
        }};
    }

    assert_field_changes_digest!(thin_split_wave_numerator);
    assert_field_changes_digest!(thin_split_wave_denominator);
    assert_field_changes_digest!(slim_split_wave_numerator);
    assert_field_changes_digest!(slim_split_wave_denominator);
    assert_field_changes_digest!(tn_split_m_wave_numerator);
    assert_field_changes_digest!(tn_split_m_wave_denominator);

    assert_ne!(
        gemm_dispatch_policy_digest(141),
        gemm_dispatch_policy_digest(142)
    );
    assert_ne!(gemm_dispatch_policy_digest(142), hash);
    assert_ne!(
        gemm_dispatch_policy_digest(142),
        Sm80TcPolicyV3::current().digest(142)
    );
}

fn route() -> GemmRouteIdentity {
    let compiler = CompilerIdentity {
        source_digest: digest(1),
        invocation_digest: digest(2),
        header_manifest_digest: digest(3),
        target: CudaTarget::new("sm_89").unwrap(),
        nvrtc_version: (13, 2),
        nvrtc_library_domain: digest(4),
        nvrtc_library_known: true,
        output_kind: ArtifactKind::Ptx,
        composer_revision: 1,
        compiler_revision: 1,
        numeric_abi_revision: NUMERIC_ABI_REVISION,
        schedule_revision: SCHEDULE_REVISION,
    };
    let artifacts = build_artifact_set(&[
        artifact(ModuleKind::Fixed, 5),
        artifact(ModuleKind::TriadScalar, 7),
        artifact(ModuleKind::TriadSm80, 9),
    ])
    .unwrap();
    GemmRouteIdentity {
        policy: GemmPolicy {
            batch_invariant: true,
            bi_tensor_cores: true,
            fast_gemm: false,
            cublas_tf32: false,
            f32_triad_policy: F32TriadPolicy::ExactScalarFmaV1,
            half_triad_policy: HalfTriadPolicy::TiledParityV1,
            bi_gemm_family: BiGemmFamily::Triad,
        },
        backend_set: BackendSet::TRIAD,
        numeric_contracts: NumericContractSet::TRIAD_SCALAR_FMA_V1
            .union(NumericContractSet::TRIAD_MMA_SYNC_V1),
        compiler,
        artifacts,
        policy_revision: POLICY_REVISION,
        policy_hash: gemm_dispatch_policy_digest(142),
        device: DeviceIdentity {
            compute_capability: (8, 9),
            multiprocessor_count: 142,
            target: CudaTarget::new("sm_89").unwrap(),
            driver: DriverIdentity {
                api_version: 13_200,
                build_sources: 1,
                build_digest: digest(8),
            },
        },
        device_caps: DeviceCaps {
            compute_capability: (8, 9),
            nvrtc_version: (13, 2),
            accepted_target: None,
            optin_shared_bytes: 101_376,
            tensor_map_access: false,
        },
        tuning_table_revision: TUNING_TABLE_REVISION,
        schedule_set_revision: SCHEDULE_REVISION,
        state_capacity: 64,
    }
}

#[test]
fn route_guard_rejects_every_policy_artifact_and_device_field() {
    let captured = route();
    let mut changed = Vec::new();

    let mut value = captured;
    value.policy.batch_invariant = false;
    changed.push(value);
    let mut value = captured;
    value.policy.bi_tensor_cores = false;
    changed.push(value);
    let mut value = captured;
    value.policy.fast_gemm = true;
    changed.push(value);
    let mut value = captured;
    value.policy.cublas_tf32 = true;
    changed.push(value);
    let mut value = captured;
    value.policy.f32_triad_policy = F32TriadPolicy::AllowDeterministicTf32V1;
    changed.push(value);
    let mut value = captured;
    value.policy.bi_gemm_family = BiGemmFamily::Fixed;
    changed.push(value);
    let mut value = captured;
    value.backend_set = BackendSet::CUBLAS;
    changed.push(value);
    let mut value = captured;
    value.numeric_contracts = NumericContractSet::CUBLAS_POLICY_V1;
    changed.push(value);
    let mut value = captured;
    value.compiler.source_digest[0] ^= 1;
    changed.push(value);
    let mut value = captured;
    value.compiler.invocation_digest[0] ^= 1;
    changed.push(value);
    let mut value = captured;
    value.compiler.header_manifest_digest[0] ^= 1;
    changed.push(value);
    let mut value = captured;
    value.compiler.target = CudaTarget::new("sm_90a").unwrap();
    changed.push(value);
    let mut value = captured;
    value.compiler.nvrtc_version.1 += 1;
    changed.push(value);
    let mut value = captured;
    value.compiler.nvrtc_library_domain[0] ^= 1;
    changed.push(value);
    let mut value = captured;
    value.compiler.nvrtc_library_known = false;
    changed.push(value);
    let mut value = captured;
    value.compiler.output_kind = ArtifactKind::Cubin;
    changed.push(value);
    let mut value = captured;
    value.compiler.composer_revision += 1;
    changed.push(value);
    let mut value = captured;
    value.compiler.compiler_revision += 1;
    changed.push(value);
    let mut value = captured;
    value.compiler.numeric_abi_revision += 1;
    changed.push(value);
    let mut value = captured;
    value.compiler.schedule_revision += 1;
    changed.push(value);
    let mut value = captured;
    value.artifacts.module_count += 1;
    changed.push(value);
    let mut value = captured;
    value.artifacts.ordered_digest[0] ^= 1;
    changed.push(value);
    let mut value = captured;
    value.artifacts.fixed.module_kind = ModuleKind::TriadScalar;
    changed.push(value);
    let mut value = captured;
    value.artifacts.fixed.artifact_kind = ArtifactKind::Cubin;
    changed.push(value);
    let mut value = captured;
    value.artifacts.fixed.compile_key[0] ^= 1;
    changed.push(value);
    let mut value = captured;
    value.artifacts.fixed.artifact_digest[0] ^= 1;
    changed.push(value);
    let mut value = captured;
    value.artifacts.triad_scalar.module_kind = ModuleKind::TriadSm80;
    changed.push(value);
    let mut value = captured;
    value.artifacts.triad_scalar.artifact_kind = ArtifactKind::Cubin;
    changed.push(value);
    let mut value = captured;
    value.artifacts.triad_scalar.compile_key[0] ^= 1;
    changed.push(value);
    let mut value = captured;
    value.artifacts.triad_scalar.artifact_digest[0] ^= 1;
    changed.push(value);
    let mut value = captured;
    value.artifacts.triad_sm80.module_kind = ModuleKind::TriadScalar;
    changed.push(value);
    let mut value = captured;
    value.artifacts.triad_sm80.artifact_kind = ArtifactKind::Cubin;
    changed.push(value);
    let mut value = captured;
    value.artifacts.triad_sm80.compile_key[0] ^= 1;
    changed.push(value);
    let mut value = captured;
    value.artifacts.triad_sm80.artifact_digest[0] ^= 1;
    changed.push(value);
    let mut value = captured;
    value.artifacts.specialized = Some(artifact(ModuleKind::TriadSm90a, 11));
    changed.push(value);
    let mut value = captured;
    value.policy_revision += 1;
    changed.push(value);
    let mut value = captured;
    value.policy_hash[0] ^= 1;
    changed.push(value);
    let mut value = captured;
    value.device.compute_capability.1 += 1;
    changed.push(value);
    let mut value = captured;
    value.device.multiprocessor_count -= 1;
    changed.push(value);
    let mut value = captured;
    value.device.target = CudaTarget::new("sm_90a").unwrap();
    changed.push(value);
    let mut value = captured;
    value.device.driver.api_version += 1;
    changed.push(value);
    let mut value = captured;
    value.device.driver.build_sources ^= 1;
    changed.push(value);
    let mut value = captured;
    value.device.driver.build_digest[0] ^= 1;
    changed.push(value);
    let mut value = captured;
    value.device_caps.compute_capability.1 += 1;
    changed.push(value);
    let mut value = captured;
    value.device_caps.nvrtc_version.1 += 1;
    changed.push(value);
    let mut value = captured;
    value.device_caps.accepted_target = Some(CudaTarget::new("compute_120").unwrap());
    changed.push(value);
    let mut value = captured;
    value.device_caps.optin_shared_bytes -= 1;
    changed.push(value);
    let mut value = captured;
    value.device_caps.tensor_map_access = true;
    changed.push(value);
    let mut value = captured;
    value.tuning_table_revision += 1;
    changed.push(value);
    let mut value = captured;
    value.schedule_set_revision += 1;
    changed.push(value);
    let mut value = captured;
    value.state_capacity += 1;
    changed.push(value);

    for live in changed {
        let error = captured
            .ensure_current(live, "graph replay")
            .expect_err("route drift must be rejected");
        assert!(error.starts_with("graph replay"));
    }
}

fn sm120_route_identity() -> Sm120RouteIdentity {
    let compiler = CompilerIdentity {
        source_digest: digest(31),
        invocation_digest: digest(32),
        header_manifest_digest: digest(33),
        target: CudaTarget::new("compute_121").unwrap(),
        nvrtc_version: (13, 2),
        nvrtc_library_domain: digest(34),
        nvrtc_library_known: true,
        output_kind: ArtifactKind::Ptx,
        composer_revision: 1,
        compiler_revision: 2,
        numeric_abi_revision: NUMERIC_ABI_REVISION,
        schedule_revision: SCHEDULE_REVISION,
    };
    let driver = DriverIdentity {
        api_version: 13_200,
        build_sources: 1,
        build_digest: digest(35),
    };
    Sm120RouteIdentity {
        numeric_contract: Sm120NumericContract::TmaMma16F32V1,
        op: Sm120Op::Nn,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S3,
            schedule: Sm120Schedule::Tiled,
        },
        shape: Sm120Shape {
            m: 257,
            k: 193,
            n: 129,
            lda: 208,
            ldb: 144,
            ldc: 144,
        },
        symbol: "gemm_bi_nn_sm120_tma_128x64_bk64_s3_bf16",
        module_kind: ModuleKind::TriadSm120,
        target: Sm120TargetCandidate {
            device_cc: (12, 1),
            nvrtc_arch: "compute_121",
            ptx_target: "sm_121",
        },
        artifact: artifact(ModuleKind::TriadSm120, 36),
        compiler,
        device: DeviceIdentity {
            compute_capability: (12, 1),
            multiprocessor_count: 84,
            target: CudaTarget::new("sm_121").unwrap(),
            driver,
        },
        device_caps: DeviceCaps {
            compute_capability: (12, 1),
            nvrtc_version: (13, 2),
            accepted_target: Some(CudaTarget::new("compute_121").unwrap()),
            optin_shared_bytes: 101_376,
            tensor_map_access: true,
        },
        tensor_map_revision: SM120_TENSOR_MAP_REVISION,
        tensor_maps_digest: digest(39),
        resources_digest: digest(40),
        tuning_revision: SM120_TUNING_REVISION,
        schedule_revision: SM120_SCHEDULE_REVISION,
    }
}

fn resolved_sm120_route() -> ResolvedGemmRoute {
    sm120_route_identity().resolved_route().unwrap()
}

#[test]
fn sm120_route_identity_resolves_the_exact_production_route() {
    let resolved = sm120_route_identity().resolved_route().unwrap();
    assert_eq!(resolved.op, ResolvedGemmOp::Nn);
    assert_eq!(resolved.dtype, PolicyDtype::Bf16);
    assert_eq!(resolved.backend, PhysicalGemmBackend::Sm120TmaMma16V1);
    assert_eq!(
        resolved.numeric_contract,
        ResolvedNumericContract::MmaSyncF32V1
    );
    assert_eq!(
        resolved.instruction_family,
        ResolvedInstructionFamily::MmaSync
    );
    assert_eq!(
        resolved.instruction_shape,
        ResolvedInstructionShape { m: 16, n: 8, k: 16 }
    );
    assert_eq!(resolved.operand_conversion, ResolvedOperandConversion::None);
    assert_eq!(resolved.shape, (257, 193, 129));
    assert_eq!(resolved.strides, (208, 144, 144));
    assert_eq!(resolved.tile, (128, 64));
    assert_eq!(
        (resolved.bk, resolved.stages, resolved.threads),
        (64, 3, 256)
    );
    assert_eq!(resolved.launch.grid_dim, (9, 1, 1));
    assert_eq!(resolved.launch.block_dim, (256, 1, 1));
    assert_eq!(resolved.launch.shared_mem_bytes, 73_856);
    assert_ne!(resolved.launch.arguments_digest, [0; 32]);
    assert_eq!(resolved.schedule_revision, SM120_SCHEDULE_REVISION);
}

#[test]
fn deterministic_tf32_identity_variants_have_stable_distinct_discriminants() {
    assert_eq!(PhysicalGemmBackend::MmaTf32RnaV1 as u8, 5);
    assert_eq!(PhysicalGemmBackend::Sm90aWgmmaTf32TmaV1 as u8, 6);
    assert_eq!(PhysicalGemmBackend::Sm100Tcgen05Tf32TmaV1 as u8, 7);
    assert_eq!(PhysicalGemmBackend::Sm120TmaMmaTf32RnaV1 as u8, 8);
    assert_eq!(ResolvedNumericContract::MmaTf32RnaV1 as u8, 5);
    assert_eq!(ResolvedNumericContract::Sm90aWgmmaTf32TmaV1 as u8, 6);
    assert_eq!(ResolvedNumericContract::Sm100Tcgen05Tf32TmaV1 as u8, 7);
    assert_eq!(ResolvedNumericContract::Sm120TmaMmaTf32RnaV1 as u8, 8);
    assert_eq!(PhysicalGemmBackend::MmaTf32RnaSplitK4V1 as u8, 10);
    assert_eq!(PhysicalGemmBackend::MmaTf32RnaSplitK2V1 as u8, 11);
    assert_eq!(PhysicalGemmBackend::MmaTf32RnaSplitK8V1 as u8, 16);
    assert_eq!(ResolvedNumericContract::MmaTf32RnaSplitK4V1 as u8, 10);
    assert_eq!(ResolvedNumericContract::MmaTf32RnaSplitK2V1 as u8, 11);
    assert_eq!(ResolvedNumericContract::MmaTf32RnaSplitK8V1 as u8, 15);
    assert_eq!(
        ResolvedOutputOwnership::LastCtaPerOutputTileFixedSplitK4ReduceV1 as u8,
        4
    );
    assert_eq!(
        ResolvedOutputOwnership::LastCtaPerOutputTileFixedSplitK2ReduceV1 as u8,
        5
    );
    assert_eq!(
        ResolvedOutputOwnership::LastCtaPerOutputTileFixedSplitK8ReduceV1 as u8,
        8
    );
    assert_eq!(ResolvedInstructionFamily::ScalarFma as u8, 1);
    assert_eq!(ResolvedInstructionFamily::MmaSync as u8, 2);
    assert_eq!(ResolvedInstructionFamily::Wgmma as u8, 3);
    assert_eq!(ResolvedInstructionFamily::Tcgen05 as u8, 4);
    assert_eq!(ResolvedOperandConversion::None as u8, 0);
    assert_eq!(ResolvedOperandConversion::RegisterCvtRnaTf32F32V1 as u8, 1);
    assert_eq!(ResolvedOperandConversion::TensorMapTfloat32V1 as u8, 2);
    assert_eq!(
        ResolvedOperandConversion::TensorMapUint32ThenCvtRnaTf32F32V1 as u8,
        3
    );
    assert_eq!(ResolvedOperandConversion::RegisterAddHalfUlpTf32V1 as u8, 4);
}

#[test]
fn scalar_split_m_identity_variants_have_stable_distinct_discriminants() {
    assert_eq!(
        PhysicalGemmBackend::ScalarFmaTnNarrowSplitMPartialV1 as u8,
        13
    );
    assert_eq!(PhysicalGemmBackend::ScalarFmaTnSplitMF64ReduceV1 as u8, 15);
    assert_eq!(PhysicalGemmBackend::Sm89Mma16HalfS3V1 as u8, 23);
    assert_eq!(PhysicalGemmBackend::Sm89Mma16HalfS2V1 as u8, 28);
    assert_eq!(
        PhysicalGemmBackend::ScalarFmaSm89ExactF32DualChunkFusedV1 as u8,
        24
    );
    assert_eq!(
        PhysicalGemmBackend::ScalarFmaSm89ExactF32DirectSplitMPartialV1 as u8,
        25
    );
    assert_eq!(
        ResolvedNumericContract::ScalarFmaTnSplitMF64ReduceV1 as u8,
        12
    );
    assert_eq!(
        ResolvedNumericContract::ScalarFmaTnNarrowSplitMPartialV1 as u8,
        13
    );
    assert_eq!(
        ResolvedNumericContract::ScalarFmaTnNarrowSplitMF64ReduceV1 as u8,
        14
    );
    assert_eq!(
        ResolvedNumericContract::ScalarFmaTnSplitMPartialV1 as u8,
        21
    );
    assert_eq!(
        ResolvedOutputOwnership::OneCtaPerOutputTilePerSplitMPartitionV1 as u8,
        13
    );
}

#[test]
fn sm120_route_identity_conversion_rejects_incoherent_inputs() {
    let baseline = sm120_route_identity();
    let mut wrong_symbol = baseline;
    wrong_symbol.symbol = "gemm_bi_nn_sm120_tma_64x64_bk32_s2_bf16";
    let mut wrong_target = baseline;
    wrong_target.device_caps.accepted_target = Some(CudaTarget::new("compute_120").unwrap());
    let mut unsupported_dtype = baseline;
    unsupported_dtype.dtype = WeightDtype::F32;
    let mut insufficient_shared = baseline;
    insufficient_shared.device_caps.optin_shared_bytes = 0;

    for identity in [
        wrong_symbol,
        wrong_target,
        unsupported_dtype,
        insufficient_shared,
    ] {
        assert!(identity.resolved_route().is_err());
    }
}

#[test]
fn resolved_launch_set_is_ordered_and_covers_every_physical_identity_field() {
    let first = resolved_sm120_route();
    let mut second = first;
    second.op = ResolvedGemmOp::Tn;
    second.symbol = "gemm_bi_tn_sm120_tma_64x128_bk32_s2_f16";
    second.dtype = PolicyDtype::F16;
    second.shape = (509, 65, 257);
    second.strides = (80, 272, 272);
    second.tile = (64, 128);
    second.bk = 32;
    second.stages = 2;

    let ordered = build_resolved_gemm_launch_set(&[first, second]).unwrap();
    assert_eq!(ordered.launch_count, 2);
    assert_ne!(
        ordered,
        build_resolved_gemm_launch_set(&[second, first]).unwrap()
    );
    assert_ne!(
        ordered,
        build_resolved_gemm_launch_set(&[first, first]).unwrap()
    );

    let mut mutations = Vec::new();
    let mut value = first;
    value.op = ResolvedGemmOp::Nt;
    mutations.push(value);
    let mut value = first;
    value.dtype = PolicyDtype::F16;
    mutations.push(value);
    let mut value = first;
    value.backend = PhysicalGemmBackend::Sm80Mma16V1;
    mutations.push(value);
    let mut value = first;
    value.numeric_contract = ResolvedNumericContract::ScalarFmaV1;
    mutations.push(value);
    let mut value = first;
    value.instruction_family = ResolvedInstructionFamily::Wgmma;
    mutations.push(value);
    let mut value = first;
    value.instruction_shape.k = 8;
    mutations.push(value);
    let mut value = first;
    value.operand_conversion = ResolvedOperandConversion::TensorMapTfloat32V1;
    mutations.push(value);
    let mut value = first;
    value.symbol = "gemm_bi_nn_tc_bf16";
    mutations.push(value);
    let mut value = first;
    value.module_kind = ModuleKind::TriadSm80;
    mutations.push(value);
    let mut value = first;
    value.target = CudaTarget::new("compute_120").unwrap();
    mutations.push(value);
    let mut value = first;
    value.artifact.artifact_digest[0] ^= 1;
    mutations.push(value);
    let mut value = first;
    value.compiler.invocation_digest[0] ^= 1;
    mutations.push(value);
    let mut value = first;
    value.device.compute_capability.1 = 0;
    mutations.push(value);
    let mut value = first;
    value.device.multiprocessor_count -= 1;
    mutations.push(value);
    let mut value = first;
    value.device.driver.build_digest[0] ^= 1;
    mutations.push(value);
    let mut value = first;
    value.device_caps.nvrtc_version.1 -= 1;
    mutations.push(value);
    let mut value = first;
    value.device_caps.accepted_target = Some(CudaTarget::new("compute_120").unwrap());
    mutations.push(value);
    let mut value = first;
    value.device_caps.optin_shared_bytes -= 1;
    mutations.push(value);
    let mut value = first;
    value.device_caps.tensor_map_access = false;
    mutations.push(value);
    let mut value = first;
    value.shape.0 += 1;
    mutations.push(value);
    let mut value = first;
    value.strides.1 += 1;
    mutations.push(value);
    let mut value = first;
    value.tile.1 = 128;
    mutations.push(value);
    let mut value = first;
    value.bk = 32;
    mutations.push(value);
    let mut value = first;
    value.stages = 2;
    mutations.push(value);
    let mut value = first;
    value.threads = 512;
    value.launch.block_dim = (512, 1, 1);
    mutations.push(value);
    let mut value = first;
    value.launch.grid_dim.0 += 1;
    mutations.push(value);
    let mut value = first;
    value.launch.block_dim = (128, 2, 1);
    mutations.push(value);
    let mut value = first;
    value.launch.shared_mem_bytes += 4;
    mutations.push(value);
    let mut value = first;
    value.launch.arguments_digest[0] ^= 1;
    mutations.push(value);
    let mut value = first;
    value.tensor_map_revision += 1;
    mutations.push(value);
    let mut value = first;
    value.tensor_maps_digest[0] ^= 1;
    mutations.push(value);
    let mut value = first;
    value.resources_digest[0] ^= 1;
    mutations.push(value);
    let mut value = first;
    value.tuning_table_revision += 1;
    mutations.push(value);
    let mut value = first;
    value.schedule_revision += 1;
    mutations.push(value);

    let baseline = build_resolved_gemm_launch_set(&[first]).unwrap();
    for mutation in mutations {
        assert_ne!(
            build_resolved_gemm_launch_set(&[mutation]).unwrap(),
            baseline
        );
    }
}

#[test]
fn streaming_launch_set_builder_matches_slice_builder_and_fails_closed() {
    let first = resolved_sm120_route();
    let mut second = first;
    second.op = ResolvedGemmOp::Tn;
    second.symbol = "gemm_bi_tn_sm120_tma_128x64_bk64_s3_bf16";

    let expected = build_resolved_gemm_launch_set(&[first, second]).unwrap();
    let mut builder = ResolvedGemmLaunchSetBuilder::new(2).unwrap();
    builder.push(&first).unwrap();
    builder.push(&second).unwrap();
    assert_eq!(builder.finish().unwrap(), expected);

    assert!(ResolvedGemmLaunchSetBuilder::new(0).is_err());
    let mut incomplete = ResolvedGemmLaunchSetBuilder::new(2).unwrap();
    incomplete.push(&first).unwrap();
    assert!(incomplete.finish().is_err());
    let mut overflow = ResolvedGemmLaunchSetBuilder::new(1).unwrap();
    overflow.push(&first).unwrap();
    assert!(overflow.push(&second).is_err());
}

#[test]
fn resolved_launch_set_rejects_placeholder_or_incoherent_launches() {
    let baseline = resolved_sm120_route();
    let mut invalid = Vec::new();

    for dimension in 0..3 {
        let mut route = baseline;
        let mut grid = [
            route.launch.grid_dim.0,
            route.launch.grid_dim.1,
            route.launch.grid_dim.2,
        ];
        grid[dimension] = 0;
        route.launch.grid_dim = (grid[0], grid[1], grid[2]);
        invalid.push(route);

        let mut route = baseline;
        let mut block = [
            route.launch.block_dim.0,
            route.launch.block_dim.1,
            route.launch.block_dim.2,
        ];
        block[dimension] = 0;
        route.launch.block_dim = (block[0], block[1], block[2]);
        invalid.push(route);
    }

    let mut wrong_thread_count = baseline;
    wrong_thread_count.threads += 1;
    invalid.push(wrong_thread_count);

    let mut overflowing_block = baseline;
    overflowing_block.launch.block_dim = (u32::MAX, 2, 1);
    invalid.push(overflowing_block);

    let mut placeholder_arguments = baseline;
    placeholder_arguments.launch.arguments_digest = [0; 32];
    invalid.push(placeholder_arguments);

    for route in invalid {
        assert!(
            build_resolved_gemm_launch_set(&[route]).is_err(),
            "invalid launch for {} must fail closed: {:?}",
            route.symbol,
            route.launch
        );
    }
}

#[test]
fn resolved_launch_guard_fails_closed_on_order_count_or_device_drift() {
    let first = resolved_sm120_route();
    let mut second = first;
    second.op = ResolvedGemmOp::Tn;
    let captured = build_resolved_gemm_launch_set(&[first, second]).unwrap();
    captured
        .ensure_current(captured, "SM120 graph replay")
        .unwrap();

    for live in [
        build_resolved_gemm_launch_set(&[second, first]).unwrap(),
        build_resolved_gemm_launch_set(&[first]).unwrap(),
        {
            let mut changed = second;
            changed.device.compute_capability = (12, 0);
            build_resolved_gemm_launch_set(&[first, changed]).unwrap()
        },
    ] {
        let error = captured
            .ensure_current(live, "SM120 graph replay")
            .expect_err("resolved launch drift must reject replay");
        assert!(error.starts_with("SM120 graph replay"), "{error}");
    }
}

#[test]
fn physical_launch_kind_discriminants_are_stable() {
    assert_eq!(PhysicalLaunchKind::Gemm as u8, 1);
    assert_eq!(PhysicalLaunchKind::InputUpcast as u8, 2);
    assert_eq!(PhysicalLaunchKind::OutputDowncast as u8, 3);
}

const _: fn(&[ResolvedGemmRoute]) -> Result<ResolvedGemmLaunchSet, String> =
    build_resolved_gemm_launch_set;
