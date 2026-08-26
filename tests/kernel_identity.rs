#![cfg(feature = "cuda")]

use mamba_rs::mamba_ssm::gpu::context::BiGemmFamily;
use mamba_rs::mamba_ssm::gpu::kernel_identity::{
    ArtifactIdentity, ArtifactKind, BackendSet, CacheEnvelope, CompileKeyMaterial,
    CompilerIdentity, CudaTarget, DeviceIdentity, DriverIdentity, FramedSha256, GemmPolicy,
    GemmRouteIdentity, LegacySm80Policy, ModuleKind, NumericContractSet, PolicyDtype, PolicyOp,
    build_artifact_set, canonical_ptx_image,
};

fn digest(seed: u8) -> [u8; 32] {
    [seed; 32]
}

fn compile_material() -> CompileKeyMaterial {
    CompileKeyMaterial {
        source: b"source".to_vec(),
        target: b"sm_89".to_vec(),
        argv: vec![
            b"--fmad=true".to_vec(),
            b"--gpu-architecture=sm_89".to_vec(),
        ],
        include_roots: vec![b"/cuda/include".to_vec()],
        header_manifest: Some(b"headers".to_vec()),
        nvrtc_version: (13, 2),
        nvrtc_library_domain: Some(b"libnvrtc.so.13".to_vec()),
        output_kind: ArtifactKind::Ptx,
        composer_revision: 1,
        compiler_revision: 1,
        numeric_abi_revision: 1,
        schedule_revision: 1,
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
    let mut mutations = Vec::new();

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
    value.include_roots[0].push(b'!');
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
    let legacy = artifact(ModuleKind::LegacyCombined, 1);
    let scalar = artifact(ModuleKind::TriadScalar, 3);
    let first = build_artifact_set(&[legacy, scalar]).unwrap();
    let second = build_artifact_set(&[scalar, legacy]).unwrap();
    assert_ne!(first.ordered_digest, second.ordered_digest);
    assert!(build_artifact_set(&[legacy, legacy]).is_err());
}

#[test]
fn legacy_policy_hash_and_admission_cover_the_frozen_table() {
    let policy = LegacySm80Policy::current();
    assert_eq!(policy.backward_cells.len(), 18);
    assert!(policy.admits(PolicyOp::Dw, PolicyDtype::Bf16, (1024, 8, 256)));
    assert!(policy.admits(PolicyOp::Dx, PolicyDtype::F16, (32, 256, 1024)));
    assert!(!policy.admits(PolicyOp::Dw, PolicyDtype::F32, (1024, 8, 256)));
    assert!(!policy.admits(PolicyOp::Dw, PolicyDtype::Bf16, (1024, 8, 257)));

    let hash = policy.digest();
    let mut changed = policy;
    changed.backward_cells[17].dims.2 += 1;
    assert_ne!(changed.digest(), hash);
    let mut changed = policy;
    changed.tile128_prefer_min_tiles += 1;
    assert_ne!(changed.digest(), hash);
    let mut changed = policy;
    changed.forward_thin_max_rows += 1;
    assert_ne!(changed.digest(), hash);
    let mut changed = policy;
    changed.forward_min_columns += 1;
    assert_ne!(changed.digest(), hash);
    let mut changed = policy;
    changed.square_tile_min += 1;
    assert_ne!(changed.digest(), hash);
    let mut changed = policy;
    changed.large_tile_min += 1;
    assert_ne!(changed.digest(), hash);
    let mut changed = policy;
    changed.reject_zero_axes = !changed.reject_zero_axes;
    assert_ne!(changed.digest(), hash);
    let mut changed = policy;
    changed.forward_thin_below_square_columns = !changed.forward_thin_below_square_columns;
    assert_ne!(changed.digest(), hash);
    let mut changed = policy;
    changed.backward_one_axis_tile64 = !changed.backward_one_axis_tile64;
    assert_ne!(changed.digest(), hash);
    let mut changed = policy;
    changed.backward_two_small_fallback = !changed.backward_two_small_fallback;
    assert_ne!(changed.digest(), hash);
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
        numeric_abi_revision: 1,
        schedule_revision: 1,
    };
    let artifacts = build_artifact_set(&[artifact(ModuleKind::LegacyCombined, 5)]).unwrap();
    GemmRouteIdentity {
        policy: GemmPolicy {
            batch_invariant: true,
            bi_tensor_cores: true,
            fast_gemm: false,
            tf32: false,
            bi_gemm_family: BiGemmFamily::Triad,
        },
        backend_set: BackendSet::TRIAD,
        numeric_contracts: NumericContractSet::TRIAD_SCALAR_FMA_V1
            .union(NumericContractSet::TRIAD_MMA_SYNC_V1),
        compiler,
        artifacts,
        policy_revision: 1,
        policy_hash: LegacySm80Policy::current().digest(),
        device: DeviceIdentity {
            compute_capability: (8, 9),
            target: CudaTarget::new("sm_89").unwrap(),
            driver: DriverIdentity {
                api_version: 13_200,
                build_sources: 1,
                build_digest: digest(8),
            },
        },
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
    value.policy.tf32 = true;
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
    value.artifacts.legacy_combined.module_kind = ModuleKind::TriadScalar;
    changed.push(value);
    let mut value = captured;
    value.artifacts.legacy_combined.artifact_kind = ArtifactKind::Cubin;
    changed.push(value);
    let mut value = captured;
    value.artifacts.legacy_combined.compile_key[0] ^= 1;
    changed.push(value);
    let mut value = captured;
    value.artifacts.legacy_combined.artifact_digest[0] ^= 1;
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
    value.state_capacity += 1;
    changed.push(value);

    for live in changed {
        let error = captured
            .ensure_current(live, "graph replay")
            .expect_err("route drift must be rejected");
        assert!(error.starts_with("graph replay"));
    }
}
