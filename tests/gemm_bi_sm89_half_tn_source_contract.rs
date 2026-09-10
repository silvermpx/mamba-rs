#[path = "support/triad_half_tn_compact_source.rs"]
mod frozen_compact;
#[path = "support/triad_half_tn_vec2_epilogue_source.rs"]
mod frozen_vec2;
#[path = "../src/mamba_ssm/gpu/gemm_bi_triad/sm89_half_tn_source.rs"]
mod production;

const SM80_OWNER: &str = include_str!("../kernels/gemm_bi_triad/sm80.cu");

#[test]
#[cfg(feature = "cuda")]
fn production_half_module_registry_includes_all_retained_tn_winners() {
    use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
        SM89_HALF_AUTO_CELLS, SM89_HALF_KERNEL_SPECS, Sm89HalfRoute,
    };
    use mamba_rs::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp;

    assert_eq!(SM89_HALF_KERNEL_SPECS.len(), 10);
    assert_eq!(SM89_HALF_AUTO_CELLS.len(), 18);
    assert_eq!(
        SM89_HALF_AUTO_CELLS
            .iter()
            .filter(|(op, _, _, _)| *op == ResolvedGemmOp::Tn)
            .count(),
        6,
    );
    assert!(SM89_HALF_KERNEL_SPECS.iter().any(|spec| {
        spec.route == Sm89HalfRoute::TnM64N64Bk64S2CompactBxor
            && spec.symbol == production::COMPACT_F16_SYMBOL
    }));
    assert!(SM89_HALF_KERNEL_SPECS.iter().any(|spec| {
        spec.route == Sm89HalfRoute::TnM64N64Bk64S2RegpipeVec2
            && spec.symbol == production::REGPIPE_VEC2_BF16_SYMBOL
    }));
}

fn normalized(source: &str) -> String {
    source.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[test]
fn production_families_are_normalized_body_identical_to_retained_winners() {
    let source = production::compose_source().expect("compose sealed SM89 half-TN source");
    let retained_compact = frozen_compact::candidate_source(SM80_OWNER)
        .expect("compose retained compact half-TN source");
    let retained_vec2 =
        frozen_vec2::candidate_source(SM80_OWNER).expect("compose retained vec2 half-TN source");

    let production_compact = production::family_source(&source, production::COMPACT_F16_SYMBOL)
        .expect("production compact family source");
    let production_vec2 = production::family_source(&source, production::REGPIPE_VEC2_F16_SYMBOL)
        .expect("production regpipe+vec2 family source");

    assert_eq!(
        normalized(&production_compact.replace(
            production::COMPACT_SYMBOL_PREFIX,
            frozen_compact::SYMBOL_PREFIX,
        ),),
        normalized(&retained_compact),
    );
    assert_eq!(
        normalized(&production_vec2.replace(
            production::REGPIPE_VEC2_SYMBOL_PREFIX,
            frozen_vec2::SYMBOL_PREFIX,
        ),),
        normalized(&retained_vec2),
    );
}

#[test]
fn owner_exports_only_the_four_retained_dtype_families() {
    production::validate_source().expect("sealed SM89 half-TN source");
    let source = production::compose_source().expect("compose sealed SM89 half-TN source");
    assert_eq!(
        production::export_inventory(&source).unwrap(),
        [
            production::COMPACT_BF16_SYMBOL,
            production::COMPACT_F16_SYMBOL,
            production::REGPIPE_VEC2_BF16_SYMBOL,
            production::REGPIPE_VEC2_F16_SYMBOL,
        ]
    );
    for forbidden in [
        "_test_",
        "_exp_",
        "gemm_bi_tn_tc64_",
        "bk32",
        "m96n64",
        "m64n96",
        "warpspecialized",
        "full_tile",
    ] {
        assert!(!source.contains(forbidden), "source retained {forbidden}");
    }
}

#[test]
fn standalone_composition_supplies_each_retained_dependency_once() {
    let source = production::compose_source().expect("compose sealed SM89 half-TN source");
    for definition in [
        "float to_f(float v)",
        "float to_f(__nv_bfloat16 v)",
        "float to_f(__half v)",
        "bool gemm_bi_is_aligned_16(const void* ptr)",
        "int gemm_bi_cp_async_valid_elems(",
        "const T* gemm_bi_cp_async_source(",
        "void gemm_bi_cp_async_16_zfill(",
        "void gemm_bi_accumulate_float2_or_scalar(",
    ] {
        assert_eq!(
            source.matches(definition).count(),
            1,
            "dependency definition count changed for {definition}",
        );
    }
    let fragment = production::compose_fragment_for_sm89_half()
        .expect("compose half-TN fragment for existing owner");
    assert!(!fragment.contains("float to_f(float v)"));
    assert_eq!(fragment.matches("bool gemm_bi_is_aligned_16").count(), 1);
    assert_eq!(
        fragment
            .matches("void gemm_bi_accumulate_float2_or_scalar(")
            .count(),
        1,
    );
}

#[test]
fn four_specs_pin_tn_driver_abi_geometry_and_resource_bounds() {
    use production::{SM89_HALF_TN_KERNEL_SPECS, Sm89HalfTnKernelKind};

    assert_eq!(SM89_HALF_TN_KERNEL_SPECS.len(), 4);
    for (symbol, kind) in [
        (
            production::COMPACT_F16_SYMBOL,
            Sm89HalfTnKernelKind::CompactBxor,
        ),
        (
            production::COMPACT_BF16_SYMBOL,
            Sm89HalfTnKernelKind::CompactBxor,
        ),
        (
            production::REGPIPE_VEC2_F16_SYMBOL,
            Sm89HalfTnKernelKind::RegpipeVec2,
        ),
        (
            production::REGPIPE_VEC2_BF16_SYMBOL,
            Sm89HalfTnKernelKind::RegpipeVec2,
        ),
    ] {
        let spec = production::kernel_spec(symbol).expect("retained half-TN symbol");
        assert_eq!(spec.kind, kind);
        assert_eq!(spec.tile, (64, 64));
        assert_eq!(spec.bk, 64);
        assert_eq!(spec.stages, 2);
        assert_eq!(spec.block, (128, 1, 1));
        assert_eq!(spec.dynamic_shared_bytes, 0);
        assert_eq!(spec.static_shared_bytes, 32_768);
        assert_eq!(spec.local_bytes, 0);
        assert_eq!(spec.register_cap, 128);
        assert_eq!(spec.occupancy_gate, 3);
        assert_eq!(spec.abi_parameter_count, 7);
        assert_eq!(spec.abi_parameter_bytes, 40);
    }
    assert_eq!(
        production::HALF_TN_DRIVER_ABI,
        [(0, 8), (8, 8), (16, 8), (24, 4), (28, 4), (32, 4), (36, 4)]
    );
    assert_eq!(production::HALF_TN_TERMINAL_ARGUMENT, 7);
    assert!(production::kernel_spec("gemm_bi_tn_tc64_f16").is_none());
    assert!(production::kernel_spec("").is_none());
}

#[test]
fn validation_rejects_duplicate_foreign_and_placeholder_exports() {
    let source = production::compose_source().expect("compose sealed SM89 half-TN source");
    let duplicate = source.replacen(
        production::REGPIPE_VEC2_SYMBOL_PREFIX,
        production::COMPACT_SYMBOL_PREFIX,
        1,
    );
    assert!(production::validate_source_text(&duplicate).is_err());

    for foreign in [
        "gemm_bi_tn_sm89_m64n64_bk64_s2_plain_regpipe_v1_f16",
        "gemm_bi_tn_test_leak_f16",
        "gemm_bi_tn_sm89_m96n64_bk64_s2_v1_bf16",
    ] {
        let mutated = format!(
            "{source}\nextern \"C\" __global__ void {foreign}(float* output) {{ output[0] = 0.0f; }}\n"
        );
        assert!(
            production::validate_source_text(&mutated).is_err(),
            "accepted foreign export {foreign}",
        );
    }
}
