#[path = "support/triad_f32_tn_direct_d768_out_bk16_source.rs"]
mod frozen_d768_out;
#[path = "support/triad_f32_tn_copyplan_dual_chunk_source.rs"]
mod frozen_dual;
#[path = "support/triad_f32_tn_direct_prism_bk16_source.rs"]
mod frozen_prism;
#[path = "../src/mamba_ssm/gpu/gemm_bi_triad/sm89_exact_f32_source.rs"]
mod production;

fn normalized(source: &str) -> String {
    source.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn production_named(source: String, frozen: &str, production: &str) -> String {
    source.replace(frozen, production)
}

#[test]
fn production_units_are_normalized_body_identical_to_frozen_winners() {
    let source = production::compose_source().expect("compose sealed production source");
    let fused = frozen_dual::compose_fused_source();
    let fused = &fused[fused.find("struct DualChunkParams").unwrap()..];
    assert_eq!(
        normalized(
            production::source_unit(&source, production::D768_IN_FUSED_SYMBOL)
                .expect("production d768-in fused source unit"),
        ),
        normalized(&production_named(
            fused.to_owned(),
            frozen_dual::DUAL_FUSED_SYMBOL,
            production::D768_IN_FUSED_SYMBOL,
        )),
    );
    assert_eq!(
        normalized(
            production::source_unit(&source, production::D768_OUT_RAW_SYMBOL)
                .expect("production d768-out raw source unit"),
        ),
        normalized(&production_named(
            frozen_d768_out::compose_source(),
            frozen_d768_out::SYMBOL,
            production::D768_OUT_RAW_SYMBOL,
        )),
    );
    assert_eq!(
        normalized(
            production::source_unit(&source, production::PRISM_RAW_SYMBOL)
                .expect("production Prism raw source unit"),
        ),
        normalized(&production_named(
            frozen_prism::compose_source(),
            frozen_prism::SYMBOL,
            production::PRISM_RAW_SYMBOL,
        )),
    );
}

#[test]
fn production_owner_is_sealed_to_exactly_three_large_tn_exports() {
    production::validate_source().expect("sealed exact-F32 source");
    let source = production::compose_source().expect("compose sealed production source");
    assert_eq!(
        production::export_inventory(&source).unwrap(),
        [
            production::D768_IN_FUSED_SYMBOL,
            production::D768_OUT_RAW_SYMBOL,
            production::PRISM_RAW_SYMBOL,
        ]
    );
    for forbidden in [
        "_test_",
        "_exp_",
        "__DUAL_SYMBOL__",
        "__DUAL_EPILOGUE__",
        "__SM89_EXACT_F32_DIRECT_SYMBOL__",
    ] {
        assert!(
            !source.contains(forbidden),
            "production source retained {forbidden}"
        );
    }
}

#[test]
fn sealed_validator_rejects_discovery_markers_and_inventory_mutations() {
    let source = production::compose_source().expect("compose sealed production source");
    let discovery = source.replacen(production::PRISM_RAW_SYMBOL, "gemm_bi_tn_test_leak", 1);
    assert!(production::validate_source_text(&discovery).is_err());

    let duplicate = source.replacen(
        production::PRISM_RAW_SYMBOL,
        production::D768_OUT_RAW_SYMBOL,
        1,
    );
    assert!(production::validate_source_text(&duplicate).is_err());
}

#[test]
fn source_specs_pin_shape_partition_abi_and_resource_contracts() {
    use production::Sm89ExactF32KernelKind::{DirectSplitMRaw, DualChunkFusedFinalize};

    let specs = production::SM89_EXACT_F32_KERNEL_SPECS;
    assert_eq!(specs.len(), 3);
    assert_eq!(
        specs[0],
        production::Sm89ExactF32KernelSpec {
            symbol: production::D768_IN_FUSED_SYMBOL,
            kind: DualChunkFusedFinalize,
            shape: (2_048, 768, 3_072),
            grid: (576, 1, 1),
            block: (128, 1, 1),
            dynamic_shared_bytes: 0,
            static_shared_bytes: 32_768,
            register_cap: 168,
            occupancy_gate: 3,
            chunks: 2,
            m_chunk: 1_024,
            abi_parameter_count: 4,
            abi_parameter_bytes: 56,
        }
    );
    assert_eq!(
        specs[1],
        production::Sm89ExactF32KernelSpec {
            symbol: production::D768_OUT_RAW_SYMBOL,
            kind: DirectSplitMRaw,
            shape: (2_048, 1_536, 768),
            grid: (288, 1, 4),
            block: (128, 1, 1),
            dynamic_shared_bytes: 0,
            static_shared_bytes: 16_384,
            register_cap: 128,
            occupancy_gate: 4,
            chunks: 4,
            m_chunk: 512,
            abi_parameter_count: 7,
            abi_parameter_bytes: 40,
        }
    );
    assert_eq!(
        specs[2],
        production::Sm89ExactF32KernelSpec {
            symbol: production::PRISM_RAW_SYMBOL,
            kind: DirectSplitMRaw,
            shape: (4_621, 384, 1_928),
            grid: (186, 1, 6),
            block: (128, 1, 1),
            dynamic_shared_bytes: 0,
            static_shared_bytes: 16_384,
            register_cap: 128,
            occupancy_gate: 4,
            chunks: 6,
            m_chunk: 784,
            abi_parameter_count: 7,
            abi_parameter_bytes: 40,
        }
    );
}

#[test]
fn source_contract_keeps_exact_fma_and_finalize_order_without_reduced_math() {
    let source = production::compose_source().expect("compose sealed production source");
    let fused = production::source_unit(&source, production::D768_IN_FUSED_SYMBOL).unwrap();
    for marker in [
        "DUAL_RUN_CHAIN(acc0, a, b, k0);",
        "DUAL_RUN_CHAIN(acc1, a1, b1, k1);",
        "__fmaf_rn(",
        "__dadd_rn((double)acc0[i][j], (double)acc1[i][j])",
        "__dmul_rn((double)alpha, sum)",
        "__double2float_rn",
        "output[idx] = __fadd_rn(output[idx], update);",
    ] {
        assert!(fused.contains(marker), "fused source lost {marker}");
    }
    assert!(
        fused.find("DUAL_RUN_CHAIN(acc0, a, b, k0);").unwrap()
            < fused.find("DUAL_RUN_CHAIN(acc1, a1, b1, k1);").unwrap()
    );

    for symbol in [
        production::D768_OUT_RAW_SYMBOL,
        production::PRISM_RAW_SYMBOL,
    ] {
        let raw = production::source_unit(&source, symbol).unwrap();
        assert!(raw.contains("__fmaf_rn(a_reg[i], b_reg[j], acc[i][j])"));
        assert!(raw.contains("partial_chunk[(long long)row * N + column] = acc[i][j]"));
        assert!(!raw.contains("__dadd_rn"));
        assert!(!raw.contains("__double2float_rn"));
    }
    for forbidden in ["mma.sync", "atomic", "atom.", "redux"] {
        assert!(
            !source.contains(forbidden),
            "exact source contains forbidden {forbidden}"
        );
    }
}
