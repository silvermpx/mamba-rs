#[path = "../src/mamba_ssm/gpu/gemm_bi_triad/sm89_exact_f32_d128_source.rs"]
mod production;

#[test]
fn forced_routes_select_their_independent_production_symbols() {
    use production::Sm89ExactF32D128Route;

    assert_eq!(
        Sm89ExactF32D128Route::D128InDirectFold.symbol(),
        "tn_sm89_f32_d128_in_m16n16_g8_s2_cg"
    );
    assert_eq!(
        Sm89ExactF32D128Route::D128OutDirectFold.symbol(),
        "tn_sm89_f32_d128_out_m16n16_g8_s2_cg"
    );
    assert_ne!(
        Sm89ExactF32D128Route::D128InDirectFold.symbol(),
        Sm89ExactF32D128Route::D128OutDirectFold.symbol()
    );
}

#[test]
fn owner_composes_exactly_two_exports_in_their_own_namespaces() {
    production::validate_source().expect("sealed exact-F32 d128 source");
    let source = production::compose_source().expect("compose sealed d128 source");
    assert_eq!(
        production::export_inventory(&source).unwrap(),
        [production::D128_IN_SYMBOL, production::D128_OUT_SYMBOL]
    );
    for (symbol, namespace) in [
        (production::D128_IN_SYMBOL, "GemmBiTnSm89ExactF32D128In"),
        (production::D128_OUT_SYMBOL, "GemmBiTnSm89ExactF32D128Out"),
    ] {
        let unit = production::source_unit(&source, symbol).expect("sealed source unit");
        assert!(unit.starts_with(&format!("namespace {namespace} {{")));
        assert_eq!(
            unit.matches("extern \"C\"").count(),
            1,
            "{symbol} exports once"
        );
    }
    for forbidden in [
        "_test_",
        "_exp_",
        "widen_exhaustive",
        "__SM89_EXACT_F32_D128_",
        "tn_sm89_f32_d128_in_m16n16_f64fold",
        "tn_sm89_f32_d128_out_m8n16_f64fold",
    ] {
        assert!(!source.contains(forbidden), "source retained {forbidden}");
    }
}

#[test]
fn wrappers_pin_driver_abi_shape_and_exact_fold_order() {
    let source = production::compose_source().expect("compose sealed d128 source");
    for (symbol, shape, grid) in [
        (production::D128_IN_SYMBOL, (1_024, 128, 512), 256),
        (production::D128_OUT_SYMBOL, (1_024, 256, 128), 128),
    ] {
        let unit = production::source_unit(&source, symbol).expect("sealed source unit");
        let (m, k, n) = shape;
        for marker in [
            format!("constexpr int MRed = {m};"),
            format!("constexpr int KOut = {k};"),
            format!("constexpr int N = {n};"),
            "constexpr int BK = 16;".into(),
            "static_assert(Chunks == 64".into(),
            "__fmaf_rn(rows[i], columns[j], partial[i * ColsPerThread + j]);".into(),
            "sums[f] = -0.0;".into(),
            "sums[f] = __dadd_rn(sums[f], widen(value));".into(),
            "double scaled = __dmul_rn((double)alpha, sums[f]);".into(),
            "output[index] = __fadd_rn(output[index], __double2float_rn(scaled));".into(),
            "using Route = Kernel<16, 16, 8, 2, 4, 2, true>;".into(),
            "static_assert(Route::Threads == 256".into(),
            "static_assert(Route::SharedBytes == 49152".into(),
            format!("static_assert(Route::Grid == {grid}"),
        ] {
            assert!(unit.contains(&marker), "{symbol} lost {marker}");
        }
        let wrapper = &unit[unit
            .find("extern \"C\"")
            .expect("export wrapper declaration")..];
        for parameter in [
            "float* output",
            "const float* a",
            "const float* b",
            "float alpha",
            "int m",
            "int k",
            "int n",
        ] {
            assert_eq!(
                wrapper.matches(parameter).count(),
                1,
                "{symbol} ABI field changed: {parameter}"
            );
        }
        assert!(wrapper.contains("extern \"C\" __global__ __launch_bounds__(256, 2)"));
        assert!(wrapper.contains("::Route::run(output, a, b, alpha, m, k, n);"));
    }
    for forbidden in ["mma.sync", "atomic", "atom.", "redux", "(double)value"] {
        assert!(
            !source.contains(forbidden),
            "exact source contains {forbidden}"
        );
    }
}

#[test]
fn validator_rejects_foreign_duplicate_and_placeholder_exports() {
    let source = production::compose_source().expect("compose sealed d128 source");
    let duplicate = source.replacen(production::D128_OUT_SYMBOL, production::D128_IN_SYMBOL, 1);
    assert!(production::validate_source_text(&duplicate).is_err());

    for foreign in [
        "tn_sm89_f32_d128_m8n32",
        "tn_test_d128_leak",
        "tn_sm89_f32_d128_in_m16n16_f64fold",
    ] {
        let mutated = format!(
            "{source}\nextern \"C\" __global__ void {foreign}(float* output) {{ output[0] = 0.0f; }}\n"
        );
        assert!(
            production::validate_source_text(&mutated).is_err(),
            "accepted foreign export {foreign}"
        );
    }
    let placeholder = format!("{source}\n// __SM89_EXACT_F32_D128_GRID__\n");
    assert!(production::validate_source_text(&placeholder).is_err());
}

#[test]
fn specs_pin_direct_fold_geometry_and_resource_limits() {
    use production::{SM89_EXACT_F32_D128_KERNEL_SPECS, Sm89ExactF32D128KernelKind};

    assert_eq!(SM89_EXACT_F32_D128_KERNEL_SPECS.len(), 2);
    for (symbol, shape, grid) in [
        (production::D128_IN_SYMBOL, (1024, 128, 512), 256),
        (production::D128_OUT_SYMBOL, (1024, 256, 128), 128),
    ] {
        let spec = production::kernel_spec(symbol).expect("d128 symbol");
        assert_eq!(spec.kind, Sm89ExactF32D128KernelKind::DirectF64FoldFinal);
        assert_eq!(spec.shape, shape);
        assert_eq!(spec.tile, (16, 16));
        assert_eq!(spec.grid, (grid, 1, 1));
        assert_eq!(spec.block, (256, 1, 1));
        assert_eq!(spec.dynamic_shared_bytes, 49_152);
        assert_eq!(spec.static_shared_bytes, 0);
        assert_eq!(spec.local_bytes, 0);
        assert_eq!(spec.register_cap, 128);
        assert_eq!(spec.occupancy_gate, 2);
        assert_eq!(spec.chunks, 64);
        assert_eq!(spec.m_chunk, 16);
        assert_eq!(spec.abi_parameter_count, 7);
        assert_eq!(spec.abi_parameter_bytes, 40);
        let (_, k, n) = shape;
        assert_eq!(
            (k / 16) * (n / 16),
            usize::try_from(grid).unwrap(),
            "{symbol} covers the output with one 16x16 tile per CTA"
        );
    }
    assert!(production::kernel_spec("tn_sm89_f32_d128_in_m16n16_f64fold").is_none());
    assert!(production::kernel_spec("tn_sm89_f32_d128_foldpipe").is_none());
    assert!(production::kernel_spec("").is_none());
}

#[test]
fn driver_abi_is_seven_separate_arguments() {
    assert_eq!(
        production::DIRECT_FOLD_DRIVER_ABI,
        [(0, 8), (8, 8), (16, 8), (24, 4), (28, 4), (32, 4), (36, 4),]
    );
    assert_eq!(production::DIRECT_FOLD_TERMINAL_ARGUMENT, 7);
}
