#[path = "support/triad_tn_d128_direct_source.rs"]
mod frozen;
#[path = "../src/mamba_ssm/gpu/gemm_bi_triad/sm89_exact_f32_d128_source.rs"]
mod production;

const DISCOVERY_OWNER: &str = include_str!("gemm_bi_scalar_tn_underfill_direct_experiment.cu");

#[test]
fn forced_routes_select_their_independent_production_symbols() {
    use production::Sm89ExactF32D128Route;

    assert_eq!(
        Sm89ExactF32D128Route::D128InDirectFold.symbol(),
        "tn_sm89_f32_d128_in_m16n16_f64fold"
    );
    assert_eq!(
        Sm89ExactF32D128Route::D128OutDirectFold.symbol(),
        "tn_sm89_f32_d128_out_m8n16_f64fold"
    );
    assert_ne!(
        Sm89ExactF32D128Route::D128InDirectFold.symbol(),
        Sm89ExactF32D128Route::D128OutDirectFold.symbol()
    );
}

fn normalized(source: &str) -> String {
    source.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn kernel_core<'a>(source: &'a str, end_marker: &str) -> &'a str {
    let start = source
        .find("constexpr int MRed")
        .expect("exact direct-fold constants");
    let end = source[start..]
        .find(end_marker)
        .map(|offset| start + offset)
        .expect("exact direct-fold alias boundary");
    &source[start..end]
}

#[test]
fn production_routes_are_normalized_body_identical_to_retained_winners() {
    let source = production::compose_source().expect("compose sealed d128 source");
    let frozen_in = frozen::compose_source(DISCOVERY_OWNER).expect("compose retained d128-in");
    let frozen_out =
        frozen::compose_out_source(DISCOVERY_OWNER).expect("compose retained d128-out");

    let production_in = production::source_unit(&source, production::D128_IN_SYMBOL)
        .expect("production d128-in source unit");
    let production_out = production::source_unit(&source, production::D128_OUT_SYMBOL)
        .expect("production d128-out source unit");

    assert_eq!(
        normalized(kernel_core(production_in, "using Retained")),
        normalized(kernel_core(&frozen_in, "using M16N16")),
    );
    assert_eq!(
        normalized(kernel_core(production_out, "using Retained")),
        normalized(kernel_core(&frozen_out, "using M16N16")),
    );

    for (unit, alias, shared_bytes, stage, grid) in [
        (
            production_in,
            "Kernel<16, 16, 64, 0, false, true>",
            4096,
            512,
            256,
        ),
        (
            production_out,
            "Kernel<8, 16, 64, 0, false, true>",
            3072,
            384,
            256,
        ),
    ] {
        assert!(unit.contains(&format!("using Retained = {alias};")));
        assert!(unit.contains(&format!("Retained::SharedBytes == {shared_bytes}")));
        assert!(unit.contains(&format!("Retained::Stage == {stage}")));
        assert!(unit.contains(&format!(
            "Retained::RowTiles * Retained::ColumnTiles == {grid}"
        )));
    }
}

#[test]
fn owner_composes_exactly_two_retained_exports_and_no_alternatives() {
    production::validate_source().expect("sealed exact-F32 d128 source");
    let source = production::compose_source().expect("compose sealed d128 source");
    assert_eq!(
        production::export_inventory(&source).unwrap(),
        [production::D128_IN_SYMBOL, production::D128_OUT_SYMBOL]
    );
    for forbidden in [
        "_test_",
        "_exp_",
        "foldpipe",
        "M8N32",
        "M16N16T128",
        frozen::M8N16_SYMBOL,
        frozen::OUT_M16N16_SYMBOL,
        "__SM89_EXACT_F32_D128_IN_SYMBOL__",
        "__SM89_EXACT_F32_D128_OUT_SYMBOL__",
    ] {
        assert!(!source.contains(forbidden), "source retained {forbidden}");
    }
}

#[test]
fn wrappers_pin_driver_abi_shape_and_exact_fold_order() {
    let source = production::compose_source().expect("compose sealed d128 source");
    for (symbol, shape, shared_bytes) in [
        (production::D128_IN_SYMBOL, (1_024, 128, 512), 4096),
        (production::D128_OUT_SYMBOL, (1_024, 256, 128), 3072),
    ] {
        let unit = production::source_unit(&source, symbol).expect("sealed source unit");
        let (m, k, n) = shape;
        for marker in [
            format!("constexpr int MRed = {m};"),
            format!("constexpr int KOut = {k};"),
            format!("constexpr int N = {n};"),
            "static_assert(Chunks == 64".into(),
            "partial[owned] = __fmaf_rn(".into(),
            "sums[owned] = __dadd_rn(sums[owned], value);".into(),
            "double scaled = __dmul_rn((double)alpha, sums[owned]);".into(),
            "__double2float_rn(scaled)".into(),
            format!("Retained::SharedBytes == {shared_bytes}"),
        ] {
            assert!(unit.contains(&marker), "{symbol} lost {marker}");
        }
        let wrapper = &unit[unit
            .find("extern \"C\"")
            .expect("retained wrapper declaration")..];
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
        assert!(wrapper.contains("extern \"C\" __global__ __launch_bounds__(64, 4)"));
    }
    for forbidden in ["mma.sync", "atomic", "atom.", "redux"] {
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
        "tn_sm89_f32_d128_foldpipe",
    ] {
        let mutated = format!(
            "{source}\nextern \"C\" __global__ void {foreign}(float* output) {{ output[0] = 0.0f; }}\n"
        );
        assert!(
            production::validate_source_text(&mutated).is_err(),
            "accepted foreign export {foreign}"
        );
    }
}

#[test]
fn retained_specs_pin_direct_fold_geometry_and_resource_limits() {
    use production::{SM89_EXACT_F32_D128_KERNEL_SPECS, Sm89ExactF32D128KernelKind};

    assert_eq!(SM89_EXACT_F32_D128_KERNEL_SPECS.len(), 2);
    for (symbol, shape, tile, shared_bytes, register_cap) in [
        (
            production::D128_IN_SYMBOL,
            (1024, 128, 512),
            (16, 16),
            4096,
            112,
        ),
        (
            production::D128_OUT_SYMBOL,
            (1024, 256, 128),
            (8, 16),
            3072,
            96,
        ),
    ] {
        let spec = production::kernel_spec(symbol).expect("retained d128 symbol");
        assert_eq!(spec.kind, Sm89ExactF32D128KernelKind::DirectF64FoldFinal);
        assert_eq!(spec.shape, shape);
        assert_eq!(spec.tile, tile);
        assert_eq!(spec.grid, (256, 1, 1));
        assert_eq!(spec.block, (64, 1, 1));
        assert_eq!(spec.dynamic_shared_bytes, shared_bytes);
        assert_eq!(spec.static_shared_bytes, 0);
        assert_eq!(spec.local_bytes, 0);
        assert_eq!(spec.register_cap, register_cap);
        assert_eq!(spec.occupancy_gate, 8);
        assert_eq!(spec.chunks, 64);
        assert_eq!(spec.m_chunk, 16);
        assert_eq!(spec.abi_parameter_count, 7);
        assert_eq!(spec.abi_parameter_bytes, 40);
    }
    assert!(production::kernel_spec(frozen::M8N16_SYMBOL).is_none());
    assert!(production::kernel_spec("tn_sm89_f32_d128_foldpipe").is_none());
    assert!(production::kernel_spec("").is_none());
}

#[test]
fn retained_driver_abi_is_seven_separate_arguments() {
    assert_eq!(
        production::DIRECT_FOLD_DRIVER_ABI,
        [(0, 8), (8, 8), (16, 8), (24, 4), (28, 4), (32, 4), (36, 4),]
    );
    assert_eq!(production::DIRECT_FOLD_TERMINAL_ARGUMENT, 7);
}
