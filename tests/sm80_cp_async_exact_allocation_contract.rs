const HARNESS: &str =
    include_str!("../tools/qualification/sm80_cp_async_exact_allocation_sanitizer.rs");

#[test]
fn exact_allocation_sanitizer_inventory_is_frozen() {
    for case in [
        "tf32_nn_exact_tail",
        "tf32_tn_exact_tail",
        "tf32_nt_exact_tail",
        "tf32_nn_splitk_exact_tail",
        "tf32_nt_splitk_exact_tail",
        "typed_tc128_a_tail",
        "typed_tc128_b_tail",
        "typed_tc64_a_tail",
        "typed_tc64_b_tail",
        "typed_thin16_a_tail",
        "typed_thin16_b_tail",
        "aligned_subview_at_allocation_end",
    ] {
        assert!(HARNESS.contains(case), "missing sanitizer case {case}");
    }
    assert!(HARNESS.contains("WeightDtype::Bf16"));
    assert!(HARNESS.contains("WeightDtype::F16"));
    assert!(HARNESS.contains("#[ignore = \"requires an SM80+ CUDA GPU under compute-sanitizer\"]"));
    assert!(HARNESS.contains(
        "--tool memcheck --report-api-errors no --target-processes all --error-exitcode 99"
    ));
}

#[test]
fn exact_allocation_harness_does_not_add_suffix_guards() {
    for forbidden in ["SUFFIX_GUARD", "RED_ZONE", "TRAILING_GUARD", "suffix_guard"] {
        assert!(
            !HARNESS.contains(forbidden),
            "exact-allocation harness contains forbidden guard marker {forbidden}"
        );
    }
}
