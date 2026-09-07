# TN compact8 source-helper report

Scope: test-only source composition for the bounded TN Prism discovery arm.
No production source, dispatcher, module holder, or runtime harness was edited.

## Files

- `tests/support/triad_tn_compact_source.rs`
  SHA256 `f6d10a1534184dca67609b5cb836268d50c7f795fa53c8fd39ab55508b1e58f0`
- `tests/gemm_bi_tf32_tn_compact_xor.cuh`
  SHA256 `0e9712b436f38f65ea3d035ced5bd83669dbad784f60a6496e9a34f39172a495`

The public test-helper API is `candidate_source(production: &str)` plus
`SYMBOL`. The symbol is
`gemm_bi_tn_test_compact_eight_warp_sm80_mma_tf32_v1_m128n64_bk32_s2`.

The composer applies eight exact-one substitutions: storage declaration, A
slot, B slot, TN S2 storage assertion, accumulator ownership, compute/row
ownership, target export, and target signature assertion. Missing or duplicate
anchors fail closed with the label and observed count. Only the exact
`SgbTf32Tn/M128/N64/S2` template receives packed 128/64 strides, XOR slots,
`MAtoms=2`, all-eight-warp compute, and 32-row warp bands. Other source text is
retained. The prepended helper implements
`axis ^ ((reduction & 3) << 3)`.

## Native TDD evidence

RED command:

`rustc --edition 2024 --test tests/support/triad_tn_compact_source.rs -o <temp>/triad_tn_compact_source_tests && <temp>/triad_tn_compact_source_tests --exact tests::candidate_is_exactly_tn_m128n64_s2_scoped --nocapture`

The real native test compiled, then failed as intended with exit 101 because
the initial composer returned `TN compact source composer is not implemented`.

GREEN command used the same native `rustc --test` build and ran the complete
binary with `--nocapture`: two passed, zero failed, exit 0. It covers the exact
target-scoped output and missing/duplicate target export counts. `rustfmt
--edition 2024` and `git diff --check` also passed.

This is source-composition evidence only. CUDA compilation, resource census,
bits, graph replay, and timing remain the sole GPU owner's separate gates.
