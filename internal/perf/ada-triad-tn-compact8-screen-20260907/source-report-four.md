# TN compact four-warp source report

Scope: one test-only source-composer ablation on base `070a4f54`. Only
`tests/support/triad_tn_compact_source.rs` changed; the existing CUDA XOR
helper and production/runtime sources were untouched.

## Interface and source identity

- `FOUR_WARP_SYMBOL` =
  `gemm_bi_tn_test_compact_four_warp_sm80_mma_tf32_v1_m128n64_bk32_s2`
- `four_warp_candidate_source(production: &str) -> Result<String, String>`
- support source SHA256:
  `cfcbe3146b513fdf51ec82e95059d535df0a8fa598f583adfdd41c0bcebd1924`
- unchanged CUDA helper SHA256:
  `0e9712b436f38f65ea3d035ced5bd83669dbad784f60a6496e9a34f39172a495`

The new composer first uses the frozen eight-warp composer, then reverses
exactly its accumulator-ownership and compute/row-ownership transformations.
It consequently retains packed 49,152-byte S2 storage and identical XOR A/B
slots while restoring `MAtoms=4`, `warp < 4`, and 64-row warp bands. Two
further exact-one substitutions rename only the target export and matching
signature assertion. Missing or duplicate inherited anchors continue to fail
closed.

The existing eight-warp output was rendered through the same six-part NVRTC
composition used by the discovery harness after this edit. Its SHA256 remains
byte-identical:
`14bf43fb3303ee23fd5fb14ffc6f662db890f9dbf5ca97edb014cd46465d3b28`.

## Native TDD

RED used real `rustc --edition 2024 --test` and ran
`tests::four_warp_candidate_keeps_compact_storage_and_restores_original_ownership`.
The placeholder returned the eight-warp source; the exact original ownership
anchor count was zero, so the test failed with exit 101.

GREEN rebuilt and ran the complete native test binary: three passed, zero
failed, exit 0. The focused assertion checks compact storage/XOR retention,
exact restoration and removal of both ownership blocks, and the distinct
DEFINE/ASSERT pair. Existing target-count failure tests also remain green.
`rustfmt --edition 2024 --check` and `git diff --check` passed.

No Cargo, CUDA, SSH, GPU, production edit, runtime-harness edit, or commit was
performed by this source-helper owner.
