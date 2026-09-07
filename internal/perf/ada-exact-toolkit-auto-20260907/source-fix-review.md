# Task8 fix1 scoped re-review

Date: 2026-09-07

## Review scope and basis

This is the requested fix-only re-review of
`ada-exact-toolkit-auto-task8-fix1.diff` against the two accepted Important
findings from the Task8 source review. The complete three-Python-file package
matches SHA-256
`71cee10764b4cfbcf8c8e855a6ade3eb7037beedb26c6f35850729b933513e5e`;
the immutable fix checkpoint matches
`c56d3693e603af7a61e02c0a21a07fd674c48de92597a25b5be1b92400fa5c76`.
The corrected `run.py`, `analyze.py`, and `test_validation.py` hashes match the
checkpoint, as does unchanged `remote.py`. I did not repeat the accepted
nine-Rust-file/source-v1 review or run a GPU, SSH, Cargo, build, or owner
suite.

## Finding verdicts

1. **ADDRESSED — the analyzer consumes the builder's real `binaries` map and
   binds the measured performance role.** `validate_binding` now requires a
   dictionary, selects paths whose basename exactly matches
   `gemm_bi_fixed_performance-<lowercase hex>`, requires exactly one such
   entry, validates its digest as 64 lowercase hexadecimal characters, and
   compares that digest to both the identity and result
   (`internal/perf/ada-exact-toolkit-auto-20260907/analyze.py:159-181`). It no
   longer reads the nonexistent top-level `binary_sha`. The primary binding
   test now uses a realistic `binaries` map and covers missing, ambiguous, and
   malformed/mismatched performance entries
   (`test_validation.py:133-180`). A separate regression loads the preserved
   actual `cuda128-binding-final1.json`, accepts its performance role, then
   proves that swapping the performance digest with another executable is
   rejected (`test_validation.py:182-212`). This closes the real CLI/schema
   mismatch without adding a synthetic legacy field.

2. **ADDRESSED — all three functional inventories use the retained Fixed RNA
   tests, while Triad cohort checks remain 13.2-only.** `functional_checks`
   includes the exact AUTO and half holder/AUTO checks plus
   `fixed_sm89_rna_wide_actual_auto_hot_a_route_and_graph` and
   `fixed_sm89_rna_wide_actual_auto_all_cells_prefix_views_and_graph_bits` for
   every supported toolkit; it appends the two historical Triad tests only
   when `toolkit == "13.2"`
   (`internal/perf/ada-exact-toolkit-auto-20260907/run.py:167-212`). The
   existing build constructor now compiles `gemm_bi_fixed_correctness`, and
   `BOUND_BINARY_STEMS` causes that executable to be found, hashed, and stored
   in every binding (`run.py:25-31`, `:286-323`). The functional operation
   consumes the same centralized inventory and bound stem map
   (`run.py:340-362`). The host regression checks both the Fixed-everywhere
   and Triad-only-13.2 sets and proves that every referenced stem is in the
   build inventory (`test_validation.py:214-241`). No Rust test body or
   production route is changed by this fix.

## New breakage in fix diff

No new Critical, Important, or Minor breakage was found in the scoped fix.

The preserved test-first logs corroborate the patch: the RED log has exactly
the two intended errors (old binding rejection and absent functional inventory),
while the final GREEN log has all seven tests passing. Their SHA-256 values
match the checkpoint (`e520b9897da0d3633bd9d20032d62a9a2fae68d6b5e427e9593e2fe414540341`
and `147a3191a7fa4054c5a74d47798c05c808a9984283a4ad561059498865f3316d`).
A scoped `git diff --check` on the three corrected Python files is clean.

## Out-of-scope observations

The matching final2 builds and live functionals were still in progress during
this review, and smoke1/post101 timing remained held. Their eventual evidence,
binary/source closure, Ada exclusivity, and release/archive manifests are not
accepted or implied by this source-fix verdict. The unchanged Rust/CUDA source,
Task7 evidence, and previously accepted Task8 source-v1 portions were not
reopened.

## Final round verdict

**Both Important findings are addressed, with no new Critical, Important, or
Minor breakage in fix1.** The corrected Python source is acceptable for the
matching functional/evidence phase. This verdict does not itself accept any
GPU functional, smoke1, post101, timing admission, archive, or final release
evidence.
