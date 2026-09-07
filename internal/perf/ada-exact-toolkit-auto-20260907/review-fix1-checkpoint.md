# Task8 fix round1 report

Status: the two Important source-review findings are repaired and the Python tool set is frozen for fix-only re-review. All nine Rust files remain byte-identical to reviewed source v1; matching final2 builds/functionals are pending. No timing has started.

## I1 — real binding schema

- `analyze.validate_binding` now requires the runner's real `binaries` map, identifies exactly one basename matching `gemm_bi_fixed_performance-<hex>`, validates its lowercase SHA256, and binds identity/result only to that role.
- Missing, ambiguous, malformed, mismatched, and wrong-role executable cases reject.
- The regression consumes the actual preserved `cuda128-binding-final1.json`; it proves a valid emitted map passes and swapping the measured performance digest with another executable fails. No synthetic top-level `binary_sha` was added.

## I2 — retained functional inventory

- The all-three-toolkit functional list now uses the existing Fixed correctness tests `fixed_sm89_rna_wide_actual_auto_hot_a_route_and_graph` and `fixed_sm89_rna_wide_actual_auto_all_cells_prefix_views_and_graph_bits`.
- `gemm_bi_fixed_correctness` is compiled, run nonignored, located, hashed, and included in each binding.
- Existing exact AUTO and half holder/AUTO checks remain on all three toolkits.
- The two Triad TF32 cohort tests run only on CUDA13.2. No Rust test body, Triad route, assertion, or production source changed.
- A host regression checks the per-toolkit functional inventory and proves every referenced binary stem belongs to the bound build inventory.

## Test-first evidence

RED command: `python3 internal/perf/ada-exact-toolkit-auto-20260907/test_validation.py`.

- Raw: `internal/perf/ada-exact-toolkit-auto-20260907/fix1-python-red.log`
- SHA256: `e520b9897da0d3633bd9d20032d62a9a2fae68d6b5e427e9593e2fe414540341`
- Exit1; exactly two new errors: real binaries-map rejected as `binary binding differs`, and missing `functional_checks`.

First GREEN after the direct repairs:

- Raw: `fix1-python-green1.log`
- SHA256: `f7018a03f10df554ea04464afa60812ffbbac98f34c5a4d6e72e4d4eb6aabb68`
- Exit0; 6/6 pass.

Final GREEN adds the actual preserved builder-binding regression, wrong-role digest swap, explicit bound-binary subset check, exact performance basename, and digest-format validation:

- Raw: `fix1-python-green2.log`
- SHA256: `147a3191a7fa4054c5a74d47798c05c808a9984283a4ad561059498865f3316d`
- Exit0; 7/7 pass.
- `python3 -m py_compile` for all four Task8 tools passes; `git diff --check` passes.

## Frozen fix1 inventory

Unchanged Rust source digest: `2ac1c93ba3e4682138a0df3341a26921438334382532d6943277268320655285`.

Unchanged nine Rust files retain their source-v1 hashes:

- `gemm_bi_fixed.rs` `d7c34ff26abb0580fcac0bbb139368ed8bc1a0fb850eab9bfa1716f15f6deed8`
- `kernel_identity.rs` `5528d6a44d0a9047080af07e868def6f4eaac24e7873e3e15e59a3f63ca772ca`
- `dispatch.rs` `48b557a8e0d97b4bc3b3a1f3ad1bd3c297d691bacf1ff2bfcb90580273cd0acd`
- `launch.rs` `816ef4474281d94e0f6462cb8695e4356f40e8ec554b2dbbfc7009fa6247b047`
- `gemm_bi_fixed_performance.rs` `72a01dd9eddc7f700b2d187b1c4e409637d2700a6f5380f2bb5cc8c741e079d2`
- `gemm_bi_fixed_sm89_exact_n64.rs` `ec97206591dbb74fd07a6f7f09a4152b5c3eb9d0a4af2e72758b013f8e9749f3`
- `gemm_bi_fixed_sm89_pipeline.rs` `26e56fb8ff5f92de10ae4103ff841dff5425d32bc59371ed9768b45961a30890`
- `gemm_bi_tf32_cohort_binding.rs` `687d495e5e46f4095324de1ac974caa8d6b785372fb467946bfd0f0ae4f4b1db`
- `fixed_sm89_toolkit_admission.rs` `2d797ba45f01c7b1983f21ce27eb39e1bd3f2c9a4f5aa82ab6abd95a5e27ddb7`

Corrected tools:

- `run.py` `d753757a717bfe18b3a28a8480f31104e252271576e45d7e53fc16396008475a`
- `analyze.py` `bb08814ac7f28cf5013b3eb5c0cb4d169c255c5b87cb7300546c36cb8ad341c9`
- `test_validation.py` `30613ac7f8a54d4a1355be039b014f7b543682775c947e6ffa203ef006a8f55f`
- Unchanged `remote.py` `a1aadafa46c41543e8665cf5ee59acb502c79c145811dd6166a99d2a627ebe28`

Historical final1 bindings/build logs remain preserved and are not relabeled. Pending: sync corrected tools, distinct final2 all-three bindings including the correctness binary, all matching functionals, source fix-only acceptance, smoke1/post101, analysis, archives/manifests, and release.
