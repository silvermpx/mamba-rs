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

Historical final1 bindings/build logs remain preserved and are not relabeled.

## Final2 matching builds and functionals

Every command used the single `env_for(toolkit, "final2")` constructor. Each toolkit passed: library 647/0/46 ignored; focused 14/14; performance nonignored 67/67; exact nonignored 2/2; half nonignored 2/2; TF32 cohort nonignored 0/0 with 3 ignored; Fixed correctness nonignored 1/1. All build wrapper/outer exits are zero.

- CUDA12.8 binding SHA `fe0773cc05411c0b4b822b607be8d696eb4299f1b3a2a0b52b12d94752c2b169`; performance binary `313f4f6c8a2e30f2bed5ce67b65135b99344674221156c39b8661e8c288aa246`; correctness binary `740a82666e12263f481d0ae11f4c2a52c0789347b5e74be71b013690e28ac02b`.
- CUDA13.0 binding SHA `c613808a5fd1c287c80ea90c5b58508f91d387a915b991165bc4fba9631be9b2`; performance binary `1b8b31e201c7d252e52a418637ebb5706651ceeb0c56dc42c0dfce145decc170`; correctness binary `be445933803b00f8a1fe3eb8c21a827711c975ecef7f80e01e621224398037a0`.
- CUDA13.2 binding SHA `486ede4eb3d5bc1c7d111d0426fbb548e0f331a868d85225c1be7ec95df2f3c6`; performance binary `4ecefea9345226073e9671017bb14450ebed562be8c5c551a2e6d9de3859868d`; correctness binary `9966b63b49711d5aa9af59f299b6c5e483b0ad644557de758112e2840b0b96d4`.

Each live functional attempt has a strict quiet PRE, same UUID/no-apps POST, all inner exits zero, wrapper complete, and outer SSH exit zero. CUDA12.8/13.0 each pass five checks: exact AUTO full/prefix/view/graph bits; half holder; half AUTO; Fixed RNA hot-A; Fixed RNA all-cells/prefix/views/graph bits. CUDA13.2 passes the same five plus the two historically qualified Triad TF32 cohort/bias checks. Result SHAs are 12.8 `9a1e895f5cc8cc70709dc2ffc2238391f13e464a28452010b49952752dea448c`, 13.0 `2a87f000ba073bd0dd84e6dcd750a07edcc1dd3f36c5eb831d8b2dd8dcb95754`, 13.2 `eda7997ca5a9137897e2e0266c9d66383da08792c5a7184f192c9216fa59d27d`.

Compiler source digest is `205f58b56429b8e74f3ac1e7ab9f0cf6a9bf193ffeb87b254bd3ab00e6b02ca5` for all three. Toolkit-specific invocation/Fixed artifact digests are:

- 12.8 `57c3e4f2177104f137d432f19398a4a36799aca4c264634a1ab25d17dbf422ba` / `60977db33de28d807ac7dd3dafe2d176914f7b7eab589917a47988050712352c`.
- 13.0 `adee1f8b255bacfce8921820d9397a0a22a0dbaac735759fdcd222a453176155` / `d1aa6e33a612d99cd44a1e9c7eebe495c05eeef2212bf2cf4296db55eedb86a9`.
- 13.2 `3a5fcbc3e7fef62dbb3542118d6d99d4ffba12b0cac04103b58f27ab4be36fb0` / `8b89aadf7d456b78ab249ff614568730946006eefd7143e35bce1e158741aff1`.

Fix1 re-review SHA `7dee4a81a2c9b9ff5e8aacd6ec1c8222ff5841054e3f251d1d10f4c635fa7b42` accepts both findings with no new findings. Pending: root functional identity acceptance, smoke1/post101, analysis, archives/manifests, and release.
