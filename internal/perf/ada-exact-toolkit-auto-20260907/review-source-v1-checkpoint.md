# Task8 Ada exact toolkit AUTO44 report

Status at source checkpoint: implementation and host validation are source-stable; matching final builds, three-toolkit functional work, smoke1, post101, archives/manifests, and release are pending. No speed timing has started.

## Scope and base

- Base: `b78aebf466b72428eaaaf2654a7e671949b62d20`.
- Production routing change: `fixed_sm89_exact_n64_auto_eligible` admits only NVRTC `(12,8)`, `(13,0)`, `(13,2)` for the existing exact A/B/D/E × bias CopyPlan whitelist and unchanged device/holder/policy/dtype/alignment guards.
- Epoch: tuning revision `43 -> 44`; numeric ABI remains `5`, schedule remains `8`.
- Post mode: actual public AUTO (required returned enum `F32Sm89N64CopyPlan`) versus explicit forced Legacy and unchanged cuBLAS Fast. Historical Task7 mode remains a distinct revision43 schema and fails closed under revision44.
- Deferred Task7 unused parent imports were removed. No kernel, loader, compiler option, numerical association, TF32, half, selector order, or forced API behavior changed.

## Exclusive Ada PRE

- Raw: `internal/perf/ada-exact-toolkit-auto-20260907/pre-task-strict-pre.log`
- SHA256: `f5685c16d328cefc47a44a2295bdc29a1e487c9f826efe25fc3c37dcb7e6d5b0`
- UTC: `2026-09-07T07:21:23Z`
- UUID `GPU-d1edd7be-e88d-aed6-047d-622163306f0e`, RTX 6000 Ada, CC8.9, GPU/memory `0%/0%`, no compute apps; both queries exit0.

## TDD history

1. Routing/epoch RED, before production changes:
   `python3 /root/evidence-ada-exact-toolkit-auto-20260907/run.py library 12.8 red1 dev1`
   - Raw `cuda128-red1/library.log`, SHA256 `efc11fabaff4ebca6baf3222db25b1419e1adefb7af2744956f816a004224ef2`.
   - Exit101; 645 passed, exactly 2 expected failures: all new 12.8 positive routes and captured43/current44 graph rejection. 46 ignored.
2. Routing/epoch GREEN after the minimal predicate and epoch change:
   same constructor, `library 12.8 green1 dev1`.
   - Raw `cuda128-green1/library.log`, SHA256 `cdf762a69a8981f93eb6065075e262dd3d5e7173f5f1ec9d4cdc25a817066287`.
   - Exit0; 647 passed, 0 failed, 46 ignored.
3. Post-AUTO support RED before implementation:
   `python3 .../run.py focused 12.8 red2 dev1`.
   - Raw `cuda128-red2/focused.log`, SHA256 `5012014a8b0adf4297403d3e836cb3642be57b021fdb128724814bd269529e76`.
   - Exit101; intended missing post parser/mode/entry symbols.
4. Post-AUTO support GREEN:
   `python3 .../run.py focused 12.8 green2 dev1`.
   - Raw SHA256 `f4e17c7ed0579e16193cac4818a0373ebe630c293c1288314d6240ffb5ba2a31`.
   - Exit0; 14 passed.
5. Pre-format support emitter/basis GREEN:
   `python3 .../run.py focused 12.8 green3 dev1`.
   - Mirrored raw `cuda128-green3-focused.log`, SHA256 `cd0e3381c1b1999d14728f0cff0b1caf3a1b2f6dfea8817ca486518961b02796`.
   - Exit0; 14 passed, 0 failed.
6. Final formatted-source focused GREEN:
   `python3 /root/evidence-ada-exact-toolkit-auto-20260907/run.py focused 12.8 sourcegreen1 final1`.
   - Raw `cuda128-sourcegreen1/focused.log`, SHA256 `c3af89ff295a50846e72f8f748593e816daca592a6afddee3861b5cf0a13ebc1`.
   - Exit0; 14 passed, 0 failed, 119 filtered out.
7. Analyzer RED before implementation: `python-red.log`, SHA256 `9ebaf11fb323416497f587cb67a1a9770e5f8b58e14899095e3b2bbda25c72c3`, exit1 (missing analyzer).
8. Promotion-basis RED: `python-red-basis.log`, SHA256 `3e0ab83fbc0d335130ace3dbcbc291fbfa5bda3190066b52cbda7ee8cf3ad549`, exit1 (basis not yet validated).
9. Analyzer GREEN: `python-green2.log`, SHA256 `17039f66b9a2c8d0f1a6a7cb0297b7193c7491f1ec708694eb150a4fb2ea3240`, exit0; 5 tests pass. A source-stable replay on the exact hashes below also passes 5/5.

All actual failures above are retained; none was retried as a timing lottery.

## Source-stable checkpoint

Direct `rustfmt --edition 2024 --config skip_children=true` was applied to all nine changed Rust files. The matching direct `rustfmt --check`, Python byte compilation, analyzer 5/5, and `git diff --check` pass.

Measured four-file source digest (the digest independently recomputed by the Rust emitter and runner):
`2ac1c93ba3e4682138a0df3341a26921438334382532d6943277268320655285`.

Controller correction before review dispatch: the earlier ad-hoc report
calculation used literal backslash-zero bytes instead of NUL separators.
The runner and Rust emitter use NUL correctly. Root independently recomputed
the value above from the four current source files; no source changed.

Nine-file Rust diff SHA256 against the base: `980a0408b8c16670a45743ed4f4e93f4c0b65f51f5f811c456b57aede0b3e0b6`.

File SHA256 values:

- `src/mamba_ssm/gpu/gemm_bi_fixed.rs` `d7c34ff26abb0580fcac0bbb139368ed8bc1a0fb850eab9bfa1716f15f6deed8`
- `src/mamba_ssm/gpu/kernel_identity.rs` `5528d6a44d0a9047080af07e868def6f4eaac24e7873e3e15e59a3f63ca772ca`
- `src/mamba_ssm/gpu/gemm_bi_triad/dispatch.rs` `48b557a8e0d97b4bc3b3a1f3ad1bd3c297d691bacf1ff2bfcb90580273cd0acd`
- `src/mamba_ssm/gpu/gemm_bi_triad/launch.rs` `816ef4474281d94e0f6462cb8695e4356f40e8ec554b2dbbfc7009fa6247b047`
- `tests/gemm_bi_fixed_performance.rs` `72a01dd9eddc7f700b2d187b1c4e409637d2700a6f5380f2bb5cc8c741e079d2`
- `tests/gemm_bi_fixed_sm89_exact_n64.rs` `ec97206591dbb74fd07a6f7f09a4152b5c3eb9d0a4af2e72758b013f8e9749f3`
- `tests/gemm_bi_fixed_sm89_pipeline.rs` `26e56fb8ff5f92de10ae4103ff841dff5425d32bc59371ed9768b45961a30890`
- `tests/gemm_bi_tf32_cohort_binding.rs` `687d495e5e46f4095324de1ac974caa8d6b785372fb467946bfd0f0ae4f4b1db`
- `tests/support/fixed_sm89_toolkit_admission.rs` `2d797ba45f01c7b1983f21ce27eb39e1bd3f2c9a4f5aa82ab6abd95a5e27ddb7`

Reviewed tool SHA256 values:

- `run.py` `b9849a455871fbdf7168b6fc4d212329d60dd9aa15327b0aca4d6b1921316967`
- `remote.py` `a1aadafa46c41543e8665cf5ee59acb502c79c145811dd6166a99d2a627ebe28`
- `analyze.py` `d661f6adc280e8cbecf5e812ff298693826cd9eb692bcdc4deff9ec246773c8a`
- `test_validation.py` `b19aae8b27ba0ffba7ce608df334a3f7408a0b19d4d05347ba2130f0ea0fa40c`

## Coverage at checkpoint

- All 24 supported toolkit-literals have positive route assertions.
- Every supported toolkit retains adjacent/thin-shape, CC, SM count, known-library, loaded-holder, policy, homogeneous dtype, mixed dtype, null, C/X/W A16, and optional-bias A4 negatives.
- Unknown `0.0/12.7/12.9/13.1/13.3/14.0` decline.
- Captured43 with otherwise identical identity rejects under current44; older rejection and current acceptance cases remain.
- Post controls require an exact full eight-literal inventory, 12.8/13.0 only, revision44, smoke1/1 or post101/101, and reject stale/cross-stage controls before context allocation.
- Raw chronology/arithmetic, physical Legacy/AUTO contracts, actual public AUTO enum, ABI/poison, promotion basis, direction, valid loss/mixed-p95 handling, and binding/exit closure have host negatives.
- The adapter imports frozen Task7 arithmetic/physical/telemetry/closure validation explicitly and transforms only an in-memory copy; Task7 archives are not changed or relabeled.

Pending at this checkpoint: all three final matching builds, all three full library/focused/nonignored matrices, all three exact AUTO/half-holder/half-AUTO/TF32 live functionals, two smoke1 timing attempts, two post101 attempts, independent analysis, complete constituent matrix, archives/manifests, final quiet release, and root review/commit.
