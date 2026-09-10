# Release test-fixture dependency notes

Read-only source audit at `779ef943`, 2026-09-10. Task903's new test is not
committed yet: Git tracks384 files under `tests/`, while the working directory
contains385. No test, fixture, experiment or evidence file was moved or deleted.

The owner approved separating public regressions, maintained benchmarks,
qualification tools and internal experiments after API assembly. File names
alone do not establish that a discovery stand is safe to remove.

There are253 top-level Rust test targets including the uncommitted903 target.
A name-only triage finds92 discovery/experiment/tournament candidates,
30 qualification/source-contract/toolkit/census candidates,22 benchmark/probe
candidates and109 remaining files. This is not a retention verdict: the
compact-XOR discovery stand described below falls in the remaining group,
and useful regressions also live inside discovery-named files.

## Library unit tests currently pull in a discovery stand

`src/mamba_ssm/gpu/gemm_bi_triad/sm89_finalist_source.rs`, its `#[cfg(test)]`
module around lines500–600, depends on:

- `tests/gemm_bi_tf32_nt_compact_xor.cu` as the frozen helper;
- `tests/support/triad_tf32_nt_compact_a_ldmatrix_sliced_source.rs` as source text;
- the entire `tests/gemm_bi_tf32_nt_compact_xor.rs` via `include!` when CUDA is
  disabled, in order to call `compact_eight_warp_s2_candidate_source` and its
  compact-A ldmatrix adapter.

That included stand in turn names eleven source-adapter modules and several
candidate CUDA/header files near its first sixty lines. Thus excluding only
the top-level experiment from the crate can break ordinary library unit tests,
even when no GPU tests run. The production composer itself consumes shipping
`kernels/` files, not these experimental bodies; this coupling is test-only.

Before archiving the stand, preserve the accepted candidate's independent
normalized source as a compact frozen fixture, or extract only the required
candidate composer and its dependencies. Retain the current regression that
compares the production composer with the accepted source, and the missing/
duplicate anchor failures. Do not replace an independent fixture with an
expected value generated from the same production function under test.

Check both non-CUDA library tests and CUDA library compilation after the split:
the frozen candidate `include!` and helper are non-CUDA-only, while the sliced
adapter source-text reference is unconditional within the unit-test module.

## Three internal receipt files are compile-time fixtures

`tests/gemm_bi_scalar_nt_copyplan_siblings_discovery.rs:175` has a
`frozen_receipt_regression` module that includes all three files below from
`internal/perf/ada-f32-nt-copyplan-siblings-batch-a-20260909/evidence/`:

- `exact-nt-prequal-f04584c8-cuda128.log`
- `exact-nt-prequal-f04584c8-cuda130.log`
- `exact-nt-prequal-f04584c8-cuda132.log`

It tests identity fields including compile/artifact/source/header/NVRTC-domain
digests. The archive remains the full raw receipt; extract the necessary
regression fixture before excluding `internal/` from a package that still
builds this test. Alternatively move the whole receipt regression with the
maintained qualification tool and keep that tool's fixtures together.

## Packaging and execution checks

The current Cargo manifest has no explicit include/exclude list or explicit
test/bench target declarations. Root-level integration Rust files are therefore
still independently discovered test targets. `#[ignore]` avoids execution but
does not avoid their build cost or compile-time fixture dependencies.

For the later cleanup task, audit the dependency closure of each retained
target, then verify the actual extracted package (not only `cargo package
--list`). Keep public source/kernels/examples and required unit/regression
fixtures. Keep internal raw performance receipts out of the published crate
without erasing their repository archive. Git ignore and package inclusion
are distinct mechanisms; adding an ignore rule does not untrack existing files.

This is a dependency preflight, not a complete classification of all384
tracked test files or a claim that the shipping test suite is already clean.
