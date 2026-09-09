# Ada SM89 TF32 joint source-only preparation

## Scope

This change prepares only the sealed CUDA source composition and native contracts for the later isolated `TriadSm89Tf32Joint` module. It does not register a module, change artifact identities, add launch or dispatch routes, populate an AUTO cohort, or execute CUDA.

The retained owner remains byte-identical:

- owner: `kernels/gemm_bi_triad/sm89_tf32_joint.cu`
- SHA-256: `5fbb9fc1481052da3591196f73f8d372a3616e07657dc13da4f2a1d8a1e95bd3`
- exports: exactly four retained symbols from the integration brief

The standalone composer prefixes one new sealed primitive owner containing only the two Fixed-TF32 definitions required by those exports:

- primitive owner: `kernels/gemm_bi_triad/sm89_tf32_joint_primitives.cuh`
- SHA-256: `80bf255d0ee7e464fbbc3225f47e4096c5886125673c92af0954770e706a77ba`
- primitives: `gbf_tf32_copy_cg`, `gbf_tf32_mma_m16n8k8`
- CUDA exports: zero

Native parity tests compare each complete primitive definition byte-for-byte with its frozen definition in `kernels/gemm_bi_fixed/tf32.cu`. Negative tests reject a missing primitive, an additional void or non-void `gbf_tf32_*` helper, and any exported kernel in the primitive owner.

## Typed contracts

The source adapter now exposes four symbol-keyed kernel specs and the two by-value ABI bundles:

- transpose params: `{rows, columns, output_stride}`, size 12, alignment 4
- GEMM params: `{alpha, beta, m, k, n, lda, ldb, ldc}`, size 32, alignment 4
- transpose Driver ABI: `[(0,8),(8,8),(16,12)]`, terminal argument 3, source order `(input, output, params)`
- GEMM Driver ABI: `[(0,8),(8,8),(16,8),(24,8),(32,32)]`, terminal argument 5

The specs bind the frozen block dimensions, dynamic/static shared memory, register caps, local-memory zero requirement, max-thread floor, and the retained occupancy gates. The transpose occupancy is deliberately `None`: discovery froze its 26 registers, zero local memory, 4,224-byte static shared allocation and `max_threads >= 256`, but did not freeze a production-module occupancy threshold.

## TDD and verification

RED was observed from the native no-CUDA integration test after adding the contracts first: 37 unresolved adapter items (`compose_source`, primitive owner, typed params/specs and ABI constants).

GREEN command (run from a temporary copy because the local workspace is a nested worktree):

```text
tmp_dir=$(mktemp -d /tmp/mamba-tf32-joint-green.XXXXXX)
rsync -a Cargo.toml Cargo.lock src kernels tests "$tmp_dir"/
/Users/silvermpx/.cargo/bin/cargo test --manifest-path "$tmp_dir/Cargo.toml" --no-default-features --test gemm_bi_tf32_joint_source_contract
```

Result: 26 passed, 0 failed, 0 ignored. This is a native macOS/no-default-features source-contract run only. No `--features cuda`, NVRTC, CUDA compiler, SSH or GPU command was used.

Formatting was limited to the two owned Rust files with `rustfmt --edition 2024 --config skip_children=true`. `git diff --check` passed.

Root independently ran `cargo test --no-default-features --test
gemm_bi_tf32_joint_source_contract` through the workspace runner: 26 passed,
0 failed on the Linux Ada worker, with no GPU execution. The imported frozen
discovery helpers emit 13 unused-item warnings; these are recorded for the
planned test-tooling cleanup, not described as a warning-free run. Root also
verified both owner SHA-256 values above against the files.

## Deferred gates

The subsequent module integration must compile this exact composition on each CUDA host, prove exact four-export PTX inventory, Driver ABI, stack/spill/local memory, per-symbol registers/shared memory/occupancy/SASS, and artifact identity before any forced route or AUTO admission. This source-only preparation makes no compile, correctness, performance, graph, or admission claim.

## Root assembly correction: five exports

The subsequent retained-winner audit recovered the missing NN TF32 d768-out
baseline N96 body. Its saved actual-AUTO improvement is 18.1–18.6%; the later
direct epilogue's sub-1% incremental result did not invalidate that baseline.
The owner now appends the exact retained baseline body with a production
export name in its own namespace. Existing four sections are unchanged and
still pass their complete-body parity tests. Updated owner SHA-256:
`e1e8a2ad1d2d03b4d0e02730f087eab1c26cfc7712f867fbbead13b032e3624c`.

Source inspection also found the missing `gbf_aligned16` dependency used by
every GEMM epilogue. The primitives now include its exact Fixed/common.cuh
definition; updated SHA-256:
`c16e81fdcc4745352c97ee7daa39f2629716d7ebe38b6eea0a91393268303b0e`.

Both new native tests were observed failing before their implementation:
missing NN export, then zero alignment-helper definitions instead of one.
After correction, the complete no-default-features source suite passes 28/28
on the Ada Linux CPU worker. The previously recorded 13 discovery-helper
unused-item warnings remain. No GPU or timing result is claimed. Module and
route integration must use five exports, not the earlier four-export census.
