# Ada SM89 exact-F32 TN d128 source promotion

Date: 2026-09-09. Scope: source assembly only; no module, artifact,
context, dispatch, launch, kernel-identity, cohort, or AUTO admission changes.

## Promoted retained bodies

The standalone owner template expands to exactly these two production-named
exports:

- `gemm_bi_tn_sm89_f32_d128_in_m16n16_f64fold_v1` for TN
  `(m,k,n)=(1024,128,512)`, `Kernel<16,16,64,0,false,true>`, grid 256,
  block 64, and 4096 bytes dynamic shared memory;
- `gemm_bi_tn_sm89_f32_d128_out_m8n16_f64fold_v1` for TN
  `(m,k,n)=(1024,256,128)`, `Kernel<8,16,64,0,false,true>`, grid 256,
  block 64, and 3072 bytes dynamic shared memory.

One sealed route template is instantiated in two distinct namespaces. The
source-contract test compares the normalized constants/helper/template body of
each instantiated route against the corresponding retained generator in
`tests/support/triad_tn_d128_direct_source.rs`, then separately pins the
retained alias, launch bounds, resource constants, seven-field Driver ABI, and
exact FP32-chunk/ascending-FP64-fold arithmetic markers.

No M8N32, 128-thread M16N16, d128-in M8N16, d128-out M16N16, delayed-fold,
test, experiment, or foreign export is present in the composed source.

## TDD evidence

RED, before the owner and adapter existed:

```text
$ cargo test --no-default-features --test gemm_bi_sm89_exact_f32_d128_source_contract --no-run
error: couldn't read tests/../src/mamba_ssm/gpu/gemm_bi_triad/sm89_exact_f32_d128_source.rs:
No such file or directory (os error 2)
exit 101
```

GREEN, after the source-only implementation:

```text
$ cargo test --no-default-features --test gemm_bi_sm89_exact_f32_d128_source_contract
running 13 tests
test result: ok. 13 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

The 13 tests comprise four new production source-contract tests plus nine
native contract tests inherited from the frozen retained adapter imported by
the integration test.

Scoped hygiene:

```text
$ rustfmt --edition 2024 --config skip_children=true \
    src/mamba_ssm/gpu/gemm_bi_triad/sm89_exact_f32_d128_source.rs \
    tests/gemm_bi_sm89_exact_f32_d128_source_contract.rs
$ git diff --check -- \
    kernels/gemm_bi_triad/sm89_exact_f32_d128.cu \
    src/mamba_ssm/gpu/gemm_bi_triad/sm89_exact_f32_d128_source.rs \
    tests/gemm_bi_sm89_exact_f32_d128_source_contract.rs
```

Both commands completed successfully.

## Frozen files and SHA-256

```text
c684cfcc1165af0ad5cdc2e9e0e1c5c4d7d2986d48fb37b6c192a2dd40718310  kernels/gemm_bi_triad/sm89_exact_f32_d128.cu
81a3ea0825087d2941120e81c990d7ce7c08c125375d59d4480ec44aafdf413d  src/mamba_ssm/gpu/gemm_bi_triad/sm89_exact_f32_d128_source.rs
fc7155e7cd0c7e59621d7c6be6b6bf4812dd022bdb67209af1a7bb030fb85093  tests/gemm_bi_sm89_exact_f32_d128_source_contract.rs
```

The CUDA owner digest is also frozen as text and bytes in the standalone Rust
adapter.

## Concerns and remaining work

- This checkpoint is deliberately unreachable from production: the new Rust
  adapter is not declared from `mod.rs` and the owner is not composed into any
  existing CUDA artifact.
- No NVRTC, PTX, SASS, resource, ABI-census, exactness, graph, or performance
  claim is made here. Those gates remain required before artifact plumbing or
  AUTO admission.
- The future artifact must keep per-symbol ABI/resource exclusion so a defect
  in one d128 export cannot darken its sibling.
- The worktree contained unrelated root-owned WIP throughout this task. This
  report and the three source/test files were not committed pending root index
  clearance.
