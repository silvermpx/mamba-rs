# Ada SM89 exact-F32 large-TN integration B2 — 2026-09-09

## Outcome

The three frozen Batch B0 winners are wired through the production scalar
physical-route machinery for eager execution and prepared graph replay:

| Cell | Production route | Frozen partition |
|---|---|---|
| TN d768-in `(2048,768,3072)` | `gemm_bi_transpose_f32_32x16_d768_v1` → `gemm_bi_tn_sm89_f32_n64_dual_chunk_fused_finalize_v1` | two ordered F32 chains, `1024+1024`, sequential FP64 finalize |
| TN d768-out `(2048,1536,768)` | `gemm_bi_tn_sm89_f32_m64n64_bk16_s2_d768_out_raw_v1` → `gemm_bi_splitm_reduce` | four ordered F32 chains, `512*4`, sequential FP64 reducer |
| TN Prism `(4621,384,1928)` | `gemm_bi_tn_sm89_f32_m64n64_bk16_s2_prism_raw_v1` → `gemm_bi_splitm_reduce` | six ordered F32 chains, `784/784/784/784/784/701`, sequential FP64 reducer |

The two exact-module backends have distinct numeric and output-ownership
identities. Both bind `ModuleKind::TriadSm89ExactF32` and the private
`SM89_EXACT_F32_TN_ROUTE_REVISION`; the portable transpose and reducer retain
their existing `TriadScalar` identities. Scratch ownership is exact: d768-in
retains transpose scratch only, while the two direct routes retain Split-M
scratch only.

## Qualification and AUTO state

The exact CUDA12.8/13.0/13.2 B1 module identities are sealed as qualification
candidates from the tracked B1 live receipts. Forced qualification requires
the exact CC8.9/142SM identity and the route-specific bound symbol. It exercises
the same production scalar controller, route observer and prepared graph
sequence as AUTO.

Public AUTO remains deliberately fail-closed: `SM89_EXACT_F32_EVIDENCE_COHORTS`
is empty until the new live selector qualification passes. Existing Fixed,
NT, SM80, SM120, TF32 and half compositions and global revisions are unchanged.

The ignored live test is
`sm89_exact_f32_large_tn_forced_correctness_and_actual_auto_admission` in
`tests/gemm_bi_sm89_exact_f32_tn_selector_qualification.rs`. With
`MAMBA_SM89_EXACT_F32_EXPECT_AUTO=0` it checks, for all three cells:

- forced eager and forced prepared-graph physical identity;
- bit equality of forced eager, forced graph and actual public AUTO;
- input immutability and all facade-owned red zones;
- the exact two-node symbol order and isolated module ownership.

After the resulting evidence is reviewed and the cohort is admitted, rerun
with `MAMBA_SM89_EXACT_F32_EXPECT_AUTO=1`; it additionally requires actual AUTO
to resolve to the exact same two symbols for each cell.

## Host verification

No GPU or SSH command was run in B2.

- CUDA host exact-F32 unit filter: 7 passed;
- kernel-identity discriminant test: 1 passed;
- live selector qualification harness: compile-only passed;
- CUDA no-default library/tests compile gate: passed.

All verification ran as native macOS host compilation/tests with CUDA feature
types enabled (`CUDARC_CUDA_VERSION=13000`) against a temporary source copy;
no CUDA driver context or GPU kernel was used. The copy is necessary because
the local workspace Cargo shim builds only committed snapshots.

## Root live CUDA13.2 wiring check

The immutable `f76f81f490df81692e072157d4122c7dde337a79` source was archived to
`/root/mamba-exact-tn-b2.oOORVG` on Ada and compiled with CUDA13.2,
`--release --no-default-features --features 'cuda,cudarc/cuda-13000'`,
`CUDARC_CUDA_VERSION=13000`, and kernel caches disabled. The exact test name
was checked from `--list`. Five consecutive preflight samples were
`0% compute, 0% memory utilization, 48463 MiB free`.

With `MAMBA_SM89_EXACT_F32_EXPECT_AUTO=0`, the live test passed in 112.36 s:
1 passed, 0 failed. All three cells reported the expected forced two-node
symbol sequence, eager/graph bit equality, equality against the actual prior
AUTO and passing red zones. The prior AUTO for each remained
`gemm_bi_tn_splitm_partial_aligned` followed by `gemm_bi_splitm_reduce`.

This is a focused production-wiring smoke, not admission or a new performance
result. Exceptional-value, repeat/accumulation and paired performance gates
remain in the assembled qualification batch; the AUTO cohorts stay empty.

- [Live receipt](exact-tn-b2-f76f81f4-cuda132.log), SHA-256
  `3181a841ee4f8b84a7535a1daf63524a26c54da7d12c272cdcd0b73799f0e1c4`.
- [CUDA host build receipt](exact-tn-b2-f76f81f4-cuda132-build.log).
