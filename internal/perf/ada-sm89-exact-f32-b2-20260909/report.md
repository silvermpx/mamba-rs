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

## Assembled pre-admission batch

The expanded selector qualification now passes on Ada/CUDA13.2. It keeps the
public AUTO cohort empty and requires `MAMBA_SM89_EXACT_F32_EXPECT_AUTO=0`.
Two earlier attempts stopped on test-contract errors. The first stopped after
the d768-in raw non-unit-alpha probe because
the test incorrectly called the public facade with an epilogue that facade
deliberately rejects; the raw candidate/oracle result itself passed and the
harness now requires that rejection instead.  The second attempt,
`/root/mamba-assembly-screen.UEcoxU/b2-cuda132-nonunit-fixed.log`, reached K0
after every d768-in raw target/tail/exception/non-unit probe passed, then exposed
a second test-oracle error.

The TN zero-reduction production kernel evaluates
`__fmaf_rn(alpha, +0.0f, output)`.  It is therefore not a bitwise no-op for
negative zero: with the production `alpha=1`, `-0.0` becomes `+0.0`.  The K0
seed intentionally contains negative zero, but the initial probe used the
generic input-immutability assertion on the output buffer.  The corrected
probe derives the complete expected output with the same fused-operation
contract, checks it after eager and graph execution, and still independently
requires X/dY immutability and both output guards.  A new native literal test
pins the signed-zero, finite and subnormal cases. The corrected live run is
recorded below; the supported-toolkit admission phase remains separate.

In one process and two reusable contexts (the qualification facade permits only
one live holder per context), the batch now requires all three routes to pass:

- the production module-loader ABI census, empty exclusion set, live register,
  local-memory, static-shared-memory and occupancy gates;
- exact ordered physical node identity for forced eager and prepared graph,
  including module owner, dtype, operation, shape, strides, tile, numeric
  contract, ownership, grid, block, dynamic shared memory and route identity;
- independent raw-symbol probes adapted from the frozen discovery harnesses:
  the d768-in transpose scratch is compared word-for-word with a CPU transpose,
  while both direct routes compare every Split-M scratch word with the retained
  aligned partial kernel before the shared reducer;
- full target, finite tail, exceptional payload, non-unit alpha and zero
  reduction; rejected strict-selector cells must stay on prior AUTO, and K0 is
  exercised through eager and captured-graph paths;
- repeated eager/graph output bits, A/B immutability, facade guards and explicit
  64-F32 leading plus trailing guards around every raw output/scratch/input
  allocation, with the active origin kept 256-byte aligned;
- one batched paired screen against current actual AUTO: ABBA and BAAB raw
  observations for eager and graph, `once3` then `once7` only if every stratum
  has both p50 and p95 below `0.99`.  Each 20-operation timed result is checked
  against a separately executed raw candidate-versus-retained 20-chain oracle.

Fast remains separately labelled and deferred to the frozen discovery receipts;
the expanded batch has no fresh Fast arm and must not be described as a full
admission result. CUDA13.2 compilation, the focused live run and raw timing
replay now pass; a separate supported-toolkit admission phase remains. SASS
instruction evidence remains owned by the existing module-loader/frozen batch;
this selector test does not add another compiler or disassembler wrapper.

Native protocol tests exercise the fail-closed three-case phase ordering, the
conditional once3/once7/Fast receipt ladder and the TN K0 fused-operation
oracle.  The exact host command is:

```text
cargo test --no-default-features --test gemm_bi_sm89_exact_f32_tn_selector_qualification -- --nocapture
```

It currently passes `4 passed; 0 failed`.

## Expanded CUDA13.2 result and raw replay

The immutable runtime base is `0c162501` in
`/root/mamba-assembly-screen.UEcoxU`, with the frozen selector/helper overlay.
Selector SHA-256:
`a32b1f5e75e21bc0893ec17d80f753221e695bcdd3d322f0abca6ced3fe3e7bd`;
helper SHA-256:
`5deff79e9d80325aa654cc9db64cb2e3dcbae187c3e836a38f10f792306b62f5`.
The later TF32 runtime changes were not copied into this snapshot. Kernel
caches were off. Five preflight samples were compute0/memory0/free48463MiB.

The exact ignored live test passes `1 passed; 0 failed` in 40.96 seconds.
All three routes pass the module resources, raw target/tail/exceptional and
non-unit probes, K0 behavior, eager/prepared identity, independent output and
scratch oracles, repeated bits, immutable inputs and two-sided guards. Both
once3 and once7 pass against the actual public pre-admission AUTO.

| TN cell | Eager once7 candidate/AUTO p50 | Graph once7 candidate/AUTO p50 | Worst once7 p95 |
| --- | ---: | ---: | ---: |
| d768-in | 0.765524–0.769026 | 0.765215–0.766430 | 0.771118 |
| d768-out | 0.766205–0.766446 | 0.762814–0.763805 | 0.768094 |
| Prism | 0.787244–0.787295 | 0.784327–0.785092 | 0.790025 |

Ratios are candidate elapsed time divided by actual AUTO elapsed time; the
ranges are ABBA/BAAB. A root-side independent JavaScript replay reconstructed
all 24 screen records, 120 brackets and 480 positive raw observations. Every
stored ratio and p50/p95 matched exactly (maximum absolute error zero); all
strata remained below `.99`. No fresh Fast result is claimed.

- [Successful live receipt](b2-cuda132-k0-fixed-run.log), SHA-256
  `0ca5ff3beb9b53d151c0935cb3dc29578e00c00c551730e265dc59fd5b884ecf`.
- [CUDA release compile receipt](b2-cuda132-k0-fixed-build.log).
- [First diagnostic stop](b2-expanded-nonunit-harness-stop-cuda132.log):
  non-unit alpha was incorrectly sent through the unit-epilogue policy facade.
  Non-unit raw-kernel oracle coverage remains, with explicit facade rejection.
- [Second diagnostic stop](b2-expanded-k0-harness-stop-cuda132.log): the test
  incorrectly assumed zero reduction preserves the output bit pattern. The
  actual TN contract computes `fma(alpha,+0,C)`; with alpha1, seeded `-0`
  correctly becomes `+0`. The repaired oracle checks that operation exactly,
  with input immutability and output guards unchanged.

The repairs changed only tests. Production arithmetic, existing route
identities and the empty AUTO evidence cohorts were not changed.
