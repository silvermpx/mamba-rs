# Canonical GEMM mode API verification

2026-09-10. Task901, source base `47d65ca4`, commit `0dcfe167`. This is context policy work,
not a new kernel performance result or a completed model-wide routing proof.

## Contract

`GemmMode::{Deterministic,CublasFast,CublasPedantic}` is the context's single
mode authority. Deterministic is the constructor default. Custom family,
tensor-core permission, exact/TF32 policy and half policy remain independent.
Context-aware vendor boundaries reject Deterministic; direct model bypass
closure and high-level mode constructors remain separate release tasks.

The host policy is versioned: policy revision6, cuBLAS policyV2 and dispatch
digestv5. CUDA source bytes, selectors, admissions and custom compiler/numeric
identities are not redefined. Existing performance receipts retain their actual
measured source SHA, not this API slice's commit.

## Tests-first and corrections

- `red/`: missing public API produced14 E0432/E0425/E0599 compiler diagnostics
  before implementation. Frozen `tests/gemm_mode_api.rs` SHA-256
  `0124f97cb89b30d321d46a61f89d2cf7df89b1068e8a4775621166eb2bf02599`
  is unchanged by GREEN.
- `green-initial/`: all-targets compilation caught a test-only E0277:
  `expect_err` requires Debug on CudaGraph. Explicit matching preserves the
  assertion without that unsupported bound. Original failure and post-failure
  source integrity check are retained; no production change was needed.
- Root source review found that a failed qualification guard entry retried
  the mode transition from Drop, potentially masking the original error.
  A separate tests-only checkpoint and narrow regression packet cover this
  error path before its repair. `lease-red/` fails on two transition attempts
  instead of one. `lease-green/` passes the unchanged regression and both
  qualification guard tests after constructing the guard only on successful
  entry; completed2026-09-10T11:21:38Z, runner0/source integrity PASS.
  The final-source CUDA all-targets check passes too (9.47s).

## Ada GREEN context/API packet

Packet `green-fix1/`, completed2026-09-10T11:17:22Z, runner0 and source/fixture
integrity checks passed. CUDA13.2, RTX6000Ada, UUID
`GPU-d1edd7be-e88d-aed6-047d-622163306f0e`; raw device/toolkit/build identities
are in the packet. GPU utilization was checked before live tests.

| Check | Result |
|---|---|
| CUDA all-targets release check | PASS |
| Public API/default/constructor/transition/adapter tests | 6 PASS |
| Capture and route-recording transition rejection | 2 PASS |
| Pure mode/env/transaction tests | 10 PASS |
| Deterministic env defaults/custom-policy validation | 1 PASS |
| Qualification policy restoration | 1 PASS |
| Real-context vendor compute selection across F32/F16/BF16 | 1 PASS |
| Kernel identity tests | 24 PASS |
| CUDA Rustdoc, broken intra-doc links denied | PASS |
| CUDA-feature doctests | 14 PASS, 3 existing ignored |
| Non-CUDA library suite | 84 PASS |

The six public API tests query real cuBLAS handle math and cover all nine
canonical mode transitions. The compute-selection check exercises the real
context and dispatch helper; it is not a fresh GEMM throughput measurement.
Non-CUDA receipt: `non-cuda-lib.log`, Fleet tree `f468976f`, 0.77s test time.
No full performance tournament was run for this API change.

## Remaining checks and warnings

Independent task review approved spec compliance and quality, with no Critical
or Important findings; see `review.md`. A stale typed-forward dispatcher
comment is assigned to the following raw-routing task. Model-wide closure,
high-level construction, benchmark migration and new RTX5090 API validation
remain explicitly outside this completed context foundation.
The final local source matches every
entry in the final remote manifest (`lease-green/local-source-match.log`).
Direct Git comparison confirms no CUDA/header changes. Existing discovery unused/dead-code warnings and
deprecated migration calls are visible in the all-targets log. Do not suppress
them: benchmark/test migration and release cleanup must resolve their causes.
Rustdoc retains one pre-existing public-to-private link warning in
`gemm_bi_triad/contract.rs` for `tf32_route_specs_for`; documentation cleanup
tracks it explicitly. No claim of pristine repository-wide warning output.

No new RTX5090 API qualification was performed. The preserved dual-GPU kernel
performance and earlier route snapshots remain available separately.
