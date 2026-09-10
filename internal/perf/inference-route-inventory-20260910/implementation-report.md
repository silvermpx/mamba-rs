# Task 904 — Inference terminal physical inventory

Status: initial implementation verified; independent review identified I1.
Fix round 1 GREEN source is frozen following root-observed RED; focused GREEN
verification is pending.

## Implemented

- Added the exact append-only backend tags 31–40, numeric tags 25–27,
  instruction tag `WmmaApi=5`, and Fixed deterministic TF32 permission bit 10.
  Existing tag values and hash domains are unchanged.
- Added the private `gemm_bi_inference/identity.rs` closed inventory of 85 exact
  terminal symbols, full storage triples, source-defined arithmetic profiles,
  ABI kinds, tile/BK/stage geometry, launch resources, and bias domains.
  Exact symbol membership resolves to the actual loaded function holder;
  optional holders remain subject to their existing loader admission.
- Added observed AUTO and forced adapters using the existing operand/shape
  structs. Threaded observation through the 17 direct terminal builders,
  legacy/WMMA boundary, and typed matvec boundary. Each terminal publishes its
  context route immediately before the actual observed enqueue.
- Preserved the cached exact-TMA bridge's prepared Triad identity and forwarded
  its observer directly into the existing cached prepared launch. Its context
  exception admits exactly the two existing NN, single-split symbols. It does
  not emit an additional Fixed/Inference alias. Borrowed half-ulp wide retains
  TriadSm80/backend5/numeric23/conversion4 and its real holder/compiler.
- Metadata construction runs only when a physical observer or context route
  recorder is active. The production conditional seam encloses all introduced
  table lookup, hashing, and binding work. Recorder borrow conflicts return
  errors. Context usability is checked before dispatch and recording.
- Context routes use a framed pointer-bound argument digest. Physical
  resolution validates that frozen binding, resolves checked nonempty spans
  through the prevalidated allocation resolver, and updates both the physical
  node and its contained GEMM route with the allocation-bound digest.
  This preserves the distinction between a GEMM context manifest and a
  conversion-inclusive physical trace with allocation-liveness evidence.
- ABI hashing is fieldwise: storage triple, symbol/backend/op, dimensions and
  strides, scalar bits, pointer presence, parameter words in the actual ABI
  order, actual opaque tensor-map fields, and literal null auxiliary slots.
  K0 never resolves fictitious zero-length input allocations. Empty output
  records and enqueues no terminal; nonempty K0 retains the actual terminal.
- Extended live permission/module/function validation, preserving backend22's
  old Triad/numeric1 pair and adding its exact Inference/numeric25 pair.
  Inference half routes remain valid with the Triad TC switch disabled;
  matvec remains a Triad fallback in both TC states. Fixed TF32 requires its
  own bit and live AllowDeterministicTf32V1 policy.
- Added eager architecture-rung preparation, reusing the existing SM100-before-
  SM90 choice, process OnceLock, disable setting, and tolerance self-check.
  Cold probes are rejected before their temporary GEMMs when recording or
  capturing. Ordinary unrecorded first use retains its existing verdict path.

## TDD evidence

The first and only edit before root authorized GREEN was
`tests/gemm_inference_route_inventory.rs`, SHA256
`a0347517cc292f217f276c4fd94d03959482795e0acc72967f9c49f58038060c`.
It holds real registered X[3,37], W[37,96], and Y[3,96] owners and calls the
existing public `gpu_gemm_typed_forward_raw` under Deterministic/Inference.

Root ran these exact ignored tests on Ada:

- `deterministic_inference_public_f32_forward_records_nn_route`
- `deterministic_inference_public_bf16_forward_records_nn_route`
- `deterministic_inference_public_bf16_to_f32_forward_records_nn_route`

Command family:
`cargo test --features cuda --test gemm_inference_route_inventory <name> -- --ignored --exact --test-threads=1 --nocapture`.
All three production forwards succeeded, then failed the intended assertion:
`Inference NN launch must be inventoried` (empty route trace). Each returned
exit101 with 1 failed/0 ignored; durations were 5.25s, 5.09s, and 5.00s.
Root reported RED build PASS (41.53s), wrapper exit0, and authorized production
GREEN only after observing all three failures.

Immutable RED source: `/root/mamba-inference-inventory-red.R7o5y8`.
RED receipts: `/root/inference-inventory-red-evidence-20260910`.

## Verification received before final source freeze

Root owns all Cargo, CUDA, GPU, immutable-snapshot, index, commit, and receipt
operations. The implementer did not run Cargo or GPU commands locally.

- Intermediate CUDA13.2 `cargo check --locked --release --features cuda --lib`:
  PASS in 6.49s, with only the pre-existing CUDA-only M3 context accessor warning.
  Immutable source `/root/mamba-inference-inventory-compile.gIarDm`;
  packet `/root/inference-inventory-compile-evidence-20260910`.
- Intermediate host snapshot (archive SHA256
  `9a48ac6e43f6d4442b11a2d1f85e6fa98b631fcd15b8ce9d58dc6642c83e2bdc`):
  CUDA library test compilation PASS; `gemm_bi_inference::identity::tests`
  7/7 PASS; append-only tag test 1/1 PASS; cold-architecture guard 1/1 PASS.
  All had 0 failed/0 ignored, wrapper exit0, and source-after manifest PASS.
  Packet `/root/inference-inventory-host-evidence-20260910`.
- Final scoped rustfmt and `git diff --check`: PASS.
- Read-only comparison of 22 existing selector/eligibility/ladder-config
  function bodies against pre-wiring source, ignoring whitespace: 0 changed.
- `git diff --name-only -- kernels build.rs Cargo.toml Cargo.lock
  src/mamba_ssm/gpu/gemm_bi_triad/contract.rs
  src/mamba_ssm/gpu/gemm_bi_triad/modules.rs`: empty.

The final freeze adds a narrow cached-bridge validation test (identity group
now 8 tests) and physical-node logical-dtype validation after that intermediate
host packet. That packet is not claimed as final GREEN evidence.

## Final verification scope

- CUDA-only and CUDA+HF compilation; final identity/validator host groups.
- All 5 ignored integration tests in `gemm_inference_route_inventory`.
- Exact GPU unit names:
  - `mamba_ssm::gpu::gemm_bi_inference::observed_inventory_cuda_tests::inference_observed_terminals_match_unrecorded_bits_on_ada`
  - `mamba_ssm::gpu::blas::matvec_inventory_cuda_tests::typed_matvec_physical_observation_preserves_public_output_and_storage`
  - `mamba_ssm::gpu::context::tests::inference_recorder_query_rejects_conflicting_borrow`

The first GPU unit exercises actual AUTO F32/BF16/F16 and half-to-F32,
forced Legacy/WMMA, Tc16/Tc64/Tc128/TcW64/TcWn64, portable TF32, loaded Ada
exact/half/RNA specialists, and borrowed half-ulp wide. It compares output
bits between disabled and enabled recording, checks output red zones and
unchanged inputs, exact symbols/modules/shapes/strides, physical/context
counts, dual digests, optional bias, F32 K0, and both backend22 family pairs.
The matvec unit covers public-baseline versus observed-terminal execution
for both input dtypes, both output types, both TC states, K0/K37, and bias.
The integration tests preserve the original three RED regressions and add
public matvec TC-state and direct Inference zero-output coverage. The latter
also verifies the canonical all-F32 public entry still rejects empty output
axes before dispatch, without recording a route.

### First final batch and narrow test correction

Root reported CUDA+HF all-target and CUDA-only library compilation PASS.
The integration target passed 4/5 tests (the original three RED regressions
and public matvec); its initial zero-output test failed because it called
the canonical all-F32 entry, whose established `F32TriadShape::validate`
rejects empty output axes before Inference dispatch. Receipt:
`/root/inference-inventory-green-evidence-20260910/routing-tests.log`.
Remaining final host/GPU unit tests had not run when that wrapper stopped.

Systematic debugging traced the public forwarding guard to the contract and
confirmed direct `inference_forward` reaches the terminal no-op guards.
Only the integration test was corrected: the fifth test is now
`deterministic_inference_direct_empty_output_has_no_terminal_record`; it
asserts canonical F32 rejection and tests direct Inference no-op recording
for all five supported storage triples at M0/N0. No production behavior or
source bytes changed. Scoped rustfmt and `git diff --check` passed.

### Accepted final verification results

Root reported the following completed results on 2026-09-10. Production
source hashes remained unchanged across these packets; only the corrected
fifth integration test changed after the first final batch.

- CUDA+HF all-target compilation: PASS, 9.83s. CUDA-only library compilation:
  PASS. Initial integration execution: 4 passed, 1 failed, 22.71s; its four
  accepted passes are the three original RED regressions plus public matvec.
  Packet: `/root/inference-inventory-green-evidence-20260910`.
- Corrected exact integration test
  `deterministic_inference_direct_empty_output_has_no_terminal_record`:
  1 passed, 0 failed, 0 ignored, 4.56s. Immutable source:
  `/root/mamba-inference-inventory-empty.2d5Ic9`; packet:
  `/root/inference-inventory-empty-evidence-20260910`.
- Unchanged-production library packet:
  `/root/inference-inventory-lib-evidence-20260910`.
  Host groups passed: identity 8, append-only tags 1, cold-architecture guard
  1, context 19. The context group left its one GPU test ignored there;
  that test was subsequently run explicitly and passed as listed below.
- Exact actual-GPU unit tests, each 1 passed, 0 failed, 0 ignored:
  `mamba_ssm::gpu::gemm_bi_inference::observed_inventory_cuda_tests::inference_observed_terminals_match_unrecorded_bits_on_ada`
  (4.77s),
  `mamba_ssm::gpu::blas::matvec_inventory_cuda_tests::typed_matvec_physical_observation_preserves_public_output_and_storage`
  (4.68s), and
  `mamba_ssm::gpu::context::tests::inference_recorder_query_rejects_conflicting_borrow`
  (4.58s), in the same library packet.
- Rustdoc: PASS, 2.40s, in the library packet.

Accepted total: **29 host tests and 8 actual GPU tests across the packets**.
This is not a claim of a fresh single 5/5 integration-target run: four accepted
integration passes came from the first batch, and the corrected fifth passed
separately. The original invalid fifth-test expectation is retained above as
a diagnosed fixture failure, not erased or counted as a pass. Root owns the
exact command receipts, their archival, source commit, and independent review;
the implementer performed no Cargo/GPU/index operations.

The final corrected integration SHA256 is
`00603815f2869f9e057681b3e6c55abba24a6e6fd3ebf0b4e0ad93dfc479f53c`.

## Final source hashes

```text
3f5d0b403f7cc467bc0b4ec63c881ab9540e7645541d8ead99a66ee494e3e615  src/mamba_ssm/gpu/gemm_bi_inference.rs
3957f1a182dc9f58e7b321ceec476d91598adb4311af10874a0e243900d7fc63  src/mamba_ssm/gpu/gemm_bi_inference/identity.rs
8c1cc9f1399b94f635397a5071a3436271f0a26cec15f3abf50ce2d0b916b7d1  src/mamba_ssm/gpu/kernel_identity.rs
5c9640b6f0db6a71165a61e7318bb14647f0a02ca58c7deef7b1a743463a9554  src/mamba_ssm/gpu/context.rs
b6022d15b0bf99c932dc5984d167472b471d08ba995a3fb1277d6e762c7a96d5  src/mamba_ssm/gpu/kernels.rs
59541b9681e54f8f936ef97553ee91797b7d5f0bb293e46d24c4bf25fb04dd93  src/mamba_ssm/gpu/blas.rs
d8d59ba04b54435f305e7e5f596f551211d94c77f3a3511875040758f3627bee  src/mamba_ssm/gpu/gemm_bi_triad/launch.rs
00603815f2869f9e057681b3e6c55abba24a6e6fd3ebf0b4e0ad93dfc479f53c  tests/gemm_inference_route_inventory.rs
```

## Self-review and remaining limits

No CUDA/header bytes, arithmetic, compiler identities, AUTO selectors,
qualification/admission tables, model graph guards, or public mode constructors
were changed. Existing `gemm_bi_triad::mod.rs` glob re-export already exposes the
GPU-private bridge seam; no redundant re-export change was needed. Existing
legacy F32 qualification-only baseline remains outside this production table,
as required by the symbol census.

The identity file is approximately 2.4k lines, mostly explicit terminal rows,
independent literal census expectations, and mutation coverage. It remains the
single cohesive private file required by the brief. No extra selector or
generic executor object was introduced.

SM90, SM100, and SM120 terminal rows (including pair-store/post-bias and the
cached bridge) have host identity coverage only in this lane. They are not
claimed as live GPU passes. Existing saved 5090 performance evidence remains
untouched. This wiring task introduces no speedup or full-release-gate claim.
Independent spec and quality review remain root-owned.

## Review fix round 1 — I1 empty output before architecture probe

Base: `7d890e6c6f3bb32eb701f503caa2c7364ddbee69`. The scoped review is
`task-904-review.md`. Only Important I1 is in this source fix; minor warning
cleanup is deferred, and root owns the evidence-index correction.

Diagnosis: homogeneous-half AUTO with M0/K64/N96 passes the architecture
alignment predicate even with null operands. With a loaded SM90/SM100 holder,
the cold preparation/probe decision runs before the terminal empty-output
guard. Recording/capture returns the eager-preparation error; unrecorded cold
execution can run two temporary probe GEMMs. The earlier K37 no-op case cannot
enter this aligned path.

Tests-first extraction mechanically moves only the existing loaded/aligned/
enabled condition into private `inference_arch_rung_for_request`, called by
the same existing AUTO branch. Its closure invokes the existing
`arch_rung_enabled`; the host test invokes the existing cold guard and counts
preparation/probe callbacks. There is no fake GPU context, alternate selector,
or generic launch executor. No empty-output skip is applied in this RED freeze.

New host names (prefix `mamba_ssm::gpu::gemm_bi_inference::`):

- `inference_arch_rung_decision_skips_cold_probe_for_aligned_empty_output`:
  BF16/F16, loaded SM90/SM100, M0/K64/N96, cold guard across recording/capture
  combinations. Requires successful `None` (no architecture terminal) and
  zero preparation/probe callbacks. Expected RED is the existing eager-
  preparation error instead of `Ok(None)` in the first recording case.
- `inference_arch_rung_decision_preserves_nonempty_cold_guard_and_probe_verdict`:
  preserves nonempty cold-record/capture rejection and allowed/disabled probe
  verdicts. Expected PASS before and after the fix.

Root-owned focused command:
`cargo test --locked --release --features cuda --lib inference_arch_rung_decision -- --nocapture --test-threads=1`.
The command above covers both tests. Root instead ran the exact empty-output
regression on the immutable RED source: 1 failed, 0 passed, 0 ignored, exit101.
Actual failure was BF16/SM90, recording=true/capturing=false: left eager-
preparation `Err` versus right `Ok(None)`. Build passed in 53.93s and the
source-after check passed. Receipt:
`/root/inference-empty-arch-red-evidence-20260910`.

Existing actual-Ada test
`deterministic_inference_direct_empty_output_has_no_terminal_record` now adds
M0/K64/N96 and M3/K64/N0 to the direct Inference loop, retaining K37. This
real-entry regression asserts zero context records; the host test supplies
the cold-loaded architecture interaction unavailable on Ada.

RED freeze SHA256:

```text
bbca49cd90c9a00f64af3730e6da5ccea6c9535790512c351068660702655182  src/mamba_ssm/gpu/gemm_bi_inference.rs
72023b1bc5ad17928f4fc1b1e6161df40c67cc91e2cb78751eeb50ae9e58e1d9  tests/gemm_inference_route_inventory.rs
```

Scoped rustfmt and `git diff --check` passed. No Cargo, GPU, index, commit,
CUDA/header, or nonempty admission changes were performed by the implementer.

After root observed this RED and authorized GREEN, the only production
decision change was an early `Ok(None)` return for `shape.m == 0 || shape.n == 0`
at the top of `inference_arch_rung_for_request`, before alignment and the
enabled/preparation callback. Empty output therefore reaches the existing
portable terminal no-op without architecture preparation/probing. Nonempty
decision logic, its companion test, and the aligned Ada regression are unchanged
from the RED freeze. Focused root-owned GREEN execution remains pending.

GREEN freeze SHA256:

```text
6dd5ebd747f05be92c0fa8515fad7357d619aa2da4ddc1eb703ecec2e2ad0287  src/mamba_ssm/gpu/gemm_bi_inference.rs
72023b1bc5ad17928f4fc1b1e6161df40c67cc91e2cb78751eeb50ae9e58e1d9  tests/gemm_inference_route_inventory.rs
```

Scoped rustfmt and `git diff --check` passed again after the three-line skip.

### Root-owned fix1 verification and source commit

On the immutable GREEN source above, root ran
`inference-architecture-host-runner.sh` with the exact aligned-empty test and
expected exit0. It compiled the CUDA+HF library, ran that test, the exact
nonempty companion and the original cold-architecture guard. All three
returned `test result: ok. 1 passed; 0 failed; 0 ignored`.
Packet: `internal/perf/inference-route-inventory-20260910/fix1-green/`.

Root then ran the existing focused integration driver on the same immutable
source, exact test
`deterministic_inference_direct_empty_output_has_no_terminal_record`.
Result: `test result: ok. 1 passed; 0 failed; 0 ignored`,4.69s; both K37 and
aligned K64 shapes were included. Packet: same evidence root `fix1-gpu/`.
No unaffected full terminal/GPU matrix was repeated.

Both runners exited0 and source-after checks passed. Root compared564 files
plus both Cargo inputs to the local worktree, reran scoped rustfmt/diff checks
and committed the two-file fix as
`086716e38e0b554009ecf034ffc50e9911790c19`. Existing build warnings remain;
no CUDA/header, nonempty selector or admission data changed. Scoped independent
re-review of I1 remains pending.
