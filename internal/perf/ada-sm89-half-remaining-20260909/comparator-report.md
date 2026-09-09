# Ada SM89 remaining-half actual-AUTO comparator harness

Date: 2026-09-09. The repaired CUDA13.2 actual-AUTO comparator batch passes:
all seven retained rows improve the real public AUTO in eager and graph.
No production selector, kernel, artifact, module, identity, or cohort was
changed by this test-only checkpoint. The first run stopped before timing on
an invalid diagnostic path; its failure receipt is retained separately.

## Coverage

One ignored batched test covers the seven missing comparator rows:

- TN d768-in F16 and BF16: retained regpipe+`float2` epilogue;
- TN d768-out F16: retained compact B-XOR;
- TN d768-out BF16: retained regpipe+`float2` epilogue;
- TN Prism F16 and BF16: retained compact B-XOR;
- NN d768-in F16: the already-loaded SM89 M128N128/BK64/S3 symbol.

For every row the harness obtains fresh physical evidence through the supported
`qualify_physical_launch(HalfPolicy)` facade with the same dtype, Tensor Core
setting, half policy and logical request. That facade proves the production
branch and its exact physical eager/graph manifest. The holder is explicitly
dropped before measurement. The harness then calls the unchanged real
`gemm_bi_backward_dw_typed` or `gemm_bi_forward_typed` public API for eager and
graph timing. The captured public graph's non-null function handle resolves to
the qualified symbol and its exact grid, block and dynamic shared-memory
configuration. The evidence rejects an empty or multi-node route and any wrong
logical op, dtype, shape, stride, execution dtype, numeric contract or output
ownership.

TN candidate, actual AUTO, and the forced TC64 exact oracle must agree bit for
bit in eager and graph execution. NN candidate and actual AUTO must agree with
the forced TC128 oracle. Each 256-byte-aligned view retains two-sided guard
checks; every observation revalidates immutable A/B words and guards.

The performance screen uses seven ABBA and seven BAAB windows for eager and
graph. Retained/actual-AUTO is labelled against the existing strict `.99`
p50/p95 decision gate. A valid performance stop is collected rather than
aborting the batch, so all six TN rows, both d768-in tournaments, and the NN
row complete before an aggregate performance result is returned. Exact-bit,
ABI, resource, graph, guard, or immutable-input failures still stop
immediately. Fast is measured and labelled independently but is not an
admission gate.

For each d768-in dtype the direct tournament measures all three unordered
pairs among compact, plain regpipe, and regpipe+vec2. It prints each pair's
four eager/graph ABBA/BAAB strata and an observed three-family order; no family
is required a priori to beat both others by 1%. No losing family is rerun on
the other four TN rows. All three candidate families are capped at the frozen
128-register ceiling for this cohort.

## TDD receipt

RED used the literal six-cell plan consumer before the plan was populated:

```text
$ cargo test --no-default-features --test gemm_bi_half_remaining_qualification_contract
running 2 tests
test remaining_tn_plan_is_the_literal_six_cell_retained_map ... FAILED
left: []
right: [six literal retained TN rows]
test result: FAILED. 1 passed; 1 failed
```

GREEN after populating the retained map was 2/2. A follow-up RED for complete
batching then failed to compile because `TournamentFamily`, the full-pair
schedule, the 128-register constant, and `finish_performance_stops` did not
exist. GREEN after implementing that bounded behavior:

```text
$ cargo test --no-default-features --test gemm_bi_half_remaining_qualification_contract
running 5 tests
test result: ok. 5 passed; 0 failed

$ cargo test --no-default-features --test gemm_bi_typed_parity --no-run
Finished `test` profile
```

The no-default contract fixes the literal dtype/shape/family/grid map, proves
that only d768-in enters the incremental three-family tournament, freezes the
complete three-pair schedule and 128-register cap, and checks that multiple
performance stops survive aggregation. The contract target is explicitly
`cfg(not(feature = "cuda"))`: CUDA builds compile the helper only as the child
of `gemm_bi_typed_parity`, where its required fixtures and `Ctx` exist. The
CUDA-only body was parsed and formatted locally, but deliberately was not
built with the CUDA feature on macOS.

## Actual-AUTO trace repair

The first root-owned CUDA13.2 attempt passed both five-sample quiet gates and
compiled the retained TN candidate at 125 registers, local 0, static shared
32,768 bytes and occupancy 3. It then stopped on the first TN F16 d768-in row
because the generic `record_eager_gemm_trace` returned zero routes. This was a
diagnostic failure, not a kernel, correctness, or timing result: the public
typed facade intentionally installs `NoPhysicalObserver`.

The repair uses the existing half physical-qualification facade described
above and leaves the actual public eager/graph/timing calls unchanged. RED was
observed from the native source contract while the unsupported generic recorder
was still present. Root independently reran the native suite (6/6), compiled
the release CUDA target on Ada and ran the repaired GPU batch (1/1).

## Live CUDA13.2 receipt

The source snapshot is `/root/mamba-assembly-screen.UEcoxU`: production base
`0c162501`, plus the frozen test-only overlay. Helper SHA-256:
`bf6ba9ce19071c8f0167877b83a51adcda083e528045ea2bd11e6ae643125f46`;
native contract SHA-256:
`a350b218e9e01d833f719db0dcf227fc08d50f2c3cc7fcfc48f6e5da42b5c305`.
No later TF32 runtime WIP was copied into the running snapshot. Kernel caches
were disabled; all five preflight samples were compute0/memory0/free48463MiB.
The test passed in 209.73 seconds, with the final quiet gate also passing.

- [Raw successful batch](half-cuda132-trace-fixed-run.log), SHA-256
  `a84ad1f10197b3193bd0d5fe2883a6d68d2ae1b9c3adc25386d3b85cce807855`.
- [CUDA release build](half-cuda132-trace-fixed-build.log).
- [Earlier observer diagnostic failure](half-cuda132-trace-harness-stop.log).

All six TN candidates retain exact TC64-oracle bits across repeated eager and
graph accumulation, and the NN candidate retains exact TC128-oracle bits.
Every actual public AUTO arm also passes its independent oracle, input and
guard checks. All seven candidate/AUTO decisions satisfy the strict four-strata
p50/p95 `<.99` gate. Both d768-in tournaments observe regpipe+vec2 first.
None of these seven rows is a strict four-strata Fast winner in this receipt;
Fast results are separate from admission against the slower current AUTO.
This closes the missing actual-AUTO comparison, not production admission or
CUDA12.8/13.0 qualification of the new TN exports.

## Root-owned CUDA13.2 commands

Compile without running:

```text
CUDARC_CUDA_VERSION=13000 cargo test --release --no-default-features \
  --features cuda,cudarc/cuda-13000 --test gemm_bi_typed_parity \
  triad_half_remaining_qualification::ada_half_remaining_seven_cells_vs_actual_auto_and_fast_once7 \
  --no-run
```

Exclusive Ada run:

```text
CUDARC_CUDA_VERSION=13000 cargo test --release --no-default-features \
  --features cuda,cudarc/cuda-13000 --test gemm_bi_typed_parity \
  triad_half_remaining_qualification::ada_half_remaining_seven_cells_vs_actual_auto_and_fast_once7 \
  -- --ignored --exact --nocapture
```

The runtime fails closed unless the device is CC8.9 with 142 SMs, the build is
release mode, and the live NVRTC version is exactly CUDA13.2. It explicitly
selects `BiGemmFamily::Triad` after constructing the Ada context so ambient
family configuration cannot change the actual-AUTO comparator.

## Files

- `tests/gemm_bi_typed_parity.rs`: one child-module declaration only;
- `tests/support/triad_half_remaining_qualification.rs`: literal plan and
  CUDA-only orchestration reusing the existing fixtures, compilers, resource
  gates, graph inspectors, exact observers, and timing functions;
- `tests/gemm_bi_half_remaining_qualification_contract.rs`: no-default native
  coverage contract;
- this report.

Root owns the shared index, immutable source snapshot and CUDA execution.
