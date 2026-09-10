# Triad final production AUTO benchmark adapter review

Date: 2026-09-10

## Scope

Read-only review of only `tests/gemm_bi_performance_matrix.rs` at frozen
SHA-256
`940873b91bc3e2a484561f61421e074354964cb835af754965a2392ea0a1b0ce`,
using `final-auto-benchmarks-brief.md`, `final-triad-harness-audit.md`, and
`triad-final-auto-review.diff`. Untouched Inference code and the rest of the
branch were not reviewed. No build, test, or GPU command was run by this
reviewer.

## Verdict

**Specification verdict: changes requested. Quality verdict: changes
requested.** There are no Critical findings and two Important findings. The
adapter's inventory, comparator mapping, paired timing protocol, reset labels,
and evidence schema otherwise match the requested design. Do not use the
current snapshot for the full 324-record release cohort until both findings are
fixed.

## Findings

### Important 1: vendor work violates the physical-holder context isolation contract

`run_final_auto_cell` creates a live `QualifiedPhysicalLaunch` at
`tests/gemm_bi_performance_matrix.rs:5803-5805`, then passes that holder and the
same `GpuCtx` into `run_final_auto_comparator` at `:5807-5818`. The comparator
uses that context for vendor allocation, seeding, an eager cuBLAS launch, and
cuBLAS graph capture at `:5680-5689`, and later for every vendor calibration,
warmup, and measured window at `:5711-5721`, `:5735-5745`, and `:5758-5769`.

That is explicitly outside the facade contract. The qualification module says
a live holder excludes unrelated work on its context
(`src/mamba_ssm/gpu/gemm_bi_triad/qualification.rs:3-6`), and the holder's own
documentation repeats “Do not enqueue unrelated work ... on that context while
the holder is live” (`:1821-1824`). A passing smoke does not turn this into a
supported use of the physical evidence/lease.

Smallest bounded correction: create a second `GpuCtx` for the vendor arm, keep
the qualified AUTO holder exclusively on the AUTO context, and thread the two
contexts explicitly through allocation/capture/calibration/collection. Keep
AB/BA sequential and keep each arm's event timing on its own stream. Both
contexts must outlive the holder, cuBLAS handle/workspace, buffers, and graph.
The existing tournament already demonstrates this ownership pattern at
`tests/gemm_bi_performance_matrix.rs:7334-7340` and `:8442-8519`.

### Important 2: the timed vendor allocations never prove required base alignment

`allocate_cublas_denominator_buffers` constructs output/A/B with direct
`DtypedBuf::zeros` allocations at
`tests/gemm_bi_performance_matrix.rs:4882-4916`. The new adapter immediately
seeds, warms, captures, and times those buffers at `:5680-5689`; it never checks
the three actual device addresses. The persisted benchmark rule requires
256-byte base alignment to be checked because a previously measured offset
materially distorted cuBLAS timing. CUDA allocation alignment is a reasonable
expectation, but an unverified expectation is not release evidence.

Smallest bounded correction: after vendor allocation and before warmup/capture,
fail closed unless `output.cached_ptr()`, `a.cached_ptr()`, and
`b.cached_ptr()` are each divisible by 256. Cover the validator with a narrow
native test if factored as a pure address check, and record the admitted
alignment in the pair metadata or otherwise make it explicit in the frozen run
receipt. Do not introduce offset guards into this timing cohort.

## Confirmed requirements

- `final_auto_cell_inventory` selects actual policy routes rather than forced
  tiles: 15 exact F32, 21 deterministic-TF32-allowed F32, 15 BF16 TC, and 15
  F16 TC cells. The tests require 66 unique IDs, 81 comparator views, and 324
  path/order records (`:9935-10020`); full mode repeats those exact assertions
  at entry and completion (`:8636-8647` and the sink completion check).
- Comparator semantics are correct: exact F32 has both Fast-TF32 throughput
  and Pedantic precision-aligned views; AllowTF32 has Fast-TF32; native BF16/F16
  use `CUBLAS_COMPUTE_32F` with `CUBLAS_GEMM_DEFAULT`. TN output is F32 for all
  input types (`:3688-3752`, `:3921-3951`, `:4919-4959`, `:9917-9932`).
- The physical facade receives each `F32Policy`/`HalfPolicy` request and
  validates the timed request before timing (`:4388-4409`, `:4465-4467`,
  `:5794-5805`). Thus emitted physical nodes, module owner, compiler identity,
  artifact set, policy revisions, and driver identity describe actual AUTO.
- Eager and whole-graph arms are both measured in AB and BA order. AUTO and
  vendor are independently calibrated once per path, the calibrated counts are
  reused for the mirrored order, raw arm samples are retained, and ratios are
  computed sample-wise as `AUTO/vendor` (`:3955-3977`, `:5452-5523`,
  `:5691-5791`). There is no threshold or staged admission.
- TN reset is outside each start event for calibration, warmup, and measured
  windows. AUTO and vendor use matching deterministic active-value generators;
  each timed window then intentionally performs repeated beta=1 accumulation.
  The record labels state both facts. NN/NT correctly use repeated beta=0
  overwrite (`:5371-5435`, `:5512-5523`, `:5639-5648`).
- Vendor graph capture occurs before timing and uses the context's retained
  cuBLAS handle/workspace. Within the comparator function, the graph is declared
  after its buffers, so normal reverse drop order releases the graph before its
  captured buffers. `capture_into_graph` rejects a missing captured graph and
  does not impose a private cuBLAS node ABI.
- Full-mode defaults to 21 windows/order, strict focused cell filters reject
  empty, duplicate, whitespace, and out-of-inventory IDs, and the board gate is
  exactly CC8.9/142SM or CC12.0/170SM (`:8611-8667`, `:10022-10052`).
- Pair records contain Git SHA, GPU UUID/CC, logical cell and dtype, requested
  vendor compute/algorithm, path/order/iterations, reset semantics, raw samples,
  physical launch evidence, compiler/NVRTC/artifact/policy/driver identity, and
  quiet pre/post snapshots. Completion adds SM count and an exact record count
  plus cohort digest (`:5219-5317`, `:5525-5665`).

## Observed root smoke (diagnostic only)

Root reported a fresh CUDA 13.2/Ada build pass, 4/4 native tests, and a
four-cell/five-view smoke yielding 20 pair records plus one completion record in
188.41s. The local receipt is
`internal/perf/final-auto-benchmarks-20260910/triad-ada-smoke/`; the JSON parses,
has 20 unique `(cell, denominator, path, order)` keys, includes both exact-F32
denominators, records TN BF16 output as F32 with the required reset labels, and
has matching three-sample arm/ratio arrays. This is useful diagnostic evidence,
but it does not clear either Important finding and must not be promoted as the
final release cohort.
