# Triad final AUTO benchmark fix 1 re-review

Date: 2026-09-10

## Scope

Read-only review of only the fix delta in
`triad-final-auto-fix1-review.diff` and immutable post-image
`triad-final-auto-fix-de4b689b.rs`. The post-image and live
`tests/gemm_bi_performance_matrix.rs` both have SHA-256
`de4b689bebdb35d0e2dd201cbdd774ab7b89b6896e3026d0da92e822566c7f45`.
No other Triad code, Inference code, or branch changes were reopened. No build,
test, or GPU command was run by this reviewer.

## Verdict

**Approved for the root-owned final hardware gate.** Both original Important
findings are addressed. No Critical or Important regression was found in the
fix delta.

## Original findings

### Important 1: vendor work shared the qualified AUTO context — addressed

The entry now creates distinct long-lived `auto_ctx` and `vendor_ctx` instances
(`triad-final-auto-fix-de4b689b.rs:8689-8694`) and passes both explicitly into
every cell (`:8703-8714`). The qualified physical holder is constructed and
validated only on `auto_ctx` (`:5825-5837`).

The complete vendor path stays on `vendor_ctx`: allocation and alignment
admission, initial seed, eager warm launch, synchronization, graph capture
(`:5694-5719`); vendor calibration and warmup (`:5741-5751`, `:5765-5775`);
and vendor TN reset plus eager/graph measured windows (`:5490-5542`). AUTO
seeding and timing in those same functions consistently use `auto_ctx`.
Therefore no cuBLAS work is enqueued through the `GpuCtx` protected by the live
`QualifiedPhysicalLaunch` lease.

Ownership remains sound: both contexts outlive every cell; the physical holder
drops before `auto_ctx`; and within each comparator, the vendor graph is
declared after its buffers and therefore drops before the captured buffers.
Sequential AB/BA semantics are retained even though each arm uses its own
stream, because each event-timed measurement completes before the next arm is
called.

### Important 2: vendor base alignment was assumed — addressed

`validate_final_auto_vendor_alignment` reads the actual output/A/B device
pointers and delegates to a fail-closed 256-byte check
(`:5417-5434`). `run_final_auto_comparator` invokes it immediately after
allocation and before any vendor seed, warm launch, or graph capture
(`:5707-5719`). Thus a misaligned arm cannot enter either eager or graph timing.

The native test accepts a fully aligned triple and independently rejects a
misaligned output, A, and B pointer (`:10073-10083`). Pair records now emit
`"vendor_base_alignment_bytes":256` only downstream of the successful gate
(`:5625-5664`). No offset allocation or timing-cohort change was introduced.

## Delta regression check

- Inventory selection, 66/81/324 count logic, comparator modes, strict filters,
  two paths, two orders, ratio orientation, and 21-window default are untouched.
- TN resets remain before each arm's start event on the correct context, and the
  separate `repeated_beta1_accumulation` label is unchanged.
- Actual AUTO route evidence and compiler/artifact identity still come from the
  physical holder on `auto_ctx`; vendor compute and graph behavior remain
  explicit.
- Added `alpha`, `beta`, and `bias` metadata accurately describe the existing
  fixed contracts: alpha 1 everywhere, beta 1 only for TN, and no bias in this
  66-cell inventory (`:5625-5664`).
- The delta does not change kernel code, production dispatch, numerical
  assertions, sample pairing, percentile calculation, or admission policy.

## Root verification status

Root reported a fresh CUDA 13.2/Ada compile pass in 5.17s and the focused
command

```text
cargo test --locked --release --features cuda --test gemm_bi_performance_matrix final_auto_ -- --nocapture --test-threads=1
```

passing **5/5**, including the new alignment validator test. The strict GPU
smoke was still running when this static re-review was completed; root owns its
result and the final hardware gate.
