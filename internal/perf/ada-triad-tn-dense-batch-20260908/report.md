# TN dense-stage three-cell discovery result

The test-only M128N64/BK32/S3 dense full-stage copy candidate passed its
resource and exact-bit gates on all three cells. It advances only for TN
d768-in; d768-out and Prism stop with no retry. This is a candidate-versus-
current-AUTO screen, not a Fast comparison or production promotion.

| Cell | Current AUTO physical route | Candidate grid | eager ABBA p50/p95 | eager BAAB p50/p95 | graph ABBA p50/p95 | graph BAAB p50/p95 | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| d768-in | M64N64/S2, grid 576 | 288 | 0.95093 / 0.95279 | 0.95009 / 0.95162 | 0.96056 / 0.97708 | 0.96179 / 0.96414 | advance |
| d768-out | M64N64/S2, grid 288 | 144 | 0.98341 / 1.00018 | 0.98337 / 0.99113 | 1.00936 / 1.02609 | 1.00754 / 1.01182 | stop |
| Prism | M128N64/S3, grid 93 | 93 | 1.00910 / 1.01806 | 1.01787 / 1.02837 | 1.03538 / 1.03702 | 1.03604 / 1.04437 | stop |

All candidates compiled to 156 registers, zero local bytes, zero static shared
bytes, 79,872 dynamic shared bytes, 256 threads, and one resident CTA/SM. Each
exact test emitted one resource record, four seven-window ABBA/BAAB timing
strata, and one decision record. Full-mantissa target, forced-portable-RNA
tail/alpha, finite K0, eager repeat, graph repeat, immutable-input, and red-zone
checks passed before timing. Each beta=1 observation independently reseeded
C+A+B and timed one GEMM.

The d768 comparisons are deliberately not described as a copy-predicate-only
ablation: live current AUTO uses M64N64/S2 with 128 threads and 36,864 shared
bytes, while the candidate uses M128N64/S3 with 256 threads and 79,872 shared
bytes. The harness asserted and emitted those distinct physical identities.
Prism compares the same tile/stage geometry.

The reused CUDA target contained a stale second executable. The inherited
build wrapper therefore declined to choose one automatically, while the build
itself succeeded. `build.log` names the 72f43542 executable, and the separately
preserved authoritative `--list` binds all three exact tests to it. See
`artifact-binding.json`. Candidate PTX was not persisted; the runtime resource
record and composed-source digest are the available candidate bindings.

Raw logs and receipts are under `evidence/`. Immediate RELEASE samples are
expectedly busy; the distinct five-second DRAIN samples are quiet for all
three cells. The production cache remained byte-stable. Root's independent
replay matched all 381 measured source rows, 84 brackets/336 observations,
all 12 p50/p95 pairs, and the single d768-in retain decision.
