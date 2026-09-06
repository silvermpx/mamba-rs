# Ada RNA-wide toolkit census — CUDA12.8 and13.0

Frozen production commit: `208c740c3f24c71f8bc090dbf597f7ee1a129fbd`.
RTX6000Ada, CC8.9/142SM, driver595.45.04,300W. Measurement only; no
production/test/dispatch changes and no AUTO promotion in this checkpoint.

Both matching release builds and both cold/warm forced correctness runs pass
(2 tests per run). Each toolkit has complete21- and101-window paired runs:
40 unique records, zero rejections, all raw/repeat/applicable graph bits pass.
The actual forced graph identifies the Fixed-owned RNA symbol, one kernel,
256 threads,98304 shared and correct tile grid. CUDA-major ABIs are not mixed.

Main independently checked all160 records, samplewise quantiles, physical
identity, shapes, matching known NVRTC, FAST/bias semantics, the requested
21/101 positive finite samples per arm, and exact completion counts. Main rehashed all167 local and
remote build inputs, the four live executables and six private-cache blobs.
See `main-independent-verification.json` and the two `main-remote-*.log` files.

## Confirmed101 results

Ratios are RNA time divided by the denominator, below1 favors RNA. Each p95
is the worst paired p95 over both paths and both launch orders. Full p50/p95
and per-cell raw samples are in `acquisition-report.md` and toolkit logs.

| CUDA | Own wins | FAST wins | A+bias RNA/FAST p95 | Worst RNA/FAST p95 |
| --- | ---: | ---: | ---: | ---: |
|12.8|10/10|1/10|0.955545|1.496884|
|13.0|10/10|1/10|0.973033|1.498848|

Only A with bias beats FAST in these two builds. B with bias still loses by
about6.3%/6.4% at worst paired p95; CUDA13.2's separate B+bias victory cannot
be transferred here. All ten cells are faster than their current ordinary
AUTO routes, but this is not an all-FAST win. ExactF32, half/mixed, Triad and
other GPUs were not measured in this task. No cross-toolkit bit-equality
claim is made from within-toolkit cross-rung repeat checks.

The vendor is explicitly `CUBLAS_COMPUTE_32F_FAST_TF32`, with bias broadcast
timed when needed. PEDANTIC is the independent numerical reference only.

## Remaining qualification before AUTO

The frozen force helper covers tail/hot-A exceptional/prefix/C4 views and
all five incumbent rungs, plus K0/empty/unsafe inputs. Finite timing covers
A–E, but full B–E exceptional/prefix/views are tied to13.2 actual-AUTO mode.
That test-corpus coupling must be removed and all three matching builds
qualified before a separate AUTO widening. This limitation is not a kernel
failure or permission to weaken the tests. Current AUTO remains13.2-only.

The initial12.8 build failed due to an omitted required test-support source
during sync; the preserved first log records the failure. After correcting
that acquisition omission, the same build passed. The production source was
not changed. The side-by-side CUDA13.0 installation transcript and verified
post-install state are in `environment/`; the default CUDA13.2 and GPU driver
were retained. No installation or build overlapped timed runs.

`SHA256SUMS` covers every evidence file except itself. Preserve raw log
whitespace; do not rewrite timing logs to satisfy a source-formatting check.
