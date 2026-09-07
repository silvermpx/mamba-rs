# Ada TF32 four-warp discovery: both arms rejected

CUDA 13.2, RTX 6000 Ada, CC8.9/142SM, E0 `(M,K,N)=(2048,2304,768)`,
no bias. Base `e79e7719`. Neither candidate changes the production kernel,
dispatcher, numeric ABI or schedule revision. Both stop after one seven-window
screen: no 21/101 extension, retry or profile. The previously committed
eight-warp N96 candidate remains the faster retained finalist.

| candidate | registers | dynamic shared B | local/static B | active CTAs/SM | median time, us | worst-order p95 / current RNA | worst-order p95 / Fast |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| M128N96/BK16/S5, four warps | 188 | 71680 | 0/0 | 1 | 121.48 | 1.020896 | 1.334692 |
| M128N128/BK16/S5, four warps | 250 | 81920 | 0/0 | 1 | 154.69 | 1.299583 | 1.698893 |

Times are descriptive medians of individual observations in the RNA pairs,
not ratios of independent medians. Current RNA is about119.05us and actual
cuBLAS Fast about91.06us in these runs. Both candidates also lose to the
committed N96: worst-order p95 ratios1.200035 and1.525871 respectively.
Static reductions in copy/conversion work did not produce a timing win;
the exact microarchitectural cause is not established without profiling.

## Small functional evidence

Both isolated GPU tests exit0. They check exact output bits versus forced
production RNA for E0/E1, M129/K36/N100 tails with both bias states, a K16
Inf/NaN case retaining the incumbent's K32-padded MMA sequence, finite M1
batch prefix and poisoned suffix, K0 with null A/B, guards/immutable inputs,
three eager/graph repeats, and nontrivial alpha/beta with old C. Candidate
symbol, grid/block/shared, five-argument ABI and runtime resources are checked.
These are discovery checks, not full all-toolkit qualification or release.

The shared test binary SHA256 is
`9957c23cd5209469d0f375c2075384f2708bfa0f851bc52a88257c07a658a4c8`.
Each run's binding records frozen Rust/CUDA sources, binary and runner hashes;
its test log records the composed-source/PTX identity. All four runs have exit0 and strict quiet
PRE/RELEASE records for UUID `GPU-d1edd7be-e88d-aed6-047d-622163306f0e`.
The local archive contains text evidence and hashes, not the CUDA binary/PTX.

## Host/build history and replay

`rust-green4.log` is the final build with six host passes and four intentionally
ignored GPU entries. Earlier RED/compile logs remain verbatim. In particular,
`host-ring-red.log` used a stale binary and is **not valid RED evidence**;
`host-ring-red2.log` is its valid behavioral replacement. Layout, K0 and K32
padding REDs are also retained. Older intermediate GREEN logs do not bind the
final runtime source.

Each timing run prints42 brackets: seven windows, both ABBA/BAAB orders, and
three comparators (production RNA, actual Fast, committed N96). Every bracket
includes all four positive finite observations and candidate/comparator ratio.
Root independently replayed all84 brackets and twelve per-order summaries,
source/runner bindings, process exit codes and quiet-lane closure. Quantiles
use sorted ratios and rounded `(n-1)*q` indices; table p95 is the worse order,
not the pooled p95 emitted by the completion record.

The manifest is repository-root-relative. No production promotion is intended
for these losing arms; retain the test-only sources/evidence to prevent repeats.
