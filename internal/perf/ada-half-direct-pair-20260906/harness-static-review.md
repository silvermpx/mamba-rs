# Ada half direct-pair harness static review

## Verdict

**APPROVE — static specification and code quality only.** I found no blocking
source defect in the packaged `tests/gemm_bi_fixed_performance.rs` change. The
workflow truthfully measures only the forced pipeline and swizzle arms, binds
both captured graphs to their live measured callsites, and emits the required
two-arm raw evidence. This does **not** complete Task 2: the interim report
correctly leaves the all-three-toolkit covering gates and quiet CUDA 12.8/13.0
21/101 measurements unverified and makes no performance or AUTO-admission claim
(`ada-half-direct-pair-report.md:3-7,136-144`).

The reviewed file SHA-256 is
`3873a40ac14a05615c5758bb3181948e6e4944486400387552de08a8ecb90d3a`,
matching the supplied milestone. The package changes only the allowed shipped
test file. I did not run builds/tests, use SSH/GPU, or inspect changing CUDA
source.

## Findings

None in the bounded static scope.

## Contract and risk checks

- **Strict cohort boundary and old-filter preservation:** the shared filter keeps
  the prior absent-means-all behavior and rejects empty, unknown, duplicate, and
  trailing-empty values (`tests/gemm_bi_fixed_performance.rs:10416-10451`). The
  new workflow exposes only homogeneous BF16/F16 rows, exact hot A-E shapes,
  bias 0/1 and eager/graph paths, bounds windows to 1..10001, and rejects any
  presence of `MAMBA_FIXED_VENDOR_TILES`
  (`tests/gemm_bi_fixed_performance.rs:13439-13473`). Release mode, exact CC
  8.9/142 SM, known NVRTC, and the 12.8/13.0/13.2 roster all fail closed
  (`tests/gemm_bi_fixed_performance.rs:13431-13437,13475-13502`). Refactoring
  `fixed_ada_filter` changes error transport but not accepted selections.

- **Truthful two-arm timing:** pipeline and swizzle use distinct output buffers
  and fixed route closures (`tests/gemm_bi_fixed_performance.rs:13547-13552,
  13603-13618`). The timed window helper invokes only those closures; reversed
  order actually executes swizzle then pipeline while returning values in stable
  arm order (`tests/gemm_bi_fixed_performance.rs:10462-10475,13780-13797`). AUTO
  and PEDANTIC launches occur only in correctness setup, outside every timed
  window (`tests/gemm_bi_fixed_performance.rs:13581-13601`). Each path warms both
  arms 128 times and independently calibrates with the existing 16-launch,
  approximately 5 ms, 1..4096 rule
  (`tests/gemm_bi_fixed_performance.rs:13742-13752`; `:9216-9222`). Event timing
  rejects non-finite or non-positive results (`:9603-9621`).

- **Actual AUTO, reference, poison and bits:** the actual production AUTO route
  is launched in the same context and checked as Tc128 for 12.8/13.0 or pipeline
  for 13.2. Both forced arms must match its raw storage bytes and all three
  outputs must meet the finite PEDANTIC F32 normalized-error bound
  (`tests/gemm_bi_fixed_performance.rs:13498-13502,13581-13638`). Complement
  poisoning covers every output byte independently before forced-repeat and each
  eager/graph path check; graph mode performs two separately poisoned replays
  (`tests/gemm_bi_fixed_performance.rs:13401-13424,13640-13663,13715-13741`). Both
  arms are re-poisoned and bit-checked before each order and checked again after
  timing (`:13754-13807`), so stale/partial writes and timing-order corruption
  fail before a record is emitted.

- **Measured physical graph identity:** both route closures are captured before
  the selected-path loop, so the gate remains mandatory for eager-only runs, and
  the graphs/buffers remain in scope through all replays and measurements
  (`tests/gemm_bi_fixed_performance.rs:13611-13618,13665-13713`). Inventory is
  passed the exact tile, dtype, live operands and shape. It rejects non-kernel
  work and, through the half descriptor, validates all five Driver parameter
  offsets/sizes, rejection of argument six, four captured pointers and the
  captured eight-word bundle (`:11559-11595,11694-11745`). The underlying
  contract requires one kernel, the typed route symbol, flat
  `(ceil(M/128)*ceil(N/128),1,1)` grid, block `(256,1,1)`, pipeline 71,680 or
  swizzle 69,632 shared bytes, and alpha/beta/dimensions/strides matching the
  measured shape (`:11284-11339`). There is no substitute identity launch.

- **Paired samples and schema truthfulness:** samples are appended at the same
  window index for each requested order. Both ratio directions are separately
  formed from those same-index pairs, sorted, and reduced with the project
  percentile function; reverse p95 is not inferred by reciprocal
  (`tests/gemm_bi_fixed_performance.rs:10477-10505,13780-13816`). Emitted records
  contain only pipeline/swizzle timing fields and preserve actual AUTO strictly
  as a correctness proof (`:13817-13873`). Completion includes tuning revision,
  count, zero rejects, and pass state (`:13880-13884`). Default loops yield the
  specified 80 records; independent analysis remains responsible for requiring
  a complete unfiltered measurement run.

## Tests inspected and remaining evidence

I inspected the three added literal-behavior tests. They prove actual reversed
closure scheduling, same-index bidirectional quantiles with a non-reciprocal
tail, strict row-filter failures, and rejection of any vendor-tile setting
(`tests/gemm_bi_fixed_performance.rs:10507-10556`). I also inspected the named
graph contract and timing/filter dependencies. The interim report records the
focused helper GREEN (3 tests) and CUDA 12.8 one-window eager/full functional
smokes; I did not duplicate them.

Still outside this static approval are CUDA 13.0/13.2 builds, nonignored suites,
focused/full smokes, old-census compatibility checks, final identity/cache/source
reconciliation, and quiet complete 21/101 direct-pair measurements and analysis.
One-window smoke evidence must not be used as speed evidence.
