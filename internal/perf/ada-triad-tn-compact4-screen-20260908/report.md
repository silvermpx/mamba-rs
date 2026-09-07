# Ada TN Prism compact-four-warp S2 screen

Outcome: **STOP, no retry or promotion.** The compact four-warp candidate
passed its resource and focused numerical gates but was slower than current
actual AUTO in every paired once7 stratum.

| Path | Order | candidate/AUTO p50 | candidate/AUTO p95 |
|---|---:|---:|---:|
| eager | ABBA | 1.010186462 | 1.026981604 |
| eager | BAAB | 1.022048799 | 1.025695419 |
| graph | ABBA | 1.042430483 | 1.045716825 |
| graph | BAAB | 1.043422481 | 1.054005563 |

The candidate used the distinct test-only symbol
`gemm_bi_tn_test_compact_four_warp_sm80_mma_tf32_v1_m128n64_bk32_s2`
at logical `(4621,384,1928)`, grid 93, 256 threads and 49,152 dynamic
shared bytes. CUDA 13.2 reported 156 registers, zero local/static shared
bytes and one resident CTA/SM, satisfying this ablation's intentional
one-CTA gate. Actual AUTO remained the production M128N64/BK32/S3 TN route
with the same grid and 79,872 dynamic shared bytes.

Focused correctness passed for the full-mantissa target and tail
`(129,65,36)` with alpha 1 and -0.75, eager2/graph2 repeats, immutable inputs
and red zones. TN zero reduction `(0,65,36)` passed eager2/graph2 against the
finite analytical `fma(1,+0,oldC)` oracle, including signed-zero
canonicalization. Each beta-1 timed observation independently restored the
same C, A and B outside the event and downloaded outputs, inputs and guards
before the next reset.

The first release-build attempt is retained as a transfer-path failure: the
new helper was initially copied to `tests/` instead of `tests/support/`, so
the compiler could not resolve its two new exports. The exact accidental copy
was removed, the frozen helper was synced to its intended path, and the
incremental build then passed. No GPU test ran from the failed build.

The accepted run executed exactly one test successfully, with quiet PRE, an
immediate busy RELEASE, a separate quiet DRAIN snapshot five seconds later, and an unchanged
private four-artifact production cache. Candidate PTX was not persisted.

Bindings:

- harness source: `8b921bbb65a0310bccf14c8e575d95fce1fed1e706c049bea9f9e8a2904a2e47`
- source helper: `cfcbe3146b513fdf51ec82e95059d535df0a8fa598f583adfdd41c0bcebd1924`
- composed CUDA source: `22d9e88d1792b842c61960309748c398020d08f30ffa20ee2f151d6d328144ee`
- test binary: `91bfbe972b96a99b9ca9a6b29e26dfc72264d28233cfee955e5a8c8ff2f729ae`
- source manifest: `deb21379513ccf6fa9c7c429e82fe1b34aa92fcb050cb0111fe07b6464df237a`
- raw test log: `624a4139cd9f134c42c0e9ad7ba272cedbc7a9139c971698d0604f3ce215d51c`

Evidence is under `evidence/`. This was a bounded test-only discovery screen,
not full qualification and not a Fast comparison. Production code and AUTO
selection were unchanged.

Commit-only formatting note: the final rustfmt check rejected the multiline
cohort assertion in measured8b921bbb. Root applied only that line wrapping,
producing b71a9253ba99561530c8671be765378955cd3b33ba0c66f614a2e0f842654a06.
The exact diff and whitespace-normalized equality confirm no token change;
fresh native31/31 and rustfmt pass. Raw evidence stays bound to8b, not b71a.
No additional GPU run was performed for this cosmetic assertion formatting.

Before that cosmetic formatting, root independently verified all379 manifest rows (378 local files plus
the committed unrelated SM120-test baseline), raw-log and manifest hashes,
all28 brackets/112 one-GEMM observations, arm means, paired ratios and the
four p50/p95 STOP decisions. Native full harness31/31 passes. Independent
source, runtime-delta and runner reviews accept this scoped checkpoint.
The source helper's native ownership RED was genuine; the initial undefined
harness enum compile failure is not described as a behavioral TDD RED.

Graph median arm means: actual AUTO249.824us in both orders, candidate
260.400/260.896us. Eager ABBA has visible drift; use the paired ratios above,
not a ratio of independently selected medians. No successful stratum was
retried or selected away. This is not proof of a silicon limit.

Classification: rejected only for this Ada/CUDA13.2 Prism TN screen. Keep the
existing AUTO route and RTX5090 winners; both compact TN variants remain
test-only, available for later deliberate cleanup. Do not resweep these same
variants in the next search.
