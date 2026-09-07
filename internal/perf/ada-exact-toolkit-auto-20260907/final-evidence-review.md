# Task8 final evidence acceptance — bounded closeout

Date: 2026-09-07

Scope: the user-approved narrow closeout supersedes the earlier prepared full-
evidence brief. I compared `internal/perf/ada-exact-toolkit-auto-20260907/final-report.md`
against the current final2 bindings, saved raw post101/smoke records, final2
build and functional receipts, and the already-accepted source/fix reviews. I
did not repeat the source review, run GPU/SSH/Cargo/build commands or test
suites, inspect a new binary/cache archive, or change source, index, branch, or
HEAD.

## Verdicts

- **Spec compliance: PASS.** The evidence supports the 16 new CUDA 12.8/13.0
  exact-F32 AUTO literals, retention of the eight CUDA 13.2 routes, tuning
  revision 44 with numerical ABI 5 and schedule 8, and no broader Fast, Triad,
  architecture, or release claim.
- **Task quality: PASS.** The final report agrees with the checked raw records,
  bindings, receipts, and accepted reviews. I found no concrete discrepancy.
- **Acceptance recommendation: ACCEPT for human commit**, after the advertised
  mechanical selected-evidence manifest generation/replay and exact staging.
  That post-review packaging step does not require another source, GPU, build,
  or performance gate.

## Findings

### Critical

None.

### Important

None.

### Minor

None.

## Evidence checked

- Final report SHA-256:
  `0fc7fe65c22a1a0f3843f4e9c26f70f1468830e9cf9cc0b6c898dbef37db14a6`.
  Accepted source review SHA-256:
  `4621e60df2995545b73496120388e0dcf52a2bdcc14bf1ddeaf6ca6b56d51aec`.
  Accepted fix1 review SHA-256:
  `7dee4a81a2c9b9ff5e8aacd6ec1c8222ff5841054e3f251d1d10f4c635fa7b42`.
- Independent local replay of both post101 runs and their smoke dependencies
  passed. The raw SHA-256 values are
  `62d5d28386c587eb7ae7764b5ec20894ae8f6bd7d8b7a6cc4bb52dc96ce468b7`
  (CUDA 12.8) and
  `a7b80ab4ad84488ebfaa006909a43dc85ba1db648793b086c44b76e8a623a21e`
  (CUDA 13.0). Each post run contains exactly 38,784 samples, 9,696 pairs,
  96 summaries, 32 completed configurations, eight decisions and one complete
  record; all eight literals are eligible. Recomputed worst-stratum p50/p95
  values agree with every displayed final-report value to its six decimals.
- Both saved analyses are `valid: true`. Direct predicates over all 16 physical
  records confirmed public-AUTO verification, custom and Fast repeat bits,
  poison readback, rejected no-op graph, guards, immutable inputs, bias
  orientation, finite ordering, and the complete one/20 observable graph
  inventories. Legacy/AUTO nodes retain their recorded pointer, parameter-word
  and driver-ABI arrays; Fast bias records correctly contain bias broadcast
  plus GEMM per logical operation rather than implying a private vendor ABI.
- Each final2 binding identifies the correct toolkit, the shared four-file
  measured digest
  `2ac1c93ba3e4682138a0df3341a26921438334382532d6943277268320655285`,
  357 matching source inputs, five executables and the current support tools.
  The all-three final2 build receipts have successful outer/SSH closure and the
  reported test counts. The all-three functional results have zero exits for
  the five Fixed/half checks; CUDA 13.2 alone additionally has zero exits for
  the two retained Triad checks.

## Staging boundary

`final-report.md:96` names `ARCHIVE_SHA256SUMS`. It was intentionally not yet
present at this pre-manifest checkpoint because the selected local-text/raw
manifest is generated only after this verdict is copied into the evidence
root. This is not a missing binary archive or a finding in the reviewed
evidence. Acceptance assumes root completes and replays that advertised
mechanical step before the human commit; omitting the file while retaining the
sentence would instead make the committed report inaccurate.

This review does not claim that any inference Fast gap is closed, does not
qualify lower-toolkit Triad behavior, and does not add a quiet-GPU release gate
before profiling the already-frozen binary.
