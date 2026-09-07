# Ada exact-F32 AUTO44 integration — 2026-09-07

## Result

All 16 new CUDA12.8/13.0 A/B/D/E bias/no-bias literals select the previously
qualified CopyPlan champion through actual public AUTO. The eight existing
CUDA13.2 routes remain selected. Tuning revision is 44; numerical ABI 5 and
schedule 8 are unchanged. No CUDA body, loader composition, other architecture
or retained TF32/half preference changed.

Both once-only actual-AUTO post101 runs passed: every literal wins p50 and
p95 against former Legacy in all four eager/graph/start-parity strata.
Worst-stratum median savings are 15.4–18.5%. These are own improvements,
NOT cuBLAS Fast victories. Fixed inference and Triad still have open Fast gaps.

## Paired results

Ratios below are the worst quantile across four strata for each literal
(lower is better). All individual strata and raw samples are preserved.
A=(4621,384,1928), B=(4621,768,2304), D=(2048,768,2304),
E=(2048,2304,768), ordered M/K/N. Suffix 0/1 is bias disabled/enabled.

| CUDA | Literal | AUTO/Legacy p50 | AUTO/Legacy p95 | AUTO/Fast p95 |
| --- | --- | ---: | ---: | ---: |
| 12.8 | hot_a:0 | 0.843559 | 0.886541 | 1.882699 |
| 12.8 | hot_a:1 | 0.841745 | 0.887635 | 1.540643 |
| 12.8 | hot_b:0 | 0.827182 | 0.849500 | 2.236558 |
| 12.8 | hot_b:1 | 0.819581 | 0.843463 | 1.961896 |
| 12.8 | hot_d:0 | 0.827371 | 0.846753 | 2.081084 |
| 12.8 | hot_d:1 | 0.829177 | 0.861947 | 1.832299 |
| 12.8 | hot_e:0 | 0.815077 | 0.835972 | 2.225878 |
| 12.8 | hot_e:1 | 0.814778 | 0.838741 | 2.102022 |
| 13.0 | hot_a:0 | 0.845826 | 0.881467 | 1.872059 |
| 13.0 | hot_a:1 | 0.842909 | 0.888879 | 1.543477 |
| 13.0 | hot_b:0 | 0.824087 | 0.848835 | 2.240107 |
| 13.0 | hot_b:1 | 0.817183 | 0.841708 | 1.960677 |
| 13.0 | hot_d:0 | 0.828871 | 0.848594 | 2.082000 |
| 13.0 | hot_d:1 | 0.829190 | 0.860462 | 1.830159 |
| 13.0 | hot_e:0 | 0.815423 | 0.830017 | 2.221576 |
| 13.0 | hot_e:1 | 0.815353 | 0.839425 | 2.108478 |

Comparator: actual public AUTO versus forced former Legacy and cuBLAS
CUBLAS_COMPUTE_32F_FAST_TF32, default math/algorithm, host pointer mode,
atomics disallowed. Bias work is inside every measured logical operation.
PEDANTIC is an untimed correctness reference, not the speed denominator.
Each run has 128 eager warmups per arm, 20 operations per observation,
101 alternating ABBA/BAAB windows with reversed comparison traversal,
38,784 samples, 9,696 paired ratios, 96 summaries and 32 configurations.
The root independently recomputed every ratio, rounded-order-statistic
quantile and admission decision from raw JSONL, including its matching
one-window smoke, identity and same-attempt exit/telemetry closure.

## Qualification and binding

All three final2 builds pass: library 647 tests (46 ignored), focused 14,
performance nonignored 67, exact nonignored 2, half nonignored 2,
Fixed correctness nonignored 1. The Triad cohort binary has 3 ignored GPU
tests; its nonignored run alone is not GPU qualification.
Live Fixed functional checks pass 5/5 on CUDA12.8, 5/5 on CUDA13.0 and 5/5 on
CUDA13.2: exact AUTO, retained half holders/AUTO and both RNA tests.
CUDA13.2 additionally passes the two retained Triad cohort/bias GPU checks;
there is no new lower-toolkit Triad cohort claim.

The final2 binding files contain 357 matching source inputs and five exact
executable hashes per toolkit. Root rechecked all input hashes against the
working tree. The nine measured Rust files remain the accepted source;
four-file NUL-framed measured digest:
`2ac1c93ba3e4682138a0df3341a26921438334382532d6943277268320655285`.

CUDA source/invocation/artifact identities remain exactly those from the
previous forced-kernel qualification (Task6A); the new AUTO44 selection and
captured graph behavior were tested on the final2 builds. Physical one/20
graph inventories, argument values, poison/guard/input-immutability,
repeat and negative controls are in each raw physical record.

Source review and its narrow fix re-review accepted the implementation.
The two fixed findings were real builder-map/performance-binary binding and
the correct Fixed RNA all-toolkit functional inventory. Original failures,
final1 build receipts and final2 corrected receipts are retained separately.
Host analyzer adversarial tests pass 7/7; direct rustfmt and tracked diff
whitespace checks pass. These host checks do not replace the saved GPU tests.

## Evidence entry points

- `cuda128-post101-final2/records.jsonl` SHA256:
  `62d5d28386c587eb7ae7764b5ec20894ae8f6bd7d8b7a6cc4bb52dc96ce468b7`.
- `cuda130-post101-final2/records.jsonl` SHA256:
  `a7b80ab4ad84488ebfaa006909a43dc85ba1db648793b086c44b76e8a623a21e`.
- Matching `cuda{128,130}-smoke-final2/`, `cuda{128,130,132}-binding-final2.json`,
  `*-build-final2/` and `*-functional-final2/` preserve source/build,
  correctness, command, telemetry and exit evidence.
- `run.py`, `remote.py`, `analyze.py`, `test_validation.py` are the exact
  final tools. The analyzer explicitly reuses the frozen Task7 analyzer.
- Prior Task7 candidate qualification is committed at b78aebf4 under
  `internal/perf/ada-f32-tf32-toolkit-20260907/`; it is not relabeled AUTO44.
- `ARCHIVE_SHA256SUMS` identifies selected local evidence.
  Large executable/cache copies, when retained separately, are provenance
  artifacts, not required to rerun the saved arithmetic. No new binary-archive
  replay or GPU release receipt is implied by this report.

## Next work, without another blanket qualification cycle

The user approved cheap discovery on CUDA13.2, minimal meaningful numerical
checks and short paired screens, then full qualification of a frozen
integrated finalist batch. See performance-playbook section 9. The GPU owner
moves directly from the completed post101 to targeted TF32 E0 production/Fast
Nsight. Do not repeat already-losing M64/N64 sweeps. Include the worst Triad
cell in the next discovery cycle; universal inference Fast victory is not a
prerequisite. No new branches, kernel deletion, 5090 qualification or release
claim is part of this integration.

