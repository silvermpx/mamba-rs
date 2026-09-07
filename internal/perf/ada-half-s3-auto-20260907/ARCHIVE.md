# Production Ada S3 AUTO43 checkpoint

Production `fixed_forward` now selects S3 for homogeneous BF16/F16 B0
`(M,K,N)=(4621,768,2304)`, no bias, on the qualified RTX6000 Ada
CC8.9/142-SM target with known NVRTC13.2. CUDA12.8/13.0 keep Swizzle for
these cells. All other preferences and holder-unavailable fallbacks remain
unchanged; S3 stays force-reachable on all three toolkits. No kernel was removed.

One fresh actual-AUTO43 post101 run confirms the production improvement:

| Dtype | AUTO/old-Swizzle worst p50/p95 | AUTO/cuBLAS Fast worst p50/p95 |
| --- | --- | --- |
| BF16 | 0.929889195 / 0.953700555 | 1.207046379 / 1.231223219 |
| F16 | 0.929018443 / 0.955993909 | 1.133473338 / 1.159192520 |

Lower time ratios are better. Every eager/graph/start-parity stratum wins both
own quantiles: at least7.01%/7.10% lower median time than the previous route.
Native-half cuBLAS Fast still leads this B0. This is not a vendor victory,
an end-to-end inference result, a new full60-cell sweep, or Triad closure.
PEDANTIC F32 is only the separate numerical reference.

The post101 raw log is `remote-fix1/cuda132-postconfirm101/test.log`, SHA256
`b6d0fca4305b6846a427c9dc8432e727dfa5434e68f380ca29511916a8c7d620`.
It has8 configurations,9696 raw observations,2424 paired ratios and24
summaries. Actual public AUTO routes, captured symbols/full arguments,
bitwise-complement upload/readback, real empty-graph overwrite negative,
repeated bits, guards, input immutability and numerical checks pass before
timing. Exact eight-record SSH/telemetry/command/result closure has all0 exits.
`root-recomputed-pairs.json` independently reconstructs smoke and101 arithmetic
and chronology; final `analyze.py` validates physical and same-attempt gates.

All three matching-toolkit full libraries pass645 tests, focused stage tests5,
nonignored performance53 and nonignored pipeline2. Nine hot-corpus GPU tests
(AUTO, forced Swizzle, forced S3 per toolkit) pass. Unavailable SM120 runtime
tests remain ignored, not claimed as GPU coverage. The selector oracle covers
60 cells across8 independent holder-availability combinations. Tuning42 graphs
reject under43 for re-capture; numeric5/schedule8 and compiled modules are
unchanged. Stale-control and pre-warmup-input gaps from Task6B are closed.

`root-build-input-checks.json` verifies all3x356 bound compiler inputs and only
seven authorized Rust changes. `root-artifact-checks.json` verifies all three
actual executables and all nine complete cache envelopes. Fixed PTX matches
Task6A exactly; every Fixed and Triad cache is byte-identical to Task6B.
The invalid wrong-environment holder loop, original report-label bug, expected
negative tests and unusable archive attempt remain clearly distinguished from
valid evidence. Fix3 changed report labels only; no timing was repeated.
The first library failure is disclosed as tool-transcript-only, not a saved
raw log. `final-report.md` contains the exact history and all24 constituents.

Independent Rust/source-fix and host-source/fix reviews are approved. Final
evidence review is also approved, with no Critical, Important or Minor findings;
the verdicts and frozen source packages are preserved under `reviews/`.

`SHA256SUMS` is the frozen95-entry complete local manifest, SHA256
`ed93f74ae7fbeb6aa0d444abab2c34e8e2cc0e49d7edd33c453dabbeb51540c5`.
It includes five local tar.gz bundles, incidental Python bytecode, and the
original SDD report path. Root verified all95 entries. Large source/binary/
cache bundles remain on disk, not in Git; nothing was deleted. The identical
report is copied to `final-report.md` for the version-controlled checkpoint.
`ARCHIVE_SHA256SUMS` covers the selected version-controlled text/source evidence
and can be verified from the worktree root without the disk-only bundles.

Final Ada release is explicitly recorded at2026-09-07T04:56:10Z: exact UUID,
CC8.9, GPU0%/memory0%, no applications, telemetry/check/outerSSH exits0.
See `remote-fix1/final-lane-release-json.log` and its saved release JSON.

Next is paired qualification of existing exact-F32 CopyPlan and TF32-C M64S2
on CUDA12.8/13.0. That is a separate measurement-only task before any further
literal selector promotion; no new branch or kernel deletion is authorized.
