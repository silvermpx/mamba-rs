# Ada retained-kernel assembly — 2026-09-10

The assembled TF32 and FP16/BF16 selectors pass actual eager/graph execution
on CUDA 12.8, 13.0 and 13.2. This closes this retained-winner integration block,
not the remaining exact-F32 assembly or the full 0.7.0 release gate.

## Scope and provenance

- Device: RTX 6000 Ada, CC8.9, 142 SMs. Exact GPU and driver are in each
  `device.csv`; compiler/artifact identities are emitted by the module tests.
- Source base: `69ed1c80` plus the retained half/TF32 assembly changes.
- Frozen CUDA source SHA-256:
  `c0f147ce21f56ba2d4a1d8a1b8b5e73708bed292a36e929c46abc2d84356e7da`.
- Module/half receipts precede the final Rust identity and graph-revision
  corrections. Forced/paired/post-AUTO receipts use the final corrected Rust
  snapshot and the same CUDA bytes. Per-run `source.sha256` records this split.
- Each toolkit used its own private kernel cache, populated by its cold
  module qualification and reused for subsequent unchanged-source checks.
  Before every GPU test, five consecutive samples required compute and memory
  utilization <=1% and free VRAM >=2048 MiB. No other workload was stopped.

## Actual production selection

| CUDA | Joint module exports | Half actual-AUTO cells | TF32 actual-AUTO cells |
|---|---:|---:|---:|
| 12.8 | 7/7 | 18/18 | 6/6 |
| 13.0 | 7/7 | 18/18 | 6/6 |
| 13.2 | 7/7 | 18/18 | 6/6 |

Half checks cover ten symbols across NN/TN/NT, FP16 and BF16: exact physical
selection, retained-oracle bits, eager/graph bits, immutable inputs and red
zones. Seven retained additions are now included beyond the original eleven
cells. These runs confirm correctness and selection; they are not new half
timing comparisons on the lower toolkits.

TF32 checks cover TN d768-in/out/Prism, NN Prism/d768-out, NT d768-in. All
six pass prior-route output bits, repeats, eager/graph manifests and guards.
CUDA13.2 selects new NT A-ldmatrix N96/S3 and TN Prism M64N96/S2. CUDA12.8/13.0
retain NT compact8, TN Prism M64N64/S3 and portable NN Prism M128N128/S3.
The other admitted joint routes remain active. No toolkit-wide performance
decision is inferred from CUDA13.2 alone.

All seven joint exports have zero local memory. NT uses 108, 108 and 110
registers on 12.8, 13.0 and 13.2 respectively; the final per-symbol cap is 110.
Initial module logs record the earlier loose cap of 131, while final post-AUTO
loads validate the tightened cap. NT uses 86,016 B dynamic shared memory;
TN M64N96/S2 uses 40,960 B. No symbols are excluded.

## Final production-body timing, CUDA13.2

Both orders (ABBA/BAAB), eager and whole graph, once3 screen followed by
once7. Ratios are candidate time divided by the retained comparator time.

| Replacement | Path | Candidate p50, us | Retained p50, us | Paired p50 ratio |
|---|---|---:|---:|---:|
| NT d768-in N96/S3 vs compact8 | eager | 186.586–186.619 | 200.863–200.876 | 0.92884–0.92907 |
| NT d768-in N96/S3 vs compact8 | graph | 186.342–186.344 | 200.634–200.634 | 0.92870–0.92879 |
| TN Prism M64N96/S2 vs M64N64/S3 | eager | 139.763–139.783 | 173.253–173.271 | 0.80670–0.80681 |
| TN Prism M64N96/S2 vs M64N64/S3 | graph | 138.711–138.720 | 172.104–172.120 | 0.80586–0.80600 |

All eight official strata pass p50 and p95 <0.99. This confirms roughly 7.1%
and 19.3–19.4% lower execution time after production promotion. No new cuBLAS
Fast timing arm was run here; earlier discovery still records Fast deficits.
The inherited raw field `candidate_over_prior_portable` is a legacy label:
these two comparators are the named retained SM89 bodies, not portable SM80.

Raw paired log:
`cuda132/paired/live_sm89_tf32_joint_two_new_production_bodies_once3_then_once7.log`;
SHA-256 `812dc1e769c2ca49d8c4056687c9002cef729acc483fd50ef9abb589bf09ed58`.

## Verification boundaries

Native source/selector/cohort/half checks: 129, 2, 1 and 20 passing tests.
CUDA-feature static checks on the 5090 host: joint 7, half 38, half source 21
and kernel identity 23 passing tests; both retained discovery targets compile.
Independent scoped review approves the TF32 wiring and refreshed identities.
Scoped formatting and `git diff --check` pass. Repository-wide formatting
still reports older discovery-file formatting differences, deferred to release
cleanup; it is not claimed green.

The six-cell smoke proves the admitted target shapes. It does not extend the
NT discovery receipt into unrun tail/exceptional/K=0 tests. Source parity pins
the measured NT body; the existing TN discovery includes its broader probes.
See the separate RTX 5090 assembly smoke report for preservation of that GPU's
routes. No deletion, API rename or release publication is part of this block.
