# Ada direct TF32 Triad: two-cell physical diagnosis

The selected current AUTO kernels are not tensor-throughput saturated in these
captures. NT has a concrete excess shared-copy transaction signal; TN also has
fewer output CTAs than SMs. Neither observation proves the entire cuBLAS Fast gap
is removable or that changing arithmetic association is necessary.

## Bound acquisition

CUDA13.2, RTX 6000 Ada CC8.9 / 142 SM, driver595.45.04. Nsight Compute
2026.1.0.0 build37166530. Both symbols belong to the direct
`RegisterCvtRnaTf32F32V1` numerical family, not scalar fallback or split8.

Source HEAD59f1a949, runtime baseb221b71e; build command exit0. Same binary SHA256
`cf38911f01f3e194834567f35fd1785a5353a14c83b3519572378343e9fef72f`
for both one-window identity tests and both NCU captures. No kernel, dispatcher
or test edit. The private cache was0700 and the existing Fixed cache untouched.
The 368-file source map includes the committed selector test, not the old local
unbuilt WIP/helper. Full binary/tool/library/cache bindings are in
[artifact-binding.json](artifact-binding.json).

NCU used exact anchored symbols, one matching launch after skip130, kernel replay,
clock/cache control none, pipeline boost dynamic, application-only and config off;
the nine sections/eight extra traffic metrics are recorded in `plan.md` and the
command receipts. Calibration is adaptive: skip130 is not an exact warmup claim.
Each report contains one expected kernel. Unprofiled identity tests each executed
one Rust test and emitted two eager/graph records. Their zero-data preflight is
not a full bit-exact qualification. Identity immediate RELEASE42% was preserved
with a distinct quiet drain. NCU RELEASE14:46:09 UTC was already quiet; the later
14:46:14 snapshot is retained separately. No successful capture was retried.

## Counters

TN prism logical M,K,N =4621,384,1928: output384x1928, reduction4621, alpha1/beta1.
NT d768_in logical M,K,N =2048,768,3072: output2048x768, reduction3072, alpha1/beta0.
Neither has bias. Symbols share suffix `_sm80_mma_tf32_v1_m128n64_bk32_s3`.

| Metric | TN prism | NT d768_in |
| --- | ---: | ---: |
| Grid CTAs / block threads | 93 / 256 | 192 / 256 |
| Registers per thread | 148 | 154 |
| Dynamic shared bytes | 79,872 | 82,944 |
| Static shared bytes | 0 | 0 |
| Register / shared residency limit, CTAs per SM | 1 / 1 | 1 / 1 |
| Waves per SM | 0.65 | 1.35 |
| Achieved active occupancy | 16.057% | 15.480% |
| Eligible warps per active cycle | 0.423522 | 0.375772 |
| HMMA active-cycle pipe activity | 37.171% | 37.626% |
| HMMA elapsed-cycle pipe activity | 21.915% | 25.865% |
| Barrier stall per issued instruction | 1.490983 | 1.610906 |
| Long-scoreboard stall per issued instruction | 0.155320 | 0.113835 |
| Short-scoreboard stall per issued instruction | 0.055952 | 0.054547 |
| Shared wavefronts actual / ideal | 12,061,692 / 10,332,198 | 19,906,560 / 14,155,776 |
| Shared excess / ideal | 16.739% | 40.625% |
| L2 sector hit rate | 99.641% | 99.797% |
| DRAM bytes (decimal Kbyte) | 35.072 | 113.536 |
| Executed warp instructions | 52,504,890 | 65,031,936 |
| HMMA instructions | 3,452,160 | 4,718,592 |
| IMAD instructions | 13,585,305 | 16,398,336 |
| FSETP instructions | 5,178,240 | 7,078,656 |
| NCU diagnostic duration, us | 246.560 | 285.568 |

Raw CSV includes a units row followed by the data row. Dynamic shared and DRAM
values are in decimal Kbyte, not bytes. `memory_l1_wavefronts_shared*` columns
carry the installed export's `sectors` unit label; values above are the exported
actual/ideal/excess counts, not byte traffic. Observed local/spill traffic metrics
are zero; `launch__stack_size=1024` is not a compiled local-byte attribute proof.
SASS opcode aggregation must strip both numbered predicates and `@!PT`;
predicate-false LDS instruction counts are not additional shared transactions.

Profile duration is diagnostic, not admission timing. NVIDIA documents that
software-instrumented counters can perturb runtime and are collected separately;
stall analysis is relevant when issue slots are being missed, and occupancy
alone does not predict performance. [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)

## Source attribution and interpretation

- NT: **all5,750,784 excess shared wavefronts** occur at18
  `LDGSTS.E.BYPASS.128.ZFILL` sites (the asynchronous global-to-shared copies).
  The96 ordinary scalar LDS compute-read sites have7,077,888 actual=ideal,
  zero excess. This is not evidence that the fragment reads need another layout.
- TN: 1,614,690 excess (93.36%) at12 wide LDGSTS sites, and114,804 (6.64%)
  at32 four-byte LDGSTS sites. Its96 ordinary LDS sites likewise have
  5,178,240 actual=ideal and zero excess.
- Source NT copies four contiguous128-byte logical rows per warp into padded
  36-float shared rows. Do not infer bad global coalescing solely from LDGSTS
  sector counters. NVIDIA staff reproduced a separate LDGSTS excessive-sector
  anomaly; that report is a caution, not proof our full-warp case is affected.
  [NVIDIA developer discussion](https://forums.developer.nvidia.com/t/excessive-sectors-reported-for-ldgsts-e/342421)
- TN's93 CTAs leave part of a142-SM GPU uncovered, but active-cycle issue is
  also low. Pure underfill is not an adequate explanation. Integer address and
  conversion/control work is substantial in both kernels. No profile supports
  weakening RNA handling, K order, epilogue rounding or the deterministic ABI.

## One next experiment

Bounded, test-only NT M128N64/BK32/S3 compact XOR staging. Replace NT's36-float
row stride by32 and map `physical_k = logical_k ^ ((row & 7) << 2)` independently
for A rows and B columns. This is a shared-copy destination hypothesis motivated
by the measured LDGSTS signal and the existing Fixed swizzle/copy-plan work,
not an assertion of bank-conflict causality or a promised speedup.

Keep the current per-thread global source copy assignment,16-byte vector groups,
RNA conversion, ascending four K8 MMA steps per BK32 tile, compute-warp mapping,
zero-fill and NT overwrite/alpha epilogue. Both writers and readers must use the
same mapping. Expected staging size73,728 bytes; this alone does not increase
residency because the current register footprint still limits one CTA per SM.

Prove bijection/bounds/vector alignment and preserve the currently conflict-free
scalar fragment-read bank mapping first, then a small actual-AUTO-versus-
candidate raw-bit/repeat/graph/tail/resource set on Ada13.2. Screen a survivor
once7; do not run another full matrix or a21/101 gate for a rejected prototype.
Only a retained finalist can proceed to production holder/dispatcher wiring and
the affected integrated qualification. No production candidate is admitted here.
The [reuse audit](fixed-reuse-audit.md) records what transfers from inference
and which numerical/ABI boundaries prohibit a direct transplant.

Current unprofiled Fast gaps are in the separate fresh
[60-cell state report](../ada-triad-state-20260907/report.md); do not divide these
NCU durations by that vendor snapshot. Selected raw evidence hashes are in
`manifest-selected.sha256`; reports/audit prose are separate derived artifacts.
