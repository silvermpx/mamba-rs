# Ada exact-F32 B0 — production Nsight diagnosis

Status: COMPLETE, counters only. No kernel, loader, selector, cache, or
qualification source changed.

## Bound invocation

CUDA13.2 used the accepted Task8 final2 performance binary
`gemm_bi_fixed_performance-46090335b71731a6`, SHA-256
`4ecefea9345226073e9671017bb14450ebed562be8c5c551a2e6d9de3859868d`,
and measured-source SHA-256
`2ac1c93ba3e4682138a0df3341a26921438334382532d6943277268320655285`.
The one-cell comparator was `f32_exact_fast/hot_b/bias0`,
`(M,K,N)=(4621,768,2304)`, forced tile `F32Sm89N64CopyPlan`, eager path,
one window, and native `CUBLAS_COMPUTE_32F_FAST_TF32`. Toolkit-admission mode
was absent. Every one of the frozen binding's 357 source hashes still matched,
with no missing or changed input. The verifier separately admitted only the
seven known additive N96/W4/half discovery-test files and rejects any other
extra input.

Nsight Compute 2026.1.0.0 build37166530 selected the 131st matching launch
with kernel replay, clock control `none`, cache control `none`, dynamic pipeline
boost, and one launch. `LaunchStats`, `Occupancy`, `SpeedOfLight`, compute,
scheduler, warp-state, memory, instruction, and `SourceCounters` sections were
collected in one NCU invocation per arm. Profile durations are diagnostic and
not admission timings.

The successful continuation PRE/RELEASE snapshots were quiet and had no
compute applications. PRE was P5,1800MHz SM,810MHz memory,40C,39.86W;
RELEASE was P5,1800MHz,810MHz,41C,54.94W. The lane was then released.

## Result

The ordinary unprofiled test passed exact AUTO/forced raw-bit and repeat gates:

| order | AUTO CopyPlan us | forced CopyPlan us | Fast us | forced/Fast |
|---|---:|---:|---:|---:|
| auto-forced-vendor | 420.523 | 418.051 | 191.921 | 2.17824 |
| vendor-forced-auto | 420.363 | 418.237 | 192.836 | 2.16888 |

The selected launches were:

| counter | CopyPlan | Fast |
|---|---:|---:|
| actual symbol | `gemm_bi_nn_fixed_sm89_f32_n64_copyplan_v1` | `cutlass_80_tensorop_s1688gemm_256x128_16x3_nn_align4` |
| grid / block | 2628 / 128 | `(296,2,1)` / 256 |
| static / dynamic shared bytes | 32,768 / 0 | 0 / 73,728 |
| registers/thread | 135 | 220 |
| theoretical / achieved occupancy | 25.00% / 24.03% | 16.67% / 16.56% |
| profiled duration, us | 438.368 | 192.704 |
| GPC elapsed frequency, GHz | 1.712836 | 1.799585 |
| executed warp instructions | 302,462,424 | 38,479,704 |
| FFMA / HMMA instructions | 258,633,216 FFMA | 8,183,808 HMMA |

CopyPlan's 258,633,216 FFMA instructions are 85.51% of all its executed warp
instructions and close to the unavoidable direct scalar work after tile-tail
padding. Its steady BK32 body is fully unrolled: SASS emits three `LDS.128`
loads per `kk` and interleaves later loads among independent FFMAs. It executes
24,219,648 LDS instructions. Those LDS instructions produced 64,585,728 actual
and 64,585,728 ideal shared wavefronts, with zero excessive wavefronts. Including
the `LDGSTS` stores, all CopyPlan shared accesses produced 76,696,416 actual and
76,696,416 ideal wavefronts, still zero excessive.

This uses Nsight's primary installed definition, not a source-address model:
`/opt/nvidia/nsight-compute/2026.1.0/sections/SourceCounters.section` defines
`derived__memory_l1_wavefronts_shared_excessive` as
`memory_l1_wavefronts_shared - memory_l1_wavefronts_shared_ideal`. Therefore the
proposed A-bank XOR is not supported by the current generated SASS/counters and
must not be implemented.

CopyPlan is limited to three CTAs by both registers and shared memory. It has
1.65 eligible warps/scheduler, 70.93% elapsed issue-slot use, and 61.67% elapsed
FMA-pipe use. Its notable warp-stall-per-issue ratios are short-scoreboard
0.4626, dispatch0.4575, barrier0.2765, and MIO-throttle0.2687. The immediate
mechanism is scalar issue/residency and load-to-use scheduling, not shared-bank
conflicts or removable scalar math. A single test-only 256-thread CopyPlan twin
is the best bounded follow-up: it halves accumulator ownership and can expose
up to24 resident warps at the unchanged three-CTA/32KiB limit, but requires live
Ada resource and speed evidence; the SM120 result is not an Ada speed claim.

## Preserved history and evidence

Run1 exited before GPU allocation because exact source-map equality rejected
the seven later additive discovery files. Run2 established a quiet PRE, passed
the only unprofiled baseline, and completed the only CopyPlan NCU capture, then
exited because the evidence parser expected three `DictReader` rows instead of
the actual units+data pair. Run3 did not rerun either; it validated their hashes,
established a fresh quiet PRE, captured Fast once, and completed quiet RELEASE.
All failures and outer exits are retained.

The selected-file manifest is `manifest-selected.sha256`. Key raw files are
`run2/unprofiled-baseline.log`, `run2/copyplan.ncu-rep`,
`run2/copyplan-{raw,sass,details}.csv`, `run3/fast.ncu-rep`,
`run3/fast-{raw,sass,details}.csv`, and all three outer transcripts/statuses.
Run3's remote manifest contains16 entries and was independently recomputed
locally with zero mismatch; outer SSH exit was0 and the completion marker was
present.
