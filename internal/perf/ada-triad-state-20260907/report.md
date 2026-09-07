# Ada Triad state — 2026-09-07

This is the historical14:51Z snapshot, not a fresh full-matrix run after
integration. The later three-cell TF32 NT compact8/S2 results and actual AUTO
closure on CUDA12.8/13.0/13.2 are in
[the finalist integration report](../ada-triad-finalist-integration-20260907/README.md).
Do not merge their separately paired Fast samples into this table's medians.

## Result

The current production AUTO routes are observed faster than cuBLAS Fast in **3/60**
cells and slower in **57/60** in this synthetic eager/eager snapshot.
There are no mixed/equal cells. This is Triad, not Fixed inference, and not a
release qualification or evidence that the 57 losses are hardware limits.

| Policy | NN faster / 5 | TN faster / 5 | NT faster / 5 | Total faster / 15 |
| --- | ---: | ---: | ---: | ---: |
| Exact F32 | 0 | 0 | 0 | 0 |
| Deterministic TF32 | 0 | 0 | 0 | 0 |
| BF16 | 1 | 0 | 0 | 1 |
| F16 | 1 | 0 | 1 | 2 |

NN is the forward product, TN the weight-gradient product, NT the input-gradient
product. All three observed winners also have a lower p95:
BF16 NN d128_in_proj (5.762 vs 9.216 us),
F16 NN d128_in_proj (5.760 vs 10.240 us), and
F16 NT prism_in_proj (73.909 vs 97.489 us).

The largest TF32 median gaps are NT d768_in_proj (+157.881 us, 2.348x Fast),
TN prism_in_proj (+142.906 us, 2.433x), and TN d768_in_proj (+132.426 us, 1.993x).
The worst exact-F32 ratio is TN d128_in_proj (95.094 vs 10.468 us, 9.085x).
Exact F32 preserves a stricter precision contract than the FAST_TF32 denominator;
this comparison must not justify silently substituting TF32 arithmetic.

## Acquisition and limits

- RTX 6000 Ada, CC8.9 / 142 SM, CUDA13.2 only; GPU UUID
  `GPU-d1edd7be-e88d-aed6-047d-622163306f0e`.
- Source HEAD `59f1a9493316b463cee0e75fa321b0429699a488`;
  runtime base `b221b71e870326c8ba984fc379596013da10fad3`.
  Same source set and binary as the preceding two-cell profiles, no rebuild.
  Binary SHA256 `cf38911f01f3e194834567f35fd1785a5353a14c83b3519572378343e9fef72f`.
- Exactly two tests, once21/orderAB: 120 custom eager/graph records and 90 vendor
  Fast/Pedantic records. Both commands exit0 with one executed test each.
  Raw logs, exact commands, identity/source map and parsed records are retained
  under `run-cuda132/`. No source or dispatcher change is part of this snapshot.
- Quiet PRE at14:51:16 UTC. Immediate RELEASE was busy (23% GPU / 1% memory);
  it remains recorded as such. Separate five-second drain at14:51:51 was quiet
  (0/0, no compute apps). No valid measurement was rerun.
- All 210 raw records, selections, positive finite21 samples, median/p95
  order statistics and eager/graph request identities independently replayed.
  The old local selector-discovery WIP was excluded: its measured source entry
  matches `git show 59f1a949:tests/gemm_bi_sm120_tf32_selector_qualification.rs`.
- Ratios below divide **independently measured quantiles**, not paired samples.
  The custom/vendor calls are separate, calibrated batches; some small Fast
  cells use only1–2 iterations per sample and can be overhead sensitive.
  These observed wins are diagnostic, not robust new champion admissions.
- Main ratios are eager/eager. Custom graph latency is shown separately;
  there is no measured vendor graph denominator and no graph/graph win claim.
- Active operands start at zero. TN beta1 C is not restored before every call.
  Preflight identity/equality is not a full-mantissa, exceptional-value,
  batch-invariance or determinism qualification, nor end-to-end training timing.
- Fast means `32f_fast_tf32` for F32/TF32 and native-half `32f` for BF16/F16;
  Pedantic records remain raw evidence but are not used in any ratio here.
- Faster/slower requires both p50 and p95 below/above1. Any ratio within3% of
  parity is marked marginal in the full table. No workload-average speedup,
  fresh CUDA12.8/13.0 result, or fresh RTX5090 result is inferred.

## Reuse and next experiment

Fixed inference contributes copy planning, compact XOR shared-memory layouts and
pipeline overlap techniques, plus its qualification protocol. It does not supply
drop-in Triad NN/TN/NT kernels or a transferable raw-bit proof.
See [the source-level reuse audit](../ada-triad-two-cell-ncu-20260907/fixed-reuse-audit.md).

The fresh two-cell profiles locate excess shared wavefronts on the asynchronous
global-to-shared copies, **not** on the MMA fragment shared reads.
NT d768_in_proj is the first bounded hypothesis: compact its padded shared-copy
destination while preserving copy ownership, the direct RNA conversion, ascending
K-MMA association and NT epilogue. This is not yet an implemented speedup.
First prove the address mapping and small bit/resource set; then screen once7.
Do not repeat this entire state matrix after each prototype.

## Full eager/Fast table

Times are microseconds; each timing pair is p50 / p95. Ratio<1 is faster.
The JSON [summary](summary.json) records all60 exact physical symbol lists,
logical shapes, call scopes, iteration counts and unrounded values.
Raw physical nodes additionally retain grid/block/shared and argument digests.

### Exact F32

| Op / shape | Ours p50 / p95 | Fast p50 / p95 | Ratio p50 / p95 | Custom graph p50 / p95 | State |
| --- | ---: | ---: | ---: | ---: | --- |
| NN d128_in_proj | 15.920 / 15.927 | 8.144 / 9.728 | 1.955 / 1.637 | 15.203 / 15.209 | slower |
| TN d128_in_proj | 95.094 / 95.306 | 10.468 / 10.581 | 9.085 / 9.007 | 88.711 / 88.890 | slower |
| NT d128_in_proj | 17.177 / 17.192 | 9.303 / 9.313 | 1.846 / 1.846 | 15.764 / 15.777 | slower |
| NN d128_out_proj | 10.185 / 10.189 | 6.782 / 6.790 | 1.502 / 1.501 | 9.471 / 9.473 | slower |
| TN d128_out_proj | 53.810 / 53.854 | 9.835 / 9.840 | 5.471 / 5.473 | 53.032 / 53.076 | slower |
| NT d128_out_proj | 12.926 / 12.931 | 5.021 / 5.045 | 2.575 / 2.563 | 11.398 / 11.401 | slower |
| NN d768_in_proj | 351.851 / 352.119 | 101.899 / 101.938 | 3.453 / 3.454 | 351.565 / 351.763 | slower |
| TN d768_in_proj | 418.797 / 420.096 | 133.342 / 133.408 | 3.141 / 3.149 | 419.499 / 421.035 | slower |
| NT d768_in_proj | 651.776 / 652.032 | 117.136 / 117.157 | 5.564 / 5.565 | 651.484 / 651.648 | slower |
| NN d768_out_proj | 248.535 / 248.588 | 62.863 / 62.876 | 3.954 / 3.954 | 248.334 / 248.393 | slower |
| TN d768_out_proj | 253.427 / 254.461 | 82.291 / 82.308 | 3.080 / 3.092 | 252.208 / 253.645 | slower |
| NT d768_out_proj | 277.235 / 277.396 | 66.799 / 66.823 | 4.150 / 4.151 | 276.911 / 277.061 | slower |
| NN prism_in_proj | 295.635 / 295.996 | 102.970 / 103.054 | 2.871 / 2.872 | 295.514 / 295.996 | slower |
| TN prism_in_proj | 304.198 / 304.550 | 99.733 / 99.755 | 3.050 / 3.053 | 302.803 / 303.159 | slower |
| NT prism_in_proj | 346.249 / 346.385 | 76.976 / 76.992 | 4.498 / 4.499 | 345.907 / 346.044 | slower |

### Deterministic TF32

| Op / shape | Ours p50 / p95 | Fast p50 / p95 | Ratio p50 / p95 | Custom graph p50 / p95 | State |
| --- | ---: | ---: | ---: | ---: | --- |
| NN d128_in_proj | 9.927 / 9.938 | 8.144 / 9.728 | 1.219 / 1.022 | 9.759 / 9.762 | slower (marginal) |
| TN d128_in_proj | 19.113 / 19.117 | 10.468 / 10.581 | 1.826 / 1.807 | 19.123 / 19.130 | slower |
| NT d128_in_proj | 12.363 / 12.371 | 9.303 / 9.313 | 1.329 / 1.328 | 12.240 / 12.248 | slower |
| NN d128_out_proj | 7.663 / 7.668 | 6.782 / 6.790 | 1.130 / 1.129 | 7.568 / 7.572 | slower |
| TN d128_out_proj | 18.988 / 18.991 | 9.835 / 9.840 | 1.931 / 1.930 | 18.798 / 18.803 | slower |
| NT d128_out_proj | 7.931 / 7.936 | 5.021 / 5.045 | 1.580 / 1.573 | 7.852 / 7.858 | slower |
| NN d768_in_proj | 123.779 / 123.829 | 101.899 / 101.938 | 1.215 / 1.215 | 121.925 / 121.931 | slower |
| TN d768_in_proj | 265.768 / 265.817 | 133.342 / 133.408 | 1.993 / 1.993 | 265.528 / 265.592 | slower |
| NT d768_in_proj | 275.016 / 275.077 | 117.136 / 117.157 | 2.348 / 2.348 | 273.836 / 273.947 | slower |
| NN d768_out_proj | 78.319 / 78.336 | 62.863 / 62.876 | 1.246 / 1.246 | 77.872 / 77.887 | slower |
| TN d768_out_proj | 170.871 / 170.899 | 82.291 / 82.308 | 2.076 / 2.076 | 170.626 / 170.659 | slower |
| NT d768_out_proj | 121.149 / 121.222 | 66.799 / 66.823 | 1.814 / 1.814 | 119.784 / 119.852 | slower |
| NN prism_in_proj | 107.133 / 107.210 | 102.970 / 103.054 | 1.040 / 1.040 | 106.173 / 106.197 | slower |
| TN prism_in_proj | 242.639 / 242.693 | 99.733 / 99.755 | 2.433 / 2.433 | 242.395 / 242.444 | slower |
| NT prism_in_proj | 190.415 / 190.464 | 76.976 / 76.992 | 2.474 / 2.474 | 190.312 / 190.388 | slower |

### BF16

| Op / shape | Ours p50 / p95 | Fast p50 / p95 | Ratio p50 / p95 | Custom graph p50 / p95 | State |
| --- | ---: | ---: | ---: | ---: | --- |
| NN d128_in_proj | 5.762 / 5.769 | 9.216 / 10.144 | 0.625 / 0.569 | 5.404 / 5.408 | faster |
| TN d128_in_proj | 18.413 / 18.417 | 9.216 / 9.547 | 1.998 / 1.929 | 18.200 / 18.204 | slower |
| NT d128_in_proj | 9.343 / 9.345 | 7.672 / 7.687 | 1.218 / 1.216 | 9.210 / 9.212 | slower |
| NN d128_out_proj | 6.587 / 6.591 | 5.385 / 5.391 | 1.223 / 1.223 | 6.542 / 6.545 | slower |
| TN d128_out_proj | 18.244 / 18.247 | 7.278 / 7.305 | 2.507 / 2.498 | 18.009 / 18.013 | slower |
| NT d128_out_proj | 5.448 / 5.450 | 4.728 / 4.734 | 1.152 / 1.151 | 5.210 / 5.211 | slower |
| NN d768_in_proj | 90.711 / 90.734 | 78.348 / 78.473 | 1.158 / 1.156 | 90.368 / 90.422 | slower |
| TN d768_in_proj | 103.524 / 103.821 | 88.432 / 88.739 | 1.171 / 1.170 | 103.280 / 103.737 | slower |
| NT d768_in_proj | 101.375 / 101.416 | 78.370 / 78.517 | 1.294 / 1.292 | 101.089 / 101.130 | slower |
| NN d768_out_proj | 54.958 / 55.000 | 47.426 / 47.469 | 1.159 / 1.159 | 54.795 / 54.817 | slower |
| TN d768_out_proj | 67.257 / 67.761 | 50.359 / 50.661 | 1.336 / 1.338 | 67.210 / 67.639 | slower |
| NT d768_out_proj | 56.559 / 56.616 | 39.889 / 40.310 | 1.418 / 1.405 | 56.168 / 56.216 | slower |
| NN prism_in_proj | 84.117 / 84.207 | 76.138 / 76.200 | 1.105 / 1.105 | 84.100 / 84.169 | slower |
| TN prism_in_proj | 81.524 / 81.689 | 54.525 / 54.593 | 1.495 / 1.496 | 80.031 / 80.144 | slower |
| NT prism_in_proj | 73.864 / 73.924 | 62.533 / 62.661 | 1.181 / 1.180 | 73.668 / 73.717 | slower |

### F16

| Op / shape | Ours p50 / p95 | Fast p50 / p95 | Ratio p50 / p95 | Custom graph p50 / p95 | State |
| --- | ---: | ---: | ---: | ---: | --- |
| NN d128_in_proj | 5.760 / 5.769 | 10.240 / 11.008 | 0.563 / 0.524 | 5.404 / 5.407 | faster |
| TN d128_in_proj | 18.409 / 18.417 | 9.216 / 9.461 | 1.998 / 1.947 | 18.200 / 18.204 | slower |
| NT d128_in_proj | 9.342 / 9.345 | 5.868 / 5.876 | 1.592 / 1.591 | 9.207 / 9.207 | slower |
| NN d128_out_proj | 6.589 / 6.592 | 4.744 / 4.746 | 1.389 / 1.389 | 6.545 / 6.546 | slower |
| TN d128_out_proj | 18.244 / 18.248 | 7.532 / 7.545 | 2.422 / 2.418 | 18.010 / 18.011 | slower |
| NT d128_out_proj | 5.447 / 5.448 | 4.524 / 4.528 | 1.204 / 1.203 | 5.212 / 5.213 | slower |
| NN d768_in_proj | 90.679 / 90.752 | 59.468 / 59.576 | 1.525 / 1.523 | 90.368 / 90.405 | slower |
| TN d768_in_proj | 103.299 / 103.633 | 88.663 / 89.229 | 1.165 / 1.161 | 103.361 / 103.880 | slower |
| NT d768_in_proj | 101.356 / 101.416 | 73.643 / 74.576 | 1.376 / 1.360 | 101.128 / 101.153 | slower |
| NN d768_out_proj | 54.968 / 54.992 | 47.033 / 47.102 | 1.169 / 1.168 | 54.784 / 54.817 | slower |
| TN d768_out_proj | 67.311 / 67.611 | 50.391 / 50.467 | 1.336 / 1.340 | 67.252 / 67.626 | slower |
| NT d768_out_proj | 56.550 / 56.607 | 39.662 / 40.121 | 1.426 / 1.411 | 56.170 / 56.216 | slower |
| NN prism_in_proj | 84.087 / 84.224 | 75.648 / 75.697 | 1.112 / 1.113 | 84.087 / 84.173 | slower |
| TN prism_in_proj | 81.539 / 81.734 | 54.510 / 54.570 | 1.496 / 1.498 | 80.035 / 80.181 | slower |
| NT prism_in_proj | 73.909 / 73.939 | 97.489 / 97.569 | 0.758 / 0.758 | 73.653 / 73.698 | faster |

## Replay

Run `ruby internal/perf/ada-triad-state-20260907/replay.rb` from the repository
(or invoke the script by absolute path). It reads raw logs, checks selections,
counts, samples, quantiles and joins, and prints the deterministic JSON summary.
It performs no GPU work or writes. The evidence manifest is a separate owner
receipt; snapshot limitations above remain applicable even when hashes verify.
