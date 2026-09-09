# Ada Triad exact-F32 TN canonical Prism direct BK16 winner — 2026-09-09

Outcome: **new retained-best winner; 21.3–21.4% faster than actual production
AUTO in every once7 stratum, exact bits preserved. Not a cuBLAS Fast win.**

The test-only `(4621,384,1928)` candidate reads row-major `X` directly and
replaces the retained generic split-M partial with one M64N64/BK16/S2 launch
over the same six deterministic chunks (`784/784/784/784/784/701`). It keeps
the production FP64 reducer unchanged. This removes the retained partial
kernel's larger tile/resource footprint while preserving every ordered
`__fmaf_rn` chain and the exact reducer order.

The corrected harness proves actual AUTO is the production two-node
`gemm_bi_tn_splitm_partial_aligned -> gemm_bi_splitm_reduce` route. An initial
pre-timing harness run accidentally forwarded batch zero and was rejected by
the physical graph identity gate; a native regression test now pins the real
fixture batch in both retained call sites.

Raw partials match for all six chunks on finite and exceptional corpora. Final
target, tail, exceptional, non-unit alpha, K0, eager/graph repeats, inputs and
guards all pass bit-exactly. The direct kernel uses 107 registers, no local
memory, 16,384 bytes static shared and four active CTAs/SM; retained partial
uses 127 registers, 33,792 bytes static shared and occupancy2.

Candidate/actual-AUTO once7 p50/p95:

| Stratum | p50 | p95 |
| --- | ---: | ---: |
| eager ABBA | 0.786449 | 0.786527 |
| eager BAAB | 0.786587 | 0.787082 |
| graph ABBA | 0.786079 | 0.786869 |
| graph BAAB | 0.786541 | 0.787238 |

Candidate/cuBLAS Fast once7 p50/p95:

| Stratum | p50 | p95 |
| --- | ---: | ---: |
| eager ABBA | 2.400689 | 2.402184 |
| eager BAAB | 2.401250 | 2.407029 |
| graph ABBA | 2.402288 | 2.404242 |
| graph BAAB | 2.405231 | 2.407012 |

Retain this candidate for the joint Triad dispatcher integration batch. It is
a large exact-F32 improvement, not a Fast victory.

Frozen identities:

- helper SHA256: `335d7cdde0364e36ad17e0f683b9f0eb57dc6cfce529be36f48b80e7acc0d12a`
- corrected harness SHA256: `c949412e703e62de598b4be06a8059886cc945b23206b25e78d9cb83f1d979e7`
- direct transformed source SHA256: `07ec10cf27ac814a80f7e9c918a6bc2268774347d284d06603fa2aa022b42eef`
- [raw qualification summary](raw.log)
