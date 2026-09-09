# Ada Triad F16 TN full-tile cp.async.cg stop — 2026-09-09

Outcome: **once3 STOP; about 16% slower than retained, no production change.**

The test-only d768-in `(2048,768,3072)` candidate changes only the two direct
16-byte full-tile staging sites from `cp.async.ca` to `cp.async.cg`, bypassing
L1 while retaining L2 caching. It starts from the previously measured
full-tile staging arm; all addresses, predicates outside the exact target,
math, synchronization and epilogue remain unchanged.

CUDA13.2 compile/SASS and resource gates pass. The candidate uses118 registers,
local0, static shared32,768, block128 and occupancy3 versus retained125
registers. It preserves32 HMMA and24 LDSM; the composed source contains48
LDGSTS sites and is1.19649x the retained SASS text. Exact target, aligned and
misaligned tails, exceptional shape, K0, eager/graph repeats,20-op accumulation,
guards and inputs pass.

Candidate/retained once3 p50/p95:

| Stratum | p50 | p95 |
| --- | ---: | ---: |
| eager ABBA | 1.162193 | 1.162873 |
| eager BAAB | 1.160509 | 1.161064 |
| graph ABBA | 1.159540 | 1.159776 |
| graph BAAB | 1.159823 | 1.160556 |

The strict retained gate failed decisively, so once7 and cuBLAS Fast were not
run. The result shows that L1 locality is valuable for this tile; do not retry
the unchanged `.cg` mechanism.

Frozen identities:

- helper SHA256: `eec32c943802245823903663e06c82deda97283ffce2a725594d740207fdf3a7`
- harness SHA256: `3d18afb3b245212920ab9a6c21765c5d5300a2d213501f18c51f1002e22a0a60`
- candidate transformed source SHA256: `e741b0c515db5505288b9b6fa08a4802803a890417d23bdb0e1b926d3def4fc6`
- candidate PTX SHA256: `098bfe86f2d8cf6d3f3a05147406ffd99ee074306130ef3ea3ff69bf4afe12c4`
- candidate CUBIN SHA256: `89457fe95a1df841604277df89c42b6b843082ba5bb79e50dc66a8db6539fa43`
- candidate SASS SHA256: `38e080b57cab15025253f7c8a4ee4b06d28c36d8d5890ca093b26c34a94a195b`
- [raw qualification summary](raw.log)
