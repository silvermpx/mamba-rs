# Ada Triad F16 TN target-only exact-entry stop — 2026-09-09

Outcome: **once3 STOP; 0.9–1.7% slower than retained, no production change.**

The test-only d768-in `(2048,768,3072)` candidate combines the exact-shape 2D
CTA mapping, direct full-tile 16-byte cp.async staging and unconditional paired
epilogue in a target-only kernel entry. Tail, exceptional-small and K0 cases
physically launch the retained symbol and one-dimensional grid instead of
embedding a runtime fallback in the candidate.

CUDA 13.2 compile/SASS gates pass. The candidate uses102 registers versus125,
local0, static shared32,768, block128 and occupancy3. It preserves32 HMMA,
24 LDSM and24 LDGSTS instructions, contains no runtime division, and its SASS
text is0.39649x retained. Exact raw bits pass target positive/negative alpha,
target exceptional values, aligned/misaligned tails, exceptional full tile, K0,
three eager/graph repeats and guarded buffers.

Candidate/retained once3 p50/p95:

| Stratum | p50 | p95 |
| --- | ---: | ---: |
| eager ABBA | 1.013204 | 1.017024 |
| eager BAAB | 1.013749 | 1.014365 |
| graph ABBA | 1.009440 | 1.011193 |
| graph BAAB | 1.011792 | 1.012991 |

The strict retained gate failed in all four strata, so once7 and cuBLAS Fast
were not run. Do not retry this unchanged combination; low registers and small
text alone do not recover the target-entry overhead.

Frozen identities:

- helper SHA256: `260dde6eb25931ad62ca876207bda2cb32fb7d69bb8575a2251dbd612bb83fd1`
- harness SHA256: `381f2c2970c390ced2c11f8090e5acab0648e47f0b02e746ca456efccb74bb40`
- candidate transformed source SHA256: `fe34c3f234cba8b8482d8d471ab30329a87508009a6f081bb4693fb02e34c996`
- candidate PTX SHA256: `bd3097baf5da0cce87606fc705186a2455b2188a1416ef1e074ef4f8e8c56e3e`
- candidate CUBIN SHA256: `b254f8e0780448bfcbfb93071d9d27a6e02471e30c6d2a2265e3478c11df46d0`
- candidate SASS SHA256: `dbd4828f3685a57079245ef0242f6c1b6cc1d7e3f05b2e35dffea6bacc1e7840`
- [raw qualification summary](raw.log)
