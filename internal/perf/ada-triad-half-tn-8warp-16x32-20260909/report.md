# Ada Triad F16 TN eight-warp 16x32 stop — 2026-09-09

Outcome: **valid exact performance STOP; no once7, Fast or production change.**

The test-only d768-in `(2048,768,3072)` candidate decomposes the retained
M64N64/BK64/S2 regpipe+`float2` CTA into eight 16x32 compute-warps instead of
four 32x32 warps. It preserves the real seven-argument TN ABI, S2 staging,
ascending K16 association, per-CTA output ownership and F32 epilogue arithmetic.

CUDA13.2 / RTX 6000 Ada compile and runtime resources pass:

- candidate 80 registers, local 0, static shared 32,768 bytes, block 256,
  occupancy 3 CTA/SM;
- retained 125 registers, local 0, static shared 32,768 bytes, block 128,
  occupancy 3 CTA/SM;
- candidate/retained static SASS counts are HMMA 16/32, LDSM 20/24 and
  LDGSTS 24/24, with no spills.

Exact target, aligned and misaligned negative-alpha tails, exceptional full
tile, K0, eager/graph repeats, 20-operation accumulation, guards and immutable
inputs pass. Despite the extra resident warps, the candidate loses retained in
every once3 stratum:

| Path/order | p50 | p95 |
| --- | ---: | ---: |
| eager ABBA | 1.107521 | 1.108632 |
| eager BAAB | 1.107593 | 1.109205 |
| graph ABBA | 1.104240 | 1.106257 |
| graph BAAB | 1.106257 | 1.110159 |

The 10.4–11.0% loss shows that duplicated B-fragment `ldmatrix` work outweighs
the additional latency hiding for this cell. The strict once3 `<0.99` gate
stopped once7 and cuBLAS Fast. Keep the retained four-warp regpipe+`float2`
kernel and do not retry this unchanged warp decomposition.

Frozen identity:

- helper SHA256: `313b1a64d0a46b6cb1051c84c86d762fe18744eefd9abb85e0c161aab5a79045`
- harness SHA256: `fcde9dc2c2813c227445b9a48cb305c9da97dda82d9054b3246dfe83045bf9d8`
- candidate source/PTX/CUBIN/SASS SHA256:
  `6f3941d4c58eb8edb6c68dd51783c60f5e261f69268e3791427499446ecbb0f7` /
  `f8f4725adfb5f7bd9486527a4d9011e4ed8b7ec0a3b99d3780b6fc9088ab8595` /
  `ad15a5477e6dd0b3f628035c615891c5366f2373b01829d5d125640374bedaa4` /
  `81d5012f200633882455f71104ddd76ac7a6b035fce9f3254cc687a9f7ab7610`
- [raw output](raw.log)
- [launch preflight](preflight.txt)
