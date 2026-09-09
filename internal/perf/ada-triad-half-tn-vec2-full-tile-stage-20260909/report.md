# Ada Triad F16 TN unpredicated full-tile staging stop — 2026-09-09

Outcome: **exact/resource PASS and a small measured improvement, but strict
retained gate FAIL; STOP before once7/Fast.**

The test-only d768-in `(2048,768,3072)` candidate adds a target-only staging
arm with direct 16-byte three-operand `cp.async` loads. It removes per-copy
extent, clamp and zero-fill work only where every M64N64/BK64 tile is full.
The retained guarded asynchronous and scalar fallbacks remain available for
all other shapes. Mapper, four-warp topology, shared XOR layout, S2 ring,
ldmatrix/HMMA order and `float2` F32 epilogue are unchanged.

CUDA13.2 / RTX 6000 Ada compile and runtime gates pass:

- candidate 118 registers versus retained 125, local 0, static shared 32,768
  bytes, block 128 and occupancy 3 CTA/SM;
- both have 32 HMMA and 24 LDSM static instructions; the composed candidate
  contains 48 LDGSTS sites because target and fallback staging arms coexist;
- SASS text ratio is 1.19649, inside the declared 1.20 cap, with no spills.

Exact target, aligned and misaligned negative-alpha tails, exceptional full
tile, K0, eager/graph repeats, 20-operation accumulation, guards and immutable
inputs pass. Candidate/retained once3 ratios are:

| Path/order | p50 | p95 |
| --- | ---: | ---: |
| eager ABBA | 0.995821 | 0.995975 |
| eager BAAB | 0.996188 | 0.996776 |
| graph ABBA | 0.992047 | 0.992628 |
| graph BAAB | 0.994093 | 0.994988 |

The candidate is consistently 0.32–0.80% faster, but no stratum satisfies the
predeclared strict `<0.99` p50+p95 gate. Once7 and cuBLAS Fast were not run.
Do not retry this unchanged staging specialization; keep it as evidence that
predicate removal helps but is insufficient alone.

Frozen identity:

- helper SHA256: `6697f33513782ef2cd22fcfabbebc6035f7d59f15700438b8c41a2702758d527`
- harness SHA256: `73881685cd8a2b639c7a66b120f482cfe6e60b513c7ef4606f33e66e34d1f838`
- candidate source/PTX/CUBIN/SASS SHA256:
  `16661d49ce157334673ea9d601e2b4211f0c899e86d2e7681cd42992de3d54ff` /
  `ed54533079db55b5ee73cf9dcb774bb9c38dd810fc9f3120f176b757d898cace` /
  `941b7dbc4a0b98301200f54dbb6b24d6206e31062621864ab1058ada08f62d5c` /
  `8244e6820946a4d957b1a9976e289f3037ca0e89c9b196384bea66958a1c42a5`
- [raw output](raw.log)
- [launch preflight](preflight.txt)
