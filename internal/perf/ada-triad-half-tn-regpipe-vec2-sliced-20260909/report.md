# Ada Triad F16 TN regpipe stage-sliced stop — 2026-09-09

Outcome: **STOP, no retry or promotion.** The test-only d768-in
`(2048,768,3072)` candidate interleaves four next-tile A+B `cp.async` slices
with the retained M64N64/BK64/S2 regpipe+`float2` K16 compute groups. It is
bit-exact and reduces register pressure, but loses the retained kernel in every
paired once3 stratum. once7 and cuBLAS Fast were therefore not launched.

## Gates

- CUDA 13.2 / SM89 RTX 6000 Ada, grid 576, block 128.
- Resource PASS: candidate 115 registers versus retained 125; both local 0,
  static shared 32,768 bytes and occupancy 3.
- SASS PASS: stack/spills 0, HMMA 32/32; candidate LDGSTS 18 with source proof
  covering all eight refill copies per thread; no LDL/STL/ATOM/RED/REDUX.
- Exact PASS: target, aligned and misaligned negative-alpha tails, exceptional
  full tile and K=0; eager/graph repeated three times, 20-op and guard checks.

## Paired once3 candidate/retained ratios

| Path/order | p50 | p95 |
| --- | ---: | ---: |
| eager ABBA | 1.025183 | 1.037243 |
| eager BAAB | 1.028446 | 1.033213 |
| graph ABBA | 1.030615 | 1.031278 |
| graph BAAB | 1.028874 | 1.028917 |

The strict `<0.99` retained gate fails throughout. Keep the prior regpipe+vec2
F16 TN d768-in winner and do not try BF16/siblings or this unchanged schedule.
Production and dispatcher are unchanged.

## Frozen evidence

- helper SHA256: `ad7cae4c68cf26bfc2d29fc9cd4260261bb33ccd5e2f2460121dab88b0e315f0`
- harness SHA256: `31a71374663f3c54da3129d0c9b3d08e9f97953186e4307381cac6c680ddee36`
- candidate source/PTX SHA256: `f4a43569f0a8234d75b221566d3c54d5c4917b5124d67d32a77bcb4446840123` / `4a302a6582a8efff5c9fc3aa5dd14208b9d3753b41ea638394969ded347a888a`
- raw log SHA256: `0db206cb807f7923eef120fdd722f2fd9bd398135a80e18cf1eed9d86e6defc4`
- [raw output](raw.log)
- [launch preflight](preflight.txt)
