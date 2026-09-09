# Ada Triad F16 NT M64N192/S3 near-miss stop — 2026-09-09

Outcome: **STOP before once7/Fast.** The test-only d768-out
`(2048,1536,768)` candidate widens the retained M64N128/BK64/S3 kernel to
M64N192 with twelve warps. It is bit-exact and faster in every once3
observation, but one graph stratum narrowly misses the strict 1% admission
margin. Preserve M64N128/S3 as retained-best; do not retry unchanged.

## Gates

- CUDA 13.2 / SM89 RTX 6000 Ada; candidate grid 256, block 384.
- Candidate resources: 127 registers, local/static 0, dynamic shared 98,304 B,
  occupancy 1. Retained: 119 registers, shared 73,728 B, occupancy 1.
- SASS: candidate/retained HMMA 64/64, LDGSTS 18/18, no stack or spills.
- Exact PASS: target, M tail with negative alpha, Kout tail, reduction tail,
  exceptional full tile and K=0; eager/graph ×3 and guards.

## Paired once3 candidate/retained ratios

| Path/order | p50 | p95 |
| --- | ---: | ---: |
| eager ABBA | 0.986460 | 0.987097 |
| eager BAAB | 0.987527 | 0.987855 |
| graph ABBA | 0.989411 | 0.989411 |
| graph BAAB | 0.990066 | 0.990735 |

Graph/BAAB misses `<0.99` by 0.0066% at p50 and 0.0735% at p95. The frozen
policy therefore stops before once7 and native-half cuBLAS Fast, despite the
directional improvement. No promotion, production or dispatcher change.

## Frozen evidence

- helper SHA256: `96650fe18ead4cdd5affad406383434e9d2050433842d5c65f92f3d72c6d4fbb`
- harness SHA256: `984a1233656ea9e76245f5821ffd1f0bbc85646ea7135ff112b68f8078c91dd4`
- candidate source SHA256: `4fe713da2307d4b56848c53731babc71c62d19b445277074ec4cf9e292efdbcf`
- raw log SHA256: `527d6996c64c9719ce395b50f13fc404ebef6ace6e7db0dd1e31575e53375984`
- [raw output](raw.log)
- [launch preflight](preflight.txt)
