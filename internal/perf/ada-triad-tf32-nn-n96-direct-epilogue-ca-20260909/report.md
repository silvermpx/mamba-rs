# Ada TF32 NN N96 direct-epilogue CA-cache stop — 2026-09-09

Outcome: **valid exact/resource candidate, rejected at once3.** This test-only
candidate changes only the two N96 stage-copy calls from L2-only
`cp.async.cg` to L1+L2 `cp.async.ca`; the retained direct-float2 epilogue,
arithmetic, geometry and dispatcher remain unchanged.

Full/tail/exceptional/K0 eager and graph bits, guards, input immutability and
graph identity pass. Candidate and retained both use124 registers/thread,
zero local bytes,86,016 bytes dynamic shared and occupancy1.

Candidate/retained once3 p50/p95:

| Path/order | Ratio |
| --- | ---: |
| eager ABBA | 1.008794 / 1.009759 |
| eager BAAB | 1.009500 / 1.011397 |
| graph ABBA | 1.018455 / 1.018649 |
| graph BAAB | 1.020059 / 1.020999 |

L1 caching is consistently slower for this high-shared-carveout N96 body.
Stop before once7/Fast and keep `cp.async.cg`; do not retry unchanged.

- helper SHA256: `3f81f33d5efd9c8c753846d34ede198987558d0f849359e699f84b5c82e2bd7e`
- harness SHA256: `0c85a3c13be322091e632c6763605df49b10abc30c18937785015ea2f0733aaf`
- candidate source SHA256: `873b84c08c3b9fff8f049f401a778a6f508ce67fa88d00cbf07fbb5e13ef9f97`
- candidate PTX SHA256: `1ddf66e9f0ec70d7f7c4ccc3120f37ca57e682b738643d0c11c83ad259e22f2a`
- CUDA13.2, RTX6000Ada, CC8.9,142 SM
