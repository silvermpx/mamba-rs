# Ada TF32 NN N96 grid-constant Prism stop — 2026-09-09

Outcome: **valid exact/resource candidate, rejected at once3.** This test-only
candidate changes only the32-byte kernel Params residency from by-value to
`__grid_constant__` with const-reference threading. The frozen direct-float2
M128N96/BK32/S3 geometry, add-half conversion, K8 MMA order, copies and
epilogue remain unchanged. Production dispatch is unchanged.

CUDA13.2 emits distinct SASS with eight fewer instructions and128 fewer text
bytes. Candidate and retained both use124regs/local0/86,016B dynamic shared/
occupancy1,48 HMMA and21 LDGSTS with no stack or spills. Full/tail/exception/
K0 eager+graph bits, repeats and guards pass.

Candidate/retained once3 p50/p95:

| Path/order | Ratio |
| --- | ---: |
| eager ABBA | 1.074787 / 1.074907 |
| eager BAAB | 1.074136 / 1.075112 |
| graph ABBA | 1.075203 / 1.075287 |
| graph BAAB | 1.075462 / 1.076230 |

The candidate loses7.4–7.6% consistently. The strongest source/statistics
inference is that avoiding one-time Params materialization adds repeated
dependent parameter-space accesses without any register/occupancy dividend;
the deleted temporary SASS listings do not support a stronger opcode claim.
Stop before once7/Fast and do not retry the qualifier unchanged.

- helper SHA256: `bd26e12f7cfda56d192802dcc6bc8701e0bef0b4f9e9234fafbb245e5663ee70`
- harness SHA256: `53df37faee159c37ffdaca825e0cd0cf12e33492b2d59e4cd6700cdc7fb55ceb`
- composed candidate source SHA256: `af21ec643427f1349dcc903eb357e0ee87b5efd34e49c6237fc02fd6b49c3643`
- CUDA13.2, RTX6000Ada, CC8.9,142 SM
