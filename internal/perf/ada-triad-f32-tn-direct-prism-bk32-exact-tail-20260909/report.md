# Ada exact-F32 TN direct Prism BK32 exact-tail stop — 2026-09-09

Outcome: **valid exact/resource candidate, rejected at once3.** This test-only
candidate keeps the retained direct M64N64 raw-partial geometry and production
FP64 reducer, but doubles the K tile from16 to32. It uses explicit exact tails:
24 full BK32 tiles plus16 values for every784 chain, and21 full tiles plus29
values for the701 chain. This avoids the83 extra FFMA and signed-zero or
exception drift that a naively padded BK32 implementation would introduce.

All six raw partials and target/tail/exceptional/non-unit/K0 eager+graph final
bits, inputs and guards pass. The candidate uses107 registers, zero local
bytes,32KiB shared and occupancy3; retained BK16 uses the same registers with
16KiB shared and occupancy4.

Candidate/retained once3 p50/p95:

| Path/order | Ratio |
| --- | ---: |
| eager ABBA | 1.038693 / 1.039573 |
| eager BAAB | 1.038365 / 1.038793 |
| graph ABBA | 1.038257 / 1.038403 |
| graph BAAB | 1.037713 / 1.038901 |

Halving tile-loop/barrier rounds does not compensate for the occupancy loss.
Stop before once7/Fast; retain BK16 and do not retry unchanged.

- helper SHA256: `97dad7f06a07d00823bd9820d2ddb6964ade7c7e684c45118c78d3c8c1a9e8a3`
- harness SHA256: `92b8f9d7c8ba5c35d757f97f80111054db25114e620768449891cd386dade137`
- candidate source SHA256: `96d29c3284266ca04addf605d679b2827c7bac9081f258406ef2f135abdeb621`
- retained direct source SHA256: `07ec10cf27ac814a80f7e9c918a6bc2268774347d284d06603fa2aa022b42eef`
- CUDA13.2, RTX6000Ada, CC8.9,142 SM
