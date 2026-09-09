# Ada Triad BF16 NT M96N128/S3 d768-out Fast winner — 2026-09-09

Outcome: **new retained-best and strict cuBLAS Fast winner.** The test-only
BF16 sibling reuses the physical M96N128/BK64/S3 geometry proven by the F16
winner, changing only the export/template instantiation while preserving the
BF16 HMMA and RNE store contract. Production dispatch is unchanged pending
joint whole-Triad integration.

Target, M/N/K tails, exceptional and K0 eager+graph bits, repeats, inputs,
256-byte alignment and guards pass against retained BF16 M64N128/S3. The
candidate uses120 registers, zero local bytes,86,016 bytes dynamic shared and
occupancy1. Its target grid is264 CTAs. It retains64 HMMA and reduces staged
traffic9.77% and LDGSTS18->15 relative to the M64N192 family.

Candidate/retained once7 p50/p95:

| Path/order | Ratio |
| --- | ---: |
| eager ABBA | 0.923176 / 0.924467 |
| eager BAAB | 0.922531 / 0.923116 |
| graph ABBA | 0.921958 / 0.922721 |
| graph BAAB | 0.920687 / 0.921958 |

Candidate/cuBLAS Fast once7 p50/p95:

| Path/order | Ratio |
| --- | ---: |
| eager ABBA | 0.915661 / 0.933944 |
| eager BAAB | 0.926098 / 0.949007 |
| graph ABBA | 0.897486 / 0.919578 |
| graph BAAB | 0.909387 / 0.913072 |

All eight retained and Fast p50/p95 values pass `<0.99`. Retain for joint
integration. This moves BF16 NT d768-out from N to Y and raises the strict Ada
Triad Fast-win inventory to11/60.

- helper SHA256: `7bd3026e5cee4ebc9df9c2e3d923453a181d067c595c406eae86e1259627f6b6`
- harness SHA256: `b9a41dfa15a77340c9225f6485e2b04079b75a6f24759e4be98f005a5c68018c`
- candidate source SHA256: `dff51cd874dd8e9f60f2a273910b407967a7dcd1da674abc3922b3546534c92e`
- retained source SHA256: `d80cc5376697b48e846191461870af579762f55796873d897f48e3de3568c570`
- CUDA13.2, RTX6000Ada, CC8.9,142 SM
