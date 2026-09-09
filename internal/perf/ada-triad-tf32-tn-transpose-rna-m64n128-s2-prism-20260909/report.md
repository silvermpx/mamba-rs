# Ada TF32 TN transpose-RNA M64N128/S2 Prism stop — 2026-09-09

Outcome: **valid exact/resource candidate, rejected at once3.** Starting from
the retained M64N64/S3 A-only transpose-RNA winner, this test-only arm doubles
N reuse with M64N128/BK32/S2. It preserves RNA conversion, K8 order and the
two-node transpose+GEMM graph. Production dispatch is unchanged.

Full/tail finite+exceptional and K0 eager+graph bits, guards, alignment and ABI
pass. Candidate resources are123 registers, zero local bytes,49,152 bytes
dynamic shared and occupancy2; retained is83regs/local0/49,152B/occupancy2.
The candidate target grid is96 CTAs versus186 retained CTAs.

Candidate/retained once3 p50/p95:

| Path/order | Ratio |
| --- | ---: |
| eager ABBA | 1.029000 / 1.038420 |
| eager BAAB | 1.027328 / 1.044892 |
| graph ABBA | 1.032207 / 1.046980 |
| graph BAAB | 1.010659 / 1.026643 |

Higher B reuse does not offset the smaller grid and larger accumulator set.
Stop before once7/Fast; retain M64N64/S3 and do not retry unchanged.

- helper SHA256: `9e6c7c260fb2abcff875450093333215032c8a91c71792d29e9762f93eab256b`
- harness SHA256: `551db85249f98293a88366c97b13d1b63a6973122a9eced198437babf4a9572d`
- candidate source SHA256: `75fe171a18e6a85d9d7dfc1ff54d8f734a191f1aa3fd0f2f6a6c07c0a23147f5`
- candidate PTX SHA256: `f7fcd250e26415ad3a17a61ab394761f0b865e4bec1a8a3e760c9fdfbe24fd30`
- CUDA13.2, RTX6000Ada, CC8.9,142 SM
