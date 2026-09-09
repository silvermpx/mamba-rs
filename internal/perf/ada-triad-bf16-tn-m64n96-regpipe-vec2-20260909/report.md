# Ada BF16 TN M64N96 regpipe+vec2 d768-out stop — 2026-09-09

Outcome: **valid exact/resource candidate, rejected at once3.** This test-only
six-warp M64N96/BK64/S2 body targets `(2048,1536,768)`. It reduces staged half
traffic16.67% relative to retained M64N64 while preserving twelve resident
warps and avoiding the M64N128 last-wave imbalance. Production dispatch is
unchanged.

Target, aligned/misaligned tails, exceptional and K0 eager+graph bits, inputs
and guards pass against retained BF16 M64N64 regpipe+vec2. Candidate resources
are118regs/local0/49,152B static shared/occupancy2; retained is
125regs/local0/32,768B/occupancy3. SASS preserves32 HMMA,24 LDSM and24 LDGSTS.

Candidate/retained once3 p50/p95:

| Path/order | Ratio |
| --- | ---: |
| eager ABBA | 1.009144 / 1.015685 |
| eager BAAB | 1.009018 / 1.010127 |
| graph ABBA | 1.006613 / 1.008205 |
| graph BAAB | 1.005514 / 1.008337 |

The candidate is0.55–0.91% slower at p50 and misses the strict retained gate.
Stop before once7/Fast and do not retry unchanged.

- helper SHA256: `cc7d112e37bb176c19253833ad6d5bd01040e032dcc5f634ea7b5f0f487534cf`
- harness SHA256: `b55edf06e9b77ade3724df44425d374ebbafa35d5517dfbf04141bd96ab7a657`
- composed candidate source SHA256: `df641e2fdfe8d663231668a05e44717074963c79dc05df56fd5baa847a9c90bc`
- CUDA13.2, RTX6000Ada, CC8.9,142 SM
