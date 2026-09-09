# Ada Triad F16 NT d768-out M96N128/S3 Fast winner — 2026-09-09

Outcome: **new retained-best and strict cuBLAS Fast winner.**

The test-only `(2048,1536,768)` candidate rotates the 12-warp output tile from
M64N192 to M96N128 while preserving BK64/S3 arithmetic. Its 3x4 warp mapping
keeps 64 HMMA sites but reduces staged operand traffic per K tile from
4,194,304 to 3,784,704 half elements (9.77%), improves the target grid from
256 to 264 CTAs, and reduces shared memory from 98,304 to 86,016 bytes.

CUDA13.2/SASS reports 120 registers, local0, 64 HMMA and 15 LDGSTS versus the
retained M64N192 body's 127 registers, local0, 64 HMMA and 18 LDGSTS. Target,
M/N/K tails, exceptional values, K0, eager/graph repeats, inputs and 256-byte
aligned guards all pass bit-exactly.

Candidate/retained M64N192 once7 p50/p95:

| Stratum | p50 | p95 |
| --- | ---: | ---: |
| eager ABBA | 0.935209 | 0.937132 |
| eager BAAB | 0.934512 | 0.937156 |
| graph ABBA | 0.931727 | 0.933066 |
| graph BAAB | 0.931773 | 0.933735 |

Candidate/cuBLAS Fast once7 p50/p95:

| Stratum | p50 | p95 |
| --- | ---: | ---: |
| eager ABBA | 0.899181 | 0.904822 |
| eager BAAB | 0.898113 | 0.907761 |
| graph ABBA | 0.905195 | 0.920687 |
| graph BAAB | 0.914098 | 0.919578 |

Every retained and Fast p50/p95 ratio is below `0.99`. Retain this candidate
for the joint Triad dispatcher integration batch; no production route changed
in discovery.

Frozen identities:

- helper SHA256: `a6ae69e3cdef747d7b0a306b3baa636b774da74ee49ae74840a12e7f77931312`
- harness SHA256: `80421373369fee76c4c9f93543e2402f20b5e47b0fa0f2ec65a239c3410ed3c2`
- candidate transformed source SHA256: `a204568d48de8e5b2011032f641218a11505daf6ce99c9cd0d36bcdcc8746328`
- candidate PTX SHA256: `6d98c9398243db94a569eac056b8f0aefe9880989c03d1028b5e2f50f2d51764`
- candidate CUBIN SHA256: `a047e32c9bf3197d88e60d2541215361703d3b3c10e5b50baa315e99b99f6ab3`
- candidate SASS SHA256: `d4da947a6746345ac0b42821af7f7c3567bc0fd6bb4ac1bc2e139b11dcfb3b46`
- [raw qualification summary](raw.log)
