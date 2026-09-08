# Half NN N64: BF16 d768-in reaches near Fast parity

Ada/CUDA13.2,2026-09-08. Fixed N64 body reused with isolated exact-shape
Triad wrappers, preserving MMA order/packed RNE epilogue. No production edits.

| Shape / dtype | Paired candidate/current p50 | Candidate/Fast p50 | Selection |
| --- | ---: | ---: | --- |
| d768-in F16 | .857–.864 | 1.292–1.315 | Stop vs Fast |
| d768-in BF16 | .857–.864 | .981–.992 | Near parity; below1%, no strict admission |
| d768-out F16 | .723–.731 | .827–.844 | Reserve; prior S3 is faster |
| d768-out BF16 | .724–.731 | .820–.835 | Reserve; prior S3 is faster |

BF16 d768-in: eager79.15–79.43us vs Fast80.62–81.11; graph78.09–78.57
vs78.65–79.26. Graph p95 .9945–.9946 fails the strict <.99 retention rule.
This is a useful new near-parity result, not a robust Fast champion.
The two d768-out raw ADVANCE decisions compare only current/Fast, not the
already retained aligned S3 (~.77–.80/Fast). Do not replace S3 with N64 or
present these as additional winning cells; cohorts are separately paired.

One exact GPU test PASS:4 candidate+4 current resources/32 screens/4 decisions,
256B aligned, once7 ×eager/graph ×ABBA/BAAB,20 GEMMs/observation. Exact current
bits/guards and Fast own bits pass during timing. Root replay224 brackets/
896 observations and64 quantiles PASS. Candidate block128/dynamic49152,
local0/occupancy2. Main source291e71c2 archived independently of newer WIP.

Attempt1 never ran GPU: next source was installed while the previous S4
build finished, causing Cargo to reuse the older binary; exact-list gate
rejected it. Attempt2 touched the same frozen source and rebuilt, then
verified the exact test before execution. No valid runtime rerun. Future
source installation must wait for the preceding build AND process to exit.

Main `291e71c2d185b0923187e1ac4dd58e9449886059310d51fd269f339a3f424348`;
helper `fc552af4f9a64eb0ea5ab965396d369729f4bbfcc24ddd7f3f994937abf8fb5c`.
[Raw](evidence/attempt2/once7-cuda132/test.log), SHA256
`b9a492c7a7e769ac7b99052e8c10c954127d81332371d7b1fe69caa4c95d67fb`.
