# Ada TF32 NN M96N96/S2 single-fragment Prism stop — 2026-09-09

Outcome: **valid exact/resource candidate, rejected at once3.** This test-only
follow-up branches from the frozen137-register M96N96/BK32/S2 candidate and
changes only fragment lifetime: two preloaded fragment banks become one
issue-scoped bank. CopyPlan/stage slicing, add-half conversion, ascending K8
MMA order, geometry and direct-float2 epilogue remain unchanged. Production
dispatch is unchanged.

The resource objective succeeds:125regs/local0/49,152B dynamic shared/
occupancy2, zero stack/spills,48 HMMA and16 LDGSTS. Full/tail/exception/K0
eager+graph bits, repeat checks and guards pass against retained direct N96.
Removing fragment-load/MMA overlap, however, costs more than occupancy2 saves.

Candidate/retained once3 p50/p95:

| Path/order | Ratio |
| --- | ---: |
| eager ABBA | 1.008376 / 1.009261 |
| eager BAAB | 1.007709 / 1.011324 |
| graph ABBA | 1.017054 / 1.017339 |
| graph BAAB | 1.016688 / 1.019506 |

The candidate is0.77–1.71% slower at p50 and misses the strict retained gate.
Stop before once7/Fast and do not retry unchanged.

- helper SHA256: `cfaacdae12d36fd99356613a04b046160bc3e22f2cfcd1411138d8030df77931`
- harness SHA256: `329e6f928bc3630ca535dae2795c2b3f093fb18cfecc9c2ab83770b6b24d257f`
- composed candidate source SHA256: `7435acb49fe521cc48aba6e2b7936f4e22800c8daf1b6279fc27ec1918733114`
- CUDA13.2, RTX6000Ada, CC8.9,142 SM
