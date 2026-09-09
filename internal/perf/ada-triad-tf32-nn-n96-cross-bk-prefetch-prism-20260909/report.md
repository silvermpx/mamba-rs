# Ada TF32 NN N96 cross-BK prefetch resource stop — 2026-09-09

Outcome: **resource stop before exact/timing.** This test-only candidate keeps
the frozen direct-float2 M128N96/BK32/S3 arithmetic and moves only ten safe
next-tile fragment0 loads across the current tile's final MMA. The final drain
boundary stays in parent order so `wait_group1` remains race-free. Production
dispatch is unchanged.

SASS is distinct and preserves48 HMMA,21 LDGSTS,86,016B dynamic shared, zero
stack/spills and a1.0027 text-size ratio. Register use, however, rises
124->149, exceeding the frozen124 cap and eliminating the intended scheduling
benefit. Exact and timing did not start. Stop this cross-BK overlap mechanism
unchanged; do not attempt a lifetime patch within the same topology.

- helper SHA256: `69046f92d863d624f5a3ffdcca04e55f0b572ae5daa4ffed8a332ab31313389e`
- harness SHA256: `8590d88b2bc345b85ec789861c7adfee1fea44875520a3a64d7a66dd65fce16b`
- CUDA13.2, RTX6000Ada, CC8.9,142 SM
