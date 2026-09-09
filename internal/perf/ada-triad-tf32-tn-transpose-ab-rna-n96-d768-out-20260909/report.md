# Ada Triad TF32 TN joint A+B RNA d768-out stop — 2026-09-09

Outcome: **once3 STOP; 1.6–2.2% slower than the new A-only RNA winner.**

The test-only candidate keeps the two-node pipeline but extends the first node
to transpose+RNA A and flat RNA-copy B into separate256-byte-aligned scratch.
The second N96 GEMM consumes pre-rounded A and B. A-only transpose-RNA N96 is
the retained comparator.

Two resource-only iterations preceded the measured source. Keeping six
double-buffered B words compiled at137 registers and stopped at the128 cap.
Using a dynamic n-atom loop reduced registers to96 but spilled288 bytes/thread
to local memory. The final source emits three explicit n-atom scopes, each
holding only two B words and immediately issuing four static-index MMAs. It
passes at114 registers, local0, dynamic shared86,016 and occupancy1 versus
retained127 registers. Candidate preprocess is28 registers/local0/static
shared4,224 versus retained transpose26 registers.

Exact full finite/exceptional tiles, tail finite/exceptional, K0, target,
eager/graph repeats, both scratch oracles, graph ABI and guards pass.
Candidate/A-only retained once3 p50/p95:

| Stratum | p50 | p95 |
| --- | ---: | ---: |
| eager ABBA | 1.021505 | 1.026882 |
| eager BAAB | 1.022345 | 1.032258 |
| graph ABBA | 1.016129 | 1.016216 |
| graph BAAB | 1.021739 | 1.027174 |

The extra B scratch traffic costs more than removing repeated B conversion.
Stop before once7/Fast; retain A-only transpose-RNA and do not retry unchanged.

Frozen final identities:

- helper SHA256: `a734caf0ebbb240601f8e96d50f6bf0cbfe6722217d3c87c7695b52581315a68`
- harness SHA256: `5b0435521bac28a342ee1941359181004c19353c58b3be99157e3e9b055c86e5`
- candidate transformed source SHA256: `4c85dc3022262b10342996fd8aaf638c814c2c4b8497b01ff7d33e2ab6b0d5ec`
- candidate PTX SHA256: `a42a6ab45af0f11bbd5aaa94d634223fa33787badc3846cd4bc574bfbfa7db15`
- retained transformed source SHA256: `3ee1e8986c3d1fb998aef408f14add2203d71bbd6f8521f08500ac972f2d2f65`
- retained PTX SHA256: `57808541ffcf29b65c0bdecb14c85e9a6e185f84da39167c4c5d3cf8df313ee6`
- [raw qualification summary](raw.log)
