# Ada Triad F16 NT d768-out M64N192 full-domain stop — 2026-09-09

Outcome: **once3 STOP; 2.9–3.1% slower than measured M64N192, no production
change.**

The test-only `(2048,1536,768)` candidate adds a target-only entry that passes
literal dimensions into the existing M64N192/BK64/S3 body. All non-target
shapes physically use the measured generic M64N192 comparator. The hypothesis
was compile-time folding of target extent, tail, stride and stage control in a
kernel that had previously missed the strict retained gate by less than0.1%.

CUDA13.2 compile/SASS passes and reduces registers127->118 while preserving64
HMMA and18 LDGSTS, local0, dynamic shared98,304 and occupancy1. Exact target,
M/K-out/reduction tails, exceptional shape, K0, eager/graph repeats, inputs and
guards pass.

Candidate/generic-M64N192 once3 p50/p95:

| Stratum | p50 | p95 |
| --- | ---: | ---: |
| eager ABBA | 1.029408 | 1.029450 |
| eager BAAB | 1.028775 | 1.028796 |
| graph ABBA | 1.030790 | 1.031480 |
| graph BAAB | 1.030790 | 1.031480 |

The specialization is consistently slower despite lower register count, so
once7 and cuBLAS Fast were not run. Do not retry this unchanged wrapper.

Frozen identities:

- helper SHA256: `8cc3ea33a1076b9be5761433fe8ac3445f1a381bb1aed4e20c9f86aa1eb79570`
- harness SHA256: `87f928d0e1d2bbab1297dfce592af5359cd90b0f5cdd7854d7747b40738bd6b0`
- candidate transformed source SHA256: `6fea2b9a028248d07fd1879004de393446151b5cb3fcf769f4653666e495a9fa`
- candidate PTX SHA256: `83394b7f66f9568d205a463acc6e34e939377997c95b55455500ef3dee6eceed`
- candidate CUBIN SHA256: `cb84a3f3b32bf5aa17f223a7228afebf8ec7ae2d122f75b7ce5b2fd953424c41`
- candidate SASS SHA256: `e0571c10e1039f4c8fe4c9d8503e7cbf61fe67fb748225f44aa68b4e8058add2`
- [raw qualification summary](raw.log)
