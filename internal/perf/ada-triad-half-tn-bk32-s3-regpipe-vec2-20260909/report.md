# Ada Triad BF16 TN compact BK32/S3 stop — 2026-09-09

Outcome: **STOP, no retry or promotion.** The test-only BF16 TN d768-in
`(2048,768,3072)` candidate changes the retained M64N64/BK64/S2
regpipe+`float2` kernel to a compact BK32/S3 ring. It preserves the padded K16
reduction order and passes exact-bit and resource gates, but loses the retained
kernel by 27.8–30.4% in every paired once3 stratum. once7 and cuBLAS Fast were
therefore not launched.

## Mechanism and gates

- CUDA 13.2 / SM89 RTX 6000 Ada, grid 576, block 128.
- Candidate resource PASS: 120 registers, local 0, static shared 24,576 bytes,
  occupancy 4. Retained: 125 registers, local 0, static shared 32,768 bytes,
  occupancy 3.
- Compile/SASS PASS: stack and spills 0, candidate/retained HMMA sites 16/32,
  candidate LDGSTS 28, and no LDL/STL/ATOM/RED/REDUX.
- Exact PASS: target, aligned and misaligned negative-alpha tails, exceptional
  BF16 values, K=0, eager/graph repeats, 20-op accumulation and guards.
- The candidate uses `2 * ceil(M_red / 64)` BK32 tiles. This retains the exact
  parent's four-K16-per-BK64 arithmetic sequence, including padded zero MMA
  groups at reduction tails.

## Paired once3 candidate/retained ratios

Ratios are candidate / retained for 20 logical GEMMs per observation.

| Path/order | p50–p95 observed range |
| --- | ---: |
| eager ABBA/BAAB | 1.2779–1.3038 |
| graph ABBA/BAAB | 1.2909–1.2987 |

Every stratum fails the predeclared strict `p50 < 0.99 && p95 < 0.99` scout
gate. The extra BK32 stage transitions dominate the occupancy improvement;
four resident CTAs do not recover the cost. Keep the prior BK64/S2
regpipe+vec2 BF16 TN d768-in kernel and do not retry this unchanged mechanism.
Production and dispatcher remain unchanged.

## Frozen identity

- helper SHA256:
  `f7e95c882ddbc292612a9036d510a121554cb58e032ebb914af4fcfe435a96fd`
- harness SHA256:
  `415bee2c2be8ded1049720f049b25c55477f3712a15f390c8940790ad3a2cbf1`
- composed candidate/retained source SHA256:
  `4e2941b424a79ae1b8f5fd148f59d7f2dc9b2236a9c134ef6b91415ba65ebfe0` /
  `9412e655faafe8b0f91c8d593a8cda0cafdb6320c04285cd9b396cfdcf36874f`
- candidate/retained PTX SHA256:
  `81023add80799f1adb35b6a668feae1c890cf55cc9ee29fad7b24b2792fc9d78` /
  `2d75e845ff58760f073898cd309a97d4c68e0a2e32d0eb39778868d10ef85e4c`
- candidate cubin/SASS SHA256:
  `f22492b71e535736371f71ff561d64245ed2177be191010c07ef85b63081f9fd` /
  `df354ac2eaaaa554e9e7e78427fa071becb62155c56c9709e81388a12002def0`
- raw log SHA256:
  `1077d0b3f4808ed3923e0c253bf171e14a414a418c235b86d855a24598727005`

Launch conditions are summarized in [preflight.txt](preflight.txt). The GPU
executor output is preserved in [raw.log](raw.log).
