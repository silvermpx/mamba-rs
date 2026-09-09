# Ada Triad F16 NN S3 B-x4 stop — 2026-09-09

Outcome: **STOP, no retry or promotion.** The test-only F16 NN d768-in
`(2048,768,3072)` candidate replaces the retained S3 kernel's four literal
`ldmatrix.x2.trans` B loads with two `ldmatrix.x4.trans` loads. It is bit-exact
and reduces registers, but loses the frozen retained S3 kernel in every paired
once3 stratum. The retained gate therefore stopped before once7 and cuBLAS Fast.

## Gates

- CUDA 13.2, compute capability 8.9, 142 SM RTX 6000 Ada.
- Resource PASS: candidate 168 registers versus retained 188; both local 0,
  dynamic shared 98,304 bytes, block 256 and occupancy 1.
- Exact PASS: target, negative-alpha M tail, N tail, K tail, exceptional full
  tile and K=0; eager/graph repeated three times with guard checks.
- Paired candidate/retained once3 ratios:

| Path/order | p50 | p95 |
| --- | ---: | ---: |
| eager ABBA | 1.010724 | 1.010747 |
| eager BAAB | 1.011187 | 1.011191 |
| graph ABBA | 1.012778 | 1.013143 |
| graph BAAB | 1.012774 | 1.012778 |

The strict `<0.99` retained gate failed in all four strata. Preserve Fixed S3
as the F16 NN d768-in retained-best and do not retry this unchanged mechanism.
Production and dispatcher are unchanged.

## Frozen evidence

- source adapter SHA256: `068b23e48c8f460066e8cc80e0239e3104d3748aef88d2b54f8a2c0c812ef765`
- harness SHA256: `09a60991c69be69f2710e59fdc70bc0054b0580412e07b708492153d60bf06de`
- raw log SHA256: `d98f67492acbbcd420f3e15a59a6121025514e77c72a4a848e01f7ec2268dd3f`
- raw output: [raw.log](raw.log)
- launch preflight: [preflight.txt](preflight.txt)
