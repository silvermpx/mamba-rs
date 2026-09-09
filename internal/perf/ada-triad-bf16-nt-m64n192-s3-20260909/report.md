# Ada Triad BF16 NT d768-out M64N192/S3 stop — 2026-09-09

Outcome: **once3 STOP; eager improves by 1.23–1.44%, but the graph retained
gate narrowly misses. No production change.**

The test-only `(2048,1536,768)` candidate ports the exact F16 M64N192/BK64/S3
geometry to BF16 and compares it directly with the retained BF16 M64N128/BK64/S3
body. CUDA13.2 compilation preserves 64 HMMA and 18 LDGSTS instructions. The
candidate uses 127 registers, no local memory, 98,304 bytes dynamic shared and
one active CTA/SM. Target, tails, exceptional values, K0, eager/graph repeats,
inputs and guards all pass bit-exactly.

Candidate/retained once3 p50/p95:

| Stratum | p50 | p95 |
| --- | ---: | ---: |
| eager ABBA | 0.986332 | 0.987097 |
| eager BAAB | 0.987088 | 0.987742 |
| graph ABBA | 0.990073 | 0.990722 |
| graph BAAB | 0.990066 | 0.990073 |

The strict retained admission requires every p50 and p95 ratio below `0.99`.
Both graph cohorts miss that threshold, so once7 and cuBLAS Fast were not run.
Do not promote or retry this unchanged BF16 adapter.

Frozen identities:

- helper SHA256: `0ef5d4440e02869d2f15d27e864b12e7f8d3f6dae6e4beee710397c272ab47f8`
- harness SHA256: `38e7f93e31068c8cb81f09201e28f7104626a9d0018ddd86119979615d3702d1`
- candidate transformed source SHA256: `0ff0d14369687a222b5fc5fc8ae0a64a0b476efa9c76c0da188e6a42906d84eb`
- candidate PTX SHA256: `e8f2cb336d9fed5b5cddc937a9014a3e46adf2d5853a8bafa986eb39817d9040`
- candidate CUBIN SHA256: `f0e476aca7bbb46f514600d15d5f0a85a0e5f12530d6ab7e236b47a69d235645`
- candidate SASS SHA256: `eeecabe58728ff452d9b5854ca3302946c6cd772d7d92d074f7fe1e37825df88`
- [raw qualification summary](raw.log)
