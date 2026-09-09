# Ada Triad F16 NT d768-out M64N192 issue-2/2/1/1 stop — 2026-09-09

Outcome: **once3 STOP; physically distinct schedule is parity/slightly slower,
with no production change.**

The test-only `(2048,1536,768)` candidate changes the M64N192/BK64/S3
`cp.async` issue distribution from `1/1/2/2` to `2/2/1/1`. It moves the full
slice 4 earlier and leaves the partial slice 5 before commit/wait. The ptxas
gate proves a distinct LDGSTS/HMMA issue signature while preserving 64 HMMA,
18 LDGSTS, 127 registers, local0, 98,304 bytes dynamic shared and occupancy1.
Target, tails, exceptional values, K0, eager/graph repeats, inputs and guards
all pass bit-exactly.

Candidate/retained M64N192 once3 p50/p95:

| Stratum | p50 | p95 |
| --- | ---: | ---: |
| eager ABBA | 1.000613 | 1.000736 |
| eager BAAB | 1.000654 | 1.001309 |
| graph ABBA | 1.000000 | 1.000669 |
| graph BAAB | 1.000000 | 1.000669 |

The schedule does not improve the retained body, so once7 and cuBLAS Fast were
not run. Do not retry this unchanged issue ordering.

Frozen identities:

- helper SHA256: `caa6ba4e72b0cdd115c2ad8f4642f0ccf3064fd7915cb9eaade4f03172abea0d`
- harness SHA256: `e7502be8dd89f022a16afa573187ffdb053c7dcf5299611c72f3e32162939097`
- candidate transformed source SHA256: `31802d9774610fa0792324be715f828cf0c3cbf1e1445ddaaafc789543fdd6fb`
- candidate PTX SHA256: `af2f5d283946bf832961be610e24114992290f545d5f999e8fde62044d6aed73`
- candidate CUBIN SHA256: `2121be1a640450dbeea178943452cb1044d76c8684c8967fe1e4ca308de8a0c1`
- candidate SASS SHA256: `9b34f8bb9390855b6b790de7e19986efba01469c182397f89d76f20e7a5cb16f`
- [raw qualification summary](raw.log)
