# Ada Triad NT padded36 dense-copy discovery

This is a test-only CUDA 13.2 discovery result for logical NT shape
`(M,K,N)=(2048,768,3072)`. It compares the dense-copy candidate with the
actual public AUTO route. It does not measure or claim a win over cuBLAS Fast,
and `promotion=false`: full qualification has not run.

The frozen 373-source build manifest is
`860fb1ee864b847d7f1d7ddb641aed66be2fc7c229f24911bb14a4e93c9e948b`;
the test binary is
`4f94cba429630aab0f1ee3964b1f5c64d141d02e9286582703b5576683b73bb5`.
The candidate composed-source SHA is
`b256264ca71464e2a4a49382ff99239151a5c44da97bff1eba0e309772482ad4`.

Correctness covered finite full-mantissa target data, eager/graph repeats and
the forced generic tail. The live resource record was 154 registers/thread,
zero local bytes, zero static shared bytes, 82,944 dynamic shared bytes,
256 maximum threads and one resident CTA/SM.

| Path | Order | AUTO median (us) | Candidate median (us) | candidate/AUTO p50 | p95 |
|---|---:|---:|---:|---:|---:|
| eager | ABBA | 274.867999554 | 248.217594624 | 0.902877339 | 0.904695784 |
| eager | BAAB | 274.837601185 | 248.266398907 | 0.903320353 | 0.904387367 |
| graph | ABBA | 273.740804195 | 247.650396824 | 0.904951208 | 0.906080434 |
| graph | BAAB | 273.659205437 | 246.961605549 | 0.902442164 | 0.903388426 |

All four strata satisfy the discovery threshold, so the bounded decision is
`advance_to_full_qualification`. This means retain as a finalist; it is not a
production admission or AUTO promotion.

Replay the preserved source and timing evidence from the worktree root with:

```sh
ruby internal/perf/ada-triad-nt-padded36-dense-copy-20260907/replay.rb
```

The source hashes are frozen against commit
`f328ce4d2abebff150d3555037e4e743d75fdfc1` plus the two listed test-only
changes. A later checkout or edited discovery file must not be treated as the
measured source.

PRE was strict quiet/no-apps. Immediate RELEASE was non-quiet and is retained;
the separate five-second DRAIN was quiet/no-apps. Candidate PTX was compiled
and loaded in memory but was not persisted. The cache was deliberately
preseeded with the three hash-identical production artifacts and is not a
cold-cache measurement.
