# Ada Triad NT compact32 eight-warp S2 discovery

This is a test-only CUDA 13.2 discovery result for logical NT shape
`(M,K,N)=(2048,768,3072)`. It compares the compact32 eight-warp S2 candidate
with the actual public AUTO route. It does not measure or claim a win over
cuBLAS Fast, and `promotion=false`: full qualification has not run.

The frozen 373-source build manifest is
`559ebcd83c8e810f324d0a733317816118ba5cfa64d91000916e987a69c27210`;
the test binary is
`6e8d56473a0fa2954162400fa559eee1cc73abadc0d6abe7ecad82b8e9cf7695`.
The candidate composed-source SHA is
`fd5bd92f7cc56276a1b95b24b59319142b247f7ddedb14d6b06cac91843205b4`.

Correctness covered finite full-mantissa target data, eager/graph repeats and
the forced generic tail. The live resource record was 121 registers/thread,
zero local bytes, zero static shared bytes, 49,152 dynamic shared bytes,
256 maximum threads and two resident CTAs/SM.

| Path | Order | AUTO median (us) | Candidate median (us) | candidate/AUTO p50 | p95 |
|---|---:|---:|---:|---:|---:|
| eager | ABBA | 275.013815273 | 226.256717335 | 0.822563921 | 0.823097796 |
| eager | BAAB | 274.943991141 | 226.261821660 | 0.822837094 | 0.823171353 |
| graph | ABBA | 273.751995780 | 226.021820849 | 0.825599198 | 0.825873428 |
| graph | BAAB | 273.687265136 | 225.978179411 | 0.825800614 | 0.826036828 |

All four strata satisfy the discovery threshold, so the bounded decision is
`advance_to_full_qualification`. This means retain as a finalist; it is not a
production admission or AUTO promotion.

Replay the preserved source and timing evidence from the worktree root with:

```sh
ruby internal/perf/ada-triad-nt-compact32-eight-warp-s2-20260907/replay.rb
```

The source hashes are frozen against commit
`e89d25a7c6e3b04ddbff725d62a95a7190c17cf0` plus the listed test-only change.
A later checkout or edited discovery file must not be treated as the measured
source.

PRE was strict quiet/no-apps. Immediate RELEASE was non-quiet and is retained;
the separate five-second DRAIN was quiet/no-apps. Candidate PTX was compiled
and loaded in memory but was not persisted. The cache was deliberately reused
from the preceding isolated discovery build and is not a cold-cache
measurement.
