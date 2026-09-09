# Ada Triad F16 NT M64N128/BK64/S2 ping-pong stop — 2026-09-09

## Decision

STOP without a cuBLAS Fast run. The test-only two-stage ping-pong candidate is
bit-exact and meets its resource target, but loses the measured M64N128/S3
retained kernel in all four paired once7 strata. Do not wire it into production,
do not run BF16 siblings, and do not retry this unchanged mechanism.

## Result

Shape: F16 NT d768-out `(M,N,K)=(2048,1536,768)`, CUDA 13.2, SM89 RTX 6000
Ada Generation, 20 GEMMs per observation.

| Path / order | candidate / retained p50 | p95 |
| --- | ---: | ---: |
| eager / ABBA | 1.04269 | 1.05290 |
| eager / BAAB | 1.04517 | 1.05735 |
| graph / ABBA | 1.06205 | 1.07606 |
| graph / BAAB | 1.06719 | 1.07708 |

The candidate is therefore 4.3–4.5% slower at eager p50 and 6.2–6.7% slower
at graph p50; worst p95 loss is 7.7%. The strict retained gate requires every
p50 and p95 to be `<0.99`, so the harness correctly stopped before allocating
or timing native-half cuBLAS Fast.

## Correctness and resources

- Exact candidate-versus-measured-S3 bits PASS for the full target, negative
  alpha tail `(67,131,69)`, exceptional/NaN corpus and K0.
- Eager, graph, three graph replays, guards, immutable inputs and the target
  20-operation accumulation check PASS.
- Candidate: 106 registers, local0, static shared0, dynamic shared49,152 bytes,
  occupancy2.
- Retained S3: 119 registers, local0, static shared0, dynamic shared73,728 bytes,
  occupancy1.

The resource hypothesis worked, but the extra CTA residency did not recover
the copy/compute overlap lost by reducing the pipeline from three stages to two.
Keep M64N128/S3. The next half-TN mechanism is the retained-pipeline B-fragment
load fusion (`ldmatrix.x2.trans` pairs to `ldmatrix.x4.trans`), which targets
instruction count without sacrificing stage overlap.

## Evidence identity

- `raw.log` SHA256 `a05654bc4af8d9e52e12fb9d3f00e5aef575ba524b35ccc9d1d8c578ec7e0e91`
- source adapter SHA256 `2e69c0b53bbbc91e389e219603c2d0ef6b75c56238cf4c32ce75c49cc873f9ec`
- standalone harness SHA256 `4591c83484eb45236d1c4dca2e606bfccb412c3fb583da832eb35a7945938bb3`
- Native source/ownership tests: 12/12 PASS.
- CUDA-feature `--no-run`: PASS.
- CUDA13.2 NVRTC candidate+retained compile-only: PASS without device/context.
- Immediate external and in-harness GPU preflight: idle with sufficient VRAM;
  see `preflight.txt` and `raw.log`.

Production and dispatcher sources were not changed.
