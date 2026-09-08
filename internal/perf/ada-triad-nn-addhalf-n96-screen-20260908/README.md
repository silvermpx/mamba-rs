# New TF32 Triad NN N96 discovery, Ada, 2026-09-08

This is a new test-only candidate, not the earlier scalar CopyPlan or half-S3
reuse results and not yet a production admission or cuBLAS Fast win.

Target `(M,K,N)=(2048,1536,768)`. Reuse the existing Fixed M128N96/BK32/S3
geometry through an exact, reversible test-source transform: replace RNA
operand conversion with the current Triad-wide `bits + 0x1000U` conversion and
rename the export. Production Fixed CUDA is unchanged. Ascending K8 MMA order,
one-CTA output ownership and the current Triad conversion are retained.

The geometry hypothesis is 128 output CTAs rather than the current wide tile's
96 CTAs on 142 SMs. More CTAs do not guarantee a win: the smaller tile also
changes reuse and scheduling. Only paired elapsed-time measurements decide.

## Invalid first comparisons, preserved

- Original89947/bcb6 failed compilation: missing bracket-order helper.
- Repair9e659/2d8b compiled, passed tail/exceptional/K0 checks and stopped on a
  target reference mismatch before any valid speed result.
- Diagnostic9768/0bbe established that candidate and current-wide outputs were
  bit-identical on the target; both differed in 662,583 words from the arm
  labelled AUTO. That arm incorrectly called `gpu_gemm_typed_forward_raw`,
  which selects the F32 matvec path for this policy. The separately qualified
  identity came from `gpu_gemm_bi_forward_raw`, a different actual launch.

This is a test-entry-point error, **not evidence that AddHalf conversion must be
changed to RNA**, and not a valid performance STOP for N96. Initial failures,
diagnostic raw and source snapshots remain in this directory.

## Corrected public-AUTO freeze

Main SHA `700f8ff8ecc59928b011ea36348e39a9f62ba8adc600b8fcb67342587c0a6ef8`.
Helper SHA `1898941bd07c6884b2b1a739da92e7be75b402af2e522d663448e0a979ec0053`.
Native6 passes; root and an independent reviewer accepted the correction.

The AUTO arm now invokes `gpu_gemm_bi_forward_raw` with separately owned,
active-origin-zero A/B/C buffers containing identical values and trailing
guards. Direct candidate/current arms retain two-sided guards. Eager and
graph output bits, input immutability, guards and the actual wide graph symbol,
launch configuration and ABI are checked. Qualification metadata is gathered
in a separate scope; its holder is dropped before comparative CUDA work.

Exact test:
`triad_nn_add_half_screen::triad_nn_d768_out_add_half_n96_discovery_once7`.
Environment: `MAMBA_TRIAD_NN_N96_DISCOVERY=1`, CUDA13.2 on quiet RTX6000 Ada.
Four strata: eager/graph x ABBA/BAAB; seven timed brackets each, raw four
observations retained. Candidate/current bits use the existing quantized/tie/
exception fixtures; a full-mantissa promotion corpus is still required.

## New measured result: ADVANCE on CUDA13.2

The corrected test passes (one exact GPU test, 7.65 seconds). Target candidate,
current-wide and actual public AUTO all produce identical bits; all pairwise
mismatch counts are zero. Tail, exceptional and null-A/B K0 checks also pass.
Resources: 124 registers, zero local/static shared bytes, 86,016 dynamic shared
bytes, 256 threads, one resident CTA. Private cache bytes remain stable.

| Measurement | candidate/AUTO p50 | candidate/AUTO p95 |
| --- | ---: | ---: |
| Eager ABBA | .819190 | .831169 |
| Eager BAAB | .819355 | .825806 |
| Graph ABBA | .814503 | .820913 |
| Graph BAAB | .814103 | .825806 |

Thus median elapsed time is **18.1–18.6% lower** than current public AUTO in
this cell: about65 microseconds versus79–80. This is NOT a cuBLAS Fast result,
not an all-shape/all-toolkit claim, and not production promotion. Next bounded
confirmation needs full-mantissa inputs, once21, an explicit Fast denominator,
and lower-toolkit compilation/bit/resource checks before admission.

Raw: `evidence/public-auto-repair/once7-cuda132/test.log`, SHA
`8e44c91e46580c590f505feeb0c7e32ac081601e0e3adb8fc4a4bf18bb66a7e2`.
Binary SHA `89c5514a39c9389e34c6763c9474d423f7e60454a33c93d785369184fc06f2f0`.
The command, source manifest, build receipt and PRE/RELEASE/DRAIN snapshots are
preserved alongside the log. RELEASE is busy immediately after process exit;
separate DRAIN obtains five quiet samples. No competing workload was stopped.
Root independently replays all28 brackets/112 timed observations via
`../ada-triad-live-reuse-20260908/replay.py`.
