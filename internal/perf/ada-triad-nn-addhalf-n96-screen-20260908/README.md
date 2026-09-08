# New TF32 Triad NN N96 discovery, Ada, 2026-09-08

## Latest: two previously unmeasured shapes, aligned timing

Final main `460a2c1f7451b33f20d77d05feb1bb89a6398e73de35c9ae93822e4c43d52fda`,
helper `086c0520ead684eb0aab86a83d2660beb65d00a6066ffa696600301768a88104`.
Logical pointers are 256B aligned (64 F32 guard words). Once7, 20 GEMMs per
observation, paired eager/graph x ABBA/BAAB; actual AUTO is qualified per cell.

| Cell | Candidate/AUTO p50 | Candidate/Fast p50 | Result |
| --- | ---: | ---: | --- |
| d768-in | 1.0567–1.0589 | 1.2780–1.2829 | Valid loss; stop |
| Prism | .9832–.9846 | .9939–1.0140 | Small AUTO improvement, not a Fast win |

Prism AUTO p95 is .9838–.9873, below .99 in every stratum. Fast eager p95
reaches1.0142 and graph p50 is1.0133–1.0140: near parity, not a champion.
Both targets pass exact candidate/current-wide/public-AUTO bits, guards and
eager/graph repeats. Resources unchanged:256 threads/124 registers/local0/
static0/dynamic86016/occupancy1. Root replayed all112 brackets/448 observations
and quantiles; cache stable and no competing compute apps in PRE/release/drain.

Exact tests under `triad_nn_add_half_screen::`:

- `triad_nn_d768_in_add_half_n96_fast_gap_once7`: complete valid samples and
  STOP decision; exits101 at the expected retain assertion, not a bit failure.
- `triad_nn_prism_add_half_n96_fast_gap_once7`: one test PASS, retain against AUTO.

[d768-in raw](evidence/zero-grid-460a/d768-in-once7/test.log), SHA256
`e1056d935e556b140159ae9b4468f679d6eb1173206bf74e1a0814d27531ab5b`.
[Prism raw](evidence/zero-grid-460a/prism-once7/test.log), SHA256
`d488da2789a3dce4bdbcddee87a8148c53e921fe82ac53636c45eed1fb0ed7a5`.
Binary `4bde257394b262ea2250ad3f714855f9ca469591c0ad712a82d5a23a37d1731b`.
Measured sources are archived with the evidence; no production/toolkit admission.

Pre-timing harness failures are preserved, not kernel timing losses:
`two-cell-final-cuda132` / `taildiag-8c717` compared forced AddHalf TF32 with
the exact-F32 fallback on tail(129,36,100). Candidate/current match, while
`gemm_bi_nn_narrow` differs at354 words. Repair separates forced-family equality
from fallback self-repeat; timed targets still require strict three-way equality.
`repair2-8d34` then passed tail/exception and all K0 bits but rejected the
legitimate linear zero-reduction grid51. Final repair pins its exact symbol,
numeric contract, block256 and ceil(M*N/256), leaving nonzero targets unchanged.
No unchanged valid measurement was rerun. The d768-out records below are older
and use their original 32-F32 guard/one-GEMM timing scope, not this aligned cohort.

## Follow-up: explicit cuBLAS Fast, short discovery only

The new CUDA13.2 once7 comparison uses literal `cublasGemmEx`,
`CUBLAS_COMPUTE_32F_FAST_TF32`, alpha=1/beta=0, and the actual public AUTO
entry point. Candidate/current/AUTO output words match; Fast passes its own
finite-output and eager/graph repeat checks. This does not assert equality
between our numeric contract and cuBLAS Fast.

| Measurement | candidate/Fast p50 | candidate/Fast p95 |
| --- | ---: | ---: |
| Eager ABBA | 1.000000 | 1.007874 |
| Eager BAAB | 1.007874 | 1.007874 |
| Graph ABBA | 1.007937 | 1.015873 |
| Graph BAAB | 1.007937 | 1.024000 |

New finding: near parity with Fast, **not a Fast win**. The earlier ~18% gain
over AUTO is not a new result. This is a candidate-screen result, not a
production admission or cross-toolkit claim. Per the active user instruction,
broader confirmation is deferred until the candidate batch is assembled.

Source SHAs: main `ce9839a655cb96864fb314a53355bda3751a211975c70cfc1f0c87dca5e14bd8`,
helper `d9098e555c7a0a447acbf4d5960c89ac4ce9fa72c27c0cf9ee69f7842df3645f`.
Raw and command receipt: `evidence/fast-gap-cuda132/once7/`.
Root independently recomputed all 56 brackets / 224 timed observations.
One exact GPU test passed; private cache unchanged; post-run drain quiet.

## Earlier discovery record

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
