# Ada Triad NT compact-XOR discovery — STOP

CUDA13.2 / RTX6000 Ada, 2026-09-07. The candidate passes the bounded resource
and storage-bit checks, but loses to actual production AUTO in all four short
paired strata. **Do not promote or retry this unchanged candidate.** No
production CUDA, dispatcher, numerical ABI or tuning revision changed.

## Result

NT logical `(M,K,N)=(2048,768,3072)`, output2048x768, reduction3072,
alpha1/beta0/no bias. Compare the test-only compact-XOR M128N64/BK32/S3
symbol with actual AUTO `gemm_bi_nt_sm80_mma_tf32_v1_m128n64_bk32_s3`.
Smaller ratios favor the candidate; p50/p95 are quantiles of paired ratios,
not ratios of independently selected latency quantiles.

| Path/order | AUTO median us | Candidate median us | Candidate/AUTO p50 | p95 |
| --- | ---: | ---: | ---: | ---: |
| eager ABBA | 274.916450 | 284.529779 | 1.034956388 | 1.035071673 |
| eager BAAB | 274.822222 | 284.418662 | 1.034742585 | 1.035196714 |
| graph ABBA | 273.746663 | 284.302221 | 1.038559589 | 1.038886947 |
| graph BAAB | 273.654222 | 284.273783 | 1.038873304 | 1.039089310 |

Each stratum has7 windows,18 iterations per bracket leg,64 warmups and a
16-iteration pilot. All28 four-leg brackets are preserved. Retention required
both p50 and p95 below0.99 in all four strata; none passed. No cuBLAS timing,
21/101 qualification, full matrix or subsequent valid timing retry ran.

Live candidate resources:155 registers,0 local bytes,0 static shared bytes,
73,728 dynamic bytes,256 maximum threads,1 active CTA/SM. Shared storage shrank
from82,944 bytes but did not add residency or improve performance. The earlier
LDGSTS profile motivated a hypothesis; it did not prove that removing padding
would speed up the kernel. No candidate NCU capture was taken.

## What was checked

- Source transformation changes only NT shared row stride36->32 and applies
  `k ^ ((row_or_column & 7) << 2)` to both stores and fragment reads. Global
  copy ownership, RNA conversion, ascending MMA and NT epilogue are retained.
  Test symbols are separate; the reference comes from the production module.
- Native host layout/source tests5/5; independently compiled actual7-line
  CUDA address helper passes the host mapping model, and a wrong-XOR mutation
  fails. This is address-model evidence, not CUDA execution of that host test.
- Exact unbiased F32 qualification upload validates all active lengths/bias
  before upload, preserves red zones and raw words. Its extracted actual host
  unit test passes. The positive upload path also executes in the GPU test.
- Target and forced same-tile tail `(129,65,36)` use the existing finite
  full-mantissa corpus, including signed zeros, subnormals and TF32 ties.
  Each has two eager and two graph repetitions, reset outputs, immutable A/B
  and red-zone checks. Target reference is actual AUTO; tail is explicitly
  forced and is not a claim about the tail's AUTO choice.
- Candidate graph checks exact symbol, one node, launch geometry/shared
  bytes, five-argument offsets/sizes and rejects a sixth argument. It does
  not independently decode captured argument values. Nonzero bit tests
  exercise the constructed argument values.
- Final target graph replays re-seed both sides and recheck golden bits,
  inputs and guards before the quiet post gate. One executed ignored test
  passes in44.67s; this is not full NaN/Inf/batch-prefix/subview qualification.

## Provenance and failed attempts

Source base311e3435607bdb36c0c45c290c77e82f0439edb1 plus the three test/hook
files in this checkpoint. The369-entry `build-cuda132/source-manifest.json`
matches the local frozen inputs; both older split8 WIP files are excluded.
Build exits0 after42s, binary SHA
`6f083ae8be2d7910d31129473c7abd03bc6258c5fa77831d026000ecc4002a66`.
The initial wrapper failure before Cargo is preserved under `build-cuda132/`.

`once7-cuda132/` is **INVALID**: the wrong `cuda_runtime::` filter ran zero
tests, despite process exit0. Its raw log and original receipt remain intact.
`once7-cuda132-valid1/` uses the listed `cuda_suite::` name on the same binary;
its receipt requires one executed test plus1 resource,4 screen,1 decision
records. It is the only actual GPU discovery run.

Valid raw log SHA
`b50c7f9bdc44471ac4a45b066091b1a1d444f362dfa7e796e23a508379cdae56`;
composed candidate source SHA
`ad931c3109b84eb421c087a2ee65621c5e7afaa6089cac7156e3cf8b74e324bf`.
PRE15:52:17Z quiet; immediate RELEASE15:53:02Z8% GPU/1% memory is preserved;
separate5s DRAIN15:53:07Z is0/0 with no apps. Internal pre/timed/post quiet
gates also pass. Separate logical contexts share the same selected GPU.

The binding records three production cache artifacts and ldd-linked host
libraries. Candidate PTX was live-loaded, not persisted; the binding is not
a saved candidate-PTX hash or a complete inventory of dlopened CUDA libraries.
No cross-toolkit/device or production-admission claim is made.

Replay: `ruby internal/perf/ada-triad-nt-compact-xor-20260907/replay.rb`.
It rederives all28 brackets/quantiles and the STOP decision, checks raw/build
bindings, and distinguishes the invalid zero-test attempt. It does not run CUDA.

## Next bounded batch

Use the original padded36 layout for two independent mechanisms from Fixed:
precomputed copy addresses and ldmatrix fragment loads. Preserve the direct
numerical association; do not combine either with this losing XOR change.
Reuse this test harness rather than clone it, screen resources/bits/once7,
then qualify only surviving finalists. Current60-cell source/evidence audit
found no already-measured, compatible Ada champion omitted from AUTO; this is
a bounded audit, not proof about every historical experiment or architecture.
