# Ada Triad: actual Fixed-holder reuse, 2026-09-08

Discovery, not completed AUTO promotion or a full Triad matrix. Existing Fixed
CUDA/module bytes and RTX5090 routes are unchanged. The next integration reuses
the loaded Fixed holder with an honest Fixed physical artifact identity.

Qualification caveat discovered during integration review: the original NN
source191a keeps its physical-qualification holder alive while running the
other timing arms on that context. This violates the holder's documented
exclusive-use contract. The raw observations below are preserved, not silently
rewritten, but they are provisional screening evidence, not final admission
proof. The repaired wrapper527d1322 first performs isolated qualified checks,
drops the holder, then captures and times the true public F32 AUTO entry point
on separately owned, identically seeded buffers. Its graph is checked against
the qualified symbol/config and exact pointer/scalar ABI. Native9 tests and
independent source review pass; integrated GPU confirmation remains pending.

## F32 NN: actual loaded CopyPlan versus public AUTO and cuBLAS Fast

Three exact shapes are `(M,K,N)`: d768-in `(2048,768,3072)`, d768-out
`(2048,1536,768)`, Prism-in `(4621,384,1928)`. Strict alpha bit+1, beta bit+0,
no bias. Other epilogues are not admitted: the earlier nontrivial alpha/beta
CopyPlan discovery failed bits, preserved in the previous broad-screen report.

Each cell has eager/graph × ABBA/BAAB, eight warmup brackets then seven timed
brackets per stratum, four one-GEMM observations per bracket. Inputs and output
are reseeded outside the event, then guarded downloads/immutability checks
precede the next reset. Actual AUTO is independently qualified and uses the
same full-mantissa input words, not a substitute generic benchmark function.

Numbers are candidate/reference elapsed-time ratios: less than one is faster.
The p50 range spans all four strata; p95 is their worst observed once7 quantile,
not an estimate from a large sample. Every AUTO comparison advances; every Fast
comparison loses. No retry was used to replace a losing observation.

| CUDA | NN cell | / actual AUTO p50 | worst p95 | / Fast p50 | worst p95 |
| --- | --- | ---: | ---: | ---: | ---: |
| 12.8 | d768-in | .6547–.6668 | .6690 | 2.4067–2.4369 | 2.4743 |
| 12.8 | d768-out | .4889–.4973 | .5010 | 2.0603–2.1029 | 2.1951 |
| 12.8 | Prism-in | .6240–.6305 | .6425 | 1.6429–1.8632 | 4.3229 |
| 13.0 | d768-in | .6597–.6707 | .6735 | 2.4141–2.4440 | 2.4683 |
| 13.0 | d768-out | .4825–.4928 | .5016 | 2.0894–2.1839 | 2.2229 |
| 13.0 | Prism-in | .6212–.6447 | .6521 | 1.8095–1.8608 | 1.8808 |
| 13.2 | d768-in | .7101–.7236 | .7256 | 2.4161–2.4439 | 2.4619 |
| 13.2 | d768-out | .5503–.5661 | .5671 | 2.1193–2.1742 | 2.2276 |
| 13.2 | Prism-in | .6508–.6819 | .6890 | 1.7881–1.8228 | 1.8980 |

Fast explicitly uses `CUBLAS_COMPUTE_32F_FAST_TF32`, `CUBLAS_GEMM_DEFAULT`,
F32 storage, host scalar pointers, and the queried TF32 tensor-op math mode.
It is a performance target, **not the raw-bit exact-F32 oracle**. Candidate
and actual AUTO must match exact-F32 reference bits; Fast has its own repeated
eager/graph bit stability and guard checks, not equality to that stricter oracle.
Do not rename this denominator PEDANTIC or silently turn exact F32 into TF32.

Before target timing, the actual loaded CopyPlan also passes tail `(67,19,137)`
and K0 `(65,0,67)` with null A/B, repeat eager/graph and red zones. The measured
production holder is `gemm_bi_nn_fixed_sm89_f32_n64_copyplan_v1`, not the prior
source-composed proxy. Its actual compiler/artifact binding is emitted separately
from the independently qualified AUTO route. All three toolkit test executions
PASS. Private cache bytes remain stable; separate post-run DRAIN is quiet.
No claim of a cold-cache run or a new RTX5090 measurement.

Frozen NN test SHA `191a27f82577f735f4187cbba0d7855a168521820d356802f19501392e5ce24d`;
Cargo SHA `2e1b5f65c9a2822e6f5a8c52828d295c38d3a49dfe784233fcf6868563a1eb39`.
Exact test: `cuda_qualification::ada_live_fixed_copyplan_vs_auto_and_fast_three_cell_once7`.
The first build failed before GPU because `serde_json` was optional under hf
and an NVRTC tuple helper used u32 rather than i32. A dev dependency (not hf)
and tuple correction repair only the test build; initial failure is retained.

Raw logs, commands, pre/release/drain snapshots and binary/source receipts:

- `../ada-scalar-nn-live-fixed-screen-20260908/evidence/repair1/` (13.2).
- `../ada-scalar-nn-live-fixed-screen-20260908/evidence/cuda128/` (12.8).
- `../ada-scalar-nn-live-fixed-screen-20260908/evidence/cuda130/` (13.0).

## Whole NT pipeline on all three toolkits; half S3 on CUDA13.2

| Cell / candidate | Comparator | p50 ratio | worst p95 | Decision |
| --- | --- | ---: | ---: | --- |
| F32 NT d768-out, transpose16 + live Fixed CopyPlan, 12.8 | actual AUTO | .4509–.4523 | .4606 | Advance |
| F32 NT d768-out, transpose16 + live Fixed CopyPlan, 13.0 | actual AUTO | .4508–.4542 | .4632 | Advance |
| F32 NT d768-out, transpose16 + live Fixed CopyPlan, 13.2 | actual AUTO | .5365–.5383 | .5466 | Advance |
| F16 NN d768-in, live Fixed S3 | forced TC128 | .8944–.9063 | .9275 | Advance |
| BF16 NN d768-in, live Fixed S3 | forced TC128 | .8806–.9101 | 1.0374 | Stop |

NT times both transpose and GEMM, not GEMM alone. Its two graph nodes carry
TriadScalar and Fixed identities respectively. Eight exact-bit/guard/repeat
records and three resource records pass. Source3167072a/binaryb441380a;
`../ada-scalar-nt-fixed-copyplan-screen-20260908/` preserves full frozen source.
This supersedes the weaker transpose16+standard-M64 discovery for this cell,
but neither variant is an already-promoted AUTO route at this checkpoint.

Lower-toolkit NT source005da52e also passes all three resource/eight bit/four
timing strata. Local paths `evidence/cuda128/repair2/once7/` and
`evidence/cuda130/repair1/once7/` hold the final raw. Initial80ea omitted a
cfg-module import and failed build before GPU; d755 corrected that import but
then stopped at a reference-only resource assumption before bits/timing.
The test-composed generic reference reports local16 bytes on12.8/13.0 versus
local0 on13.2. Final005da records and requires exactly those observed reference
values; all candidate/transpose gates still require local0. This reference is
not the actual AUTO holder, which is independently qualified and timed through
the public wrapper. No arithmetic change, retry of a valid timing loss, or
candidate spill allowance. Final helper native5PASS; initial failures preserved.

Half has sixteen bit records and four resource records, with actual loaded
Fixed S3 F16/BF16 holders and their five-argument/32-byte ABI. Both bit checks
pass. F16 advances; BF16 eager/BAAB p95 loses, so no BF16 promotion or retry.
The mixed-result test intentionally exits101 solely for that speed decision.
This half comparator is forced TC128, **not a fresh AUTO/Fast comparison**.
Full tails/exceptional-value qualification and lower-toolkit finalists remain.
Sources431ef535/helper759a1bbd, binary9163b8a0;
`../ada-half-nn-fixed-s3-screen-20260908/` includes raw and frozen sources.

## Independent replay and next batch

Run `python3 internal/perf/ada-triad-live-reuse-20260908/replay.py` from the
worktree. Root independently recomputes **644 brackets / 2576 observations**,
including order-sensitive arm selection, all p50/p95 values, and decisions.
The raw SHA256s are printed. NN/half ABBA starts with the candidate; NT ABBA
starts with AUTO. The replay explicitly distinguishes these encodings.

Integrate the three NN winners with separate backend/plan/private revision,
actual Fixed compiler/artifact, strict operand/shape/environment guards and
old-route fallback. Preserve global tuning45 and all old numeric/schedule ABI.
Run one assembled actual-AUTO/bit/Fast batch on all three toolkits after focused
selector/identity tests. NT lower-toolkit screening is complete. TF32 N96 research
preserves current Triad AddHalfUlp conversion, including NaN payload behavior;
RNA Fixed N96 is not silently interchangeable with that current contract.
No production kernel or losing experimental source was deleted here.
