# Ada Triad broad discovery checkpoint

GPU: RTX 6000 Ada, CC8.9 / 142 SMs. CUDA13.2 discovery only. Fixed inference
and existing RTX5090 production routes are unchanged. Ratios below are
candidate/reference; lower is better. None is a new cuBLAS Fast comparison.

| Precision / operation / cell | Candidate | Reference | Paired p50 range, eager + graph | Decision |
|---|---|---|---:|---|
| TF32 TN d768-in | Dense-stage M128N64/S3 | Actual AUTO M64N64/S2 | .9501–.9618 | Retain for qualification |
| TF32 TN d768-out | Dense-stage M128N64/S3 | Actual AUTO M64N64/S2 | .9834–1.0094 | Stop: graph loses |
| TF32 TN Prism | Dense-stage M128N64/S3 | Actual AUTO M128N64/S3 | 1.0091–1.0360 | Stop |
| F32 NN d768-in | Existing scalar M64 | Generic exact NN | 1.1820–1.1837 | Stop |
| F32 NN d768-out | Existing scalar M64 | Generic exact NN | .9390–.9536 | Passes screen; superseded by CopyPlan screen below |
| F32 NN Prism | Existing scalar M64 | Generic exact NN | 1.0541–1.0573 | Stop |
| F32 NN d768-in | Unchanged Fixed CopyPlan | Generic exact NN | .7373–.7405 | Retain for qualification |
| F32 NN d768-out | Unchanged Fixed CopyPlan | Generic exact NN | .5598–.5670 | Retain for qualification |
| F32 NN Prism | Unchanged Fixed CopyPlan | Generic exact NN | .6709–.6811 | Retain for qualification |
| F32 NT d768-out | Existing transpose16 + scalar M64 | Actual public AUTO NT | .8625–.8715 | Retain whole two-node pipeline |
| F16 NN d768-in | Existing TC64 | Forced TC128 | 1.4529–1.4962 | Stop |
| BF16 NN d768-in | Existing TC64 | Forced TC128 | 1.4737–1.4997 | Stop |

Cross-run ranking is discovery, not a paired candidate-versus-candidate
admission. The Fixed CopyPlan reuse is strictly alpha bit-pattern +1,
beta bit-pattern +0, no bias; all other epilogues keep their old route.
Its nontrivial alpha/beta discovery tail failed exact bits, so this is not
a universal Triad epilogue replacement. The restricted run passed its tail,
K0 with null A/B, target repeat, graph, guard, and immutable-input checks.
Resources: 135 registers, zero local memory, 32,768 static shared bytes,
zero dynamic shared bytes, 128 threads, three resident CTAs/SM.

Scalar M64 uses 120 registers, zero local/static memory, 17,408 dynamic
shared bytes, 128 threads and four resident CTAs/SM. NT timing includes the
full transpose16 and inner GEMM; it is not an inner-GEMM-only speed claim.
Its reference is the unchanged public AUTO wrapper, physically bound to
`gemm_bi_nt`, not a guessed raw kernel. Both produce identical bits in
two eager and two graph executions. Transpose resources are 18 registers,
4,224 static shared bytes, 512 threads, three resident CTAs/SM, zero local.

## Evidence and failures preserved

- Dense TF32: `../ada-triad-tn-dense-batch-20260908/report.md`, committed
  as `30df4596` with its raw and independent replay.
- Standard scalar NN: `../ada-scalar-nn-m64-screen-20260908/evidence/`.
  Initial source `71dcd709` / binary `134225c3` produced valid d768-in/out
  rows, then failed the Prism guard because the harness chose a specialized
  backward-NT inner symbol for a forward-NN tail. Prism-only repair source
  `6a7354d8` / binary `9ce31937` passed correctness but lost timing. These
  completed d768 rows were not rerun. The original batch is NOT a passing test.
- CopyPlan: source `6473d602` failed unsupported alpha=-.75/beta=.5 tail
  bits before target timing; no speed verdict came from that failed run.
  Source `6729c227` restricts the epilogue before kernel loading and records
  expected/actual bit words on a future mismatch. Successful binary
  `7bd0f02e716ee127a021dea0e1da7d1b42ea1e919faef6d026912ffc4ec8a97f`.
  A prior checked-integer-type build error is separate from GPU correctness.
  Successful raw and source snapshot:
  `../ada-scalar-nn-fixed-copyplan-overwrite-screen-20260908/`.
- NT: `../ada-scalar-nt-d768-out-screen-20260908/evidence/` preserves the
  initial wrong target gate (`compute_89` versus actual `sm_89`), then the
  graph-prepared-cache failure. Qualification allocated different operand
  pointers from the benchmark fixture. Source `a1f20ca1` warms the exact
  fixture through public AUTO immediately before capturing it. No cache
  gate or public wrapper was bypassed. Successful binary
  `1c2ce6c3a9bec3c1a177b674541c936344207699a07f5304808f6861fd1d89a2`;
  frozen source and raw in `evidence/graph-warmup-repair/`.

`python3 internal/perf/ada-triad-broad-screen-20260908/replay.py` independently
recomputes all standard-M64 NN, CopyPlan NN and NT brackets, quantiles, and decisions:
196 brackets / 784 observations. It explicitly expects the first NN batch
to have failed after its two completed cells. It does not convert that
failure into a whole-test PASS.

Half TC64 screen: `../ada-half-nn-tc64-screen-20260908/evidence/`; source
`982a1b43`, helper `157ef344`, binary `ef703c17`. Both dtypes pass their
resource and 16 total bit/repeat/graph records, but all eight timing strata
lose. The libtest deliberately returns failure for the negative speed verdict;
this is a valid STOP, not a whole-test PASS. No retry or promotion.
TC64:128 registers,36,864 static shared,128 threads,occupancy2. TC128:162
registers,71,680 dynamic shared,256 threads,occupancy1; both zero local bytes.
NCU diagnostics and raw CSV are in `../ada-triad-half-nn-ncu-20260908/`.
The counter run is not timing-admission evidence: ptxas already interleaves
fragment loads, and shared wavefronts have no excess. Therefore merely adding
source-level fragment ping-pong or shrinking the tile is not the next candidate.
Next is the complete already-existing Fixed S3 pipeline package, not another
TC64 resweep.

## Next batch and cleanup boundaries

Qualify retained routes against actual AUTO and cuBLAS Fast on each installed
CUDA12.8/13.0/13.2, then check the admitted public AUTO path. Do not repeat
losing once7 screens or run the full matrix for each prototype. Source or
artifact changes must not silently invalidate existing portable/NT/Fixed
admissions. Keep the original modules and RTX5090 selectors intact.

All discovery code remains under `tests/`; no losing prototype was inserted
into production. Compact4/8 TN, losing dense cells, and standard-M64 NN
losses are cleanup candidates after release assembly, not globally bad
kernels on unmeasured GPUs. Raw evidence and decisions remain the audit log.
