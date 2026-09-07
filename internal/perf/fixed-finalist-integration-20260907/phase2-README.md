# Fixed AUTO45 assembly — five measured replacements qualified

Phase1 source/evidence is frozen in commit `0a95a00f`; its README, original
receipts and manifest are immutable historical evidence. This second phase
changes host routing only. Numerical ABI5, schedule8 and all CUDA bodies stay
unchanged. No Triad optimization is part of this change.

## Exact intended AUTO replacements

All rows require CC8.9/142SM, a known qualified matching NVRTC library,
the independent live holder, exact shape, no bias and aligned operands.

| Mode / `(M,K,N)` | CUDA | New AUTO | Old forced control |
| --- | --- | --- | --- |
| TF32 `(2048,2304,768)` | 12.8,13.0,13.2 | `Tf32RnaM128N96S3` | `Tf32RnaM128N128S3` |
| F16 `(2048,768,2304)` | 13.2 | `TcM64N64Sm89S3` | `Tc128Sm89Swizzle` |
| F16 `(2048,2304,768)` | 13.2 | `TcM128N64Sm89S2` | `Tc128Sm89Pipeline` |

CUDA12.8/13.0 half D/E keep old AUTO because the new candidates did not pass
their own timing thresholds there. Other dtypes, bias modes, shapes, toolkits,
devices and failed-holder fallbacks retain their previous selection. This is
five literal rows using three new kernels, not five independent implementations.

## Observed test-first checkpoint

`phase2-red-cuda132` records successful old-AUTO44 compilation followed by
two expected behavioral failures: the half selector returned old Swizzle
instead of D N64/S3, and actual TF32 E0 returned old RNA N128 instead of N96.
The initial shorthand half test filter matched zero tests and is explicitly
invalid; the corrected full path ran one test and failed as expected. Initial
busy release is preserved; `phase2-red-cuda132-drain` records later quiet GPU
release without rerunning the numerical test.

## Final closure protocol

All three matching selected-binary builds succeeded. The first CUDA12.8 full
library run then found two stale phase1 inventory expectations (650 passed,
2 failed,46 ignored), before any GPU test. The repair adds the two new Ada
paths to the complete expected inventory, its CC12 exclusion filter and the
expected old-N64 suffix: six lines, all inside `cfg(test)`. Root and independent
review verified that production composition and CUDA are unchanged. Final
test-only module hash is
`2aab1470061486dbb5df1958133a119cacaf7d78012223dbf8fcc433b14c006f`.
Rebuild/rerun the affected library tests on all three toolkits; retain the
original integration binaries with explicit before/after source provenance.
Do not claim those integration binaries were rebuilt from the repaired test
module or rerun unchanged CUDA qualification just for this test-only edit.

After the combined source freeze, rebuild each matching toolkit once. Check
the selector and graph-recapture truth tables, current host tests, affected
actual-AUTO storage bits/prefixes/views/graphs and retained module identities.
Reuse unchanged phase1 forced resource/numerical qualification.

Run the existing `fixed_ada_forced_rungs_paired_precision_cublas` once with
101 windows on each of the five rows, forcing the **old** route while public
AUTO selects the new one. Both eager/graph and A-F-V/V-F-A orders are required.
The protocol is paired, not ABBA. Compute AUTO/old and AUTO/Fast from each raw
triplet before quantiles; do not invert published old/AUTO quantiles.
Actual captured new AUTO symbol, geometry and five-argument ABI are mandatory.
Own acceptance requires worst-stratum p50<=0.985 and p95<=1.0 plus bit gates;
a valid loss is retained and not retried. Timed Fast remains native-half
GemmEx for F16 and explicit FAST_TF32 for TF32, never PEDANTIC.

All five once101 post-AUTO runs pass. Root and independent review replay the
raw evidence separately; the table below is derived from per-window ratios,
not inverted forced/AUTO quantiles. Every cell has four strata,101 triplets
per stratum, zero rejected records and passing storage/repeat/graph bits.
Across the batch this is20 records and2020 timing triplets. Each number is
the worst of its four strata; smaller is better,1.0 is parity.

| CUDA / new AUTO cell | AUTO/old p50 | AUTO/old p95 | AUTO/Fast p50 | AUTO/Fast p95 |
| --- | ---: | ---: | ---: | ---: |
|12.8 TF32 E0 N96|0.773862106|0.774189644|1.108350442|1.108773655|
|13.0 TF32 E0 N96|0.773759020|0.774043641|1.108399746|1.108750071|
|13.2 TF32 E0 N96|0.852934841|0.853256795|1.114448163|1.114884342|
|13.2 F16 D0 N64/S3|0.912581056|0.952713753|1.044123987|1.068794161|
|13.2 F16 E0 N64/S2|0.970985030|0.979662588|0.969903644|0.984771828|

Raw directories: `phase2-post101-cuda{128,130,132}-n96`,
`phase2-post101-cuda132-d` and `phase2-post101-cuda132-e`.
All five improve the replaced route at both required quantiles. Only F16 E0
beats Fast in all four strata; TF32 E0 and F16 D0 retain the explicit losses
above. This is not universal superiority over Fast or complete inference
closure. ExactF32/BF16 and other TF32/F16 cells did not receive new timing in
this batch; their prior evidence and remaining gaps remain separate.
Historical E Fast results used different pointer alignment/window topology;
do not present their ratio reversal as a controlled before/after improvement.
No old production route is deletion-ready from these five replacements.

## Actual-AUTO functional evidence and provenance

`phase2-functional2-cuda{128,130,132}` each passes four real checks: the
literal performance oracle, live holders/resources, N96 actual-AUTO corpus
and half actual-AUTO corpus. N96 has224 unique groups per toolkit (672 total).
CUDA13.2 additionally passes both retained Triad cohort/bias GPU checks.
All three repaired library suites pass652 tests, fail0, ignore46; the two
composition tests and the correctly targeted current-revision test pass by
their full names. These are focused checks plus the affected library suite,
not a new full hardware/architecture/precision certification.

Root independently replayed all14 current source hashes, complete build-input
maps, binary/receipt bindings, all672 N96 groups, and all nine unchanged
phase1 compiled cache hashes. The only local/remote build-input exception is
the separately preserved, unbuilt Triad discovery WIP test and its untracked
helper; neither is an input to these selected Fixed/cohort binaries. The
cfg(test)-only module repair is bound explicitly as before/after source;
production prefix bytes remain identical to phase1. Do not mislabel retained
integration binaries as rebuilt from the repaired test module.

Every timed run has a quiet PRE. Busy immediate releases are preserved; each
separate `-drain` receipt records0%/0% and no compute applications. No valid
workload result was retried. The same distinction applies to functional and
RED receipts; a later quiet drain is not a rewrite of an earlier busy release.

The extra `cohort-revision-host` command used a library test name against an
integration binary and ran zero tests. Its exit0 is **not a passing test** and
is excluded on all three toolkits, with explicit invalid metadata. The
correctly targeted full library suite is the relevant epoch test evidence;
no GPU work was rerun to repair this command metadata.

## Handoff

The three finalist implementations are assembled and their five qualified
AUTO45 rows are ready to retain. Phase1's four losing lower-toolkit half
rows keep their faster old AUTO. No source is deletion-ready: old RNA N128,
Swizzle/Pipeline and generic rungs remain reachable for other rows and
independent-holder fallbacks. Rejected discovery variants stay outside
production and are tracked in the retirement ledger. No new sweep is required
to finish this batch; next work is the bounded Triad profile, with numerical
family and fixed-association constraints explicit. Ada evidence does not
constitute new RTX5090 or other-architecture performance qualification.
