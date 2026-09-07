# Task6C — Ada half S3 AUTO43 implementation and qualification report

Status: implementation and required Task6C qualification complete; independent
final evidence review and human commit remain with root.

## Scope and source identity

Base `4386e41a43f1b3b95baa829b1e7ab2b470eecf29`, branch
`codex/gemm-bi-triad-sm80`. The initial four-file freeze is recorded below;
the final authorized seven-file Rust freeze, including fix1's current-revision
consumers, is recorded later. No CUDA fragment, loader, composition, compiler
option, ABI, numeric contract, schedule or old kernel was changed.

Frozen local and remote SHA-256 values:

- `src/mamba_ssm/gpu/gemm_bi_fixed.rs` `21127b055f53c6a8fb3968083dbdd043bac39eb01ddd26c978c37be9c724e477`
- `src/mamba_ssm/gpu/kernel_identity.rs` `d3b821bab65bcdfb09cfdebf9a6be5e6260c611b244bf24754c47d43efcf966b`
- `tests/gemm_bi_fixed_sm89_pipeline.rs` `51400bb8d9e2d6f357381b14bb02b0f133b3c1720a26b457fe16390c4e1a485c`
- `tests/gemm_bi_fixed_performance.rs` `3888a55389044274bd74e9feb14c69133ca8ae4404889ecdfaad65644ec291f1`

The preserved initial four-file remote freeze is
`/root/mamba-ada-half-s3-auto-final-20260907`; remote evidence is
`/root/evidence-ada-half-s3-auto-20260907`; isolated targets and private caches
are `/root/target-ada-half-s3-auto-final-cuda{128,130,132}-20260907` and
`/root/mamba-kcache-ada-half-s3-auto-final-cuda{128,130,132}-20260907`.
Development RED source/target/cache remain separately preserved without the
`-final-` component.

## Ada lane and TDD

Fresh live PRE at `2026-09-07T03:38:19Z`: exact UUID
`GPU-d1edd7be-e88d-aed6-047d-622163306f0e`, RTX6000 Ada, CC8.9,
driver595.45.04, GPU0%, memory0%, no compute applications, SSH exit0.

Matching CUDA13.2 RED1 returned exit101 because literal BF16 B0/no-bias got
`Some(Tc128Sm89Swizzle)` instead of `Some(Tc128Sm89S3)`. RED2 returned exit101
because the captured42 rejection fixture still observed current42 instead of43.
Exact commands and outputs are retained in
`internal/perf/ada-half-s3-auto-20260907/host-red-green.log`.

After the minimal change, the selector suite passed3/3, including the literal
60-cell oracle across all8 independent Pipeline/Swizzle/S3 availability states
(480 states total); captured42 rejection passed1/1. The shared pre42/post43
harness host suite passed5/5. The Python validator suite passed7/7 after one
test-helper correction (`[]` had incorrectly selected its default dtype list);
the failed helper run is preserved in task history and was not runtime data.

## Implemented behavior

After the unchanged common eligibility guard, only CUDA13.2 homogeneous BF16
and F16 `(4621,768,2304)` no-bias requests select independently available
`Tc128Sm89S3`. With S3 absent, the old Pipeline/Swizzle table runs unchanged,
including the CUDA13.2 Pipeline-unavailable asymmetry. Public `fixed_forward`
passes the live S3 holder, calls the existing `launch_sm89_half_s3`, and returns
the selected S3 enum. The tuning revision is43; numeric/schedule identities are
still5/8. Captured42 and older identities reject while current43 accepts.

The actual hot-cell oracle names exactly the two CUDA13.2 B0/no-bias cells.
Forced Pipeline, Swizzle and S3 routes remain separate.

The Task6B timing mechanics now take an explicit stage policy. Historical
pre42 schema/record metadata remain unchanged and invocation fails closed on
current43. New ignored entry `fixed_ada_half_s3_post_auto_fast_paired` uses
schema `MambaBiFixedAdaS3PostAutoPairedV1`, stage `post_auto`, and exact arms
Swizzle/actualAUTO/Fast with directions AUTO/Swizzle, Swizzle/Fast and
AUTO/Fast. It allows only CUDA13.2, windows1/101 and both dtypes. M1 stale and
cross-stage controls are rejected. M2 saved A/B bytes are compared after every
pre-timing correctness block and before warmup; a focused negative mutates each
input independently. Existing post timing immutability, guard, poison, no-op,
repeat, graph, captured-argument and PEDANTIC-reference gates remain.

The new analyzer explicitly adapts the post schema/labels to Task6B's frozen
chronology/physical validator after checking stage43 semantics. Task6A's
compiled-artifact qualification retains historical tuning42 while the live
routing record requires43. Host negatives cover wrong stage/revision/AUTO arm/
direction, windows21/dtype subsets, stale controls, genuine own loss/mixed p95,
malformed captured arguments/symbols/poison closure and all exit layers.

## Interim checkpoint (subsequently completed)

Completed after this interim text: all-three matching-toolkit full/nonignored
and live forced/AUTO functional columns, the CUDA13.2 one-window smoke and
single fresh101 confirmation, final artifact comparison, rooted manifest, and
quiet lane release. Their final results are recorded below.

## Source fix1 — global epoch consumers

The first frozen CUDA12.8 full-library command was:

`env CUDA_HOME=/usr/local/cuda-12.8 CUDA_PATH=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:... LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64 CARGO_TARGET_DIR=/root/target-ada-half-s3-auto-final-cuda128-20260907 MAMBA_RS_KERNEL_CACHE=/root/mamba-kcache-ada-half-s3-auto-final-cuda128-20260907 cargo test --release --features cuda,cudarc/cuda-12080 --lib -- --test-threads=1`

It returned exit101:642passed/3failed/46ignored. That first failure exists in
the exact tool transcript but was not redirected to a standalone raw file; all
corrected library logs below are standalone unedited raw artifacts. The exact
failure was three tests exposing four live/current assertion sites outside the
original ownership still expecting42: the
global/F32 alias pair in `dispatch.rs` and one resolved identity assertion in
each of two `launch.rs` tests. Root's bounded ruling authorized those four
assertion changes plus the same current assertion in
`tests/gemm_bi_tf32_cohort_binding.rs`. No production Triad logic changed.
The current direct-pair oracle in `gemm_bi_fixed_performance.rs` was also named
v42 and expected Swizzle for the two promoted cells; it is now v43 and expects
S3 only for CUDA13.2 BF16/F16 B0/no-bias. Its forced Pipeline/Swizzle controls
and schema are unchanged.

Corrected CUDA12.8 full library at fix1 returned exit0:645passed/0failed/
46ignored. The focused current direct-pair oracle returned exit0:1passed/
117filtered. Python analyzer/wrapper adversarial tests now return exit0:8 test
groups, including exact eight-record same-attempt closure and mutated command,
test/wrapper/SSH exits.

Final fix1 local/remote source is
`/root/mamba-ada-half-s3-auto-fix1-20260907`; evidence, targets and caches use
the same `fix1` component. Original frozen source/bindings and the failed
library run remain separate. Fix1 SHA-256 additions/changes are:

- `dispatch.rs` `8341aa0f651f2cf1c7b07a174c3be3739a6591dec2e94feb8a4544e76e616505`
- `launch.rs` `cfed2608b390e7c1694b53edd54e113f396bc4e3db0d6c6a1cdd96d4a8af77fa`
- `gemm_bi_tf32_cohort_binding.rs` `d473f3ca0042c4b3b95650c5f231f0a43068c6625321c9735c51ebdedb0e8604`
- `gemm_bi_fixed_performance.rs` `7d61e9f1241c9b6b3e3018b48cc14d8c7ca29149c56a3a6fc3af2e6188dc40b2`

The unchanged three original implementation file hashes remain those listed
above. This is the second and final Rust source freeze before corrected all-
toolkit bindings and runtime measurement.

## Host fix2 — frozen dependency and inherited-gate closure

The reviewed Task6B analyzer and wrapper are now fail-closed inputs, pinned
before import/use to SHA-256
`0c1391f47ec0253720b66733e950b8f9ad67ba112027b81e2d67eda9a1e63f30`
and `ce0de6882fa45835ba81b90d02a6cc085ed32226e2462b72260ce9cacd0fd713`.
The post analyzer rejects a foreign bound wrapper digest. The runtime wrapper
checks both the current imported file and the binding pin before and after the
measured command. Existing bindings already contain the approved wrapper pin,
so no rebuild or relabel was performed.

The TDD RED command was
`env PYTHONDONTWRITEBYTECODE=1 python3 internal/perf/ada-half-s3-auto-20260907/test_validation.py`.
It returned exit1 with one expected failure and two expected errors: a foreign
bound digest was accepted and both pre-execution pinned loaders were absent.
The identical GREEN command returned exit0 with all11 groups passing. The new
coverage directly corrupts sample chronology, physical `repeat_bits`,
`poison_upload_verified`, and `guards`, plus nonfinite and over-tolerance
`numerical_error`, through the post adapter. Raw output is in
`fix2-host-red.log` and `fix2-host-green.log` beside the utilities.

Final fix2 hashes are:

- `analyze.py` `a18899f51be393fe72d5db55b2450e63f553168d89e3ed80daed5d013afb1a2e`
- `run.py` `2d0634fa26b219b4d6a55eb7099b8a075337e937255b4656d565f64f98124fea`
- `test_validation.py` `844c22c3a4c342d2c2c896044fcbd63004e34da4f9658582e44a26505be89302`

Their pre-fix2 copies are preserved under `fix2-pre/` with hashes
`9c73297e...`, `a4e322bc...`, and `dc3fb73c...` respectively. Fix2 changed no
Rust, CUDA, binary, binding, cache, or GPU timing artifact.

## Corrected all-toolkit functional qualification

One initial holder invocation omitted the wrapper's toolkit library environment,
so all three processes loaded NVRTC13.2. Those raw logs are explicitly retained
as `cuda{128,130,132}-pipeline-holders-unbound-env-invalid.log` and are not
counted. They used the wrong cache variable name and did not populate the bound
private caches. The corrected commands used the wrapper's exact `CUDA_HOME`,
`CUDA_PATH`, toolkit-first `PATH`, `LD_LIBRARY_PATH`, `CARGO_TARGET_DIR`, and
`MAMBA_RS_KERNEL_CACHE` constructor.

The corrected holder logs directly report matching NVRTC12.8,13.0,13.2. Each
passed1/1 and proves BF16/F16 Swizzle and S3 holders live with valid resource
bounds. On each toolkit, the actual-AUTO and forced-Swizzle and forced-S3 hot
A–E prefix/view/eager/graph bit tests each passed1/1, for nine exact ignored GPU
tests with exit0. The three nonignored integration targets passed2/2, and the
three cohort-binding targets compiled with the three unavailable cohort tests
properly ignored (0fail, exit0). The three corrected full libraries each passed
645 with 0fail and46ignored.

The additional CUDA13.2 direct-pair hot-B window1 smoke passed1/1, exit0, and
emitted exactly16 records: both dtypes, both bias states, eager+graph, and both
orders. Every raw/repeat/graph-where-applicable bit gate passed. Actual AUTO43
was S3 in exactly the two literal dtype/no-bias cells, represented by eight
path/order records, and remained Swizzle in the eight bias-control records. The
raw log SHA-256 is
`30c9697154a289e11b3cdff70760122823baaaefbd200ecf339f9375fc7a30ed`.
All corrected evidence is mirrored under
`internal/perf/ada-half-s3-auto-20260907/remote-fix1/`.

Cache provenance is captured in `cache-provenance-final-v2.log` (the earlier
malformed inventory is preserved but not authoritative). The invalid mixed-
environment loop set `MAMBA_KERNEL_CACHE`, which this source never reads; the
accepted variable is `MAMBA_RS_KERNEL_CACHE`. The default cache's newest file
predates this task (2026-08-27), so the invalid 2026-09-07 calls changed neither
it nor the bound caches. Each corrected, matching-toolkit holder run was the
cold population of its separate mode0700 cache: three files appeared after the
invalid log and before the corrected holder completed. The subsequent AUTO/
forced functional and direct-pair calls were the corresponding warm reads.
The authoritative inventory records every bound filename, size and timestamp,
plus invalid/corrected/functional log times for each toolkit.

## Host fix3 — exact returned direction identities

The first valid smoke analysis revealed a reporting-only alias bug: Task6B's
global summaries and per-cell constituents share the same dictionary objects,
so two post-label remap loops transformed comparison2 twice. Raw direction
validation, chronology, physical gates, ratios and admission arithmetic were
unaffected. The original analyzer/test and original smoke analysis are
preserved under `fix3-pre/`.

The unchanged host-suite command first returned exit1 with exactly one failure:
comparison2 was reported as `Swizzle/Fast` instead of `AUTO/Fast`. The minimal
fix tracks object identity and remaps each returned shared summary once. The
same command then passed all11 groups, exit0. The regression checks all24 global
summaries, all24 per-cell constituents, and verifies the raw input rows are not
mutated. Saved smoke evidence was revalidated without a GPU rerun: 8 configs,
96 samples,24 pairs,24 summaries, all three exact directions, analyzer exit0.

Final fix3 hashes are analyzer
`0a8c16dffa7fe7226fce975a252408fac6a751832ee032b9b731eeb17819e71a`
and test
`037b7a4a0f777f0206ff8cd03f07b388875ce31332432b53fb0d55878b74294b`;
the wrapper remains
`2d0634fa26b219b4d6a55eb7099b8a075337e937255b4656d565f64f98124fea`.
Raw RED/GREEN logs are `fix3-host-red.log` and `fix3-host-green.log`; the fixed
smoke report is `remote-fix1/cuda132-postsmoke1-analysis-fix3.json`.

## Post-AUTO43 timing result

The exact one-window smoke completed with all exit layers0 and validated
8configurations/96samples/24pairs/24summaries. The single fresh101 command then
completed once, with no retry: test/post/wrapper/outer-SSH exits were all0 and
the final analyzer accepted 8configurations/9696samples/2424pairs/24summaries.
Raw test SHA-256 is
`b6d0fca4305b6846a427c9dc8432e727dfa5434e68f380ca29511916a8c7d620`;
the exact eight-line SSH transcript is
`35f09354dea490b51ffa18d808f8c1c6584c5f68e941ba85e05017df0b4cf525`.

Full 101 constituents (`p50`, `p95`; lower is better):

|dtype|path|start|direction|p50|p95|
|---|---|---:|---|---:|---:|
|bf16|eager|0|AUTO/Swizzle|0.927458038371086|0.952269787119487|
|bf16|eager|0|Swizzle/Fast|1.30839258366045|1.3482905172724|
|bf16|eager|0|AUTO/Fast|1.20517320027239|1.22608022967833|
|bf16|eager|1|AUTO/Swizzle|0.927274015561574|0.953554088686561|
|bf16|eager|1|Swizzle/Fast|1.31192320914935|1.35345269982913|
|bf16|eager|1|AUTO/Fast|1.20283921061199|1.22893112218001|
|bf16|graph|0|AUTO/Swizzle|0.929889194840401|0.95370055537639|
|bf16|graph|0|Swizzle/Fast|1.31087693807664|1.36348638506306|
|bf16|graph|0|AUTO/Fast|1.20704637900001|1.23122321853941|
|bf16|graph|1|AUTO/Swizzle|0.929705192828039|0.946809960723235|
|bf16|graph|1|Swizzle/Fast|1.30967754066252|1.35996563038971|
|bf16|graph|1|AUTO/Fast|1.20120585960643|1.22764833849855|
|f16|eager|0|AUTO/Swizzle|0.924617647729043|0.945308702669699|
|f16|eager|0|Swizzle/Fast|1.241112946577|1.27375197087621|
|f16|eager|0|AUTO/Fast|1.12437271497145|1.15919252001431|
|f16|eager|1|AUTO/Swizzle|0.925311217620799|0.947876729324836|
|f16|eager|1|Swizzle/Fast|1.23844335770121|1.27168832708835|
|f16|eager|1|AUTO/Fast|1.12561527422198|1.15788766893408|
|f16|graph|0|AUTO/Swizzle|0.929018443238828|0.955993909109527|
|f16|graph|0|Swizzle/Fast|1.24749801961977|1.27613232320885|
|f16|graph|0|AUTO/Fast|1.13178454322126|1.15657761268089|
|f16|graph|1|AUTO/Swizzle|0.928527929638008|0.953288860485244|
|f16|graph|1|Swizzle/Fast|1.24751849737445|1.27460588057574|
|f16|graph|1|AUTO/Fast|1.13347333829252|1.15774986475749|

All eight AUTO/Swizzle strata have p50<1 and p95<1. Worst own values are
0.929889194840401/0.953700555376390 BF16 and
0.929018443238828/0.955993909109527 F16, so both literal cells qualify. Fast
still leads: worst AUTO/Fast p50/p95 are1.20704637900001/1.23122321853941
BF16 and1.13347333829252/1.15919252001431 F16. This is an own-kernel
improvement, not a vendor win or a claim about other shapes, bias, precisions,
architectures, Triad, or end-to-end inference.

Final six-cell selector matrix for B0/no-bias is therefore:

|CUDA|BF16|F16|
|---|---|---|
|12.8|Swizzle unchanged|Swizzle unchanged|
|13.0|Swizzle unchanged|Swizzle unchanged|
|13.2|S3 promoted|S3 promoted|

All bias cells and every other literal shape retain the prior selector/fallback
table. The historical pre42 entry was also invoked once against current43 and
failed cleanly before timing at the explicit 43-versus42 revision assertion,
exit101.

## Artifact, archive, and lane closure

All three corrected bindings contain356 compiler inputs and the final seven
authorized Rust changes. Binary SHA-256 values are
`48ccd2956deb650d65b79c547bcf1591b44e025c5738069ba8162c9d9f793ede`
(12.8), `99acb666d50cf79ed954c09e03a9490c0dfc037443eced5111b75832888d724b`
(13.0), and
`a6c29a2dc4bb6ab2f9fbde8a4ed2ae82625ae85e9471d7321af449305170eb91`
(13.2). Compiled Fixed source/composition/options and payload identities match
Task6A exactly on all three toolkits; tuning42 in the historical artifact
qualification is intentionally distinct from live routing43.

`source-final.tar.gz` preserves all356 bound inputs, SHA-256
`d90cc387433547f5d9af9fe8d15d99a0c89df6c1f86920897d561d529a9188c4`.
The per-toolkit binary+three-cache-envelope archives hash to
`2df2a1d098af0c8971f3854b7aa7fd776b6de56cdc457ccc32263df0eaa53475`,
`51c609d6346117ad04096050dc4e206cf30cc86d70388915326da552743551ba`,
and `e2de16e4788c206d1346ac00c32e223f7100dc08acf9b158432263d794a1ba73`.
`archive-verification.json` returns PASS: every archived/live source and binary
hash matches its binding, all nine cache envelopes match live bytes, and each
Fixed payload/invocation matches Task6A. An initial archive path construction
failed before a usable bundle; its partial artifact is retained with
`-invalid-path` and is not authoritative.

Rooted `internal/perf/ada-half-s3-auto-20260907/SHA256SUMS` covers the final
report and source/evidence/archive artifacts; `manifest-verification.log`
records a clean full replay and exit0 while remaining outside the manifest.

Independent root artifacts `root-build-input-checks.json`,
`root-recomputed-pairs.json`, and `root-artifact-checks.json` verify the
all3x356 inputs/seven-file diff, both smoke/101 raw arithmetic, all three
executables, all nine complete cache envelopes, exact Task6A Fixed payloads,
byte-identical Task6B Triad payloads, and saved-run cache hashes. The final
audited lane release at
`2026-09-07T04:56:10Z` records exact UUID/RTX6000 Ada/CC8.9, GPU0%, memory0%,
empty applications, both telemetry query exits0, release-check exit0, and outer
SSH exit0 in `final-lane-release-json.log` and its saved `release.json`.
