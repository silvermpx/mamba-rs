# Task6C — measured Ada S3 AUTO43 and post-AUTO qualification

Read this first. This implements stage4/5 of the approved
`ada-half-s3-integration-plan.md`; consult
`ada-half-s3-auto-interface-audit.md` for source anchors. Start only after root
accepts/reviews and commits Task6B and grants the Ada lane. Root supplies the
actual base commit on dispatch. The measured winner matrix is
`internal/perf/ada-half-s3-paired-20260907/winner-matrix.json`.

## Ownership and unchanged contracts

Existing worktree `/Users/silvermpx/IdeaProjects/mamba-rs/internal/worktrees/gemm-bi-triad-sm80`,
branch `codex/gemm-bi-triad-sm80`; no new branch/stage/commit/push/deletion/
process signals/subagents. Root commits with human identity and noAI trailers.
Use apply_patch. No Mac cargo/CUDA: Mac cargo starts unrelated application.
Direct rustfmt is allowed; CUDA/Rust builds on isolated Ada remote sources,
per-toolkit targets and private0700 caches.

Own `src/mamba_ssm/gpu/gemm_bi_fixed.rs`,
`src/mamba_ssm/gpu/kernel_identity.rs`,
`tests/gemm_bi_fixed_sm89_pipeline.rs`,
`tests/gemm_bi_fixed_performance.rs`, and new evidence under
`internal/perf/ada-half-s3-auto-20260907/`. Write report next to this brief:
`ada-half-s3-auto-task6c-report.md`. Root owns handoff/ledger/archive selection.
Do not change CUDA fragments, loader/composition/compiler/ABI/resource rules,
numerical or schedule revisions. No kernel removal or unrelated tuning.

## Literal production change

Only homogeneous BF16 and homogeneous F16, B0 `(M,K,N)=(4621,768,2304)`,
no bias, known NVRTC13.2 on CC8.9/142SM with existing aligned/nonzero operand
guards, may prefer `FixedTile::Tc128Sm89S3`. Both dtypes on12.8/13.0 retain
the existing Swizzle choice: BF16 failedscreen21; F16 failedconfirm101 p95.
Do not infer wins for other toolkits/shapes/devices/bias values.

The exact owned selector is `fixed_select_sm89_half_auto_tile`. Extend its
independent availability arguments with `s3_available: bool`. After the
unchanged common eligibility guard and before the unchanged old preference/
fallback table, the new preference condition is:

```rust
if s3_available
    && nvrtc == (13, 2)
    && (shape.m, shape.k, shape.n) == (4621, 768, 2304)
    && operands.bias_ptr.is_none()
{
    return Some(FixedTile::Tc128Sm89S3);
}
```

The existing guard already requires homogeneous non-F32 operands, exactdevice,
known toolkit/library and alignment. Do not bypass it. When S3 is unavailable,
execute exactly the old preference/fallback table; in particular preserve the
13.2 Pipeline-unavailable asymmetry. S3 availability must not affect any
non-promoted cell. In public `fixed_forward`, pass the actual optional S3
holder and handle `Some(Tc128Sm89S3)` through existing `launch_sm89_half_s3`,
returning the selected enum. No surrogate/manual route in AUTO tests.

Change global `TUNING_TABLE_REVISION`42->43. Update only current-revision
fixtures and names, not historical captured revisions38/39/40/41 or unrelated
byte fixtures such as `[42;32]`. Add explicit captured42 rejection while
compiler/artifact/numeric/schedule identities stay unchanged. Current tuple
becomes `(5,43,8)`; actual AUTO hot-cell route oracle must name exactly the two
new winning cells. Keep oldS2/S3 forced tests independently meaningful.

## TDD and host selection qualification

- Add a literal failing test for both new B0/no-bias13.2 choices before
  implementing selector behavior; run it on matching remote13.2 and retain RED.
- Update the complete60 preferred-cell oracle (only two entries become S3).
  Test all60 cells across all8 independent Pipeline/Swizzle/S3 availability
  combinations, including old fallback behavior when S3 absent and unchanged
  choices for all58 nonpromoted cells when S3 present. Literal old preference
  expectations must be independent of the new selector implementation.
- Preserve and extend common negatives: wrongdevice/SMcount, unknownlibrary,
  unsupportedNVRTC, zeros/misalignment, dtype/mixedoutput, biastrue/nulloptional
  semantics as existing, and adjacentdimensions. Do not expandS3 to B1 or
  prefixes/views unless the public exactdims meet the existing literal rule.
- Add a meaningful old42 graph identity rejection test; keep current identity
  acceptance, earlier captured revision rejection and Fixed-only artifact
  replacement invalidation tests. Full matching library suite onall3toolkits.

## Post-AUTO measurement without historical relabeling

Reuse the Task6B tested timing/physical helpers without copying the entire
700-line protocol. Separate explicit stage/arm policy from shared measurement
mechanics. Keep historical pre-promotion entry/schema meanings intact: its
actualAUTO42 is Swizzle and it must fail if invoked againstwrongrevision.
Add ignored entry `fixed_ada_half_s3_post_auto_fast_paired` with schema
`MambaBiFixedAdaS3PostAutoPairedV1`; enable explicitly
`MAMBA_FIXED_ADA_S3_POST_PAIR=1`. Require actualAUTO43 S3 on13.2 only.

Shared arm indices may remain0baseline/1candidate/2Fast. For poststage use:

```text
arm0=Swizzle  -> public forced Tc128Sm89Swizzle (unchanged old control)
arm1=AUTO     -> actual public fixed_forward, require Tc128Sm89S3
arm2=Fast     -> same native-half cuBLAS contract as Task6B
comparison0 A=Swizzle B=AUTO -> AUTO/Swizzle
comparison1 A=Fast B=Swizzle -> Swizzle/Fast
comparison2 A=Fast B=AUTO    -> AUTO/Fast
```

Do not label forcedS3 as AUTO. Same B0,biasfalse,alpha1beta0,homogeneousBF16/F16,
eight dtype/path/startparity configs,128eagerwarmups/arm,20logicalops/observation,
mirrored ABBA/BAAB brackets and reversed comparisontraversal. Event-onlytimed
intervals, rawchronology, roundedquantile convention, exactphysicalsymbol/ABI/
capturedpointers/bundle, guardedC, separatePEDANTICnumericreference andnative
Fast queriedmodes all remain as Task6B. Do not invert p95. Oldrawrecords and
bound source/binary hashes remain unchanged historical evidence.

Carry Task6B deferred source-review findings M1/M2 into this change:
- Reject stale `MAMBA_FIXED_AUTO_VENDOR_ROW/CELL/BIAS` and
  `MAMBA_FIXED_HALF_TILE_CANDIDATE` as well as the existing legacycontrol
  families; stage-specific controls must not silently enable/mute the other
  experiment. Cover the control classifier/parser with host tests.
- Compare saved A/B bytes after each pre-timing correctness block and BEFORE
  warmup for each configuration, as well as existing postchecks. Include a
  focused negative proving changed input is rejected by this gate. Original
  final outputimmutability checks remain; no tolerance relaxation.

Factor stage parameters only as needed, avoid unrelated harness refactoring.
Explicitly reject wrongtoolkit/revision/schema/stage/actualAUTO return. Retain
unit tests for prehistorical and newpost schedules/configurations/directions.
New independent analyzer must validate all3 comparisons, exact8configclosure,
source/binary/production artifact/UUID and corrected same-attempt exact SSH
closure. Reuse tested Task6B hostvalidator pieces by explicit import/adapter
if practical; do not silently edit/reinterpret its frozen evidence schema.
Cover newstage badrevision, wrongAUTO/symbol, wrongdirection, stalecontrols,
genuine ownloss/mixedp95, malformedcapturedargs/poison/closure identities.

## Runtime and artifact closure

Exact Ada UUID `GPU-d1edd7be-e88d-aed6-047d-622163306f0e`, RTX6000Ada CC8.9/142SM.
Acquire lane only after previousowner releases; PRE0%GPU/0%memory/noapps,
POST sameidentity/noapps with residualutilallowed, finalreleasequiet/noapps.
No concurrent GPU workloads. Toolkits12.8/13.0/13.2 use matching features
`cuda,cudarc/cuda-12080`, `cuda,cudarc/cuda-13000`,
`cuda,cudarc/cuda-13020` and matching CUDA_HOME/PATH/LD_LIBRARY_PATH.

All3: full library, focused+nonignored performance tests; actualAUTO half
hotA-E/bothbias/prefix/view/exceptional/poisonedgraph bits and unchangedforced
half inventory. Root's Task6A CUDA qualification remains for byte-identical
kernels; verify all compiled source/composition/options identities are exact
matches, andcold/warmproductionholders/physicalargs agree. Explicitly retain
S3 forceavailability onall3toolkits though AUTO only13.2. No global13.2-only
test invoked on12.8/13.0 without respecting its existing versionprecondition.

13.2: newpost one-window smoke forbothdtypes/all8configs, then one fresh101
postAUTO confirmation, not a newcandidate lottery. ActualAUTO/forcedSwizzle
p50<1 ANDp95<1 must hold in every eager/graph/start0/1 stratum for eachdtype;
vendorcomparison reported independently. If ownwin fails, preserve it and
report exactfailedstrata; do not repeatuntil lucky or call closed. Do not
rollback/remove kernel without root ruling. Sameartifact rawbits/poison/guards/
immutableinput andactualcapturedAUTOargs mustpassbeforeadmission.

Report all commands/exits, RED/GREEN, failurehistory, source-stablehashes for
rootreview, six-cell unchanged/promotedmatrix, postAUTO101 fullconstituents and
rawhashes, allsource/buildinput+binary/cache/compiler/PTXidentities, exactsource
diff closure versusTask6B andall3functionalqualification. Produce rootedSHA256
manifest, preserve actualsource/binaries/caches andfinalexclusive lanerelease.
No success claim for allinference/allprecision/allarchitecture, noFastwin
claim for remaining B0deficit. Root reviews/commits before nextbacklogtask.
