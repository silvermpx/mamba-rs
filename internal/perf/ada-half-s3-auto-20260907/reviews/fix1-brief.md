# Task6C source fix1 — current epoch consumers

Original frozen Rust review:
`ada-half-s3-auto-task6c-rust-review.md`. Original four-file source package:
`ada-half-s3-auto-task6c-rust-review-v1.md`. Original performanceSHA
3888a55389044274bd74e9feb14c69133ca8ae4404889ecdfaad65644ec291f1.
Root grants the existing implementer this additional narrow ownership; no
second implementer or root source edits. All original Task6C constraints stay.

## I1 Important, independently reproduced

Full CUDA12.8 library returns642passed/3failed/46ignored,exit101. The current
global tuning43 disagrees with live/current expectations42 in:

- `src/mamba_ssm/gpu/gemm_bi_triad/dispatch.rs`:
  `tf32_tn_underfill_qualification_uses_current_tuning_revision`, both
  TUNING_TABLE_REVISION and F32_TF32_TUNING_REVISION assertions.
- `src/mamba_ssm/gpu/gemm_bi_triad/launch.rs`: two physical launch identity
  assertions at priorlines13855/13922.
- `tests/gemm_bi_tf32_cohort_binding.rs`: current captured evidence tuning
  assertion at priorline207, in the existing SM120 ignored runtime fixture.

Ruling: the original four-file ownership omitted necessary consumers of a
global epoch. Extend only these three test files and change those five live
expectations42->43. F32_TF32_TUNING_REVISION remains the alias of global tuning;
do not change contract.rs, CUDA, physical routes, numeric/schedule revisions,
historical captured42 fixtures or unrelated byte arrays.

## Additional current direct-pair oracle

Root's focused revision-consumer scan found the existing active
`fixed_ada_half_forced_direct_pair` calls `expected_ada_half_auto_v42` at
priorperformance.rs:14580. That helper at9689 still expects Swizzle for both
13.2B0/no-bias cells, so it fails against current productionAUTO43. This is
the current AUTO oracle inside a forcedS2 control benchmark, not historical
pre42 `ada_s3_pair` metadata.

Ruling: update only this helper/current test names/diagnostics to43 and the
two13.2homogeneoushalfB0/no-bias choices toS3. Other58 choices unchanged.
Keep existing Pipeline/Swizzle forced controls, test modes and timing schema.
Keep old `ada_s3_pair` pre42 schema/meaning/revision rejection untouched.

## Verification and review boundary

Preserve failed12.8 library output/exits. Run corrected full matchinglibrary
onall3toolkits plus focused/new/nonignored performance suite, including direct
oracle hosttests. Compile and run nonignored `gemm_bi_tf32_cohort_binding` on
matchingtoolkits; the actual SM120 ignoredtest cannot run onAda, report that
hardware limitation rather than weakening its precondition.

Add one functional windows1 direct-pair smoke on13.2hot_b,bothhalf dtypes,
bothbiases/eager+graph. This verifies currentAUTO43 with retainedforcedS2
controls, not new performance admission. Continue original Task6C all3functional
and postAUTOsmoke101 proof only after all finalsource/binarybindings match.

Reopen source freeze and preserve earlier build bindings/failurehistory; no
historical Task6B timing reruns. Sourcefreeze after minimal changes/formatting,
append exact sourcehashes, commands/exits to the same implementation report.
Root will review only the fixdiff for I1/additional currentoracle and any new
breakage; the alreadyreviewed four-file implementation is not re-reviewed.
