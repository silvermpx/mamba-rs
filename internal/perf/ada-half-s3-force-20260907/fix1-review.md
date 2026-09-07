# Task6A fix round1 review

## Spec Compliance

✅ Source compliant for the reviewed Task6A source snapshot plus fix1. The sole prior Important finding I1 is closed. No new material issue was found in the scoped delta.

⚠️ This closes the source gate only. Complete CUDA12.8/13.0/13.2 matrix, retained-artifact and cohort qualification, sanitizers, final identity reconciliation and GPU-lane release remain a separate pending acceptance gate.

## I1 disposition

`src/mamba_ssm/gpu/gemm_bi_fixed.rs:2981` now validates checked K+127. For accepted nonnegative K, the maximum accepted value is 2,147,483,520, yielding at most 33,554,430 slabs; the final unconditional `(kt+2)*64` is therefore at most 2,147,483,584 and fits i32. This also protects K+63 tile rounding. Negative values cannot enter this helper from the public path because the existing `FixedArgs::try_new` converts a usize dimension to i32.

`src/mamba_ssm/gpu/gemm_bi_fixed.rs:3027` calls the helper before holder lookup and launch. This tightens only S3 and leaves the frozen CUDA body unchanged. The source hash in the reviewed package remains `e163497e2f092c911146704b37691a645d57a3868379d9cf59acc31e2dddeba8` for `sm89_half_s3.cu`.

`src/mamba_ssm/gpu/gemm_bi_fixed.rs:2990` rejects the first value beyond the new bound, the original reproducer and INT_MAX; it accepts the largest safe bound and normal K values, independently calculating lookahead in i64. `tests/gemm_bi_fixed_sm89_pipeline.rs:457` also drives both invalid boundary values through the real public S3 dispatcher and asserts its specific prelaunch error. These checks cover the production call site as well as the pure arithmetic helper.

## Other scoped changes

- `src/mamba_ssm/gpu/gemm_bi_triad/modules.rs:1713` replaces the explicitly provisional 255-register cap with measured188. `tests/gemm_bi_fixed_sm89_pipeline.rs:141` matches that bound. The independent initial cold censuses record182 registers for both dtypes on12.8/13.0 and188 on13.2, all with local0/static0/maxthreads256/occupancy1/dynamic98304. This supports the chosen source admission threshold without implying the entire final matrix is complete.
- `tests/gemm_bi_fixed_sm89_pipeline.rs:81` adds compiler/artifact identity logging to the holder test without changing admission or launch behavior.

## Evidence inspected

- `internal/perf/ada-half-s3-force-20260907/reviewfix-k-boundary-red.log:13,16,20`: the focused boundary test fails at the expected assertion with the old guard, exit101.
- `internal/perf/ada-half-s3-force-20260907/reviewfix-k-boundary-green.log:8,10`: the focused boundary test passes with the fix, exit0.
- `internal/perf/ada-half-s3-force-20260907/cuda128-attempt1/half-full.log:7,22`: public unsafe-operands/dimensions test completes successfully; full pipeline suite11passed/0failed. `matrix-cuda128-attempt1.log:80` records `EXIT half-full 0`.
- `internal/perf/ada-half-s3-force-20260907/cuda128-attempt1/half-full.log:15,16` records both12.8 S3 resource rows. `first-live-cuda130-132.log:6,7,12,18,19,24` records both13.0/13.2 rows and successful cold exits.
- Read only the supplied fix1 delta, the report's new fix section and focused existing evidence records; located exact line anchors. No builds, reruns, GPU/SSH actions, source changes, staging, commits or agents. Root supplied independently verified reconstructed-v1/current source hashes; this review did not duplicate that reconstruction.

## Issues and assessment

Critical: none. Important: none remaining. Minor: none material to this source gate.

**Task quality: Approved for source.** The narrow host fix closes the accepted-input overflow while preserving the previously tested CUDA body, and meaningful negative/positive boundary coverage plus the completed public-dispatch test support the change. The original v1 source review remains the record for all unchanged paths; its I1 disposition is superseded by this review.
