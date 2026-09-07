# Task6A source review — supplied v1 snapshot

## Spec Compliance

- ❌ One source issue: the S3 public launcher's K overflow check does not cover the frozen S3 mainloop's two-tile lookahead arithmetic. See Important I1 below.
- The remaining reviewed force wiring matches the brief: distinct homogeneous-half enum/symbols, Fixed/sm89-only suffix, independent ABI census and holder/rejection, five arguments, 256 threads, 98,304 dynamic shared bytes, and complete forced physical-identity wiring. AUTO is untouched and tuning revision remains 42.
- ⚠️ This is a source verdict on `ada-half-s3-force-task6a-review-source-v1.md`, based on 52d57d09 with documentation-only HEAD 3c0b1523. Final CUDA12.8/13.0/13.2 qualification, retained-artifact proofs, final resource cap, sanitizers, manifests and GPU release remain pending. This review does not grant final Task6A acceptance.

## Strengths

- `src/mamba_ssm/gpu/gemm_bi_triad/modules.rs:490,672,782,925` carries the S3 ABI result separately through both warm and cold branches and final module construction. The separate loader at `:2568` returns only S3 availability/rejection, preserving incumbent holders.
- `src/mamba_ssm/gpu/gemm_bi_triad/modules.rs:1863,2066,2529` validates the exact two-symbol inventory, dtype-correct MMA and both waits, parameter layout, local/static memory, register/thread limits and occupancy. New negative fixtures cover missing/extra exports, instruction drift, ABI offsets/sizes and the terminal Driver probe (`:11905,12048`).
- `src/mamba_ssm/gpu/gemm_bi_triad/modules.rs:6395` appends S3 after the unchanged swizzle provider only in the Fixed/sm89 branch. The supplied diff contains no edits to existing CUDA fragments or other target composition paths.
- `kernels/gemm_bi_fixed/sm89_half_s3.cu:13,40,45,72,94,130` retains the distinct three-slot ring and 49,152-byte B base while reusing the provider's fragment/copy/epilogue helpers. A focused check of the unchanged provider confirmed stage-index arithmetic accepts slot 2 and `sm89_half_swizzle.cu:237` retains the shared-alias barrier. The S3 final transition waits for group 0 before consuming the last tile.
- `src/mamba_ssm/gpu/gemm_bi_fixed.rs:252,2981` makes S3 reachable through the actual typed public force API. `tests/gemm_bi_fixed_performance.rs:10903,11131,11408,11469,13163,13266` covers the reverse-complete registry, exact typed symbols, required physical parameters and eager/graph inventory arms.
- The supplied pipeline-test hunks add S3 to the existing full/hot/rounding/sanitizer corpora, preserve incumbent candidates, add short K128/256, and replace potentially equal output poison with reference-complement bits and upload readback. The raw ABI test in v1 at `tests/gemm_bi_fixed_sm89_pipeline.rs:664` checks independent strides/nonunit scalars and captured arguments; nonzero-beta replays reset to identical old-C input and assert it differs from gold.

## Issues

### Critical

None found.

### Important

**I1 — Accepted large K causes signed overflow in S3 lookahead.** `src/mamba_ssm/gpu/gemm_bi_fixed.rs:3007` checks only `args.k.checked_add(63)`, whereas `kernels/gemm_bi_fixed/sm89_half_s3.cu:72` unconditionally evaluates `int next_k = (kt + 2) * 64` on every fast-path iteration, including the final one when `refill` is false. For K=2,147,483,584, the host check succeeds; `num_k_tiles` is 33,554,431, and final kt=33,554,430 produces 2,147,483,648, beyond signed int. This introduces C++ signed-overflow undefined behavior in an input admitted by the public API. The value is unused on that iteration, so this review does not claim observed GPU corruption, but the accepted source-level arithmetic is undefined. A focused check of `FixedArgs::try_new` at `src/mamba_ssm/gpu/gemm_bi_fixed.rs:1039` confirms it only narrows dimensions to i32 and does not reject this K. M=1/N=8 with aligned inputs reaches the fast path and passes the grid checks.

Preserve the frozen Task5B kernel by tightening the S3 host lookahead bound to checked K+127 (maximum safe K=2,147,483,520), or explicitly avoid evaluating unused out-of-range lookahead in a separately reviewed kernel change. Add a focused boundary rejection check for the first invalid K=2,147,483,521; the existing unsafe-dimension fixture tests K=i32::MAX and therefore misses this accepted band. No huge allocation or long GPU run is needed to verify rejection.

### Minor

None material to this source gate.

## Assessment

**Task quality: Needs fixes** for I1. The implementation otherwise follows the existing independent admission and launch boundaries, reuses the CUDA helpers without duplicating them, and tests actual captured launch parameters and overwritten output bits. Final acceptance also requires the explicitly pending runtime evidence and final cap.

## Review scope and pending follow-up

- Read the brief, implementation claims, parent integration plan/wiring audit and the supplied complete seven-path diff once, in consecutive chunks. No builds, test reruns, GPU/SSH calls, source edits, staging, commits or subagents were performed.
- Named focused checks outside the diff: helper compatibility/shared-alias safety in unchanged `sm89_half_swizzle.cu`; earlier admission of the concrete overflowing K in `FixedArgs::try_new`; current tuning revision at `kernel_identity.rs:35`; supplied snapshot identity via SHA256. The production body's existing `body-equivalence.log:1` records comparison to candidate hash `a90f504d6d2d0363908fd00d230263164105b41c6b5a6085a83ae130f2ce4222`; this review read that record, without rerunning candidate qualification or reconstructing the comparison.
- Six current path hashes matched v1 at the review hash check. During review, `tests/gemm_bi_fixed_sm89_pipeline.rs` changed from v1 SHA256 `5c6e3cd14e8c5153197ca5410d654dfed3567c1ecea4a94182602a40deb27038` to `7e6aeda987edf4c5197c9df6a9793e148314ffa6cc20378c364805f36bf086f2`. Root reports this is a five-line module-identity logging insertion; the later delta remains outside this verdict and will accompany final review. Pipeline test line references above refer to supplied v1.
- `src/mamba_ssm/gpu/gemm_bi_triad/modules.rs:1712` has provisional cap255 by explicit task direction. This is a pending final-qualification condition, not a separately invented source defect. The reported 12.8 live182-register result does not substitute for final counts and zero local/static/stack/spill proof across all three toolkits.
- Root should reconcile final exact source/binary/cache/compiler/driver hashes, cold/warm ABI and holder evidence, full S3 and retained Fixed suites, unchanged Triad SM89/SM120 cohorts, supported sanitizer statuses, final resource-cap delta and precise GPU-lane release before full acceptance.
