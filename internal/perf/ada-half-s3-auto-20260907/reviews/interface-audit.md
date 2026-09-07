# S3 AUTO admission interface audit

Read-only preflight at0892986d, while Task6B owns the performance test and GPU.
This records interfaces for the already-planned next stage, not an admission
decision or an implementation dispatch. The six dtype/toolkit decisions must
come from reviewed production paired results.

## Existing selection and fallback

- `src/mamba_ssm/gpu/gemm_bi_fixed.rs:477` defines
  `fixed_select_sm89_half_auto_tile`. Its existing common guard requires
  known NVRTC library, CC8.9/142SM, literal12.8/13.0/13.2, homogeneous half,
  nonzero16-byte-aligned C/A/B, bias alignment, and exactly five hot shapes.
- Its preferred-route match selects Swizzle for literal B0
  `(4621,768,2304)`, no bias, on all six dtype/toolkit combinations. This
  agrees with Task6B's required actual-AUTO baseline.
- Existing optional arguments are independent pipeline/swizzle availability.
  The fallback table is intentionally asymmetric on13.2: unavailable preferred
  Pipeline does not fall back to Swizzle. Preserve this behavior for every
  non-promoted cell and when an S3 holder is unavailable.
- `fixed_forward` at3688 supplies live device/compiler/holder state. Its
  homogeneous-half branch currently handles only Pipeline and Swizzle before
  the general architecture/portable fallback. S3 public-force launch and its
  independently admitted holder already exist from Task6A.
- `sm89_pipeline_auto_tests` at546 has the complete60-cell preferred-route
  table, independent holder fallbacks and common-gate negatives. Future S3
  promotion needs both new positive cells and unchanged-cell/fallback proof,
  including bias=true and neighboring dimensions declining S3.

## Identity revision and fixtures

- `src/mamba_ssm/gpu/kernel_identity.rs:35` owns the global
  `TUNING_TABLE_REVISION`, currently42. A host-only AUTO change must invalidate
  old captured dispatch without changing numeric/schedule/compiled identities.
- The identity tests around4950–5070 contain historical captured revisions
  38/39/40/41 and current42 assertions. Preserve historical values; add the
  previous42 captured-graph rejection when the current revision moves to43.
  Do not mechanically replace unrelated fixture bytes such as `[42; 32]`.
- `gemm_bi_fixed.rs:6325` pins the current numeric/tuning/schedule tuple
  `(5,42,8)`. Only tuning changes in the planned host-only promotion.
- `tests/gemm_bi_fixed_sm89_pipeline.rs:181` pins current AUTO revision42
  before checking all hot shapes. Its actual-AUTO route oracle must reflect
  exactly the measured promoted cells, while forced S3/Swizzle/Pipeline tests
  keep their own independent expectations.

## Timing interface boundary

Task6B's named pre-promotion harness explicitly requires actual AUTO Swizzle
at revision42. Its records are historical evidence once AUTO changes. Future
post-promotion confirmation must identify actual AUTO S3 against the unchanged
forced Swizzle control and native-half Fast, with explicit ratio direction;
do not relabel old records or silently reinterpret the pre-promotion schema.

The production CUDA, source composition, compiler options, ABI and existing
holders need no further change for an AUTO-only selection. Task6A's three
artifact identities remain valid only if those compiled inputs stay byte
identical. A new source change would require separate artifact qualification.

No file outside this scratch audit was modified. No GPU/build/test was run.
