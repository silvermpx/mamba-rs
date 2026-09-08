# Ada exact-F32 TN d768-in CopyPlan probe, 2026-09-08

Correctness stop before timing. The three-node transpose -> full-K Fixed
CopyPlan -> exact FMA epilogue candidate passes native proofs, resource gates,
small normal/exception/K0 and non-unit-alpha eager/graph checks, but it does
**not** match actual public AUTO on the full d768-in target. Word0 differs:
`d76bd726` candidate versus `d76bd729` AUTO. No speed result exists and this
candidate must not be integrated or retried unchanged.

## Root cause

On RTX6000Ada the exact policy resolves target `(2048,768,3072)` to
`TnSplitM { m_chunk:1024, chunks:2 }`, not to a single linear TN reduction:

- `gemm_bi_tn_splitm_partial_aligned`, grid `(144,1,2)`, block256,
  static shared33,792B, forms two independent ascending BK16 F32-FMA partials;
- `gemm_bi_splitm_reduce`, grid9216/block256, sums partial0 then partial1 in
  FP64, casts once to F32, then accumulates into dW.

The rejected CopyPlan instead forms one ascending m0..2047 F32-FMA chain and
then applies a fused seed epilogue. Both are deterministic, but they are
different arithmetic contracts. The small probes selected `gemm_bi_tn_narrow`
and therefore could not prove equality to the target SplitM route.

A distinct feasible follow-up must preserve the exact two1024-sample partial
layout and reuse the existing FP64 reducer. Possible mechanics are a raw-partial
CopyPlan per identical chunk or a faster SplitM partial body; either must first
prove each partial word identical, including NaN payloads. This result gives no
permission to weaken bit equality or compare timing before that proof.

## Evidence

All candidate component resource gates pass: Fixed135regs/occ3/static32768,
transpose18regs/occ3/static4224, epilogue10regs/occ6; all have local0. Exact
test stopped on the first full-target candidate bit comparison, before any
paired AUTO/Fast screen.

- Main test SHA256:
  `1f5c25999ef04ba75c07d8b5284347a79d90fdc7b6befdea7ff3defbf7ba68a9`.
- Epilogue helper SHA256:
  `95a47682178bbc4e068649583302678b81a64e3295040e979c564d0fa19e480c`.
- Binary SHA256:
  `256501230d70d1093d4d8fcd44b97319f078937447e42a0c2a52713802215c4b`.
- [Raw log](evidence/cuda132/run1/test.log), SHA256
  `e03a4b572bca50db9e6ef635be484b9d5529425e113b8b8ce014be8409697d89`.

Cache before/after hashes match. Release and drain have no compute apps. This
is an invalid candidate for the selected target contract, not a performance
loss, Fast comparison, dispatcher change or CUDA12.8/13.0 claim.
