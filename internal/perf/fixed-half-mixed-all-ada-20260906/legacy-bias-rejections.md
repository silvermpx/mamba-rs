# Legacy bias rejection triage

The all22 screen completed with800timing records and20rejected candidate/cell
attempts. Each rejected attempt suppresses4eager/graph/order timings, so the
planned880timing cohorts reconcile as800+20*4. Do not call this880passes.

All20rejections are `Legacy`, `bias=true`, `gate=auto_bit_identity`: four
dtype rows (BF16,F16,BF16->F32,F16->F32) times five A-E shapes. There are no
other rejection kinds in the completion log. The forced Legacy output does
not match current AUTO bit-for-bit on this corpus, so no timing result or
promotion credit is assigned to it. This is not evidence of run-to-run
nondeterminism; it is a cross-route bit-equivalence failure.

Source-level cause consistent with the entire observed pattern:

- `kernels/gemm_bi_fixed/wmma_legacy.cu`: WMMA accumulators are seeded with0;
  bias is added with `__fadd_rn` after the dot product in the scalar epilogue.
- `kernels/gemm_bi_fixed/mma16.cu`: Tc16/Tc64/Tc128 seed their F32
  accumulators withbias before ascendingK MMA. The Ada pipeline preserves
  this ladder's arithmetic.
- Different bias insertion points can change final rounding, including when
  the final output is half. Both can satisfy a numerical tolerance without
  being interchangeable under the Fixed rawbit contract.

The current `fixed_forward` uses the portable/architecture half ladder for
every supported homogeneous-half or same-half-input/F32-output request. Its
generic Legacy tail is for unsupported dtype triples; the two half pickers
do not select Legacy. Therefore these20forced-baseline exclusions are not
20broken AUTO cells. The source mapping was checked in fixed_forward and
fixed_pick_tile/fixed_pick_f32out_tile on the C-promotion tree.

Disposition: retain the gate and raw rejection records. Do not weaken it to
tolerance, do not silently substitute Legacy for a ladder rung withbias,
and do not delete the legacy source until other API/fallback users have
been audited. No production source or selector was changed by this triage.
