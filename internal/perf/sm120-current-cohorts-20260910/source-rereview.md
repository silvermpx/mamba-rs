# SM120 current-cohort integration re-review

Scope: only the delta from the two frozen `*-green-fix1.rs` snapshots to
current `dispatch.rs` and `gemm_bi_tf32_cohort_binding.rs`.

Current hashes match the supplied freeze:

- dispatch: `c4ef0dbe74b0d20a19888a84d8c137fd4da8c7839606bc542a917b947979f90e`
- binding: `e79597052dfa926b9f28f94c0a80a9e129f8ff43c2e184201140b1db0e03f6ee`

## Prior findings

1. **ADDRESSED — non-vacuous G10 exact-route proof.** The rejected G10 arm now
   asserts `!nodes.is_empty()` before applying its existing all-nodes exact-F32
   contract and no-`tf32` checks. An empty launch-evidence set can no longer pass.

2. **ADDRESSED — lower-toolkit portable-twin isolation.** The portable baseline
   and full portable mutation inventory now run over the exact current
   driver-595.84 specialized/portable pairs for CUDA 12.8, 13.0 and 13.2. Each
   exact pair must resolve the retained portable route and every mutated twin must
   fall back to scalar F32. The separate CUDA 13.2/595.58.03 cohort is not folded
   into or altered by this loop.

No new finding is present in the scoped delta. The dispatch delta is confined to
test-module imports and the test loop; its production prefix is unchanged. The
binding delta is only the four-line nonempty assertion.

## Verdicts

- **Specification compliance: PASS.** Both previously open requirements are
  addressed exactly and without changing the production cohort tables.
- **Code quality: PASS.** The corrections are narrow, explicit and fail closed.

Host and GPU reruns remain root-owned and were not repeated for this review.
