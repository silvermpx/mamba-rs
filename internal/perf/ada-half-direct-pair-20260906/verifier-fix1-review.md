# Half direct-pair verifier fix round 1 review

## Verdict

**APPROVE. Prior P1 ADDRESSED.** The bounded fix closes the record, completion,
and external-control schemas exactly as requested, retains all three actual-AUTO
correctness fields, and introduces no new defect found in the reviewed change.
This is an analysis-helper verdict, separate from harness runtime qualification
and any AUTO decision.

I reviewed only `half-direct-verifier-fix1.diff`, the appended fix section in
`half-direct-verifier-report.md`, and the prior finding/context. I did not rerun
the reported focused 1-test/6-assertion or full 9-test/48-assertion GREEN suites.

## Prior finding

### P1 — open record, completion, and external-control schemas: ADDRESSED

- `verify-half-direct-pair.rb:9-19` defines the record set as the exact inherited
  identity keys plus every field enumerated by the V1 brief, and defines the
  completion set as identity plus only `schema`, `records`, `rejected`, and
  `passed`. The record set retains `actual_auto_tile`, `auto_bits_equal`, and
  `auto_normalized_error`, while admitting no AUTO/vendor timing, ratio, winner,
  or admission field.
- `verify-half-direct-pair.rb:25-26,43-46` now requires exact external-control
  and per-record key-set equality. This rejects both missing and extra keys
  before accepting the evidence cohort.
- `verify-half-direct-pair.rb:113-130` continues to reject unknown schema
  objects and now also requires exact completion keys before validating pass,
  zero rejects, record count, and full identity.
- `test-verify-half-direct-pair.rb:155-164` supplies focused negatives for extra
  AUTO timing, vendor timing, admission and arbitrary record fields, an extra
  external-control field, and an admission-bearing completion. The pre-existing
  valid fixture remains exercised by `test_completion_and_actual_harness_are_required`
  (`:135-153`). These cases directly cover the previously demonstrated hole.

## New-breakage check

None found. Exact schema closure does not weaken the existing 80-key cohort,
identity, graph, bits/reference, sample, quantile, or winner checks. Array
construction uses non-mutating concatenation, so freezing `RECORD_KEYS` and
`COMPLETION_KEYS` does not alter the inherited `KEYS`. Parser ordering still
allows the record analyzer to enforce record closure, while completion closure
is enforced during parsing. The helper still returns
`auto_admission_authorized=false` and makes no vendor-performance claim
(`verify-half-direct-pair.rb:91-106`).
