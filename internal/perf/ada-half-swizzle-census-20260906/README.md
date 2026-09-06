# Ada half-swizzle paired production census (2026-09-06)

This directory preserves the measurement-only CUDA 12.8/13.0/13.2 census for
the production `Tc128Sm89Pipeline` and `Tc128Sm89Swizzle` forced rungs.  It uses
the source, release binaries, and private caches frozen at Task-1 checkpoint
`6570ce872aafe77bb94ecc5b6fee27fd48c054f7` and the external identity controls
in `../ada-half-swizzle-force-20260906/identity-cuda{128,130,132}.json`.

## Outcome

All six whole runs passed the strict external-control verifier.  The 21-window
screen contains 400 records and the 101-window confirmation contains 400
records, with no rejected, missing, duplicate, or foreign cohorts.

- CUDA 12.8 and 13.0: both forced candidates beat actual `Tc128` AUTO at paired
  p50 and p95 in all 20 dtype/cell/bias aggregates.  Therefore all 20 aggregates
  require a direct same-run pipeline-vs-swizzle comparison; these independent
  AUTO ratios do not rank the two candidates.
- CUDA 13.2: swizzle beats actual `Tc128Sm89Pipeline` AUTO in 10/20 confirmation
  aggregates: BF16 B/D/E and F16 B/D, each with bias off and on.  It loses A/C
  for both dtypes and F16 E for both bias states.
- Vendor results are separate: in the 101-window confirmation, pipeline wins
  12/20 on both CUDA 12.8 and 13.0; swizzle wins 15/20 on CUDA 12.8 and 13.0,
  and 13/20 on CUDA 13.2.  Vendor performance is not the internal admission gate.

`confirmation-cell-results.tsv` gives every cell's worst paired p50/p95 over
both eager/graph paths and both execution orders.  The `verification-*.json`
files retain the same exact per-cell results plus identity and direct-pair
obligations.  `analysis/main-*.json` is main's independent reproduction using
the reviewed verifier copy in `analysis/`.

## Raw evidence

- `screen-cuda{128,130,132}-w21.log`: exact command/environment, immediate
  before/after telemetry, complete raw sample arrays, completion, and exit.
- `confirm-cuda{128,130,132}-w101.log`: corresponding full confirmations.
- `preflight-identities-telemetry.log`: 174 source and all 14 binary/9 cache
  checks.  Its first telemetry query intentionally preserves the diagnostic
  that this `nvidia-smi` lacks `multiprocessor_count` as a query field.
- `preflight-telemetry-corrected.log`: supported live telemetry plus a read-only
  CUDA Driver query proving CC 8.9 and 142 SMs before any timing began.
- `postflight-identities-telemetry.log`: final unchanged source/binary/cache
  checks and idle released-lane state.
- `SHA256SUMS`: final integrity manifest for the evidence directory.
- `SHA256SUMS.worker`: historical worker-manifest snapshot, before final
  README/report corrections. Its original paths and document hashes are
  provenance, not a validator for the amended documents. Use `SHA256SUMS`
  from the worktree root to validate the current complete archive.

Every run used GPU `GPU-d1edd7be-e88d-aed6-047d-622163306f0e`.  Immediate
preflight snapshots showed 0% GPU and memory utilization, P5, and no other
compute application.  Post-run snapshots showed 0% utilization and no compute
application (with expected residual P0 clocks/heat).  No process matching,
signaling, workload termination, source edit, build, or cache mutation occurred.

The full narrative and command report is [census-report.md](census-report.md).
This census authorizes no AUTO or epoch change.
