# Ada half-swizzle census archive fixround1 review

## Verdict

**APPROVE.** Both prior findings are addressed and I found no breakage introduced
by the scoped README/report/manifest changes. This approval remains limited to
the frozen three-arm census archive; it does not review the concurrent direct-pair
work, authorize AUTO, or change the previously approved measurement conclusions.

I performed local read-only document and digest checks only. No benchmark,
verifier, test suite, build, GPU, or SSH command was rerun.

## Finding dispositions

### P1 final manifest did not validate current archive — ADDRESSED

- `internal/perf/ada-half-swizzle-census-20260906/SHA256SUMS` is now a distinct
  current-archive manifest with 29 entries. Local SHA-256 validation found 29 OK,
  zero mismatches, and zero missing paths.
- It binds the amended README, the archived corrected report, the archived initial
  evidence review, and the original worker manifest in addition to the unchanged
  raw/analyzer/TSV/telemetry evidence (`SHA256SUMS:1-29`). The current hashes match
  the fix report: README `8fccb444...c65285`, report
  `bd0fe6aa...a65d`, initial review `8de9b4ed...e5c4`, and worker manifest
  `4a154b71...9897`.
- `SHA256SUMS.worker` remains byte-preserved at the reported hash. The README now
  explicitly calls it a historical pre-correction snapshot whose old paths and
  document hashes are provenance, not validation of amended documents, and directs
  current validation to `SHA256SUMS` from the worktree root (`README.md:44-48`).
  This removes the former false “final” implication.
- Both report copies are byte-identical at SHA-256
  `bd0fe6aa0367e3525146cc1c99941f3008e90d757dbc553bfc8e02f59c93a65d`.

After this review is archived, the planned final regeneration must naturally add
that new archive path before the evidence-only commit. That expected post-review
step is not a remaining defect in the presently reviewed 29-entry snapshot.

### P2 completion was said to bind revision 41 — ADDRESSED

Both report copies now state the exact split:

- every ordinary record binds the full external identity, including tuning
  revision 41;
- each completion binds its emitted identity subset;
- V2 completion omits `tuning_table_revision`, and the verifier intentionally
  excludes that field for completion matching
  (`ada-half-swizzle-census-report.md:90-98`, identically
  `census-report.md:90-98`).

This matches the six raw completion objects and the reviewed verifier behavior.
The edit changes no roster, ratio, graph, bit, reference, or admission claim.

## New-breakage check

No new issue found. The README still identifies `SHA256SUMS` as the current final
integrity manifest while accurately distinguishing the historical worker snapshot;
all referenced current paths exist and validate. The report correction is narrowly
worded and preserves the established external-control/no-timing-inference boundary.
The remaining document changes are manifest bookkeeping only; the raw logs,
worker/main analyzer JSON, TSV, identity/telemetry evidence, and measurement claims
are unchanged by this fix.
