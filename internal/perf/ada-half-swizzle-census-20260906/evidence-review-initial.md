# Ada half-swizzle census evidence review

## Verdict

**REQUEST CHANGES for archive finalization: one P1 manifest-integrity issue and
one P2 narrative overstatement.** The six measurement runs and their substantive
results are otherwise approved. I found no error in the 400-record screen,
400-record confirmation, confirmed win/loss rosters, direct-pair obligations,
physical launch/bit/reference binding, or quiet-lane evidence.

This verdict is limited to the frozen three-arm census. It does not review or
authorize the separate direct-pair helper/run, AUTO selection, an epoch change,
or a fastest-candidate claim on CUDA 12.8/13.0. I performed local read-only
parsing and digest checks only; I did not rerun a verifier, test suite, build, or
GPU/SSH command.

## Findings

### P1 — the file called the final integrity manifest does not validate the current archive

`internal/perf/ada-half-swizzle-census-20260906/README.md:39-43` describes
`SHA256SUMS` as the final integrity manifest and says the archived report is bound
by it. However, both `SHA256SUMS` and the intentionally preserved
`SHA256SUMS.worker` are byte-identical and currently validate only 26 of their 27
entries. Their line 2 expects README SHA-256
`0c4f4b116e60a387c0ae42ee131f426b10e01f7407a441c5ea0a601a322c825d`,
while the current README is
`f3bbfd54f7d85a5563cf2c018ddbdd905f5bd82ee01da9058c8f12f296784876`.
The current `census-report.md` path is also absent; line 1 binds the byte-identical
source report at its ledger path instead.

This does not alter or orphan the measurement data: all six raw logs, all twelve
worker/main analyzer outputs, the TSV, telemetry/identity logs, analyzer copies,
and the source ledger report entry have matching manifest hashes. It does mean
the in-directory artifact labeled “final” is not presently a passing integrity
manifest, which conflicts with the brief's immutable-manifest requirement.

Bounded correction: preserve `SHA256SUMS.worker` byte-for-byte as the historical
worker manifest, but regenerate/update `SHA256SUMS` after README/report/review
finalization so all intended current archive paths validate, or rename/reword it
as historical and make the forthcoming root manifest the explicitly authoritative
final manifest. In either case, the README must not claim that a currently failing
manifest is final. Main's planned root manifest can resolve this if it binds the
new README, archived report, review, raw data, and preserved worker manifest and
the narrative is adjusted accordingly.

### P2 — completion records do not carry tuning revision 41 as the report claims

`ada-half-swizzle-census-report.md:94-100` says the verifier matched “every record
and completion” against all listed control fields, including tuning revision 41.
All six raw completion objects omit `tuning_table_revision`; only their ordinary
records carry value 41. The reviewed census verifier intentionally compares
completion identity with `IDENTITY_KEYS - ['tuning_table_revision']`
(`verify-half-census.rb:43-47`). Worker and main analyzer outputs are therefore
correctly successful, but the prose overstates what the completion schema proves.

Bounded correction: say that every ordinary record matches the full external
identity including revision 41, while the single successful completion in each run
matches all identity fields emitted by the completion schema; do not attribute the
missing revision field to completion. No measurement or roster needs changing.

## Evidence and claims verified

- **Whole-run completeness:** raw parsing found 160/160/80 ordinary records for
  CUDA 12.8/13.0/13.2 in both the 21-window screen and 101-window confirmation,
  exactly one completion per run, `passed=true`, `rejected=0`, one successful Rust
  test result, and `TEST_EXIT=0`. This is exactly 400 records per stage. The six
  worker verifier outputs report the same counts and
  `required_census_complete=true`; each main reproduction is JSON-identical to its
  worker counterpart.

- **External identity:** every verifier output's raw-log SHA matches the named raw
  file, and every identity-file SHA matches the committed Task-1 toolkit control.
  All ordinary records carry CC8.9, 142 SMs, the appropriate NVRTC tuple, the
  per-toolkit source/invocation/artifact/header/library identity, and revision 41.
  Preflight and postflight transcripts each contain 174 successful source checks,
  23 successful binary/cache checks (14 executables plus 9 caches), and zero check
  exit statuses (`preflight-identities-telemetry.log:170-206`,
  `postflight-identities-telemetry.log:176-207`). No timing-derived control is used.

- **Raw samples and aggregation:** all records have arrays equal to their declared
  21 or 101 windows. The reviewed analyzer recomputed medians and paired p50/p95
  from same-index raw samples and took maxima across eager/graph and both execution
  orders. The main and worker outputs agree exactly, so this review did not treat
  ratios of independent quantiles as paired ratios. All six analyzer outputs retain
  `auto_admission_authorized=false`.

- **Confirmation table:** `confirmation-cell-results.tsv` has exactly 100 data
  rows: 40 each for CUDA 12.8/13.0 and 20 for CUDA 13.2. Every key, Boolean result,
  and nine-decimal numeric value matches the corresponding full-precision
  confirmation JSON (within the stated decimal rounding). The report's rosters
  follow those outputs exactly:

  - CUDA 12.8 and 13.0 pipeline: 20/20 internal, 12/20 vendor;
  - CUDA 12.8 and 13.0 swizzle: 20/20 internal, 15/20 vendor;
  - CUDA 13.2 swizzle: 10/20 internal, 13/20 vendor.

  The listed vendor-loss cells match the JSON. CUDA 13.2 F16 E is an internal win
  in both 21-window screen aggregates but a loss in both 101-window confirmation
  aggregates, so the report correctly refuses to promote the screen result.

- **No false fastest claim:** the CUDA 12.8/13.0 analyzer outputs identify all 20
  dtype/cell/bias groups as `direct_pair_required`; CUDA 13.2 identifies none
  because the census has only swizzle versus the actual pipeline AUTO. The report
  correctly states that separate pipeline/AUTO and swizzle/AUTO pairings cannot
  rank pipeline against swizzle.

- **Graphs and bits:** local inspection found no custom-graph mismatch in any of
  the 800 records. AUTO and forced graphs each have exactly one kernel and no
  non-kernel nodes, the typed symbol, flat
  `(ceil(M/128)*ceil(N/128),1,1)` grid, block 256, and 71,680 bytes for Tc128 or
  pipeline versus 69,632 bytes for swizzle. Raw-storage, AUTO, repeat and vendor
  repeat flags are true throughout; graph replay is true exactly on graph-path
  records and false on eager records. This is physical and bit evidence for the
  measured callsites, not merely an enum claim.

- **FAST/reference boundary:** all 800 records use native-half vendor
  `CUBLAS_COMPUTE_32F` as the timed performance denominator and
  `CUBLAS_COMPUTE_32F_PEDANTIC` only as the F32 numerical reference. Bias records
  have `vendor_gemm_beta=1` and `vendor_bias_broadcast_timed=true`; non-bias records
  use zero/false. All AUTO/forced/vendor normalized errors are finite and within
  0.01 BF16 or 0.0025 F16. The report correctly avoids calling vendor comparison a
  raw-bit FP32-bias-preseed oracle and makes vendor wins separate from internal
  admission.

- **Quiet-lane evidence:** every raw log begins with the fully expanded matching
  toolkit command and required environment (`{screen,confirm}-*.log:1-2`). Each
  immediate pre snapshot is the same GPU UUID at CC8.9/142 SM, 0% GPU/memory,
  P5, and an empty active-compute-app section; each immediate post snapshot is 0%
  with no compute app. Residual P0 clocks, power and temperatures are retained.
  The initial unsupported telemetry-field diagnostic is preserved, and the
  corrected preflight before timing proves CC8.9/142 SM with exit 0
  (`preflight-telemetry-corrected.log:1-5`). Main's telemetry summary binds the six
  raw hashes and records every run quiet with exit 0
  (`analysis/main-identity-telemetry-verification.json:1-79`).

- **Narrative boundaries:** `README.md:10-29` and the report correctly distinguish
  screening from confirmation, internal from vendor wins, and paired-to-AUTO from
  direct candidate ranking. They authorize neither AUTO nor epoch changes and make
  no Triad/global-FAST/5090 claim. The archived `census-report.md` is byte-identical
  to the source report (SHA-256
  `6b199a00cb30d0eda1598c441288951bacdc89735d70ddcd936ca4b0e718f8cc`).

After the two bounded archive/prose corrections, the census package is suitable as
the immutable measurement input to the separate direct-pair task.
