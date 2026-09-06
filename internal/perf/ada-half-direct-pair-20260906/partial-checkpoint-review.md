# Half direct-pair partial-checkpoint review

## Verdict

**APPROVE for the partial checkpoint.** The corrected wrapper is statically
fail-closed for its intended live timing run, the partial archive clearly
excludes every telemetry-invalid timing attempt, and the report explicitly
releases the idle Ada lane. The documents do not claim that the unexecuted final
live probe, fresh performance runs, or final remote reconciliation have passed.
No must-fix issue remains in this bounded scope.

I reviewed only `run-direct-pair.sh`, the partial `README.md`,
`SHA256SUMS.partial`, and `ada-half-direct-pair-report.md`. I did not revisit the
already-approved Rust harness/analyzer, use SSH/GPU, or run a benchmark. The
final reviewed report SHA-256 is
`70ede18f8ae35033dc59340f2c932d4e6e5b69706c60cc7a54dee15a7f993426`;
the README SHA-256 is
`0bb59ab76ca0b810aae8262f06aad174a106c21a29ab434c21b4c83defe1f469`;
and `SHA256SUMS.partial` itself is
`782b69a234270c98de1bc61db5a9e7c1a44aadb6454d47812a16a8e3062d605d`.
All 29 entries in `SHA256SUMS.partial` exist and passed `sha256sum -c` locally.

## Resolved finding

### P2 — report named the bound tuning-revision field as an epoch: ADDRESSED

`ada-half-direct-pair-report.md:150-152` now accurately calls the bound value
“tuning revision 41,” consistent with the V1 `tuning_table_revision` identity
field and the report's earlier terminology. No separate epoch claim remains.

## Verified checkpoint boundaries

- **Fail-closed wrapper:** toolkit and window inputs are restricted before any
  run, paths and expected source/binary/control hashes are pinned per toolkit,
  and all relevant harness filters are exported while foreign vendor-tile state
  is unset (`run-direct-pair.sh:6-46,121-140`). Hash failure exits before
  telemetry or benchmark.
- **Pre/post telemetry:** the supported `nvidia-smi` query checks the exact UUID,
  model/CC and zero GPU/memory utilization, rejects a nonempty compute-app list,
  and separately obtains CC 8.9/142 SM through CUDA Driver attributes
  (`run-direct-pair.sh:48-119`). Any date, GPU query, app query, or Driver query
  error makes telemetry return 70. Failed preflight exits before
  `BENCHMARK_START`; postflight still runs after the benchmark and its failure is
  propagated (`:142-162`). Omitting `set -e` is deliberate here: it permits the
  wrapper to preserve the benchmark exit and still collect postflight state.
- **Functional evidence versus timing:** the README labels eager/full
  one-window runs as functionality only (`README.md:9-28`). It separately names
  the two 21-window and one 101-window telemetry-invalid attempts, preserves
  them as diagnostics, and explicitly excludes their verifier output from
  timing qualification (`README.md:30-38`; report lines 154-162).
- **Failure-probe chronology and unexecuted final probe:** both documents now
  distinguish the missing-control stop, the injected-false run that also found
  the invalid `date` argument, and the corrected-`ns` injected-false run whose
  date and Driver checks passed while both injected `nvidia-smi` calls failed
  closed before `BENCHMARK_START` (`README.md:46-57`; report lines 171-186).
  They explicitly state that the subsequently strengthened utilization/app
  checks have not received their final live probe because the remote command
  never ran. No diagnostic is promoted to final live-wrapper or timing proof.
- **Pending reconciliation and conclusions:** the README reserves fresh 12.8
  and 13.0 21/101 logs, strict verification, reconciliation, final checksums and
  per-cell reporting for later (`README.md:59-64`). The report repeats those
  gates and makes neither a fastest-route nor AUTO-admission claim
  (`ada-half-direct-pair-report.md:199-205`). This is an appropriate partial
  checkpoint boundary rather than an incomplete Task 2 approval.
- **Lane state:** the amended report says no worker GPU job is active and the
  exclusive Ada lane is explicitly released (`ada-half-direct-pair-report.md:
  197-200`). This resolves the stale reservation wording and does not imply the
  blocked runtime work is complete.

No other must-fix issue was found in the bounded files.
