# Half direct-pair wrapper v2 review

## Verdict

**REQUEST ONE P2 TEST ADDITION; wrapper implementation otherwise approve.** The
v2 phase branch implements the ledger ruling correctly and I found no execution
or failure-propagation defect. The supplied behavior test sources and calls the
actual shipped helper, but it does not independently exercise the memory half of
PRE's binding `0% GPU / 0% memory` condition.

This is a static wrapper/test review only. I did not revisit the Rust harness or
analyzer, run GPU/SSH work, or assess the worker's fresh timing results. The
package changes only the new v2 wrapper and its behavior test; the old wrapper
and excluded raw archive remain unchanged.

## Finding

### P2 — behavior test does not lock PRE's zero-memory-utilization gate

`test-run-direct-pair-v2.sh:10-12,27-34` supplies quiet `0%/0%` and residual
`14%/0%` snapshots. It therefore proves that PRE rejects nonzero **GPU**
utilization and POST accepts it, but never supplies `0% GPU / nonzero memory`.
The implementation currently checks both values correctly with the exact
`0 %, 0 %,` prefix (`run-direct-pair-v2.sh:18-23`), so this is not a present
wrapper logic failure. It is a gap in the new behavior regression that is meant
to preserve the full phase-specific admission rule: a later change that checks
only GPU utilization, or that improperly rejects POST memory residuals, would
still pass all eight assertions.

Bounded fix: add one literal memory-only residual snapshot, for example `0 %,
14 %`, and assert PRE returns 71 while POST returns 0. No wrapper, harness,
archive, build, or GPU rerun is needed.

## Verified behavior

- **Phase policy:** `direct_pair_validate_gpu_snapshot` first requires a
  successful `nvidia-smi` query and exact target UUID/model/CC for both phases.
  PRE additionally requires both utilization fields to be zero; POST deliberately
  applies no utilization threshold (`run-direct-pair-v2.sh:6-31`). This matches
  the ruling that immediate POST utilization can describe the benchmark's past
  NVML sample rather than a competing live process.
- **Remaining POST gates:** POST still records the full raw GPU snapshot, queries
  compute applications and rejects nonempty output, and independently verifies
  8.9/142 through CUDA Driver attributes (`run-direct-pair-v2.sh:81-151`). The
  phase relaxation affects only utilization; it does not relax query success,
  UUID/CC, active-app, date, or Driver checks.
- **Failure propagation:** a nonzero GPU-query status is returned unchanged by
  the helper, copied into `smi_exit`, and makes telemetry return 70
  (`run-direct-pair-v2.sh:11-13,97-100,145-149`). Failed PRE exits before
  `BENCHMARK_START`; benchmark, POST, and final-date statuses remain collected
  and propagated with benchmark failure taking precedence
  (`:170-194`). The actual old corrected run's POST 14%/0% would now pass only
  the utilization branch, but remains excluded as required; fresh whole runs are
  still necessary.
- **Actual-function test:** the source guard returns only when the wrapper is
  sourced, so `test-run-direct-pair-v2.sh:6-8` imports the production helper
  without executing argument validation or a benchmark. Assertions at lines
  27-34 call that helper directly and cover quiet PRE/POST, 14%/0% PRE rejection,
  14%/0% POST acceptance, wrong UUID in both phases, and exact propagation of a
  simulated query exit 9 in both phases. This is not a duplicated model.
- **Scope and identity:** the v2 main path retains the strict toolkit/window
  roster, immutable source/binary/control hashes, exact harness filters and
  foreign-tile unset from v1 (`run-direct-pair-v2.sh:37-79,153-172`). The new
  `WRAPPER=v2` marker distinguishes fresh output (`:168`). Source-on-import is
  bounded before all main-path state changes (`:33-35`).

The reported `bash -n`, ShellCheck, and 8-assertion pass are consistent with the
reviewed code; I did not duplicate those already observed checks. Final timing
evidence remains a separate review gate.

## Fix round 1 re-review

**APPROVE — P2 ADDRESSED; no open finding.** The test file at SHA-256
`e5d674b34841142ef3e73338bbe919de8a7e0807adf27f251b6bb5b75245aa4e`
adds a literal `0% GPU / 14% memory` snapshot and invokes the actual sourced
`direct_pair_validate_gpu_snapshot` helper with both phases
(`test-run-direct-pair-v2.sh:12,32-33`). PRE must return 71 and POST must return
0, exactly closing the missing half of the phase-specific utilization contract.
The existing quiet, GPU-residual, wrong-identity, and nonzero-query cases remain
unchanged around it (`:28-37`). I found no new test breakage.

The wrapper remains unchanged at SHA-256
`f6a23119cae22f36528c1bd4b461f0083bf63458e7020f12ad8bdbbb5046e71d`;
its previously approved logic was not re-reviewed. I inspected, but did not
rerun, `green-run-direct-pair-v2-review-fix.log`, which records
`PASS assertions=10`. GPU timing and final performance evidence remain separate.
