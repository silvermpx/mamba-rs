# Task6B fix round 1 scoped re-review

### Important I1 — bind exit and telemetry evidence to this attempt and parse exact closure records — ADDRESSED

`internal/perf/ada-half-s3-paired-20260907/analyze.py:157-189` removes the prior substring predicate and fail-closes on exactly one ordered eight-record attempt. It parses PRE, COMMAND, COMMAND_EXIT, POST, RUN_RESULT, WRAPPER_COMPLETE, WRAPPER_EXIT, and OUTER_SSH_EXIT in their emitted order; exact line count and full-match exit parsing reject extra/conflicting attempts, duplicate or missing records, suffixed markers, and nonzero wrapper/SSH exits (`analyze.py:160-180`).

The same-attempt bindings are substantive:

- PRE and POST objects must equal the corresponding saved JSON objects, retain the correct phase labels, and pass the existing telemetry checks, including quiet PRE/no-apps and allowed residual POST utilization (`analyze.py:162-168`).
- COMMAND must decode to the exact bound binary and literal ignored-test argv; COMMAND_EXIT must be a complete anchored record and equal saved `test_exit` (`analyze.py:169-174,181`).
- RUN_RESULT must equal the loaded `result.json`, whose toolkit/binary/source and raw test-log hash are independently checked against the selected binding and files (`analyze.py:153-156,175-176`).
- Saved test/post exits and parsed wrapper/outer-SSH exits are passed together to the four-layer zero-exit closure before any matrix result is returned (`analyze.py:181-184`).
- Accepted output records the corrected analyzer plus the exact SSH, binding, and qualification hashes as validator provenance (`analyze.py:185-189`).

The fix tests exercise the actual `verify_run` path. `test_validation.py:128-157` supplies conflicting and nonzero exits, duplicate/missing/suffixed records, a foreign attempt, wrong binary/test command, mismatched result/telemetry/phase, swapped PRE/POST, and an added failure marker. `test_validation.py:159-171` separately mutates the saved PRE/POST/result files, proving transcript-to-file binding rather than transcript-only consistency.

### New Breakage in the Fix Diff

None. Exact eight-line closure matches the unchanged wrapper's emitted run shape, while POST remains permitted to carry residual utilization. The adjacent independent-fixture extension from 21 to 101 windows preserves the literal round-nearest indices and distinguishes `advance101` from `admission`; it introduces no issue in the reviewed fix delta.

### Out-of-Scope Observations

M1 and M2 remain the previously classified Minor preflight gaps in untouched Rust and are explicitly deferred in the controller ledger to immediate Task6C. They do not alter this I1 verdict. Final archive/evidence admission is a separate controller gate and is not granted by this source re-review.

### Checks

- Read-only SHA-256 check matched the frozen fix identities: analyzer `0c1391f47ec0253720b66733e950b8f9ad67ba112027b81e2d67eda9a1e63f30`, tests `4461fe0d3562e27be393743c97fcd47016a9d75fbfa3cdffc69a430c3ecbe131`; unchanged run.py `ce0de6882fa45835ba81b90d02a6cc085ed32226e2462b72260ce9cacd0fd713` and Rust `f43a2a22dea7435716ff0c99d721e1d255ad9311f18e9f0b53a15280eafb582b` also match the package.
- Inspected, but did not rerun, `fix1-red.log`: both targeted test methods produced 19 meaningful old-validator failures across the malformed/mixed evidence cases.
- Inspected, but did not rerun, `fix1-green.log`: all 11 test groups passed on the corrected analyzer.
- Inspected, but did not rerun, `fix1-revalidation.log`: all 10 archived actual runs validate with the corrected analyzer hash and retain the reported six-cell decisions.
- No GPU, SSH, build, benchmark, or test command was run; no source, index, HEAD, branch, binary, raw log, or derived evidence was mutated by this review.

### Verdict

**Fix round:** All findings addressed, no new Critical/Important breakage.
