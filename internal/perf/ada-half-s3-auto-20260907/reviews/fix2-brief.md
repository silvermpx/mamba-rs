# Task6C host fix2 — pin validator dependencies and exercise inherited gates

Read `ada-half-s3-auto-task6c-host-source-review.md` for exact Important
findings. This fix is owned by the same implementation worker at its next
safe run boundary. Only new Task6C analyze.py/run.py/test_validation.py and
report/evidence may change. All seven Rust files, binaries, CUDA, Task6B source
and historical evidence remain frozen. No GPU timing reruns or library rebuilds
are needed for these host-only validator fixes; revalidate saved post runs if
any have finished before the fix. Preserve original utility versions/hashes.

## I2: imported frozen validator/wrapper must actually be pinned

Reviewer finding: "the security/validity boundary rests on dynamically imported
Task6B code, but neither utility compares that code to an approved frozen digest.
The analyzer merely reports whatever analyzer hash it just executed. The wrapper
writes frozen_pre_wrapper_sha at build time, yet the inherited validate_binding
ignores that field, so a Task6B wrapper change between build and run is accepted
and executed. This makes frozen validator/wrapper imported unchanged an external
convention rather than a fail-closed contract and can silently change the meaning
of accepted post evidence."

Required pins before import/use:

- Task6B analyzer.py:
  0c1391f47ec0253720b66733e950b8f9ad67ba112027b81e2d67eda9a1e63f30
- Task6B run.py:
  ce0de6882fa45835ba81b90d02a6cc085ed32226e2462b72260ce9cacd0fd713

Check the wrapper digest against the existing build binding before and after
the measured run. Analyzer also rejects a foreign bound wrapper digest. Existing
all-three bindings already carry the correct field; do not rebuild/relabel them.
Use a narrow helper if useful, no broad refactoring. Cover actual digest mismatch
before executing imported code and mismatched binding digest with focused host
negative tests. Never edit the historical Task6B files to manufacture a test.

## I3: exercise physical and chronological gates through the post adapter

Reviewer finding: the named poison test currently changes captured-node bundle,
pointers, ABI, sixth-argument rejection, symbol and aggregate completion flags,
but not direct physical poison/raw-repeat/guard/numeric fields or chronology.

Add focused mutations which must fail through post verify:

- a wrong raw sample chronology index;
- physical repeat_bits=false;
- physical poison_upload_verified=false;
- physical guards=false;
- numerical_error nonfinite and above the existing dtype tolerance.

Retain current captured-argument/aggregate-closure tests and valid recomputed
own win/loss/mixed-p95 decisions. No tolerance changes or parser weakening.

## Covering verification and handoff

Run meaningful host RED against the current adapter/boundary, then GREEN for
the complete new host suite, recording exact commands/output/exits. These tests
are CPU-only and must not launch GPU work. Final hashes/report additions go into
the same Task6C report. Root will review only this host fix diff against I2/I3
and new breakage, then use the corrected validator on actual frozen Rust data.
No change to original all3 functional qualification or post1/101 task scope.
