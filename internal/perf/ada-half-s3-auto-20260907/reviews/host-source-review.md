### Spec Compliance

- ❌ Issues found: the post43 arm/direction adapter, CUDA13.2-only policy, exact two-dtype/window closure, routing43 versus compiled-qualification42 split, and eight-record same-attempt transcript checks are implemented, but the utilities do not enforce that the imported Task6B analyzer/wrapper are the reviewed frozen versions (`internal/perf/ada-half-s3-auto-20260907/analyze.py:13`, `internal/perf/ada-half-s3-auto-20260907/analyze.py:148`, `internal/perf/ada-half-s3-auto-20260907/run.py:12`, `internal/perf/ada-half-s3-auto-20260907/run.py:98`, `internal/perf/ada-half-s3-auto-20260907/run.py:116`). The explicitly required synthetic chronology/raw-bit/poison/guard/numeric gate coverage is also incomplete (`internal/perf/ada-half-s3-auto-20260907/test_validation.py:167`).
- ⚠️ Cannot verify from this source package: GPU functional qualification, post window1/window101 results, archived same-attempt evidence, manifest closure, and lane release are intentionally still pending and must be judged only from the later final-evidence package.

### Strengths

- `internal/perf/ada-half-s3-auto-20260907/analyze.py:44` validates post records first and adapts a deep copy only; its exact `Swizzle -> AUTO`, `AUTO -> S3`, `Fast -> Fast` and direction mappings preserve the frozen pre42 validator's semantics without rewriting raw evidence.
- `internal/perf/ada-half-s3-auto-20260907/analyze.py:59` enforces schema/stage/routing43, CUDA13.2, windows1/101, exact BF16+F16 closure, and historical compiled qualification42 before delegation. The inherited verifier then supplies exact physical, captured-argument, chronology, quantile, and eight-configuration closure.
- `internal/perf/ada-half-s3-auto-20260907/analyze.py:90` binds the saved result and raw-log hash to an exact eight-line PRE/command/test-exit/POST/result/wrapper/outer-SSH transcript and rejects any nonzero exit layer (`internal/perf/ada-half-s3-auto-20260907/analyze.py:111`, `internal/perf/ada-half-s3-auto-20260907/analyze.py:139`).
- `internal/perf/ada-half-s3-auto-20260907/run.py:30` rejects the named stale controls and all unrecognized `MAMBA_FIXED_ADA_*` controls; runtime policy is correctly limited to CUDA13.2, windows1/101, and both dtypes (`internal/perf/ada-half-s3-auto-20260907/run.py:107`).
- The loss/mixed-p95 test uses genuine recomputed timing ratios and confirms admission remains false rather than expecting validation failure (`internal/perf/ada-half-s3-auto-20260907/test_validation.py:130`). The three packaged file hashes exactly match the frozen package declarations.

### Issues

#### Critical (Must Fix)

- None.

#### Important (Should Fix)

- `internal/perf/ada-half-s3-auto-20260907/analyze.py:13`, `internal/perf/ada-half-s3-auto-20260907/analyze.py:148`, `internal/perf/ada-half-s3-auto-20260907/run.py:12`, `internal/perf/ada-half-s3-auto-20260907/run.py:98`, `internal/perf/ada-half-s3-auto-20260907/run.py:116`: the security/validity boundary rests on dynamically imported Task6B code, but neither utility compares that code to an approved frozen digest. The analyzer merely reports whatever analyzer hash it just executed. The wrapper writes `frozen_pre_wrapper_sha` at build time, yet the inherited `validate_binding` ignores that field, so a Task6B wrapper change between build and run is accepted and executed. This makes "frozen validator/wrapper imported unchanged" an external convention rather than a fail-closed contract and can silently change the meaning of accepted post evidence. Pin the reviewed Task6B analyzer and wrapper digests (currently `0c1391f47ec0253720b66733e950b8f9ad67ba112027b81e2d67eda9a1e63f30` and `ce0de6882fa45835ba81b90d02a6cc085ed32226e2462b72260ce9cacd0fd713`) before import/use, and verify the wrapper digest against the build binding both before and after the run. Add focused digest-mismatch negatives.
- `internal/perf/ada-half-s3-auto-20260907/test_validation.py:167`: the method named for poison coverage mutates only captured-node `bundle`, `pointers`, `abi`, `sixth_rejected`, the AUTO symbol, and aggregate completion flags. It never corrupts `poison_upload_verified`, `numerical_error`, sample `chronology`, or the physical `repeat_bits`/`guards` fields. Thus several gates expressly required to be exercised by synthetic negatives are only inherited code paths, not demonstrated through the post adapter. Add focused mutations for a wrong chronology index, `repeat_bits=False`, `poison_upload_verified=False`, `guards=False`, and non-finite/over-tolerance `numerical_error`; retain the existing aggregate raw-bit/guard closure mutations.

#### Minor (Nice to Have)

- None.

### Assessment

**Task quality:** Needs fixes

**Reasoning:** The post43 mapping and exact-attempt closure are compact and technically coherent, but the unpinned frozen dependencies undermine the adapter boundary, and the required adversarial gate coverage is incomplete. No tests were rerun; review used the frozen package, its matching source hashes, and one focused inspection of the reused Task6B validation boundary.
