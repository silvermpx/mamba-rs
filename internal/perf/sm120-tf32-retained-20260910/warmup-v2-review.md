# SM120 selector V2 frozen-delta review

**Frozen post-image SHA-256:** `2c7bd723ac3caed76c136df5226c8017f34ce475340a504f7095de2baf2d6b01` — independently matched `tests/gemm_bi_sm120_tf32_selector_qualification.rs`.  
**Reviewed artifact:** `sm120-warmup-v2-review.diff` (one file, 105 additions / 5 deletions).  
**Scope:** static review only. No source/index mutation, build, or GPU execution was performed.

## Findings

No blocking or non-blocking findings in the frozen delta.

## Spec-compliance verdict: PASS

The V2 correction implements the approved protocol exactly:

- retains the calibration and mandatory timed quiet gate;
- adds exactly four fixed post-quiet paired warmup cycles before discovery;
- uses calibrated `window_iterations` and the actual eager/graph path;
- executes `candidate → scalar → scalar → candidate` each cycle, covering both contexts in both order positions;
- reseeds both arms in the warmup helper, after which the unchanged `paired_samples` reseeds again before recording;
- leaves the 21-window discovery, 101-window final cohort, percentile thresholds, numerical checks, and admission logic unchanged;
- bumps the artifact schema to `MambaBiSm120Tf32SelectorQualificationV2` and records both warmup count and order in the completion record;
- adds focused source-contract checks for `timed_preflight < paired_warmup < discovery_samples`, count 4, and the symmetric call order.

The fixture-only G10 change also matches `final-auto-benchmarks-brief.md`: `PROJECTION_CELLS` changes from 45 to 46 and adds only `tn_m8192_k128_n128` with `Tn`, public dimensions `(8192,128,128)`, alpha 1, beta 1, and no bias. Its focused test pins the derived TN exact key `(128,128,8192)`. The existing `tn_split_candidate`, candidate inventories, gates, and bias derivation are unchanged.

## Code-quality verdict: PASS

The helper is small, single-purpose, fallible in the same style as the existing timing helpers, and avoids allocating or retaining warmup samples. Its fixed loop prevents data-dependent retries. The call is placed directly after the timed quiet gate and directly before recorded discovery, making the intended lifecycle explicit. Protocol metadata distinguishes new evidence from archived V1 receipts.

The added source-contract tests are appropriately narrow for a test harness whose measurement order is itself part of the evidence protocol. The frozen diff applies cleanly in reverse, `git diff --check` reports no whitespace errors, and static inspection confirmed the exact helper call sequence and G10 fixture literals.

## Verification boundary

This verdict covers static spec compliance and code quality only. Host-test and one-shot GPU results being run by the root owner were not available to, executed by, or inferred by this reviewer. G06/G09 admission remains conditional on their predeclared one-shot V2 receipts passing all unchanged gates; prior V1 receipts remain immutable failures.
