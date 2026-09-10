### Spec Compliance

- ❌ Issues found: the production repair matches the required selection-level behavior, but the task is not fully compliant because the required loader/availability regression does not prove that the specialized binding and every accepted exact-F32 sibling remain live (`src/mamba_ssm/gpu/gemm_bi_triad/modules.rs:11980`), and the claimed frozen RED source does not match the current S2-preserving test (`src/mamba_ssm/gpu/gemm_bi_triad/dispatch.rs:13201`; `internal/perf/sm120-symbol-reachability-20260910/red/tests.log:16`).
- ⚠️ Cannot verify from diff: the required actual CUDA 13.0 production probe and the two affected policy-cell runtime receipts are explicitly pending root execution (`tests/gemm_bi_tf32_contract.rs:13647`). Source review finds the probe pins the generic symbol, M128N64 tile, grid 768, fixed-split numeric contract/ownership, sampled independent CPU FMA/fold bits, complete eager-repeat/graph output equality, and red zones (`tests/gemm_bi_tf32_contract.rs:13619`, `tests/gemm_bi_tf32_contract.rs:13647`).

### Strengths

- `src/mamba_ssm/gpu/gemm_bi_triad/contract.rs:1832` represents exclusions as a compact typed inventory-position mask with an empty default; `src/mamba_ssm/gpu/gemm_bi_triad/modules.rs:9027` propagates only the specialized loader's known exclusions into its retained binding.
- `src/mamba_ssm/gpu/gemm_bi_triad/dispatch.rs:3871` filters the measured arm before fallback, preserves the measured split count on the retained generic tile, filters that generic arm too, and naturally reaches the existing scalar floor. The shared floor means Exact and unmatched AllowTF32 receive the same behavior.
- `src/mamba_ssm/gpu/gemm_bi_triad/dispatch.rs:3441` rejects a forced excluded exact route with the resolved symbol and precise toolkit-exclusion message before preparation.
- The diff is tightly scoped: CUDA arithmetic, resource gates, ABI/identity/tables, launch lookup, specialized-module enablement, and performance harnesses are unchanged; the extra literals in launch/selector tests are mechanical defaults.
- Focused outside-diff check — named risk: split normalization could alias distinct symbols. Checked `src/mamba_ssm/gpu/gemm_bi_triad/contract.rs:3084`: all 12 literal inventory entries are unique by `(op, tile, kvec)`, and split count is launch-only, so normalized lookup is unambiguous.
- Focused outside-diff check — named risk: an unexplained missing function could become a silent exclusion. Checked `src/mamba_ssm/gpu/gemm_bi_triad/modules.rs:9841`: symbol-load/query failures still return errors, while only explicit resource-admission failures enter the exclusion list; the unchanged launch map therefore remains the fail-closed backstop.

### Issues

#### Critical (Must Fix)

- None.

#### Important (Should Fix)

- `src/mamba_ssm/gpu/gemm_bi_triad/modules.rs:11980` — the new loader-mask test exercises only the pure symbol-to-mask helper and checks one sibling plus one cross-operation route. It never exercises availability propagation, never asserts `specialized` remains bound, never checks the accepted function map, and does not prove that all other 11 exact inventory entries remain unmasked. This misses the brief's explicit loader/availability regression and leaves the most delicate integration seam unguarded. Add a narrow testable seam around binding propagation (without CUDA execution) and assert one excluded exact symbol, all 11 sibling bits clear, accepted-map preservation, and `availability.specialized.is_some()`.
- `src/mamba_ssm/gpu/gemm_bi_triad/dispatch.rs:13201` — the current fallback test requires `splits: 2`, but the purported frozen RED receipt expected `splits: 1` (`internal/perf/sm120-symbol-reachability-20260910/red/tests.log:16`). Thus the supplied RED run is not evidence for the final measured-split-preservation requirement, and the report's claim that the RED test source was frozen is inaccurate. Supply an immutable pre-implementation source/hash and RED receipt that already contains the S2 expectation if one exists; otherwise correct the report and treat S2 TDD evidence as missing rather than frozen.

#### Minor (Nice to Have)

- `internal/perf/sm120-symbol-reachability-20260910/green/tests.log:7` — the GREEN run emits an expected unmatched-cohort warning. The behavior is intentional, but the receipt is not pristine and the test does not assert the diagnostic. Capture/assert the expected warning or route it through a test sink so future unexpected warnings remain visible.

### Assessment

**Task quality:** Needs fixes

**Reasoning:** The production code is compact and appears behaviorally correct, including the crucial measured-S2 generic fallback and fail-closed semantics. Approval is blocked by one explicitly required integration-test gap and by RED evidence that does not correspond to the final S2 assertion; CUDA 13.0 runtime acceptance also remains with root.

---

## Fix Round 1 Re-review

### Spec Compliance

- ✅ Spec compliant. Both prior Important findings are addressed, the actual CUDA 13.0 production probe and both affected policy cells passed, and no fix-round production regression was found.

### Prior Important Findings

- **ADDRESSED — loader/availability preservation:** `src/mamba_ssm/gpu/gemm_bi_triad/modules.rs:103` adds a narrow retention seam used by the production specialized-loader path at `src/mamba_ssm/gpu/gemm_bi_triad/modules.rs:9051`. The amended regression at `src/mamba_ssm/gpu/gemm_bi_triad/modules.rs:12000` passes the accepted function map, binding, known exclusions, and successful qualification through that same seam; it asserts the specialized binding remains present, the map is unchanged, the unrelated TF32 sibling remains, exactly the named exact-F32 inventory bit is masked, and all 11 accepted exact siblings remain mapped and unmasked.
- **ADDRESSED — RED/S2 provenance:** the updated implementation report now explicitly states that immutable RED predates the S2 numeric requirement, covers only exclusion-aware selection/forced rejection, and is not claimed as S2 TDD evidence. That is the accurate fallback requested by the original finding. The final S2 behavior is instead supported by the fresh 5/5 host receipt and the actual CUDA 13.0 probe: 1/1 in 49.25s with the measured symbol excluded, generic M128N64/kvec/S2, grid 768, eight independent CPU-bit samples, complete eager repeat/graph equality, and guards. The two-cell matrix receipt also records both Exact and unmatched AllowTF32 using that symbol/tile/grid with eager/graph equality.

### Strengths

- `src/mamba_ssm/gpu/gemm_bi_triad/modules.rs:103` factors only the state transition the test must exercise and delegates the existing function-map/binding retention behavior, avoiding duplicated production logic.
- Focused new-breakage check — moving exclusion assignment after artifact qualification is safe: `src/mamba_ssm/gpu/gemm_bi_triad/modules.rs:10215` validates module/artifact identity and probes the already-filtered function map; it does not consult `sm120_fma_exclusions`. The final retained binding receives the mask through the new seam.
- The source-contract correction at `tests/gemm_bi_tf32_contract.rs:4137` is precise: structural tokens remain checked on masked source, while the required literal error text is checked on the raw forced-resolver scope at `tests/gemm_bi_tf32_contract.rs:4145`. Fresh receipts show that exact contract test and the existing selector integration target each pass 1/1; the manifest pins the amended contract test to SHA-256 `489fef58977d6db393640b1b4367c98b3e1bb4ba1d1c36112b7b00ca42ba214c`.

### Issues

#### Critical (Must Fix)

- None.

#### Important (Should Fix)

- None.

#### Minor (Nice to Have)

- No new fix-round Minor issue. The previously recorded synthetic unmatched-cohort warning remains deferred to the existing ledger and is not reopened by this scoped re-review.

### Assessment

**Spec-compliance verdict:** Approved

**Code-quality verdict:** Approved

**Task quality:** Approved

**Reasoning:** The fix closes the requested test seam without changing selection, launch, arithmetic, identity, resource gates, or accepted function ownership. Evidence is internally consistent, the historical RED limitation is now stated accurately, and focused host plus CUDA 13.0 runtime receipts cover the repaired behavior.
