# Final Inference AUTO fix1 scoped re-review

**Frozen post-image SHA-256:** `c92442da094de530e5388960b343f82eec163766c6e5ceafd4043f0d1d17de73` — independently matched both `tests/gemm_bi_fixed_performance.rs` and `inference-final-auto-fix-c92442da.rs`.  
**Comparison base:** frozen `inference-final-auto-candidate-ef2b82d2.rs`.  
**Scope:** only the three findings in `inference-final-auto-review.md`; no broader adapter review, source/index mutation, build, or GPU execution.

## Finding dispositions

1. **Exact graph-symbol collision — ADDRESSED.** `fixed_production_auto_inventory_has_exact_symbol` searches for the complete serialized token `"symbol":"<escaped-name>"`, including the closing quote. A longer suffixed symbol therefore cannot satisfy a base-symbol query. `fixed_production_auto_graph_symbol_membership_is_exact` covers the concrete SM120 base, producer-warp, and pair-store collision set and proves the base is absent when only the two suffixed routes are present. The AUTO graph assertion now uses this helper.

2. **Visible CUDA ordinal UUID — ADDRESSED.** `fixed_production_auto_cuda_uuid(0)` initializes the CUDA driver, resolves CUDA ordinal 0, queries that device's UUID with the driver API, and formats it in canonical `GPU-...` form. The adapter records this value for the same ordinal passed to `GpuDevice::new(0)`, eliminating dependence on global `nvidia-smi` row order and remaining correct under CUDA visibility remapping. The formatter has a deterministic native test.

3. **Tuning revision metadata — ADDRESSED.** `fixed_production_auto_tuning_metadata()` emits `"tuning_table_revision":45` from `TUNING_TABLE_REVISION`. It is inserted into `device_metadata`, which is embedded in every timing record and the completion record. The focused test pins the emitted field and current revision.

No new defect was found within this scoped delta. The prior-to-current delta changes only helpers, focused tests, UUID acquisition, metadata composition, and the exact AUTO symbol assertion needed for these findings.

## Spec-compliance verdict: PASS

All three requested receipt-integrity corrections are implemented. Exact physical graph-route proof is no longer prefix-fail-open; device provenance is bound to the benchmarked visible CUDA ordinal; and production tuning-table revision is present in both record classes.

## Code-quality verdict: PASS

The fixes are small and reusable, use the repository's existing CUDA UUID formatting/query pattern, escape symbol text before matching serialized inventory, and include focused regression tests for the concrete collision and metadata formats. No production dispatch or unrelated benchmark behavior was changed.

## Evidence boundary

This is a static re-review. The root owner's in-flight host tests and focused GPU smoke were not executed or inferred by this reviewer.
