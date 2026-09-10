# Final Inference AUTO adapter review

**Frozen source:** `tests/gemm_bi_fixed_performance.rs` SHA-256 `ef2b82d25e53495a6a79cbc60c1e1695dbac4a44c5fc8b0970f4ff3b5a1588cb` — independently matched both the worktree file and `inference-final-auto-candidate-ef2b82d2.rs`.  
**Reviewed diff:** `inference-final-auto-review.diff`, limited to the Inference adapter file.  
**Review boundary:** static source review only; no build, host test, GPU job, source/index mutation, selector review, or cohort review.

## Findings

### 1. [High] Captured AUTO symbol proof is substring-based and can accept a different physical route

At `tests/gemm_bi_fixed_performance.rs:14373-14380`, the adapter resolves the selected tile's expected symbol but checks it with:

```rust
auto_inventory.contains(expected_auto_symbol)
```

`auto_inventory` is serialized JSON, so this is not exact symbol membership. There are concrete prefix collisions in the admitted SM120 inventory:

- `gemm_bi_nn_sm120_tma_tf32_v1_m64n64_bk32_s2`
- `gemm_bi_nn_sm120_tma_tf32_v1_m64n64_bk32_s2_producer_warp`
- `gemm_bi_nn_sm120_tma_tf32_v1_m64n64_bk32_s2_pair_store`

If AUTO reports the base tile while capture contains either suffixed route, the assertion passes. That defeats the required link between returned production `FixedTile` and actual captured GPU work.

Minimal fix: test an exact serialized symbol token, including both delimiters, for example `auto_inventory.contains(&format!("\"symbol\":\"{}\"", fixed_sm120_tf32_bd_json_escape(expected_auto_symbol)))`, or return structured symbol names from the inventory helper and compare equality. Add a native regression test using the three colliding names.

### 2. [Medium] GPU UUID metadata is not bound to the CUDA device being benchmarked

At lines 14182-14194, the adapter runs an unfiltered global `nvidia-smi --query-gpu=uuid` and records `.lines().next()`. The benchmark device is separately opened as CUDA ordinal 0. On a multi-GPU host, especially with `CUDA_VISIBLE_DEVICES` remapping, the first global `nvidia-smi` row need not be CUDA ordinal 0. The record can therefore attach valid timings and CC/SM data to the wrong UUID.

Minimal fix: obtain the UUID for CUDA ordinal 0 through the CUDA driver (`cuDeviceGetUuid_v2`), as the existing repository quiet-GPU helpers do, then optionally use that UUID as the `nvidia-smi -i` selector. Add a native formatter/selection contract where practical; do not fix this by assuming physical NVIDIA index 0.

### 3. [Medium] The final record dropped the production tuning-table revision

The replaced adapter emitted `"tuning_table_revision": TUNING_TABLE_REVISION`; the final per-order record at lines 14480-14498 and completion record at lines 14550 onward omit it. This benchmark is evidence for actual production AUTO selection, so the dispatch-table revision is part of the route provenance. Git/source digests make the run recoverable, but silently dropping the existing explicit revision makes cross-receipt comparisons needlessly ambiguous.

Minimal fix: emit `tuning_table_revision` in the common metadata or in both the timing and completion records, and pin it in the focused metadata test.

## Spec-compliance verdict: NEEDS FIX

Finding 1 leaves the required physical graph-route proof fail-open, and finding 2 can misidentify the measured device. Correct both before treating output as final evidence. Finding 3 should be restored with the final provenance fields.

The remainder of the requested behavior is present and correctly scoped:

- all seven required row/comparator views use the specified input/output dtypes, policies, cuBLAS compute modes, and tolerances;
- the custom arm calls production `fixed_forward`; there is no forced launch arm, and any forced-tile filter is rejected;
- five hot cells, bias/no-bias, eager/whole-graph, and both mirrored orders are selectable with strict existing parsers;
- AUTO and vendor graph objects/buffers remain alive, capture is outside timing, biased vendor capture includes the typed broadcast plus GEMM, and graph replays are poisoned and compared twice against eager raw bytes;
- independent Pedantic/F32 reference checks, eager repeat checks, post-timing raw checks, per-arm calibrated iterations, raw samples, sample-wise `AUTO/vendor` ratios, and p50/p95 values are retained;
- the final default is 21 windows/order, filtered counts are derived from selected dimensions, and the full inventory is asserted at 280 records;
- CUDA 13.2, known NVRTC, CC8.9/142SM or CC12.0/170SM, external `GEMM_BI_FINAL_AUTO_GIT_SHA`, toolkit paths, compiler/artifact digests, driver identity, tile, and graph inventories are fail-closed or emitted as required.

## Code-quality verdict: NEEDS FIX

The adapter otherwise reuses the existing typed row, launch, graph, filter, error, and timing helpers cleanly. The ratio helper validates equal nonempty positive finite samples, and focused tests pin row/dtype inventory, the 21-window override, full count, and ratio orientation. However, a string-substring assertion is too weak for identity evidence, global first-row UUID selection is brittle provenance code, and dropping an existing selector revision reduces auditability. These are localized fixes; no redesign or broader adapter change is needed.

## Evidence boundary

The root owner's reported RED phase and in-flight host/GPU checks were not used as proof of GREEN behavior here. This verdict is based on the frozen source/diff and requirements only.
