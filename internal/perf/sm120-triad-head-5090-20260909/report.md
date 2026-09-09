# RTX 5090 SM120 production-dispatch snapshot — 2026-09-09

This snapshot was measured on the live RTX 5090 box at committed branch state
`3f4c9892` with CUDA 13.2, release code, 21 windows, path order `ab`, and five
preceding quiet samples at 0% GPU / 0% memory-controller utilization. Ratios
below are `cuBLAS p50 / production AUTO p50`, so values above `1.00x` favor
Mamba.

## Evidence integrity

- `full742-auto-w21.log`: 1,484 JSON rows, 742 unique cells, exactly one eager
  and one graph row per cell.
- All 742 pairs report `eager_graph_equal=true`; physical nodes, launch count,
  module owner, symbol, and launch digest agree between eager and graph.
- `vendor45-w21.log`: 90 JSON rows, 45 unique dtype/op/shape cells, exactly one
  `cublas_fast` and one `cublas_pedantic` denominator per cell.
- Every row is release, CC 12.0, and 21 windows. Both Rust tests passed.
- The canonical release comparison is 4 policy routes x 3 operations x 5
  shapes x 2 paths x 2 denominators = 240 comparisons.

## Canonical production ratios

| policy | path | denominator | wins | within 5% | median | worst | worst cell |
|---|---|---:|---:|---:|---:|---:|---|
| BF16 TC | eager | cuBLAS Fast | 6/15 | 8/15 | 0.98x | 0.31x | NT/d128-out |
| BF16 TC | graph | cuBLAS Fast | 9/15 | 14/15 | 1.06x | 0.82x | TN/Prism |
| F16 TC | eager | cuBLAS Fast | 6/15 | 8/15 | 0.98x | 0.31x | NT/d128-out |
| F16 TC | graph | cuBLAS Fast | 10/15 | 14/15 | 1.06x | 0.82x | TN/Prism |
| deterministic TF32 | eager | cuBLAS Fast | 11/15 | 14/15 | 1.07x | 0.93x | TN/d768-out |
| deterministic TF32 | graph | cuBLAS Fast | 10/15 | 12/15 | 1.07x | 0.81x | NT/d128-out |
| exact F32 | eager | cuBLAS Fast | 2/15 | 6/15 | 0.91x | 0.43x | TN/d128-in |
| exact F32 | graph | cuBLAS Fast | 2/15 | 4/15 | 0.82x | 0.41x | TN/d128-in |
| BF16 TC | graph | cuBLAS Pedantic | 15/15 | 15/15 | 3.60x | 1.63x | NN/d128-in |
| F16 TC | graph | cuBLAS Pedantic | 15/15 | 15/15 | 3.74x | 1.92x | NT/d128-out |
| deterministic TF32 | eager | cuBLAS Pedantic | 14/15 | 15/15 | 1.38x | 0.98x | NT/d128-out |
| deterministic TF32 | graph | cuBLAS Pedantic | 12/15 | 14/15 | 1.37x | 0.79x | NT/d128-out |
| exact F32 | eager | cuBLAS Pedantic | 11/15 | 12/15 | 1.06x | 0.38x | TN/d128-in |
| exact F32 | graph | cuBLAS Pedantic | 10/15 | 11/15 | 1.05x | 0.37x | TN/d128-in |

The omitted eager half-vs-pedantic rows are 12/15 wins for both BF16 and F16,
with medians of 3.58x and 3.61x. The small eager half cells expose fixed host
dispatch/launch overhead; their graph counterparts are the relevant steady
replay comparison and are within 5% of Fast in 14/15 cells.

## Change from the saved 2026-09-06 SM120 snapshot

The same canonical 120 production rows were compared with
`../sm120-triad-current21-20260906/cuda-13.2/auto60-v1.log` by
`(cell_id, path)`. Eleven logical cells changed launch count/owner/symbol; all
eleven are faster in both eager and graph. Deterministic-TF32 median speedup is
1.277x eager and 1.251x graph. Across all four policies, every canonical row is
within 5% of the old snapshot; the worst observed delta is 0.9969x and is
measurement noise. Physical digests were not used as the semantic-change key
because the newer route/identity epochs intentionally reframe them.

## Raw evidence hashes

- `build.log`: `20aeb7c1dc40db3440dd02a084f4fc564c4f51f1938128cfa0d2abbd0c373852`
- `full742-auto-w21.log`: `303890f9bdb2f7b3426a07c06c45afaaae61fc29a2c5a017c4d8dcdda406befa`
- `vendor45-w21.log`: `59f54de2b256a85966adcd12e0b7b9795708536421788285328834ef5e4082f5`

This snapshot is a production-route audit and same-box performance baseline;
it is not evidence for the uncommitted Ada Module-B admission that followed.
