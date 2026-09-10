# RTX 5090 Triad assembly smoke — 2026-09-10

CUDA 13.2, release build, seven windows, path order `ab`. This checks the
frozen integration snapshot, not a final release or a paired speedup claim.
Before launch, five successive samples showed 0% GPU and memory utilization
and 32,110 MiB free. The exact source digests are in `evidence/source.sha256`.

The existing `gemm_bi_deterministic_performance_matrix` test passed in 76.16 s:

- 66 unique cells, exactly 132 rows: one eager and one graph row per cell.
- 60 canonical cells: exact F32, deterministic TF32, F16 and BF16 across
  NN/NT/TN and d128-in/out, d768-in/out and Prism.
- Six additional deterministic-TF32 cells: NN/NT/TN underfill and large-deep.
- Every row reports `eager_graph_equal=true`.

Compared by cell/path with the saved `3f4c9892` snapshot, 65 of 66 cells retain
the same physical symbols, owners, tiles, grids, blocks and shared-memory
requirements. The only change is the previously admitted TF32 TN underfill
route: both eager and graph select
`gemm_bi_tn_sm80_mma_tf32_v1_m16n32_bk32_s4` instead of the older scalar
SplitM route. No Ada-specific kernel displaced a 5090 route in this cohort.

The log records absolute timings, but this run was not paired with the old
binary or a fresh vendor arm. Do not turn historical median ratios into a
new performance claim. Inference, the remaining matrix cells and final release
validation are outside this smoke's scope.

Primary raw log SHA-256:
`c438794c171d4c06bb3c822a0a6db3afe8a05605d0fe70dc1d9ff5ae23e1e501`.

## Subsequent focused verification

After the final Ada identity/graph fixes, the same source was compiled on
this host with CUDA13.2. CUDA-feature static tests pass joint7, half38,
half-source21, kernel-identity23 and route-inventory1. Both retained TF32
discovery targets compile. Logs and source hashes are under `static/`.

All 52 TF32 dispatcher tests also pass after correcting the three stale
24-cell/6-portable inventory assertions. The earlier failing logs are retained
beside `static/dispatch-final.log`. That last correction is test-only: no
production route, kernel body or qualification identity changed afterward.
Source and tests are committed in `70fc963a`.

Inference regression checks also pass:

- `bridge_fixed_ladder_bit_identical_to_triad`:14 forced FP16/BF16 cases,
  bit-for-bit against the corresponding Triad tiles (9.00s, earlier snapshot).
- `misaligned_subview_matches_aligned_bits`: BF16 aligned/misaligned safety
  comparison, final source (8.47s).
- `fixed_tile_matches_cpu_across_tails`:8 F32 shapes against CPU tolerance,
  final source (9.29s); this is not a CPU bit-exact assertion.

The initial extra-probe preflight stopped on a transient2% utilization
sample before executing the test. The later final-source runs obtained all
five idle samples and passed. Logs remain in `inference-initial/` and
`inference/`; no process was killed. These are focused regressions, not the
full Inference release matrix or new Inference speed measurements.
