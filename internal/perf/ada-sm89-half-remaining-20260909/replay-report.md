# Ada SM89 half remaining: independent replay

Source receipt: `half-cuda132-trace-fixed-run.log`

- SHA-256: `a84ad1f10197b3193bd0d5fe2883a6d68d2ae1b9c3adc25386d3b85cce807855`
- Live result recorded by the receipt: `1 passed`, `0 failed`, `209.73s`.
- Replay scope: all raw timing observations, seven cell decisions, six pair decisions, and two tournaments. No GPU run was repeated.

## Replay method

Each seven-window screen was recomputed directly from its four raw samples:

- `ABBA = (A0 + A3) / (B1 + B2)`
- `BAAB = (A1 + A2) / (B0 + B3)`
- nearest-rank `p50 = sorted[ceil(7 * 0.50) - 1]`
- nearest-rank `p95 = sorted[ceil(7 * 0.95) - 1]`
- strict win requires `p50 < 0.99` and `p95 < 0.99` in every one of `eager/ABBA`, `eager/BAAB`, `graph/ABBA`, and `graph/BAAB`.

Parsed inventory: 116 JSON records, 80 screens (28 actual AUTO, 28 Fast, 24 tournament), 7 cell decisions, 6 pair decisions, and 2 tournament decisions. Every screen contained exactly seven finite positive four-sample observations. Maximum absolute difference was `4.906978157137587e-10` between raw replay and rounded screen quantiles, and `1.078470646120877e-11` between raw replay and decision strata.

## Cell results

Ratios are `candidate/comparator`; lower is faster. Ranges are across the four strata.

| Route / dtype / shape | Candidate | Actual AUTO p50 min-max | Actual AUTO worst p95 | Fast p50 min-max | Fast worst p95 | Decision |
|---|---|---:|---:|---:|---:|---|
| TN d768-in F16 `(2048,768,3072)` | `tc64_bk64_s2_regpipe_vec2_epilogue` | 0.837648-0.843631 | 0.847157 | 0.956664-0.989887 | 0.997978 | AUTO win; Fast no |
| TN d768-in BF16 `(2048,768,3072)` | `tc64_bk64_s2_regpipe_vec2_epilogue` | 0.835032-0.841063 | 0.845670 | 0.950688-0.985580 | 0.993564 | AUTO win; Fast no |
| TN d768-out F16 `(2048,1536,768)` | `tc64_bk64_s2_compact_xor` | 0.823255-0.827726 | 0.833922 | 1.089086-1.126343 | 1.128188 | AUTO win; Fast no |
| TN d768-out BF16 `(2048,1536,768)` | `tc64_bk64_s2_regpipe_vec2_epilogue` | 0.809911-0.814992 | 0.823648 | 1.071996-1.106702 | 1.112342 | AUTO win; Fast no |
| TN Prism F16 `(4621,384,1928)` | `tc64_bk64_s2_compact_xor` | 0.950584-0.957797 | 0.961034 | 1.358885-1.412367 | 1.419830 | AUTO win; Fast no |
| TN Prism BF16 `(4621,384,1928)` | `tc64_bk64_s2_compact_xor` | 0.950939-0.958028 | 0.963209 | 1.359265-1.415873 | 1.423911 | AUTO win; Fast no |
| NN d768-in F16 `(2048,768,3072)` | `fixed_sm89_tc128_s3` | 0.673337-0.708286 | 0.720660 | 0.762108-1.046358 | 1.062362 | AUTO win; Fast no |

All seven `actual_auto_win` fields independently recompute to `true`. All seven `fast_win` and `promotion` fields independently recompute to `false`. The NN d768-in candidate beats Fast in eager strata but loses in graph strata, so it is correctly not a strict four-strata Fast win.

## D768-in TN tournaments

Pair representative ratios are geometric means of the four pairwise p50 values. Ranking was independently recomputed from pairwise wins (descending), then summed log-ratio score (ascending).

| Dtype | Winner and replayed order | Pairwise wins | Pair representative ratios: compact/regpipe, compact/vec2, regpipe/vec2 |
|---|---|---|---|
| F16 | `regpipe_vec2 > regpipe > compact` | compact 0, regpipe 1, vec2 2 | 1.000080941, 1.012789171, 1.012815865 |
| BF16 | `regpipe_vec2 > compact > regpipe` | compact 1, regpipe 0, vec2 2 | 0.999936652, 1.013225569, 1.012849261 |

Both tournament winners and complete orders match the receipt. The replay therefore confirms all seven actual-AUTO decisions and both tournaments without relying on the emitted summary fields.
