# Half NN Fast alignment diagnostic

The earlier NN7 ratios were inflated by benchmark guard alignment, not a
44–56% production speed improvement. F16 d768-out `(2048,1536,768)`, CUDA13.2
RTX6000 Ada; three graph samples per setting, five GEMMs per timed sample:

| Logical pointer mod256 | cuBLAS algorithm flag | Candidate us | Fast us |
| --- | --- | ---: | ---: |
| 16 | DEFAULT | 45.5–46.0 | 85.4–85.7 |
| 16 | DEFAULT_TENSOR_OP | 45.4–46.8 | 85.7–86.8 |
| 0 | DEFAULT | 38.9–39.8 | 48.3–48.6 |
| 0 | DEFAULT_TENSOR_OP | 39.0–39.8 | 48.5–48.8 |

The aligned Fixed S3 candidate is still about18–20% faster in this diagnostic.
That is not a once7 admission or a result for the other three provisional
NN7 winners. Alignment has a material effect on both implementations; the
cuBLAS algorithm flag does not explain this discrepancy. All inputs/outputs
are guarded, actual pointer mod256 is logged, reset is outside each five-call
window, and output/self-repeat bits are checked. No production change.

Exact `ada_half_nn_d768_out_fast_denominator_diagnostic` passed with4 records;
all12 ratios independently replayed, quiet PRE/DRAIN, stable private cache.
Measured typed source SHA256
`cdcc889e51ed2cd144f982c81590feceb1e0afb89deeb59a8df2687dcc650892`.
[Raw](evidence/cuda132/test.log) SHA256
`8984f1412818272dd50cc317dce6cd6efd2c9e44bb9bba5519a6f3de997bcd41`.
New performance fixtures must retain256-byte GPU allocation alignment;
misaligned views remain correctness tests, not the sole release denominator.
