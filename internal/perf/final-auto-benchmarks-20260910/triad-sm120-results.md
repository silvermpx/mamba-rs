# RTX 5090 production AUTO Triad, CUDA 13.2

The full final paired packet `triad-sm120-full/` passed:66 cells,81 cuBLAS
comparator views,324 pair records plus completion,21 windows per mirrored
order. All rows, statistics and metadata independently replay with
`verify-triad.jq`, including actual256-byte vendor alignment. The completion
digest matches the exact324 pair-record lines:
`22ac7e5fb6cf9f801f04e430e9e134ba88d77ccd6283e0526a1179eb0fad75fb`.

Measured production source:
`732c11462dd9a47376e6d15b36040ca69bde3e34`.
Benchmark source SHA-256:
`de4b689bebdb35d0e2dd201cbdd774ab7b89b6896e3026d0da92e822566c7f45`.
GPU UUID, driver595.84, compiler and actual physical node identities are
recorded with the raw data.

## Aggregate comparison

Speedup is cuBLAS/AUTO; above1 favors AUTO. Per-cell/path ratios pool the42
raw paired observations, use the lower empirical median, then invert.
Aggregates are equally weighted geometric means. Win counts describe
medians, not a confidence-based qualification decision.

| AUTO precision | cuBLAS comparator | Cells | Eager speedup | Graph speedup | Median wins, eager / graph |
|---|---|---:|---:|---:|---:|
| BF16 | Fast COMPUTE_32F |15|0.689×|0.929×|6 /6|
| F16 | Fast COMPUTE_32F |15|0.690×|0.929×|6 /6|
| Deterministic TF32 | Fast TF32 |21|1.051×|0.939×|18 /10|
| Exact F32 | Fast TF32 |15|0.793×|0.698×|0 /0|
| Exact F32 | Pedantic F32 |15|1.019×|0.919×|11 /10|

This does not establish a whole-Triad win against cuBLAS Fast. TF32's eager
aggregate is faster, but its graph aggregate remains slower. Several exact
F32 cells are close to Fast parity without exceeding it. Exact-F32 and
Fast-TF32 input semantics differ; Pedantic F32 is reported separately.

As on Ada, TN half inputs produce F32 outputs, C is reset before each event
window, and repeated beta1 accumulation is measured within that window.
NN/NT use beta0 overwrites. Whole graphs include all selected physical
launches. Performance evidence is separate from numerical qualification:
the physical `eager_graph_equal` flag compares launch metadata, not output
words.

The subsequent live CUDA12.8/13.0 cohort additions reuse unchanged CUDA
bodies. Removing true G10 from the current CUDA13.2 cohort cannot change
this66-cell packet: G10 is not in its inventory, and the other current
CUDA13.2 selections are preserved. The actual measured SHA above is kept;
post-admission24-case binding checks provide separate final-dispatch proof.
