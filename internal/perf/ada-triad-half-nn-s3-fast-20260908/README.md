# Ada half Triad NN: Fixed S3 reuse, short Fast screen

CUDA13.2 / RTX6000 Ada, 2026-09-08 08:46Z. This is candidate discovery,
not dispatcher admission or a claim that all NN cells beat cuBLAS Fast.
The already-loaded `gemm_bi_nn_fixed_sm89_tc128_s3_v1_{f16,bf16}` kernels
are reused unchanged. No production CUDA or SM120 routes changed.

## Observed paired ratios

Each range contains the ABBA and BAAB median candidate/Fast ratios.
Lower is better; eager and CUDA Graph are independently paired.

| Precision / NN shape | Eager p50 | Graph p50 | Screen outcome |
| --- | ---: | ---: | --- |
| F16 d768-in | 0.986–0.986 | 1.224–1.235 | Stop |
| F16 d128-out | 0.953–0.957 | 1.893–1.919 | Stop |
| BF16 d128-out | 0.900–0.909 | 1.651–1.678 | Stop |
| F16 d768-out | 0.442–0.458 | 0.559–0.561 | Provisional advance |
| BF16 d768-out | 0.439–0.461 | 0.548–0.553 | Provisional advance |
| F16 Prism | 0.535–0.540 | 0.649–0.651 | Provisional advance |
| BF16 Prism | 0.536–0.540 | 0.643–0.649 | Provisional advance |

**Baseline discrepancy identified: pointer alignment.** For F16 d768-out, graph medians
were candidate 51.056/51.312 us versus Fast 91.040/91.456 us. Historical
Fast eager was about47 us and the historical custom route about55 us.
Those are different cohorts, but the discrepancy is large enough to block
a broad champion claim. A separate one-cell diagnostic varies guard alignment,
cuBLAS DEFAULT vs DEFAULT_TENSOR_OP, and uses five-call graph windows.
The [alignment diagnostic](../ada-triad-half-nn-fast-diag-20260908/README.md)
finds an18–20% aligned F16 d768-out advantage, not the earlier44–56% figure.
The raw four `advance` decisions must not be treated as final admissions.

## Contract and evidence

- Literal cuBLAS `gemm_ex`, native-half A/B/C, FP32 compute,
  DEFAULT_TENSOR_OP, NN alpha1/beta0/no bias; not Pedantic.
- Guarded 16-byte-aligned pointers, reseed before every single-GEMM observation.
- Candidate/current eager2+graph2 raw bits pass; current is forced TC64 for
  d128-out and TC128 otherwise, **not a fresh public-AUTO qualification**.
  Fast has its own finite/nonzero/repeated-bit checks.
- Exact test `ada_half_nn_fixed_s3_shape_fast_batch_discovery_once7` passed:
  14 resource records, 56 bit records, 28 screens and seven decisions.
  All196 four-observation brackets (784 observations) and p50/p95 were
  independently replayed from raw JSON. Quiet PRE/DRAIN and stable private cache.
- Measured main SHA256 `ce3c8b3a1d4d7fbeaf8a0ab9d6b8c2618a77012d08fabf1fea940f892075f70a`.
  Binary SHA256 `721f6d3b36686ba0546740eb024e988d95376f8a008b075aa5e60a85d86c5b77`.
  Raw SHA256 `d5fb6e186bf22b612269a760e9172249bdf737d06ab9b0a997c8cb3decee45b1`.
- [Raw observations](evidence/once7-cuda132/test.log),
  [exact command and receipt](evidence/once7-cuda132/command.json).
  No BF16 d768-in, d128-in, other toolkit, cold-cache or fresh5090 result
  is inferred from this batch. Valid losers are not rerun.
