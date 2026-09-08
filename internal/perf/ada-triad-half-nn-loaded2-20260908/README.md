# F16 NN d768-in: loaded Pipeline and Swizzle lose to retained S3

Ada / CUDA13.2, 2026-09-08. Existing Fixed holders were forced through
`fixed_forward_with_tile` on Triad NN shape (2048,768,3072), no bias.
No new CUDA kernel or production route was added.

| Candidate | Candidate/S3 p50 | Candidate/Fast p50 | Decision |
| --- | ---: | ---: | --- |
| Pipeline | 1.3047–1.3087 | 1.3324–1.3643 | Stop |
| Swizzle | 1.0775–1.0829 | 1.1028–1.1226 | Stop |

Both candidates match forced TC128 output bits on this target. Retained S3
is checked against the same reference; cuBLAS Fast has its own finite,
nonzero eager/graph output oracle. Ratios are independent candidate/S3 and
candidate/native-half-Fast pairs across eager/graph and ABBA/BAAB. These
losses do not change S3's previously measured 2–4% Fast gap or authorize
deletion of Pipeline/Swizzle on other cells or devices.

One focused build and exact test PASS:
`ada_half_nn_loaded_pipeline_swizzle_f16_d768_in_vs_s3_and_fast_discovery_once7`.
4 resource / 8 candidate-bit / 16 screen / 2 decision records, once7,
20 GEMMs/observation. All actual timed pointers are 256B aligned. Root
replayed all112 brackets /448 observations and both quantiles. PRE/release/
drain show no competing compute application; private cache hashes unchanged.
No full tails/sanitizer/toolkit qualification is claimed for this target screen.

Resources: all block256, local/static0, occupancy1. Register/dynamic-shared
pairs: TC128162/71680, S3188/98304, Pipeline164/71680, Swizzle180/69632.

Measured typed SHA256:
`7859864fbc27f0b32304f8c0a0d6a3936b18553a207c9a2def7e6e78003f2196`.
Shared test helper SHA256:
`ff7b4ce1add5734ce46991d07559e925af302a901f81421500e97083d21780c9`.
Binary SHA256:
`d17c29ce3415021bdbaa653a5714437f1b63e877c5546e008947d56aaff8bf84`.
[Raw samples](evidence/once7/test.log), SHA256
`5cd5cb65268b19fdabffb94c0dae911cd90762c5a7b39a0a65413642f0d8abcf`.
Measured source snapshots are retained under `evidence/source/`.
