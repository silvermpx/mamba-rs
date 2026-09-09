# Ada deterministic-TF32 NT stage-sliced production-AUTO qualification

Date: 2026-09-09

Device: GeForce RTX 6000 Ada Generation, SM89, 142 SMs

Toolkit cohorts: CUDA 12.8, 13.0, and 13.2

## Production change

The existing SM89 finalist route and symbol remain stable:

`gemm_bi_nt_sm89_mma_tf32_compact8_v1_m128n64_bk32_s2`

The module now installs the measured stage-sliced mainloop for two exact
contiguous NT shapes:

- `(2048, 1536, 768)` with strides `(768, 768, 1536)`;
- `(4096, 3072, 1536)` with strides `(1536, 1536, 3072)`.

The d768-in and canonical Prism shapes retain the former async mainloop inside
the same symbol. The dispatcher admits all four shapes only when the exact
SM89 device, toolkit, compiler, module artifact, shape, stride, scalar and
alignment identity matches. The private finalist tuning revision is 2; the
portable tuning revision remains 45.

## Frozen toolkit identities

The generated source digest is common to all three toolkits:
`1a83c5ec7a753e513ad3d4d976b97f3782c7aac6a729a2ac407ae006284ea669`.

| CUDA | compile/invocation | artifact | header manifest | NVRTC domain |
|---|---|---|---|---|
| 12.8 | `362d4e1fb046662995f7cdb97f2de0f648b3aee4f2902c0a2a036c4798603bf7` | `d1a7098f4e216ba3b9ad5a7c21b78136f3530a67e1edd5670afb6ba280314414` | `b6620b4b1d35304ab1cb98ef1ee5f32059807e441f61490ea067e943364cb2d3` | `26b0a3a02044ffcbc1693fd83e9261beffa692a4fbcfe3ac5e9d8c87980bb155` |
| 13.0 | `858018be20b7f32f7d13c322d57c0364ac6df1d43b5d81bc9867ea6932753793` | `c546dcf867cc376998257223d2e7575941f93f9dd3a55ca0dcac9591634a94e0` | `027807b25006df6eb6b77490997d3ff531fc0e277cf59fb4fd0d46910178b7a8` | `709b91c36bfb0ed966ee69adc8d6f87ff110eecf3dfb5060367f183ce614eb0d` |
| 13.2 | `a0bb06e5f2ad02cfb7291228c268059e057daede78355b52947d07e7b9c552e1` | `ddc86b75375b35b5df375e2af008cb73633564ac7ea5c46337c66a5c4d0fd7ec` | `6c00a2b09a568726056c1c999a5e08a572d7b6dcf5ad6d396f1fe90a2bd8d9b9` | `d031a53eb97235b70f62f652932db1bdf728ea229c8ca809d53c5ffd91642687` |

All identities bind driver API 13020 and driver-build digest
`d1edc5a5bc3e10a2688e21568dccbde5e280dc39144dd81e853843ca98b2d0e1`.

## Correctness and resources

The forced finalist passed bit-for-bit comparison with the portable
deterministic RNA route on all three toolkits for all four production shapes,
eager and captured graph, tail `(129, 65, 36)`, prefix `(1, 65, 36)`, alpha
`1.0` and alpha `-0.75`. The focused CUDA 12.8 and 13.0 sliced-mainloop tests
also covered aligned and misaligned tails, exceptional values and K=0. The
CUDA 13.2 production resource gate independently covered K=0, exceptional
values and live tuning revisions.

On CUDA 13.2 the route uses 122 registers, zero local memory, zero static
shared memory, 49,152 bytes dynamic shared memory and 256 threads, with two
active blocks per SM. The frozen cap is 125 registers and occupancy floor is 2.

## Once21 timing

The CUDA 13.2 paired once21 run admitted 4/4 cells and completed 32 strata,
672 paired observations and 1,344 timed arm windows. Every candidate/current
comparison had the same deterministic output digest.

| NT cell | candidate / current deterministic AUTO | candidate / cuBLAS Fast TF32 | disposition |
|---|---:|---:|---|
| d768-in `(2048,768,3072)` | about 0.835 | about 1.96 | admit |
| d768-out `(2048,1536,768)` | about 0.751 | about 1.36 | admit |
| Prism `(4621,384,1928)` | about 0.834 | about 2.09 | admit |
| large-deep `(4096,3072,1536)` | about 0.69--0.72 | about 1.50 | admit |

The ratios are candidate divided by denominator, so lower is faster. This
qualification establishes substantial deterministic-AUTO speedups, not a
cuBLAS Fast victory. cuBLAS Fast uses the vendor TF32 contract and remains the
faster denominator in these four NT cells.

## Lower-toolkit focused timing

The two shapes whose mainloop changed were independently timed against the
former retained A-ldmatrix implementation on CUDA 12.8 and 13.0. Each entry is
the range of `(p50, p95)` candidate/retained ratios over eager/graph and both
ABBA/BAAB orders from the strict once7 gate.

| CUDA | NT cell | candidate / retained range | candidate / cuBLAS Fast TF32 range | result |
|---|---|---:|---:|---|
| 12.8 | d768-out `(2048,1536,768)` | `(0.9797,0.9802)`--`(0.9804,0.9812)` | `1.4005`--`1.4099` | pass |
| 12.8 | large-deep `(4096,3072,1536)` | `(0.9851,0.9874)`--`(0.9870,0.9876)` | `1.5725`--`1.5770` | pass |
| 13.0 | d768-out `(2048,1536,768)` | `(0.9789,0.9807)`--`(0.9810,0.9823)` | `1.4005`--`1.4093` | pass |
| 13.0 | large-deep `(4096,3072,1536)` | `(0.9839,0.9855)`--`(0.9856,0.9879)` | `1.5746`--`1.5768` | pass |

All eight retained comparisons satisfy the strict `p50 < 0.99` and
`p95 < 0.99` rule. All candidate/Fast ratios remain above one and are reported
as gaps, not wins.

## Verification

Every GPU command was preceded immediately by five samples at no more than 1%
compute and memory utilization, an explicit free-memory check and a process
census. No external process was stopped or unloaded.

The production AUTO binding test was run independently for CUDA 12.8, 13.0
and 13.2. It requires all four public AUTO launches to resolve to the finalist
symbol and checks repeated eager and captured-graph outputs bit-for-bit.

The non-CUDA library suite passed 84/84 tests. The CUDA 13.2 resource/K=0/live
revision gate passed with finalist revision 2, portable revision 45, 122
registers, zero local memory, 49,152 bytes dynamic shared memory and occupancy
2. The production performance harness also compiles after pinning its retained
comparator explicitly and requiring all four cells to satisfy the strict gate.
