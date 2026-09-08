# Ada exact-F32 TN fused GROUP_M12, 2026-09-08

Valid exact performance loss on RTX6000 Ada/CUDA13.2.  Keep the retained
three-node fused-finalize `GROUP_M=8` winner from commit `84b6227e`; do not
integrate or retry unchanged `GROUP_M=12`.

The candidate changes both compute nodes to `GROUP_M=12` and is mechanically
byte-reversible to its frozen `GROUP_M=8` parents after only the symbol and
grouping substitutions.  It remains exact, but loses every paired once7
stratum:

| Comparator | Candidate/comparator p50 | Worst p95 | Decision |
| --- | ---: | ---: | --- |
| fused-finalize GROUP_M8 | `1.00155-1.00201` | `1.00338` | stop_no_retry |

The range covers eager/graph x ABBA/BAAB, two warm-up windows and20 logical
GEMMs per observation.  Because the strict retained gate failed, the harness
correctly did not launch the cuBLAS Fast screen.

## Verification

Both GROUP_M12 raw planes match the production SplitM oracle for finite and
exceptional target/tail corpora.  Final target, tail, exceptional/NaN,
non-unit-alpha and independently-oracled K0 checks pass exact bits in eager
and graph paths, including20-operation accumulation and all guards.

GROUP_M12 raw and fused kernels use135 and129 registers respectively,32768B
static shared, zero local/dynamic shared,128 threads and occupancy3: identical
resources to their GROUP_M8 counterparts.

## Frozen identity

- Harness SHA256: `481c64853b8db7020bd032bc9b15574974c736e19b238db517f7c4ea818330d8`.
- Source helper SHA256:
  `af9da62de7f16f76d49c941804fcedf3529969acc237b248e360496dde07e3f6`.
- Composed GROUP_M12 raw source SHA256:
  `56b3699183ef141f1d0b5c776a033583728d304528d80949278cc13d3c1e8ba3`.
- Composed GROUP_M12 fused source SHA256:
  `5ab9db5d5ff7aad84019dcccc167b9fa5c7e08091978a27bf2fc953f91790822`.
- Test binary SHA256:
  `e0bcdf41f2383e428cecf6a6d4b16c41b047f928d1fbeb2a21bdf544d618414f`.
- Authoritative [once7 log](attempt1/once7.log), SHA256
  `4484b2e08bc30ba98b3a671a610330e7a2ba1506c09b2439b44675149967170c`.
- [Manifest](attempt1/manifest.json) binds source, binary, toolkit, GPU UUID
  and log.  PRE/POST quiet gates pass.

Next exact-F32 TN structural screen:64x32x32/S2, but only if compilation
reaches <=128 registers, <=25KiB shared and occupancy4 without local spills.
