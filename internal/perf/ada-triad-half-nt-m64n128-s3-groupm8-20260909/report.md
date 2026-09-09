# Ada Triad F16 NT M64N128/S3 GROUP_M8 stop — 2026-09-09

## Decision

STOP after the once3 retained scout. The test-only GROUP_M8 CTA raster is exact
and resource-neutral, but is consistently 0.26–0.46% slower than the frozen
row-major M64N128/S3 parent. Do not run once7/Fast or wire this raster into
production. Keep the measured row-major S3 kernel.

## Result

Shape: F16 NT d768-out `(M,N,K)=(2048,1536,768)`, CUDA13.2, SM89 RTX6000 Ada,
20 GEMMs per observation.

| Path / order | candidate / retained p50 | p95 |
| --- | ---: | ---: |
| eager / ABBA | 1.00258 | 1.00387 |
| eager / BAAB | 1.00278 | 1.00382 |
| graph / ABBA | 1.00397 | 1.00397 |
| graph / BAAB | 1.00464 | 1.00464 |

The strict scout gate requires every p50 and p95 `<0.99`. It failed in all four
strata, so the harness correctly stopped before retained once7 and cuBLAS Fast.

## Correctness and resources

- Exact candidate-versus-measured-S3 bits PASS for full target, ragged M9
  group, separate M/N/K tails, exceptional corpus and K0.
- Eager, graph, three graph replays, input/output guards and target 20-op PASS.
- Candidate and retained are identical at119 registers, local0, static shared0,
  dynamic shared73,728 bytes and occupancy1.
- The reversible source transform changes only CTA coordinate mapping and the
  test export; S3 pipeline, MMA sequence and epilogue remain unchanged.

The earlier hypothesis that grouped traversal might improve B-tile locality is
not supported for this 32x12 CTA grid. Preserve row-major raster. A further
GROUP_M sweep is not justified by a consistent loss with unchanged resources.

## Evidence identity

- raw log SHA256 `57db4d7697ee9d94df2c40f0d4a49a325d0130e856c9dcdd7ea43b1ef8c9e095`
- source adapter SHA256 `23f8677b646ea2c043a19d36195cf19c2b4a7b1229ee9b3b8a456543162183da`
- standalone harness SHA256 `f3247d963934cb3a88ea437e90e4d197163b78b6ee5070be82646e48adb8deb8`
- Native source/harness tests: 14/14 PASS.
- CUDA-feature no-run and CUDA13.2 NVRTC candidate+retained compile-only: PASS.
- Immediate external and in-harness GPU quiet gates: PASS with ample VRAM.

Production and dispatcher were unchanged.
