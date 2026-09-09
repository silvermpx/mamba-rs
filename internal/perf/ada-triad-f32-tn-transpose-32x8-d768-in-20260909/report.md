# Ada exact-F32 TN 32x8 transpose d768-in stop — 2026-09-09

Outcome: **bit-exact/resource-valid near parity, rejected at once3.** This
test-only candidate replaces only the retained 32x16/512-thread transpose with
a canonical padded 32x8/256-thread transpose processing four elements per
thread. The retained M64N64/BK32 dual-chunk fused-finalize GEMM and its exact
arithmetic tree are unchanged. Production dispatch is unchanged.

The candidate transpose uses 24 registers, zero local/stack/spill bytes,
4,224 bytes static shared memory and six CTA/SM. Actual-AUTO, retained target,
tail, exceptional, non-unit-alpha and K0 eager/graph bits, guards and graph
identity all pass.

Candidate/retained once3 p50/p95:

| Path/order | Ratio |
| --- | ---: |
| eager ABBA | 0.998475 / 0.999598 |
| eager BAAB | 0.998633 / 0.999920 |
| graph ABBA | 0.998551 / 0.999597 |
| graph BAAB | 0.999035 / 0.999678 |

The measured p50 improvement is only 0.10–0.15%, and every stratum misses the
frozen `<0.99` retained-admission threshold. Stop before once7 and both vendor
denominators. Keep the retained 32x16 transpose and do not retry this unchanged
mechanism. The final assembled exact-F32 census must still pair production AUTO
independently against both cuBLAS Pedantic and cuBLAS Fast TF32.

- harness SHA256: `dcfde5d434b96b3d7d47777d8d8530fc0b7157caecd248e12e852ed64976979f`
- helper SHA256: `96a4c09acf63ef436a10a43e6efc58777f2ab19cf49f245ab5ef420dd4204560`
- composed candidate source SHA256: `f234afe01d7157d5db4e77bf48264b29ed50272e6e030ceaf476f008c1d52e96`
- CUDA13.2, RTX6000Ada, CC8.9, 142 SM
