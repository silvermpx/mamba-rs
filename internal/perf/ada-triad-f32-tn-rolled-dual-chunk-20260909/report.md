# Ada exact-F32 TN rolled dual-chunk compile stop — 2026-09-09

Outcome: **STOP before GPU.** The test-only candidate keeps the winning
transpose + fused dual-chunk two-node graph and exact arithmetic order, but
rolls the two 1024-row chains through one mainloop body with `#pragma unroll 1`.
This substantially shrinks code footprint, yet introduces local spills and
therefore fails the mandatory compile/resource gate.

## CUDA 13.2 / SM89 compile evidence

| Kernel | text bytes candidate/retained | ratio | FFMA candidate/retained | registers | stack | spill store/load |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw twin | 72,960 / 135,808 | 0.5372 | 2,048 / 4,096 | 168 | 16 B | 16 / 16 B |
| fused finalize | 75,776 / 138,624 | 0.5466 | 2,048 / 4,096 | 168 | 16 B | 16 / 16 B |

Both candidates retain 32,768 bytes static shared and real LDGSTS, but both
contain LDL/STL. The required text `<=0.70` and static-FFMA `<=0.60` gates pass;
the required local/spill-free gate fails. One bounded selector-lifetime cleanup
produced identical SASS/resources, so no timing, NCU, exact GPU corpus or Fast
comparison was launched. Keep the earlier unrolled dual-chunk retained-best and
do not retry this unchanged mechanism.

## Verification and frozen identity

- native source/harness tests: 15/15 PASS
- helper SHA256: `4002c0cb2eb93865f42a81643e98f9c8ab2a06e02cc2ccac7a378431c123732c`
- harness SHA256: `31b59c535f7c37df18210b4c6fb460b5405bbda6bb117bfb4f2833c09b1a7ffa`
- compile-only log SHA256: `6da756ec854f8aaa9dbbdd4d1700e67254d1b438cf4a5b44b34c31efece921ed`
- [compile-only raw output](compile-only.log)

Production and dispatcher are unchanged.
