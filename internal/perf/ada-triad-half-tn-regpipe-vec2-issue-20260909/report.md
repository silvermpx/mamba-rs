# Ada Triad F16 TN regpipe issue-order stop — 2026-09-09

Outcome: **STOP, no retry or promotion.** The test-only actual-Triad d768-in
`(2048,768,3072)` candidate changes only the order of the eight independent
HMMA accumulator tiles inside each K16 group. It traverses `fn` before `fm`,
alternating the two accumulator rows while reusing each B fragment. Every
output retains the exact ascending K16 order `0,1,2,3`; staging, the `float2`
epilogue, geometry and the real seven-argument TN ABI are unchanged.

## Correctness and resource gates

- CUDA 13.2 / SM89 RTX 6000 Ada, grid 576, block 128.
- Candidate and frozen regpipe+`float2` comparator both use 125 registers,
  local 0, static shared 32,768 bytes, dynamic shared 0 and occupancy 3.
- CUDA13.2 SASS is spill-free; candidate/retained both contain 32 HMMA and
  24 LDGSTS instructions.
- Exact PASS for target, aligned negative-alpha tail, misaligned negative-alpha
  tail, exceptional full tile and K=0. Eager/graph repeats, 20-operation target
  accumulation, input immutability and guards pass.

## Paired once3 candidate/retained ratios

Ratios are candidate / frozen regpipe+`float2`; lower is better.

| Path/order | p50 | p95 |
| --- | ---: | ---: |
| eager ABBA | 1.000000 | 1.000293 |
| eager BAAB | 1.000550 | 1.001110 |
| graph ABBA | 0.999705 | 1.000295 |
| graph BAAB | 1.000295 | 1.001476 |

No stratum meets the predeclared strict `p50 < 0.99 && p95 < 0.99` retained
gate. The mechanism is resource-neutral but runtime-neutral to slightly slower,
so once7 and cuBLAS Fast were not launched. Keep the prior M64N64/BK64/S2
regpipe+`float2` kernel and do not retry this unchanged issue order.
Production and dispatcher remain unchanged.

## Frozen evidence

- helper SHA256:
  `b36cd82bbadd5b2e181277b75e9532d8a981be259126cb8c8fef55ad0f34f7fd`
- harness SHA256:
  `18cb9ee80cc9a760bd34d01002066858b06eef01b9dd161f20e7d239e8f8c4aa`
- candidate source/PTX/CUBIN/SASS SHA256:
  `b1c88702cd3a033d94b03d9e6a35614f039fd054d528fe1ab45ceca766f4d5a0` /
  `1451e7c3883fc927c1f9d241f3fa18f69aeb0bf0875cb2f64060c6b9fe8a398e` /
  `a2b4c98ac7aa6abc3c1a03469e38b2e53e1cfd2ee938f1f4c6c0438c5533509f` /
  `19440ecc5fcd7cadc41a1ab884f50906d11dcf154bdb2e8300abd3e37d355449`
- retained source/PTX/CUBIN/SASS SHA256:
  `9412e655faafe8b0f91c8d593a8cda0cafdb6320c04285cd9b396cfdcf36874f` /
  `2d75e845ff58760f073898cd309a97d4c68e0a2e32d0eb39778868d10ef85e4c` /
  `8712e90afb71706cd3d6440d44931e151e335e4c86900e18ee5137306e79756b` /
  `4942f0b14da97ea82a4cc40d5352d681abfaec5b3b06a79190a32d5cc77ed785`
- raw log SHA256:
  `c4dfee49591bd9f77547495950e0500d74cdbed44f5622728787a11358fc7724`
- [raw output](raw.log)
- [launch preflight](preflight.txt)
