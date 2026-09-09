# Ada Triad TF32 NT d768-in stage-sliced stop — 2026-09-09

Outcome: **valid strict-threshold stop.** The exact four-slice CUDA body that
became the d768-out retained-best was screened on aligned d768-in
`(2048,768,3072)`. It is consistently about 0.45–0.65% faster than the frozen
A-only ldmatrix comparator, but does not meet the declared 1% admission margin.
Stop before once7 and cuBLAS Fast; preserve the prior d768-in retained kernel.

## Gates

- CUDA 13.2 / SM89 RTX 6000 Ada; grid 192, block 256.
- Candidate and retained: 98 registers, local/static 0, dynamic shared 49,152
  bytes, occupancy 2.
- Exact PASS: target, aligned and misaligned tails, exceptional input and K=0;
  eager/graph repeats and guards pass.
- Candidate CUDA source/PTX identities are unchanged from the d768-out winner:
  `521b592024c6ad946972cbc2454e9aa62905b6d1305e5ba5b0b60392cbc64ad8` /
  `abf41b272d3857a2c9af0fdbe9e9eb752c58bc74dd9547c12b2c9bc483db17e3`.

## Paired once3 candidate/retained ratios

| Path/order | p50 | p95 |
| --- | ---: | ---: |
| eager ABBA | 0.993873 | 0.993977 |
| eager BAAB | 0.993492 | 0.993626 |
| graph ABBA | 0.993996 | 0.994098 |
| graph BAAB | 0.994505 | 0.995525 |

All ratios are below 1.0, but none satisfies the strict `<0.99` admission gate.
No unchanged retry, sibling inference, production change or dispatcher change
is justified by this screen. An inherited d768-out environment guard was caught
before any kernel launch, repaired with a native shape-specific regression, and
the final measured run used the frozen corrected harness.

## Frozen evidence

- adapter SHA256: `e2ef4f0290867c0f23b3aa152e32fea91d1b8bb11b2dcc02d08e815358a83233`
- harness SHA256: `2a48df840d887dc01a963304fdd74e90101ccbab38469995761ee9e2fef4f04c`
- raw log SHA256: `7b2b4dc89f87d22e5d2d5eceaa644b1ac3da45f459915996abe9e2f18ad1c5f6`
- [raw output](raw.log)
- [launch preflight](preflight.txt)
