# Half TN: TC64/BK32/S4 rejected

Ada/CUDA13.2,2026-09-08. All six aligned cells pass bits/resources but lose
cuBLAS Fast. Shared36864/block128; same ascending K16 math and F32 dW epilogue.

| Shape, F16/BF16 | Paired candidate/Fast p50 | Unpaired candidate/current diagnostic |
| --- | ---: | ---: |
| d768-in | 1.490–1.545 | 1.293–1.307 |
| d768-out | 1.700–1.773 | 1.300–1.329 |
| Prism | 1.687–1.769 | 1.174–1.195 |

Exact one test PASS:6 resources/48 bits/24 Fast screens/6 decisions. Once7,
eager/graph ×ABBA/BAAB,20 GEMMs per observation,256-byte logical alignment.
Root independently replayed168 brackets/672 observations and48 quantiles,
PASS. Helper native4PASS. Valid stop: do not retry this unchanged candidate.

Measured main `8c7b95ac61e4fc2f1f48de036cfe0646fb7f7cd2344a2b202280ca0619e6c639`;
helper `c48545d9d9dd6477b9150e315d0c905253dfffc1b2d3ed8ee8b4953643464201`.
[Raw](evidence/once7-cuda132/test.log), SHA256
`fa71aed1d39baac16f67eaa9b1224d75ee707ab17906a6b0ba0edde178d16005`.
No production or SM120 modification.
