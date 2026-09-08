# Exact F32 TN: one-chunk-delayed fold rejected

Ada/CUDA13.2,2026-09-08. Named previous/current register arrays were intended
to overlap FP64 folding with next-chunk FP32 work, preserving each16-FMA
chunk and ascending64-chunk FP64 association. Both exact tests pass but lose
to retained direct candidates; keep the previous M16N16-in/M8N16-out finalists.

| Cell | Candidate/retained p50, eager | Graph | Candidate/Fast p50, all strata |
| --- | ---: | ---: | ---: |
| d128-in | 1.374–1.408 | 1.138–1.153 | 3.415–4.243 |
| d128-out | 1.030 | 1.003–1.004 | 2.176–2.419 |

Each cell:2 resource/1 corner-bits/8 screen/2 decision records; separate exact
processes. Guards256B aligned; finite timing bits and ±zero/subnormal/Inf/NaN
payload/chunk-boundary eager2+graph2 bits match actual AUTO and retained route.
Root replayed112 brackets/448 observations and32 quantiles across both logs,
PASS. Native16 tests PASS. No production edits or unchanged-candidate retry.

Main `bea57f42f30b947ed4dde4c7ebf13faf6ed7da9331c9e082f696c574b8008bd1`;
helper `1243b969c16747388a07c46a2fe491a076421bb97b6a907b8c2aa0db6ea5f287`.
[In raw](evidence/once7-cuda132-d128-in/test.log), SHA256
`ed82ad8972ca9665f6915d74d8d30aa3f412c127afe98822c79ea7e6c2dffa04`;
[out raw](evidence/once7-cuda132-d128-out/test.log), SHA256
`af05fe18b4827ed70b71d95a57c28642ae70994749c0d942a1acf5adf01c1c6d`.
