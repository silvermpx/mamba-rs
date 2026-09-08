# Ada half TN: loaded Rect128x64 rejected

2026-09-08, CUDA13.2 / RTX6000 Ada. Existing loaded rectangular kernel,
six large cells,256-byte-aligned logical pointers. All six valid losers;
no production change and no unchanged-candidate rerun.

| TN shape, F16/BF16 | Paired candidate/Fast p50 | Candidate/current diagnostic |
| --- | ---: | ---: |
| d768-in | 1.379–1.449 | 1.214–1.225 |
| d768-out | 1.431–1.492 | 1.102–1.108 |
| Prism | 2.139–2.245 | 1.498–1.510 |

Current ratios are two UNPAIRED diagnostic observations, not champion evidence.
Fast ratios cover eager/graph ×ABBA/BAAB, seven windows,20 GEMMs/observation.
F16/BF16 inputs, F32 dW output. Candidate/current bits, guards and Fast own
bits pass.256 threads/static shared39936; no new NVRTC candidate compilation.

Exact `ada_half_tn_rect128x64_three_cell_vs_current_and_fast_discovery_once7`
GPU test PASS. Wrapper97 was stale schema names/counts, not a kernel failure:
actual6 resource/48 bits/24 screens/6 decisions, independently verified.
Root replayed168 brackets/672 observations and all p50/p95 values, PASS.

Measured main `2d733a49e2eb15ff788c8d4dd9115d9f40e3c3a74050736fef794cfb4311a89d`;
retained unchanged in aligned-NN4 source667510c2.
Raw SHA256 `6b8e4562138d019a45d010584d0bb6e3d5f9ccb450019fba55ee11dc984e9a7d`.
[Raw](evidence/once7-cuda132/test.log),
[independent receipt correction](evidence/once7-cuda132/independent-verification.json).
