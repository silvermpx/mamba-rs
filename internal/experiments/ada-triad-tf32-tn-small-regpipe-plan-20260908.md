# TF32 TN small register-pipeline discovery plan

Goal: reduce the remaining d128-in/out TF32 TN gap to explicit cuBLAS Fast
without changing RNA conversion, ascending K8 MMA association or the epilogue.
Existing compatible loaded rungs were all measured previously; no resweep.

Scope: a test-only source adapter for existing TN M16N32/BK32/S4. Double-buffer
A/B register fragments, preload issue(k+1) before consuming issue(k), and retain
the existing shared-memory staging, barriers, copy paths and output ownership.
No production/inference/SM120 changes. Whole-Triad discovery boundary applies.

- [x] Add native failing tests for exact source restoration, load/MMA text,
  ordered slot consumption and missing/duplicated source-boundary rejection.
- [x] Implement a count-checked generator restricted to TN M16N32/S4; run native
  tests and bounded independent review. Do not claim instruction scheduling yet.
- [x] Reuse the TF32 TN harness after its transpose freeze: d128-in/out only,
  actual AUTO plus explicit Fast, exact bits/guards, aligned256B, eager/graph,
  once7. Local0/static0/dynamic32768,128threads, occupancy>=3.
- [x] Sole GPU executor measures once on Ada13.2. Preserve raw source and
  observations; stop valid losses, retain only measured improvements.
