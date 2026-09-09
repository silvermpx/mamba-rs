# Ada TF32 NN M96N96/S2 copy-lifetime resource stop — 2026-09-09

Outcome: **resource stop before exact/timing.** This test-only follow-up keeps
the frozen M96N96/BK32/S2 geometry, add-half conversion, K8 MMA order and
direct-float2 epilogue. It removes the persistent copy-plan state and computes
copy addresses inside bounded A/B issue scopes. Production dispatch is
unchanged.

CUDA13.2 ptxas/SASS reports140 registers, zero stack and spill bytes,
49,152 bytes dynamic shared,48 HMMA and16 LDGSTS. The parent M96N96 candidate
used137 registers, so the intended lifetime contraction instead raised the
count by three. The frozen admission cap was128 registers. Exact and timing
did not start; this is not a performance result. Do not retry the on-demand
address-computation mechanism unchanged.

- helper SHA256: `646e70a61ade446b5d791b09f932370846520187b7fa550ccf9d2b7e01d93fda`
- harness SHA256: `281fa9749e2299622c57f300ce3f64c5823081a580f78694561a42246aafa083`
- CUDA13.2, RTX6000Ada, CC8.9,142 SM
