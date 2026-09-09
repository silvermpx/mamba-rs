# Ada TF32 NN M96N96/S2 direct Prism resource stop — 2026-09-09

Outcome: **resource stop before exact/timing.** This test-only candidate keeps
the retained add-half TF32 conversion, ascending K8 MMA order and direct-float2
epilogue, while changing M128N96/BK32/S3 to six-warp M96N96/BK32/S2. The goal
was occupancy2 and finer wave coverage for the remaining near-Fast Prism gap.

The candidate compiled with137 registers/thread, zero local bytes,49,152 bytes
dynamic shared and occupancy2. The frozen admission cap was128 registers; the
retained direct N96 body uses124 registers,86,016 bytes shared and occupancy1.
The candidate therefore stopped before exact and timing. This is not a measured
performance loss and does not justify weakening the cap after observing the
compile result. A retry requires a concrete source-level reduction of at least
nine registers.

- helper SHA256: `7dc395ce7cdc822a2e43bfdfdb8333c8b50893f8071ccbcdd25529c5d441df8b`
- harness SHA256: `590b9ca24da6b714c70f1033bb9f645c5c52729a745372f963ef5371a2dd38f7`
- CUDA13.2, RTX6000Ada, CC8.9,142 SM
