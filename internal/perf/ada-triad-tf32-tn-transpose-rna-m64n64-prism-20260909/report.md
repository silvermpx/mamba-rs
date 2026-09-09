# Ada TF32 TN transpose-RNA M64N64 canonical Prism winner — 2026-09-09

Outcome: **new retained-best canonical-Prism winner; cuBLAS Fast miss.** The
test-only candidate preserves the retained A-only pre-RNA transpose and K8
order `[0,8,16,24]`, but changes the GEMM from M128N96/S3 to an eight-warp
M64N64/S3 wave-quantized body. Production dispatch is unchanged pending joint
whole-Triad integration.

All full/tail/exceptional/K0 exact checks pass in eager and graph paths, with
independent RNA scratch oracle, input immutability, ABI and 256-byte guards.
The candidate uses 83 registers/thread, zero local bytes, 49,152 bytes dynamic
shared and occupancy2. The retained N96 body uses 127 registers, zero local
bytes, 86,016 bytes dynamic shared and occupancy1. Canonical Prism launches
186 candidate CTAs versus 63 retained CTAs.

Candidate/retained once7 p50/p95:

| Path/order | Ratio |
| --- | ---: |
| eager ABBA | 0.918651 / 0.946645 |
| eager BAAB | 0.913590 / 0.947053 |
| graph ABBA | 0.912703 / 0.925526 |
| graph BAAB | 0.912384 / 0.933405 |

Every stratum passes the strict `<0.99` retained gate. Candidate/cuBLAS Fast
once7 p50/p95 is `1.738473/1.751374`, `1.675168/1.784323`,
`1.699439/1.721239`, and `1.705918/1.723499` in the same order. It is therefore
not a Fast win, but it reduces the retained pipeline by 8.1–8.8% at p50 and
5.3–7.4% at p95.

Frozen identities:

- helper SHA256: `72b6bc0b42b3c6b54acf066f2941da7ee5413571209284c51243bd104c6a0486`
- harness SHA256: `ab9ade1f1e9718877369f861ffea81051af8fc4ba583df347b90f630cb99fc4e`
- candidate transformed source SHA256: `af96f3a6d7f2858922721c8879f3a7d51380a86962bd75809a592ba82bd4ac67`
- candidate PTX SHA256: `d8e94bec59902c2d176baf4e39734ad26a97749935bea4f57447ad75da08c7b4`
- retained transformed source SHA256: `3ee1e8986c3d1fb998aef408f14add2203d71bbd6f8521f08500ac972f2d2f65`
- retained PTX SHA256: `57808541ffcf29b65c0bdecb14c85e9a6e185f84da39167c4c5d3cf8df313ee6`
- CUDA toolkit: 13.2; device: RTX 6000 Ada, CC8.9, 142 SM
