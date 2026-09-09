# Ada Triad TF32 NT canonical Prism full-domain resource stop — 2026-09-09

Outcome: **resource STOP before exactness or timing; no production change.**

The test-only canonical `prism_in_proj` `(M,K,N)=(4621,384,1928)` candidate
uses a two-dimensional `(6,37,1)` grid. The first 216 full CTAs use direct
16-byte copies for the first 60 complete BK32 stages; the final M strip and
final K stage keep the retained guarded A-ldmatrix path. The transform is
reversible and leaves the retained HMMA order and arithmetic intact.

CUDA 13.2 NVRTC compile-only passed and retained the expected cp.async,
ldmatrix and TF32 HMMA PTX anchors. The runtime resource gate then rejected the
candidate at 141 registers and one active CTA/SM, versus the predeclared caps
of at most 128 registers and at least two active CTAs/SM. Local memory is zero
and dynamic shared memory is 49,152 bytes. Per the stop rule, no exactness
launch, once3, once7 or cuBLAS Fast timing was run. Do not retry this unchanged
full-domain mechanism; the next canonical-Prism attempt needs a source-backed
register-lifetime reduction.

Frozen identities:

- helper SHA256: `66f446928e03b18d5a9472355aefa48e4ecc3ac81dc4d7e4acaf7706739a8208`
- harness SHA256: `4f5dae59886b9e4f8ff14e25f22ab91a38cdb9a9a2a863a5b71f489075681a86`
- candidate transformed source SHA256: `dbeb2f86acbae04b9ac503c69fbbff1f71e835c81a0af54128277e17827b793c`
- candidate PTX SHA256: `35e60c4cc48815f9af50510df30ab05c93274ee6d6dd0427190420485ea47194`
- retained transformed source SHA256: `263cdf63dedfbebd84fd668fd3e5aa00a5f4b80241d17f5c336e5e9d60945619`
- retained PTX SHA256: `17ba9734e6e628dcd88761475e2b4a518cf66b46eb613a2aec67cea7ccfa1f32`
- [raw CUDA output](raw.log)
