# Ada Triad TF32 NT canonical Prism K8 regpipe stop — 2026-09-09

Outcome: **once3 STOP at parity/slightly slower; no production change.**

The test-only canonical `prism_in_proj` `(4621,384,1928)` candidate starts
from the frozen A-only ldmatrix parent and keeps its linear grid222. It
double-buffers only A fragments across ascending K8 issues while loading and
consuming B one n-atom at a time. Every accumulator retains K8 order
`0,8,16,24`, and the candidate contains none of the rejected 2D/full-domain
mechanism.

CUDA13.2 NVRTC compile-only passes. Candidate and retained both use98
registers, local0, dynamic shared49,152 and occupancy2. Exact target finite and
exceptional values, aligned/misaligned tails, exceptional small shape, K0,
eager/graph repeats and guards all pass.

Candidate/retained once3 p50/p95:

| Stratum | p50 | p95 |
| --- | ---: | ---: |
| eager ABBA | 1.001632 | 1.004461 |
| eager BAAB | 1.001688 | 1.002247 |
| graph ABBA | 1.001583 | 1.001848 |
| graph BAAB | 1.001291 | 1.001930 |

All four strata miss the strict `<0.99` retained gate, so once7 and cuBLAS
Fast were not run. Do not retry this unchanged scheduling-only mechanism.

Frozen identities:

- helper SHA256: `2f93aeb779965c56e2f74b7d6ae5ace566e56805c3945bf35f1f7a144c8edb24`
- harness SHA256: `5133f1ae745641214f434db136efe941cfce05b220468011e9ef820bea0ff516`
- candidate transformed source SHA256: `264d662829d4a6c15acdac0e79bae8624b0cf164ba7e817223d678c6c1053a28`
- candidate PTX SHA256: `1722a739fae0f29e691165d8d12b12cd915c9e00a79b68e7385996f7c8345a0a`
- retained transformed source SHA256: `263cdf63dedfbebd84fd668fd3e5aa00a5f4b80241d17f5c336e5e9d60945619`
- retained PTX SHA256: `17ba9734e6e628dcd88761475e2b4a518cf66b46eb613a2aec67cea7ccfa1f32`
- [raw qualification summary](raw.log)
