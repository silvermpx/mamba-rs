# Ada Triad TF32 NT stage-sliced A-ldmatrix — 2026-09-09

Outcome: **new test-only retained-best for d768-out, not a cuBLAS Fast win.**
The candidate interleaves four bounded next-stage copy slices before ascending
K8 MMA issues while preserving the compact A-x4 ldmatrix parent, S2 staging,
RNA conversion, MMA order and epilogue.

## Gates and identity

- Ada RTX 6000, CUDA 13.2, compute capability 8.9.
- Target `(M,K,N)=(2048,1536,768)`, grid 384, block 256.
- Resource PASS for candidate and retained: 98 registers, local 0, dynamic
  shared 49,152 bytes, occupancy 2.
- Exact PASS: target, aligned tail, misaligned tail, exceptional case and K=0;
  eager/graph repeats and guards pass.
- Candidate source/PTX SHA256:
  `521b592024c6ad946972cbc2454e9aa62905b6d1305e5ba5b0b60392cbc64ad8` /
  `abf41b272d3857a2c9af0fdbe9e9eb752c58bc74dd9547c12b2c9bc483db17e3`.
- Retained source/PTX SHA256:
  `263cdf63dedfbebd84fd668fd3e5aa00a5f4b80241d17f5c336e5e9d60945619` /
  `17ba9734e6e628dcd88761475e2b4a518cf66b46eb613a2aec67cea7ccfa1f32`.

## Paired once7 ratios

Ratios are candidate/comparator; strict win requires both p50 and p95 `<0.99`.

| Path/order | vs retained p50 | p95 | vs Fast p50 | p95 |
| --- | ---: | ---: | ---: | ---: |
| eager ABBA | 0.986096 | 0.987166 | 1.411686 | 1.412710 |
| eager BAAB | 0.985565 | 0.986841 | 1.412328 | 1.413252 |
| graph ABBA | 0.985263 | 0.986046 | 1.418658 | 1.419688 |
| graph BAAB | 0.985795 | 0.986303 | 1.418220 | 1.419430 |

Retain this candidate for joint integration as the new TF32 NT d768-out leader.
It is 1.28–1.47% faster than the prior A-only ldmatrix kernel, while cuBLAS Fast
remains 1.41–1.42x faster. Production and dispatcher are unchanged here.

## Frozen evidence

- helper SHA256: `0c930ee799ba3b7741ddeb9eecc35e9cac503bd48e8c083f592fe00fcbd7e6bf`
- harness SHA256: `de5f976911bbfdbbf7d96f91d584409b9858f67c584557d149a8e6669ece0d37`
- raw log SHA256: `3faa137d1711460ae05045791dd05d518b8f1b622ab4ed875e0fec388196d50a`
- [raw output](raw.log)
- [launch preflight](preflight.txt)
