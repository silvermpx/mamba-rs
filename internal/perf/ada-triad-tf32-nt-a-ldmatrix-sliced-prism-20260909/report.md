# Ada Triad TF32 NT Prism stage-sliced A-ldmatrix — 2026-09-09

Outcome: **new test-only retained-best for Prism, not a cuBLAS Fast win.**
The actual training-Triad candidate applies the already frozen four-slice
A-ldmatrix schedule to Prism `(M,K,N)=(4096,3072,1536)`. It preserves the
compact A-x4 ldmatrix parent, S2 staging, RNA conversion, ordered MMA and
epilogue while interleaving bounded next-stage copy slices before the ascending
K8 MMA issues.

## Gates and identity

- CUDA 13.2 / SM89 NVIDIA RTX 6000 Ada Generation; grid 1,536, block 256.
- Candidate and frozen Prism A-only ldmatrix comparator both pass resources:
  98 registers, local/static 0, dynamic shared 49,152 bytes and occupancy 2.
- Exact PASS: target, separate M/N/K tails, exceptional input and K=0;
  eager/graph repeats, 20-operation checks and guards pass.
- The `gemm_bi_nt_*` symbol, source derivation from
  `kernels/gemm_bi_triad/sm80.cu` and five-argument
  `Sm80Tf32KernelParams` launch contract prove this is the training-Triad
  family, not standalone Fixed/inference.
- Candidate source/PTX SHA256:
  `521b592024c6ad946972cbc2454e9aa62905b6d1305e5ba5b0b60392cbc64ad8` /
  `abf41b272d3857a2c9af0fdbe9e9eb752c58bc74dd9547c12b2c9bc483db17e3`.
- Retained source/PTX SHA256:
  `263cdf63dedfbebd84fd668fd3e5aa00a5f4b80241d17f5c336e5e9d60945619` /
  `17ba9734e6e628dcd88761475e2b4a518cf66b46eb613a2aec67cea7ccfa1f32`.

## Paired once7 ratios

Ratios are candidate/comparator; strict admission requires both p50 and p95
`<0.99` in all four path/order strata.

| Path/order | vs retained p50 | p95 | vs Fast p50 | p95 |
| --- | ---: | ---: | ---: | ---: |
| eager ABBA | 0.978819 | 0.979435 | 1.561288 | 1.565702 |
| eager BAAB | 0.978074 | 0.979631 | 1.560529 | 1.562227 |
| graph ABBA | 0.978471 | 0.981452 | 1.560726 | 1.562250 |
| graph BAAB | 0.978268 | 0.979108 | 1.561354 | 1.562782 |

The candidate beats the frozen Prism A-only ldmatrix comparator by about
1.85–2.19% across the paired p50/p95 results and passes the retained gate in
every stratum. Native cuBLAS Fast remains about 1.56x faster, so retain this as
the new test-only Prism leader for joint integration but do not claim a Fast
win or production promotion.

## Frozen evidence

- helper SHA256:
  `0c930ee799ba3b7741ddeb9eecc35e9cac503bd48e8c083f592fe00fcbd7e6bf`
- harness SHA256:
  `b383ecc70c75ea2bcfa956c5924791d48fd7b255075dff523f21bfa11aa154ae`
- raw log SHA256:
  `f233e7e99dfde8fb5efae18b14d226e5e0e03d3f5912afff42f058d6117f8ee0`
- [raw output](raw.log)
- [launch preflight](preflight.txt)
