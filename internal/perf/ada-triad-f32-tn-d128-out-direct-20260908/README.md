# Ada exact F32 TN d128-out: direct fold reuse

NEW short discovery on CUDA13.2, RTX6000 Ada; shape `(M,K,N)=(1024,256,128)`.
M8N16 is the preferred finalist: 39–42% lower paired median time than the
actual public AUTO route. It still takes about2.15–2.21x cuBLAS Fast TF32 time.
This is progress toward exact-F32 parity, **not a cuBLAS Fast win**.

| Candidate | Eager median us | Graph median us | Paired p50 / AUTO | Paired p50 / Fast |
| --- | ---: | ---: | ---: | ---: |
| M8N16 / 64 threads | 35.1–35.3 | 35.8–36.0 | 0.579–0.613 | 2.147–2.205 |
| M16N16 / 64 threads | 40.0–40.4 | 40.8–41.5 | 0.664–0.672 | 2.426–2.568 |

The AUTO arm was about57–62 us, Fast about16 us. Quantile ratios are paired
per bracket, not ratios of the independently summarized medians in this table.
Both candidates pass eager2/graph2 and every timed observation against actual
public SplitM64 output bits. Fast has separate finite/self-repeat bit checks.
All112 brackets /448 observations and16 p50/p95 pairs independently replayed.

The source adapter changes only output geometry/exports: each chunk still starts
at positive FP32 zero, performs16 ascending FMAs, then64 chunks are folded in
ascending FP64 order. Alpha scaling and final FP32 C addition are unchanged.
One launch replaces the two-launch partial-buffer/reducer path. The M8N16 grid
has256 CTAs,80 registers,3072 dynamic shared bytes and no local/static storage.

Measured main SHA256 `7f4b21ac730a93527a01c4875fd1a301ab64a8b23cdded8f9902490efbce84d5`;
helper `22847b7c051dff6db92665f6ea694da1a34075861a7c7677456b04598202a757`.
The existing64-F32-word guards preserve256-byte alignment; this run does NOT
have the half-NN16-byte-offset baseline defect. One-GEMM event timing includes
launch overhead. No admission, exceptional-value full qualification, other
toolkit, SM120 or end-to-end training claim is made.

Exact `cuda_tournament::ada_d128_out_direct_fold_two_arm_once7` passed in6.88s.
The wrapper expected old one-candidate record counts and returned97; its raw
receipt is preserved. Correct counts are2 resources/16 screens/4 decisions;
[independent verification](evidence/once7-cuda132/independent-verification.json)
records that distinction without rerunning the GPU.
[Raw](evidence/once7-cuda132/test.log) SHA256
`4e64d67d9a589e9d6b5db9afacdd46d15a5ee7496f3d0bc60ce0e1795a0c2893`.
