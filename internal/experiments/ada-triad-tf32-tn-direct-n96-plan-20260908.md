# Ada TF32 TN direct RNA-N96 short screen

## Hypothesis

Reuse the retained eight-warp M128xN96/BK32/S3 N96 pipeline directly on
the original TN A layout, avoiding the separately timed raw-bit transpose.
Only A's global copy plan, shared-memory index and scalar fragment reads
change. The B path, three-stage ring, register ping-pong, `cvt.rna.tf32.f32`,
ascending K8 MMA sequence, output ownership and accepted TN
`__fmaf_rn(alpha, accumulator, old_c)` epilogue remain unchanged.

The direct A shared index is
`k*128 + (((row>>2)^((k&3)<<1))*4) + (row&3)`. Native enumeration must prove
complete 16-byte copy ownership and a 32-bank permutation for every scalar A
fragment instruction. The bank mapping is a source-derived inference, not a
performance claim. NVIDIA documents 32-bit shared-memory banks in the
[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html),
and CUTLASS documents XOR/permuted shared layouts in its
[implicit GEMM convolution guide](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/implicit_gemm_convolution.md).

## Screen boundary

- Test-only distinct export; no production or Fixed source changes.
- First cell only: TN d768-in `(2048,768,3072)`.
- Resource gate precedes bits and timing: local/static zero, 256 threads,
  86,016 dynamic bytes, occupancy at least one; register count recorded.
- Candidate must match actual AUTO on the target and the existing forced
  RNA-wide route on tail, exceptional and K0 probes.
- Compare candidate with actual AUTO and explicit cuBLAS Fast in eager/graph
  ABBA/BAAB once7 strata, one GEMM per observation, with resets outside events.
- A valid loss stops this arm. Any later sibling or production qualification is
  a separate decision.

## Outcome

Main36126907/helpera153aeb2 passed the focused13.2 screen; direct-only tail
changed to `(129,68,36)` for its required16B strides. Actual AUTO improves
about27%, but graph remains35–37% above Fast. No paired comparison to the
retained transpose+N96, so no incremental-best claim. Valid STOP; no repeat.
See `../perf/ada-triad-tf32-tn-direct-n96-20260908/report.md`.
