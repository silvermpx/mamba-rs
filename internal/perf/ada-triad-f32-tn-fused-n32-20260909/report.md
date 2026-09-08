# Ada exact-F32 TN fused-finalize N32 screen — 2026-09-09

## Decision

Valid loss; stop this unchanged candidate and keep the retained N64 GROUP_M8
fused-finalize pipeline. Nothing from this experiment is admitted to production
or the dispatcher.

The test-only candidate was the exact three-node pipeline
`transpose -> N32 raw chunk0 -> N32 fused-finalize chunk1` for d768-in
`(batch,k_out,n_out)=(2048,768,3072)`. It preserves the two fixed-order F32
FMA chains and the existing FP64 finalizer.

## Correctness and resources

- Raw chunk0 and chunk1 planes are bit-identical to the production SplitM
  partial oracle on full-mantissa and exceptional corpora.
- Final target, tail, exceptional payload, non-unit alpha, K0, eager replay,
  graph replay, 20-operation accumulation, guards, transpose and unused-slab
  checks pass exactly.
- Raw N32: 105 registers, 0 local bytes, 24,576 static shared bytes,
  4 active blocks/SM.
- Fused N32: 101 registers, 0 local bytes, 24,576 static shared bytes,
  4 active blocks/SM.

## Paired once7 result

Candidate/retained-N64 ratios, 20 logical GEMMs per observation:

| Path/order | p50 | p95 |
| --- | ---: | ---: |
| eager/ABBA | 1.08795 | 1.09201 |
| eager/BAAB | 1.08919 | 1.09013 |
| graph/ABBA | 1.08975 | 1.09106 |
| graph/BAAB | 1.09052 | 1.09097 |

The N32 candidate is consistently 8.8–9.1% slower than the retained N64
pipeline. Per the discovery stop rule, cuBLAS Fast was not screened after the
candidate failed the retained-best gate.

## Environment and evidence

- RTX 6000 Ada, SM89, 142 SMs, CUDA 13.2, 1800 MHz.
- User-authorized idle-resident mode: the preflight and each timed boundary
  required five consecutive samples at no more than 1% compute and memory
  utilization. The pre-context sample had 2,669 MiB free; the timed/post
  samples had 1,992 MiB free. External CUDA contexts were not stopped.
- This is strong evidence for a loss because all four paired strata agree by
  roughly nine percent, but it is not a new release qualification run.
- Raw receipt: `raw.log`.
- Harness SHA-256:
  `103a7b3c6d38934231bfe73d6e74f5e1236ecddb099df825aa3982f8e13b9d2c`.
- Candidate source SHA-256:
  `f09aa58152ecee11d83ed46a752ab33623417d56680e3513a9ae2ce360773eb0`.
