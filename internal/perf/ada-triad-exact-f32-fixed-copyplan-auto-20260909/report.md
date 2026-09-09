# Ada exact-F32 Fixed CopyPlan production-AUTO qualification

Date: 2026-09-09

Device: GeForce RTX 6000 Ada Generation, SM89, 142 SMs

Toolkit cohort in this report: CUDA 13.2 (`cudarc/cuda-13020`)

## Production routes

- NN `(2048, 768, 3072)`, `(2048, 1536, 768)`, and
  `(4621, 384, 1928)` select
  `gemm_bi_nn_fixed_sm89_f32_n64_copyplan_v1` from the `Fixed` module.
- NT `(2048, 1536, 768)` selects the two-node exact pipeline
  `gemm_bi_transpose_f32_32x16_d768_v1` (`TriadScalar`) followed by
  `gemm_bi_nn_fixed_sm89_f32_n64_copyplan_v1` (`Fixed`).
- Selection is fail-closed on the exact artifact/compiler/toolkit/device,
  shape, stride, scalar, bias, and pointer-alignment contract.

## Correctness and route identity

The public AUTO route was inspected after qualification and before timing.
All eager and captured-graph output digests matched the generic exact-F32
reference bit-for-bit. The NT graph manifest contained the expected two nodes,
exact launch configurations, shared scratch pointer, and exact argument values.

The Fixed kernel used 135 registers, zero local bytes, 32,768 bytes static
shared memory, and achieved occupancy 3 on the qualification device. The NT
transpose used 18 registers, zero local bytes, 4,224 bytes static shared
memory, and achieved occupancy 3.

## Timing result

The NN once21 qualification retained all three cells in every eager/graph and
ABBA/BAAB stratum. Public AUTO divided by the composed generic exact route was
approximately 0.68--0.75, a 1.33--1.46x speedup. This exact-F32 route remained
approximately 1.82--2.50x slower than cuBLAS Fast TF32, which has a different
numerical contract and is not used as its exactness acceptance denominator.

The NT d768-out once21 qualification produced the following public-AUTO over
generic-exact ratios:

| Path / order | p50 | p95 |
|---|---:|---:|
| eager / ABBA | 0.549288 | 0.554277 |
| eager / BAAB | 0.549601 | 0.552761 |
| graph / ABBA | 0.534173 | 0.542000 |
| graph / BAAB | 0.536932 | 0.539277 |

This is a 1.82--1.87x p50 speedup over the generic exact NT route. The NT
qualification did not include a vendor comparator, so this report makes no
cuBLAS claim for that cell.

## Commands

The clean remote snapshot was built with:

```text
cargo test --release --no-default-features --features 'cuda,cudarc/cuda-13020' \
  --test gemm_bi_scalar_nn_m64n64_qualification \
  --test gemm_bi_scalar_nt_d768_out_m64n64_tournament --no-run
```

The two ignored qualification tests were then run individually with
`--exact --nocapture --test-threads=1`:

```text
cuda_qualification::ada_integrated_copyplan_auto_three_cell_once21
cuda_tournament::ada_d768_out_integrated_fixed_copyplan_actual_auto_once21
```

Both passed. Five consecutive quiet GPU samples were required before each
launch; no foreign compute process was present.
