# Vendor terminology for the0.7.0 documentation pass

Checked against NVIDIA's pinned CUDA13.2.1 documentation on2026-09-10.
This is wording support, not a benchmark or an acceptance of pending model API.

- cuBLAS documents repeatable bits under stated toolkit, hardware and execution
  conditions, not an unconditional cross-toolkit guarantee. Stream/workspace
  and certain algorithm choices matter. Do not describe all cuBLAS or TF32
  execution as inherently nondeterministic.
  [NVIDIA cuBLAS, Results Reproducibility](https://docs.nvidia.com/cuda/archive/13.2.1/cublas/index.html#results-reproducibility).
- `CUBLAS_COMPUTE_32F_PEDANTIC` specifies arithmetic precision and restricts
  optimizations; its name does not establish a separate universal determinism
  guarantee. `CUBLAS_COMPUTE_32F_FAST_TF32` permits TF32 Tensor Core arithmetic
  for F32 inputs/outputs. A speed comparison against exact F32 must disclose
  this difference in numerical permission.
  [NVIDIA cuBLAS, compute types](https://docs.nvidia.com/cuda/archive/13.2.1/cublas/index.html#cublascomputetype-t).

Project mode names must describe actual source dispatch. Our own determinism
scope must come from the implemented contracts and recorded tests, not from
contrast with an overstated description of vendor behavior. Preserve separate
claims for storage dtype, compute policy, batch invariance, repeatability and
cross-toolkit/cross-device qualification.
