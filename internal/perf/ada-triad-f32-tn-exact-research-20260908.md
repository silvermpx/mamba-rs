# Ada exact-F32 TN optimization order, 2026-09-08

This is the source-backed next-candidate order after the retained d768-in
SplitM CopyPlan result. It is not performance evidence and does not admit a
dispatcher route.

The d768-in candidate executes about 9.664 GFLOP in about354.1us on its graph
path (~27.3TFLOP/s), while the measured Fast comparator is about136.4us
(~70.9TFLOP/s). Small cache-operator changes alone are unlikely to close that
gap. The next screens must remove whole-memory passes or reduce the physical
work while preserving the exact arithmetic tree.

## Candidate order

1. **Direct-TN CopyPlan.** Load each original row-major X chunk directly into
   a transposed shared tile with coalesced 16-byte `cp.async`; keep ascending
   `__fmaf_rn` order, the exact SplitM boundaries and the existing FP64 reducer.
   This removes the global transpose node and its temporary write/read.
2. **Fuse the final chunk and reducer.** Write partial0 normally; after the
   unchanged second ascending FMA chain, perform the same ordered FP64 sum and
   final update inside chunk1. Keep an explicit chunk0-to-chunk1 dependency.
   Do not combine this with independent `grid.z=2` execution.
3. **Screen `GROUP_M=12`** for the d768-in physical M=768. This preserves the
   M-fast raster while reusing one B tile across all twelve M tiles rather than
   an 8+4 grouping.
4. **Screen a 64x32x32/S2 tile.** The current 64x64 body uses135 registers and
   32KiB shared per128-thread block, both capping residency at three blocks/SM.
   A 64x32 candidate can target <=128 registers and <=25KiB shared, which are
   both required to make four blocks/SM possible. Do not use `maxrregcount`
   alone or accept spills.
5. Only after the structural screens, inspect SASS/Nsight evidence for scalar
   shared loads, long-lived 64-bit address state and shared-bank wavefronts.
   Then try exact-neutral vector loads or `.ca`/`.cg`/L2 prefetch hints one at a
   time.

A cheaper alternative to step2 is one `grid.z=chunks` raw-partial launch plus
the old reducer. It removes launch overhead but not partial traffic, and must
not reorder the reducer.

## Numeric and hardware constraints

- Never change chunk boundaries, ascending `__fmaf_rn` association, or the
  FP64 reducer order. No even/odd split, tree reduction or atomic reduction.
- `cp.async` fast-path operands must remain 16-byte aligned; tail bytes use the
  instruction's zero-fill mechanism and commit/wait must remain converged.
- TF32 tensor cores cannot implement the full-mantissa exact-F32 contract.
- TMA/WGMMA are Hopper-class mechanisms, not Ada SM89 mechanisms.
- Triple-buffering the current 64x64 tile would raise shared use to about48KiB
  and likely reduce residency to two blocks/SM; do not screen it without new
  evidence.
- If a transpose remains, the current 32x33 padded layout already addresses
  the canonical bank-conflict problem. A 32x8/256-thread geometry is a bounded
  exact-neutral alternative to the current 32x16/512-thread launch.

## Primary sources

- [NVIDIA Ada GPU Tuning Guide](https://docs.nvidia.com/cuda/ada-tuning-guide/)
- [NVIDIA Ada GPU Architecture whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)
- [NVIDIA RTX 6000 Ada specifications](https://www.nvidia.com/en-us/products/workstations/rtx-6000/)
- [CUDA C++ Best Practices: asynchronous copy and shared memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [PTX ISA: `cp.async`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
- [CUDA Programming Guide: asynchronous copies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)
- [CUDA Programming Guide: pipelines](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html)
- [CUTLASS efficient GEMM design](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md)
- [NVIDIA canonical padded transpose sample](https://github.com/NVIDIA-developer-blog/code-samples/blob/master/series/cuda-cpp/transpose/transpose.cu)
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
- [CUDA single-precision intrinsic reference for `__fmaf_rn`](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html)
- [NVIDIA Ampere Tuning Guide: TF32 precision](https://docs.nvidia.com/cuda/ampere-tuning-guide/)

## Working protocol

For each step: source/profile hypothesis, native structural tests, one focused
GPU exact/resource screen, then paired once7 only if exact passes. Stop a valid
loser; freeze and report a winner immediately. Run dispatcher integration and
the full CUDA12.8/13.0/13.2 qualification batch only after the retained-best
shortlist is complete.
