# Inference identity source census

Source baseline: `de8b463ec1ad37448a448bc4e39a6415b789808a`.
Read-only source inspection on 2026-09-10, while Task 902 owns raw F32 routing.
This supplements `inference-m3-physical-recording-design-20260910.md`.
It does not qualify performance or assert cross-architecture bit equality.

## Module ownership

`MambaKernels::new` compiles `ModuleKind::Fixed`, then its `get` closure loads
functions from `fixed.module` (`kernels.rs`, around lines 696 and 969).
This includes legacy WMMA, legacy FFMA, and `matvec_bi_*`, not just the named
Inference ladder. The existing module name and artifact identity must remain.
The two exceptions already identified in the recording design remain real
Triad module launches: the cached SM120 exact bridge and the forced non-RNA
wide TF32 arm. Never relabel these as Fixed or record the bridge twice.

## Arithmetic classes which must not be conflated

| Source family | Observed arithmetic / bias placement | Identity consequence |
|---|---|---|
| `ffma.cu`, SM89/SM120 N64 copyplans, SM120 sliced | Ascending K scalar FMA, then explicit multiply by alpha, add bias, optional beta FMA | Needs a post-dot-bias scalar contract; ordinary Triad scalar bias-seeded identity is not interchangeable. Slice/copy scheduling is not split-K arithmetic. |
| `sm120_f32_postbias.cu` | Ascending scalar FMA with TMA staging; post-dot bias, plus explicit no-bias twin; unused partial/flag ABI slots | Same arithmetic class can share a semantic tag if all supported scalar/zero cases agree, but needs its actual TMA backend and map/argument binding. Do not use Triad one-split bias-seeded identity. |
| `wmma_legacy.cu` | WMMA `m16n16k16` fragments, zero accumulator, post-dot alpha/bias/beta and one output cast | A distinct WMMA backend/numeric class. Do not claim this is the ladder's inline `mma.sync.m16n8k16` contract merely because both use Tensor Cores. |
| `mma16.cu`, `tcw64.cu` | Inline MMA `m16n8k16`, ascending K, bias pre-seeded; alpha=1 required with bias | Existing `MmaSyncF32V1` may be reusable after comparison with its complete definition. Backend is nevertheless Fixed-owned. Native-half and F32-output symbols must retain different dtype triples. |
| SM89 half pipeline/swizzle/S3 | Same stated k16 MMA and pre-seeded bias; N64 forced finalists accept no bias, alpha1/beta0 only | Bind the actual selected symbol and supported domain. No extra semantic class solely for a copy/swizzle change. |
| `sm120_tma.cu` | TMA staging into inline `m16n8k16` MMA; pre-seeded bias, output multiply and optional beta FMA | Distinct Fixed TMA backend, with actual maps; homogeneous/F32-output selected function matters. |
| `tf32.cu`, RNA wide, RNA N96 | Explicit register `cvt.rna.tf32.f32`, bias-seeded MMA; N96/wide alpha/beta epilogues | Keep explicit RNA conversion distinct from the borrowed half-ulp wide arm. Reuse a numeric tag only after checking its full definition/domain. |
| `tf32_sm120.cu` | TMA plus register `cvt.rna.tf32.f32`, bias-seeded MMA; optional pair store | Record the final pair-store function rather than just the returned tile. Map data type/conversion must come from actual descriptors. |
| `sm90_wgmma.cu` | `m64n128k16` WGMMA, first group scale-D=0, bias added after the dot and alpha | Distinct Fixed WGMMA backend and post-dot-bias semantic class. The source explicitly distinguishes it from bias-seeded MMA ladder arithmetic. |
| `sm100_tcgen05.cu` | Bias pre-seeded into TMEM before tcgen05; alpha/beta epilogue | Distinct Fixed TCGEN05 backend. Do not inherit the WGMMA post-dot-bias class. |
| `matvec.cu` | Eight contiguous even-rounded K ranges, one per warp; pairwise fixed tree `(p0+p1)+(p2+p3)` and `(p4+p5)+(p6+p7)`, then final sum; post-dot alpha/bias/beta | Separate fixed-eight-warp reduction contract, even though instruction family is scalar FMA. Not an ascending single FMA chain. |

The table is a source-backed classification, not the final enum allocation.
Existing backend discriminants end at 30 and have retired holes 12/14; append
new values rather than filling holes. Numeric tags currently end at 24.
`ScalarFmaSm89FixedCopyPlanV1` (backend 22) is an existing narrow qualified
contract, not a universal alias for all Fixed functions.

## Additional terminal launch missed by the earlier change map

`blas.rs::gpu_gemm_typed_forward_raw` can reach `launch_bi_matvec` in
Deterministic/Triad mode for small or uncovered homogeneous-half shapes and
half-input/F32-output triples. This lies outside `inference_forward`, so
instrumenting the Inference ladder alone does not close model route inventory.
Task 902 is changing the F32 branch; inspect its final source before defining
the remaining matrix, but do not drop the half/F32-output matvec path.

The actual matvec module is Fixed even when the selected logical family is
Triad. Validator authorization must follow this existing permitted fallback,
not reject it just because FIXED is not the selected logical family. Conversely,
this exception must not broadly authorize arbitrary Inference families.

Matvec dynamic shared memory is K times input item size; grid is
`(ceil(N/32), M, 1)`, block 256. Record actual pointers, dtype triple, strides,
bias/scalars and launch configuration. Zero dimensions and excessive K/shared
memory need the existing host guard behavior preserved and tested; this census
does not claim arbitrary K is a valid launch.

## Validator and tests

`context.rs::validate_resolved_gemm_route` currently assumes TRIAD permission
bits and module-bound backend tags. Extend this narrowly using the backend's
actual module and arithmetic contract. Review logical-F32 scalar admission,
TF32 policy, tensor-core-off scalar paths, live module binding, revision checks,
and zero-reduction epilogues together. Do not let a new tag bypass artifact,
compiler, device, function, allocation or argument checks.

Required negative fixtures include a post-dot-bias route mislabeled as seeded
scalar, WMMA mislabeled as inline MMA, wrong Fixed/Triad module, changed output
dtype/symbol, and changed pair-store function. Actual launch fixtures should
exercise the remaining matvec fallback as well as ladder and cached bridge.

No CUDA bytes, compiler keys, selector thresholds, or measured winners were
changed by this census. Generated PTX inspection is still needed before naming
a WMMA instruction family/shape as a lower-level physical opcode; the source
only establishes the WMMA fragment operation here.
