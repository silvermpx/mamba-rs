# Inference physical GEMM identity vocabulary

Design-only closure for the 0.7.0 deterministic API work. The source census is
against `5ba6c5e2bacdc6274c732c1741d92a7f42f2e540`; the four relevant files are
unchanged at the later workspace HEAD inspected while writing this note. This
document assigns identities and validation rules only. It does not change CUDA
bytes, selectors, artifact/compiler identities, revisions, or performance
claims.

## Decisions

### Append-only enum allocation

Keep every existing discriminant, including the retired backend holes 12 and
14. Keep `NUMERIC_CONTRACT_DOMAIN` and all route/digest domains that already
exist. Append exactly these values:

```rust
// PhysicalGemmBackend; existing maximum is 30.
InferenceScalarFmaV1 = 31,
InferenceWmmaV1 = 32,
InferenceMma16V1 = 33,
InferenceSm90aWgmmaV1 = 34,
InferenceSm100Tcgen05V1 = 35,
InferenceMmaTf32RnaV1 = 36,
InferenceSm120TmaFmaV1 = 37,
InferenceSm120TmaMma16V1 = 38,
InferenceSm120TmaMmaTf32RnaV1 = 39,
FixedMatvecEightWarpV1 = 40,

// ResolvedNumericContract; existing maximum is 24.
ScalarFmaPostDotBiasV1 = 25,
WmmaF32PostDotBiasV1 = 26,
ScalarFmaEightWarpTreePostDotBiasV1 = 27,

// ResolvedInstructionFamily; existing maximum is 4.
WmmaApi = 5,
```

Also append one permission bit to `NumericContractSet`:

```rust
FIXED_DETERMINISTIC_TF32_V1 = Self(1 << 10)
```

No `ModuleKind`, `ResolvedOperandConversion`, or
`ResolvedOutputOwnership` addition is needed. Every newly represented terminal
owns its output through `OneCtaPerOutputTileV1`. All new backend tags above map
to `ModuleKind::Fixed`. Backend 22,
`ScalarFmaSm89FixedCopyPlanV1`, remains the identity of its one existing Fixed
copy-plan symbol; it must not become the generic Fixed scalar backend.

`WmmaApi` deliberately names the source-level operation and not a PTX opcode.
The local CUDA establishes `nvcuda::wmma::mma_sync` with fragment shape
`m16n16k16`; it does not contain a stable lower-level opcode whose name can be
truthfully recorded. This is a complete physical distinction from inline
`mma.sync`, without inventing PTX.

### Exact new numeric definitions

`ScalarFmaPostDotBiasV1` means:

1. Each output starts from `+0.0f` and consumes K in ascending order through
   explicit F32 fused multiply-add.
2. The completed dot is multiplied by alpha with round-to-nearest F32.
3. Optional F32 bias is added with round-to-nearest F32.
4. Optional beta times the prior output is incorporated by F32 FMA.
5. The result is converted to the storage dtype exactly once at the store.

Copy, slice, TMA staging, K4 load grouping, and CTA thread count do not alter
this numeric definition. They remain distinguishable through backend, symbol,
tile/BK/stages, launch configuration, maps/resources, and arguments.

`WmmaF32PostDotBiasV1` means:

1. F32 accumulator fragments are filled with zero.
2. WMMA `m16n16k16` operations consume K tiles in ascending order.
3. The completed dot receives the same explicit alpha, optional bias, optional
   beta, and single output conversion sequence defined above.

`ScalarFmaEightWarpTreePostDotBiasV1` means:

1. `k_per_warp = round_up_even(ceil(K / 8))`.
2. Warp `i` consumes the contiguous range
   `[i*k_per_warp, min(K, (i+1)*k_per_warp))` in ascending order with F32 FMA,
   starting at `+0.0f`.
3. Warp 0 folds the eight partials as
   `s01=p0+p1`, `s23=p2+p3`, `s45=p4+p5`, `s67=p6+p7`,
   `s0123=s01+s23`, `s4567=s45+s67`, `sum=s0123+s4567`.
4. `sum` receives the same explicit post-dot epilogue and single output
   conversion defined above.

For matvec route metadata, use tile `(1, 32)`, `bk = 0`, and `stages = 1`.
Here zero BK explicitly means “no fixed K tile”: the whole A row is staged once
and the shape-dependent per-warp span is part of numeric contract 27. Recording
8 (the loop unroll) or `k_per_warp` as BK would give the field a different
meaning from tiled GEMM routes.

### Exact existing numeric reuse

Reuse is permitted only for these source-equal definitions:

| Physical source | Numeric contract | Instruction / conversion | Why reuse is exact |
|---|---|---|---|
| Portable `mma16.cu` / `tcw64.cu`, SM89 pipeline/swizzle/S3/finalists, SM120 half TMA | `MmaSyncF32V1` (2) | `MmaSync`, `(16,8,16)`, `None` | Ascending K16 MMA chain, F32 accumulation, bias pre-seeded under the existing alpha-with-bias restriction, one output conversion. |
| Fixed SM90a rung | `WgmmaF32V1` (3) | `Wgmma`, `(64,128,16)`, `None` | First WGMMA group uses zero/scale-D=0, then ascending K16 groups; alpha and bias are post-dot, matching the existing Triad WGMMA definition. |
| Fixed SM100 rung | `Tcgen05F32V1` (4) | `Tcgen05`, `(128,128,16)`, `None` | Bias is seeded into TMEM before the ascending K16 tcgen sequence, matching the existing Triad TCGEN05 definition. |
| Portable TF32, Fixed RNA-wide, Fixed RNA-N96 | `MmaTf32RnaV1` (5) | `MmaSync`, `(16,8,8)`, `RegisterCvtRnaTf32F32V1` | Both operands use explicit register `cvt.rna.tf32.f32`; the F32 accumulator is bias-seeded. |
| Fixed SM120 TMA TF32, including pair-store | `Sm120TmaMmaTf32RnaV1` (8) | `MmaSync`, `(16,8,8)`, `TensorMapUint32ThenCvtRnaTf32F32V1` | Maps carry raw 32-bit values and the kernel performs register RNA before MMA; pair-store changes the schedule/symbol, not arithmetic. |

The forced borrowed wide symbol
`gemm_bi_nn_sm80_mma_tf32_v1_m128n128_bk32_s3` is intentionally not assigned a
new backend. It keeps existing backend `MmaTf32RnaV1` (5), numeric
`MmaTf32AddHalfUlpV1` (23), instruction `MmaSync` `(16,8,8)`, conversion
`RegisterAddHalfUlpTf32V1`, and its actual `TriadSm80` binding. Its integer
half-ulp conversion is not the explicit-RNA contract of the Fixed twin.

The cached exact-TMA bridge likewise keeps its existing
`PreparedF32TriadLaunch` identity, including backend, numeric contract,
TriadSm120 module, scoped revisions, and argument/resource identity. Forward
the observer and record that existing route once; do not manufacture an
Inference alias.

Do not reuse `ScalarFmaV1` for the new post-dot scalar paths and do not reuse
`MmaSyncF32V1` for WMMA. The same Fixed SM89 copy-plan symbol already has an
older Triad-authored `(backend 22, ScalarFmaV1)` route. Preserve that existing
route encoding. New Inference recording of the exact symbol uses
`(backend 22, ScalarFmaPostDotBiasV1)`; validation must admit these two exact
pairs only in their respective family contexts rather than redefining an old
numeric tag.

## Backend-to-terminal mapping

Every row below has `op=Nn`, shape `(M,K,N)`, strides `(K,N,N)`, ownership
`OneCtaPerOutputTileV1`, Fixed artifact identity, and the compiler identity of
the loaded Fixed module, unless the row explicitly says it is an existing
Triad identity. “S2/S3/S4” is copied into `stages`; dynamic shared bytes and
threads are the values passed to the launch builder.

### Scalar and legacy terminals

| Symbols | Backend / numeric | Instruction | Tile, BK, stages; threads; dynamic shared |
|---|---|---|---|
| `gemm_bi_f32_f32_s2` | `InferenceScalarFmaV1` / `ScalarFmaPostDotBiasV1` | `ScalarFma (1,1,1)`, no conversion | `64x64, BK32, S2`; 128; 0 |
| `gemm_bi_f32_f32_n128_s2` | same | same | `64x128, BK32, S2`; 256; 0 |
| `gemm_bi_nn_fixed_sm89_f32_n64_copyplan_v1` | existing `ScalarFmaSm89FixedCopyPlanV1` / new post-dot scalar | same | `64x64, BK32, S2`; 128; 0 |
| `gemm_bi_nn_fixed_sm120_f32_n64_copyplan_v1` | `InferenceScalarFmaV1` / new post-dot scalar | same | `64x64, BK32, S2`; 128; 0 |
| `gemm_bi_nn_fixed_sm120_f32_n64_copyplan_t256_v1` | same | same | `64x64, BK32, S2`; 256; 0 |
| `gemm_bi_nn_fixed_sm120_f32_n64_copyplan_m128n64_t256_v1` | same | same | `128x64, BK32, S2`; 256; 0 |
| `gemm_bi_nn_fixed_sm120_f32_n64_sliced_v1` | same | same | `64x64, BK32, S2`; 128; 0 |
| `gemm_bi_bf16_bf16`, `gemm_bi_f16_f16`, `gemm_bi_bf16_f32`, `gemm_bi_f16_f32` | `InferenceWmmaV1` / `WmmaF32PostDotBiasV1` | `WmmaApi (16,16,16)`, no conversion | `64x64, BK32, S1`; 256; 0 |

The loaded `gemm_bi_f32_f32` legacy symbol is not selected by
`pick_bi_gemm`; do not emit a route for a function that did not execute.

All six Fixed post-bias SM120 symbols use
`InferenceSm120TmaFmaV1 / ScalarFmaPostDotBiasV1`, instruction
`ScalarFma (1,1,1)`, no conversion, BK16, S2, and two actual tensor maps:

| Symbol suffix after `gemm_bi_nn_sm120_tma_fma_v1_` | Tile; threads; dynamic shared | Additional domain |
|---|---|---|
| `fixed_postbias_m128n64_bk16_s2` | `128x64`; 128; 24,592 | bias required by selected route |
| `fixed_postbias_m64n128_bk16_s2` | `64x128`; 128; 24,592 | bias required by selected route |
| `fixed_postbias_m128n96_bk16_s2` | `128x96`; 256; 28,688 | bias required by selected route |
| `fixed_postbias_m128n64_bk16_s2_k4` | `128x64`; 128; 24,592 | exact K4 schedule and symbol |
| `fixed_postbias_m128n64_t256_bk16_s2` | `128x64`; 256; 24,592 | bias required by selected route |
| `fixed_nobias_m128n64_t256_bk16_s2` | `128x64`; 256; 24,592 | bias must be null |

Their unused partial and flag ABI slots are literal nulls and must be present
in the argument identity. They do not imply split ownership.

### Inline half tensor-core terminals

All rows use numeric `MmaSyncF32V1`, instruction `MmaSync (16,8,16)`, and no
operand conversion.

| Symbol family | Backend | Tile, BK, stages; threads; dynamic shared |
|---|---|---|
| `gemm_bi_nn_tc128_{bf16,f16}` and `gemm_bi_nn_tc128_f32out_{bf16,f16}` | `InferenceMma16V1` | `128x128, BK64, S2`; 256; 71,680 |
| `gemm_bi_nn_tcw64_{bf16,f16}` | same | `128x128, BK64, S2`; 128; 65,536 |
| `gemm_bi_nn_tcwn64_{bf16,f16}` | same | `128x256, BK64, S2`; 256; 98,304 |
| `gemm_bi_nn_tc64_{bf16,f16}` and `gemm_bi_nn_tc64_f32out_{bf16,f16}` | same | `64x64, BK64, S2`; 128; 0 |
| `gemm_bi_nn_tc16_{bf16,f16}` and `gemm_bi_nn_tc16_f32out_{bf16,f16}` | same | `16x32, BK64, S4`; 128; 0 |
| `gemm_bi_nn_fixed_sm89_tc128_pipeline_v1_{bf16,f16}` | same | `128x128, BK64, S2`; 256; 71,680 |
| `gemm_bi_nn_fixed_sm89_tc128_swizzle_v1_{bf16,f16}` | same | `128x128, BK64, S2`; 256; 69,632 |
| `gemm_bi_nn_fixed_sm89_tc128_s3_v1_{bf16,f16}` | same | `128x128, BK64, S3`; 256; 98,304 |
| `gemm_bi_nn_fixed_sm89_m64n64_bk64_s3_v1_f16` | same | `64x64, BK64, S3`; 128; 49,152 |
| `gemm_bi_nn_fixed_sm89_m128n64_bk64_s2_v1_f16` | same | `128x64, BK64, S2`; 128; 49,152 |
| `gemm_bi_nn_sm120_tma_64x64_bk64_s2{,_f32out}_{bf16,f16}` | `InferenceSm120TmaMma16V1` | `64x64, BK64, S2`; 128; 32,896 |
| `gemm_bi_nn_sm120_tma_64x128_bk64_s2{,_f32out}_{bf16,f16}` | same | `64x128, BK64, S2`; 256; 49,280 |
| `gemm_bi_nn_sm120_tma_128x64_bk32_s3{,_f32out}_{bf16,f16}` | same | `128x64, BK32, S3`; 256; 36,992 |
| `gemm_bi_nn_sm120_tma_128x128_bk32_s2{,_f32out}_{bf16,f16}` | same | `128x128, BK32, S2`; 256; 32,896 |
| `gemm_bi_nn_sm120_tma_128x128_bk32_s3{,_f32out}_{bf16,f16}` | same | `128x128, BK32, S3`; 256; 49,280 |

The brace notation denotes all four real spellings for an SM120 base:
`BASE_bf16`, `BASE_f16`, `BASE_f32out_bf16`, and `BASE_f32out_f16`.

### WGMMA and TCGEN05 terminals

| Symbols | Backend / numeric | Instruction | Tile, BK, stages; threads; dynamic shared |
|---|---|---|---|
| `gemm_bi_nn_sm90a_wgmma_wg1_{bf16,f16}` | `InferenceSm90aWgmmaV1` / `WgmmaF32V1` | `Wgmma (64,128,16)`, no conversion | `64x128, BK64, S2`; 128; 49,152 |
| `gemm_bi_nn_sm100_tcgen_c4_{bf16,f16}` | `InferenceSm100Tcgen05V1` / `Tcgen05F32V1` | `Tcgen05 (128,128,16)`, no conversion | `128x128, BK64, S2`; 128; 65,536 |

Both sources explicitly define two staging buffers. That common staging count
does not make WGMMA post-dot bias and TCGEN05 seeded bias interchangeable.

### TF32 terminals

All Fixed TF32 symbols have storage triple F32/F32/F32 and require live
`AllowDeterministicTf32V1`. Portable and Fixed SM89 routes use
`InferenceMmaTf32RnaV1 / MmaTf32RnaV1`; SM120 routes use
`InferenceSm120TmaMmaTf32RnaV1 / Sm120TmaMmaTf32RnaV1`.

| Symbol | Tile, BK, stages; threads; dynamic shared |
|---|---|
| `gemm_bi_nn_tf32_v1_m128n64_bk32_s2` | `128x64, BK32, S2`; 256; 55,296 |
| `gemm_bi_nn_tf32_v1_m128n64_bk32_s3` | `128x64, BK32, S3`; 256; 82,944 |
| `gemm_bi_nn_tf32_v1_m64n64_bk32_s2` | `64x64, BK32, S2`; 128; 32,768 |
| `gemm_bi_nn_tf32_v1_m64n64_bk32_s3` | `64x64, BK32, S3`; 128; 55,296 |
| `gemm_bi_nn_tf32_v1_m16n32_bk32_s4` | `16x32, BK32, S4`; 128; 29,696 |
| `gemm_bi_nn_fixed_rna_wide_tf32_v1_m128n128_bk32_s3` | `128x128, BK32, S3`; 256; 98,304 |
| `gemm_bi_nn_fixed_sm89_rna_tf32_v1_m128n96_bk32_s3` | `128x96, BK32, S3`; 256; 86,016 |
| `gemm_bi_nn_sm120_tma_tf32_v1_m128n64_bk32_s2` | `128x64, BK32, S2`; 128; 49,280 |
| `gemm_bi_nn_sm120_tma_tf32_v1_m128n64_bk32_s3` | `128x64, BK32, S3`; 256; 73,856 |
| `gemm_bi_nn_sm120_tma_tf32_v1_m64n128_bk32_s2` | `64x128, BK32, S2`; 128; 49,280 |
| `gemm_bi_nn_sm120_tma_tf32_v1_m64n128_bk32_s3` | `64x128, BK32, S3`; 256; 73,856 |
| `gemm_bi_nn_sm120_tma_tf32_v1_m64n64_bk32_s2_producer_warp` | `64x64, BK32, S2`; 160; 32,896 |
| `gemm_bi_nn_sm120_tma_tf32_v1_m64n64_bk32_s2` | `64x64, BK32, S2`; 128; 32,896 |
| `gemm_bi_nn_sm120_tma_tf32_v1_m64n64_bk32_s2_pair_store` | `64x64, BK32, S2`; 128; 32,896 |

The SM120 dispatcher may return logical tile `Tf32Sm120M64S2` and then choose
the pair-store function. The route symbol and argument digest must be taken
after that choice. A route naming the ordinary function when the pair-store
function executed is invalid even though all geometry fields match.

### Fixed eight-warp matvec terminal

The five symbols `matvec_bi_bf16_bf16`, `matvec_bi_f16_f16`,
`matvec_bi_bf16_f32`, `matvec_bi_f16_f32`, and `matvec_bi_f32_f32` map to
`FixedMatvecEightWarpV1 / ScalarFmaEightWarpTreePostDotBiasV1`, instruction
`ScalarFma (1,1,1)`, no conversion, tile `(1,32)`, BK0/S1, 256 threads, grid
`(ceil(N/32), M, 1)`, and dynamic shared `align_up(K * input_item_bytes, 16)`.
The F32 binary exists, but the current typed public call graph delegates an
all-F32 request through the raw F32 seam before reaching this matvec branch;
do not claim it as currently reachable under logical Triad. The four half-input
triples are reachable under logical Triad, including TC-off and N<32 fallback.

## Complete storage-dtype binding

`ResolvedGemmRoute::dtype` remains the input/execution storage dtype. Add no
public output-dtype field. The physical observation must bind all three storage
dtypes before enqueue and validation must use this exact symbol table:

| Symbol spelling | `(input, weight, output)` |
|---|---|
| Every scalar, TF32, and exact-TMA F32 symbol above; both existing cached bridge symbols; borrowed wide symbol | `(F32,F32,F32)` |
| `*_bf16_bf16` and `*_bf16` without `f32out` | `(Bf16,Bf16,Bf16)` |
| `*_f16_f16` and `*_f16` without `f32out` | `(F16,F16,F16)` |
| `*_bf16_f32` and `*_f32out_bf16` | `(Bf16,Bf16,F32)` |
| `*_f16_f32` and `*_f32out_f16` | `(F16,F16,F32)` |

This table is applied as exact symbol membership, not as a globally permissive
suffix parser. In particular, SM89 N64 finalists, WGMMA, and TCGEN05 have only
homogeneous outputs; portable F32-output exists only for Tc128/Tc64/Tc16; SM120
half has both forms for all five bases. Legacy WMMA and matvec have the four
half/mixed spellings shown. `route.dtype` must equal input dtype, input must
equal weight dtype, and the physical node's logical dtype must equal input
dtype. The symbol table alone decides whether output may be F32.

## Permission, ownership, binding, and revision validation

Replace the numeric-only permission decision in
`GpuCtx::validate_resolved_gemm_route` with an exhaustive decision over at
least `(backend, numeric_contract, symbol, live_family)`. Required backend-set
membership follows `expected_route_module`: Fixed requires `BackendSet::FIXED`;
a Triad module requires `BackendSet::TRIAD`. Do not retain the unconditional
TRIAD requirement.

The allowed new pairs are:

| Backend(s) | Required contract-set bit | Required live family / policy |
|---|---|---|
| `InferenceScalarFmaV1`, and backend 22 paired with numeric 25 | `FIXED_SCALAR_FMA_V1` | `Inference` |
| `InferenceWmmaV1`, `InferenceMma16V1`, `InferenceSm90aWgmmaV1`, `InferenceSm100Tcgen05V1`, `InferenceSm120TmaMma16V1` | `FIXED_MMA_SYNC_V1` | `Inference` |
| `InferenceMmaTf32RnaV1`, `InferenceSm120TmaMmaTf32RnaV1` | `FIXED_DETERMINISTIC_TF32_V1` | `Inference` plus F32 TF32 Allow policy |
| `InferenceSm120TmaFmaV1` | `FIXED_SCALAR_FMA_V1` | `Inference` |
| `FixedMatvecEightWarpV1` | `FIXED_MATVEC_TREE_V1` | `Triad` only |
| existing backend 5 + borrowed wide exact symbol/numeric 23/conversion 4 | `FIXED_DETERMINISTIC_TF32_V1` | explicit `Inference` exception plus F32 TF32 Allow policy |
| existing cached bridge routes | their existing Triad contract bit(s) | explicit `Inference` exception for the exact existing prepared route |

For `BiGemmFamily::Inference`, add
`FIXED_DETERMINISTIC_TF32_V1` to `route_backend_contract_sets` when and only
when F32 policy is `AllowDeterministicTf32V1`, independent of
`bi_tensor_cores`. Do not add either Triad TF32 bit merely because Inference is
selected. Preserve both current FIXED scalar/MMA bits in the Inference false
and true cases.

Do not apply the current blanket “typed non-scalar requires
`bi_tensor_cores`” check to new Inference backends. The production call graph
enters `inference_forward` before the generic typed tensor-core gate, and its
half ladder does not consult that boolean. Existing Triad tensor-core backends
still require the live tensor-core permission. The matvec exception is allowed
for Triad in both boolean states because it serves the scalar tier and the
narrow-N fallback.

For backend 22, retain the old exact `(ScalarFmaV1, Triad)` validation used by
the pre-existing Triad route and add only `(ScalarFmaPostDotBiasV1, Inference)`.
Do not allow either numeric contract with an arbitrary backend-22 symbol.

For every new backend, expected tuning and schedule revisions are the existing
global `TUNING_TABLE_REVISION = 45` and `SCHEDULE_REVISION = 8`. Add no selector
or qualification revision. Backend 22 and both Triad exceptions keep their
existing specialized revision rules. All new routes use the live Fixed binding
from `artifact_set_identity().fixed` and `kernels.compiler_identity()`.

Validation also needs an exhaustive Fixed function lookup matching the symbol
to the actual loaded holder (including `Option` admission for SM89/SM120
specialists). Acceptance by module plus a suffix is insufficient. The same
terminal specification used to build a route should validate backend, numeric,
instruction family/shape, conversion, dtype triple, tile/BK/stages, threads,
map requirement, bias domain, and exact loaded function. This rejects a valid
Fixed symbol paired with another Fixed symbol's otherwise plausible metadata.

## Minimal `PhysicalLaunchObservation` extension

Keep the existing `PhysicalLaunchObservation::gemm` constructor and payload for
existing Triad call sites. Add one private Inference-specific constructor whose
payload includes:

```text
storage_dtypes = { input, weight, output }
arguments = { c, a, b, bias, alpha_bits, beta_bits, abi_kind }
```

`abi_kind` is a closed private enum describing the actual builder layout, not a
free-form label. Its variants need only cover the existing terminal layouts:

- legacy/portable 12 arguments: C, A, B, bias, alpha, beta, M, N, K, lda,
  ldb, ldc;
- five arguments: C, A, B, bias, then the exact 24- or 32-byte parameter
  fields for portable TF32, TF32 wide, SM89 half, or exact F32;
- SM120 TF32: C, map A, map B, bias, `{M,K,N,ldc}`;
- SM120 half: C, map A, map B, bias,
  `{a_x,a_y,b_x,b_y,alpha,beta,M,K,N,ldc}`;
- SM120 post-bias: C, null partials, null flags, map A, map B, bias,
  `{alpha,beta,M,N,K,ldc,splits,tiles_per_split}`.

Hash fields individually; never hash Rust structure padding. At resolve time:

1. Require the observed launch config to equal the route's grid/block/shared
   fields.
2. Apply the exact terminal table and dtype-triple checks above.
3. Require NN, `(M,K,N)`, `(K,N,N)`, alpha/beta and all ABI parameter fields to
   agree with the values actually submitted. This includes the post-bias null
   partial/flag slots and tensor-map coordinates.
4. Compute checked spans: C=`M*N*output_width`, A=`M*K*input_width`,
   B=`K*N*weight_width`, bias=`N*4` when present. Reject every integer or byte
   conversion overflow. C is non-null for nonempty output. A/B are required
   only for K>0; for K=0 bind their actual null/non-null presence rather than
   asking the observer for a fictitious zero-length allocation. Resolve each
   non-null span through `argument_identity_digest`; retain the launcher's
   existing alignment/domain checks. Do not introduce an alias rule that the
   launcher does not currently impose.
5. Recompute a framed digest under a new domain such as
   `mamba-rs.inference-nn-gemm-arguments.v1`, including exact symbol, backend,
   op, the three dtype discriminants, dimensions/strides, alpha/beta bits,
   pointer-null mask, allocation identities for C/A/B/optional bias, the
   fieldwise ABI bundle, tensor-map digest/revision when maps are passed, and
   literal null auxiliary slots. Pair-store and ordinary-store symbols must
   therefore differ even with identical descriptors and geometry.
6. Replace `physical_route.launch.arguments_digest` with this resolved digest,
   just as the current GEMM observer may replace `resources_digest`; return the
   same digest in the physical node's launch. Do not compare a raw-address-free
   context placeholder digest with the allocation-bound digest and fail it.

M=0 or N=0 remains a no-launch/no-record path. K=0 with nonempty output records
the real terminal that launches. Zero-reduction numeric 9 may be reused only
when that terminal implements its exact established epilogue; otherwise keep
the terminal's family numeric contract. The observer must not synthesize a
separate zero kernel or a fake A/B identity.

## Required tests

Pure identity/validator tests:

- assert every old enum discriminant and the appended values 31–40, 25–27,
  instruction 5, and permission bit 10;
- table-test every new backend/numeric/instruction/conversion/module/permission
  tuple and its exact global or retained scoped revision;
- prove Inference Fixed MMA validates with `bi_tensor_cores=false`, while an
  existing Triad MMA does not; prove Fixed matvec validates under logical Triad
  with TC both off and on and rejects under logical Inference;
- reject post-dot scalar as `ScalarFmaV1`, WMMA as `MmaSyncF32V1`, WGMMA as
  seeded TCGEN05, TCGEN05 as post-dot WGMMA, Fixed RNA as half-ulp conversion,
  borrowed half-ulp wide as Fixed RNA, and every wrong Fixed/Triad module;
- retain the exact old backend-22/Triad pairing and admit only the new
  backend-22/Inference pairing for the same exact symbol;
- reject a symbol that is valid in the Fixed artifact but does not match the
  route's backend, geometry, storage triple, or loaded optional holder;
- reject ordinary-store versus pair-store substitution even though launch
  geometry is identical.

Physical observation tests must show failure before the enqueue callback for:

- route input dtype unequal to the storage input, unequal input/weight dtypes,
  illegal output dtype, or an F32-output symbol paired with homogeneous output;
- changed grid/block/shared bytes, alpha/beta bits, dimension/stride/parameter
  field, null mask, tensor map, or pair-store symbol;
- insufficient or overflowing C/A/B/bias spans; nonempty K with null A/B;
- K=0 incorrectly requiring A/B allocation identities;
- a resolved allocation-bound argument digest left only in the physical node
  but not copied into its contained route.

Real CUDA fixtures, when the implementation task is allowed to run them, must
cover legacy F32 and WMMA, every portable half tile class, half-to-F32, SM89
specialists, WGMMA, TCGEN05, Fixed explicit-RNA TF32, borrowed half-ulp wide,
SM120 half/TF32/post-bias, pair-store, the cached exact bridge, and the reachable
half/mixed matvec. Assert unchanged output, exact symbol/config/dtype triple,
and one record per actual terminal enqueue. The cached bridge must not produce
a duplicate. M/N zero produces no record; a supported K0 nonempty output
produces exactly its actual epilogue launch.

## Source anchors

- Existing discriminants, route fields, observer payload/resolve, and route
  hashing: `src/mamba_ssm/gpu/kernel_identity.rs:2941-3100`,
  `:3350-3493`, and `:4791-4950`.
- Current permission/module/revision assumptions:
  `src/mamba_ssm/gpu/context.rs:1408-1660` and `:1930-2020`.
- Fixed module ownership and terminal function loading:
  `src/mamba_ssm/gpu/kernels.rs:969-1040`, `:1167-1285`, and `:1341-1356`;
  optional exact symbol lists and resource admission are in
  `src/mamba_ssm/gpu/gemm_bi_triad/modules.rs:2167-2189`, `:2420-2458`, and
  `:3999-4024`.
- Legacy and matvec selection/configuration:
  `src/mamba_ssm/gpu/blas.rs:3260-3542`; arithmetic bodies:
  `kernels/gemm_bi_inference/wmma_legacy.cu:2-137` and
  `kernels/gemm_bi_inference/matvec.cu:35-196`.
- Inference selectors and terminal builders:
  `src/mamba_ssm/gpu/gemm_bi_inference.rs:1214-1265`, `:1293-1877`,
  `:2568-2889`, `:3090-3582`, and `:3969-4341`. The last range is the source
  for the Inference-before-TC-gate decision.
- Arithmetic definitions: `kernels/gemm_bi_inference/ffma.cu:100-430`,
  `mma16.cu:1-807`, `tcw64.cu:1-580`, `sm90_wgmma.cu:110-242`,
  `sm100_tcgen05.cu:250-314`, `tf32.cu:83-430`,
  `tf32_rna_wide.cu:180-410`, `tf32_rna_n96.cu:135-370`,
  `tf32_sm120.cu:90-445`, `sm120_tma.cu`, and
  `sm120_f32_postbias.cu:1-680`.
- Existing contracts used for exact reuse:
  `kernels/gemm_bi_triad/contract.cuh:21-90`,
  `kernels/gemm_bi_triad/sm90a.cu:148-280`,
  `kernels/gemm_bi_triad/sm100.cu:269-675`, and
  `kernels/gemm_bi_triad/sm80.cu:1914-2420`.

## Blockers

There is no unresolved identity-design blocker. In particular, absence of a
verified WMMA PTX opcode is resolved by the honest `WmmaApi` instruction-family
tag and source-established fragment shape. Implementation still must preserve
the exact live function-holder checks and source-defined K0/domain guards; that
is mechanical validation work, not a new design decision.
