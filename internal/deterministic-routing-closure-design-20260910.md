# Deterministic routing closure after Task901

2026-09-10. Read-only design against the post-Inference-rename tree; no build,
GPU run, or source edit. Consume Task901's canonical mode and unusable-context
guard. Mode defaults and public constructor propagation are outside this task.

Recommendation: three sequential tasks. The third is a release prerequisite,
not a diagnostic polish task. None requires changing CUDA source bytes.

## 1. Borrowed F32 NN/NT cores and removal of reachable vendor bypasses

Current facts: `blas.rs::gpu_gemm_bi_forward_ptr` unconditionally calls SGEMM;
the tied-head `_raw` wrappers call their `_blas` twins unconditionally.
M3's private `sgemm_no_bias` and its typed no-bias call sites bypass context.
The F32 cached Triad wrappers require `GpuBuffer`, although their prepared
requests already carry `F32TriadOperands` pointers and checked shapes.

Factor existing bodies in `gpu/gemm_bi_triad/launch.rs`, keeping existing public
buffer wrappers as adapters. Proposed internal interfaces (CUptr is CUdeviceptr):

```rust
pub(crate) unsafe fn launch_cached_f32_forward_ptrs(
    ctx: &GpuCtx, y: CUptr, x: CUptr, w: CUptr, bias: CUptr,
    dims: (usize, usize, usize),
) -> Result<(), String>;

pub(crate) unsafe fn launch_cached_f32_backward_dx_ptrs(
    ctx: &GpuCtx, dx: CUptr, dy: CUptr, w: CUptr,
    dims: (usize, usize, usize),
) -> Result<(), String>;
```

These feed `launch_cached_f32_triad`, `F32TriadRequest`,
`F32TriadShape::contiguous`, and `F32TriadOperands` exactly as the current buffer
wrappers do. Keep the private selection-bearing versions, including ExactScalar,
so task 2 can reuse them without changing the user's global F32 policy.

Factor `gemm_bi_forward_sub_with_control` and
`gemm_bi_backward_dx_with_control` down to pointer bodies. Preserve their entire
`ScalarLaunchController` plan/validation/enqueue path, symbol/argument order,
zero-reduction rules, scratch use, and alpha/beta semantics. In particular,
`ScalarLaunchArgs::arg_buffer_mut(dx)` currently both submits an allocation and
captures its pointer for prepared arguments. The raw body must instead submit
`arg(&dx_ptr)`, which already captures identical pointer bytes. Apply this to
every scalar branch, including split-K and transpose branches, not just a
convenient scalar fallback. Existing owned-buffer callers keep ownership and
extract cached pointers while their owners are borrowed.

Raw APIs document that spans belong to the context's allocation domain, do not
overlap contrary to GEMM semantics, and remain live through stream completion
and all captured replays. Validate dimensions, checked span arithmetic, null and
alignment rules using existing shape/allocation validation. Do not manufacture
a `GpuBuffer` from a pointer or use forgotten owners, transmute, or extended
lifetimes. No per-call device copies or allocations for F32.

Add one crate-internal context dispatch seam in `blas.rs`:

```rust
pub(crate) unsafe fn gpu_gemm_f32_forward_ptrs(
    ctx: &GpuCtx, y: CUptr, x: CUptr, w: CUptr, bias: Option<CUptr>,
    dims: (usize, usize, usize),
) -> Result<(), String>;
```

Check Task901's guard before work. Deterministic NN follows the selected family:
Inference reuses `inference_forward` with F32 `TypedPtr`s; Triad uses the raw
cached NN core. Vendor modes use the existing SGEMM branch under the configured
handle. `gpu_gemm_bi_forward_raw`, `gpu_gemm_bi_forward_ptr`, and the all-F32
case of `gpu_gemm_typed_forward_raw` delegate to this seam. This also avoids
sending a large all-F32 NN through the typed matvec fallback.

Tied F32 heads use the raw cached Triad NT core in both families; the Inference
family never receives NT. The tied-head dimensions are easy to swap incorrectly:

| Logical operation | Existing Triad dims `(M,K,N)` | Physical A/B/output |
|---|---|---|
| untied NN logits | `(B,D,Vpad)` | hidden[B,D], head[D,Vpad], logits[B,Vpad] |
| tied NT logits | `(B,Vpad,D)` | hidden[B,D], embed[Vpad,D], logits[B,Vpad] |
| M3 NN projection | `(B,input_width,output_width)` | input, projection weights, output |

For tied NT: `lda=D`, `ldb=D`, `ldc=Vpad`, alpha=1, beta=0, no bias.
Use automatic F32 policy for F32 tied inputs. The old Inference dX branch
currently calls a noncached Triad wrapper; route it through the same cached NT
core so its actual launches participate in capture recording too.

In `mamba3_siso/gpu/inference.rs`, remove its private vendor-only helper (or make
it a trivial context wrapper) and use the pointer seam for all three F32 call
sites. Replace typed no-bias calls with existing
`gpu_gemm_typed_forward_raw(ctx, c, x, w, None, dims)`. In
`module/gpu_lm3.rs`, use context-aware tied/untied wrappers, not `_blas` helpers.
Add `pub(crate) fn ctx(&self) -> &GpuCtx` to `Mamba3Backbone`, matching its
existing `blas()` match over F32/mixed engines; the underlying F32 engine already
has a context accessor. M1's existing call sites then benefit from fixed wrappers.
Keep no-context vendor helpers explicitly vendor-only; no high-level model path
may call them. Their removal from the public API is not needed for this task.

## 2. Typed F32-output tied logits and workspace lifetime

Do not reuse typed NT output followed by an upcast: that rounds logits to half.
Use exact BF16/F16-to-F32 input conversions, then the ExactScalar F32 NT core,
writing directly to the caller's F32 logits. This is a declared composition of
existing conversion and scalar contracts, not a claim of cuBLAS bit parity.
It does not change the homogeneous-half NT/tensor-core contract.

Keep `gpu_gemm_ex_tied_lm_head_raw`'s public signature. Add a private observed
body beside `bi_upcast_to_f32` in `blas.rs`:

```rust
fn gemm_bi_tied_half_f32_in<O: PhysicalLaunchObserver>(
    ctx: &GpuCtx, logits: CUptr, temporal: TypedPtr, embed: TypedPtr,
    dims: TiedLmDims, observer: &mut O,
) -> Result<(), String>;
```

Require matching half input types. Build `HalfPhysicalContext { op: Nt, dtype,
dims: (B,Vpad,D) }`; reserve `with_bi_upcast_scratch((B*D,Vpad*D,0), ...)`.
Reuse `bi_upcast_to_f32` twice, then factor the pointer counterpart of
`record_physical_exact_scalar_f32_backward_dx` to consume those two scratch
pointers and the real logits pointer. Its observer must accept a logical half
dtype with F32 execution/output. Record both conversions, then every physical
NT kernel (including any transpose, partial, and reduce kernels); no output
downcast and no third logits staging allocation. All products are checked.
F32 dtype passed to the typed tied wrapper delegates to the F32 tied wrapper.
Vendor modes use Task901's central compute choice, including fast half compute.

Untied half-to-F32 already has native Inference `launch_f32out_ladder` kernels
(Tc16/Tc64/Tc128), and the Triad-context typed dispatch has a matching matvec
route. Reuse these for the reachable supported triple; do not convert all
untied heads to the slower upcast NT composition. Preserve fail-closed errors
for unsupported mixed input triples.

The LM head currently runs eagerly outside the backbone graph in both models.
Nevertheless, its context scratch is shared with the captured backbone:
`with_bi_upcast_scratch` rejects growth after `freeze_graph_scratch`. Therefore
before either LM `capture_graph`, reserve the maximum of backbone and tied-head
scratch by making a no-launch head workspace reservation, then let the backbone
presize and freeze. Prefer a narrow `pub(crate) fn presize_tied_lm_head_scratch(
ctx: &GpuCtx, dtype: WeightDtype, dims: TiedLmDims) -> Result<(), String>` in
`blas.rs`, called from both LM wrappers; avoid adding head logic to generic
context construction. Existing scratch grows monotonically and never shrinks.

The owned context resource anchor already retains scratch for graph lifetime.
Keep original hidden/embedding/logit owners live; serialize cast/GEMM/backbone
work on that context's stream. An unexpected larger shape after freeze errors
before enqueue. A future head-inclusive graph can reuse these buffers and the
existing prepared conversion graph launches, but is not part of this release
closure. Measure/document the additional `(B+Vpad)*D*4` bytes and per-call input
conversion cost; avoid a persistent converted-weight cache in this first fix.

## 3. Complete Inference route recording and M3 capture manifests

**Release-blocking fact:** `gemm_bi_inference.rs` has no
`record_resolved_gemm_route` calls. `require_f32_triad_graph_plan` requires a plan
only for family Triad and logical F32. Consequently, successful M1/M3 graph
smoke and a correct `gemm_route()` label do not prove that default Inference
execution has a physical inventory. Copying the M1 capture wrapper alone leaves
this hole. Also keep distinct the context's GEMM route manifest and the fuller
`RecordedPhysicalTrace` that includes casts/transforms.

Reuse `InferenceTile`, `InferenceFwdOperands`, `InferenceShape`, checked
`FixedArgs`, the existing per-architecture preparation functions and tensor-map
caches. Do not add a second tile selector or change AUTO admission. The missing
seam is immediately at each existing selected launch helper's builder/config:
bind its actual module, symbol, tile, numeric contract, arguments, grids, and
resources into `ResolvedGemmRoute`, validate it, record it, then enqueue through
`enqueue_with_physical_observation` when a physical observer is active.

Add an internal observed dispatch entry using existing types:

```rust
pub(crate) fn inference_forward_observed<O: PhysicalLaunchObserver>(
    ctx: &GpuCtx, operands: InferenceFwdOperands, shape: InferenceShape,
    observer: &mut O,
) -> Result<InferenceTile, String>;
```

Existing `inference_forward` delegates with `NoPhysicalObserver`; context route
recording must still happen in this production path. Thread the observer to the
selected launch helpers, including `fixed_legacy_forward`, F32 exact, TF32,
homogeneous half, native half-to-F32, and architecture/TMA helpers. Construct
identity from actual arguments, not from the returned tile after execution.
Reuse route/physical argument builders and allocation-domain checks; extend
validation narrowly for Inference symbols and distinct input/output dtypes
where the current Triad-only assumptions reject them. Allocate new versioned
identities where representation requires them, never relabel frozen identities.

Warm architecture self-checks/tensor maps before recording the manifest; otherwise
the one-time self-check's extra real kernels make eager and capture inventories
differ. A capture-time cold preparation must error before enqueue, not allocate
or silently omit launches. Record all actual AUTO rungs; an unsupported identity
must reject capture explicitly until covered. Do not claim the capture task
complete with the default Inference branches still unrecorded.

In both M3 F32 and mixed engines, mirror M1's
`Cell<Option<PreparedGemmCaptureManifest>>` and
`Option<CapturedGemmGraphPlan>` fields. Successful eager step paths record the
actual body via `record_eager_gemm_manifest`. Capture uses
`capture_into_graph_with_gemm_plan` and its exact capacity/digest check; replay
uses `CapturedGemmGraphPlan::with_validated_launch` via the existing wrapper.
Cover both host-input and GPU-input step paths. Retain state/scratch pointer
guards, mixed scratch freeze/pointer checks, context/resource ownership, and
graph destruction order. Require nonempty inventory for every deterministic
model body known to execute GEMMs, regardless of family or storage dtype.

Nonempty alone is insufficient: tests below must verify the complete per-model
projection inventory. For conversion-inclusive graph qualification, reuse
`PreparedPhysicalCaptureManifest` / `prepare_conversion_graph_launch` /
`RecordedPhysicalTrace` rather than presenting the GEMM-only context manifest as
proof that every cast was recorded. No new universal tracing framework is needed.

## Concrete tests that expose the old bypasses

1. CUDA integration fixtures hold real `GpuBuffer`/`DtypedBuf` owners and pass
   interior pointers with nonzero offsets into NN and NT public wrappers. Choose
   asymmetric shapes, e.g. B=3, D=37, Vpad=96, with output guards. Compare F32
   results to explicitly selected scalar reference execution, repeat bitwise,
   and require nonempty resolved routes with NN versus NT and correct strides.
   Test both families, optional NN bias, zero reduction, and invalid spans.
   The old raw wrappers produce an empty custom trace, so this is a real red
   test rather than a test of enum mapping.
2. For tied BF16/F16 use B=2, D=1, Vpad=96 and exactly representable half inputs
   `1 + 2^-p` with p=7 for BF16 or p=10 for F16. Their product contains a low bit
   representable in F32 but lost by a half output. Require exact F32 product
   bits, two InputUpcast observations, NT GEMM output bound to the F32 logits
   allocation, and **zero OutputDowncast** observations. Add irregular D=37
   cases and a scalar multi-launch NT shape. Use the real observed production
   body, not a parallel reconstructed launch; extend the existing ignored CUDA
   physical tests in `tests/kernel_identity_cuda.rs` or colocated unit tests.
3. Build small one-layer M1/M3 models for F32/BF16/F16 and tied/untied heads;
   exercise real `compute_logits` through step APIs. For M3 use nonidentity
   input_dim != d_model to exercise the optional input projection. Assert the
   ordered per-layer in/out projection inventory plus that input projection,
   and the separate head NN/NT inventory; compare eager and captured replay
   outputs after resetting recurrent state. Module-local tests may access the
   private logits function without exposing another production API.
4. Run a real first step, reserve the large tied-head workspace, capture the
   backbone, then compute logits/replay repeatedly. Assert scratch addresses do
   not change and no growth error occurs. Negative tests grow after freezing,
   omit one expected recorded launch, mutate the family/mode, and change a
   bound operand/physical manifest; reject before graph launch. Eager and capture
   manifest equality must fail for an omitted projection even if other routes
   remain and the context mode is unchanged.
5. A custom trace records only observed custom calls; it cannot prove absence
   of an additional hidden vendor launch. Pair complete inventory assertions
   with a test vendor-call interception/tripwire at every remaining production
   cuBLAS GEMM boundary (or existing external CUDA API trace in GPU CI). Run
   each high-level deterministic model fixture with vendor GEMM forbidden;
   vendor-mode controls must trip/count the hook so a broken hook cannot pass.
   Include an `rg` census of SGEMM/GemmEx call sites as supporting review evidence,
   not as the runtime proof. No-vendor means no vendor GEMM execution; merely
   owning/initializing a cuBLAS handle is already allowed by Task901.

Acceptance is all three tasks plus those real routes/graph checks. Source/plan
fixtures, a mode label, or positive graph smoke alone are insufficient.
