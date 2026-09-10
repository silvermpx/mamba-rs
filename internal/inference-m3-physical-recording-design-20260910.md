# Inference physical recording and M3 replay: concrete change map

2026-09-10; design only. No source, Git, test, or GPU execution. Consume the
current `GemmMode`/`ensure_gemm_usable` foundation. This narrows section 3 of
`internal/deterministic-routing-closure-design-20260910.md`.

## Corrected facts and boundaries

Most `gemm_bi_inference.rs` launch helpers enqueue directly and never publish a
resolved route. There is an important exception: AUTO SM120 exact A0/B0 calls
`gemm_bi_triad::launch_cached_fixed_sm120_exact_tma`; its prepared launch path
already records a route. Preserve that bridge and count it exactly once.

M1's `PreparedGemmCaptureManifest` / `CapturedGemmGraphPlan` records ordered
resolved GEMM launches. It is not the conversion-inclusive
`PreparedPhysicalCaptureManifest` / `RecordedPhysicalTrace`. M3 currently has
only logical `captured_gemm_route` and pointer guards. Its mixed decode body
asserts `identity_proj=true`; nonidentity input-projection coverage belongs to
the F32 engine, not an unsupported mixed-decode fixture.

Raw F32 routing task must land before end-to-end M3 manifest acceptance: otherwise
F32 projection calls never enter these recorders. Its typed no-bias replacements
are similarly required for mixed decode. Tied-logit upcast/F32-output task is not
needed to capture the backbone, since heads run eagerly outside that graph;
it is required for LM no-vendor acceptance and pre-capture workspace reservation.
Inference route instrumentation itself can be implemented/tested first with
direct real `inference_forward` calls. Do not expand this task into mode plumbing.

## Smallest implementation sequence

### A. Bind identities where the actual Inference launch is built

Keep `InferenceTile`, `InferenceFwdOperands`, `InferenceShape`, `FixedArgs`, AUTO
selectors, qualified admission, and tensor-map caches. Add the private observed
entry proposed in section 3, with existing public `inference_forward` delegating
through `NoPhysicalObserver`. No second selector, prepared-launch framework, or
owning pointer wrapper is needed.

Thread an observer through the existing direct-launch helpers below. Bind the
resolved identity from the same locals used by the builder, publish through
`ctx.record_resolved_gemm_route`, and submit through
`enqueue_with_physical_observation`. Perform validation/recording before enqueue;
an enqueue error makes the surrounding trace/capture fail and discard its
partial inventory. `NoPhysicalObserver` must still publish context GEMM routes.
Call `ensure_gemm_usable` at the entry.

| Existing hook | Required binding / special case |
|---|---|
| `blas.rs::fixed_legacy_forward` → `launch_bi_gemm` | Bind selected `pick_bi_gemm` symbol, dtype triple and actual launch; this final fallback is part of Inference coverage. Keep generic helper behavior scoped to Inference calls. |
| `launch_f32_n128_s2` | Actual 64×128/256-thread scalar launch; zero M/N returns no record. |
| `launch_sm89_exact_n64`, `launch_sm120_exact_n64`, `launch_sm120_copyplan_t256`, `launch_sm120_sliced` | Reuse their checked `prepare_*`/parameter objects, actual configuration, admission and function holder. Force-only branches get the same instrumentation as AUTO. |
| `launch_sm120_tma_postbias` | Exact selected bias/no-bias/K4/T256 symbol, parameter bundle, two tensor maps, null partial/flag slots, actual threads/shared bytes. |
| `launch_ladder` | Tc16/Tc64/Tc128/TcW64/TcWn64 and SM90/SM100 selected symbols; instruction family is not universally MMA-sync. |
| `launch_f32out_ladder` | Input half, output F32, actual `_f32out` symbol. Never label this as homogeneous half storage. |
| `launch_sm89_half_pipeline`, `launch_sm89_half_swizzle`, `launch_sm89_half_s3`, `launch_sm89_half_n64` | Actual finalist symbol/parameter ABI, tile and resource state; no renaming to a Triad finalist just because tile geometry matches. |
| `launch_tf32`, `launch_tf32_rna_n96`, `launch_tf32_wide` | Bind actual portable/RNA symbol and ABI. `launch_tf32` is partly a dispatcher: record in its terminal launcher, not again at its caller. |
| `launch_sm120_tf32` | Bind final function after pair-store selection, maps and actual launch. A returned `Tf32Sm120M64S2` tile can execute the pair-store symbol, so tile-only recording is wrong. |
| `launch_sm120_half` | Bind inner `InferenceSm120HalfTile`, input/output dtypes, selected homogeneous or F32-output function, maps and parameters. |
| exact-TMA cached bridge | Existing `PreparedF32TriadLaunch` routes remain authoritative. Add only observer forwarding if needed; never construct a second Inference alias route. |

All these routes are NN. Record each physical launch once, in enqueue order.
If a selected path has several physical kernels, each gets its own record;
do not synthesize one record for the high-level GEMM. The shared cached bridge
already follows this model. Existing zero-output no-ops contribute zero records;
nonempty zero-reduction epilogues contribute their real launch.

### B. Extend the existing identity vocabulary, without false aliases

There is no existing universal “Inference prepared identity” to call.
`ResolvedGemmRoute` is the reusable record. It already carries operation, dtype,
backend, numeric contract, instruction/ownership/conversion, symbol/module,
artifact/compiler/device, shape/strides, tile/BK/stages, launch argument digest,
map/resource digests, tuning and schedule revisions.

Use `ModuleKind::Fixed` and `artifact_set_identity().fixed` for actual functions
compiled into that existing module. Keep this versioned name. The SM120 cached
bridge intentionally uses its real TriadSm120 qualified module and identity.
The force-only `Tf32M128N128S3` arm of `launch_tf32_wide(..., false)` also
borrows `gemm_bi_nn_sm80_mma_tf32_v1_m128n128_bk32_s3` through `tf32_function`;
retain that symbol's actual Triad binding and half-ulp conversion, distinct from
the Inference-owned RNA-wide twin. Module ownership follows the function holder.
`live_gemm_module_binding(ModuleKind::Fixed)` already returns its actual artifact
and compiler. Do not substitute the Triad artifact for an Inference symbol.

Current backend enums are module-bound: `ScalarFmaV1` requires TriadScalar,
`Sm80Mma16V1` requires TriadSm80, WGMMA/TCGEN require specialized Triad modules.
Thus ordinary Inference functions cannot simply reuse those backend tags.
Append narrowly named versioned Inference backend tags for the physical classes
actually encountered (scalar, MMA-sync half, WGMMA, TCGEN05, MMA TF32, and SM120
TMA classes), mapping them to their actual module. Reuse existing numeric
contract tags only where their defined reduction/conversion/epilogue contract
is identical; otherwise append a new numeric tag. Preserve all old discriminants,
digests, artifact bytes, and compiler identities. The existing
`ScalarFmaSm89FixedCopyPlanV1` is reusable only for that exact existing copy-plan
contract, not as a catch-all Fixed backend.

`context.rs::validate_resolved_gemm_route` currently maps numeric contracts to
TRIAD permission bits and assumes Triad module/backend combinations. After
Task901 ownership is released, extend that function plus
`expected_route_module`, tuning/schedule revision checks, and F32/typed backend
admission to the new Inference tags. Use the existing FIXED scalar/MMA policy
bits for their contracts. Explicit Inference TF32 needs its own permitted
contract mapping; do not grant it by enabling the Triad TF32 bit regardless of
family. Retain exact F32 policy, actual loaded symbol, artifact/compiler and
device checks. Separate backend ownership from the semantic arithmetic tag.

No new output-dtype field is necessary for the minimum slice: record input
dtype in the existing dtype slot, distinguish F32-output by exact symbol and
include the full `(input_dtype, weight_dtype, output_dtype)` in a versioned
argument digest and checked spans. This matches the existing ability to record
typed-input/F32-output TN. Update symbol/physical-node validation to admit the
actual F32-output Inference symbols; do not weaken suffix validation globally.

Hash actual scalar arguments and parameter fields (not Rust padding), allocation
identities, bias presence, and map bindings using existing framed digests and
physical range machinery. Pair-store, descriptor, output dtype, or pointer
changes must change identity. Add identity metadata in Rust; no CUDA edits.

### C. Install the existing M1-style GEMM manifest in both M3 engines

In `mamba3_siso/gpu/inference.rs`, add eager manifest and captured plan fields
to `Mamba3GpuInferenceEngine` and `Mamba3GpuInferenceMixed`. Four eager entry
branches require recording: `step`, `step_gpu_only`, `step_mixed_native`, and
`step_gpu_only_mixed_native`. Record their existing private kernel body, keeping
H2D/D2H outside. Clear the cached eager manifest before attempting a new body;
publish it only on success, so a failed step cannot leave an apparently current
capture permit. Do not wrap both the public step and its body in nested recorders.

Both capture methods consume the successful eager manifest, use
`capture_into_graph_with_gemm_plan`, and store its plan only after the body and
manifest validation succeed. Require a plan whenever deterministic execution
has a nonzero expected GEMM workload. Replace the family-specific helper with
a shared `require_deterministic_gemm_graph_plan(ctx, has_gemm_work, plan, label)`;
use it for M1 too, since Inference can no longer be exempt from the guard.
Move/reuse M1's small `with_validated_launch` wrapper in `graph_capture.rs` for
all four M3 replay branches, ensuring usability and plan validation precede
`graph.launch`. Keep logical route checks, state/scratch pointer assertions,
mixed scratch freeze/addresses, resource anchoring and drop-time synchronization.

Expected ordered workload for L layers: F32 nonidentity projection is first
NN `(B,input_dim,d_model)`; then for each layer NN in-proj
`(B,d_model,cfg.in_proj_out_dim())` followed by NN out-proj
`(B,cfg.d_inner(),d_model)`. Identity F32 and mixed decode omit the first GEMM;
mixed therefore has exactly 2L logical projections. Physical record counts may
exceed logical counts when a route decomposes. Heads are a separate eager
NN `(B,D,Vpad)` or NT `(B,Vpad,D)` inventory, never part of backbone capture.

Architecture rung self-check uses two real custom GEMMs on temporary buffers.
Resolve it before recording the serving manifest (and before CUDA capture),
using the existing once-only gate. Do not hide its launches under a recorder
and call the result a complete physical trace. Warm descriptor caches before
capture; a cold map or prepared-cache miss errors before enqueue. If a first
step performs a self-check, do not publish that contaminated inventory as the
serving capture manifest: require a subsequent successful warm eager step.

## Test inventory / runtime proof

- Direct real Inference fixtures in `tests/kernel_identity_cuda.rs` or module
  CUDA tests: F32 Legacy, each portable half rung, half-to-F32 output, then
  hardware-admitted exact/TF32/half specialized rungs. Use forced APIs to cover
  helpers and AUTO fixtures to prove the selected production branch uses them.
  Verify exact symbol/config/argument binding, correct module/compiler, ordered
  count, and identical output before/after recording. SM120 exact bridge must
  produce one set of existing records, never duplicates. Pair-store fixture
  verifies the executed symbol differs when its schedule cell applies.
- Extend `tests/inference_graph_route.rs`: its current setup explicitly selects
  Triad. Add Inference for F32/BF16/F16, B=1 and B=3. Reuse `m3_config`/seeded
  weights and an identity projection; add nonidentity F32 weights separately.
  Check the actual eager projection sequence above, not merely `has_graph`,
  route flags or finite output. Capture/replay after resetting recurrence and
  compare with eager output. Cover all four step entry points.
- Existing kernel-identity mutation tests provide the pattern: assert callback
  was not invoked for a tampered route order, symbol/module, output-dtype digest,
  descriptor/resource binding, live family/mode, or unusable context. An empty
  deterministic manifest errors. Remove the middle projection record in a
  multi-layer fixture; exact manifest comparison must reject despite nonempty
  inventory. Invalid/cold preparation must not leave a stale eager manifest.
- Complete inventories are necessary but cannot detect an extra hidden vendor
  call. Recommended minimum tripwire: a test-only thread-local count/deny hook
  immediately before every production cuBLAS SGEMM/GemmEx boundary, including
  retained no-context twins. Run real deterministic model step, capture and
  head fixtures under deny, expecting zero hits. Positive control first runs
  real CublasFast/Pedantic calls with counting enabled and sees >0, then with
  deny sees the hook error before FFI. Test-only module tests can reach hooks;
  integration tests need an explicit testing feature rather than assuming
  `cfg(test)` is active in the library. Census all vendor FFI sites to ensure
  hook placement is complete; do not route the new raw wrappers around it.
- Optional stronger GPU-CI evidence is library/API interception with a positive
  control. `LD_PRELOAD` alone is not sufficient proof for cudarc's dynamically
  loaded symbols; validate that the chosen interception actually observes the
  explicit vendor controls. No existing usable interception harness was found
  in the inspected source/tests. Do not present this option as already available.

Scope of the result: complete **GEMM** physical inventory and guarded M3 replay,
plus vendor-call evidence. It does not claim to trace every normalization/SSM
kernel or cast. The typed tied-output task owns conversion-inclusive evidence
using existing physical observers, and its workspace reservation is required
before shared scratch is frozen. No new universal graph tracer is proposed.

Critical implementation detail still requiring a bounded contract census is the
new Inference backend/numeric tag list: current enums do not represent all these
module/epilogue combinations. This is required identity work, not permission to
alias them to Triad or silently skip unrepresented branches.
