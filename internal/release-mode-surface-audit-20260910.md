# 0.7.0 GEMM release-mode surface audit

Read-only audit against base `b47ff405`. The concurrent mechanical `Fixed` ->
`Inference` rename may move lines or replace that identifier; this report uses
`Inference` for the intended role and notes the base name where useful. No new
default is recommended here.

## Minimal public integration seams

- `src/mamba_ssm/gpu/context.rs:42-65` owns `BiGemmFamily`. `Triad` is the
  NN/TN/NT/backward-capable family and current default; base `Fixed` (renaming to
  `Inference`) is forward-NN-only. A deterministic trainer must therefore use
  `Triad`; it must not inherit the inference role family.
- `src/mamba_ssm/gpu/context.rs:493-555,631-668` owns context construction and
  environment application. `GpuCtx::new` ignores the environment, while
  `new_from_env*` consumes the legacy boolean/policy variables. The no-env state
  is currently a hybrid: batch invariance off, `fast_gemm=false`, cuBLAS TF32
  math enabled, F32 exact policy, and half tiled policy. It does not cleanly
  equal any proposed public mode.
- `src/mamba_ssm/gpu/context.rs:902-990,1462-1518` owns policy mutation.
  `set_fast_gemm` controls typed half/BF16 compute but not F32 handle math;
  `disable_tf32` changes the handle to default math and is one-way. A coherent
  mode setter/constructor must atomically configure both dimensions and needs a
  reversible, fallible handle-math setter.
- `src/mamba_ssm/gpu/context.rs:130-170,493-555,1947-2105` owns strict env
  parsing and its unit tests. A single mode variable belongs here. Conflicting
  legacy knobs should be rejected (or have explicitly documented precedence),
  not silently form a fourth hybrid mode.
- Public M1 inference construction reaches an env-aware context through
  `src/mamba_ssm/gpu/inference.rs:252-309,894-952,1732-1797`; M1 LM construction
  is in `src/module/gpu_lm.rs:98-179`. Neither surface takes a mode, although the
  LM exposes `ctx()` at `gpu_lm.rs:121-127` for post-construction mutation.
- Public M3 inference uses `GpuCtx::new(device)` and therefore ignores the env at
  `src/mamba3_siso/gpu/inference.rs:336-401` (call at 379). Its high-level
  backbone constructors are at `:1663-1724`; they have no mode argument and no
  context accessor. `Mamba3LmBuild` at `src/module/gpu_lm3.rs:88-170` contains
  dtype/batch but no mode and builds the backbone at `:199-200`.
- Public M1 trainer constructors are at
  `src/mamba_ssm/gpu/trainer.rs:282-337`; their internal contexts are env-aware
  at `:952` and `:2149`. Public M3 trainer constructors are at
  `src/mamba3_siso/gpu/trainer.rs:83-155`; their internal contexts are env-aware
  at `:762` and `:1870`. Both expose `ctx()` but neither constructor takes a
  mode. Trainer deterministic construction must resolve to `Triad`, not
  `Inference`.

The smallest coherent API seam is one canonical public `GemmMode` in
`gpu/context.rs`, plus an atomic `GpuCtx` configure/new-with-mode operation.
Thread it through the public inference/backbone, LM, and trainer construction
points above (a field on `Mamba3LmBuild`; additive `*_with_mode` or shared
options where existing APIs are positional). Keep deterministic kernel policy
and role family orthogonal: inference NN may choose `Inference`; training and
transpose/tied-head work require `Triad`. Selection of the release default is a
separate pending decision.

## What the proposed labels mean today

- Device handles are created in TF32 tensor-op math mode at
  `src/mamba_ssm/gpu/device.rs:230-255`.
- `src/mamba_ssm/gpu/dtype.rs:48-88` maps normal BF16/F16 calls to
  `CUBLAS_COMPUTE_32F_PEDANTIC`, fast BF16/F16 calls to
  `CUBLAS_COMPUTE_32F`, and both F32 choices to `CUBLAS_COMPUTE_32F`.
- `src/mamba_ssm/gpu/blas.rs:25-37` applies `fast_gemm` only to typed compute
  selection. Thus `CublasFast` requires both `fast_gemm=true` and TF32 handle
  math enabled; `CublasPedantic` requires `fast_gemm=false` and TF32 disabled.
  Merely toggling `fast_gemm` mislabels F32 behavior, and merely toggling TF32
  mislabels half/BF16 behavior.
- For F32, current vendor calls are SGEMM or GemmEx with `COMPUTE_32F`, not a
  literal `CUBLAS_COMPUTE_32F_PEDANTIC`. If the public word “Pedantic” is meant
  literally rather than “default-math F32 plus pedantic half accumulation,” the
  current implementation does not satisfy it and needs a defined conversion.
  `tests/cublas_compute_probe.rs:1-34,78-102` already records the relevant
  distinctions, including that pedantic compute makes TF32 handle mode inert.
- `tests/classifier_gemm_tier_bench.rs:54-58` records the present one-way TF32
  limitation.

## Deterministic no-cuBLAS gaps

Context-aware F32 NN and backward routes are fail-closed/custom in
`src/mamba_ssm/gpu/blas.rs:39-138,196-315`; typed backward and forward routes
are fail-closed/custom at `:317-453,3361-3507`. In the base implementation,
`Inference` forward uses its NN kernel, while its dX/dW paths deliberately fall
back to `Triad` (`:217-224,278-285`). That is correct evidence that the family is
a role/capability choice, not a global deterministic synonym.

The following reachable entry points bypass the context policy and still call
cuBLAS unconditionally. All must be routed through context-aware dispatch (or
made unreachable under `Deterministic`) before the label can promise no cuBLAS:

- F32 pointer NN: `src/mamba_ssm/gpu/blas.rs:140-194` (cuBLAS at 174), used by
  the M1 untied F32 LM head at `src/module/gpu_lm.rs:542-563`.
- Tied F32 LM head: `src/mamba_ssm/gpu/blas.rs:2299-2350` (cuBLAS at 2334),
  used at `src/module/gpu_lm.rs:567-575` and
  `src/module/gpu_lm3.rs:497-508`.
- Tied half/BF16 LM head: `src/mamba_ssm/gpu/blas.rs:3024-3077` (cuBLAS at
  3054), used at `src/module/gpu_lm.rs:631-642` and
  `src/module/gpu_lm3.rs:538-549`; it always uses the dtype's normal pedantic
  compute and ignores `fast_gemm`.
- Generic typed no-bias path: `src/mamba_ssm/gpu/blas.rs:3126-3163` (cuBLAS at
  3140), used by M3 mixed projection and logits paths, including
  `src/mamba3_siso/gpu/inference.rs:1213,1473` and
  `src/module/gpu_lm3.rs:524-532`.
- M3 F32 projection helper: `src/mamba3_siso/gpu/inference.rs:27-63` (cuBLAS at
  44), used at `:559,681,880`.

The M1 half untied head is already context-aware
(`src/module/gpu_lm.rs:615-628`). The tied half path has an extra shape/type
seam: the existing custom typed NT route produces typed output, whereas logits
expect F32. Reusing that route to write half/BF16 logits and then upcasting adds
an output-rounding step that the existing half-input-to-F32 logits path does not
have. That changes the numeric output contract and must not be presented as an
equivalent mixed-precision replacement. A true custom NT kernel with F32 output,
or exact input upcast followed by custom F32 NT, preserves the output
representation; the latter also requires explicit workspace, captured-graph,
and performance treatment. Whichever contract is chosen, cuBLAS cannot remain
as a hidden fallback under `Deterministic`.

## Captured graph identity

- `src/mamba_ssm/gpu/kernel_identity.rs:2793-2921` includes batch invariance,
  tensor-core enablement, fast compute, TF32 state, F32/half policies, and family
  in `GemmPolicy`/`GemmRouteIdentity`. Its declared deterministic backend sets
  contain custom `Inference` and `Triad` backends, not cuBLAS
  (`:2846-2894`). Exact equality is conservative: irrelevant flag changes can
  invalidate a graph, but cannot silently reuse a less strict route.
- M1 capture validates physical route/launch identity through
  `kernel_identity.rs:4741-4779`, with capture/replay integration in
  `src/mamba_ssm/gpu/inference.rs:360-409,465-500,1526-1559`.
- M3 inference only snapshots and compares the logical `GemmRoute` at
  `src/mamba3_siso/gpu/inference.rs:414-438,925-943,1595-1625`. Because several
  M3 operations currently bypass that route and call cuBLAS directly, the
  identity check can be non-causal. Once deterministic M3 dispatch is wired,
  extend M3 to the physical manifest/plan validation used by M1.
- Training replay checks already exist in
  `src/mamba_ssm/gpu/training_graph.rs:519-523,764`,
  `src/mamba3_siso/gpu/training_graph.rs:385-392,688-695`, and the split trainer
  paths tested below. The new mode must be represented in, or fully and uniquely
  derive, the exact route identity; mutating modes after capture must reject
  replay.

## Existing coverage and bounded additions

Existing relevant tests:

- Environment/parser combinations: `src/mamba_ssm/gpu/context.rs:1947-2105`.
- F32/BF16/F16 deterministic training and cuBLAS comparisons, plus batch
  invariance: `tests/gemm_bi_determinism.rs:117-205,209+`.
- Typed custom parity and `Inference`/base-`Fixed` backward fallback to Triad:
  `tests/gemm_bi_typed_parity.rs:377-470,584+`.
- Family invariance boundaries and bridge behavior:
  `tests/gemm_bi_invariance_matrix.rs:273-512` and
  `tests/gemm_bi_fixed_bridge.rs:60-180` (file/name may follow the rename).
- Inference route-drift rejection: `tests/inference_graph_route.rs:96-227`.
  It checks logical policy changes but does not prove M3's physical calls obey
  that policy.
- Training graph route drift: `tests/f32_training_graph_parity.rs:320-337`,
  `tests/m3_training_graph_safety.rs:147-180`, and
  `tests/trainer_split.rs:436-473` plus the analogous M3 split tests.
- Physical custom route traces: `tests/kernel_identity_cuda.rs:489-614`.
- Vendor compute/math semantics: `tests/cublas_compute_probe.rs:1-34,78-102`.

Minimal additions for the release API:

1. Unit-test each mode's complete mapping (including handle math), the eventual
   default, the single env spelling, and conflicts with legacy knobs.
2. Test mode propagation through every public M1/M3 inference, LM, and trainer
   constructor/builder; assert deterministic role family is `Inference` for
   inference NN and `Triad` for training/backward-capable contexts.
3. Probe `CublasFast` and `CublasPedantic` for F32, BF16, and F16 so both compute
   type and handle math match the label.
4. Add end-to-end deterministic LM-head/projection tests for every direct-call
   surface listed above, asserting no vendor call, including tied/untied and
   F32/half cases.
5. Capture under each mode and assert replay rejection after every mode
   transition and role-family change; for M3, validate the physical launch
   manifest rather than only the logical snapshot.

## Primary-source check for the API/docs phase

Checked NVIDIA's [cuBLAS13.2 documentation](https://docs.nvidia.com/cuda/archive/13.2.0/cublas/index.html)
on2026-09-10. `CUBLAS_COMPUTE_32F_PEDANTIC` specifies single-precision
arithmetic throughout computation and disables certain algorithmic shortcuts.
The supported GemmEx table explicitly permits F16/BF16 inputs with F32 output
under both32F and32F_PEDANTIC. Therefore the release can name its vendor
precision contract explicitly instead of inferring it from half output type.

The same documentation's reproducibility section states conditional bitwise
repeatability for a fixed toolkit, GPU architecture and SM count, with
concurrency/workspace and emulation caveats. Do not describe all cuBLAS Fast
or TF32 operation as intrinsically nondeterministic, or imply Pedantic alone
establishes cross-toolkit/cross-architecture invariance. The custom kernel
contract and batch-invariance scope must be described separately.
