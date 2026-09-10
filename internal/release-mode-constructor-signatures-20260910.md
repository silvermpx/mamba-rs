# Concrete public GEMM-mode constructor map

Read-only source preparation against `7d890e6c` on 2026-09-10. This is a
signature/forwarding handoff for the existing release item, not an independent
implementation plan. No source, tests, Cargo metadata, index, or HEAD was
changed while preparing it.

## Fixed behavior to preserve

- `GemmMode` is already the canonical public enum, re-exported from
  `mamba_ssm::gpu` and `mamba_ssm::gpu::context`. Its variants and default are
  `Deterministic` (default), `CublasFast`, and `CublasPedantic`
  (`src/mamba_ssm/gpu/gemm_mode.rs:3-31`).
- A model/inference context stores `BiGemmFamily::Inference`; a trainer or a
  generic `GpuCtx` stores `BiGemmFamily::Triad`. This role choice is orthogonal
  to `GemmMode` and should remain stored even while a cuBLAS mode makes it
  dormant.
- Existing no-mode model/trainer constructors remain environment-aware.
  Missing GEMM environment selects `Deterministic`; the role supplies the
  default family. M3 model construction intentionally becomes environment-aware
  for consistency with M1.
- Every new explicit-mode constructor bypasses the mode, precision,
  tensor-core, and family environment selectors. It must construct the one real
  context with the requested mode and role family, not construct from env and
  mutate afterward.
- Generic public `GpuCtx::new*` behavior remains `Deterministic` + `Triad` and
  nonambient (`context.rs:607-647,680-710`). Generic `new_from_env*` continues
  to resolve with a `Triad` default (`context.rs:649-678`).
- All paths continue to end in the single owned body
  `GpuCtx::new_with_state_cap_and_config(&GpuDevice, usize,
  ResolvedGemmEnv) -> Result<GpuCtx, String>` (`context.rs:712-825`). There is
  no second environment parser, process-environment mutation, or throwaway
  context.

## Narrow context seam

`ResolvedGemmEnv` is already the complete internal construction value
(`context.rs:70-77`), and `resolve_gemm_env(values, default_family)` already
accepts the role default (`context.rs:124-173`). Keep both private. Add only
these crate-private forwarding seams alongside the existing constructors:

```rust
pub(crate) fn new_from_env_with_state_cap_and_family(
    device: &GpuDevice,
    state_cap: usize,
    default_family: BiGemmFamily,
) -> Result<Self, String>;

pub(crate) fn new_with_state_cap_mode_and_family(
    device: &GpuDevice,
    state_cap: usize,
    mode: GemmMode,
    family: BiGemmFamily,
) -> Result<Self, String>;
```

The first performs exactly
`resolve_gemm_env(GemmEnvValues::read(), default_family)` and then calls
`new_with_state_cap_and_config`. The second creates exactly the current
explicit-mode defaults (`tensor_cores: true`, `ExactScalarFmaV1`,
`TiledParityV1`) with the supplied `mode` and `family`, then calls the same
body. Refactor the existing public context constructors to use these with
`BiGemmFamily::Triad`; do not change their signatures.

This seam also retains the important environment distinction implemented at
`context.rs:219-243` and already unit-tested at `context.rs:2394-2417`:

- absent `MAMBA_RS_BI_GEMM_FAMILY` -> the supplied role default;
- present but empty/whitespace family -> `BiGemmFamily::Triad`;
- explicit mode construction -> this variable is not read at all and the
  supplied role family wins, even if the variable is empty, invalid, or names
  the other family.

An empty `MAMBA_RS_GEMM_MODE` is not absent; it remains a parse error. Existing
canonical/legacy conflicts and the rejection of deterministic-only precision,
family, and tensor-core controls under a resolved vendor mode remain unchanged
(`context.rs:83-173,2458-2563`).

## Exact additive public signatures and forwarding

All return types below are `Result<Self, String>`. Existing signatures remain
source-compatible and use the environment lane. A private `Option<GemmMode>`
dispatcher is acceptable as a narrow forwarding detail (`None` = env,
`Some(mode)` = explicit); it must choose context construction before the common
upload/allocation body, not parse env and overwrite fields.

### Mamba-1 inference and backbone

Current construction sites are `GpuMambaInference::new` at
`src/mamba_ssm/gpu/inference.rs:296-309`,
`GpuMambaInferenceMixed::new` at `:941-954`, and the public backbone entry
points at `:1738-1798`. Add:

```rust
// GpuMambaInference
pub fn new_with_mode(
    device: &GpuDevice,
    cpu_weights: &MambaWeights,
    cfg: MambaConfig,
    input_dim: usize,
    batch: usize,
    mode: GemmMode,
) -> Result<Self, String>;

// GpuMambaInferenceMixed
pub fn new_with_mode(
    device: &GpuDevice,
    cpu_weights: &MambaWeights,
    cfg: MambaConfig,
    input_dim: usize,
    batch: usize,
    bulk_dtype: WeightDtype,
    mode: GemmMode,
) -> Result<Self, String>;

// GpuMambaBackbone (f32 shortcut)
pub fn new_with_mode(
    gpu_ordinal: usize,
    cpu_weights: &MambaWeights,
    cfg: MambaConfig,
    input_dim: usize,
    batch: usize,
    mode: GemmMode,
) -> Result<Self, String>;

// GpuMambaBackbone (full storage choice)
pub fn new_with_dtype_and_mode(
    gpu_ordinal: usize,
    cpu_weights: &MambaWeights,
    cfg: MambaConfig,
    input_dim: usize,
    batch: usize,
    dtype: WeightDtype,
    mode: GemmMode,
) -> Result<Self, String>;
```

Forwarding rule:

1. `GpuMambaInference::new` validates `cfg`, computes the existing
   `state_capacity(cfg.d_state)`, and constructs via the env/family seam with
   `Inference`; `new_with_mode` uses the explicit mode/family seam with the same
   state capacity and `Inference`. Both then enter one shared weight-upload and
   allocation body.
2. The mixed explicit constructor calls the f32 engine's explicit constructor,
   just as current mixed construction reuses the f32 engine at `:950-954`.
   This is the one real context; the retained f32 engine also supplies f32
   weights/bias views and is not a throwaway context.
3. Backbone `new_with_mode` forwards F32 to
   `new_with_dtype_and_mode`. A single private dtype dispatcher serves both
   existing env-aware `new_with_dtype` and the explicit overload and calls the
   matching f32/mixed engine constructor. No signature exceeds seven
   positional arguments.

The existing public `ctx()` accessors are sufficient:
`GpuMambaInference::ctx` (`:878-881`), mixed `ctx` (`:1615-1617`), and
backbone `ctx` (`:1878-1884`). Callers inspect the selected mode with the
already-public `GpuCtx::gemm_mode() -> GemmMode` (`context.rs:1125-1131`) and
the family with `bi_gemm_family()` (`:1189-1192`). Do not add duplicate
top-level mode fields.

### Mamba-3 inference and backbone

Current engine construction is
`src/mamba3_siso/gpu/inference.rs:327-363`; mixed construction is `:976-1011`;
public backbone construction is `:1636-1691`. Add the same signatures, with
the M3 types:

```rust
// Mamba3GpuInferenceEngine
pub fn new_with_mode(
    device: &GpuDevice,
    cpu_weights: &Mamba3Weights,
    cfg: Mamba3Config,
    input_dim: usize,
    batch: usize,
    mode: GemmMode,
) -> Result<Self, String>;

// Mamba3GpuInferenceMixed
pub fn new_with_mode(
    device: &GpuDevice,
    cpu_weights: &Mamba3Weights,
    cfg: Mamba3Config,
    input_dim: usize,
    batch: usize,
    bulk_dtype: WeightDtype,
    mode: GemmMode,
) -> Result<Self, String>;

// GpuMamba3Backbone (f32 shortcut)
pub fn new_with_mode(
    gpu_ordinal: usize,
    cpu_weights: &Mamba3Weights,
    cfg: Mamba3Config,
    input_dim: usize,
    batch: usize,
    mode: GemmMode,
) -> Result<Self, String>;

// GpuMamba3Backbone (full storage choice)
pub fn new_with_dtype_and_mode(
    gpu_ordinal: usize,
    cpu_weights: &Mamba3Weights,
    cfg: Mamba3Config,
    input_dim: usize,
    batch: usize,
    dtype: WeightDtype,
    mode: GemmMode,
) -> Result<Self, String>;
```

Forward identically to M1 with the `Inference` role. The existing M3 state-cap
logic must remain distinct and unchanged: `GpuCtx` currently uses its default
capacity 64, while `Mamba3Kernels::compile_with_state_cap` receives
`state_capacity(cfg.d_state)` (`inference.rs:340-347`). Use `64` when calling
the new context family seam and keep the M3 compile call and its error ordering
in the one shared engine body. Do not silently replace either capacity with the
other.

Accessor changes needed for an honest public high-level surface:

```rust
// Mamba3GpuInferenceMixed (new convenience/accessibility parity with M1)
pub fn ctx(&self) -> &GpuCtx;

// GpuMamba3Backbone: widen current pub(crate), keep body unchanged
pub fn ctx(&self) -> &GpuCtx;
```

The f32 engine already has `pub fn ctx(&self) -> &GpuCtx` at `:417-421`. The
backbone accessor is currently only `pub(crate)` at `:1874-1880`. As with M1,
`ctx().gemm_mode()` is the mode accessor; no extra stored mode is needed.

### Mamba-1 LM

Current chain is `src/module/gpu_lm.rs:136-179`. Add:

```rust
pub fn from_hf_with_mode(
    dir: &Path,
    gpu_ordinal: usize,
    mode: GemmMode,
) -> Result<Self, String>;

pub fn from_hf_with_dtype_and_mode(
    dir: &Path,
    gpu_ordinal: usize,
    dtype: WeightDtype,
    mode: GemmMode,
) -> Result<Self, String>;

pub fn from_hf_with_dtype_batch_and_mode(
    dir: &Path,
    gpu_ordinal: usize,
    dtype: WeightDtype,
    batch: usize,
    mode: GemmMode,
) -> Result<Self, String>;
```

Keep one private full body taking `Option<GemmMode>`. Existing `from_hf`,
`from_hf_with_dtype`, and `from_hf_with_dtype_batch` feed the env lane; the new
overloads feed the explicit lane and call
`GpuMambaBackbone::new_with_dtype_and_mode`. The LM already exposes
`pub fn ctx(&self) -> &GpuCtx` at `gpu_lm.rs:121-127`; update its legacy
batch-invariant wording to describe `ctx().gemm_mode()`.

### Mamba-3 LM

`Mamba3LmBuild<'a>` already carries eight fields
(`src/module/gpu_lm3.rs:88-109`). Adding a required mode field would break
every external struct literal, and appending mode to the seven-argument
`from_weights_with_dtype` (`:133-153`) would violate the positional-argument
limit. Keep the struct unchanged and add:

```rust
pub fn from_weights_with_mode(
    cpu_weights: &Mamba3Weights,
    cfg: Mamba3Config,
    embed: Vec<f32>,
    lm_head: Option<Vec<f32>>,
    vocab_size: usize,
    gpu_ordinal: usize,
    mode: GemmMode,
) -> Result<Self, String>;

pub fn build_with_mode(
    args: Mamba3LmBuild<'_>,
    mode: GemmMode,
) -> Result<Self, String>;
```

`from_weights_with_mode` is the F32 shortcut. Explicit dtype/batch callers use
`build_with_mode`; do not add an eight-argument dtype constructor or a lint
suppression. `build` and `build_with_mode` feed one private body, whose only
branch is whether line `199-200` calls env-aware
`GpuMamba3Backbone::new_with_dtype` or explicit
`new_with_dtype_and_mode`. Keep the input-projection behavior at `:183-197`
ahead of that call.

Add the missing public inspection seam:

```rust
pub fn ctx(&self) -> &GpuCtx {
    self.backbone.ctx()
}
```

This mirrors M1 and requires the backbone visibility widening above. The
current module-private tests reach `lm.backbone.ctx()` only because they are in
the same module (`gpu_lm3.rs:662-675`); external users cannot.

### Mamba-1 and Mamba-3 trainers

Both `new_with_dtype` methods already have seven positional arguments
(`src/mamba_ssm/gpu/trainer.rs:286-310` and
`src/mamba3_siso/gpu/trainer.rs:87-110`). Do not add an eighth. Add only the
compact explicit overload beside each existing `new_full`:

```rust
// MambaTrainer
pub fn new_full_with_mode(
    gpu_ordinal: usize,
    cpu_weights: &MambaWeights,
    cfg: MambaConfig,
    session: TrainSessionCfg,
    dtype: WeightDtype,
    mode: GemmMode,
) -> Result<Self, String>;

// Mamba3Trainer
pub fn new_full_with_mode(
    gpu_ordinal: usize,
    cpu_weights: &Mamba3Weights,
    cfg: Mamba3Config,
    session: TrainSessionCfg,
    dtype: WeightDtype,
    mode: GemmMode,
) -> Result<Self, String>;
```

Existing `new_full` and the explicit overload call one private dispatcher with
`Option<GemmMode>`, preserving all validation before precision dispatch.
Private F32/mixed `new_full` bodies receive that choice and create their real
context through the env/family or explicit mode/family seam with `Triad`.

Preservation points:

- M1 mixed and F32 both compute `state_capacity(cfg.d_state)` before context
  creation (`mamba_ssm/gpu/trainer.rs:948-952,2145-2149`) and pass that exact
  capacity to `GpuCtx`; explicit mode must do the same.
- M3 keeps generic `GpuCtx` capacity 64 and separately compiles M3 kernels with
  `state_capacity(cfg.d_state)` in both mixed and F32 branches
  (`mamba3_siso/gpu/trainer.rs:759-768,1867-1876`).
- M3's shared public dispatcher must continue `cfg.validate`, launch-capacity
  validation, and especially the F32 explicit-input-projection rejection before
  it calls the private F32 body (`mamba3_siso/gpu/trainer.rs:119-153`). Do not
  move this check into only the env or only the explicit lane.
- Both trainers already expose public `ctx() -> &GpuCtx`
  (`mamba_ssm/gpu/trainer.rs:364-371`,
  `mamba3_siso/gpu/trainer.rs:181-187`), so mode/family are inspectable without
  new duplicated accessors.

## Architecture-rung environment honesty

The explicit context path bypasses `GemmEnvValues::read`, including constructor
validation of `MAMBA_RS_ARCH_RUNG` (`context.rs:44-66,124-129,200-217`). That
does **not** make inference architecture-rung selection nonambient. The
Inference kernel has a separate process-wide `ARCH_RUNG_OK: OnceLock<bool>` and
reads `MAMBA_RS_ARCH_RUNG` on the first qualifying launch
(`gemm_bi_inference.rs:4033-4073`). Therefore Rustdoc for every explicit model
constructor should say:

> GEMM mode, custom precision/tensor-core controls, and family selectors are
> ignored. `MAMBA_RS_ARCH_RUNG` is a separate first-use process policy for the
> Inference architecture rung and is not captured by this constructor.

Do not claim that explicit construction ignores all `MAMBA_RS_*` variables,
and do not move or duplicate that `OnceLock` behavior in this API item.

## Bounded real-fixture test plan

No downloads and no mutation of the parent Rust test process environment.

1. Extend the parser unit tests already colocated in `context.rs` rather than
   introducing a public config object. `bi_gemm_family_environment_accepts_only_semantic_family_names`
   (`:2394-2417`) already locks absent-vs-empty behavior; add a focused test for
   the new explicit config helper showing both model `Inference` and trainer
   `Triad` with each of the three modes and the unchanged exact/tiled defaults.
2. Add a CUDA integration test binary dedicated to constructor propagation.
   Use `std::process::Command::new(std::env::current_exe())` with one exact child
   test and a case-selector variable. On every child command, call
   `.env_remove` for all eight existing GEMM variables, then add only that
   case's variables. This gives each case a fresh process and a fresh
   architecture-rung `OnceLock`; never call `set_var`/`remove_var` in the test
   process.
3. In the missing-env child cases, construct actual M1/M3 F32 and Bf16
   backbones and F32/Bf16 trainer branches, then assert
   `ctx().gemm_mode() == Deterministic`; backbones must report `Inference` and
   trainers `Triad`. For the M1 cases reuse the small owned pattern from
   `trainer_smoke::m1_trainer_bf16_smoke` (`tests/trainer_smoke.rs:26-54`) and
   `m1_trainer_f32_smoke` (`:357-385`). For M3 reuse `small_m3_cfg` and the
   owned `Mamba3Weights::init` fixture from
   `tests/gpu_mamba3_lm_test.rs:11-46`, plus the trainer shapes from
   `m3_trainer_bf16_smoke` (`trainer_smoke.rs:402-430`).
4. In an explicitly empty-family child, leave mode absent and set
   `MAMBA_RS_BI_GEMM_FAMILY=""`; construct at least both F32 model backbones and
   assert `Deterministic + Triad`. The parser unit test proves the full spelling
   table; these two real constructions prove that the model role default is
   passed only for an actually absent value.
5. In conflicting-env children, set a genuinely conflicting lane such as
   `MAMBA_RS_GEMM_MODE=deterministic` plus `MAMBA_RS_FAST_GEMM=true` and set
   `MAMBA_RS_BI_GEMM_FAMILY=triad`. Construct every new explicit public surface
   once with `CublasFast` and once with `CublasPedantic`, split across F32 and
   Bf16 to cover both precision branches. Assert the requested mode and the
   role family before any step. A representative legacy env-aware constructor
   in the same isolated lane must still return the existing conflict error.
6. Cover both LM builders with owned local data. For M1, reuse
   `write_synthetic_checkpoint` from `tests/hf_integration.rs:53-176` in a
   tempfile; it explicitly documents zero network access (`:1-4`). For M3,
   extend `small_m3_cfg` / `build_synthetic_lm` from
   `tests/gpu_mamba3_lm_test.rs:11-47` to accept the new build mode and assert
   `lm.ctx()` before generation. Exercise both F32 and one half dtype across the
   explicit fast/pedantic pair.
7. Preserve error contracts with constructor-only cases: M1 and M3 invalid
   config/state-capacity errors must be identical between env and explicit
   entry points. For M3 F32, reuse the identity-matrix fixture at
   `tests/trainer_smoke.rs:460-497` for success and separately keep empty
   `input_proj_w` to assert the exact `f32 M3 trainer requires an explicit input
   projection` failure from `mamba3_siso/gpu/trainer.rs:131-141`. The mode must
   not bypass or reorder this validation. Keep the existing mixed identity
   projection convention (`gpu_mamba3_lm_test.rs:29-35`).
8. Assert M1 `ctx().state_cap()` equals the rounded
   `state_capacity(cfg.d_state)` in both inference and trainer explicit lanes.
   For M3, test an out-of-range `d_state` through both default and explicit
   construction and require the same M3 kernel state-cap error; do not assert
   that the generic M3 context's fixed 64 equals the independently compiled M3
   kernel capacity.

This matrix tests construction and inspection before the first step, so it does
not depend on the serial-905 model graph-guard implementation. Existing graph,
numeric, and no-vendor-route suites remain the place for end-to-end dispatch
claims.

## Confirmed trainer `None`-plan replay risk

Both trainer-local wrappers currently do this:

```rust
match plan {
    Some(plan) => plan.with_validated_launch(ctx, label, launch),
    None => launch(),
}
```

Locations are `src/mamba_ssm/gpu/trainer.rs:110-120` and
`src/mamba3_siso/gpu/trainer.rs:49-59`; their only callers are the f16 graph
replays at M1 `:1681-1689` and M3 `:1159-1167`.

The risk is confirmed through the actual call chain:

- f16 capture stores the `Option<CapturedGemmGraphPlan>` returned by
  `capture_into_graph_with_gemm_plan` (M1 `:1810-1889`, M3 `:1265-1347`).
- `GemmRouteRecordingGuard::finish_against_manifest` legitimately returns
  `Ok(None)` when `resolved_gemm_launches` finds no custom physical routes
  (`context.rs:437-461`), which is the vendor-graph case.
- `CapturedGemmGraphPlan::with_validated_launch` starts with
  `ctx.ensure_gemm_usable()` (`kernel_identity.rs:4821-4837`), but the local
  `None` branch omits it.
- `set_gemm_mode` marks the context unusable after an unverified cuBLAS math
  rollback without publishing the requested canonical mode
  (`context.rs:1103-1121`). The surrounding trainer replay comparison uses
  `ctx.gemm_route()`, whose logical policy derives from the still-old canonical
  mode and does not itself call `ensure_gemm_usable` (`context.rs:1778-1817`).
  Thus pointer checks and route equality can pass before the vendor graph is
  launched on an unhealthy handle.

Serial task 905 is planned but not present at this source revision. Do not name
or depend on a guessed helper. After 905 lands, replace both local trainer
wrappers with its actual reviewed shared optional-plan launch guard if it owns
this exact contract. If it does not expose such a guard, the smallest complete
fallback is to add `ctx.ensure_gemm_usable()?;` unconditionally before the
existing `match` in both local wrappers. That preserves the stronger `Some`
plan validation and closes only the missing `None` health check. Add/extend the
shared guard's unit test so an unusable context rejects the `None` case without
invoking the closure; no trainer graph fixture is needed to prove that narrow
branch.

## Remaining concrete risks / integration checks

- Rebase line numbers and the trainer guard choice after serial 905; no model
  graph implementation was assumed here.
- Ensure every default model lane passes `Inference` to the env resolver.
  Merely changing explicit constructors would leave M1 missing-env on Triad and
  M3 nonambient.
- Ensure every explicit model lane stores `Inference` even for cuBLAS modes;
  tests should inspect the dormant family instead of assuming it is irrelevant.
- Avoid routing explicit overloads through existing env-aware constructors and
  then calling `set_gemm_mode`/`set_bi_gemm_family`: that can fail on env
  conflicts, temporarily configure the wrong cuBLAS math, and violate the
  one-context construction contract.
- Keep M3 LM's mixed non-identity input-projection rejection and M3 trainer
  F32's explicit projection requirement exactly where they are relative to GPU
  allocation.
- Rustdoc must separate storage `WeightDtype`, GEMM `GemmMode`, deterministic
  family, and the process-wide architecture-rung admission policy.
