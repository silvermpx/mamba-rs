# Public model-mode constructor map

Source: `779ef943`, inspected 2026-09-10 while Task903 owns tied-head source.
Read-only preparation for the existing release API item. No constructor,
default family, environment behavior or public signature was changed.

## Current propagation

| Public surface | Construction path | Remaining seam |
|---|---|---|
| M1 F32 inference / `GpuMambaBackbone::new` | `GpuMambaInference::new` → `GpuCtx::new_from_env_with_state_cap` | No explicit mode argument; env-absent family remains Triad. |
| M1 mixed inference / `new_with_dtype` | `GpuMambaInferenceMixed::new` → F32 engine constructor | Must forward the same resolved choice, not construct once then mutate a different context. |
| M3 F32 inference / `GpuMamba3Backbone::new` | `Mamba3GpuInferenceEngine::new` → `GpuCtx::new` | Currently ignores GEMM env; no explicit mode argument. |
| M3 mixed inference / `new_with_dtype` | `Mamba3GpuInferenceMixed::new` → F32 engine constructor | Same forwarding requirement. |
| M1 LM | `from_hf*` → `from_hf_with_dtype_batch` → M1 backbone | Explicit mode must reach the backbone at construction; public `ctx()` already exists. |
| M3 LM | `from_weights*` → `build(Mamba3LmBuild)` → M3 backbone | Build struct contains dtype/batch but no mode; additive `build_with_mode(args, mode)` can avoid breaking all existing struct literals. |
| M1 trainer | `new_with_dtype` → `new_full` → private F32/mixed `new_full` | Both private paths independently construct env-aware contexts; explicit mode must reach both. |
| M3 trainer | `new_with_dtype` → `new_full` → private F32/mixed `new_full` | Same seam; preserve existing F32 explicit-input-projection validation. |

Public trainer `new_with_dtype` already has seven positional arguments.
Do not add an eighth argument plus a lint suppression. The existing
`new_full(..., TrainSessionCfg, dtype)` is the compact additive mode seam.
Backbone/mixed-engine dtype constructors have six arguments, so an additive
mode argument still fits the existing argument-count limit. No new general
configuration framework is needed to propagate one enum.

## Existing reusable context machinery

`resolve_gemm_env(values, default_family)` already takes a role default.
`bi_gemm_family_from_result` honors this default when the variable is absent;
an explicitly empty family value currently resolves to Triad. Keep that
compatibility distinction explicit instead of treating empty as absent.

`GpuCtx::new_with_state_cap_and_config` is the single owned construction body.
`ResolvedGemmEnv` contains mode, tensor-core permission, family and the F32/half
policies. Extend a narrow crate-private construction seam around these existing
types if needed; do not add a second parser, mutate process environment, or
construct a throwaway GPU context merely to configure the real one.

For the planned role defaults, inference NN selects Inference and training
selects Triad; explicit expert family selection remains a separate policy.
Generic public `GpuCtx::new*` defaults need not change from Triad. Defaulting
model inference to Inference depends on completion of its physical route
recording and guarded graph replay, not just the renamed module's availability.

An explicit-mode constructor must document whether it ignores all GEMM env
settings; the existing context explicit-mode constructors do. Do not parse env
then overwrite only mode, leaving a hidden mix of precision/family selectors.
Existing env-aware constructors should preserve strict conflict diagnostics.
Making the M3 model constructor env-aware is an intentional documented
consistency change, not an already-implemented behavior.

## Acceptance to carry into the implementation task

- Test actual M1/M3 F32 and mixed construction, both LM builders, and both
  trainer branches. Assert canonical mode and role family before first step.
- Use small owned synthetic weights and colocated LM fixtures; no downloads.
- Check explicit Fast/Pedantic propagation under a conflicting process-env
  lane using isolated test processes; do not mutate shared Rust test env.
- Preserve config/state-capacity/shape errors, especially M3 F32 trainer's
  explicit input-projection requirement and mixed decode's identity domain.
- Document defaults, env precedence, storage precision versus GEMM mode,
  graph transition restrictions and errors in IDE-visible Rustdoc. Keep
  Deterministic's custom kernels distinct from either explicit vendor mode.
- M3's crate-visible `ctx()` from Task902 is sufficient for internal forwarding;
  decide its public accessor together with the high-level mode API, not as an
  accidental visibility expansion in tied-head work.

The final implementation task still needs its concrete signatures and focused
tests; this map records source facts and avoids another constructor census.
