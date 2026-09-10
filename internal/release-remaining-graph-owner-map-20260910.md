# Remaining prefill/training graph owner map for 0.7.0

Read-only audit of `b7397d4fd53e` plus the in-progress Task905 working-tree
changes, 2026-09-10. No source, tests, Cargo metadata, documentation, index, or
HEAD was changed. The only write is this handoff. Line numbers below are the
inspected working tree and may move when Task905 is committed.

## Release decision summary

Two changes are required with Task906's model-role default, rather than being a
new graph subsystem:

1. The genuine inference prefill recorders must prepare the Inference
   architecture rung before starting their eager route recorder. Otherwise a
   first deterministic Inference launch on an SM90/SM100 context reaches the
   cold self-check while recording, and Task904 intentionally rejects it.
2. Every remaining graph capture/replay should use Task905's reviewed
   deterministic admission/health seams. The current five
   `require_f32_triad_graph_plan` calls stop protecting model prefill as soon as
   Task906 gives those contexts the `Inference` family, and every local
   optional-plan replay wrapper bypasses `GpuCtx::ensure_gemm_usable()` in its
   legitimate vendor `None` branch.

Trainer construction itself must continue to supply the `Triad` role default.
That is not presently broken: M1 mixed/F32 construct with
`new_from_env_with_state_cap` at `trainer.rs:952,2149`, and M3 mixed/F32 use
`new_from_env` at `mamba3_siso/gpu/trainer.rs:762,1870`; all four currently
resolve absent family as Triad. Task906's private dispatcher must route both
precision branches through its new role-aware seam with `BiGemmFamily::Triad`.
An explicitly configured `MAMBA_RS_BI_GEMM_FAMILY=inference` must remain
honored. It exposes the same cold-rung issue in the trainer recorders, but that
is an expert override, not the default-role regression.

## Exact inference-prefill owners

### M1 pooled prefill: `src/mamba_ssm/gpu/prefill.rs`

- `PrefillPooledGraph::capture` (`:368-441`) performs its own eager pass inside
  `ctx.record_eager_gemm_manifest` at `:388-401`, immediately feeds that local
  manifest to `capture_into_graph_with_gemm_plan` at `:405-420`, then calls the
  old admission helper at `:421-426` with fixed `true`.
- This graph always calls raw input projection at `:304-311` before the layer
  chain. Its real nonzero-work predicate is
  `dims.batch != 0 && dims.seq_len != 0 && dims.mamba_input_dim != 0 &&
  dims.d_model != 0`. This is available as `scratch.dims` in capture. It is not
  a dtype flag: `bulk_dtype` describes the layer weights, while the raw input
  projection is explicitly required to be F32 at `:297-303`.
- Minimal change points: import and call
  `prepare_inference_arch_rung(ctx)` after shape/dtype checks and scratch
  presizing but before `record_eager_gemm_manifest`; replace the old capture
  admission with `require_deterministic_gemm_graph_plan(ctx, has_gemm_work,
  ...)`; store `has_gemm_work` on the holder (replay no longer has weights or
  dimensions); replace local `with_validated_launch` (`:35-45`) at replay
  `:486-495` with `with_validated_gemm_graph_launch(ctx,
  self.has_gemm_work, ...)`.
- There is no stale saved permit here: the successful eager result is a local
  value in the same capture call. Do not add a graph manager or a second
  manifest mechanism.

### M3 full and pooled prefill: `src/mamba3_siso/gpu/prefill.rs`

- The single genuine eager recorder is `Mamba3Prefill::run_full` at `:247-258`;
  it records `run_full_body` at `:253-255` and saves the manifest at `:256`.
  It currently does not prepare the architecture rung.
- `run_full_body` exposes the exact logical work choices: input projection is
  skipped only when `run.identity_proj` is true (`:329-396`); each of
  `dims.n_layers` executes at least in-projection `(bt, d_model, in_proj_dim)`
  at `:403,448-473` and out-projection `(bt, d_inner, d_model)` at
  `:1110-1151`. With validated model dimensions, use
  `dims.bt() != 0 && (!run.identity_proj || dims.n_layers != 0)`, matching the
  reviewed M3 decode predicate. If this helper is made independently callable
  before validation, spell out the corresponding nonzero projection
  dimensions rather than weakening it to a dtype check.
- Full capture reads (does not consume) `prefill.eager_gemm_manifest` at
  `:1302-1304`, captures at `:1308-1320`, and calls the old helper at
  `:1321-1326` with `weights_dtype == F32`. Pooled capture repeats the pattern
  at `:1506-1530`. Thus deterministic half work is unprotected already, and
  deterministic F32 work becomes unprotected under Task906's default
  `Inference` family.
- Minimal required change points: in `run_full`, compute the predicate, call
  `prepare_inference_arch_rung(run.ctx)` before the recorder when it is true,
  then record. Both captures use the predicate with
  `require_deterministic_gemm_graph_plan`; both holders retain it for replay.
  Replace local `with_validated_launch` (`:41-51`) in full replay
  (`:1422-1431`) and pooled replay (`:1622-1631`) with the shared launch seam.
- One-shot permit discipline is a focused improvement, not necessary to fix
  the cold-rung/default-family regression: clear `eager_gemm_manifest` before
  each attempted eager run, and use `.take()` in either capture. Today a failed
  later run leaves an older permit, and one successful run can feed either
  capture repeatedly. `capture_into_graph_with_gemm_plan` still compares the
  actual capture against the manifest, so a changed route/inventory is rejected;
  the remaining issue is stale permission semantics, not an unvalidated
  mismatched physical plan.

## Exact training owners

Validated trainer entry points reject zero batch/sequence through
`validate_kernel_arg_capacity`; both model configs require positive model and
layer dimensions. The useful predicate should nevertheless name the actual
projection choice:

```text
M1/M3 F32:   batch != 0 && seq_len != 0
M1 mixed:    batch != 0 && seq_len != 0 &&
             (weights.compute.input_proj_w.len_elems() != 0 || cfg.n_layers != 0)
M3 mixed:    dims.bt() != 0 &&
             (weights.compute.input_proj_w.len_elems() != 0 || dims.n_layers != 0)
```

The F32 predicates are justified by their unconditional explicit input
projection. M3 rejects an empty F32 input projection before dispatch at
`mamba3_siso/gpu/trainer.rs:130-142`; M1 F32 calls it unconditionally in the
forward. Mixed M1/M3 explicitly treat an empty projection as identity, so the
layer count must remain part of the predicate. The same mixed predicate serves
BF16 and F16; storage dtype alone is not evidence of work.

### M1 training graph and trainer

- `src/mamba_ssm/gpu/training_graph.rs` BF16 capture is
  `GpuMambaTrainingStepGraph::capture` (`:246-401`), with capture at
  `:323-370` and no missing-plan admission. Its replay is `:413-533` through
  the local wrapper at `:72-82`.
- Its F32 twin captures at `:639-748`, calls old fixed-`true` admission at
  `:715-720`, and replays through the same local wrapper at `:750-850`.
- `src/mamba_ssm/gpu/trainer.rs` has exactly three eager manifest preparers:
  F16 forward/backward `eager_f16_forward_backward` at `:1542-1580`, ordinary
  mixed `step_eager` at `:1969-2026`, and F32 `step_eager` at `:2552-2592`.
  All save a manifest only after success, but none clears the old value before
  an attempted recording or prepares a selected Inference rung.
- BF16/F32 capture reads the saved manifest without consuming it at
  `:1179-1204` and `:2268-2293`. F16 does the same at `:1790-1792`, captures at
  `:1810-1887`, installs an optional plan at `:1888-1889`, and has no
  deterministic missing-plan admission.
- The F16-only trainer wrapper (`:110-120`) is called at `:1681-1689`. Its
  `None => launch()` branch skips health.

Minimal change points: call `prepare_inference_arch_rung(&self.ctx)` before
each of the three recorders when the corresponding predicate is true (a no-op
for the normal Triad trainer); use deterministic admission after all three
capture forms; route BF16/F32 training-graph replays and the F16 trainer replay
through `with_validated_gemm_graph_launch`, carrying/storing the predicate as
needed. Merely adding health to the F16 wrapper, as Task906 currently specifies,
closes that confirmed vendor replay hole but leaves the identical two
`training_graph.rs` replay holes.

### M3 training graph and trainer

- `src/mamba3_siso/gpu/training_graph.rs` BF16 capture is
  `GpuMamba3TrainingStepGraph::capture` (`:143-269`), with capture at
  `:205-237`, no missing-plan admission, and replay through local
  `with_validated_launch` (`:32-42`) at `:392-401`.
- F32 capture is `GpuMamba3F32TrainingStepGraph::capture` (`:499-594`), with
  old fixed-`true` admission at `:562-567`; replay uses the same local wrapper
  at `:695-704`.
- `src/mamba3_siso/gpu/trainer.rs` has exactly three eager preparers: F16
  `eager_f16_forward_backward` at `:1021-1075`, ordinary mixed `step_eager` at
  `:1430-1495`, and F32 `step_eager` at `:2115-2168`. Their saved manifests
  have the same non-clearing behavior as M1 and lack Inference-rung preparation.
- BF16/F32 capture reads without consuming at `:929-935` and `:1978-1983`.
  F16 reads at `:1245-1247`, captures at `:1265-1345`, stores its optional plan
  at `:1346-1347`, and has no missing-plan admission.
- The F16-only trainer wrapper (`:49-59`) is called at `:1159-1167`; its
  `None` branch skips health.

Use the same minimal changes and predicates as M1. Calling the preparer in all
three eager methods preserves an explicit trainer `Inference` family override;
it remains a no-op under the required default `Triad` role.

## What is already protected

- Task905's `with_validated_gemm_graph_launch(ctx, has_gemm_work, plan,
  label, closure)` first calls `ctx.ensure_gemm_usable()`, then requires a plan
  for deterministic nonzero work, then delegates `Some(plan)` to its full
  validation; vendor and true zero-work `None` remain legal.
- Every existing `Some(plan)` branch already reaches
  `CapturedGemmGraphPlan::with_validated_launch`, which itself checks context
  health and route/launch identity. The uncovered health path is specifically
  `None`.
- All listed holders already check capture context/stream, logical GEMM route,
  and their applicable weight/module/pointer/scratch identities before launch.
  Preserve those checks; the shared helper is the final submission seam, not a
  replacement for them.
- `capture_into_graph_with_gemm_plan` validates the observed capture against
  the eager manifest. Saved-manifest one-shot consumption is therefore
  lifecycle hardening; it is not evidence that current mismatched routes launch.
- Task905 decode owners already clear/take their permits, prepare the rung, use
  actual work predicates, and route replay through the shared helper. They are
  outside this remaining-owner change.

## Focused existing fixtures / exact tests

No new manager or broad CUDA matrix is needed. The smallest owned coverage is:

- Host source contract: update
  `exact_graph_inventory_wires_decode_and_existing_prefill_training_holders`
  in `tests/gemm_bi_tf32_contract.rs`. Its existing `direct_holders` table is
  exactly the five old-helper sites; `conditional_holders` is exactly the two
  BF16 training and two raw-F16 trainer captures. Require the shared
  deterministic capture guard and shared replay launch helper for all nine,
  and assert `has_gemm_work` reaches both seams. Add ordered preparer-before-
  recorder assertions for M1 pooled, M3 `run_full`, and the three recorder
  methods in each trainer. If one-shot semantics are accepted, this same test
  already has the Task905 `.take()`/clear-before-work patterns to mirror.
- Shared guard behavior is already covered by the Task905 unit
  `model_graph_guard_zero_vendor_and_poison_controls`; it proves healthy
  deterministic zero-work `None`, healthy Fast/Pedantic `None`, and poisoned
  `None` without callback execution. `deterministic_inference_half_work_rejects_empty_graph_plan`
  covers the half/Inference admission hole. If local wrappers remain instead
  of being deleted, add actual-wrapper poison tests; otherwise source wiring
  plus this shared unit is the narrower proof.
- The fail-closed rule itself already has the host unit
  `cold_architecture_probe_is_rejected_before_recording_or_capture` in
  `gemm_bi_inference.rs`.
- Existing focused GPU behavior fixtures, sufficient after the wiring/source
  RED-GREEN check: M1 `pooled_graph_replays_bitwise`; M3 full
  `gpu_prefill_graph_replay_is_bitwise`; M3 half pooled
  `typed_pooled_graph_replay_and_container_guard`; M1 BF16
  `training_graph_bf16_one_step_matches_eager`; M3 BF16
  `m3_training_graph_bf16_one_step_matches_eager`; F32
  `m1_f32_training_graph_matches_eager` and
  `m3_f32_training_graph_matches_eager`; F16 trainer paths
  `graph_determinism_f16` and `m3_f16_training_graph_replays_deterministically`.
  The existing first-recording cache tests are
  `m1_f16_first_triad_step_prepares_graph_cache` and
  `m3_f16_first_triad_step_prepares_cache_before_graph_scratch`.
- Task906's constructor regressions `m1_default_constructor_uses_inference_family`
  and `m3_default_constructor_uses_inference_family` prove the actual default
  role before any graph work. They should not be overloaded to prove graph
  admission.

## Scope/risk classification

- **Must accompany the Task906 model default:** prepare the two genuine
  prefill recorders and migrate their three captures/replays to all-dtype,
  all-family deterministic guarding. Otherwise the advertised default model
  role either fails its first cold SM90/SM100 prefill recording or accepts an
  empty deterministic Inference graph inventory.
- **Confirmed release health gap already assigned to Task906:** both trainer
  F16 optional-plan wrappers must reject an unusable context in the vendor
  `None` branch. The same defect is present in both prefill wrappers and both
  training-graph wrappers; migrating all six definitions to the reviewed
  shared helper avoids knowingly retaining equivalent holes.
- **Pre-existing completeness gap:** BF16 training graphs and raw F16 trainer
  captures have no deterministic missing-plan guard; F32 training uses the old
  family-specific fixed-true guard. This is not caused by trainer role
  defaulting, but is a small direct migration once the shared seam exists.
- **Optional lifecycle hardening:** clear saved manifests before a new eager
  attempt and consume them with `.take()` in M3 prefill and all trainer capture
  paths. Current capture-time manifest comparison already rejects changed
  inventories; do not make this optional cleanup a prerequisite for the
  bounded constructor API if release scope must stay smaller.
