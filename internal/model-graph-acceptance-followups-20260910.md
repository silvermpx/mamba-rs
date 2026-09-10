# Model graph acceptance after Inference inventory

Source inspection at6c282b3c, while Task904 owns terminal identity changes.
This is bounded preparation for the next implementation brief, not a claim
that model-wide graph/no-vendor acceptance is complete.

## Paths that must be covered

| Owner | Eager/replay entry points | Captures |
|---|---|---|
| M1 F32 | `step`, `step_gpu_only` | `capture_graph` |
| M1 mixed legacy F32 activations | `step`, `step_gpu_only` | `capture_graph` |
| M1 mixed native half activations | `step_mixed_native`, `step_gpu_only_mixed_native` | `capture_graph_mixed_native` |
| M3 F32 | `step`, `step_gpu_only` | `capture_graph` |
| M3 native mixed | `step_mixed_native`, `step_gpu_only_mixed_native` | `capture_graph_mixed_native` |

M1 native already has a separate eager manifest/plan. Its legacy mixed branch
has neither and directly invokes `graph.launch`; these two public legacy paths
must not be omitted when closing M3. Both M1 native and M3 native mixed require
identity input projection. F32 and M1 legacy mixed can exercise a nonidentity
input projection. Preserve that distinction in fixtures.

The M1 mixed object shares one `graph` field between legacy and native capture.
Its next guarded design must pair a captured plan with the path it describes;
do not leave a native plan attached to a subsequent legacy graph or vice versa.
Keep independently typed eager manifests or an explicit private path tag, not
two public graph APIs or a generic graph-state framework.

## Real projection sequence

Dimensions below use `(M,K,N)` and all operations are NN. For L layers:

- Optional nonidentity input projection: `(B,input_dim,d_model)` first.
- M1 each layer: `(B,d_model,2*d_inner)`, `(B,d_inner,xdbl_dim)`,
  `(B,dt_rank,d_inner)`, `(B,d_inner,d_model)`, in that order.
- M3 each layer: `(B,d_model,in_proj_out_dim)`, `(B,d_inner,d_model)`.
- M1/M3 native mixed omit the optional input projection.
- Untied head: separate eager NN `(B,D,Vpad)`.
- Tied head: separate eager NT `(B,Vpad,D)`, including Task903 input upcasts
  only in the conversion-inclusive observer, not the GEMM-only context manifest.

Do not assume physical count equals logical count for every scalar fallback.
Use fixed small single-launch Inference fixtures for exact ordered tables and
explicit grouped projections for composed fallback coverage. A nonempty trace
by itself is insufficient: dropping the middle projection must be rejected.

## Recorder and replay rules

- Call Task904 eager architecture preparation before the serving recorder;
  its two one-time self-check GEMMs must not contaminate the capture manifest.
- Clear the corresponding eager manifest before any new step attempt, including
  an upload/validation failure; assign it only after the real body succeeds.
- Record the existing private body once. Do not wrap a public step in another
  context recorder, which would be nested with its own internal recorder.
- Capture uses `capture_into_graph_with_gemm_plan`, a successful matching eager
  manifest, and complete comparison. A known nonzero deterministic GEMM body
  cannot accept `None` as its plan under any family or storage dtype.
- Before replay, check usability, current logical route, captured plan and
  existing state/scratch pointers. Preserve scratch freeze and graph lifetime
  anchors. Vendor graphs may have no custom plan, but still check usability
  and policy/handle route before launch.
- Keep H2D/D2H outside the recorded body and do not turn debug-only body entry
  points into nested recorders. Correct touched SGEMM-only/default comments.

## Tests-first and acceptance

The first real M3 regression can warm its private production body to prepare
caches, then call the actual capture method without any public eager step.
Current M3 capture permits it; the new manifest prerequisite must reject it.
A second focused guard regression calls the current family/F32-specific guard
with Deterministic/Inference and a known half GEMM workload; empty plan must
be rejected. Root observes these intended REDs before production edits.

For positive fixtures, run the private real body with `record_eager_gemm_trace`
to inspect ordered projections, then exercise the public step and capture on
the same owned buffers. Compare its eager manifest/captured routes against
the expected table and the trace. Reset recurrence between eager and replay
numerical comparisons. Use B1/B3, two layers to expose a missing middle GEMM,
F32/BF16/F16, and a distinct input width for the supported nonidentity cases.
Colocated tests can inspect private plans without adding public testing APIs.

Mutation tests must reject before a supplied launch callback: missing/reordered
middle route, wrong symbol/module/storage/argument binding, changed mode/family,
unusable context, cold preparation and stale manifest after a failed step.
Preserve valid state/pointer fixtures so unrelated assertions do not mask the
intended guard.

## Vendor boundary proof

The source census at6c282b3c has nine production SGEMM/GemmEx result calls in
`src/mamba_ssm/gpu/blas.rs` (lines136,225,288,353,422,2360,3673,3759,4100).
Place one test-only thread-local count/deny seam immediately before every
actual FFI, including no-context compatibility helpers. No production branch,
environment knob, global test state, or public testing feature is required.

Positive controls must actually execute both explicit vendor modes with
counting enabled and observe hits; denial controls must stop before FFI.
Under deny, actual deterministic M1/M3 step, capture and tied/untied head
fixtures must report zero vendor GEMM calls. Integration tests do not compile
the library with cfg(test), so put hook consumers in colocated library tests.
Retain a call-site census as supporting evidence, not as a substitute for the
positive and negative runtime controls.

The next task's exact function signatures, test names and field changes are
written after Task904's reviewed implementation interface is frozen. This
note prevents another broad source census and preserves the newly found M1
legacy-path requirement without starting a second source writer.
