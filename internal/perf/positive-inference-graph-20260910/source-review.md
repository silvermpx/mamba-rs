# Positive inference graph test review

Date: 2026-09-10

## Scope and verdict

Reviewed only the delta to `tests/inference_graph_route.rs`, frozen SHA-256
`4e92673fd2643726c109982725d565c108377414f01e4976f2f8bee1d9da75bd`,
against `positive-inference-graph-brief.md`.

**Specification: approved. Code quality: approved.** No Critical or Important
finding. No production code or public API is added or changed.

## Findings

None.

## Confirmed behavior

- One helper explicitly configures fast GEMM off, batch-invariant mode on,
  Triad family, exact-F32 policy, tiled-parity half policy, and the requested
  tensor-core tier (`inference_graph_route.rs:15-22`). The companion assertion
  reads the existing public route identity and pins every configured field
  (`:24-50`), so the test does not depend on ambient environment handling.
- All four existing cases are covered in place: Mamba F32 (`:107-129`), Mamba
  BF16 native mixed (`:131-163`), Mamba-3 F32 (`:170-193`), and Mamba-3 BF16
  native mixed (`:195-227`). F32 explicitly uses `tensor_cores=false`; BF16
  explicitly uses `tensor_cores=true`.
- Each case warms the existing step API, captures with the original live
  state/scratch, asserts the existing `has_graph()` accessor, and confirms the
  route still equals the pre-capture identity before replay.
- Each positive replay poisons the nonempty 32-element host output, calls the
  ordinary `step` or `step_mixed_native` with no route change, requires success,
  and then requires every output element to be finite and to have replaced the
  exact poison word (`:52-67` and the four case bodies). This is a real positive
  replay check, not merely graph construction.
- The negative route-drift check remains after the positive replay in all four
  cases. It changes `batch_invariant` to false as required and retains each
  existing exact error-prefix assertion.
- State and scratch remain alive through capture, positive replay, and rejected
  drift replay. Each engine is explicitly dropped before its referenced state
  and scratch can be destroyed (`:129`, `:163`, `:193`, `:227`), so graph
  lifetime ordering is explicit.
- The scope is represented honestly: it proves successful deterministic-mode
  model decode graph replay and complete route-drift rejection. It does not add
  eager equivalence, bitwise reconstruction, performance, or a claim that
  single-token decode selected a particular Tensor Core GEMM.

## Evidence status

Root's assertion-only RED against the old setup failed at the expected explicit
M1 F32 batch-invariant assertion after 38.21s. Final GREEN on Ada and RTX 5090
was still root-owned/running when this static review was completed; no hardware
result is claimed here.
