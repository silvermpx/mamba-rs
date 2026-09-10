# Positive deterministic-mode model decode graph check

Date: 2026-09-10. Test: `inference_graph_route` /
`decode_graphs_reject_complete_route_drift`, CUDA 13.2.

The existing model test now covers successful graph replay as well as rejection
of a changed GEMM route. Its four cases are Mamba and Mamba-3, each with exact
F32 and native BF16 mixed inference, using the existing tiny deterministic corpus.

## Frozen source and scope

GREEN test SHA-256:
`4e92673fd2643726c109982725d565c108377414f01e4976f2f8bee1d9da75bd`.
Assertions-only RED SHA-256:
`9436e95aeb1932fcee8303943c3cf0729ede1df76d5282df0a23df4a08f3cbe3`.
The RED source is retained as `assertions-only-red.rs`.

Each context explicitly selects batch-invariant mode, the Triad family,
exact-F32 policy, tiled-parity half policy, and fast GEMM off. Tensor cores
are disabled for F32 and enabled for BF16. The complete configured route is
asserted and pinned before capture. Each engine must report a captured graph,
successfully replay with that unchanged route, and replace all 32 poisoned host
outputs with finite values. Changing batch-invariant mode then must produce the
existing route-drift error. Engines are dropped while their captured state and
scratch buffers are still alive.

There is no production or public API change. This check does not claim whole-model
eager/graph bit equality, model throughput, or a particular Tensor Core GEMM
selection for single-token decode. The independent kernel gates carry their own
bitwise guarantees.

## Actual runs

| Board | Image | Result | Test duration |
|---|---|---|---:|
| RTX 6000 Ada | assertions-only RED | Expected failure: `M1 f32 batch-invariant route` | 38.21s |
| RTX 6000 Ada | final GREEN | PASS: all four cases | 27.76s |
| RTX 5090 | final GREEN | PASS: all four cases | 92.32s |

Each directory contains the full source and executable manifests, GPU/driver
identity, idle preflight, build log, test listing, raw test output and exit status.
These durations are correctness-test runtimes, not performance comparisons.
The four route-drift warnings in each GREEN run are expected negative checks.
The RTX 5090 BF16 single-token case reports its ordinary portable tensor-core
route for an unmeasured specialized shape; the test succeeds on that route.

Both GREEN images contain dispatch `e86bc0d2` and binding test `e4922a1a`.
The subsequent cohort-review fixes (`c4ef0dbe` / `e7959705`) change only test
assertions/imports, not any production route or CUDA source. Their separate
host/live checks are in `../sm120-current-cohorts-20260910/`.

Independent scoped source review: `source-review.md`, specification and quality
approved. The raw RED failure was observed before the GREEN hardware runs.
