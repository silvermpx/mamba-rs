# Kernel optimization continuation

For GEMM/CUDA optimization work, first read
`internal/agent-operational-rules.md`, especially its binding research-first
search method and latest user overrides. Resume measured state from
`internal/handoff-codex-gemm-bi-triad-2026-09-05.md` and remaining choices from
`internal/perf/ada-triad-discovery-shortlist-20260908.md`.

The user explicitly requires source/profile-guided search with primary
NVIDIA/CUTLASS/PTX research and parallel independent agent work. Reuse existing
winners; prepare a small concrete candidate shortlist; run focused exact-bit,
resource and paired eager/graph once7 checks against retained-best and cuBLAS
Fast. Stop valid unchanged losers. Do not repeatedly run full gates after each
prototype or present old validated results as new speedups.

Whole-Triad discovery precedes joint integration and one supported-toolkit
qualification batch. Keep one timed GPU executor, preserve bit contracts,
measured source snapshots, other-architecture winners and unrelated WIP. Work
in this existing branch/worktree and commit bounded completed changes without
AI/co-author trailers. No new branch, push or destructive cleanup is implied.
