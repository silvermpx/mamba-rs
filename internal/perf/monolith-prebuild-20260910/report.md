# Old-main comparison: compilation preflight

On 2026-09-10, root exported unchanged main commit
`d8f2efbeaecc04a53cec9890097f2913cd3f09e6` to an isolated Ada source directory
using `git archive`. No main source, branch or worktree was modified.

`cargo test --locked --release --features cuda --test m1_gpu_benchmark --no-run`
passed with CUDA13.2 in57.07s. The executable lists the expected
`m1_gpu_benchmark` test. Source-after SHA checks and the runner pass.
This is a build preflight only: no benchmark body or GPU kernel was run, no
old/new speed measurement was collected, and no output parity is claimed.

Source directory: `/root/mamba-monolith-main-prebuild.0pF6ga`.
Private Cargo target: `/root/target-monolith-main-prebuild-cuda132-20260910`.
The source manifest, compiler/toolkit versions, build log and binary SHA are
kept beside this report.

Before timing, use one identical adapted inference harness at both source
endpoints with explicit custom exact-F32/family/tensor-core settings. The old
constructor ignores route environment variables, so environment flags alone
do not select the old custom backend. Compare reset output checks first, then
paired eager/graph measurements with idle GPU preflight. Do not run the old
full benchmark unchanged and label its default cuBLAS path the monolithic
custom baseline. The new source endpoint must be the completed API assembly.
