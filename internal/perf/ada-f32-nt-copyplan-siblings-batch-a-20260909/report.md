# Ada exact-F32 NT Fixed CopyPlan siblings — Batch A pre-admission

Date: 2026-09-09. This batch wires the retained exact-F32 NT d768-in
`(2048,768,3072)` and canonical Prism `(4621,384,1928)` plans, but leaves
their production admission deliberately fail-closed. It does not add a CUDA
kernel and does not start the TN export batch.

## Production routes

| Cell | plan tag | physical route | transpose scratch |
| --- | ---: | --- | ---: |
| d768-in | 39 | `gemm_bi_transpose_f32_32x16_d768_v1` (`TriadScalar`) → `gemm_bi_nn_fixed_sm89_f32_n64_copyplan_v1` (`Fixed`) | 2,359,296 f32 |
| Prism | 40 | same ordered symbols/modules | 740,352 f32 |

The dormant selector is exact-shape and exact-stride only. It requires CC8.9/142 SM,
no bias, `alpha=+1` and `beta=+0` by bits, non-null 16-byte-aligned output/A/B
pointers, a loaded Fixed function and a sealed composed identity. The
production evidence cohort is currently empty, so both cells resolve to their
prior AUTO plans: `NtFinal { slim: false }` and `NtFinal { slim: true }`.

Three CUDA12.8/13.0/13.2 identity rows are retained only as qualification
candidates. Each candidate pins both the `TriadScalar`
transpose compiler/artifact and the Fixed CopyPlan compiler/artifact in the
same NVRTC/library domain. A later commit may move an individual row into the
production cohort only after that exact domain passes the ignored live gate.
Existing NN and NT d768-out Fixed admissions retain their previous Fixed-only
gate unchanged.

When admitted, eager execution and prepared graph construction consume the same
`ScalarDispatchPlan` and the same two-node `scalar_physical_nodes` description.
The eager branch now recognizes both sibling tags as Fixed CopyPlan routes;
the ordered modules, configs, argument layouts, scratch pointer and function
lifetime behavior therefore match prepared graph rehydration.

## Evidence provenance

The retained CUDA13.2 correctness/performance evidence remains
`internal/perf/ada-f32-nt-copyplan-siblings-20260908/report.md`: d768-in was
`.4317–.4331` of prior actual AUTO p50 with worst p95 `.4334`; Prism was
`.8700–.8717` with worst p95 `.8722`. These are retained-best improvements,
not cuBLAS Fast wins.

Fixed CUDA12.8/13.0/13.2 identities are the existing production cohorts from
`internal/perf/ada-scalar-nn-live-fixed-screen-20260908/evidence/`. The
unchanged `TriadScalar` identities are frozen in the CUDA12.8/13.0/13.2 Ada
artifact logs under `internal/perf/ada-half-swizzle-force-20260906/`; the
CUDA13.2 compiler/artifact pair is also recorded directly in the sibling
discovery log. No identity was synthesized from a different target.

## Required live qualification

The ignored
`cuda_suite::ada_f32_nt_copyplan_siblings_pre_admission_qualification` entry is
the only promotion gate. Run it independently on CUDA12.8, CUDA13.0 and
CUDA13.2 while the production cohort remains empty. It requires:

- live Driver ABI and resource caps for both physical functions before timing;
- target, tail, exceptional-payload, non-unit-alpha and K0 exactness;
- A/B immutability and two-sided 256-byte-aligned candidate output/scratch red
  zones for eager and captured-graph execution;
- the exact prior public AUTO symbol/config/module in eager/prepared evidence
  and the operation-level captured graph;
- paired whole-pipeline candidate/prior-public-AUTO eager and graph ABBA plus
  BAAB, strict p50 and p95 below `.99` at once3 and then once7.

The candidate timing arm launches both transpose and Fixed nodes. The AUTO arm
uses `gpu_gemm_bi_backward_dx_raw`; because admission is empty, it remains the
actual prior production fallback rather than aliasing the candidate.

No toolkit row is authorized by the current CPU-only commit. The live logs,
artifact/compiler identities and timing strata must be frozen before a
separate minimal admission commit.

## TDD and host verification

The review regression started RED because the composed production cohort was
non-empty. After separating qualification candidates from the empty production
cohort, host tests require both new cells to resolve exactly to their prior
fallback while existing NN and d768-out routes remain unchanged.

The final current-tree checks were run from an isolated temporary copy because
the repository-local Cargo shim executes a committed fleet snapshot and native
Cargo rejects this worktree's nested workspace location. No CUDA test or GPU
work was executed.

```text
cargo test --no-default-features --features 'cuda,cudarc/cuda-13000' \
  --lib nt_fixed_copyplan_sibling -- --nocapture
4 passed before the concurrent Batch B1 WIP entered the shared tree

cargo test --no-default-features \
  --test gemm_bi_scalar_nt_copyplan_siblings_discovery -- --nocapture
4 passed

cargo test --no-default-features --features 'cuda,cudarc/cuda-13000' \
  --test gemm_bi_scalar_nt_copyplan_siblings_discovery --no-run
deferred: the shared tree contains concurrent, uncommitted Batch B1 module
plumbing that currently fails before this test is compiled
```

No live CUDA test or GPU timing was executed in this fail-close repair.
