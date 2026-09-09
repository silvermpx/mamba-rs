# Ada exact-F32 NT Fixed CopyPlan siblings — Batch A integration

Date: 2026-09-09. This batch integrates only the retained exact-F32 NT
d768-in `(2048,768,3072)` and canonical Prism `(4621,384,1928)` cells. It
does not add a CUDA kernel and does not start the TN export batch.

## Production routes

| Cell | plan tag | physical route | transpose scratch |
| --- | ---: | --- | ---: |
| d768-in | 39 | `gemm_bi_transpose_f32_32x16_d768_v1` (`TriadScalar`) → `gemm_bi_nn_fixed_sm89_f32_n64_copyplan_v1` (`Fixed`) | 2,359,296 f32 |
| Prism | 40 | same ordered symbols/modules | 740,352 f32 |

The selector is exact-shape and exact-stride only. It requires CC8.9/142 SM,
no bias, `alpha=+1` and `beta=+0` by bits, non-null 16-byte-aligned output/A/B
pointers, a loaded Fixed function and one of three sealed CUDA12.8/13.0/13.2
composed identities. Each composed identity pins both the `TriadScalar`
transpose compiler/artifact and the Fixed CopyPlan compiler/artifact in the
same NVRTC/library domain. Existing NN and NT d768-out Fixed admissions retain
their previous Fixed-only gate unchanged.

Both eager execution and prepared graph construction consume the same
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

## TDD and host verification

RED was observed before production admission: the new selector test resolved
d768-in to `NtFinal { slim: false }` instead of the requested retained plan.
After adding the enum variants, compilation remained RED until all exhaustive
launch surfaces were wired.

The final current-tree checks were run from an isolated temporary copy because
the repository-local Cargo shim executes a committed fleet snapshot and native
Cargo rejects this worktree's nested workspace location. No CUDA test or GPU
work was executed.

```text
CUDARC_CUDA_VERSION=13000 cargo test --no-default-features --features cuda \
  --lib nt_fixed_copyplan_sibling -- --nocapture
3 passed

CUDARC_CUDA_VERSION=13000 cargo test --no-default-features --features cuda \
  --lib fixed_copyplan_selector -- --nocapture
3 passed

cargo test --no-default-features \
  --test gemm_bi_scalar_nt_copyplan_siblings_discovery -- --nocapture
3 passed

CUDARC_CUDA_VERSION=13000 cargo test --no-default-features --features cuda \
  --test gemm_bi_scalar_nt_copyplan_siblings_discovery --no-run
compiled
```

The ignored live entry is
`cuda_suite::ada_f32_nt_copyplan_siblings_integrated_auto_qualification`. On
an exclusive RTX6000Ada it fail-closes on the three sealed toolkit domains,
resource caps, exact eager output, input/scratch/output guards, public AUTO
node order/config/module identity, and eager/prepared-graph identity. It was
intentionally not run in this CPU-only integration task.
