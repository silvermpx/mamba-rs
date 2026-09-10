### Spec Compliance

- ✅ Spec compliant. The required crate-private unsafe F32 NN/NT pointer seams are present and mode-routed through one central NN boundary (`src/mamba_ssm/gpu/blas.rs:50`, `src/mamba_ssm/gpu/gemm_bi_triad/launch.rs:5244`, `src/mamba_ssm/gpu/gemm_bi_triad/launch.rs:6201`); the public buffer/raw and all-F32 typed paths converge on it (`src/mamba_ssm/gpu/blas.rs:156`, `src/mamba_ssm/gpu/blas.rs:170`, `src/mamba_ssm/gpu/blas.rs:3469`). Deterministic tied F32 dispatch uses the cached NT seam with `(B,Vpad,D)` (`src/mamba_ssm/gpu/blas.rs:2295`), and M3 F32/typed projections and tied/untied heads now consume `GpuCtx` (`src/mamba3_siso/gpu/inference.rs:521`, `src/mamba3_siso/gpu/inference.rs:1177`, `src/mamba3_siso/gpu/inference.rs:1875`, `src/module/gpu_lm3.rs:490`, `src/module/gpu_lm3.rs:508`, `src/module/gpu_lm3.rs:532`, `src/module/gpu_lm3.rs:550`).
- ✅ The supplied regression change exercises real owned buffers, route/stride identity, both deterministic families, asymmetric CPU references, repeat bits, an interior input, guarded output subspans, bias, zero reduction, and pointer rejection (`tests/gemm_context_routing.rs:73`, `tests/gemm_context_routing.rs:111`, `tests/gemm_context_routing.rs:152`, `tests/gemm_context_routing.rs:240`, `tests/gemm_context_routing.rs:327`, `tests/gemm_context_routing.rs:369`, `tests/gemm_context_routing.rs:425`).
- ⚠️ Cannot verify from diff: the tests-first Ada RED, seven-test Ada GREEN, final CUDA+HF all-target compile, 54 prepared-control tests, Rustdoc run, source-manifest equality, and runtime CUDA/header-byte equality are execution claims outside the change view. The controller should adjudicate them from the root-owned receipts named in `internal/perf/gemm-context-routing-20260910/report.md`.

### Strengths

- The owned/raw scalar adapter is narrow and preserves both contracts: owned `GpuBuffer` arguments still reach cudarc slice submission, while raw arguments submit the same `CUdeviceptr` bytes (`src/mamba_ssm/gpu/gemm_bi_triad/launch.rs:1835`, `src/mamba_ssm/gpu/gemm_bi_triad/launch.rs:1852`, `src/mamba_ssm/gpu/gemm_bi_triad/launch.rs:1862`, `src/mamba_ssm/gpu/gemm_bi_triad/launch.rs:1872`, `src/mamba_ssm/gpu/gemm_bi_triad/launch.rs:1882`). The shared forward and NT bodies retain one dispatcher each (`src/mamba_ssm/gpu/gemm_bi_triad/launch.rs:8349`, `src/mamba_ssm/gpu/gemm_bi_triad/launch.rs:9707`).
- The central F32 seam validates context/mode at the routing boundary, keeps vendor modes explicit, and sends deterministic NN to the selected family without introducing an alternate arithmetic path (`src/mamba_ssm/gpu/blas.rs:50`). The tied path sends both deterministic family selections to cached Triad NT and never passes NT to the Inference NN ladder (`src/mamba_ssm/gpu/blas.rs:2295`).
- The raw-boundary Rustdoc states row-major dimensions, F32 spans/alignment, aliasing, managed-owner and stream/replay lifetimes, zero-reduction nullability, errors, and `unsafe` obligations (`src/mamba_ssm/gpu/blas.rs:36`, `src/mamba_ssm/gpu/gemm_bi_triad/launch.rs:5226`, `src/mamba_ssm/gpu/gemm_bi_triad/launch.rs:6183`).
- Named-risk outside-diff check — legacy cudarc synchronization: inspected `src/mamba_ssm/gpu/buffers.rs:177`, `src/mamba_ssm/gpu/context.rs:748`, and pinned cudarc 0.19.9 `driver/safe/launch.rs:100` and `driver/safe/launch.rs:117`. The adapter's owned arms preserve cudarc's read/write wait-and-record path; the raw arms are used by `GpuCtx`, whose context intentionally disables that tracking. The derived NT tail pointer remains downstream of an owned `dy` submission on the same stream (`src/mamba_ssm/gpu/gemm_bi_triad/launch.rs:10322`, `src/mamba_ssm/gpu/gemm_bi_triad/launch.rs:10362`).
- Named-risk outside-diff check — warm-cache validation: inspected the raw seam-to-cache chain and cache refresh (`src/mamba_ssm/gpu/gemm_bi_triad/launch.rs:5101`, `src/mamba_ssm/gpu/gemm_bi_triad/launch.rs:4944`). Cached Triad routing does not call the standalone full pointer validator at `src/mamba_ssm/gpu/gemm_bi_triad/launch.rs:3311`; it retains the existing epoch-gated cached validation, so warm calls do not gain a duplicate allocation scan.
- Named-risk outside-diff check — hidden M3 vendor routing: an exact census of `src/mamba3_siso` and `src/module/gpu_lm3.rs` found no remaining `sgemm_no_bias`, `gpu_gemm_typed_raw_no_bias`, `gpu_gemm_bi_tied_lm_head_blas`, or `gpu_gemm_ex_tied_lm_head_blas` call.

### Issues

#### Critical (Must Fix)

- None.

#### Important (Should Fix)

- None.

#### Minor (Nice to Have)

- None.

### Assessment

**Task quality:** Approved

**Reasoning:** The implementation matches the requested routing topology and preserves the two highest-risk cross-cutting contracts: legacy owned-slice synchronization and single warm-cache allocation validation. The diff adds focused behavioral coverage without duplicating dispatch logic, hiding a vendor GEMM, or changing the arithmetic implementation.
