# Inference retained-toolkit audit

Date: 2026-09-10. Bounded read-only audit of current selectors and already-retained evidence; no build, GPU run, source change, or new tuning search.

Bare review-note filenames below refer to the working-session notes under
`.superpowers/sdd/handoff-codex-gemm-bi-triad-2026-09-05/`. They are context,
not substitutes for the committed source and raw qualification packets.

## Verdict

**No concrete already-qualified faster-than-prior-AUTO Inference winner is omitted from the supported CUDA 12.8/13.0/13.2 routes in the inspected evidence.** The apparent CUDA-13.2-only selectors are either backed by explicitly 13.2-only performance admission or sit above a lower-toolkit route that was independently retained. I found one worthwhile **unresolved qualification gap**, not a release wiring defect: SM120 Fixed TF32 hot-B has favorable old lower-toolkit screen data, but not the current-source 101-confirmation and post-AUTO chain required to call it qualified.

## Reachability versus qualification

- Ada homogeneous BF16/F16 is explicitly reachable on all three toolkits in `fixed_select_sm89_half_auto_tile`: its common guard admits only CUDA 12.8/13.0/13.2, and its lower-toolkit table selects the independently retained Pipeline/Swizzle winners (`src/mamba_ssm/gpu/gemm_bi_fixed.rs:498-568`). The S3 override is correctly 13.2-only: `ada-half-s3-paired-task6b-final-evidence-review.md:5,15-23` records lower-toolkit p95 failures and says 12.8/13.0 retain their incumbents.
- Ada deterministic-TF32 RNA-wide is explicitly admitted for 12.8/13.0/13.2 (`gemm_bi_fixed.rs:3656-3690`), with the three-toolkit paired-101 receipts named in the source. The older Fixed TF32 C M64 selector remains 13.2-only, but Task7's accepted lower-toolkit evidence says those C candidates lost the screen against actual RNA AUTO and did not advance to 101 (`ada-f32-tf32-toolkit-task7-final-evidence-review-brief.md:47-51`). This is not an omitted lower-toolkit winner.
- Ada exact-F32 CopyPlan admits the exact finite toolkit set 12.8/13.0/13.2 and all A/B/D/E bias rows (`gemm_bi_fixed.rs:5740-5773`; table test `fixed_sm89_exact_n64_auto_routes_all_twenty_four_qualified_toolkit_literals`). `ada-exact-toolkit-auto-task8-fix1-report.md:69-78` records matching builds and lower-toolkit actual-AUTO correctness/graph coverage.
- The retained Ada TN-d128 exact routes are not stranded in a 13.2-only cohort: `f32-d128-integration-report.md:7-11,43-60,81-82,99-104` records separate 12.8/13.0/13.2 module, performance, AUTO-admission, and post-admission runs.
- The retained Ada joint-TF32 map also distinguishes qualification from mere loadability: `tf32-final-integration-report.md:20-23` preserves the lower-toolkit prior routes and admits all six new joint winners only on 13.2; its post-admission test is explicitly three-toolkit (`:87-91`).
- SM120 generic half selection is not toolkit-gated (`gemm_bi_fixed.rs:2941` onward), while the literal exact-half overlay is intentionally CUDA 13.2-only (`:2893-2938`). The lower-toolkit full-census logs physically execute the generic AUTO routes, e.g. `internal/perf/sm120-fixed-full-census-20260906/cuda-12.8/fixed-full-census-cuda12.8-bf16_f32-postauto-v1.log`; I found no accepted lower-toolkit 101 receipt authorizing the 13.2 exact overlay.
- SM120 Fixed TF32 B/D specialization is explicitly limited to CC12.0/170SM/NVRTC13.2 (`gemm_bi_fixed.rs:2483-2515`), and the selector test deliberately requires the generic SM120 schedule on 12.8/13.0 (`sm120_tf32_selector_promotes_only_the_qualified_cuda_132_b_and_d_cells`). The newly integrated lower-toolkit 23-key TF32 cohorts in `sm120-current-cohorts-report.md:30-64` qualify **Triad dispatch** identities/routes; they do not by themselves qualify these distinct Fixed selector arms.

## Unresolved gap, not an accepted omission

`internal/perf/sm120-fixed-full-census-20260906/cuda-{12.8,13.0}/fixed-full-census-cuda{12.8,13.0}-tf32-postauto-v1.log` contains 21-window hot-B forced `Tf32Sm120M128S2` ratios below 1 for both bias rows, eager/graph, and both orders. That makes lower-toolkit B a credible bounded follow-up. It does **not** establish an already-qualified retained winner: the records bind the older Fixed source digest `8a9a5186...`, are screen-21 rather than accepted 101 confirmation, and lack a current-source selector/post-AUTO receipt. Hot-D PairStore is not even a clean screen winner: several lower-toolkit p95 ratios are at or above 1. Therefore neither B nor D can be widened from 13.2 on the evidence inspected.

## Hardware-proof boundary

“Reachable in code” above means the selector admits the exact toolkit identity and required holder. “Physically tested” is claimed only where a cited report/log records execution on the named Ada or SM120 device. Static unit tests and the fresh three-toolkit Triad cohort integration do not substitute for a Fixed-route performance admission. The bounded audit did not recompute raw quantiles, inspect every historical handoff line, or rerun any hardware test; the SM120 hot-B lower-toolkit qualification chain remains the only identified open item.
