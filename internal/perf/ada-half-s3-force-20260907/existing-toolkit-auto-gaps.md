# Ada SM89 existing-toolkit AUTO gaps

Date: 2026-09-07  
Scope: bounded local audit of the existing Fixed exact-F32 N64 and TF32 C0/C1 SM89 families only. This is backlog evidence, not authority to widen selectors. No GPU, SSH, build, source/test edit, or SM120 analysis was performed. Source anchors are from a concurrent Task6A snapshot and may drift.

## Bottom line

Both CUDA 12.8 and 13.0 gaps are **qualification/selector gaps, not missing-loader gaps** on the tested RTX 6000 Ada stack:

- Exact F32's `F32Sm89N64CopyPlan` is an optional, independently admitted holder, but current 12.8/13.0 retained-route logs prove that the holder loads and its forced correctness suite passes. AUTO deliberately rejects every NVRTC version except 13.2 and falls back to `Legacy`.
- TF32's `Tf32M64S2` is one of the mandatory portable Fixed TF32 functions, has no optional holder/rejection path, and is force-selectable on CC8.9. AUTO deliberately chooses it for C0/C1 only on known NVRTC 13.2; 12.8/13.0 fall back to `Tf32M128S2`.

Force availability alone does not authorize promotion. Each toolkit still needs literal-cell performance qualification and post-selector actual-AUTO requalification.

## Inventory

| Family | Exact AUTO guard | Physical candidate | 12.8/13.0 incumbent fallback | Existing evidence | Minimum next qualification |
|---|---|---|---|---|---|
| SM89 exact F32 N64 | `loaded && library_known && nvrtc==(13,2) && CC8.9 && 142 SM && ExactScalarFmaV1`; homogeneous non-null 16-byte-aligned F32 C/A/B; absent or non-null 4-byte-aligned bias; exact A/B/D/E shapes, both bias states | `FixedTile::F32Sm89N64CopyPlan`; `gemm_bi_nn_fixed_sm89_f32_n64_copyplan_v1`; 128 threads, 32,768 B static / 0 dynamic shared. Current 12.8/13.0 live census: 135 registers, 0 local, 3 active CTAs, MaxShared carveout | `FixedTile::Legacy`; `gemm_bi_f32_f32_s2` on CC8.9 | Full 13.2 B/E production paired101 plus independent 13.2 A/D paired101. Current 12.8 and 13.0 forced correctness/resource suites pass; no rooted 12.8/13.0 paired performance promotion record found | Per toolkit, run forced CopyPlan vs actual Legacy for all eight A0/A1/B0/B1/D0/D1/E0/E1 rows, eager+graph and both timing parities/orders: 21, then 101 only on p50+p95 owner wins. Preserve exact bit/guard/immutability, ABI/resource and graph-identity gates. Only after success consider a toolkit-literal guard, then rerun actual AUTO and retained-route production-NVRTC qualification |
| SM89 TF32 C0/C1 | Before generic geometry: not SM120 TMA, `CC8.9 && 142 SM && nvrtc==(13,2) && library_known`, exact `(4621,1928,384)`, bias false or true. Output alignment is not part of this C promotion | `FixedTile::Tf32M64S2`; `gemm_bi_nn_tf32_v1_m64n64_bk32_s2`; grid 438 for C, 128 threads, 32,768 B dynamic shared | `FixedTile::Tf32M128S2`; `gemm_bi_nn_tf32_v1_m128n64_bk32_s2`; grid 222, 256 threads, 55,296 B dynamic shared | 13.2 forced-candidate paired101 and three actual-AUTO paired101 confirmations cover both bias states/eager+graph/both orders. 12.8/13.0 records prove the M64S2 function executes/captures on other Ada shapes, but no C0/C1-specific 12.8/13.0 candidate-vs-M128 paired record was found | Per toolkit, run literal C0/C1 forced M64S2 vs actual M128S2 with raw-bit and physical-graph identity, eager+graph, both orders/parities, paired21 then paired101 on p50+p95 owner wins. If admitted, widen only that toolkit literal and rerun actual AUTO plus production-NVRTC retained routes |

## Exact-F32 anchors

Selector and fallback:

- `src/mamba_ssm/gpu/gemm_bi_fixed.rs:5225-5261`, `fixed_sm89_exact_n64_auto_eligible`, contains the complete 13.2-only predicate and eight admitted rows.
- `src/mamba_ssm/gpu/gemm_bi_fixed.rs:5376-5442` explicitly rejects NVRTC 12.8, 13.0, 13.1, 13.3 and unknown/absent holders in selector tests.
- `src/mamba_ssm/gpu/gemm_bi_fixed.rs:3652-3681` launches CopyPlan only when that predicate succeeds, otherwise calls `fixed_pick_f32_exact`; `fixed_pick_f32_exact` at 2305-2313 returns `Legacy` for CC8.9.
- `src/mamba_ssm/gpu/gemm_bi_fixed.rs:190-192` exposes the same candidate through `fixed_forward_with_tile`; the public force registry/spec includes it at `tests/gemm_bi_fixed_performance.rs:10911` and 11169-11174.

Physical availability, not a missing loader:

- `src/mamba_ssm/gpu/kernels.rs:462-464,690-691` owns and initializes the optional function plus rejection reason.
- `src/mamba_ssm/gpu/gemm_bi_triad/modules.rs:2652-2658` composes it for `sm_89` without an NVRTC-version predicate. Loader admission at 3641-3710 checks Fixed/sm_89/CC8.9, exact Driver ABI and resources; it does not require NVRTC 13.2.
- `kernels/gemm_bi_fixed/sm89_f32_n64_copyplan.cu:116-128` is the unique physical body/ABI.
- `internal/perf/ada-half-s3-force-20260907/cuda128-continuation1/retained-exact.log:7,575-596` records NVRTC 12.8 live admission and `5 passed; 0 failed; 1 filtered` for the forced-only retained suite. `cuda130-attempt1/retained-exact.log:7,575-596` records the same for NVRTC 13.0.
- The earlier `cuda128-attempt1/retained-exact.log:5-16` is not a loader/candidate failure: it mistakenly invoked `fixed_sm89_exact_n64_auto_prefix_view_graph_bits`, whose explicit assertion expected `(13,2)` but observed `(12,8)`. The corrected forced-only run above is the admissible availability result.

13.2 evidence:

- `internal/perf/exact-n64-copyplan/production-paired-101-v1.jsonl` is the rooted B/E production-NVRTC record; SHA-256 `91493d7994999bfa48473803a1e29a7cc516a98b5264478c264457c3ba34db86`. Its manifest pins CC8.9/142 SM/NVRTC13.2, 128 warmups and 20 operations/window; completion closes 48 pair records. B0/B1/E0/E1 cell summaries report owner wins; all four CopyPlan ÷ Legacy and CopyPlan ÷ actual-AUTO p95 ratios per cell are below 0.835.
- `internal/perf/fixed-exact-promotions-ada-20260906-318b3fbd09c8/ad-copyplan-confirm101.log` adds A0/A1/D0/D1 under NVRTC13.2: 32 records, 101 windows, eager+graph, both orders, exact raw/repeat/graph identity; CopyPlan ÷ AUTO Legacy p50 ratios are `0.830110-0.846557`, and the same-direction p95 ratios are `0.832349-0.854944`. Summary: sibling `analysis.md`; log SHA-256 `9224161eec3a578f472c07810647952c56604a980d699876b187887a929c575a`.

These records justify the existing 13.2 guard, not 12.8 or 13.0. The current forced correctness passes close loader/functionality risk for those two toolkits but do not close performance promotion.

## TF32 C0/C1 anchors

Selector and fallback:

- `src/mamba_ssm/gpu/gemm_bi_fixed.rs:2224-2256`, `fixed_pick_tf32`, places the C0/C1 literal before generic geometry and requires CC8.9/142 SM, known NVRTC13.2, exact `(4621,1928,384)` and either bias state.
- `src/mamba_ssm/gpu/gemm_bi_fixed.rs:6987-7048` proves C0/C1 route to M64S2 on 13.2 regardless of output alignment, while 12.8/13.0 and adjacent/device/library variants return M128S2.
- For this shape, the generic fallback at 2294-2302 computes 222 M128x64 CTAs (`ceil(4621/128)*ceil(384/64)`), which is not below 142 SM, so it selects `Tf32M128S2`.
- `src/mamba_ssm/gpu/gemm_bi_fixed.rs:2343-2373` maps M64S2 and M128S2 to their exact function holders and launch geometry.

Physical availability, not a missing loader:

- `src/mamba_ssm/gpu/kernels.rs:967-989` loads all five portable TF32 symbols with mandatory `get(...)`; M64S2 has no optional/rejection holder and receives its 32,768 B shared-memory attribute.
- `src/mamba_ssm/gpu/gemm_bi_triad/modules.rs:1552-1558` inventories M64S2 and M128S2 in the Fixed module; `kernels/gemm_bi_fixed/tf32.cu:350-365` defines both bodies.
- `src/mamba_ssm/gpu/gemm_bi_fixed.rs:276-299` permits forced TF32 tiles for homogeneous F32; force mapping is explicit at `tests/gemm_bi_fixed_performance.rs:11007-11020` and its CC8.9 inventory at 11984-11995.
- `internal/perf/ada-rna-toolkit-census-20260906/cuda-12.8/timing101.log:8` and `cuda-13.0/timing101.log:8` record real M64S2 eager/graph-capable production functions under those NVRTC versions on other Ada shapes. This proves availability, not the C0/C1 win.

13.2 evidence:

- Pre-promotion: `internal/perf/fixed-tf32-all6-ada-20260906/c-m64s2-confirm101.log`, SHA-256 `908b0ffd1fd1e5127b51d7e440d530080a6bd07d985e2aa93720f095accd926a`: 8 C0/C1 records, both paths/orders and both bias states, candidate M64S2 vs AUTO M128S2; M64S2 ÷ AUTO M128S2 p50 ratios are `0.971979-0.977209`, and the same-direction p95 ratios are `0.973704-0.978250`; all raw/repeat/applicable graph bits pass.
- Actual AUTO: `internal/perf/fixed-tf32-c-postauto-ada-20260906/README.md` and its 101-window log compare the old forced/incumbent M128S2 against production AUTO M64S2. The raw `forced_over_auto_*` direction is M128S2 ÷ AUTO M64S2: p50 ratios are `1.036022-1.042052`, and p95 ratios are `1.038318-1.045031`. All eight owner-win cohorts and physical graph identities pass; ratios above one therefore favor AUTO M64S2.
- Independent cache-state repeats use that same raw M128S2 ÷ AUTO M64S2 direction: `fixed-tf32-c-postauto-privatecache-ada-20260906/README.md` (uncached/in-memory; p50 ratios `1.026771-1.032084`, p95 ratios `1.029164-1.033931`) and `fixed-tf32-c-postauto-privatecache-v2-ada-20260906/README.md` (active private cache; p50 ratios `1.025144-1.035193`, p95 ratios `1.026725-1.037257`). Both explicitly decline extrapolation to other toolkits.

No CC8.9 C0/C1 JSON performance record with `nvrtc:[12,8]` or `[13,0]` and `forced_tile:"Tf32M64S2"` was found in `internal/perf`. Therefore the correct backlog item is cross-toolkit C qualification, not a loader addition and not an unconditional guard relaxation.

## Safe backlog boundary

Leave both 13.2 guards unchanged until toolkit-specific evidence passes. A proposed 12.8 or 13.0 promotion must remain literal to CC8.9/142 SM, known compiler-library identity, the already admitted shapes/bias states and the exact tested production NVRTC artifact. If either toolkit loses p50 or p95, preserve its current fallback; a successful force launch is only availability evidence.
