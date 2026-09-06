# RTX 6000 Ada Fixed exact smoke (CUDA 13.2)

Source commit: `318b3fbd09c8fe0bfac7eaabefdec2bba457f212`

This is a three-window smoke, not a 21/101-window performance qualification. Lower ratios are better; a ratio below 1.0 means production AUTO was faster than the named cuBLAS denominator.

## Coverage and integrity

- Device: RTX 6000 Ada, CC 8.9, 142 SMs; driver 595.45.04; NVCC 13.2.51; runtime NVRTC `[13,2]`.
- Rows: `f32_exact` against `CUBLAS_COMPUTE_32F_PEDANTIC`, and the same exact custom outputs in `f32_exact_fast` against `CUBLAS_COMPUTE_32F_FAST_TF32`.
- All A-E shapes, bias false/true, eager/graph, both launch orders, and all three SM89 exact force candidates (`Legacy`, `F32N128S2`, `F32Sm89N64CopyPlan`).
- 240 records, 0 rejected, completion `passed:true`.
- All graph replay, raw-storage identity, AUTO repeat-bit, and forced repeat-bit checks passed.
- Binary SHA-256: `d0368f56ed46c35f536ca2331aeb7e1f34bbc03026c9820d409316eb26f05ca7`.
- Runtime Fixed digests: source `35d4fd870cd7c81aaf0ef5e7c877ed77ed0d95e80b9c522823af2a7c85a1210f`; invocation `f094b565e36deecb487d852e4fb67cbd775c4f35ebbc3fb852e06974d74dd6a4`; artifact `f933cc6ec76459f2b6a28ee36a77474e1804d6ca2ba44d95cd5113325b635dd7`.

## Production AUTO symbols

The mapping was identical for both exact comparator rows, both bias states, and eager/graph paths.

| Cells | AUTO tile | Captured CUDA graph symbol |
|---|---|---|
| A, C, D | `Legacy` | `gemm_bi_f32_f32_s2` |
| B, E | `F32Sm89N64CopyPlan` | `gemm_bi_nn_fixed_sm89_f32_n64_copyplan_v1` |

## Paired AUTO / vendor ratios

Each entry is paired p50/p95 over 18 samplewise ratios: 3 physical force-census contexts x 2 balanced launch orders x 3 windows. This preserves pairing. Separately medianing all AUTO and vendor samples gives 16/20 PEDANTIC wins; paired p50 gives 14/20. No FAST_TF32 comparison won (0/20).

| Cell | Bias | PEDANTIC eager | PEDANTIC graph | FAST eager | FAST graph |
|---|---:|---:|---:|---:|---:|
| A | 0 | 1.007 / 1.099 | 0.998 / 1.062 | 2.112 / 2.114 | 2.112 / 2.146 |
| A | 1 | 0.947 / 0.948 | 0.950 / 0.984 | 1.698 / 1.723 | 1.708 / 1.745 |
| B | 0 | 0.833 / 0.931 | 0.939 / 1.088 | 2.183 / 2.315 | 2.193 / 2.365 |
| B | 1 | 0.861 / 0.930 | 0.875 / 0.987 | 1.871 / 2.073 | 1.881 / 2.214 |
| C | 0 | 1.478 / 1.534 | 1.507 / 1.587 | 3.609 / 3.611 | 3.622 / 3.668 |
| C | 1 | 1.416 / 1.514 | 1.455 / 1.508 | 3.279 / 3.282 | 3.310 / 3.313 |
| D | 0 | 0.998 / 1.011 | 1.000 / 1.033 | 2.420 / 2.682 | 2.435 / 2.726 |
| D | 1 | 0.936 / 0.977 | 0.938 / 0.984 | 2.081 / 2.381 | 2.110 / 2.413 |
| E | 0 | 0.977 / 1.020 | 0.989 / 1.065 | 2.192 / 2.280 | 2.197 / 2.475 |
| E | 1 | 0.959 / 1.006 | 0.964 / 1.033 | 2.038 / 2.191 | 2.050 / 2.342 |

PEDANTIC paired-p50 best: B/no-bias/eager at 0.833 (about 16.7% faster). Worst: C/no-bias/graph at 1.507 (about 50.7% slower). FAST_TF32 best is still a loss: A/bias/eager at 1.698. Worst: C/no-bias/graph at 3.622.

The p95 values are intentionally not described as qualification-grade: three windows expose outliers but are too short for a release decision. They do show only 7/20 PEDANTIC path/bias comparisons below 1.0 at paired p95, while FAST has none.

## Compatible forced candidates not selected by AUTO

The force census establishes that the following candidates are runnable on this Ada, produce the same raw storage bits as current AUTO, and expose the expected graph symbol. The ranges below are forced/AUTO p50 across both balanced launch orders and eager/graph; both bias states showed the same conclusion in both exact-comparator repetitions.

| Cell | Current AUTO | Compatible candidate | forced/AUTO p50 range | Smoke conclusion |
|---|---|---|---:|---|
| A | `Legacy` | `F32Sm89N64CopyPlan` | 0.846–0.849 | Consistent candidate win; max smoke p95 below 0.963 |
| C | `Legacy` | `F32N128S2` | 0.954–0.959 | Consistent p50 win; PEDANTIC repetition max p95 1.066, so confirmation is needed |
| D | `Legacy` | `F32Sm89N64CopyPlan` | 0.830–0.845 | Consistent candidate win; max smoke p95 below 0.952 |

This is shape-specific, not a blanket promotion: `F32N128S2` is 1.12–1.16x AUTO on A/D, while `F32Sm89N64CopyPlan` is about 1.05x AUTO on C.

The current production gate explains the non-selection. `fixed_sm89_exact_n64_auto_eligible` admits only B/E on the measured CC8.9/NVRTC13.2 cohort (`src/mamba_ssm/gpu/gemm_bi_fixed.rs:4533`), and `fixed_pick_f32_exact` selects `F32N128S2` only for two CC12 shapes (`src/mamba_ssm/gpu/gemm_bi_fixed.rs:2013`). Thus A/C/D have compatible production candidates, but no current AUTO allowlist entry. A quiet 21-window screen followed by 101-window confirmation is required before changing that conservative dispatcher.
