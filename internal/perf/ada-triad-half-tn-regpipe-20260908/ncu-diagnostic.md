# F16 TN d768-in register-pipeline diagnostic

One exact-symbol launch, Nsight Compute2026.1.0.0, RTX6000Ada/CUDA13.2.
The profile is diagnostic, not another performance benchmark.

| Observation | Measured value |
| --- | ---: |
| Tensor pipe active / elapsed | 42.1986% |
| Eligible warps / scheduler | .5687 |
| Active warps / scheduler | 2.4057 |
| Issue-active cycles | 40.3968% |
| Achieved / theoretical occupancy | 20.3394% /25% |
| Shared wavefronts, actual / ideal | 7,077,888 /7,077,888 |
| Excessive shared wavefronts | 0 |

Inference: dependency/issue latency and work distribution are useful next
targets; another shared-bank swizzle is not supported by this capture. This
does not prove a hardware performance limit. Profile durations95.488/95.904us
are not substituted for paired benchmark times.

The profiled symbol is `gemm_bi_tn_test_tc64_bk64_s2_regpipe_f16`. The unchanged
helper is `f3e66016fc6d9f2e584ac79486f09261c3be207790aad01f784ffcf3a63fddee`;
the newer typed harness is15cb1a3d and executableec4928ea, as recorded in the
[receipt](evidence/ncu-f16-d768-in/command.json). No old executable was rebuilt
just to reproduce its timing identity. A metadata-export NameError was repaired
after capture/drain without another GPU capture.

Raw evidence: [main counters](evidence/ncu-f16-d768-in/regpipe-f16-raw.csv),
[shared counters](evidence/ncu-f16-d768-in-source/regpipe-f16-source-raw.csv),
[SASS](evidence/ncu-f16-d768-in/regpipe-f16-sass.csv). Root independently checked
the raw counter values above. NCU reports and command/telemetry receipts are
retained beside these exports.
