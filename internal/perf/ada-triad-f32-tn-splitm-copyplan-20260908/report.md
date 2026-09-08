# Ada exact-F32 TN d768-in SplitM CopyPlan, 2026-09-08

New retained-best discovery result on RTX6000Ada/CUDA13.2; this is **not** a
cuBLAS Fast win.  The candidate preserves the selected exact arithmetic tree:

1. transpose the full input once;
2. run the production exact-F32 CopyPlan on the first 1024-row chunk;
3. run the same CopyPlan on the second 1024-row chunk;
4. use the unchanged production FP64 SplitM reducer.

This replaces the rejected full-K probe, whose single F32 chain changed the
output by 3 ULP.  Each candidate raw partial is independently bit-identical to
`gemm_bi_tn_splitm_partial_aligned` before the common reducer is applied.

| Comparator | Candidate/comparator p50 | Worst p95 | Decision |
| --- | ---: | ---: | --- |
| actual public AUTO | .84635-.85354 | .85527 | **retain: 14.5-15.4% faster** |
| cuBLAS Fast | 2.58130-2.59599 | 2.60175 | not Fast-qualified |

All four eager/graph x ABBA/BAAB strata pass the strict `.99` retained gate.
The paired screen uses seven measured windows after two warm-up windows and 20
complete logical GEMMs per observation.

## Exact and physical verification

The target `(2048,768,3072)`, tail `(2047,68,69)`, exceptional payload,
non-unit alpha, and K=0 cases pass exact output bits, repeated accumulation,
input immutability, guards, eager execution and captured-graph execution.  The
AUTO target graph is the expected two-node `TnSplitM { m_chunk:1024,
chunks:2 }` route.  The candidate graph contains transpose, two CopyPlan
launches and the unchanged reducer in that order.

CUDA13.2 Driver-JIT resources are:

| Symbol | Registers | Static shared | Local | Threads | Occupancy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gemm_bi_nn_fixed_sm89_f32_n64_copyplan_v1` | 135 | 32768 B | 0 B | 128 | 3 |
| `gemm_bi_transpose_f32_32x16_d768_v1` | 18 | 4224 B | 0 B | 512 | 3 |
| `gemm_bi_tn_splitm_partial_aligned` | 127 | 33792 B | 0 B | 256 | 2 |
| `gemm_bi_splitm_reduce` | 40 | 0 B | 0 B | 256 | 6 |
| `gemm_bi_tn_zero_reduction_v1` | 16 | 0 B | 0 B | 256 | 6 |

## Frozen identity and failed-attempt accounting

- Harness SHA256: `14d5082eecf33d1bea9b605477fbdf0b3f190744de71372064f6e81eeadaf0bb`.
- Raw-store proof helper SHA256:
  `0ceade475a0fce4ec2fe5a25e0defe7ea950aa3c9fed09c93230eb863fb44c0d`.
- Test binary SHA256:
  `162ddb353554cf3cc2c7ce3d2e644fef00e560c0d33a98caf93ada5b8aee8125`.
- Authoritative [once7 log](attempt3/once7.log), SHA256
  `26cef0f690b03a971173364f1c0c44000add0437cdfe9223120813c2e141b0f9`.
- [Manifest](attempt3/manifest.json) binds the source, helper, binary, toolkit,
  GPU UUID and log. PRE/POST telemetry records the correct idle CC8.9 GPU with
  no competing compute process.

The first attempt stopped before timing because its manual K=0 probe used
`alpha=-0.75`, while the public dW API fixes `alpha=beta=1.0`; its signed-zero
mismatch is a harness failure, not a kernel loss. The repaired test also checks
the complete AUTO zero-kernel graph ABI. Attempt2 accidentally omitted
`--ignored` and executed zero tests. Both non-authoritative logs are preserved.

This is a frozen discovery winner, not yet a public dispatcher promotion.
Preserve it for the joint retained-best integration and supported-toolkit
qualification batch.
