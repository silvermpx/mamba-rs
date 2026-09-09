# Ada exact-F32 TN canonical-Prism vec4 reducer stop — 2026-09-09

Outcome: **valid exact/resource candidate, retained-pipeline parity at once3.**
The test-only arm preserves the retained direct M64N64/BK16 raw partials and
replaces only the production scalar FP64 reducer with a four-output/thread
reducer. It performs six aligned float4 plane loads, four independent ordered
five-DADD chains and one float4 store. Production dispatch is unchanged.

Raw partials and final target/tail/exceptional/non-unit/K0 eager+graph bits,
guards and inputs pass. The vector reducer uses56 registers, zero local/shared
bytes and occupancy9; its grid is1,446x128 instead of2,892x256. The retained
raw kernel remains107regs/local0/16KiB/occupancy4.

Candidate/retained once3 p50/p95:

| Path/order | Ratio |
| --- | ---: |
| eager ABBA | 1.000374 / 1.001199 |
| eager BAAB | 1.000984 / 1.001890 |
| graph ABBA | 1.001158 / 1.001188 |
| graph BAAB | 0.999913 / 1.002314 |

The raw GEMM dominates the pipeline, so reducer-only vectorization does not
clear the strict gate. Stop before once7/Fast; keep the production reducer.

- helper SHA256: `b67163c37a3664e72e227402866f0698ae9e51232a2be703bc482b40c938041c`
- harness SHA256: `cd0d6f828e75f5d32b4a6f02495fb73445fa33fdc39734deaa40879a63b9b1a3`
- vector reducer source SHA256: `531cb7d9caaf5883fa90bbf5f77cbb7a0a6276ee9e7f6669dce3590e2e2b1b27`
- CUDA13.2, RTX6000Ada, CC8.9,142 SM
