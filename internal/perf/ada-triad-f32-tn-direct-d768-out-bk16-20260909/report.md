# Ada exact-F32 TN direct d768-out BK16 winner — 2026-09-09

Outcome: **new retained-best exact-F32 d768-out winner; cuBLAS Fast miss.** The
test-only direct row-major-X M64N64/BK16/S2 raw kernel preserves the four exact
512-FFMA chains and unchanged production FP64 reducer. It replaces the retained
six-node transpose + four Fixed CopyPlan launches + reducer with a two-node raw
launch + reducer. Production dispatch is unchanged pending joint integration.

All four raw partials and target/tail/exceptional/non-unit/K0 eager+graph final
bits, repeated20-op output, inputs and guards pass against actual AUTO and the
retained CopyPlan pipeline. Candidate resources are107regs/local0/16KiB shared
and occupancy4.

Candidate/retained once7 p50/p95:

| Path/order | Ratio |
| --- | ---: |
| eager ABBA | 0.850732 / 0.851366 |
| eager BAAB | 0.850671 / 0.851388 |
| graph ABBA | 0.869166 / 0.869396 |
| graph BAAB | 0.868509 / 0.868883 |

The new pipeline is14.9–15.0% faster eager and13.1–13.2% faster in graph.
Candidate/cuBLAS Fast p50 is `2.3532–2.3608` with worst p95 `2.3634`, so this
is not a Fast win. Retain for joint integration.

- helper SHA256: `6822ab11a67430c6171198a0d71f18ff1aba5a7511c405ed3a33fedcc5cb8551`
- harness SHA256: `2051fec72dc6a09362bb18ae6db0857e73106344de7aebbaf32e2e5408b88f74`
- candidate source SHA256: `43ee744b6911fefe2c93262a739fe73b958df4fd95f298d6afaf4baff58f598a`
- CUDA13.2, RTX6000Ada, CC8.9,142 SM
