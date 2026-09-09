# Ada exact-F32 TN direct d768-in BK16 stop — 2026-09-09

Outcome: **valid exact/resource candidate, rejected at once3.** The test-only
direct row-major-X M64N64/BK16/S2 raw kernel emits two independent partial
planes and uses the unchanged production FP64 reducer. It preserves the two
ordered1024-FFMA chains. The comparator is the actual retained transpose plus
dual-chunk fused-finalize pipeline, not public AUTO. Production dispatch is
unchanged.

Both raw planes and target/tail/exceptional/non-unit/K0 eager+graph final bits,
repeat20 output, inputs and guards pass. Candidate resources are
107regs/local0/16KiB shared/occupancy4. Despite those resources, the separate
raw-plane global round trip plus reducer loses to the retained fused finalize
on this wide-output cell.

Candidate/retained once3 p50/p95:

| Path/order | Ratio |
| --- | ---: |
| eager ABBA | 1.109240 / 1.110718 |
| eager BAAB | 1.108154 / 1.110185 |
| graph ABBA | 1.108816 / 1.111380 |
| graph BAAB | 1.109483 / 1.110109 |

The candidate loses10.8–11.1% in every stratum. Stop before once7/Fast and do
not retry this direct raw-plus-reducer mechanism unchanged on d768-in.

- helper SHA256: `e26af561235d50ad59a154200d2808abcbd531103ce617328660325108440e61`
- harness SHA256: `bcdf68d5496cba32016e5154c99fd3a5ed8fab2557f1d54892ce605b7c160bfe`
- composed direct source SHA256: `29019451f56ae0d97a5c168e897c075e644779425a291bfb881d46d8868f3fa8`
- CUDA13.2, RTX6000Ada, CC8.9,142 SM
