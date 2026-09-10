# Ada model CUDA Graph checks

Source commit: `732c11462dd9a47376e6d15b36040ca69bde3e34`.
RTX6000 Ada, CC8.9/142SM, driver595.45.04, CUDA13.2.
The immutable source manifest, device record, per-test build/list/preflight
logs and four executable hashes are retained in this directory.

All seven selected tests passed:

| Test | Observed result | Test runtime, s |
|---|---|---:|
| Mamba exact-F32 training graph vs eager | Sampled norm/in-projection weight max error0 | 160.20 |
| Mamba-3 exact-F32 training graph vs eager | Sampled norm/in-projection weight max error0 | 20.03 |
| Mamba BF16 five-step graph vs eager | Zero bit-divergent master weights | 4.64 |
| Mamba-3 BF16 five-step graph vs eager | Master-weight max error0 | 5.09 |
| Mamba-3 BF16 graph reconstruction/repeat | Output/gradient/weight digests repeat | 21.50 |
| Mamba-3 F16 graph reconstruction/repeat | Output/gradient/weight digests repeat | 10.12 |
| Mamba-3 F32 graph reconstruction/repeat | Output/gradient/weight digests repeat | 10.08 |

The first test includes cold context/kernel initialization; these runtimes
are test durations, not model latency or GPU-performance comparisons.

The F32 tests execute complete small-model forward/backward/AdamW steps with
batch-invariant Triad and require sampled-weight errors below1e-5, not full
model bit equality. The Mamba-3 BF16 eager comparison uses its existing5e-5
tolerance. The repeated-capture tests compare same-process digests.

All seven tests disable tensor cores and use the exact-F32/upcast Triad path.
They therefore do not qualify native-half winning kernels, production-size
shapes, cross-device bits or positive custom inference graph replay. Those
remain separate checks. The expected Mamba route-drift warning belongs to
its deliberate post-capture policy-change rejection assertion.
