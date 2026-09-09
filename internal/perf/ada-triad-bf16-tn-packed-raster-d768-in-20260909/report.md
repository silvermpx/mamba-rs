# Ada BF16 TN packed-raster d768-in stop — 2026-09-09

Outcome: **exact/resource-valid parity, rejected at once3.** This final
test-only discovery candidate preserves the retained M64N64/BK64/S2
regpipe+vec2 body and changes only target CTA traversal from a linear 576-block
grid to packed `(4,48,3)`, with `pid_m = blockIdx.x + 4*blockIdx.z` and
`pid_n = blockIdx.y`. Production dispatch is unchanged.

Candidate and retained both use 125 registers, zero local bytes, 32,768 bytes
static shared memory, 128 threads and three CTA/SM. Target, aligned and
misaligned negative-alpha tails, exceptional values and K0 pass exact eager/
graph repeats, guards, graph symbol/grid/ABI and resource/SASS gates.

Candidate/retained once3 p50/p95:

| Path/order | Ratio |
| --- | ---: |
| eager ABBA | 1.000587 / 1.000883 |
| eager BAAB | 1.000294 / 1.000855 |
| graph ABBA | 0.999705 / 1.001185 |
| graph BAAB | 0.998523 / 0.999409 |

The packed raster is parity: the measured range is approximately 0.15% slower
to 0.15% faster and misses the frozen `<0.99` gate in every stratum. Stop
before once7/Fast and do not retry the unchanged traversal. This closes the
current discovery shortlist before joint production-AUTO integration.

- harness SHA256: `e8612bf3784efaa4662d818fa24479028d88c2a80d9a6adfa536dee09c97e3dc`
- helper SHA256: `c7fc44cba99509b3f71151bd6809a9d58a7473c452c58bd7cc49af7f47d4f117`
- CUDA13.2, RTX6000Ada, CC8.9, 142 SM
