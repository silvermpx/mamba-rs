# Ada TF32 TN M64N64 d768-out reuse stop — 2026-09-09

Outcome: **valid exact/resource reuse, rejected at once3.** The frozen
M64N64/BK32/S3 A-only RNA wave kernel that won on canonical Prism was paired
against the retained M128N96/BK32/S3 A-only RNA route on d768-out
`(2048,1536,768)`. Production dispatch is unchanged.

The candidate uses 83 registers, zero local bytes, 49,152 bytes dynamic shared
memory and two CTA/SM. The retained route uses 127 registers, zero local bytes,
86,016 bytes dynamic shared memory and one CTA/SM. Full, exceptional, tail and
K0 eager/graph bits, RNA scratch contents, guards and two-node graph identity
all pass.

Candidate/retained once3 p50/p95:

| Path/order | Ratio |
| --- | ---: |
| eager ABBA | 1.283422 / 1.290323 |
| eager BAAB | 1.284946 / 1.289889 |
| graph ABBA | 1.279570 / 1.279570 |
| graph BAAB | 1.286486 / 1.293478 |

The smaller wave-quantized body loses 27.9–29.3% on d768-out even though it
wins on canonical Prism. Stop before once7/Fast. Preserve M64N64 for Prism and
the retained N96 route for d768-out; do not infer cross-shape admission.

- harness SHA256: `897c6f6df842892bc619475cb148ef28c5d152f6200ed3d88807394fa548d04d`
- reused helper SHA256: `72b6bc0b42b3c6b54acf066f2941da7ee5413571209284c51243bd104c6a0486`
- composed candidate source SHA256: `af96f3a6d7f2858922721c8879f3a7d51380a86962bd75809a592ba82bd4ac67`
- candidate PTX SHA256: `d8e94bec59902c2d176baf4e39734ad26a97749935bea4f57447ad75da08c7b4`
- CUDA13.2, RTX6000Ada, CC8.9, 142 SM
