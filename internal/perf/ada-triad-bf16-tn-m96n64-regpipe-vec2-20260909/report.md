# Ada BF16 TN M96N64 regpipe+vec2 d768-in stop — 2026-09-09

Outcome: **valid exact/resource candidate, rejected at once3.** This test-only
six-warp M96N64/BK64/S2 body reduces target staged half traffic16.67% while
preserving twelve resident warps. It is materially distinct from the earlier
wide-per-warp M64N128 loss. Production dispatch is unchanged.

Target, aligned/misaligned tails, exceptional and K0 eager+graph bits, inputs
and guards pass against retained BF16 M64N64 regpipe+vec2. Candidate resources
are124regs/local0/49,152B static shared/occupancy2; retained is
125regs/local0/32,768B/occupancy3.

Candidate/retained once3 p50/p95:

| Path/order | Ratio |
| --- | ---: |
| eager ABBA | 1.116597 / 1.121973 |
| eager BAAB | 1.115617 / 1.118488 |
| graph ABBA | 1.107080 / 1.107774 |
| graph BAAB | 1.108387 / 1.108785 |

The larger CTA loses9.3–11.7% despite lower staged traffic. Stop before
once7/Fast and do not retry unchanged.

- helper SHA256: `84c73f819087da8f0adfbcc9e08c1a40d7d3cd74314f42186e0aa441578a4fef`
- harness SHA256: `d3f26fd41cf601cb180d8ff98e29995cb479db11778901f471171b06975eef2e`
- CUDA13.2, RTX6000Ada, CC8.9,142 SM
