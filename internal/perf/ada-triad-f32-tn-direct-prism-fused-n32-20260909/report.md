# Ada exact-F32 TN direct Prism fused N32 stop — 2026-09-09

Outcome: **valid exact/resource candidate, rejected at once3.** This was a
test-only attempt to replace the retained direct M64N64 raw-partial plus FP64
reducer with one M64N32 kernel that computes all six ordered
`784/784/784/784/784/701` chains and performs the exact finalize in registers.
Nothing from this experiment is admitted to the production dispatcher.

The candidate passes target, tail, exceptional, non-unit-alpha and K0 bit-exact
checks in eager and graph paths, including input immutability and 256-byte
guards. Its SASS resource census is 127 registers/thread, zero local bytes,
12,288 bytes static shared and occupancy4, satisfying the frozen resource gate.

Candidate/retained once3 p50/p95:

| Path/order | Ratio |
| --- | ---: |
| eager ABBA | 1.120229 / 1.121375 |
| eager BAAB | 1.120937 / 1.122896 |
| graph ABBA | 1.121476 / 1.123197 |
| graph BAAB | 1.122178 / 1.122496 |

The one-node fused arm is consistently 12.0–12.2% slower than the retained
direct M64N64 raw-partial plus reducer. Stop before once7 and Fast; do not retry
this unchanged M64N32 design.

Frozen identities:

- helper SHA256: `1ddc70919f44fc60a86757bbcf4f958156a8d9416beb9ec5933b571fcd1a4067`
- harness SHA256: `9b4546097ae784eca6598de53d2688aad3ca56e315f659bb4746890bd1a8dad6`
- transformed direct source SHA256: `80e1e578d68914c185b63a53da6743ba0676c798cb54da0ffd87f6daa89b9b5d`
- CUDA toolkit: 13.2; device: RTX 6000 Ada, CC8.9, 142 SM
