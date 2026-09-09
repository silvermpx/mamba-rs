# Ada exact-F32 TN M128N64/BK16 dual-fused resource stop — 2026-09-09

Outcome: **resource stop before exact/timing.** This test-only d768-in
candidate keeps the retained transpose and replaces only the fused GEMM with a
M128N64/BK16/S2/256-thread GROUP_M6 body. It halves GEMM CTAs576->288 and
reduces modeled staged traffic576->432MiB while preserving two independent
ordered1024-FFMA chains and the exact FP64 finalize. Production dispatch is
unchanged.

The first CUDA13.2 resource gate found a24-byte stack frame in the raw
verification twin. Replacing the dynamic `kk < valid_k` loop with a statically
unrolled16-step loop plus uniform tail guard preserves the exact FFMA sequence
and reduced the frame to8 bytes, but did not meet the mandatory stack0 gate.
Per the frozen falsification criterion, the entire M128N64 mechanism stops;
exact and timing did not start and no further register/lifetime patch is
authorized.

- helper SHA256: `41be32910993e0d2139b5c0bba4ca9828c154a6b4f94f496cdde11a6d4e82f32`
- harness SHA256: `9b2219a7df9010db6a48e680a2d8ed968f06e360b6dea876a50c6de0bfd5eb74`
- CUDA13.2, RTX6000Ada, CC8.9,142 SM
