# Ada BF16 TN one-wave atlas resource stop — 2026-09-09

Outcome: **resource stop before exact/timing.** The test-only d768-out atlas
maps exactly142 CTAs to Ada's142 SMs:100 M96N96 tiles plus42 M64N96 tiles,
with1152 active compute warps and29.6875% less logical staged half traffic than
retained M64N64. Non-target shapes retain the previous kernel. Production
dispatch is unchanged.

The initial compile exposed an invalid generated-source collision because the
F16 and BF16 macro instances declared the same `atlas_smem`. A dtype-tokenized
identifier fixed only that collision. A second harness defect treated ptxas'
omitted zero static-smem metric as missing data; the parser now follows existing
dynamic-shared harnesses and maps omission to static0 while retaining the
runtime65,536B dynamic-shared gate.

The repaired CUDA13.2 resource check reaches the real decision and reports
130 registers versus the frozen128 cap, static shared0. Exact and timing did
not start. Stop the one-wave atlas unchanged rather than weakening the cap or
shaving two registers after observing it.

- helper SHA256: `31c0e9b8f6b465ed36253e2b531e6bfb8ae7088821d8179402b102a6c232e253`
- harness SHA256: `da1359392a4f9fc2b07af1e5b713de4b22ea495c134daeaf22d47b5ee2cb8861`
- CUDA13.2, RTX6000Ada, CC8.9,142 SM
