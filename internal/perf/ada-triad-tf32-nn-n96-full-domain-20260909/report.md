# Ada TF32 NN N96 full-domain cp.async compile stop — 2026-09-09

Outcome: **STOP before GPU.** The test-only Prism candidate adds a CTA-uniform
full-domain arm that uses three-operand 16-byte `cp.async` while retaining the
direct-`float2` N96 parent for edge and K-tail tiles. The first whole-mainloop
form compiled at 136 registers and duplicated static HMMA. One bounded factoring
refinement shares the MMA body, but the final candidate still exceeds the
predeclared 128-register cap. No timing or GPU exact corpus was launched.

## CUDA 13.2 / SM89 compile evidence

| Metric | retained | candidate |
| --- | ---: | ---: |
| registers | 124 | 131 |
| SASS text bytes | 47,616 | 49,536 |
| text ratio | — | 1.0403 |
| HMMA sites | 48 | 48 |
| LDGSTS sites | 21 | 42 |
| local/static shared/stack/spills | 0 | 0 |
| dynamic shared ABI | 86,016 B | 86,016 B |

The text `<=1.20`, HMMA parity and spill-free gates pass. The candidate fails
only `registers<=128`; this gate is not relaxed after observing the build.
Preserve the direct-float2 N96 retained-best and do not retry unchanged.

## Verification and frozen identity

- native structural tests: 17/17 PASS
- helper SHA256: `d1091f0093fbf17823e3998834173e2b7364e3be88a7e35c6f17824cc05f282c`
- harness SHA256: `2dd7c85cab5ffb5bc09580fe4c04f901d624f6a3cbdf5cf565dd0aa73ade6a0c`
- compile-only log SHA256: `3dfadb11431a2cf3d9fbb2117c0434824a22a4f9fb9ea0b79b7b8789b24e8e86`
- [compile-only raw output](compile-only.log)

Production and dispatcher are unchanged.
