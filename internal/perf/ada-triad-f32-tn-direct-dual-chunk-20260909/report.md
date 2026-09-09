# Ada exact-F32 TN one-node direct dual-chunk stop — 2026-09-09

## Decision

STOP without cuBLAS Fast. The test-only one-node direct row-major-X dual-chunk
kernel is bit-exact and passes SASS/resources, but loses the current two-node
transpose+dual-chunk retained-best by16.2–16.6% in every paired once7 stratum.
Do not wire it into production or retry the unchanged direct-staging mechanism.

## Result

Shape: exact-F32 TN d768-in `(batch,k_out,n_out)=(2048,768,3072)`, CUDA13.2,
SM89 RTX6000 Ada, 20 logical GEMMs per observation.

| Path / order | candidate / retained p50 | p95 |
| --- | ---: | ---: |
| eager / ABBA | 1.16181 | 1.16245 |
| eager / BAAB | 1.16185 | 1.16369 |
| graph / ABBA | 1.16521 | 1.16607 |
| graph / BAAB | 1.16496 | 1.16546 |

Candidate observations are about369us; retained observations are about317us.
The strict retained gate failed, so cuBLAS Fast was not initialized or timed.

## Exactness, graph and resources

- Candidate graph is one node:
  `gemm_bi_tn_test_direct_sm89_f32_n64_dual_chunk_fused_finalize_v1`.
- Comparator is the actual retained-best from `edaa4d26`, two nodes:
  `transpose -> gemm_bi_tn_test_fixed_sm89_f32_n64_dual_chunk_fused_finalize_v1`.
- Both direct raw planes `[2,k_out,n_out]` match the production SplitM partial
  oracle bit-for-bit for finite/exceptional target and tail chunks.
- Final target, tail, exceptional payload, non-unit alpha, K0, eager/graph,
  20-op accumulation, guards and input immutability PASS.
- Direct raw:151 regs, local0, shared32768, occupancy3.
- Direct fused:168 regs, local0, shared32768, occupancy3.
- SASS has FFMA and LDGSTS; stack/spills are zero and LDL/STL/ATOM/RED/REDUX
  are absent.
- Retained fused remains163 regs/local0/shared32768/occupancy3.

This result establishes that eliminating the transpose pass and graph node is
not sufficient: direct row-major-X staging is slower than the transposed
CopyPlan access pattern even after the two exact chains are fused into one CTA.
Without a profiler attribution, do not claim a single microarchitectural cause.

## Evidence identity

- raw log SHA256 `47d632829a907ba37deae1e1e84c630199670cdcb8577f53fc5badda7a623135`
- source helper SHA256 `621e636c9870a0efcf0abf4f7e60e6558895c6f19f73b1671d0e99a76ad751ed`
- standalone harness SHA256 `07ca72ebd71b6d8e3b49e45480cf0ac714022e89e7d4b239ce921593b17695bd`
- composed direct raw/fused source SHA256:
  `b15408367aea23bebdb2a08d50c038e17f9ccb0b192a78d13f271765d6df358a` /
  `09f254026a3676c26bb777347dff95a55818be3d2847c492d45b785b5e90b1fe`.
- Native source/harness tests:13/13 PASS; CUDA feature no-run PASS.
- Immediate external and in-harness GPU quiet gates: PASS with ample VRAM.

Production and dispatcher were unchanged.
