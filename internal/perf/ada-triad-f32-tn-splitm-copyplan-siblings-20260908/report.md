# Ada exact-F32 TN SplitM CopyPlan siblings, 2026-09-08

One new retained-best result and one valid loss on RTX6000Ada/CUDA13.2. Both
candidates preserve the public AUTO SplitM partitions and reuse the unchanged
FP64 reducer. Neither beats cuBLAS Fast.

| Cell | Exact AUTO plan | Candidate/AUTO p50 | Worst p95 | Candidate/Fast p50 | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| d768-out `(2048,1536,768)` | `512 x 4` | .88484-.90297 | .90506 | 2.71131-2.77403 | **retain: 9.5-11.5% faster than AUTO** |
| Prism `(4621,384,1928)` | `784 x 6` | 1.34781-1.36362 | 1.36499 | 4.11623-4.15566 | stop: 34.8-36.4% slower than AUTO |

Each range covers eager/graph x ABBA/BAAB paired once7 strata. The timing
protocol uses two warm-up windows and20 complete logical GEMMs per observation.
The raw-store fallback was not used by either measured candidate.

## Exact and physical verification

For both cells, every raw partial matches
`gemm_bi_tn_splitm_partial_aligned` bit-for-bit before the shared reducer. Full
target, tail, exceptional payload, non-unit alpha, K=0, guards, eager/graph and
20-op accumulation checks pass. AUTO graph inspection proves the expected
partition and reducer arguments. Candidate graphs are transpose + one exact
CopyPlan launch per chunk + the unchanged reducer: six nodes for d768-out and
eight for Prism.

d768-out uses the production transpose and four BK32-aligned CopyPlan chunks.
Prism uses the existing raw bit-preserving padded transpose with stride4624 so
the CopyPlan body enters its async path. Its five784-row chunks and final701-row
chunk remain exact despite padded zero FMAs, but the larger eight-node pipeline
is decisively slower than the retained AUTO route. Do not retry or integrate
this Prism candidate. The next Prism hypothesis is a direct/tail-safe BK16 body.

The shared resources match the previous d768-in screen. The additional padded
transpose uses26 registers,4224B static shared, zero local/dynamic shared,
256 threads and occupancy6. Both partial and transpose scratch extents fit the
existing production capacities.

## Frozen identity

- Harness SHA256: `5e9a0661e94775d83b54ae6218ba805bbebc70f9fae68aa6bb5309e972bd01a8`.
- Padded-transpose helper SHA256:
  `c97995e0c20ddad382c6ef5b55d18719d094d06c38d882627f2816280693afc6`.
- Unchanged raw-store helper SHA256:
  `0ceade475a0fce4ec2fe5a25e0defe7ea950aa3c9fed09c93230eb863fb44c0d`.
- Test binary SHA256:
  `1569040aba165833538b4f2b0ef1d64436c6897f9eccb233aab39374b205aae6`.
- Authoritative [once7 log](attempt1/once7.log), SHA256
  `ea1989bee3e41338ee916378b484eb8e31e2aef637b83f360d9e9c3eab70cfcc`.
- [Manifest](attempt1/manifest.json) binds the source, helpers, binary, toolkit,
  GPU UUID and log. PRE/POST telemetry records the correct idle CC8.9 GPU with
  no competing compute process.

The d768-out route is a frozen discovery winner, not yet a public dispatcher
promotion. Preserve it for the joint retained-best integration and toolkit
qualification batch. Prism is a valid exact performance loss and is excluded.
