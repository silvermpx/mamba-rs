# Ada exact-F32 TN direct CopyPlan, 2026-09-08

Valid exact performance loss on RTX6000Ada/CUDA13.2. Keep the retained
transpose + two CopyPlan chunks + FP64 reducer pipeline; do not integrate or
retry this unchanged direct candidate.

The candidate removes the global transpose and merges both independent partial
chunks into one `grid.z=2` launch, followed by the unchanged production FP64
reducer. It stages row-major X directly into shared `[BK][BM]` form using
coalesced16-byte `cp.async`, but that direct shared-load path consumes the
expected memory-pass saving.

| Comparator | Candidate/comparator p50 | Worst p95 | Decision |
| --- | ---: | ---: | --- |
| retained SplitM CopyPlan | .99956-1.00715 | 1.00817 | stop: graph is0.67-0.72% slower |
| cuBLAS Fast | 2.57827-2.61195 | 2.61653 | not Fast-qualified |

Eager is effectively tied but fails the strict p95 gate; both graph orders are
slower. Each range covers eager/graph x ABBA/BAAB paired once7 strata, with two
warm-up windows and20 logical GEMMs per observation.

## Verification

Both full-mantissa and exceptional target raw partial planes match production
`gemm_bi_tn_splitm_partial_aligned` bit-for-bit. The exceptional tail
`(2047,68,132)`, target, non-unit alpha and K=0 final checks pass exact bits,
guards, eager/graph execution and20-op accumulation. The candidate graph is
exactly direct partial → unchanged reducer; the retained comparator graph is
transpose → CopyPlan0 → CopyPlan1 → reducer.

The direct kernel uses107 registers,32768B static shared, zero local/dynamic
shared,128 threads and occupancy3. The retained CopyPlan uses135 registers and
the same shared/threads/occupancy, so lower register pressure alone does not
improve residency or whole-pipeline time.

## Frozen identity

- Harness SHA256: `09c6f5ce609194b0b32ecfe49fc3dd561afd22b0f256199d99db62c6828e258d`.
- Direct CUDA helper SHA256:
  `8109bb4d248e356ddc981029b721083f732d3517ee2d2b164ce0a8210754f821`.
- Composed CUDA source SHA256:
  `b31acd1f1909a0afe086557168b12f951e6851a954e50ede30467039cc7aa19b`.
- Test binary SHA256:
  `0d022e4a82cbef51319b28e68dee8e743008da9ada28d4c05e8f991d3eb13e60`.
- Authoritative [once7 log](attempt2/once7.log), SHA256
  `8f738a390f6803d5c0a614394ac46e74dc69c9632eb57ffd5ac87f79692610f3`.
- [Manifest](attempt2/manifest.json) binds source, binary, toolkit, GPU UUID and
  log. PRE/POST quiet gates pass with no competing compute process.

Attempt1 used a wrong test module prefix and executed zero tests. It is retained
as a non-authoritative harness invocation, not performance evidence.

The next structural candidate should preserve the retained CopyPlan compute
path and fuse the second chunk with the exact ordered FP64 finalize, removing
partial1 traffic and the separate reducer instead of retrying direct staging.
