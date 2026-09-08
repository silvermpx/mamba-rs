# Ada exact-F32 TN fused finalize, 2026-09-08

Retained-best win on RTX6000 Ada/CUDA13.2 for d768-in
`(M,K,N)=(2048,768,3072)`.  The candidate preserves the transpose and the
two ascending1024-element F32 FMA chains, but executes them as:

`transpose -> CopyPlan chunk0 raw partial -> CopyPlan chunk1 + exact FP64 finalize`

This removes the second partial write/read and the standalone reducer from the
previous four-node retained pipeline.

| Comparator | Candidate/comparator p50 | Worst p95 | Decision |
| --- | ---: | ---: | --- |
| retained SplitM CopyPlan | `.93356-.93661` | `.93782` | retain: `6.2-6.6%` faster |
| cuBLAS Fast | `2.42657-2.43007` | `2.43313` | not Fast-qualified |

Each range covers eager/graph x ABBA/BAAB paired once7 strata, with two warm-up
windows and20 logical GEMMs per observation.

## Verification

Both raw partial planes match production `gemm_bi_tn_splitm_partial_aligned`
bit-for-bit.  The raw-store accumulator path is proved unconditionally for
finite and exceptional target inputs plus finite and exceptional tails.  The
fused candidate passes target, tail, exceptional/NaN-payload, non-unit-alpha,
K0, eager/graph,20-operation accumulation, input/transpose/output/scratch
guards, and unused-partial-slab checks.

The exact finalize order is pinned as FP64 add, FP64 multiply, RN FP64-to-F32
conversion, then F32 add.  The fused kernel uses129 registers,32768B static
shared, zero local/dynamic shared,128 threads and occupancy3.  The retained
CopyPlan uses135 registers with the same shared memory, threads and occupancy.

## Frozen identity

- Harness SHA256: `9aa7a7955adb1867161deddf79662c4cbf4a541ec97354e7e1416e1fb7e8bb3f`.
- CUDA helper SHA256:
  `563d63f196f344505d832474ad87fdc7f20754490d2973358ae0e72c71c36a53`.
- Composed CUDA source SHA256:
  `d00b3e22260df21b4a59b03aee8b37962e078961ba611c871a6bf7879bed2730`.
- Test binary SHA256:
  `b3c3fc501a98b0f64453133948fb9f44d7fd28859cddc99fa2b26413b9a74e82`.
- Authoritative [once7 log](attempt1/once7.log), SHA256
  `1e209187d867a2bd081207887195b880c33651e9220766e5ce0810dca2e0fc44`.
- [Manifest](attempt1/manifest.json) binds source, helper, binary, toolkit,
  GPU UUID and log.  PRE/POST quiet gates pass with no competing compute
  process.

This is a discovery winner, not yet a production dispatcher admission.  The
next exact-F32 TN screen is `GROUP_M=12`; integration waits until the focused
shortlist is complete.
