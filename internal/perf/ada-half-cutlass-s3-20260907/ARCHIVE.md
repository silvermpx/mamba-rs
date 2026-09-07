# Version-controlled Task5B checkpoint

This archive preserves the final source, reproducible harness and tests,
corrected functional evidence, four raw timing runs, source/binary hashes and
resource/SASS proofs for the S3 experiment. `ARCHIVE_SHA256SUMS` checks the
selected version-controlled files. The original `SHA256SUMS` records the full
129-file local evidence set, including executable/SASS/intermediate artifacts
that remain on disk and are not all part of this Git checkpoint.

Root independently verified all129 original manifest entries and recomputed
the final paired101 ratios. Independent kernel/harness and final evidence
reviews accepted retention of both own improvements after correcting the
poisoned-graph and bias-corpus gaps. Only final/ timing is admitted.

| CUDA13.2 B0, no bias | S3 / production p50 | p95 | S3 / cuBLAS Fast p50 | p95 |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 0.964968888968 | 0.996412833347 | 1.267528694145 | 1.290857051084 |
| F16 | 0.964326220305 | 0.981133357944 | 1.187129654181 | 1.221770514572 |

These are standalone same-binary NVCC comparisons against the unchanged
production Swizzle body. Production NVRTC integration, per-toolkit timing and
AUTO promotion are the next stage. BF16 p95 margin is only0.36%; cuBLAS Fast
is still faster for both dtypes. No release-wide or cross-device win is claimed.

The final candidate is bit-identical to production across the exercised
ordinary/exceptional/bias/view/tail/eager/poisoned-graph corpus. This is not a
claim of untested cross-architecture bit equivalence.

No artifact was deleted. The full local manifest SHA is
`80d4652c24e35b68d997e38c3a98f15bb27b6908a919c8e90c403960b08a3833`.
