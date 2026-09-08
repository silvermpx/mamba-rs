# Ada TF32 NT: compact A+B ldmatrix, 2026-09-08

Valid measured loser; **stop without retry**. Logical d768-in
(2048,768,3072), RNA TF32, alpha1/beta0/no bias. Replacing the retained
A-only candidate's two scalar B words with `ldmatrix.m8n8.x2.shared.b16`
preserves exact output bits but is slower in every paired cohort.

| Paired comparison | Eager p50 | Graph p50 | Worst p95 | Outcome |
| --- | ---: | ---: | ---: | --- |
| Candidate / retained A-only | 1.02001–1.02006 | 1.02056–1.02067 | 1.021318 | 2.0–2.1% slower |
| Candidate / cuBLAS FAST_TF32 | 1.71999–1.72052 | 1.74527–1.74681 | 1.747486 | Fast remains ahead |

The retained A-only body stays the TF32 NT d768-in discovery baseline at about
200us. This experiment does not alter the central count of nine robust Fast
wins and is not a production dispatcher change.

## Focused verification

The CUDA13.2 RTX6000Ada test
`cuda_suite::ada_tf32_nt_compact_ab_ldmatrix_d768_in_discovery_once7` passes
target, forced-family tail, exceptional-value, direct-K0, repeat, graph and
guard checks. Native helper proofs pass5/5; the main native tree passes83/83.
Compiled resources are106 registers, zero local/static shared,49,152B dynamic
shared,256 threads and occupancy2. Candidate grid is192.

The paired screen uses seven windows for eager/graph x ABBA/BAAB,20 complete
GEMMs per observation and explicit retained-A-only and native cuBLAS
`CUBLAS_COMPUTE_32F_FAST_TF32` denominators. PRE, RELEASE and DRAIN snapshots
are quiet with no compute applications; four private cache files are unchanged.

## Frozen identities and evidence

- Main test SHA256: `812145f0b1a6c0729a93c64e2d6fdf36bc7137a3d053fc9966e26fb7772bdda9`.
- A+B helper SHA256: `687c7b4ad8f56c7932294ee7c0ff874e95299f92adbe96cb8e77271d857aaff9`.
- Composed candidate SHA256: `c2decbb1217501041ec1c973922dab15d565f58582c77023d609c5648ac4ece5`.
- Retained A-only composed SHA256: `b66739ed9d6433d157b3a96ca9bad6448d5b0e24e141ff0bde095e5161ead5f8`.
- Test binary SHA256: `4037d6f1f804393b9ca49014a3c72785e4d512cb12cc670b3be341b9f2d5a167`.
- [Raw log](evidence/cuda132/test.log), SHA256
  `da873e535eb439c29e196d0a0aef0e75167887eae966da8bdd38ad979d4e0f39`.

The measured sources are committed by frozen index content because the shared
test harness may advance independently. No losing route is wired into AUTO.
