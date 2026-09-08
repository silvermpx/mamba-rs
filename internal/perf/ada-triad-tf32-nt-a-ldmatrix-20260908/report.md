# Ada TF32 NT: A-only ldmatrix, 2026-09-08

NEW actual-AUTO improvement, **not a cuBLAS Fast win**. Logical d768-in
(2048,768,3072), RNA TF32, alpha1/beta0/no bias. Retain this source as the next
NTin baseline; do not repeat the unchanged losing Fast comparison.

| Paired comparison | Eager p50 | Graph p50 | Worst p95 | Outcome |
| --- | ---: | ---: | ---: | --- |
| Candidate / actual AUTO | .88616–.88641 | .88626–.88632 | .886833 | 11.3–11.4% less time |
| Candidate / cuBLAS FAST_TF32 | 1.69056–1.69225 | 1.71235–1.71258 | 1.713474 | Fast remains ahead |

AUTO cohort: candidate about200.2–200.6us versus225.9–226.3us. Fast cohort:
candidate about200.4–201.1us versus117.0–119.4us. Do not combine independent
cohort samples into a new paired comparison. The raw decision's `retain:false`
requires winning BOTH denominators: it does not erase the measured AUTO gain.
The central robust Fast-winning replacement count remains nine.

## Mechanism and focused verification

Only replace compact8/S2's scalar A-fragment shared loads with
`ldmatrix.sync.aligned.m8n8.x4.shared.b16`. B loads, RNA conversion and per-output
K8 MMA order remain unchanged. All-lane native coordinate/address proofs pass4/4.
Compiled resource gate passes98 registers, zero local/static shared,49,152B
dynamic shared,256 threads, occupancy2. Candidate grid192; actual AUTO identity
is `gemm_bi_nt_sm89_mma_tf32_compact8_v1_m128n64_bk32_s2`.

The focused CUDA test passes target equality against actual AUTO, forced-family
tail and exceptional-value equality, repeats, graphs and guards. K0 uses the
direct literal +zero oracle (8385 outputs, two eager and two graph repeats),
because the physical qualification facade rejects zero logical N. Timed
guarded candidate/vendor pointers are checked256B aligned. Fast explicitly uses
`CUBLAS_COMPUTE_32F_FAST_TF32`, rejects TF32 override0/PEDANTIC/non-HOST pointer
mode, warms up before nonempty capture and checks its own eager/graph bits.

One once7 screen per eager/graph × ABBA/BAAB for each denominator. The existing
AUTO screen uses25 iterations; the Fast screen uses20 complete GEMMs per
observation with C/A/B reseeding before the start event and checks after timing.
Root independently replayed all8 screen rows,56 brackets/224 observations,
including explicit Fast arm order and nearest-rank p50/p95. All ratios match.

## Reproducible evidence

CUDA13.2 on RTX6000Ada CC8.9, GPU UUID
`GPU-d1edd7be-e88d-aed6-047d-622163306f0e`.

- Test: `cuda_suite::ada_tf32_nt_compact_a_ldmatrix_d768_in_discovery_once7`.
- Measured test SHA256: `66edee469c9700aa85678ae5d80a915d72d94d75cd937eedc1f86a3a2d7d8ee5`.
- A-only helper SHA256: `c853f8b824d9a1717f1594dd063457cc0b6ba2713f57884adaa58e6e795d3cfa`.
- Composed CUDA SHA256: `b66739ed9d6433d157b3a96ca9bad6448d5b0e24e141ff0bde095e5161ead5f8`.
- Binary SHA256: `89ed67898e799e198b5c126a28fe51e31079a6b02eb44f844245139c77ac9299`.
- [Valid raw log](evidence/cuda132/run2-k0-repair/test.log), SHA256
  `48ba62ee7798609ac85ac9782619fb5739d4ebbd67fcb9173772e5946779bc02`.

PRE13:20:57UTC and DRAIN13:21:45UTC are quiet/no-apps; RELEASE13:21:40UTC
has no compute app but10% sampled utilization, so it is not labeled quiet.
Internal pre/timed/post quiet gates pass. Four private cache artifacts unchanged;
this is discovery evidence, not a new cold-cache qualification.

Preserved harness failures: initial Rust build rejected immutable Fast buffers
reset through mutable methods; repaired build then reached an unsupported K0
qualification request before timing. Both receipts remain under `evidence/`.
Neither is a kernel performance loss. Valid run2 uses unchanged A-only CUDA.

No production selector, Fixed inference or SM120 route changed. Next independent
hypothesis removes duplicate RNA conversions per shared stage; it must be ranked
against this retained A-only candidate, not just against the slower old AUTO.
