# Ada Fixed/inference F16 NN N96/S3 retained win and Fast near miss — 2026-09-09

## Decision

Retain the test-only F16 NN d768-in `(2048,768,3072)` M128N96/BK64/S3
candidate as the new **Fixed/inference** shortlist leader over the frozen Fixed
S3 comparator. It passes the strict retained-best gate in every once7 stratum,
but does not beat native-half cuBLAS Fast. Its `gemm_bi_nn_fixed_sm89_*` symbol
and `FixedSm89HalfSwizzleParams` ABI belong to the standalone Fixed dispatcher,
so this result must not be counted as a training-Triad win.
Production and dispatcher remain unchanged.

The candidate narrows the parent CTA from N128 to N96 and uses eight 64x24
warp tiles. On the target shape this raises the launch from 96 to 128 CTAs,
reduces per-CTA B/accumulator work, and cuts the compiled register count while
preserving the parent's BK64/S3 ordered MMA pipeline.

## Correctness and resource gates

- CUDA 13.2, SM89, 142-SM NVIDIA RTX 6000 Ada Generation.
- Exact PASS against frozen Fixed S3 for target, negative-alpha M tail, N tail,
  K tail, exceptional full tile and K=0.
- Eager and captured-graph execution are bit-exact across three repeats; target
  also passes eager/graph 20-operation repeats, input immutability and guards.
- Candidate: 156 registers, local/static shared 0, dynamic shared 86,016 B,
  block 256 and occupancy 1.
- Frozen Fixed S3 comparator: 188 registers, local/static shared 0, dynamic
  shared 98,304 B, block 256 and occupancy 1.
- CUDA 13.2 SASS is spill-free. Candidate/comparator HMMA, LDSM and LDGSTS
  counts are `96/128`, `63/72` and `21/24`, respectively.

## Paired once7 results

Candidate/frozen Fixed S3:

| Path/order | p50 | p95 |
| --- | ---: | ---: |
| eager ABBA | 0.862817 | 0.864211 |
| eager BAAB | 0.863538 | 0.863831 |
| graph ABBA | 0.866423 | 0.866740 |
| graph BAAB | 0.866375 | 0.866740 |

All four strata pass the strict `<0.99` retained gate. The candidate is about
13.3–13.7% faster than the previous retained-best at the median.

Candidate/native-half cuBLAS Fast:

| Path/order | p50 | p95 |
| --- | ---: | ---: |
| eager ABBA | 1.008720 | 1.010038 |
| eager BAAB | 1.009345 | 1.011727 |
| graph ABBA | 1.007640 | 1.010204 |
| graph BAAB | 1.008071 | 1.011470 |

The strict Fast `<1.0` gate fails in every stratum. This is a consistent near
miss of roughly 0.76–0.93% at p50, not a cuBLAS Fast win.

## Frozen evidence and source-hash correction

- helper SHA256:
  `ce9c6ce3ddf158371d3f3844ab7e0232f99383429391d6777fa80d38d52e7476`
- harness SHA256:
  `bf1750195e3590f8a40060317f43b255ccdad1c2279fb59ff2a7c3d16840bd1d`
- runtime composed candidate source SHA256:
  `efa92dfc2022374bac411b115668a65fe6398500197e6806832daadb0ecc8260`
- runtime composed retained source SHA256:
  `e09931f8fcdad3737257e2e2892f5e896b25d405c83f337e3752adb18096b60b`
- raw log SHA256:
  `08c8adefd9264091cc006a823d3459f6ef761e7efb2392cea46d678517b05292`
- [raw output](raw.log)
- [launch preflight](preflight.txt)

The initial compile-only receipt incorrectly labelled hashes of the transformed
fragments as `candidate_source_sha`/`retained_source_sha`. The reporter now uses
the SHA returned by the same composed-source NVRTC path as runtime, so its
source labels agree with the measured receipt above. This evidence-only change
did not alter generated kernel bytes: candidate PTX/CUBIN/SASS remain
`8d35e2f471c86a7a2798c18ec122b5613be17787ed36d04cb0c55002511366ce` /
`d3e8388f865bdd1a08729d772b7991908af10626d8d63c677eb00dc15afbb2a1` /
`e6aaa5fa1c55ba269ca18f5d59f1e3341b2c951dba2a557f7273523f20e9f452`;
retained PTX/CUBIN/SASS remain
`fe1cea43193613bd6b94bc477991d8ccb7a03d03976f565fd47e1f49b0c9101e` /
`eb2fe8997d55b7a1852a6a85f8e3613fdf74f203b734ae1a4a95d61c95da58d8` /
`8db079e2ef918a39dcadd2bee355bc12b81ecd02cc85e01a7376838c253857ac`.
