# F16 NN d768-in M128N64/S2: valid loss, stop

Ada/CUDA13.2, logical(2048,768,3072). Reused unchanged Fixed N64 body via a
test-only exact-shape wrapper; no production route change. Candidate132regs,
local/static0, dynamic49,152B,128 threads, grid768, occupancy2. Target exact
TC128/retainedS3 eager/graph bits pass; all five timed pointers are256B aligned.

| Candidate / comparator | Eager p50 | Graph p50 | Worst p95 |
| --- | ---: | ---: | ---: |
| Retained Fixed S3 | 1.02499–1.02560 | 1.03329–1.03353 | 1.034729 |
| Native-half cuBLAS Fast | 1.04441–1.04846 | 1.07537–1.07557 | 1.078501 |

Candidate64.2–65.0us versus retained62.1–63.6us: finer grid/occupancy does not
pay for this cell. Keep S3. No unchanged retry and no integration of this loser.
Focused once7,20 GEMMs/observation, eager/graph × ABBA/BAAB. Root replayed
8screens/56brackets/224observations and nearest-rank quantiles; native helper6/6.

Exact test `ada_half_nn_m128n64_s2_f16_d768_in_vs_s3_and_fast_discovery_once7`.
Measured main SHA256 `f5d3303c2fa2cb45343908eff1aa707f8f1e9183000f16408c52bd4f3ea04584`;
helper `122083c95d74e970340d4ac19c69aa29edcf44740f5df86d1d8b1a354b8e23df`;
composed CUDA `959b8a453cfb8612423be70f415a87bb6fd068b02470c2534bccac1c1ef1bf35`;
binary `1e073b4ab2f17a17c2dae0d3f6fef64aeb88f5db901916f582bb30836129bb40`.
[Raw log](evidence/cuda132/run1/test.log) SHA256
`5974fa79f0813331c4062444bab85b20e449b83d7b8a9aa8a44748755e9da1de`.
Exact1 PASS, private cache unchanged, PRE13:33:10/DRAIN13:33:49UTC quiet;
RELEASE13:33:44 has no compute app. Discovery only, not full qualification.
