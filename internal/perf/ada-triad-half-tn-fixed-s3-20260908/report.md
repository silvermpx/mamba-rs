# Ada half TN Fixed-S3 discovery

Valid loss on d768-in for both F16 and BF16. Keep the earlier compact/regpipe
shortlist; do not repeat this unchanged M128N128/BK64/S3 candidate.

| Dtype / comparator | Eager p50 ratio | Graph p50 ratio | Worst p95 |
| --- | ---: | ---: | ---: |
| F16 / current TC64 | 1.0771–1.0793 | 1.0784–1.0788 | 1.0870 |
| BF16 / current TC64 | 1.0794–1.0803 | 1.0742–1.0783 | 1.0883 |
| F16 / native-half Fast | 1.2380–1.2387 | 1.2789–1.2794 | 1.2874 |
| BF16 / native-half Fast | 1.2392–1.2404 | 1.2796–1.2816 | 1.2844 |

Ratios are candidate/comparator from paired eager/graph ABBA/BAAB once7,
20 GEMMs per observation, 256B-aligned pointers. The candidate takes about
112 microseconds versus current TC64 about104 microseconds. Current TC64 is
the forced family reference, not a new public AUTO admission.

Exact test `ada_half_tn_fixed_s3_bxor_d768_in_vs_current_and_fast_discovery_once7`
passed on RTX6000Ada / CUDA13.2. Both half inputs produce F32 dW matching the
reference bit-for-bit in focused eager/graph repeats. Target `(2048,768,3072)`:
grid144, block256, 162 registers, zero local/static shared bytes, 98,304 dynamic
shared bytes, occupancy1. Root native helper tests passed10/10; source/runtime
review preceded the GPU run.

Root replayed112 brackets /448 observations and both nearest-rank quantiles,
with16 bit records,16 timing screens,2 resource records and4 STOP decisions.
Receipts show unchanged private cache, quiet PRE/DRAIN and no competing
application. Immediate RELEASE records11% GPU utilization after the test.
Raw evidence is under `evidence/cuda132/`.

The exact measured source is the source snapshot in this report's commit:

- Main: `15cb1a3d5e9840cf3342ed605fc727afee509f9a04dfead1f5dee0afbbd8f092`
- Helper: `9147da28e1948a1774d45d87823b3c1cb67fbdcaea4e34802151574bfc14493e`
- Composed CUDA: `c3084901ac0f98ed524756a54920ee220b6b8724c8ace324608e5a2c97b32da5`
- Binary: `ec4928ea17a61ce654419e9067540dfb0a9aeea838f83f8f009448aa550fae56`
- Raw log: `2a3a19ee5c9395c84bd3467fa7179d17a6cf508a623bf3d11bd1e484d5871277`

No production/Fixed/SM120 change, full gate or other-toolkit qualification.
