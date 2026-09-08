# Ada half TN compact register-pipeline screen

CUDA 13.2 ran the exact ignored test
`ada_half_tn_tc64_bk64_s2_regpipe_d768_in_vs_current_and_fast_discovery_once7`
and passed. Main/helper SHA-256 values were `8096311fb5d83dee96911403749569a3f641658a24d24fbea170477cea1c5580`
and `f3e66016fc6d9f2e584ac79486f09261c3be207790aad01f784ffcf3a63fddee`;
binary SHA-256 was `e39aaed5985a6589b623110ccf3b9c470b7d5bc55c20aae93271a7bd186ef021`.

F16 and BF16 both passed all eager/graph repeat-bit checks. Each candidate
used 125 registers, 0 local bytes, 32,768 static shared bytes, 128 threads,
and occupancy 3.

The register-pipeline candidate improved on the test's current TC64 arm:
F16 p50 ratios were 0.8527--0.8545 and BF16 ratios were 0.8497--0.8536.
This comparison does not establish a new best over the previously measured
compact candidate because those two candidates were not paired here.

Against Fast, eager favored the candidate (F16 0.9630--0.9653; BF16
0.9569--0.9628), while graph regressed (F16 1.0158--1.0161; BF16
1.0097--1.0129). Both decisions are therefore valid `stop_no_retry`; there
is no Fast win.

The exact log is `evidence/cuda132/run1/test.log` (SHA-256
`4a3e04732f6190e48a291944cea2e2d5ffd80728191e2ae628c52a6cd98182d8`).
Root replayed112 paired brackets /448 observations and both quantiles.
The run contains2 resource,16 bit,16 screen and4 comparator-specific
decision records;20 GEMMs/observation with256B-aligned logical pointers.
Native source tests8/8 pass. PRE/RELEASE/DRAIN show no competing application
and the private cache is unchanged. Exact measured sources are archived
in `evidence/source/`. No production admission or full toolkit gate ran.
