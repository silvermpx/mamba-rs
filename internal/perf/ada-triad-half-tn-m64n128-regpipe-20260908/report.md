# Half TN four-warp M64N128 regpipe: valid loss

Ada/CUDA13.2, F16 TN d768-in(2048,768,3072). The new four-warp N128
adaptation passes focused target bits/resources, but loses retained N64 vec2
by19.2–19.6%. **Keep retained vec2. Stop; no unchanged retry or BF16 sibling.**
This is distinct from the older already-losing eight-warp M64N128 probe.

| Candidate / comparator | Eager p50 | Graph p50 | Worst p95 |
| --- | ---: | ---: | ---: |
| Retained N64 regpipe+vec2 | 1.19537–1.19620 | 1.19173–1.19210 | 1.200511 |
| Native-half cuBLAS Fast | 1.15015–1.15817 | 1.19185–1.19265 | 1.197893 |

Candidate104.3–105.9us versus retained87.4–88.7us. Reuse savings do not pay
for this geometry: actual register count rises125→200; occupancy falls3→2.
This is an observed tradeoff, not proof that registers alone caused the loss.
Candidate local0/static49,152B/dynamic0,128 threads,grid288, occupancy2. Native
copy/output ownership, alignment and unchanged K16/vec2 source checks pass15/15.

GPU candidate/retained/currentTC64 exact target eager2+graph2 checks pass;
both own graphs have exact symbol/config and Fast is a nonempty whole graph
with its own bit oracle. All five timed pointers have mod256=0. Once7×4
paired strata per comparator,20 complete accumulated TN calls/observation,
pre-event reset, exact repeated outputs and input/output guards. Root replayed
8screens/56brackets/224observations and nearest-rank p50/p95. No full gates.

Exact test `ada_half_tn_m64n128_bk64_s2_four_warp_regpipe_vec2_f16_d768_in_discovery_once7`.
Measured main SHA256 `6c7ebc5bf348600bab33c53c96c8b8e9cd3a3879793ebcd1ae003f1e75421cfe`;
helper `ef9e3f7ff7e2f42b551f3ff949ff495b5f0855106894ed8039a66f7eafdd8ec3`;
composed CUDA `6cbe2fbb1139b63c899749d6d0fad74146e06e350c70bbd8ac30acdd8cebace5`;
binary `e6bf02729d031bd195cfacf031fd0c35ca3ed88bac819c4abc1d1d9a0b3d8be1`.
[Raw log](evidence/cuda132/run1/test.log) SHA256
`ac409cde76d37164935c6c3c5d079834b9f6ca5e64fc54a64d14b10720f13be7`.
Exact1 PASS, quiet/no-apps PRE/RELEASE/DRAIN and unchanged cache receipts.
No production route, Fixed inference or SM120 change; Fast-win count unchanged.
