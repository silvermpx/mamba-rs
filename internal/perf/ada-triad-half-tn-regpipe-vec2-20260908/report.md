# Ada half TN vector-pair epilogue — 2026-09-08

Valid CUDA13.2 discovery result, not a public AUTO admission. Both F16/BF16
`d768-in (2048,768,3072)` preserve CurrentTC64 output bits and guards, including
two eager and two graph repetitions. Candidate output/dW is F32.

| Dtype | Comparator | Eager p50 | Graph p50 | Worst graph p95 |
| --- | --- | ---: | ---: | ---: |
| F16 | CurrentTC64 | .8412–.8449 | .8420–.8433 | .8479 |
| BF16 | CurrentTC64 | .8393–.8398 | .8404–.8433 | .8466 |
| F16 | native-half cuBLAS Fast | .9643–.9654 | 1.0014–1.0022 | 1.0068 |
| BF16 | native-half cuBLAS Fast | .9612–.9657 | .9988–.9997 | 1.0035 |

Ratios are candidate/comparator; lower is better. Graph is near parity, not a
strict Fast winner (all four p50/p95 strata must be below .99). Candidate was
not paired against the retained compact/regpipe variants: no incremental-best
claim. Do not rerun unchanged or replace the retained family on this evidence.

125 registers, zero local, 32,768 static shared, zero dynamic shared, block128,
three resident CTAs/SM. The change only packs adjacent F32 epilogue RMWs into
float2 operations; per-output K16 MMA ordering is unchanged. Parent SASS had
separate adjacent 32-bit LDG/STG instructions, so this was a distinct hypothesis.

## Reproducibility

- Exact measured source is the commit containing this report; live experiments
  may advance separately. Main `gemm_bi_typed_parity.rs`:
  `b7fd0276ead84fa16c3134d64d4fbcd82db32393bf771cf71e9f258cdfd79010`.
- Helper `triad_half_tn_vec2_epilogue_source.rs`:
  `e0589f179020be2e1841095a831097f7c1dabf21bd7a2bf42d5358328acae9b2`.
- CUDA source: `5c5f4b289698a88b7f3c6d4e460ea5182cc0ccbb17417d6651eed7ff7f95475a`.
- Binary: `249e685526dc9d987e83408c6ba0cad2d62f7c9677f5880a72ceda9518234824`.
- [Raw log](evidence/cuda132/run1/test.log):
  `c7b83697990d3c0b5866b5e5cac7b98e3a56336ac732b9797f0d07aa9e9ddd57`.
- Exact bare test:
  `ada_half_tn_tc64_bk64_s2_regpipe_vec2_d768_in_vs_current_and_fast_discovery_once7`.
- Root independently replayed all112 brackets/448 event observations and
  nearest-rank p50/p95. Two resource,16 bit,16 screen and4 decision records.
  Twenty GEMMs/observation;256B logical alignment; whole graphs.
- Private artifact cache unchanged; pre/drain quiet, release had no compute
  apps but9% residual GPU utilization, so release is not labelled quiet.
- No CUDA12.8/13.0, cold-cache, SM120, dispatcher or release claim.
