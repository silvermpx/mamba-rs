# Half NT BK32/S3 d768-out: stop

CUDA13.2 / RTX6000 Ada, `(2048,1536,768)`. Same TC64 output geometry and
ascending K16 MMA association; BK64/S2 replaced by BK32/S3. Resources pass:
100 registers,0 local,30720 static shared bytes,128 threads,3 resident CTAs.
F16/BF16 candidate/current eager2+graph2 bits pass, including input/output guards.

It loses to native-half cuBLAS Fast: F16 paired p50 1.543–1.621,
BF16 1.595–1.664; worst p95 1.629/1.690. Both stop; no production admission.
This historical screen used guard8 half elements (pointer offset16 bytes),
before the NN diagnostic identified the alignment-sensitive denominator.
It is not a final production-aligned NT performance matrix.

Exact `ada_half_nt_d768_out_bk32_s3_vs_current_and_fast_discovery_once7` passes:
2 resource/16 bit/8 screen/2 decision records. Root replayed all56 brackets,
224 observations, p50/p95. Each observation is20 GEMMs; output beta0 is
overwritten each call. Quiet PRE/DRAIN, stable private cache.
Measured main `be132a7a6cb93a57f98869ab73034108c4d414a8716bb5706dba82d4e5fd3de1`;
helper `5937918033882f36f8fcb97d3cf29e29982486b6d83274990ef0705fe434a546`.
[Raw](evidence/once7-cuda132/test.log) SHA256
`c5fe501d4ed53937b9a4493374a774db95bad2b58d8765dd9d2ad440ba21b847`.
