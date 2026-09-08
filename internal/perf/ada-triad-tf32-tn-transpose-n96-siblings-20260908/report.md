# TF32 TN transpose + N96: two more AUTO improvements, not Fast wins

Ada / CUDA13.2. The unchanged RNA-N96 plus raw-A-transpose pipeline was
screened on two previously unmeasured shapes, without repeating d768-in.

| Shape | Candidate/AUTO p50 | Worst AUTO p95 | Candidate/Fast graph p50 | Decision |
| --- | ---: | ---: | ---: | --- |
| d768-out | .5933–.6047 | .6104 | 1.1977–1.2016 | Keep AUTO gain; not Fast |
| Prism | .8337–.8522 | .8587 | 1.9959–1.9985 | Keep AUTO gain; not Fast |

d768-out's eager/Fast ratios .9429–.9520 do not override its graph loss.
Prism also loses eager/Fast (1.5825–1.6156). Both tests pass exact target
AUTO bits plus the focused tail/alpha/exception/K0/input/guard checks.
Transpose scratch is checked word-for-word including zero padding before
timing. Both transpose and GEMM are inside each measured pipeline.

Actual AUTO identity: d768-out M64N64/S2 grid288/block128/shared36864;
Prism M128N64/S3 grid93/block256/shared79872. Candidate N96 GEMM grids96/63,
block256,128 registers,local/static0,dynamic86016,occupancy1. Transpose uses
4224 static shared bytes and0 local bytes. Prism A stride is padded to4624.

One build, two exact PASS processes:
`cuda_suite::ada_tf32_tn_d768_out_transpose_rna_n96_discovery_once7` and
`cuda_suite::ada_tf32_tn_prism_transpose_rna_n96_discovery_once7`.
Root replayed112 paired brackets /448 observations, p50/p95 and decisions.
Once7,1 GEMM/observation,256B actual pointer checks, eager/graph ABBA/BAAB,
explicit Fast TF32. Private cache unchanged and no competing application
in PRE/RELEASE/DRAIN. Native combined harness53/53 passed before GPU work.

Main SHA256: `e5f1a811f7a49f43425e6885caaeedcc559df3775cd5225820016da9aacc3d0a`.
Unchanged helper SHA256: `f969041b0b149dcfef4da65465df6ee47397299124c6658e56b4fa3a3a77773d`.
Binary SHA256: `8f7c463d7b522632764137fb332396aa797acf82814de1ed89a1341213cbb2fd`.
[Out raw](evidence/cuda132/run1/test.log), SHA256
`76e2bb2ba3a351c6f539331297b01e57036292a94a20e9fd714b78c8f8307c4c`.
[Prism raw](evidence/cuda132/run2/test.log), SHA256
`f97a99bb3c923a99c28bc66ba012a697aa4b534701652e9e166dc1f97e9db971`.
Frozen sources under `evidence/source/`. No production admission or full
toolkit qualification; the next distinct hypothesis is direct TN staging
inside N96, removing global transpose, not a retry of these valid losses.
