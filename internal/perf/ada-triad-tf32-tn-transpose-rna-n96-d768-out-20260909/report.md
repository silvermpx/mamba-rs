# Ada Triad TF32 TN transpose-RNA N96 d768-out winner — 2026-09-09

Outcome: **new retained-best; 6.5–7.0% lower whole-pipeline time.**

For `d768_out_proj` `(M,K,N)=(2048,1536,768)`, the test-only candidate moves
the deterministic add-half-ULP TF32 conversion of A from every GEMM load into
the preceding transpose store. The retained comparator is the frozen raw
transpose plus N96 GEMM. Both arms remain two-node pipelines; B conversion,
HMMA issue order and output arithmetic are unchanged.

Exact raw bits pass for finite and exceptional full tiles, a generic tail, K0,
the target, eager repeats, graph repeats and guarded buffers. The independent
scratch oracle verifies candidate RNA-transposed A separately from the retained
raw-transposed A. CUDA 13.2 resources are candidate/retained GEMM 127/128
registers, local0, dynamic shared86,016 and occupancy1. Both transpose kernels
use26 registers, local0 and static shared4,224.

Candidate/retained ratios:

| Cohort | eager ABBA | eager BAAB | graph ABBA | graph BAAB |
| --- | ---: | ---: | ---: | ---: |
| once3 p50 / p95 | .930000 / .934673 | .930000 / .931250 | .934343 / .934343 | .929293 / .939394 |
| once7 p50 / p95 | .930000 / .933501 | .930000 / .935000 | .929648 / .934343 | .929648 / .934673 |

Candidate/cuBLAS Fast once7 p50/p95 is respectively
`1.088316/1.110655`, `1.077675/1.092311`, `1.079434/1.114878`, and
`1.091445/1.114248`. This is a robust retained win, not yet a Fast win.
Keep it in the whole-Triad integration shortlist; production dispatch remains
unchanged until discovery closes.

Frozen identities:

- helper SHA256: `5f0d37ecad9ddabb5f7728e34b1920c41f8049bae37b80448ac0361fb0a713d3`
- harness SHA256: `0f3d525521ccba30115284d8317d9a3b6436cea4c075daec947aab80b9ff5c11`
- candidate transformed source SHA256: `3ee1e8986c3d1fb998aef408f14add2203d71bbd6f8521f08500ac972f2d2f65`
- candidate PTX SHA256: `57808541ffcf29b65c0bdecb14c85e9a6e185f84da39167c4c5d3cf8df313ee6`
- retained transformed source SHA256: `303801f92790331a1e4ffa3eb26036f7f2148da30fafa14ce50833dfe3117fd3`
- retained PTX SHA256: `273722a7022e615096fa61f2740f0a916d518517bec1dcbe234c82b1ecced963`
- [raw qualification summary](raw.log)
