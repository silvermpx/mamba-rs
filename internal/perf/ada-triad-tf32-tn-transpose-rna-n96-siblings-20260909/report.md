# Ada Triad TF32 TN transpose-RNA N96 sibling winners — 2026-09-09

Outcome: **two new retained-best wins; the A-only RNA recipe now wins all
three large TF32 TN cells.**

The unchanged test-only d768-out winner moves deterministic A RNA conversion
from repeated N96 GEMM loads into the preceding transpose store. One CUDA13.2
runtime compiled candidate and retained once, then qualified d768-in
`(2048,768,3072)` and canonical `prism_in_proj` `(4621,384,1928)` sequentially.
Each arm remains a two-node transpose+GEMM pipeline.

Candidate/retained resources are127/128 GEMM registers, local0, dynamic shared
86,016 and occupancy1; both transpose kernels use26 registers, local0 and
static shared4,224. Shared finite/exceptional full-tile, tail and K0 gates pass;
each target also passes exact eager/graph repeats, scratch oracle, graph ABI and
guards.

Candidate/retained once7 p50/p95:

| Cell | eager ABBA | eager BAAB | graph ABBA | graph BAAB |
| --- | ---: | ---: | ---: | ---: |
| d768-in | .927027/.932170 | .927083/.932249 | .926431/.928962 | .926431/.928962 |
| canonical Prism | .927030/.943108 | .922936/.935518 | .925948/.931736 | .925334/.928276 |

Both cells pass the strict retained gate. Candidate/cuBLAS Fast p50 is
`1.2610–1.2677` for d768-in and `1.8776–1.8854` for canonical Prism, so neither
is a Fast win. Combined with the separately qualified d768-out result
(`.92965–.93000` p50), the same deterministic recipe is the retained champion
for all three large TF32 TN cells and should be integrated once whole-Triad
discovery closes.

Frozen identities:

- sibling harness SHA256: `16a0e19d75ff7ac65555d122c8b72af9328c2719dba83ccd0b8aafafad847867`
- unchanged helper SHA256: `5f0d37ecad9ddabb5f7728e34b1920c41f8049bae37b80448ac0361fb0a713d3`
- candidate transformed source SHA256: `3ee1e8986c3d1fb998aef408f14add2203d71bbd6f8521f08500ac972f2d2f65`
- candidate PTX SHA256: `57808541ffcf29b65c0bdecb14c85e9a6e185f84da39167c4c5d3cf8df313ee6`
- retained transformed source SHA256: `303801f92790331a1e4ffa3eb26036f7f2148da30fafa14ce50833dfe3117fd3`
- retained PTX SHA256: `273722a7022e615096fa61f2740f0a916d518517bec1dcbe234c82b1ecced963`
- [raw qualification summary](raw.log)
