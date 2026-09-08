# Ada half NT Fixed-S3 B-XOR screen

CUDA 13.2 ran the single exact ignored test
`ada_half_nt_fixed_s3_bxor_d768_out_vs_current_and_fast_discovery_once7`.
The test passed with 2 resource, 16 bit, 16 timing-screen, and 4 decision
records. Source SHA-256 was `c1fd1b86f85d6aee259d76b9c91e5c1c98020d5780552003c63b4f30815e7af9`,
helper SHA-256 was `2c41646460a23f92f43417b08b4389bc306386b26d521282c904881b2e1c2fc0`,
and binary SHA-256 was `d692f4819d4200d84dbe5edd09e4222e30ea14962a52cd44648c0049024f8274`.

Both F16 and BF16 matched the forced TC64 reference bits in eager and graph
repeats. The candidate used 167 registers, 0 local bytes, 0 static shared
bytes, 98,304 dynamic shared bytes, 256 threads, and occupancy 1.

Against forced TC64, the candidate measured about 41.4--42.1 us versus
56.3--56.9 us. All four F16 median ratios were 0.7354--0.7394 and all four
BF16 median ratios were 0.7355--0.7397, so both advance against current.

It did not beat native-half Fast. F16 was approximately tied in eager
(median ratios 0.9948--0.9956) but regressed in graph (1.0202--1.0227), with
every p95 above 1.0. BF16 regressed in all four strata (median ratios
1.0314--1.0562). These are valid stop decisions against Fast, not a Fast
winner.

The exact test log is `evidence/cuda132/run1/test.log` (SHA-256
`a54034697c2fc3f51952d8bc6f5e1157f2c6bbe0c44f988327b69e7430f3191f`).
Root independently replayed all112 brackets /448 observations and both
quantiles; native source tests5/5 passed. Each observation is20 GEMMs with
256B-aligned logical pointers. The command receipt records a stable private
cache and no competing application in PRE/RELEASE/DRAIN. PRE and five-second
DRAIN are quiet; immediate RELEASE still shows9% utilization after the test.
This is target discovery against forced TC64, not public AUTO admission,
whole-Triad qualification or a claim for CUDA12.8/13.0 or RTX5090.
Exact measured source snapshots are retained in `evidence/source/`.

Root independently replayed 112 timing brackets / 448 observations and
verified the 16 bit, 2 resource, and 4 decision records, including all p50 and
p95 ratios plus the cache and quiet receipts.
