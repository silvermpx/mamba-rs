# Ada half TN regpipe+vec2 siblings, 2026-09-08

One new retained-best cell on RTX6000Ada/CUDA13.2; **no new cuBLAS Fast
win**. The unchanged TC64/BK64/S2 regpipe body with vector-pair F32 epilogue
was screened on d768-out and Prism for both native half input dtypes.

| Cell | Dtype | Candidate/retained p50 | Worst p95 | Candidate/Fast p50 | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| d768-out | F16 | .98174–.98576 | .99124 | 1.06044–1.10972 | stop; one p95 misses |
| d768-out | BF16 | .98166–.98412 | .98758 | 1.06798–1.11093 | **advance retained best** |
| Prism | F16 | .99229–.99766 | .99972 | 1.36267–1.40352 | stop |
| Prism | BF16 | .99365–.99819 | 1.00413 | 1.35767–1.40199 | stop |

The BF16 d768-out result is a robust1.6–1.8% median time reduction against
the retained compact BK64/S2 comparator across eager/graph x ABBA/BAAB;
all four p95 values are below.99. It remains6.8–11.1% slower than Fast at
the median. The other three unchanged candidates must not be retried.

## Verification and identities

Candidate and retained both match CurrentTC64 output bits for two eager and
two graph repeats in every cell. Their20-operation accumulated outputs also
match CurrentTC64; Fast is checked against its own finite, nonzero repeatable
output. All five logical pointers are256B aligned and guards/input immutability
pass. Candidate graphs are single-node grid288 for d768-out and186 for Prism.

Resources pass for both dtypes and both compared kernels: candidate125regs,
retained123regs, each local0/static32768/dynamic0/block128/occupancy3. CUDA
bodies are unchanged: vec2 helper `e0589f17...`, parent regpipe `f3e66016...`,
compact `a8edb14a...`; composed candidate and retained sources are
`5c5f4b289698a88b7f3c6d4e460ea5182cc0ccbb17417d6651eed7ff7f95475a`
and `d6f98ddf27278b927ac0f79b7fbb4036be03b1478acb32989e13ace34696d4fb`.

- Main test SHA256:
  `8132d81f677359806fbce010c55b56ee0667529cecb2a8bf6607f9fad0f970a5`.
- Plan helper SHA256:
  `f2df60014f7287996511dbe149c6a600e5749f128a42f2437239e8acec4fe7be`.
- Binary SHA256:
  `37332da0ec801560decd82456f72146a75a56742c9d3284adb49345be7e334c8`.
- [Authoritative raw log](evidence/cuda132/run2-repaired/test.log), SHA256
  `e0aa84db34d8c3d5947b9053fcda1b049b671289aa252cdf864e6eec028c7900`.

Root independently replayed32 screen rows,224 brackets and896 event
observations, including order-dependent candidate/comparator placement and
nearest-rank p50/p95. Cache hashes are unchanged. PRE, RELEASE and DRAIN are
quiet/no-apps. Exact test passes in88.79s.

The preserved run1 is explicitly non-authoritative: an independent review
found that it omitted the retained compact resource gate. The repaired source
adds that gate, rebuilds, and reruns once. Run1 is a harness-omission receipt,
not a second performance sample or kernel loss.

No production selector, Fixed inference, lower toolkit or SM120 route changed.
The accepted BF16 cell remains test-only until the joint Triad integration and
supported-toolkit qualification batch.
