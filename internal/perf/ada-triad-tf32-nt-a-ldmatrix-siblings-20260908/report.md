# Ada TF32 NT A-only ldmatrix siblings, 2026-09-08

Two new retained-best Triad cells on RTX6000Ada/CUDA13.2; neither is a
cuBLAS Fast win. The unchanged d768-in A-only ldmatrix body was screened on
d768-out and Prism against the actual public AUTO compact8 route and explicit
native cuBLAS `CUBLAS_COMPUTE_32F_FAST_TF32`.

| Cell | Candidate / actual AUTO p50 | Worst p95 | Candidate / Fast p50 | Decision |
| --- | ---: | ---: | ---: | --- |
| d768-out | .90317–.90558 | .90638 | 1.39618–1.43910 | **retain: 9.4–9.7% faster than AUTO** |
| Prism | .89950–.90100 | .90236 | 1.84705–1.88401 | **retain: 9.9–10.0% faster than AUTO** |

All four eager/graph x ABBA/BAAB p50 and p95 strata beat actual AUTO by the
strict `.99` gate. Fast remains ahead, so `fast_win=false` is recorded
separately and these results do not increase the strict Fast-win count.

## Verification and frozen identity

Both isolated tests pass target/tail/exceptional/K0 exact-bit checks, input
immutability, red zones, 256-byte logical pointer alignment, graph identity and
repeat checks. The compiled candidate is 98 registers, zero local/static
shared, 49,152B dynamic shared, 256 threads and occupancy2. Grids are384 for
d768-out `(2048,1536,768)` and222 for Prism `(4621,384,1928)`.

The paired screens use seven windows, 20 complete GEMMs per observation and
both ABBA/BAAB orders for eager and graph paths. PRE/RELEASE/DRAIN show the
correct idle GPU with no compute applications; private cache hashes are stable.

- Main test SHA256: `5ce3c4222d041090a80d2bb9101d52ff0b382ee38342865e62b5a95b7569b456`.
- Unchanged A-only helper SHA256: `c853f8b824d9a1717f1594dd063457cc0b6ba2713f57884adaa58e6e795d3cfa`.
- Composed candidate SHA256: `b66739ed9d6433d157b3a96ca9bad6448d5b0e24e141ff0bde095e5161ead5f8`.
- CUDA body SHA256: `0c3907f60f4f6f47abafdc3d476bb1956172c443acfd10b11481fd1121117b76`.
- Test binary SHA256: `cc102367d4b336ee8ea810ec60a12d9928d28e66b93fbb96de0b4024d8fa4728`.
- [d768-out raw log](evidence/cuda132/d768-out/test.log), SHA256
  `401a09fe9a7d1bda0d5dcf31c2c9f3f8a19795de4b1271d6c6ec6da367716c4a`.
- [Prism raw log](evidence/cuda132/prism/test.log), SHA256
  `80e631db46c3ede70c379cb11cc3e55a0d308abfc529ea0847a3bf8915ef6aca`.

These are frozen discovery winners, not dispatcher promotion. Preserve both
cells for the joint integration and supported-toolkit qualification batch.

