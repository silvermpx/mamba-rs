# Ada TF32 NN N96 B-ldmatrix, 2026-09-08

Valid measured loser; **stop without retry**. Replacing the retained N96
kernel's two scalar B words with `ldmatrix.m8n8.x2.shared.b16` plus a warp
shuffle preserves exact bits but regresses both screened cells.

| Cell | Candidate / retained p50 | Worst p95 | Candidate / Fast p50 | Decision |
| --- | ---: | ---: | ---: | --- |
| d768-out | 1.09521–1.09887 | 1.09928 | 1.10855–1.11193 | stop |
| Prism | 1.08906–1.09049 | 1.09243 | 1.10480–1.11898 | stop |

The retained scalar-B N96 body remains selected. This experiment changes no
dispatcher route and adds no strict Fast win.

## Verification and frozen identity

Candidate and retained pass target/tail/exceptional/K0 exact bits, independent
positive-zero K0 oracle, guards, 256B alignment, eager/graph repeats and graph
identity. Both compile to 124 registers, zero local/static shared, 86,016B
dynamic shared, 256 threads and occupancy1. Grids are128 and777.

The paired once7 screen uses 20 GEMMs/observation and eager/graph x ABBA/BAAB
against both retained N96 and explicit native cuBLAS Fast. PRE/RELEASE/DRAIN
are valid and cache hashes are unchanged.

- Main test SHA256: `cc20f710e69639815e2508e9588eb45563dbbe426543175427165cce63b7696e`.
- B-ldmatrix helper SHA256: `bfe91e98b9f81af41b5bc0ce0a4b3aae03f091e7021c6d9770c8396657b19d10`.
- Test binary SHA256: `7e3ce77e209529c2162d1600912b1e11db9a0c4a28383b2228460ed867f593df`.
- [Authoritative raw log](evidence/cuda132/attempt2/test.log), SHA256
  `d5f965bdea539783583b663a6475b643091db475c06ea3a0c92e6c307687ef09`.

Attempt1 is retained as a non-authoritative invocation failure: the discovery
environment flag was omitted, so no NVRTC candidate compile or timing ran.

