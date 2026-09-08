# Half TN M64N128/BK64/S2 compact: six valid losses

Ada / CUDA13.2, 2026-09-08. New output-column-wide candidate preserves the
TC64 ascending K16 MMA chain and F32 accumulation epilogue. It reuses A across
more output columns, but the measured implementation is slower in every cell.

| Cell | Dtype | Candidate/current p50 | Candidate/Fast p50 |
| --- | --- | ---: | ---: |
| d768-in | F16 | 1.0403–1.0439 | 1.1949–1.2394 |
| d768-in | BF16 | 1.0387–1.0411 | 1.1924–1.2347 |
| d768-out | F16 | 1.0152–1.0279 | 1.3212–1.3741 |
| d768-out | BF16 | 1.0165–1.0265 | 1.3291–1.3710 |
| Prism | F16 | 1.2526–1.2743 | 1.8037–1.8827 |
| Prism | BF16 | 1.2523–1.2681 | 1.8047–1.8862 |

Current is forced TC64, not a fresh public AUTO qualification. Fast is native
half cuBLAS CUBLAS_COMPUTE_32F, not Pedantic. Ratios cover both paired orders
and eager/graph paths. All six cells STOP; do not repeat these unchanged
candidates or remove incumbent routes for other devices/cells.

One focused build and two exact test processes PASS. Logical pointers are
256-byte aligned. All six resource gates pass: 256 threads, 103 registers,
49152 static shared bytes, zero dynamic/local bytes, occupancy 2 CTAs/SM.
Candidate/current eager and graph bits match, with independent 20-operation
accumulation oracles and vendor self-repeat checks. Counts: 6 resource,
48 bit, 48 paired screen and 12 comparator-decision records. Once7,
20 GEMMs/observation; root independently replayed 336 brackets / 1344
observations and both quantiles. Quiet/drain and stable-cache checks pass.
Stable cache is not cold-cache qualification.

Exact bare test names:

- `ada_half_tn_m64n128_bk64_s2_compact_d768_in_discovery_once7`
- `ada_half_tn_m64n128_bk64_s2_compact_d768_out_and_prism_discovery_once7`

Measured typed harness SHA256:
`ee3d984aaed5dfcb4ec20cf4f78ceec1ad453bb2ce4300d6d0bbfc7436de5da5`.
Source helper SHA256:
`a307af837767982142f7dbeeecd4f5d8fdf5a665b0bbe85b7b08eb5faf13a24e`.
Binary SHA256:
`8910ae253e5fc7272d98842a287a9f3570c1ea9c97901b44781ebd95352226cc`.

[d768-in raw](evidence/d768-in-once7/test.log), SHA256
`18a83f0d71ce4e49ac3a289385db658921abbf2d2a360948bc714f266b617426`.
[d768-out and Prism raw](evidence/d768-out-prism-once7/test.log), SHA256
`ffa32504a4808ee11acc904eeca30978a2edb2024698b4fd45a2b62da26216b1`.
No production, Fixed inference, SM120, or toolkit admission changes.
