# Ada half Triad NN: four aligned S3 finalists

2026-09-08, RTX6000 Ada / CUDA13.2. Candidate discovery, not dispatcher
admission. Existing Fixed S3 math reused unchanged; no production/SM120 edits.
All logical A/B/C pointers are 256-byte aligned (128 half-element guards).
This supersedes the alignment-sensitive NN7 performance interpretation.

| NN cell | Candidate us | Fast us | Paired candidate/Fast p50 | Candidate/forced TC128 p50 |
| --- | ---: | ---: | ---: | ---: |
| F16 d768-out | 38.0–38.5 | 47.5–49.3 | .778–.800 | .686–.687 |
| BF16 d768-out | 38.0–38.4 | 48.0–49.7 | .771–.792 | .687 |
| F16 Prism | 59.7–60.3 | 76–77 | .779–.791 | .702–.706 |
| BF16 Prism | 59.4–60.5 | 75.7–77.9 | .774–.787 | .701–.705 |

Ranges span eager/graph and ABBA/BAAB; lower ratios are better. All four
advance against BOTH separately paired comparators, with every p95 < .803
against Fast. Approximately20–23% less time than aligned cuBLAS Fast.
Current comparator is forced TC128, **not a fresh public-AUTO qualification**.

- Exact `ada_half_nn_fixed_s3_aligned_four_cell_confirmation_once7`: one test
  PASS, 8 resource / 32 screen / 4 decision records. Seven windows,20 GEMMs
  per observation. Root independently replayed224 brackets/896 observations
  and all64 p50/p95 values from raw, PASS.
- Candidate/current raw bits and guards checked eager/graph and during every
  timed observation; Fast has its own reference bits. No separate bit-record
  schema in this test. No whole-model/cross-architecture determinism claim.
- Candidate188 registers, local0, dynamic shared98304, occupancy1.
  Private cache stable; PRE/DRAIN quiet. Not a cold-cache result.
- Main SHA256 `667510c2e99547064e58b05eb1937cffd7cb1357e92215983d917043bb100938`.
  Binary `f05a628a0e50acbd3edcb465779e8fb20715679983e1c02696d3f3b38fcadc21`.
  Raw `9b1f6d35b6a5eaf20586a3ca395f84d747e8bba1e2b6c582fcf158aacd64ced1`.
- [Raw](evidence/once7-cuda132/test.log),
  [command/receipt](evidence/once7-cuda132/command.json).
  No other shape, toolkit or fresh5090 result is inferred.
