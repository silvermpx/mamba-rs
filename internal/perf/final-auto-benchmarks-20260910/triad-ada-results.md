# RTX 6000 Ada production AUTO Triad, CUDA 13.2

The full paired benchmark completed successfully:66 operation/precision/shape
cells,81 explicit comparator views,324 pair rows,21 windows in each mirrored
order. Raw packet: `triad-ada-full/`. The324 rows and completion metadata
independently replay with `verify-triad.jq`; all actual vendor base pointers
were admitted at256-byte alignment. The completion `cohort_digest` matches
the exact pair-record bytes:
`9b29337fe0e61cf943c3ba709986cf9ca2d608c9105a7fae27e4b1ca9c3998b2`.

Measured production source:
`732c11462dd9a47376e6d15b36040ca69bde3e34`.
Measured adapter source SHA-256:
`de4b689bebdb35d0e2dd201cbdd774ab7b89b6896e3026d0da92e822566c7f45`.
The run finished in624.56s, including host/quiet-gate overhead. This duration
is not kernel execution time. The device, compiler, artifact, runner and
physical route provenance remain in the raw packet.

## Aggregate comparison

Speedup is cuBLAS time divided by AUTO time: above1 means AUTO is faster.
For each cell/path, combine its42 recorded paired AUTO/vendor ratios and
take the lower empirical median; invert that value. Aggregate speedup is
the equally weighted geometric mean over those cells. Counts below are
descriptive median wins, not an admission/confidence test.

| AUTO precision | cuBLAS comparator | Cells | Eager speedup | Graph speedup | Median wins, eager / graph |
|---|---|---:|---:|---:|---:|
| BF16 | Fast COMPUTE_32F |15|0.890×|0.869×|7 /7|
| F16 | Fast COMPUTE_32F |15|0.881×|0.851×|6 /6|
| Deterministic TF32 | Fast TF32 |21|0.718×|0.709×|0 /1|
| Exact F32 | Fast TF32 |15|0.422×|0.422×|0 /0|
| Exact F32 | Pedantic F32 |15|0.775×|0.772×|4 /4|

The single TF32 graph median win is only1.0047× and should be treated as
parity, not a robust Fast win. The strongest F16 cell is1.988× eager /
1.971× graph. These measurements do not support a claim that the whole
Triad is faster than cuBLAS Fast. Release assembly accepts retained kernels
that improve the previous AUTO; that historical improvement is a different
comparison and is not inferred from this table.

Exact F32 versus Fast TF32 also compares different input precision
semantics. Its Pedantic F32 comparison is reported separately. Half-input
TN produces F32 outputs. TN initializes C before each event window and
measures repeated beta1 accumulation inside it; NN/NT measure beta0
overwrites. Whole graphs include all selected physical launches.

This is descriptive performance evidence. The benchmark's physical
`eager_graph_equal` field compares launch metadata; independent qualification
and model/repeat tests provide numerical and determinism evidence.

The pending current-toolkit cohort change is SM120-only. This Ada packet
keeps its actual measured source SHA; it must not be relabeled as a later
commit. If subsequent changes affect Ada selection or arithmetic, affected
cells need replacement measurements.
