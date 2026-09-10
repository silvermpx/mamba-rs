# Production AUTO Inference on RTX 6000 Ada and RTX 5090

Both full CUDA13.2 runs passed:280 pair records plus completion per board,
seven precision/comparator rows, five shapes, bias off/on, eager/whole
graph, mirrored order,21 windows per order. The root independently replayed
all raw timing ratios, quantiles, counts, precision metadata, GPU UUID,
tuning revision45, output tolerances and repeat/graph bit-check fields with
`verify-inference.jq`. Original logs remain in `inference-ada-full/` and
`inference-sm120-full/`.

Production source: `732c11462dd9a47376e6d15b36040ca69bde3e34`.
Adapter source SHA-256:
`c92442da094de530e5388960b343f82eec163766c6e5ceafd4043f0d1d17de73`.
These are fresh final AUTO measurements, not new kernel implementations.

## Aggregate comparison

Values are cuBLAS/AUTO speedups: above1 means AUTO is faster. Each
shape/bias/path pools its42 paired ratios, takes the lower empirical median,
then inverts it. A precision aggregate is the equally weighted geometric
mean of its ten shape/bias speedups. These descriptive medians are not a
confidence-based admission claim. Both bias states count independently.

| Input → output | cuBLAS comparator | Ada eager | Ada graph | RTX5090 eager | RTX5090 graph |
|---|---|---:|---:|---:|---:|
| BF16 → BF16 | Fast COMPUTE_32F |1.187×|1.185×|1.243×|1.235×|
| F16 → F16 | Fast COMPUTE_32F |1.167×|1.165×|1.234×|1.238×|
| BF16 → F32 | Fast COMPUTE_32F |0.828×|0.825×|1.288×|1.276×|
| F16 → F32 | Fast COMPUTE_32F |0.815×|0.812×|1.278×|1.270×|
| Deterministic TF32 → F32 | Fast TF32 |0.900×|0.900×|1.132×|1.126×|
| Exact F32 → F32 | Fast TF32 |0.462×|0.459×|0.772×|0.776×|
| Exact F32 → F32 | Pedantic F32 |1.004×|1.002×|1.083×|1.074×|

RTX5090: all ten shape/bias medians win for each of the four half-input
rows in both paths; TF32 wins eight of ten. Exact F32 wins six of ten
against Pedantic and none against Fast TF32.

Ada: native BF16/F16 each win eight of ten; mixed half→F32 and TF32 each
win two of ten. Exact F32 wins eight medians against Pedantic, but the
overall1.002–1.004× aggregate is parity, not a meaningful overall win.
Exact F32 does not beat Fast TF32 on these ten cases. Different precision
contracts must not be conflated.

The AUTO graph inventory identifies the exact selected physical symbol,
not a forced candidate. Whole vendor graphs may have a different node
count. Bias comparison includes the vendor's required bias preparation.
No NVIDIA_TF32_OVERRIDE was imposed: cuBLAS compute type is explicit.

These numbers are kernel/graph event timings, not whole-model training or
inference throughput. The pending Triad-only SM120 cohort insertion does
not change these Inference kernel or selector bytes. Keep each packet's
actual measured SHA; do not relabel it as a later release commit.
