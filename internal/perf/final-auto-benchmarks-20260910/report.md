# Final production AUTO benchmark adapters

This packet validates the benchmark tools. It is not yet the full release
performance matrix, and none of its three-window smoke timings is a final
cuBLAS speed claim. Production kernel/dispatch bytes in these runs are from
`732c11462dd9a47376e6d15b36040ca69bde3e34`; each raw packet separately pins
the benchmark source, complete source manifest, binary and runner.

## Triad adapter

The new ignored test measures actual AUTO for66 cells and81 explicit cuBLAS
comparator views. Exact F32 gets separate Fast-TF32 and Pedantic views;
deterministic TF32 gets Fast-TF32; native half inputs get COMPUTE_32F Fast.
Eager/whole graph and both mirrored orders produce324 final pair records
plus one completion record at the default21 windows per order.

The frozen RED source failed compilation on Ada/CUDA13.0 with the17 expected
undefined new helper/type errors. This is compile-time RED, not executed
behavioral assertions. Raw output is under `red/`.

The first candidate compiled on Ada/CUDA13.2, passed4 native tests and a
four-cell/five-view smoke. Independent review then found two Important
problems: vendor work shared the physical-holder's exclusive AUTO context,
and vendor base alignment had not been explicitly checked. Preserve
`triad-ada-smoke/` as diagnostic-only evidence; it is not an approved final
benchmark protocol.

The fixed snapshot is SHA-256
`de4b689bebdb35d0e2dd201cbdd774ab7b89b6896e3026d0da92e822566c7f45`.
AUTO and vendor now use separate contexts, each event window uses its own
arm's stream, and the actual three vendor pointers must be256-byte aligned
before seeding or graph capture. Pair records include that admitted alignment.

Fresh Ada/CUDA13.2 results for the fixed snapshot:

- compile: pass,5.17s;
- focused native tests:5/5 pass;
- strict GPU smoke: pass,153.66s including context initialization;
- four cells/five comparator views/two paths/two orders:20 pair records plus
  one completion, three samples per order;
- all raw sample ratios and p50/p95 values independently replay;
- completion SHA-256 matches the exact20 pair lines:
  `9732cc9d2cdae021030f442f2b4f225d1338a4323f215c2c8752d0be378d57c1`;
- independent scoped re-review: both findings addressed, no new blocking issue.

The smoke covers exact-F32 NN with both vendor modes, deterministic-TF32 NT,
native-F16 NN, and BF16 TN with F32 output. It preserves the recorded
TN reset-before-event and repeated-beta1-within-window semantics. This
descriptive performance tool does not replace the independent numerical
qualification receipts. Physical `eager_graph_equal` is launch-metadata
equality, not an assertion about downloaded output bits.

## Inference adapter

Work is separate and still under review/runtime verification. Its frozen RED
`1e655261df26912501becce5729f7be906378a9e52c3b302bdec67130860f9e8`
failed on Ada/CUDA13.0 with exactly5 expected undefined-symbol errors for the
new default, count and ratio helpers. Raw output is in `inference-red/`.
No full280-record Inference matrix is claimed by this packet yet.
