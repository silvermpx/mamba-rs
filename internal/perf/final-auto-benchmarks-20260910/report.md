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

Its frozen RED
`1e655261df26912501becce5729f7be906378a9e52c3b302bdec67130860f9e8`
failed on Ada/CUDA13.0 with exactly5 expected undefined-symbol errors for the
new default, count and ratio helpers. Raw output is in `inference-red/`.
No full280-record Inference matrix is claimed by this packet yet.

The adapter measures seven precision/comparator rows, five hot shapes, both
bias states, eager and whole-graph execution, and both mirrored orders:
280 pair records at21 windows per order. Exact F32 has separate Fast-TF32
and Pedantic comparisons. The TF32 row uses explicit Fast-TF32; the four
native/mixed half rows use COMPUTE_32F with their actual output types.

The first candidate `ef2b82d2` passed two native tests and a56-record Ada
smoke covering all seven rows, `hot_d`, both bias states, both paths and
orders. Static review found three receipt-integrity issues: symbol-prefix
collisions, UUID discovery not tied to the visible CUDA ordinal, and absent
tuning revision metadata. Keep `inference-ada-smoke/` as the original
pre-fix numerical coverage, not a final-approved release table.

Fixed source SHA-256:
`c92442da094de530e5388960b343f82eec163766c6e5ceafd4043f0d1d17de73`.
The complete serialized symbol token is now matched exactly, the driver
provides the visible CUDA ordinal0 UUID, and both timing/completion records
include tuning revision45. Independent scoped review closed allthree
findings with no new blocking issue.

Fresh CUDA13.2 / RTX5090 verification in `inference-sm120-smoke-fix1/`:

- compile: pass,11.78s;
- four focused native tests:4/4 pass;
- TF32/hot_d/bias0 focused eager/graph smoke: pass,70.83s including startup;
- four pair records plus completion, three windows per order;
- full sample/ratio/quantile and metadata replay: true, revision45 and
  actual GPU UUID pinned, repeat/graph/output-storage checks passed.

The launch in `inference-sm120-aborted-bad-sha/` was stopped during build
because the command accidentally supplied a placeholder production SHA.
It contains no completed GPU measurement and is not accepted evidence.
The replacement uses the real production SHA stated above. Original files
are retained untouched; no measurement was relabeled after acquisition.

Example strict post-fix replay:

```sh
jq -Rse --argjson count 4 --argjson windows 3 --arg cc 12.0 \
  --argjson revision 45 -f verify-inference.jq \
  inference-sm120-smoke-fix1/inference.log
```

For the retained pre-fix diagnostic only, pass `--argjson revision null`.
Full final inference runs on both boards remain separate evidence packets.
