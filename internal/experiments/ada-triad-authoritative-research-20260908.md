# Ada Triad: source-guided next candidates

User requested parallel authoritative research tied to actual kernel changes.
Discovery remains on Ada/CUDA13.2; Fixed inference and SM120 routes are frozen.
No full gates or toolkit sweep for each prototype.

## Selected hypotheses and falsification

| Priority / cell | Concrete mechanism | Expected benefit / risk | First evidence |
| --- | --- | --- | --- |
| Half NT d768-out | M64N128/BK64/S3, eight compute warps | Grid192→384; accumulators64→32/thread; shared98,304→73,728B. Better final-wave filling is an inference, not a guarantee. Still one CTA/SM; B tile requests double. | F16 bits + retainedS3/Fast eager/graph once7; inspect L2 only if this loses ambiguously. |
| TF32 NT d768-in | A-only ldmatrix.x4 on retained compact8/S2 | Reduce scalar shared-load instructions without extra fragment buffers. Keep B loads, RNA rounding and K8 MMA chain unchanged. Occupancy must remain at least2. | All-lane native mapping proof, target/tail/exception/K0 bits, actual AUTO/Fast once7. |
| Exact F32 NT d768-in / Prism | Existing transpose32x16 + existing production Fixed CopyPlan | Reuse the faster NN inner body; pay and time the entire transpose. Prism has a partial K32 slab, so its gain is not assumed from d768-out. | New isolated sibling test, generic exact/public AUTO bits, full2-node graph timing against AUTO/Fast. |
| F16 NN d768-in, next | Existing Fixed M128N64/BK64/S2 body | Existing49,152B/two-CTA body, finer grid; extra A tile traffic. | One new wrapper/cell, not a new arithmetic kernel. |
| Half TN d768-in, investigate next | M128N128/BK32/S3 | NVIDIA heuristic suggests49,152B mainloop, unlike the losing BK64/S3 at98,304B. Potential two-CTA residency, subject to registers and epilogue memory. | Source feasibility and occupancy calculation before implementing. |

NVIDIA explains tile-reuse versus parallelism and partially filled final waves
in its [matrix performance guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html).
[CUTLASS efficient GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)
describes register/shared pipelines, coalesced epilogues and CTA rasterization.
Those are mechanisms, not evidence that our candidate will win. Rasterization
is deferred unless doubled B traffic proves material.

## Analytical shortlist, actually queried

Installed only an isolated advisory tool on Ada, not a large CUTLASS build:
`nvidia-matmul-heuristics 0.1.0.27`, library0.1.0. The official
[overview](https://docs.nvidia.com/cuda/nvidia-matmul-heuristics/) supports Ada,
CUTLASS2/3 and these precision families. Internal discovery data calibrates the
library backends, not our custom kernels; predicted runtimes are not measurements.

[Query JSON](../perf/ada-nvmmh-shortlist-20260908/query.json), SHA256
`85ca2127b9f0204d6ecadcdf36dc5911165c47cb68bb48eaf9b0d98b14e5d61f`:

- GPU is explicit `RTX_6000_ADA`. Internal discovery loading returned true.
- Disable `HAS_SLICE_K`, `SPLIT_K_KIND`, workspace and clusters in the backend;
  verify splitK1, cluster1x1, CTA.K=warp.K in returned configurations.
- Our logical TN(2048,768,3072) maps to standard GEMM(768,3072,2048), TN_ROW,
  precisionHSS (half input, F32 compute/output).
- Logical NT(2048,1536,768) maps to standard(2048,1536,768), NT_ROW,
  HSH (half input/output, F32 compute).
- CUTLASS TN ranks include128x128x32/S3 and128x128x64/S2. NT ranks largely
  support the existing96KiB deep-stage direction. CUTLASS3 runtime estimates
  are inconsistent enough that they are not used to promise speedups.
- Shared-memory estimates in the JSON are our tile formula, excluding
  epilogue space; actual compiled resources remain mandatory.

API interpretation follows NVIDIA's [Python API](https://docs.nvidia.com/cuda/nvidia-matmul-heuristics/api_python.html)
and [backend properties](https://docs.nvidia.com/cuda/nvidia-matmul-heuristics/api.html).

## Contract and search protocol

Selected next TF32 structural hypothesis: convert each shared-stage word to
RNA bits once, then consume those raw bits with the unchanged K8 MMA chain.
Compact8 NT currently makes16,384 source-level conversion calls per BK32 for
6,144 unique words. This count is source analysis, not a measured speedup.
The async producer thread must convert only its own completed copy chunks
after wait_group and before the existing publishing CTA barrier; stage-reuse
barriers remain intact. Unconverted scalar fallbacks retain their original
consumer-side RNA conversion. No global prepack buffer or extra kernel.

The [PTX wait-group contract](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-wait-group-cp-async-wait-all)
is thread-local completion/visibility for async copies, not a substitute for
the CTA publishing barrier. Verify ownership, every ring stage, zero-fill and
fallback paths before timing. Risks are extra shared read/write traffic,
conversion on the critical path, register growth and lost occupancy.

If that fails, consider interleaving target-specific copy slices with K8 MMA
issues using the existing Fixed-N96 schedule, while retaining compact32/S2
storage. Do not combine both mechanisms in the first experiment.

Keep per-output reduction and rounding unchanged. SplitK/SliceK, tensor
decomposition of exact F32, and atomics are not drop-in bit-exact optimizations.
NVIDIA demonstrates how changing association/FMA grouping changes results in
[Floating Point and IEEE754](https://docs.nvidia.com/cuda/floating-point/index.html#dot-product-an-accuracy-example).
An independently deterministic alternative would still be a different contract.

For each next candidate: identify one limiting mechanism from source/profile;
consult primary documentation; native mapping/resource proof; focused GPU bits;
one paired eager/graph once7 screen. Stop unchanged valid losers. Do not infer
new-best status from separate current ratios. Preserve every measured source
before the shared harness advances. Integrate finalists across the whole Triad
then run one combined supported-toolkit qualification batch.

The new half-TN float2 epilogue screen reaches graph parity, not a strict win;
its [report](../perf/ada-triad-half-tn-regpipe-vec2-20260908/report.md) and the
[NCU diagnostic](../perf/ada-triad-half-tn-regpipe-20260908/ncu-diagnostic.md)
separate measured outcomes from the hypotheses above.

Completed first F32 reuse screen: [NT sibling report](../perf/ada-f32-nt-copyplan-siblings-20260908/report.md).
Actual AUTO time falls56.7–56.8% on d768-in and12.8–13.0% on Prism, with target,
tail, exceptional and K0 bits preserved. Fast gaps remain2.36x/3.92x. These
measurements validate reuse for these cells, not a general performance guarantee.
