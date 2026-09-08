# Ada Triad: source-guided next candidates

User requested parallel authoritative research tied to actual kernel changes.
Discovery remains on Ada/CUDA13.2; Fixed inference and SM120 routes are frozen.
No full gates or toolkit sweep for each prototype.

The user explicitly reaffirmed this as the persistent working method, not a
one-off experiment. Its canonical rules are the first section of
`internal/agent-operational-rules.md`; the handoff and full-wiring plan point there.
Continue from measured remaining gaps and retained champions, research concrete
mechanisms in parallel, screen briefly, stop valid losers, and qualify once after
the whole-Triad shortlist is assembled. Do not substitute repeated verification
of old wins for new candidate search.

## Selected hypotheses and falsification

| Priority / cell | Concrete mechanism | Expected benefit / risk | First evidence |
| --- | --- | --- | --- |
| Half NT d768-out | M64N128/BK64/S3, eight compute warps | Grid192→384; accumulators64→32/thread; shared98,304→73,728B. | Measured BOTH F16/BF16:4.5–4.6% retained-best win, Fast graph p95 still fails. Stop unchanged screen. |
| TF32 NT d768-in | A-only ldmatrix.x4 on retained compact8/S2 | Reduce scalar shared-load instructions without extra fragment buffers. Keep B loads, RNA rounding and K8 MMA chain unchanged. Occupancy must remain at least2. | All-lane native mapping proof, target/tail/exception/K0 bits, actual AUTO/Fast once7. |
| Exact F32 NT d768-in / Prism | Existing transpose32x16 + existing production Fixed CopyPlan | Reuse the faster NN inner body; pay and time the entire transpose. Prism has a partial K32 slab, so its gain is not assumed from d768-out. | New isolated sibling test, generic exact/public AUTO bits, full2-node graph timing against AUTO/Fast. |
| F16 NN d768-in | Existing Fixed M128N64/BK64/S2 body | Existing49,152B/two-CTA body, finer grid; extra A tile traffic. | Measured valid2.5–3.4% retainedS3 loss. Stop; keep S3. |
| Half TN d768-in, rejected before build | M128N128/BK32/S3, four warps | Heuristic49,152B mainloop is valid, but ownership requires128 accumulators/thread (not64), and grid144 cannot fill two CTAs on142 SMs. | Source/resource analysis rejects this four-warp geometry; no speed claim. |
| Half TN d768-in, stop | M64N128/BK64/S2 FOUR-warp compact regpipe/vec2 |64 accumulators/thread,49,152B shared,grid288;25% less source-level staging for equal output work. | Measured200regs/occ2/local0, target bits PASS, but19.2–19.6% retainedvec2 loss. NoBF16/no retry. [Report](../perf/ada-triad-half-tn-m64n128-regpipe-20260908/report.md). |

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

Additional advisory query: [TF32 TN Prism JSON](../perf/ada-nvmmh-shortlist-20260908/query-tf32-tn-prism.json),
SFS/TN_ROW standard(384,1928,4621), same explicit GPU and no-SplitK/SliceK/cluster
constraints. Only four legal CUTLASS configurations returned: three M128N64
grids93 and one M64N64 grid186, against142 Ada SMs. The latter improves grid
coverage but its predicted runtime is worse; this is a tradeoff to investigate,
not proof of a win. Audit found stock TF32 TN M64N64/BK32/S2 was already forced
on Prism in `internal/perf/gemm-bi-triad-sm89-candidate-prelocal-20260827/candidate-prelocal-{ab,ba}.jsonl`:
grid186,343.2–344.0us versus wide S3's316.5–317.2us in that old cohort. These
are independent measurements, not a modern paired comparison with retained N96.
Still, stock M64 is not untested and is not a priority retry. The existing
four-warp compact candidate is separately M128N64/grid93, not stock M64.

Completed A-only ldmatrix screen: [TF32 NT report](../perf/ada-triad-tf32-nt-a-ldmatrix-20260908/report.md).
ActualAUTO time falls11.3–11.4%, target/tail/exception/K0 bits pass, occupancy2.
Candidate/Fast time remains1.69–1.71. Retain this as the
new baseline for shared-stage RNA; a gain versus old AUTO alone is not enough
to call the next candidate a new best.

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

First compiled shared-RNA source (`03e6aa25` main / `96ae0a10` helper) fails
the resource gate:138regs, local0,49,152B shared, occupancy1 instead of2.
No bit or speed verdict exists for that attempt. One targeted refinement is
the candidate-only `__launch_bounds__(256,2)` rather than `(256,1)`, retaining
the same local0/occupancy2 gate. NVIDIA's
[launch-bound guidance](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html)
explains the register-budget effect and possible instruction/spill costs;
this is a distinct compiler-scheduling hypothesis, not permission to weaken
resource gates or run a blind register-count sweep.

That refinement is now measured:125regs/local0/occupancy2 and focused GPU bits
PASS, but35.4–35.5% slower than retained A-ldmatrix (~272vs201us), Fast2.27–2.32x.
See [both attempts](../perf/ada-triad-tf32-nt-shared-rna-20260908/report.md).
Stop the shared-stage RNA direction for this candidate; do not repeat/profile
this decisive loss. Keep A-ldmatrix, and investigate distinct fragment-load
instruction reduction on that retained body instead.

Selected next TF32 load hypothesis: retain A-only x4, replace only B scalar
loads with non-transposed `ldmatrix.m8n8.x2.shared.b16`, then perform the SAME
consumer RNA conversion. The padded helper already contains this B-load
pattern; compact XOR alignment and all-lane mapping are checked separately.
For consumer laneL, registers map to `(col=warp_n+nAtom*8+L/4,k=k8+L%4)`
and the same column/k+4, exactly the two old scalar words. No shared-stage
conversion, new buffer, MMA-order change or old-AUTO-only speed comparison.
Use the [PTX ldmatrix mapping](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-ldmatrix)
and native coordinate/16B-row proofs before the retainedA-only/Fast shortscreen.

That A+B load experiment is now complete and is a valid loser. Exact-bit,
guard and resource gates pass (106regs/local0/occupancy2), but it is2.0–2.1%
slower than retained A-only in every eager/graph ordering cohort and1.72–1.75x
cuBLAS Fast. See the [A+B report](../perf/ada-triad-tf32-nt-compact-ab-ldmatrix-20260908/report.md).
Stop this unchanged candidate without retry; retain A-only.

If that fails, consider interleaving target-specific copy slices with K8 MMA
issues using the existing Fixed-N96 schedule, while retaining compact32/S2
storage. Do not combine both mechanisms in the first experiment.

Keep per-output reduction and rounding unchanged. SplitK/SliceK, tensor
decomposition of exact F32, and atomics are not drop-in bit-exact optimizations.
NVIDIA demonstrates how changing association/FMA grouping changes results in
[Floating Point and IEEE754](https://docs.nvidia.com/cuda/floating-point/index.html#dot-product-an-accuracy-example).
An independently deterministic alternative would still be a different contract.

Future exact-F32 TN reuse: source analysis finds that direct CopyPlan output
is NOT a general TN epilogue replacement. CopyPlan uses multiply then FMA,
whereas TN requires `fma(alpha, accumulator, oldC)`. A potential d768-in
experiment is raw A transpose -> CopyPlan(alpha1,beta0) into accumulator
scratch -> exact TN FMA epilogue, timing all three nodes. Preserve NaN payloads
through the intermediate alpha1 multiplication or reject/adapt raw-acc output;
do not assume payload identity from ordinary numeric correctness. Prism also
needs a padded transpose stride4624 instead of4621. This is feasibility only,
not an implemented/measured winner; prioritize the current short candidate queue.

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

The same unchanged regpipe+vec2 body is now screened on d768-out and Prism for
F16/BF16. Only BF16 d768-out advances: candidate/retained p50.98166–.98412,
worst p95.98758. It is not a Fast win (p501.06798–1.11093). F16 out misses
one retained p95 at.99124; both Prism dtypes lose. See the
[sibling report](../perf/ada-triad-half-tn-regpipe-vec2-siblings-20260908/report.md).
Stop the three losers; retain the BF16-out result for joint integration.

Completed first F32 reuse screen: [NT sibling report](../perf/ada-f32-nt-copyplan-siblings-20260908/report.md).
Actual AUTO time falls56.7–56.8% on d768-in and12.8–13.0% on Prism, with target,
tail, exceptional and K0 bits preserved. Fast gaps remain2.36x/3.92x. These
measurements validate reuse for these cells, not a general performance guarantee.
