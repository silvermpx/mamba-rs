# Ada Triad discovery shortlist — 2026-09-08

This is a selection index, not a new assembled-dispatch benchmark or release
qualification. CUDA13.2 / RTX6000Ada only unless a row explicitly says otherwise.
Keep Fixed inference and SM120 routes unchanged. Whole-Triad discovery first;
then integrate the selected replacements and run one full toolkit batch.

## Paired Fast-winning replacement cells

All nine cells below pass their focused current-family exact-bit checks and
beat explicit native-half cuBLAS Fast in eager/graph × ABBA/BAAB once7, including
the p95 screen. All use256B-aligned timing buffers and20 GEMMs/observation.
They are candidates, not nine newly admitted public AUTO routes.

| Op | Dtype | Shape | Candidate | Candidate/Fast p50 | Evidence |
| --- | --- | --- | --- | ---: | --- |
| NN | BF16 | d768-in | Fixed S3 | .775–.786 | [BF16 in](ada-triad-half-nn-s3-bf16in-20260908/README.md) |
| NN | F16/BF16 | d768-out, Prism (4 cells) | Fixed S3 | about .77–.80 | [Aligned NN](ada-triad-half-nn-s3-aligned-20260908/README.md) |
| NT | F16 | d768-in | Fixed-S3 B-XOR | .8802–.8926 | [NT sibling raw](ada-triad-half-nt-fixed-s3-siblings-20260908/evidence/cuda132/run1/test.log) |
| NT | BF16 | d768-in | Fixed-S3 B-XOR | .8955–.9270 | [NT sibling raw](ada-triad-half-nt-fixed-s3-siblings-20260908/evidence/cuda132/run1/test.log) |
| NT | F16 | Prism | Fixed-S3 B-XOR | .5000–.5092 | [NT sibling raw](ada-triad-half-nt-fixed-s3-siblings-20260908/evidence/cuda132/run1/test.log) |
| NT | BF16 | Prism | Fixed-S3 B-XOR | .7781–.8018 | [NT sibling raw](ada-triad-half-nt-fixed-s3-siblings-20260908/evidence/cuda132/run1/test.log) |

This count excludes old stock-route wins, misaligned historical fixtures,
unpaired timing ratios, marginal parity, and merely faster-than-AUTO results.

## Important improvements that are not Fast wins

| Scope | Keep / remaining issue | Evidence |
| --- | --- | --- |
| F16 NN d768-in | Keep S3. New M128N64/S2 reuse loses retainedS3 by2.5–3.4%; Pipeline/Swizzle also weaker. B-`ldmatrix.x4` reduces registers188->168 but loses S3 by1.01–1.31%. No unchanged retry. | [S3](ada-triad-half-nt128-f16in-20260908/README.md), [N64 loss](ada-triad-half-nn-m128n64-s2-20260908/report.md), [loaded losses](ada-triad-half-nn-loaded2-20260908/README.md), [B-x4 stop](ada-triad-half-nn-s3-bx4-20260909/report.md) |
| Half NT d768-out | NEW M64N128/S3 beats retained B-XOR S3 by4.5–4.6% for BOTH F16/BF16. Fast graph worst p95 F16 1.0105/BF16 1.0217; neither a strict Fast winner. A two-stage F16 ping-pong variant improves resources to106regs/49,152B/occupancy2 but loses retained S3 by4.3–7.7%; valid stop before Fast, no BF16/siblings. GROUP_M8 traversal is exact/resource-neutral but loses row-major S3 by0.26–0.46%; stop before once7/Fast. | [M64N128 F16](ada-triad-half-nt-m64n128-s3-20260908/report.md), [BF16](ada-triad-half-nt-m64n128-s3-bf16-20260908/report.md), [S2 stop](ada-triad-half-nt-m64n128-s2-20260909/report.md), [GROUP_M8 stop](ada-triad-half-nt-m64n128-s3-groupm8-20260909/report.md), [retained out](ada-triad-half-nt-fixed-s3-bxor-20260908/report.md) |
| Half TN | d768-in regpipe+vec2 reaches Fast graph parity but has no strict Fast win. The SM89 five-warp producer/consumer variant is exact and occupancy3, but loses retained15.4–17.3% and Fast12.0–16.8%; valid stop, no BF16/siblings. B-ldmatrix x4 fusion preserves occupancy3/local0 but compiles to137 regs above its128 cap in both loop and literal forms; resource stop before exact/timing. NEW BF16 d768-out beats retained compact1.6–1.8% robustly; F16 out and both Prism siblings fail the strict retained gate. No half-TN Fast win yet. | [Compact](ada-triad-half-tn-compact-20260908/README.md), [d768-in vec2](ada-triad-half-tn-regpipe-vec2-20260908/report.md), [WS5 stop](ada-triad-half-tn-ws5-20260909/report.md), [B-x4 stop](ada-triad-half-tn-bx4-20260909/report.md), [siblings](ada-triad-half-tn-regpipe-vec2-siblings-20260908/report.md), [S3 loss](ada-triad-half-tn-fixed-s3-20260908/report.md) |
| TF32 NN d768-out / Prism | N96 reduces AUTO time; near Fast parity. NEW direct-float2 epilogue is a strict retained-best improvement on Prism (1.1–1.8%) and reaches graph/Fast near parity, but eager remains1.2–1.3% slower than Fast. Removing its final CTA barrier adds only0.0–0.4%, has p95 at1.00002–1.00005, and is a valid stop. d768-out gains only0.2–0.4% and stops below the1% retained gate. N96 d768-in loses and is excluded. | [N96](ada-triad-nn-addhalf-n96-screen-20260908/README.md), [direct epilogue](ada-triad-tf32-nn-n96-direct-epilogue-20260909/report.md), [no-barrier stop](ada-triad-tf32-nn-n96-direct-epilogue-nobarrier-20260909/report.md) |
| TF32 NT large shapes | Keep A-only ldmatrix for all three: it cuts actual AUTO time11.3–11.4% on d768-in,9.4–9.7% on d768-out and9.9–10.0% on Prism, exact bits. Fast remains ahead by1.40–1.89x on the new siblings. Shared-RNA and A+B ldmatrix lose the d768-in A-only baseline and stop. | [d768-in](ada-triad-tf32-nt-a-ldmatrix-20260908/report.md), [out/Prism](ada-triad-tf32-nt-a-ldmatrix-siblings-20260908/report.md), [shared RNA](ada-triad-tf32-nt-shared-rna-20260908/report.md), [A+B](ada-triad-tf32-nt-compact-ab-ldmatrix-20260908/report.md) |
| TF32 TN large shapes | Whole raw-transpose+N96 reduces actual AUTO time29–30% in,40% out,15–17% Prism. Graph/Fast respectively1.34,1.20,2.00; no Fast win. | [In](ada-triad-tf32-tn-transpose-n96-20260908/report.md), [out/Prism](ada-triad-tf32-tn-transpose-n96-siblings-20260908/report.md) |
| TF32 TN direct-N96 d768-in | Direct staging removes transpose but still loses Fast graph35–37%. Not paired against retained transpose+N96; no new-best claim. | [Direct probe](ada-triad-tf32-tn-direct-n96-20260908/report.md) |
| Exact F32 TN d128-in/out | Direct fixed-order fold substantially improves AUTO; still roughly2–3x Fast. | [In](ada-triad-f32-tn-d128-direct-20260908/README.md), [out](ada-triad-f32-tn-d128-out-direct-20260908/README.md) |
| Exact F32 TN large | NEW d768-in dual-chunk fused candidate reduces the retained three-node N64 GROUP_M8 pipeline to two nodes and wins by4.0–4.4% with exact bits. Fast remains2.30–2.33x ahead. A one-node direct row-major-X combination is also exact/occupancy3 but loses this retained-best by16.2–16.6%; direct staging stops. Earlier SplitM CopyPlan cuts actual AUTO14.5-15.4% for d768-in and9.5-11.5% for d768-out. Prism is exact but loses AUTO34.8-36.4% and stops. | [dual-chunk retained-best](ada-triad-f32-tn-dual-chunk-20260909/report.md), [direct dual stop](ada-triad-f32-tn-direct-dual-chunk-20260909/report.md), [d768-in CopyPlan](ada-triad-f32-tn-splitm-copyplan-20260908/report.md), [out/Prism](ada-triad-f32-tn-splitm-copyplan-siblings-20260908/report.md), [earlier full-K correctness stop](ada-triad-f32-tn-copyplan-d768-in-20260908/report.md) |
| Exact F32 TN d768-in fused N32 | Exact bits/resources pass, but the three-node M64N32 GROUP_M8 pipeline loses the retained fused N64 pipeline by8.8–9.1% in all eager/graph paired strata. Valid stop; do not retry unchanged. | [N32 loss](ada-triad-f32-tn-fused-n32-20260909/report.md) |
| Exact F32 TN d768-in direct | Direct shared staging removes transpose and merges partial launches, but only ties retained eager and loses retained graph0.67-0.72%. Stop unchanged candidate; next try fused exact finalize. | [Direct loss](ada-triad-f32-tn-direct-copyplan-20260908/report.md) |
| Exact F32 NN large / NT d768-out | Preserve the earlier CopyPlan reuse finalists and their existing three-toolkit evidence; these are not new Fast wins. | [Live reuse](ada-triad-live-reuse-20260908/README.md) |
| Exact F32 NT d768-in / Prism | New whole transpose+CopyPlan candidates cut actual AUTO time56.7–56.8% /12.8–13.0%, with exact bits. Candidate/Fast time ratios remain2.36 /3.92. Retain for joint integration, no Fast-win count. | [Sibling reuse](ada-f32-nt-copyplan-siblings-20260908/report.md) |

This table is not exhaustive closure of60 operation/dtype/shape cells. The old
[whole-matrix snapshot](ada-triad-state-20260907/report.md) remains historical;
do not merge its independent quantiles with these paired screens to invent
an average speedup or claim that the remaining cells are finished.
The current strict Y/N/U inventory is maintained in the
[60-cell coverage audit](ada-triad-60-cell-coverage-20260908.md).

## Integration and cleanup boundary

- Selected half NN uses the existing Fixed S3 holder; NT B-XOR and new TN
  source adapters currently live only in tests/support. Their public physical
  identities and precision/shape/toolkit guards still need joint integration.
- Preserve measured source by its committed hash/snapshot before another
  owner changes a shared harness. Never qualify a different live source as
  if it were the measured one.
- Valid losing experiments stay in the separate evidence/test area for now.
  No Ada loss authorizes removing a route used on another shape or GPU.
- The TF32 NN N96 B-ldmatrix probe is a valid exact loser on d768-out/Prism
  (8.9–9.9% slower than retained); keep scalar B loads and do not retry it.
  See [the frozen loss](ada-triad-tf32-nn-n96-b-ldmatrix-20260908/report.md).
- Neither source availability nor a CUDA13.2 win proves all-architecture or
  CUDA12.8/13.0 performance. The final batch must verify those supported
  toolkits, preserve fallbacks and reject foreign device-specific identities.
