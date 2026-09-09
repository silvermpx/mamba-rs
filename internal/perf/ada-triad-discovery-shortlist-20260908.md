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

The later [F16 NN N96/S3 result](ada-triad-half-nn-n96-s3-20260909/report.md)
uses the `gemm_bi_nn_fixed_sm89_*` symbol and Fixed parameter ABI. It is a valid Fixed/inference
retained win and Fast near miss, but is outside this training-Triad shortlist
and does not change the Triad win count.

## Important improvements that are not Fast wins

| Scope | Keep / remaining issue | Evidence |
| --- | --- | --- |
| F16 NN d768-in | Keep S3. New M128N64/S2 reuse loses retainedS3 by2.5–3.4%; Pipeline/Swizzle also weaker. B-`ldmatrix.x4` reduces registers188->168 but loses S3 by1.01–1.31%. No unchanged retry. | [S3](ada-triad-half-nt128-f16in-20260908/README.md), [N64 loss](ada-triad-half-nn-m128n64-s2-20260908/report.md), [loaded losses](ada-triad-half-nn-loaded2-20260908/README.md), [B-x4 stop](ada-triad-half-nn-s3-bx4-20260909/report.md) |
| Half NT d768-out | NEW M64N128/S3 beats retained B-XOR S3 by4.5–4.6% for BOTH F16/BF16. Fast graph worst p95 F16 1.0105/BF16 1.0217; neither a strict Fast winner. M64N192 is exact and faster in every F16 once3 observation, but graph BAAB p50/p95 `0.99007/0.99073` narrowly miss the1% retained gate. The direct BF16 sibling likewise improves eager1.23–1.44% but graph p50/p95 only0.99007/0.99072, so it also stops before once7. Changing F16 M64N192 `cp.async` issue distribution from1/1/2/2 to2/2/1/1 produces distinct SASS but stays parity/slightly slower (`1.0000–1.00065` p50). A literal full-domain F16 M64N192 entry reduces127->118regs but loses generic M64N192 by2.9–3.1%. S2 loses retained4.3–7.7%; GROUP_M8 loses0.26–0.46%. | [M64N128 F16](ada-triad-half-nt-m64n128-s3-20260908/report.md), [BF16](ada-triad-half-nt-m64n128-s3-bf16-20260908/report.md), [F16 N192 near miss](ada-triad-half-nt-m64n192-s3-20260909/report.md), [BF16 N192 stop](ada-triad-bf16-nt-m64n192-s3-20260909/report.md), [issue-order stop](ada-triad-f16-nt-m64n192-issue2211-20260909/report.md), [N192 full-domain stop](ada-triad-half-nt-m64n192-full-domain-20260909/report.md), [S2 stop](ada-triad-half-nt-m64n128-s2-20260909/report.md), [GROUP_M8 stop](ada-triad-half-nt-m64n128-s3-groupm8-20260909/report.md), [retained out](ada-triad-half-nt-fixed-s3-bxor-20260908/report.md) |
| Half TN | d768-in regpipe+vec2 reaches Fast graph parity but has no strict Fast win. Five-warp loses retained15.4–17.3%; B-ldmatrix x4 exceeds its register cap. Stage-sliced cp.async improves resources125->115regs but loses regpipe+vec2 by2.5–3.7%. A resource-neutral HMMA issue-order change stays at parity/slightly slower (`0.9997–1.0006` p50, worst p95 `1.00148`). Exact-shape 2D mapping plus unpredicated epilogue raises registers125->128 and stops at compile. Eight 16x32 compute-warps reach80regs/occupancy3 but duplicate B-fragment loads and lose retained10.4–11.0%. Exact-shape unpredicated full-tile staging improves resources125->118regs and speed0.32–0.80%, but misses the strict1% once3 gate. Combining full-domain/staging/epilogue in a target-only entry cuts125->102regs and text to.396x, yet loses retained0.9–1.7% and stops once3. Changing full-tile staging to L2-only `.cg` loses retained about16%, proving L1 reuse matters. Compact BF16 BK32/S3 reaches120regs/24,576B shared/occupancy4 but loses retained27.8–30.4%. NEW BF16 d768-out beats retained compact1.6–1.8% robustly; F16 out and both Prism siblings fail the strict retained gate. No half-TN Fast win yet. | [Compact](ada-triad-half-tn-compact-20260908/README.md), [d768-in vec2](ada-triad-half-tn-regpipe-vec2-20260908/report.md), [WS5 stop](ada-triad-half-tn-ws5-20260909/report.md), [B-x4 stop](ada-triad-half-tn-bx4-20260909/report.md), [sliced stop](ada-triad-half-tn-regpipe-vec2-sliced-20260909/report.md), [issue-order stop](ada-triad-half-tn-regpipe-vec2-issue-20260909/report.md), [full-domain compile stop](ada-triad-half-tn-vec2-full-domain-20260909/report.md), [8-warp stop](ada-triad-half-tn-8warp-16x32-20260909/report.md), [full-tile staging stop](ada-triad-half-tn-vec2-full-tile-stage-20260909/report.md), [exact-entry stop](ada-triad-half-tn-vec2-exact-entry-20260909/report.md), [full-tile cg stop](ada-triad-half-tn-full-tile-cg-20260909/report.md), [BK32/S3 stop](ada-triad-half-tn-bk32-s3-regpipe-vec2-20260909/report.md), [siblings](ada-triad-half-tn-regpipe-vec2-siblings-20260908/report.md), [S3 loss](ada-triad-half-tn-fixed-s3-20260908/report.md) |
| TF32 NN d768-out / Prism | N96 reduces AUTO time; near Fast parity. NEW direct-float2 epilogue is a strict retained-best improvement on Prism (1.1–1.8%) and reaches graph/Fast near parity, but eager remains1.2–1.3% slower than Fast. Removing its final CTA barrier adds only0.0–0.4%. A full-domain unpredicated-cp.async arm reaches clean131-reg SASS but exceeds its128-reg cap and stops before GPU. d768-out gains only0.2–0.4% and stops below the1% retained gate. N96 d768-in loses and is excluded. | [N96](ada-triad-nn-addhalf-n96-screen-20260908/README.md), [direct epilogue](ada-triad-tf32-nn-n96-direct-epilogue-20260909/report.md), [no-barrier stop](ada-triad-tf32-nn-n96-direct-epilogue-nobarrier-20260909/report.md), [full-domain stop](ada-triad-tf32-nn-n96-full-domain-20260909/report.md) |
| TF32 NT large shapes | Keep A-only ldmatrix for d768-in. NEW stage-sliced A-ldmatrix wins d768-out retained by1.28–1.47% and `large_deep` retained by1.85–2.19%, exact and resource-neutral. Fast remains1.41–1.42x ahead on d768-out and about1.56x ahead on `large_deep`. The same body improves d768-in only0.45–0.65%, below the strict1% admission gate. Shared-RNA and A+B ldmatrix lose the d768-in parent and stop. The frozen `large_deep` harness/raw schema used the wrong `Prism` label; canonical `prism_in_proj` `(4621,384,1928)` is a different cell. A canonical-Prism 2D/full-domain candidate compiles at141regs/occupancy1 and stops at the resource gate; a separate linear-grid K8 A-regpipe candidate is exact/resource-neutral but slightly slower at once3 p50`1.00129-1.00169`. | [d768-in](ada-triad-tf32-nt-a-ldmatrix-20260908/report.md), [out/Prism](ada-triad-tf32-nt-a-ldmatrix-siblings-20260908/report.md), [sliced d768-out](ada-triad-tf32-nt-a-ldmatrix-sliced-20260909/report.md), [sliced large_deep (legacy path)](ada-triad-tf32-nt-a-ldmatrix-sliced-prism-20260909/report.md), [sliced d768-in stop](ada-triad-tf32-nt-a-ldmatrix-sliced-in-20260909/report.md), [canonical Prism full-domain stop](ada-triad-tf32-nt-prism-full-domain-20260909/report.md), [canonical Prism K8-regpipe stop](ada-triad-tf32-nt-prism-k8-regpipe-20260909/report.md), [shared RNA](ada-triad-tf32-nt-shared-rna-20260908/report.md), [A+B](ada-triad-tf32-nt-compact-ab-ldmatrix-20260908/report.md) |
| TF32 TN large shapes | Whole raw-transpose+N96 reduces actual AUTO time29–30% in,40% out,15–17% Prism. NEW A-only transpose-RNA recipe wins all three retained cells: d768-in7.3–7.4%, d768-out6.5–7.0%, canonical Prism7.3–7.7%, exact in every gate. It remains26–27%,7.8–9.1%, and88% slower than Fast at p50 respectively. Joint A+B preprocessing reaches114regs/local0 but loses A-only1.6–2.2% because extra B scratch traffic dominates. A d768-out M128N192 S3/S2 route is a quantified no-go: S3 exceeds Ada shared/block; S2 would need an implausible21.6% win over already-slower N128 to close Fast. No Fast win yet. | [Earlier in](ada-triad-tf32-tn-transpose-n96-20260908/report.md), [earlier out/Prism](ada-triad-tf32-tn-transpose-n96-siblings-20260908/report.md), [d768-out A-RNA winner](ada-triad-tf32-tn-transpose-rna-n96-d768-out-20260909/report.md), [in/Prism A-RNA winners](ada-triad-tf32-tn-transpose-rna-n96-siblings-20260909/report.md), [A+B stop](ada-triad-tf32-tn-transpose-ab-rna-n96-d768-out-20260909/report.md) |
| TF32 TN direct-N96 d768-in | Direct staging removes transpose but still loses Fast graph35–37%. Not paired against retained transpose+N96; no new-best claim. | [Direct probe](ada-triad-tf32-tn-direct-n96-20260908/report.md) |
| Exact F32 TN d128-in/out | Direct fixed-order fold substantially improves AUTO; still roughly2–3x Fast. | [In](ada-triad-f32-tn-d128-direct-20260908/README.md), [out](ada-triad-f32-tn-d128-out-direct-20260908/README.md) |
| Exact F32 TN large | NEW d768-in dual-chunk fused candidate reduces the retained three-node N64 GROUP_M8 pipeline to two nodes and wins by4.0–4.4% with exact bits. Fast remains2.30–2.33x ahead. A one-node direct row-major-X combination loses retained by16.2–16.6%. Rolling the two exact chains halves SASS text/FFMA sites but introduces16B spills and stops before GPU. Earlier SplitM CopyPlan cuts actual AUTO14.5-15.4% for d768-in and9.5-11.5% for d768-out. Prism is exact but loses AUTO34.8-36.4% and stops. | [dual-chunk retained-best](ada-triad-f32-tn-dual-chunk-20260909/report.md), [direct dual stop](ada-triad-f32-tn-direct-dual-chunk-20260909/report.md), [rolled compile stop](ada-triad-f32-tn-rolled-dual-chunk-20260909/report.md), [d768-in CopyPlan](ada-triad-f32-tn-splitm-copyplan-20260908/report.md), [out/Prism](ada-triad-f32-tn-splitm-copyplan-siblings-20260908/report.md), [earlier full-K correctness stop](ada-triad-f32-tn-copyplan-d768-in-20260908/report.md) |
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
