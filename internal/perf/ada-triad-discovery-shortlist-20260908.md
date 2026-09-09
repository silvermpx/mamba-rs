# Ada Triad discovery shortlist — 2026-09-08

This is a selection index, not a new assembled-dispatch benchmark or release
qualification. CUDA13.2 / RTX6000Ada only unless a row explicitly says otherwise.
Keep Fixed inference and SM120 routes unchanged. Whole-Triad discovery first;
then integrate the selected replacements and run one full toolkit batch.

## Paired Fast-winning replacement cells

All eleven cells below pass their focused current-family exact-bit checks and
beat explicit native-half cuBLAS Fast in eager/graph × ABBA/BAAB once7, including
the p95 screen. All use256B-aligned timing buffers and20 GEMMs/observation.
They are candidates, not ten newly admitted public AUTO routes.

| Op | Dtype | Shape | Candidate | Candidate/Fast p50 | Evidence |
| --- | --- | --- | --- | ---: | --- |
| NN | BF16 | d768-in | Fixed S3 | .775–.786 | [BF16 in](ada-triad-half-nn-s3-bf16in-20260908/README.md) |
| NN | F16/BF16 | d768-out, Prism (4 cells) | Fixed S3 | about .77–.80 | [Aligned NN](ada-triad-half-nn-s3-aligned-20260908/README.md) |
| NT | F16 | d768-in | Fixed-S3 B-XOR | .8802–.8926 | [NT sibling raw](ada-triad-half-nt-fixed-s3-siblings-20260908/evidence/cuda132/run1/test.log) |
| NT | BF16 | d768-in | Fixed-S3 B-XOR | .8955–.9270 | [NT sibling raw](ada-triad-half-nt-fixed-s3-siblings-20260908/evidence/cuda132/run1/test.log) |
| NT | F16 | d768-out | M96N128/BK64/S3 | .8981–.9141 | [F16 out](ada-triad-f16-nt-m96n128-s3-20260909/report.md) |
| NT | BF16 | d768-out | M96N128/BK64/S3 | .8975–.9261 | [BF16 out](ada-triad-bf16-nt-m96n128-s3-20260909/report.md) |
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
| Half NT d768-out | NEW F16 and BF16 M96N128/BK64/S3 are strict Fast winners. F16 beats its retained M64N192 by6.5–6.8% and Fast by8.6–10.2% p50; BF16 beats retained M64N128 by7.7–7.9% and Fast by7.4–10.3% p50. Both keep every once7 p95 below.95. The geometry reduces staged traffic9.77%, uses120regs/86,016B and15 LDGSTS. Earlier direct BF16 M64N192, F16 issue-order, full-domain, S2 and GROUP_M8 attempts stop. | [F16 M96N128 winner](ada-triad-f16-nt-m96n128-s3-20260909/report.md), [BF16 M96N128 winner](ada-triad-bf16-nt-m96n128-s3-20260909/report.md), [M64N128 F16](ada-triad-half-nt-m64n128-s3-20260908/report.md), [BF16](ada-triad-half-nt-m64n128-s3-bf16-20260908/report.md), [F16 N192 near miss](ada-triad-half-nt-m64n192-s3-20260909/report.md), [BF16 N192 stop](ada-triad-bf16-nt-m64n192-s3-20260909/report.md), [issue-order stop](ada-triad-f16-nt-m64n192-issue2211-20260909/report.md), [N192 full-domain stop](ada-triad-half-nt-m64n192-full-domain-20260909/report.md), [S2 stop](ada-triad-half-nt-m64n128-s2-20260909/report.md), [GROUP_M8 stop](ada-triad-half-nt-m64n128-s3-groupm8-20260909/report.md), [retained out](ada-triad-half-nt-fixed-s3-bxor-20260908/report.md) |
| Half TN | d768-in regpipe+vec2 reaches Fast graph parity but has no strict Fast win. Five-warp loses15.4–17.3%; B-ldmatrix x4 exceeds its cap; stage-sliced loses2.5–3.7%; issue-order stays parity; 8-warp loses10.4–11.0%; full-tile staging gains only0.32–0.80%; full-domain target loses0.9–1.7%; `.cg` loses~16%; compact BF16 BK32 loses27.8–30.4%. NEW M96N64/BK64/S2 reduces staged traffic16.7% with twelve resident warps, but loses retained9.3–11.7%. BF16 d768-out still beats retained compact1.6–1.8%; no half-TN Fast win yet. | [Compact](ada-triad-half-tn-compact-20260908/README.md), [d768-in vec2](ada-triad-half-tn-regpipe-vec2-20260908/report.md), [M96N64 stop](ada-triad-bf16-tn-m96n64-regpipe-vec2-20260909/report.md), [WS5 stop](ada-triad-half-tn-ws5-20260909/report.md), [B-x4 stop](ada-triad-half-tn-bx4-20260909/report.md), [sliced stop](ada-triad-half-tn-regpipe-vec2-sliced-20260909/report.md), [issue-order stop](ada-triad-half-tn-regpipe-vec2-issue-20260909/report.md), [8-warp stop](ada-triad-half-tn-8warp-16x32-20260909/report.md), [full-tile staging stop](ada-triad-half-tn-vec2-full-tile-stage-20260909/report.md), [exact-entry stop](ada-triad-half-tn-vec2-exact-entry-20260909/report.md), [full-tile cg stop](ada-triad-half-tn-full-tile-cg-20260909/report.md), [BK32/S3 stop](ada-triad-half-tn-bk32-s3-regpipe-vec2-20260909/report.md), [siblings](ada-triad-half-tn-regpipe-vec2-siblings-20260908/report.md) |
| TF32 NN d768-out / Prism | N96 reduces AUTO time; near Fast parity. Direct-float2 epilogue improves Prism1.1–1.8%, but eager remains1.2–1.3% slower than Fast. No-barrier gains only0.0–0.4%; `ca` loses0.9–2.1%; full-domain reaches131regs and stops. M96N96/S2 reaches occupancy2/local0 but compiles at137regs over its128 cap and stops before exact/timing; retry only with a concrete >=9-register lifetime reduction. d768-out gains only0.2–0.4%. | [N96](ada-triad-nn-addhalf-n96-screen-20260908/README.md), [direct epilogue](ada-triad-tf32-nn-n96-direct-epilogue-20260909/report.md), [no-barrier stop](ada-triad-tf32-nn-n96-direct-epilogue-nobarrier-20260909/report.md), [CA stop](ada-triad-tf32-nn-n96-direct-epilogue-ca-20260909/report.md), [M96N96 resource stop](ada-triad-tf32-nn-m96n96-s2-direct-prism-20260909/report.md), [full-domain stop](ada-triad-tf32-nn-n96-full-domain-20260909/report.md) |
| TF32 NT large shapes | Keep A-only ldmatrix for d768-in. NEW stage-sliced A-ldmatrix wins d768-out retained by1.28–1.47% and `large_deep` retained by1.85–2.19%, exact and resource-neutral. Fast remains1.41–1.42x ahead on d768-out and about1.56x ahead on `large_deep`. The same body improves d768-in only0.45–0.65%, below the strict1% admission gate. Shared-RNA and A+B ldmatrix lose the d768-in parent and stop. The frozen `large_deep` harness/raw schema used the wrong `Prism` label; canonical `prism_in_proj` `(4621,384,1928)` is a different cell. A canonical-Prism 2D/full-domain candidate compiles at141regs/occupancy1 and stops at the resource gate; a separate linear-grid K8 A-regpipe candidate is exact/resource-neutral but slightly slower at once3 p50`1.00129-1.00169`. | [d768-in](ada-triad-tf32-nt-a-ldmatrix-20260908/report.md), [out/Prism](ada-triad-tf32-nt-a-ldmatrix-siblings-20260908/report.md), [sliced d768-out](ada-triad-tf32-nt-a-ldmatrix-sliced-20260909/report.md), [sliced large_deep (legacy path)](ada-triad-tf32-nt-a-ldmatrix-sliced-prism-20260909/report.md), [sliced d768-in stop](ada-triad-tf32-nt-a-ldmatrix-sliced-in-20260909/report.md), [canonical Prism full-domain stop](ada-triad-tf32-nt-prism-full-domain-20260909/report.md), [canonical Prism K8-regpipe stop](ada-triad-tf32-nt-prism-k8-regpipe-20260909/report.md), [shared RNA](ada-triad-tf32-nt-shared-rna-20260908/report.md), [A+B](ada-triad-tf32-nt-compact-ab-ldmatrix-20260908/report.md) |
| TF32 TN large shapes | Whole raw-transpose+N96 reduces actual AUTO time29–30% in,40% out,15–17% Prism. A-only transpose-RNA wins all three retained cells. NEW canonical-Prism M64N64/S3 wave body then beats N96 by8.1–8.8% p50 and5.3–7.4% p95, with83regs/local0/49,152B/occupancy2; it remains1.68–1.74x slower than Fast. An M64N128/S2 reuse follow-up keeps occupancy2 but loses M64N64 by1.1–3.2% p50, proving the96-CTA grid is too small. Joint A+B preprocessing loses1.6–2.2%. | [Earlier in](ada-triad-tf32-tn-transpose-n96-20260908/report.md), [earlier out/Prism](ada-triad-tf32-tn-transpose-n96-siblings-20260908/report.md), [d768-out A-RNA winner](ada-triad-tf32-tn-transpose-rna-n96-d768-out-20260909/report.md), [in/Prism A-RNA winners](ada-triad-tf32-tn-transpose-rna-n96-siblings-20260909/report.md), [Prism M64N64 winner](ada-triad-tf32-tn-transpose-rna-m64n64-prism-20260909/report.md), [M64N128 stop](ada-triad-tf32-tn-transpose-rna-m64n128-s2-prism-20260909/report.md), [A+B stop](ada-triad-tf32-tn-transpose-ab-rna-n96-d768-out-20260909/report.md) |
| TF32 TN direct-N96 d768-in | Direct staging removes transpose but still loses Fast graph35–37%. Not paired against retained transpose+N96; no new-best claim. | [Direct probe](ada-triad-tf32-tn-direct-n96-20260908/report.md) |
| Exact F32 TN d128-in/out | Direct fixed-order fold substantially improves AUTO; still roughly2–3x Fast. | [In](ada-triad-f32-tn-d128-direct-20260908/README.md), [out](ada-triad-f32-tn-d128-out-direct-20260908/README.md) |
| Exact F32 TN large | d768-in dual-chunk fused wins retained4.0–4.4%; Fast remains2.30–2.33x ahead. Canonical Prism direct M64N64/BK16+reducer wins AUTO21.3–21.4%; Fast remains2.40x. NEW d768-out direct M64N64/BK16 removes transpose+four CopyPlan launches, reduces six nodes to two and beats retained13.1–15.0%; Fast remains2.35–2.36x. One-node M64N32 loses12.0–12.2%; vector reducer stays parity; BK32 exact-tail loses3.8–4.0%. | [dual-chunk retained-best](ada-triad-f32-tn-dual-chunk-20260909/report.md), [d768-in CopyPlan](ada-triad-f32-tn-splitm-copyplan-20260908/report.md), [out/Prism](ada-triad-f32-tn-splitm-copyplan-siblings-20260908/report.md), [d768-out direct winner](ada-triad-f32-tn-direct-d768-out-bk16-20260909/report.md), [Prism direct winner](ada-triad-f32-tn-direct-prism-bk16-20260909/report.md), [Prism fused N32 stop](ada-triad-f32-tn-direct-prism-fused-n32-20260909/report.md), [vec4 reducer stop](ada-triad-f32-tn-prism-vec4-reducer-20260909/report.md), [BK32 exact-tail stop](ada-triad-f32-tn-direct-prism-bk32-exact-tail-20260909/report.md) |
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

Latest bounded follow-ups on2026-09-09 do not add a winner. BF16 TN d768-out
M64N96 reaches118regs and equal resident warps but loses retained0.55–0.91%
at p50. Exact-F32 TN d768-in direct BK16 raw+reducer is exact at107regs/
occupancy4 but loses the retained transpose+dual-fused pipeline10.8–11.1%.
For TF32 NN Prism, on-demand copy-address computation worsens M96N96/S2 to
140regs; a separate single-fragment refinement reaches125regs/occupancy2 and
passes exact, but loses direct N96 by0.77–1.71% p50. All timing candidates stop
at once3 before Fast. Evidence: [BF16 TN M64N96](ada-triad-bf16-tn-m64n96-regpipe-vec2-20260909/report.md),
[exact-F32 d768-in direct](ada-triad-f32-tn-direct-d768-in-bk16-20260909/report.md),
[TF32 copy lifetime](ada-triad-tf32-nn-m96n96-s2-copy-lifetime-prism-20260909/report.md),
[TF32 single fragment](ada-triad-tf32-nn-m96n96-s2-single-fragment-prism-20260909/report.md).

The next four bounded mechanisms also stop without a new winner. TF32 N96
`__grid_constant__` Params is exact/resource-neutral but loses7.4–7.6%;
cross-BK fragment prefetch rises124->149regs. BF16 TN one-wave atlas models
29.69% less staging but compiles at130regs over its128 cap. Exact-F32 TN
M128N64/BK16 dual-fused reduces modeled staging25%, but its raw verification
twin retains an8-byte stack frame after one bounded scalarization fix. Evidence:
[grid constant](ada-triad-tf32-nn-n96-direct-grid-constant-prism-20260909/report.md),
[cross-BK prefetch](ada-triad-tf32-nn-n96-cross-bk-prefetch-prism-20260909/report.md),
[one-wave atlas](ada-triad-bf16-tn-one-wave-atlas-20260909/report.md),
[M128N64 dual-fused](ada-triad-f32-tn-m128n64-bk16-dual-fused-20260909/report.md).

Two final reuse/transpose checks also stop at once3. The canonical-Prism TF32
TN M64N64 wave winner does not transfer to d768-out: despite 83regs and
occupancy2, it loses the retained N96 route 27.9–29.3%. The exact-F32 TN
32x8/256-thread transpose is bit-exact and resource-clean at 24regs/occupancy6,
but improves the retained 32x16 pipeline only 0.10–0.15% at p50 and misses the
strict 1% gate in every stratum. Evidence: [TF32 d768-out reuse](ada-triad-tf32-tn-transpose-rna-m64n64-d768-out-20260909/report.md) and
[exact-F32 32x8 transpose](ada-triad-f32-tn-transpose-32x8-d768-in-20260909/report.md).

The final BF16 TN d768-in packed-CTA-raster check is exact and resource-neutral
at 125regs/local0/32KiB/occupancy3, but stays within roughly ±0.15% of the
retained linear raster. It misses the strict 1% gate and stops once3 before
Fast. The current discovery shortlist is now frozen for joint production-AUTO
integration. Evidence: [packed raster](ada-triad-bf16-tn-packed-raster-d768-in-20260909/report.md).

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
