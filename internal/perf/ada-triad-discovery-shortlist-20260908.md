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
| F16 NN d768-in | S3 beats TC128; Fast still2–4% ahead. Pipeline/Swizzle are weaker. | [S3](ada-triad-half-nt128-f16in-20260908/README.md), [loaded losses](ada-triad-half-nn-loaded2-20260908/README.md) |
| Half NT d768-out | B-XOR S3 is26% below TC64; graph Fast remains2–6% ahead. Direct epilogue loses retained S3 by14.5–15.6%; keep vector epilogue. | [NT out](ada-triad-half-nt-fixed-s3-bxor-20260908/report.md), [epilogue loss](ada-triad-half-nt-direct-epilogue-20260908/report.md) |
| Half TN d768-in | New regpipe-float2 screen reaches graph parity: Fast p50 .9988–1.0022, worst p95 1.0068. No strict Fast win and no paired compact/regpipe ranking. M128N128 Fixed-S3 loses current TC64; excluded. | [Compact](ada-triad-half-tn-compact-20260908/README.md), [float2 parity](ada-triad-half-tn-regpipe-vec2-20260908/report.md), [S3 loss](ada-triad-half-tn-fixed-s3-20260908/report.md) |
| TF32 NN d768-out / Prism | N96 reduces AUTO time; near Fast parity, not a strict win. N96 d768-in loses and is excluded. | [N96](ada-triad-nn-addhalf-n96-screen-20260908/README.md) |
| TF32 TN large shapes | Whole raw-transpose+N96 reduces actual AUTO time29–30% in,40% out,15–17% Prism. Graph/Fast respectively1.34,1.20,2.00; no Fast win. | [In](ada-triad-tf32-tn-transpose-n96-20260908/report.md), [out/Prism](ada-triad-tf32-tn-transpose-n96-siblings-20260908/report.md) |
| TF32 TN direct-N96 d768-in | Direct staging removes transpose but still loses Fast graph35–37%. Not paired against retained transpose+N96; no new-best claim. | [Direct probe](ada-triad-tf32-tn-direct-n96-20260908/report.md) |
| Exact F32 TN d128-in/out | Direct fixed-order fold substantially improves AUTO; still roughly2–3x Fast. | [In](ada-triad-f32-tn-d128-direct-20260908/README.md), [out](ada-triad-f32-tn-d128-out-direct-20260908/README.md) |
| Exact F32 NN large / NT d768-out | Preserve the earlier CopyPlan reuse finalists and their existing three-toolkit evidence; these are not new Fast wins. | [Live reuse](ada-triad-live-reuse-20260908/README.md) |

This table is not exhaustive closure of60 operation/dtype/shape cells. The old
[whole-matrix snapshot](ada-triad-state-20260907/report.md) remains historical;
do not merge its independent quantiles with these paired screens to invent
an average speedup or claim that the remaining cells are finished.

## Integration and cleanup boundary

- Selected half NN uses the existing Fixed S3 holder; NT B-XOR and new TN
  source adapters currently live only in tests/support. Their public physical
  identities and precision/shape/toolkit guards still need joint integration.
- Preserve measured source by its committed hash/snapshot before another
  owner changes a shared harness. Never qualify a different live source as
  if it were the measured one.
- Valid losing experiments stay in the separate evidence/test area for now.
  No Ada loss authorizes removing a route used on another shape or GPU.
- Neither source availability nor a CUDA13.2 win proves all-architecture or
  CUDA12.8/13.0 performance. The final batch must verify those supported
  toolkits, preserve fallbacks and reject foreign device-specific identities.
