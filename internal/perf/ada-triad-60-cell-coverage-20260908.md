# Ada Triad 60-cell coverage audit — 2026-09-08

CUDA13.2 / RTX6000Ada only. This is an evidence index, not an average-speedup
claim or release qualification. Five shapes are `i128=(1024,128,512)`,
`o128=(1024,256,128)`, `i768=(2048,768,3072)`,
`o768=(2048,1536,768)`, and `P=(4621,384,1928)`.

> **Scope correction:** this matrix tracks logical Triad workload/candidate
> evidence, not production-Triad physical-route ownership or admission. Of the
> ten `Y` cells, five use Fixed NN kernels with the five-argument Fixed ABI and
> five use test-only NT adapters; zero are already integrated production-Triad
> replacements. The authoritative historical inventory of the public AUTO
> paths is the [current21 production-route smoke](ada-triad-current21-20260906/smoke-inventory.md),
> which records production Triad physical routes for all 60 cells.

| `Y` candidate ownership | Cells | Physical status |
| --- | ---: | --- |
| Fixed NN, F16/BF16 | 5 | Existing Fixed five-argument holder, not a production-Triad admission |
| Test-only NT adapter, F16/BF16 | 5 | Test-only seven-argument symbol, not a production-Triad admission |
| Integrated production-Triad replacement | 0 | None of the ten `Y` candidates is integrated yet |

Legend: **Y** = strict paired cuBLAS Fast win in every eager/graph x ABBA/BAAB
p50+p95 stratum; **N** = explicit paired Fast evidence exists but strict win
fails; **U** = no qualifying paired Fast evidence. Retained-best improvements
remain N when cuBLAS Fast still wins. Historical independent eager quantiles do
not change U into Y/N. These letters describe the logical workload result only;
they do not encode kernel source family, ABI or dispatcher admission.

| Precision / op | i128 | o128 | i768 | o768 | P |
| --- | ---: | ---: | ---: | ---: | ---: |
| exact F32 NN | U | U | N* | N* | N* |
| exact F32 TN | N | N | N | N | N |
| exact F32 NT | U | U | N | U | N |
| TF32 NN | U | U | N | N | N |
| TF32 TN | N | N | N | N | N |
| TF32 NT | U | U | N | N | N |
| F16 NN | U | U | N | **Y** | **Y** |
| F16 TN | N | N | N | N | N |
| F16 NT | U | U | **Y** | **Y** | **Y** |
| BF16 NN | U | U | **Y** | **Y** | **Y** |
| BF16 TN | N | N | N | N | N |
| BF16 NT | U | U | **Y** | **Y** | **Y** |

Totals: **11 Y**, **32 N**, **17 U**. The three exact-F32 NN `N*` rows have
valid paired no-win evidence but their candidate timing is provisional because
the original wrapper retained an exclusive Fixed holder; rerun the repaired
holder-lifetime wrapper before any admission claim.

The F16 and BF16 NT d768-out M96N128/BK64/S3 candidates are now strict Fast
winners in all four once7 strata, moving both cells from N to Y. The latest half-TN sibling
screen converts four former U cells (F16/BF16 x
o768/P) to N. BF16 o768 gains1.6–1.8% over retained compact but remains slower
than Fast; the other three stop. TF32 NT o768/P now have A-only retained-best
improvements of9.4–10.0% over actual AUTO, but remain N because Fast is still
1.40–1.89x ahead. TF32 NN N96 B-ldmatrix is a valid exact loss on o768/P and
does not change their existing N status. Exact-F32 TN i768 is now N: a new
two1024-sample CopyPlan pipeline preserves the AUTO SplitM+FP64-reducer bits
and improves actual AUTO by14.5-15.4%, but cuBLAS Fast remains2.58-2.60x ahead.
The earlier full-K CopyPlan probe remains rejected because it changed the
selected arithmetic tree and failed exact bits before timing. Exact-F32 TN
o768 now retains the same exact SplitM CopyPlan mechanism with a9.5-11.5% AUTO
improvement; Prism's exact candidate loses AUTO34.8-36.4%. Both have paired
Fast evidence and therefore close from U to N.

## Highest-value next work

1. Integrate and jointly qualify the ten strict winners plus all distinct
   retained-best improvements; do not delete routes used by other shapes/GPUs.
2. Resolve exact-F32 blind spots: NN i128/o128, NT i128/o128 and paired Fast for
   NT o768. For TN i768/o768/P, preserve each selected SplitM partition and its
   FP64 reducer; a full-length F32 chain is a different bit contract.
3. Integrate the three TF32 NT A-only retained-best cells. Do not retry the
   measured NN N96 B-ldmatrix loss; small TF32 shapes still need profiling.
4. After the discovery shortlist is frozen, perform one combined dispatcher
   integration and CUDA12.8/13.0/13.2 qualification batch. CUDA13.2 source
   availability alone is not cross-toolkit or cross-architecture proof.

Detailed measurements and links are in
[the discovery shortlist](ada-triad-discovery-shortlist-20260908.md). The base
historical matrix is [the state snapshot](ada-triad-state-20260907/report.md);
newer focused reports supersede it only for their exact cells and cohorts.
