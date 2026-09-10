# Ada TF32 final-candidate screen — 2026-09-10

Board: NVIDIA RTX 6000 Ada Generation, CC 8.9, CUDA 13.2. Each GPU run
started only after five consecutive samples at 0% compute, 0% memory
utilization, and 48 MiB resident memory. Timings are paired ABBA/BAAB screens;
lower ratios are faster. All candidates passed target-shape exact eager/graph
output bits, guard zones and input immutability. The TN protocol additionally
checked tails, exceptional values and zero reduction. The focused NT N96
protocol did not run those off-target probes; do not infer their coverage from
the broader NT harness. Its proposed AUTO admission is exact-shape only.

## Decisions

| Cell | Candidate | Comparator | Eager ratio p50/p95 | Graph ratio p50/p95 | Decision |
|---|---|---|---|---|---|
| TF32 NT d768-in `(2048,768,3072)` | A-ldmatrix M128N96/BK32/S3 | public AUTO compact M128N64/BK32/S2 | `0.92783–0.92823` / `0.92853–0.92927` | `0.92968–0.93014` / `0.93040–0.93053` | promote for the exact CUDA 13.2 Ada cell |
| TF32 TN Prism `(4621,384,1928)` | pre-RNA M64N96/BK32/S2 | retained production M64N64/BK32/S3 | `0.80537–0.82003` / `0.83540–0.85558` | `0.81790–0.82074` / `0.83866–0.83950` | promote for the exact CUDA 13.2 Ada cell |
| TF32 NT Prism `(4621,384,1928)` | A-ldmatrix M128N96/BK32/S3 | scalar-A M128N96/BK32/S3 | `0.98687–0.98710` / `0.98736–0.98758` | `0.98674–0.98688` / `0.98744–0.98768` | reject: both N96 bodies are much slower than public AUTO |

The promoted candidates do not beat cuBLAS Fast on these cells. The selection
criterion for this assembly pass is an exact, repeatable improvement over the
current deterministic production AUTO route; cuBLAS Fast gaps remain future
optimization targets.

## Resource receipts

- NT M128N96/S3 A-ldmatrix: 110 registers, 86,016 B dynamic shared memory,
  one resident CTA.
- TN M64N96/S2: 96 registers, 40,960 B dynamic shared memory, two resident
  CTAs. Its shared-B stride defect was corrected from 64 to 96 before this
  successful run.

## Evidence

- `evidence/ada-tf32-nt-a-ldmatrix-n96-s3-d768-in-auto-20260910.log`
- `evidence/ada-tf32-nt-a-ldmatrix-n96-s3-prism-20260910.log`
- `evidence/ada-tf32-tn-pre-rna-m64n96-s2-prism-fixed-20260910.log`
