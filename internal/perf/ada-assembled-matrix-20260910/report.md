# Ada assembled AUTO route matrix — 2026-09-10

Source commit `14b72136`, CUDA13.2, RTX6000Ada CC8.9/142. Exact device,
source manifest and full raw logs are under `cuda132/`.

The frozen committed source was built in `/root/mamba-ada-assembled.MqPPl6`.
The existing `gemm_bi_deterministic_performance_matrix` test ran with
`GEMM_BI_QUAL_WINDOWS=3`, path order `ab`, all60 canonical cells plus six
TF32 underfill/large-deep cells. Result:1 passed,0 failed in9.34s; build55.85s.

Root independently parsed132 rows across66 unique cells, allCC8.9 and all
`eager_graph_equal=true`. Both exact-F32 TN d128 rows select the newly admitted
one-node direct-fold symbols, not the prior SplitM64 multi-launch path.

This is a combined production-route/eager-graph smoke check after assembly.
Its timings pair eager with graph for AUTO; they are not paired AUTO/cuBLAS
measurements and do not establish a Fast win or an average release speedup.
Keep the separate candidate qualification receipts and forthcoming explicit
cuBLAS Fast/Pedantic tables for those claims. Lower-toolkit focused admissions
remain documented in their individual assembly reports.
