# RTX 5090 assembled-route matrix

These are assembly diagnostics, not paired cuBLAS results. Each successful
cell produces one eager and one graph row. Source and device manifests are
stored beside the corresponding logs.

| Toolkit | Completed cells / rows | Outcome |
|---|---:|---|
| CUDA13.2 | 66 / 132 | Pass; every eager/graph equality flag is true |
| CUDA12.8 | 66 / 132 | Pass; every eager/graph equality flag is true |
| CUDA13.0, initial | 50 / 100 | Stopped at the unavailable exact NT d768-out symbol |

CUDA12.8 passing this matrix did not prove that its fastest TF32 routes were
admitted: the unmatched TF32 evidence cohort selected the exact-F32 floor.
Fresh retained-route qualification for CUDA12.8/13.0 is tracked separately in
`../sm120-tf32-retained-20260910/`.

The CUDA13.0 symbol-reachability repair and focused two-policy rerun are in
`../sm120-symbol-reachability-20260910/`. That rerun passes four eager/graph
rows, preserving the measured two-part reduction. It does not complete the
other14 cells that the initial matrix never reached. Those remain required
in the post-admission assembly check.

`cuda132/compile-initial/` preserves an earlier formatter enum-coverage build
failure. The corrected source subsequently passed the full CUDA13.2 matrix;
the failed build is not a GPU correctness failure.
