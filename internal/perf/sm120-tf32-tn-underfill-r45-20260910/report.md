# Retained RTX 5090 TF32 TN underfill admission

This archives the previously collected receipt accompanying the assembly WIP;
it is not a new measurement from the final Ada integration run.

- Logical TN dimensions `(M,K,N)=(256,512,384)`; alpha1, beta1, no bias,
  exact contiguous/aligned operand cohort.
- RTX5090, CC12.0/170 SMs, CUDA13.2, driver595.84 qualified cohort,
  tuning revision45. Complete compiler/artifact/device identity is in the raw
  selector receipt.
- Retained selection: `gemm_bi_tn_sm80_mma_tf32_v1_m16n32_bk32_s4`.
- The paired protocol used21 discovery and101 final windows per order.
  Final reported median speedup against the scalar comparator: eager1.80806x,
  graph1.99966x; p05 speedup1.78635x/1.99902x. This is not a cuBLAS Fast ratio.
- The receipt marks this cell qualified/admitted. The separate actual-AUTO
  log confirms the production symbol on eager and graph. Its one timing window
  is routing evidence, not a fresh statistically supported speedup claim.
- The later66-cell assembly smoke retains this route. Other physical routes
  in that smoke are unchanged relative to the saved5090 snapshot.

The admission is specific to the frozen595.84/CUDA13.2 cohort, not a blanket
extension to every Blackwell GPU or toolkit. The live manifest now has24cells,
including6 portable routes; the old23/5 test expectation was corrected.
