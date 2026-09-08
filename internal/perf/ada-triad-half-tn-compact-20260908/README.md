# Half TN compact BK64/S2: near parity in, losses out/Prism

Ada / CUDA13.2, 2026-09-08. Six cells: three shapes x F16/BF16.
Shared-layout compaction preserves original K16 MMA ordering and F32 dW epilogue.

| Shape / dtype | Candidate/Fast eager p50 | Candidate/Fast graph p50 |
| --- | ---: | ---: |
| d768-in F16 | .976–.980 | 1.016–1.016 |
| d768-in BF16 | .972–.974 | 1.011–1.012 |
| d768-out, both | 1.085–1.134 (all strata) | loss |
| Prism, both | 1.355–1.404 (all strata) | loss |

All six STOP vs Fast; do not rerun unchanged candidates. Current diagnostics
(.835–.958) are unpaired, NOT a robust current-win claim. Candidate resources:
128 threads, 32768 static shared bytes, local0, occupancy3. 256B guarded
alignment, eager/graph exact current bits and Fast own-bit checks pass.

Attempt1 was a harness compile error (u32 occupancy compared with i32), not
a kernel loss; no GPU run. Owner fixed the type only, with the already approved
BF16 NN singleton sharing the final build. Preserve the failed build log.
Exact `ada_half_tn_tc64_bk64_s2_compact_three_cell_vs_current_and_fast_discovery_once7`
PASS in attempt2: 6 resources / 48 bits / 24 screens / 6 decisions.
Root replayed 168 brackets / 672 observations and all quantiles. No full gates.

Main `e77e1961d03d7f26d19ffd1a5fe6514127ead50871768f44ad4eab6d70d160b6`;
helper `a8edb14a581ed7ed5f50cb52f5849c55a4184383bfbc5b34e3ed98ef104a1b36`.
[Raw](evidence/attempt2/once7-cuda132/test.log), SHA256
`bc4da63989621ac4fd30fe6fdc700cbdcf55ac4884558934115c734b56f0b21d`.
