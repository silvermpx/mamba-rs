# Ada half TN microtiles: short discovery, 2026-09-08

CUDA13.2 only; one exact eight-cell GPU test passed. M32N32 and M16N32
BK32/S3 candidates preserve TC64's ascending K16 MMA order and F32 dW
epilogue. Eager/graph repeated outputs match TC64 exactly for both F16/BF16,
both d128-in `(1024,128,512)` and d128-out `(1024,256,128)`. Inputs and guards
are checked; Fast has independent finite/nonzero and repeat-bit checks.

| Candidate / shape | F16 candidate/Fast p50 range | BF16 candidate/Fast p50 range |
| --- | ---: | ---: |
| M32N32 / d128-in | 1.421–1.553 | 1.422–1.558 |
| M32N32 / d128-out | 1.571–1.832 | 1.615–1.849 |
| M16N32 / d128-in | 1.540–1.695 | 1.531–1.698 |
| M16N32 / d128-out | 1.687–1.973 | 1.729–1.988 |

All eight are **Fast losses**, not champion admissions. Ratios cover
eager/graph × ABBA/BAAB, seven paired brackets, twenty beta1 GEMMs per
observation; C is restored before each observation. Exact twenty-call
candidate and Fast outputs are checked after timed observations.

M32N32 has a roughly .69 candidate/TC64 ratio in the two explicitly UNPAIRED
diagnostics. This suggests a useful internal improvement, but is not a
paired-confirmed current-route win. Retain the measured source as a possible
finalist; do not present it as a cuBLAS Fast winner. M16N32 is the weaker arm.

Resources: M32N32 128 threads, 92 registers, 15360 static shared bytes,
occupancy5; M16N32 64 threads, 116 registers, 12288 static bytes, occupancy7.
Both have zero local and dynamic shared bytes. Production CUDA is unchanged.

Measured main SHA `1721ab31dfecaa138ea16d64551a7ff8588093a42a9e58e13640215fa67e5c85`;
helper SHA `b1d7337982d2b0d26d1b536f710487af6e8adabf4d07762266b5f00e7ed3745e`.
Raw `evidence/final/test.log`, SHA
`3df7af8993d622a47e366a98383ba3b760acbb1519070e14456c0ff1f2e8fe64`.
Root independently recomputed 224 brackets / 896 timed observations. Exact
test count1, 64 correctness rows, eight resource and decision rows, 32 timing
rows; cache unchanged and PRE/DRAIN quiet. Native adapter5/5 also passes.

Preserved harness failures: initial Rust pointer-lifetime/unit-result errors;
runtime1 lacked the required CUDA source preambles; runtime2 incorrectly
required the vendor graph to contain exactly one node. Its actual two-node
native-half cuBLAS graph is legitimate. Final validation keeps one node for
each custom arm and permits a nonempty vendor graph. Failures are not kernel
performance losses. No once21/all-toolkit/full-suite promotion run was made.
