# Ada half three-arm source checkpoint

Scope is test-only. No production loader, dispatcher, arithmetic provider, or
selector was edited. No local Cargo, CUDA, GPU, or SSH command was run.

Frozen source SHA-256 values:

- `tests/gemm_bi_fixed_half_batch_discovery.rs`: `1604c00de80ab03a819881500b696422dfac2ef189535f12441ce20c2325f079`
- `tests/gemm_bi_fixed_half_batch_discovery.cu`: `9345a184cd50dd744a72e2ff0e784fe4440493616a03f55364f094df8e388a32`
- `tests/support/fixed_half_batch_layout.rs`: `a72b6b38bb44851ead5ec45b7fdc822842b845c218b6db5738f72435127c0ecb`

Standalone host behavior tests: RED 7/7 failed, GREEN 7/7 passed. A review
regression then exposed the nonzero-stage fragment-XOR precedence bug (RED
7 pass / 1 fail); the corrected helper is GREEN 8/8. The schedule test now
asserts the independent current-S3 transition order and rejects a reordered
last-MMA mutant rather than comparing two aliases of one model.

GPU-owner build command (under the accepted CUDA 13.2 / private-cache env):

```text
cargo test --release --features cuda --test gemm_bi_fixed_half_batch_discovery --no-run
```

GPU-owner single batch command:

```text
MAMBA_FIXED_HALF_BATCH_DISCOVERY=1 cargo test --release --features cuda --test gemm_bi_fixed_half_batch_discovery fixed_half_three_arm_batch_discovery -- --ignored --exact --nocapture
```

The harness validates each arm independently, and a resource, correctness, or
timing rejection is recorded without aborting the other arms. It runs only the
four approved CUDA 13.2 no-bias cells and uses actual AUTO S3/Swizzle/Pipeline
controls plus the native-half Fast facade.
