# RTX 5090 model training graph check — 2026-09-10

CUDA13.2, RTX5090 CC12.0/170, driver595.84. Source snapshot:
`/root/mamba-release-sm120.J6z6IG`, frozen before the SM120 symbol-reachability
repair. The exact source inventory is in `assembly-source.sha256`. This is
the same SM120 routing source as the passing66-cell CUDA13.2 assembly matrix,
not a claim about a future final release commit.

Command:

```sh
cargo test --locked --release --no-default-features --features cuda \
  --test f32_training_graph_parity -- --nocapture --test-threads=1
```

Result:20 passed,0 failed,1 ignored in53.87s. The target includes18 passing
shared host tests as well as the two model tests. The ignored shared test is
the live CUDA-ordinal/NVML-UUID check; it is not either model graph check.

| Model test | Result | Sampled-weight max absolute difference |
| --- | --- | ---: |
| `m1_f32_training_graph_matches_eager` | PASS | 0 |
| `m3_f32_training_graph_matches_eager` | PASS | 0 |

Both explicitly select batch-invariant Triad, exact-F32 policy, tensor cores
off and fast GEMM off. Both capture and replay a forward/backward/AdamW step,
record a nonempty custom GEMM manifest, reject replay through a foreign
context, and compare two selected weights with eager. Mamba also checks
captured-route drift rejection. Five idle/free-memory preflight samples and
the complete test/build logs are retained.

These are tiny synthetic model integration tests: d_model32, batch1, sequence4
for Mamba and64 for Mamba-3. The assertions use a1e-5 tolerance on sampled
weights. Observed zero difference does not establish whole-model bit equality,
all production GEMM shapes, half tensor-core winners, or successful custom
inference graph replay. This is not performance evidence.

The initial wrapper expected a two-test Cargo summary and therefore its final
grep exited unsuccessfully after the actual test binary passed. The wrapper
was corrected to accept the complete shared-test count; the valid GPU run was
not repeated. `training-f32.log` is the authoritative test result.
