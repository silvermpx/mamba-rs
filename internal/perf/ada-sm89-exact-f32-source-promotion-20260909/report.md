# Ada SM89 exact-F32 large-TN source promotion (Batch B0)

Date: 2026-09-09

## Scope

Source-only promotion of exactly three retained Ada exact-F32 TN winners. This
batch deliberately does not add module ownership, compilation, dispatcher,
launch, graph, selector, or d128 integration.

| Cell `(M,K,N)` | Production export | Frozen source | Contract |
| --- | --- | --- | --- |
| `(2048,768,3072)` | `gemm_bi_tn_sm89_f32_n64_dual_chunk_fused_finalize_v1` | `triad_f32_tn_copyplan_dual_chunk_source::compose_fused_source` | two ordered 1024 chains, fused sequential FP64 finalize, 128 threads, 32768 static shared bytes, register cap 168, occupancy gate 3 |
| `(2048,1536,768)` | `gemm_bi_tn_sm89_f32_m64n64_bk16_s2_d768_out_raw_v1` | `triad_f32_tn_direct_d768_out_bk16_source::compose_source` | four ordered 512 chunks, raw partial output, 128 threads, 16384 static shared bytes, register cap 128, occupancy gate 4 |
| `(4621,384,1928)` | `gemm_bi_tn_sm89_f32_m64n64_bk16_s2_prism_raw_v1` | `triad_f32_tn_direct_prism_bk16_source::compose_source` | six ordered 784/784/784/784/784/701 chunks, raw partial output, 128 threads, 16384 static shared bytes, register cap 128, occupancy gate 4 |

The sealed CUDA owner stores one dual-chain template, its exact fused
epilogue, and one direct raw template. The Rust adapter expands only the three
production symbols and the frozen fused epilogue. The two direct cells retain
distinct exports even though their arithmetic bodies are identical.

Owner SHA-256:
`cdcb768216699f41553e73492a32d92717c62889a4a329ca1990360b361541c7`.

## TDD and verification

RED was observed before implementation: the scoped integration test failed to
compile because `sm89_exact_f32_source.rs` did not exist.

GREEN command (run from a temporary native-Cargo mirror of the worktree):

```text
/Users/silvermpx/.cargo/bin/cargo test --manifest-path <scratch>/Cargo.toml \
  --no-default-features --test gemm_bi_sm89_exact_f32_source_contract -- --nocapture
```

Result: 12 passed, 0 failed. The suite checks normalized frozen-body parity,
exactly three callable exports, no discovery placeholders, shape/partition,
launch/resource/ABI metadata, ordered F32 FMA chains, sequential FP64 fused
finalize, and absence of reduced-precision/atomic mechanisms.

No CUDA compiler, remote host, or GPU was used in this batch.
