# Mamba-1 fold qualification

This manual target compares the functions selected by `MambaKernels` with an
independent frozen Fixed-module composition. It does not install candidates
or change the model's sequential/parallel scan policy.

`legacy_fixed.cu` is the complete Ada Fixed composition at capacity 32, where
no fold admission applies; its fold kernels are the ones that shipped before
fold admission, and the file is regenerated from the capacity-32 composition
whenever the composed source moves (a kernel rename moves it without moving
a bit). SHA-256, the capacity-32 source digest the test pins:
`1e6c83d81ee034fd0fdee498cfcb12b4d9bd52c3070238332b38b6820f043d61`.
The full source is intentional: compiling just the fold function would change
its compiler context. It is a test fixture, not another production module.

Run on an idle RTX 6000 Ada with native `sm_89` and the selected NVRTC library:

```sh
cargo test --release --features cuda,qualification \
  --test m1_fold_transport_qualification -- --ignored --nocapture --test-threads=1
```

The default checks capacity16 raw outputs in F32, BF16 and F16, with full and
slim tapes. It also checks that capacities32/64 keep the legacy function
selection, that capacity32 still compiles the legacy source digest, and that
capacity64 compiles the qualified capacity-64 composition (the retained
inference overlay on the same base). Fixtures start from a nonzero recurrent state and use
the frozen forward kernel to make saved inputs. Each arm owns separate input
and output allocations with guards. All six gradient outputs must match byte
for byte, inputs must remain unchanged, and a one-node graph must reproduce
the result after output poisoning.

Optional controls:

- `M1_FOLD_MODE=compare`: paired ABBA graph-only timings, with exact output
  checks before and after timing. Default `check` does no timing.
- `M1_FOLD_CAPS=16`: reuse the capacity16-only numerical lane after the full
  capacity selector check. This is not evidence for the omitted capacities.
- `M1_FOLD_DTYPE=all|half|f32|bf16|f16`, default `all`.
- `M1_FOLD_TAPE=both|full|slim`, default `both`.
- `M1_FOLD_SHAPES='B,T,DI,DS;...'`: a bounded shape list.
- `M1_FOLD_NODES`, `M1_FOLD_REPLAYS`, `M1_FOLD_ROUNDS`: positive timing counts.
- `M1_FOLD_NEGATIVE_SKIP=1` omits the candidate eager launch;
  `M1_FOLD_NEGATIVE_GRAPH=1` captures the baseline arm twice. Both must fail
  the relevant byte comparison. They must never be set in a positive run.

Archive compiler identities, selected routes, launch resources, all raw
results, and source/binary hashes with each run. Repeat final-artifact
acceptance for CUDA12.8,13.0,13.2; do not infer one toolkit's results from
another. Kernel timings are not whole-model training speedups.
