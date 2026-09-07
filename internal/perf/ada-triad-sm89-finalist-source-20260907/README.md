# Isolated Ada NT finalist source checkpoint

This checkpoint transfers the selected compact8/S2 CUDA source from discovery
into a pure production source builder. It does not register/load the new
module, change AUTO, compile CUDA or claim new performance evidence.

The candidate is the one retained for three heavy NT TF32 cells in
`ada-triad-nt-compact32-eight-warp-s2-20260907` and
`ada-triad-nt-sibling-two-mechanism-20260907`. The builder reuses immutable
SM80 source with eight count-one transformations; every missing/duplicate
anchor rejects. Only the new helper/target names and helper comments differ
from the measured candidate. Existing production CUDA fragments stay unchanged.

Root's independent `replay-source.rb` checks eight frozen reference file hashes
and exact equality of the complete composed translation unit after those
allowed name/comment changes. Production composed SHA256:
`d20cb39f390ef45181be94e4bd7baeb51548c0e6efac64bad72aae71a889e3ce`,
203,981 bytes. This is a plain SHA256 of actual source bytes, not the later
framed compiler invocation/cache identity.

Native TDD started with four intended failures after successful compilation.
A root pre-review additionally caught a private composition helper parameter
being validated but not emitted; a focused marker test failed before the fix
and passed after threading the selected helper through. Production composed
bytes did not change. Final native tests:28/28; native Rust with the CUDA cfg
flag:5/5 structural tests. The latter is not CUDA compilation or GPU testing.

Reproduce from the worktree root:

```sh
ruby internal/perf/ada-triad-sm89-finalist-source-20260907/native-tests.rb
ruby internal/perf/ada-triad-sm89-finalist-source-20260907/replay-source.rb
```

Source-builder registration, strict PTX/Driver/resource validation, separate
module holder, physical route, three-toolkit qualification and AUTO admission
remain the next tasks in
`internal/gemm-bi-triad-ada-finalist-integration-2026-09-07.md`.
