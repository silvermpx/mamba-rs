# Ada RNA cross-toolkit full qualification report

Date: 2026-09-06. Result: the test-only corpus change is qualified on Ada
with matching CUDA 12.8, 13.0, and 13.2 builds. This report does not authorize
AUTO widening.

## Scope and source identity

- Production source base was `208c740c3f24c71f8bc090dbf597f7ee1a129fbd`.
  While this task ran, the main worker committed census evidence only as
  `f3248a16a2b64d4790f6fb7a99a04f0d3bdd39ad`; production source remained
  unchanged.
- The only source diff is `tests/gemm_bi_fixed_correctness.rs`, SHA256
  `7f0566985b7c80ed858258104f166fc46bc7328cd8548eab778fae94e4eb977f`.
- Remote source: `/root/mamba-ada-rna-full-qualification-20260906`.
- The 167-build-input source manifest has SHA256
  `44f1a4632512169d56fea483df599507e7cd718cb4853640f2eb3ff733002105`.
  It matches the local inputs exactly and includes
  `tests/support/fixed_sm89_exact_n64_admission.rs` with SHA256
  `22c9d2c9b045f6e40d1f8fdf7cf380e0928911ad2b19282f8a12a2efa5aaaa2f`.
- Remote source, targets, and caches were newly created mode 0700. Frozen
  census/13.2 source, targets, and caches were read only for the final identity
  comparison.

The change makes `rna_wide_qualification_shapes()` unconditionally return the
same ordered tail/A-E corpus. Both force and 13.2 actual-AUTO wrappers now use
the full tail prefix matrix, the common seven-view matrix for every non-tail
family, and both C4/C16 output offsets. `actual_auto` remains only around the
actual-AUTO launch/selection assertions and the existing misaligned-A/B AUTO
fallback checks. The separate forced unsafe-input/K0 test is unchanged. A
success marker is emitted only after all five incumbent comparisons, full
guard comparison, two eager repeats, graph geometry/ABI assertion, and two
poisoned graph replays have succeeded (and after the optional AUTO checks in
the 13.2 wrapper).

## TDD evidence

The first implementation extracted the old conditional shape builder without
changing its behavior and added the real host regression. It was built and run
on Ada with CUDA 13.2:

```sh
cargo test --release --features cuda,cudarc/cuda-13020 \
  --test gemm_bi_fixed_correctness \
  rna_wide_force_corpus_contains_every_hot_shape_family \
  -- --exact --nocapture --test-threads=1
```

RED exited 101 after a successful build. The assertion reported
`actual_auto=false`, actual `[(tail), (hot_a_boundary)]`, and expected all six
families. After the minimal unconditional-corpus change, the same command
exited 0 with `1 passed; 0 failed`. These are behavior executions, not source
string assertions. Raw logs are `raw/regression-red-cuda132.log` and
`raw/regression-green-cuda132.log`.

## Toolkits, paths, builds, and runtime identity

Every invocation explicitly set `CUDA_HOME`, `CUDA_PATH`, `LD_LIBRARY_PATH`,
`PATH`, `CARGO_TARGET_DIR`, and `MAMBA_RS_KERNEL_CACHE`. The common PATH suffix
was `/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin`.

| CUDA | Feature | Target | Cache | Correctness executable SHA256 |
|---|---|---|---|---|
| 12.8 | `cuda,cudarc/cuda-12080` | `/root/target-ada-rna-full-cuda128-20260906` | `/root/mamba-kcache-ada-rna-full-cuda128-20260906` | `1af67c50c686a7269363faa24d69070402d42a786472f1810d539ce5b3572364` |
| 13.0 | `cuda,cudarc/cuda-13000` | `/root/target-ada-rna-full-cuda130-20260906` | `/root/mamba-kcache-ada-rna-full-cuda130-20260906` | `555cee220e8c9a41efc243f9f9cc1216e63c2cf89893b42429a4ee86ff0199e7` |
| 13.2 | `cuda,cudarc/cuda-13020` | `/root/target-ada-rna-full-cuda132-20260906` | `/root/mamba-kcache-ada-rna-full-cuda132-20260906` | `750c1e69b52333b5b505d0734c532cda04a4a7cc46a70fac8febfc648be54d06` |

The per-toolkit build command was:

```sh
cargo test --release --features <matching-feature> \
  --test gemm_bi_fixed_correctness --no-run
```

All three exited 0. CUDA 13.2's existing selection was verified from the prior
AUTO build/report and by the successful explicit `cudarc/cuda-13020` build; it
was not assumed from the global symlink.

Ada identity was RTX 6000 Ada, CC 8.9, driver 595.45.04, Rust 1.98.0. Runtime
NVRTC symlinks resolved inside each explicitly selected toolkit:

| CUDA | Runtime NVRTC | SHA256 |
|---|---|---|
| 12.8 | `/usr/local/cuda-12.8/targets/x86_64-linux/lib/libnvrtc.so.12.8.93` | `2bb82d1a34b9fefa46aca357299aed66763d4d6613d015d5a591224c00fa7e5a` |
| 13.0 | `/usr/local/cuda-13.0/targets/x86_64-linux/lib/libnvrtc.so.13.0.88` | `a49e67e8e74590f1e98de55c39c6287efd3f59e3c3797464d7bbe0fe01349b11` |
| 13.2 | `/usr/local/cuda-13.2/targets/x86_64-linux/lib/libnvrtc.so.13.2.51` | `b93dae6f3435a5b7b52a689af7f769cc7876fcfd7b01381ca7d6a3814d405c89` |

## GPU qualification

CUDA 12.8 and 13.0 each ran this exact filter cold and warm, serially:

```sh
cargo test --release --features <matching-feature> \
  --test gemm_bi_fixed_correctness fixed_tf32_rna_wide_ \
  -- --ignored --nocapture --test-threads=1
```

CUDA 13.2 ran that force filter once and then the exact actual-AUTO test:

```sh
cargo test --release --features cuda,cudarc/cuda-13020 \
  --test gemm_bi_fixed_correctness \
  fixed_sm89_rna_wide_actual_auto_all_cells_prefix_views_and_graph_bits \
  -- --exact --ignored --nocapture --test-threads=1
```

| CUDA | Phase | Test summary | Runtime | Markers | Unique | Mode |
|---|---|---|---:|---:|---:|---|
| 12.8 | cold, cache 0 -> 3 | 2 passed; 0 failed | 87.45s | 448 | 448 | all `actual_auto=false` |
| 12.8 | warm, cache 3 -> 3 | 2 passed; 0 failed | 28.13s | 448 | 448 | all `actual_auto=false` |
| 13.0 | cold, cache 0 -> 3 | 2 passed; 0 failed | 85.71s | 448 | 448 | all `actual_auto=false` |
| 13.0 | warm, cache 3 -> 3 | 2 passed; 0 failed | 28.87s | 448 | 448 | all `actual_auto=false` |
| 13.2 | force, cache 0 -> 3 | 2 passed; 0 failed | 88.71s | 448 | 448 | all `actual_auto=false` |
| 13.2 | actual AUTO, cache 3 -> 3 | 1 passed; 0 failed | 36.62s | 448 | 448 | all `actual_auto=true` |

Every row exited 0. In every run the independent family counts were tail 168
and 56 each for hot A, B, C, D, and E. This is
`(21 + 5*7) * 2 output offsets * 2 input modes * 2 bias states = 448`.
The Rust harness places the first printed marker after the test name on the
same physical line, so the independent `.groups` normalization starts at the
marker token before counting and uniqueness comparison; it neither fabricates
nor drops a record.

The 13.2 actual-AUTO pass retains and executes the existing symbol, five-arg
Driver ABI, geometry, bit, repeated eager/graph, and misaligned A/B fallback
assertions. No actual-AUTO wrapper ran on 12.8 or 13.0.

## Kernel-cache identity against frozen proof

All three new cache directories contain exactly three blobs. Each cache-key
filename and blob is byte-identical to the frozen matching-toolkit proof:

| CUDA | Cache-key suffix | Blob SHA256 |
|---|---|---|
| 12.8 | `73fbec7557a95462a555e39f1b3c94e8d2bcc7047a14f8f3e3fa51a768ec5b9a` | `c02fb7a0ef6a50cb52dc3fec2e83ebb1eb6b4015a9f200d48d65fd4493d489c9` |
| 12.8 | `ae8e2e3db0255db26419c8292c70ea945f46076770ddb3522d34443addeaf374` | `22827bf4ea23b939d8ec862b276f39b02c53ac2389345df1c16f62ca293c23fb` |
| 12.8 | `da5dabeff55570d1fb6ca43a1d4a8aa7a2a9d25efb6436ed7f03c4a28a73d986` | `48a388fc610d110bb884ea3cd0f49eda299c0c7ce0c8e2beb24d34a925198a54` |
| 13.0 | `4d6815a9cdc06297113b72dc2d9ae9fac46b50a0583c561f358727d4a6f2cf49` | `cc518c145df2f13d1a76334dd6b5572a27cfc9d6887b6d72c273f51a39b58549` |
| 13.0 | `ceeda6e9d55cfd166dafd638a9055daf8a1bb044f573c3349e147bff03b8fe7b` | `c599a8b8807dd944ae79238241b36f77599cb5cab0511d3ab4f8716bc7545c3c` |
| 13.0 | `feac83eb41fcc629d0c4bf9be1758d9e92c431728e17c1f46cfcd5c5a9ebc27d` | `32e61498d102e2343b2c3b65ec062af81b1351b1039a2503c0657628b4d3a9c0` |
| 13.2 | `3809af034718aa08dfdfc1b6baeb407715436dc7db87ef39825b033c2fd72541` | `5851ebd49595a0b2ef665073289e52f903eec3266deadfc5ad5d9f2096ce6d16` |
| 13.2 | `4e99acb556ad1f6bb2aafa382f1618db9c9ecf37286fa7da37a62d86399c14ce` | `220ccfc809011d7738b7447a319ecb0c1b2b12c49b3766ca0315d28fa3d5153f` |
| 13.2 | `822ebf0dda7dcf9c4f5c5a2213884dde42ab45f5cf238479a81856042c302b7c` | `5035921fc8e078e5454c9c5392f9e9ee6ca6a3de54c1e08cb54bba432fa37546` |

The frozen comparators were the census caches for 12.8/13.0 and
`/root/mamba-kcache-ada-rna-wide-auto-20260906` for 13.2. Full paths, modes,
both hashes, and `cmp` results are in `raw/cache-identity-comparison.log`.

## Evidence, verification, and remaining boundary

Complete raw evidence is mirrored at
`internal/perf/ada-rna-full-toolkit-qualification-20260906/raw/`. Its
21-entry `SHA256SUMS` manifest has SHA256
`348935b9e11389dcb4c643c2333d69258b45e05d91f66d71076383d508055278`;
every copied file passed `shasum -a 256 -c SHA256SUMS`.

Direct local `rustfmt --edition 2024` and `rustfmt --check` passed, as did
`git diff --check`. `git diff --name-only` lists only the assigned test source.
No production/CUDA/epoch/harness source was changed. There are no qualification
blockers for this test-only task. Main still owns independent raw-log review,
selection of evidence/docs, and the commit. Actual AUTO widening, cross-toolkit
AUTO mapping, revision/epoch changes, and post-promotion confirmation remain
explicitly outside this task.
