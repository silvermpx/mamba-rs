# Ada RNA-wide AUTO promotion report

Status: implementation, correctness, quiet final101 and final focused fixture
cleanup GREEN. Required REDs confirmed; independent implementation review found
no issues. Ready for main review/integration; Ada build/GPU lane released.

Base: `82fc400fa891795764c7b0c6dc359768e18e253e`, branch
`codex/gemm-bi-triad-sm80`, existing isolated local worktree.

## Owned diff

Nine files (including two subsequently authorized fixture fixes): `src/mamba_ssm/gpu/gemm_bi_fixed.rs`,
`src/mamba_ssm/gpu/kernel_identity.rs`,
`src/mamba_ssm/gpu/gemm_bi_triad/dispatch.rs`,
`src/mamba_ssm/gpu/gemm_bi_triad/launch.rs`,
`src/mamba_ssm/gpu/gemm_bi_triad/modules.rs` (test only),
`tests/gemm_bi_tf32_cohort_binding.rs`,
`tests/gemm_bi_fixed_correctness.rs`, `tests/gemm_bi_fixed_performance.rs`,
`tests/arch_compile_gates.rs` (test only).
No CUDA/compiler/composer/numeric/schedule or qualification hash edits.

## Environment and RED

SSH: `ssh -o BatchMode=yes -o ConnectTimeout=10 ada`.
New source `/root/mamba-ada-rna-wide-auto-20260906`, new target
`/root/target-ada-rna-wide-auto-20260906`, private 0700 cache
`/root/mamba-kcache-ada-rna-wide-auto-20260906`. Frozen force source/target
were not modified. Preflight: RTX 6000 Ada, GPU UUID
`GPU-d1edd7be-e88d-aed6-047d-622163306f0e`, driver 595.45.04, GPU 0%, 90 MiB.

Remote environment for builds/tests:

```sh
export PATH=/root/.cargo/bin:/usr/local/cuda-13.2/bin:$PATH
export CUDA_HOME=/usr/local/cuda-13.2 CUDA_PATH=/usr/local/cuda-13.2
export LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64
export CARGO_TARGET_DIR=/root/target-ada-rna-wide-auto-20260906
export MAMBA_RS_KERNEL_CACHE=/root/mamba-kcache-ada-rna-wide-auto-20260906
```

Before production edits, against base dispatch:

```sh
cargo test --release --features cuda,cudarc/cuda-13020 --lib only_the_tuning_table_revision_moved -- --nocapture
cargo test --release --features cuda,cudarc/cuda-13020 --test gemm_bi_fixed_correctness fixed_sm89_rna_wide_actual_auto_hot_a_route_and_graph -- --exact --ignored --nocapture --test-threads=1
```

`red-epoch.log`: expected 40, actual 39, 0 passed/1 failed.
`red-auto-hot-a.log`: expected `Tf32RnaM128N128S3`, actual `Tf32M64S2`,
0 passed/1 failed; GPU test 65.66 seconds including cold NVRTC initialization.
Logs are archived under `internal/perf/ada-rna-wide-auto-20260906/`; hashes below.

## Build and checks so far

```sh
cargo test --release --features cuda,cudarc/cuda-13020 --lib --test gemm_bi_fixed_correctness --test gemm_bi_fixed_performance --test gemm_bi_tf32_cohort_binding --test arch_compile_gates --no-run
```

PASS, 49.25 seconds. Five executable test binaries built in the new target.
Direct `/Users/silvermpx/.cargo/bin/rustfmt --edition 2024 --check` on the seven
owned files and `git diff --check -- src tests`: PASS.

First full library run: 631 passed, 2 failed, 46 ignored. Failures:
`sm120_half_exact_overlay_promotes_only_qualified_full_context_cells` (remaining
39 tuple pin) and `module_sources_have_exact_deterministic_boundaries` (CC12
test retains Ada half suffix because it slices len-2 instead of len-3).
The exact composition test also failed on the frozen force binary (0/1),
`baseline-force-composition-test.log`. Main authorized the test-only explicit
exclusion of the three Ada fragments; production composition is unchanged.
The tuple pin is now (5,40,8). Both fixes passed the final rebuild/suite below.

## Final source hashes (local and isolated remote match)

| File | SHA256 |
|---|---|
| gemm_bi_fixed.rs | 751266956443ea9f753ba5809692598491f9cd54664f66dcf45748e52f6c186e |
| modules.rs | 07914f8fcd979856065392fb71e3fdbb69f2bde2998885c50edfe89b21ff032a |
| kernel_identity.rs | cf5b82ac2b70fc2dde4634666991b4c1fd5a539eed06761b03e3afcfd9a986aa |
| dispatch.rs | 7e46c4c4f86323187d3519cc896faa3f728ada41fcffaa0a4fd9af8a3f71fdea |
| launch.rs | aaaa43b1f35300fe7ef5edd87d3aff39697cfbd52ba1259e1fc6f4bfeee292db |
| gemm_bi_tf32_cohort_binding.rs | e6ff5806ef88732ff52f8194418817f3722c5dd3ef288e55d9f90e57b0ce24f9 |
| gemm_bi_fixed_correctness.rs | 9a87b3c3af7a074e689877e8085928397e38e989e31c467e27c4d9eb251886c1 |
| gemm_bi_fixed_performance.rs | 8ba5fe376737c29919db307db3867c62b51c1f9dc4295d29e9d8215cecf69de9 |
| arch_compile_gates.rs | 36b33f96852fa8c75b77fdb250f4ed707667b91f9e2eb302a9b6cd05a867f6f9 |

## Final build, correctness and cohort results

Final release rebuild: PASS, 50.29 seconds, `green-build-final.log`.
The following commands use binaries under
`/root/target-ada-rna-wide-auto-20260906/release/deps/` with the environment above:

```sh
mamba_rs-87c2fddad15adab2 --nocapture --test-threads=1
gemm_bi_fixed_correctness-7860b14bbe37841d fixed_sm89_rna_wide_actual_auto --ignored --nocapture --test-threads=1
gemm_bi_fixed_performance-429f558dd590762a fixed_sm89_tf32_c_auto_prefix_special_bias_graph_bits --exact --ignored --nocapture --test-threads=1
gemm_bi_tf32_cohort_binding-c96a6105c026b3ba tf32_cohort_binds_on_this_board --exact --ignored --nocapture --test-threads=1
gemm_bi_tf32_cohort_binding-c96a6105c026b3ba sm89_tf32_bias_cohort_serves_the_qualified_wide_epilogues --exact --ignored --nocapture --test-threads=1
gemm_bi_fixed_performance-429f558dd590762a --nocapture --test-threads=1
arch_compile_gates-4a95865368ee0138 --nocapture --test-threads=1
```

Results: lib 633/633, 46 ignored (0.94s); actual AUTO 2/2 (38.95s);
retained C prefix/special-bias graph 1/1 (2.78s); Triad cohort 1/1 (3.03s),
Triad bias cohort 1/1 (3.01s); performance static 43/43, 62 ignored (0.01s).
The architecture suite completed 56 passed/1 failed in 1432.60s. Its only failure
is a pre-existing stale source assertion in
`fixed_sm120_tf32_pair_store_production_source_contract`: it forbids the public
tile `Tf32Sm120M64S2PairStore`, already present at 82fc400f and selected for
qualified D0/D1 AUTO by existing production dispatch. The same exact test
fails in the frozen force arch binary (0/1,0.00s), proving baseline failure.
Main authorized removal of only that obsolete source-absence assertion and stale
private wording; existing CUDA arithmetic/store/inventory checks remain intact.
All actual NVRTC/PTX/assembly/resource gates pass,
including RNA-wide registers 153, static shared 0, stack 0, spills 0. The library suite includes
old-epoch graph rejection, all cohort revision tests, and exhaustive guard
declines before launch. Both real Triad cohort tests retain their old symbols.

Actual AUTO tests exercise all A-E cells/bias0,1, finite nonrepresentable and
exceptional A/B/bias, all five ordinary TF32 rungs versus forced RNA raw bits,
M-1/M/M+1 and small prefixes, aligned row subviews, C4-only output, shifted
F32-valid A/B views, guarded outputs/input immutability, two eager repeats and
two poisoned graph replays. Admitted calls assert the actual RNA graph symbol,
five Driver arguments ending in the 32-byte bundle, exact grid, 256 threads,
98,304 shared bytes, one node and no sixth argument. Shifted A/B calls assert
old M64S2 AUTO and identical RNA reference bits. C4 calls decline RNA.

| Log (under `internal/perf/ada-rna-wide-auto-20260906/`) | SHA256 |
|---|---|
| red-epoch.log | 2c23b90ee5e19eed0b6363c4e98bd561943dcee278ad6b7fcf466b5b2a16f304 |
| red-auto-hot-a.log | ee657281822decf2d30f796f790c3d88fbbe961a5abce17ecfa9e3a3a6ac2f42 |
| baseline-force-composition-test.log | 68ace74c21cf899ecb6fadfd75141a3ff4c1aa46a0548ea3f2a51367168707d2 |
| green-lib-all-final.log | 99902126710067b8b940e340204ea35d4009b93517200c01d0c8be7144236217 |
| green-auto-correctness.log | 5c8d903a7632435672753e41a8c6444073214c4d7e4df525b3d18ade4232e38d |
| green-c-auto-prefix.log | a78b0804599fb6a3911f6443121fb0b8325f6a91cf4b247557745df61e3ca8ab |
| green-triad-cohort.log | ba323b22f334f9de6de6f014fa2fdd511d2f5375935fbf41c82348a99de7ca25 |
| green-triad-bias-cohort.log | a221a9da682af360f85fb4b15d7af2e5affb2508160fb1e8ff0436cfc56e1ff2 |
| green-performance-units.log | cdf8a6b813329d51cd39c9d1ff6b9035a5cd94a6af269996bc95a5714a2d756b |
| green-arch-units.log (56 pass / 1 baseline failure) | 1d1a08fbafc830378557f69c85d200e8b8065578529682e9679ff6bcfdf841be |
| baseline-force-pair-store-contract.log | ee12d6b4a610f077bbcaa8f50221da6049965c11f5c1898dd33b1b9a044c72b3 |
| green-pair-store-contract-final.log | 677bb65afe8fde8d3151d12e3baa4973d81254ac855418929e38ef9ffa21fe86 |

After the quiet final timing run, only the architecture test binary was rebuilt
(4.89s) and its exact corrected test passed (1/1, 0.00s):

```sh
cargo test --release --features cuda,cudarc/cuda-13020 --test arch_compile_gates fixed_sm120_tf32_pair_store_production_source_contract -- --exact --nocapture --test-threads=1
```

The unchanged 24-minute compile/assembly matrix was not rerun. Evidence is
56 passed plus one baseline fixture failure in that full run, followed by the
corrected focused 1/1 GREEN, not a claim of a fresh full 57/57 run. Final arch
binary SHA256 `fcd4386dcbae5aaf1f080a477b8d5c3ca64637dd5d4b6efc938a39fd202984c9`.
The performance binary remained `2dfbc274...`, verified after this build.
Final direct rustfmt checks and code-only diff whitespace checks pass.

## Frozen production identity

Final performance binary
`/root/target-ada-rna-wide-auto-20260906/release/deps/gemm_bi_fixed_performance-429f558dd590762a`,
SHA256 `2dfbc274dfc1773939491fe2fbc0eb86d6536c7bb6f380720b448898d1cc68e2`.
The frozen force binary still hashes to
`db3dbafb1b1ccf3c0ea422d2860e8a7789e1eca9b9c14f51480711f18ac5dde1`.

Actual loaded Fixed composed source digest
`7ccad9938c4ee7cb0080d62b7053e082537fcb2ad184288ffc59a513cf526301`,
invocation `4e99acb556ad1f6bb2aafa382f1618db9c9ecf37286fa7da37a62d86399c14ce`,
artifact `c6dc1ee707ceceb512b58f226f8b5288097162c5257c41ef89a2e235fda1945f`,
header manifest `e893dcebd4b2eb9d2e8cd84721c99684550413437d318113a8c45a6fa9ac1f73`:
all match the force101 checkpoint. The new private cache was compiled cold.
All three produced Ada cache blobs are byte-identical to the frozen force cache:

| Invocation key | SHA256 of complete cache blob in both caches |
|---|---|
| 3809af034718aa08dfdfc1b6baeb407715436dc7db87ef39825b033c2fd72541 | 5851ebd49595a0b2ef665073289e52f903eec3266deadfc5ad5d9f2096ce6d16 |
| 4e99acb556ad1f6bb2aafa382f1618db9c9ecf37286fa7da37a62d86399c14ce | 220ccfc809011d7738b7447a319ecb0c1b2b12c49b3766ca0315d28fa3d5153f |
| 822ebf0dda7dcf9c4f5c5a2213884dde42ab45f5cf238479a81856042c302b7c | 5035921fc8e078e5454c9c5392f9e9ee6ca6a3de54c1e08cb54bba432fa37546 |

## Timing admission note

`inadmissible-overlap-confirm101-old-m64-vs-auto-rna.log` is explicitly INADMISSIBLE for timing:
although its idle GPU preflight passed, it overlapped the running architecture
NVRTC compilation suite. It completed 40 records in 71.97s; no performance
conclusion will use it. No process termination/signals were used. A new quiet
final101 log completed after the architecture suite exited. Inadmissible
log SHA256 `5d075e116ced4e14ad8a5f7f483463b392f1346d5358fe2b983675a96ced1eea`.

## Quiet final post-AUTO confirmation

The arch suite had fully exited before the final run. Preflight: GPU 0%, 90 MiB,
1800 MHz, 41 C. No host build, NVRTC test, or competing GPU task overlapped.
Final command, with the common environment above:

```sh
export MAMBA_FIXED_ADA_VENDOR=1 MAMBA_FIXED_ADA_ROWS=tf32
export MAMBA_FIXED_ADA_CELLS=hot_a,hot_b,hot_c,hot_d,hot_e
export MAMBA_FIXED_ADA_BIAS=0,1 MAMBA_FIXED_ADA_WINDOWS=101
export MAMBA_FIXED_VENDOR_TILES=Tf32M64S2 MAMBA_FIXED_VENDOR_PATHS=eager,graph
export MAMBA_FIXED_VENDOR_EXACT_CC=8.9
/root/target-ada-rna-wide-auto-20260906/release/deps/gemm_bi_fixed_performance-429f558dd590762a fixed_ada_forced_rungs_paired_precision_cublas --exact --ignored --nocapture --test-threads=1
```

PASS, 96.54s, 40 records, 0 rejected, completion `passed:true`. All 40 records have
101 samples per arm, revision 40, actual AUTO RNA, forced old M64, one expected
kernel in each custom graph, identical raw/AUTO/repeat bits and unchanged
source/compile/artifact/header/NVRTC library identities versus force101.
All 20 graph-path replay flags are true; eager records use false for this
graph-replay field as N/A in the existing schema (their physical graphs and
raw/repeat bits still pass). RNA graph census verifies the 32-byte ABI and exact
geometry. Maximum AUTO normalized error versus independent PEDANTIC reference
is 0.00027250802550237064, below 0.0025; the performance denominator is FAST_TF32.

`confirm101-old-m64-vs-auto-rna-final.log` SHA256
`3dbb8754b651706d67c83ab1f96fb26ed66a63bc161e8ced015f36ef50831f46`.
`preflight-final-post-auto101.log` SHA256
`635c2b5bfa18b9bd2c922254cf6a8ff96b3b868e10eccab96e2f7e9e2f8ade3f`.

Each number below is the worst of eager/graph and both orders. Quantiles are
computed directly from matched samplewise ratios; 101 samples use indices 50/95.
AUTO/old p95 is not the inverse of old/AUTO p95. All ten cells win against old
AUTO; only A1/B1 win against FAST in every p50/p95 cohort.

| Cell | AUTO/old p50 | AUTO/old p95 | old/AUTO p50 | old/AUTO p95 | AUTO/FAST p50 | AUTO/FAST p95 |
|---|---:|---:|---:|---:|---:|---:|
| A0 | 0.828687 | 0.848324 | 1.213291 | 1.232536 | 1.080677 | 1.108626 |
| A1 | 0.837823 | 0.856835 | 1.197807 | 1.220026 | 0.868782 | 0.904797 |
| B0 | 0.726012 | 0.732621 | 1.385496 | 1.393455 | 1.098057 | 1.110621 |
| B1 | 0.730103 | 0.738436 | 1.377416 | 1.387907 | 0.966568 | 0.980568 |
| C0 | 0.621531 | 0.626263 | 1.633333 | 1.648897 | 1.314221 | 1.334471 |
| C1 | 0.619660 | 0.628503 | 1.633990 | 1.644553 | 1.207602 | 1.213196 |
| D0 | 0.896758 | 0.911599 | 1.131510 | 1.163695 | 1.262692 | 1.301159 |
| D1 | 0.899866 | 0.913177 | 1.123208 | 1.154117 | 1.091251 | 1.126994 |
| E0 | 0.870655 | 0.885280 | 1.150418 | 1.178687 | 1.307817 | 1.342478 |
| E1 | 0.873246 | 0.887069 | 1.147451 | 1.172885 | 1.225008 | 1.254475 |

Revision 40 changes the global dispatch/graph epoch, including aliased TF32
evidence cohorts, without changing compiler/composer/numeric/schedule revisions.
Revision 39 graphs reject and current-epoch identities match. Existing Triad
qualification hashes/routes remain unchanged and their real Ada cohort tests pass.
No cell regressed versus its own prior route. C4 and all unqualified contexts
retain ordinary AUTO; every old force route remains reachable.

No commit/push performed; main owns final independent review and integration.
