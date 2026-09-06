# Ada Fixed homogeneous-half swizzle force report

## Result

Task 1 is complete on the requested CC8.9/142-SM Ada board for matching CUDA
12.8, 13.0 and 13.2. The existing experimental swizzle math/layout is integrated
as a distinct, force-only production route. All final host, static, functional,
retained-route and sanitizer gates passed. No AUTO admission or routing-epoch
change was made; timing qualification remains Task 2.

Baseline: `86fd7f5605ef5aba77f4e51211af362edb929003`.

## Production change

- Added production `sm89_half_swizzle.cu` and its constexpr layout header. The
  CUDA namespace body is byte-identical to the prototype. The exported BF16/F16
  wrappers use four pointers plus the align-4 32-byte Fixed bundle in exact
  `alpha,beta,M,N,K,lda,ldb,ldc` order.
- Added a distinct `FixedTile::Tc128Sm89Swizzle`, force-only homogeneous-half
  launch with flat grid `ceil(M/128)*ceil(N/128)`, block 256 and 69,632 dynamic
  shared bytes. Empty output, K0 null inputs, alignment, i32, padding and grid
  checks retain incumbent semantics.
- Composed layout then CUDA source only in the Ada Fixed suffix. Exact tests prove
  the old Fixed base/suffix bytes, non-Ada Fixed bytes and every Triad composition
  remain unchanged apart from the two intended fragments.
- Threaded an independent cold/warm Driver ABI result and independent optional
  holder/rejection through module loading. Admission requires exactly the two
  typed exports, exact five-argument PTX and live Driver ABI, typed MMA,
  `cp.async`, A x4/B x2-transposed `ldmatrix`, no local/atomic/reduction work,
  zero local/static shared, <=224 registers, >=256 max threads and occupancy >=1.
- Extended the force-performance registry fail-closed by architecture/dtype.
  Measured-callsite identity graphs are now mandatory for eager-only RNA and both
  Ada half physical routes. The live half descriptor validates exact symbol,
  one node, flat grid, block/shared, five offsets/sizes, terminal sixth rejection,
  all four captured pointer values and bundle bytes before any timing.
- Parameterized the existing half hot-cell helper, retaining its CUDA13.2 AUTO
  behavior while adding a forced-swizzle A-E wrapper usable on all three toolkits.
- Added a shipped pure-host C++ proof that directly includes the production layout
  header and exhaustively checks staging bijection, chunk alignment, fragment lane
  mapping and bank groups.

Changed source/test paths:

- `kernels/gemm_bi_fixed/sm89_half_swizzle.cu`
- `kernels/gemm_bi_fixed/sm89_half_swizzle_layout.cuh`
- `src/mamba_ssm/gpu/gemm_bi_fixed.rs`
- `src/mamba_ssm/gpu/gemm_bi_triad/modules.rs`
- `src/mamba_ssm/gpu/kernels.rs`
- `tests/arch_compile_gates.rs`
- `tests/gemm_bi_fixed_performance.rs`
- `tests/gemm_bi_fixed_sm89_pipeline.rs`
- `tests/cuda/gemm_bi_fixed_sm89_swizzle_layout.cpp`

No prototype, CUDA composer base, loader outside the independent new holder,
numeric policy, AUTO selector or epoch fixture was changed.

## TDD and review fixes

The focused CUDA12.8 RED logs preserve exit 101 for:

- missing production source/layout (`red-source-layout-cuda128.log`);
- missing distinct force enum (`red-force-enum-cuda128.log`);
- missing production-bound exhaustive layout proof
  (`red-production-layout-binding-cuda128.log`);
- eager-only swizzle not requesting an identity graph
  (`red-half-eager-identity-graph-cuda128.log`).

Implementation debugging is preserved rather than rewritten: an initial rsync put
`modules.rs` at an unused remote path, then the correct nested build input was
synced; immutable composition fixtures required suffix-only updates; one count
fixture required the new inventory entry; and the first final CUDA12.8 release
batch exposed a missing `mut` in the newly amended poison reset. The covering
CUDA12.8 rerun has all six exit statuses zero.

Independent static review initially requested four proof improvements: poison
reset per candidate, direct production-header host proof, forced hot A-E coverage,
and full measured-callsite graph arguments including eager-only mode. All four
were implemented. `half-swizzle-force-fix1-review.md` approves the final fix
round with no new finding.

## Exact command profiles

Every remote command used:

```text
ssh -o BatchMode=yes -o ConnectTimeout=10 ada bash -lc <command>
```

For each toolkit the command exported the exact `CUDA_HOME`, `CUDA_PATH`,
`LD_LIBRARY_PATH`, CUDA-first `PATH`, listed `CARGO_TARGET_DIR`, and listed
`MAMBA_RS_KERNEL_CACHE`, then ran in
`/root/mamba-ada-half-swizzle-force-20260906`. Matching features were:

```text
CUDA12.8: --features cuda,cudarc/cuda-12080
CUDA13.0: --features cuda,cudarc/cuda-13000
CUDA13.2: --features cuda,cudarc/cuda-13020
```

The final release host/static commands per toolkit were:

```text
cargo test --release <feature> --lib
cargo test --release <feature> --test gemm_bi_fixed_performance
cargo test --release <feature> --test arch_compile_gates fixed_sm89_half_swizzle_production_source_and_layout_contract -- --exact --nocapture
cargo test --release <feature> --test arch_compile_gates compiles_for_sm89 -- --exact --nocapture
cargo test --release <feature> --test gemm_bi_fixed_sm89_pipeline fixed_sm89_half_swizzle_and_pipeline_holders_are_independently_live -- --ignored --exact --nocapture
c++ -std=c++17 -O2 tests/cuda/gemm_bi_fixed_sm89_swizzle_layout.cpp -o /tmp/gemm_bi_fixed_sm89_swizzle_layout-final-20260906
/tmp/gemm_bi_fixed_sm89_swizzle_layout-final-20260906
```

Final functional commands used the same release/profile/toolkit environment:

```text
cargo test --release <feature> --test gemm_bi_fixed_sm89_pipeline fixed_sm89_half_swizzle_rounding_edges_match_incumbent_across_store_paths -- --ignored --exact --nocapture
cargo test --release <feature> --test gemm_bi_fixed_sm89_pipeline fixed_sm89_half_pipeline_forced_cross_rung_prefix_view_graph_bits -- --ignored --exact --nocapture
cargo test --release <feature> --test gemm_bi_fixed_sm89_pipeline fixed_sm89_half_swizzle_forced_hot_a_e_prefix_view_graph_bits -- --ignored --exact --nocapture
cargo test --release <feature> --test gemm_bi_fixed_correctness fixed_sm89_rna_wide_actual_auto -- --ignored --nocapture
```

The eager-only identity smoke additionally set:

```text
MAMBA_FIXED_ADA_VENDOR=1
MAMBA_FIXED_VENDOR_EXACT_CC=8.9
MAMBA_FIXED_ADA_ROWS=bf16,f16
MAMBA_FIXED_ADA_CELLS=hot_a
MAMBA_FIXED_ADA_BIAS=0,1
MAMBA_FIXED_ADA_WINDOWS=1
MAMBA_FIXED_VENDOR_TILES=Tc128Sm89Swizzle
MAMBA_FIXED_VENDOR_PATHS=eager
```

and ran exact ignored `fixed_ada_forced_rungs_paired_precision_cublas`. This was
an identity/bit-path smoke, not a quiet timing qualification.

CUDA13.2 additionally ran exact ignored:

```text
gemm_bi_tf32_cohort_binding::tf32_cohort_binds_on_this_board
gemm_bi_tf32_cohort_binding::sm89_tf32_bias_cohort_serves_the_qualified_wide_epilogues
gemm_bi_fixed_performance::fixed_sm89_tf32_c_auto_prefix_special_bias_graph_bits
gemm_bi_fixed_sm89_pipeline::fixed_sm89_half_pipeline_auto_hot_cell_prefix_view_graph_bits
gemm_bi_fixed_sm89_exact_n64::fixed_sm89_exact_n64_forced_bits_graph_views
gemm_bi_fixed_sm89_exact_n64::fixed_sm89_exact_n64_auto_prefix_view_graph_bits
```

Sanitizers invoked the exact release test executables directly:

```text
compute-sanitizer --tool <memcheck|racecheck|initcheck|synccheck> \
  --report-api-errors no --error-exitcode 99 \
  <release-pipeline-test-binary> --ignored --exact \
  fixed_sm89_half_pipeline_sanitizer_smoke --nocapture
```

`--report-api-errors no` is intentional and limited to CUDA API diagnostics: the
loader performs a required invalid sixth-parameter query to prove that the Driver
ABI has exactly five arguments. The first 12.8 run, without that flag, exited 99
only because compute-sanitizer reported those expected negative probes. The final
runs retain `--error-exitcode 99`; GPU memory/race/init/sync errors remain fatal.

## Results

| Gate | CUDA 12.8 | CUDA 13.0 | CUDA 13.2 |
|---|---:|---:|---:|
| Release library | 636 pass, 46 ignored | 636 pass, 46 ignored | 636 pass, 46 ignored |
| Performance non-ignored | 43 pass, 62 ignored | 43 pass, 62 ignored | 43 pass, 62 ignored |
| production source/layout | 1 pass | 1 pass | 1 pass |
| `compiles_for_sm89` | 1 pass, 113.11 s | 1 pass, 98.63 s | 1 pass, 95.88 s |
| live independent holders | 1 pass | 1 pass | 1 pass |
| amended rounding edges | 1 pass, 3.29 s | 1 pass, 3.42 s | 1 pass, 3.38 s |
| full forced corpus | 1 pass, 8.69 s | 1 pass, 8.60 s | 1 pass, 8.46 s |
| forced hot A-E | 1 pass, 13.59 s | 1 pass, 14.40 s | 1 pass, 14.39 s |
| eager-only identity smoke | 1 pass, 8 records | 1 pass, 8 records | 1 pass, 8 records |
| retained RNA/AUTO | 2 pass, 448 unique groups | 2 pass, 448 unique groups | 2 pass, 448 unique groups |
| four sanitizer tools | zero errors/hazards | zero errors/hazards | zero errors/hazards |

The production host layout proof passed 4,096 copied chunks, 98,304 checked
fragment halves and 1,536 conflict-free `ldmatrix` groups.

Live swizzle resources for both BF16 and F16 were local 0, static shared 0,
max-threads 256 and occupancy 1. Registers were 177 on CUDA12.8/13.0 and 180 on
CUDA13.2, all below the unchanged cap 224.

The bounded rounding corpus includes raw minimum/maximum subnormals, minimum
normal, maximum finite, signed cancellation and FP32 bias not representable in
half. One finite-only row prevents dense maximum-finite overflow from reducing
the corpus to NaNs. Before every pipeline/swizzle launch the result region is
filled with the bitwise complement of its expected u16, while guards remain fixed;
therefore any missing store necessarily fails.

## Source and artifact identities

Production files:

```text
1ab11e6729a5b25689dff8b978bf0cdd4676dae9425ac8286bbeb297e57d64a2  kernels/gemm_bi_fixed/sm89_half_swizzle.cu
243e9b0680a3ecd1d34bd325244d221903adab59c35a691295ec334996115410  kernels/gemm_bi_fixed/sm89_half_swizzle_layout.cuh
00383b367bbdeb2bfd094dc289779c5aef1d6ec0561629f96dddaa863c1a9454  tests/cuda/gemm_bi_fixed_sm89_swizzle_layout.cpp
```

Main independently established prototype namespace-body SHA
`e6a26205430d04d6bdeccb613a4dc0284f1747b92d932e7900f92987b72bcf2f`
and layout-body SHA
`353603409734a365acae0fa00a25525237ca347a49d4b0539338913764a188ed`.

The composed Fixed source is 688,275 bytes with SHA
`d1180079f0067afbc3f54210ce41a80dbc6bb422d53085824082089a0ff97f2a`
on all three toolkits.

| CUDA | invocation/cache key | PTX artifact digest | header-manifest digest | NVRTC library domain |
|---|---|---|---|---|
| 12.8 | `53c404fc45b06ef3faf5ef8842f166da9fa77df7f2965f64f39c4aae608261af` | `323538314fb45b937e7eaca4e3c646c8e1d055a56e980e8bf72432bb6f879964` | `cef6c4772487e8993abaf6afcb41d3a91ec304e731420165b5f512e5da90ef64` | `26b0a3a02044ffcbc1693fd83e9261beffa692a4fbcfe3ac5e9d8c87980bb155` |
| 13.0 | `e96609c91e1e74d22a2aa435f7a723fb6a78e08f1e66c818d6a6930dc5b7e0e6` | `45ccedfa2904b03e2eb0d25a331e54ade876158a482ea821e255626c0aea8e59` | `8a02b290406f6a9459bae976b18cf882786d6d90b48b1ec6c50d38671d4a5255` | `709b91c36bfb0ed966ee69adc8d6f87ff110eecf3dfb5060367f183ce614eb0d` |
| 13.2 | `58c7853250f2f225830390289de0c1dfce34d81efede704436222e54d53ac991` | `c213b3c33d94e94e257d30e472cee583b5f3634d8dd5216e91a08b7c286d21c2` | `4d2b8b1c3d1ec175d3556f37139b52b832d4fccd0112e68ca8474b3276b58b5b` | `d031a53eb97235b70f62f652932db1bdf728ea229c8ca809d53c5ffd91642687` |

The header-manifest values intentionally differ from the earlier RNA checkpoint:
that identity serializes paste analysis over the newly composed source, not only
filesystem headers. Compiler targets and NVRTC library domains remain pinned.

Release pipeline-test executable SHA-256:

```text
CUDA12.8 fc6d0bd9bcae309ccc1d7083a084b980cf283e4f760a2bec8884781272b6e6bf
CUDA13.0 7f8c8f36482b89914f9537226cc1bbf11462b9f8359bc4f0e0853afe684384d8
CUDA13.2 54fc8a995dc9227a2942afb846415ef5619f5167e49dd845d7f818e371aa0985
```

All nine cache blob filenames and file SHA-256 values, plus every relevant release
test executable, are recorded in `internal/perf/ada-half-swizzle-force-20260906/final-identities.log`.
Every warm functional/sanitizer log shows the exact three Fixed/Triad cache hits.
Each cache directory remained mode 0700. The source rsync copy intentionally has
no `.git`; the authoritative HEAD is the local baseline above.

One early rsync placed an unused duplicate at
`/root/mamba-ada-half-swizzle-force-20260906/src/mamba_ssm/gpu/modules.rs`.
The actual build input is the correctly synced nested
`src/mamba_ssm/gpu/gemm_bi_triad/modules.rs`; the stray file is outside module
resolution and was left untouched because this task prohibited deletion.

## Remaining work and claim boundary

This report establishes one-board, three-toolkit production force availability
and functional equivalence for the named corpora. It does not claim universal
cross-GPU MMA identity, a quiet performance result, a vendor win, or AUTO
eligibility. The one-window eager-only records exist solely to execute the
identity path.

Task 2 must run the reviewed paired 21/101-window census, including the incumbent
pipeline on CUDA12.8/13.0 and direct candidate pairing when both internal routes
win. AUTO admission requires internal p50 and p95 wins, with vendor gaps reported
separately. Task 3 alone may change the selector and routing epoch after that
evidence.
