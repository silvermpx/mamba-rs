# Ada Fixed RNA-wide force integration report

## Status

GREEN and ready for independent review/commit as a force-only checkpoint. The new
`FixedTile::Tf32RnaM128N128S3` is reachable only through the Fixed `sm_89`
module. No AUTO selector was changed. The old Triad wide force tile/symbol and
every Triad composed-source byte sequence remain unchanged; the RNA fragment is
absent from all non-`sm_89` Fixed compositions, including CC12.

The production-loaded symbol is
`gemm_bi_nn_fixed_rna_wide_tf32_v1_m128n128_bk32_s3`: 128x128/BK32/S3,
256 threads, 98,304 bytes opt-in dynamic shared, no scratch or auxiliary launch.
It retains the approved ascending-k8 pipeline and explicit
`cvt.rna.tf32.f32` conversion.

## Changed files

- `kernels/gemm_bi_fixed/tf32_rna_wide.cu` (new isolated Fixed source)
- `src/mamba_ssm/gpu/gemm_bi_triad/modules.rs` (conditional composition,
  PTX/Driver/resource admission, optional loader and rejection reason)
- `src/mamba_ssm/gpu/kernels.rs` (optional admitted holder)
- `src/mamba_ssm/gpu/gemm_bi_fixed.rs` (distinct force enum and guarded launch)
- `tests/arch_compile_gates.rs` (target-aware composition and real compile/resource gate)
- `tests/gemm_bi_fixed_correctness.rs` (raw-bit, prefix/view/graph and rejection coverage)
- `tests/gemm_bi_fixed_performance.rs` (force census, exact physical graph ABI and
  eager-only identity gate; authored/synchronized by main)

Final local and isolated-Ada hashes match:

| File | SHA256 |
|---|---|
| `tf32_rna_wide.cu` | `d6c96659d2d6c9bec9aed8abbf1d03f2d2384443aca771b3f4bc3ecad4c2aa34` |
| `modules.rs` | `9cdcd16e75f12186829feac76ea02ac2e1b1ab2f159bf0187c3cdc33441bfde5` |
| `kernels.rs` | `65eb0ad129851a443705a5b90c40015328bc7db4d0130166ffc43299b1e576e1` |
| `gemm_bi_fixed.rs` | `6eb12bc70325ec4e2aaa04dbf96a7115b12b7912f1537b66d78bdb7c6bba0a8c` |
| `arch_compile_gates.rs` | `607734545fda2f8f75be10e2f1c561cda5b7935f20f6c1a599145998f51e5ff1` |
| `gemm_bi_fixed_correctness.rs` | `2799861b06857ee19c8db26a788689692322b2c58f90c403c5e67da0823726ba` |
| `gemm_bi_fixed_performance.rs` | `aaea227608ea9b5ea66e96b55df3ba3f2e8cce86bf178c74c8f80395a7808899` |

Frozen performance test binary:
`/root/target-ada-rna-wide-force-20260906/release/deps/gemm_bi_fixed_performance-429f558dd590762a`,
SHA256 `db3dbafb1b1ccf3c0ea422d2860e8a7789e1eca9b9c14f51480711f18ac5dde1`.
The earlier 21-window screen used pre-empty-no-op binary SHA256
`2d7721f42aa08dccef786a0dead4e9242eb7c54d607ad2e494fecb35f1745982`;
the final 101-window confirmation uses the final binary above.

## RED evidence

All CUDA-feature tests ran on the isolated Ada source
`/root/mamba-ada-rna-wide-force-20260906`, target
`/root/target-ada-rna-wide-force-20260906`, CUDA/NVRTC 13.2, and private 0700
cache `/root/mamba-kcache-ada-rna-wide-force-20260906`.

- Composer/validator RED: 0/2 because the valid new export was not composed or
  admitted. Log `internal/perf/ada-rna-wide-force-20260906/red-composer-validator.log`,
  SHA256 `0d02308b897303d4d8fd8eec41e4dc5532b677f99675ca3c0b8d7b6a1505f8b2`.
- Census filter RED on the frozen pre-integration source: unknown RNA tile.
  Log `red-harness-filter.log`, SHA256
  `9a6954fe9453958c7f96de758d35381e7d9215f467af61309ac20240bdc7b2e4`.
- Eager-only physical identity regression RED: RNA eager returned false from
  the identity-graph predicate. Log `red-harness-eager-identity.log`, SHA256
  `2771b4a148f4331617eb7d949dd5dd467ee4e0188ba05e40b79de46e5ba8302f`.
- Empty-output RED on actual Ada: `M=0` with null operands and unused
  `K/N=usize::MAX` failed with `Fixed N exceeds i32` before reaching the launch
  no-op. Log `red-rna-empty-output.log`, SHA256
  `24f68fe36074ea82f7019a43567067193fbb11bf15be91e38b3ed06f3d4d496d`.

## GREEN evidence

Commands used the common environment:

```sh
export PATH=/root/.cargo/bin:/usr/local/cuda-13.2/bin:$PATH
export CUDA_HOME=/usr/local/cuda-13.2
export CUDA_PATH=/usr/local/cuda-13.2
export LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64:${LD_LIBRARY_PATH:-}
export CARGO_TARGET_DIR=/root/target-ada-rna-wide-force-20260906
export MAMBA_RS_KERNEL_CACHE=/root/mamba-kcache-ada-rna-wide-force-20260906
```

- `cargo test --release --features cuda,cudarc/cuda-13020 --lib fixed_sm89_ -- --nocapture`:
  36/36. `green-lib-fixed-sm89.log`, SHA256
  `a1d5404717f6d416070491899f5affaa8bcde151becfb30ad33248e6bd7bb898`.
- `cargo test --release --features cuda,cudarc/cuda-13020 --test arch_compile_gates fixed_sm89_ -- --nocapture`:
  3/3. `green-arch-fixed-sm89.log`, SHA256
  `ac32279f57754c2e220d2cfedc6c6df9884da8e316990f681e35ef089d229306`.
  The independent production-source compile
  also reports RNA registers 153, static shared 0, stack 0, spills 0, native
  HMMA and LDGSTS.
- `cargo test --release --features cuda,cudarc/cuda-13020 --test gemm_bi_fixed_correctness --no-run`:
  PASS. `green-correctness-no-run-v1.log`, SHA256
  `601d3ac853d3e7136a02f63bbee6e7e4a9fe94f974cb1f35c87fc6da2554cd87`.
- Real production loader/safety/K0 test: 1/1 in 76.59 s from a cold private
  cache. `green-correctness-k0-unsafe-v1.log`, SHA256
  `560fb1e2930d0d83c025f8fb083e31ee6e8fa994f94281f04825545167067114`.
- Real raw-bit comparison against all five incumbent Fixed TF32 rungs, both
  bias states, exceptional inputs, aligned subviews/C4 output, full prefix
  ladder, repeated eager and poisoned graph replay: 1/1 in 2.77 s.
  `green-correctness-rungs-prefix-graph-v1.log`, SHA256
  `236bca76baccff22d70f9b839aa2803eae168b3c4edf9556bc493050c95fb3b5`.
- Final expanded empty/K0 test: 1/1. RNA empty M/N now exits before conversion
  of unused dimensions; K0 compares all five incumbent rungs twice and RNA
  eager twice plus two poisoned graph replays, with exact five-argument bundle,
  geometry, output guards and bias immutability. Log
  `green-rna-empty-k0-expanded.log`, SHA256
  `44c096b25acd7df716bdd8558e8aef37dbbe00c0dd948b7b9fda05b4eec3e0a5`.
- Final expanded cross-route boundary matrix: 1/1. In addition to the retained
  K36/N132 tiny/view/tail matrix, every one of the five incumbent rungs is
  compared at the N132 AUTO underfill boundary M6016/6017/6018 and hot-A
  M4620/4621/4622, for finite/exceptional inputs and both bias states; RNA
  eager/graph repeats and guards remain checked. Log
  `final-green-rna-rungs-crossroute-boundaries.log`, SHA256
  `ab192de237695887155b10ea752244cb912da7aa5e601844271362a7509fdd06`.
- Harness RNA contract/filter/eager identity: 3/3; arch census: 1/1.
  `green-performance-rna-static-v2.log` and
  `green-performance-rna-census-v2.log`, SHA256 respectively
  `2e36e457f64cc89e1081c836c60ece0165f9b93ad693b404b935e6407bd3ea9d`
  and `6807496643b10ec060d5392da929143373d2b9445f81a798d1027594791e9e4c`.
- All explicit vendor static units: 12/12; complete forward/reverse force
  registry: 6/6. Logs `green-performance-explicit-vendor-units.log` and
  `green-performance-force-units.log`, SHA256 respectively
  `c78aa38d9f58d28f976ef316f07eac593de31032a8658278813750926475d7e7`
  and `64c6deac25e78af32a00f4afcc558e9d0e40b7f6155566e59ba6c41417f54309`.
- Final post-no-op module units: 36/36, RNA performance units: 3/3, census:
  1/1. Final logs SHA256 are respectively
  `82c04c30037076e879a6f7819b9c71cc4b3af54efa482aa229b7bc8d41f80935`,
  `53e36504d8f38ed6364d0953097bda6ca09eab3c027166528fe454e3f8aefedf`,
  and `6807496643b10ec060d5392da929143373d2b9445f81a798d1027594791e9e4c`.
- Direct `rustfmt --edition 2024 --check` on all six changed Rust files and
  code-only `git diff --check`: PASS. The final correctness test hash above is
  the rustfmt-only successor of the passing runtime source hash
  `ee8c5370406e743ea1d5f024582cf4d937f1451993ecffbdf869022f13f511c1`;
  no executable statement changed after the runtime run.

The live loader admitted the exact five-argument Driver ABI at offsets/sizes
`(0,8),(8,8),(16,8),(24,8),(32,32)` with a strict terminal sixth-argument
probe. Runtime/compile resources are registers 153, static shared 0, local 0,
stack 0 and spills 0; the 224-register, 256-thread, 98,304-byte opt-in-shared
and occupancy-at-least-one gates pass.

Current production identity from the real Ada run:

- Fixed source digest: `7ccad9938c4ee7cb0080d62b7053e082537fcb2ad184288ffc59a513cf526301`
- invocation digest: `4e99acb556ad1f6bb2aafa382f1618db9c9ecf37286fa7da37a62d86399c14ce`
- artifact digest: `c6dc1ee707ceceb512b58f226f8b5288097162c5257c41ef89a2e235fda1945f`
- header manifest: `e893dcebd4b2eb9d2e8cd84721c99684550413437d318113a8c45a6fa9ac1f73`

## Bounded eager-only smoke

With `MAMBA_FIXED_ADA_ROWS=tf32`, `MAMBA_FIXED_ADA_CELLS=hot_a`,
`MAMBA_FIXED_ADA_BIAS=0,1`, `MAMBA_FIXED_ADA_WINDOWS=21`,
`MAMBA_FIXED_VENDOR_TILES=Tf32RnaM128N128S3`,
`MAMBA_FIXED_VENDOR_PATHS=eager`, and exact CC8.9 admission, the harness emitted
four paired records and completed with `rejected:0, passed:true`.

Although only eager timing was requested, every record contains the forced
physical inventory and validates it before timing: one exact RNA kernel,
grid `[592,1,1]`, block `[256,1,1]`, shared 98,304; AUTO/vendor graph fields
remain null. Raw storage and repeat bits pass. Forced/AUTO p95 ratios were
0.82886 and 0.83341 (bias0 orders) and 0.83771 and 0.84637 (bias1 orders).
Log `green-rna-eager-hot-a01-w21-v2.log`, SHA256
`752f78910839b1487eceb5541dd1503d555b763dbf4e77e8220c0fab533355e8`.

The retained `green-rna-eager-a01-w21.log` is deliberately not GREEN evidence:
it documents an operator typo (`CELLS=A`) being rejected by the strict filter
before any GPU work. The corrected run uses the authoritative `hot_a` label.

## Qualification results and remaining policy boundary

- No correctness/admission blocker is known. The implementation is intentionally
  force-only, so no production route changes until a separate bounded promotion.
- The requested current-build 21-window screen is complete: hot A-E, bias0/1,
  eager+graph and both orders produced 40/40 records, `rejected:0`, `passed:true`.
  Every output/repeat bit gate passed, every graph replay bit gate passed, and
  every record carried the exact forced RNA physical identity. Worst p95
  forced/AUTO by cell and bias was: A0 0.84055, A1 0.85420, B0 0.73088,
  B1 0.73593, C0 0.62135, C1 0.62177, D0 0.89812, D1 0.90165, E0 0.87115,
  E1 0.87366. Log `screen21-rna-all5-both-paths.log`, SHA256
  `76942789166062f41734188a93e43f19a3de9dac44570b5c6a7ba4876aeb082c`.
- Final-binary 101-window confirmation is complete: 40/40 records,
  `rejected:0`, `passed:true`; every raw/AUTO/repeat bit gate and every graph
  replay gate passed, with the exact RNA symbol in all records. Worst p95
  forced/AUTO by cohort: A0 0.849957, A1 0.860678, B0 0.732515, B1 0.736051,
  C0 0.627831, C1 0.628870, D0 0.903182, D1 0.907060, E0 0.891869,
  E1 0.888943. Log `confirm101-rna-all5-both-paths-final.log`, SHA256
  `4983d29fa241538f170e087372c70baa92858033a7b075c7e53945088ba6d910`.
- The force-only checkpoint itself still does not authorize an AUTO selector
  change. Any AUTO promotion is a separate reviewed/TDD change after this
  checkpoint is committed.
- The new fragment intentionally changes the Ada Fixed source/artifact identity.
  The reported header-manifest change is also source-derived, not evidence of a
  toolkit change: `header_manifest()` conservatively encodes preprocessor/paste
  analysis of the combined source itself (including its recognized call stream),
  in addition to reached header bytes. Appending the RNA source therefore changes
  that digest even though the fragment adds no `#include`; the NVRTC library
  domain remains `d031a53e...`.
- Tests prove unchanged composed-source bytes for Ada Triad and CC12 Fixed/Triad.
  This task did not rebuild and compare their artifact digests, so it makes no
  cross-build artifact-identity claim for those modules. Existing graph
  invalidation correctly follows the changed aggregate Ada Fixed artifact identity.
