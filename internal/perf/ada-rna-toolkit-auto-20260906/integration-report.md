# Ada RNA toolkit AUTO promotion report

Date: 2026-09-06. Status: source implementation, TDD, matching-toolkit
functional qualification, and quiet post-AUTO101 acquisition complete. The
exclusive Ada lane was explicitly released after all five timing invocations.
No commit, push, branch operation, deletion, signal, or unrelated-project
operation was performed.

Base/branch/worktree:

- base `c35646367b30acff74136520f1ad6e9cbb376b9e` (`c3564636`), branch
  `codex/gemm-bi-triad-sm80`;
- existing linked worktree
  `/Users/silvermpx/IdeaProjects/mamba-rs/internal/worktrees/gemm-bi-triad-sm80`;
- remote source `/root/mamba-ada-rna-toolkit-auto-20260906`;
- all Cargo builds/tests ran through
  `ssh -o BatchMode=yes -o ConnectTimeout=10 ada`; no local Cargo command ran.

## Result and source scope

Production AUTO now admits the existing Fixed-owned RNA-wide route only for
explicit NVRTC `(12,8)`, `(13,0)`, or `(13,2)` after the unchanged loaded
holder, known-library, CC8.9/142-SM, deterministic-TF32, homogeneous-F32,
C/A/B16, bias4, and exact A-E shape gates. The global route/graph epoch is 41.
Revision40, revision39, and revision38 captured identities are independently
rejected. No CUDA, composer, loader, compiler, resource/ABI, numeric contract,
schedule, old picker, other architecture, other precision, or Triad selector
was changed.

The implementation source/test diff contains exactly these six assigned files:

| File | Final SHA256 |
| --- | --- |
| `src/mamba_ssm/gpu/gemm_bi_fixed.rs` | `6a09002c52f16fe91af8601630df63b5c81865db0cffa5dd10ac99c9a314f93a` |
| `src/mamba_ssm/gpu/kernel_identity.rs` | `40371448e851d8d5af6cfea7648439a904604b7b0094d9157f7f81eb34857f19` |
| `src/mamba_ssm/gpu/gemm_bi_triad/dispatch.rs` | `c6c238b96763fe99b80fe680ea578df89eb0b1e3c463604fe1026e1dbc7a7b9c` |
| `src/mamba_ssm/gpu/gemm_bi_triad/launch.rs` | `65d1a442527c553b8a20f1d42aec4944f27326c19c8a57fb835a0e28c27bb17c` |
| `tests/gemm_bi_fixed_correctness.rs` | `4d07e82a2623f19c83ee9c7062935ab7077e7075ccc58fd57861d8a416781050` |
| `tests/gemm_bi_tf32_cohort_binding.rs` | `bd203d2c023f4352b70b59d355e915a46a76eb0576076b3f7ab5466e6e11f427` |

`git diff --exit-code HEAD -- kernels src/mamba_ssm/gpu/gemm_bi_triad/modules.rs
src/mamba_ssm/gpu/kernels.rs` exited 0. Direct
`/Users/silvermpx/.cargo/bin/rustfmt --edition 2024 --check` on all six files
and `git diff --check` both exited 0. The unchanged production RNA CUDA file,
composer, and loader hashes are recorded in `local-source-scope.log`. The
24-minute architecture suite was deliberately not rerun because none of those
sources changed.

## Source closure and cache isolation

The build-input closure contains 168 files: Cargo metadata/config, package
README/licenses, all `src/` and `kernels/` inputs, correctness/performance/
cohort-binding test targets, and
`tests/support/fixed_sm89_exact_n64_admission.rs`. The final sorted manifest
file hashes to
`21f828115c983114175615bf6b6c044c461b811d3cc11aad787697dff3f7a574`;
remote `sha256sum -c source-manifest-final.sha256` passed all 168 entries.

Targets and private root-owned mode-0700 caches were:

| CUDA | Feature | Target | Cache |
| --- | --- | --- | --- |
| 12.8 | `cuda,cudarc/cuda-12080` | `/root/target-ada-rna-auto-cuda128-20260906` | `/root/mamba-kcache-ada-rna-auto-cuda128-20260906` |
| 13.0 | `cuda,cudarc/cuda-13000` | `/root/target-ada-rna-auto-cuda130-20260906` | `/root/mamba-kcache-ada-rna-auto-cuda130-20260906` |
| 13.2 | `cuda,cudarc/cuda-13020` | `/root/target-ada-rna-auto-cuda132-20260906` | `/root/mamba-kcache-ada-rna-auto-cuda132-20260906` |

Each new cache was populated by copying, never mutating, the three blobs from
the corresponding `/root/mamba-kcache-ada-rna-full-cuda{128,130,132}-20260906`
cache before any run. Filename and complete-blob SHA comparisons remained
equal after all functional runs:

| CUDA | Cache key | Blob SHA256 |
| --- | --- | --- |
| 12.8 | `73fbec7557a95462a555e39f1b3c94e8d2bcc7047a14f8f3e3fa51a768ec5b9a` | `c02fb7a0ef6a50cb52dc3fec2e83ebb1eb6b4015a9f200d48d65fd4493d489c9` |
| 12.8 | `ae8e2e3db0255db26419c8292c70ea945f46076770ddb3522d34443addeaf374` | `22827bf4ea23b939d8ec862b276f39b02c53ac2389345df1c16f62ca293c23fb` |
| 12.8 | `da5dabeff55570d1fb6ca43a1d4a8aa7a2a9d25efb6436ed7f03c4a28a73d986` | `48a388fc610d110bb884ea3cd0f49eda299c0c7ce0c8e2beb24d34a925198a54` |
| 13.0 | `4d6815a9cdc06297113b72dc2d9ae9fac46b50a0583c561f358727d4a6f2cf49` | `cc518c145df2f13d1a76334dd6b5572a27cfc9d6887b6d72c273f51a39b58549` |
| 13.0 | `ceeda6e9d55cfd166dafd638a9055daf8a1bb044f573c3349e147bff03b8fe7b` | `c599a8b8807dd944ae79238241b36f77599cb5cab0511d3ab4f8716bc7545c3c` |
| 13.0 | `feac83eb41fcc629d0c4bf9be1758d9e92c431728e17c1f46cfcd5c5a9ebc27d` | `32e61498d102e2343b2c3b65ec062af81b1351b1039a2503c0657628b4d3a9c0` |
| 13.2 | `3809af034718aa08dfdfc1b6baeb407715436dc7db87ef39825b033c2fd72541` | `5851ebd49595a0b2ef665073289e52f903eec3266deadfc5ad5d9f2096ce6d16` |
| 13.2 | `4e99acb556ad1f6bb2aafa382f1618db9c9ecf37286fa7da37a62d86399c14ce` | `220ccfc809011d7738b7447a319ecb0c1b2b12c49b3766ca0315d28fa3d5153f` |
| 13.2 | `822ebf0dda7dcf9c4f5c5a2213884dde42ab45f5cf238479a81856042c302b7c` | `5035921fc8e078e5454c9c5392f9e9ee6ca6a3de54c1e08cb54bba432fa37546` |

## TDD and debugging evidence

Against the test-first CUDA12.8 snapshot, this build exited 0:

```sh
cargo test --release --features cuda,cudarc/cuda-12080 \
  --lib --test gemm_bi_fixed_correctness --no-run
```

The fully qualified selector regression then exited 101 because the first
newly positive 12.8 assertion was false:

```sh
cargo test --release --features cuda,cudarc/cuda-12080 --lib \
  mamba_ssm::gpu::gemm_bi_fixed::sm89_rna_auto_tests::fixed_sm89_rna_auto_exact_cells_and_independent_declines \
  -- --exact --nocapture
```

The revision40 replay regression exited 101 with actual current epoch40 versus
expected41:

```sh
cargo test --release --features cuda,cudarc/cuda-12080 --lib \
  mamba_ssm::gpu::kernel_identity::physical_launch_tests::ada_rna_toolkit_auto_epoch_rejects_revision40_graph_identity \
  -- --exact --nocapture
```

The live CUDA12.8 hot-A AUTO regression exited 101 with actual
`Tf32M64S2` versus expected `Tf32RnaM128N128S3`:

```sh
cargo test --release --features cuda,cudarc/cuda-12080 \
  --test gemm_bi_fixed_correctness \
  fixed_sm89_rna_wide_actual_auto_hot_a_route_and_graph \
  -- --exact --ignored --nocapture --test-threads=1
```

An earlier unqualified library filter combined with `--exact` selected zero
tests and exited 0. It is preserved as `red-selector-cuda128.log`, explicitly
excluded from RED evidence. `--list` identified the full namespace, after
which the command above produced the intended behavioral RED.

After the minimal production change, the focused selector and revision40
tests each passed 1/1. The first complete library attempt then exposed two
current-epoch fixtures in assigned files: `(5,40,8)` versus actual `(5,41,8)`
and `F32_TF32_TUNING_REVISION=40` versus its global alias value41. That run is
preserved as `green-lib-cuda128.log` (632 passed/2 failed/46 ignored, exit101).
Only those two current-epoch expectations were updated; historical 38/39/40
replay values were retained.

## Matching builds and functional verification

Every command used an explicit toolkit environment of this form, with the
version-specific paths from the table above:

```sh
export CUDA_HOME=/usr/local/cuda-<version>
export CUDA_PATH=/usr/local/cuda-<version>
export LD_LIBRARY_PATH=/usr/local/cuda-<version>/lib64
export PATH=/usr/local/cuda-<version>/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CARGO_TARGET_DIR=<exact target>
export MAMBA_RS_KERNEL_CACHE=<exact cache>
```

The matching final-source commands and results were:

Chronology clarification by main: the initial green build logs describe the
pre-amend source manifest0f8762c1dada48707d404a39ba1357b3b5d65e8c11f198998f27e9b84281e483.
The final library jobs recompiled after the two epoch fixtures were corrected;
subsequent test targets and final binary hashes correspond to final21f828...
source. The earlier logs are retained as history, not substituted for final
source verification.

```sh
cargo test --release --features <matching-feature> --lib \
  -- --nocapture --test-threads=1
# each CUDA version: exit 0, 634 passed, 0 failed, 46 ignored

cargo test --release --features <matching-feature> \
  --test gemm_bi_fixed_performance -- --nocapture --test-threads=1
# each CUDA version: exit 0, 43 passed, 0 failed, 62 ignored

cargo test --release --features <matching-feature> \
  --test gemm_bi_fixed_correctness fixed_sm89_rna_wide_actual_auto \
  -- --ignored --nocapture --test-threads=1
# each CUDA version: exit 0, exactly 2 passed, 0 failed, 448/448 unique groups
```

Every actual-AUTO run covered the complete tail/A-E, finite/exceptional,
bias0/1, C4/C16, row/prefix, five-rung, guard/input, eager-repeat and
poisoned-graph corpus. The admitted exact hot rows physically launched the RNA
symbol; misaligned A/B retained M64S2 for A/B/D/E and for C on13.2, while C on
12.8/13.0 retained M128S2. The wrapper now requires a known runtime library and
one of the three exact NVRTC versions.

The following CUDA13.2 retention commands each exited 0 and passed 1/1:

```sh
cargo test --release --features cuda,cudarc/cuda-13020 \
  --test gemm_bi_tf32_cohort_binding tf32_cohort_binds_on_this_board \
  -- --exact --ignored --nocapture --test-threads=1
cargo test --release --features cuda,cudarc/cuda-13020 \
  --test gemm_bi_tf32_cohort_binding \
  sm89_tf32_bias_cohort_serves_the_qualified_wide_epilogues \
  -- --exact --ignored --nocapture --test-threads=1
cargo test --release --features cuda,cudarc/cuda-13020 \
  --test gemm_bi_fixed_performance \
  fixed_sm89_tf32_c_auto_prefix_special_bias_graph_bits \
  -- --exact --ignored --nocapture --test-threads=1
```

Final executable identities after all builds/functional jobs:

| CUDA | Library test SHA256 | Correctness SHA256 | Performance SHA256 | Cohort SHA256 |
| --- | --- | --- | --- | --- |
| 12.8 | `6f85e7dc251850a0ebfea45194f3eff3c79a3c44a046610b2b17dec6348c9183` | `96721d176a28f0e1a1627f93c7e60f637820f455afa108d25c7cdf98a059b40f` | `a013bdae3f6da79adfe877854c7a07589ddd269456c27ec64d8519bce531f228` | `2100bc4443edef1fcc7c1207a9cc384b21c3469a02d68082be8a8e9c0b28aee0` |
| 13.0 | `d4efb887b239cdb6d5bc41f8b79e8955dc368433448366e2c2b218b744ffa9e6` | `73afb5ff7364b4ec355fb3041f760f8aaaece64e68552557b1a83c47c9acd868` | `c0cf51e9eb69772df604dfd4f42e7ac92ee82aede726cd3dbe0051af1ae5ebc3` | `31b7b46c1264442eb84fc5f5669f3fb0ea617162d8fc3ae1fa90f54378044411` |
| 13.2 | `541c00e1bcc719d14b9081367448f24a0173ad9479f8bba58c4ad5d28400f374` | `fb468ff9ffad5813909d1cec0684402b805291f80e15582b138295868846d495` | `b8d8cd24b5e44ca27749c1950fb17802477dba432e469ed613c396910e542b38` | `4f685222acedc7e5c3302346910339ed5610f5854a9e53c6b08f2e1eceed7d6c` |

The literal executable paths are recorded in `final-identities.log`; each stem
resolved to exactly one executable in its listed target.

## Quiet post-AUTO101

All compilation and functional jobs had exited before timing. Each toolkit
preflight showed GPU/memory utilization 0%, 90 MiB used, P5, 1800 MHz, 300 W,
and the exact performance binary SHA above. No build or competing GPU task
overlapped timing.

For CUDA12.8 and13.0, A/B/D/E used this direct-binary command (32 records):

```sh
MAMBA_FIXED_ADA_VENDOR=1 MAMBA_FIXED_VENDOR_EXACT_CC=8.9 \
MAMBA_FIXED_ADA_ROWS=tf32 MAMBA_FIXED_ADA_BIAS=0,1 \
MAMBA_FIXED_VENDOR_PATHS=eager,graph MAMBA_FIXED_ADA_WINDOWS=101 \
MAMBA_FIXED_ADA_CELLS=hot_a,hot_b,hot_d,hot_e \
MAMBA_FIXED_VENDOR_TILES=Tf32M64S2 \
<exact performance binary> \
fixed_ada_forced_rungs_paired_precision_cublas \
--exact --ignored --nocapture --test-threads=1
```

C used an otherwise identical invocation with
`MAMBA_FIXED_ADA_CELLS=hot_c` and
`MAMBA_FIXED_VENDOR_TILES=Tf32M128S2` (8 records). CUDA13.2 used one invocation
with all five cells and `Tf32M64S2` (40 records). All five invocations exited
0. Each toolkit independently produced exactly 40 unique
`(cell,bias,path,order)` records and zero rejection records. The12.8/13.0
completion records individually contain32 and8 records;13.2 contains40.
Every completion has `rejected=0,passed=true`.

All 120 records have actual AUTO `Tf32RnaM128N128S3`, epoch41, known matching
NVRTC, the RNA physical graph with one kernel/zero non-kernel nodes, 256
threads and 98,304 shared bytes, the exact old tile per cell, all raw/AUTO/
repeat/vendor bits, graph replay on graph records, 101 positive finite samples
per arm, explicit `CUBLAS_COMPUTE_32F_FAST_TF32`, PEDANTIC reference, and bias
broadcast timed iff bias is present.

The loaded Fixed identities remained exactly the committed pre-promotion
identities:

| CUDA | Source | Invocation | Artifact | Header manifest | NVRTC library domain |
| --- | --- | --- | --- | --- | --- |
| 12.8 | `7ccad9938c4ee7cb0080d62b7053e082537fcb2ad184288ffc59a513cf526301` | `ae8e2e3db0255db26419c8292c70ea945f46076770ddb3522d34443addeaf374` | `c71288517eb76b839ce23b2f915b010ca73b817b08d5aa0eb2e88a7b2c9f2a3e` | `9924f331b7c7e70041f74e8a9b39d072930c493c921ddf19beb34b6263682fc8` | `26b0a3a02044ffcbc1693fd83e9261beffa692a4fbcfe3ac5e9d8c87980bb155` |
| 13.0 | same | `4d6815a9cdc06297113b72dc2d9ae9fac46b50a0583c561f358727d4a6f2cf49` | `c90cd431d3c2df95e849b99f6fcef8e6a7e28f64d97e851ec39f2445a7dc5822` | `7801fef3bdeb57597ff2028997aa9d6190fe63dec216b8bf0d1d5a07235f8685` | `709b91c36bfb0ed966ee69adc8d6f87ff110eecf3dfb5060367f183ce614eb0d` |
| 13.2 | same | `4e99acb556ad1f6bb2aafa382f1618db9c9ecf37286fa7da37a62d86399c14ce` | `c6dc1ee707ceceb512b58f226f8b5288097162c5257c41ef89a2e235fda1945f` | `e893dcebd4b2eb9d2e8cd84721c99684550413437d318113a8c45a6fa9ac1f73` | `d031a53eb97235b70f62f652932db1bdf728ea229c8ca809d53c5ffd91642687` |

Ratios below are recomputed from matched sample arrays as `AUTO/forced-old`
and `AUTO/FAST`, never by inverting a reported forced/AUTO p95. Each entry is
the worst of eager/graph and both launch orders; lower than one wins.

| CUDA | Cell | Bias | AUTO/old p50 | AUTO/old p95 | AUTO/FAST p50 | AUTO/FAST p95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
|12.8|A|0|0.884219|0.891391|1.163780|1.174750|
|12.8|A|1|0.892445|0.910683|0.934008|0.956254|
|12.8|B|0|0.784974|0.796153|1.196046|1.211875|
|12.8|B|1|0.788382|0.799958|1.049718|1.066434|
|12.8|C|0|0.648405|0.648678|1.437987|1.438571|
|12.8|C|1|0.649443|0.649841|1.316131|1.316724|
|12.8|D|0|0.965933|0.969599|1.370208|1.384029|
|12.8|D|1|0.969123|0.970660|1.189273|1.198244|
|12.8|E|0|0.946648|0.966892|1.437137|1.479948|
|12.8|E|1|0.949455|0.972474|1.343938|1.390535|
|13.0|A|0|0.884315|0.908074|1.165462|1.204894|
|13.0|A|1|0.893030|0.919966|0.944355|0.975619|
|13.0|B|0|0.784842|0.791617|1.199961|1.213308|
|13.0|B|1|0.788547|0.794647|1.056413|1.067244|
|13.0|C|0|0.649070|0.660606|1.438515|1.465293|
|13.0|C|1|0.650017|0.654364|1.316998|1.326611|
|13.0|D|0|0.965539|0.972845|1.370312|1.417571|
|13.0|D|1|0.968968|0.992156|1.190065|1.234843|
|13.0|E|0|0.947190|0.965989|1.437677|1.487687|
|13.0|E|1|0.949573|0.971106|1.345848|1.384659|
|13.2|A|0|0.829316|0.847653|1.086109|1.111950|
|13.2|A|1|0.838617|0.866844|0.884519|0.908820|
|13.2|B|0|0.725168|0.728916|1.102039|1.112077|
|13.2|B|1|0.729183|0.733408|0.969868|0.982497|
|13.2|C|0|0.624798|0.632038|1.314799|1.337789|
|13.2|C|1|0.625519|0.629951|1.213824|1.228891|
|13.2|D|0|0.897008|0.918121|1.266337|1.298603|
|13.2|D|1|0.900102|0.905076|1.100613|1.135001|
|13.2|E|0|0.870559|0.887461|1.308243|1.345904|
|13.2|E|1|0.873217|0.889227|1.227312|1.259524|

Every one of the ten shape/bias cells wins against its own prior route at both
p50 and p95 in all four cohorts for every toolkit, satisfying full promotion.
Actual FAST wins remain only A1 on12.8, A1 on13.0, and A1/B1 on13.2. The other
9/9/8 cells remain FAST gaps; worst AUTO/FAST p95 is 1.479948 on12.8 E0,
1.487687 on13.0 E0, and 1.345904 on13.2 E0. This is not an all-FAST,
all-precision, all-GPU, or all-inference completion claim.

## Evidence

Main archival addendum: the complete immutable worker bundle described below
is now copied to this tracked bundle's `raw/` directory, with all39 file
hashes and its original manifest retained. Main's independent13.0 actualAUTO
recheck passes2/2,448groups,38.69s. `main-remote-check.log` gives the exact
168-source verification and six live binary/nine cache-blob hashes, while
`main-independent-verification.json` names the prerequisite cache comparison
basis. The root README and manifest define the final archive; the following
ignored-path description records the worker's original handoff location.

The complete raw bundle is in the ignored directory
`.superpowers/sdd/handoff-codex-gemm-bi-triad-2026-09-05/ada-rna-toolkit-auto-evidence/`.
It includes every RED/build/library/static/actual-AUTO/retention/preflight/
timing log, both source manifests, final executable/cache identity logs, the
validator and its output. Its 39-entry `SHA256SUMS` file hashes to
`3173d5830bb4272ed088abe37c76e102635c581cefe701f2c1bcd8baa39acf38` and
`shasum -a 256 -c SHA256SUMS` exits 0. Raw log whitespace was not rewritten.

Main independently validated the source diff, exact 448-group corpus for all
three toolkits, all 120 timing records, old-route mapping, identities, epoch,
physical graph, bits, FAST/bias semantics, and samplewise quantiles. An
independent read-only reviewer reported no source or amended-helper issues.
Main owns selection into tracked evidence/docs, retirement-note updates, and
the human-authored commit.
