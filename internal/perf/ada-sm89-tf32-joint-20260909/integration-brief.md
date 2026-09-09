# Ada SM89 Joint TF32 Retained Winners Integration Plan

## Assembly correction: fifth retained cell and complete primitives

The retained-winner census found a missing row in the original four-cell plan:
NN TF32 d768-out `(2048,1536,768)`. Its retained baseline N96 add-half body
beats the actual prior AUTO by 18.1–18.6% in the saved screen. The subsequent
direct-epilogue variant did not clear the incremental 1% gate for this shape;
do not discard the baseline because that follow-up stopped.

The joint owner now has **five exports for five logical cells**, adding
`gemm_bi_nn_sm89_tf32_addhalf_m128n96_bk32_s3_v1` in an isolated `NN_N96`
namespace. It is byte-identical to the retained source after export rename,
with the shared-memory epilogue intact. Launch: grid `(128,1,1)`, block
`(256,1,1)`, 86,016 dynamic shared bytes, five-argument/64-byte ABI, alpha +1,
beta +0, no bias, contiguous strides `(1536,768,768)`. Resource bounds match
the retained N96 screen: 124 registers, local/static zero, occupancy one.
Its identity/selector/oracle/actual-AUTO qualification is required alongside
the other four cells. It adds no new module kind. The original four-count
statements below describe the earlier baseline and are superseded here.

The standalone primitives also require `gbf_aligned16`, copied exactly from
Fixed `common.cuh`, in addition to the two TF32 helpers. All four original
source sections remain unchanged. Current owner SHA-256:
`e1e8a2ad1d2d03b4d0e02730f087eab1c26cfc7712f867fbbead13b032e3624c`;
primitives SHA-256:
`c16e81fdcc4745352c97ee7daa39f2629716d7ebe38b6eea0a91393268303b0e`.
Native source contracts pass 28/28; this is not live module qualification.
Evidence: `../ada-triad-nn-addhalf-n96-screen-20260908/README.md` and
`../ada-triad-tf32-nn-n96-direct-epilogue-20260909/report.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task after exact-F32 TN Batch B2 lands. Keep AUTO fail-closed until each exact toolkit cohort passes live qualification.

**Goal:** Integrate the four retained Ada TF32 cells materialized by commit `560169d5` without changing any frozen existing CUDA module or route.

**Architecture:** Compile the sealed four-export source as a new exact-SM89 optional module. Model the three TN cells as a two-node pre-RNA transpose plus GEMM pipeline and the NN cell as one direct N96 GEMM; admit exact toolkit identities only after actual-AUTO eager/graph qualification.

**Tech Stack:** Rust, CUDA C++, NVRTC/Driver API, CUDA graphs, deterministic TF32 `mma.sync.m16n8k8`.

**Spec:** `.superpowers/sdd/handoff-codex-gemm-bi-triad-2026-09-05/task-4-tf32-brief.md`, corrected below for current HEAD after exact-F32 Batch B1.

## Global constraints and current source state

- Source owner: `kernels/gemm_bi_triad/sm89_tf32_joint.cu`, SHA-256 `5fbb9fc1481052da3591196f73f8d372a3616e07657dc13da4f2a1d8a1e95bd3`.
- Adapter: `src/mamba_ssm/gpu/gemm_bi_triad/sm89_tf32_joint_source.rs`; native parity/inventory test: `tests/gemm_bi_tf32_joint_source_contract.rs`.
- The owner has exactly four exports and exactly one physical transpose export. It is normalized byte-for-byte from the retained discovery transforms; it contains no `_test_` or `_exp_` export.
- The owner is not yet a standalone compilable module: it calls global `gbf_tf32_copy_cg` and `gbf_tf32_mma_m16n8k8`, whose definitions currently live in `kernels/gemm_bi_fixed/tf32.cu`. The joint composer must prepend only frozen, parity-tested copies of those two primitives. Including the whole Fixed TF32 source would add foreign exports and is forbidden.
- Current `ModuleKind::TriadSm89ExactF32` already owns value `10`; append `TriadSm89Tf32Joint = 11`. Preserve every existing value and module-local digest.
- Current `ArtifactSetIdentity` already carries finalist, half and exact-F32 optionals. Append `sm89_tf32_joint`; accept up to seven artifacts and preserve ordering `Fixed, TriadScalar, TriadSm80, TriadSm89Finalist?, TriadSm89Half?, TriadSm89ExactF32?, TriadSm89Tf32Joint?`.
- Do not append these exports to `Fixed`, `TriadSm80`, `TriadSm89Finalist`, `TriadSm89Half`, or `TriadSm89ExactF32`.

## Exact retained-cell map

All tuples below use the repository request convention `(M,K,N)`. Logical contiguous strides are `(lda,ldb,ldc)=(K,N,N)`. All operands and outputs are F32; policy permission is `AllowDeterministicTf32V1`; pointers must be non-null and at least 16-byte aligned (qualification uses 256-byte alignment); bias is absent.

| Cell | Exact request and operands | Production export/path | Physical params and launch | Discovery result | Current pre-integration AUTO that mutations must preserve |
| --- | --- | --- | --- | --- | --- |
| TN d768-in | `Tn (2048,768,3072)`, strides `(768,3072,3072)`, alpha `+1`, beta `+1` | common pre-RNA transpose → `gemm_bi_tn_sm89_tf32_pre_rna_m128n96_bk32_s3_v1` | transpose scratch stride `2048`, `1,572,864` f32; transpose grid `(24,64,1)`; GEMM params `(m,k,n,lda,ldb,ldc)=(768,2048,3072,2048,3072,3072)`; grid `(192,1,1)` | candidate/retained once7 p50 `.926431–.927083`, worst p95 `.932249`; Fast p50 `1.2610–1.2677` | portable `MmaTf32RnaV1(M64N64,S2)` |
| TN d768-out | `Tn (2048,1536,768)`, strides `(1536,768,768)`, alpha `+1`, beta `+1` | common pre-RNA transpose → same N96 export | scratch stride `2048`, `3,145,728` f32; transpose grid `(48,64,1)`; GEMM params `(1536,2048,768,2048,768,768)`; grid `(96,1,1)` | candidate/retained once7 p50 `.929648–.930000`, worst p95 `.935000`; Fast p50 `1.077675–1.091445` | portable `MmaTf32RnaV1(M64N64,S2)` |
| TN canonical Prism | `Tn (4621,384,1928)`, strides `(384,1928,1928)`, alpha `+1`, beta `+1` | common pre-RNA transpose → `gemm_bi_tn_sm89_tf32_pre_rna_m64n64_bk32_s3_v1` | scratch stride `4624`, `1,775,616` f32; transpose grid `(12,145,1)`; GEMM params `(384,4621,1928,4624,1928,1928)`; grid `(186,1,1)` | candidate/N96 once7 p50 `.912384–.918651`, worst p95 `.947053`; Fast p50 `1.675168–1.738473` | portable `MmaTf32RnaV1(M128N64,S3)` |
| NN canonical Prism | `Nn (4621,384,1928)`, strides `(384,1928,1928)`, alpha `+1`, beta `+0` | `gemm_bi_nn_sm89_tf32_addhalf_m128n96_bk32_s3_direct_v1`, one node | GEMM params `(4621,384,1928,384,1928,1928)`; grid `(777,1,1)` | candidate/N96 once7 eager `.98584–.98949`, graph `.98235–.98434`; Fast eager `1.01166–1.01359`, graph `.99614–.99971` | portable `MmaTf32RnaV1(M64N64,S2)` |

TN numerical order is frozen: the transpose applies `cvt.rna.tf32.f32` only to A and writes zero-padded `round_up4(M)` rows; the GEMM loads those A bits unchanged, applies `cvt.rna.tf32.f32` to B, and issues K8 steps `[0,8,16,24]` with FP32 accumulation. NN uses the retained integer add-half conversion (`bits + 0x1000`) for both operands, the same K8 order, and direct `float2` stores only on its proven aligned full-tile branch; tails retain the shared epilogue. None of the four is a strict all-path cuBLAS Fast win.

## Frozen ABI and resource contracts

| Export | Driver ABI `(offset,size)`; terminal | Block / dynamic shared | Retained CUDA13.2 resources |
| --- | --- | --- | --- |
| pre-RNA transpose | `[(0,8),(8,8),(16,12)]`; arg 3 invalid. Source argument order is **input, output, params**, correcting the older Task-4 brief. Params are `{rows,columns,output_stride}`, 12 B/alignment 4. | `(32,8,1)` / 0 | 26 regs, local 0, static shared 4,224 B, max threads ≥256 |
| TN N96 | `[(0,8),(8,8),(16,8),(24,8),(32,32)]`; arg 5 invalid | `(256,1,1)` / 86,016 B | 127 regs, local 0, static shared 0, occupancy 1 |
| TN M64N64 | same five-argument ABI | `(256,1,1)` / 49,152 B | 83 regs, local 0, static shared 0, occupancy ≥2 |
| NN direct N96 | same five-argument ABI | `(256,1,1)` / 86,016 B | 124 regs, local 0, static shared 0, occupancy 1 |

The five GEMM arguments are `(output, A, B, bias, GbfTf32N96Params)` and the 32-byte/alignment-4 bundle is `{alpha,beta,m,k,n,lda,ldb,ldc}`. Qualification must additionally prove stack 0, spills 0, sufficient max-dynamic shared, exact SASS mechanism (`HMMA`/`mma.sync`, async loads, `ldmatrix`, no atomics/reducer) and unchanged issue order. Discovery did not freeze production-module PTX/SASS or Driver ABI for the new exports.

## Required production touch points after exact-F32 TN B2

### 1. Isolated module and artifacts

- `sm89_tf32_joint_source.rs`: add four typed kernel specs, the 12-byte transpose params and 32-byte GEMM params, exact ABI/resource fields, and a `compose_source()` that prefixes only the two parity-frozen TF32 primitives. Keep `SOURCE` and its SHA unchanged.
- `gemm_bi_triad/mod.rs`: register/re-export the adapter/spec inventory.
- `kernel_identity.rs`: append `ModuleKind=11`, `ArtifactSetIdentity.sm89_tf32_joint`, seven-entry build/parsing/digest tests. Never renumber B2 identities.
- `gemm_bi_triad/modules.rs`: exact-`sm_89` target gate, four-export PTX inventory, instruction/ABI census, per-symbol resource exclusions, separate function map/compiler identity/rejection, and module anchor.
- `kernels.rs`, `context.rs`, `tests/kernel_identity*.rs`, `tests/arch_compile_gates.rs`: compile/surface the optional module and bind its independent artifact/compiler identity. A failed joint module must not disable finalist, half, exact-F32, portable, or scalar modules.

Per-symbol failure must be local: missing transpose disables all three TN cells only; missing N96 disables d768-in/out only; missing M64N64 disables TN Prism only; missing direct N96 disables NN Prism only.

### 2. Truthful routes and exact selectors

- `contract.rs`: append `Tf32PhysicalRoute::{Sm89TnPreRnaN96V1, Sm89TnPreRnaM64N64V1, Sm89NnDirectN96V1}`, joint kernel specs, `SM89_TF32_JOINT_TUNING_REVISION=1`, exact tiles/BK32/S3/configs, and `ModuleKind::TriadSm89Tf32Joint` mapping.
- `kernel_identity.rs`/`context.rs`: append GEMM backends `Sm89MmaTf32PreRnaV1` and `Sm89MmaTf32DirectN96V1`, plus numerical contracts for pre-RNA-A and add-half direct N96. Model transpose as an appended `PhysicalLaunchKind::InputTransform` with its own `Tf32RnaPreprocessV1` transform identity; do not label it as a GEMM backend or ordinary dtype upcast.
- `dispatch.rs`: add exact joint cell/cohort tables and lookup at the start of `measured_tf32_route_with_operands`, before finalist/specialized/portable lookup but after the existing K0 handling. Acceptance is exact op/dimensions/contiguous strides/scalar bits/bias/alignment, CC8.9/142 SM, exact `sm_89`, all compiler/artifact/header/NVRTC-library-domain/device identities, shared-memory capability and required symbols.
- Selector tests must snapshot the four current AUTO routes listed in the table, accept only the exact target, then mutate one field at a time and assert the byte-identical prior route. Explicitly reject NN/TN cross-op aliases, TN d768-in/out interchange, Prism N96 substitution, NN d768-out direct-N96 promotion, and neighboring dimensions/strides.

Keep the joint evidence cohort empty through module/route/launch work. Add CUDA12.8/13.0/13.2 rows only from that toolkit's own passing log; never copy a digest across toolkits.

### 3. Eager, graph and scratch ownership

- `launch.rs`: add `PreparedF32Kind::Tf32TnPreRna` holding both configs/parameter bundles, ordered physical identities, transpose pointer/required extent/allocation identity/epoch and launch-set digest. TN enqueue is exactly `A -> scratch`, then `scratch + B -> output`; no memset or reducer. The GEMM exclusively owns output. NN stays on the existing single-node `PreparedF32Kind::Tf32` machinery.
- Reuse `GpuCtx`/`GpuKernels::transpose_scratch_buf`; extend TN scratch sizing and `qualification_scratch_envelope`/`presize_physical_qualification_suite` for the three exact extents. Do not add an allocator.
- `blas.rs` and physical observer code: record the transform then GEMM in enqueue order with joint module identity, exact configs, argument digests and scratch resource identity. `prepare_prepared_f32_direct_graph_sequence`, eager launch and replay validation must preserve the same two nodes and reject stale pointer/extent/allocation epoch, managed epoch, artifact or symbol.
- Native tests must prove eager/prepared manifest equality, one default graph edge transpose→GEMM, captured scratch pointer equality, exact Driver arguments, rehydration, stale/undersized scratch rejection and unchanged existing routes.

### 4. Live qualification and admission

Add `tests/gemm_bi_tf32_joint_cohort_binding.rs`. Run independently on CUDA12.8, CUDA13.0 and CUDA13.2 with exact RTX 6000 Ada CC8.9/142-SM identity and an idle-GPU preflight. For every toolkit require:

Build the joint module once per toolkit and run the four cells sequentially in one batched qualification process, reusing the loaded artifact and presized scratch. Collect per-symbol ABI/resource/SASS records from that build, but do not rerun the full compile/correctness/timing suite separately for each export.

1. Exact four-export artifact, compiler/library-domain/driver/device identities; ABI and resource/SASS gates above.
2. Full, tail, exceptional, K0 and target exactness; independent pre-RNA scratch oracle; A/B immutability; two-sided output/scratch guards.
3. Actual AUTO eager and prepared identities: exact node count/order/module/symbol/config/arguments, one TN dependency edge, scratch allocation identity, and repeatable bits.
4. Paired candidate versus **actual current AUTO** and frozen retained comparator, eager and graph ABBA+BAAB once3→once7, with every p50 and p95 `<.99`. Report cuBLAS Fast separately without requiring or claiming a win.
5. Regression of the four already-admitted Ada TF32 NT finalist cells and the exact-F32/half optional modules.

Only after a toolkit passes all five gates: freeze its source/compile/artifact/header/library/driver/device digests, populate that one joint evidence row, rebuild, and run a post-admission actual-AUTO eager/graph qualification. CUDA13.2 discovery ancestry is not production qualification; CUDA12.8 and CUDA13.0 have no discovery coverage for these bodies.

## Safe commit boundaries

- [ ] Commit A: composable joint source/specs, `ModuleKind=11`, artifact loading, ABI/resources/exclusions and native module tests. AUTO cohort remains empty.
- [ ] Commit B: route/transform identities, forced launches, TN scratch ownership, eager observer and prepared graph rehydration. AUTO cohort remains empty.
- [ ] Commit C per toolkit: frozen pre-admission evidence and exact identity row only after live PASS.
- [ ] Commit D: enable only passed cohorts; run post-admission actual-AUTO qualification and update matrix/report/handoff.

At each boundary run scoped native tests, CUDA-feature `--no-run`, `git diff --check`, and verify that no existing CUDA source hash changed. Do not combine source materialization, unqualified identity and AUTO enablement in one commit.

## Evidence ledger and remaining gap

| Evidence | CUDA12.8 | CUDA13.0 | CUDA13.2 |
| --- | --- | --- | --- |
| retained discovery correctness/performance | missing | missing | present for all four cells; test-only composed modules |
| sealed joint source parity/inventory (`560169d5`) | native only | native only | native only |
| joint production module compile/PTX/SASS/ABI/resources | missing | missing | missing |
| forced production eager/prepared exactness | missing | missing | missing |
| candidate vs actual pre-integration AUTO | missing | missing | missing |
| admitted actual AUTO post-check | missing | missing | missing |

Authoritative retained reports:

- `internal/perf/ada-triad-tf32-tn-transpose-rna-n96-siblings-20260909/report.md`
- `internal/perf/ada-triad-tf32-tn-transpose-rna-n96-d768-out-20260909/report.md`
- `internal/perf/ada-triad-tf32-tn-transpose-rna-m64n64-prism-20260909/report.md`
- `internal/perf/ada-triad-tf32-nn-n96-direct-epilogue-20260909/report.md`
