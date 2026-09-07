# Kernel replacement and retirement ledger

Owner instruction 2026-09-05: immediately route verified winners, stop choosing
the beaten candidate in its verified cell, and retain an explicit deletion
list. No source has been deleted by this ledger. Times below are microseconds.

## Current Fixed assembly checkpoint — 2026-09-07

Phase1 commit `0a95a00f` qualifies three independent forced routes on
CUDA12.8/13.0/13.2: RNA TF32 M128N96/S3, F16 M64N64/S3 (D), and
F16 M128N64/S2 (E). AUTO45 now connects exactly five rows: N96 E0/no-bias
on all three toolkits, F16 D0/E0/no-bias only13.2. ActualAUTO bit/graph gates
and all five once101 comparisons pass against their replaced routes.
See `internal/perf/fixed-finalist-integration-20260907/phase2-README.md`.
Old RNA N128 is no longer selected for these three exact aligned E0 rows;
old Swizzle/Pipeline are no longer selected for the two exact13.2 F16 rows
while their corresponding finalist holders are live. Lower-toolkit D/E
remain unchanged because their new candidates lost the admission test.

- Keep all three production finalists: each has a measured successful row.
  D/E losses on12.8/13.0 do not make the kernels deletion candidates.
- Keep RNA N128, Pipeline, Swizzle and generic rungs: other hot cells, bias,
  toolkit-specific preferences and independent-holder fallbacks still use them.
- Keep rejected W4 TF32, compact B0 half and exact-F32 T256 experiments separate
  from production. Their tests/source/evidence remain archival cleanup
  candidates, not active AUTO implementations. T256 stopped at local-memory
  resources before timing; do not relabel it a measured performance loss.
- No production CUDA deletion is authorized by this batch. Removal requires
  a complete remaining-use audit, including the unavailable architectures.

## Rules

`screened` means standalone only; `production-qualified` means actual NVRTC
loaded symbol passed bits/ABI/resources; `AUTO` means captured production launch
proved routing. Only AUTO entries can retire an incumbent in that cell.
`delete-ready` additionally requires every remaining live use to be replaced.
Different architecture, compiler, bias, shape and numeric policy are separate.

| Candidate / incumbent | Scope | Evidence | Status / disposition |
| --- | --- | --- | --- |
| Fixed half Tc128 -> pipeline_v1 | Ada BF16/FP16, exact hot A-E, bias 0/1, CC8.9/142SM/NVRTC13.2, ABC16 | `internal/perf/fixed-half-production-20260905/`:80 pre+80 post-promotion NVRTC records, eager+graph, both orders101windows,20/20 old wins (post worst p95 0.8993),12/20 robust nativecuBLAS wins; all raw gates PASS | AUTO actual hot-boundary and final paired graph-symbol proof PASS; Tc128 removed from AUTO only in these20 cells, not delete-ready |
| Old Fixed TF32 -> Fixed-owned RNA wide | Ada TF32 hot A-E, bias0/1, CC8.9/142SM/known NVRTC13.2, admitted holder, C/A/B16, bias4 | force integration `82fc400f`; `internal/perf/ada-rna-wide-auto-20260906/`: actual AUTO all-five-rung/prefix/view/replay gates,40/40 quiet post-AUTO101 records; AUTO/old worst p95 0.626263--0.913177, FAST wins A1/B1 only | AUTO with routing epoch40; old M64S2 removed only from these ten qualified aligned cells. Original add-based Triad wide remains distinct; ordinary rungs remain essential for decode, C4 and other shapes/alignments/devices/toolkits. Nothing deletion-ready |
| RNA wide fastfinite twin | Ada TF32 hot A bias1 | `internal/perf/nn-rna-fastfinite-probe-20260905/`:145.45 vs explicit RNA112.94, FAST131.78 in one build; all bits passed | rejected performance candidate; never wired into production; keep ignored experiment as negative evidence |
| TN Parts4 align16 -> align128 | Ada TF32 TN large beta1 | `internal/perf/tn-wide-probe-20260905/`:214.68 ->185.83 vs FAST141.58; exact same Parts4 output | screened own improvement, still loses vendor; no production replacement yet |
| Original half baseline/vec/pipe experiment arms | Ada standalone comparative controls | same-build half census above | keep evidence controls; only pipe_vec proposed for production, not four redundant exports |
| Fixed half pipe_vec -> packed XOR staging | Ada hotB BF16 bias1 | `internal/perf/half-swizzle-probe-20260905/`: old157.62,pipe135.99,swizzle109.21,nativecuBLAS128.79; paired swizzle/vendor0.839568; full strict gates PASS | screened vendor win; separate ignored candidate, no production composition change yet |
| Same packed XOR staging | Ada hotB BF16/FP16 bias0 | BF16 pipe137.11->swizzle112.08/vendor92.31; FP16 pipe149.22->128.15/vendor110.87; all strict gates PASS | screened own wins, both still lose nativecuBLAS; no deletion-ready source |
| Packed XOR align16 -> align128 only | Ada hotB BF16/FP16 bias0 | `align128-hot-b-*-bias0.jsonl`: all strict gates/504samples each; direct paired new/old p50 1.000230204 and1.000163918 | rejected as an improvement; aggregate medians are not a paired causal result; never AUTO, retain ignored negative evidence |
| Fixed exact N64 -> N64 copy-plan | Ada exactF32 hotE/B, bias0/1 | Actual NVRTC force291fixtures+all4sanitizers PASS; pre-AUTO `production-paired-101-v1.jsonl` SHA91493d… and post-AUTO SHA376610… each48ABBA/BAABpairs/19392positions; post-AUTO own worstp95 .8232-.8340; PEDANTIC p95 E0 .9916,E1 .9671,B0 1.1155 LOSS,B1 .9832 | AUTO commit `35b291b1`; old N64 removed from AUTO only in these4cells, not delete-ready because all other exact Ada shapes/alignments/toolchains still use it |
| New5090 exact fallback -> native TF32 S2 | CC12.0/170SM/CUDA13.2/driver595.58.03; NN M/K/N2048/768/3072,2048/1536/768,4621/384/1928; alpha1,beta+0,no bias | retained full36+fourtools; fresh guarded vendor-r39/v2/vendor-101.jsonl SHA9209b997…24records+completion, all eager/graph/orders win installed cuBLAS130400 FAST_TF32; worst AUTO p95 .903631/.943909/.973642; actual AUTO3symbols PASS,568lib/clippyPASS | AUTO738fe5ad; guarded vendor proof42a02d0d. Old facade zero-guard claims remain retracted; new proof includes actual F32 trailing32 guards/input readback and vendor prefix/suffix32 guards. Both1024-row small cells keep exactfallback. No source delete-ready |
| Fixed SM120 A1 postbias M128N96 | RTX5090 CC12.0/170SM/NVRTC13.2; exact-F32 `(4621,384,1928,bias)`, alpha1/beta0 | `internal/perf/sm120-fixed-tma-fma-a1-postbias-20260906/sm120-fixed-a1-m128n96-fast21-v1.log`, SHA256 `388d10281ed4089fafc5aea6eff8c81e918094b299c0d40bb0367b11172f733e`: strict bits/graph/symbol/ABI PASS; versus current AUTO worst p95 1.0393, versus Legacy 0.9669, versus PEDANTIC 0.7944, versus FAST_TF32 1.2598 | rejected for AUTO; retain force-only until owner-visible cleanup because it improves Legacy but is slower than the incumbent and FAST |
| Fixed SM120 postbias shared align1024 -> align128 | RTX5090 CC12.0/170SM/NVRTC13.2; M128N64 and M64N128 | emitted-PTX patch plus live Driver occupancy query: M128 regs128/dyn24592/active3 and M64 regs124/dyn24592/active3 unchanged; cubin still accounts 1024 B shared allocation | rejected before timing; source restored to align1024 because the hypothesis did not change residency |
| Fixed SM120 A1 postbias M128N64 K4 loop twin | RTX5090 CC12.0/170SM/NVRTC13.2; exact-F32 `(4621,384,1928,bias)`, alpha1/beta0 | `internal/perf/sm120-fixed-tma-fma-a1-postbias-20260906/sm120-fixed-a1-k4-fast21-v1.log`, SHA256 `5f7c3570d06fbbd836f7e37f4a88d9229190a4dbc8393a601a2dd875845af57f`: strict bits/graph/symbol/ABI PASS; versus current AUTO worst p95 1.0545, versus Legacy 0.9803, versus PEDANTIC 0.8098, versus FAST_TF32 1.2370 | rejected for AUTO; smaller runtime-loop body is slower than the fully unrolled incumbent, retain force-only as negative evidence pending cleanup |

## Production AUTO replacements already implemented

- Ada Fixed explicit-RNA toolkit expansion (supersedes the13.2-only routing
  epoch below): the same guarded A-E/bias0,1 cells now select
  `gemm_bi_nn_fixed_rna_wide_tf32_v1_m128n128_bk32_s3` on known
  NVRTC12.8/13.0/13.2, CC8.9/142SM, homogeneousF32/TF32 policy, admitted
  holder, C/A/B16 and bias4. Global epoch41 requires graph recapture;
  no CUDA/composer/loader/numeric/schedule identity changed. Actual AUTO
  two-wrapper448-group proof passes per toolkit; main repeated13.0. All120
  quiet101 records pass and each toolkit retains10/10 own p50+p95 wins.
  Worst p95 AUTO/old ranges12.8:0.648678--0.972474,
  13.0:0.654364--0.992156,13.2:0.629951--0.918121. FAST wins are A1 only
  on12.8/13.0 (p950.956254/0.975619), A1/B1 on13.2
  (0.908820/0.982497). The replaced old route is M128S2 for C on12.8/13.0,
  otherwise M64S2. Both remain essential for decode/unqualified shapes,
  C4/misaligned A/B, other devices/toolkits/policies and forced controls;
  neither is deletion-ready. All nine matching-toolkit cache blobs retain
  their qualified hashes. Evidence:
  `internal/perf/ada-rna-toolkit-auto-20260906/` including raw physical launch
  records and main's explicit live source/binary/cache hashes. No deletion.

- Ada Fixed explicit-RNA wide: actual AUTO at A-E/bias0,1 under the guard
  above; one256-thread/98304-shared launch with five-argument32-byte bundle.
  Post-AUTO101 SHA `3dbb8754b651706d67c83ab1f96fb26ed66a63bc161e8ced015f36ef50831f46`,
  all ten own wins; A1/B1 FAST worst p950.904797/0.980568. Old M64S2,
  M128S2 and the other rungs remain live outside the exact admission. Global
  epoch40 requires recapture; all three Ada compiled cache blobs unchanged.
  Evidence `internal/perf/ada-rna-wide-auto-20260906/`.

- Earlier Ada ordinary TF32 C0/C1 `(4621,1928,384)`, CC8.9/142SM/knownNVRTC13.2:
  M128S2 -> M64S2. Actual AUTO post101 passes all8bias/path/order records and
  strict bits; old/new paired p50 1.036022--1.042052, p95
  1.038318--1.045031. Prefix/specialbias/graph gate1/1 PASS. FAST remains
  1.912826--2.121482x faster by the AUTO/vendor median-time ratio. Existing
  M128S2 remains essential for other cells/toolchains; not deletion-ready.
  Evidence `internal/perf/fixed-tf32-c-postauto-ada-20260906/`. The newer RNA
  AUTO checkpoint supersedes this M64 choice on qualified C16 calls; C4 and
  other declines still use the ordinary picker. These older FAST ratios are
  historical, not the current RNA AUTO result.

- `6e7ed213` expands Ada exact-F32 CopyPlan from B/E to A/B/D/E,
  bothbias, CC8.9/142SM/knownNVRTC13.2 and existing aligned/loaded/policy gates.
  Post-AUTO A/D101: all32 records PASS, oldLegacy/AUTO p50
  1.181364--1.204765; AUTO/PEDANTIC16/16 wins, AUTO/FAST0/16. Expanded
  prefix/view/poisoned-graph suite passes704 boundary records. OldLegacy
  remains required for C, nonqualified shapes, alignments and toolchains;
  it is not deletion-ready. Evidence:
  `internal/perf/fixed-sm89-ad-copyplan-postauto-20260906/`.

- Committed `35b291b1`: Ada-only exact-F32 N64 copy-plan
  `gemm_bi_nn_fixed_sm89_f32_n64_copyplan_v1` is AUTO at E/B x bias0/1,
  CC8.9/142SM, known NVRTC13.2, exact policy and admitted/aligned operands.
  Actual five-argument graph ABI, force291 corpus, four sanitizers, resources
  135regs/32768static/0local/3CTA, cold/warm identity and retained half/Triad
  regressions pass. Post-AUTO101 reproduces all4 own wins and3 PEDANTIC wins;
  B/no-bias remains an explicit vendor loss. Old N64 stays required elsewhere.

- Committed `738fe5ad`: fresh595.58.03 cohort admits exactly the three SM120
  NN alpha1/beta+0/no-bias cells above. Revision39 moves both global andTF32
  epoch together; priorcohort identities retained, carryforward is NOT fresh
  other-card qualification. Actual specialized artifact1cbfd231… unchanged,
  complete current-r39 proof in `internal/perf/sm120-restart-20260905/current-r39/`.
  Fixed/oldAdaTF32 regression564lib+twoAdaGPUcohorttests PASS. Exact source
  remains necessary for nonqualified cells, policies and other hardware.

- Committed `e72fbab4` Fixed source+AUTO phase: exact symbols
  `gemm_bi_nn_fixed_sm89_tc128_pipeline_v1_{bf16,f16}`, one256-thread
  launch/71680 shared bytes. Five (M,K,N): (4621,384,1928),
  (4621,768,2304),(4621,1928,384),(2048,768,2304),(2048,2304,768).
  Both bias modes, homogeneous half only, allABCnonnull16, optionalbias4.
  Admission is VERSION-SCOPED with known libraries and the live optional
  ABI/resource loader, not a digest-frozen selector. Measured Fixed source
  c191aeb5b0c0be2adc12f2a887c6b93ee7bd072e4ac89d79a2a228a406d3824a,
  artifact3f3538334f6612bc2aa9fb9db6d34c8963b76aa09a5d32f441033944d658ab1a.
  Full557 normal lib tests and3 explicitGPU tests passed; GPU25.92s includes
  actualAUTO hotM±1, smallM, row17, C2 fallback, finite/exceptions, graph
  symbols/poison/replay/guards. New Fixed source and AUTO ship TOGETHER:
  aggregate artifact identity changes, global tuning38/othercohorts retained.
  Later host-only retune needs intentional dispatch-identity revision.
  Post-AUTO101 records independently captured the candidate in all40 path/cell
  combinations, old forcedTc128 rawbits equal, zero rejection, exit0. Existing
  Ada Triad cohort2/2 live tests PASS5.80s; G1 TF32 contract85 normal tests
  PASS299.63s,10hardwaretests remain ignored; CLI11/11 andclippyalltargets PASS.
  Cold/warm persistent-cache actualGPU PASS,sameartifact; all50CUDA13.2arch
  PASS1349.80s,12.8arch48PASS/2FAIL (existing scalarNT16Bspill; unsupported
  sm_110a in extension test), neither waived. FullTF3219route+all4sanitizersPASS;
  compactnewhalfnormalPASS70.38s+all4sanitizersPASS0errors/hazards. NVRTC12.8
  forcedhalf3PASS240.62s; native12.8Rust557lib+clippyPASS, contract85PASS175.71s.

- `67cc4d77`: latest qualified Ada TF32 evidence wins duplicate shape lookup;
  eight previously shadowed measured winners become selectable. Evidence rows
  are retained as history, not competing AUTO precedence.
- `5b81925b`: five separately qualified Ada TF32 NN bias cells select the
  existing wide symbol, replacing prior scalar choices. No scalar source is
  deletion-ready: other shapes, precisions and architectures still need it.

## Next update required

### Newly production-qualified, not yet AUTO

Ada homogeneous half packed-XOR swizzle now has an independent production holder,
forced dispatcher route and BF16/F16 exports on tested CUDA12.8/13.0/13.2. Full
bits/graph/ABI/resource/cold-warm and four-tool sanitizer qualification is recorded
in `internal/perf/ada-half-swizzle-force-20260906/`. Earlier standalone swizzle
rows remain historical timing evidence, not current-module timing admission.
The matching three-toolkit paired21/101 census is now complete at frozen6570ce87,
800records total,zero rejects; evidence in
`internal/perf/ada-half-swizzle-census-20260906/`. Both pipeline andswizzle win
20/20 internally vsTc128 on12.8/13.0,so all20 cells need direct pairing before
selecting a replacement. On13.2 swizzle wins internally forBF16 B/D/E andF16 B/D,
bothbias;pipeline remains faster/safer forA/C bothdtypes andF16 E. AUTO promotion
and post-AUTO proof are still pending. Both Tc128 andpipeline remain essential;
neither is retired or deletion-ready. The four original swizzle experiment files
are preserved unchanged as provenance.

After each production promotion append exact symbol/module/shape/dtype/bias,
source/compiler identity, old/new/vendor measurements, strict-bit evidence,
actual AUTO launch proof, and remaining uses of the beaten kernel. Never mark
delete-ready from an experimental number or a host enum alone.
