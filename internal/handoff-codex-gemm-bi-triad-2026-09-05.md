# Handoff: the deterministic GEMM triad (gemm_bi_triad) — state on 2026-09-05

MANDATORY continuation method, reaffirmed by the user on2026-09-08: read
`internal/agent-operational-rules.md#binding-search-method-research---candidates---short-measurements-2026-09-08`
before taking the next kernel task. Find the bottleneck -> primary-source
research plus parallel agent analysis -> a small concrete candidate shortlist ->
focused bits/resources and paired once7 -> record new wins / stop valid losers.
Use retained-best and cuBLAS Fast comparisons, preserve exact bits, keep one GPU
timing owner. Whole-Triad shortlist first; joint integration and ONE toolkit
qualification batch later. Older per-candidate full-gate instructions below are
historical and do not override this method. Do not spend the next session
reconfirming the same completed kernels instead of finding replacements.

Written for the next engineer (Codex) taking the branch. Everything below was
measured or verified in this tree; where a number is quoted, the evidence
directory that holds it is named. Read this file, then
`internal/gemm-bi-full-wiring-plan-2026-09-03.md` (the decision of record and
the dated progress log, 900 lines), then `internal/agent-operational-rules.md`
(the lane rules for the two CUDA boxes).

## 1. What the work is

Latest2026-09-08: authoritative parallel research and measured follow-through
are in `experiments/ada-triad-authoritative-research-20260908.md` (NVIDIA,
CUTLASS, actual constrained nvMatmulHeuristics query; not predicted speed claims).
Commitde64fbfe retains TWO NEW exact-F32 NT candidates: d768-in actualAUTO
~640→277us, Prism346→301us, exact target/tail/exception/K0 bits PASS.
Fast gaps remain2.36x/3.92x; no Fast victory claimed. Report:
`perf/ada-f32-nt-copyplan-siblings-20260908/report.md`.
Commit4504cef1 records half-TN float2 graph parity (not strict Fast win), NCU
showing zero excessive shared wavefronts, and the first research shortlist.
Discovery still precedes joint integration/full-toolkit gates. Half-NT
M64N128/S3 F16out now beats retainedS3 by4.5%, but Fastgraph p95 fails;
see `perf/ada-triad-half-nt-m64n128-s3-20260908/report.md`. BF16 sibling also
PASSES and gains4.5–4.6% over retainedS3, but Fastgraph p95 still fails:
`perf/ada-triad-half-nt-m64n128-s3-bf16-20260908/report.md`, main027f/helpera97c.
TF32 A-only ldmatrix now PASSES repaired K0/target/tail/exception bits and cuts
actualAUTO11.3–11.4% (~226→200us), but candidate/Fast time remains1.69–1.71;
no Fast victory. Main66ed/helperc853, report
`perf/ada-triad-tf32-nt-a-ldmatrix-20260908/report.md`. Retain this new baseline
for the next shared-stage RNA comparison. Half-NNin M128N64/S2 is a valid
2.5–3.4% loss versus retainedS3; stop, report
`perf/ada-triad-half-nn-m128n64-s2-20260908/report.md`, mainf5d3/helper1220.
TF32 once-per-stage RNA03e6/96ae first failed at138regs/occ1. Refined helper2fbf
with target-only launchbounds256/minBlocks2 restores125regs/local0/occ2 and
passes focused bits, but loses retainedA-ldmatrix35.4–35.5% in paired timing.
ValidSTOP, no unchanged retry; keep A-ldmatrix. See
`perf/ada-triad-tf32-nt-shared-rna-20260908/report.md`.
Half-TNin M64N128/BK64/S2 FOUR-warp regpipe/vec2 also passes bits/resources
but loses retainedvec2 by19.2–19.6%,200regs/occ2; STOP/noBF16/no retry.
Main6c7e/helperef9e, `perf/ada-triad-half-tn-m64n128-regpipe-20260908/report.md`.
Next TF32 candidate retains A-x4 and replaces only B scalar loads with x2;
all-lane mapping is source-proven, implementation pending, no timing claim.
F32TN three-node CopyPlan feasibility is in the research journal; unimplemented.
Fixed inference/SM120 unchanged. Older
progress paragraphs below are historical; do not rerun their completed probes.

Direct TF32 TN N96 d768-in passed bits/resources but is NOT a Fast win:
actualAUTO27% lower; graphFast35–37% slower. Not paired with retained
transpose+N96, hence no new-best claim. Main36126907/helpera153aeb2;
`perf/ada-triad-tf32-tn-direct-n96-20260908/report.md`. Stop unchanged probe.
Next: A-only ldmatrix on retained compact TF32 NT and paired-F32 epilogue on
half TN regpipe, plus one targeted regpipe NCU diagnostic. No full gates.

Half TN Fixed-S3 d768-in is a valid loss for both half types:7–8% slower than
current TC64,24–28% above Fast. Bits/resources PASS; keep compact/regpipe.
Measured typed15cb1a3d/helper9147da28, raw2a3a19ee; root112brackets replayed.
See `perf/ada-triad-half-tn-fixed-s3-20260908/report.md`. Next bounded hypothesis
is paired F32 dW epilogue on regpipe; do not resweep the losing S3 tile.

Half NT direct-epilogue probe is a valid loss:14.5–15.6% slower than retained
S3, also loses Fast; bits/resources PASS. Keep the vector S3 epilogue, do not
repeat. Typed ada3769c/helper087678c8; see
`perf/ada-triad-half-nt-direct-epilogue-20260908/report.md`.
Next focused GPU queue: half TN Fixed-S3 d768-in typed15cb1a3d/helper9147da28;
direct TF32 TN N96 d768-in main36126907/helpera153aeb2. Direct-only tail uses
(129,68,36) so raw-A stride satisfies16B copies; old variants unchanged.

TF32 TN transpose+N96 now screened on all3 large shapes: NEW d768-out40%
and Prism15–17% actualAUTO improvements; focused bits/guards/scratch PASS.
Still NOT Fast wins (graph1.20x/2.00x). Main e5f1a811/helperf969041b,
`perf/ada-triad-tf32-tn-transpose-n96-siblings-20260908/report.md`.
Do not repeat these; next source hypothesis is direct TN N96 staging without
the global transpose. Half TN all8warp Fixed-S3 adaptation is also in progress.

TF32 TN small register pipeline d128-in/out: bits/resources PASS but neither
passes all AUTO p95 strata and both lose Fast strongly in graph. Valid STOP,
no repeat. Main88e238cf/helperbfc9052d, see
`perf/ada-triad-tf32-tn-small-regpipe-20260908/report.md`.

FOUR NEW FAST WINS: half NT Fixed-S3 B-XOR d768-in+Prism ×F16/BF16.
Candidate/Fast p50: in F16 .880–.893, BF16 .896–.927; Prism F16 .500–.509,
BF16 .778–.802. All four eager/graph strata and p95 pass,32 target bits pass.
Measured typeddbbcfaf8/helper2c416464, raw9d6cb191, root224brackets replayed.
See `perf/ada-triad-half-nt-fixed-s3-siblings-20260908/report.md` and the new
selection index `perf/ada-triad-discovery-shortlist-20260908.md`. There are
now nine aligned paired Fast-winning replacement cells in that index (five
earlier NN plus these four NT), not nine public AUTO admissions. Continue
remaining TN/TF32/ExactF32 discovery before joint integration. Fixed/SM120
untouched. No all60/three-toolkit/whole-Triad-complete claim.

Half TN d768-in register pipeline passed both dtype bit/resource screens,
125regs/occ3; still1–2% slower than Fast in graph. It beats forcedTC64 but
does NOT prove an incremental win over the earlier compact candidate.
Stop/no unchanged retry. Typed8096311f/helperf3e66016, raw4a3e0473;
`perf/ada-triad-half-tn-regpipe-20260908/report.md`. Next queued: new NT sibling
coverage, smallTF32 TN register pipeline, NT direct epilogue versus retainedS3.

NEW TF32 TN d768-in raw-A-transpose + RNA-N96:29–30% below actualAUTO,
target/tail/exception/K0/scratch-word bits PASS. Entire2node pipeline timed,
not just GEMM. Fast still loses: graph1.344–1.345x. Preserve AUTO improvement,
not a Fast win. Measured mainc009c10f/helperf969041b, raw3cd94f88; root56brackets
replayed. See `perf/ada-triad-tf32-tn-transpose-n96-20260908/report.md`.
Frozen measured source is separate from new small-TN register-pipeline edits.

NEW half NT d768-out Fixed-S3 B-XOR improves forcedTC64 by26% for F16/BF16,
41–42us versus56–57us, all target bits/guards pass. Not a Fast win: F16 eager
near parity but graph2% slower; BF16 remains3–6% slower. Keep both candidates.
Measured typed c1fd1b86/helper2c416464, raw a5403469; root replay112brackets.
See `perf/ada-triad-half-nt-fixed-s3-bxor-20260908/report.md`.
Next: same NT source on unmeasured d768-in/Prism; TF32 TN transpose-to-N96;
half TN compact register pipeline; TF32 TN small register pipeline. These
remain test-only discovery. Do not repeat d768-out or run full gates now.

New TF32 NN Prism N96 shortlist entry:1.5–1.7% lower actualAUTO time with
strict target bits and all paired p95<.99. Near Fast parity, NOT Fast win;
graph remains1.3–1.4% slower. N96 d768-in losesAUTO5.7–5.9% and is stopped.
Final main460a2c1f/helper086c0520; source, raw samples and the tail/K0 harness
repairs are recorded in `perf/ada-triad-nn-addhalf-n96-screen-20260908/README.md`.
Continue new halfNT-S3 / halfTN-register-pipeline / TF32TN-transpose-N96,
not another old N96 sweep. Whole-Triad discovery boundary still applies.

Latest follow-up: loaded Fixed Pipeline/Swizzle on aligned F16 NN d768-in
both pass target bits but lose retained S3 (30–31% /8% slower) and Fast.
Keep S3; see `perf/ada-triad-half-nn-loaded2-20260908/README.md`, measured
typed7859864f/helperff7b4ce1. These are existing-route screens, not new wins.
Next source work is direct half NT S3 reuse and TF32 TN transpose-to-N96;
root's separate half TN compact register-pipeline helper is also unmeasured.

Latest discovery checkpoint: half TN M64N128/BK64/S2 compact passes all six
bit/resource cells but loses current and Fast everywhere; do not resweep it.
Measured typed ee3d984a/helper a307af83; raw and paired ranges in
`perf/ada-triad-half-tn-m64n128-20260908/README.md`. Next: loaded NN Pipeline/
Swizzle on F16 d768-in, then direct NT adaptation of the Fixed S3 pipeline.
TF32 N96 tail diagnostic found candidate/current identical but public AUTO
selects exact ScalarFmaV1 on the unqualified (129,36,100) tail. Repair that
cross-family test expectation; keep strict candidate/current/AUTO bits on
the timed target cells. Then continue new TN transpose-to-RNA-N96 discovery.
No full gates/integration until the whole-Triad replacement shortlist.

Latest checkpoint after 7f5d2e29: NEW aligned S3 BF16 NN d768-in beats Fast
by21–23% (62–63us vs79–81us), exact current bits PASS. This supersedes N64
near-parity and brings aligned S3 NN finalists to FIVE cells; forcedTC128
reference, not dispatcher admission. Measured typed e77e1961 is archived
independently of the next edits. See `perf/ada-triad-half-nn-s3-bf16in-20260908/README.md`.
Half NT compact TWO cells and half TN compact SIX cells are valid Fast losses;
TN d768-in is near parity but graph remains1–2% slower. TF32 TN Prism single
barrier improves actualAUTO1.5–3.4% but remains1.86–2.32x Fast; STOP/no retry.
Raw evidence and source hashes are in the corresponding short reports.
Next measured checkpoint (0ccb source): aligned S3 F16 NN d768-in reduces
forcedTC128 time32%, but remains2–4% aboveFast. Retain the current benefit,
not a Fast-win count. Loaded NT128 d768-out xhalf is6% slower than TC64;
STOP on this cell only, preserve other-device routes. See
`perf/ada-triad-half-nt128-f16in-20260908/README.md`.
Next discovery: compact halfTN M64N128/BK64/S2 (better A reuse, same K16 bits)
and existing TF32 NN N96 on unmeasured large shapes. WHOLE-Triad shortlist
before joint integration/full gates; do not trigger gates after a small batch.
Fixed inference and SM120 routes remain untouched. Below is historical context.

Latest checkpoint2026-09-08 after2a42d272: four NEW aligned half NN S3
finalists, d768-out/Prism ×F16/BF16, are20–23% below cuBLAS Fast in independently
paired eager+graph once7. Allbits/guards PASS; forcedTC128 reference, NOT public
AUTO admission. See `perf/ada-triad-half-nn-s3-aligned-20260908/README.md`.
Measured typed667510c2 is retained independently of next S4/N64 working edits.
Half TN Rect128x64 all6cells and TF32 TN A-prepack bothcells are valid losers;
new reports record them, no reruns. Current GPU queue: compact half NT/TN
BK64/S2 reuse (shared32KB for3CTA/SM). ExactF32 d128 fold-pipeline bothcells
and halfTN BK32/S4 all6cells completed bits/resources but lost; keep earlier
finalists. N64 BF16 NN d768-in is now nearFastparity (.981–.992,p95up.9946),
no strict<.99 admission. N64 d768-out beatsFast but is weaker than retainedS3,
NOT another champion. See `perf/ada-triad-half-nn-n64-20260908/README.md`
and corresponding loser reports.
Discovery first remains in force; batch finalists before one integration gate.
Fixed inference and SM120 remain unchanged. Older checkpoints below are history.

Previous checkpoint after c09af9eb: NEW exact F32 TN d128-out direct
M8N16 candidate is39–42% below actualAUTO, still2.15–2.21x Fast; see
`perf/ada-triad-f32-tn-d128-out-direct-20260908/README.md`. No rerun: wrapper97
was stale record counts, exact GPU test passed. Native source tests13PASS.
Half NN7's four apparent Fast wins are provisional: guard offset16 slowed
cuBLAS; aligned F16 d768-out diagnostic still shows18–20% advantage. See
`perf/ada-triad-half-nn-fast-diag-20260908/README.md`; use256-aligned timing
buffers henceforth. Half NT BK32/S3 passes bits/resources but loses Fast;
`perf/ada-triad-half-nt-s3-20260908/README.md`. Do not rerun unchanged losers.
Current short pipeline: TF32 TN A-prepack (vendor graph validation repaired,
first valid timing pending), existing half TN Rect128x64 on three large shapes,
and test-only half TN TC64/BK32/S4 helper. Fixed/SM120 remain unchanged.
Older "no result yet" statements below are historical, not current status.

Latest 2026-09-08, commit df8948ff: short discovery FIRST. This overrides the
older next-step instructions below: do not restart once21/all-CUDA/full gates
or repeat the old three NN wins. Batch new performance finalists on Ada13.2,
then integrate and confirm together. Fixed inference and SM120 are unchanged.

Later checkpoint ce247d0d: half TN eight-cell batch completed, all bits PASS,
all candidate/Fast pairs lose. Best M32N32 p50/Fast 1.42–1.85 across dtypes,
d128-in/out and eager/graph. Its ~.69/current ratio is UNPAIRED diagnostic only,
not a robust win. Final main1721ab31/helperb1d73379; see
`perf/ada-triad-half-tn-d128-microtiles-20260908/README.md`. Current work expands
NN S3 to untested half shapes with Fast, plus A-only raw-RNA prepack for TF32
TN Prism/d768-in (whole pack+GEMM pipeline timed). Neither has a result yet.
F32 M8N16/64 and M16N16/128 also passed bits but had p95 losses to retained
M16N16/64. Do not resweep them or present their small median gains as champions.

NEW F32 TN d128-in direct fold, `(1024,128,512)`: M16N16 removes the 16 MiB
partial buffer and second launch, preserves existing FP32-chunk/FP64-fold bits.
Candidate/public AUTO p50 .3813–.4644 (54–62% lower observed time), but
candidate/explicit cuBLAS Fast p50 2.5625–3.0086. One exact once7 GPU test PASS;
raw and limitations: `perf/ada-triad-f32-tn-d128-direct-20260908/README.md`.
These one-GEMM event timings include launch overhead, not kernel-only duration.
Targeted NCU precedes another tile hypothesis. M8N32 is the weaker alternative.

NEW denominator for TF32 NN N96 d768-out: explicit cuBLAS Fast gives p50
1.0000–1.00794, worst p95 1.0240: near parity, NOT a Fast win. Candidate/current/
true AUTO match bits. Main ce9839a6/helper d9098e55, short13.2 only; see the
new top section of `perf/ada-triad-nn-addhalf-n96-screen-20260908/README.md`.
The old 18% gain versus AUTO below is not a newly discovered improvement.

Half TN next batch: test-only M32N32/128-thread and M16N32/64-thread BK32/S3
microtiles, d128-in/out × F16/BF16 (eight candidate cells), preserving current
TC64 K16 MMA order and F32 dW output. Owner Carver; sole GPU executor Avicenna.
Current freeze main e27eb891/helper b1d73379 includes Fast finite/self-bit
checks and separate 20-GEMM timing oracles. No result or production admission
claimed until its short GPU run finishes. Preserve all unrelated WIP.

Latest2026-09-08, NEW N96 result (not a recap of CopyPlan/half wins): test-only
TF32 Triad NN M128N96/BK32/S3 AddHalf candidate advances on Ada13.2 d768-out
`(M,K,N)=(2048,1536,768)`. Candidate/truepublicAUTO p50 .814103-.819355,
worstp95 .831169:18.1-18.6%less median time, about65us versus79-80us.
Target candidate/currentwide/publicAUTO allbit-identical, mismatch0; tail,
exception,K0-null eager2/graph2pass.124regs/local0/static0/dynamic86016/occ1.
Main700f8ff8/helper1898941b, raw8e44c91e, binary89c5514a. NOT Fastwin orAUTOyet.
Earlier apparent mismatch was a harness bug: typed-forward invoked F32 matvec,
while separatelyqualifiedidentitydescribed trueTriadF32forward. Do NOTinfer
RNAconversionisneeded; preserving existingAddHalf matched the correct AUTO.
Evidence:`internal/perf/ada-triad-nn-addhalf-n96-screen-20260908/README.md`.
User explicitly rejected oldwins presentedasnew; report only genuinelynew
measurements asnew. Next N96 fullmantissa/once21/Fast/3CUDA confirmation.

NN3+NT1 scalar CopyPlan dispatch integration source is independentlyreviewed,
not yet GPUconfirmed. Fiveproductionfiles useFixedbackend22/private1, NNtag37,
NTtag38; exact3CUDAcohorts, strict3NN+1NTcells, fallbackelsewhere. NTinnergrid768.
Assembled dispatchb4f5d584, noFixedCUDA/SM120/global45change. Avicenna soleGPU.
Original NN191a discovery holder remainedlive during other timingarms,
violating its exclusive-use contract: preserve rawbut treatasprovisional.
Rootrepaired NNwrapper527d1322 dropsholderafterisolatedchecks, then truepublic
entry withownedbuffers anddriverABIgraphchecks; native9/reviewPASS.
Newintegratedonce21actualAUTO/generic/Fast3CUDA isnext; NTowner addsseparateproof.

Latest2026-09-08T00:49Z: actual loaded Fixed CopyPlan now beats actual public
AUTO for all3 F32 NN shapes on CUDA12.8/13.0/13.2. It does NOT beat cuBLAS
Fast (`CUBLAS_COMPUTE_32F_FAST_TF32`): median ratios remain about1.64–2.45x.
Strict alpha+1/beta+0/noBias only, full-mantissa target/tail/K0-null/guard and
repeat/eager/graph bits pass. Source191a27f8/Cargo2e1b5f65, no Fixed CUDA changes.
Real AUTO is independently qualified, true Fixed holder/artifact separately
bound. Root replays644brackets/2576observations including all3CUDA NN/NT and below.
NTd768out transpose16+liveFixedCopyPlan wholepipeline13.2 now advances
candidate/actualAUTO p50 .5365–.5383 (46.2–46.3%less time), worstp95 .5466;
source3167072a, eightbitrecords/resources pass. Lower12.8/13.0 nowalsoadvance:
p50 .4509–.4523/.4508–.4542, worstp95 .4606/.4632, bits/resources pass.
FinalNTsource005da52e/native5PASS; initial80ea cfg-helper importfail and d755
reference-resource16-versus0 failurepreserved. Final explicitlyexpectslocal16
onlyfor test-composed generic reference onlowerCUDA, candidate/transpose0still.
No arithmetic/timingchange, no retryofvalidloss. All3NT arediscovery, notAUTOyet.
HalfFixedS3 d768in13.2: F16 advances vsforcedTC128 p50 .8944–.9063,
worstp95 .9275; BF16 STOP p95 1.0374 despite fastermedians. Both bits pass;
no half Fast claim or BF16 retry. Frozen431ef535/helper759a1bbd.
Full current report and raw locations:
`internal/perf/ada-triad-live-reuse-20260908/README.md`.
Carver now integrates NN3 into true AUTO with distinct Fixed backend/tag/private
revision; global45/numericABI/schedule and Fixed/SM120 untouched. Beauvoir owns
test-only TF32 NN N96 preserving current AddHalfUlp conversion (not silentlyRNA),
firstd768out only. Avicenna soleGPUexecutor. Root owns evidence/replay and NT
toolkit-helper repair. No newbranch, no deletion, no completedTriad claim.

Latest2026-09-08T00:08Z: F32 CopyPlan reuse advances all3NN cells on Ada13.2:
candidate/generic exact p50 ranges .7373-.7405 / .5598-.5670 / .6709-.6811.
Strict alpha bit+1/beta bit+0/no-bias only; tail and K0/null A/B, repeat/graph,
guards/immutability allpass. Nontrivialalpha/beta earlierfailed beforetiming;
do not claim universalepilogue compatibility. Source6729c227/binary7bd0f02e.
StandardM64 NN wins onlyd768out but is superseded in discovery by CopyPlan;
do not integrate the weaker candidate merelybecause it also beatsgeneric.
NTd768out transpose16+M64 wholepipeline advances vs actualpublicAUTO:
p50 .8625-.8715 (~12.8-13.7%less time), eightbitrecords/resourcesPASS.
Initialtargetgate andpreparedcache errors fixedwithoutbypassingAUTO; exact
fixture nowwarms beforecapture. Sourcea1f20ca1/binary1c2ce6c3, frozenrawkept.
Half TC64 STOPs F16/BF16 (~45-50%slowerthanforcedTC128), bits/resourcespass;
no retry. Nextshortscreens: NTinnerFixedCopyPlan and existingFixedhalfS3reuse.
All newresultsremain discovery, not Fast comparisons orAUTOadmissions.
Root independentlyreplays196brackets/784observations forscalarNN/NT/CopyPlan.
Summary/evidence/cleanup boundaries:
`internal/perf/ada-triad-broad-screen-20260908/README.md`.
PreserveFixed/SM120/globalepoch45. PreferexistingFixedholder/artifactreuse
overduplicatingCUDA source; newTriadphysicalidentity musttruthfullynameFixed.
TF32TN denseintegration willchangeexistingAdaoptionalmoduleidentity; batch
requalNT3+TN1alltoolkits together, nofifthmodule or staleNTcohortliteral.

Latest2026-09-07T23:45Z: broad Ada Triad discovery, not another Fixed retune.
TF32 TN dense-stage batch is committed30df4596. All3 resource/bits gates pass;
only d768-in advances (candidate/actualAUTO p50 .9501-.9618; worst p95 .9771).
d768-out graph is slower and Prism is slower; both STOP, no retries. Candidate
M128N64/S3 is still test-only; no CUDA12.8/13.0 admission or Fast win inferred.
Evidence: `internal/perf/ada-triad-tn-dense-batch-20260908/report.md`.

Existing scalar M64 NN CUDA13.2 discovery: d768-in STOP (~18% slower than
generic exact); d768-out advances (p50 .9390-.9536, p95 .9608-.9715).
Initial Prism test incorrectly chose the specialized backward-NT inner-GEMM
symbol, whose column stores require a full tile, for forward N1928. The guard
caught it before timing. This is a harness-selection error, not a production
dispatcher regression. Repair6a7354d8 runs ONLY Prism with the standard safe
NN symbol: bits/resource pass, speed STOP (~5.4-5.7% slower). Preserve the failed
batch rawc6ebbdb0 and repaired rawc3064318; do not rerun valid d768 observations.
Root independently replayed all84 brackets/336 observations. Comparator is
honestly `generic_exact`, not newly qualified public AUTO or cuBLAS Fast.
Evidence: `internal/perf/ada-scalar-nn-m64-screen-20260908/`.

Next source freezes: NT d768-out064b9c6e (existing full transpose16+M64 pipeline
versus actual AUTO, independent review ACCEPT), and NN6473d602 (existing Fixed
exact-F32 CopyPlan source reused unchanged, no-bias3cell screen; static32768,
dynamic0,128threads). Both have GPU GO after independent review, sole executor
Avicenna. A test-only resource i32/usize compile error in3db54a83 was repaired
in6473d602 before GPU use. No arithmetic contract or production route changed.

Half-NN F16 d768-in NCU now captured: TC128 has162regs, one resident CTA,
low ready-warp count, and zero excessive shared wavefronts. SASS already
interleaves fragment loads with MMA; do NOT spend a new candidate on merely
explicit fragment ping-pong or claim XOR is a demonstrated bank fix. Next
bounded half screen is the already-built TC64 versus TC128, both BF16/F16,
with unchanged BK64 association. Carver owns additive typed-parity harness;
root owns NN CopyPlan harness, Beauvoir read-only TF32 route inventory.
Evidence: `internal/perf/ada-triad-half-nn-ncu-20260908/evidence/attempt1/`.
Keep all experiments test-only until qualified; batch finalists for all3 CUDA
toolkits and Fast comparison. Preserve RTX5090 routes and unrelated SM120 WIP.

Latest2026-09-07T22:31Z: the distinct TN compact-four-compute-warp S2
ablation also STOPs. Focused bits, finite K0, guards and repeats pass;
resource156regs/0local/0static/49152dynamic/256threads/occupancy1. Paired
once7 candidate/AUTO p50: eager1.0102/1.0220, graph1.0424/1.0434;
p95 ranges1.0257–1.0540. Graph AUTO249.824us versus260.400/260.896us.
This is not a Fast comparison. No production or AUTO change, no retry.
Root verifies379 source rows and28 brackets/112 observations from raw
624a4139cd9f134c42c0e9ad7ba272cedbc7a9139c971698d0604f3ce215d51c.
One failed transfer-path build was repaired before GPU execution; its log is
preserved. Runtime75.87s, source/main8b921bbb, composed22d9e88d, binary91bfbe97.
Commit-only rustfmt wraps one host assertion: finalmainb71a9253, token-identical
to GPU-built8b; native31/31 plus rustfmt pass. No extra GPU run for formatting.
Independent source/runtime/runner review accepts the scoped implementation.
Evidence: `internal/perf/ada-triad-tn-compact4-screen-20260908/`.
Compact4 and compact8 remain test-only negative results, not a global
rejection on other GPUs. RTX5090 winners, portable SM80 and Fixed unchanged.
The next TN search must change mechanism; do not resweep these two variants.

Latest2026-09-07T22:20Z: TN Prism compact-eight-warp S2 discovery is complete
and committed as070a4f54: resource104regs/0local/49152dynamic/occ2 and focused
RNA target/tail/alpha/K0 eager+graph bits pass, but candidate/AUTO p50 is
1.045–1.064 and p95 is1.052–1.067. STOP, no production promotion and no Fast
comparison. Evidence: `internal/perf/ada-triad-tn-compact8-screen-20260907/`.
The initial K0 harness rejection and repaired once7 raw are both retained.
Root native29/29 passes. Existing RTX5090 kernels/routes remain untouched.
Next in progress: one compact-four-compute-warp S2 ablation, retaining compact
staging while restoring original ownership; no new result yet. Contract and
ownership: `internal/perf/ada-triad-tn-compact4-screen-20260908/brief.md`.
No Fixed retuning, matrix resweep, deletion or new branch in this work.

Latest2026-09-07T21:30Z: compact8/S2 NT integration now closes on ALL THREE
installed toolkits12.8/13.0/13.2. Each has its own measured identity and all
three exact cells in real AUTO, seven forced RNA bit cases, resource/K0 and
paired once21 qualification. Final-source batch on each:663 library passes,
47 ignored, three exact host passes, actualAUTO three cells/no skips with
repeat/eager/graph bit equality. Final common source map3ed80f44, all377rows
accounted; raw commands and hashes in the README below. Lower-toolkit oldAUTO
was exactF32; finalist/current~.282/.319/.425, versus13.2~.824/.88/.837.
These are wins over oldAUTO, NOT Fast: Fast is still~1.6–2.1x ahead here.
Measured register counts125/125/121 give final cap125; zero local/static,
49152dynamic,256threads,occupancy2 on each. Physical1/global45 unchanged.
One stale cap assertion was repaired after the final host batch caught it;
the failed raw run is retained. No unrelated WIP or losing experiments deleted.
Next: one test-only TN Prism compact-packed M128N64/BK32/S2 eight-compute-warp
screen. Preserve RNA/K order/TN old-C FMA. No new TN speedup is established yet.
Evidence: `internal/perf/ada-triad-finalist-integration-20260907/README.md`.

Latest2026-09-07T20:50Z: CUDA13.2 compact8/S2 passed the integrated paired
once21 qualification for all three target NT cells. GraphAB p50 current→new:
d768-in274.324→226.077us, d768-out119.954→105.791us,
Prism192.385→160.867us. Allfour eager/graph/AB/BA p50+p95 current ratios win.
Fast remains ahead: independently paired p50 finalist/Fast about1.93x/1.59x/2.10x.
Root replay verifies24cohorts/504pairs and all measured binding fields. Actual
resource/K0/live-revision gate passes121regs/0local/0static/49152dynamic/256/occ2.
Production dispatch now contains the observed13.2 identity and three exact
NT cells; post-admission actual-AUTO eager/graph/bits and three focused host
tests now pass (GPU9.29s, source map4ea35e39,377rows accounted). CUDA12.8/13.0 have no new
AUTO admission yet; run those next and add only their observed identities.
No new5090 run or new candidate search in this integration phase. Keep global45,
physical finalist1 and unchanged portable cohorts. Evidence/commands/remaining
scope: `internal/perf/ada-triad-finalist-integration-20260907/README.md`.

Earlier2026-09-07T20:17Z: the integrated compact8/S2 module has now actually
loaded and run on Ada/CUDA13.2. The scoped forced-route smoke passed all7
full-mantissa target/tail/prefix cases with two eager and two graph launches
per arm, unchanged inputs/guards and exact portable-RNA bit equality. It ran
in58.10s; original three cache artifacts were preserved and exactly one new
finalist artifact appeared. Source818b64d6 and host evidencec1faa724 are already
committed. AUTO remains unadmitted pending paired current/Fast measurements
and the remaining toolkit/resource checks. This is not a new speedup claim.
See `internal/perf/ada-triad-finalist-integration-20260907/README.md`.

Source-integration checkpoint2026-09-07: compact8/S2 production source is
isolated and committed as04c71ebe, independently reviewed with no findings.
Exact composed bytes match the measured candidate after only helper
comments/names and target symbol normalization: SHA256
`d20cb39f390ef45181be94e4bd7baeb51548c0e6efac64bad72aae71a889e3ce`,203981bytes.
Root native28 plus CUDA-cfg-only Rust5 tests and eight frozen reference hashes
pass; these are not NVRTC/runtime tests. Evidence:
`internal/perf/ada-triad-sm89-finalist-source-20260907/README.md`.
The optional module/forced route is now wired under
`internal/gemm-bi-triad-ada-finalist-integration-2026-09-07.md`. CUDA13.2 release
build/list and eight focused host tests pass; no NVRTC/runtime load in that
checkpoint. Public AUTO has not admitted this candidate yet. Keep existing
epoch45 and module bytes. The frozen all-toolkit runtime batch is next; see
`internal/perf/ada-triad-finalist-integration-20260907/README.md` for exact scope.
Use forced portable RNA for bit comparisons, actual AUTO and cuBLAS Fast as
separate timing references on each toolkit. Existing NT requires beta0/no
bias; use the internal zero-reduction qualifier for NT N0, not a public-facade
contract expansion. No new GPU performance result is claimed by this checkpoint.

Latest four-arm sibling batch (2026-09-07): complete, three ADVANCE and
one valid STOP. Compact32/eight-compute-warp/S2 is the preferred finalist
for all three measured heavy NT TF32 shapes, with unchanged composed CUDA
`fd5bd92f7cc56276a1b95b24b59319142b247f7ddedb14d6b06cac91843205b4`.
New siblings use the same source/symbol as the earlier d768-in finalist:

| NT case / candidate | AUTO eager median us | Candidate eager median us | paired p50 range, eager + graph | Decision |
|---|---:|---:|---:|---|
| d768-out / dense S3 | 120.912-121.149 | 110.283-110.490 | .911986-.919154 | ADVANCE, slower than compact in discovery |
| Prism / dense S3 | 191.941-192.649 | 194.560-195.289 | 1.013413-1.015495 | STOP, no retry |
| d768-out / compact S2 | 120.994-121.082 | 105.810-105.940 | .874592-.882708 | preferred ADVANCE |
| Prism / compact S2 | 191.787-192.413 | 160.816-160.834 | .835888-.840650 | preferred ADVANCE |

All four actual GPU tests pass bounded finite full-mantissa target/tail
eager/graph bit equality, repeated execution, input immutability and guards.
Compact remains121regs/0local/49152dynamic/occupancy2; dense remains
154regs/0local/82944dynamic/occupancy1. Root and independent reviewer replay
374 source hashes,112 raw paired brackets and16 strata. Each arm ran once7
exactly once; one shared fresh build/list, reused private cache, no cold-cache
claim. Immediate RELEASE is retained as nonquiet; separate five-second DRAIN
is quiet for each arm. Source rs4643558f..., new Prism header790bb241...,
binarya9e7b2c3..., manifestb4a7d40d.... Root native23 checks and actual-header
host stage replay34,062,336 dense-copy tuples,12 negative guards and two
compiled tail mutations PASS. Evidence and durable replayers:
`internal/perf/ada-triad-nt-sibling-two-mechanism-20260907/README.md`.

These are improvements over actual AUTO, NOT cuBLAS Fast wins. No21/101,
full qualification, new5090 run or production promotion in this checkpoint.
Cross-arm rank is provisional discovery, not direct candidate-vs-candidate
admission. Freeze compact S2 for the three NT cells, then batch qualify and
wire only admitted cells/toolkits. Keep experiments separate; do not delete
dense kernels globally on one cell's loss. Production integration must not
invalidate existing portable SM89/SM120 or Fixed routes: changing shared
SM80 source changes literal cohort identities. The current bounded design
investigation is an optional separate Ada finalist module, a distinct physical
route identity and independently versioned selector, leaving existing module
bytes and epoch45 intact. The forced-only integration above now implements it;
runtime qualification and AUTO admission are still pending.
Older entries below describe their own checkpoint's then-next action.

Latest compact32/eight-compute-warp/S2 NT screen (2026-09-07): ADVANCE
finalist, about17.4-17.7% lower runtime versus actual AUTO on
(2048,768,3072). Eager226.257-226.262us vs274.944-275.014us;
graph225.978-226.022us vs273.687-273.752us. Allfourpairedonce7 strata win:
eagerABBA p50/p95 .822563921/.823097796,
eagerBAAB .822837094/.823171353,
graphABBA .825599198/.825873428,
graphBAAB .825800614/.826036828. Runtime confirms the intended threshold:
121regs/0local/0static/49152dynamic/256threads/occupancy2(required2).
Bounded finite target/tail eager/graph bitsPASS, exactly1GPUtest48.67s;
root373source/28rawbracket replay and native20checks PASS. Actual generated
storage/slot host proof covers all9NT specializations and alloriginalNN/TN
extent assertions, with2compiled behavioral mutations caught.
Source025a8f44..., binary6e8d5647..., map559ebcd8...,
composedCUDAfd5bd92f..., rawca50cb02.... Evidence:
`internal/perf/ada-triad-nt-compact32-eight-warp-s2-20260907/README.md`.
Retain=true/promotion=false, no Fast/21/101/fullqualification yet. Faster
than the earlier dense-copy screen provisionally; not a direct paired
candidate-vs-candidate admission. Next one-build shortbatch: d768_out and
prism, each comparing both retained mechanisms with its actual AUTO.
Compact sibling labels reuse this exactsource/symbol and occupancy2 gate;
dense siblings use exactguards and prism needs per-stage generic tails.
Only frozen selected winners proceed to batch qualification/dispatcher.

Latest NT dense full-tile copy screen (2026-09-07): ADVANCE finalist, not
production promotion. Same four-compute-warp/padded36/S3 layout, RNA/MMA
order and epilogue, but exact full target uses six unconditional16B copies
per thread; all other cases keep generic staging. Target(2048,768,3072),
strides(3072,3072,768), existing wide-alignment eligibility required.
All four paired once7 strata beat actual AUTO: eager ABBA p50/p95
0.902877339/0.904695784, eager BAAB0.903320353/0.904387367,
graph ABBA0.904951208/0.906080434, graph BAAB0.902442164/0.903388426.
Median times: eager248.22-248.27us vs274.84-274.87us; graph246.96-247.65us
vs273.66-273.74us. This is about9.5-9.8% lower runtime versus our AUTO,
NOT a new cuBLAS Fast comparison/win. Resource154regs/0local/0static/
82944dynamic/1CTA. Bounded finite target+forced-generic-tail eager/graph
bits and input/guard checks PASS; exactly one GPU test48.49s. Root replays
373source hashes and28rawbrackets; native17tests plus actual-header
28,311,552copy tuples/seven negative guards/three behavioral mutations PASS.
Source rs3d433d7b..., header5aa5d735..., binary4f94cba4...,
composed CUDAb256264c..., raw e837429a.... Evidence:
`internal/perf/ada-triad-nt-padded36-dense-copy-20260907/README.md`.
Retain=true, promotion=false: keep this finalist for frozen-batch qualification.
Next bounded design crosses a different resource threshold: compact32/S2
plus eight compute warps, exact NT128x64/S2 specialization,49152dynamic;
require actual occupancy>=2 and zero local memory before bits/timing.
The separate S3 compact/eight-warp losses do not establish this joint regime.
Explicit register pingpong was skipped after existing rolling LDS/HMMA
interleaving was observed. Many FSETP are RNA Inf checks, not copy bounds;
do not attribute the entire profile FSETP count to staging predicates.
Older checkpoints and their then-next steps below are historical.

Latest NT eight-compute-warp screen (2026-09-07): finite target/forced-tail
eager/graph bitsPASS,100regs versus profiled154,0local/82944shared/1CTA,
but all4once7 strata lose: p501.080783..1.083239,p951.081299..1.083544.
STOP_NO_RETRY, no21/101/Fast/production promotion. Oneactualtest49.52s,
rootreplay372sources/28brackets and native14checksPASS. Source989acbcf...,
binary11b99112..., raw89c6bd60.... Evidence:
`internal/perf/ada-triad-nt-padded36-eight-warp-20260907/README.md`.
Four recent NT mechanisms now screened without a retained winner (compact,
copyplan,ldmatrix,eightwarp); production AUTO remains unchanged. Next bounded
research is existing full-tile staging adaptation and warp-fragment pipelining,
not another retry of these losses. Old checkpoints below are historical.

Latest padded36 two-arm NT screen (2026-09-07): both candidates pass bounded
finite target/tail eager/graph bits but are STOP_NO_RETRY, no promotion.
Copy-plan185regs loses all4strata, p501.093789..1.095568 and
p951.094197..1.096079. Ldmatrix160regs is eager parity and graphslower:
p500.999331..1.001887, p950.999987..1.002079. Both0local/82944shared/1CTA.
Root replays2actualtests/56rawbrackets/8strata and both372-source maps.
The pre-repair Rust build was never GPU-run: stale signature-assertion names
were caught and repaired before tests; original source/receipt are preserved.
Measured commonrs37dec61a..., binary34a67271..., CUDA13.2only. Evidence and
native actual-header mutation proofs:
`internal/perf/ada-triad-nt-padded36-two-arm-20260907/README.md`.
Next independent candidate is eight compute warps instead of four on the
original padded36 target; not measured by this checkpoint. No retries of
these losers, no production/dispatcher changes or Fast closure claimed.

Latest Triad discovery (2026-09-07): NT compact-XOR preserves the bounded
finite target/tail/eager/graph bits and has155regs/0local/73728shared/1CTA,
but loses to actual AUTO in all4once7 strata: p501.034743..1.038873,
p951.035072..1.039089. STOP, no retry/21/101/production promotion. Exactly
one real GPU test passes44.67s; the earlier wrong-filter zero-test attempt
is preserved and explicitly invalid. Raw/replay/source binding:
`internal/perf/ada-triad-nt-compact-xor-20260907/README.md`.
Next bounded batch designs use the original padded36 layout: precomputed
copy plans and ldmatrix fragment loads, separate mechanisms, same RNA/MMA
association. Reuse the discovery harness. Do not restart the six old direct
tile losses or treat the rejected compact layout as a retained finalist.
The current60-cell audit found no measured compatible Ada champion omitted
from AUTO. Stream-K has connected winners on other training dimensions under
its distinct policy; SM120 TMA winners are not Ada-compatible. RTX5090's
preserved current21 fingerprint independently replays17/60 both-quantile
Fast wins (Exact1,TF324,BF166,F166), versus Ada3/60; not same-binary or new
RTX5090 measurements. Older profile/state checkpoints below remain valid.

Fresh Ada Triad state and two-cell profiles are complete (2026-09-07), using
the unchanged CUDA13.2 binary built at59f1a949. Once21 raw snapshot independently
replays120 custom eager/graph +90 vendor Fast/Pedantic records. Against Fast
eager/eager, both p50+p95 win in3/60 cells: BF16 NN d128_in, F16 NN d128_in,
F16 NT prism. ExactF32 and TF32 each0/15; BF161/15; F162/15. The other57 lose.
TF32 NT d768_in is275.016 vs117.136us; TN prism242.639 vs99.733us. These are
independent quantile ratios on zero-initialized synthetic operands, not paired
admission, full-mantissa bit proof, end-to-end training or fresh5090 results.
Full physical symbols, all60 rows, raw logs and replay:
`internal/perf/ada-triad-state-20260907/report.md`.

NCU attributes all NT excess shared wavefronts and all TN excess to LDGSTS
copy sites; the ordinary MMA fragment LDS sites have actual=ideal. NT excess
is5,750,784 (+40.625% relative to ideal), TN1,729,494 (+16.739%). Both are
one CTA/SM by registers/shared, about37% active-cycle HMMA; TN has only93CTAs
for142SM. No claim that this isolates all Fast loss or proves bad global
coalescing. Profile/report/binding/reuse audit:
`internal/perf/ada-triad-two-cell-ncu-20260907/`. The field measured_source_sha
fb878... is a legacy FOUR-Fixed-file digest, NOT the complete Triad binding;
full368-file map SHA is163519fc90a7e35f0cde5587b66afe9a0ab0bda98a727161e648fe0d95dc74b5.

Next bounded experiment: test-only direct TF32 NT M128N64/BK32/S3 compact XOR
shared-copy destination (36->32 floats/row, physical_k=k^((row&7)<<2)), reusing
Fixed staging lessons but preserving per-thread global copy ownership, RNA,
ascending MMA and NT epilogue. It is NOT implemented/admitted by this checkpoint.
Existing NT tournament is split-K-specific and its cuBLAS companion SM120-sealed;
do not weaken those guards. Use one focused test, small full-mantissa actual-AUTO
bit/repeat/graph/resource checks then once7; no unchanged full-state rerun per
prototype. Keep the old separate split8 discovery WIP untouched. No kernel was
deleted and no new production route is claimed by these measurement reports.

Assembled Fixed AUTO45 source and evidence are committed as `b221b71e`
(`promote qualified Ada Fixed AUTO45 routes`), human identity/no trailers.
The 131-entry phase2 manifest verifies against that commit; immutable raw logs
retain their harmless terminal blank lines. The subsequent holder-comment
cleanup changes Rust documentation only and is not a new GPU qualification.
Prepared Triad discovery WIP remains separate and uncommitted. The subsequently
completed two-cell current-AUTO profile is summarized above; its acquisition
plan remains `internal/perf/ada-triad-two-cell-ncu-20260907/plan.md`.

Fixed AUTO45 assembly (2026-09-07) is now qualified: exact five rows are
connected to public AUTO and all five once101 runs pass against the replaced
route. TF32 E0 N96 on12.8/13.0/13.2: worst own p50
0.773862/0.773759/0.852935, Fast p951.108774/1.108750/1.114884.
F16 D0/E0 on13.2: own p500.912581/0.970985, Fast p951.068794/0.984772.
Thus E wins Fast, N96/D still lose Fast, and inference as a whole remains
open. No exactF32/BF16 or other-cell fresh result is implied. All three
matching builds, corrected lib652/0/46, actualAUTO N96 corpus224/toolkit,
half actualAUTO/holders and retained13.2 Triad cohort tests pass. Root and
independent reviewer replay20 records/2020 raw triplets, sources/binaries,
physical graphs and unchanged compiled module cache hashes. Numerical ABI5,
schedule8 and CUDA bodies unchanged from phase1; tuning45 requires recapture.
See internal/perf/fixed-finalist-integration-20260907/phase2-README.md.
Lower-toolkit half D/E keep old AUTO; no production kernels are deleted.
This completes the assembled finalist batch: proceed to the bounded Triad
physical profile, not another unchanged Fixed sweep. The older checkpoints
below are historical, not pending work or new full-survey evidence.

Fixed finalist integration phase1 (2026-09-07) is now qualified on all three
CUDA versions. Three independent forced holders (TF32 RNA N96, F16 D N64/S3,
F16 E N64/S2) pass resource/ABI/bit/prefix/view checks; 18/18 focused functional
tests pass. All nine once21 measurements are preserved and independently
replayed. Exact five AUTO promotion rows: TF32 E0/no-bias on12.8/13.0/13.2;
F16 D0/E0/no-bias only13.2. TF32 own median improves22.6%/22.6%/14.7%;
F16 D/E13.2 improve at least4.85%/2.96%. N96 and D remain slower than Fast;
E is ahead of native-half Fast in this fresh integration protocol. Lower
toolkit D/E losses retain their old AUTO and are not retried. Production
still AUTO44 at this checkpoint: immediately implement these five rows as45,
verify actual AUTO and once101, then Triad. Numerical ABI5/schedule8 unchanged.
Source, all losses, metadata supplements and raw receipts are under
internal/perf/fixed-finalist-integration-20260907/README.md. Do not repeat
unchanged phase1 gates or restart discovery sweeps. Prepared Triad WIP remains
untouched and uncommitted separately.

Exact-F32 B0 profile follow-up (2026-09-07): current frozen AUTO44 CopyPlan
has no excessive shared-memory wavefronts in the measured generated SASS;
258.63M of302.46M executed warp instructions are scalar FFMA. One-window
unprofiled diagnostic is418.05/418.24us forced versus191.92/192.84us actual
cuBLAS FAST_TF32, not an admission run. Fast and CopyPlan physical symbols,
resources, clocks, raw counters, the failed parser attempt and successful
Fast-only continuation are retained in
internal/perf/ada-f32-b0-ncu-20260907/report.md. Do not repeat the captures or
implement the earlier unsupported A-bank-XOR idea. One test-only existing
CopyPlan-T256 transplant tested resources without changing arithmetic and
STOPPED at24localbytes despite80registers/3CTA; no GPUbits or7/21timing ran.
See internal/perf/ada-f32-t256-discovery-20260907/README.md. This is an
unqualified resource STOP, not a measured loss. The three TF32/F16 finalists
are being assembled into forced production routes using the existing test
harnesses, with all-three-toolkit qualification before literal AUTO changes.
The active execution brief/ownership is in the existing SDD ledger; keep
working autonomously without a terminal status pause while useful work remains.

Half discovery follow-up (2026-09-07): one CUDA13.2 batch leaves two F16
short-screen survivors, D0 M64N64/BK64/S3 and E0 M128N64/BK64/S2. Worst-stratum
candidate/current-AUTO median/p95 is0.957428/0.959236 and0.966673/0.970062;
candidate/Fast worst medians remain1.053728 and1.089085. B0 compact-S3 BF16/F16
both stop. Allfour small GPU correctness/resource checks pass; D/E each have
two active CTAs/SM,49152 shared bytes,zero local bytes. These are not AUTO
routes or all-toolkit/full batch-invariance qualifications. The special
corpus is finite, not NaN/Inf. Root replayed224brackets/32strata; evidence is
internal/perf/ada-half-batch-discovery-20260907/. Together with retained TF32
N96 these form the discovery finalist set; inference is not closed.

Latest discovery (2026-09-07): both four-warp TF32 BK16/S5 arms are rejected
after their one CUDA13.2 screen7, despite passing the small bit-exact GPU set.
E0 M128N96 is about121.48us, M128N128 about154.69us, current RNA119.05us,
Fast91.06us. Worst-order candidate/RNA p95 is1.020896/1.299583. No21,101,
retry or production promotion for these losers; the earlier eight-warp N96
at about101.5us remains the retained finalist, not yet AUTO. See
internal/perf/ada-tf32-w4-discovery-20260907/README.md and its integrity
manifest. The separate three-arm half batch follows; Triad remains deferred.

Latest integration (2026-09-07): actual Fixed AUTO44 now selects the qualified
exact-F32 CopyPlan for A/B/D/E, bias/no-bias, on CUDA12.8/13.0 as well as13.2.
All16 new toolkit-literals pass once-only post101 p50 AND p95 against former
Legacy in every eager/graph/parity stratum; worst-stratum median ratios are
0.814778..0.845826. cuBLAS Fast still wins: worst-per-cell AUTO/Fast p95 is
1.540643..2.240107. All3 matching builds and retained Fixed functional checks
pass; Triad cohort retention is13.2-only. CUDA bodies and numerical ABI5 /
schedule8 are unchanged. See internal/perf/ada-exact-toolkit-auto-20260907/
final-report.md and raw/binding files. The older Task7/43 paragraphs below
are historical checkpoints, not missing current aligned-hot wiring.

User-approved process correction: candidate discovery now uses one CUDA13.2
build, a small meaningful bit/repeat/eager/graph set and short paired timing;
full qualification is for a frozen integrated finalist batch. Reuse unchanged
evidence and rerun affected checks after fixes. Latest user priority is now
inference first: find a small research-informed batch, assemble its qualified
winners into the production dispatcher, verify the assembled inference set,
then resume Triad. This supersedes both the earlier interleaved Triad plan
and older per-prototype full-gate ordering; see docs/performance-playbook.md
section9. The prepared test-only Triad probe is retained, not being timed.

TF32 E0 profile is now complete on the unchanged AUTO44 CUDA13.2 build:
unprofiled one-window AB/BA is about119us AUTO versus91us cuBLAS Fast.
Nsight finds28.822M versus15.968M instructions for the same3.538944M HMMA;
see internal/perf/ada-tf32-e0-ncu-20260907/report.md. Both arms have96 useful
tiles; Fast's128 physical CTAs include32 padding early exits. The next
bounded probe, test-only RNA M128N96/BK32/S3, now passes its small CUDA13.2
bit/repeat/graph set and short7 then21 timing. E0 is about101.51us versus
119.06us production RNA and91.07us actual Fast: about14.7% less time than
the incumbent, still11.5% slower than Fast. Worst-order paired21 p95 ratios
are0.852894 versus RNA and1.114887 versus Fast. Root replayed all emitted
ratios/quantiles. Resources:128 registers, zero local bytes,86016 dynamic
shared bytes,128 useful CTAs. This is a finalist discovery, not production
qualification or an AUTO route. Source and evidence are under
tests/gemm_bi_fixed_tf32_n96_discovery.{rs,cu} and
internal/perf/ada-tf32-n96-discovery-20260907/. The next Triad probe remains
the existing TN prism split8 M64N64/S3; no new Triad timing is claimed here.

Latest measured checkpoint (2026-09-07): existing exact-F32 CopyPlan passed
paired21 and fresh101 against actualLegacy AUTO on CUDA12.8 and13.0 for
A/B/D/E, bothbias: all16toolkit-literals win p50 ANDp95 in every eager/graph
and start-parity stratum. Worst-per-cell median ratios are0.814324..0.844589;
worstp95 ratios0.828924..0.891455. cuBLAS Fast still wins these F32cells:
worst-per-cell candidate/Fastp95 is1.543210..2.239997. TF32 M64 C0/C1 loses
against actualRNA AUTO onboth toolkits (worstp95~1.518), so noTF32101 or
promotion. This measurement-only Task7 changes no selector or epoch; the
next integration connects the16exact winners with actualpostAUTO proof.
All20decisions, rawdata, source/binary/cache bindings and reviews are preserved
in `internal/perf/ada-f32-tf32-toolkit-20260907/ARCHIVE.md` and `final-report.md`.

Latest production checkpoint: Ada half production AUTO now uses tuning
revision43, selecting S3 for the two qualified CUDA13.2 B0/no-bias BF16/F16
cells. Matching three-toolkit functional tests, a single actual-AUTO post101
confirmation, artifact checks and independent source/evidence reviews pass.
The local commit is recorded at the top of the active SDD ledger.
`c8244b52` adds the reusable profile-guided kernel
optimization protocol to `docs/performance-playbook.md` section 9.
The earlier full60-cell AUTO42 paired101 versus native-half cuBLAS Fast reports 15 wins / 4 losses /
1 mixed result on each of CUDA 12.8 and 13.0, and 13 wins / 7 losses on 13.2.
This is 43 wins, 15 losses, and 2 mixed results across 60 BF16/F16 cells;
inference and Triad are not fully closed. Authoritative report:
`internal/perf/ada-half-auto-post101-20260907/final-report.md`.

Subsequent D0 small-tile, B0 16-warp, F16 B0 rectangular, and schedule-only S2
experiments did not qualify for production. The release B0 shape is exactly
`(M,K,N)=(4621,768,2304)`; the historical K256 16-warp screen is not B0 evidence.
The latest S2 experiment passed exactness and physical gates, then lost to
production by ratios 1.036883635 / 1.063118530 at p50/p95. Its report and
59-entry integrity manifest are preserved under
`internal/perf/ada-half-cutlass-schedule-s2-20260907/`.

The following S3 experiment is a confirmed own improvement on CUDA13.2 B0:
BF16 candidate/production paired101 p50/p95 `0.964968888968/0.996412833347`,
F16 `0.964326220305/0.981133357944`. Candidate/cuBLAS Fast remains
`1.267528694145/1.290857051084` and `1.187129654181/1.221770514572` respectively.
Final corrected tests include285 functional cases per dtype,126 with bias,
independently poisoned graph output and captured-argument checks. Source and
final evidence review accept retention; original incomplete graph-test runs
are excluded. See `internal/perf/ada-half-cutlass-s3-20260907/ARCHIVE.md`.
Production force wiring and independent CUDA12.8/13.0/13.2 functional
qualification are now complete: `FixedTile::Tc128Sm89S3`, BF16/F16, exact
five-argument ABI, 98304 dynamic shared bytes. Final full-half suites pass
11 tests on each toolkit, with clean memcheck/racecheck/synccheck and retained
Fixed routes. Live S3 register counts are182/182/188 across the toolkits,
with zero local/stack/spills. All old CUDA and Triad compositions remain
unchanged; actual SM89 cohorts and SM120 compile contracts pass on13.2.
Source review closed a near-INT_MAX lookahead overflow through an S3-only
host K+127 guard; the CUDA body remains the tested candidate.
See `internal/perf/ada-half-s3-force-20260907/ARCHIVE.md` and `final-report.md`.
Production paired S3 / actual AUTO / native-half cuBLAS Fast timing is now
complete on all three toolkits. Only CUDA13.2 B0/no-bias qualifies for S3:
BF16 paired101 worst p50/p95 `0.929639720306/0.945012678690`,
F16 `0.927334867696/0.958864521137` (S3/AUTO; lower is better).
Both12.8/13.0 BF16 failedscreen21 p95, and F16 failedfresh101 p95, so those
four cells retain the old Swizzle choice. All failed/mixed results are valid
and preserved; no repeated screening. S3/Fast worst101 on13.2 remains
`1.213927713009/1.232313497321` forBF16 and
`1.131168505705/1.167174487390` forF16: own improvement, not vendor victory.
The new measurement harness passes all3 matching builds and functionalsmokes;
its exact same-attempt SSH/telemetry validator closes all10savedruns after
host RED/GREEN. Original source/binaries/rawdata remain frozen and all3Fixed
PTX identities exactly match the approved production force artifacts.
See `internal/perf/ada-half-s3-paired-20260907/ARCHIVE.md` and`final-report.md`.
Those two measured13.2 cells are now selected by production AUTO43. All other
literal preferences and the old fallback table remain unchanged, including
12.8/13.0 B0 Swizzle and S3-unavailable behavior. The selector oracle covers
all60 cells across8 independent holder-availability states. All3 full libraries
pass645 tests, focused stage tests5, nonignored performance53, and the three
AUTO/forced-Swizzle/forced-S3 hot-corpus GPU tests per toolkit pass bit gates.
The compiled CUDA modules remain byte-identical to Task6A; only the host
tuning epoch changes, and captured42 graphs correctly reject for re-capture.

ActualAUTO43 post101 on13.2 has8 complete dtype/path/start-parity strata,
9696 raw samples and2424 paired ratios. BF16 AUTO/old-Swizzle worst p50/p95
is `0.929889194840/0.953700555376`; F16 is `0.929018443239/0.955993909110`.
Every stratum wins both own quantiles. AUTO/cuBLAS Fast still loses: BF16
worst p50/p95 `1.207046379000/1.231223218539`, F16
`1.133473338293/1.159192520014`. This is at least7.01%/7.10% lower median
time than the old route, not a Fast victory or a new global60-cell sweep.
Raw post101 SHA is
`b6d0fca4305b6846a427c9dc8432e727dfa5434e68f380ca29511916a8c7d620`.
See `internal/perf/ada-half-s3-auto-20260907/`; raw numerical/physical/poison/
guard/immutable-input gates and exact same-attempt exit closure pass. Previous
preflight gaps (unused legacy controls and pre-warmup input checks) are closed.
The host analyzer's shared-summary label bug was fixed and reviewed without
changing or repeating the measured GPU run; original reports are preserved.

Cross-toolkit backlog correction (2026-09-07): exact-F32 CopyPlan loads and is
force-reachable on12.8/13.0, but its literal AUTO promotion remains13.2-only
until the separate Task7-winner integration. The paired measurements above
are now complete. TF32 is different: commit86fd7f56 already enabled
the faster RNA-wide AUTO for all aligned A-E/bias cells on all three toolkits.
It takes precedence over the ordinary TF32 picker. The M64S2 C preference in
that lower picker remains13.2-only, but that is a fallback-view/holder domain,
not missing aligned-hot AUTO wiring. The earlier
`internal/perf/ada-half-s3-force-20260907/existing-toolkit-auto-gaps.md`
overstated the TF32 gap by omitting this precedence; retain it as history, not
the current aligned dispatch inventory. The authoritative prior promotion is
`internal/perf/ada-rna-toolkit-auto-20260906/`, whose49-entry manifest was freshly
verified. Its historical C comparison uses oldM128S2 on12.8/13.0. Task7 compared
the existing M64S2 candidate against actualRNA-wide AUTO, not against that
obsolete aligned incumbent, and the candidate lost. No TF32 route is disabled.

P0 Nsight evidence under `internal/perf/ada-half-b0-ncu-20260907/` shows
53.40% tensor-pipe activity for production versus 73.48% for cuBLAS, with
similar requested memory traffic and substantially more executed instructions
in production. Resume from the current entry at the top of
`.superpowers/sdd/handoff-codex-gemm-bi-triad-2026-09-05/progress.md`; it records
the sole Ada owner and the profile-driven next experiment. Historical active
task paragraphs farther down that ledger are not the current execution state.

mamba-rs ships two GEMM families: the Fixed (inference) family
(`src/mamba_ssm/gpu/gemm_bi_fixed.rs`, tuned to the maximum earlier) and the
deterministic training triad (`src/mamba_ssm/gpu/gemm_bi_triad/`, kernels under
`kernels/gemm_bi_triad/`). The triad computes the three training GEMMs
(NN forward, TN dW, NT dX) with a fixed summation order per output element, so
a training run is bit-reproducible. Precisions: half (bf16 / f16 tensor
cores), TF32 (f32 inputs rounded to TF32 in registers, tensor cores), and exact
F32 (scalar FMA, the reference).

The owner's target for release 0.7 (see the release order in section 8):

- every precision except exact F32 must be AHEAD of cuBLAS FAST on the
  measured boards (for f32 inputs FAST means `CUBLAS_COMPUTE_32F_FAST_TF32`,
  i.e. TF32 against TF32);
- exact F32 retains its scalar-FMA bit contract and is squeezed toward cuBLAS
  FAST too; PEDANTIC remains a separately labelled diagnostic denominator;
- no CUTLASS or any other external kernel in the tree: only our own
  deterministic kernels; cuBLAS stays the fallback (`GemmBackend::Cublas`);
- every compiled kernel is reachable by a production dispatcher for its card
  class and toolkit; an unreachable kernel is a defect;
- nothing is removed without a paired measurement on the same box, and the
  owner sees the table before removal;
- web research of the state of the art is expected before kernel work and is
  recorded in the plan note; every measured plus is kept, even 1%; only a
  measured minus or noise is dropped.

## 2. Where things are

- Repository: `~/IdeaProjects/mamba-rs`. Work happens in the worktree
  `internal/worktrees/gemm-bi-triad-sm80`, branch `codex/gemm-bi-triad-sm80`.
  `main` is not touched until the 0.7 merge. Never `git push`; commit locally
  after every green phase. No AI authorship trailers in commits. Commit
  subjects describe the defect or the design, never a plan index.
- `internal/` is git-ignored by default. Selected handoff and evidence files
  are explicitly tracked in the September 6 checkpoint commits; other evidence
  remains disk-only until explicitly added. Copy untracked evidence before
  deleting a worktree.
- Boxes:
  - `ada`: `ssh ada`, RTX 6000 Ada (SM89, 142 SMs, power cap 300 W; under a
    sustained GEMM the SM clock holds 1800 MHz, so every harness number is at
    1800 MHz). Synced tree at `/root/mamba-rs-triad`; toolkit env in
    `/root/triad-env.sh` (CUDA 13.2; 12.8 and13.0 also installed). Scripts in
    `/root/run-*.sh`, logs in `/root/logs`, SASS blobs in `/root/wide-check`.
  - RTX 5090 (SM120, 170 SMs): rented by the owner's word only; the last
    rental ended 2026-09-04 09:15Z. Its scripts and evidence are in
    `internal/perf/sm120-*` and `internal/agent-operational-rules.md`.
- Sync from the Mac: `scratchpad/sync_box.sh ada` (rsync + touch of changed
  files; without the touch cargo keeps a stale binary because rsync preserves
  the Mac's older mtimes — this bit us once).
- The Mac cannot build with `--features cuda` (cudarc's build script needs a
  toolkit): every CUDA compile, test and measurement runs on a box.
- Documentation and comments are English only; task identifiers are Latin.

## 3. The architecture you will touch

### Modules and fragments

The triad's portable module (`ModuleKind::TriadSm80`) is composed from
fragments by `compose_module_source_for` (modules.rs):
`_typed_prelude.cuh`, `contract.cuh`, `common.cuh`, `epilogue.cuh`,
`mma16.cuh`, `sm80.cu`, then the extension fragments `sm80_streamk.cu`,
`sm80_tf32_wide.cu`, `sm80_tn_splitk.cu`, which are composed for every
sm80-family target EXCEPT CC 12.x (`sm80_target_composes_extensions`).

Why the split matters: the SM120 TF32 cohort (the 5090's qualification record)
is frozen against the compute_120 composition of the portable module. Any byte
change to `sm80.cu` (or the shared headers) moves that composition's source
digest and unbinds every SM120 cohort; a refreeze is only possible on a 5090
(the compile key digests the capturing box's NVRTC library set). So: new
portable kernels go into extension fragments; `sm80.cu` stays byte-identical
unless a 5090 is available to refreeze. Both the wide tile and the TN split-K
family hit this trap once each before moving to fragments.

Other modules: `TriadSm120` (`sm120.cu`, TMA + mma.sync; the 5090's own
kernels), `TriadSm90a` (wgmma), `TriadSm100` (tcgen05) — code and compile
gates exist, no board has run them; `TriadScalar` (`scalar.cu`, exact F32).

### Routes, specs, contract

`contract.rs` holds the kernel inventory: `Tf32KernelSpec` rows (symbol,
tile, bk, stages, threads, dynamic shared bytes, instruction family, operand
conversion, numeric contract fields) in `SM80_TF32_ROUTE_SPECS` (the base 18,
portable on every target), `SM80_TF32_WIDE_ROUTE_SPECS` (the extension routes),
`TF32_SPLITK_CANDIDATE_SPECS` (6 portable split-K) + `TF32_SPLITK_EXTENSION_SPECS`
(4 TN split-K). `tf32_route_specs_for(module_kind, extensions)` and
`tf32_splitk_specs_for(extensions)` are the only accessors the loader, the
Driver-ABI census, the PTX validators, the identity fixtures and the selector
harness use — add a route there and everything follows the table.

`Tf32PhysicalRoute` names a route family: `MmaTf32RnaV1(Tf32PortableRoute
{tile, stages})` (tiled), `MmaTf32RnaSplitK{2,4,8}V1`, the SM120/SM90a/SM100
variants, `Sm120TmaMmaTf32RnaStreamKV1`. `Tf32PortableTile` currently:
M128N64, M64N64, M16N32, M16N16, M32N32, M128N128 (the wide eight-warp tile).
A measurement twin is added as one more variant + spec + register-cap arm +
symbol pin (see section 6 for the ones measured and removed this week).
Register caps per symbol live in modules.rs
(`ModuleKind::TriadSm80 if symbol.contains(...) => Ok(N)`).

Operand conversion (`kernel_identity.rs::ResolvedOperandConversion`):
`RegisterCvtRnaTf32F32V1` (the base routes: `cvt.rna.tf32.f32` in
registers) and, new this session, `RegisterAddHalfUlpTf32V1` (= 4): TF32
rounding as one unguarded integer add of 0x1000. Measured on sm_89 with a
probe kernel: the tensor core reads only the upper 19 bits of a TF32 operand
(0x3f800fff multiplies as 1.0, 0x7f800001 multiplies as +inf even after
cvt.rna), so the add reaches the mma bit-identical to cvt.rna for every finite
value and both infinities; the only divergence is a NaN whose payload sits in
bits 12-13 (the add keeps it a NaN where cvt.rna + truncation yields +inf).
The validator (`validate_tf32_feature_instructions`) accepts the add
conversion for the portable module's wide routes only, requires the mma token
in such an entry and rejects cvt.rna there.

### Validators, census, identity

- `modules.rs` validates every module's PTX: exact export set, per-entry
  required/forbidden instruction tokens, parameter ABI (`Sm80Tf32KernelParams`
  is 32 bytes, eight 4-byte fields; the split-K and stream-K kernels take the
  scratch and counter pointers before the operand pointers), no atomics in
  TF32 kernels (the split-K fused fixup uses exactly one `atomicInc` with an
  immediate limit), per-symbol local-memory (spill) admission.
- The Driver-ABI census (`census_tf32_driver_abi`) checks every route symbol
  of the composed module against the driver's view; it is extension-aware.
- `kernel_identity.rs` + `tests/kernel_identity*.rs`: the compile key, source
  digest, header manifest and artifact digests of each module; the SM89 and
  SM120 TF32 cohorts (`dispatch.rs::SM89_TF32_QUALIFICATION_IDENTITY` and the
  SM120 records) must match the live tree or the TF32 policy declines with a
  `mamba-rs WARNING:` naming the field that differs. After any kernel or
  compile-argv change on a board: refreeze from a one-cell selector capture
  (`scratchpad/refreeze_identity.py --allow-source-move`).
- NVRTC in-process nondeterminism (fixed 2026-09-04): the register allocator
  re-loads fragments differently per compile; the cure is inline-PTX
  `ld.shared.v4/v2` loads (`gemm_bi_scalar_load_fragment<Count>` in
  scalar.cu); `arch_compile_gates` refuses the element-wise form. Cache trace:
  `MAMBA_RS_CACHE_TRACE=1`, `MAMBA_RS_CACHE_TRACE_DIR=<dir>`.

### Dispatch and evidence

`dispatch.rs` holds the measured cell tables: `SM89_TF32_EVIDENCE_CELLS`
(69 rows, gate counts `[5, 3, 36, 25]`, each row keyed by a `// <cell_id>:
<symbol>` comment), the SM120 tables, the half tables, the scalar (exact F32)
cells, and the policy rules (`Sm80TcPolicyV3` incl. the stream-K rule fields).
Rows are produced by the selector harness and applied with
`scratchpad/apply_sm89_rows.py`; expected-value fixtures regenerate with
`scratchpad/regen_sm89_expected.py`. An unmeasured shape falls to the generic
rules or to the exact scalar family — always visibly (warn_once), never
silently, since commit c447234.

### The measurement harness

`tests/gemm_bi_sm120_tf32_selector_qualification.rs`
(`sm120_tf32_projection_selector_qualification`, board-generic despite the
name): for each cell it times the exact scalar, every TF32 candidate of the
bound module (eager and CUDA-graph paths), gates each candidate numerically
(the tiled reference family must be BITWISE equal to the forced portable
reference; the split-K and stream-K families are held to 0.25% against the
EXACT scalar output, because the tiled TF32 reference itself drifts past the
tolerance at K = 10400), checks bit-stability across repeats, and records a
JSONL row per cell with every candidate's `order_stats`, `raw_samples`,
`discovery` and `final` speedups. Env: `MAMBA_RS_SM120_TF32_SELECTOR_QUALIFICATION=1`,
`MAMBA_RS_SM120_TF32_SELECTOR_CELLS=<comma list>`, `GEMM_BI_QUAL_WINDOWS=<n>`,
`MAMBA_RS_SM120_TF32_SELECTOR_JSONL=<path>`. `python3 /root/requal_table.py
<jsonl> <cublas evidence dir>` prints the table against the scalar and cuBLAS.
cuBLAS denominators come from `tests/gemm_bi_performance_matrix.rs`
(`gemm_bi_cublas_performance_denominators`, `GEMM_BI_CUBLAS_CELL_IDS=...`);
the SM89 record is `internal/perf/sm89-hotall-20260904`.

MEASUREMENT PROTOCOL (learned the hard way this week): on the power-capped
Ada, separate runs drift 3-5% while two identical kernels inside one run
agree within 0.1%. So every kernel variant is compared as a second route
symbol in the SAME selector run as the incumbent (the harness times all
candidates per cell), never as a separate run. Three "minus" verdicts from
separate runs are being re-measured this way (section 6).

Profiling recipes on ada (all in `/root/run-ncu-*.sh`, `/root/ncu-*.sh`):
`ncu --kernel-name regex:<symbol part> --launch-skip 4 --launch-count 1 --set
full --import-source yes -o <rep> <selector binary ...>`, then `ncu --import
<rep> --page source --print-source sass --csv` for per-line stall samples and
`--page raw` for `smsp__average_warps_issue_stalled_*` (the stall-reason
split). ncu runs the board at 2.1-2.4 GHz, the harness at 1.8: compare like
with like. Standalone SASS of the composed module: concatenate the fragments
as `/root/run-plan.sh` does and `nvcc -arch=sm_89 -cubin -Xptxas -v`, then
`cuobjdump -sass` and `python3 /root/loop_mix.py <sass>` (instruction mix of
the hot loop per kernel).

Gates before a commit (all on a box, `--features cuda`): `cargo fmt`, `cargo
clippy -- -D warnings`, lib tests, `arch_compile_gates`, `kernel_identity`,
`kernel_identity_cuda`, `gemm_bi_tf32_contract`, the SM120 cohort test
(`at_least_one_sm120_tf32_cohort_matches_this_tree`), the selector suite,
`gemm_bi_tc`, determinism and invariance suites; `/root/run-ada-gate*.sh`
chains them. Full workspace runs at phase ends.

## 4. What was found and fixed (the audit of 2026-09-03 and after)

The five-audit census found 420 GEMM symbols in six NVRTC modules of which 231
were compiled but unreachable by any production selector (SM120 half TMA 78
of 96, SM120 exact F32 6 of 12, SM120 TF32 13 of 18, SM80 portable TF32 5 of
18, every TF32 split-K, all SM90a and SM100 kernels, the Rect128x64 tile, the
tuned scalar cells under the TF32 policy). Nine gates were pinned to exactly
170 SMs and three to NVRTC 13.2, so any other board declined silently. The
fixed family's SM120 half kernels ignore alpha/beta and were safe only by the
launcher's constants. Two arch derivations disagreed; the loader keyed SM120
kernels on "any major > 12"; no `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`
handling; every decline on the selection path was a bare `None`.

Done since (commits on the branch, all gated on both boards while the 5090
was rented; none pushed):

- Visibility and portability (A1-A7): declines report once per process
  (`diagnostics::warn_once`) naming the failing identity field; one CC ->
  family predicate for loader and selectors; live SM count in heuristics; the
  no-op env flag combinations fail at context creation; CC 11.0 on sm_110a;
  compile gates cover sm_86, sm_87, sm_110a; the GEMM compile key no longer
  digests the machine's include paths or the SSM state cap (the Fixed module
  keeps the cap: the SSM kernels read it).
- SM120 (5090): 30 half cells for the product shapes (table of 60, all
  qualified eager == graph); nine exact-F32 product cells with the generic
  floor lowered 3 -> 1 tiles per SM; the SM120 TF32 evidence widened to 21
  shapes and requalified under driver 595.84; the deep-reduction cells; the
  Fixed-versus-Triad pairwise census (B9: on the 5090 the SM120 half tiles and
  tc64/tc16 rungs are one body compiled twice; the portable TF32 bodies differ
  by 3-5% in the Fixed family's favour except m64n64 s3 where Triad is 19%
  faster); half TN stream-K on SM120 (persistent grid, fixed-order slab fold,
  own numeric contract `MmaSyncF32StreamKFixedOrderV1`; product TN shapes
  0.32-0.96 of the best tiled route; opt-in policy
  `MAMBA_RS_BI_HALF_POLICY=tiled|streamk`, contract bit
  `TRIAD_MMA_SYNC_STREAM_K_V1`); paired half TN/NT against cuBLASLt at
  vendor parity on the projection shapes (0.95-1.02, the TN batch_in_proj
  1.41 gap remains). Final 5090 full run: 2242 passed.
- SM90a / SM100 (D1, D2): `resolve_sm90a_auto` / `resolve_sm100_auto` with
  per-CC tables (empty, fail-closed), observed enqueues sealed by the
  qualification harness, capture-time declines, cohort search following the
  bound module family, SM100 offered only on CUDA >= 12.9 (nvrtc 12.8 does
  not know `compute_100f`). No board has run them.
- SM89 (ada): the 21-shape SM89 TF32 cohort (41 cells) replacing a five-cell
  synthetic one; the SM89 hot requalification (18 product cells; 15 TF32
  winners); the sm80 tc64 TN dW stream-K twin in its own fragment
  (`sm80_streamk.cu`; batch_input_proj 4.04x -> 1.39x of cuBLAS; rule =
  tiles64 x 8 <= SMs x 9 and tiles64 x slabs >= 32 x SMs on `Sm80TcPolicyV3`);
  the identity flake fixed (inline-PTX fragment loads); the wide TF32 tile
  (128x128, eight warps, `sm80_tf32_wide.cu`) taking every large NN cell; the
  TN split-K family (four TN candidates, eight partitions, `sm80_tn_splitk.cu`)
  taking all six TN hot cells; the harness gate on the exact reference for the
  split and stream families; the wide tile pipelined (ldmatrix A, fragment
  double buffer, cp.async spread, staged epilogue) and then, this session,
  the half-ulp add rounding.

## 5. Where each lane stands on SM89 (ada, 1800 MHz)

Reference records: `internal/perf/sm89-hotall-20260904` (22 shapes x 3 ops,
cuBLAS FAST and PEDANTIC denominators), `sm89-requal-wide-20260904`,
`sm89-requal-tn2-20260904`, `sm89-wide4-20260904`, and today's
`sm89-wide-quad-20260905` (same-run twins).

| lane | state | evidence |
|---|---|---|
| NN TF32 | wide tile + add rounding wins 11 of 15 cells; against cuBLAS FAST: d768_in_proj 1.23, d768_out_proj 1.24, prism_in_proj 1.03, large_deep 1.37, large 1.25, batch_in_proj 1.39, prism_out_proj 1.20, prism_input_proj 1.22, batch_input_proj 1.18, batch_out_proj 1.23, rect_tall 1.21; small cells d128_in_proj 1.74 (m64n64 s2), d128_out_proj 1.09, underfill 1.08 (split-K m16n32), split_candidate stays scalar (1.93). Was 2.9-4.2x on 2026-09-03 | sm89-wide5-20260905 (the requalification record), sm89-wide-quad-20260905 (same-run twins) |
| TN TF32 | split-K m64n64 s3 on every hot cell: 2.8-3.1x of FAST (prism_out_proj 106.4 us, rect_tall 123.2, batch_out_proj 213.0); the split fixup folds beta; bias NN-only | sm89-requal-tn-20260904, -tn2- |
| NT TF32 | m64n64 s2 / m128n64 s3 tiles: 1.9-2.3x of FAST | sm89-hot-tf32-20260904 |
| half bf16/f16 | 1.1-1.5x of FAST on projections; TN batch shapes fixed by the stream-K twin (4.0x -> 1.4x); stream-K default under the tensor-core tier (`resolve_half_triad_policy`) | sm89-streamk-tn-20260904 |
| exact F32 | scalar family 0.96-2.25x of PEDANTIC on the hot cells (median 1.51 over 66 cells); NT worst (1.55-2.25x); split-K/stream-K on underfilled mid shapes untouched | sm89-hotall-20260904 |

The NN TF32 lane is the one below cuBLAS FAST on every product shape and has
had the most work. What is known about its remaining 1.24-1.37x:

- cuBLAS's kernels on these very shapes (ncu, `/root/ncu-cublas-name.sh`):
  d768_out_proj and large_deep run `cutlass_80_tensorop_s1688gemm_128x128_16x5`
  (128 threads = four warps of 64x64, BK 16, five stages, 224 registers, its
  128/512-CTA grids are the CUTLASS swizzle round-up of the same 96/384
  tiles); batch_out_proj runs `128x256_16x3` (eight warps of 64x64). Its loop
  is 4.7 SASS per HMMA with conversions interleaved one-for-one between the
  HMMAs and fragment loads as `[R + UR + imm]`; its warps are pipe-throttled
  (stall split: math_pipe_throttle 4.80, wait 0.85, barrier 0.48 cycles per
  instruction). In-loop tensor utilisation about 86% against our 63-67%; wave
  quantisation is identical for both (same tiles).
- Our eight-warp body (2 x 4 warps of 64x32): cycles per issued instruction
  5.68 = math pipe throttle 2.05, wait 1.13 (fixed-latency dependency),
  selected 1.00, short scoreboard 0.39, barrier 0.36; 6.4 SASS per HMMA after
  the add rounding (407 static instructions per 64 HMMA: 127 IADD3 of which 64
  are the rounding, 58 IMAD, 44 LDS incl. 12 ptxas dummies, 16 LDSM, 8 LDGSTS
  + the k-clamp arithmetic of the copy path).
- Measured and dropped this week (each in the plan note with numbers): four
  warps of 64x64 with cvt (-15%) and with the add rounding (-5%); the same at
  BK 16 x five stages, cuBLAS's exact configuration (-13%: with one warp per
  scheduler our body's dependency stalls dominate, wait 1.70 cycles per
  instruction); 128x256 and 256x128 tiles; TF32 rounding once per stage in
  shared (+ a barrier); the stage boundary before the last mma group (-4% on
  the eight-warp body, twice); fragment loads written per atom group between
  mma groups (ptxas emits the identical loop); the cp.async ignore-src
  predicate form; interleaved column atoms with 16-byte B loads (a TF32 mma B
  pair comes from k rows t and t+4, so a vector load can never produce an
  adjacent register pair: ptxas inserts a move per HMMA).
- The lever that is NOT in-loop: SM fill. d768_out_proj is 96 tiles on 142
  SMs (68% of the device for the whole kernel), batch_out_proj 246 tiles (1.73
  waves, 87%), large_deep 384 (2.7 waves, 90%). A stream-K schedule over the
  wide tile recovers that: on d768_out_proj alone 78.4 x 0.68 = 53 us would
  beat cuBLAS's 63. The design is written (section 7).

## 6. State of the tree at the checkpoint (commit 417eafa1)

Commit 417eafa1 "Round TF32 operands by a half-ulp add and measure the wide
tile's twins in one run" on top of 72e0420; the working tree is clean. Gate
on ada (13.2) for it: fmt, clippy, lib 538, arch_compile_gates 50,
kernel_identity 22, kernel_identity_cuda 2, tf32 contract 82, gemm_bi_tc 33,
selector harness unit tests 27, cohort binding smoke 1 — all green. The plan
note has today's log; scratchpad copies of every variant measured today are
under the session scratchpad (`sm80_tf32_wide_*.cu`).

What the tree holds:

1. The eight-warp wide body (`gemm_bi_nn_sm80_mma_tf32_v1_m128n128_bk32_s3`)
   with the copy plan (pointers, destinations and bounds computed once per
   thread, a stage = pointer advance + k clamp) and the half-ulp add rounding
   (`tf32w_round`). Route conversion `RegisterAddHalfUlpTf32V1`. This is the
   measured winner: 78.4 / 488.4 / 82.8 us on the three control cells.
2. The four-warp twin (`..._m128n128w4_bk32_s3`, templated body
   `tf32q_nn_kernel<Bk, Stages>`, row-padded B stage, one warp per scheduler,
   clean copy path, stage boundary before the last mma group, add rounding)
   ran in the full requalification and won no cell (5% behind on the hot
   cells), so it was removed from the tree; its source is in the session
   scratchpad (`sm80_tf32_wide_three_routes.cu`) and its numbers in the plan
   note. The clamped-copy twin measured -2.7% in the same run and was removed
   the same way (`sm80_tf32_wide_three_routes_c.cu`).
3. The 14 NN rows of `SM89_TF32_EVIDENCE_CELLS` were replaced from the
   requalification record and the SM89 identity refrozen from a one-cell
   capture of the final tree (`internal/perf/sm89-identity-halfulp-20260905`);
   the count pins are the committed ones (one extension wide route).

## 6a. An open question to settle FIRST (found at the checkpoint smoke)

The cohort binding smoke (`tests/gemm_bi_tf32_cohort_binding.rs`,
`tf32_cohort_binds_on_this_board`, run with `--ignored`) on the committed
tree reports for dims (2048, 768, 3072) (= d768_in_proj) under the TF32
policy: `served=TriadSm80:gemm_bi_nn_sm80_mma_tf32_v1_m64n64_bk32_s2`,
although the table row for that cell names the wide tile
`m128n128_bk32_s3`. Every NN row generated by `sm89_rows_from_selector.py`
carries the gate `RequiresNoBiasAndVectorAlignmentEvidence` ("both bias and
pointer-alignment evidence are absent"), so a request WITH a bias (the
training forward has one) falls past the wide-tile row to whatever the
generic rules serve. The wide kernel itself supports bias (accumulators
start from the bias row). Check: (1) what the smoke's request carries (bias,
alignment) and what `tf32_auto_operands_match` requires per gate; (2) whether
the requalification cells (`PROJECTION_CELLS`) qualify with bias = None; if
so, qualify the wide tile WITH bias and vector-aligned pointers so the rows
can carry `RequestContractSafe`, or split the evidence per epilogue. Until
this is settled, the NN numbers of section 5 may not reach a production
forward that passes a bias. This is the first thing to verify.

## 7. What to do next, in order

1. (Done at the checkpoint, see the last commit on the branch.) The recipe
   for the next round is the same: measure twins in one run
   (`/root/run-plan7.sh` on three control cells), prune, full NN
   requalification (`/root/run-nn-wide5.sh`: 15 NN cells, 7 windows, about
   an hour, new evidence directory), rows (`sm89_rows_from_selector.py` ->
   `apply_sm89_rows.py` -> `regen_sm89_expected.py`), one-cell identity
   capture and `refreeze_identity.py <jsonl> specialized_identity
   SM89_TF32_QUALIFICATION_IDENTITY --allow-source-move`, the gate list of
   section 3, commit.
2. Stream-K over the wide tile (NN TF32). Draft kernel in the session
   scratchpad `wide_streamk_draft.cu`, to be appended to
   `sm80_tf32_wide.cu`: units = tiles x k-tiles dealt by
   `gemm_bi_streamk_range` (already in `sm80_streamk.cu`), segments walked
   from the end of the range, the tiled body's mainloop factored into
   `tf32w_nn_mainloop(k_begin, k_end)`, one partial slab per CTA (256 threads x
   64 floats; 142 x 16384 floats fits the 8M-float `SPLITK_SCRATCH_CAP`),
   release/acquire flags (`gemm_bi_streamk_raise/await/clear`), the tile's
   last CTA folds lower contributors in ascending CTA order then its own
   accumulators (bias only in the owner's init), the staged epilogue. Host
   side: a new `Tf32PhysicalRoute` variant for the portable stream-K
   (module TriadSm80, own numeric contract like the SM120
   `Sm120TmaMmaTf32RnaStreamKV1`: `ResolvedNumericContract::MmaSyncF32StreamKFixedOrderV1`,
   ownership `OwnerCtaPerOutputTileStreamKFixedOrderV1`), grid = SM count,
   scratch and flags from the split-K buffers as `prepare_tf32_splitk_direct_
   graph_sequence` does (arguments: output, partial, flags, a, b, bias,
   params), the selector harness holding it to the exact reference like the
   split family, identity and census entries. Expected: d768_out_proj below
   cuBLAS; batch_out_proj and large_deep gain 10-13% from the wave tail alone.
3. In-loop efficiency of the eight-warp body (the other half of the NN gap):
   the per-line stall profile of the add body is the next instrument. Ideas
   not yet measured: uniform-register stage offsets for the B loads (cuBLAS's
   `[R + UR + imm]` form), predicated copies with precomputed per-thread masks
   instead of the per-stage clamp arithmetic (the clamped-copy twin is the
   first step), a 2-CTA-per-SM configuration (needs <= 128 registers and <=
   48 KB per CTA: a 128x64 eight-warp tile).
4. TN TF32 (2.8-3.1x): a k-major 128x128 eight-warp body under the split
   family (the analogue of what took NN from 3-4x to 1.3-1.4x), then
   stream-K on it. NT (1.9-2.3x): the wide tile's NT twin.
5. Half lanes: a 128x128 tensor-core tile, split-K for the deep-thin shapes,
   small-M tiles for the thin cells (the 4-5x losers of sm89-hotall).
6. Exact F32: split-K / stream-K on the underfilled mid shapes (tn underfill
   10.1x, nn rect_wide 9.8x, tn d128_in_proj 9.2x of PEDANTIC).
7. Fixed-versus-Triad on SM89 (B9 table in the plan note): the Fixed portable
   TF32 bodies are 8-26% faster than Triad's old m128n64/m64n64 bodies on
   Ada; with the wide tile in place, re-measure before any removal, and show
   the owner the table.
8. Then the release order (section 8).

## 8. Release order (owner, 2026-09-03; do not reorder)

1. Every compiled kernel reachable by a production dispatcher for every card
   class (SM80/86/89, SM90a, SM100/103/110, SM120/121) and toolkit (12.8 /
   13.0 / 13.2); duplicates measured and the loser removed with the numbers in
   front of the owner; no silent declines. (Mostly done; SM90a/SM100 wait for
   boards.)
2. The triad ahead of cuBLAS FAST on the 5090 and on ada for every precision
   but exact F32 (in progress: NN TF32 on ada is the open lane).
3. Full retest on both boxes.
4. Rentals of other SM classes by the owner's word only; final runs.
5. Release 0.7: merge to main, rename `gemm_bi_fixed` -> inference, API
   (`GemmBackend{Deterministic, Cublas}` + `GemmPrecision`), documentation.
   Never push without the owner's word.

## 9. Standing lessons (each cost a day or a night)

- Measure every textbook step; half of them were slower here. Keep only a
  same-run plus.
- The reference of a numeric gate can be the wrong party: at K = 10400 the
  tiled TF32 chain drifts past 0.25% on 0.05% of elements while an
  eight-partial split sits within a tenth of the tolerance. Different
  summation families are held to the exact scalar.
- The tensor core ignores the low 13 bits of a TF32 operand; the
  finiteness guard of cvt.rna is dead weight in front of an mma.
- ptxas rematerialises anything cheaply derivable from the thread index
  inside a loop, branch blocks split its scheduling regions, and a dependent
  chain inside a copy block stalls the whole warp: keep addresses in
  registers (a shuffle from the lane makes a value opaque to
  rematerialisation) and keep copy blocks branch-free.
- An `if` around only part of a stage's copies leaves stale rows in shared
  memory and shows up as a repeat-bits gate failure, not as a wrong answer.
- Editing `sm80.cu` unbinds the SM120 cohorts; extension fragments only.
- The runtime rejects a whole module on any local-memory report unless the
  spilled symbol is admitted per symbol (done); ptxas 12.8 spills where 13.2
  does not.
- QuietGpu treats any other process on the GPU as a competitor: performance
  runs on ada only with the serve stopped and nothing else on the board.

## 10. Fixed exact-F32 SM120 promotion checkpoint (2026-09-06)

This checkpoint is newer than sections 5--7. Fixed remains the active lane;
do not switch to Triad until the remaining exact-F32 vendor gaps are closed or
explicitly dispositioned by the owner.

- Added production symbol
  `gemm_bi_nn_fixed_sm120_f32_n64_copyplan_v1` from
  `kernels/gemm_bi_fixed/sm120_f32_n64_copyplan.cu`, source SHA256
  `b49017ddfb688c2814b50d732aa5027817b3d4adbbb997042ea83a041baad0ff`.
  It is exact ascending scalar FMA, 64x64x32/S2, 128 threads, compact five
  argument ABI, and is composed only into the Fixed compute_120 module.
- RTX5090 holder admission: NVRTC13.2 known library, CC12.0/170 SM,
  111 registers, 32768 static shared bytes, zero local bytes, three CTA/SM,
  max-shared carveout. Strict PTX ABI/resource and module-identity gates pass.
- Forced correctness: 283 fixtures pass across raw bits, eager/captured graph,
  row views, alignment/tails, guards, input immutability, order and exceptional
  values. Compute Sanitizer memcheck/initcheck/synccheck/racecheck is clean on
  the eight-fixture suite (`--report-api-errors no` only suppresses the
  intentional terminal ABI-probe Driver error).
- Paired 101-window admission file:
  `internal/perf/sm120-fixed-exact-n64-20260906/sm120-fixed-eb-admission101-20260906-v1.jsonl`,
  SHA256 `62c96372caf6ea8fa6020299d266fb8303c99674d1ca927f1faa4ae91b45249d`.
  All 48 paired cohorts are complete. The candidate beats the prior production
  AUTO/Legacy on E0, E1, B0 and B1. It beats PEDANTIC cuBLAS only on B1
  (B shape with bias), by roughly 5.9--7.1%; the other three are still open.
- Production AUTO is fail-closed to exact CC12.0/170 SM, NVRTC13.2 known
  library, admitted holder, ExactScalarFmaV1, homogeneous F32, A/B/C A16,
  null-or-A4 bias, and exactly E0/E1/B0/B1. Post-promotion 21-window route
  evidence has 48/48 complete cohorts and actual graph symbol/geometry/ABI:
  `sm120-fixed-eb-post-auto21-20260906-v2.jsonl`, SHA256
  `0502519d8cc6fb98cd7b71ab5d147bfb11ab0de8970a2ff799e73986e403b92a`.
- RTX5090 regressions after AUTO promotion: library 596 pass/46 ignored;
  all 55 architecture compile gates pass for compute_80 through compute_121;
  kernel identity 2 pass/3 ignored; SM120 exact test 2 pass/3 ignored;
  performance harness CPU tests 20 pass/1 ignored.
- Ada regression after the shared host/module changes: selector unit tests
  5/5 pass; the ignored production AUTO prefix/view/graph/raw-bit boundary
  test passes (1/1, 211.74 s). Target Ada rows still choose the Ada N64
  copy-plan and neighboring/misaligned requests remain Legacy.

Next SM120 exact experiment priority is a private full-M/N-tile unmasked
`cp.async` path with the existing masked fallback, paired directly against the
promoted copy-plan and PEDANTIC cuBLAS. Keep production frozen until it wins
correctness, resources, eager+graph ABBA/BAAB screening and long admission.
Then try full-tile `float4` B0 stores and smaller K-loop unroll. Current honest
status: Fixed is improved and one new SM120 vendor win is live, but Fixed is
not yet globally above cuBLAS on either board.

### SM120 exact follow-up experiments after promotion

The private full-M/N unmasked `cp.async` path compiled at 109 registers,
32768 static shared bytes, zero spills and three CTA/SM. Its 176-fixture hot
suite passes exact eager/graph bits, poisoned replay, guards and input
immutability. Compute Sanitizer is clean on all four tools over 14 fixtures.
At B0, 101-window paired medians are 0.9908/0.9922 eager and
0.9954/0.9974 graph versus PEDANTIC cuBLAS, but p95 reaches
1.0015/0.9986 and 1.0028/1.0027. This is a promising near-win, not an
admission: three of four p95 cohorts are still above 1.0. Evidence SHA256
`aecad72103bd0ebfe83dee6bdb239b4d6fd9350872c050a144bb165c4387b8aa`.

The follow-up full-tile `float4` epilogue compiled at 111 registers and kept
zero spills/three CTA/SM, but its 21-window direct ratio against the frozen
unmasked control was 1.0032--1.0046 at p50 in all eager/graph orders. Reject
it for production; it moved in the wrong direction. Keep it only as ignored
negative experimental evidence pending cleanup. Next independent lanes are
B0 CTA M-group swizzle and E0/E1 smaller K-loop unroll.

The smaller K-loop unroll was rejected: it reduced registers 109 -> 108 but
slowed E and B0 by about 4.6--5.3% versus unmasked. The group1/2/4/8/16/32
CTA swizzle screen was valid (group8 reproduced the frozen resource profile)
but produced no group that beat both controls in all eager/graph orders.

The sliced next-stage copy experiment is the B0 winner. It issues A/B copy
pairs before kk=0/8/16/24, commits after the fourth pair, retains the existing
wait-group-zero and CTA barrier at the next slab, advances bases only after all
32 ascending FMAs, and leaves masked fallback/epilogue unchanged. B0/no-bias
101-window ratios versus PEDANTIC cuBLAS are:

- eager ABBA/BAAB p50 0.9760/0.9781, p95 0.9849/0.9846;
- graph ABBA/BAAB p50 0.9813/0.9811, p95 0.9889/0.9874.

It also beats the frozen unmasked control in all four cohorts. Evidence file
SHA256 is `dcb1007489cb7994d300276814268520211328933558623bf902fd070b5ea865`.
The CUDA body uses 113 registers, 32768 static shared bytes, zero spills and
three CTA/SM. Its expanded hot suite has 226 fixture records and passes raw
bits/eager/graph/poison/guards/input immutability. All four Compute Sanitizer
tools are clean over 18 fixtures (racecheck zero hazards/errors/warnings).
Production integration is in progress and must route only B0 without bias;
E0/E1 regress and B1 already has the prior copy-plan winner.

### SM120 Fixed B0 production correction and exact-TMA promotion

The standalone sliced result above did not survive the real production
NVRTC13.2 admission. In the 101-window Fixed harness its worst paired p95 was
1.005009/1.000447 eager and 1.013953/1.013734 graph versus PEDANTIC cuBLAS.
It is therefore retained only as an explicitly forced diagnostic route, with
an empty AUTO admission table. Do not promote it from the standalone result.

The existing deterministic Triad exact scalar-FMA TMA implementation was then
screened as a Fixed bridge. All three one-split exact tiles were bit-identical;
M128xN64/BK16/S2 won the balanced 21-window tournament. Fixed AUTO now selects
that literal physical route (never Triad AUTO) only for B0
`(M,K,N)=(4621,768,2304)` without bias, homogeneous aligned F32,
`ExactScalarFmaV1`, known NVRTC13.2, and the CC12.0/170-SM holder. The prepared
cache key includes the forced physical route, and one-split preparation binds
zero partial/flag workspace. B0 with bias and every neighboring shape retain
their prior fallbacks.

Post-AUTO 101-window paired evidence passes every eager/graph ABBA/BAAB cohort:
worst p95 is 0.894264 versus the prior copy-plan and 0.909777 versus
`CUBLAS_COMPUTE_32F_PEDANTIC`. Raw eager and poisoned CUDA Graph replay bits,
the actual seven-argument Driver ABI, tensor maps, grid/block/shared geometry,
and zero scratch/flag pointers were checked. Evidence:
`internal/perf/sm120-fixed-tma-fma-b0-20260906/sm120-fixed-auto-b0-post101-v1.log`,
SHA256 `20245b1fe378434130f0e74bdfe3117790506afccf34224c27fd516bdf480932`.

This adds a second RTX5090 exact-F32 cuBLAS victory: B0/no-bias joins the
existing B1/bias copy-plan win. Fixed is still the active lane; A/C/D and E,
plus the remaining bias/precision cells and Ada gaps, are not globally closed.

### SM120 Fixed A0 exact-TMA promotion

The same Fixed bridge was screened over the remaining exact-F32 hot rows.
M128xN64 lost on C0 and D0; those rows retain Legacy. M64xN128 won A0
`(M,K,N)=(4621,384,1928)` without bias and is now selected by production AUTO
only for the exact CC12.0/170-SM/NVRTC13.2 holder and aligned homogeneous F32
under `ExactScalarFmaV1`. The physical tile is literal in the prepared-cache
key. A1 is explicitly excluded: the borrowed Triad kernel adds bias before the
FMA chain, while Fixed requires post-dot bias, and the raw-bit screen caught
the mismatch. The public force gate now rejects every biased launch until a
separate post-dot kernel is qualified.

Post-AUTO 101-window evidence passes raw eager/AUTO/Legacy equality, CUDA Graph
replay, actual symbol, tensor-map ABI and all paired eager/graph ABBA/BAAB
cohorts. Worst p95 is 0.895330 versus Legacy and 0.880051 versus
`CUBLAS_COMPUTE_32F_PEDANTIC`. Evidence:
`internal/perf/sm120-fixed-tma-fma-a0-20260906/sm120-fixed-fma-a0-postauto101-v1.log`,
SHA256 `0974cae2ed67c3b1c1b0325c662b882b7d832f09c8c2f7ca6644675e3c5f410c`.

Honest denominator note: this exact-F32 admission is against explicit
PEDANTIC cuBLAS. The separate current-AUTO census against
`CUBLAS_COMPUTE_32F_FAST_TF32` still shows an open gap; do not describe A0 as a
FAST win. Next Fixed implementation lane is a distinct post-dot-bias exact-TMA
variant, followed by renewed FAST-facing work.

### SM120 Fixed A1 post-dot-bias exact-TMA promotion

A distinct deterministic Fixed post-dot-bias mode now exists in the SM120 TMA
scalar-FMA body. It starts each accumulator at zero, performs the same
ascending-k `__fmaf_rn` chain as Legacy, and only then applies
`__fmul_rn(alpha, acc)` followed by `__fadd_rn(scaled, bias)`. This is separate
from the original Triad-compatible bias-seeded mode in the route digest,
numeric contract, prepared-cache identity and public forced rung. The Driver
ABI remains seven arguments / 32 bytes. Host validation restricts this route
to NN, one split, beta zero, alpha one, and a non-null aligned bias pointer.

The two new physical exports are M128xN64 and M64xN128. Their PTX/module
admission requires exact FMA and exact add instructions, preserves the existing
register/resource caps, and rejects local memory, MMA/WMMA, TF32 conversions,
atomics and reductions. Existing exports retain their original arithmetic.

For A1 `(M,K,N)=(4621,384,1928)` with bias, M128xN64 won the long pre-AUTO
screen and production AUTO now selects that literal symbol only on the known
CC12.0/170-SM/NVRTC13.2 holder with aligned homogeneous F32 under the Fixed
scalar-FMA contract. A0/no-bias and B0/no-bias mappings are unchanged; all
other requests retain their prior fallbacks.

The post-AUTO 101-window replay passes raw eager/AUTO/Legacy equality, CUDA
Graph replay, actual symbol and seven-argument graph ABI, and every paired
eager/graph ABBA/BAAB cohort. Worst p95 is 0.914063 versus Legacy and 0.765326
versus `CUBLAS_COMPUTE_32F_PEDANTIC`. Evidence:
`internal/perf/sm120-fixed-tma-fma-a1-postbias-20260906/sm120-fixed-fma-a1-postbias-postauto101-v1.log`,
SHA256 `7af2dd4afc99448ec530102f2b7aaec771bbf9833dbd97c033de8ef91d561ac4`.
The pre-AUTO 101-window artifact is
`sm120-fixed-fma-a1-postbias-m128-admission101-v1.log`, SHA256
`8887b5a16c246a07d55a47a3b442642bbb53d53ec75565098e9acff6f53ac7f4`.

This is an exact-F32 PEDANTIC win, not yet a FAST-TF32 win. Measure
`CUBLAS_COMPUTE_32F_FAST_TF32` separately after the A1 integration regression;
do not infer FAST victory from these ratios.

### SM120 Fixed A1 postbias FAST-facing follow-up

The postbias implementation was isolated into the Fixed-only
`kernels/gemm_bi_fixed/sm120_f32_postbias.cu` fragment so its experiments do
not move the frozen Triad SM120 compositions. The hardened comparator measures
explicit PEDANTIC and FAST_TF32 independently and validates both vendor graph
inventories plus poisoned replay.

Three hypotheses were screened on the live RTX5090. Reducing shared alignment
1024 -> 128 did not change Driver occupancy: the cubin still accounts the 1 KiB
allocation granularity and both control symbols remain at three CTAs/SM, so the
source was restored without timing it. M128N96/256 and the M128N64 BK16
chunk-by-four runtime-loop twin retained exact bits but were slower than the
current A1 AUTO; they remain force-only negative evidence.

The M128N64/T256 twin (WM32/WN32, TM4/TN8, eight compute warps) is a genuine
own-kernel winner and passes the strict live gate of <=85 registers, zero local
memory and at least three CTAs/SM with 24,592 dynamic shared bytes. Its fresh
101-window results are worst p95 0.930242 eager and 0.919701 graph versus the
prior A1 AUTO, 0.856751 versus Legacy, and 0.706538 versus PEDANTIC cuBLAS.
Raw eager bits, poisoned graph replay, actual symbol and seven-argument Driver
ABI all pass. It still loses explicit FAST_TF32: worst p95 1.265263. Evidence:
`internal/perf/sm120-fixed-tma-fma-a1-postbias-20260906/sm120-fixed-a1-t256-fast101-v1.log`,
SHA256 `5c68b8ec54f699a0399425b9c3f6b975830f0b0195587fa21d2f73a7b63e7030`.
Promote T256 only after the independent integration review, then continue the
FAST-facing schedule from that stronger production control.

### RTX5090 Fixed multi-CUDA census and Triad baseline (2026-09-06)

Fixed inference was rebuilt and exercised on the same RTX5090 under installed
CUDA 12.8.93, 13.0.88 and 13.2.78 toolkits. The production graph smoke passes
all three toolkits after making the postbias tensor-map ABI validator explicit
for the CUDA-major layout (CUDA 12: alignment 64/layout end 360; CUDA 13:
alignment 128/layout end 424). The force inventory is now compiler-exhaustive
and reverse-checked: SM120 exact 14 variants, SM89 exact 3, SM120 half 11,
SM89 half 7, SM120 mixed half-to-F32 9, SM89 mixed 4, SM120 TF32 12 and SM89
TF32 6. Every runnable force tile records its actual graph symbol before any
numeric rejection. AUTO admission remains benchmark-qualified and therefore
separate from physical holder admission.

The full CUDA 13.2/13.0/12.8 Fixed census covers BF16, F16, BF16-to-F32,
F16-to-F32, TF32, exact F32 versus PEDANTIC, and exact F32 versus explicit
`CUBLAS_COMPUTE_32F_FAST_TF32`, A--E, both bias states, eager and graph. There
were no launch or bit-identity failures. CUDA 13.2 production AUTO promotions
that survived 101-window confirmation are TF32 A0/A1 to
`Tf32Sm120M128S2`, exact B1 to `PostBiasM128N64`, exact C1 to
`PostBiasM128N96`, and the previously qualified TF32 D0/D1 PairStore route.
Exact F32 still loses explicit FAST_TF32 in every A--E/bias cell; that is the
remaining arithmetic-contract/silicon gap and must not be described as a FAST
win. Commits `7a07ad16` and `3304a29a` contain the hardened census,
dispatcher and qualified promotions.

Before the temporary board expired, the current production Triad dispatcher
received a broad CUDA 13.2 baseline over five projection shapes, NN/TN/NT,
exact F32, AllowTF32, BF16 and F16. AUTO produced 120/120 successful records
(60 cells, eager plus graph, 21 windows, AB order); all eager/graph identity
checks passed and every actual physical node/symbol/geometry is recorded.
The separate vendor run produced 90/90 successful FAST/PEDANTIC records over
45 cells. The current production artifact is
`1cbfd2318610ff5e105eed18ec269453246fef30ca23ac94ac4d3378dfe89246`.

These generic ratios compare independently measured quantiles, not paired
windows; the vendor harness is eager-only, so graph/vendor ratios are only a
proxy and are not graph-versus-graph admission evidence. Eager p95 wins out of
15 versus FAST/PEDANTIC are: exact F32 1/11, AllowTF32 4/11, BF16 6/12 and
F16 6/12. The graph/eager-vendor proxy counts are exact F32 2/11, AllowTF32
5/11, BF16 11/15 and F16 11/15. Worst eager/FAST cells are BF16 NT
`d128_out_proj` 3.144797x, F16 at the same cell 3.074104x, exact F32 TN
`d128_in_proj` 1.892838x and AllowTF32 at the same cell 1.895693x. Best
eager/FAST cells are BF16/F16 NT `d768_out_proj` at 0.709505x/0.706644x and
exact/AllowTF32 NN `d128_in_proj` at 0.808046x/0.808109x.

Evidence is under `internal/perf/sm120-triad-current21-20260906/`:
`generic-matrix.md` is the complete 120-row human table,
`generic-matrix.json` retains full precision and physical nodes, and
`SHA256SUMS` verifies both source logs, the runner and both matrix files. Half precision
uses physical SM120 TMA kernels in all 30 cells. AllowTF32 uses SM120 TF32 TMA
only for NN `d768_in_proj`, NN `prism_in_proj` (M64N128), and NN
`d768_out_proj` (M64N64); d128 NN still uses scalar split-K/reduce, while all
TN/NT AllowTF32 cells retain exact-FMA/scalar routes. Those fallbacks are the
first Triad optimization targets. A strict paired TF32 run was started against
the same artifact and private cache. The rental nearly expired before its
third cell completed, so the first 16/24 records are preserved in
`cuda-13.2/paired-partial/`. Both completed cells are strict all-four FAST
wins: `d768_in_proj` AUTO/forced worst p95 0.902121/0.902750, and
`d768_out_proj` AUTO/forced worst p95 0.928441/0.927627. Their actual forced
symbols are respectively the SM120 M64N128 and M64N64 TF32 TMA tiles. All
numeric/guard/AUTO-versus-forced/eager-versus-graph raw-bit checks passed.
These remain 21-window screening results with `dispatch_admission=false`, not
long dispatcher admissions; the third cell and long confirmation remain open.

### Ada resumption: missing compatible AUTO rows found (2026-09-06)

The current tree was synced from `318b3fbd` to `/root/mamba-rs-triad` without
deleting remote-only files; all 394 tracked regular files were hash-verified.
Ada is CC8.9 / 142 SMs, NVRTC13.2, driver 595.45.04. The first exact Fixed
smoke covers both PEDANTIC and FAST_TF32, A--E, both biases, all three exact
force tiles and eager/graph: 240 records, zero rejected, all applicable
repeat/storage/graph gates passed. Actual AUTO was Legacy at A/C/D and the
SM89 copy-plan at B/E. Evidence:
`internal/perf/fixed-smoke-ada-20260906T125444Z-318b3fbd09c8/`.

The missing A/D acceleration is not solely an architectural limitation.
Although the recent SM120 TMA kernels cannot run on Ada, an already loaded,
compatible SM89 copy-plan beats Legacy at A and D. The AUTO allowlist had
only B/E. Fresh production 101-window qualification of A/D, both biases,
both orders and eager/graph passed all 32 paired p50 and p95 comparisons
against incumbent AUTO: p50 0.830110--0.846557, p95 0.832349--0.854944.
PEDANTIC is beaten in all 16 corresponding cohorts (worst p95 0.914202);
FAST_TF32 is still faster than this exact-F32 route (worst p95 2.118521).
Evidence:
`internal/perf/fixed-exact-promotions-ada-20260906-318b3fbd09c8/`.

C's existing F32N128S2 wins all 16 medians but only 14/16 p95 cohorts in
the 21-window screen; its worst paired p95 versus AUTO is 1.022126.
It is not promoted. The A/D dispatch change adds the two shapes with both
biases to the existing CUDA13.2/CC8.9/142-SM allowlist, updates admission
support and expands prefix/view/graph tests; CUDA bodies stay unchanged.
New selector and admission regression tests both failed before the change,
then passed (6 selector tests and 16 support tests). Independent review found
no blocker. Post-AUTO GPU qualification from the isolated source
`/root/mamba-ada-dispatch-review-gxJWgz` completed successfully: all 32 A/D
101-window records capture AUTO `F32Sm89N64CopyPlan` and its actual graph
symbol, with Legacy as the forced control. Legacy/AUTO paired p50 is
1.181364--1.204765 in every cohort; AUTO beats PEDANTIC in all 16 p50/p95
cohorts (worst p95 0.91652), while exact-versus-FAST remains open. B/E
regression21 passes all 32 records with the original copy-plan route intact.
The expanded prefix/view/misalignment eager/poisoned-graph test passes its
704 boundary records (134.60 s), including A's partial M tile. Library tests,
40 nonignored performance-harness tests and formatting pass. Actual source
hashes and all GPU logs are preserved in
`internal/perf/fixed-sm89-ad-copyplan-postauto-20260906/`. A/D promotion is
therefore verified through production AUTO, not only through forced timing.

Ada Triad baseline also completed: 120 smoke records, 120 AUTO records at
21 windows, 90 vendor records, all applicable eager/graph bit checks passed.
Actual TF32 and half routes use the portable tensor-core family. Independent
eager p95 comparisons win FAST/PEDANTIC in respectively 0/0 of 15 exact-F32
cells, 0/11 TF32, 1/15 BF16, and 2/13 F16. The generic vendor harness is
eager-only, so graph ratios remain proxies, not paired graph evidence.
The complete table, symbols and source artifacts are saved in
`internal/perf/ada-triad-current21-20260906/`.

The owner's clarified priority is deterministic TF32 as the flagship mode,
especially for the planned RL use of Triad. BF16/F16/mixed and exact F32
remain supported performance targets. Finish the already qualified A/D
dispatch fix, then prioritize Ada TF32 Fixed and Triad NN/TN/NT existing
candidate qualification. A broad claim that every compatible winner on every
architecture is already in AUTO would still be false: reachability, numerical
qualification, AUTO promotion and measured speed must be tracked separately.

TF32 inventory audit found an existing force-only Fixed wide tile,
`Tf32M128N128S3`, with strong historical finite-input results in
`internal/perf/fixed-wide-rungs-20260905.log` (11-window eager, not a current
admission). Keep it in the next full TF32 screen, but its exceptional-value
conversion contract blocks a silent Fixed AUTO promotion: it uses
`RegisterAddHalfUlpTf32V1`, while ordinary Fixed TF32 uses
`RegisterCvtRnaTf32F32V1`. Existing `gemm_bi_fixed_correctness.rs` tests compare
exceptional values only within the wide route and cross-route bits only for
finite inputs. Finite performance wins do not prove the full Fixed rung/prefix
bit contract. Preserve this distinction; qualify a compatible conversion
implementation before using this wide body as an interchangeable Fixed rung.

### Recovered RNA-compatible Fixed candidate (2026-09-06)

Before the new RNA integration, the existing ordinary Fixed TF32 C0/C1
promotion is complete and independently reviewed: exact `(4621,1928,384)`,
CC8.9/142SM/knownNVRTC13.2 now chooses M64S2 instead of M128S2. All other
shapes/toolkits/architectures keep their prior routes. Ada selector4/4 and
the actualAUTO special-bias/first-row-prefix/graph test pass. Post-AUTO101
captures M64S2 in all8bias/path/order records with0rejections; old/new paired
p50 is1.036022--1.042052, p95 is1.038318--1.045031. Every required rawbit gate
passes. It remains an internal win: newAUTO/FAST median1.912826--2.121482.
Evidence `internal/perf/fixed-tf32-c-postauto-ada-20260906/`, performance log
SHA256 `1758732b2150c0b668c3051e85f6085dfb682403a9deea275d2434392ec65471`.
The prepromotion all6TF32 screen and C101 are in
`internal/perf/fixed-tf32-all6-ada-20260906/`; noCUDA bodies changed.
An uncached repeat (Driver cache disabled, persistent kernel caching declined
because its empty directory had mode0755) independently confirms all8wins
with identical loaded artifacts: old/new paired p50 1.026771--1.032084,
p95 1.029164--1.033931. Preserve the truthful cache-mode distinction in
`internal/perf/fixed-tf32-c-postauto-privatecache-ada-20260906/README.md`;
raw log SHA256 `bb74e5e4bdcdfedf54439496e3c5caca0426351c1dfea9149ea3b8e32d38851a`.

Do not recreate the already completed standalone experiment:
`internal/experiments/sm89-nn-wide-rna-compatible.cu`, SHA256
`c0c9eb735374620eaf8a023345ee86359af9f748df53fc5cf7d07ee7051f65c4`.
It preserves the wide pipeline and uses explicit `cvt.rna.tf32.f32`.
`internal/experiments/fixed-rna-production-integration.md` documents the
Fixed-owned isolated fragment/holder/ABI design; its historical line numbers
need to be interpreted against the live tree.

Saved NVCC graph-only 21-window evidence is under
`internal/perf/nn-rna-probe-20260905/` and
`internal/perf/nn-rna-fastfinite-probe-20260905/`. A1 RNA/FAST paired p95 was
0.955720; B1 was 1.062197. These are experimental results, not current
production NVRTC/AUTO admission. The fastfinite experiment was slower than
explicit RNA (A1 paired median ratio 1.288064), so keep it as a documented
loser, not the next production candidate. Its same-run log SHA256 is
`7c326e4f7f5b47c7971614867cb69fd12eef6295fc09b209fdcfb84e2315c103`.

The saved explicit-RNA probe passed 448 raw conversion cases, 175 GEMM
contract records and 7 numeric validations, including exceptional A/B/bias,
prefixes/subviews and graph/repeat bits. It covered only three of the five
ordinary Fixed rungs and is not a substitute for live Fixed-owned NVRTC
resource/ABI/capture and complete cross-rung/view qualification. Integrate
force-only first; promote measured cells after same-build 21/101-window
eager/graph comparisons against current AUTO and explicit cuBLAS FAST.

### Ada Fixed production census checkpoint (2026-09-06 14:10 UTC)

Commit `26b44c4d` contains the qualified TF32 C0/C1 AUTO promotion. A further
101-window confirmation with active mode-0700 private caching and Driver
caching disabled passes all eight bias/path/order records. Old/new paired
p50 is 1.025144--1.035193 and p95 is 1.026725--1.037257, with identical loaded
artifacts. This remains an internal improvement, not a FAST victory. See
`internal/perf/fixed-tf32-c-postauto-privatecache-v2-ada-20260906/README.md`.

The complete already-shipped half/mixed force inventory has now been screened:
22 candidate specifications across A-E, both bias states, eager/graph and two
timing orders. Of 880 planned timings, 800 passed; 20 Legacy+bias attempts were
rejected before timing by cross-AUTO bit identity, excluding four timings
each. All 400 applicable graph replay checks pass. These rejections are not
run-to-run nondeterminism or broken AUTO cells: the Legacy post-dot bias
arithmetic differs from the bias-seeded native tensor-core family, and AUTO
does not select Legacy for these four supported input/output dtype triples.

No non-incumbent compatible candidate wins both paired p50 and p95 in every
cohort. Actual AUTO robustly beats native half cuBLAS in 14/40 shape-bias-dtype
cells: BF16 7/10, F16 4/10, BF16->F32 2/10, F16->F32 1/10. The comparator is
explicit `CUBLAS_COMPUTE_32F` for half inputs, not PEDANTIC or F32 FAST_TF32.
The remaining largest gaps are mixed B/D/E and homogeneous-half B/D0.
Full samples, source identities, all candidate/cell summaries and rejection
analysis are in `internal/perf/fixed-half-mixed-all-ada-20260906/`.
Standalone RNA and half-swizzle prototypes are not included in this shipped
inventory result; do not claim every experiment is already in AUTO.

Current work remains Ada Fixed/inference, prioritizing explicit-RNA wide
TF32 integration. The new symbol is force-only until current production
NVRTC correctness, physical ABI/resource/capture gates and paired 21/101
performance justify a promotion. Frozen Triad and CC12 source compositions
must remain unchanged. Ada and expired-5090 Triad baselines remain saved;
Triad optimization is not yet the active implementation stage.

### Ada Fixed RNA-wide production qualification (2026-09-06)

The explicit-RNA wide candidate is now integrated as the distinct
`Tf32RnaM128N128S3` force route in the Fixed sm89 module, with strict optional
PTX/Driver/resource admission. The original Triad wide route is unchanged.
The new symbol uses a five-argument/32-byte bundle, block 256, 98,304 bytes
dynamic shared, 153 registers and no local memory/spills. Composed Triad and
CC12 source bytes are retained; no claim of unmeasured artifact equality.

Reviewed GPU gates cover all five ordinary TF32 rungs, finite/exceptional
inputs, both biases, actual hot-A M4620/4621/4622, underfill M6016/6017/6018,
all-prefix cross-route bits, aligned row views/C4 output, repeat/poisoned graph,
K0 and empty/no-op/unsafe input rejection. The eager-only benchmark filter now
also verifies real physical function/ABI/geometry before timing.

Production 21-window and final 101-window screens each pass all 40 records
(A-E, bias0/1, eager/graph, two orders), no rejections. Every cohort wins
against incumbent AUTO at paired p50 and p95. A1 and B1 also win against
explicit cuBLAS FAST at both quantiles in every cohort. Final worst p95
RNA/FAST is 0.887512 for A1 and 0.976818 for B1; remaining cells still lose
FAST. This is not all-precision inference closure.

Evidence: `internal/perf/ada-rna-wide-force-20260906/README.md` and its
integration report, raw logs and manifest. Final 101 log SHA256
`4983d29fa241538f170e087372c70baa92858033a7b075c7e53945088ba6d910`;
frozen test binary SHA256
`db3dbafb1b1ccf3c0ea422d2860e8a7789e1eca9b9c14f51480711f18ac5dde1`.

Next bounded task: promote the ten measured Ada/142-SM/known-NVRTC13.2
shape-bias cells through a pre-launch availability/alignment guard, leaving
the old picker unchanged on decline. Increment routing revision 39->40;
preserve CUDA sources/identities and unrelated Triad/SM120 qualification
selectors and hashes. Actual AUTO boundary/graph/bits and post-AUTO paired
confirmation remain required. Do not remove any superseded route: it remains
an ordinary fallback or a candidate for other architectures/toolkits.

### Ada Fixed RNA-wide AUTO checkpoint (2026-09-06)

The ten measured A-E/bias0,1 cells now select the Fixed-owned RNA-wide symbol
through actual AUTO on CC8.9/142SM/known NVRTC13.2 under deterministic TF32.
The guard also requires the admitted holder, homogeneous F32, C/A/B16 and
optional bias4. Every decline keeps the old picker unchanged. Global routing
epoch39->40 invalidates old graphs; CUDA/compiler/composer/numeric/schedule
identities and unrelated Triad/SM120 selectors are unchanged. All three Ada
cache blobs match the frozen force checkpoint byte-for-byte by main's hashes.

Quiet post-AUTO101:40/40 unique records, zero rejections, exact actual RNA
versus forced old M64S2 graph symbols/geometry, all raw/repeat/applicable graph
bits pass. Samplewise actual AUTO/old worst paired p95 ranges
0.626263--0.913177: all ten own wins. Actual AUTO/FAST robust wins remain
A1 and B1 only, worst paired p95 respectively0.904797 and0.980568. Remaining
eight cases still lose FAST; worst ratio1.342478. This is not all-inference
closure and is not an extrapolation to CUDA12.8/13.0 or another physical GPU.

Evidence: `internal/perf/ada-rna-wide-auto-20260906/README.md`, raw logs,
summary, integration report and manifest. Final101 log SHA256
`3dbb8754b651706d67c83ab1f96fb26ed66a63bc161e8ced015f36ef50831f46`;
frozen performance binary SHA256
`2dfbc274dfc1773939491fe2fbc0eb86d6536c7bb6f380720b448898d1cc68e2`.
Main independently verified all records and nine local/remote source hashes,
reran633 library tests and actual hot-A AUTO/graph1/1. Full AUTO GPU matrix2/2,
retained C prefix1/1 and both actual Ada Triad cohorts pass. Static review and
the subsequent test-fixture amendment review are clean.

Full architecture run:56 passed/1 baseline failure. All actual compile,
assembly and resource gates passed; the lone obsolete PairStore enum-absence
assertion also fails at82fc. It was corrected without touching production;
focused1/1 GREEN followed. Do not claim a new full57/57 rerun. Other stale
revision/composition fixtures were corrected and full633 lib tests then pass.
The first post-AUTO timing overlapped host compilation and is retained as
explicitly inadmissible, never used in performance conclusions.

Next bounded work: matching-build RNA census on installed CUDA12.8 and
side-by-side CUDA13.0, then full exceptional/prefix/view qualification before
any toolkit AUTO widening. Forced RNA has no toolkit lock, but current AUTO
admission is13.2-only. Existing forced tests cover tail/hot-A; B-E exceptional
coverage is currently tied to the13.2 actual-AUTO helper and must be decoupled
before promoting those toolkits. The half-swizzle prototype remains the next
separate kernel integration candidate. Preserve all old routes and evidence.

### Ada RNA-wide CUDA12.8/13.0 census checkpoint (2026-09-06)

Completed on frozen208c740c with distinct matching builds, source/targets and
private0700 caches. CUDA13.0 was installed side-by-side; no GPU driver upgrade,
package removal or default CUDA13.2 change. Both builds and cold/warm force
correctness pairs pass2/2. The first12.8 build was a preserved sync-omission
failure, then passed after supplying the required test-support file.

Both21/101 runs per toolkit complete40/40 unique records with zero rejection.
Main independently verified all160 raw records, exact physical RNA graph,
samplewise quantiles, repeat/raw/applicable graph bits, matching NVRTC domains
and explicit FAST/bias denominator;167 local/remote source hashes,4 live
binary hashes and6 cache blobs were checked. Production source is unchanged.

At101 windows all ten shape/bias cells beat current ordinary AUTO for both
toolkits. FAST wins are only A+bias: worst paired p950.955545 on12.8 and
0.973033 on13.0. B+bias still loses FAST at1.063045/1.063894; worst remaining
FAST ratio is E without bias at1.496884/1.498848. CUDA13.2's separate B+bias
win cannot be extrapolated to these builds. Evidence and manifest:
`internal/perf/ada-rna-toolkit-census-20260906/README.md`.

This census does not promote AUTO. The frozen force helper's B–E exceptional/
prefix/views were coupled to13.2 actual-AUTO assumptions. A test-only task
now separates that corpus and will rerun full qualification on all three
toolkits before the separate dispatch/epoch/post-AUTO confirmation step.
Keep all old candidates and frozen evidence; no cleanup or other-architecture
qualification was performed here. Half/mixed and Triad gaps remain open.

### Ada RNA full-corpus three-toolkit proof (2026-09-06)

The known force-test gap is closed by a one-test-file change. Force mode now
uses the full tail/A–E corpus and the same C4/C16/prefix/row-view matrix as
the13.2 AUTO wrapper, without inheriting its AUTO assertions. The intended
host regression first failed for the old two-family force list, then passed.

Matching12.8/13.0 cold+warm force tests each pass2/2 with exactly448 unique
view groups.13.2 force2/2 and actual-AUTO1/1 each pass448 groups. Full five-
rung raw-bit comparisons, finite/exceptional inputs, bias0/1, guarded views,
two eager repeats and poisoned graph replays are retained; K0/empty/unsafe
inputs and immutable-input checks remain. Main independently checked all
raw groups against the expected set,167 local/remote sources,21 raw hashes
and9 byte-identical matching-toolkit cache blobs, then reran13.0 force2/2
with448 groups in28.02s. Static source review found no issues.

Evidence: `internal/perf/ada-rna-full-toolkit-qualification-20260906/README.md`.
No production CUDA/dispatcher/epoch changes or new performance measurements
in this task. The next separate task is qualified12.8/13.0 RNA AUTO promotion
and post-AUTO confirmation. Preserve the ordinary C fallback distinction:
M128S2 on12.8/13.0, M64S2 on13.2; A/B/D/E ordinary routes areM64S2 throughout.

### Ada RNA AUTO on all three qualified toolkits (2026-09-06)

The full-corpus prerequisite is committed atc3564636. Actual Fixed AUTO now
admits the same ten A-E/bias0,1 cells on known NVRTC12.8,13.0 and13.2,
CC8.9/142SM, deterministic TF32 policy, homogeneousF32, admitted holder,
C/A/B16 and optional bias4. The only production changes are this finite
toolkit membership and global routing epoch40->41. Every other decline and
ordinary picker is unchanged; old graphs must recapture. Revision38/39
rejection tests are retained and revision40 rejection is added.

Final matching builds pass634 library tests and43 performance-static tests
on each toolkit. Actual-AUTO two-wrapper GPU checks pass2/2 with exactly448
expected unique groups per toolkit: full five-rung raw bits, prefixes/views,
finite/exceptional values, bias0/1, guards, repeats and poisoned graphs. Both
existing13.2 Ada Triad cohorts and C-prefix retention pass. Main independently
reran13.0 actualAUTO2/2,448groups in38.69s. Initial full12.8 library RED exposed
two stale current-epoch fixtures; these were corrected, not waived. Earlier
pre-amend build logs are retained and distinguished from final binaries.

Quiet post-AUTO101 completes exactly40 unique records per toolkit, zero
rejections, after all builds/functional checks finished. The correct old
comparator is M128S2 for C on12.8/13.0; M64S2 for all other cells/toolkits.
Main and independent reviewer verified physical AUTO/old graph geometry,
epoch41, known matching NVRTC, FAST with timed bias, all bit flags and the
samplewise ratios. Worst paired quantiles are over all four path/order cohorts.

| CUDA | Own p50+p95 wins | Range of worst p95 AUTO/old across10cells | FAST wins | Worst p95 AUTO/FAST in winning cells |
| --- | ---: | ---: | --- | --- |
|12.8|10/10|0.648678--0.972474|A1 only|A1:0.956254|
|13.0|10/10|0.654364--0.992156|A1 only|A1:0.975619|
|13.2|10/10|0.629951--0.918121|A1,B1|A1:0.908820; B1:0.982497|

FAST losses remain9/10,9/10 and8/10 respectively. Worst remaining ratios are
E0:1.479948,1.487687 and1.345904.13.0 D1's internal p95 margin is only0.8%;
do not describe it as a large gain. This is one TF32 inference promotion,
not all-precision inference or Triad closure and not fresh5090 qualification.

Evidence: `internal/perf/ada-rna-toolkit-auto-20260906/README.md`, raw logs,
worker and main summaries, independent review, executable validation helpers
and manifest. Final168-input source manifest SHA256
`21f828115c983114175615bf6b6c044c461b811d3cc11aad787697dff3f7a574`;
main checked all168 local/remote hashes,162 unchanged inputs againstc3564636,
six live correctness/performance binaries and nine byte-identical matching
cache blobs. CUDA/composer/loader/ABI/numeric/schedule and compiled module
identities are unchanged. No full architecture suite or clippy rerun is
claimed for this host-only task. No kernel was deleted or made unreachable.

Next separate kernel work remains the half-swizzle/mixed-output follow-up
and TF32 FAST gaps, then Triad NN/TN/NT. Keep the saved5090 baseline. Local
read-only notes describe possible RNA tile-geometry follow-ups, not measured
or admitted candidates; no new tile was implemented in this checkpoint.

### Precision and determinism terminology

TF32 is a compute mode for F32 tensors; it is not a distinct tensor storage
dtype here. Its reduced multiplication precision differs from exact F32 even
when both implementations are bit-reproducible. The public
`AllowDeterministicTf32V1` policy permits qualified TF32 routes and falls back
to the exact-F32 family for unqualified requests (`context.rs` and
`gemm_bi_triad/dispatch.rs`). Half typed Triad calls prefer eligible native
tensor-core/typed routes, then use an exact-F32 upcast fallback. NN and dX
round back to the requested BF16/F16 output; dW is F32 by design (`blas.rs`).
These are kernel-route fallbacks, not an automatic BF16-to-F16 switch.

Do not equate cuBLAS FAST or TF32 with nondeterminism. NVIDIA documents
bitwise repeatability under specified toolkit/device/stream/workspace
conditions: https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
Batch invariance, reproducibility after changing a physical route, and
cross-toolkit/cross-architecture equality are distinct guarantees. Claim only
the numerical contract and configuration coverage demonstrated by tests;
deterministic GEMMs alone do not prove full RL-training reproducibility.

## Ada TF32 NT A+B ldmatrix closure (2026-09-08)

The isolated d768-in A+B ldmatrix candidate passes exact bits, guards and the
resource gate (106regs, local0, occupancy2), but loses the retained A-only
candidate by2.0–2.1% in all paired eager/graph once7 cohorts. It remains
1.72–1.75x slower than native cuBLAS Fast. This is a decisive valid loss:
do not retry or integrate it. Retain A-only (~200us,11.3–11.4% faster than
actual AUTO). Full evidence and frozen identities are in
`internal/perf/ada-triad-tf32-nt-compact-ab-ldmatrix-20260908/report.md`.

## Ada half TN regpipe+vec2 sibling screen (2026-09-08)

The repaired four-cell CUDA13.2 screen adds one retained-best result: BF16
d768-out is1.6–1.8% faster than compact BK64/S2 with every paired p50+p95
below.99. It is still1.068–1.111x cuBLAS Fast at p50. F16 d768-out misses one
retained p95 (.99124); F16/BF16 Prism also fail the strict retained gate.
Stop those three without retry. No half-TN Fast win is claimed. Full evidence:
`internal/perf/ada-triad-half-tn-regpipe-vec2-siblings-20260908/report.md`.

## Ada exact-F32 TN CopyPlan correctness stop (2026-09-08)

Do not time or integrate the full-K d768-in transpose+CopyPlan+epilogue probe.
It differs from actual AUTO at target word0 (`d76bd726` vs `d76bd729`). AUTO is
the deterministic two-chunk SplitM route with an FP64 reducer; the candidate is
one full-length F32 chain. Preserve the failed source/evidence, but the next
candidate must reproduce two1024-sample raw partials and reuse the exact reducer.
See `internal/perf/ada-triad-f32-tn-copyplan-d768-in-20260908/report.md`.

## Ada exact-F32 TN SplitM CopyPlan retained winner (2026-09-08)

The corrected d768-in candidate preserves AUTO's two1024-row raw partials and
unchanged FP64 reducer while replacing each partial body with the production
exact-F32 CopyPlan. It passes raw-partial and final target/tail/exceptional/
non-unit/K0 bits in eager and graph paths. Candidate/AUTO p50 is
`.84635-.85354`, worst p95 `.85527`: retain a14.5-15.4% latency reduction.
Candidate/Fast remains`2.581-2.596x`, so this is not a Fast win. Frozen source,
binary and authoritative attempt3 evidence are in
`internal/perf/ada-triad-f32-tn-splitm-copyplan-20260908/`; commit `3285af45`.
The first K0 attempt used the wrong non-unit alpha for a public API that fixes
alpha=beta=1, and is preserved as a harness failure rather than a kernel loss.

The next exact-F32 TN screens are d768-out `m_chunk=512,chunks=4` and Prism
`m_chunk=784,chunks=6`. Prism requires a padded transpose stride4624 to enter
the CopyPlan async path, and its raw-partial gate must run before timing because
BK32 adds padded zero-FMAs relative to AUTO's BK16 body. The longer-term
source-backed order is direct-TN staging, fused final chunk+FP64 finalize,
GROUP_M=12, then a 64x32 tile; see
`internal/perf/ada-triad-f32-tn-exact-research-20260908.md`.

Those sibling screens are now complete. d768-out is a new retained-best:
candidate/AUTO p50 `.88484-.90297`, worst p95 `.90506`, exact across all raw
partials and final gates. It remains`2.71-2.77x` Fast. Prism is also exact but
loses AUTO by34.8-36.4% and stops without integration. Frozen evidence is in
`internal/perf/ada-triad-f32-tn-splitm-copyplan-siblings-20260908/`; the next
Prism mechanism, if revisited, is direct/tail-safe BK16 rather than CopyPlan.

The next d768-in direct-TN candidate is also complete and stops. It is fully
exact, uses one grid.z=2 direct-partial node plus the old reducer, and lowers
kernel registers to107, but only ties retained eager and loses retained graph
by0.67-0.72%; candidate/Fast remains2.58-2.61x. Preserve retained transpose+
CopyPlan. Frozen evidence:
`internal/perf/ada-triad-f32-tn-direct-copyplan-20260908/`. Next candidate is
retained CopyPlan chunk0 plus a chunk1 body with the identical ordered FP64
finalize, to remove partial1 traffic and the standalone reducer.

That fused-finalize candidate is now a new retained-best winner on d768-in.
It is exact across both raw planes, exceptional/NaN payloads, tails, non-unit
alpha, K0, eager/graph,20-op accumulation and guards.  Candidate/previous
retained p50 is `.93356-.93661`, worst p95 `.93782`: a `6.2-6.6%` latency
reduction in every paired once7 stratum.  It uses129 registers,32KiB shared,
local0 and occupancy3.  Candidate/Fast remains`2.42657-2.43007x` at p50, so
this is not a Fast win.  Frozen evidence is in
`internal/perf/ada-triad-f32-tn-fused-finalize-20260908/`.  Keep it as the
exact-F32 TN d768-in shortlist leader; next screen is `GROUP_M=12`, then the
64x32 occupancy4 geometry if the resource thresholds are met.  Production
dispatcher integration waits until this focused shortlist is complete.

The `GROUP_M=12` screen is complete and is a valid exact loss.  Both compute
nodes used G12, all raw/final/exceptional/tail/K0/graph/guard gates passed, and
resources stayed identical to G8, but candidate/G8 p50 was
`1.00155-1.00201` with worst p95 `1.00338`.  Fast was intentionally not run
after the retained gate failed.  Stop without retry and keep G8.  Frozen
evidence is in
`internal/perf/ada-triad-f32-tn-fused-groupm12-20260908/`.  Next exact-F32 TN
screen is64x32x32/S2, gated before timing on <=128regs, <=25KiB shared,
occupancy4 and local0.

## Ada half-swizzle production force checkpoint (2026-09-06)

The existing homogeneous BF16/F16 XOR-staging prototype is now reachable through
the production loader and `FixedTile::Tc128Sm89Swizzle` forced dispatcher. Exact
exports are `gemm_bi_nn_fixed_sm89_tc128_swizzle_v1_{bf16,f16}`. This is a verified
force checkpoint, not AUTO promotion or inference completion. Revision remains41;
the incumbent pipeline and every ordinary route remain available.

Matching CUDA12.8/13.0/13.2 each pass release636 library/43 static performance
tests, the SM89 compile gate, cold/warm holder proof, full forced/rounding/hotA-E
bit corpora, eager-only physical identity and retained RNA AUTO448unique groups.
The six extra13.2 retained gates cover both TriadTF32 cohorts, ordinary TF32C,
old halfAUTO, and exact-N64 force/AUTO. Allfour sanitizer tools pass on allthree
artifacts. Sanitizer CUDA API reporting is disabled for deliberate terminal
arity-negative probes; device memory/race/init/sync checks remain enabled with
error-exitcode99. The original unsuppressed99API-probe diagnostic is preserved.

Actual swizzle resources: block256, dynamic shared69632, local/staticshared0,
occupancy1, registers177/177/180 by toolkit. Main independently reran13.0 hotA-E
(1pass,11.27s) and live holder/cache proof (1pass,2.94s), reconciled174 local/remote
inputs,14release binaries and9cache blobs. Allsix Triad blobs match the prior
checkpoint. Fixed source has26fragments/688275bytes, SHA
`d1180079f0067afbc3f54210ce41a80dbc6bb422d53085824082089a0ff97f2a`.
Header-manifest identity also changes because it includes source macro analysis;
do not carry forward the old RNA header digest into new timing controls.

Evidence and exact commands: `internal/perf/ada-half-swizzle-force-20260906/`.
The original four `internal/experiments/sm89-fixed-half-swizzle*` files are retained
as provenance, not shipped tests or new AUTO admissions. No kernel was deleted.

Continue immediately with the paired21/101-window homogeneous-half census on all
three toolkits. Include old pipeline on12.8/13.0, whose AUTO is still Tc128;13.2
AUTO is pipeline. If both candidates beat AUTO, pair them directly before picking
the fastest. Compare native-half cuBLAS FAST with timed bias, retain the bias
rounding caveat, and promote robust internal p50+p95 winners before considering
deletion. Mixed F32-output needs a separate epilogue/qualification. No new5090
measurement and no Triad optimization is claimed by this checkpoint.

## Ada half-swizzle three-toolkit timing census (2026-09-06)

Frozen production force checkpoint6570ce87 completed allsix paired runs:400
21-window screening records and400 final101 records,zero rejects. Main separately
recomputed all samplewise quantiles, checked every external module identity and
exact cohort/physical graph/bit proof,174local sources and197remote pre/post hash
check keys(174source,14binary,9cache). No source, AUTO or revision changed.

| CUDA | Candidate | Actual AUTO | Internal p50+p95 wins | cuBLAS FAST wins |
| --- | --- | --- | ---: | ---: |
|12.8|Pipeline|Tc128|20/20|12/20|
|12.8|Swizzle|Tc128|20/20|15/20|
|13.0|Pipeline|Tc128|20/20|12/20|
|13.0|Swizzle|Tc128|20/20|15/20|
|13.2|Swizzle|Pipeline|10/20|13/20|

Each cell aggregate covers BF16/F16 x A-E x bias0/1, with the worst paired p50
and p95 over eager/graph and both ordering cohorts. CUDA13.2 confirmed internal
swizzle wins:BF16 B/D/E andF16 B/D,bothbias. Retain pipeline forA/C bothdtypes and
F16 E;its short-screen E wins did not survive101. CUDA12.8/13.0 both candidates
beat oldTc128 in every cell, but separate vsAUTO ratios do not rank candidates.
The required direct pipeline/swizzle paired workflow is the immediate next task.

The vendor denominator is native-half CUBLAS_COMPUTE_32F,not PEDANTIC. F32
PEDANTIC is only the accuracy reference. Timed vendor bias broadcast rounds the
FP32bias to half; it is not our FP32-bias-preseed raw-bit oracle. This is not an
allprecision inference/Triad/5090 completion claim or permission to delete routes.

Evidence:internal/perf/ada-half-swizzle-census-20260906/README.md and
census-report.md, all raw arrays,100-cell confirmation table and independently
reproduced main summaries. Immediate pre/post GPU snapshots were idle with
no compute apps;1800MHz SM clocks,post-run temperatures up to77C are preserved.
No competing workload was stopped. Continue direct pairing then actualAUTO
promotion and101 confirmation on allthree toolkits before mixed/remaining gaps.
