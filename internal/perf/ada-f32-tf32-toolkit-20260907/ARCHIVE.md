# Ada exact-F32 / TF32 cross-toolkit qualification checkpoint

Task7 measures existing kernels on RTX6000Ada CC8.9/142SM with CUDA12.8 and
13.0. It does not change AUTO, kernel source, loader, compiler options or the
current tuning43/numeric5/schedule8 tuple. The exact20literal matrix is in
final-matrix.json; all432 constituent quantile pairs are in
final-constituents.json. Raw JSONL is authoritative.

| Candidate and literal domain | Candidate / actualAUTO | Candidate / cuBLAS Fast | Decision |
|---|---|---|---|
| Exact-F32 CopyPlan, A/B/D/E, bothbias, eachtoolkit (16cells) | All101 p50 andp95 win in allfourstrata | Every constituent loses | Qualified for separate AUTO integration |
| TF32 M64S2, C, bothbias, eachtoolkit (4cells) | All21 lose against actualRNA | Every constituent loses | Keep RNA; no101 |

For exactF32 the worst-per-cell101 median ratio ranges0.814324..0.844589,
and worstp95 ranges0.828924..0.891455, relative to actualLegacy AUTO.
Worst-per-cell candidate/Fastp95 ranges1.543210..2.239997. These are robust
own improvements, not Fast wins. TF32 M64/RNA worstp95 is~1.518 across its
fourcells. CurrentRNA was already connected onall3toolkits; the initial
TF32-incumbent audit omitted its precedence and is preserved with an explicit
correction in reviews/incumbent-ruling.md.

All4 final5b smokes, all4screen21 and both exactconfirm101 are complete.
Confirm101 binds the exact recomputed21subset and unchanged source/binary/
compiledartifact. There was no timing retry or pooling; TF32losses did not
advance. Each run checks actual publicAUTO/forced/vendor launches, custom
one/20node captured symbols/fullABI/pointers/bundles/geometry, Fast public
modes and bias workflow, deterministic bits, outputcomplementreadback,
successfulemptygraph overwrite negative, fullguards/inputimmutability and
strict same-attempt telemetry/test/wrapper/outer exit closure. PEDANTIC is
only the untimed numerical reference; timedF32vendor is FAST_TF32.

The main report and original corrected source/fix reviews are copied
byte-identically to final-report.md and reviews/. Source review's three
Important findings are addressed. Its unused-import Minor is disclosed and
deferred to the next already-needed integration edit. Final independent
evidence review is accepted: specPASS, qualityPASSWITHMINOR, noCritical or
Important findings and no newevidencefinding. See reviews/final-evidence-review.md,
SHA750e0d02b524229c7a987894eee214af5e33e57f75499779b7263352b33697ac.
reviews/controller-checks.md records root's independent arithmetic, source,
archive and closure checks; reviews/root-replay.rb reproduces the local raw
arithmetic/closure check without GPU access.

manifest-files.sha256 is the frozen243-entry complete local snapshot,
SHA95c8356fa8546ce343c3946e9ab9fd1f4432ca6999cc18ab027362803d69c537;
manifest-root.json binds it. Root verified243/243. It includes original SDD
paths, the two frozen testsourcefiles, incidentalbytecode and three local
tar.gz archives. For this historical manifest, source paths refer to this
checkpoint, not a future changedAUTO44 working tree. All357 archivedbuild
inputs, both exactbinaries, six cacheenvelopes and Fixedinvocation/artifacts
were independently matched. The tar.gz files remain on local disk, not inGit;
no raw data, archives or losing kernels were deleted. ARCHIVE_SHA256SUMS covers
only the selected version-controlled text/source evidence, including the
byte-identical report/review copies. It excludes itself and disk-only files.

Final quietrelease2026-09-07T06:56:16Z records exactUUID/CC8.9/0%GPU/0%memory,
noapps and queryexits0. Actualsource/binary hashes, allcommands, original
development failures and the final20literal table are in final-report.md.
No 5090, other GPU architecture, end-to-end inference or Triad qualification
is claimed by this checkpoint.

Next bounded integration connects these16exact winners to actualAUTO on12.8
and13.0, retaining the already-qualified13.2 routes, allTF32RNA andhalf choices.
It advances only hosttuning43->44 and requires actualpostAUTO smoke/101 before
acceptance. That integration is not part of Task7's measured43 evidence.
