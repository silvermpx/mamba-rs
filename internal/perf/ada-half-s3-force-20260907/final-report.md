# Task6A — production Ada half S3 force qualification

## Final status

Implementation and functional qualification COMPLETE on CUDA12.8,13.0,13.2.
Ready for root's final evidence review and Task6B production paired timing.
This is force-only functional admission, not performance/AUTO admission.
AUTO and tuning revision remain42. Final S3 register cap is188; final S3
K guard is checked_add(127). Earlier cap255 and K+63 statements were historical
interim states, superseded before the final matrix.

Root's independent source review and scoped fix1 review approved specification
and quality; finding I1 is closed, with no remaining material source findings.
No stage/commit/push/new branch, deletion, process signal, Mac Cargo/CUDA run,
or subagent was used. Root owns integration.

GPU lane released at2026-09-07T02:48:19Z:
GPU-d1edd7be-e88d-aed6-047d-622163306f0e, NVIDIA RTX6000 Ada, CC8.9/142SM,
driver595.45.04; utilization0%, no compute applications, explicit identity and
noapps checks both exit0. See lane-release.log. No further Task6A GPU work.

## Source and implementation

Worktree: /Users/silvermpx/IdeaProjects/mamba-rs/internal/worktrees/gemm-bi-triad-sm80.
Branch codex/gemm-bi-triad-sm80; production baseline
52d57d09e0e95e11c048d55e6901dd4c0e8fc430. Current HEAD includes root's
documentation-only3c0b1523389a6186c2cff147da2b79d0a450d0e7.

Seven authorized source/test paths only:

- kernels/gemm_bi_fixed/sm89_half_s3.cu (new)
- src/mamba_ssm/gpu/gemm_bi_fixed.rs
- src/mamba_ssm/gpu/gemm_bi_triad/modules.rs
- src/mamba_ssm/gpu/kernels.rs
- tests/arch_compile_gates.rs
- tests/gemm_bi_fixed_performance.rs
- tests/gemm_bi_fixed_sm89_pipeline.rs

FixedTile::Tc128Sm89S3 exposes distinct
gemm_bi_nn_fixed_sm89_tc128_s3_v1_bf16 and _f16. It reuses the unchanged
sm89_fixed_half_swizzle helpers and FixedSm89HalfSwizzleParams. The exact
five-argument ABI is C,A,B,bias then align4/32-byte alpha,beta,m,n,k,lda,ldb,ldc;
live layout0:8,8:8,16:8,24:8,32:32 with terminal next argument absent.

CTA128x128/BK64/S3, block256,98304B dynamic shared, B base49152,
ascending m16n8k16 recurrence and Task5B's distinct S3 ring/prologue/wait
schedule are preserved. Task5B candidate SHA256:
a90f504d6d2d0363908fd00d230263164105b41c6b5a6085a83ae130f2ce4222.
body-equivalence.log proves normalized mainloop equality after namespace
qualification; normalized SHA256:
08a71606ccf5d0de226d5251cfe24de4b19505904983c7f6b90c2c917f8514d4.

The fragment is appended only to Fixed/sm89 after its unchanged swizzle
provider. Cold/warm ABI census, independent optional holder and rejection,
exact PTX inventory/ABI/instructions/local-atomic gates, live Driver resources,
typed force dispatch and own launcher are wired. Complete force inventories,
physical eager symbols and graph parameter contracts include S3. Mixed F32
output remains incompatible. The stale already-promoted S2 comment is fixed.

All50 old CUDA files remain byte-identical to baseline
(unchanged-old-cuda-v2.json). The two retained Triad cache envelopes per toolkit
remain byte-identical to prior integration evidence, while the one Fixed
envelope is new (final-triad-cache-byte-equivalence.log, exit0). Architecture
tests also enforce non-Ada Fixed and Triad source exclusion/composition.

## TDD and source review fix

red.log records three meaningful initial REDs: missing S3 source, absent public
force inventory and missing mandatory S3 PTX inventory, exit101 each.
green-first.log records initial CUDA12.8 library642pass/46ignored,
force48pass/63ignored, pipeline host2pass and source1pass, all exit0.
The final matrix below includes later tests and supersedes these initial counts.

I1: K+63 protected ceil(K/64) but did not protect unconditional (kt+2)*64.
Only the S3 host guard changed to validate_sm89_half_s3_k using checked_add(127);
frozen candidate CUDA and all other launcher guards were preserved.
The regression rejects first invalid2147483521, reproducer2147483584 and
INT_MAX; accepts largest valid2147483520 and K0/64/128/192/256, and independently
checks lookahead using i64 arithmetic.

Exact Ada CUDA12.8 command:
cargo test --release --features cuda,cudarc/cuda-12080 --lib sm89_half_s3_rejects_overflow_in_unconditional_refill_index -- --nocapture

reviewfix-k-boundary-red.log: old+63 helper fails, exit101.
reviewfix-k-boundary-green.log: final+127 helper1pass/0fail, exit0.
The public unsafe-operand test also rejects both invalid K values with an
S3-specific prelaunch error. Final full12.8 command:
cargo test --release --features cuda,cudarc/cuda-12080 --test gemm_bi_fixed_sm89_pipeline -- --include-ignored --test-threads=1 --nocapture

cuda128-attempt1/half-full.log contains:
test fixed_sm89_half_pipeline_rejects_unsafe_operands_and_dimensions ... ok
and11passed/0failed,76.93s, EXIT half-full0. It also passes on13.0/13.2.
The reviewed post-v1 changes were the boundary fix/tests, measured cap188 and
five-line cold/warm compiler/artifact identity logging, all covered by fix1.

## Functional and retained coverage

Existing Pipeline and Swizzle remain in the full/rounding/sanitizer corpus;
S3 adds hotA-E, BF16/F16, both bias states, K0/64/128/192/256/tails,
odd/misaligned and padded strides, prefixes/row/output views, exceptional NaNs,
unrounded bias, guards and immutable inputs. Every independently poisoned
full/hot/sanitizer output uses reference bits XOR0xffff and GPU readback before
eager/replay. Hot expected force selection uses the actual supplied enum.
Raw five-argument launches cover nonunit alpha/beta, independent strides,
K0 and captured actual C/A/B/bias/scalars/function/grid/block/shared.
For beta!=0 both routes receive identical old-C inputs, each active seed
asserted different from gold.

All counts below are completed final-host tests. Every listed successful gate
has explicit exit0 in the corresponding matrix log.

| Gate | CUDA12.8 | CUDA13.0 | CUDA13.2 |
|---|---:|---:|---:|
| Release library |644pass/46ignored|644pass/46ignored|644pass/46ignored|
| Force/static performance suite |48pass/63ignored|48pass/63ignored|48pass/63ignored|
| Half source contracts |2pass|2pass|2pass|
| compiles_for_sm89 |1pass|1pass|1pass|
| Fresh private-cache cold + warm holders |1+1pass|1+1pass|1+1pass|
| Full half pipeline, include-ignored |11pass|11pass|11pass|
| Retained RNA-wide actual AUTO |2pass/448groups|2pass/448groups|2pass/448groups|
| Retained exact supported corpus |5pass|5pass|6pass|
| Retained exact AUTO13.2-only test |not applicable|not applicable|included above|
| Retained TF32-C AUTO13.2-only fixture |not applicable|not applicable|1pass|
| S3 eager physical identity control |8records|8records|8records|
| memcheck/racecheck/synccheck |0/0/0 errors|0/0/0 errors|0/0/0 errors|
| Production PTXAS/resource/SASS/cache proof |PASS|PASS|PASS|
| Triad SM120 arch + generic PTX contracts |not scheduled|not scheduled|1+1pass|
| Retained live TF32 cohort + bias cohort |not scheduled|not scheduled|1+1pass|

SM120 coverage is compile/PTX/static validation on Ada, not SM120 hardware runtime.
The ignored performance inventory is not a timing qualification claim.
Eager physical control has graph_replay_bits_equal=false because that control
runs eager only; the full pipeline separately validates actual graph replay.

CUDA12.8 matrix-cuda128-attempt1.log intentionally remains a FAILED attempt:
its original runner mistakenly included fixed_sm89_exact_n64_auto_prefix_view_graph_bits,
whose existing fixture requires NVRTC13.2. Assertion at exact test line1226 was
left(12,8),right(13,2), exit101; the other five exact tests passed.
No test was weakened. Corrected continuation ran the five supported exact
tests, physical control, all sanitizers and artifact gates using the same
validated source/cache in NEW cuda128-continuation1 evidence, all exit0.
Thus completed12.8 acceptance combines attempt1 through RNA and continuation1.
matrix-cuda130-attempt1.log and matrix-cuda132-attempt1.log each end final exit0;
13.2 completed02:47:21Z. The13.2 broad exact corpus took138.00s and passed6/6.

## Commands, isolation and runner integrity

Evidence root (all references above/below relative to it):
internal/perf/ada-half-s3-force-20260907/
Remote source: /root/mamba-ada-half-s3-force-20260907.
Remote evidence: /root/evidence-ada-half-s3-force-20260907.
Targets: /root/target-ada-half-s3-force-cuda{128,130,132}-20260907.
Final caches: /root/mamba-kcache-ada-half-s3-force-final-cuda{128,130,132}-attempt1-20260907.

Matching CUDA_HOME,PATH,LD_LIBRARY_PATH point to /usr/local/cuda-X.Y.
CUDA Cargo features respectively cuda,cudarc/cuda-12080, cuda,cudarc/cuda-13000,
cuda,cudarc/cuda-13020. Full COMMAND records and every exit are retained in
matrix-cuda128-attempt1.log, matrix-cuda128-continuation1.log,
matrix-cuda130-attempt1.log and matrix-cuda132-attempt1.log.
The final runner is run-toolkit.sh locally/run-toolkit-v4.sh remotely;
historical runner versions and run-continuation128.sh are preserved.
Each attempt rejects existing cache/evidence directories and uses no-clobber
logs; critical identity/tool/telemetry exits are explicit. Sanitizer binaries
are uniquely resolved and hashed, not selected by find|head.

Main suites use cargo test --release --features <matching-feature>:
--lib; --test gemm_bi_fixed_performance;
--test arch_compile_gates fixed_sm89_half_;
--test arch_compile_gates compiles_for_sm89;
--test gemm_bi_fixed_sm89_pipeline -- --include-ignored --test-threads=1 --nocapture;
--test gemm_bi_fixed_correctness fixed_sm89_rna_wide_actual_auto -- --ignored;
--test gemm_bi_fixed_sm89_exact_n64 -- --include-ignored --test-threads=1 --nocapture
(with --skip fixed_sm89_exact_n64_auto_prefix_view_graph_bits only12.8/13.0).
See runner/COMMAND logs for exact cold/warm, retained13.2 and environment forms.

Sanitizers use the exact archived pipeline executable and
--tool {memcheck,racecheck,synccheck} --report-api-errors no --error-exitcode99
--ignored --exact fixed_sm89_half_pipeline_sanitizer_smoke --nocapture.
API reporting is suppressed for the intentional terminal Driver ABI query;
memory/race/synchronization errors still fail the process.

The physical control uses fixed_ada_forced_rungs_paired_precision_cublas with
MAMBA_FIXED_ADA_VENDOR=1, EXACT_CC8.9, ROWSbf16,f16, CELLShot_a, BIAS0,1,
WINDOWS1, TILESTc128Sm89S3 and PATHSeager (full variable names in runner).
It is strictly identity control, not measured production paired admission.

## Artifact identities and resources

Final fixed source digest on all three:
205f58b56429b8e74f3ac1e7ab9f0cf6a9bf193ffeb87b254bd3ab00e6b02ca5.

| Toolkit | BF16/F16 registers | Fixed compile key | Fixed PTX artifact SHA256 |
|---|---|---|---|
|12.8|182/182|57c3e4f2177104f137d432f19398a4a36799aca4c264634a1ab25d17dbf422ba|60977db33de28d807ac7dd3dafe2d176914f7b7eab589917a47988050712352c|
|13.0|182/182|adee1f8b255bacfce8921820d9397a0a22a0dbaac735759fdcd222a453176155|d1aa6e33a612d99cd44a1e9c7eebe495c05eeef2212bf2cf4296db55eedb86a9|
|13.2|188/188|3a5fcbc3e7fef62dbb3542118d6d99d4ffba12b0cac04103b58f27ab4be36fb0|8b89aadf7d456b78ab249ff614568730946006eefd7143e35bce1e158741aff1|

All six entries: zero stack/static shared/local/spill stores/spill loads;
live block limit256,98304 dynamic shared, occupancy1. SASS per entry:
commit3,wait0=2,wait1=2,HMMA128,LDSM72,LDGSTS24,BAR5.
Final measured cap188 covers all required toolkits.

Each attempt's artifact-proof.log binds cache envelope/key/payload hash to
fixed.ptx and validates per-symbol PTXAS/resources/SASS. identity-cuda*.json
binds all eight physical records to compiler,header,NVRTC library,source,
invocation and artifact identities and revision42. cold.log/warm.log retain
exact S3_MODULE_IDENTITIES and live ABI/resources. identities-final.log records
full source/build inputs, executable and cache hashes.

## Final evidence and caveats

source-stable-seven.sha256 freezes the exact seven source paths.
source-final-v2.tar.gz retains356 actual source/build input files;
final-v1.diff retains the tracked source diff and the new CUDA is in the archive.
cuda128-final-binaries-cache-v2.tar.gz contains6 test executables and3 cache
envelopes; cuda130-final-binaries-cache.tar.gz and cuda132-final-binaries-cache.tar.gz
each contain7 test executables and3 cache envelopes.
archive-verification.json PASS reconciles every actual source/build input,
all20 archived test binaries and9 cache envelopes against each remote final
hash log, plus each Fixed key/artifact against the physical identity control.
macOS AppleDouble ._ metadata is explicitly excluded as non-compiler input.

Full raw logs, saved PTX/cubin/SASS/resources, failed attempts and archives are
retained locally. archive-verification-first.log preserves the initial checker
failure on AppleDouble metadata, corrected without changing artifacts.
archive-cuda128-first.log and its partial archive preserve an initial incorrect
guessed executable filename; resolved archive-cuda128-final.log exits0.
source-final-v1.tar.gz was an initial tar attempt that referenced nonexistent
root build.rs; source-final-v2.tar.gz is authoritative. Earlier scratch
preparation failures are retained, not acceptance evidence.

SHA256SUMS records the final rooted evidence manifest; manifest-verification.log
records its complete verification. The manifest intentionally excludes itself
and its verification output, and includes a final-report.md copy of this report.
Full archives remain local; root selects bounded text/checksum evidence for Git.

No material source or functional blocker remains. Root final evidence review
and Task6B production direct21/101 paired timing, confidence/physical identity
checks and any measured AUTO43 decision remain outside Task6A. This task makes
no S3 latency, vendor victory, general-toolkit AUTO, or SM120 runtime claim.
