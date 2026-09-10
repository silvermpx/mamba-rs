# SM120 G06/G09 post-quiet warmup audit

**Audited source/evidence:** `a265125e5541d706a9cb335de6bb6eed69939e94` plus the current test-fixture-only G10 worktree delta  
**Scope:** read-only diagnosis of the CUDA 13.0 G06 and G09 retained qualification anomalies. No source edit, build, GPU job, clock lock, sample deletion, or admission was performed.

## Decision

This is a concrete qualification-harness ordering defect. The harness calls its only warmup and calibration before `quiet.require_cohort(.../timed)`. That gate then requires five consecutive quiet telemetry/process samples and sleeps 100 ms between unsuccessful/intermediate attempts. In the best case, four sleeps plus command overhead separate the warmup from discovery. The recorded discovery cohort therefore begins after an intentionally idle interval, not after the declared warmup.

The two independent failures have the exact signature expected from that ordering:

| Receipt | Recorded eager discovery failures | First later discovery speedup | Final eager minimum | Final conservative median / p05 |
|---|---|---:|---:|---:|
| G06 `nn_d128_in_proj` | AB indices 0–2; BA indices 0–2 | AB 1.03845; BA 1.12937 | AB 1.11944; BA 1.11672 | 1.13792 / 1.12453 |
| G09 `tn_d128_out_proj` | AB indices 0–1; BA indices 0–1 | AB 1.32221; BA 1.30331 | AB 1.60451; BA 1.60605 | 1.60859 / 1.60673 |

For G06, all six sub-1 samples are the first three paired cycles. For G09, all four are the first two paired cycles. Every remaining eager discovery sample is above 1, all 202 final eager samples in each receipt are above 1, and both graph discovery/final cohorts pass. The 101-window final cohort immediately follows discovery with no intervening quiet gate and shows no corresponding transient. Of the 23-cell CUDA 13.0 retained batch, 21 admitted; G06 and G09 were the only failures. This does not prove a particular clock mechanism, but it does prove that the existing warmup does not warm the state at the point that is being qualified; the replicated front-only transient makes that defect the parsimonious cause of these two false negatives. NVIDIA's profiler guidance independently notes that prior launches affect GPU clock state and first launches can execute at lower clocks: <https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#clock-control>.

The old V1 receipts remain failed receipts and must remain archived unchanged. Do not admit G06 or G09 from their later samples, do not trim their leading samples, and do not reinterpret their p05 values. The prospective corrected run is new evidence.

## Exact source cause

In `tests/gemm_bi_sm120_tf32_selector_qualification.rs`:

- lines 1872–1880 perform the calibration quiet gate, operand seeding, 128-launch sequential warmups, and candidate/scalar calibration;
- line 1881 then performs `quiet.require_cohort(.../timed)`;
- line 1882 immediately starts the recorded 21-window `paired_samples` discovery;
- lines 1895–1903 run the 101-window final cohort immediately after discovery.

`paired_samples` at lines 743–777 is already the correct recorded ordering: each cycle records candidate→scalar (AB), then scalar→candidate (BA), using the same calibrated iteration count. It reseeds both operands before each cohort. The strict percentile gate is also behaving as written: with 21 values, nearest-rank p05 is the second-smallest sample, so two or three cold cycles necessarily reject the discovery cohort.

In `tests/common/gpu_quiet.rs`, `REQUIRED_QUIET_SAMPLES` is 5 (line 11); `require_quiet` samples telemetry/process state at lines 282–315 and sleeps 100 ms at line 309 until the five-sample streak completes. `require_cohort` delegates to it at lines 320–322. Thus moving or removing a statistical gate is neither necessary nor appropriate; the defect is solely that warmup is on the wrong side of this idle gate.

## Minimal prospective correction

Keep calibration where it is and keep the timed quiet gate. Add one fixed, non-adaptive helper call between `timed_preflight` and `discovery_samples`, separately for eager and graph:

```rust
const POST_QUIET_PAIRED_WARMUP_WINDOWS: usize = 4;

fn paired_warmup(
    candidate: &mut QualifiedPhysicalLaunch<'_>,
    candidate_ctx: &GpuCtx,
    scalar: &mut QualifiedPhysicalLaunch<'_>,
    scalar_ctx: &GpuCtx,
    path: Path,
    iterations: usize,
) -> Result<(), String> {
    candidate.seed_f32_operands(candidate_ctx, CORPUS_SALT)?;
    scalar.seed_f32_operands(scalar_ctx, CORPUS_SALT)?;
    for _ in 0..POST_QUIET_PAIRED_WARMUP_WINDOWS {
        measure(candidate, candidate_ctx, path, iterations)?;
        measure(scalar, scalar_ctx, path, iterations)?;
        measure(scalar, scalar_ctx, path, iterations)?;
        measure(candidate, candidate_ctx, path, iterations)?;
    }
    Ok(())
}
```

The call site must be exactly:

```rust
let timed_preflight = quiet.require_cohort(&format!("{label}/{path_name}/timed"))?;
paired_warmup(
    &mut candidate,
    &candidate_ctx,
    &mut scalar_gate,
    &scalar_ctx,
    path,
    window_iterations,
)?;
let discovery_samples = paired_samples(/* unchanged 21-window arguments */)?;
```

Four paired cycles are fixed in advance, not selected from results. Three cycles are the observed G06 settling horizon; the fourth establishes one complete already-stable AB/BA cycle before recording. Each cycle exercises both physical launch contexts twice, once in each order position, through the actual eager or graph measurement path and with the already calibrated `window_iterations`. This avoids the asymmetric state that would result from merely repeating sequential candidate then scalar calibration after quiet. The normal `paired_samples` reseed then restores the qualification corpus before any recorded sample, which matters for beta-1 cases.

This preserves the quiet gate: it now proves exclusivity immediately before the local warmup-plus-measurement cohort. The existing postflight still proves exclusivity after measurement. Do not lock clocks globally, add sleeps, recalibrate after quiet, add a discovery/final intermission, or warm only the candidate. None addresses both arms and both order positions as directly.

Treat the protocol change as auditable evidence, not an invisible implementation detail:

- bump `SCHEMA` from `MambaBiSm120Tf32SelectorQualificationV1` to `...V2`;
- add `post_quiet_paired_warmup_windows: 4` and `post_quiet_paired_warmup_order: "candidate-scalar-scalar-candidate"` to the completion record (and update its golden JSON assertion);
- extend `runtime_qualification_requires_gpu_quiet_gates` to assert source order `timed_preflight < paired_warmup < discovery_samples` and pin the count/order literals.

No production selector/source identity changes belong in this correction.

## One-shot confirmation protocol

1. Freeze the test-only V2 patch and compile it once. Preserve the current G06/G09 V1 JSONL files and hashes; never overwrite them.
2. Pre-register two new unique CUDA 13.0 output paths, one for exact filter `nn_d128_in_proj` forced to `gemm_bi_nn_sm80_mma_tf32_v1_m64n64_bk32_s3`, and one for `tn_d128_out_proj` forced to `gemm_bi_tn_sm80_mma_tf32_v1_m16n16_bk32_s4`. Use the same target, cache, driver, source body, and all other qualification environment as the retained CUDA 13.0 packet.
3. Run each cell exactly once in a predetermined order. A harness/process/setup failure may be classified as such, but a completed statistical failure is final evidence; do not rerun it.
4. Require the normal numerical, guard, eager, graph, discovery, and final gates unchanged. Require exactly 21 AB + 21 BA discovery and 101 AB + 101 BA final samples per path, strict median `>1.01` and p05 `>1.0`, a completion record, and identities equal to the retained CUDA 13.0 packet. The absence of a front-loaded sub-1 transient is a causal diagnostic, not a replacement acceptance gate.
5. Admit only a cell whose new V2 receipt passes. If either fails, leave that cell on exact and retain the new receipt; investigate the new sample shape without trimming, threshold changes, or retry.

The existing 21 passing CUDA 13.0 V1 receipts need not be rerun for this bounded repair: they cleared every strict gate despite the colder protocol, and this finding supplies no basis to invalidate them. G10 runs already started with the frozen V1 fixture are likewise preserved as V1 evidence; do not relabel them or mix them into the V2 G06/G09 confirmation.

## Verdict

**Harness defect: CONFIRMED.** The implemented warmup is separated from recorded discovery by a mandatory sustained-idle gate, and G06/G09 independently show only a front-of-first-cohort transient.

**Correction design: SUFFICIENT IF IMPLEMENTED AS ABOVE.** A fixed four-cycle post-quiet calibrated paired warmup covers candidate and scalar in both AB/BA positions without weakening gates or deleting recorded samples. G06/G09 remain unadmitted until their single prospective V2 runs pass unchanged qualification gates.
