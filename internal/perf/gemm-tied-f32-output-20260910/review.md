### Spec Compliance

- ✅ Spec compliant. The deterministic BF16/F16 path casts the two inputs into the first two shared F32 scratch slots and invokes the existing ExactScalar NT implementation directly on caller-owned F32 logits with `(B,Vpad,D)`, contiguous NT strides, alpha 1, beta 0, and no bias (`src/mamba_ssm/gpu/blas.rs:3476`, `src/mamba_ssm/gpu/blas.rs:3502`, `src/mamba_ssm/gpu/blas.rs:3526`, `src/mamba_ssm/gpu/gemm_bi_triad/launch.rs:6260`). The public wrapper checks context health first, delegates F32, selects the composition only in deterministic mode, and leaves vendor modes on canonical GemmEx compute selection (`src/mamba_ssm/gpu/blas.rs:3592`, `src/mamba_ssm/gpu/blas.rs:3593`, `src/mamba_ssm/gpu/blas.rs:3604`, `src/mamba_ssm/gpu/blas.rs:3621`). Checked `B*D`, `Vpad*D`, and `B*Vpad` products, byte spans, allocation coverage, alignment, and output non-aliasing precede scratch growth or enqueue (`src/mamba_ssm/gpu/blas.rs:3373`, `src/mamba_ssm/gpu/blas.rs:3416`, `src/mamba_ssm/gpu/blas.rs:3494`). Zero reduction skips both casts and is handed to the existing F32 NT epilogue semantics (`src/mamba_ssm/gpu/blas.rs:3505`, `src/mamba_ssm/gpu/blas.rs:3515`, `src/mamba_ssm/gpu/gemm_bi_triad/launch.rs:6322`).
- ✅ The existing exact-scalar owned-buffer entry remains thin and retains cudarc's owned `GpuBuffer` argument submission; the shared generic body also accepts raw pointer arguments for the new raw composition without constructing fake owners or introducing another selector/dispatcher (`src/mamba_ssm/gpu/gemm_bi_triad/launch.rs:1832`, `src/mamba_ssm/gpu/gemm_bi_triad/launch.rs:1847`, `src/mamba_ssm/gpu/gemm_bi_triad/launch.rs:1857`, `src/mamba_ssm/gpu/gemm_bi_triad/launch.rs:6232`, `src/mamba_ssm/gpu/gemm_bi_triad/launch.rs:6247`, `src/mamba_ssm/gpu/gemm_bi_triad/launch.rs:6312`).
- ✅ Scratch presizing uses `(B*D,Vpad*D,0)`, is a no-op for F32 or non-deterministic/vendor modes, and is called before the actual M1 and M3 backbone capture while the head remains outside the captured backbone (`src/mamba_ssm/gpu/blas.rs:3542`, `src/mamba_ssm/gpu/blas.rs:3554`, `src/module/gpu_lm.rs:260`, `src/module/gpu_lm.rs:271`, `src/module/gpu_lm.rs:281`, `src/module/gpu_lm3.rs:286`, `src/module/gpu_lm3.rs:293`, `src/module/gpu_lm3.rs:303`).
- ✅ Required test surfaces are present: exact low-bit BF16/F16 cases, irregular `D=37` asymmetric guarded repeats, zero reduction, null/overflow rejection, observer ordering including multi-launch NT/no downcast, frozen scratch reuse/growth rejection, and synthetic M1/M3 capture paths (`tests/gemm_tied_f32_output.rs:13`, `tests/gemm_tied_f32_output.rs:84`, `tests/gemm_tied_f32_output.rs:209`, `tests/gemm_tied_f32_output.rs:256`, `src/mamba_ssm/gpu/blas.rs:3165`, `src/mamba_ssm/gpu/blas.rs:3209`, `src/mamba_ssm/gpu/blas.rs:3228`, `src/module/gpu_lm.rs:767`, `src/module/gpu_lm3.rs:631`). Rustdoc states the direct F32 output contract, pointer lifetime, errors, capture reservation, two conversions, and `(B+Vpad)*D*4` scratch cost without a speed claim (`src/mamba_ssm/gpu/blas.rs:3542`, `src/mamba_ssm/gpu/blas.rs:3565`).
- ⚠️ Cannot verify from diff: actual Ada execution and feature-lane compilation/Rustdoc outcomes are external runtime evidence. The root execution report records the two intended RED failures, twelve actual GREEN CUDA cases, both compilation lanes, and Rustdoc (`internal/perf/gemm-tied-f32-output-20260910/report.md:8`, `internal/perf/gemm-tied-f32-output-20260910/report.md:31`). The controller should retain/check the named raw `red/`, `green/`, and `green-resume/` receipts; this review intentionally did not rerun Cargo or GPU work.
- ⚠️ Cannot verify from diff: byte-for-byte CUDA identity requires the root manifest receipts. The package changes only the six listed Rust files, and the root report says CUDA/header manifest comparison passed (`internal/perf/gemm-tied-f32-output-20260910/report.md:34`), but the external immutable-source packet was not opened in this scoped review.
- ⚠️ Cannot verify from diff: branch/index state was not queried because the review contract forbids Git regeneration/state mutation and the task explicitly assigns commit/index ownership to root. The supplied package contains one commit, `21a3f0e5`, over base `779ef943`.

### Strengths

- The new path composes existing primitives at a narrow boundary: input conversion remains in `blas.rs`, while exact NT planning/dispatch remains centralized in `gemm_bi_triad/launch.rs` (`src/mamba_ssm/gpu/blas.rs:3502`, `src/mamba_ssm/gpu/gemm_bi_triad/launch.rs:6247`).
- Validation is deliberately front-loaded, including managed-allocation coverage and overlap checks, so malformed requests fail before shared scratch mutation or enqueue (`src/mamba_ssm/gpu/blas.rs:3416`, `src/mamba_ssm/gpu/blas.rs:3494`).
- The argument abstraction preserves the existing owned path's mutable/output and immutable/input submission hooks instead of degrading all callers to raw pointer launch arguments (`src/mamba_ssm/gpu/gemm_bi_triad/launch.rs:1917`, `src/mamba_ssm/gpu/gemm_bi_triad/launch.rs:1923`, `src/mamba_ssm/gpu/gemm_bi_triad/launch.rs:6232`).
- Capture regression fixtures warm only the backbone, capture through the real LM wrapper, make their first head call after freeze, and assert both reserved slots remain stable (`src/module/gpu_lm.rs:816`, `src/module/gpu_lm.rs:819`, `src/module/gpu_lm.rs:823`, `src/module/gpu_lm3.rs:667`, `src/module/gpu_lm3.rs:670`, `src/module/gpu_lm3.rs:674`).
- The six reviewed source hashes exactly match the implementer's frozen-source manifest (`.superpowers/sdd/handoff-codex-gemm-bi-triad-2026-09-05/task-903-report.md:67`). A focused forbidden-pattern check found no `allow(dead_code)` in those six files.

### Issues

#### Critical (Must Fix)

None.

#### Important (Should Fix)

None.

#### Minor (Nice to Have)

- `internal/perf/gemm-tied-f32-output-20260910/report.md:62`: the verification output is not pristine; it retains deprecated-setter, CUDA-only unused-accessor, and known Rustdoc-link warnings. The report identifies them as pre-existing and no suppression was added, so they do not undermine Task903's behavior, but the review rubric requires warning noise to remain visible as a finding. Resolve them in the already-planned API/docs/test-cleanup phase and keep that phase linked to these receipts.

### Focused Checks

- Exact F32 composition risk: inspected `gemm_bi_tied_half_f32_in` and the borrowed exact-scalar pointer entry; confirmed two conditional input casts, direct caller logits output, ExactScalar selection, and no output-downcast/temporary-logits stage (`src/mamba_ssm/gpu/blas.rs:3476`, `src/mamba_ssm/gpu/gemm_bi_triad/launch.rs:6312`).
- Owned cudarc synchronization risk: inspected `ScalarInputArgument`/`ScalarOutputArgument`, `ScalarLaunchArgs`, and the owned wrapper; confirmed `GpuBuffer` still submits `inner()`/`inner_mut()` while only the explicit raw entry submits pointer bytes (`src/mamba_ssm/gpu/gemm_bi_triad/launch.rs:1835`, `src/mamba_ssm/gpu/gemm_bi_triad/launch.rs:1917`, `src/mamba_ssm/gpu/gemm_bi_triad/launch.rs:6232`).
- Scratch policy/order risk: inspected the presizer plus both LM capture methods; confirmed mode/dtype no-op semantics and reservation before backbone capture (`src/mamba_ssm/gpu/blas.rs:3549`, `src/module/gpu_lm.rs:264`, `src/module/gpu_lm3.rs:286`).
- Allocation-liveness risk: inspected `ManagedAllocationEpochStamp` and `with_bi_upcast_scratch`; the new code validates all live caller spans before scratch access, and scratch remains persistent/grow-only under the context freeze guard (`src/mamba_ssm/gpu/buffers.rs:17`, `src/mamba_ssm/gpu/context.rs:855`, `src/mamba_ssm/gpu/blas.rs:3495`).

### Assessment

**Task quality:** Approved

**Reasoning:** The implementation matches the requested composition, validation, observer, scratch-capture, testing, and documentation contracts without broadening CUDA policy or introducing a duplicate dispatch path. No Critical or Important finding remains; only pre-existing warning noise is carried as a non-blocking Minor.
