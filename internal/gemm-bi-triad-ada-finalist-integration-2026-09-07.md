# Ada compact NT finalist integration plan

> For agentic workers: use superpowers:subagent-driven-development or
> superpowers:executing-plans. Continue in the existing worktree and branch.

**Goal:** Make the selected compact8/S2 kernel reachable through the real
Triad dispatcher for qualified Ada NT cells, preserving all existing routes.

**Architecture:** An optional, separately identified Ada module holds one
selected compact kernel. A count-checked source builder reuses immutable
SM80 code; no copy of the complete SM80 file and no edits to its existing
composition. A separate route identity/selector epoch prevents the new
module from invalidating existing Fixed and portable evidence.

**Tech stack:** Rust, CUDA C++/NVRTC, Driver ABI/resource checks, existing
physical-launch qualification and paired timing harnesses.

**Spec:** `internal/handoff-codex-gemm-bi-triad-2026-09-05.md`, latest sibling
checkpoint; user instructions to wire measured winners, preserve deterministic
bits, qualify all installed toolkits, and avoid full gates per prototype.

## Global constraints

- Branch `codex/gemm-bi-triad-sm80`; no new worktree, push, AI/coauthor trailer.
- Preserve unrelated SM120 test WIP and untracked discovery-sample support.
- Root commits exact reviewed paths; only the designated GPU owner runs
  SSH/Cargo/CUDA. Native macOS rustc/rustfmt and actual host shims are allowed.
- Global tuning epoch45, numeric ABI5 and schedule8 remain unchanged.
- No old SM80/scalar/Fixed/specialized CUDA fragment is edited.
- Existing portable SM89/SM120 cohorts remain byte-for-byte and revision45.
- New module failure removes only the finalist, never the portable route.
- Three candidate cells, NT `(2048,768,3072)`, `(2048,1536,768)`,
  `(4621,384,1928)`, contiguous, aligned, alpha1/beta0/no bias.
- One kernel: M128N64/BK32/S2,256 threads,49152 dynamic shared, local/static0,
  runtime occupancy at least2. Preserve RNA and per-output ascending K8/MMA.
- CUDA12.8,13.0,13.2 are separate evidence identities. No new5090 claims.
- No admission literal is fabricated. Until qualification, expose the new
  physical route for qualification but retain existing public AUTO.

## Design decisions

Changing shared SM80 source would invalidate the current literal cohort and
can send TF32 requests to exact F32. Keeping an old literal is insufficient.
Copying a standalone compact kernel would duplicate hundreds of sensitive
staging/math/epilogue lines. Use a narrow fail-closed transformation of the
already measured template, hashing the actual complete output bytes through
the normal compiler path. Unused old exports in this separate module are
module-local; only the new exact symbol enters its holder.

The proposed names are `ModuleKind::TriadSm89Finalist`,
`Tf32PhysicalRoute::Sm89MmaTf32Compact8V1`, matching distinct physical backend,
and `SM89_FINALIST_TUNING_REVISION = 1`. Numeric contract stays
`MmaTf32RnaV1`; backend-to-module validation remains strict. Finalist revision
checks use1; every existing module still uses the live global45. The artifact
set may use this Ada-only fourth module, mutually exclusive with the existing
SM90a/SM100/SM120 fourth module. Never allow five modules.

## Task 1: isolate the selected source (native-testable)

Files: create `src/mamba_ssm/gpu/gemm_bi_triad/sm89_finalist_source.rs` and
`kernels/gemm_bi_triad/sm89_nt_compact.cuh`; later register the Rust module
in `src/mamba_ssm/gpu/gemm_bi_triad/mod.rs` during Task2.

Interface:

```rust
pub(super) const SM89_FINALIST_SYMBOL: &str =
    "gemm_bi_nt_sm89_mma_tf32_compact8_v1_m128n64_bk32_s2";
pub(super) fn compose_sm89_finalist_source() -> Result<String, String>;
```

- [ ] Write native tests first: missing/duplicate anchors must return Err;
  reversed transformation restores exact immutable SM80 source; export/assert
  sets balance; only the target storage, slots, warp ownership and symbol pair
  differ from the frozen `CompactEightWarpS2` candidate.
- [ ] Implement the tiny XOR helper with production naming and the exact
  measured count-checked replacements. Keep all non-target instantiations
  unchanged. Compose the same five preambles with transformed body, rejecting
  unexpected include seams. Do not depend on `tests/` from production.
- [ ] Native rustc RED/GREEN, rustfmt and scoped diff-check; normalize only
  helper comments/name and target symbol to compare with the frozen candidate.
- [ ] Review and checkpoint. This task alone does not load or promote a kernel.

## Task 2: physical route and optional-module integration

Files: `src/mamba_ssm/gpu/{kernel_identity.rs,context.rs,kernels.rs}` and
`src/mamba_ssm/gpu/gemm_bi_triad/{mod.rs,contract.rs,modules.rs,dispatch.rs,launch.rs,qualification.rs}`.

- [ ] Add the distinct module/backend/physical route and one NT kernel spec.
  Thread it through existing exhaustive identity/spec/launch matches. Do not
  weaken the old MmaTf32RnaV1 backend-to-TriadSm80 invariant.
- [ ] Compile only on actual CC8.9, using the complete Task1 source bytes in
  source digest/compile key. Load only the finalist symbol, validate the five
  argument Driver ABI and runtime resource contract. Keep optional rejection
  reason, module lifetime anchor and `F32TriadAvailability::finalist` separate
  from the existing `specialized` field, which suppresses portable fallback.
- [ ] Add live module/binding access and route-specific revision validation.
  Existing physical routes with epoch1 must fail, finalist epoch1 must pass,
  finalist epochs0/2 must fail; all old routes remain45.
- [ ] Add selector hook before the old portable lookup with an initially
  empty qualified cohort. Wrong identity/shape/stride/alignment/operands,
  absent holder, and rejected holder must return the exact previous selection.
- [ ] Focused host tests cover source failure isolation, artifact fourth-slot
  constraints, unchanged Fixed/SM89/SM120 selections, stale route rejection,
  exact spec/ABI/resource contracts, and generated source digest binding.
- [ ] One build/list per installed toolkit after the source freeze. Do not
  run the full qualification matrix after every Rust edit.

## Task 3: frozen qualification and literal AUTO admission

Files: existing `tests/gemm_bi_tf32_contract.rs`,
`tests/gemm_bi_tf32_cohort_binding.rs`, `tests/gemm_bi_performance_matrix.rs`,
and a scoped report under `internal/perf/`.

- [ ] Qualify the one physical kernel on the three selected NT cells plus
  edge cases: finite full-mantissa eager/graph repeats, same per-output prefix,
  forced tails, input/redzone checks, immutable inputs, scalar epilogue cases
  allowed by the physical route, and rejected/misaligned neighbouring requests.
- [ ] Keep numerical and timing references distinct: portable forced RNA
  establishes the same numeric contract on a toolkit where public AUTO may
  fall back to exact F32. Never demand TF32 bits equal exact F32 bits. Time
  actual public AUTO and cuBLAS Fast as separate denominators.
- [ ] Run a single paired finalist/current and finalist/Fast batch on
  CUDA12.8/13.0/13.2. Record actual loaded compiler/module/Driver identities,
  source and binary hashes, resources, graph and eager results, raw samples,
  quiet PRE/raw immediate RELEASE/separate quiet DRAIN. No cold-cache claim
  on the reused private discovery cache.
- [ ] Admit only passing literal toolkit/cell identities. Verify actual AUTO
  now resolves and launches the new symbol on each admitted cell, and that
  all existing portable runtime gates still pass on their qualified stack.
- [ ] One final selected-matrix/library regression batch, independent raw
  replay and code review; commit source + evidence + handoff. Report remaining
  Fast losses honestly. Do not delete losing experiments in this phase.

## Afterwards

Resume bounded TN work only after the three NT cells are assembled. The
read-only TN proposal uses transposed XOR `axis ^ ((reduction & 3) << 3)`,
not the NT XOR. Prism TN has only93 CTAs for142 SMs: occupancy2 does not cure
its grid underfill, and no speedup is established by that proposal.
