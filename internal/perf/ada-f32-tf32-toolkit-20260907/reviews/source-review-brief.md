# Task7 first source gate — corrected final4 package

Use the task-reviewer method: assess spec compliance and task code quality,
with file:line evidence and separate Critical/Important/Minor findings. This
is a task-scoped source gate before screen21, not whole-branch merge review
or final performance-evidence acceptance. No21/101 was authorized or run.

Worktree /Users/silvermpx/IdeaProjects/mamba-rs/internal/worktrees/gemm-bi-triad-sm80.
All following relative files are inside
.superpowers/sdd/handoff-codex-gemm-bi-triad-2026-09-05/:

- Requirements: ada-f32-tf32-toolkit-task7-brief.md, with the binding
  ada-f32-tf32-toolkit-task7-incumbent-ruling.md correction. The original
  audit is historical and wrong about TF32 AUTO precedence; the amended
  brief/ruling contains the requested current behavior.
- Implementer report: ada-f32-tf32-toolkit-task7-report.md. Its top final4
  section is current; final3 sections below preserve superseded history.
- Diff: ada-f32-tf32-toolkit-task7-source-review-package-v2.diff. Read once,
  including all six files. It contains full additions and tracked entry diff
  against e1d503c78694936b46ee79ca6b1a320a494ce7f5. HEAD is still that base;
  root owns commits after verification, so this is frozen uncommitted WIP.

Source and validation-tool hashes were independently matched by root to the
report. Direct rustfmt --check --edition2024 --config skip_children=true on
both Rust files passed. Actual final4 GPU smokes are still executing under
the sole implementer; report final21/101 performance requirements as pending
the later evidence gate, while evaluating that their code implements them.

Binding constraints copied from the task:

Produce a complete per-literal winner matrix for existing CopyPlan and M64S2
against actual incumbent AUTO and independently against cuBLAS Fast. No source
kernel, loader, selector, compiler option, numeric/schedule revision change.
No new benchmark entry: retain the existing ignored
`fixed_ada_forced_rungs_paired_precision_cublas` as the measurement entry and add
an explicit narrow qualification mode. Ordinary historical mode remains intact.

Each literal must beat actual incumbent in every path/parity p50+p95 to
advance21->fresh101. Fast denominator independent, losses preserved. For
current TF32 C16, AUTO is RNA-wide and candidate is ordinary M64S2, with
different32/24-byte final argument bundles. Never disable the holder or alter
alignment to force the old ordinary-picker incumbent. ExactF32 AUTO is Legacy.

Any functional/physical/numeric/identity/guard/immutability/telemetry/exit or
exact-completion failure invalidates the affected run/stage; preserve reason
and raw data before a justified rerun. Confirm requires full storedscreen
recomputation and exact eligible subset/source/binary/artifact/toolkit binding
before launch, independently repeated by the analyzer. A digest alone is not
admission. All known custom captured args are validated; cuBLAS private ABI
is opaque and uses actual inventory plus public modes and workflow proof.

Review is read-only on source/index/HEAD; write only your report path below
using apply_patch. No SSH/GPU/build/Cargo/branches/worktrees/staging/commit/push,
deletion or subagents. The implementer owns Ada; no reviewer GPU work. Do not
rerun package-wide suites whose results are reported. A focused local host
probe is allowed only for a concrete code doubt not answered by existing logs.

Read the diff rather than reopening changed files. Inspect unchanged code
only for a concrete named risk, report the risk and focused location checked.
Two known cross-file interfaces warrant focused inspection: actual public
fixed_forward's full TF32 precedence (the original audit missed RNA), and
the reused parent launch/timing/vendor-graph helpers' numerical/bias/event
contracts. Do not repeat full loader or architecture inventories. If raw
evidence is not yet locally mirrored, name the precise missing path rather
than rerunning a GPU suite or inferring it never existed.

Output ada-f32-tf32-toolkit-task7-source-review.md with Spec compliance,
Strengths, Critical/Important/Minor findings and Task quality Approved/Needs
fixes. Label genuinely unverifiable cross-task evidence separately for root.
Judge the code on its merits, not on the implementer's claims. Preserve the
source-only boundary and return the report path and concise verdict.
