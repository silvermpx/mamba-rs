**I1 — Empty output reaches a cold architecture probe.** — ADDRESSED. `src/mamba_ssm/gpu/gemm_bi_inference.rs:4103` returns `Ok(None)` for M0/N0 before alignment and before invoking the architecture-enabled/preparation closure; the production AUTO branch uses that decision seam at `src/mamba_ssm/gpu/gemm_bi_inference.rs:4925`. The cold-loaded regression covers BF16/F16 and both SM90/SM100 tiles across recording/capture states, asserting success and zero preparation/probe callbacks at `src/mamba_ssm/gpu/gemm_bi_inference.rs:4121`. The production-entry regression adds aligned `(0,64,96)` and `(3,64,0)` cases and asserts zero context routes at `tests/gemm_inference_route_inventory.rs:188`. The nonempty companion preserves cold recorder/capture rejection and enabled/disabled probe verdicts at `src/mamba_ssm/gpu/gemm_bi_inference.rs:4160`.

### New Breakage in the Fix Diff

None. For nonempty requests, `inference_arch_rung_for_request` retains the prior loaded-holder, K/N alignment, X/W pointer alignment, and enabled-verdict ordering at `src/mamba_ssm/gpu/gemm_bi_inference.rs:4106`; the production closure still performs the same recorder-aware `arch_rung_enabled` call at `src/mamba_ssm/gpu/gemm_bi_inference.rs:4927`. No Critical, Important, or Minor regression was introduced by the two-file fix diff.

### Out-of-Scope Observations

None. Previously recorded warning cleanup and evidence-index maintenance are unchanged and remain nonblocking deferred work.

### Focused Checks and Evidence

- Supplied fix diff read once; no Git, Cargo, or GPU command was rerun and no source or index file was changed.
- Root-owned RED receipt shows the aligned BF16/SM90 recording case failed with the eager-preparation error before the three-line production guard.
- Root-owned GREEN receipts show the aligned empty-output regression, nonempty companion, and original cold guard each passed 1/1; the actual-GPU production-entry empty-output regression also passed 1/1 with the new K64 shapes.
- The report records identical frozen production bytes across the three GREEN host checks and actual-GPU no-op check, plus the root-owned 564-file/source-after and scoped formatting checks.

### Verdict

**Fix round:** All findings addressed, no new Critical/Important breakage.

**Assessment:** Fix round 1 is accepted; I1 is closed.
