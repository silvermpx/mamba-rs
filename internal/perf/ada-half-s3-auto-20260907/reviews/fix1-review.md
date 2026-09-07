- **I1: live/current global epoch consumers in Triad dispatch/launch and cohort binding still assert revision 42** — ADDRESSED. The global and F32 alias assertions now require 43 at `src/mamba_ssm/gpu/gemm_bi_triad/dispatch.rs:10123` and `src/mamba_ssm/gpu/gemm_bi_triad/dispatch.rs:10124`; both resolved physical launch identities require 43 at `src/mamba_ssm/gpu/gemm_bi_triad/launch.rs:13855` and `src/mamba_ssm/gpu/gemm_bi_triad/launch.rs:13922`; and the current captured cohort evidence requires 43 at `tests/gemm_bi_tf32_cohort_binding.rs:207`.
- **Additional current direct-pair oracle: forced-S2 control benchmark still uses the revision-42 AUTO oracle and expects Swizzle for the two CUDA 13.2 B0/no-bias half cells** — ADDRESSED. The current helper and literal-guard test are named for revision 43 at `tests/gemm_bi_fixed_performance.rs:9689` and `tests/gemm_bi_fixed_performance.rs:9728`; exactly the CUDA 13.2 BF16/F16 hot_b no-bias choices are S3 at `tests/gemm_bi_fixed_performance.rs:9712` and `tests/gemm_bi_fixed_performance.rs:9718`, while their bias=true counterparts remain Swizzle at `tests/gemm_bi_fixed_performance.rs:9713` and `tests/gemm_bi_fixed_performance.rs:9719`. The forced direct-pair benchmark consumes this helper at `tests/gemm_bi_fixed_performance.rs:14595`, and its Pipeline/Swizzle controls are unchanged by the fix diff.

### New Breakage in the Fix Diff

None. No new Critical or Important issue is present in the exact fix diff; all edits are confined to test/oracle code and do not change CUDA or Triad production behavior.

### Preservation Checks

- Numeric ABI revision 5 and schedule revision 8 literal guards remain at `src/mamba_ssm/gpu/gemm_bi_triad/dispatch.rs:9492`, `src/mamba_ssm/gpu/gemm_bi_triad/dispatch.rs:9493`, `src/mamba_ssm/gpu/gemm_bi_triad/dispatch.rs:9675`, `src/mamba_ssm/gpu/gemm_bi_triad/dispatch.rs:9676`, `src/mamba_ssm/gpu/gemm_bi_triad/dispatch.rs:11153`, and `src/mamba_ssm/gpu/gemm_bi_triad/dispatch.rs:11154`.
- Historical `ada_s3_pair` pre-promotion capture remains revision 42 at `tests/gemm_bi_fixed_performance.rs:45`, `tests/gemm_bi_fixed_performance.rs:59`, and `tests/gemm_bi_fixed_performance.rs:576`; its historical AUTO incumbent assertion remains Swizzle at `tests/gemm_bi_fixed_performance.rs:634`.
- Current literal coverage remains fail-closed for unsupported toolkit, dtype, and shape inputs at `tests/gemm_bi_fixed_performance.rs:9728`, and now explicitly guards both promoted no-bias choices and retained biased choices at `tests/gemm_bi_fixed_performance.rs:9773`.
- SHA-256 checks for all four touched files match the fix report. No runtime, GPU, SSH, build, or suite command was rerun. The report records the corrected CUDA 12.8 full library as 645 passed / 0 failed / 46 ignored, exit 0, and the focused current-oracle test as 1 passed / 117 filtered, exit 0; remaining runtime and post-101 qualification are correctly outside this fix1 review gate.

### Out-of-Scope Observations

None.

### Verdict

**Fix round:** All findings addressed, no new Critical/Important breakage.

**Spec verdict:** PASS.

**Quality verdict:** PASS.
