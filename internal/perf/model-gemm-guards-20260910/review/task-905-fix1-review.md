### Finding Verdicts

- **M1 mixed stale eager permit survives an opposite-path failed step/upload and remains usable for recapture** — ADDRESSED. All ten model entry points now clear only their matching permit unconditionally before upload: M1 F32 at `src/mamba_ssm/gpu/inference.rs:992` and `:1038`, M1 mixed legacy at `:1591` and `:1629`, M1 mixed native at `:2047` and `:2088`, M3 F32 at `src/mamba3_siso/gpu/inference.rs:1304` and `:1354`, and M3 mixed native at `:1963` and `:2002`. The amended M1 regression enumerates both attempted paths, both entry forms, and wrong-path versus invalid-upload failure at `src/mamba_ssm/gpu/inference.rs:133-135`; it requires the corresponding permit to be absent and recapture to fail with the eager prerequisite at `:215-229`, while the separate successful recapture control has no intervening step attempt at `:316-321`.
- **Check — intended RED:** the supplied exact M1 lifecycle run exited 101 at the first stale-permit assertion after 9.15s; its RED protocol runner exited 0.
- **Check — focused GREEN:** the supplied expanded M1 lifecycle run passed in 42.75s and printed all eight installed-opposite combinations; the M3 lifecycle run passed in 9.56s; the exact owner-map host test passed in 0.03s; the packet runner exited 0.
- **Check — build/freeze evidence:** the supplied CUDA+HF and CUDA-only checks, Rustdoc, source checks, and Cargo input checks completed successfully; the already logged warning debt is unchanged. Earlier accepted matrices were not rerun, consistent with the fix's permit-clear/test-only scope.
- **Check — reviewer actions:** read the supplied `96640d627d45cfe68f3c413a2e9bd18eacde8d80..a651b13395a6403de2852ab4edbe3ff2e606dc68` fix diff once and inspected the supplied reports/raw receipts; no Git, Cargo, GPU, build, or test command was run.

### New Breakage in the Fix Diff

- None.

### Out-of-Scope Observations

- None.

### Verdict

- **Fix round:** All findings addressed, no new Critical/Important breakage.
