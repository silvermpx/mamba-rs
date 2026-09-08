# Half TN compact register pipeline implementation plan

> Agentic workers: use executing-plans for this bounded discovery task.

Goal: test whether register double buffering closes compact TN d768-in's
remaining graph/Fast gap without changing its K16 chain or F32 dW epilogue.
Spec: `internal/agent-operational-rules.md`; whole-Triad discovery precedes
integration. Fixed inference, production kernels and SM120 remain unchanged.

Architecture: derive a test-only source from `triad_half_tn_compact_source.rs`.
Keep BK64/S2, 64x64 output, 128 threads, compact XOR layout, cp.async schedule
and all arithmetic. Use two A/B register-fragment slots: prime issue0, load
issue(k+1) into the other slot, consume issue(k), in order0,1,2,3.

- [x] Add `tests/support/triad_half_tn_regpipe_source.rs` with failing generator
  checks: unchanged outside-loop source, unchanged MMA operand order after
  stripping only the buffer subscript, rejected missing/duplicate anchors.
- [x] Run native `rustc --edition 2024 --test` on that helper; observe RED.
- [x] Implement `candidate_source(production: &str) -> Result<String,String>`
  by extracting the existing load/MMA text and inserting buffer subscripts;
  add no arithmetic, copy or epilogue rewrite. Run the same tests to GREEN.
- [x] After the typed harness owner's NT freeze, add one variant to its
  existing TN runner, d768-in x F16/BF16 only, paired current TC64 and Fast.
  Require exact bits/repeats/guards, local0/static32768/dynamic0/occupancy>=3,
  256B logical alignment, once7 and20 GEMMs/observation.
- [x] Sole executor runs one focused Ada13.2 test. Retain only measured
  improvements; preserve a valid loss without resweeping. Commit exact
  measured source and raw evidence with no production admission.
