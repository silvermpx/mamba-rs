# Task 2 review-fix report: SM89 half graph, ABI isolation, and oracle

## Outcome

The Task 2 production integration now admits `TriadSm89Half` through the
one-node native prepared-graph path. NN graph rehydration shares the exact
`#[repr(C)]` 32-byte parameter type used by eager launch and marshals four
pointers plus that single by-value bundle. The legacy SM80/scalar NN graph ABI
is unchanged.

The live Driver ABI census preserves module-level fail-closed behavior for
module load, architecture, procedure lookup, inventory, and unload failures,
but records export lookup/query/validation results independently for each of
the six SM89-half symbols. Loading applies ABI and resource admission to each
symbol separately, so one rejected export falls back only for its own cells.

The ignored actual-AUTO harness still asserts the literal 11 special identities
before executing them. It now also binds a separate retained SM80 Tile128
forced route in an independent context, seeds both routes identically, compares
the special eager and graph output bits against that retained oracle, and
checks exact A/B snapshots after both phases. Both candidates also retain their
red-zone validation. No GPU execution was performed in this fix.

Selector negative coverage additionally mutates `ldb`, `ldc`, composer,
compiler, numeric, and schedule revisions individually.

## TDD evidence

- Graph admission/ABI RED: the focused graph tests first failed because
  `TriadSm89Half` was not in the native-module set and because the SM89 NN base
  did not select the five-argument bundle ABI. They pass through the real
  argument encoder after the fix and verify all four pointer slots plus the
  exact 32-byte bundle bytes.
- Per-symbol ABI RED: a synthetic valid NN entry was rejected when an unrelated
  NN sibling contained a Driver ABI error. The revised census lookup and real
  exclusion path retain the valid NN/NT siblings and exclude only the malformed
  symbol.
- Qualification-hook RED: the host test did not compile while the deterministic
  public half seeding seam was absent. It now proves salt stability, distinct
  salts, finite active values, and unchanged guard words. The ignored live test
  uses the same hook for the independent retained oracle and A/B snapshots.

## Verification

- `cargo test --features cuda,cudarc/fallback-latest --lib sm89_half -- --nocapture`
  — 38 passed.
- `cargo test --features cuda,cudarc/fallback-latest --lib public_half_seed_pattern_preserves_guards_and_is_salt_stable -- --nocapture`
  — 1 passed.
- `cargo test --no-default-features --test sm89_half_source_freeze -- --nocapture`
  — 22 passed.
- `cargo test --features cuda,cudarc/fallback-latest --test gemm_bi_sm89_half_selector_qualification --no-run`
  — compiled successfully.
- `cargo test --features cuda,cudarc/fallback-latest --lib --no-run`
  — compiled successfully.
- Scoped `rustfmt --check` over the eight touched Rust files — passed.
- Scoped `git diff --check` — passed.

## GPU handoff

On an exclusive RTX 6000 Ada, run under each selected CUDA 12.8/13.0/13.2
toolkit:

```text
cargo test --features cuda,cudarc/fallback-latest \
  --test gemm_bi_sm89_half_selector_qualification \
  sm89_half_actual_auto_qualification -- --ignored --nocapture
```

The live harness must bind all 11 actual-AUTO cells to `TriadSm89Half`, bind
each independent oracle to `TriadSm80`'s retained Tile128 symbol, and pass exact
eager/graph/oracle bits, immutable A/B snapshots, and both red-zone checks.
