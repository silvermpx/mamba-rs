# Ada RNA full-corpus qualification on three CUDA toolkits

One test-source change on top of f3248a16: the forced RNA correctness corpus
now includes tail/A–E independently of the13.2 actual-AUTO assertions.
No production CUDA/Rust dispatcher/epoch change and no new timing claim.

The real host regression first failed because force mode supplied only tail/A
instead of all six families. After decoupling the shared shape list and view
matrix, it passed. Source SHA256:
`7f0566985b7c80ed858258104f166fc46bc7328cd8548eab778fae94e4eb977f`.

## Full GPU results

| Matching toolkit | Force tests | Actual-AUTO test | Completed view groups per run |
| --- | --- | --- | ---: |
|12.8|cold2/2, warm2/2|not run; not yet promoted|448|
|13.0|cold2/2, warm2/2|not run; not yet promoted|448|
|13.2|force2/2|1/1|448|

Every run has exactly the expected448 unique groups: tail168 and56 per hot
family. Each group retains all five incumbent rung comparisons, full output
guards, finite/exceptional inputs, both bias forms, C4/C16 and row views, two
eager repeats, exact physical graph ABI/geometry and two poisoned replays.
The separate unsafe-input/K0/empty test and immutable-input checks remain.
The13.2 AUTO wrapper additionally retains actual route, eager/graph and
misaligned-input fallback assertions. Cross-toolkit bit equality is not
claimed; these are within-toolkit cross-route proofs on one Ada board.

Main independently checked every raw run against the exact expected448-key
set, not merely marker totals or generated `.groups` files; verified the
intended RED/GREEN, all167 local/remote source hashes and raw manifest; and
matched all9 cache filenames/blob hashes to the already committed matching
toolkit evidence. Main then reran the13.0 force filter directly:2/2,448 unique
groups,28.02s; both live source and executable hashes matched the report.

The authoritative original bundle is `raw/`, with its own21-entry manifest.
`main-independent-verification.json` and `main-cuda130-force-recheck.log`
record main's additional checks. `review.md` records independent review.
The root manifest covers the selected archived bundle/docs/main evidence;
early acquisition mirrors outside `raw/` are not part of that manifest.

This closes the known full B–E exceptional/prefix/view coverage gap. It does
not itself widen AUTO. The next separate change may promote the measured
12.8/13.0 winners, bump the routing epoch, preserve toolkit-specific C
fallbacks, and rerun actual-AUTO/FAST confirmation. Existing13.2 production
acceleration, other precisions and Triad kernels are unchanged here.
