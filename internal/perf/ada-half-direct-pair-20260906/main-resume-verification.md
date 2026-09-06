# Main verification — restored-access Task 2 completion

Main independently ran the committed strict analyzer against all four fresh v2
whole-run logs and their committed external controls. Each produced exactly 80
records, revision 41, `required_census_complete:true`, and
`auto_admission_authorized:false`. All wrappers and test suites exited zero.

The final 101-window result is identical on CUDA 12.8 and 13.0: swizzle is the
robust direct winner for all B/C/D/E dtype/bias cells plus BF16 A1 (17/20).
BF16 A0, F16 A0 and F16 A1 have no robust direct winner. Their worst
swizzle-over-pipeline p95 values are respectively 1.000779, 1.005588 and
1.000275 on 12.8, and 1.000613, 1.005206 and 1.001090 on 13.0.

The two 21-window screens each had 18/20 swizzle winners but are not used as the
final route matrix. The invalid original attempts and the later v1 post-residual
attempt remain excluded. The worker's 22-entry resume manifest passed from the
evidence-directory working directory; an initial invocation from the worktree
root produced only path-open failures, not content mismatches.

The clean source reconciliation records `SOURCE_EXPECTED=174`, all source
checks OK and `SOURCE_EXIT=0`. The final reconciliation binds the wrapper/test,
three binaries, two controls, nine cache blobs and 0700 modes, and the final
idle Ada UUID/driver/CC8.9/142-SM state with zero substantive exits. Independent
evidence review is APPROVE with no blocking finding. Task 2 changes no AUTO,
dispatcher, production CUDA or tuning revision; those belong to Task 3.
