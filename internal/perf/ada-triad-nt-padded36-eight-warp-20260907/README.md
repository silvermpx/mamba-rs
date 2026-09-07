# Ada Triad NT: eight compute warps, 2026-09-07

**STOP_NO_RETRY.** The bounded bits/resource test passes, but this candidate
loses to actual production AUTO in every timing stratum. No production route
or numerical/tuning ABI changes are made.

## Result

Target logical M/K/N=(2048,768,3072), NT output2048x768, reduction3072.
CUDA13.2, RTX6000 Ada CC8.9. Ratios are candidate/actual AUTO, not cuBLAS:

- Eager ABBA: p501.081097417, p951.081298543; medians297.271/274.972us.
- Eager BAAB: p501.080782645, p951.081894046; medians297.190/274.955us.
- Graph ABBA: p501.083137674, p951.083544202; medians296.444/273.691us.
- Graph BAAB: p501.083239146, p951.083445400; medians296.448/273.692us.

Registers fall from the profiled production154 to100, but speed does not
improve. Local0, static shared0, dynamic shared82,944B, max threads256,
one resident CTA. Reduced registers alone did not solve this case.
Predeclared retention required both quantiles<0.99 in all four strata.
No retry,21/101-window qualification, new cuBLAS/profile/full-matrix run
or production promotion follows this loss.

## Scope and proof

Only the test-generated source changes: exact NT/M128/N64/S3 ownership becomes
eight compute warps, MAtoms2, warp_m=(warp>>1)*32. Original padded36 storage,
four A/two B vector copies per thread, scalar RNA fragment conversion,
ascending K8 MMA order, staging/barriers and epilogue stay unchanged.
The unique export and matching signature assertion are both renamed.

Host tests cover all8192 outputs once and compare all four A/two B fragment
register coordinates at each K8 issue against the old ownership. Root's
independent actual-generated-source proof reverses the three ownership
declarations and two symbol references to recover production source exactly,
and verifies all18 exports equal their signature-assertion targets.
Native suite13/13 plus that root test14/14 pass; rustfmt/scoped diff checks pass.
Author observed a genuine three-test behavioral RED before implementation.

The actual GPU test executed once49.52s: resource gate, finite full-mantissa
target and forced-tail(129,65,36) eager/graph repeated bits, immutable inputs,
guards and post-timing graph replay all pass. Alpha1/beta0/no bias.
The tail reference is forced same tile, not tail AUTO. This is not full
NaN/Inf/K0/prefix/stride/epilogue/toolkit/architecture qualification.
Graph identity/ABI layout checks do not independently decode every captured
argument value. Candidate PTX and a full dlopened CUDA-library inventory are
not persisted; this stopped screen is not artifact-complete admission.

## Bindings and replay

Source checkpoint base27926301; sole changed test source
SHA989acbcf79c07607dc08125e2e8d5141761416dbc7422e44e3e05cde072f32c5.
372-input manifest SHA94a6e459d47b1bced440b76607af444e7e7e6b9c6d0da27765a44c2d7aedd758.
Binary SHA11b99112c4de62e0e5644fb7711f61a944343286f1c4890a54cc4693d24b4869.
Raw SHA89c6bd600cbe593da44a6ff7e68973b4c8f59e8e41c59fa27a086cc7593cd9fe.
Composed candidate source SHAed845491499db900868cd74722529862f14d23d890dea7cc8bf32fa770a4977d.

The existing isolated build target and private0700 preseeded production cache
were deliberately reused and verified; this is not cold-cache timing.
The validated previous arm runner is imported by the small run-once7 wrapper.
PRE is quiet; immediate busy RELEASE is preserved; distinct5s DRAIN is quiet.
No GPU process was killed and no clocks changed.

At this checkpoint's worktree revision:

    ruby internal/perf/ada-triad-nt-padded36-eight-warp-20260907/replay.rb
    ruby internal/perf/ada-triad-nt-padded36-eight-warp-20260907/host-source-replay.rb

Replay checks372 source hashes, binary/source/raw bindings, one actual exact
test,28 raw four-leg brackets, all four quantile pairs and STOP.

Next read-only research: reuse the existing split-K full-tile staging idea
without its reduction or128-thread assumption, and inspect explicit warp
register-fragment pipelining. Neither is implemented/measured here.
