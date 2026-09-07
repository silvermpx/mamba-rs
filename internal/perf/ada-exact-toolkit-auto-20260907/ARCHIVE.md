# Selected AUTO44 integration evidence

This directory preserves the completed exact-F32 dispatcher integration and
its source/build, functional, numerical, physical and performance evidence.
Start with final-report.md and final-evidence-review.md. Source-v1 and fix1
reviews, patches and checkpoints retain the original findings and their fixes.
Final1 build checkpoints are historical, not mislabeled final2 qualification.

ARCHIVE_SHA256SUMS is relative to the repository root. It covers every selected
text/source/raw record in this directory, excluding itself and Python bytecode.
Verify with `shasum -a 256 -c` from the repository root. The nine Rust files and
documentation are preserved by the accompanying Git commit; the final2 bindings
give the measured source and exact remote executable identities.

The saved numerical/performance records use the accepted final2 binary builds.
This commit does not claim a new local replay of executable or CUDA-cache tar
archives. The earlier immutable Task7 archive remains separately committed.

Root additionally checked all 357 source inputs per toolkit, replayed both
post101 raw files and their smoke dependencies with root-replay.rb, and ran
the seven host analyzer tests, direct rustfmt and tracked diff checks. The
independent reviewer accepted this bounded integration with no findings.

No GPU suite was repeated for packaging. The next activity is profile-guided
candidate discovery under the user-approved tiered protocol, not a release.
