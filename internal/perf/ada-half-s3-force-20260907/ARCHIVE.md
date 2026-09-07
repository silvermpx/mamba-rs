# Production Ada half S3 force checkpoint

The BF16/F16 S3 kernel is now reachable through the production public force
dispatcher as `FixedTile::Tc128Sm89S3` on Ada. AUTO remains revision42 and
keeps its existing selections until production paired timing qualifies a win.
This checkpoint establishes functional integration, not a new latency or
cuBLAS Fast victory.

| Final gate | CUDA 12.8 | CUDA 13.0 | CUDA 13.2 |
| --- | ---: | ---: | ---: |
| Library / force-static tests passed | 644 / 48 | 644 / 48 | 644 / 48 |
| Full half pipeline tests passed | 11 | 11 | 11 |
| Retained RNA / exact tests passed | 2 / 5 | 2 / 5 | 2 / 6 |
| S3 BF16 / F16 registers | 182 / 182 | 182 / 182 | 188 / 188 |
| memcheck / racecheck / synccheck | clean | clean | clean |

Every S3 entry has zero local memory, stack and spills, 98304 dynamic shared
bytes, 256 threads and one resident CTA. Both cold and warm production
holders pass their ABI/resource gates. Actual eager and independently
poisoned graph paths are tested, including captured arguments, both biases,
views, guards, tails, nonunit alpha/beta and independent strides.

The original 12.8 runner failed by invoking an existing 13.2-only exact-AUTO
fixture. Its failed log is preserved. The corrected continuation covers only
the applicable forced exact tests and remaining gates using the same
source/cache. No numerical assertion was weakened. CUDA13.2 additionally
passes actual exact-F32 AUTO, TF32-C AUTO, live SM89 Triad cohorts and two
SM120 compile/PTX contracts. Ada compilation is not SM120 runtime proof.

All50 previous CUDA files and all Triad composed sources remain unchanged.
The source review's only material finding, extreme-K lookahead overflow,
is closed with an S3-only host K+127 check and RED/GREEN boundary plus public
dispatch coverage. The tested S3 CUDA body remains unchanged.

`final-report.md` records commands, failure history, source/binary/cache and
compiler identities. `source-review.md`, `fix1-review.md` and
`evidence-review.md` preserve the independent reviews. Existing exact-F32 and
TF32-C cross-toolkit AUTO qualification gaps are explicitly recorded in
`existing-toolkit-auto-gaps.md`; this half-only task does not close inference
or Triad as a whole.

`ARCHIVE_SHA256SUMS` covers the selected version-controlled files, relative
to the worktree root. `SHA256SUMS` covers the original complete122-file local
evidence set, including the large PTX/SASS/cubin and source/binary/cache
archives that remain on disk rather than in Git. Root independently checked
all122 entries; full manifest SHA256:
`9899ead9a0d4e9a5ec29fcb4eae68dfc34b9b450ae432ef88d14c182c7992aff`.
No source, kernel or evidence was deleted. The Ada lane was released idle
with no compute apps at2026-09-07T02:48:19Z.
