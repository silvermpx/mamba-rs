# Ada TF32 E0 N96 discovery

This checkpoint is a test-only CUDA 13.2 discovery of an M128N96/BK32/S3
explicit-RNA kernel. It does not modify or select production code.

The candidate passed the bounded bit/resource probe against the forced current
M128N128/S3 RNA implementation: E0 and E1, M129/K36/N100 tails with and
without bias, an M1 prefix with an untouched poison suffix, three eager repeats,
three captured-graph repeats, exact graph geometry/ABI, and unchanged guarded
inputs. Its compiled resource record is 128 registers, zero local/static shared
bytes, 86,016 dynamic shared bytes, and one resident block per SM. The actual E0
AUTO route remained `Tf32RnaM128N128S3`.

The one 7-window screen and one permitted 21-window extension agreed:

| Run | Comparator | candidate p50 | comparator p50 | pooled ratio p50 | pooled ratio p95 |
| --- | --- | ---: | ---: | ---: | ---: |
| 7 | production RNA | about 101.51 us | about 119.05 us | 0.852663 | 0.852831 |
| 7 | actual Fast TF32 | about 101.51 us | about 91.06 us | 1.114683 | 1.114876 |
| 21 | production RNA | about 101.52 us | about 119.06 us | 0.852642 | 0.852864 |
| 21 | actual Fast TF32 | about 101.52 us | about 91.07 us | 1.114657 | 1.114887 |

Thus N96 is retained as a batch finalist: roughly 14.7% faster than the current
RNA incumbent at E0, but still roughly 11.5% slower than true Fast TF32. No
101-window run, full qualification, production promotion, or selector change
was performed. The table's ratios pool the ABBA and BAAB observations; the
worst individual-order 21-window RNA ratio p95 was 0.852893993. The small input
probe intentionally is not a finalist-level
special-value corpus; most generated finite values are 1/1024 multiples, with
only the explicit prefix covering signed zero, RNA ties, and subnormals.

All GPU attempts used the accepted Task8 `final2` CUDA 13.2 environment and the
same test binary recorded in each binding. Timing used 128 warmups, event
windows targeting 5 ms, and both ABBA and BAAB orders. Each timing run had a
fresh quiet/no-apps PRE and quiet/no-apps RELEASE.

The first correctness command itself exited zero, but its immediate RELEASE
check correctly failed at residual 9%/1% utilization. That receipt is preserved
unchanged in `cuda132-correctness1/release.json`; the test was not rerun. The
separate `cuda132-correctness1-release/release.json` closes the lane at 0%/0%
with no compute apps.

Focused TDD history is preserved verbatim: `host-red.log`, `gpu-red.log`, and
`gpu-red2.log` establish missing-helper/runtime seams; `gpu-green-build.log`
records the subsequent tuple-format compile failure; `gpu-green-build2.log` and
`host-green2.log` are the final compile and three-test GREEN transcripts.

`sha256-manifest.txt` is the selected, repo-root-relative source/evidence
inventory. It deliberately excludes binaries, target directories, and caches;
binary identity remains captured inside the immutable run bindings. The
binding-addressed historical wrappers are preserved as `run-correctness1.py`
and `run-timing7.py`; their hashes exactly match those bindings.
