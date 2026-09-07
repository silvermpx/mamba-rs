# Task7 source-gate fix round 1 report

Status: **FIX ROUND ACCEPTED AND QUALIFICATION COMPLETE**. Formatted final5b
builds and four smokes pass; the fix-only re-review accepted all three
Important fixes with no new Critical/Important finding. Root accepted the
source+smoke gate. The same frozen source and binaries were used for all Task7
screen21/eligible101 evidence. Pre-format final5 identities remain preserved
as intermediate history.

## Scope and accepted findings

This round implements only the three Important findings in
`ada-f32-tf32-toolkit-task7-source-review.md` and the binding
`ada-f32-tf32-toolkit-task7-fix1-brief.md`. It changes no kernel, loader,
selector, holder, or tuning/numeric/schedule revision. Root's byte-identical
final4 freeze and all final4 attempts remain preserved.

1. The runtime now checks actual compiler numeric ABI revision 5 and schedule
   revision 8, together with tuning revision 43, before any case allocation or
   workflow. Both actual checked values are emitted in identity JSONL and are
   independently required by `analyze.py`.
2. A `prelaunch_gate` checks revisions and the live Fixed artifact against the
   confirm screen artifact before invoking `single_term_controls`, the first
   kernel workflow. Its host negative counts launches and proves an artifact
   mismatch leaves the count zero.
3. A `timing_boundary_gate` calls `Case::timing_boundary` immediately after
   timed observations and before every post-timing output read/eager/graph/
   poison workflow. That boundary compares complete saved A/B/bias bytes,
   input guards, common operand pointers, and every output guard. The later
   input check is retained. A/B/bias negatives model a later restoring replay
   and prove it cannot run before the mutation rejection.

## RED and GREEN

Python command:

```text
PYTHONDONTWRITEBYTECODE=1 python3 internal/perf/ada-f32-tf32-toolkit-20260907/test_validation.py
```

- RED: `internal/perf/ada-f32-tf32-toolkit-20260907/fix1-python-red.log`,
  exit 1. Four expected subtest failures showed missing/wrong numeric ABI and
  missing/wrong schedule revisions were accepted.
- GREEN: `internal/perf/ada-f32-tf32-toolkit-20260907/fix1-python-green.log`,
  9/9 tests pass, exit 0.

Focused Rust command, through the sole toolkit environment constructor:

```text
python3 /root/evidence-ada-f32-tf32-toolkit-20260907/run.py host 12.8 fix1-red
python3 /root/evidence-ada-f32-tf32-toolkit-20260907/run.py host 12.8 fix1-green
```

- RED: `internal/perf/ada-f32-tf32-toolkit-20260907/cuda128-fix1-red/host.log`,
  exit 101. Four compile errors identified the absent `prelaunch_gate` and
  `timing_boundary_gate` used by the new focused tests.
- GREEN: `internal/perf/ada-f32-tf32-toolkit-20260907/cuda128-fix1-green/host.log`,
  12/12 focused tests pass, exit 0.

## Final5 frozen source and builds

Direct rustfmt was run on the focused support module. Root's direct
`rustfmt --check --edition 2024 --config skip_children=true` found one import
wrap in the parent; that two-line formatting-only patch was applied before GPU
smokes. The same check now passes, as do support rustfmt, `git diff --check`,
and all four Python tool syntax checks.

| File | SHA-256 |
|---|---|
| `tests/gemm_bi_fixed_performance.rs` | `317cb44c60926bca32da6b3f7828187be7ab61b4173cf6b07297a86961698419` |
| `tests/support/fixed_sm89_toolkit_admission.rs` | `cb20805c079b200f1bb77f02912c16fb56113d26cf3204c04d046cc872cee73f` |
| `internal/perf/ada-f32-tf32-toolkit-20260907/run.py` | `24275ce7ed506336f64297a0eb29afaefe12d366ca4ed47860a0a16031dbd1aa` |
| `internal/perf/ada-f32-tf32-toolkit-20260907/analyze.py` | `e8bebda48987a699392f7c8bdbdf8f74b61349a2d3aebc20a36bf1e222026182` |
| `internal/perf/ada-f32-tf32-toolkit-20260907/remote.py` | `cb91f07492a9e9b661d98e6be58d7b71f48511eae082ffbeca133d50e7bfdf0e` |
| `internal/perf/ada-f32-tf32-toolkit-20260907/test_validation.py` | `43afdd008451cd89195200e687b3fe8142072969fbb701b78d5f3c33e5ab4608` |

Formatted composite runtime SHA:
`97f3d43fc315c96238f41f9f39bc15418518b5493e216fb9bc0a8f03a46e5bc7`.
Final5b uses distinct target/cache/binding paths and does not overwrite final4
or the pre-format final5 intermediate.

Build command per toolkit:

```text
python3 /root/evidence-ada-f32-tf32-toolkit-20260907/run.py build TOOLKIT final5b-build
```

| Toolkit | Feature | Focused | Nonignored | Binary SHA-256 |
|---|---|---:|---:|---|
| 12.8 | `cuda,cudarc/cuda-12080` | 12/12, exit 0 | 65 pass + 65 ignored, exit 0 | `0f3f2e617797fcce4e70789cc507cd25704429f98663812013d6df9851321bb5` |
| 13.0 | `cuda,cudarc/cuda-13000` | 12/12, exit 0 | 65 pass + 65 ignored, exit 0 | `cf90b10de3cb8afb0a4a68575a1c9db9597750ff3b929dfff85864a423009154` |

The build output is not warning-free: the parent imports
`NUMERIC_ABI_REVISION` and `SCHEDULE_REVISION` but does not use them, because
the required runtime gate compares actual compiler identity against literal
5/8. This warning was preserved and classified below Important; root explicitly
deferred cleanup rather than invalidating the accepted functional source.

All four final5b functional smokes passed strict PRE, test/POST/wrapper/outer
SSH, exact record closure and independent analysis, including emitted numeric
5/schedule8:

| Toolkit/family | JSONL SHA-256 |
|---|---|
| 12.8 exact | `dc87b9e2f2a9e41ec7923cf37100fe641fc6a50f5aa288fb481ba46b91174261` |
| 12.8 TF32 | `875155cbf0a8cf1387223473f9dc04cf7fb25195ba1d903917aa3761aabf623d` |
| 13.0 exact | `7b66bcd85135f74219ad96363f6ab89cf4e307b34787e8a873f870b20cb89aa7` |
| 13.0 TF32 | `ca7854c0a041fa682cbb3ad5a140cc3547c6142813f2fdf378ccfcd2c6b8a15b` |

Final5b binding SHAs are
`243778b249c06534329f2bba18775adc3ef661d0996aef87ae070fbd4f61e2fd`
(12.8) and
`eae1bd397e66a01fd5433b8e74163c56f7542af503aec3e6a514326260ff6f4b`
(13.0). All build/smoke files and closure receipts are locally mirrored.
The four authorized screen21 runs and both exact eligible101 runs later passed
complete closure on this unchanged source. TF32 had an empty eligible subset
on both toolkits and therefore stopped at 21. Final release telemetry is quiet
with no compute apps. The full matrix, constituents, artifact archives, rooted
manifest, commands, and release recommendation are in
`ada-f32-tf32-toolkit-task7-report.md` and the adjacent evidence directory.
