# Ada homogeneous-half swizzle force qualification

Task-1 evidence for the force-only production integration of
`FixedTile::Tc128Sm89Swizzle` on the CC8.9/142-SM Ada host. The checkout baseline
is `86fd7f5605ef5aba77f4e51211af362edb929003`. No AUTO selector or routing epoch
was changed.

## Final evidence

- `final-release-host-static-warm-cuda{128,130,132}.log`: release library,
  performance and SM89 source/compile gates plus live holder/resources. Use
  `final-release-host-static-warm-cuda128-rerun.log` for the final CUDA 12.8
  result; the unsuffixed CUDA 12.8 file preserves an intermediate Rust compile
  error that was corrected before the rerun.
- `final-rounding-edges-cuda{128,130,132}.log`: min/max half subnormal, minimum
  normal, maximum finite, cancellation and non-half-representable FP32 bias,
  with a bitwise-complement result poison reset before every candidate.
- `final-full-forced-corpus-cuda{128,130,132}.log`: K=0/64/65/192,
  finite/exceptional values, bias modes, independent A/B/C misalignment, odd
  strides, guards, repeats, graph replay and exact graph arguments.
- `final-forced-hot-a-e-cuda{128,130,132}.log`: original hot A-E shapes, both
  homogeneous dtypes and bias modes, finite/exceptional inputs, prefixes,
  views, guards and exact forced-swizzle graphs.
- `final-eager-identity-smoke-cuda{128,130,132}.log`: minimal one-window smoke;
  eight records per toolkit prove the measured eager-only callsite still captures
  and validates an untimed physical swizzle graph for BF16/F16 and bias 0/1.
  These samples are not timing-admission evidence.
- `final-retained-rna-auto-cuda{128,130,132}.log`: the two retained RNA/AUTO
  tests and 448 unique groups per toolkit.
- `final-retained-extra-cuda132.log`: retained TF32 cohort binding, TF32-C AUTO,
  incumbent half AUTO hot A-E, and exact-N64 forced/AUTO gates.
- `final-sanitizers-cuda128-rerun.log`, `final-sanitizers-cuda130.log`, and
  `final-sanitizers-cuda132.log`: memcheck, racecheck, initcheck and synccheck.
  `final-sanitizers-cuda128.log` preserves the first diagnostic run, where the
  intentionally invalid terminal Driver-ABI query was reported as an API error.
- `final-identities.log`: remote source, release executable and private-cache
  SHA-256 identities. Its attempted `git rev-parse` records that the rsynced
  source is intentionally not a Git checkout; the authoritative local baseline
  above is from the worktree.

Earlier `red-*`, `green-*`, `fixround1-*`, and unsuffixed `cuda*` files preserve
the complete TDD/debug chronology. `SHA256SUMS.raw` binds every immutable raw log
present when this README was finalized.

## Environment

Remote source: `/root/mamba-ada-half-swizzle-force-20260906`.

| Toolkit | CUDA root | Cargo feature | Target | Private cache |
|---|---|---|---|---|
| 12.8 | `/usr/local/cuda-12.8` | `cuda,cudarc/cuda-12080` | `/root/target-ada-half-swizzle-force-cuda128-20260906` | `/root/mamba-kcache-ada-half-swizzle-force-cuda128-20260906` |
| 13.0 | `/usr/local/cuda-13.0` | `cuda,cudarc/cuda-13000` | `/root/target-ada-half-swizzle-force-cuda130-20260906` | `/root/mamba-kcache-ada-half-swizzle-force-cuda130-20260906` |
| 13.2 | `/usr/local/cuda-13.2` | `cuda,cudarc/cuda-13020` | `/root/target-ada-half-swizzle-force-cuda132-20260906` | `/root/mamba-kcache-ada-half-swizzle-force-cuda132-20260906` |

All three cache directories were created new with mode 0700. The preceding
frozen RNA source, targets and caches were not modified.

## Main independent verification

`main-independent-verification.json` records the independent live comparison of
174 source inputs,14 release binaries and9 cache files, including six unchanged
Triad blobs. `source-manifest-final.sha256` pins those174 local build/test inputs;
`main-live-sha256.log` is the independent remote checksum response. Main reran
CUDA13.0 hot A-E (1pass,11.27s) and exact live holders/cache (1pass,2.94s); separate
`main-hot-a-e-cuda130.log` and `main-holder-cache-cuda130.log` preserve transcripts.
The JSON records exact commands and minor diagnostic-command corrections.

`identity-cuda{128,130,132}.json` are controls for future paired timing, established
from this completed qualification, not from future timing rows. The independent
26-fragment source reconstruction matches their Fixed source digest; all24
eager-only smoke records match these controls. Header manifests include source
macro analysis and therefore differ from the preceding RNA source checkpoint.

`integration-report.md` contains the worker's full command/result report;
`static-fix-review.md` records the independent source-fix review and
`final-evidence-review.md` approves the final force-only qualification evidence.
`SHA256SUMS.raw`
is preserved unchanged at48 entries. The final root `SHA256SUMS` additionally
binds subsequent main transcripts, controls, manifests and reports. No timing
admission, vendor victory or AUTO selection follows from this force checkpoint.
