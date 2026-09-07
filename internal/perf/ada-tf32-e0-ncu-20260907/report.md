# Ada Fixed TF32 E0 — production Nsight diagnosis

Status: COMPLETE, counters only; no admission claim and no production source,
kernel, loader, selector, or cache change.

## Bound invocation

- CUDA 13.2, Nsight Compute 2026.1.0.0 build 37166530, driver 595.45.04.
- RTX 6000 Ada, CC 8.9, 142 SMs, UUID
  `GPU-d1edd7be-e88d-aed6-047d-622163306f0e`.
- Accepted Task8 final2 production test binary SHA-256
  `4ecefea9345226073e9671017bb14450ebed562be8c5c551a2e6d9de3859868d`.
- Accepted measured-source SHA-256
  `2ac1c93ba3e4682138a0df3341a26921438334382532d6943277268320655285`.
- Existing private final2 cache and binding were revalidated before acquisition
  and again before release.
- Existing comparator only: row `tf32`, cell `hot_e`, `(M,K,N)=(2048,2304,768)`,
  bias false, one window. It reported public AUTO
  `Tf32RnaM128N128S3`, true vendor
  `CUBLAS_COMPUTE_32F_FAST_TF32`, equal repeat bits, and equal normalized
  error `0.0002385988569540601` for AUTO and Fast against the PEDANTIC
  reference.

All four counter acquisitions selected exactly the 131st matching launch
(`--launch-skip 130 --launch-count 1`). The explicit settings were kernel
replay, `--clock-control none`, `--cache-control none`, and dynamic pipeline
boost state. The main reports used 17 replay passes; the separate traffic
reports used four. Nsight consequently warned that caches were uncontrolled;
this was the deliberate warmed-cache acquisition, and the warning is retained
in both raw logs. Profiled event durations are not qualification timing.

PRE at `2026-09-07T09:02:08Z` and RELEASE at
`2026-09-07T09:02:38Z` were quiet, no-app snapshots on the same GPU. The
outer SSH exit was zero and the completion marker was present.

One preliminary path-discovery invocation, `ssh ada ncu --version`, exited
127 because the non-login shell did not contain the CUDA toolkit in `PATH`.
It launched no GPU work. That failure exists only in the agent tool transcript;
the final acquisition records the absolute CUDA 13.2 binary and saves its full
version and help output.

## Result

The separate unprofiled one-window baseline was:

| order | RNA AUTO (us) | Fast TF32 (us) | AUTO/Fast |
|---|---:|---:|---:|
| AB | 119.003 | 91.117 | 1.3060 |
| BA | 119.058 | 91.372 | 1.3030 |

The selected warmed Nsight launches showed:

| counter | RNA AUTO | Fast TF32 |
|---|---:|---:|
| grid / block | 96 / 256 | 128 / 128 |
| waves per SM | 0.68 | 0.90 |
| dynamic shared bytes | 98,304 | 81,920 |
| registers/thread | 153 | 224 |
| profiled duration, us | 119.456 | 91.648 |
| executed instructions | 28,822,272 | 15,968,256 |
| HMMA instructions | 3,538,944 | 3,538,944 |
| global load/store, MB | 226.492 / 6.291 | 226.492 / 6.291 |
| warm-cache DRAM, MB | 19.473 | 19.446 |
| L2 hit rate | 99.70% | 99.81% |
| tensor pipe, elapsed | 46.51% | 60.71% |
| ALU pipe, elapsed | 19.18% | 9.91% |
| short-scoreboard stall/issue | 0.388 | 0.016 |
| barrier stall/issue | 0.358 | 0.054 |
| MIO-throttle stall/issue | 0.190 | 0.012 |

The RNA launch performs the same matrix work and essentially the same global
traffic, but executes 1.805x as many instructions. It also has much larger
short-scoreboard, barrier and MIO-throttle stalls. Although Fast reports a
128-CTA physical grid, its SASS predicates 32 padding CTAs before the main body:
512 warps execute the first CTA-ID reads but only 384 warps reach the CTA-Z
check. Thus both arms have 96 useful tiles. Their average-SM-active / elapsed
cycle fractions are also nearly identical (RNA 0.659, Fast 0.655). The evidence
therefore points primarily to non-MMA instruction plus
synchronization/shared-memory-pipeline overhead—not extra HMMA work, DRAM
traffic, or more useful Fast CTAs—as the immediate gap.

The next narrow candidate hypothesis is M128N96/S3: for E0 it statically makes
128 useful CTAs (`16 * 8`) and its three-stage A/B footprint is 86,016 bytes
(`3 * (128*32 + 32*96) * 4`). That geometry is not measured here and is not a
speed claim. Reducing the 1.805x instruction burden is at least as important as
raising useful tile count from 96 to 128; unlike the Fast grid's padding, those
extra N96 tiles would perform work.

## Evidence

- Human/machine summary: `summary.json`.
- Installed profiler: `run/ncu-version.log`, `run/ncu-help.log`.
- Exact environment and settings: `run/identity.json`,
  `run/environment-pre.log`, `run/environment-release.log`.
- Unprofiled truth: `run/unprofiled-baseline.log` and `.json`.
- Raw reports: `run/custom-rna.ncu-rep`, `run/fast-tf32.ncu-rep`, and the two
  `-traffic.ncu-rep` files.
- CSV exports: each arm's `-details.csv`, `-raw.csv`, `-sass.csv`,
  `-traffic.csv`; session exports are also retained.
- Commands/exits: `run/commands.json` (19/19 exit zero), `outer-ssh.log`,
  `outer.json`, `run/status.json`.
- Mirrored remote manifest: `run/manifest.json`, 28 entries, SHA-256
  `e4685f6fea351d649afe136b74dc2b27b84b805e15d67936aa2d68a33d3485b2`;
  every entry was recomputed locally and matched.
- Main report SHA-256 values: custom
  `8c0bded51ea3ad99f9cb0677b2abbcd69f0fa6cffa5e6613d284e12a00d0e7ad`,
  Fast `b4e5f1dac480159e4a9ba09dad2185579a858fae6a973e1f995da6c7c55032f9`.
- Outer transcript SHA-256
  `805fa547c062bf724267fcaec51d165e0c915177e78be212c731abfcf138e657`.
