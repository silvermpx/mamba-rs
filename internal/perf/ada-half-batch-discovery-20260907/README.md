# Ada half three-arm discovery batch

This directory preserves one bounded CUDA 13.2 screening batch for three
test-only homogeneous-half candidates. It is discovery evidence, not a
production selector or kernel admission. The production loader, dispatcher,
numeric contract, and existing kernels were unchanged.

Shapes are `(M,K,N)`: B0 `(4621,768,2304)`, D0 `(2048,768,2304)` and
E0 `(2048,2304,768)`, homogeneous BF16/F16 as labeled, no bias.

## Result

All four candidate/dtype cells passed the physical resource gate, raw-bit
equality to their actual production control, finite normalized-error gate,
guards, poison overwrite, immutable-input checks, and three eager plus three
graph repeats. The `exceptional` corpus in these records is deliberately
finite: signed zeros, subnormals, and rounding-tie values. It does **not**
contain NaN or infinity.

| cell | actual production control | short7 worst median / p95 vs control | decision |
| --- | --- | --- | --- |
| B0 BF16 compact S3 | S3 | 0.994435 / 1.007763 | **STOP**; misses the 0.985 advance threshold |
| B0 F16 compact S3 | S3 | 0.997467 / 1.032544 | **STOP**; misses median threshold and p95 cap |
| D0 F16 M64N64/BK64/S3 | Swizzle | 0.957428 / 0.959236 | short-screen survivor |
| E0 F16 M128N64/BK64/S2 | Pipeline | 0.966673 / 0.970062 | short-screen survivor |

The D0 and E0 results mean only that these test-only arms met the bounded
short-screen criteria against current AUTO. They are not AUTO routes and are
not Fast wins: their worst Fast-relative groups remained above 1.0. They still
require the full raw-bit and batch-invariance matrix, all three CUDA toolkits,
and a fresh paired-21 qualification before any production consideration.

D0's worst-stratum candidate/Fast median and p95 are1.053728/1.056548;
E0's are1.089085/1.111951. Fast here is the native-half non-PEDANTIC cuBLAS
facade; PEDANTIC is used only as an untimed numerical reference.

B0 absolute observations varied from roughly100us to139us **within this
single invocation**. The paired within-run control result is retained for
screening, not final performance admission. Do not infer a historical gap
reduction by comparing these absolute times with earlier measurement epochs.

## Evidence and provenance

`host-red.log`, `host-review-red.log`, and `host-green.log` preserve the real
standalone-rustc TDD sequence. `build1.log` retains the initial host type error;
`build2.log` is the corrected successful build. `cuda132-batch1/` contains the
source/binary binding, live pre/post/drain/release state, raw test transcript,
and exit result. No binary or kernel cache is packaged here.

Root independently replayed 224 timing brackets across 32 strata and all four
decisions, and verified quiet-lane state plus frozen source hashes. A fresh
standalone host run also passed all 8 tests.

See `sha256-manifest.txt` for repository-root-relative checksums of the three
frozen sources and selected text/JSON evidence.
