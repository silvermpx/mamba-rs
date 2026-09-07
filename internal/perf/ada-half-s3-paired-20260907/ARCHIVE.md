# Production Ada S3 paired timing checkpoint

Only CUDA13.2 BF16/F16 B0/no-bias qualify as robust improvements over actual
production AUTO42. All three toolkits were measured independently, with both
eager/graph execution and both mirrored-bracket starting parities. No production
CUDA or AUTO change is included in this checkpoint.

| CUDA | Dtype | Screen21 worst p50/p95 | Fresh101 worst p50/p95 | Decision |
| --- | --- | --- | --- | --- |
|12.8|BF16|0.983487847 / 1.006867571|not eligible|keep Swizzle|
|12.8|F16|0.981792374 / 0.997414724|0.981555906 / 1.006347845|keep Swizzle|
|13.0|BF16|0.983097932 / 1.011073358|not eligible|keep Swizzle|
|13.0|F16|0.981527365 / 0.994274018|0.981151367 / 1.015680726|keep Swizzle|
|13.2|BF16|0.930532163 / 0.951183161|0.929639720 / 0.945012679|qualifies for S3 AUTO|
|13.2|F16|0.929007877 / 0.955216865|0.927334868 / 0.958864521|qualifies for S3 AUTO|

Ratios are measured S3/AUTO time; lower is better. Both p50 andp95 must be<1
in every path/parity at21 andfresh101. Failed/mixed cells remain valid evidence
and were not repeated. B0 is literally `(M,K,N)=(4621,768,2304)`, biasfalse,
alpha1beta0, homogeneous half storage. No other shape/bias/device is promoted.

On13.2 the candidate's worst median time is7.04% lower inBF16 and7.27% lower
inF16 than actual oldAUTO. Native-half cuBLAS Fast still leads this B0: worst
candidate/Fast101 p50/p95 are1.213927713/1.232313497 and
1.131168506/1.167174487 respectively. PEDANTIC F32 is only the numerical
reference. This is an own-kernel improvement, not a vendor victory or closure
of inference/Triad/allprecisions/allarchitectures.

The new isolated harness records actual publicAUTO return and captured kernel
arguments, independently poisoned graph overwrite, repeat bits, guarded output,
input immutability and numerical error. The timed graph contains20 GEMMs and
no reset/readback/print/reference work. NativeFast modes are explicitly queried.
All3 builds pass focused3/nonignored51/64; all3 smokes pass8configs. Every
screen has2016raw/504pairs;12.8/13.0 confirms4848/1212,13.2confirm9696/2424.

`final-report.md` records exact commands, all failures and identities.
`winner-matrix.json` includes full own/Fast constituents and raw hashes.
`root-recomputed.json` independently reconstructs every final smoke/screen/
confirm bracket and quantile. The corrected hostvalidator binds all eight
ordered SSH records and actual exit values to the same command, raw log,
result and PRE/POST telemetry;19 RED failures become11 passing test groups.
All10 saved actual runs pass revalidation, without changing measured Rust,
binaries or raw logs. Original derived reports remain preserved in an archive.

The source review's Minor M1/M2 are explicitly carried into the immediate
AUTO43 task: reject unused legacy controls and check input immutability before
warmup as well as afterwards. Neither gap changes a timed cohort or permits
corrupted final inputs to pass this checkpoint. Independent source, scoped
fix and final evidence reviews are archived beside this note.

`ARCHIVE_SHA256SUMS` covers selected version-controlled text/source evidence.
`SHA256SUMS` preserves the original123-file complete local evidence manifest,
including large source/binary/cache archives retained on disk rather than in
Git. Root verified all123 entries; fullmanifest SHA256:
`710391bba6854434ea2340332413edcd57f98e8218321a17a208553fe2d976a5`.
The archives match all356 measured inputs, three actual binaries, nine cache
envelopes and exact Task6A Fixed PTX identities onall3toolkits. Nothing was
deleted. Ada was released quiet/noapps at2026-09-07T03:28:29Z.

Next is literal13.2 BF16/F16 B0/no-bias AUTO43 selection, preservation of every
other cell/fallback, and actual post-AUTO graph/bit/performance qualification.
