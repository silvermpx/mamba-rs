# TF32 E0 profile evidence

This read-only profile follows AUTO44 integration commit44416c22. It used the
same accepted CUDA13.2 final2 test binary, not a newly compiled prototype.
Start with report.md and summary.json. Their corrected interpretation accounts
for cuBLAS's padding CTAs: 128 physical launches are not128 useful tiles.

The four Nsight reports, raw CSV/SASS/session exports, separate unprofiled
baseline, commands, telemetry, binding and outer closure are retained.
run/manifest.json is the original 28-entry remote manifest; root independently
verified all28 hashes and all19 command exits. ARCHIVE_SHA256SUMS additionally
roots the local summaries and acquisition scripts, relative to repository root.
Python bytecode and the manifest itself are excluded from that selected list.

sass-opcode-counts.json independently aggregates dynamic warp instruction
counts from the two SASS CSV files (skip the first kernel-name row; group the
Source column's unpredicated opcode prefix and sum Instructions Executed).
Its totals match the main Nsight instruction counts. FSETP includes explicit
RNA special-value handling and must not simply be removed as an optimization.

This is diagnostic evidence, not paired performance admission, cross-toolkit
qualification, proof of an N96 speedup, or a release. A one-window unprofiled
baseline and a kernel-replay duration must not be presented as a101-window
win. No production kernel source changed for this profile.
