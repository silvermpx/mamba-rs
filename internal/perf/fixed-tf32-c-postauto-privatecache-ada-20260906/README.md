# Uncached C0/C1 post-AUTO confirmation

The historical directory/log names say `privatecache`, but this first run
had a mode0755 empty kernel-cache directory. The loader declined persistent
cache use; `private-cache-files.txt` is empty. Driver cache was explicitly
disabled (`CUDA_CACHE_DISABLE=1`). This is uncached/in-memory compilation
evidence, NOT proof of an active private persistent cache. The parent/main
independently checked remote mode755 and all archived checksums.

Same immutable source/binary as `../fixed-tf32-c-postauto-ada-20260906/`.
Every performance record reproduces Fixed source digest35d4fd87...,
artifactf933cc6e... and invocationf094b565... from the prior run.

Result:8/8records,0rejections,completion PASS. C0/C1 × eager/graph × both
timing orders,101windows. Actual AUTO M64S2 symbol in all8records;
rawstorage/AUTO/repeat bits pass, and replay bits pass on all graph rows.
Old/new paired p50=1.026771--1.032084, p95=1.029164--1.033931. This confirms
approximately2.6--3.1% lower median time without shared caches. FAST still
wins (new/FAST median approximately1.931--2.139).

Raw log SHA256:
`bb74e5e4bdcdfedf54439496e3c5caca0426351c1dfea9149ea3b8e32d38851a`.
No CUDA source or numerical arithmetic changes are part of this promotion.
A separate mode0700 persistent-cache run is supplementary evidence; do not
discard or relabel this uncached record.
