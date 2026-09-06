# Ada TF32 C post-AUTO: active private-cache confirmation

2026-09-06. This supplements the warm and uncached qualification of the
`26b44c4d` dispatch change; it is not a new kernel implementation.

Unlike the historical `privatecache` directory from the first repeat, this
run really used a mode-0700 application cache. `preflight.log` records its
permissions and immutable source/binary hashes. `private-cache-files.txt`
records the three resulting mode-0600 cache artifacts; the binary blobs are
not committed. CUDA Driver caching was disabled with `CUDA_CACHE_DISABLE=1`.

The production AUTO route is M64S2 for C0/C1 on the 142-SM Ada, known NVRTC
13.2. The forced control is the old M128S2 route. Both bias states, eager and
graph paths, both timing orders, and 101 paired windows give eight records.
All eight pass raw output, AUTO, repeat and applicable graph replay bits;
the captured symbols confirm the physical M64S2/M128S2 functions.

Old/new paired p50 is 1.025144--1.035193; p95 is 1.026725--1.037257.
This confirms the internal improvement, not a cuBLAS FAST victory:
AUTO/vendor median latency ratios are still approximately 1.93--2.14.

The loaded Fixed source, invocation and artifact digests match the earlier
qualification exactly. Full samples and completion metadata are preserved in
`c-postauto-old-m128-control101-privatecache-v2.log`. The corresponding
benchmark invocation is documented in the sibling
`fixed-tf32-c-postauto-ada-20260906/commands.md`; this repeat changes only the
cache environment described above. No AUTO gate was widened to another
toolkit, architecture or shape based on this repeat.
