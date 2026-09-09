# Ada Triad F16 TN full-domain mapper/epilogue compile stop — 2026-09-09

Outcome: **compile STOP before GPU, no timing and no production change.**

The test-only d768-in `(2048,768,3072)` candidate replaced the generic linear
CTA div/mod mapping with grid `(48,12,1)` and direct `blockIdx.{x,y}` ownership.
It also removed tail/alignment predicates from the fully covered `float2`
epilogue. The transform is reversible and leaves staging, HMMA operands,
ascending K16 association and arithmetic unchanged. Non-target and tail cases
remain on the retained regpipe+`float2` route in the harness.

The predeclared CUDA13.2 resource gate rejected the candidate at 128 registers,
above the retained 125-register cap. Static shared is the expected 32,768 bytes.
Per the stop rule, no CUDA context, correctness launch, once3, once7 or cuBLAS
Fast timing was run. Do not retry this unchanged mechanism; a new source-backed
lifetime refinement would be required before another compile.

Frozen source identity:

- helper SHA256: `d6cf7dcd2b2d4c0ed707691e37adc9cf503adb4270ffa67325f235efe7e1ef77`
- harness SHA256: `f63f22e717da030e6f9a765d9a6445f4e5855483e19270442cf5a769b3e83032`
- [raw compile output](raw.log)
